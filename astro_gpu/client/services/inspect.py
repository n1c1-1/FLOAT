"""The worker side of Inspect and Frames: everything that reads the disk or takes time.

The views (tune.py, frames.py, sources.py) never touch a file. They ask an ``InspectService``; it runs the job in a
worker thread and answers with a signal in the GUI thread:

* images: the array of a block (stack, denoised, the starless image...) or the engine's preview, with the automatic
  display stretch, the histograms and the astrometric solution (``ShownImage``); the last images are kept (least recently used);
* the metrics of a done result (``api.stage_outputs``), the frame table of Selection, the sessions with their headers
  and health checks, the frames of a group, their thumbnails and statistics.

Four workers: ``images`` (arrays, metrics, tables), ``thumbs`` (the thumbnails of the grid and the filmstrip),
``view`` (the statistics of the frame shown) and ``prefetch`` (the statistics of many frames), so that a long load never
keeps another one waiting. Each request carries a channel and the latest one wins (``ApiWorker``).

``GuiDiskWatch`` is a debug hook (tests use it): it counts the file accesses the GUI thread makes under some folders.
"""
from __future__ import annotations

import os
import re
import sys
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from weakref import WeakKeyDictionary

import numpy as np
from PySide6.QtCore import QCoreApplication, QObject, QTimer, Signal

from ... import api
from ..stretch import mtf
from .worker import ApiWorker, norm_path

# The service's data live in the modules below by subject; this module keeps the GUI disk watch, the service and
# its registry, and gives every name it had, so its importers read them from here.
# ruff: noqa: F401
from .inspect_display import (VisualSpec, GridLine, SkyData, ShownImage, PixelSource, reduce2, stretch_params,
                              neutral_params, neutral_preview, CAMERA_PREVIEWS, _lut, render8, histogram8,
                              stretch_image, _rgb_of_file, _wcs_of, GRID_LINES, _DEC_STEPS, _RA_STEPS,
                              CATALOGUE_G_MAX, CATALOGUE_MAX, _step, ra_label, dec_label, _pieces, sky_grid,
                              catalogue_stars, sky_data, add_sky, load_visual, ImageCache)
from .inspect_frames import (SessionRow, ProjectFrames, _files, _levels, _calib_of, _keys_of, dark_lines, flat_pct,
                             calib_checks, _first_readable, project_frames, FrameItem, browse_group,
                             inventory_headers, calibration_left_out, set_calibration_excluded, frame_stats)
from .inspect_sources import (diag, block_of, not_yet, _MISSING, missing_text, missing_of, not_computed, _unavailable,
                              _w, _array, array_source, _one_plane, preview_source, apply_matrix, grey_sky,
                              color_source, _sky_model, calibrated_source, gradient_source, removed_source, MASTERS,
                              _masters_lock, _masters, _calib_masters, forget_masters, calibrated_frame,
                              _calibrated_colour, calframe_source, wcs_source, ScaledWcs, _stk, _stack_meta,
                              _stack_zone, SKY_BLOCK, crop_frame, stack_place, reference_frame_place, render_place,
                              preview_wcs_source, final_wcs_source, light_thumb, post_key, _debayered, post_source,
                              shown_neutral, resolve_key, _try, _which_done)
from .inspect_blocks import (CalibrationData, _master_stats, calibration_data, CalibrationReport, calibration_report,
                             registration_from_table, RegistrationBundle, registration_bundle, sensor_size,
                             _offsets_anyway, frame_table_any, StackBundle, stack_bundle, ColorBundle, color_bundle,
                             color_fit_of, lights_of, warm_annotations, post_annotations, sky_bundle,
                             final_sky_bundle)

# ---------------------------------------------------------------------------------------------- the GUI disk watch
_WATCHED = ("open", "os.listdir", "os.scandir", "glob.glob", "os.mkdir", "os.rename", "os.remove", "os.rmdir",
            "shutil.copyfile")
_watch: dict[str, Any] = {"on": False, "tid": None, "roots": (), "hits": [], "where": []}
_installed = False


def _norm(p: str) -> str:
    return os.path.normcase(os.path.abspath(p))


def _hook(event: str, args: tuple):
    if not _watch["on"] or event not in _WATCHED or threading.get_ident() != _watch["tid"]:
        return
    if not args or not isinstance(args[0], str | bytes | os.PathLike):
        return                                              # a file descriptor, or nothing to compare
    try:
        path = _norm(os.fsdecode(args[0]))
    except (TypeError, ValueError):
        return
    if any(path.startswith(r) for r in _watch["roots"]):
        _watch["hits"].append((event, path))
        if len(_watch["where"]) < 400:                      # who asked: the calls of the client that led here
            import traceback
            fr = [f"{Path(f.filename).name}:{f.lineno} {f.name}" for f in traceback.extract_stack()
                  if "/astro_gpu/client/" in f.filename.replace("\\", "/") and f.name != "_hook"]
            _watch["where"].append((path, fr))


class GuiDiskWatch:
    """``with GuiDiskWatch(folders) as w: ...``: ``w.hits`` lists the (event, path) the calling thread opened, listed,
    created, renamed or removed under ``folders`` (no disk access in the GUI thread); ``w.where`` says, for
    the first of them, which calls of the client led there. The hook is installed once (``sys.addaudithook`` cannot be
    removed) and costs nothing while no watch is open."""

    def __init__(self, folders):
        self.roots = tuple(_norm(str(f)) for f in folders)
        self.hits: list[tuple[str, str]] = []
        self.where: list[tuple[str, list[str]]] = []

    def __enter__(self):
        global _installed
        if not _installed:
            sys.addaudithook(_hook)
            _installed = True
        _watch.update(on=True, tid=threading.get_ident(), roots=self.roots, hits=self.hits, where=self.where)
        return self

    def __exit__(self, *exc):
        _watch.update(on=False, tid=None, roots=(), hits=[], where=[])

    @property
    def count(self) -> int:
        return len(self.hits)


# ---------------------------------------------------------------------------------------------- the service
class InspectService(QObject):
    """The requests of the views, and their answers as signals (GUI thread). One service per model (``service_of``)."""

    imageReady = Signal(int, int, object)           # slot, token, ShownImage
    imageFailed = Signal(int, int, str)             # slot, token, reason
    outputsReady = Signal(int, str, str)            # workflow id, stage, which: the metrics are in ``outputs``
    framesReady = Signal(object)                    # ProjectFrames
    tableReady = Signal(int, int, object)           # workflow id, token, FrameTable (or an error text)
    groupReady = Signal(str, int, object)           # kind, token, [FrameItem]
    thumbReady = Signal(str, object)                # path, (h, w, 3) uint8
    statsReady = Signal(str, object)                # path, {level, noise, pct}
    previewReady = Signal(int, int, int)            # token, kept, total: what the thresholds of Selection keep
    jobReady = Signal(str, int, object)             # name, token, result of ``request_job``
    jobFailed = Signal(str, int, str)               # name, token, reason (its first line)
    # a reading that failed: what (frames, group, outputs, preview, stats), its request (the token; (workflow id, stage,
    # which) for outputs; the path for stats; None for frames), the reason
    readFailed = Signal(str, object, str)

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.images = ApiWorker(self)               # arrays, metrics, tables
        self.thumbs = ApiWorker(self)
        self.view = ApiWorker(self)                 # the statistics of the frame shown
        self.prefetch = ApiWorker(self)             # background batches (the statistics of the frames)
        for w in (self.images, self.thumbs, self.view, self.prefetch):
            w.done.connect(self._done)
            w.failed.connect(self._failed)
        self.cache = ImageCache()
        self.outputs: dict[tuple, Any] = {}          # (wf, stage, which, key) -> StageOutputs (or None)
        self.headers: dict = {}
        self._tokens: dict[str, int] = {}
        self._thumb_token = 0
        self.thumb_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self.stats: dict[str, dict] = {}
        self._wb = None                              # the gains that make the sky of a light neutral
        self.requests = 0                            # jobs sent to the workers (the tests count them)
        self._rw = None                              # the render worker of the final images (see ``use_renderer``)
        self._rw_own = False                         # made here (and stopped here), not lent by the window
        self._rq: dict[int, tuple] = {}              # slot -> the final image waiting to be rendered
        self._rin: tuple | None = None               # (slot, token, key, wid, spec, which) being rendered
        self._rin_req: tuple | None = None           # its request as it waited in ``_rq`` (asked again elsewhere)

    # ------------------------------------------------------------ the workers' answers
    def _done(self, channel: str, _n: int, result):
        head = channel.split(":", 1)[0]
        if head == "inspect.image":
            slot, token, im = result
            self.imageReady.emit(slot, token, im)
        elif head == "inspect.outputs":
            wid, stage, which, key, out = result
            self.outputs[(wid, stage, which, key)] = out
            self.outputsReady.emit(wid, stage, which)
        elif head == "inspect.frames":
            self.framesReady.emit(result)
        elif head == "inspect.table":
            wid, token, t = result
            self.tableReady.emit(wid, token, t)
        elif head == "inspect.group":
            kind, token, items = result
            self.groupReady.emit(kind, token, items)
        elif head == "inspect.thumbs":
            pass
        elif head == "inspect.stats":
            path, st = result
            self.stats[path] = st
            self.statsReady.emit(path, st)
        elif head == "inspect.preview":
            token, kept, total = result
            self.previewReady.emit(token, kept, total)
        elif head == "inspect.job":
            name, token, value = result
            self.jobReady.emit(name, token, value)

    def _failed(self, channel: str, _n: int, text: str):
        head = channel.split(":", 1)[0]
        why = text.splitlines()[0] if text else "?"
        if head == "inspect.job":
            self.jobFailed.emit(channel.split(":", 1)[1], self._tokens.get(channel, 0), why)
        elif head == "inspect.image":
            slot = int(channel.split(":")[1])
            self.imageFailed.emit(slot, self._tokens.get(channel, 0), why)
        elif head == "inspect.table":
            self.tableReady.emit(int(channel.split(":")[1]), self._tokens.get(channel, 0), why)
        elif head == "inspect.frames":
            self.readFailed.emit("frames", None, why)
        elif head in ("inspect.group", "inspect.preview"):
            self.readFailed.emit(head.split(".")[1], self._tokens.get(channel, 0), why)
        elif head == "inspect.outputs":
            _h, wid, stage, which = channel.split(":", 3)
            self.readFailed.emit("outputs", (int(wid), stage, which), why)
        elif head == "inspect.stats":
            self.readFailed.emit("stats", self._stats_path, why)

    _stats_path = None                               # the frame whose statistics were last asked (``request_stats``)

    def _token(self, channel: str) -> int:
        self._tokens[channel] = self._tokens.get(channel, 0) + 1
        return self._tokens[channel]

    def stop(self):
        self._drop_renderer()
        for w in (self.images, self.thumbs, self.view, self.prefetch):
            w.stop()
        root = getattr(getattr(self.model, "p", None), "root", None)
        if root is not None:                         # the project closed: the masters read of it go too
            forget_masters(root)

    def wait(self, timeout: float = 60.0) -> bool:
        """Process events until the workers are idle (tests and scripted walkthroughs)."""
        end = time.monotonic() + timeout
        ok = all(w.wait(max(end - time.monotonic(), 0.1)) for w in (self.images, self.thumbs, self.view, self.prefetch))
        while self._rendering() and time.monotonic() < end:
            QCoreApplication.processEvents()
            time.sleep(0.005)
        QCoreApplication.processEvents()
        return ok and not self._rendering()

    def busy(self) -> bool:
        return self._rendering() or any(w.busy() for w in (self.images, self.thumbs, self.view, self.prefetch))

    # ------------------------------------------------------------ requests
    def request_job(self, name: str, fn, *args) -> int:
        """Run ``fn(project, *args)`` in the worker (the data of a block: ``calibration_data``,
        ``registration_bundle``...). Answer: ``jobReady(name, token, result)`` or ``jobFailed``; a newer request of
        the same name makes the older answer stale."""
        channel = f"inspect.job:{name}"
        token = self._token(channel)
        self.requests += 1
        self.images.submit(channel, self._job, name, token, fn, args)
        return token

    def _job(self, name, token, fn, args):
        return name, token, fn(self.model.p, *args)

    def request_image(self, slot: int, wid: int, spec: VisualSpec, which: str = "current") -> int:
        """Load a picture into ``slot`` (0: the shown one, 1: the one it is compared with). Answer: ``imageReady`` or
        ``imageFailed``, with the token returned here; a newer request of the slot makes the older answer stale. (A view
        looks in ``cache`` first: a picture seen lately is there.)"""
        if spec.kind == "render":
            return self._request_render(slot, wid, spec, which)
        channel = f"inspect.image:{slot}"
        token = self._token(channel)
        self.requests += 1
        self.images.submit(channel, self._image_job, slot, token, wid, spec, which)
        return token

    def _image_job(self, slot, token, wid, spec, which):
        p = self.model.p
        out = api.stage_outputs(p, wid, spec.stage, which)
        if out is not None:
            hit = self.cache.get((spec.stage, out.key, spec.name, which))
            if hit is not None:
                add_sky(hit, p, wid, spec, which)           # the stack seen by Stacking has no solution drawn yet
                return slot, token, hit
        im = load_visual(p, wid, spec, which)
        self.cache.put(im)                          # at once: the next request of the worker finds it
        return slot, token, im

    # ------------------------------------------------------------ the final image of a workflow
    def use_renderer(self, worker) -> None:
        """Render the final images on this worker: the project's render worker, the one Post uses, so that the engines
        and the images it keeps are shared (they are gigabytes). Without one the service makes its own at the first
        render and stops it with itself."""
        if worker is None or worker is self._rw:
            return
        if self._rin is not None:                    # under way on the worker let go: its answer never comes here,
            slot = self._rin[0]                      # asked again of this one (unless a newer one waits in its slot)
            self._rq.setdefault(slot, self._rin_req)
            self._rin = None
        self._drop_renderer()
        self._set_renderer(worker, own=False)
        self._pump()

    def _set_renderer(self, worker, own: bool):
        self._rw, self._rw_own = worker, own
        worker.rendered.connect(self._rendered)
        worker.failed.connect(self._render_failed)

    def _drop_renderer(self):
        w, own = self._rw, self._rw_own
        self._rw, self._rw_own = None, False
        if w is None:
            return
        for sig, slot in ((w.rendered, self._rendered), (w.failed, self._render_failed)):
            try:
                sig.disconnect(slot)
            except (RuntimeError, TypeError):
                pass
        if own:
            w.stop()

    def _renderer(self):
        if self._rw is None:
            from .render import RenderWorker, default_device_pref
            m = self.model
            self._set_renderer(RenderWorker(m.p, device=default_device_pref() if m.gpu else "cpu"), own=True)
        return self._rw

    def _rendering(self) -> bool:
        return bool(self._rq) or self._rin is not None

    def _request_render(self, slot: int, wid: int, spec: VisualSpec, which: str) -> int:
        """The final image of a workflow at its default render settings (the suggestion, every slider in the middle):
        what Post shows for that layer, rendered by the render worker at the half resolution. One at a time (the worker
        has one lane for Inspect); the newest request of a slot replaces the one waiting."""
        from ..postengine import new_workflow_layer
        token = self._token(f"inspect.image:{slot}")
        self.requests += 1
        self._rq.pop(slot, None)                       # the one still waiting in this slot is not wanted any more
        v = self.model.view(wid)
        st = self.model.state_info(v, spec.stage) if v is not None else None
        key = None if st is None else (st.previous_key if which == "previous" else (st.key if st.state == "done" else None))
        if not key:
            QTimer.singleShot(0, lambda: self.imageFailed.emit(slot, token, "the result is not computed"))
            return token
        self._rq[slot] = (token, wid, spec, which, (spec.stage, key, spec.name, which),
                          new_workflow_layer(wid, v.name, True))
        self._pump()
        return token

    def _pump(self):
        if self._rin is not None or not self._rq:
            return
        from .render import HALF_LEVEL, Request
        slot = min(self._rq)
        self._rin_req = self._rq.pop(slot)
        token, wid, spec, which, key, layer = self._rin_req
        self._rin = (slot, token, key, wid, spec, which)
        self._rin_seq = self._renderer().request(Request(kind="inspect", layers=[layer], level=HALF_LEVEL, tag="inspect"))

    _rin_seq = None                                  # the render worker's number of the request in ``_rin``

    def _rendered(self, res):
        if res.kind != "inspect" or self._rin is None or res.seq != self._rin_seq:
            return                                   # another request's answer (an older one, another page's)
        slot, token, key, wid, spec, which = self._rin
        self._rin = None
        try:
            if res.rgb is None:
                self.imageFailed.emit(slot, token, "this workflow has no result to show")
            else:
                im = self._final_image(res, key, spec)
                self.cache.put(im)
                self.imageReady.emit(slot, token, im)
        finally:
            self._pump()

    @staticmethod
    def _final_image(res, key: tuple, spec: VisualSpec) -> ShownImage:
        a = res.rgb
        src = PixelSource(a)
        im = ShownImage(key, spec, stretched=src, size=src.size(), hist={"stretched": histogram8(a)},
                        nbytes=src.nbytes)
        cv = res.canvas
        if cv is not None:                 # where the picture sits on the stack grid (the sky readout)
            im.grid, im.origin = (cv.W, cv.H), (float(cv.origin[0]), float(cv.origin[1]))
        return im

    def _render_failed(self, seq: int, kind: str, text: str):
        if kind not in ("inspect", "start") or (kind == "inspect" and seq != self._rin_seq):
            return
        why = text.splitlines()[0] if text else "?"
        if self._rin is not None:
            slot, token = self._rin[:2]
            self._rin = None
            self.imageFailed.emit(slot, token, why)
        self._pump()

    def request_outputs(self, wid: int, stage: str, which: str, key: str):
        """The metrics of a done result (``api.stage_outputs``); answer: ``outputsReady``, then ``outputs_of``."""
        self.requests += 1
        self.images.submit(f"inspect.outputs:{wid}:{stage}:{which}", self._outputs_job, wid, stage, which, key)

    def _outputs_job(self, wid, stage, which, key):
        return wid, stage, which, key, api.stage_outputs(self.model.p, wid, stage, which)

    def outputs_of(self, wid: int, stage: str, which: str, key: str):
        return self.outputs.get((wid, stage, which, key))

    def request_frames(self, wid: int | None):
        """The sessions with their headers and the health checks; answer: ``framesReady``."""
        self.requests += 1
        self.images.submit("inspect.frames", self._frames_job, wid)

    def _frames_job(self, wid):
        return project_frames(self.model.p, wid, self.headers)

    def request_table(self, wid: int) -> int:
        channel = f"inspect.table:{wid}"
        token = self._token(channel)
        self.requests += 1
        self.images.submit(channel, self._table_job, wid, token)
        return token

    def _table_job(self, wid, token):
        return wid, token, frame_table_any(self.model.p, wid)

    def request_preview(self, wid: int, min_transparency: float, max_fwhm_ratio: float) -> int:
        """How many frames these thresholds keep (``api.selection_preview``); answer: ``previewReady``."""
        token = self._token("inspect.preview")
        self.requests += 1
        self.images.submit("inspect.preview", self._preview_job, wid, min_transparency, max_fwhm_ratio, token)
        return token

    def _preview_job(self, wid, tmin, ratio, token):
        t = api.selection_preview(self.model.p, wid, tmin, ratio)
        return token, t.kept, t.total

    def request_group(self, wid: int | None, kind: str, session: int | None = None) -> int:
        """The frames of a kind (``session``: of that session); answer: ``groupReady``."""
        token = self._token("inspect.group")
        self.requests += 1
        self.images.submit("inspect.group", self._group_job, wid, kind, token, session)
        return token

    def _group_job(self, wid, kind, token, session=None):
        if kind == "lights":
            return kind, token, lights_of(self.model.p, wid, self.headers, session)
        return kind, token, browse_group(self.model.p, wid, kind, self.headers, session)

    def _balanced(self, arr: np.ndarray, balance: bool) -> np.ndarray:
        """A raw frame is green (its pixels are not white balanced): the sky of the first light seen is made neutral and the
        same gains serve every light, so that blinking still shows what changes from frame to frame."""
        if not balance:
            return arr
        if self._wb is None:
            flat = arr.reshape(-1, 3)[::7].astype(np.float32)
            med = np.median(flat, axis=0)
            hi = np.percentile(flat, 99.5, axis=0)
            gain = np.clip((hi[1] - med[1]) / np.maximum(hi - med, 1.0), 0.4, 4.0)
            self._wb = (med, gain.astype(np.float32), med[1])
        med, gain, ref = self._wb
        return np.clip(ref + (arr.astype(np.float32) - med) * gain, 0, 255).astype(np.uint8)

    def request_thumbs(self, paths: list[str], size: int = 192, balance: bool = False) -> int:
        """Thumbnails, in this order, the ones already made left out; a new request drops what is not made yet. Each
        answer is a ``thumbReady`` signal."""
        self._thumb_token += 1
        token = self._thumb_token
        todo = [x for x in paths if x not in self.thumb_cache]
        if todo:
            self.requests += 1
            self.thumbs.submit("inspect.thumbs", self._thumbs_job, todo, size, token, balance)
        return token

    def _thumbs_job(self, paths, size, token, balance):
        p = self.model.p
        for path in paths:
            if token != self._thumb_token:
                break
            if path in self.thumb_cache:
                continue
            try:
                arr = self._balanced(api.frame_thumb(p, path, size), balance)
            except Exception:  # noqa: BLE001 - one unreadable frame does not stop the others
                continue
            self.thumb_cache[path] = arr
            while len(self.thumb_cache) > 4000:
                self.thumb_cache.popitem(last=False)
            self.thumbReady.emit(path, arr)
        return None

    def request_stats_batch(self, paths: list[str]):
        """The level and noise of these frames, one after the other in the background; each answer is ``statsReady``."""
        todo = [x for x in paths if x not in self.stats]
        if todo:
            self.requests += 1
            self.prefetch.submit("inspect.statsbatch", self._stats_batch_job, todo)

    def _stats_batch_job(self, paths):
        for path in paths:
            if path in self.stats:
                continue
            try:
                st = frame_stats(path)
            except Exception:  # noqa: BLE001
                continue
            self.stats[path] = st
            self.statsReady.emit(path, st)
        return None

    def request_stats(self, path: str):
        if path in self.stats:
            self.statsReady.emit(path, self.stats[path])
            return
        self.requests += 1
        self._stats_path = path
        self.view.submit("inspect.stats", self._stats_job, path)

    def _stats_job(self, path):
        return path, frame_stats(path)


_services: WeakKeyDictionary = WeakKeyDictionary()


def service_of(model) -> InspectService:
    """The one service of a model (made at the first request)."""
    sv = _services.get(model)
    if sv is None:
        sv = _services[model] = InspectService(model)
    return sv
