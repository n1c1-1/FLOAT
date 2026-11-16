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


# ---------------------------------------------------------------------------------------------- the viewer's sources
# The common viewer (imageview.ImageView) reads its images in its own worker threads: a ``ViewImage`` source may be a
# callable, called there. The callables below read the engine's results (nothing runs in the GUI thread). A result
# that is not there raises ``Unavailable`` with the words the view shows.

def diag(name: str):
    """A diagnostic of the engine (``api.calibration_frames``, ``registration_table``, ``stack_diagnostics``,
    ``color_fit``), or None while the engine does not have it."""
    return getattr(api, name, None)


def block_of(stage: str) -> str:
    """The title of the block a stage belongs to (English; translated where shown)."""
    from ..graphnode import STEPS
    return next((b["title"] for b in STEPS if stage in b["stages"]), stage)


def not_yet(stage: str) -> str:
    """The words for a diagnostic the engine does not give yet, or that this result does not hold (translated)."""
    from ..i18n import _
    return _("To recompute: {block}").format(block=_(block_of(stage)))


_MISSING = re.compile(r"^(.*): missing: recompute (.*) to get it\.?$")


def missing_text(entry: str) -> str:
    """One ``missing`` entry of a diagnostic ("rejection maps: missing: recompute Stacking to get it") in the
    interface's language: "Rejection maps: not in this result. Recompute Stacking to get them." """
    from ..i18n import _
    m = _MISSING.match(str(entry).strip())
    if m is None:
        return _("Not in this result.")
    what, stage = m.group(1).strip(), m.group(2).strip()
    w = _(what)
    return _("{what}: older result · recompute {stage}").format(what=w[:1].upper() + w[1:], stage=_(stage))


def missing_of(obj, word: str = "") -> list[str]:
    """The ``missing`` entries of a diagnostic (those naming ``word``), in the interface's language."""
    return [missing_text(m) for m in (getattr(obj, "missing", None) or []) if word in str(m)]


def not_computed(stage: str) -> str:
    from ..i18n import _
    return _("{block}: not computed here").format(block=_(block_of(stage)))


def _unavailable(text: str):
    from ..imageview import Unavailable
    return Unavailable(text)


def _w(which, stage: str) -> str:
    """``which`` is one word for every stage, or {stage: current | previous} (a picture made of several results)."""
    return which.get(stage, "current") if isinstance(which, dict) else which


def _array(p, wid: int, stage: str, name: str, which) -> np.ndarray:
    which = _w(which, stage)
    try:
        return np.asarray(api.stage_array(p, wid, stage, name, which=which))
    except api.NotDone:
        if api.stage_outputs(p, wid, stage, which) is None:
            raise _unavailable(not_computed(stage)) from None
        raise _unavailable(not_yet(stage)) from None


def array_source(p, wid: int, stage: str, name: str, which: str = "current", channel: int | None = None):
    """An array of a done result (one channel of it with ``channel``)."""
    def load():
        a = _array(p, wid, stage, name, which)
        if channel is not None and a.ndim == 3:
            return a[channel]
        return _one_plane(p, wid, a)
    return load


def _one_plane(p, wid: int, a: np.ndarray) -> np.ndarray:
    """A monochrome sensor's stack is stored as three equal planes (the engine's files are all (3, H, W)): shown as the
    one plane it is, with one column of statistics and no channels."""
    try:
        mono = a.ndim == 3 and a.shape[0] == 3 and p.is_mono(wid)
    except (KeyError, AttributeError):
        return a
    if mono and np.array_equal(a[0, ::7, ::7], a[1, ::7, ::7]) and np.array_equal(a[0, ::7, ::7], a[2, ::7, ::7]):
        return a[:1]
    return a


def preview_source(p, wid: int, stage: str, name: str, which: str = "current"):
    """An engine preview (a JPEG), the ones still in the camera's colours with a neutral sky."""
    def load():
        out = api.stage_outputs(p, wid, stage, _w(which, stage))
        f = out.previews.get(name) if out is not None else None
        if f is None:
            raise _unavailable(not_computed(stage) if out is None else not_yet(stage))
        a = _rgb_of_file(f)
        return neutral_preview(a) if (stage, name) in CAMERA_PREVIEWS else a
    return load


def apply_matrix(M: np.ndarray, a: np.ndarray) -> np.ndarray:
    """The colour matrix of the Color stage on a (3, H, W) image in the camera's colours: linear display RGB."""
    M = np.asarray(M, np.float32)
    out = np.empty_like(a, dtype=np.float32)
    for c in range(3):
        out[c] = M[c, 0] * a[0] + M[c, 1] * a[1] + M[c, 2] * a[2]
    return out


def grey_sky(a: np.ndarray) -> np.ndarray:
    """The sky of a colour image set grey by an offset per channel (its median brought to the mean of the three), as a
    background neutralisation does: the colours of the stars and the nebulae are not touched (no gain)."""
    if a.ndim != 3 or a.shape[0] != 3:
        return a
    med = np.array([float(np.median(a[c, ::4, ::4])) for c in range(3)])
    shift = med.mean() - med
    for c in range(3):
        a[c] += np.float32(shift[c])
    return a


def color_source(p, wid: int, base: tuple[str, str], which: str = "current"):
    """An image of the workflow with the colour matrix of its Color result applied (the colours the stage measured on
    the Gaia stars), its sky set grey (``grey_sky``: the light pollution's colour is the Gradient block's)."""
    def load():
        M = _array(p, wid, "color", "M", which)
        return grey_sky(apply_matrix(M, _array(p, wid, base[0], base[1], which).astype(np.float32, copy=False)))
    return load


def _sky_model(p, wid: int, which: str, H: int, W: int) -> tuple[np.ndarray, np.ndarray]:
    """(the sky variation the Gradient block removes, (3, H, W) in the camera's colours, read back as the stage reads
    it; the sky level it keeps, (3,))."""
    import json

    from ...stages.gradient import INFO_FILE, SKY_FILE, read_grid
    out = api.stage_outputs(p, wid, "gradient", _w(which, "gradient"))
    if out is None:
        raise _unavailable(not_computed("gradient"))
    info = json.loads((Path(out.dir) / INFO_FILE).read_text(encoding="utf-8"))
    return read_grid(np.load(Path(out.dir) / SKY_FILE), info, H, W), np.asarray(info["level"], np.float32)


def calibrated_source(p, wid: int, base: tuple[str, str], which: str = "current"):
    """An image of the workflow in calibrated colours: the colour matrix of its Color result applied, nothing else (its
    sky stays where the step left it: an image whose gradient is removed keeps its zero point)."""
    def load():
        M = _array(p, wid, "color", "M", which)
        return apply_matrix(M, _array(p, wid, base[0], base[1], which).astype(np.float32, copy=False))
    return load


def gradient_source(p, wid: int, stage: str = "gradient", which: str = "current"):
    """The stack with its sky gradient removed, in the camera's colours, as the later blocks read it: the stack minus
    the Gradient block's sky variation (the level stays)."""
    def load():
        stack = _array(p, wid, _stk(p, wid, which), "drz_full", which).astype(np.float32)
        stack -= _sky_model(p, wid, which, *stack.shape[1:])[0]
        return stack
    return load


def removed_source(p, wid: int, stage: str = "gradient", which: str = "current"):
    """What the Gradient block took away: its sky model (the variation, on the level it keeps), in the camera's
    colours."""
    def load():
        stack = _array(p, wid, _stk(p, wid, which), "drz_full", which)
        var, level = _sky_model(p, wid, which, *stack.shape[1:])
        var += level[:, None, None]
        return var
    return load


#: the masters the calibration keeps (arrays of its result): key, file, caption
MASTERS = [("master_dark", "master_dark", "Master dark"), ("master_flat", "master_flat", "Master flat"),
           ("master_bias", "master_bias", "Master bias"), ("master_flat_dark", "master_flat_dark", "Master dark flat"),
           ("badmask", "badmask", "Defective pixels")]

_masters_lock = threading.Lock()
_masters: OrderedDict = OrderedDict()                     # calibrate dir -> (dark, flat, bad), the last two


def _calib_masters(cal_dir: str):
    with _masters_lock:
        hit = _masters.get(cal_dir)
        if hit is not None:
            return hit
    d = Path(cal_dir)
    m = (np.load(d / "master_dark.npy").astype(np.float32), np.load(d / "master_flat.npy").astype(np.float32),
         np.load(d / "badmask.npy").astype(bool))
    with _masters_lock:
        _masters[cal_dir] = m
        while len(_masters) > 2:
            _masters.popitem(last=False)
    return m


def forget_masters(root) -> None:
    """The masters read from a project's results go with it (closed): a gigabyte for a large sensor, kept until the
    program ended."""
    top = os.path.normcase(os.path.abspath(str(root)))
    with _masters_lock:
        for k in [k for k in _masters if os.path.normcase(os.path.abspath(k)).startswith(top + os.sep)]:
            del _masters[k]


def calibrated_frame(p, wid: int, path: str, which: str = "current") -> tuple[np.ndarray, str, str]:
    """One light frame as the calibration made it: (mosaic in ADU above the dark, Bayer pattern, how it was got); a frame
    already in colour: (its image (H, W, 3), no pattern, how it was got). From the calibrated cache when it is on disk,
    else made again from the stage's own masters with its formula ((raw − master dark) / master flat, defective pixels
    left empty)."""
    import json

    from ...io.frames import read_cfa
    from ...stages.calibrate import VALID_MAX
    which = _w(which, "calibrate")
    out = api.stage_outputs(p, wid, "calibrate", which)
    if out is None:
        raise _unavailable(not_computed("calibrate"))
    d = Path(out.dir)
    info = json.loads((d / "cache.json").read_text(encoding="utf-8"))
    cfa = str(info.get("cfa") or "")
    white = float(info.get("white_level") or 65535.0) - float(info.get("dark_level") or 0.0)
    cache = Path(info.get("path") or "")
    n, H, W = int(info["n"]), int(info["height"]), int(info["width"])
    colour = bool(info.get("rgb_path"))                    # frames already in colour (issue #57): their three planes
    if colour:
        cache = Path(info["rgb_path"])
    idx = None
    if cache.is_file() and cache.stat().st_size == n * (3 if colour else 1) * H * W * 2:
        inv = inventory_headers(p, wid, "lights")
        order = list(inv)
        norm = norm_path(path)
        idx = order.index(norm) if norm in order and len(order) == n else None
    if colour:
        return _calibrated_colour(cache, info, d, path, idx, white), "", "cache" if idx is not None else "masters"
    if idx is not None:
        mm = np.memmap(cache, dtype=np.float16, mode="r", shape=(n, H, W))
        x = np.asarray(mm[idx], np.float32)
        sat = x >= VALID_MAX                                   # the saturation marks of the cache
        x /= float(info.get("scale") or 1.0)
        x[sat] = white
        how = "cache"
    else:
        dark, flat, bad = _calib_masters(str(d))
        raw = read_cfa(path).astype(np.float32)
        x = (raw - dark) / flat
        x[bad] = np.nan
        how = "masters"
    return np.nan_to_num(x, nan=0.0, posinf=white, neginf=0.0), cfa, how


def _calibrated_colour(cache: Path, info: dict, d: Path, path: str, idx: int | None, white: float) -> np.ndarray:
    """A frame already in colour as the calibration made it (issue #57): (H, W, 3) in ADU above the dark, from its
    three planes in the cache (``idx``: its rank there), else made again colour by colour from the masters (a joined
    set keeps its masters on the luminance only: the same for the three colours)."""
    from ...io.frames import read_colour
    from ...stages.calibrate import MASTER_RGB, VALID_MAX
    n, H, W = int(info["n"]), int(info["height"]), int(info["width"])
    if idx is not None:
        mm = np.memmap(cache, dtype=np.float16, mode="r", shape=(n, 3, H, W))
        x = np.array(mm[idx], np.float32)
        sat = x >= VALID_MAX
        x /= float(info.get("scale") or 1.0)
        x[sat] = white
    else:
        dark, flat, bad = _calib_masters(str(d))
        if all((d / f).is_file() for f in MASTER_RGB):
            dark, flat = (np.load(d / f).astype(np.float32) for f in MASTER_RGB)
        x = (read_colour(path) - dark) / flat
        x[:, bad] = np.nan
    return np.moveaxis(np.nan_to_num(x, nan=0.0, posinf=white, neginf=0.0), 0, -1)


def calframe_source(p, wid: int, path: str, which: str = "current"):
    def load():
        return calibrated_frame(p, wid, path, which)[0]
    return load


def wcs_source(p, wid: int, which: str = "current"):
    """The astrometric solution of the workflow, read in the viewer's thread (None without one)."""
    def load():
        return _wcs_of(p, wid, _w(which, "astrometry")) or _wcs_of(p, wid, "current")
    return load


class ScaledWcs:
    """The solution of the stack for a reduced, cropped picture of its grid (the final image): pixel (x, y) of the
    picture is grid pixel (ox + (x + 0.5) f − 0.5, oy + (y + 0.5) f − 0.5). Only what the readout and the compare
    ask."""

    def __init__(self, wcs, origin, f: float):
        self.wcs, self.ox, self.oy, self.f = wcs, float(origin[0]), float(origin[1]), float(f)

    def pixel_to_world_values(self, x, y):
        f = self.f
        return self.wcs.pixel_to_world_values(self.ox + (np.asarray(x) + 0.5) * f - 0.5,
                                              self.oy + (np.asarray(y) + 0.5) * f - 0.5)

    def world_to_pixel_values(self, ra, dec):
        gx, gy = self.wcs.world_to_pixel_values(ra, dec)
        f = self.f
        return (np.asarray(gx) + 0.5 - self.ox) / f - 0.5, (np.asarray(gy) + 0.5 - self.oy) / f - 0.5


# ---------------------------------------------------------------------------------------------- where a picture sits
def _stk(p, wid: int, which="current") -> str:
    """The stage whose image the blocks after Stacking read (Chains.resolve): the aligned image when the workflow has an
    Align on block done, else the stack (issue #165)."""
    return "align_on" if api.stage_outputs(p, wid, "align_on", _w(which, "align_on")) is not None else "stack"


def _stack_meta(p, wid: int, which: str, stage: str | None = None) -> tuple[str, float, int, int]:
    """(the registration's key, the drizzle scale, the stack grid's width and height) of a workflow's stack result:
    its grid is the reference frame of that registration enlarged by the scale (stage stack: X = (x + 0.5) s - 0.5)."""
    import json
    stage = stage or _stk(p, wid, which)
    st = api.stage_outputs(p, wid, stage, which) or api.stage_outputs(p, wid, stage)
    if st is None:
        raise LookupError("no stack result")
    meta = json.loads((Path(st.dir) / "drz_meta.json").read_text(encoding="utf-8"))
    reg = (st.manifest.get("inputs") or {}).get("register") or ""
    if stage == "align_on":                           # the reference's grid: a frame of its own, its zone is none
        return str(st.key), float(meta.get("scale") or 1.0), int(meta["width"]), int(meta["height"])
    if not reg:
        out = api.stage_outputs(p, wid, "register", which) or api.stage_outputs(p, wid, "register")
        reg = out.key if out is not None else ""
    W, H = int(meta["width"]), int(meta["height"])
    sensor = meta.get("sensor") or [W, H]
    s = float(meta.get("scale") or (W / float(sensor[0])))
    return str(reg or st.key), s, W, H


def _stack_zone(p, wid: int, which: str, stage: str | None = None) -> tuple[int, int]:
    """Where the stack starts on the reference frame (a Crop block's zone: issue #79), in its pixels."""
    import json
    stage = stage or _stk(p, wid, which)
    if stage == "align_on":
        return 0, 0
    st = api.stage_outputs(p, wid, stage, which)
    meta = json.loads((Path(st.dir) / "drz_meta.json").read_text(encoding="utf-8")) if st is not None else {}
    zone = meta.get("zone") or [0, 0]
    return int(zone[0]), int(zone[1])


#: the blocks in which the Crop block looks for the sky, in pixels of the image it shows
SKY_BLOCK = 32


def crop_frame(p, wid: int) -> dict:
    """What the Crop block shows (issue #79): the reference frame in colour, stretched (``rgb``, ``step`` sensor pixels
    per pixel of it), and where the sky is (``sky``, one value per block of ``SKY_BLOCK`` pixels of ``rgb``)."""
    import json

    from ..stretch import superpixel
    out = api.stage_outputs(p, wid, "register", "current")
    if out is None:
        raise _unavailable(not_computed("register"))
    ref = int(np.load(Path(out.dir) / "reg.npz")["ref"])
    inv = api.stage_outputs(p, wid, "inventory", "current")
    lights = json.loads((Path(inv.dir) / "frames.json").read_text(encoding="utf-8"))["lights"]
    a, cfa, _how = calibrated_frame(p, wid, lights[ref]["path"])
    a = np.nan_to_num(np.asarray(a, np.float32))
    rgb = a if a.ndim == 3 else np.moveaxis(superpixel(a, cfa), 0, 2)
    step = round(a.shape[1] / rgb.shape[1])
    level = np.median(rgb, axis=(0, 1))
    spread = 1.4826 * np.median(np.abs(rgb - level), axis=(0, 1))
    shown = np.arcsinh(np.clip((rgb - level) / np.maximum(spread, 1e-6) + 2.0, 0, None) / 3.0) / 6.0
    grey = rgb.mean(2)
    H, W = grey.shape
    n, m = max(1, H // SKY_BLOCK), max(1, W // SKY_BLOCK)
    blocks = grey[:n * SKY_BLOCK, :m * SKY_BLOCK].reshape(n, -1, m, SKY_BLOCK).mean((1, 3))
    low = float(np.quantile(blocks, .1))
    noise = 1.4826 * float(np.median(np.abs(blocks[blocks <= np.median(blocks)] - low)))
    return {"rgb": (np.clip(shown, 0, 1) * 255).astype(np.uint8), "step": step, "size": (W * step, H * step),
            "sky": blocks <= low + 3 * max(noise, 1e-6)}


def stack_place(p, wid: int, which="current", stage: str | None = None):
    """Where a picture on the stack grid of a workflow sits (``imageview.Placement``), read in the viewer's thread: on
    the grid of the registration's reference frame, at the drizzle scale. Its size (w, h) tells a reduced copy of the
    grid (an engine preview of it) from the grid itself."""
    from ..imageview import Placement
    which = _w(which, "stack")

    def place(w: int, h: int):
        reg, s, W, H = _stack_meta(p, wid, which, stage)
        fx, fy = (W / float(w), H / float(h)) if (w, h) != (W, H) else (1.0, 1.0)
        x0, y0 = _stack_zone(p, wid, which, stage)
        return Placement.scaled(("reference frame", reg), fx / s, float(x0), float(y0), fy=fy / s)
    return place


def reference_frame_place(p, wid: int, which="current"):
    """The reference frame of the registration (calibrated, on the sensor): the grid the stacks are placed on."""
    from ..imageview import Placement
    which = _w(which, "stack")

    def place(w: int, h: int):
        reg, _s, _W, _H = _stack_meta(p, wid, which)
        return Placement(("reference frame", reg))
    return place


def render_place(p, wid: int, stack_which: str, grid, origin, size):
    """Where a final image sits: a part of the stack grid (``grid`` its size there, ``origin`` its corner: the render
    canvas), reduced to ``size`` (the render's half size); then the stack grid's own placement (``stack_which``: the
    stack result the render was made from, current or previous)."""
    from ..imageview import Placement
    on_stack = stack_place(p, wid, stack_which)

    def place(w: int, h: int):
        f = float(grid[0]) / float(size[0] or w)
        fy = float(grid[1]) / float(size[1] or h)
        sx, sy = (float(v) for v in origin)
        _reg, _s, W, H = _stack_meta(p, wid, _w(stack_which, "stack"))
        return Placement.scaled(None, f, sx, sy, fy=fy).then(on_stack(W, H))
    return place


def preview_wcs_source(p, wid: int, which_ast: str, stage: str, name: str, which="current"):
    """The solution of the stack for an engine preview of its grid (a reduced copy: the H-alpha survey), read in the
    viewer's thread: the preview's size against the stack grid's gives the reduction."""
    def load():
        w = _wcs_of(p, wid, which_ast) or _wcs_of(p, wid, "current")
        out = api.stage_outputs(p, wid, stage, _w(which, stage))
        f = out.previews.get(name) if out is not None else None
        if w is None or f is None:
            return None
        from PIL import Image
        with Image.open(f) as im:
            pw = im.size[0]
        _reg, _s, W, _H = _stack_meta(p, wid, _w(which, "stack"))
        return ScaledWcs(w, (0.0, 0.0), W / float(pw))
    return load


def final_wcs_source(p, wid: int, which: str, grid, origin, size):
    """The sky readout of a final image (``grid``: the canvas size on the stack grid, ``size``: the picture's)."""
    def load():
        w = _wcs_of(p, wid, which) or _wcs_of(p, wid, "current")
        if w is None or not grid or not size or not size[0]:
            return None
        return ScaledWcs(w, origin, grid[0] / size[0])
    return load


def light_thumb(p, path: str, cfa: str = ""):
    """The film strip's thumbnail of a light, made from the raw with the viewer's automatic stretch (a neutral
    sky)."""
    from ..imageview import ViewImage, quick_thumbnail

    def load():
        return quick_thumbnail(ViewImage(path, cfa=cfa, camera=True))
    return load


# ---------------------------------------------------------------------------------------------- Send to Post
def post_key(vi, wid: int | None) -> str:
    """The key Post keeps for an image Inspect sends (``PostPage3.add_image_layer``): "<stage>/<store key>/<array>"
    for an array of a result (Post reads it from the store), else "inspect:{...}" or "file:<path>", which
    ``resolve_key`` turns into the source again when the project is opened again."""
    import json
    ident = vi.identity()
    if isinstance(ident, tuple) and ident[:1] == ("file",):
        return "file:" + str(vi.source)
    if isinstance(ident, tuple) and ident[:1] == ("inspect",):
        kind = ident[1]
        if kind == "array" and ident[3] is None and len(ident[5]) == 1:
            stage, key = ident[5][0]
            return f"{stage}/{key}/{ident[2]}"
        if kind == "master":
            return f"calibrate/{ident[2]}/{ident[3]}"
        if kind == "calframe":
            return "inspect:" + json.dumps({"wid": wid, "kind": "calframe", "path": ident[3]})
        if kind in ("array", "preview", "color", "gradient", "removed", "calibrated"):
            return "inspect:" + json.dumps({"wid": wid, "kind": kind, "name": ident[2], "channel": ident[3],
                                            "base": list(ident[4]), "stages": [s for s, _k in ident[5]]})
        return "inspect:" + json.dumps({"wid": wid, "kind": str(kind)})
    return "inspect:" + json.dumps({"wid": wid, "kind": "other", "name": vi.name})


def _debayered(load, cfa: str = ""):
    """A source for Post: a Bayer mosaic made a colour image (one pixel per 2 x 2 photosites)."""
    from ..stretch import superpixel

    def get():
        a = np.asarray(load())
        return superpixel(a, cfa) if a.ndim == 2 and cfa else a
    return get


def post_source(vi, neutral: bool = False):
    """What Post reads for an image of a viewer: its source, read in Post's render worker (a raw or calibrated mosaic
    made a colour image). ``neutral``: the viewer showed it with a neutral sky (still in the camera's colours), so Post
    stretches its channels apart (``PostView.add_image_layer``)."""
    from ..imageview import _read

    def load():
        arr, cfa, _wcs = _read(vi)
        return _debayered(lambda: arr, cfa)()
    load.neutral = bool(neutral)
    return load


def shown_neutral(viewer) -> bool:
    """Whether a viewer shows its image with a neutral sky now (its stretch's Neutral sky)."""
    try:
        st = viewer.stretch()
    except (AttributeError, RuntimeError):
        return False
    return bool(getattr(st, "neutral", False))


def resolve_key(p, key: str):
    """The source of an image layer of Post from its key (``post_key``), when the project is opened again; None when
    it cannot be made again (Post says the image is missing). Called in Post's render worker."""
    import json
    if key.startswith("file:"):
        path = key[5:]
        if not Path(path).is_file():
            return None
        from ..io.frames import read_cfa, read_info
        return _debayered(lambda: read_cfa(path), read_info(path).cfa or "")
    if not key.startswith("inspect:"):
        return None
    try:
        d = json.loads(key[len("inspect:"):])
    except ValueError:
        return None
    wid, kind = d.get("wid"), d.get("kind")
    if wid is None:
        return None
    which = {s: _which_done(p, wid, s) or "current" for s in d.get("stages") or []}
    if kind == "calframe":
        w = _which_done(p, wid, "calibrate") or "current"

        def cal():
            x, cfa, _how = calibrated_frame(p, wid, d["path"], w)
            return _debayered(lambda: x, cfa)()
        return cal
    stage = (d.get("stages") or [""])[0]
    if kind == "array":
        return array_source(p, wid, stage, d.get("name") or "", which, d.get("channel"))
    if kind == "preview":
        return preview_source(p, wid, stage, d.get("name") or "", which)
    if kind == "color":
        return color_source(p, wid, tuple(d.get("base") or ("stack", "drz_full")), which)
    if kind == "gradient":
        return gradient_source(p, wid, stage, which)
    if kind == "calibrated":
        return calibrated_source(p, wid, tuple(d.get("base") or ("background", "astro")), which)
    if kind == "removed":
        return removed_source(p, wid, stage, which)
    if kind in ("rejection_low", "rejection_high"):
        fn = diag("stack_diagnostics")
        w = _which_done(p, wid, "stack") or "current"
        return (lambda: getattr(fn(p, wid, w), kind)) if fn is not None else None
    return None


# ---------------------------------------------------------------------------------------------- the blocks' data
# The engine's diagnostics (``diag``) take ``which``: an Outdated block shows the result it had. Each answer is read
# whole in the worker; what an older result did not keep comes back as None (the view says "recompute ... to get it").

def _try(fn, *a, **k):
    """A diagnostic, or None when the result is not there or predates it."""
    try:
        return fn(*a, **k)
    except (api.NotDone, OSError, ValueError, KeyError, IndexError, TypeError):
        return None


def _which_done(p, wid: int, stage: str) -> str | None:
    """current when the workflow's result of ``stage`` is done, else previous when it had one, else None."""
    for w in ("current", "previous"):
        if api.stage_outputs(p, wid, stage, w) is not None:
            return w
    return None


@dataclass
class CalibrationData:
    """What the Calibration block shows: the masters (key, caption, statistics), the frames of each kind (left-out
    ones included), and how a calibrated light is got. The engine's report (quality flags, the statistics of each
    frame) comes apart: ``calibration_report`` reads every raw frame the first time."""

    masters: list[tuple[str, str, dict]] = field(default_factory=list)
    groups: dict[str, list[FrameItem]] = field(default_factory=dict)
    cache: bool = False                  # the calibrated cache is on disk (else a light is calibrated again)
    cfa: str = ""


def _master_stats(a: np.ndarray) -> dict:
    s = a[::4, ::4].astype(np.float32)
    s = s[np.isfinite(s)]
    if not s.size:
        return {}
    med = float(np.median(s))
    return {"median": med, "noise": float(1.4826 * np.median(np.abs(s - med))), "min": float(s.min()),
            "max": float(s.max())}


def calibration_data(p, wid: int, which: str = "current", headers: dict | None = None) -> CalibrationData:
    """The Calibration block's data (worker thread): the masters with their statistics and the calibration frames of
    each kind (a few files read: the masters' arrays and the folders' listings)."""
    import json
    out = CalibrationData()
    cal = api.stage_outputs(p, wid, "calibrate", which)
    if cal is not None:
        d = Path(cal.dir)
        for key, fname, caption in MASTERS:
            f = d / (fname + ".npy")
            if f.is_file():
                try:
                    a = np.load(f)
                    st = {"bad_pct": 100.0 * float(np.mean(a))} if key == "badmask" else _master_stats(a)
                except (OSError, ValueError):
                    st = {}
                out.masters.append((key, caption, st))
        try:
            info = json.loads((d / "cache.json").read_text(encoding="utf-8"))
            colour = bool(info.get("rgb_path"))           # frames already in colour (issue #57): no mosaic to show
            out.cfa = "RGB" if colour else str(info.get("cfa") or "")
            c = Path(info.get("rgb_path" if colour else "path") or "")
            out.cache = c.is_file() and c.stat().st_size == (3 if colour else 1) * int(info["n"]) * int(info["height"]) * \
                int(info["width"]) * 2
        except (OSError, ValueError, KeyError):
            out.cache = False
    sid = str(which).partition("@")[2] if isinstance(which, str) else ""
    for kind in ("darks", "flats", "bias", "flat_darks"):
        out.groups[kind] = browse_group(p, wid, kind, headers, int(sid) if sid else None)
    return out


@dataclass
class CalibrationReport:
    """The engine's word on each calibration frame (``api.calibration_frames`` of every session), by normalised
    path; its notes. None from ``calibration_report`` while the engine does not give it."""

    frames: dict[str, Any] = field(default_factory=dict)            # path -> api.CalFrame
    notes: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)


def calibration_report(p, wid: int, which: str = "current") -> CalibrationReport | None:
    fn = diag("calibration_frames")
    if fn is None:
        return None
    out = CalibrationReport()
    for s in p.sessions:
        r = _try(fn, p, s.id, wf=wid, which=which)
        if r is None:                                   # Calibration not computed: the frames are measured anyway
            r = _try(fn, p, s.id, wf=wid, which=which, require_done=False)
        if r is None:
            continue
        for cf in getattr(r, "frames", None) or []:
            out.frames[norm_path(cf.path)] = cf
        out.notes += [str(n) for n in (getattr(r, "notes", None) or [])]
        out.missing += [str(n) for n in (getattr(r, "missing", None) or [])]
    return out


def registration_from_table(p, wid: int, which: str, table) -> Any:
    """The charts' data (``charts.RegistrationData``) from the engine's registration table: it reads an Outdated
    result too (``charts.registration_data`` reads the current one)."""
    from ..charts import RegistrationData
    from ..chartview import to_seconds
    rows = list(table)
    aligned = np.array([bool(r.used) for r in rows], bool)
    tr = np.array([float(r.transparency) for r in rows], float)
    ok = tr[aligned & np.isfinite(tr)]
    p90 = float(np.percentile(ok, 90)) if len(ok) else 1.0
    rnd = [r.roundness for r in rows]
    scale = None
    ast = api.stage_outputs(p, wid, "astrometry", which) or api.stage_outputs(p, wid, "astrometry")
    if ast is not None and ast.metrics.get("sensor_scale_arcsec"):
        scale = float(ast.metrics["sensor_scale_arcsec"])
    else:
        inv = inventory_headers(p, wid, "lights")
        first = next(iter(inv.values()), {})
        if first.get("focal_mm") and first.get("pixel_um"):
            scale = 206.265 * float(first["pixel_um"]) / float(first["focal_mm"])
    secs = np.array([np.nan if (s := to_seconds(r.time)) is None else s for r in rows], float)
    ref = next((i for i, r in enumerate(rows) if getattr(r, "reference", False)), 0)
    return RegistrationData(
        names=[r.name for r in rows], iso=[r.time or "" for r in rows], secs=secs,
        fwhm=np.array([float(r.fwhm) for r in rows], float), transp=np.where(aligned, 100 * tr / p90, np.nan),
        nmatch=np.where(aligned, np.array([float(r.nmatch) for r in rows], float), np.nan),
        nstars=np.array([float(r.nstars) if r.nstars else np.nan for r in rows], float),
        rms=np.where(aligned, np.array([float(getattr(r, "rms", np.nan)) for r in rows], float), np.nan),
        aligned=aligned, dx=np.array([float(r.dx) for r in rows], float), dy=np.array([float(r.dy) for r in rows]),
        ref=ref, scale=scale,
        roundness=None if all(x is None for x in rnd) else np.array([np.nan if x is None else float(x) for x in rnd]))


@dataclass
class RegistrationBundle:
    """What the Registration block shows: the charts' data, the engine's per-frame table (``api.registration_table``,
    None until the engine gives it), the frame table (current results only) and the reference frame."""

    data: Any = None                     # charts.RegistrationData
    table: list | None = None            # [api.RegRow]
    frames: Any = None                   # api.FrameTable, or None
    ref_path: str = ""
    ref_name: str = ""
    which: str = "current"


def registration_bundle(p, wid: int) -> RegistrationBundle:
    from ..charts import registration_data
    which = _which_done(p, wid, "register") or "current"
    b = RegistrationBundle(which=which)
    fn = diag("registration_table")
    if fn is not None:
        b.table = _try(fn, p, wid, which)
    b.frames = _try(api.frame_table, p, wid)
    if b.table:
        b.data = registration_from_table(p, wid, which, b.table)
        ref = next((r for r in b.table if getattr(r, "reference", False)), None)
        if ref is not None:
            b.ref_path, b.ref_name = getattr(ref, "path", "") or "", ref.name
    else:
        b.data = registration_data(p, wid)                  # NotDone when it is not current: said by the view
        rows = b.frames.rows if b.frames is not None else []
        if 0 <= b.data.ref < len(rows):
            b.ref_path, b.ref_name = rows[b.data.ref].path, rows[b.data.ref].name
    if b.data is not None and not np.isfinite(b.data.dx).any():
        _offsets_anyway(p, wid, which, b.data)
    return b


def sensor_size(p, wid: int, which: str = "current") -> tuple[float, float] | None:
    """The sensor's size (w, h) from whichever result of the workflow gives it: the calibration's cache description,
    its master dark, the stack's grid description. None when none does."""
    import json
    cal = api.stage_outputs(p, wid, "calibrate", which) or api.stage_outputs(p, wid, "calibrate")
    if cal is not None:
        d = Path(cal.dir)
        try:
            c = json.loads((d / "cache.json").read_text(encoding="utf-8"))
            return float(c["width"]), float(c["height"])
        except (OSError, ValueError, KeyError):
            pass
        try:
            shape = np.load(d / "master_dark.npy", mmap_mode="r").shape
            return float(shape[-1]), float(shape[-2])
        except (OSError, ValueError, IndexError):
            pass
    st = api.stage_outputs(p, wid, "stack", which) or api.stage_outputs(p, wid, "stack")
    if st is not None:
        try:
            s = json.loads((Path(st.dir) / "drz_meta.json").read_text(encoding="utf-8")).get("sensor")
            if s and len(s) == 2:
                return float(s[0]), float(s[1])
        except (OSError, ValueError):
            pass
    return None


def _offsets_anyway(p, wid: int, which: str, data) -> None:
    """The drift of the frames when the registration table could not give it (the sensor size unknown to it: no
    inventory result): made from the registration's polynomials with the sensor size another result gives, as the
    engine makes it (``core.diagnostics.frame_geometry``). Left NaN when no result gives the size."""
    size = sensor_size(p, wid, which)
    out = api.stage_outputs(p, wid, "register", which)
    if size is None or out is None or not (Path(out.dir) / "reg.npz").is_file():
        return
    from ...core.diagnostics import frame_geometry
    with np.load(Path(out.dir) / "reg.npz") as z:
        coef = np.asarray(z["coef"], float)[:data.n]
    if len(coef) != data.n:
        return
    dx, dy, _rot, _sc = frame_geometry(coef, size[0], size[1], np.asarray(data.aligned, bool))
    data.dx, data.dy = dx, dy


def frame_table_any(p, wid: int):
    """The frame table of Selection: the current one (``api.frame_table``), else, for an Outdated registration, the
    one its previous result gives (``api.registration_table``, the stack's weights and reasons when it kept them)."""
    try:
        return api.frame_table(p, wid)
    except api.NotDone:
        pass
    fn = diag("registration_table")
    t = _try(fn, p, wid, "previous") if fn is not None else None
    if not t:
        raise api.NotDone("registration is not computed")
    sd = diag("stack_diagnostics")
    sdiag = _try(sd, p, wid, _which_done(p, wid, "stack") or "previous") if sd is not None else None
    weights = {str(x[0]): x[1] for x in (getattr(sdiag, "weights", None) or [])}
    left = dict(getattr(sdiag, "excluded", None) or {})
    ex = {(int(a), str(b)) for a, b in p.excluded}
    sess = {}
    for s in p.sessions:
        for f in _files(p, s.lights):
            sess.setdefault(f.name, s.id)
    rows = []
    for r in t:
        sid = sess.get(r.name)
        hand = (sid, r.name) in ex
        status = "excluded by hand" if hand else (left.get(r.name) or ("" if r.used else (r.reason or "not aligned")))
        w = weights.get(r.name)
        rows.append(api.FrameRow(r.index, r.path, r.name, sid, r.time, float(r.fwhm), float(r.transparency),
                                 float(getattr(r, "rms", np.nan)), float(w) if w is not None else 0.0, status, hand))
    try:
        mt, mf = api._stack_thresholds(p, wid)
    except Exception:  # noqa: BLE001 - the defaults of the stage
        mt, mf = 0.0, 99.0
    return api.FrameTable(rows, sum(1 for r in rows if not r.status), float(mt), float(mf), "previous")


@dataclass
class StackBundle:
    frames: Any = None                   # api.FrameTable (or the previous one), the weights the stack gave
    diag: Any = None                     # api.StackDiag, None until the engine gives it
    ref_path: str = ""
    ref_name: str = ""
    which: str = "current"


def stack_bundle(p, wid: int) -> StackBundle:
    which = _which_done(p, wid, "stack") or "current"
    b = StackBundle(which=which)
    b.frames = _try(frame_table_any, p, wid)
    fn = diag("stack_diagnostics")
    if fn is not None:
        b.diag = _try(fn, p, wid, which)
    rw = _which_done(p, wid, "register") or "current"
    try:
        with api.stage_array(p, wid, "register", "reg.npz", which=rw) as z:
            ref = int(z["ref"])
    except (api.NotDone, OSError, KeyError, ValueError):
        ref = -1
    lights = list(inventory_headers(p, wid, "lights").values())
    if 0 <= ref < len(lights):
        b.ref_path = str(lights[ref].get("path") or "")
    elif lights:
        b.ref_path = str(lights[len(lights) // 2].get("path") or "")
    b.ref_name = Path(b.ref_path).name if b.ref_path else ""
    return b


@dataclass
class ColorBundle:
    curves: Any = None                   # core.spectral.SpectralResponse


def color_bundle(p, wid: int, which: str = "current") -> ColorBundle:
    """The sensor's curves with their uncertainty (worker thread: about a second of fitting)."""
    return ColorBundle(curves=api.spectral_response(p, wid, which))


def color_fit_of(p, wid: int, which: str = "current"):
    """The per-star colour fit (``api.color_fit``: for an older result it measures the stars again on the stack),
    None while the engine does not give it."""
    fn = diag("color_fit")
    return _try(fn, p, wid, which) if fn is not None else None


def lights_of(p, wid: int | None, headers: dict | None = None, session: int | None = None) -> list[FrameItem]:
    """The light frames (``session``: of that session) with what is known of each: headers, star size, transparency,
    weight, state, roundness. For an Outdated registration or stack, the numbers their previous results give."""
    items = browse_group(p, wid, "lights", headers, session)
    if wid is None:
        return items
    fn = diag("registration_table")
    rw = _which_done(p, wid, "register")
    t = _try(fn, p, wid, rw) if fn is not None and rw else None
    by = {str(r.name): r for r in (t or [])}
    sd = diag("stack_diagnostics")
    sw = _which_done(p, wid, "stack")
    sdiag = _try(sd, p, wid, sw) if sd is not None and sw else None
    weights = {str(x[0]): x[1] for x in (getattr(sdiag, "weights", None) or [])}
    left = dict(getattr(sdiag, "excluded", None) or {})
    for it in items:
        r = by.get(it.name)
        if r is not None:
            if getattr(r, "roundness", None) is not None:
                it.roundness = float(r.roundness)
            if it.fwhm is None:
                it.fwhm, it.transparency = float(r.fwhm), float(r.transparency)
                it.index = r.index
                it.status = "" if r.used else (r.reason or "not aligned")
        if it.weight is None and it.name in weights and weights[it.name] is not None:
            it.weight = float(weights[it.name])
        if not it.status and it.name in left:
            it.status = str(left[it.name])
    return items


def warm_annotations(p) -> None:
    """Load the catalogue of named objects (read from its cache, or made from the shipped files the first time in a
    session: 1 to 3 s) before Sky grid or Object names is first ticked (worker thread; the result is kept in memory)."""
    from ..annotate import field_annotations  # noqa: F401 - its imports are the rest of the first use's cost
    from ...core import skyobjects as so
    try:
        so.load_catalogue()
    except so.CatalogueUnavailable:
        pass                                           # the ticked box says why, as before


def post_annotations(p, wid: int, w: int, h: int, origin=(0, 0)):
    """The sky grid and the named objects of Post's canvas (w x h pixels of the bottom workflow's grid from
    ``origin``): ``annotate.Annotations`` (worker thread)."""
    from ..annotate import Annotations, crop_wcs, field_annotations
    wcs = _wcs_of(p, wid, "current") or _wcs_of(p, wid, "previous")
    if wcs is None:
        return Annotations(why="No astrometric solution for this image.")
    return field_annotations(crop_wcs(wcs, origin), w, h, grid=True, names=True)


def sky_bundle(p, wid: int, which: str, w: int, h: int) -> SkyData | None:
    """The grid and the catalogue stars a picture of the stack shows (None without a solution)."""
    wcs = _wcs_of(p, wid, which) or _wcs_of(p, wid, "current")
    return sky_data(p, wid, which, wcs, w, h)


def final_sky_bundle(p, wid: int, which: str, grid, origin, w: int, h: int) -> SkyData | None:
    """The same for a final image, which is a part of the stack grid (``grid``: its size there, ``origin``: its corner)
    reduced to w x h pixels: the solution is shifted and scaled with it (the grid and the names through the cropped,
    scaled solution, the catalogue stars by the same shift and scale)."""
    from ..annotate import crop_wcs
    wcs = _wcs_of(p, wid, which) or _wcs_of(p, wid, "current")
    if wcs is None or not grid or not w or not h:
        return None
    ox, oy = float(origin[0]), float(origin[1])
    f = float(grid[0]) / float(w)                                  # one factor, as the final image's readout has
    sky = sky_data(p, wid, which, crop_wcs(wcs, origin, ((w, h), tuple(grid))), w, h)
    try:
        s = catalogue_stars(p, wid, which, int(ox + grid[0]) + 1, int(oy + grid[1]) + 1)
    except (OSError, ValueError, KeyError):
        s = None
    if s is not None:
        s = s[(s[:, 0] >= ox) & (s[:, 1] >= oy)].copy()
        s[:, 0] = (s[:, 0] + 0.5 - ox) / f - 0.5
        s[:, 1] = (s[:, 1] + 0.5 - oy) / f - 0.5
        s = s[(s[:, 0] >= 0) & (s[:, 0] < w) & (s[:, 1] >= 0) & (s[:, 1] < h)]
    sky.stars = s
    return sky


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
