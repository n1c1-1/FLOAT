"""Inspect's data of the blocks, read in the worker: calibration, registration, stacking, colour, the
lights, the sky over the stack and the final image."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ... import api
from .worker import norm_path
from .inspect_display import SkyData, _wcs_of, catalogue_stars, sky_data
from .inspect_frames import _files, FrameItem, browse_group, inventory_headers
from .inspect_sources import diag, MASTERS, _try, _which_done


# ---------------------------------------------------------------------------------------------- the blocks' data
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
