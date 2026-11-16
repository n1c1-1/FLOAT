"""Inspect's sessions and frames: the header readings of a session, the health checks of its calibration,
the frames of a group, their statistics."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ... import api
from .worker import norm_path


# ---------------------------------------------------------------------------------------------- sessions and frames
@dataclass
class SessionRow:
    id: int
    name: str
    lights: int = 0
    darks: int = 0
    flats: int = 0
    bias: int = 0
    flat_darks: int = 0
    first: str = ""              # time of the first light, ISO
    last: str = ""
    exposure: float | None = None
    iso: float | None = None
    gain: bool = False           # ``iso`` is a camera gain (FITS frames of an astro camera), said "Gain", not "ISO"
    temperature: float | None = None
    filter: str = ""
    fwhm: float | None = None    # median star size of its frames, once Registration is done
    kept: int | None = None      # lights the stack takes (None: not known yet)
    #: its own calibration frames, by kind (darks, flats, bias), as the middle one of them says (``_calib_of``):
    #: exposure, iso, time, file, median and low (0.01 % percentile) levels in ADU, stacked (a master: its frames)
    calib: dict = field(default_factory=dict)
    #: {"exposure", "iso"} of its lights (a sample: io/frames.light_facts) and of each of its darks, which the
    #: Calibration groups by exposure to give each light the dark of its own (stages/calibrate.match_darks)
    light_keys: list = field(default_factory=list)
    dark_keys: list = field(default_factory=list)
    bias_keys: list = field(default_factory=list)      # the same for its bias and its flats (``long_bias``)
    flat_keys: list = field(default_factory=list)
    #: its frames' own date against the one their file names say, when they disagree (io/frames.clock_disagreement:
    #: {"camera", "names"} ISO dates), from the first and last lights' headers read for ``first`` and ``last`` when
    #: the date is their own (EXIF, DATE-OBS), never the file's date on disk
    clock: dict | None = None
    #: the nights its lights were taken in (``frames.nights_of``: ISO days, first to last) when its headers give their times
    nights: list = field(default_factory=list)
    #: its lights' header pointing far from the project's target (``api.target_far``): {"sep", "target", "object"}, or None
    target_far: dict | None = None


@dataclass
class ProjectFrames:
    """The sessions of a project with what their headers say, and the health checks (worker thread)."""

    sessions: list[SessionRow] = field(default_factory=list)
    checks: list = field(default_factory=list)                      # api.Check
    camera: dict[str, Any] = field(default_factory=dict)           # camera, width, height, cfa, black, white
    flat_pct: float | None = None                                   # median level of a flat, % of saturation
    dark_level: float | None = None                                 # median level of a dark (ADU)
    darks: dict[str, Any] = field(default_factory=dict)            # exposure, iso of the first dark
    flats: dict[str, Any] = field(default_factory=dict)
    folders: dict[str, list[str]] = field(default_factory=dict)    # kind -> folders as the sessions list them
    counts: dict[str, int] = field(default_factory=dict)           # kind -> files in the project
    unreadable: list[tuple[str, str]] = field(default_factory=list)  # (session, file name) of lights that cannot be read


def _files(p, entries) -> list[Path]:
    from ...core.project import list_frames
    return list_frames([p.abspath(x) for x in entries])


def _levels(path) -> tuple[float, float]:
    """(median, lowest levels) of a frame in ADU: its 0.01 % percentile, a few pixels in a million (stages/calibrate:
    ``LOW_PCT``), on one sample in 64."""
    from ...io.frames import read_cfa
    from ...stages.calibrate import LOW_PCT
    a = read_cfa(path)[::8, ::8]
    return float(np.median(a)), float(np.percentile(a, LOW_PCT))


def _calib_of(files: list, hdr: dict) -> dict | None:
    """What the middle one of a session's calibration frames of a kind says: its header (exposure, ISO, a master) and
    its levels (one decode each, remembered in ``hdr`` by path: darks shared by sessions are read once)."""
    from ...io.frames import read_info
    if not files:
        return None
    mid = files[len(files) // 2]
    k = ("calib", str(mid))
    if k not in hdr:
        info = read_info(mid)
        med, low = _levels(mid)
        hdr[k] = {"exposure": info.exposure, "iso": info.iso, "time": info.time, "file": mid.name, "level": med,
                  "low": low, "stacked": info.stacked, "black": info.black, "white": info.white,
                  "time_from_file": info.time_from_file, "gain": info.iso_is_gain}
    return hdr[k]


def _keys_of(files: list, hdr: dict, limit: int = 400) -> list[dict]:
    """{"exposure", "iso"} of each of a session's calibration frames of a kind (headers only, remembered in ``hdr`` by
    path; at most ``limit``, evenly spread)."""
    from ...io.frames import read_info
    if len(files) > limit:
        files = [files[i] for i in sorted({int(i) for i in np.linspace(0, len(files) - 1, limit)})]
    out = []
    for f in files:
        k = ("key", str(f))
        if k not in hdr:
            try:
                info = read_info(f)
                hdr[k] = {"exposure": info.exposure, "iso": info.iso, "filter": info.filter}
            except Exception:  # noqa: BLE001 - an unreadable frame is said by the import, not here
                hdr[k] = None
        if hdr[k] is not None:
            out.append(hdr[k])
    return out


def dark_lines(row: SessionRow, camera: dict) -> list[dict]:
    """Which dark the Calibration gives the lights of each exposure (stages/calibrate.dark_plan, its own rule): one
    {"lights": exposure s, "gain", "how": "same" | "scaled" | "none", "dark": exposure s of the darks used or None,
    "others": the exposures of the darks of the lights' gain, "several": the lights or the darks are of several
    exposures} a group of lights, when they are, or when the lights get no dark of theirs; [] else (the one line of
    the darks says it)."""
    from ...stages.calibrate import _key, dark_plan, exposure_groups
    if not row.light_keys or not row.dark_keys:
        return []
    lg = exposure_groups([_key(f) for f in row.light_keys])
    dg = exposure_groups([_key(f) for f in row.dark_keys])
    plan = dark_plan(lg, dg, bool(row.bias) or float(camera.get("black") or 0) > 0)
    if len(lg) < 2 and len(dg) < 2 and plan[0][0] == "same":
        return []
    out = []
    for L, (how, j) in sorted(zip(lg, plan), key=lambda x: -(x[0]["exposure"] or 0)):
        others = sorted({D["exposure"] for D in dg if D["exposure"] and (L["gain"] is None or D["gain"] is None
                                                                          or D["gain"] == L["gain"])}, reverse=True)
        out.append({"lights": L["exposure"], "gain": L["gain"], "how": how, "others": others,
                    "dark": dg[j]["exposure"] if j is not None else None, "several": len(lg) > 1 or len(dg) > 1})
    return out


def flat_pct(c: dict, camera: dict) -> float:
    """A flat's level as % of the sensor's range: above the camera's black level (a master flat, already calibrated:
    above zero), over the range of the lights' sensor (a master in floats is read up to 65535 ADU, a 14-bit camera
    saturates at 16383)."""
    black, white = float(camera.get("black") or c["black"] or 0), float(camera.get("white") or c["white"] or 0)
    zero = 0.0 if c.get("stacked") is not None else black
    return 100 * (c["level"] - zero) / max(white - black, 1.0)


def calib_checks(row: SessionRow, camera: dict) -> dict:
    """What a session's calibration frames contradict, by the Calibration's own rules (stages/calibrate): "dark_far"
    (darks far under the bias, or the black level: their bias was taken off) and "flat_under" (flats that are not a
    master go under the level they will be corrected with: their bias was taken off already), each with the reference
    level {"ref": ADU, "what": "bias" | "black level" | "dark level"}; {} for none. "darks_by_exposure": which dark the
    lights of each exposure get (``dark_lines``), when the lights or the darks are of several exposures;
    "mixed_exposures": the exposure times of lights of several (information)."""
    from ...stages.calibrate import dark_far, flat_under, long_bias, mixed_exposures
    out: dict = {}
    lb = long_bias(row.bias_keys, row.flat_keys)
    if lb:
        out["long_bias"] = lb
    mixed = mixed_exposures(row.light_keys)
    if mixed:
        out["mixed_exposures"] = mixed
    by = dark_lines(row, camera)
    if by:
        out["darks_by_exposure"] = by
    bias, black = row.calib.get("bias"), float(camera.get("black") or 0)
    ref, what = (bias["level"], "bias") if bias else (black, "black level")
    d = row.calib.get("darks")
    if d and dark_far(d["level"], ref):
        out["dark_far"] = {"ref": ref, "what": what}
    f = row.calib.get("flats")
    if not bias and d:                     # without bias the flats are corrected with the darks' level (master_flat)
        ref, what = d["level"], "dark level"
    if f and f.get("stacked") is None and not row.flat_darks and f["level"] > ref \
            and flat_under((f["low"] - ref) / (f["level"] - ref)):
        out["flat_under"] = {"ref": ref, "what": what}
    return out


def _first_readable(files: list, like=None, bad: list | None = None, tries: int = 8):
    """The header of the first of ``files`` that can be read (``bad`` gets the names of those that cannot; ``tries``
    of them at most), or None."""
    from ...io.frames import read_info
    for f in files[:tries]:
        try:
            return read_info(f, like=like) if like is not None else read_info(f)
        except Exception:  # noqa: BLE001 - a damaged or foreign file: said in the checks (exploration 6.10)
            if bad is not None and f.name not in bad:
                bad.append(f.name)
    return None


def project_frames(p, wid: int | None, headers: dict | None = None) -> ProjectFrames:
    """The sessions, headers, star sizes and health checks of a project. ``headers`` caches the headers read (a dict
    owned by the service). A light that cannot be read never stops it (exploration 6.10: the checks stayed on "Reading
    the frames…"): it is said in the checks; the files the engine does not read (a JPEG) are not counted as kept."""
    from ...core.project import colour_lights, frame_kind
    from ...io.frames import clock_disagreement, light_facts
    from ..frames import nights_of
    hdr = headers if headers is not None else {}
    out = ProjectFrames()
    table = None
    if wid is not None:
        try:
            table = api.frame_table(p, wid)
        except (api.NotDone, OSError, ValueError, KeyError):
            table = None
    by_session: dict[int, list] = {}
    if table is not None:
        for r in table.rows:
            by_session.setdefault(r.session, []).append(r)
    first_info = None
    points: dict[int, dict] = {}                                  # what the lights' headers say of where, by session
    dates: dict[int, list[tuple[str, str]]] = {}      # the lights' own dates read here, for the health checks
    for s in p.sessions:
        row = SessionRow(s.id, s.name, filter=s.filter)
        lights = _files(p, s.lights)
        row.lights = len(lights)
        colour = colour_lights(lights)                            # issue #57: JPEGs only when they are the lights
        frames = [f for f in lights if frame_kind(f) and (colour or frame_kind(f) != "colour")]   # what the engine reads
        row.darks, row.flats = len(_files(p, s.darks)), len(_files(p, s.flats))
        row.bias, row.flat_darks = len(_files(p, s.bias)), len(_files(p, s.flat_darks))
        for kind in ("lights", "darks", "flats", "bias", "flat_darks"):
            out.counts[kind] = out.counts.get(kind, 0) + getattr(row, kind)
        if frames:
            k = (str(frames[0]), str(frames[-1]), len(frames))
            if k not in hdr:
                bad: list[str] = []
                a = _first_readable(frames, bad=bad)
                b = (_first_readable(frames[::-1], like=a, bad=bad) if a is not None and len(frames) > 1 else a)
                hdr[k] = (a, b, bad)
            a, b, bad = hdr[k]
            out.unreadable += [(s.name, n) for n in bad]
            if a is not None:
                first_info = first_info or a
                row.exposure, row.iso, row.temperature = a.exposure, a.iso, a.temperature
                row.gain = a.iso_is_gain
                times = [x for x in (a.time, b.time if b is not None else None) if x]
                row.first, row.last = (min(times), max(times)) if times else ("", "")
                dates[s.id] = [(x.path, x.time) for x in (a, b)        # the file's date on disk is not a clock
                               if x is not None and x.time and not x.time_from_file]
                row.clock = clock_disagreement(dates[s.id])
                facts = light_facts(frames)
                row.nights = nights_of(facts.get("times") or [])
                row.light_keys = facts.get("keys") or []
                points[s.id] = facts
        rows = by_session.get(s.id)
        if rows:
            fw = [r.fwhm for r in rows if np.isfinite(r.fwhm)]
            row.fwhm = float(np.median(fw)) if fw else None
            row.kept = sum(1 for r in rows if not r.status)
        else:
            ex = sum(1 for sid, _n in p.excluded if sid == s.id)
            row.kept = max(len(frames) - ex, 0)
        for kind in ("darks", "flats", "bias"):           # its own, as the Calibration of the session reads them
            try:
                c = _calib_of(_files(p, getattr(s, kind)), hdr)
            except Exception:  # noqa: BLE001 - a hint: an unreadable frame is reported by the import, not here
                c = None
            if c is not None:
                row.calib[kind] = c
        try:
            row.dark_keys = _keys_of(_files(p, s.darks), hdr)
        except Exception:  # noqa: BLE001 - a hint, as the levels above
            row.dark_keys = []
        try:
            row.bias_keys = _keys_of(_files(p, s.bias), hdr)
            row.flat_keys = _keys_of(_files(p, s.flats), hdr)
        except Exception:  # noqa: BLE001 - a hint, as the levels above
            row.bias_keys = row.flat_keys = []
        out.sessions.append(row)
    far = {x["sid"]: x for x in api.target_far(p, points)}
    for row in out.sessions:
        row.target_far = far.get(row.id)
    if first_info is not None:
        out.camera = dict(camera=first_info.camera, width=first_info.width, height=first_info.height,
                          cfa=first_info.cfa, black=first_info.black, white=first_info.white)
    for kind in ("darks", "flats", "bias", "flat_darks"):
        seen = []
        for s in p.sessions:
            seen += [p.abspath(x) for x in getattr(s, kind)]
        out.folders[kind] = list(dict.fromkeys(seen))
    # the project's: those of the first session that has some (each session's own are in its row, ``calib``)
    for kind in ("darks", "flats"):
        c = next((r.calib[kind] for r in out.sessions if kind in r.calib), None)
        if c is None:
            continue
        d = {k: c[k] for k in ("exposure", "iso", "time", "file")}
        if kind == "darks":
            out.dark_level, out.darks = c["level"], d
        else:
            out.flat_pct, out.flats = flat_pct(c, out.camera), d
    try:
        out.checks = api.health_checks(p, dates)
    except Exception:  # noqa: BLE001
        out.checks = []
    out.checks[:0] = [api.Check("warn", "unreadable", f"{name}: {f} cannot be read", {"session": name, "name": f})
                      for name, f in out.unreadable]      # 6.10: said, never a silent stop
    out.checks = [c for c in out.checks if not (c.key == "ok" and any(x.level == "warn" for x in out.checks))]
    return out


@dataclass
class FrameItem:
    """One frame of a group of the browser."""

    path: str
    name: str
    kind: str                         # lights | darks | flats | bias | flat_darks
    session: int | None = None
    time: str = ""                    # ISO, or the file's date for a calibration frame
    fwhm: float | None = None
    transparency: float | None = None
    weight: float | None = None
    status: str = ""                  # '' kept, else the reason the stack leaves it out
    excluded: bool = False            # by hand
    index: int | None = None          # in the frame table
    level: float | None = None        # median level above black, ADU (lights: from the calibration statistics)
    exposure: float | None = None     # what the header says (the Inventory result, else the file)
    iso: float | None = None
    temperature: float | None = None
    cfa: str = ""
    white: float | None = None
    black: float | None = None
    roundness: float | None = None    # once the engine's registration table gives it
    stats: dict = field(default_factory=dict)        # a calibration frame: the engine's statistics (diagnostics)
    flag: str = ""                    # a calibration frame: the engine's quality flag, and why
    reason: str = ""


def browse_group(p, wid: int | None, kind: str, headers: dict | None = None, session: int | None = None
                 ) -> list[FrameItem]:
    """The frames of a group, in name order, with what is known of each (worker thread). Lights come with their star
    size, transparency and weight once Registration is done. ``session``: that session's frames only (issue #43: a
    session's Calibration block)."""
    import datetime as dt

    from ...io.frames import read_info
    hdr = headers if headers is not None else {}
    by_path: dict[str, Any] = {}
    if kind == "lights" and wid is not None:
        try:
            t = api.frame_table(p, wid)
            by_path = {norm_path(r.path): r for r in t.rows}
        except (api.NotDone, OSError, ValueError, KeyError):
            by_path = {}
    items: list[FrameItem] = []
    levels = None
    if by_path:                                    # the sky level of each frame, measured at calibration
        cal = api.stage_outputs(p, wid, "calibrate")
        f = Path(cal.dir) / "frame_stats.npy" if cal is not None else None
        if f is not None and f.is_file():
            levels = np.load(f)[:, :, 0].mean(1)
    heads = inventory_headers(p, wid, kind)
    like = None
    excluded = {(a, b) for a, b in p.excluded}
    listed = []
    for s in p.sessions:
        if session is not None and s.id != session:
            continue
        # a Calibration placed in parallel lists its own files
        for f in _files(p, getattr(p.own_session(wid, s.id) if wid is not None else s, kind)):
            listed.append((s.id, f, False))
    if kind != "lights":                           # a calibration frame left out: still shown, to take it back
        seen = {norm_path(f) for _sid, f, _x in listed}
        own = (p.workflow(wid).calibration or {}) if wid is not None else {}
        for sid, f in calibration_left_out(p, kind, seen):
            if str(sid) not in own:                # its own frames are taken out of its list, not left out
                listed.append((sid, f, True))
    for sid, f, left_out in listed:
        norm = norm_path(f)
        it = FrameItem(str(f), f.name, kind, sid)
        h = heads.get(norm)
        if h is None and kind == "lights" and not by_path:
            k = str(f)
            if k not in hdr:
                try:
                    info = read_info(f, like=like)
                    like = like or info
                    hdr[k] = {"time": info.time or "", "exposure": info.exposure, "iso": info.iso,
                              "temperature": info.temperature, "cfa": info.cfa, "white": info.white,
                              "black": info.black}
                except Exception:  # noqa: BLE001
                    hdr[k] = {}
            h = hdr[k]
        if h:
            it.time = h.get("time") or ""
            it.exposure, it.iso, it.temperature = h.get("exposure"), h.get("iso"), h.get("temperature")
            it.cfa, it.white, it.black = h.get("cfa") or "", h.get("white"), h.get("black")
        r = by_path.get(norm)
        if r is not None:
            it.time = r.time or it.time
            it.fwhm, it.transparency, it.weight = r.fwhm, r.transparency, r.weight
            it.status, it.index = r.status, r.index
            if levels is not None and r.index < len(levels):
                it.level = float(levels[r.index])
        if not it.time and kind != "lights":
            # the header's own date when it has one (a master flat of 2022 is not "taken" on the day it was copied);
            # else the file's date on disk
            k = ("date", str(f))
            if k not in hdr:
                try:
                    info = read_info(f, like=like)
                    hdr[k] = "" if info.time_from_file else (info.time or "")
                except Exception:  # noqa: BLE001 - a frame that cannot be read has no date of its own
                    hdr[k] = ""
            it.time = hdr[k]
        if not it.time and kind != "lights":
            try:
                it.time = dt.datetime.fromtimestamp(f.stat().st_mtime).isoformat(timespec="seconds")
            except OSError:
                it.time = ""
        it.excluded = (sid, f.name) in excluded if kind == "lights" else left_out
        items.append(it)
    if kind != "lights":
        items.sort(key=lambda x: (x.name, x.path))
    return items


def inventory_headers(p, wid: int | None, kind: str) -> dict[str, dict]:
    """What the Inventory result read in the headers of the frames of a kind (time, exposure, ISO, temperature,
    Bayer pattern, black and white levels), by normalised path; {} when it is not computed."""
    import json
    inv = api.stage_outputs(p, wid, "inventory") if wid is not None else None
    if inv is None:
        inv = api.stage_outputs(p, wid, "inventory", "previous") if wid is not None else None
    if inv is None:
        return {}
    try:
        fj = json.loads((Path(inv.dir) / "frames.json").read_text(encoding="utf-8")).get(kind) or []
    except (OSError, ValueError):
        return {}
    return {norm_path(f["path"]): f for f in fj if f.get("path")}


def calibration_left_out(p, kind: str, seen: set[str]) -> list[tuple[int, Path]]:
    """The calibration frames of a kind left out by hand (``ProjectModel.set_calibration_excluded``) that no session
    lists and that are still on the disk. (session id, path)."""
    out = []
    for s in p.sessions:
        for x in s.left_out.get(kind, []):
            f = Path(p.abspath(x))
            if norm_path(f) not in seen and f.is_file():
                seen.add(norm_path(f))
                out.append((s.id, f))
    return out


def set_calibration_excluded(model, item: FrameItem, excluded: bool, group: list[FrameItem]) -> bool:
    """Leave a calibration frame out of its master (or take it back): ``ProjectModel.set_calibration_excluded``, given
    the frames of the item's folder as the worker listed them (``group``: the kind's listing, left-out frames
    included), so that the GUI thread reads no folder. False: its session is not known."""
    if item.session is None:
        return False
    here = norm_path(Path(item.path).parent)
    folder = [(g.path, g.excluded) for g in group if g.session == item.session
              and norm_path(Path(g.path).parent) == here]
    return model.set_calibration_excluded(item.session, item.kind, item.path, excluded, folder=folder)


def frame_stats(path: str) -> dict[str, float]:
    """The median level above black and the noise of one frame (ADU): the decode is the cost (worker thread)."""
    from ...io.frames import read_cfa, read_info
    info = read_info(path)
    a = read_cfa(path)[::4, ::4]
    med = float(np.median(a))
    noise = float(1.4826 * np.median(np.abs(a - med)))
    return {"level": med - info.black, "noise": noise, "pct": 100 * (med - info.black) / max(info.white - info.black, 1.0),
            "black": info.black, "white": info.white, "median": med, "stacked": info.stacked}
