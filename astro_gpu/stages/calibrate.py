"""Calibration of the raw Bayer frames (no debayering): master dark / bias / flat, bad pixel map, calibrated cache."""
from __future__ import annotations

import json
import hashlib
import os
import shutil
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

import cv2
import numpy as np

from ..core import cfa as cfa_mod
from ..core import devices
from ..core.imgutil import robust_sigma
from ..core.params import ADVANCED, Auto, P, StageParams
from ..core.stage import Context, Stage
from ..io.frames import read_cfa, read_colour

SAT_MARK = 65000.0      # value written for saturated pixels in the cache
VALID_MAX = 60000.0     # anything above is treated as saturated downstream
MIN_USABLE = 0.5        # the share of a frame's samples below which it has no statistics (issue #64)


#: how the hot pixels are found in the lights: "persistence" (the pixels that stand out in the lower quartile of the
#: session's frames) or "per_frame" (Siril's cosmetic correction without darks, on each light)
HOT_METHODS = ("persistence", "per_frame")
#: Siril's cosmetic correction, "Hot Sigma" (src/gui-gtk4/uifiles/cosmetic_dialog.ui, adjSigCosmeHot, value 3) and
#: "Amount" 1 (adjCosmeAmount): the defaults of the dialog, of the documentation's "use the default settings"
SIRIL_HOT_SIGMA = 3.0


class CalibrateParams(StageParams):
    hot_sigma: float = P(0.6, label="Hot pixel threshold", level=ADVANCED, ge=0.1, le=10, step=0.1, unit="x sky noise",
                         help="A pixel whose dark current exceeds this fraction of the sky noise of a light frame is "
                              "excluded from the stack.",
                         tip="Low values (0.5-1) exclude more pixels; with many frames that costs nothing because "
                             "the drizzle fills the gaps. Raise it if more than ~5 % of the pixels get excluded.")
    flat_defect: float = P(0.3, label="Flat defect threshold", level=ADVANCED, ge=0.05, le=0.9, step=0.05,
                           help="Pixels whose flat response differs from their neighbourhood by more than this fraction "
                                "(dust specks, dead pixels) are excluded.",
                           tip="0.3 = 30 %. Lower it if dust donuts remain visible after calibration.")
    hot_from_lights: bool = P(True, label="Hot pixels (lights)", level=ADVANCED,
                              help="Also flag the pixels that stand out from their neighbours in the light frames "
                                   "themselves, besides the darks'. Darks taken at another temperature miss pixels "
                                   "that are hot only when the sensor is warmer; in the stack they draw the drift "
                                   "path of the mount. Each flagged pixel is replaced by the mean of its "
                                   "same-colour neighbours.",
                              tip="The method is chosen by Hot pixels.")
    hot_method: str = P("persistence", label="Hot pixels", level=ADVANCED, choices=list(HOT_METHODS),
                        help="How the hot pixels are found in the lights. Persistence: a pixel that stands out from "
                             "its neighbours in the lower quartile of 40 frames spread over the session (a defect is "
                             "bright in every frame; a star covers a given pixel in some frames only). Per frame: "
                             "Siril's cosmetic correction without darks, applied to each light "
                             "(https://siril.readthedocs.io/en/stable/processing/cc.html).",
                        tip="Persistence needs 15 frames or more. Per frame works on any session; it needs a Bayer "
                            "sensor.")
    flat_scale_px: float = P(0.0, label="Flat scale", level=ADVANCED, ge=0, le=500, step=5, unit="px",
                             help="Keep only the scales of the master flat larger than this (Gaussian sigma, sensor "
                                  "pixels). 0 = the whole flat. For flat frames borrowed from another night of the "
                                  "same optics: they hold the vignetting of the instrument, but not this night's dust.",
                             tip="About 40 px removes dust donuts and keeps the vignetting.")
    saturation: float = P(0.976, label="Saturation level", level=ADVANCED, ge=0.5, le=1.0, step=0.001, unit="x white level",
                          help="Raw values above this fraction of the sensor white level are flagged as saturated; the "
                               "stacker then treats them specially instead of averaging clipped values.",
                          tip="Keep the default unless star cores look flat-topped or rings appear around bright stars.")
    apply_masters: bool = P(True, label="Apply calibration", level=ADVANCED,
                            help="Off when the Calibration block is cut (issue #43): the frames are taken as they are "
                                 "(already calibrated), without master dark, flat or defective pixels.")
    workers: int = P(8, label="Parallel workers", level=ADVANCED, ge=1, le=64,
                     help="Number of CPU processes used to calibrate frames in parallel.",
                     tip="Defaults to the number of CPU cores minus two (max 12).")


class Calibrate(Stage):
    name = "calibrate"
    title = "Calibration"
    description = ("Builds the master dark, bias and flat, finds hot, cold and dusty pixels, then subtracts the dark "
                   "and divides by the flat for every light frame. Frames stay as raw Bayer mosaics (no debayering): "
                   "colour is reconstructed later by the drizzle, without interpolation. The calibrated frames are "
                   "cached on the fast scratch disk; an interrupted calibration resumes frame by frame.")
    deps = ("inventory",)
    Params = CalibrateParams
    # 2: hot pixels also seen in the lights; 3: a bad pixel stays excluded when saturated; 4: hot pixels seen in the
    # lights when the sky did not move too (a lone pixel above its neighbours of every colour: m8 kept 45-57 coloured
    # dots per square degree), and then filled from their same-colour neighbours (left out, they were holes). The
    # master bias and flat dark (EXTRA_MASTERS, for the Inspect mode) are extra files: an older result lacks them. 5: a
    # FITS mosaic in floats on the 0-1 scale read in ADU (io/frames.fits_adu): a master dark of another program was read
    # as 0.008 for 508 ADU, so the lights kept their black level, which the flat then turned into an inverted vignetting.
    # 6: a master given (io/frames.master_count) is used as it is: a master flat had the bias taken off a second time
    # 7: each light is calibrated with the dark of its exposure (``match_darks``): a session of lights of 300 and 60 s
    # with a master dark of each got their mean, of 180 s, scaled by 1.667 with the camera's offset (no bias, a black
    # level of 0): every calibrated sky below 0. A dark of another exposure is scaled only above a bias or a known
    # black level, else the lights go without dark, said (scaled with the offset before). A session of one exposure
    # with darks of its exposure gets the same frames, to the byte. 8: a flat with no bias nor flat darks loses the black
    # level only, no longer the level of a dark (its thermal signal: a 300 s dark's 848 ADU taken off a 1 s flat)
    # 9: the hot pixels of the lights: the persistence's movement test and lone-pixel condition taken out (a saturated
    # defect lights its four nearest neighbours by crosstalk, which the condition took for a star), Siril's cosmetic
    # correction per frame as the other method, and every excluded pixel filled with the mean of its same-colour
    # neighbours (the sky moved or not). 10: the persistence's threshold is in sigmas of one frame's noise, no longer of
    # the lower quartile's residual (a third of it): a Sony A7S session at ISO 4000 had 178,000 pixels flagged, 110,000
    # of them within 2 to 5 sigmas of a frame
    # 11: the persistence leaves the pixels under a star of the lower quartile to the stack's rejection (the frames
    # hardly moved there: a Sony A7S series flagged cores of its bright stars, 10 to 26 % of their flux)
    version = "11"

    def auto(self, ctx: Context) -> dict[str, Auto]:
        n = max(1, min(12, (os.cpu_count() or 4) - 2))
        return {"workers": Auto(n, "{cores} CPU cores detected", {"cores": os.cpu_count()})}

    def run(self, ctx: Context, p: CalibrateParams) -> None:
        frames = ctx.load_json("inventory", "frames.json")
        if cfa_mod.is_colour(ctx.metric_of("inventory", "cfa")):       # frames already in colour (issue #57)
            (ctx.dir / MASTERS).unlink(missing_ok=True)
            calibrate_colour(ctx, p, frames)
            return
        given = given_masters(ctx, p)
        if given is not None:                # the masters of an earlier result: rebuild the calibrated cache only
            rebuild_cache(ctx, p, frames["lights"], given, ctx.metric_of("inventory", "cfa"),
                          float(ctx.metric_of("inventory", "white_level")))
            return
        H, W = int(ctx.metric_of("inventory", "height")), int(ctx.metric_of("inventory", "width"))
        black = float(ctx.metric_of("inventory", "black_level"))
        if not p.apply_masters:
            pass_through(ctx, p, frames["lights"], H, W, black)
            return
        ms = given_masters_of(ctx, frames)            # a master given: used as it is, its kind's single frames left out
        # issue #50: the masters' phase moves the bar too, file by file (its share: its files against the lights)
        cfa0 = ctx.metric_of("inventory", "cfa")
        n_hot = min(40, len(frames["lights"])) if hot_method(p, cfa0, len(frames["lights"]))[0] == "persistence" else 0
        nm = sum(len(frames[k]) for k in ("bias", "darks", "flats", "flat_darks")) \
            + (1 if frames["darks"] and "darks" not in ms else 0) + n_hot
        share = nm / max(nm + len(frames["lights"]), 1)
        ctx._files_total, ctx._files_read, ctx._lights_from = nm, 0, share
        ctx.phase(0.0, share)
        bias = master_bias([f["path"] for f in frames["bias"]], ctx, ms.get("bias"))
        if bias is not None:
            ctx.save("master_bias.npy", bias)
        # each light with the dark of its exposure (a session of lights of 300 and 60 s with a master of each got their
        # mean, of 180 s, scaled with its offset: every calibrated sky below 0)
        darks, rank, dark_noise, real = match_darks(ctx, frames["lights"], frames["darks"], bias, H, W, black)
        dark = darks[0]
        kept: dict = {}
        fd = flat_darks_of(ctx, frames["flat_darks"], frames["flats"])
        flat = master_flat([f["path"] for f in frames["flats"]], [f["path"] for f in fd], bias,
                           black, H, W, p.flat_scale_px, ctx, keep=kept,
                           cfa=ctx.metric_of("inventory", "cfa") or "RGGB", master="flats" in ms,
                           fd_masters=[f for f in fd if f.get("stacked") is not None] or None)
        if "flat_dark" in kept:
            ctx.save("master_flat_dark.npy", kept.pop("flat_dark"))
        calibrate_lights(ctx, p, frames["lights"], dark, flat, real[0], dark_noise,
                         ctx.metric_of("inventory", "cfa"), float(ctx.metric_of("inventory", "white_level")),
                         darks=darks, rank=rank)


def pass_through(ctx: Context, p: CalibrateParams, lights: list[dict], H: int, W: int, black: float) -> None:
    """The Calibration block cut (issue #43): the frames as they are (already calibrated), the black level taken off,
    no flat, no defective pixel; the calibrated cache the later stages read is written all the same."""
    ctx.log("Calibration cut: the frames are taken as they are (no master dark, flat or defective pixels)")
    dark = np.full((H, W), black, np.float32)
    flat = np.ones((H, W), np.float32)
    ctx.save("master_dark.npy", dark)
    ctx.save("master_flat.npy", flat)
    bad = np.zeros((H, W), bool)
    ctx.save("badmask.npy", bad)
    ctx.metric("dark_level", round(black, 1))
    ctx.metric("bad_pixels_pct", 0.0)
    ctx.metric("calibration_cut", True)
    _calibrated_cache(ctx, p, lights, dark, flat, bad, ctx.metric_of("inventory", "cfa"),
                      float(ctx.metric_of("inventory", "white_level")), time.time())


#: the calibration frames a master can stand for, and their names in a sentence
CAL_KINDS = {"bias": "bias", "darks": "darks", "flats": "flats", "flat_darks": "flat darks"}


def given_masters_of(ctx: Context, frames: dict) -> dict[str, list[dict]]:
    """The masters among the calibration frames, by kind (io/frames.master_count, from their headers: a file another
    program stacked). A master is not calibrated nor stacked again: it is used as it is (several of a kind: their mean,
    weighted by their frames), and the single frames given beside it are left out (it holds them already, or they
    would count twice). ``frames`` keeps the masters only, for the kinds that have one; each is said in the log, in one
    line."""
    out: dict[str, list[dict]] = {}
    for kind, word in CAL_KINDS.items():
        ms = [f for f in frames.get(kind) or [] if f.get("stacked") is not None]
        if not ms:
            continue
        # a master dark holds the darks of its exposure only: single darks of another one (and gain) are kept, for
        # the lights of theirs (``match_darks``)
        mk = [_key(m) for m in ms]
        own = [f for f in frames[kind] if f.get("stacked") is None and kind == "darks"
               and not any(_same_key(k, _key(f)) for k in mk)]
        left = len(frames[kind]) - len(ms) - len(own)
        out[kind] = ms
        frames[kind] = ms + own
        n = sum(int(f["stacked"]) for f in ms)
        what = (("a master" if len(ms) == 1 else f"{len(ms)} masters") +
                (f" of {n} frames" if n and all(f["stacked"] for f in ms) else ""))
        done = ", already calibrated" if kind == "flats" else ""
        ctx.log(f"the {word} are {what}{done}: used as {'it is' if len(ms) == 1 else 'they are'}" +
                ("" if len(ms) == 1 else " (their mean, weighted by their frames)"))
        if left:
            ctx.log(f"{left} single {word} given beside the master are left out (it holds its frames already)",
                    "warning")
    return out


#: two exposures within this share of the lights' are one (a camera writes 299.99 s for 300 s)
SAME_EXPOSURE = 0.05


def _key(f: dict) -> tuple[float | None, float | None]:
    """(exposure s, ISO or gain) of a frame of the inventory, None for what its header does not record."""
    e, g = f.get("exposure"), f.get("iso")
    return (float(e) if e else None, float(g) if g is not None else None)


def _same_exposure(t_l: float | None, t_d: float | None) -> bool:
    """Darks of ``t_d`` s fit lights of ``t_l`` s (an exposure not recorded is no mismatch)."""
    return t_l is None or t_d is None or abs(t_d - t_l) <= SAME_EXPOSURE * t_l


def _same_gain(a: float | None, b: float | None) -> bool:
    return a is None or b is None or a == b


def _same_key(a: tuple, b: tuple) -> bool:
    return _same_exposure(a[0], b[0]) and _same_gain(a[1], b[1])


def exposure_groups(keys: list[tuple]) -> list[dict]:
    """Frames by exposure and gain (``keys``: their ``_key``): [{"exposure": the median of its frames' exposures, or
    None, "gain", "of": the frames' ranks}], in the order of their first frame; exposures within SAME_EXPOSURE of a
    group's first are one. Frames whose exposure is not recorded join the largest group of their gain (they were
    calibrated with the darks of the others: the lights of one exposure stay one group)."""
    out: list[dict] = []
    for i, (e, g) in enumerate(keys):
        grp = next((x for x in out if x["gain"] == g and (x["first"] is None) == (e is None)
                    and (e is None or _same_exposure(x["first"], e))), None)
        if grp is None:
            out.append({"first": e, "gain": g, "of": [i]})
        else:
            grp["of"].append(i)
    for grp in [x for x in out if x["first"] is None]:
        known = [x for x in out if x["first"] is not None and x["gain"] == grp["gain"]]
        if known:
            big = max(known, key=lambda x: len(x["of"]))
            big["of"] = sorted(big["of"] + grp["of"])
            out.remove(grp)
    for grp in out:
        t = [keys[i][0] for i in grp["of"] if keys[i][0]]
        grp["exposure"] = float(np.median(t)) if t else None
    return out


def mixed_exposures(frames: list[dict]) -> list[float]:
    """The exposure times (s, longest first) of lights of several exposures (groups within SAME_EXPOSURE, whatever
    their gain, as ``exposure_groups`` makes them); [] for one exposure, or none recorded."""
    groups = exposure_groups([(_key(f)[0], None) for f in frames])
    times = sorted({g["exposure"] for g in groups if g["exposure"]}, reverse=True)
    return times if len(times) > 1 else []


#: A bias frame is the shortest exposure the camera can make; one longer than this is not "near zero" (0.1 s: far above
#: the 1/4000 s to 1/1000 s of a camera's shortest exposures, below the shortest flat or dark anyone takes).
BIAS_MAX_EXPOSURE = 0.1


def long_bias(bias: list[dict], flats: list[dict]) -> dict | None:
    """Bias frames whose exposure is not near zero (over ``BIAS_MAX_EXPOSURE``), which are often flat darks or darks named
    bias: {"n": how many, "exposure": the median one (s), "flats": whether the flats have that exposure (within
    SAME_EXPOSURE)}; None for none. Information only: nothing is reclassified."""
    long = sorted(f["exposure"] for f in bias if (f.get("exposure") or 0) > BIAS_MAX_EXPOSURE)
    if not long:
        return None
    t = long[len(long) // 2]
    return {"n": len(long), "exposure": t,
            "flats": any(f.get("exposure") and _same_exposure(t, f["exposure"]) for f in flats)}


def dark_plan(lights: list[dict], darks: list[dict], base: bool) -> list[tuple[str, int | None]]:
    """How each group of lights (``exposure_groups``) is calibrated against the groups of darks, by the usual practice
    of calibration: ("same", j) with the darks of group j, of its exposure and gain, as they are; ("scaled", j) when no
    dark is of its exposure, with those of group j (of its gain, the nearest exposure), their thermal signal scaled,
    only when ``base`` (a bias, or a known black level) tells it from the offset; ("none", None) without dark, rather
    than with a wrong one: a dark of another exposure scaled with its offset, or not scaled (the two masters of 300 and
    60 s of M101 averaged into one of 180 s and scaled by 1.667 with a black level of 0 put the calibrated sky at -260
    ADU). Darks are never averaged across exposures. A gain or an exposure one side does not record is no mismatch."""
    out: list[tuple[str, int | None]] = []
    for L in lights:
        ok = [j for j, D in enumerate(darks) if _same_gain(L["gain"], D["gain"])]
        same = [j for j in ok if _same_exposure(L["exposure"], darks[j]["exposure"])]
        if same:               # a recorded exposure before one not recorded, then the most frames
            out.append(("same", max(same, key=lambda j: (darks[j]["exposure"] is not None, len(darks[j]["of"])))))
        elif base and ok:
            out.append(("scaled", min(ok, key=lambda j: abs(np.log(darks[j]["exposure"] / L["exposure"])))))
        else:
            out.append(("none", None))
    return out


def _group_name(grp: dict, gains: bool, word: str) -> str:
    """A group of frames in a sentence: "300 s", "60 s at gain 139" (``gains``: the session's groups differ in gain)."""
    t = f"{grp['exposure']:g} s" if grp["exposure"] else "an exposure not recorded"
    return t + (f" at {word} {grp['gain']:g}" if gains and grp["gain"] is not None else "")


def match_darks(ctx: Context, lights: list[dict], darks: list[dict], bias: np.ndarray | None, H: int, W: int,
                black: float) -> tuple[list[np.ndarray], np.ndarray, float | None, list[bool]]:
    """The dark of each group of lights of one exposure and gain (``dark_plan``): (the darks, the main group's first:
    the group of the most lights, whose dark is master_dark.npy and finds the hot pixels; each light's rank in them;
    the noise of one dark of the main group, or None; whether each has a dark). The masters made by
    another program (``stacked``) stand for their group's darks, as they are. A session of one exposure gets the dark
    it got before, to the byte (the same files, in the same order, the same scaling)."""
    n = len(lights)
    lg = exposure_groups([_key(f) for f in lights])
    lg.insert(0, lg.pop(max(range(len(lg)), key=lambda i: len(lg[i]["of"]))))
    rank = np.zeros(n, int)
    for i, grp in enumerate(lg):
        rank[grp["of"]] = i
    if not darks:
        dark, _ = master_dark([], bias, H, W, black, ctx)
        return [dark], np.zeros(n, int), None, [False]
    dg = exposure_groups([_key(f) for f in darks])
    plan = dark_plan(lg, dg, bias is not None or black > 0)
    mixed = len(lg) > 1 or len(dg) > 1
    from ..core.project import frame_kind
    word = "ISO" if frame_kind(lights[0]["path"]) == "raw" else "gain"
    gains = len({g["gain"] for g in lg + dg if g["gain"] is not None}) > 1
    made: dict[int, tuple[np.ndarray, float | None]] = {}
    base = bias if bias is not None else np.float32(black)
    out, noise = [], None
    for i, (L, (how, j)) in enumerate(zip(lg, plan)):
        name = _group_name(L, gains, word)
        if how == "none":
            others = sorted({_group_name(dg[k], gains, word) for k in range(len(dg)) if _same_gain(L["gain"], dg[k]["gain"])})
            why = (" (only the master bias taken off)" if bias is not None else
                   " (only the black level taken off)" if black > 0 else
                   f" (no bias frames nor black level: the darks of {', '.join(others)} would scale the offset with "
                   "their thermal signal)" if others else "")
            ctx.log(f"no dark of {name}: the lights of {name} are calibrated without dark{why}", "warning")
            out.append(bias.copy() if bias is not None else np.full((H, W), black, np.float32))
            continue
        if j not in made:
            fs = [darks[k] for k in dg[j]["of"]]
            ms = [f for f in fs if f.get("stacked") is not None]
            made[j] = master_dark([f["path"] for f in fs], bias, H, W, black, ctx, ms or None)
            check_dark(ctx, made[j][0], bias, black)
        dark = made[j][0]
        if i == 0:
            noise = made[j][1]
        if how == "scaled":           # the classic dark scaling of the reduction guides (``scaled_dark``)
            r = L["exposure"] / dg[j]["exposure"]
            ctx.log(f"no dark of {name}: the lights of {name} are calibrated with the darks of "
                    f"{_group_name(dg[j], gains, word)}, their thermal signal scaled by {r:.3f}"
                    + ("" if bias is not None else " (no bias frames: above the black level)"))
            if i == 0:
                ctx.metric("dark_scale", round(r, 4))
            dark = (base + (dark - base) * np.float32(r)).astype(np.float32)
        elif mixed:
            ctx.log(f"the lights of {name} are calibrated with the darks of {_group_name(dg[j], gains, word)}")
        out.append(dark)
    return out, rank, noise, [how != "none" for how, _j in plan]


def flat_darks_of(ctx: Context, flat_darks: list[dict], flats: list[dict]) -> list[dict]:
    """The flat darks of the flats' exposure: all of them when they are of one (the same files as before); of several,
    those of the flats' exposure and gain, else of the nearest one, said (never averaged across exposures)."""
    groups = exposure_groups([_key(f) for f in flat_darks])
    if len(groups) < 2:
        return flat_darks
    t = exposure_groups([_key(f) for f in flats])
    want = max(t, key=lambda x: len(x["of"])) if t else {"exposure": None, "gain": None}
    ok = [g for g in groups if _same_gain(want["gain"], g["gain"])] or groups
    same = [g for g in ok if _same_exposure(want["exposure"], g["exposure"])]
    pick = (max(same, key=lambda g: len(g["of"])) if same else
            min(ok, key=lambda g: abs(np.log((g["exposure"] or 1.0) / (want["exposure"] or 1.0)))))
    e = f"{pick['exposure']:g} s" if pick["exposure"] else "an exposure not recorded"
    ctx.log(f"the flats are corrected with the flat darks of {e}" + ("" if same else ", the nearest to theirs"))
    return [flat_darks[i] for i in pick["of"]]


def _masters_mean(ms: list[dict], ctx: Context, label: str, read=None) -> np.ndarray:
    """The master of a kind given as masters: the one file as it is, or their mean weighted by the frames each holds
    (one when its header does not say)."""
    read = read or read_cfa
    out, tot = None, 0.0
    for i, f in enumerate(ms):
        w = float(f.get("stacked") or 1)
        a = read(f["path"]) * np.float32(w) if len(ms) > 1 else read(f["path"])
        out = a if out is None else out + a
        tot += w
        _tick(ctx, f"reading the {label} {i + 1}/{len(ms)}")
    return out if len(ms) == 1 else (out / np.float32(tot)).astype(np.float32)


#: a dark whose median level is under this share of the bias level (or of the black level without bias frames) had its
#: bias taken off: it does not take the lights' off (a master made by another program with its bias removed)
DARK_FAR = 0.5
#: a master flat over its centre whose lowest levels (its 0.01 % percentile) go under minus this share went under the
#: level it was corrected with: the flats had their bias taken off already, and the calibration took it off again
FLAT_UNDER = 0.005
#: the percentile of a flat's lowest levels: a few pixels in a million, not the single dead pixel
LOW_PCT = 0.01


def dark_far(level: float, ref: float) -> bool:
    """Whether darks of median ``level`` cannot take off the lights' bias level ``ref`` (``DARK_FAR``): the rule of the
    stage's warning and of the Inspect's matching line."""
    return ref > 0 and level < DARK_FAR * ref


def flat_under(low: float) -> bool:
    """Whether a flat whose lowest levels are ``low``, as a share of its central level once the bias (or the level it is
    corrected with) is taken off, went under it (``FLAT_UNDER``): its bias was taken off twice."""
    return low < -FLAT_UNDER


def master_bias(paths: list[str], ctx: Context, masters: list[dict] | None = None) -> np.ndarray | None:
    """Master bias (offset): median of the bias frames, or None without any; ``masters``: the given master, as it is."""
    if masters:
        return _masters_mean(masters, ctx, "master bias")
    return _median_stack(paths, ctx, "bias") if paths else None


def master_dark(paths: list[str], bias: np.ndarray | None, H: int, W: int, black: float,
                ctx: Context, masters: list[dict] | None = None) -> tuple[np.ndarray, float | None]:
    """Master dark (median of the darks) and the noise of one dark frame; without darks, the master bias or the black
    level stands for it. ``masters``: the given master dark, as it is; the noise of one dark is then not known (a
    master's is the noise of its mean, and its fixed pattern does not go down with it: none rather than a guess)."""
    if masters:
        return _masters_mean(masters, ctx, "master dark"), None
    if paths:
        dark = _median_stack(paths, ctx, "darks")
        one = read_cfa(paths[0])
        _tick(ctx, "the noise of one dark")
        return dark, robust_sigma(one[::3, ::3])[1]
    dark = bias.copy() if bias is not None else np.full((H, W), black, np.float32)
    ctx.log("no dark frames: using " + ("the master bias" if bias is not None else "the black level") +
            " as dark (hot pixels will not be removed by calibration)", "warning")
    return dark, None


def _exposure(frames: list[dict]) -> float | None:
    t = [float(f["exposure"]) for f in frames if f.get("exposure")]
    return float(np.median(t)) if t else None


def scaled_dark(ctx: Context, dark: np.ndarray, bias: np.ndarray | None, black: float, frames: dict) -> np.ndarray:
    """The master dark brought to the lights' exposure when the darks were taken at another one: its thermal signal
    (the dark minus the bias, or minus the black level without bias frames) scaled by the ratio of the exposures, the
    bias put back (dark current grows in proportion to the exposure; the classic dark scaling of the reduction
    guides, e.g. the astropy CCD reduction guide, "scaling dark frames"). Darks of the lights' exposure: as they are."""
    t_l, t_d = _exposure(frames.get("lights") or []), _exposure(frames.get("darks") or [])
    if not frames.get("darks") or not t_l or not t_d or abs(t_d - t_l) <= 0.05 * t_l:
        return dark
    base = bias if bias is not None else np.float32(black)
    r = t_l / t_d
    ctx.log(f"darks of {t_d:g} s for lights of {t_l:g} s: their thermal signal scaled by {r:.3f}"
            + ("" if bias is not None else " (no bias frames: above the black level)"))
    ctx.metric("dark_scale", round(r, 4))
    return (base + (dark - base) * np.float32(r)).astype(np.float32)


def check_dark(ctx: Context, dark: np.ndarray, bias: np.ndarray | None, black: float) -> None:
    """Darks whose level is far under the bias (or the black level) had their bias taken off: said, never guessed
    (``dark_far``): the lights then keep their own, which the flat turns into an inverted vignetting."""
    level = float(np.median(dark))
    ref = float(np.median(bias)) if bias is not None else black
    if dark_far(level, ref):
        ctx.log(f"the master dark's level ({level:.0f} ADU) is far under the {'bias' if bias is not None else 'black level'}"
                f" ({ref:.0f} ADU): its bias was taken off, so it does not take the lights' off; give darks that hold it",
                "warning")


def master_flat(paths: list[str], flat_dark_paths: list[str], bias: np.ndarray | None, black: float, H: int,
                W: int, flat_scale_px: float, ctx: Context, keep: dict | None = None, cfa: str = "RGGB",
                master: bool = False, fd_masters: list[dict] | None = None) -> np.ndarray:
    """Master flat normalised to 1 at the centre of each position of the colour pattern (``cfa``: its 4 Bayer
    positions, its 36 X-Trans ones), all ones without flats. ``keep``: receives the master flat dark under
    "flat_dark" when there are flat darks (``fd_masters``: given as a master, as it is). ``master``: the flats are a
    master (``given_masters_of``): already
    calibrated (the stacking programs take the bias or the flat dark off each flat before they stack them, which is
    what makes a master flat), so nothing is taken off it: the bias taken off again was 7 to 20 % of a flat's level."""
    P = cfa_mod.period(cfa)
    if paths:
        if master:
            fdark = np.zeros((H, W), np.float32)
            if flat_dark_paths:
                ctx.log("the flat darks are not used: the master flat is already calibrated")
        elif flat_dark_paths:
            fdark = (_masters_mean(fd_masters, ctx, "master flat dark") if fd_masters else
                     _median_stack(flat_dark_paths, ctx, "flat darks"))
            if keep is not None:
                keep["flat_dark"] = fdark
        elif bias is not None:
            fdark = bias
        else:
            fdark = np.full((H, W), black, np.float32)   # the black level only: never a dark's thermal signal
        flat = _master_flat(paths, fdark, ctx)
        low = float(np.percentile(flat[::2, ::2], LOW_PCT))
        if not master and flat_under(low):
            ctx.log(f"the flats go under the {'flat darks' if flat_dark_paths else 'bias' if bias is not None else 'black level'}"
                    f" they are corrected with ({100 * low:.1f} % of their level): their bias was taken off already, "
                    "and is taken off again", "warning")
        if flat_scale_px > 0:
            # borrowed flat: vignetting only (each colour plane smoothed; its dust is not this night's dust)
            for k in range(P * P):
                v = _plane(flat, k, P)
                v[...] = cv2.GaussianBlur(np.ascontiguousarray(v), (0, 0), flat_scale_px / P)
            ctx.log(f"master flat kept above {flat_scale_px:.0f} px only (vignetting, not dust)")
    else:
        flat = np.ones((H, W), np.float32)
        ctx.log("no flat frames: vignetting and dust will not be corrected (the sky model will absorb part of "
                "the vignetting)", "warning")
    r = 200 // P                                        # the centre's 400 x 400 sensor pixels, in plane pixels
    for k in range(P * P):
        v = _plane(flat, k, P)
        c = v[v.shape[0] // 2 - r:v.shape[0] // 2 + r, v.shape[1] // 2 - r:v.shape[1] // 2 + r]
        m = np.median(c)
        if not m > 0:
            raise ValueError(f"the master flat has no light at the centre of its colour position {k} (it reads {m:.3g})")
        v /= m
    _flat_finite(flat, "master flat")
    return flat


def calibrate_lights(ctx: Context, p: CalibrateParams, lights: list[dict], dark: np.ndarray, flat: np.ndarray,
                     have_darks: bool, dark_noise: float | None, cfa: str, white: float,
                     darks: list[np.ndarray] | None = None, rank: np.ndarray | None = None) -> None:
    """Bad pixel map and calibrated cache of the light frames from the given masters (saved with them in the
    stage's folder: master_dark, master_flat, badmask, frame_stats, cache.json). ``darks``, ``rank``: the darks of
    the lights' exposures (``match_darks``; ``dark`` the main group's, the first) and each light's; one dark for all
    without. The hot pixels are those of the main group's dark, the sky noise that of one of its lights."""
    H, W = dark.shape
    t0 = time.time()
    level = float(np.median(dark))
    ctx.save("master_dark.npy", dark)
    darks, rank = _one_dark(dark, darks, rank, len(lights))
    _save_darks(ctx, darks, rank)
    ctx.metric("dark_level", round(level, 1))
    if dark_noise:
        ctx.metric("dark_frame_noise", round(dark_noise, 2))
    ctx.save("master_flat.npy", flat)
    ctx.metric("vignetting_min", round(float(flat.min()), 3))

    # ---------------- bad pixels: hot (dark), cold / dust (flat)
    P = cfa_mod.period(cfa)
    main = np.flatnonzero(rank == 0)
    sky_noise = _sky_noise(lights[int(main[len(main) // 2])]["path"], dark, flat, P)
    bad = np.zeros((H, W), bool)
    if have_darks:
        bad |= (dark - level) > p.hot_sigma * sky_noise
    from scipy.ndimage import median_filter
    for k in range(P * P):
        v = _plane(flat, k, P)
        loc = median_filter(v, size=9)
        _plane(bad, k, P)[...] |= (v < (1 - p.flat_defect) * loc) | (v > (1 + p.flat_defect) * loc)
    # a pixel the flat's light hardly reached (a sensor's overscan, a dead column: a band the local test above cannot
    # see) would be multiplied by ten or more
    bad |= flat < FLAT_UNLIT * float(np.median(flat))
    hot, why = hot_method(p, cfa, len(lights))
    if why:
        ctx.log(why, "warning")
    if hot == "persistence":
        n0 = int(bad.sum())
        msg = _hot_from_lights([f["path"] for f in lights], dark, flat, bad, median_filter, cfa=cfa,
                               tick=lambda what: _tick(ctx, what),
                               dark_of=(lambda i: darks[rank[i]]) if len(darks) > 1 else None)
        ctx.metric("hot_pixels_from_lights", int(bad.sum()) - n0)
        ctx.log(msg)
    # a pixel left out would be a hole in the stack: every excluded pixel is filled from its same-colour neighbours
    ctx.save("badmask.npy", bad)
    ctx.metric("bad_pixels_pct", round(100 * float(bad.mean()), 2))
    ctx.metric("sky_noise_adu", round(sky_noise, 2))
    ctx.log(f"master dark level {level:.1f} ADU, flat min {flat.min():.3f}, {100 * bad.mean():.2f} % pixels "
            f"excluded, sky noise {sky_noise:.1f} ADU")
    _calibrated_cache(ctx, p, lights, dark, flat, bad, cfa, white, t0, darks, rank, hot)


def hot_method(p: CalibrateParams, cfa: str, n_lights: int) -> tuple[str, str]:
    """(the method that runs: "persistence", "per_frame" or "off", why it is not the one asked for, or ""). The
    persistence needs a session of 15 lights or more; Siril's correction is made for a Bayer sensor ("It does not work
    for X-Trans sensor", its documentation): an X-Trans session gets the persistence."""
    if not p.hot_from_lights:
        return "off", ""
    if p.hot_method == "per_frame" and cfa_mod.is_xtrans(cfa):
        if n_lights >= 15:
            return "persistence", "Per frame hot pixels work on a Bayer sensor only: the persistence is used instead"
        return "off", ("Per frame hot pixels work on a Bayer sensor only, and the session is too short for the "
                       "persistence: no hot pixel search in the lights")
    if p.hot_method == "persistence" and n_lights < 15:
        return "off", ""
    return p.hot_method, ""


#: the darks of the lights of each exposure when they are of several (``match_darks``): "darks" (G, H, W), the main
#: group's first (master_dark.npy), and "rank" (each light's); not there for a session of one exposure
GROUP_DARKS = "master_darks.npz"


def _one_dark(dark: np.ndarray, darks: list | None, rank, n: int) -> tuple[list[np.ndarray], np.ndarray]:
    if not darks or rank is None or len(rank) != n:
        return [dark], np.zeros(n, int)
    return list(darks), np.asarray(rank, int)


def _save_darks(ctx: Context, darks: list[np.ndarray], rank: np.ndarray) -> None:
    """GROUP_DARKS in the stage's folder for a session of several exposures (the cache-only rebuild reads it back),
    none else."""
    f = ctx.out(GROUP_DARKS)
    if len(darks) < 2:
        f.unlink(missing_ok=True)
        return
    np.savez(f, darks=np.stack(darks).astype(np.float32), rank=rank)


#: what a v2 runner copies from an earlier result whose calibrated cache was lost, with MASTERS (their settings
#: and metrics): calibrate then rebuilds the cache only (``given_masters``)
MASTER_FILES = ("master_dark.npy", "master_flat.npy", "badmask.npy")
MASTERS = "masters.json"
#: masters kept for the Inspect mode only (there when the project has bias or flat dark frames): the cache-only
#: rebuild copies them from the earlier result itself (``from`` of MASTERS)
EXTRA_MASTERS = ("master_bias.npy", "master_flat_dark.npy")


def result_values(values: dict) -> dict:
    """Of a stage's settings (a dict, its manifest's or ``model_dump``'s), those that decide the result (the worker
    count does not)."""
    d = dict(values)
    d.pop("workers", None)
    if d.get("apply_masters", True):                 # the default: as the results made before the setting existed
        d.pop("apply_masters", None)
    return d


def result_params(p: CalibrateParams) -> dict:
    return result_values(p.model_dump(mode="json"))


def given_masters(ctx: Context, p: CalibrateParams) -> dict | None:
    """{"params", "metrics"} of masters placed in the stage's folder with MASTERS, when they were made with the same
    settings; None otherwise (a full calibration, which replaces them)."""
    f = ctx.dir / MASTERS
    if not f.exists():
        return None
    try:
        d = json.loads(f.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        d = {}
    if d.get("params") == result_params(p) and all((ctx.dir / n).is_file() for n in MASTER_FILES):
        return d
    f.unlink(missing_ok=True)
    ctx.log("the masters of the earlier result were made with other settings: full calibration", "warning")
    return None


def rebuild_cache(ctx: Context, p: CalibrateParams, lights: list[dict], given: dict, cfa: str, white: float) -> None:
    """The cache-only path: the calibrated frames from the masters already in the stage's folder (the same files,
    so the same cache as a full run), their metrics restored."""
    t0 = time.time()
    dark, flat, bad = (np.load(ctx.out(n)) for n in MASTER_FILES)
    src = Path(str(given.get("from") or ""))
    if given.get("from") and (src / GROUP_DARKS).is_file() and not ctx.out(GROUP_DARKS).is_file():
        shutil.copy2(src / GROUP_DARKS, ctx.out(GROUP_DARKS))      # a session of several exposures: each its dark
    darks = rank = None
    if ctx.out(GROUP_DARKS).is_file():
        with np.load(ctx.out(GROUP_DARKS)) as z:
            darks, rank = list(z["darks"]), z["rank"]
    for k, v in (given.get("metrics") or {}).items():
        if k not in ("cache_scale", "cache_gb"):
            ctx.metric(k, v)
    ctx.log("calibrated frames rebuilt with the masters of the earlier result (same settings)")
    hot, why = hot_method(p, cfa, len(lights))
    for n in EXTRA_MASTERS:
        if given.get("from") and (src / n).is_file() and not ctx.out(n).is_file():
            shutil.copy2(src / n, ctx.out(n))
    _calibrated_cache(ctx, p, lights, dark, flat, bad, cfa, white, t0, darks, rank, hot)
    (ctx.dir / MASTERS).unlink(missing_ok=True)


def _calibrated_cache(ctx: Context, p: CalibrateParams, lights: list[dict], dark: np.ndarray, flat: np.ndarray,
                      bad: np.ndarray, cfa: str, white: float, t0: float,
                      darks: list[np.ndarray] | None = None, rank: np.ndarray | None = None,
                      hot: str = "off") -> None:
    """The calibrated cache of the light frames (float16 memmap on the scratch disk), its frame statistics and
    cache.json, from the masters and the bad pixel map; the bad pixels filled with the mean of their same-colour
    neighbours that are not bad (Siril's replacement). ``hot``: "per_frame" corrects each light's hot pixels the way
    Siril's cosmetic correction does (``siril_hot_pixels``). ``darks``, ``rank``: each light's dark among the darks of
    the lights' exposures (``match_darks``), ``dark`` for all without."""
    H, W = dark.shape
    darks, rank = _one_dark(dark, darks, rank, len(lights))
    # the lowest dark's level: the widest range above it, which the cache's scale keeps in the float16 range
    level = min(float(np.median(d)) for d in darks) if len(darks) > 1 else float(np.median(dark))
    # values are scaled so that the brightest valid value stays below the float16 range
    scale = float(min(1.0, 16000.0 / max(white - level, 1.0)))
    n = len(lights)
    cache = ctx.scratch / "lights.f16"
    done_p = ctx.ckpt / "done.npy"
    stats_p = ctx.ckpt / "stats.npy"
    hot_p = ctx.ckpt / "hot.npy"
    resume = cache.exists() and done_p.exists() and cache.stat().st_size == n * H * W * 2
    if resume:
        done = np.load(done_p)
        stats = np.load(stats_p)
        hots = np.load(hot_p) if hot_p.exists() else np.zeros(n, np.int64)
        ctx.log(f"resuming calibration: {int(done.sum())}/{n} frames already done")
    else:
        done = np.zeros(n, bool)
        stats = np.zeros((n, 4, 2), np.float32)
        hots = np.zeros(n, np.int64)
        mm = np.memmap(cache, dtype=np.float16, mode="w+", shape=(n, H, W))
        del mm
    paths_dir = ctx.ckpt
    np.save(paths_dir / "dark.npy", dark)
    for g in range(1, len(darks)):
        np.save(paths_dir / f"dark_{g}.npy", darks[g])
    np.save(paths_dir / "flat.npy", flat)
    np.save(paths_dir / "bad.npy", bad)
    todo = [(i, lights[i]["path"], int(rank[i])) for i in range(n) if not done[i]]
    sat_raw = p.saturation * white
    plain = plain_writes(ctx, n * H * W * 2, p.workers, H * W)
    init = (str(cache), n, H, W, str(paths_dir), sat_raw, scale, hot, cfa, plain, len(darks))
    ctx.phase(getattr(ctx, "_lights_from", 0.0), 1.0)          # issue #50: the lights' share of the bar
    ctx.progress(done.mean(), f"calibrating {len(todo)} frames ({p.workers} processes)", force=True)
    with devices.one_maths_thread():                           # issue #125: 1 GB less reserved a worker
        pool = get_context("spawn").Pool(p.workers, initializer=_init, initargs=init)
    with pool:
        for c, (idx, st, nh) in enumerate(pool.imap_unordered(_calib_one, todo, chunksize=2)):
            stats[idx] = st
            hots[idx] = nh
            done[idx] = True
            if (c + 1) % 20 == 0 or c + 1 == len(todo):
                np.save(done_p, done)
                np.save(stats_p, stats)
                np.save(hot_p, hots)
            ctx.progress(done.mean(), f"{int(done.sum())}/{n} frames calibrated")
    for f in ("dark.npy", "flat.npy", "bad.npy", *(f"dark_{g}.npy" for g in range(1, len(darks)))):
        (paths_dir / f).unlink(missing_ok=True)
    ctx.save("frame_stats.npy", stats)
    if hot == "per_frame":
        ctx.metric("hot_pixels_from_lights", int(round(float(hots.mean()))))
        ctx.log(f"hot pixels, per frame: {int(hots.sum())} pixels corrected in the {n} lights (between "
                f"{int(hots.min())} and {int(hots.max())} in a light, {hots.mean():.0f} on average), each by the mean "
                f"of its same-colour neighbours")
    ctx.save_json("cache.json", {"path": str(cache), "n": n, "height": H, "width": W, "scale": scale,
                                 "cfa": cfa, "sat_mark": SAT_MARK, "valid_max": VALID_MAX,
                                 "white_level": white, "dark_level": level})
    ctx.metric("cache_scale", scale)
    ctx.metric("cache_gb", round(n * H * W * 2 / 1e9, 1))
    ctx.log(f"{n} frames calibrated in {(time.time() - t0) / 60:.1f} min (cache {n * H * W * 2 / 1e9:.1f} GB)")
    from .inventory import _superpixel_rgb
    ctx.preview("master_flat", _flat_view(flat, cfa))
    fr = np.nan_to_num(CachedFrames({"path": str(cache), "n": n, "height": H, "width": W})[n // 2])
    ctx.preview("calibrated_frame", _superpixel_rgb(np.where(fr > VALID_MAX, 0, fr), cfa))


def _plane(a: np.ndarray, k: int, P: int = 2) -> np.ndarray:
    """The samples of position k of the colour pattern of period P (core/cfa.plane)."""
    return cfa_mod.plane(a, k, P)


#: the files of the masters read at a time (their reading and decoding are what these phases wait for)
READERS = max(4, min(os.cpu_count() or 4, 8))


def _read_frames(paths: list[str], put, start: int = 0, tick=None, read=None) -> None:
    """The files from rank ``start`` read on several threads: ``put(i, frame)`` stores each one in its place (so the
    result does not depend on which is read first); ``tick(c)``: c files of ``paths`` are done."""
    with ThreadPoolExecutor(READERS) as pool:
        jobs = [pool.submit(lambda i=i, q=q: put(i, (read or read_cfa)(q))) for i, q in enumerate(paths) if i >= start]
        for c, job in enumerate(as_completed(jobs)):
            job.result()
            if tick is not None:
                tick(start + c + 1)


def _master_name(kind: str) -> str:
    """The calibration frames a shared master is made of, as the logs say them: bias, darks, flat darks, flats."""
    return "flats" if kind == "flat" else kind.replace("colour_median_", "").replace("median_", "").replace("_", " ")


def _shared(ctx: Context, kind: str, paths: list[str], build, extra: bytes = b"") -> np.ndarray:
    """A master built once for identical inputs: the same files (name, size, date), the same ``extra`` (what else it
    depends on) and the same code version give the same master, kept under the project's store and read by every
    session that asks for it (three sessions sharing 48 darks read them three times, 45 s each). Without a store
    (no project folder) it is simply built."""
    store = getattr(getattr(ctx, "project", None), "store", None)
    if store is None:
        return build()
    h = hashlib.blake2b(digest_size=16)
    h.update(f"{kind}|{Calibrate.version}|".encode())
    for q in paths:
        st = os.stat(q)
        h.update(f"{os.path.abspath(q)}|{st.st_size}|{st.st_mtime_ns};".encode())
    h.update(extra)
    d = Path(store.path) / "_masters"
    f = d / f"{kind}_{h.hexdigest()}.npy"
    if f.is_file():
        try:
            got = np.load(f)
            ctx.log(f"master of the {_master_name(kind)} reused: " + (
                "its file was" if len(paths) == 1 else f"its {len(paths)} files were") + " already read by another session")
            return got
        except Exception:
            pass
    out = build()
    d.mkdir(parents=True, exist_ok=True)
    tmp = f.with_name(f.name + f".{os.getpid()}.tmp.npy")
    np.save(tmp, out)
    tmp.replace(f)
    ctx.log(f"master of the {_master_name(kind)} built from " + ("1 file" if len(paths) == 1 else f"{len(paths)} files")
            + " and kept for the other sessions")
    return out


def _median_stack(paths: list[str], ctx: Context, label: str) -> np.ndarray:
    return _shared(ctx, "median_" + label.replace(" ", "_"), paths, lambda: _median_stack_built(paths, ctx, label))


def _median_stack_built(paths: list[str], ctx: Context, label: str) -> np.ndarray:
    """The pixel-wise median of the frames. They are read into one array (no list then stack), and the median is taken
    in bands of rows, in place: the memory is the frames once, not three times (28 darks of 12 Mpx: 1.3 GB, not 4)."""
    first = read_cfa(paths[0])
    stack = np.empty((len(paths), *first.shape), np.float32)
    stack[0] = first
    _tick(ctx, f"reading {label} 1/{len(paths)}")

    def put(i, a):
        stack[i] = a

    _read_frames(paths, put, 1, lambda c: _tick(ctx, f"reading {label} {c}/{len(paths)}"))
    return _median_stack_of(stack)


def _median_stack_of(stack: np.ndarray) -> np.ndarray:
    """The pixel-wise median of the frames (n, H, W), by bands of rows on the machine's cores (the stack is sorted in
    place, band by band)."""
    out = np.empty(stack.shape[1:], np.float32)

    def band(r0, r1):
        out[r0:r1] = np.median(stack[:, r0:r1], axis=0, overwrite_input=True)

    by_bands(out.shape[0], band)
    return out


#: the rows a pixel-wise statistic of a stack of frames is taken on at a time (its temporary arrays stay small)
BAND = 256
#: the same on several threads (numpy's sorts and scipy's filters let go of the interpreter): thinner bands, so that
#: the threads' temporary arrays together stay as small as one band of BAND rows
THREAD_BAND = 64


def by_bands(H: int, work, threads: int | None = None) -> None:
    """``work(r0, r1)`` for the bands of rows of 0..H, on the machine's cores: the pixel-wise statistics of the masters
    were taken band after band on one core, which left a CPU-only machine at a tenth of its power. The bands are
    independent, so the result does not depend on the threads."""
    n = threads or max(1, min(os.cpu_count() or 1, 8))
    cuts = [(r, min(r + THREAD_BAND, H)) for r in range(0, H, THREAD_BAND)]
    if n == 1 or len(cuts) == 1:
        for r0, r1 in cuts:
            work(r0, r1)
        return
    with ThreadPoolExecutor(n) as pool:
        for job in [pool.submit(work, r0, r1) for r0, r1 in cuts]:
            job.result()
#: a flat pixel under this share of the flat's median received no light: a bad pixel
FLAT_UNLIT = 0.1


def _master_flat(paths: list[str], fdark: np.ndarray, ctx: Context) -> np.ndarray:
    return _shared(ctx, "flat", paths, lambda: _master_flat_built(paths, fdark, ctx),
                   hashlib.blake2b(np.ascontiguousarray(fdark).tobytes(), digest_size=16).digest())


def _master_flat_built(paths: list[str], fdark: np.ndarray, ctx: Context) -> np.ndarray:
    """The flats, each over its central level, averaged without the values over 3 MAD from the pixel's median. Read
    into one array and reduced in bands of rows: the memory is the frames once (46 flats of 12 Mpx: 2 GB, not 10)."""
    H, W = fdark.shape
    fl = np.empty((len(paths), H, W), np.float32)

    def put(i, a):
        _flat_finite(a, paths[i])
        fl[i] = a - fdark

    _read_frames(paths, put, 0, lambda c: _tick(ctx, f"reading flats {c}/{len(paths)}"))
    level = np.median(fl[:, H // 2 - 200:H // 2 + 200, W // 2 - 200:W // 2 + 200], axis=(1, 2), keepdims=True)
    for i, v in enumerate(level.ravel()):
        if not v > 0:                       # nothing divides by it: a flat is its own light over its centre's
            raise ValueError(f"the flat {Path(paths[i]).name} has no light left at its centre once the level it is "
                             f"corrected with ({float(np.median(fdark)):.0f} ADU) is taken off (it reads "
                             f"{float(v):.3g} ADU): a flat already calibrated (a master) must be given as a master, "
                             "otherwise give its flat darks or a bias")
    fl /= level
    _flat_finite(fl, paths[0] if len(paths) == 1 else f"{len(paths)} flats")
    return _flats_mean(fl)


def _flat_finite(a: np.ndarray, what: str) -> None:
    """A flat with a value that is not a number (NaN, infinite) would turn the lights it divides into holes, said later
    by a stage that cannot say why: it stops here, naming the flat."""
    if not np.isfinite(a).all():
        raise ValueError(f"the flat {Path(what).name} holds values that are not numbers "
                         f"({int((~np.isfinite(a)).sum())} pixels): it cannot correct the lights")


def _flats_mean(fl: np.ndarray) -> np.ndarray:
    """The flats (n, H, W), each over its central level, averaged without the values over 3 MAD from the pixel's
    median (by bands of rows)."""
    H, W = fl.shape[1:]
    out = np.empty((H, W), np.float32)

    def band(r0, r1):
        b = fl[:, r0:r1]
        med = np.median(b, axis=0)
        dev = np.abs(b - med)
        keep = dev < 3 * (1.4826 * np.median(dev, axis=0) + 1e-6)
        out[r0:r1] = np.where(keep, b, 0).sum(0) / np.maximum(keep.sum(0), 1)

    by_bands(H, band)
    return out


def _tick(ctx: Context, what: str) -> None:
    """One more file of the masters' phase read (issue #50): the bar moves by one file."""
    ctx._files_read = getattr(ctx, "_files_read", 0) + 1
    tot = getattr(ctx, "_files_total", 0)
    ctx.progress(min(1.0, ctx._files_read / tot) if tot else 0.0, what, force=ctx._files_read == 1)


def _hot_from_lights(paths, dark, flat, bad, median_filter, n=40, k=5.0, cfa: str = "RGGB", tick=None,
                     dark_of=None) -> str:
    """The persistence: flag in ``bad`` the pixels that stand out from their same-colour neighbours in the lower
    quartile of ``n`` light frames spread over the session, in sensor coordinates: a defect is bright in every frame,
    a star covers a given pixel in some of them only. Returns the message of the log. Each position of the pattern
    (``cfa``) is a plane of its own: its neighbours in the plane are of its colour. ``dark_of(i)``: the dark of the
    light of rank i (a session of several exposures), ``dark`` for all without."""
    P = cfa_mod.period(cfa)
    picks = np.linspace(0, len(paths) - 1, min(n, len(paths))).astype(int)
    sub = [paths[i] for i in picks]
    dk = [dark_of(i) for i in picks] if dark_of is not None else [dark] * len(picks)
    st = np.empty((len(sub), *dark.shape), np.float32)          # one array, the quartile in bands: the frames once

    def put(i, a, st=st):
        st[i] = (a - dk[i]) / flat

    _read_frames(sub, put, 0, (lambda c: tick(f"hot pixels: frame {c}/{len(sub)}")) if tick is not None else None)
    one = st[len(sub) // 2].copy()                    # a single frame: the unit of the threshold is ITS noise
    q = np.empty(dark.shape, np.float32)

    def quartile(r0, r1):
        q[r0:r1] = np.percentile(st[:, r0:r1], 25, axis=0, overwrite_input=True)

    by_bands(q.shape[0], quartile)
    del st
    cand = np.zeros(dark.shape, bool)
    med_all = np.empty_like(q)
    def position(c):
        v = _plane(q, c, P)
        med = median_filter(v, size=5)
        f = _plane(one, c, P)
        # the published thresholds (3 to 5 sigma) are in units of one exposure's noise, as Siril's per-frame test is;
        # the quartile's own residual is a third of that (the frames' noise averaged out, the defects' fixed
        # pattern kept), and 5 of its sigmas were 2 of a frame's: every warm pixel of the dark current was flagged
        s = robust_sigma((f - median_filter(f, size=5))[::2, ::2])[1]
        hot = (v - med) > k * s
        _plane(cand, c, P)[...] = hot
        _plane(med_all, c, P)[...] = med
    # the positions of the colour pattern are independent planes: one thread each (the filters let go of the
    # interpreter)
    with ThreadPoolExecutor(min(P * P, max(1, os.cpu_count() or 1))) as pool:
        for job in [pool.submit(position, c) for c in range(P * P)]:
            job.result()
    # a saturated defect lights its nearest neighbours by crosstalk: the candidates and their 8 neighbours are
    # replaced by the neighbours' median before the stars are looked for, else a defect would be taken for one
    qc = np.where(cv2.dilate(cand.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0, med_all, q)
    # The published masks need shifted exposures: a defect stays on its pixel while the sky moves off it. Where a
    # star is in the LOWER QUARTILE (it covers the pixel in at least three frames of four) the sky has not left, and
    # a pixel there cannot be told from the star by persistence. Those pixels are not judged here; the stack's
    # rejection, which works on the registered frames (shifted by construction), is left to them.
    star = _star_footprints(qc, P)
    held = cand & star & ~bad
    hot = cand & ~star & ~bad
    bad |= hot
    n_new, n_held = int(hot.sum()), int(held.sum())
    return (f"hot pixels, persistence: {n_new} more pixels flagged, standing out from their neighbours in the lower "
            f"quartile of {len(sub)} frames spread over the session; each is replaced by the mean of its same-colour "
            f"neighbours. {n_held} more stand out where a star covers the pixel in at least three frames of four (the "
            f"frames hardly moved there): not judged by persistence, left to the stack's rejection")


def _star_footprints(img: np.ndarray, P: int) -> np.ndarray:
    """The pixels of the detected stars of ``img`` (the lower quartile of the lights, its isolated defects already
    replaced by their neighbours' median): SExtractor's detection through sep at 5 sigma of the background's noise (the usual star detection level of
    DAOFIND; at SExtractor's 1.5 the fixed pattern of the sensor's warm pixels is detected as objects), 5 pixels at
    least, the 3 x 3 convolution, on the image averaged over the colour pattern's cell; the footprint is the objects' segmentation, holes filled."""
    import sep
    from scipy.ndimage import binary_fill_holes
    lum = np.ascontiguousarray(cv2.blur(img, (P, P)), np.float32)
    bkg = sep.Background(lum, bw=64, bh=64, fw=3, fh=3)
    from ..core.imgutil import sep_extract
    obj, seg = sep_extract(lum - bkg.back(), 5.0, err=bkg.globalrms, minarea=5, segmentation_map=True)
    # a star is round: a line of hot pixels along a row or column of the sensor (a Canon 40D: elongation 70 to 150,
    # half of its pixels flagged) is a defect pattern, not a sky that stayed (stars: 1.0 to 1.2). No source gives the
    # limit; the two kinds are two orders of magnitude apart, so that 2 is no tuning
    keep = np.zeros(len(obj) + 1, bool)
    keep[1:] = obj["a"] < 2.0 * np.maximum(obj["b"], 1e-3)
    return binary_fill_holes(keep[seg])


def siril_hot_pixels(x: np.ndarray, P: int = 2, sigma: float = SIRIL_HOT_SIGMA):
    """Siril's cosmetic correction without darks, hot pixels, on one light (src/filters/cosmetic_correction.c of Siril
    1.5.0, ``autoDetect`` with the cold detection off, Amount 1; documented at
    https://siril.readthedocs.io/en/stable/processing/cc.html). With the whole frame's median ``bkg`` and average
    deviation from it ``avgDev`` (``statistics`` STATS_BASIC | STATS_AVGDEV, gsl_stats_absdev_m), a pixel is a candidate
    above ``bkg + avgDev``; it is hot when it also exceeds ``m + max(avgDev, sigma * avgDev)``, ``m`` being the median
    of the 24 pixels of the 5 x 5 block of its colour around it (the same-colour pixels, 2 apart: the CFA option), and
    when the mean ``a`` of its 8 same-colour neighbours (in the frame) stays under ``m + avgDev / 2``: a star's core
    lights its neighbours too. All is read in the frame as it comes (the corrections are made on a copy), and the pixel
    is replaced by ``a``. Returns (rows, columns, the new values). A pattern of period ``P`` gives each of its
    positions a plane of its own, whose pixels are the same colour. NaN pixels are neither read nor corrected."""
    ok = np.isfinite(x)
    bkg = float(np.median(x[ok]))
    avg_dev = float(np.mean(np.abs(x[ok] - bkg)))
    k2, k4 = avg_dev / 2, max(avg_dev, sigma * avg_dev)
    ys, xs, vals = [], [], []
    for c in range(P * P):
        v = _plane(x, c, P)
        cand = np.argwhere(v > bkg + avg_dev)                   # only these need the 24 neighbours
        if not len(cand):
            continue
        pad = np.pad(v, 2, constant_values=np.nan)              # outside the frame: not a neighbour
        for i in range(0, len(cand), 200_000):
            cy, cx = cand[i:i + 200_000].T
            win = np.stack([pad[cy + 2 + dy, cx + 2 + dx] for dy in range(-2, 3) for dx in range(-2, 3)
                            if dy or dx], 1)                    # (n, 24)
            full = ~np.isnan(win).any(1)
            m = np.empty(len(cy), np.float32)
            if full.any():
                part = np.partition(win[full], (11, 12), axis=1)
                m[full] = (part[:, 11] + part[:, 12]) / 2       # median24 (the mean of the two middle values)
            if (~full).any():                                   # the 2 outer rows and columns, fewer neighbours
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    m[~full] = np.nanmedian(win[~full], axis=1)
            near = win[:, [6, 7, 8, 11, 12, 15, 16, 17]]        # the 8 nearest of the 24 (3 x 3 minus the centre)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                a = np.nanmean(near, axis=1)
            hot = (v[cy, cx] > m + k4) & (a < m + k2)
            ys.append(cy[hot] * P + c // P)
            xs.append(cx[hot] * P + c % P)
            vals.append(a[hot])
    if not ys:
        return np.zeros(0, int), np.zeros(0, int), np.zeros(0, np.float32)
    return np.concatenate(ys), np.concatenate(xs), np.concatenate(vals).astype(np.float32)


def _sky_noise(path: str, dark: np.ndarray, flat: np.ndarray, P: int = 2) -> float:
    x = (read_cfa(path) - dark) / flat
    step = max(1, 4 * 2 // P)                         # about the same samples per plane as on a Bayer sensor
    return float(np.median([robust_sigma(_plane(x, k, P)[::step, ::step])[1] for k in range(P * P)]))


def _flat_view(flat: np.ndarray, cfa: str) -> np.ndarray:
    """Flat field shown as a colour image normalised to its maximum (vignetting and dust)."""
    rgb = np.moveaxis(cfa_mod.superpixel(flat, cfa), -1, 0)   # frames already in colour: their three planes (issue #57)
    return np.clip(rgb / np.percentile(rgb, 99.9), 0, 1)


# ---------------------------------------------------------------------- worker processes
_G: dict = {}


def _init(cache, n, H, W, paths_dir, sat_raw, scale, hot="off", cfa="RGGB", plain=False, n_darks=1):
    d = Path(paths_dir)
    bad = np.load(d / "bad.npy")
    darks = [np.load(d / "dark.npy")] + [np.load(d / f"dark_{g}.npy") for g in range(1, n_darks)]
    _G.update(darks=darks, flat=np.load(d / "flat.npy"), bad=bad,
              mm=FrameWriter(cache, n, (H, W), plain), sat=sat_raw, scale=scale,
              fill=_neighbours(bad, cfa), cfa=cfa, hot=hot)


def _neighbours(bad: np.ndarray, cfa: str = "RGGB"):
    """(rows, columns) of the bad pixels and of their 8 same-colour neighbours (2 px away; clipped at the edges). On
    an X-Trans sensor, whose same-colour neighbours are not on a lattice of step 2: the 8 nearest of its colour within
    2 px (each 3 x 3 window holds every colour, so a 5 x 5 one holds 2 red and 2 blue at least, besides the centre)."""
    H, W = bad.shape
    ys, xs = np.nonzero(bad)
    if cfa_mod.is_xtrans(cfa):
        g = cfa_mod.channels(cfa)
        near = sorted(((dy, dx) for dy in range(-2, 3) for dx in range(-2, 3) if dy or dx),
                      key=lambda o: (o[0] ** 2 + o[1] ** 2, o))
        table = np.zeros((6, 6, 8, 2), int)
        for i in range(6):
            for j in range(6):
                same = [o for o in near if g[(i + o[0]) % 6, (j + o[1]) % 6] == g[i, j]][:8]
                same += [same[-1]] * (8 - len(same))           # the nearest repeated: a median of the same samples
                table[i, j] = same
        off = table[ys % 6, xs % 6]                            # (n, 8, 2)
        ny, nx = ys[:, None] + off[..., 0], xs[:, None] + off[..., 1]
    else:
        off = [(dy, dx) for dy in (-2, 0, 2) for dx in (-2, 0, 2) if dy or dx]
        ny = np.stack([ys + dy for dy, _ in off], 1)
        nx = np.stack([xs + dx for _, dx in off], 1)
    ny = np.where((ny < 0) | (ny >= H), ys[:, None], ny)
    nx = np.where((nx < 0) | (nx >= W), xs[:, None], nx)
    return ys, xs, ny, nx


def _calib_one(args):
    idx, path, g = args                     # g: the light's dark among those of the session's exposures
    raw = read_cfa(path)
    sat = raw >= _G["sat"]
    x = (raw - _G["darks"][g]) / _G["flat"] * _G["scale"]
    x = np.minimum(x, VALID_MAX - 1)
    # an excluded pixel is filled with the mean of its same-colour neighbours that are not excluded nor saturated (a
    # saturated one is marked below, read here as it is): Siril's replacement. NaN again when every neighbour is bad
    ys, xs, ny, nx = _G["fill"]
    x[_G["bad"]] = np.nan
    nb = np.where(sat[ny, nx], np.nan, x[ny, nx])
    n_ok = np.isfinite(nb).sum(1)
    x[ys, xs] = np.where(n_ok > 0, np.nansum(nb, axis=1) / np.maximum(n_ok, 1), np.nan)
    n_hot = 0
    if _G.get("hot") == "per_frame":
        # Siril's cosmetic correction, before the saturated pixels are marked: a saturated defect is corrected, not
        # taken for the core of a star
        hy, hx, hv = siril_hot_pixels(x, cfa_mod.period(_G.get("cfa") or "RGGB"))
        x[hy, hx] = hv
        sat[hy, hx] = False
        n_hot = len(hy)
    # a bad pixel stays out of the saturated ones even when it saturates: marked saturated, the stack took a hot pixel
    # for the core of a bright star and drew the drift of the mount with it (green dotted trails on the Flaming Star)
    x[sat & ~_G["bad"]] = SAT_MARK
    _G["mm"][idx] = x
    stats = []
    for v in _stat_samples(x, _G.get("cfa") or "RGGB"):
        stats.append(_frame_stat(v))
    return idx, stats, n_hot


def _frame_stat(v: np.ndarray) -> tuple[float, float]:
    """(level, noise) of a frame's samples; NaN for a frame with too few usable ones (a white frame, taken on purpose
    to mark where a series starts): Selection leaves it out."""
    ok = np.isfinite(v) & (v < VALID_MAX)
    return robust_sigma(v[ok]) if ok.mean() >= MIN_USABLE else (np.nan, np.nan)


def _stat_samples(x: np.ndarray, cfa: str) -> list[np.ndarray]:
    """The four sets of samples frame_stats measures (level, noise): the four Bayer positions (one in four of their
    samples each way); on an X-Trans sensor its red, green (twice: the same four entries as a Bayer sensor's, so that
    their mean weighs the colours alike) and blue samples, about as many."""
    if not cfa_mod.is_xtrans(cfa):
        return [_plane(x, k)[::4, ::4] for k in range(4)]
    sub = x[::5, ::5]                                  # a step prime to 6: every position of the pattern
    col = cfa_mod.colour_map(cfa, *x.shape)[::5, ::5]
    r, g, b = (sub[col == c] for c in range(3))
    return [r, g, g, b]


def open_cache(ctx: Context, mode: str = "r"):
    """(memmap (N, H, W) float16, cache info) of the calibrated frames."""
    info = ctx.load_json("calibrate", "cache.json")
    mm = np.memmap(info["path"], dtype=np.float16, mode=mode, shape=(info["n"], info["height"], info["width"]))
    return mm, info


# ---------------------------------------------------------------------- frames already in colour (issue #57)
#: what a colour frame's masters are, colour by colour, besides the luminance (H, W) kept under the usual names (the
#: stack's saturation level, Inspect)
MASTER_RGB = ("master_dark_rgb.npy", "master_flat_rgb.npy")


def open_colour(info: dict) -> np.memmap:
    """(memmap (N, 3, H, W) float16) of the three planes of the calibrated frames already in colour (``info``:
    cache.json)."""
    return np.memmap(info["rgb_path"], dtype=np.float16, mode="r", shape=(info["n"], 3, info["height"], info["width"]))


class CachedFrames:
    """The calibrated frames of a cache (cache.json ``info``; ``colour``: the three planes of frames already in
    colour), frame i as float32 (a new array) by ``frames[i]``, one thread at a time.

    Each frame is one explicit read of its bytes into a float16 buffer reused from frame to frame (issue #125). Through
    a memory map of the cache, the stack's three passes brought every frame in by page faults, and its pages stayed in
    the process's working set until the system trimmed them: on a night of 754 frames (a 17 GB cache), the stack's
    working set reached 19.9 GB, only 8.2 GB of it private, on a machine of 27.7 GB whose free memory stayed under
    1 GB, about 4,000 pages read back from the disk a second. Read this way, the cache's pages are only in the
    system's file cache, which it gives up first; the file is open only during a read. The values are the same to the
    bit: the same float16, converted to float32 the same way. ``mapped``: through the memory map as before (the tests
    compare both). ``seconds``: the time spent reading the frames, for the passes' lines in the log."""

    def __init__(self, info: dict, colour: bool = False, mapped: bool = False):
        self.path = info["rgb_path"] if colour else info["path"]
        self.n = int(info["n"])
        self.shape = (3, info["height"], info["width"]) if colour else (info["height"], info["width"])
        self.mm = np.memmap(self.path, dtype=np.float16, mode="r", shape=(self.n, *self.shape)) if mapped else None
        self.buf = None if mapped else np.empty(self.shape, np.float16)
        self.seconds = 0.0

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i) -> np.ndarray:
        t = time.perf_counter()
        i = int(i)
        if not 0 <= i < self.n:
            raise IndexError(f"frame {i} of a cache of {self.n}")
        if self.mm is not None:
            out = np.asarray(self.mm[i], np.float32)
        else:
            view = memoryview(self.buf).cast("B")
            with open(self.path, "rb", buffering=0) as f:
                f.seek(i * view.nbytes)
                got = 0
                while got < view.nbytes:                        # a read may return less than asked: the rest after
                    k = f.readinto(view[got:])
                    if not k:
                        raise OSError(f"{self.path}: frame {i} cut short (a cache of {self.n} frames expected)")
                    got += k
            out = self.buf.astype(np.float32)
        self.seconds += time.perf_counter() - t
        return out


class FrameWriter:
    """Where a calibration worker writes its frames into the cache ``path`` ((n, *shape) float16): ``writer[i] = x``
    writes frame i, x converted to float16 as ``astype`` does.

    Through a memory map (``plain`` False, as before), each frame written stays in the worker's working set as modified
    pages of the map until the system writes them out and trims them: on a night of 796 frames (a 19 GB cache) on a
    machine of 27.7 GB, Calibration's used memory reached 19 GB, 0.4 GB free and 18,000 pages read back from the disk
    a second. With ``plain``, each frame is converted into a float16 buffer reused from frame to frame and written by
    one plain write: the system's file cache holds it until it is on the disk and the workers hold nothing of the
    cache (measured on 16 threads, a pool of 12, frames of 12 Mpx: the working set of the lights' phase 4.9 GB for 32
    frames and 6.0 GB for 96 through the map, 4.3 and 4.4 GB written plainly, the same private memory). The file is the
    same to the bit. Plain writes can wait for the disk sooner (a test of writing alone: 3.25 GB in 28 to 43 s, 5 to
    7 s into a map), so they are used only when memory is short (``memory_short``)."""

    def __init__(self, path, n: int, shape: tuple[int, ...], plain: bool):
        self.path, self.plain = str(path), plain
        self.mm = None if plain else np.memmap(path, dtype=np.float16, mode="r+", shape=(n, *shape))
        self.buf = np.empty(shape, np.float16) if plain else None

    def __setitem__(self, i: int, x: np.ndarray) -> None:
        if self.mm is not None:
            self.mm[i] = x.astype(np.float16)
            return
        np.copyto(self.buf, x, casting="unsafe")             # the float16 of astype, into the same buffer
        view = memoryview(self.buf).cast("B")
        with open(self.path, "r+b", buffering=0) as f:
            f.seek(int(i) * view.nbytes)
            done = 0
            while done < view.nbytes:                         # a write may take less than given: the rest after
                done += f.write(view[done:])


#: what a calibration worker holds per pixel of a frame, besides the cache (the masters, the frame read and calibrated,
#: their temporary arrays): measured 52 bytes a pixel on a 12 Mpx sensor, 7.6 GB for a pool of 12 (issue #125)
WORKER_BYTES = 52


def memory_short(cache_bytes: int, workers: int, pixels: int) -> float | None:
    """The memory available (GB) when the calibrated cache (``cache_bytes``) and a pool of ``workers`` on frames of
    ``pixels`` do not fit in the memory the engine may fill now (core/devices.memory_budget): the pools then write
    and read the cache plainly (``FrameWriter``, ``CachedFrames``) instead of through memory maps. None when they fit:
    a machine with the room keeps the maps, as before (no slower)."""
    m = devices.memory_budget()
    if m is None or cache_bytes + workers * WORKER_BYTES * pixels <= m[0]:
        return None
    return m[1] / 2 ** 30


def plain_writes(ctx: Context, cache_bytes: int, workers: int, pixels: int) -> bool:
    """Whether Calibration's workers write the frames plainly (``memory_short``), said in one line when they do."""
    gb = memory_short(cache_bytes, workers, pixels)
    if gb is not None:
        ctx.log(f"memory: {gb:.1f} GB available, the calibrated frames ({cache_bytes / 2 ** 30:.1f} GB) are written "
                f"straight to the disk")
    return gb is not None


def calibrate_colour(ctx: Context, p: CalibrateParams, frames: dict) -> None:
    """The calibration of frames already in colour, channel by channel (issue #57): the master dark, bias and flat of
    their own three planes, each frame's dark taken off and its flat divided by colour by colour; written as a
    luminance cache (N, H, W) (what the registration reads) and the planes beside it (``_colour_cache``). The frames
    are not linear when they were stretched (every JPEG is): the correction is then approximate (said by the session
    and the inventory). Hot pixels are not looked for in the lights: a frame already demosaiced has spread them over
    their neighbours."""
    H, W = int(ctx.metric_of("inventory", "height")), int(ctx.metric_of("inventory", "width"))
    black = float(ctx.metric_of("inventory", "black_level"))
    white = float(ctx.metric_of("inventory", "white_level"))
    if not p.apply_masters:
        ctx.log("Calibration cut: the frames are taken as they are (no master dark, flat or defective pixels)")
        ctx.metric("calibration_cut", True)
        _colour_cache(ctx, p, frames["lights"], np.full((3, H, W), black, np.float32), np.ones((3, H, W), np.float32),
                      np.zeros((H, W), bool), white, time.time())
        return
    ms = given_masters_of(ctx, frames)                # as for mosaics: a master given is used as it is
    nm = sum(len(frames[k]) for k in ("bias", "darks", "flats", "flat_darks"))
    ctx._files_total, ctx._files_read, ctx._lights_from = nm, 0, nm / max(nm + len(frames["lights"]), 1)
    ctx.phase(0.0, ctx._lights_from)
    bias = (_masters_mean(ms["bias"], ctx, "master bias", read_colour) if "bias" in ms else
            _colour_median([f["path"] for f in frames["bias"]], ctx, "bias") if frames["bias"] else None)
    if bias is not None:
        ctx.save("master_bias.npy", cfa_mod.luminance(bias))
    if "darks" in ms:
        dark = _masters_mean(ms["darks"], ctx, "master dark", read_colour)
    elif frames["darks"]:
        dark = _colour_median([f["path"] for f in frames["darks"]], ctx, "darks")
        one = cfa_mod.luminance(read_colour(frames["darks"][0]["path"]))
        ctx.metric("dark_frame_noise", round(robust_sigma(one[::3, ::3])[1], 2))
    else:
        dark = bias.copy() if bias is not None else np.full((3, H, W), black, np.float32)
        ctx.log("no dark frames: using " + ("the master bias" if bias is not None else "the black level") +
                " as dark (hot pixels will not be removed by calibration)", "warning")
    if frames["darks"]:
        check_dark(ctx, dark, bias, black)
    dark = scaled_dark(ctx, dark, bias, black, frames)
    flat = _colour_flat([f["path"] for f in frames["flats"]], [] if "flats" in ms else
                        [f["path"] for f in frames["flat_darks"]], None if "flats" in ms else bias,
                        np.zeros_like(dark) if "flats" in ms else dark, H, W, p.flat_scale_px, ctx)
    ctx.log("hot pixels seen in the lights: not looked for in frames already in colour (demosaicing has spread them)")
    level = np.median(dark, axis=(1, 2), keepdims=True)
    t0 = time.time()
    sky = read_colour(frames["lights"][len(frames["lights"]) // 2]["path"])
    noise = float(np.median([robust_sigma(((sky - dark) / flat)[c, ::4, ::4])[1] for c in range(3)]))
    bad = np.zeros((H, W), bool)
    if frames["darks"]:
        bad |= ((dark - level) > p.hot_sigma * noise).any(0)
    from scipy.ndimage import median_filter
    for c in range(3):
        loc = median_filter(flat[c], size=9)
        bad |= (flat[c] < (1 - p.flat_defect) * loc) | (flat[c] > (1 + p.flat_defect) * loc)
        bad |= flat[c] < FLAT_UNLIT * float(np.median(flat[c]))
    ctx.metric("bad_pixels_pct", round(100 * float(bad.mean()), 2))
    ctx.metric("sky_noise_adu", round(noise, 2))
    ctx.metric("dark_level", round(float(np.median(dark)), 1))
    ctx.metric("vignetting_min", round(float(flat.min()), 3))
    _colour_cache(ctx, p, frames["lights"], dark, flat, bad, white, t0)


def _colour_median(paths: list[str], ctx: Context, label: str) -> np.ndarray:
    return _shared(ctx, "colour_median_" + label.replace(" ", "_"), paths, lambda: _colour_median_built(paths, ctx, label))


def _colour_median_built(paths: list[str], ctx: Context, label: str) -> np.ndarray:
    """The pixel-wise median (3, H, W) of frames already in colour, in bands of rows (as ``_median_stack``)."""
    first = read_colour(paths[0])
    stack = np.empty((len(paths), *first.shape), np.float32)
    stack[0] = first
    _tick(ctx, f"reading {label} 1/{len(paths)}")

    def put(i, a):
        stack[i] = a

    _read_frames(paths, put, 1, lambda c: _tick(ctx, f"reading {label} {c}/{len(paths)}"), read_colour)
    out = np.empty(stack.shape[1:], np.float32)
    by_bands(out.shape[1], lambda r0, r1: out.__setitem__((slice(None), slice(r0, r1)),
                                                          np.median(stack[:, :, r0:r1], axis=0, overwrite_input=True)))
    return out


def _colour_flat(paths: list[str], fd_paths: list[str], bias: np.ndarray | None, dark: np.ndarray, H: int, W: int,
                 flat_scale_px: float, ctx: Context) -> np.ndarray:
    """The master flat (3, H, W) of frames already in colour, each colour normalised to 1 at its centre; all ones
    without flats. Each flat over its own central level, per colour, then ``_flats_mean``."""
    if not paths:
        ctx.log("no flat frames: vignetting and dust will not be corrected (the sky model will absorb part of "
                "the vignetting)", "warning")
        return np.ones((3, H, W), np.float32)
    if fd_paths:
        fdark = _colour_median(fd_paths, ctx, "flat darks")
    elif bias is not None:
        fdark = bias
    else:
        fdark = np.broadcast_to(np.median(dark, axis=(1, 2), keepdims=True), (3, H, W)).astype(np.float32)
    fl = np.empty((len(paths), 3, H, W), np.float32)

    def put(i, a):
        fl[i] = a - fdark

    _read_frames(paths, put, 0, lambda c: _tick(ctx, f"reading flats {c}/{len(paths)}"), read_colour)
    y0, y1, x0, x1 = H // 2 - 200, H // 2 + 200, W // 2 - 200, W // 2 + 200
    fl /= np.median(fl[:, :, max(y0, 0):y1, max(x0, 0):x1], axis=(2, 3), keepdims=True)
    flat = np.stack([_flats_mean(fl[:, c]) for c in range(3)])
    for c in range(3):
        if flat_scale_px > 0:
            flat[c] = cv2.GaussianBlur(flat[c], (0, 0), flat_scale_px)
        flat[c] /= np.median(flat[c, max(y0, 0):y1, max(x0, 0):x1])
    if flat_scale_px > 0:
        ctx.log(f"master flat kept above {flat_scale_px:.0f} px only (vignetting, not dust)")
    return flat


def _colour_cache(ctx: Context, p: CalibrateParams, lights: list[dict], dark: np.ndarray, flat: np.ndarray,
                  bad: np.ndarray, white: float, t0: float) -> None:
    """The calibrated frames already in colour: their luminance in ``lights.f16`` (N, H, W), what the registration
    reads as a monochrome mosaic, their three planes in ``lights_rgb.f16`` (N, 3, H, W), read by the stack; the masters
    colour by colour (MASTER_RGB) and their luminance under the usual names."""
    _, H, W = dark.shape
    level = float(np.median(dark))
    scale = float(min(1.0, 16000.0 / max(white - level, 1.0)))
    n = len(lights)
    cache, rgb = ctx.scratch / "lights.f16", ctx.scratch / "lights_rgb.f16"
    ctx.save("master_dark.npy", cfa_mod.luminance(dark))
    ctx.save("master_flat.npy", cfa_mod.luminance(flat))
    ctx.save("badmask.npy", bad)
    for name, a in zip(MASTER_RGB, (dark, flat)):
        ctx.save(name, a)
    mm = np.memmap(cache, dtype=np.float16, mode="w+", shape=(n, H, W))
    mr = np.memmap(rgb, dtype=np.float16, mode="w+", shape=(n, 3, H, W))
    del mm, mr
    d = ctx.ckpt
    np.save(d / "dark.npy", dark)
    np.save(d / "flat.npy", flat)
    np.save(d / "bad.npy", bad)
    stats = np.zeros((n, 4, 2), np.float32)
    plain = plain_writes(ctx, n * H * W * 8, p.workers, 3 * H * W)        # the luminance and the three planes
    init = (str(cache), str(rgb), n, H, W, str(d), p.saturation * white, scale, plain)
    ctx.phase(getattr(ctx, "_lights_from", 0.0), 1.0)
    ctx.progress(0.0, f"calibrating {n} frames ({p.workers} processes)", force=True)
    with devices.one_maths_thread():                           # issue #125: 1 GB less reserved a worker
        pool = get_context("spawn").Pool(p.workers, initializer=_init_colour, initargs=init)
    with pool:
        for c, (idx, st) in enumerate(pool.imap_unordered(_calib_colour, list(enumerate(f["path"] for f in lights)),
                                                          chunksize=2)):
            stats[idx] = st
            ctx.progress((c + 1) / n, f"{c + 1}/{n} frames calibrated")
    for f in ("dark.npy", "flat.npy", "bad.npy"):
        (d / f).unlink(missing_ok=True)
    ctx.save("frame_stats.npy", stats)
    # the cache's pattern is its luminance's: the stages that walk it read one monochrome mosaic (core/cfa.MONO)
    ctx.save_json("cache.json", {"path": str(cache), "rgb_path": str(rgb), "n": n, "height": H, "width": W, "scale": scale,
                                 "cfa": cfa_mod.MONO, "colour": True, "sat_mark": SAT_MARK, "valid_max": VALID_MAX,
                                 "white_level": white, "dark_level": level})
    ctx.metric("cache_scale", scale)
    ctx.metric("cache_gb", round(n * H * W * 8 / 1e9, 1))
    ctx.log(f"{n} frames already in colour calibrated in {(time.time() - t0) / 60:.1f} min")
    from .inventory import _superpixel_rgb
    ctx.preview("master_flat", _flat_view(flat, cfa_mod.RGB))
    fr = np.nan_to_num(CachedFrames({"rgb_path": str(rgb), "n": n, "height": H, "width": W}, colour=True)[n // 2])
    ctx.preview("calibrated_frame", _superpixel_rgb(np.where(fr > VALID_MAX, 0, fr), cfa_mod.RGB))


def _init_colour(cache, rgb, n, H, W, paths_dir, sat_raw, scale, plain=False):
    d = Path(paths_dir)
    _G.update(dark=np.load(d / "dark.npy"), flat=np.load(d / "flat.npy"), bad=np.load(d / "bad.npy"),
              mm=FrameWriter(cache, n, (H, W), plain), rgb=FrameWriter(rgb, n, (3, H, W), plain), sat=sat_raw,
              scale=scale)


def _calib_colour(args):
    """One frame already in colour, calibrated colour by colour: its three planes and its luminance written, a
    saturated sample marked (the luminance too: a star clipped in one colour), the bad pixels left out."""
    idx, path = args
    raw = read_colour(path)
    sat = raw >= _G["sat"]
    x = (raw - _G["dark"]) / _G["flat"] * _G["scale"]
    x = np.minimum(x, VALID_MAX - 1)
    x[sat] = SAT_MARK
    y = cfa_mod.luminance(np.where(sat, 0.0, x))
    y[sat.any(0)] = SAT_MARK
    x[:, _G["bad"]] = np.nan
    y[_G["bad"]] = np.nan
    _G["rgb"][idx] = x
    _G["mm"][idx] = y
    stats = []
    for c in (0, 1, 1, 2):                                  # the same four entries as a Bayer sensor's (their mean)
        stats.append(_frame_stat(x[c, ::4, ::4]))
    return idx, stats

