"""The Post of the client, without its widgets: layers, masks, filters and the composition.

The images come from a ``Provider`` (the engine's layers, see ``services/render.py``), the composition is numpy and
runs in the render worker, never in the GUI thread.

Layers (a list, top first, as in the reference image editor; saved as it is in ``project.post["layers"]``):

* a workflow layer ``{id, kind: "workflow", pid, part, blend, name, visible, opacity, mode, base, sliders, mask}``:
  an image of a workflow, rendered by the engine with its ``base`` (a named rendering of the project, or None for the
  suggestion) moved by its simplified ``sliders`` (positions 0-4, 2 = the base). ``part`` None: the Output block's
  image; else one of the Stars block's (``PARTS``): its starless image, its stars alone, or the two recombined (the
  Stars and background block: ``blend`` says how its stars go back, render/stars.BLENDS; ``shown`` which of its two
  images are put together: issue #3, its two images are sub-layers of their own, each with its filters). ``stretch``
  (held only when it is not the automatic one: "arcsinh") is the base stretch of its linear image, for the image the
  stars are not (``LAYER_STRETCHES``);
* an image layer ``{id, kind: "image", key, name, visible, opacity, mode, sliders, linked, mask}``: the linear image of
  a stage result (a stack, a gradient-removed or a denoised image) sent from Inspect, stretched automatically and moved
  by the Sky, Contrast and Saturation sliders; ``key`` names the stage result (the pixels are never saved);
* a filter layer ``{id, kind: "filter", filter, name, visible, opacity, mode, params, mask, target}``: one of the
  filters (``FILTERS``: each a published or documented method, named in its ``source``; postfilters/ holds the new
  ones), applied to the one pixel layer (workflow or image) ``target`` names, or to everything below (``target`` is
  None). In a Stars and background layer, ``sub`` names the one of its two images (``SUBS``) the filter applies to,
  before they are put together; None: to the recombined image.

The stars of a Combine layers layer are a group (issue #93), in its ``blend`` over the starless image: the real stars
at its bottom (their eye: ``shown["real"]``), and, when the workflow has the Synthetic stars block, a Synthetic stars
layer beside them (``is_synth``: saved as a filter row of the stars, ``filter`` "synth", whose image is the block's
stars), composed in Normal through its mask over what is under it in the group; the group's other rows are filters on
what is under them in it. The panel shows the group as ``rows`` says, the composition reads it from the same rows.

The model is the reference image editor's: a filter that applies to one layer is that layer's child; a filter that applies to
everything below is a layer of its own. In the list a layer's children sit just above it, top child first; the panel
shows each pixel layer with its children indented under it (``rows``), and reading the panel from the bottom up is the
order of the composition: a pixel layer, then its children from the lowest to the highest, then the next row up.

A Combine layers layer may also have ``side_masks = {"stars": mask, "starless": mask}`` (either key absent: no mask on
that side): each is applied to its side's image, after that side's filters and before the two are put together, so
the stars (or the background) are absent where it is black. An absent field is no side mask (older projects).

A Combine layers layer has a layer tree (``nest``, given by the model from the graph: client/postfusion.py,
graphcore/post.mix_tree; never saved): a side that is a combination of filters (S-H-O, H-O, the recipes mixed through hues
or a matrix; the stars' combination too) is a mix Post makes again, any combination that reads another combination is a
mix inside it, and a combination Post cannot remake (a block after it, its depth, a recipe composed as lightness and
colour) is a fixed image, said on its row. Each node of the tree (its path names it in every saved key:
``postfusion.child``, "f:SII", "f:Ha/f:OIII", "stars/f:R") is a sub-image of its own under its parent, with its eye
(``shown``), its mask (``side_masks``) and its filters (``sub``), which act on it before its parent mixes it. A mix's hues
and intensities are its own (``tune`` for the starless side's, ``tunes`` for the others: written once changed). The
filters are stretched by one factor per pixel, the same for the whole mix at any depth (client/postmix.py;
render/simple.SimpleRenderEngine.colour_factor): a factor is linear, so the colour ratios keep their meaning.

Every layer may have a ``mask = {parts, feather, density, invert}`` of parts (H-alpha, luminance, color, gradient,
brush), each added, subtracted or intersected. Pixel sizes are full-resolution pixels: at a reduced resolution they are
divided by the reduction factor, so the draft looks like the full image.

Resolutions: level 0 is the full canvas (only a region of it, with a 64 px margin for the filters), level 1 the half
(the screen), level 2 the quarter (the draft while something moves), level 3 the eighth (a draft on a slow machine).
The canvas is the area every workflow layer covers, taken on multiples of 8 px so that the levels line up.

``Composer`` composes the layers (one per render worker: it owns the caches); ``Provider`` is what it reads the
images from; ``FILTERS`` lists the filters and their settings.
"""
from __future__ import annotations

import copy
import hashlib
import itertools
import json
import math
import time
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from ..render.sliders import top as slider_top
from . import postfusion as pf
from .postfilters import colour as pf_colour
from .postfilters import curves as pf_curves
from .postfilters import dehaze as pf_dehaze
from .postfilters import detail as pf_detail
from .postfilters import heal as pf_heal
from .postfilters import mixer as pf_mixer
from .postfilters import tone as pf_tone

MODES = ["Normal", "Lighten", "Screen", "Linear dodge (add)", "Multiply", "Soft light"]   # plain English: _() where shown
#: the Stars block's images a workflow layer can show (render/stars.PARTS), and how the recombined one puts its stars
#: back (render/stars.BLENDS)
PARTS = ("starless", "stars", "combined")
#: the sub-layers of a Stars and background layer, top first (the stars go over the starless image)
SUBS = ("stars", "starless")
BLENDS = ("Screen", "Linear dodge (add)", "Lighten")
OPS = {"add": "Add", "sub": "Subtract", "inter": "Intersect"}
ALIGN = 8                                  # the canvas starts and ends on a multiple of this (the deepest level)
MAX_LEVEL = 3
REGION_MARGIN = 64                         # full-resolution pixels around a region, so that the filters see neighbours
THUMB = (54, 36)                           # the layer and mask thumbnails (width, height)
_ids = itertools.count(1)


def new_id() -> int:
    return next(_ids)


def with_new_ids(layers: list[dict]) -> list[dict]:
    """A copy of these layers with ids of their own (a filter follows its layer, a mask's parts are renumbered)."""
    out = copy.deepcopy(layers)
    ids = {lay["id"]: new_id() for lay in out}
    for lay in out:
        lay["id"] = ids[lay["id"]]
        if lay.get("target") in ids:
            lay["target"] = ids[lay["target"]]
        for part in (lay.get("mask") or {}).get("parts", []):
            part["id"] = new_id()
        for m in (lay.get("side_masks") or {}).values():
            for part in m.get("parts", []):
                part["id"] = new_id()
    return out


def reseed_ids(layers: list[dict]) -> None:
    """Make the next new id greater than every id of these layers (after loading a saved composition)."""
    global _ids
    top = 0
    for lay in layers:
        top = max(top, int(lay.get("id", 0)))
        for m in [lay.get("mask") or {}, *(lay.get("side_masks") or {}).values()]:
            for p in m.get("parts", []):
                top = max(top, int(p.get("id", 0)))
    _ids = itertools.count(max(top, next(_ids) - 1) + 1)


class Superseded(Exception):
    """A newer request replaced the one being computed: leave at once."""


# ======================================================================== small image helpers
_LUM = np.array([[0.2126, 0.7152, 0.0722]], np.float32)


def lum(x: np.ndarray) -> np.ndarray:
    """The luminance of an RGB image (h, w, 3) -> (h, w), float32."""
    if x.dtype == np.float32 and x.ndim == 3 and x.flags.c_contiguous:
        return cv2.transform(x, _LUM)
    return (0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2]).astype(np.float32)


def _gray3(a: np.ndarray) -> np.ndarray:
    """A (h, w) image as (h, w, 3), the same value in each channel."""
    return cv2.merge([a, a, a])


def _mix(a: np.ndarray, wa: float, b: np.ndarray, wb: float) -> np.ndarray:
    """wa * a + wb * b in one pass (two float32 images of the same shape)."""
    return cv2.addWeighted(_f32(a), float(wa), _f32(b), float(wb), 0.0)


def _torch(x: np.ndarray):
    import torch
    return torch.from_numpy(_f32(x))


def hue(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Hue in degrees and saturation (0..1) of an RGB image."""
    mx, mn = x.max(-1), x.min(-1)
    d = mx - mn + 1e-6
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    h = np.where(mx == r, ((g - b) / d) % 6, np.where(mx == g, (b - r) / d + 2, (r - g) / d + 4)) * 60
    return h, d / (mx + 1e-6)


def hue_weight(h, center, width):
    dist = np.abs((h - center + 180) % 360 - 180)
    return np.clip(1 - dist / max(width, 1), 0, 1)


_f32 = pf_colour.f32                  # a C-contiguous float32 array (no copy when it is one already)


def gauss(x: np.ndarray, sigma: float) -> np.ndarray:
    """A Gaussian blur of an (h, w) or (h, w, c) float image, reflecting at the borders. A wide blur is made on a
    reduced copy (a Gaussian is smooth: the result is the same within a fraction of a level)."""
    if sigma < 0.3:
        return x
    x = _f32(x)
    one = x.ndim == 3 and x.shape[2] == 1
    a = x[..., 0] if one else x
    if sigma > 12:
        m = int(sigma // 6)
        h, w = a.shape[:2]
        small = cv2.resize(a, (max(1, w // m), max(1, h // m)), interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (0, 0), sigma / m, borderType=cv2.BORDER_REFLECT)
        out = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        out = cv2.GaussianBlur(a, (0, 0), sigma, borderType=cv2.BORDER_REFLECT)
    return out[..., None] if one else out


# ======================================================================== filters
HUES = [("All", None), ("Reds", 0), ("Yellows", 60), ("Greens", 120), ("Cyans", 180), ("Blues", 230),
        ("Magentas", 300)]


def P(key, label, lo, hi, default, unit="", step=None, help=""):                   # noqa: A002 - the catalogue's word
    return dict(key=key, label=label, type="float", lo=lo, hi=hi, default=default, unit=unit, step=step, help=help)


def C(key, label, choices, default):
    return dict(key=key, label=label, type="choice", choices=choices, default=default)


def B(key, label, default):
    return dict(key=key, label=label, type="bool", default=bool(default))


def CURVE(key):
    """The curves of the Curves tool: {channel: {"pts", "interp"}}; {} is the identity (postfilters.curves)."""
    return dict(key=key, label="Curves", type="curve", default={})


def SPOTS(key):
    """The spots of Spot healing: [[x, y, r, dx, dy], ...] in full-resolution canvas pixels (postfilters.heal)."""
    return dict(key=key, label="Spots", type="spots", default=[])


def clean_spots(v) -> list[list[float]]:
    out = []
    for s in v if isinstance(v, (list, tuple)) else []:
        try:
            x, y, r, dx, dy = (float(t) for t in s)
        except (TypeError, ValueError):
            continue
        if all(math.isfinite(t) for t in (x, y, r, dx, dy)) and 1 <= r <= 2000:
            out.append([round(x, 1), round(y, 1), round(r, 1), round(dx, 1), round(dy, 1)])
    return out


def f_stretch(x, p, ctx):
    """An arcsinh stretch around the symmetry point (the transcendental on all cores, through torch)."""
    d, sp = p["strength"], p["symmetry"]
    if d < 1e-3:                                       # the default: no stretch at all
        return x
    a0 = float(np.arcsinh(-d * sp))                    # python floats: a numpy scalar would promote the image to float64
    a1 = float(np.arcsinh(d * (1 - sp)))
    t = (_torch(x) - sp).mul_(d).asinh_().sub_(a0).mul_(1.0 / (a1 - a0))
    return t.numpy()


def f_curves(x, p, ctx):
    """The Curves tool (postfilters.curves): a table while a point moves, exact on release."""
    return pf_curves.apply(x, p["curves"], draft=ctx.draft)


def curves_of_old(sh: float, mid: float, hi: float) -> dict:
    """The curves of a composition saved before the Curves tool (three sliders moving the curve at .25, .5 and .75):
    the same piecewise-linear curve, so that it renders as it did."""
    if not (sh or mid or hi):
        return {}
    ys = np.clip([0, 0.25 + sh, 0.5 + mid, 0.75 + hi, 1], 0, 1)
    return {"RGB/K": {"pts": [[x, float(y)] for x, y in zip((0.0, 0.25, 0.5, 0.75, 1.0), ys)], "interp": "Linear"}}


def f_levels(x, p, ctx):
    if p["black"] <= 0 and p["white"] >= 1 and abs(p["gamma"] - 1) <= 1e-9:      # the default: the identity
        return x
    t = (_torch(x) - p["black"]).mul_(1.0 / max(p["white"] - p["black"], 1e-3)).clamp_(0, 1)
    if abs(p["gamma"] - 1) > 1e-9:
        t = t.pow_(1.0 / p["gamma"])
    return t.numpy()


def per_pixel(fn, *arrays, min_rows: int = 256):
    """``fn`` on bands of rows of ``arrays`` (images of the same height) in the processor's threads, the bands put
    back together: the same pixels as ``fn`` on the whole, for a function of each pixel alone (numpy and OpenCV give
    the processor back while they compute)."""
    rows = arrays[0].shape[0]
    n = min(_THREADS, max(1, rows // min_rows))
    if n <= 1:
        return fn(*arrays)
    import threading
    cuts = [rows * i // n for i in range(n + 1)]
    parts: list = [None] * n
    errors: list = []

    def band(i):
        try:
            parts[i] = fn(*(a[cuts[i]:cuts[i + 1]] for a in arrays))
        except BaseException as e:  # noqa: BLE001 - raised again in the caller's thread
            errors.append(e)
    threads = [threading.Thread(target=band, args=(i,), name="post-pixels", daemon=True) for i in range(1, n)]
    for t in threads:
        t.start()
    band(0)
    for t in threads:
        t.join()
    if errors:
        raise errors[0]
    return np.concatenate(parts, 0)


_THREADS = max(1, min(16, int(__import__("os").environ.get("ASTRO_GPU_POST_THREADS") or 0) or
                    (__import__("os").cpu_count() or 1)))


def _gauss_margin(sigma: float) -> int | None:
    """The rows a ``gauss`` of ``sigma`` reads around a pixel (OpenCV's kernel: 4 sigma each side); None for a wide
    blur, made on a reduced copy whose grid is the whole image's (not a band's)."""
    if sigma < 0.3:
        return 0
    return None if sigma > 12 else int(math.ceil(4 * sigma)) + 2


def _starlet_margin(n: int) -> int:
    """The rows n starlet layers read around a pixel (the B3 kernel's 2 x 2^j each, client/postfilters/detail)."""
    return 2 * (2 ** (n + 1)) + 8


#: the filters whose result on a band of rows, given the rows it reads around it, is the whole image's bit for bit:
#: (params, level) -> those rows, None when not (a wide blur). Each is checked against the whole image by the tests
#: (tests/client/test_post_bands.py); a filter not here (a whole-image measure, a grid of the whole image: CLAHE, the
#: veil map, the noise reduction's noise, the background guard) is made on the whole, and so is a filter computed by
#: torch (Levels, the multiscale ones): its elementwise functions take another path at the end of each thread's share,
#: so that its pixels follow how the work is cut.
BAND_MARGINS = {
    "balance": lambda p, lv: 0,
    "curves": lambda p, lv: 0,
    "scnr": lambda p, lv: 0,
    "denoise": lambda p, lv: _gauss_margin(4 / 2 ** lv),
    "sharpen": lambda p, lv: _gauss_margin(max(p["radius"] / 2 ** lv, 0.35)),
    "local_contrast": lambda p, lv: _gauss_margin(p["scale"] / 2 ** lv),
    "faint_smooth": lambda p, lv: int(math.ceil(4 * max(p["radius"] / 2 ** lv, 0.5))) + 2,
    "highlights": lambda p, lv: 3 * max(1, int(round(p["radius"] / 2 ** lv))) + 8,
}
#: a band's least rows (under it, the filter is made on the whole: the threads would cost more than they give)
BAND_ROWS = 128


def run_filter(key: str, x: np.ndarray, p: dict, ctx) -> np.ndarray:
    """A filter on an image: in bands of rows in the processor's threads when it can be (``BAND_MARGINS``: the same
    pixels as on the whole), else on the whole. Between two bands a composition abandoned stops (``Superseded``):
    a full-resolution render never holds the next draft for a whole heavy filter."""
    fn = FILTERS[key]["fn"]
    margin_of = BAND_MARGINS.get(key)
    rows = x.shape[0]
    margin = margin_of(p, getattr(ctx, "level", 0)) if margin_of is not None and x.ndim == 3 else None
    n = 0 if margin is None else min(4 * _THREADS, rows // max(BAND_ROWS, margin))
    if n < 2 or _THREADS < 2:
        return fn(x, p, ctx)
    check = getattr(ctx, "check", None)
    cuts = [rows * i // n for i in range(n + 1)]
    out = np.empty(x.shape, np.float32)
    import threading
    nxt = [0]
    lock = threading.Lock()
    errors: list = []

    def worker():
        while True:
            with lock:
                if errors or nxt[0] >= n:
                    return
                i = nxt[0]
                nxt[0] += 1
            try:
                if check is not None and check():
                    raise Superseded()
                a, b = max(0, cuts[i] - margin), min(rows, cuts[i + 1] + margin)
                y = fn(np.ascontiguousarray(x[a:b]), p, ctx)
                out[cuts[i]:cuts[i + 1]] = y[cuts[i] - a:cuts[i + 1] - a]
            except BaseException as e:  # noqa: BLE001 - raised again in the caller's thread
                with lock:
                    errors.append(e)
                return
    k = min(_THREADS, n)
    with _one_thread_each():
        threads = [threading.Thread(target=worker, name="post-bands", daemon=True) for _ in range(k - 1)]
        for t in threads:
            t.start()
        worker()
        for t in threads:
            t.join()
    if errors:
        raise errors[0]
    return out


class _one_thread_each:
    """While bands run in threads, the libraries' own threads one each (the processor shared, not each band asking
    for all of it): torch's and OpenCV's thread counts set back after."""
    _depth = 0

    def __enter__(self):
        import torch
        _one_thread_each._depth += 1
        if _one_thread_each._depth == 1:
            self._torch, self._cv = torch.get_num_threads(), cv2.getNumThreads()
            torch.set_num_threads(1)
            cv2.setNumThreads(1)
        return self

    def __exit__(self, *a):
        import torch
        _one_thread_each._depth -= 1
        if _one_thread_each._depth == 0:
            torch.set_num_threads(self._torch)
            cv2.setNumThreads(self._cv)
        return False


def f_saturation(x, p, ctx):
    k = p["strength"] / 100
    if k == 0:                                         # the default: no change
        return x
    L = lum(x)
    center = dict(HUES)[p["hues"]]
    guard = background_guard(L, p.get("protect", 0.0), ctx)    # (measured on the whole image)
    if center is None and guard is None:               # every hue alike: L + (x - L) * (1 + k) in one pass
        return _mix(x, 1 + k, _gray3(L), -k)
    return per_pixel(lambda xa, La, ga: _saturate(xa, La, ga, center, k), x, L,
                     guard if guard is not None else np.empty((x.shape[0], 0), np.float32), min_rows=64)


def _saturate(x, L, guard, center, k):
    """``f_saturation`` on pixels: the weight of each (its hue's, its guard's), then the saturation around ``L``."""
    w = np.ones_like(L)
    if center is not None:
        h, s = hue(x)
        w = hue_weight(h, center, 45) * np.clip(s * 3, 0, 1)
    if guard.size:
        w = w * guard
    L = L[..., None]
    return L + (x - L) * (1 + k * w[..., None])


def background_guard(L, factor: float, ctx=None):
    """Issue #23: the saturation of the astronomy software's colour tools spares the background: only the pixels
    brighter than ``factor`` x (the median + one sigma of the image's luminance) change, so that the noise of the
    background is not coloured. A smooth start over 20 % around that level; None at 0 (everything changes). The
    median and sigma are the whole image's (``ctx.whole_stats``), also when a region is rendered."""
    if factor <= 0:
        return None
    ws = getattr(ctx, "whole_stats", None)                # (a filter applied outside a composition: its own image)
    med, sig = ws("background", L, _median_sigma) if ws is not None else _median_sigma(L)
    thr = factor * (med + sig)
    return np.clip((L - 0.8 * thr) / max(0.4 * thr, 1e-6), 0, 1).astype(np.float32)


def _median_sigma(L) -> tuple[float, float]:
    sub = L[::4, ::4] if L.size > 4_000_000 else L
    med = float(np.median(sub))
    return med, 1.4826 * float(np.median(np.abs(sub - med)))


def mixer_params() -> list[dict]:
    """The Colour mixer's settings: for each band its hue (degrees), saturation and luminance (percent)."""
    words = {"hue": "hue", "sat": "saturation", "lum": "luminance"}
    out = []
    for k in pf_mixer.keys():
        what, band = k.split("_")
        top = pf_mixer.HUE_MAX if what == "hue" else 100
        out.append(P(k, f"{band.capitalize()} · {words[what]}", -top, top, 0, "°" if what == "hue" else "%", 1))
    return out


def f_mixer(x, p, ctx):
    return pf_mixer.mix(x, p, _THREADS)


def f_scnr(x, p, ctx):
    """SCNR, the reference program's and the reference processing suite's four protections (postfilters.colour.scnr)."""
    return pf_colour.scnr(x, p["method"], p["strength"], bool(p["lightness"]))


def f_ghs(x, p, ctx):
    return pf_tone.ghs(x, p)


def f_asinh(x, p, ctx):
    return pf_tone.asinh(x, p)


def f_histogram(x, p, ctx):
    return pf_tone.histogram_transform(x, p)


def f_clahe(x, p, ctx):
    """CLAHE; on a region (100 %) the tile grid is the whole canvas's, so that it looks as in the whole image."""
    ax0, ay0 = getattr(ctx, "abox", None)[:2] if getattr(ctx, "abox", None) is not None else (0, 0)
    return pf_detail.clahe(x, p["amount"], p["clip"], p["radius"], ctx.level, (ax0, ay0),
                           getattr(ctx, "canvas_wh", None))


def f_highlights(x, p, ctx):
    return pf_detail.highlights(x, p["strength"], p["knee"], p["detail"], p["radius"], ctx.level)


def f_faint_smooth(x, p, ctx):
    return pf_detail.faint_smoothing(x, p["amount"], p["threshold"], p["radius"], ctx.level)


def _whole_measure(ctx, name, x, measure):
    """``Ctx.whole_measure``; (None, the level) for a filter applied outside a composition."""
    wm = getattr(ctx, "whole_measure", None)
    return wm(name, x, measure) if wm is not None else (None, getattr(ctx, "level", 0))


def f_dehaze(x, p, ctx):
    if p["amount"] <= 0:
        return x
    if not float(getattr(ctx, "level", 0)).is_integer() and getattr(ctx, "canvas_wh", None) is not None:
        # the screen's picture (or a draft) between two powers of two: always the whole canvas; its meshes measured
        # on it reduced to the next power of two (a measure kept at this level would serve no region: theirs are at
        # whole levels; measured here too, it took 120 ms of a 300 ms draft)
        coarse = math.ceil(ctx.level)
        W, H = ctx.comp_canvas.size(coarse) if getattr(ctx, "comp_canvas", None) is not None else (
            max(1, round(x.shape[1] / 2 ** (coarse - ctx.level))), max(1, round(x.shape[0] / 2 ** (coarse - ctx.level))))
        return pf_dehaze.dehaze_between(x, p["amount"], p["scale"] / ctx.f, 2 ** (coarse - ctx.level), (H, W))
    m = pf_dehaze.mesh_of(p["scale"] / ctx.f)
    got, at = _whole_measure(ctx, f"veil {p['scale']}", x, lambda a: (pf_dehaze.background_grid(a, m), m, a.shape[:2]))
    if got is None or at == ctx.level:               # the whole image: its own map, as always
        return pf_dehaze.dehaze(x, p["amount"], p["scale"] / ctx.f)
    g, gm, hw = got                                   # a region: the whole image's map, measured smaller
    return pf_dehaze.take_off(x, p["amount"], pf_dehaze.map_in_region(g, gm, hw, 2 ** (at - ctx.level), ctx.abox,
                                                                      x.shape))


def f_heal(x, p, ctx):
    return pf_heal.heal(x, p["spots"], 1.0 / ctx.f, (ctx.abox[0], ctx.abox[1]))


ML_KEYS = ("l1", "l2", "l3", "l4", "l5")
ML_LABELS = ("1 px", "2 px", "4 px", "8 px", "16 px")


def f_ml_sharpen(x, p, ctx):
    return pf_detail.sharpen(x, [p[k] for k in ML_KEYS], p["target"], ctx.level)


#: the noise reduction's layers: 1 to 8 px on the first layer's noise, 16 to 64 px each on its own
ML_NR_KEYS = ML_KEYS + ("l6", "l7")
ML_NR_LABELS = ML_LABELS + ("32 px", "64 px")


def _nr_coarse(p) -> bool:
    return float(p.get("amount", 0.0)) > 0 and any(float(p.get(k, 0)) > 0 for k in ML_NR_KEYS[4:])


def f_ml_denoise(x, p, ctx):
    ks = [float(p.get(k, 0)) for k in ML_NR_KEYS]
    sig, at = None, ctx.level
    if p["amount"] > 0 and any(t > 0 for t in ks[:4]):
        sig, at = _whole_measure(ctx, f"noise {p['target']}", x, lambda a: pf_detail.noise_of(a, p["target"]))
    if sig is not None and at != ctx.level:          # a white noise halves at each halving of the resolution
        sig = [s * 2 ** (at - ctx.level) for s in sig]
    own = None
    if _nr_coarse(p):                                # a coarse layer's noise is the same at any resolution
        lv = ctx.level
        own, _at = _whole_measure(ctx, f"noise layers {p['target']}", x,
                                  lambda a: pf_detail.layer_noise_of(a, p["target"], lv))
    return pf_detail.denoise(x, ks, p["amount"], p["target"], ctx.level, sigma=sig, own=own)


def _nr_margin(p) -> float:
    """The reach of the coarsest layer the noise reduction changes (the starlet's support: 2 (2^n - 1) px)."""
    if not _nr_coarse(p):
        return 0
    n = max(i + 1 for i, k in enumerate(ML_NR_KEYS) if float(p.get(k, 0)) > 0)
    return 2 * (2 ** n - 1)


def f_balance(x, p, ctx):
    t, g = p["temperature"], p["tint"]
    if not (t or g):                                   # the default: neutral
        return x
    return x * np.array([1 + 0.15 * t, 1 + 0.1 * g, 1 - 0.15 * t], np.float32)


def f_local_contrast(x, p, ctx):
    k = p["strength"]
    if k == 0:                                         # the default: no change
        return x
    return _mix(x, 1 + k, gauss(x, p["scale"] / ctx.f), -k)            # x + k * (x - blur)


def f_sharpen(x, p, ctx):
    k = p["strength"]
    if k == 0:                                         # the default: no change
        return x
    return _mix(x, 1 + k, gauss(x, max(p["radius"] / ctx.f, 0.35)), -k)


def f_denoise(x, p, ctx):
    if not (p["luminance"] or p["color"]):             # the default: nothing smoothed
        return x
    Y = lum(x)
    C_ = cv2.subtract(_f32(x), _gray3(Y))                              # the colour, what is left without the luminance
    a, b = p["luminance"], p["color"]
    Y2 = _mix(Y, 1 - a, gauss(Y, 1.6 / ctx.f), a)
    C2 = _mix(C_, 1 - b, gauss(C_, 4 / ctx.f), b)
    return cv2.add(_gray3(Y2), C2)


def f_halpha(x, p, ctx):
    if p["strength"] == 0:                             # the default: no red added
        return x
    y = x.copy()
    y[..., 0] = x[..., 0] + p["strength"] * 0.35 * ctx.halpha() * (1 - x[..., 0])
    return y


GHS = ("The generalised hyperbolic stretch (D. Payne and M. Cranfield): a strong stretch around a chosen level, the "
       "shadows and highlights protected")
FILTERS = {
    # ---- stretch
    "ghs": dict(name="Generalised hyperbolic stretch", family="Stretch", fn=f_ghs, graph=pf_tone.ghs_curve, hist=True,
                source=GHS,
                params=[P("stretch", "Stretch factor", 0, 4, 0, step=0.01,
                          help="ln(D+1) in the formula of this stretch."),
                        P("local", "Local intensity", -5, 15, 0, step=0.1, help="b in the formula of this stretch."),
                        P("symmetry", "Symmetry point", 0, 1, 0, step=0.001),
                        P("shadows", "Shadow protection", 0, 1, 0, step=0.001),
                        P("highlights", "Highlight protection", 0, 1, 1, step=0.001),
                        P("black", "Black point", 0, 0.5, 0, step=0.001, help="BP in the formula of this stretch."),
                        C("model", "Color model", pf_tone.COLOUR_MODES, pf_tone.COLOUR_MODES[0])]),
    "asinh": dict(name="Arcsinh stretch", family="Stretch", fn=f_asinh, graph=pf_tone.asinh_curve, hist=True,
                  source="An arcsinh stretch (after Lupton et al. 2004): the colours kept",
                  params=[P("stretch", "Stretch factor", 0, 40, 0, step=0.1),
                          P("black", "Black point", 0, 0.2, 0, step=0.001),
                          C("model", "Luminance", pf_tone.COLOUR_MODES[2:0:-1], pf_tone.COLOUR_MODES[2])]),
    "histogram": dict(name="Histogram transformation", family="Stretch", fn=f_histogram, graph=pf_tone.ht_curve,
                      hist=True, source="Shadows, midtones and highlights, through the midtones transfer function",
                      params=[P("shadows", "Shadows", 0, 1, 0, step=0.001),
                              P("midtones", "Midtones", 0.001, 0.999, 0.5, step=0.001),
                              P("highlights", "Highlights", 0, 1, 1, step=0.001),
                              C("channel", "Channel", pf_tone.HT_CHANNELS, "RGB/K")]),
    # hidden: not in the filter menu, kept so that the compositions that hold it still open
    "stretch": dict(name="Arcsinh stretch (symmetry point)", family="Stretch", fn=f_stretch, hidden=True,
                    source="An arcsinh stretch around a symmetry point (the first Post's)",
                    params=[P("strength", "Strength", 0, 12, 0, step=0.1),
                            P("symmetry", "Symmetry point", 0, 0.6, 0.08, step=0.01)]),
    # ---- contrast and curves
    "curves": dict(name="Curves", family="Stretch", fn=f_curves, editor="curves", hist=True,
                   source="A curve per channel through control points",
                   params=[CURVE("curves")]),
    "levels": dict(name="Levels", family="Stretch", fn=f_levels,
                   source="Black point, white point, gamma",
                   params=[P("black", "Black point", 0, 0.3, 0, step=0.005),
                           P("white", "White point", 0.7, 1, 1, step=0.005),
                           P("gamma", "Gamma", 0.5, 2, 1, step=0.01)]),
    "local_contrast": dict(name="Local contrast", family="Contrast", fn=f_local_contrast,
                           source="A large-radius unsharp mask",
                           params=[P("strength", "Strength", 0, 1.5, 0, step=0.01),
                                   P("scale", "Scale", 5, 120, 40, "px", 1)]),
    "dehaze": dict(name="Dehaze", family="Contrast", fn=f_dehaze, margin=lambda p: 2 * p["scale"],
                   source="The veil (residual sky glow, haze) measured as the background of SExtractor (Bertin and "
                          "Arnouts 1996: meshes, 3-sigma clipping, mode, bicubic map) and taken off with He's haze "
                          "model (He, Sun, Tang 2009), the white kept",
                   params=[P("amount", "Amount", 0, pf_dehaze.AMOUNT_MAX, 0, step=0.01),
                           P("scale", "Scale", 32, 512, 128, "px", 1)]),
    "clahe": dict(name="Local histogram equalization", family="Contrast", fn=f_clahe,
                  source="CLAHE (Zuiderveld 1994): the contrast equalised tile by tile, with a limit, on the "
                         "lightness",
                  params=[P("amount", "Amount", 0, 1, 0, step=0.01),
                          P("clip", "Contrast limit", 1, 8, 2, step=0.1),
                          P("radius", "Kernel radius", 16, 512, 64, "px", 1)]),
    "highlights": dict(name="Highlight compression", family="Contrast", fn=f_highlights,
                       source="Brings the bright areas down above a start level and keeps their detail (a base and "
                              "its detail by the guided filter, He et al. 2013)",
                       params=[P("strength", "Strength", 0, 1, 0, step=0.01),
                               P("knee", "Start", 0.2, 0.9, 0.5, step=0.01),
                               P("detail", "Detail kept", 0, 1, 0.3, step=0.01),
                               P("radius", "Scale", 8, 120, 40, "px", 1)]),
    # ---- colour
    "saturation": dict(name="Saturation", family="Color", fn=f_saturation,
                       source="Saturation around the luminance, by hue range",
                       params=[P("strength", "Strength", -100, 150, 0, "%", 1),
                               C("hues", "Hues", [h for h, _ in HUES], "All"),
                               P("protect", "Background protection", 0, 3, 0, "", 0.1)]),
    "mixer": dict(name="Colour mixer", family="Color", fn=f_mixer,
                  source="Hue, saturation and luminance of eight colour ranges in CIE L*c*h*, as the colour mixers of "
                         "photo editors; the grey background and white star cores left alone",
                  params=mixer_params()),
    "scnr": dict(name="Green removal", family="Color", fn=f_scnr,
                 source="Removes a green cast: the green held under a neutral made of red and blue",
                 params=[P("strength", "Amount", 0, 1, 0, step=0.01),
                         C("method", "Protection", pf_colour.SCNR_METHODS, pf_colour.SCNR_METHODS[0]),
                         B("lightness", "Preserve lightness", True)]),
    "balance": dict(name="Color balance", family="Color", fn=f_balance,
                    source="Gains on red and blue (temperature) and on green (tint)",
                    params=[P("temperature", "Temperature", -3, 3, 0, step=0.01),
                            P("tint", "Green / magenta", -3, 3, 0, step=0.01)]),
    # ---- detail
    "sharpen": dict(name="Sharpen", family="Detail", fn=f_sharpen,
                    source="Unsharp mask",
                    params=[P("strength", "Strength", 0, 2, 0, step=0.01),
                            P("radius", "Radius", 0.5, 4, 1.2, "px", 0.1)]),
    "ml_sharpen": dict(name="Multiscale sharpening", family="Detail", fn=f_ml_sharpen,
                       source="Starlet wavelet layers (B3 spline a trous): each scale of detail raised or lowered "
                              "on its own",
                       params=[P(k, lb, -1, 3, 0, step=0.01) for k, lb in zip(ML_KEYS, ML_LABELS)] +
                       [C("target", "Applies to", pf_detail.TARGETS, pf_detail.TARGETS[0])]),
    # ---- noise
    "denoise": dict(name="Noise reduction", family="Noise", fn=f_denoise,
                    source="A Gaussian smoothing of the luminance and of the colour. Added at zero it changes nothing: raise "
                           "Luminance or Color",
                    params=[P("luminance", "Luminance", 0, 1, 0, step=0.01),
                            P("color", "Color", 0, 1, 0, step=0.01)]),
    "ml_denoise": dict(name="Multiscale noise reduction", family="Noise", fn=f_ml_denoise, margin=_nr_margin,
                       source="Soft thresholding of the starlet wavelet layers (Starck and Murtagh): the noise "
                              "lowered scale by scale",
                       params=[P(k, lb, 0, 10, 0, "σ", 0.1) for k, lb in zip(ML_NR_KEYS, ML_NR_LABELS)] +
                       [P("amount", "Amount", 0, 1, 1, step=0.01),
                        C("target", "Applies to", pf_detail.NR_TARGETS, pf_detail.NR_TARGETS[0])]),
    "faint_smooth": dict(name="Faint area smoothing", family="Noise", fn=f_faint_smooth,
                         source="Smooths the areas fainter than a level, the brighter ones untouched: the faint "
                                "extensions can then be raised without their grain. Added at zero it changes nothing: "
                                "raise Amount",
                         params=[P("amount", "Amount", 0, 1, 0, step=0.01),
                                 P("threshold", "Below", 0.05, 0.8, 0.3, step=0.01),
                                 P("radius", "Radius", 2, 64, 24, "px", 1)]),
    # ---- combine
    # ---- retouch
    "heal": dict(name="Spot healing", family="Detail", fn=f_heal, editor="spots",
                 margin=lambda p: pf_heal.margin(p["spots"]),
                 source="A spot replaced by a patch of its neighbourhood, seamlessly (Poisson image editing, Perez et "
                        "al. 2003, as the healing brush of photo editors); the source found automatically, movable",
                 params=[SPOTS("spots")]),
    # the Synthetic stars layer of a Stars group (is_synth, issue #93): never in the filter menu
    "synth": dict(name="Synthetic stars", family="Detail", fn=lambda x, p, ctx: ctx.synth() if ctx.synth else x, hidden=True,
                  source="The stars of the Synthetic stars block: where the mask is white they replace the real stars, "
                         "where it is painted black the real stars show",
                  params=[]),
    "halpha_red": dict(name="H-alpha to red", family="Combine", fn=f_halpha, needs="halpha",
                       source="The engine's H-alpha map added to the red channel",
                       params=[P("strength", "Strength", 0, 1, 0, step=0.01)]),
}
FAMILIES = ["Stretch", "Contrast", "Color", "Detail", "Noise", "Combine"]
#: a filter name older saved compositions hold (a layer keeps its name) -> the filter's name today
#: the part names older saved compositions hold in their layers' names -> today's (issue #36)
OLD_PART_NAMES = {"Stars and background": "Combine layers", "Combine stars & background layers": "Combine layers",
                  "Recombine": "Combine layers"}
OLD_NAMES = {"Stretch (GHS)": "Arcsinh stretch (symmetry point)",
             # issue #15: no acronyms in the names (their sources still cite the methods)
             "Local histogram equalization (CLAHE)": "Local histogram equalization",
             "SCNR (green removal)": "Green removal"}

# ======================================================================== masks
MASK_SOURCES = {
    "halpha": "H-alpha", "luminance": "Luminance range", "color": "Color range", "gradient": "Linear gradient",
    "brush": "Brush", "stars": "Stars",
}
#: the Stars part (``find_stars``) finds the stars of the layer's stars side as the combination finds those of its
#: continuum (stages/combine.continuum_scale): SExtractor's detection (sep), the background in 64 px meshes, the stars
#: above STAR_SIGMA times its noise (here 1.4826 median absolute deviations), four pixels at least, its deblending; a
#: star's size is its area above that threshold. It reads the stars as their block computed them (linear, at full
#: resolution, once per result: ``Composer.star_discs``), never as Post's star settings show them: Strength, Star size
#: or Star color change nothing in the mask and cost no detection, and the screen and the export draw the same discs
#: the least step of an 8-bit display: the noise taken for a stars image without any (a drawing)
DISPLAY_STEP = 1.0 / 255
DIRECTIONS = ["Top → bottom", "Bottom → top", "Left → right", "Right → left"]
#: the settings of a mask part: key → (label, lo, hi, unit); the defaults are those of ``new_part``
PART_SETTINGS = {
    "luminance": [("lo", "From", 0, 100, "%"), ("hi", "To", 0, 100, "%")],
    "color": [("hue", "Hue", 0, 359, "°"), ("width", "Width", 10, 120, "°")],
    "gradient": [("position", "Position", 0, 100, "%"), ("softness", "Softness", 1, 100, "%")],
    "stars": [("grow", "Grow", 1, 10, "×"), ("feather", "Feather", 0, 60, "px")],
}


def new_part(src: str) -> dict:
    p = {"id": new_id(), "src": src, "op": "add"}
    if src == "luminance":
        p.update(lo=40.0, hi=100.0)
    elif src == "color":
        p.update(hue=0.0, width=40.0)
    elif src == "gradient":
        p.update(direction=DIRECTIONS[0], position=50.0, softness=40.0)
    elif src == "brush":
        p.update(strokes=[], size=60.0, hardness=50.0, paint=True)
    elif src == "stars":
        p.update(grow=1.0, feather=0.0)                 # each star to its own extent, a hard edge: neutral
    elif src != "halpha":
        raise ValueError(f"unknown mask source {src!r}")
    return p


def grey3(x: np.ndarray) -> np.ndarray:
    """A grey image (h, w) as an (h, w, 3) one, each channel the grey: a master layer's filter as a layer's image. The
    same values as numpy's repeat on the last axis, made by OpenCV's merge (about three times faster: five filters of a
    16 Mpx image took 1.4 s of a release's full-resolution picture)."""
    x = np.ascontiguousarray(x, np.float32)
    return cv2.merge([x, x, x])


# the images kept are read only (``_frozen``); torch reads them without a copy, and no filter writes into its input
__import__("warnings").filterwarnings("ignore", message="The given NumPy array is not writable", category=UserWarning)


def _frozen(a: np.ndarray) -> np.ndarray:
    """``a`` made read only, kept as it is: the image a step, a backdrop or a layer keeps is handed on without a copy
    (a 27 Mpx image is 82 MB at the screen's level: a copy each filter took 0.2 s of a release); a filter that would
    write into its input raises instead of changing what is kept (each filter writes into images of its own)."""
    if isinstance(a, np.ndarray):
        a.flags.writeable = False
    return a


def _torch_of(a: np.ndarray):
    import torch
    return torch.from_numpy(a)


def _stretch_reads(s) -> str:
    """What the colour-preserving stretch of a master layer reads of its settings (render/simple.SimpleRenderEngine.
    colour_factor): the stretch, Sky, Faint tones, the Arcsinh's Strength, and Saturation for the Arcsinh's room."""
    d = {k: getattr(s, k, None) for k in ("stretch", "sky", "faint", "stretch_strength")}
    if d["stretch"] == "arcsinh":
        d["saturation"] = getattr(s, "saturation", None)
    return json.dumps(d, sort_keys=True, default=str)


def _finish_reads(s) -> str:
    """What the end of the stretch reads (render/simple.SimpleRenderEngine.finish): Contrast, Saturation."""
    return json.dumps({k: getattr(s, k, None) for k in ("contrast", "saturation")}, sort_keys=True, default=str)


class _SideHeld:
    """A side that is a mix of the layer tree before the end of its stretch, kept (``Composer._mix_side``): the mix
    (3, h, w) and its covered pixels (bool); ``nbytes`` for the cache."""
    __slots__ = ("y", "valid", "nbytes")

    def __init__(self, y: np.ndarray, valid: np.ndarray):
        self.y, self.valid = y, valid
        self.nbytes = y.nbytes + valid.nbytes


class _LazySources(Mapping):
    """The images a mask part may read (``Composer.mask``'s ``sources``), each made when a part first reads it."""

    def __init__(self, makers: dict):
        self._makers, self._got = makers, {}

    def __getitem__(self, k):
        if k not in self._got:
            self._got[k] = self._makers[k]()
        return self._got[k]

    def __contains__(self, k):
        return k in self._makers

    def __iter__(self):
        return iter(self._makers)

    def __len__(self):
        return len(self._makers)


#: the detection of a large stars image is made tile by tile: its time grows faster than its size (the owner's M27,
#: 16 Mpx, 23 700 stars: 48 s at once, the message "Finding the stars…" up for minutes while Post rendered; 2.7 s in
#: tiles of 512 px)
STAR_TILE = 512
STAR_TILE_SLACK = 32


def _tiled_extract(sub: np.ndarray, thresh: float, err: float, **kw):
    """sep's extraction (``core.imgutil.sep_extract``) of ``sub`` tile by tile: each tile extracted over its own box
    grown to every pixel group above the threshold that reaches into it (and ``STAR_TILE_SLACK`` px more), so that
    each object is found whole, with its neighbours, as on the whole image; an object belongs to the tile its centre is
    in. (Two extractions of the whole image do not give the same measures either: sep's deblending varies from one run
    to the next; the tiles stay within that.)"""
    from ..core.imgutil import sep_extract
    H, W = sub.shape
    if H * W <= 4 * STAR_TILE * STAR_TILE:
        return sep_extract(sub, thresh, err=err, **kw)
    over = cv2.dilate((sub > thresh * err).astype(np.uint8), np.ones((5, 5), np.uint8))
    _n, _lab, st, _c = cv2.connectedComponentsWithStats(over, connectivity=8)
    left, top, right, bottom = st[1:, 0], st[1:, 1], st[1:, 0] + st[1:, 2], st[1:, 1] + st[1:, 3]
    out = []
    m = STAR_TILE_SLACK
    for y0 in range(0, H, STAR_TILE):
        y1 = min(H, y0 + STAR_TILE)
        rows = (top < y1) & (bottom > y0)
        for x0 in range(0, W, STAR_TILE):
            x1 = min(W, x0 + STAR_TILE)
            sel = rows & (left < x1) & (right > x0)
            ya, yb, xa, xb = y0, y1, x0, x1
            if sel.any():
                ya, yb = min(ya, int(top[sel].min())), max(yb, int(bottom[sel].max()))
                xa, xb = min(xa, int(left[sel].min())), max(xb, int(right[sel].max()))
            ya, yb, xa, xb = max(0, ya - m), min(H, yb + m), max(0, xa - m), min(W, xb + m)
            o = sep_extract(np.ascontiguousarray(sub[ya:yb, xa:xb]), thresh, err=err, **kw)
            x, y = o["x"] + xa, o["y"] + ya
            o = o[(y >= y0) & (y < y1) & (x >= x0) & (x < x1)]
            o["x"] += xa
            o["y"] += ya
            out.append(o)
    return np.concatenate(out)


def find_stars(L: np.ndarray) -> np.ndarray:
    """(n, 3) float32: the stars of a stars image (``L``: its luminance at full resolution), each (x, y, r): found as
    the combination finds those of its continuum (the comment above: SExtractor's detection, eight sky sigmas), r its
    equivalent radius at that threshold, sqrt(area / pi), which a brighter star has larger (its profile stays above
    the threshold farther out). An image with no noise (a drawing): one display step of its brightest pixel taken for
    it."""
    import sep

    from ..stages.combine import STAR_SIGMA
    none = np.zeros((0, 3), np.float32)
    data = np.ascontiguousarray(L, np.float32)
    if data.size == 0 or not np.isfinite(data).all() or float(data.max()) <= float(data.min()):
        return none
    try:
        bkg = sep.Background(data, bw=64, bh=64)
        sub = data - bkg.back()
        v = sub[::4, ::4] if sub.size > 1_000_000 else sub
        rms = 1.4826 * float(np.median(np.abs(v - np.median(v))))     # the background's noise
        if rms <= 0:
            rms = DISPLAY_STEP * max(float(np.abs(sub).max()), 1e-12)
        objs = _tiled_extract(sub, STAR_SIGMA, rms, minarea=4, deblend_cont=0.005)
    except Exception:  # noqa: BLE001 - sep refuses the image (too small a box): no star
        return none
    if len(objs) == 0:
        return none
    return np.stack([np.asarray(objs["x"], np.float32), np.asarray(objs["y"], np.float32),
                     np.sqrt(np.asarray(objs["npix"], np.float32) / np.pi)], axis=1)


#: the stars found in a stars side's result (``Composer.star_discs``), shared by every composer of the session (Post's,
#: Inspect's editor's): {key: (n, 3)}
_STARS: OrderedDict = OrderedDict()
_STARS_LOCK = __import__("threading").Lock()
#: the searches running: {key: True}; each in a process of its own (client/starsearch.py: a search holds the
#: interpreter for long stretches, so it never runs in the window's process), waited for in a thread
_SEARCHING: dict = {}
#: called (from another thread) when a search ends: Post and Inspect's editor render again with the stars found
STARS_FOUND_HOOKS: list = []
#: what the picture says while the stars of a result are searched (the mask is drawn without them meanwhile)
STARS_SEARCHING = "Finding the stars of the Stars mask (once for this result): the mask shows them when found"
#: what it says when a search failed (the mask without its stars; searched again in the next session, never kept)
STARS_FAILED = "The stars of the Stars mask could not be found: the mask shows none (searched again next time)"
_FAILED: set = set()


def _search_stars(key, L, folder, name):
    """A search of stars, in a thread of the window: the process that finds them (client/starsearch.py), then what
    it found kept (for the session, on disk) and the composers told to draw again."""
    import tempfile

    from .starsearch import search
    try:
        hit = search(L, Path(folder) if folder is not None else Path(tempfile.gettempdir()) / "float_stars")
    except Exception:  # noqa: BLE001 - the search failed (its process ended): never kept as "no stars"
        with _STARS_LOCK:
            _FAILED.add(key)                              # said in the picture's line; searched again next session
            _SEARCHING.pop(key, None)
        for c in list(_COMPOSERS):
            c._stars_new = True
        for fn in list(STARS_FOUND_HOOKS):
            try:
                fn()
            except RuntimeError:
                pass
        return
    if folder is not None:
        try:
            Path(folder).mkdir(parents=True, exist_ok=True)
            np.save(Path(folder) / name, hit)
        except OSError:                                   # not kept on disk: searched again in the next session
            pass
    with _STARS_LOCK:
        _STARS[key] = hit
        _SEARCHING.pop(key, None)
        while len(_STARS) > 16:
            _STARS.popitem(last=False)
    for c in list(_COMPOSERS):
        c._stars_new = True
    for fn in list(STARS_FOUND_HOOKS):
        try:
            fn()
        except RuntimeError:                              # a view gone meanwhile
            pass


#: every composer, told when a search ends (its images made without the stars are made again)
_COMPOSERS = __import__("weakref").WeakSet()


def draw_stars(stars: np.ndarray, level: int, abox, grow: float, feather: float) -> np.ndarray:
    """The Stars mask part over the box at the level: 1 over each star (``find_stars``, full-resolution pixels of the
    canvas), a disc of radius ``grow`` r around its centre: at 1 its own extent, at 2 twice its radius. No source was
    found for a published law of a star mask's growth by the star's own size: this one is structural (a radius
    proportional to the star's measured size at the detection's threshold), nothing in it is fitted on an image.
    ``feather``: the edge smoothed by a Gaussian of that radius (px at the level), as the mask's own Feather."""
    x0, y0, x1, y1 = abox
    f = 2 ** level
    out = np.zeros((y1 - y0, x1 - x0), np.float32)
    g = max(float(grow), 1.0)
    for x, y, r in stars:
        cv2.circle(out, (int(round(float(x) / f - x0)), int(round(float(y) / f - y0))),
                   max(1, int(round(g * float(r) / f))), 1.0, -1, lineType=cv2.LINE_8)
    if feather > 0:
        out = gauss(out, feather)
    return np.clip(out, 0, 1).astype(np.float32)


def stars_cover(L: np.ndarray, grow: float, feather: float) -> np.ndarray:
    """The Stars part drawn over a whole stars image, found in it (``find_stars`` then ``draw_stars``)."""
    return draw_stars(find_stars(L), 0, (0, 0, L.shape[1], L.shape[0]), grow, feather)


def new_mask() -> dict:
    return {"parts": [], "feather": 0.0, "density": 100.0, "invert": False}


def is_synth(lay: dict) -> bool:
    """The Synthetic stars layer of a Combine layers layer's Stars group (issue #93): a sibling of the real stars in the
    panel. It is kept as a row nested in the stars (``sub`` "stars", see the module's docstring) so that its mask,
    eye, opacity and mode are a layer's as any other; its image is the Synthetic stars block's, composed in Normal over
    what is under it in the group: where its mask is white the synthetic stars replace the real ones, never add to
    them."""
    return lay.get("kind") == "filter" and lay.get("filter") == "synth"


def new_synth_layer(gid: int) -> dict:
    """A Synthetic stars layer in the Stars group of the layer ``gid``. Its mask starts open for both methods (issue
    #93; the image's shape started closed, the instrument's open): the synthetic stars replace all the real ones, the
    block's effect shows at once; its brush subtracts, so that what is painted brings the real stars back."""
    row = new_filter_layer("synth", gid, "stars")
    row["mask"] = dict(new_mask(), parts=[dict(new_part("brush"), op="sub")])
    return row


def stars_side(lay: dict):
    """The workflow whose stars a Combine layers layer puts back: its stars side's (``stars_pid``: a stars row, or a
    combination of stars), else its own; None when the side is unwired. This is the workflow a Synthetic stars block
    of the layer lives in."""
    return lay.get("stars_pid", lay.get("pid"))


def with_synth_rows(layers: list[dict], wanted: dict[int, str]) -> list[dict]:
    """The layers with one Synthetic stars layer in the Stars group of each combined layer whose stars side's workflow
    (``stars_side``: its own when the layer has no other) has the block (``wanted``: {workflow id: its method}), none
    in the others. A new one goes at the bottom of the group's rows above
    the real stars (applied first: the filters of the group then act on the stars it leaves). A composition saved
    before the group (#93) holds the same row, its mask and settings: it is kept as it is, so the image is the same;
    one put on the starless image by an older drag goes back to the stars, a second one in a group goes."""
    out, seen = [], set()
    for lay in layers:
        if is_synth(lay):
            t = next((x for x in layers if x["id"] == lay.get("target")), None)
            if t is None or not is_group(t) or stars_side(t) not in wanted or t["id"] in seen:
                continue
            seen.add(t["id"])
            if lay.get("sub") != "stars":
                lay = dict(lay, sub="stars")
        if is_group(lay) and stars_side(lay) in wanted and not any(
                is_synth(x) and x.get("target") == lay["id"] for x in layers):
            out.append(new_synth_layer(lay["id"]))
        out.append(lay)
    return out


def has_synth(layers: list[dict], gid: int) -> bool:
    """Whether the Stars group of the layer ``gid`` holds a Synthetic stars layer (then its real stars are a row of
    their own, with their eye)."""
    return any(is_synth(x) and x.get("target") == gid for x in layers)


def new_workflow_layer(pid: int, name: str, visible=False, part=None, base=None) -> dict:
    """A workflow's layer: its Output block's image (``part`` None) or one of its Stars block's images (``PARTS``). The
    stars alone go in Screen mode (they are black around the stars); the recombined image puts them back in Screen."""
    if part is not None and part not in PARTS:
        raise ValueError(f"unknown layer part {part!r} (allowed: {PARTS})")
    return {"id": new_id(), "kind": "workflow", "pid": pid, "part": part, "name": name, "visible": visible,
            "opacity": 100.0, "mode": "Screen" if part == "stars" else "Normal", "base": base, "sliders": {},
            "mask": None,
            **({"blend": BLENDS[0], "shown": {k: True for k in SUBS}, "starless": "starless",
                              "stars_pid": pid} if part == "combined" else {})}


def new_image_layer(key: str, name: str, visible: bool = True) -> dict:
    """A layer of a stage result's image (``key`` names it; the pixels come from the render worker's sources)."""
    return {"id": new_id(), "kind": "image", "key": str(key), "name": name, "visible": visible, "opacity": 100.0,
            "mode": "Normal", "sliders": {}, "linked": True, "mask": None}


PIXEL = ("workflow", "image")
#: the base stretch of a workflow layer (render/simple.STRETCHES), applied to its linear image: the automatic screen
#: transfer, or the colour-preserving arcsinh stretch (Lupton et al. 2004). A layer holds ``stretch`` only when it is
#: not the automatic one (a composition saved before the choice, or left at Automatic, is unchanged); the Arcsinh's
#: Strength is the layer's slider ``STRETCH_KEY`` (0-4, 2 = its starting strength)
LAYER_STRETCHES = ("auto", "arcsinh")
STRETCH_NAMES = {"auto": "Automatic", "arcsinh": "Arcsinh"}
STRETCH_KEY = "stretch"


def layer_stretch(lay: dict) -> str:
    """The base stretch of a workflow layer: "auto" unless it holds another one."""
    v = lay.get("stretch")
    return v if v in LAYER_STRETCHES[1:] else "auto"


def output_stretch(layers: list[dict], pid: int) -> dict:
    """The base stretch of the layer that shows a workflow's Output image in these layers (the first one: its own
    layer, or the Combine layers layer whose starless image is that Output), as the fields a layer holds for it:
    ``{"stretch": ..., "sliders": {STRETCH_KEY: ...}}``; {} for Automatic, or when no layer shows it. The Output
    block's picture takes it, so that it stays Post's layer at its default settings whichever stretch is chosen."""
    for lay in layers or []:
        if lay.get("kind") != "workflow" or lay.get("pid") != pid:
            continue
        if lay.get("part") is None or (lay.get("part") == "combined" and lay.get("starless") == "output"):
            if layer_stretch(lay) == "auto":
                return {}
            pos = (lay.get("sliders") or {}).get(STRETCH_KEY)
            return {"stretch": lay["stretch"], "sliders": {} if pos is None else {STRETCH_KEY: float(pos)}}
    return {}


def stretch_tag(st: dict) -> str:
    """A short text for a base stretch that is not the automatic one ("" for it), for the keys of what is kept by
    stretch (``output_stretch``'s answer)."""
    if not st:
        return ""
    return f"+{st['stretch']}:{st['sliders'].get(STRETCH_KEY, 2.0):.3f}"


def set_layer_stretch(lay: dict, kind: str) -> None:
    """The base stretch chosen for a workflow layer: Automatic leaves no trace in it (nor its Strength)."""
    if kind in LAYER_STRETCHES[1:]:
        lay["stretch"] = kind
    else:
        lay.pop("stretch", None)
        (lay.get("sliders") or {}).pop(STRETCH_KEY, None)
#: the sliders of an image layer (its automatic stretch; the simple rendering's Sky, Contrast and Saturation, 0-4)
IMAGE_SLIDERS = (("sky", "Sky", ("darker", "brighter")), ("contrast", "Contrast", ("softer", "stronger")),
                 ("saturation", "Saturation", ("paler", "more vivid")))


def is_pixel(lay: dict) -> bool:
    """A layer with an image of its own (a workflow's or a stage result's), which filters can be nested in."""
    return lay.get("kind") in PIXEL


def source_id(lay: dict):
    """What the composition asks the provider for: a workflow's id (its Output block's image), ("stars", id) for its
    Stars block's images, or ("image", key). A Combine layers layer asks for its stars side's (``sub_layer``)."""
    if lay["kind"] == "workflow":
        if lay.get("part") == FUSION:                    # a mix of the layer tree and its filters (``sub_layer``)
            n = mix_at(lay, pf.root_of(lay.get("node") or "starless"))
            return (FUSION, n["pid"] if n is not None else lay["pid"])
        if lay.get("part") == "combined" and "stars_pid" in lay:
            return ("stars", lay["stars_pid"]) if lay["stars_pid"] is not None else ("stars", lay["pid"])
        return ("synth" if lay.get("part") == "synth" else "stars", lay["pid"]) if lay.get("part") else lay["pid"]
    return ("image", lay["key"])


def sub_layer(lay: dict, sub: str) -> dict:
    """The layer to render for one image of a Combine layers layer (issue #165, two branches): its stars are its stars
    side's workflow's stars (``stars_pid``); its starless image is its first workflow's Stars/background starless image,
    or its Output (a combination's: ``starless`` "output"); a starless image that is a mix of the layer tree is its
    filters mixed (``FUSION``, ``node`` "starless"), and each node under it an image of its own (``node``: its
    path, postfusion.child). A part of any other layer is the layer's own."""
    if sub == "starless" and mix_at(lay, "starless") is not None or sub not in SUBS and sub in nodes_of(lay):
        return dict(lay, part=FUSION, node=sub)                  # a node of the starless mix, as that mix makes it
    if sub in ("stars", "synth"):
        sp = lay["stars_pid"] if lay.get("stars_pid") is not None else lay["pid"]
        if lay.get("stars_image") == "output" and lay.get("stars_pid") is not None:
            # a combination of stars: its Output's image (the Synthetic stars block's, redrawn from it, in the same
            # layout), stretched with the statistics of the starless combination they go onto (stars_over), so that the
            # synthetic stars replace the real ones with no jump
            return dict(lay, pid=sp, part="synth" if sub == "synth" else None, stars_over=lay["pid"])
        return dict(lay, pid=sp, part=sub)
    return dict(lay, part=None if lay.get("starless") == "output" and sub == "starless" else sub)


def source_ids(lay: dict) -> list:
    """Everything the provider must have for a layer: a Combine layers layer's two sides (the shown ones, the stars
    side when it has a workflow), else its own."""
    if is_group(lay):
        shown = lay.get("shown") or {}
        subs = [x for x in ("starless", "stars") if shown.get(x, True)]
        if lay.get("stars_pid", lay["pid"]) is None and "stars" in subs:
            subs.remove("stars")
        return [source_id(sub_layer(lay, x)) for x in subs]
    return [source_id(lay)]


def side_layers(lay: dict) -> list:
    """(side, the layer that is that image) of what ``source_ids`` lists, in the same order: a Combine layers layer's
    shown sides, else (None, the layer)."""
    if is_group(lay):
        shown = lay.get("shown") or {}
        subs = [x for x in ("starless", "stars") if shown.get(x, True)]
        if lay.get("stars_pid", lay["pid"]) is None and "stars" in subs:
            subs.remove("stars")
        return [(x, sub_layer(lay, x)) for x in subs]
    return [(None, lay)]


def new_filter_layer(key: str, target=None, sub=None) -> dict:
    """A filter layer. target: the id of the one layer it applies to (it sits just above it, as a reference image editor clipping
    mask); None: it applies to everything below it. sub: in a Stars and background layer, the one of its two images it
    applies to (``SUBS``)."""
    f = FILTERS[key]
    return {"id": new_id(), "kind": "filter", "filter": key, "name": f["name"], "visible": True, "opacity": 100.0,
            "mode": "Normal", "params": {q["key"]: copy.deepcopy(q["default"]) for q in f["params"]}, "mask": None,
            "target": target, "sub": sub}


#: the mask parts that read the image they are on (``Composer.mask``'s ``sources``)
READS_IMAGE = ("luminance", "color")
#: what such a part of a combination's filter's mask reads (``from``): the combination's picture without its stars (its
#: filters, starless images, mixed, before the stars are put back: as a luminosity mask is taken from the composite
#: picture in the reference photo editors; the default), the filter itself, or another filter ("f:<role>")
READS = ("picture", "self")


#: the rows of a Combine layers layer's images in the panel: its stars (a group, in Screen over the starless image,
#: issue #93), the real stars in that group when it holds Synthetic stars, its starless image
SUB_NAMES = {"starless": "Starless image", "stars": "Stars", "real": "Real stars"}
SUB_ROWS = ("stars", "real", "starless")


def is_group(lay: dict) -> bool:
    """A Stars and background layer: its two images are sub-layers, each with its own filters."""
    return lay.get("kind") == "workflow" and lay.get("part") == "combined"


#: the prefix of a combination's filter as a sub-image of its master layer ("f:SII"; postfusion.child)
ROLE_SUB = pf.ROLE
#: the pseudo part of the images Post makes of the nodes of a layer's tree (``sub_layer``)
FUSION = "fusion"
#: the layer tree of a Combine layers layer (graphcore post.mix_tree, given by the model: client/postfusion.py, never saved)
NEST = "nest"


def nodes_of(lay: dict) -> dict[str, dict]:
    """{path: node} of the layer tree a layer holds (the model gives it: ``nest``), {} for a layer without one."""
    nest = lay.get(NEST)
    return pf.node_paths(nest) if isinstance(nest, dict) else {}


def mix_at(lay: dict, path: str) -> dict | None:
    """The node at ``path`` when it is a combination Post mixes (its filters are the nodes under it), else None."""
    n = nodes_of(lay).get(path)
    return n if n is not None and n.get("kind") == "mix" else None


def is_fusion(lay: dict) -> bool:
    """A combination's master layer: a Combine layers layer whose starless image is a combination Post mixes again
    (the layer tree has a mix at "starless": client/postfusion.py)."""
    return is_group(lay) and mix_at(lay, "starless") is not None


def has_mix(lay: dict) -> bool:
    """A Combine layers layer with a mix on either side."""
    return is_group(lay) and any(mix_at(lay, s) is not None for s in SUBS)


def fusion_roles(lay: dict) -> list[str]:
    """The filters of a layer's starless mix (or of those of its images, ``sub_layer``), in its recipe's order; []
    for any other layer."""
    if not (is_group(lay) or lay.get("part") == FUSION):
        return []
    n = mix_at(lay, "starless")
    return list(n["fusion"].get("roles") or []) if n is not None else []


def role_sub(role: str) -> str:
    return ROLE_SUB + role


def sub_role(sub) -> str | None:
    """The filter a sub-image of the starless mix names ("f:SII" -> "SII"), None for another one."""
    return sub[len(ROLE_SUB):] if isinstance(sub, str) and sub.startswith(ROLE_SUB) and len(sub) > len(ROLE_SUB)         and "/" not in sub else None


def is_side(sub) -> bool:
    """A sub-image of a Combine layers layer: its stars, its starless image, or a node of a combination's tree (a
    path; whether the layer has that node is the tree's: ``normalize``)."""
    return sub in SUBS or pf.is_path(sub)


def subs_of(lay: dict) -> tuple[str, ...]:
    """The sub-images of a Combine layers layer: its two, and every node of its tree under them."""
    return SUBS + tuple(p for p in nodes_of(lay) if p not in ("", *SUBS))


def sub_name(lay: dict, sub: str) -> str:
    """The name of a sub-image's row: "Stars", "Starless image", or a node's, which is its filter's ("S II")."""
    if sub in SUB_NAMES:
        return SUB_NAMES[sub]
    above = mix_at(lay, pf.parent(sub))
    role = sub.rpartition("/")[2][len(ROLE_SUB):]
    if above is not None:
        return str((above["fusion"].get("names") or {}).get(role, role))
    return role


def sibling_paths(lay: dict, sub: str) -> list[str]:
    """The nodes that share ``sub``'s parent in the layer's tree (the eyes Alt+click on one shows alone): the sides of the
    layer, or the other inputs of a node's mix, the node itself among them, in their order."""
    nodes = nodes_of(lay)
    if sub in SUBS:
        return [x for x in SUBS if x in nodes]
    above = mix_at(lay, pf.parent(sub))
    return [pf.child(pf.parent(sub), r) for r in above["fusion"]["roles"]] if above is not None else [sub]


def read_paths(lay: dict, sub: str) -> list[str]:
    """The images a part of the mask of ``sub`` (a side or a node) that reads an image may read besides the node itself and
    the picture, as their paths: the filters of the starless mix (and of the stars' mix, for the stars' mask) for a
    side, the other inputs of its mix and its own inputs for a node (the choices of ``Composer.mask``'s ``sources``)."""
    out: list[str] = []
    if sub in SUBS:
        for root in ("starless", "stars"):
            n = mix_at(lay, root)
            if n is not None and root in ("starless", sub):
                out += [pf.child(root, r) for r in n["fusion"]["roles"]]
        return out
    above = mix_at(lay, pf.parent(sub))
    if above is not None:
        out += [c for c in (pf.child(pf.parent(sub), r) for r in above["fusion"]["roles"]) if c != sub]
    inner = mix_at(lay, sub)
    if inner is not None:
        out += [pf.child(sub, r) for r in inner["fusion"]["roles"]]
    return out


def fusion_core(lay: dict, path: str | None = None) -> str:
    """What the mix under ``path`` (the side its node is under, by default the starless one) depends on, of its
    combinations' settings: their filters, matrix, weights and balance, and those of the combinations nested in it."""
    node = mix_at(lay, pf.root_of(path or "starless"))
    return nest_core(node) if node is not None else ""


def nest_core(node: dict) -> str:
    """``fusion_core`` of a mix node."""
    def one(n):
        f = n["fusion"]
        return {"pid": n.get("pid"), "f": {k: f.get(k) for k in ("roles", "matrix", "weights", "balance")},
                "in": {r: one(c) for r, c in pf.mix_kids(n).items()}}
    return json.dumps(one(node), sort_keys=True)


def part_sliders(sliders: dict | None, part: str | None) -> dict:
    """The sliders an image of a workflow layer depends on: the stars alone, on the star sliders and the Sky (their
    stretch takes the whole image's sky, render/stars.star_tone); the starless image, on all but the star sliders
    (issue #19: a starless slider leaves the stars as they are)."""
    sliders = sliders or {}
    if part in ("stars", "synth"):
        return {k: v for k, v in sliders.items() if k.startswith("star_") or k == "sky"}
    if part in ("starless", FUSION):
        return {k: v for k, v in sliders.items() if not k.startswith("star_")}
    return dict(sliders)


def sub_row(lay: dict, sub: str) -> dict:
    """The row of one of a Stars and background layer's two images in the panel (not a layer of the list: its id is
    its layer's)."""
    return {"id": lay["id"], "kind": "sub", "sub": sub, "target": lay["id"], "name": sub_name(lay, sub),
            "visible": (lay.get("shown") or {}).get(sub, True), "mode": "Normal", "opacity": 100.0,
            "mask": (lay.get("side_masks") or {}).get(sub), "mkey": (lay["id"], sub)}


def mask_selection(what: str) -> bool:
    """Whether a selection kind of the panel is a mask's: "mask" (the layer's own) or "mask_stars" / "mask_starless"."""
    return what == "mask" or what.startswith("mask_")


def selection_side(what: str) -> str | None:
    """The side of a Combine layers layer a selection kind is about ("stars", "starless", or their masks), else None."""
    side = what[5:] if what.startswith("mask_") else what
    return side if is_side(side) else None


class SideLayer(dict):
    """What the mask tools act on when a side of a Combine layers layer is chosen: a layer-like view whose ``mask`` is
    that side's, written back to its layer (``side_masks``; the key is dropped with the last mask, so an old file's
    layer stays as it was)."""

    def __init__(self, lay: dict, side: str):
        super().__init__(id=lay["id"], kind="sub", sub=side, name=sub_name(lay, side), visible=True, opacity=100.0,
                         mode="Normal", mask=(lay.get("side_masks") or {}).get(side))
        self.lay, self.side = lay, side

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        if key == "mask":
            sides = self.lay.setdefault("side_masks", {})
            if value is None:
                sides.pop(self.side, None)
            else:
                sides[self.side] = value
            if not sides:
                self.lay.pop("side_masks", None)


def normalize(layers: list[dict]) -> list[dict]:
    """Each layer's own filters go just above it, in their order; a filter whose layer is gone applies to everything
    below. A filter on a node its layer's tree no longer has goes to the nearest node of the tree above it (the layer's
    tree is the graph's: a level the graph lost; the other keys naming it wait for the graph's change, never moved).
    The list is top first."""
    ids = {lay["id"] for lay in layers if is_pixel(lay)}
    groups = {lay["id"]: lay for lay in layers if is_group(lay)}
    for lay in layers:
        if lay["kind"] == "filter" and lay.get("target") not in ids:
            lay["target"] = None
        if lay["kind"] == "filter" and (lay.get("target") not in groups or not is_side(lay.get("sub"))):
            lay["sub"] = None
        elif lay["kind"] == "filter" and lay.get("sub") not in SUBS:
            nodes = nodes_of(groups[lay["target"]])
            if nodes:
                sub = lay["sub"]
                while sub not in nodes and sub not in SUBS:
                    sub = pf.parent(sub)
                lay["sub"] = sub
    out = []
    for lay in layers:
        if lay["kind"] == "filter" and lay.get("target"):
            continue
        if is_pixel(lay):
            out += [f for f in layers if f["kind"] == "filter" and f.get("target") == lay["id"]]
        out.append(lay)
    return out


def children(layers: list[dict], lid: int) -> list[dict]:
    """The filters nested in a pixel layer, top first (the top one is applied last)."""
    return [f for f in layers if f["kind"] == "filter" and f.get("target") == lid]


def tree(layers: list[dict]) -> list[tuple[dict, list[dict]]]:
    """The panel's structure: the top-level rows, top first, each with its children (pixel layers only)."""
    return [(lay, children(layers, lay["id"]) if is_pixel(lay) else []) for lay in normalize(list(layers))
            if not (lay["kind"] == "filter" and lay.get("target"))]


def rows(layers: list[dict], collapsed=()) -> list[tuple[dict, int]]:
    """The rows of the panel, top first, with their depth: a pixel layer, then its children indented (unless it is
    collapsed), then the next top-level row. A Stars and background layer's own filters come first, then its two
    images (``sub_row``), each with its filters one step further in, then the nodes of its tree under the side that is a
    mix (a node: its row, its filters one step further in, then the nodes under it: as a group's rows in a reference
    editor). ``collapsed``: the layer ids whose rows under them are hidden, and (id, path) for a side or a node."""
    out = []
    for lay, kids in tree(layers):
        out.append((lay, 0))
        if lay["id"] in collapsed:
            continue
        out += [(k, 1) for k in kids if not k.get("sub")]
        if is_group(lay):
            for sub in SUBS:
                out.append((sub_row(lay, sub), 1))
                if (lay["id"], sub) in collapsed:
                    continue
                out += [(k, 2) for k in kids if k.get("sub") == sub]
                if sub == "stars" and any(map(is_synth, kids)):     # the group's bottom row (#93)
                    out.append((sub_row(lay, "real"), 2))
                out += _node_rows(lay, kids, sub, 2, collapsed)
    return out


def _node_rows(lay: dict, kids: list[dict], path: str, depth: int, collapsed) -> list[tuple[dict, int]]:
    """The rows of the nodes under the mix at ``path`` of a layer's tree, each with its filters and the nodes under it."""
    node = mix_at(lay, path)
    out = []
    for r in (node["fusion"]["roles"] if node is not None else ()):
        cp = pf.child(path, r)
        out.append((sub_row(lay, cp), depth))
        if (lay["id"], cp) in collapsed:
            continue
        out += [(k, depth + 1) for k in kids if k.get("sub") == cp]
        out += _node_rows(lay, kids, cp, depth + 1, collapsed)
    return out


def move(layers: list[dict], lid: int, parent: int | None, index: int) -> list[dict]:
    """The layers with one moved, as a drop in the panel does. ``parent`` None: it becomes the top-level row at
    ``index`` (0 = the top, counted without it; a filter then applies to everything below it); else it becomes the child
    of that pixel layer at ``index`` among its children (0 = the top child). A pixel layer moves with its children and
    stays top level."""
    by_id = {x["id"]: x for x in layers}
    lay = by_id[lid]
    if parent is not None and (is_pixel(lay) or parent == lid or not is_pixel(by_id[parent])):
        raise ValueError("only a filter is nested, and only in a pixel layer")
    group = [lay] + (children(layers, lid) if is_pixel(lay) else [])
    gid = {x["id"] for x in group}
    top = [(x, [k for k in kids if k["id"] not in gid]) for x, kids in tree(layers) if x["id"] not in gid]
    if parent is None:
        if lay["kind"] == "filter":
            lay["target"] = lay["sub"] = None
        top.insert(max(0, min(index, len(top))), (lay, group[1:]))
    else:
        if lay.get("target") != parent:
            lay["sub"] = None
        lay["target"] = parent
        kids = next(k for x, k in top if x["id"] == parent)
        kids.insert(max(0, min(index, len(kids))), lay)
    out = []
    for x, kids in top:
        out += kids + [x]
    return out


def copy_mask(m: dict) -> dict:
    return dict(m, parts=[dict(p, strokes=list(p["strokes"])) if "strokes" in p else dict(p) for p in m["parts"]])


def snapshot(layers: list[dict]) -> list[dict]:
    """A copy of the layers, for Undo and for a render request: nothing in it is shared with the layers being edited."""
    out = []
    for lay in layers:
        d = dict(lay)
        if "sliders" in d:
            d["sliders"] = dict(d["sliders"])
        if "shown" in d:
            d["shown"] = dict(d["shown"])
        if "tune" in d:
            d["tune"] = {k: dict(v) for k, v in d["tune"].items()}
        if "tunes" in d:                                    # the own settings of the nested mixes (None: none of its own)
            d["tunes"] = {p: ({k: dict(v) for k, v in t.items()} if t else None) for p, t in d["tunes"].items()}
        if "params" in d:
            d["params"] = {k: copy.deepcopy(v) if isinstance(v, (dict, list)) else v for k, v in d["params"].items()}
        m = lay.get("mask")
        if m is not None:
            d["mask"] = copy_mask(m)
        if lay.get("side_masks"):
            d["side_masks"] = {k: copy_mask(v) for k, v in lay["side_masks"].items()}
        out.append(d)
    return out


# ---------------------------------------------------------------------- saved compositions
def _num(v, default, lo=None, hi=None):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(x):
        return default
    return min(max(x, lo), hi) if lo is not None else x


def _clean_mask(m) -> dict | None:
    if not isinstance(m, dict):
        return None
    out = new_mask()
    out["feather"] = _num(m.get("feather"), 0.0, 0, 60)
    out["density"] = _num(m.get("density"), 100.0, 0, 100)
    out["invert"] = bool(m.get("invert"))
    for p in m.get("parts") or []:
        src = p.get("src") if isinstance(p, dict) else None
        if src not in MASK_SOURCES:
            continue
        ranges = {k: (lo, hi) for k, _n, lo, hi, _u in PART_SETTINGS.get(src, [])}
        q = new_part(src)
        q["id"] = int(p.get("id") or q["id"])
        q["op"] = p.get("op") if p.get("op") in OPS else "add"
        for k, v in q.items():
            if k in ("id", "src", "op") or k not in p:
                continue
            if k == "strokes":
                q[k] = [(float(s[0]), float(s[1]), float(s[2]), float(s[3]), bool(s[4]))
                        for s in p[k] if isinstance(s, (list, tuple)) and len(s) == 5]
            elif k == "paint":
                q[k] = bool(p[k])
            elif k == "direction":
                q[k] = p[k] if p[k] in DIRECTIONS else v
            else:
                q[k] = _num(p[k], v, *ranges.get(k, (None, None)))        # within its slider's range
        if src in READS_IMAGE and isinstance(p.get("from"), str) and (p["from"] in READS or pf.is_path(p["from"])):
            q["from"] = p["from"]                       # (a node's mask: what the part reads: ``READS`` or a node's path)
        out["parts"].append(q)
    return out


def load_layers(raw, alive: set[int]) -> list[dict]:
    """The layers of a saved composition (``project.post["layers"]``), made safe: unknown filters and the layers of
    workflows that no longer exist are dropped, every field is brought to its type and range."""
    out: list[dict] = []
    for d in raw or []:
        if not isinstance(d, dict):
            continue
        if d.get("kind") == "workflow":
            if int(d.get("pid", -1)) not in alive or d.get("part") not in (None, *PARTS):
                continue
            name = str(d.get("name") or "?")
            for old, new in OLD_PART_NAMES.items():           # issue #36: a layer saved with the old name
                if name.endswith(" · " + old):
                    name = name[: -len(old)] + new
            lay = new_workflow_layer(int(d["pid"]), name, bool(d.get("visible")), d.get("part"),
                                     d.get("base"))
            if lay.get("part") == "combined":
                lay["starless"] = d.get("starless") if d.get("starless") in ("starless", "output") else "starless"
                sp = d["stars_pid"] if "stars_pid" in d else d["pid"]      # the stars side: its workflow, else none
                lay["stars_pid"] = int(sp) if sp is not None and int(sp) in alive else None
                if d.get("stars_image") == "output" and lay["stars_pid"] is not None:   # written only when not the default
                    lay["stars_image"] = "output"
                lay["blend"] = d.get("blend") if d.get("blend") in BLENDS else BLENDS[0]
                shown = d.get("shown") if isinstance(d.get("shown"), dict) else {}
                lay["shown"] = {k: bool(shown.get(k, True)) for k in SUBS}
                if "real" in shown:                              # the real stars' eye, in a Stars group (#93)
                    lay["shown"]["real"] = bool(shown["real"])
                for k, v in shown.items():                       # the eyes of the nodes of the layer's tree
                    if pf.is_path(k) and k not in SUBS:
                        lay["shown"][k] = bool(v)
                tune = d.get("tune") if isinstance(d.get("tune"), dict) else None
                if tune is not None:                             # a combination's own hues and intensities
                    lay["tune"] = {k: {str(r): _num(v, 0.0) for r, v in (tune.get(k) or {}).items()}
                                   for k in ("hues", "weights") if isinstance(tune.get(k), dict)}
                if isinstance(d.get("tunes"), dict):             # the nested mixes' own (the syntax only: the layer tree
                    lay["tunes"] = {str(p): (None if not isinstance(t, dict) else    # is attached later)
                                    {k: {str(r): _num(v, 0.0) for r, v in (t.get(k) or {}).items()}
                                     for k in ("hues", "weights") if isinstance(t.get(k), dict)})
                                    for p, t in d["tunes"].items()}
                sides = d.get("side_masks") if isinstance(d.get("side_masks"), dict) else {}
                sides = {k: m for k in [*SUBS, *(x for x in sides if pf.is_path(x) and x not in SUBS)]
                         if (m := _clean_mask(sides.get(k))) is not None}
                if sides:                                        # written only when there is one (older files)
                    lay["side_masks"] = sides
            lay["sliders"] = {k: _num(v, 2.0, 0, slider_top(k)) for k, v in (d.get("sliders") or {}).items()}
            if d.get("stretch") in LAYER_STRETCHES[1:]:          # written only when it is not the automatic one
                lay["stretch"] = d["stretch"]
        elif d.get("kind") == "image" and isinstance(d.get("key"), str) and d["key"]:
            lay = new_image_layer(d["key"], str(d.get("name") or "?"), bool(d.get("visible", True)))
            lay["sliders"] = {k: _num(v, 2.0, 0, 4) for k, v in (d.get("sliders") or {}).items()}
            lay["linked"] = bool(d.get("linked", True))
        elif d.get("kind") == "filter" and d.get("filter") in FILTERS:
            lay = new_filter_layer(d["filter"], d.get("target"), d.get("sub") if is_side(d.get("sub")) else None)
            name = str(d.get("name") or lay["name"])
            lay["name"] = OLD_NAMES.get(name, name)
            lay["visible"] = bool(d.get("visible", True))
            raw = dict(d.get("params") or {})
            if d["filter"] == "curves" and "curves" not in raw:          # saved before the Curves tool
                raw["curves"] = curves_of_old(_num(raw.get("shadows"), 0.0, -0.2, 0.2),
                                              _num(raw.get("midtones"), 0.0, -0.2, 0.2),
                                              _num(raw.get("highlights"), 0.0, -0.2, 0.2))
            if d["filter"] == "scnr":                                    # saved before its protections: as it was
                raw.setdefault("method", pf_colour.SCNR_METHODS[0])
                raw.setdefault("lightness", False)
            for q in FILTERS[d["filter"]]["params"]:
                v = raw.get(q["key"], q["default"])
                if q["type"] == "choice":
                    v = v if v in q["choices"] else q["default"]
                elif q["type"] == "bool":
                    v = bool(v)
                elif q["type"] == "curve":
                    v = pf_curves.clean_curves(v)
                elif q["type"] == "spots":
                    v = clean_spots(v)
                else:
                    v = _num(v, q["default"], q["lo"], q["hi"])
                lay["params"][q["key"]] = v
        else:
            continue
        lay["id"] = int(d.get("id") or lay["id"])
        lay["opacity"] = _num(d.get("opacity"), 100.0, 0, 100)
        lay["mode"] = d.get("mode") if d.get("mode") in MODES else "Normal"
        lay["mask"] = _clean_mask(d.get("mask"))
        out.append(lay)
    ids = {lay["id"] for lay in out}
    if len(ids) != len(out):                                  # duplicated ids: give them new ones
        seen: set[int] = set()
        for lay in out:
            if lay["id"] in seen:
                lay["id"] = new_id()
            seen.add(lay["id"])
    reseed_ids(out)
    return normalize(out)


def save_layers(layers: list[dict]) -> list[dict]:
    """The composition as JSON-able dicts (a filter on no sub-layer saved as before them). A Synthetic stars row as it
    comes (``new_synth_layer``, nothing changed in it) is not saved: ``with_synth_rows`` gives it at each load, so the
    definition holds what the user made, and a change undone gives back the definition it had (it was written there at
    the first save after the load, whatever the change)."""
    out = json.loads(json.dumps(snapshot(layers), default=list))
    for d in out:
        d.pop("fusion", None)                           # the model's, given at each load (client/postfusion.py)
        d.pop(NEST, None)                               # the layer tree, derived from the graph
        d.pop("retitle", None)                          # the name the graph shows its workflow by (post3.RETITLE)
        if d.get("kind") == "filter" and d.get("sub") is None:
            d.pop("sub", None)
    return [d for d in out if not _untouched_synth(d)]


def _untouched_synth(d: dict) -> bool:
    if not is_synth(d) or d.get("sub") != "stars" or any(p.get("strokes") for p in (d.get("mask") or {}).get("parts", [])):
        return False                                    # painted: changed (and no copy of its strokes at each save)

    def plain(x):
        x = json.loads(json.dumps(x, default=list))
        x.pop("id", None)
        for part in (x.get("mask") or {}).get("parts", []):
            part.pop("id", None)
        return x
    global _SYNTH_AS_IT_COMES
    if _SYNTH_AS_IT_COMES is None:                      # made once: its ids are taken out of the comparison
        _SYNTH_AS_IT_COMES = plain(snapshot([new_synth_layer(0)])[0])
    return plain(d) == dict(_SYNTH_AS_IT_COMES, target=d.get("target"))


_SYNTH_AS_IT_COMES: dict | None = None


# ======================================================================== what the composition reads
@dataclass
class Canvas:
    """The image of Post: the area every workflow layer covers, in the pixels of the first (bottom) layer's grid."""
    W: int
    H: int
    origin: tuple[int, int] = (0, 0)

    def size(self, level: float) -> tuple[int, int]:
        """The canvas at a level: 1 / 2^level of its pixels each way, rounded up. A level between two powers of two
        (the screen's picture at the size it is shown, ``services.render.screen_level``) is a reduction by any factor."""
        if float(level).is_integer():
            f = 2 ** int(level)
            return -(-self.W // f), -(-self.H // f)
        f = 2.0 ** level
        return max(1, math.ceil(self.W / f - 1e-9)), max(1, math.ceil(self.H / f - 1e-9))


def crop_within(crop, W: int, H: int) -> tuple[int, int, int, int] | None:
    """The part of a crop (x0, y0, x1, y1) that is inside an image of W x H pixels: the picture that is written and
    the size that is said. A crop is kept in the pixels of Post's image, which is the area every layer covers: when
    that area gets smaller afterwards (a layer added, a workflow computed again), the box kept goes past its edge.
    None: no crop, or nothing of it inside (the whole image then)."""
    if not crop:
        return None
    x0, y0, x1, y1 = (int(v) for v in crop)
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, int(W)), min(y1, int(H))
    return (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None


class Unreadable(Exception):
    """A source (``source_id``) whose result is there but one of its files is not: the layers reading it are left out,
    and said not computed."""

    def __init__(self, sid):
        super().__init__(f"the result of {sid!r} cannot be read")
        self.sid = sid


class Provider:
    """What the composition needs from the engine. ``services/render.py`` implements it on the layers service."""

    def version(self, pid) -> int:
        """Changes when the result of this workflow (or image, ``("image", key)``) changes."""
        return 0

    def available(self, pid) -> bool:
        raise NotImplementedError

    def readable(self, lay: dict) -> bool:
        """Whether the image of this layer (or of one side of a Combine layers layer, see ``sub_layer``) can be read."""
        return self.available(source_id(lay))

    def canvas(self, pids: list) -> Canvas:
        """The canvas of these workflows and images (bottom first, see ``source_id``)."""
        raise NotImplementedError

    def layer(self, lay: dict, level: int, box: tuple[int, int, int, int], draft: bool = False,
              stop=None) -> np.ndarray:
        """The pixel layer ``lay`` at the level, over the box (x0, y0, x1, y1) of the canvas at that level:
        (h, w, 3) float32 in [0, 1]. Above level 0 the box is the whole canvas. ``draft``: a quick look is enough (a
        slider moves); ``stop()``: True abandons the rendering (raise ``Superseded``)."""
        raise NotImplementedError

    def linear_stars(self, lay: dict, level: int, box) -> np.ndarray:
        """(h, w, 3): a stars side's stars as its block computed them, none of Post's star settings: where the Stars
        mask part finds its stars (``find_stars``). A provider without the block's linear image gives the side at its
        suggestion (its own image, every star setting at the middle)."""
        return self.layer(dict(lay, sliders={}), level, box)

    def stars_id(self, lay: dict) -> str | None:
        """What a stars side's result is, for a detection kept on disk (``stars_dir``): its folder and its time; None
        when the provider cannot say (the detection is then kept for the session only)."""
        return None

    #: the folder where the stars found in a result are kept (the project's cache), None: not kept on disk
    stars_dir = None
    #: the Stars part's search made in another process, the mask without its stars meanwhile (the window's results:
    #: a search of a 16 Mpx result takes a minute and holds the interpreter); False: at once, in the caller's thread
    search_apart = False

    def cover(self, lay: dict, level: int, box) -> np.ndarray | None:
        """What the layer covers of the box, (h, w) in [0, 1] (an image of another shape is fitted inside the canvas,
        its aspect kept: transparent around it); None when it covers all of it."""
        return None

    def data_mask(self, name: str, level: int, box) -> np.ndarray | None:
        """The 'halpha' map over the box, (h, w) float32 in [0, 1], or None when the layers have none."""
        return None

    def fusion_planes(self, lay: dict, level: int, box) -> tuple[dict, np.ndarray]:
        """A master layer's balanced filters ({role: (h, w) float32}, linear, before their weights) over the box of
        the canvas at the level (above level 0 the whole canvas), and where every one has data ((h, w) bool)."""
        raise NotImplementedError

    def fusion_form(self, lay: dict) -> dict:
        """How the roles of a combination (``lay``: ``pid`` and ``fusion`` its description) read its inputs, asked for a
        combination that has another combination among its inputs: {"maps": {role: ({channel read: coefficient},
        offset)}, "channels": {role: the channel it reads of its input's image}} (postfusion.balance_maps)."""
        raise NotImplementedError

    def fusion_reads(self, lay: dict, level: int, box) -> dict:
        """{role: (h, w) float32}: the channels a combination reads of its inputs, linear, before its balance, over the
        box of the canvas at the level (as ``fusion_planes``): the planes a role that reads another combination takes
        of its other inputs."""
        raise NotImplementedError

    def fusion_settings(self, lay: dict):
        """The render settings of a master layer's starless image (render/simple.SimpleSettings): its base stretch,
        its sliders."""
        from ..render.sliders import settings_at
        return settings_at("simple", None, part_sliders(lay.get("sliders"), "starless"), has_denoise=False,
                           origin="palette", stretch=lay.get("stretch"))

    def has(self, name: str) -> bool:
        """Whether a data mask exists ('halpha')."""
        return False

    def notes(self) -> set[str]:
        """What the engine said while rendering (a fallback to the CPU...), read and cleared by the composer."""
        return set()


class ByteLRU:
    """Images kept by content, the least recently used dropped first; an image no layer holds any more goes at once.

    Scan-resistant: a composition reads its images in the same order every time, and when they do not all fit, a plain
    LRU drops exactly the one the next composition reads first (none is ever found again). ``begin(level)`` opens a
    composition at a level. To make room it drops, in this order: for a screen level's composition, an image at full
    resolution (level 0: the exact view's, the export's), then one of a coarser level (cheaper to make again), the
    least recently used first; one of its level not used by the last two compositions there; one the last composition
    used, the most recently used first (in a scan, the one needed last: Belady's choice). Never one this composition
    used, nor one of a finer screen level; a composition at full resolution never drops a screen level's image (a
    27 Mpx image is 327 MB: kept, it pushed out the screen's steps and every release made the whole stack again). The
    images kept stay the same from one composition to the next, and are found again."""

    def __init__(self, cap: int = 1_500_000_000):
        self.cap = cap
        self.items: OrderedDict = OrderedDict()
        self.bytes = 0
        self.level = 0                            # the level of the composition being made (``begin``)
        self.gens: dict = {}                      # level -> its compositions so far
        self._used: dict = {}                     # key -> (level, composition) that last used it
        self._held: dict = {}                     # (layer id, its part, level) -> key
        self._holders: dict = {}                  # key -> {(layer id, its part, level)}

    def begin(self, level: int = 0):
        self.level = level
        self.gens[level] = self.gens.get(level, 0) + 1

    def _mark(self, key):
        self._used[key] = (self.level, self.gens.get(self.level, 0))

    def get(self, key):
        a = self.items.get(key)
        if a is not None:
            self.items.move_to_end(key)
            self._mark(key)
        return a

    def _victim(self):
        lvl, now = self.level, self.gens.get(self.level, 0)
        full = coarser = old = last = None
        for k in self.items:                                  # the least recently used first
            t, g = self._used.get(k, (lvl, -1))
            if t == 0 and lvl > 0:                            # full resolution: the first to go for the screen's
                if full is None:
                    full = k
            elif t > lvl:
                if lvl == 0:                                  # (never a screen level's for a full-resolution image)
                    continue
                if coarser is None:
                    coarser = k
            elif t == lvl and g != now:
                if g < now - 1 and old is None:
                    old = k
                last = k
        for k in (full, coarser, old):
            if k is not None:
                return k
        return last

    def put(self, key, a: np.ndarray, holder=None) -> bool:
        """Keep ``a``; False when it was not kept (what is kept serves this composition, or a finer level)."""
        if holder is not None:
            self._unhold(holder, key)
        if key not in self.items:
            while self.items and self.bytes + a.nbytes > self.cap:
                k = self._victim()
                if k is None:
                    return False
                self._drop(k)
            self.items[key] = a
            self.bytes += a.nbytes
        self.items.move_to_end(key)
        self._mark(key)
        if holder is not None:
            self.hold(key, holder)
        return True

    def room(self, holder=None) -> int:
        """The bytes free, with those of what ``holder`` holds (it gives it up for what it takes next)."""
        k = self._held.get(holder) if holder is not None else None
        mine = self.items[k].nbytes if k in self.items and len(self._holders.get(k, ())) <= 1 else 0
        return self.cap - self.bytes + mine

    def _unhold(self, holder, key):
        """The holder takes another image: the one it held goes when nobody else holds it."""
        old = self._held.get(holder)
        if old is not None and old != key:
            hs = self._holders.get(old, set())
            hs.discard(holder)
            del self._held[holder]
            if not hs:
                self._drop(old)

    def hold(self, key, holder):
        self._unhold(holder, key)
        if key not in self.items:
            return
        self._held[holder] = key
        self._holders.setdefault(key, set()).add(holder)
        self._mark(key)

    def drop_level(self, level) -> None:
        """Every image last used by a composition at ``level`` goes (a screen level the view left)."""
        for k in [k for k in self.items if self._used.get(k, (None,))[0] == level]:
            self._drop(k)

    def _drop(self, key):
        a = self.items.pop(key, None)
        if a is not None:
            self.bytes -= a.nbytes
        self._used.pop(key, None)
        for h in self._holders.pop(key, set()):
            if self._held.get(h) == key:
                del self._held[h]

    def clear(self):
        self.items.clear()
        self._used.clear()
        self._held.clear()
        self._holders.clear()
        self.bytes = 0


@dataclass
class Composite:
    rgb: np.ndarray                                           # (h, w, 3) float32 in [0, 1]: the box at the level
    level: int
    box: tuple[int, int, int, int]
    masks: dict = field(default_factory=dict)          # layer id -> (h, w) float32, those wanted
    mask_thumbs: dict = field(default_factory=dict)    # layer id -> (36, 54) uint8
    notes: set[str] = field(default_factory=set)
    layer_ms: float = 0.0                                     # time spent in the engine's renderings
    drawn: int = 0                                            # how many workflow layers were rendered (not cached)
    hists: dict[int, dict] = field(default_factory=dict)      # filter id -> the histograms of its input (``hists``)


class Ctx:
    """What a filter or a mask part needs, at one resolution and for one region."""

    def __init__(self, comp: Composer, level: int, abox, draft: bool = False):
        self.comp, self.level, self.abox = comp, level, abox
        self.f = 2 ** level
        self.draft = draft                                    # a quick look while something moves (tables, not exact)
        self.hist_ids: set = set()                            # the filters whose input's histograms are wanted
        self.hists: dict = {}
        self.synth = None                                     # the Synthetic stars of the layer being composed
        self.fusion_halpha = None                             # (a combination's master layer: its H-alpha map)
        self.stars_fn = None                                  # the stars image of the layer being composed (Stars part)
        self._memo: dict[str, np.ndarray] = {}
        self.canvas_wh: tuple[int, int] | None = None         # the whole canvas at this level
        self.stats_key: str | None = None                     # the input of the filter being applied, any region
        self.row_edit: str | None = None                      # (the ``edit`` of the image layer being composed)
        self.input_sig: str | None = None                     # what the input of the filter being applied depends on
        self.check = None                                     # True: the composition is abandoned (``Superseded``)

    def whole_stats(self, name: str, x: np.ndarray, measure):
        """``measure`` of a filter's input taken on the whole image, whatever region is rendered (review 10.1: at
        100 % Post renders the visible area only, and a measure of it differed from the export's): on a whole-canvas
        rendering, measured on the input reduced to the measure level (``stats_level``: the same pixels at any level)
        and kept under the input's key; on a region, the kept value (``Composer.compose`` makes it first)."""
        key = None if self.stats_key is None else (self.stats_key, name)
        whole = self.canvas_wh is not None and tuple(self.abox) == (0, 0, *self.canvas_wh)
        if not whole and key is not None and key in self.comp.stats:
            return self.comp.stats[key]
        if not whole or self.canvas_wh is None:
            return measure(x)
        mine = None if self.input_sig is None else ("input", self.input_sig, name)
        if mine is not None and mine in self.comp.stats:      # the same input as before (a drag of the filter's own
            v = self.comp.stats[mine]                         # setting): measured once
            self._keep_stats(key, v)
            return v
        k = stats_level(*self.canvas_wh)
        v = measure(x if k == 0 else cv2.resize(x, (max(1, x.shape[1] >> k), max(1, x.shape[0] >> k)),
                                               interpolation=cv2.INTER_AREA))
        self._keep_stats(key, v)
        self._keep_stats(mine, v)
        return v

    def _keep_stats(self, key, v) -> None:
        if key is not None:
            self.comp.stats[key] = v
            self.comp.stats.move_to_end(key)
            while len(self.comp.stats) > STATS_KEPT:
                self.comp.stats.popitem(last=False)

    def whole_measure(self, name: str, x: np.ndarray, measure):
        """(``measure`` of a filter's input, the level it was taken at) for a measure that belongs to its resolution
        (a noise, a mesh of pixels): a whole-canvas rendering measures what it renders (as it always did) and keeps it
        at its level; a region takes the whole image's, kept at the measure level (``stats_level``: ``compose``
        renders it first), for the filter to bring to its own level (review 10.1)."""
        whole = self.canvas_wh is None or tuple(self.abox) == (0, 0, *self.canvas_wh)
        if whole:                         # measured (and kept) only at the measure level: elsewhere the filter does
            if self.canvas_wh is None or self.stats_key is None or stats_level(*self.canvas_wh):
                return None, self.level
            mine = None if self.input_sig is None else ("input", self.input_sig, name, self.level)
            v = self.comp.stats.get(mine) if mine is not None else None
            if v is None:                                     # (the same input as before: measured once)
                v = measure(x)
            self._keep_stats((self.stats_key, name, self.level), v)
            self._keep_stats(mine, v)
            return v, self.level
        at = self.level + stats_level(*self.canvas_wh)
        got = None if self.stats_key is None else self.comp.stats.get((self.stats_key, name, at))
        return (got, at) if got is not None else (None, self.level)

    def _data(self, name: str) -> np.ndarray:
        if name not in self._memo:
            m = self.comp.provider.data_mask(name, self.level, self.abox)
            x0, y0, x1, y1 = self.abox
            self._memo[name] = (np.zeros((y1 - y0, x1 - x0), np.float32) if m is None
                                else np.ascontiguousarray(m, np.float32))
        return self._memo[name]

    def stars(self):
        """(their key, the stars) found in the stars of the layer being composed (its Stars block's stars alone; a
        Combine layers layer's, and its filters', its Stars side: ``Composer.star_discs``), None when it has none."""
        if self.stars_fn is None:
            return None
        if "stars" not in self._memo:
            self._memo["stars"] = self.stars_fn()
        if self._memo["stars"] is None and self.comp._stars_wait:
            self.comp._stars_missing = True              # (what is made now without them is not kept: ``compose``)
        return self._memo["stars"]

    def halpha(self) -> np.ndarray:
        if self.fusion_halpha is not None:              # a combination's master layer: its H-alpha filter
            if "fusion halpha" not in self._memo:
                self._memo["fusion halpha"] = self.fusion_halpha()
            return self._memo["fusion halpha"]
        return self._data("halpha")


#: the long side of the image a whole-image measure is taken on (``Ctx.whole_stats``), whatever the level rendered
STATS_SIDE = 1600
STATS_KEPT = 512


def stats_level(w: int, h: int) -> int:
    """How many halvings bring an image of w x h to ``STATS_SIDE`` or under: at level l the measure is taken at
    level l + this, the same level for every rendering of the image."""
    k = 0
    while max(w, h) >> k > STATS_SIDE:
        k += 1
    return k


#: the filters a measure of their whole input decides (``Ctx.whole_stats``), when that measure is used
def _wants_whole(fl) -> list[tuple]:
    """The whole-image measures a filter will use with its settings: ("background",) for ``Ctx.whole_stats``,
    (name, True) for ``Ctx.whole_measure`` (kept at the measure level)."""
    p = fl["params"]
    if fl["filter"] == "saturation" and float(p.get("protect", 0.0)) > 0 and float(p.get("strength", 0.0)) != 0:
        return [("background",)]
    if fl["filter"] == "ml_denoise":
        out = []
        if float(p.get("amount", 0.0)) > 0 and any(p.get(k, 0) for k in ML_KEYS[:4]):
            out.append((f"noise {p['target']}", True))
        if _nr_coarse(p):
            out.append((f"noise layers {p['target']}", True))
        return out
    if fl["filter"] == "dehaze" and float(p.get("amount", 0.0)) > 0:
        return [(f"veil {p['scale']}", True)]
    return []


def thumb_of(a: np.ndarray) -> np.ndarray:
    """A 54 x 36 uint8 thumbnail of a float image (h, w[, 3])."""
    u = (np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8)
    return cv2.resize(u, THUMB, interpolation=cv2.INTER_AREA)


def default_cache_bytes() -> int:
    """What Post's caches (the layers' images, the image after each filter, the masks' planes) may hold together: 8 % of
    the machine's memory, 0.3 to 6 GB (a 16 GB laptop: 1.3 GB; 64 GB: 5.1 GB). Measured on a laptop (CPU, 3 Mpx
    synthetic project, 60 actions), the caches were two LRUs of 1.5 GB each and filled over the first 30 actions: +1.2 GB
    (the second one alone held 1.4 GB, 39 images of 36 MB), for no gain a user feels: a slider reuses the steps below the
    filter it moves, a dozen images. The top was 1.5 GB until a 27 Mpx image with twelve filters on its layer: the
    steps of one screen level are 12 x 82 MB, and every release made the whole stack again (the owner's NGC 7000)."""
    try:
        import psutil
        total = psutil.virtual_memory().total
    except Exception:  # noqa: BLE001 - no figure: the smallest cap that keeps Post quick
        return 600_000_000
    return int(min(6_000_000_000, max(300_000_000, 0.08 * total)))


@dataclass
class _Item:
    """One row of the composition as ``compose`` walks it, bottom first: a visible pixel layer with its own filters
    ("pixel"), a filter on everything below ("filter"), or a hidden layer whose mask is still asked for ("hidden")."""
    lay: dict
    kind: str
    own: list = field(default_factory=list)
    sig: str = ""                       # what the composite after it depends on (the step cache's ``comp_sig``)
    src: str = ""                       # a pixel layer's: what its image after its own filters depends on
    edit: str = ""                      # the same, without the level: what was set up to it (where an edit is)
    tkey: str = ""                      # what its mask thumbnails depend on
    opaque: bool = False                # Normal, 100 %, no mask: what is under it never shows
    needs: bool = False                 # a mask or a histogram of it is asked for: never skipped


THUMB_ITEMS = 4096                      # the mask thumbnails kept, by the rows that made them (2 KB each at most)


from .postmix import MixRendering  # noqa: E402 - the composer's mixes of the layer tree (it reads this module)


class Composer(MixRendering):
    """The composition of the layers. One per render worker: it owns the caches (the workflow layers' images by
    content, the brushes' rasters), so it is used from one thread only. ``cache_bytes``: all the caches together
    (``default_cache_bytes``): 30 % the layers' images, 30 % the steps, 25 % the composites (``backdrops``), 15 % the
    masks' planes.

    As in a layer editor, a change recomputes only what it changes: the composite under the row being edited (its
    backdrop) is kept, so a change of that row starts from it, and a row under an opaque one (Normal, 100 %, no mask)
    is never computed. Each layer's image after its own filters is in the steps (by all that made it)."""

    def __init__(self, provider: Provider, cache_bytes: int | None = None):
        cache_bytes = default_cache_bytes() if cache_bytes is None else cache_bytes
        self.provider = provider
        self.images = ByteLRU(int(cache_bytes * .30))
        self.steps = ByteLRU(int(cache_bytes * .30))        # the image after each filter, by all that made it (#87)
        self.backdrops = ByteLRU(int(cache_bytes * .25))    # the composite after a row: under the edited one, the last
        self._brush: dict = {}                              # (part id, level) -> (n strokes, last stroke, raster)
        self.parts = ByteLRU(min(int(cache_bytes * .15), 400_000_000))  # a mask part's plane, by its settings and what it reads
        self.part_draws = 0                                 # how many mask part planes were computed (not cached)
        self._canvas: tuple | None = None                   # (pids, canvas)
        self._drawn = 0
        self._edits: list[str] = []                         # the rows' ``edit`` of the last composition followed
        self._focus: str | None = None                      # ``edit`` of the row under the one edited last
        self._edited: str | None = None                     # ``edit`` of the row edited last (its steps are kept)
        self._keep = True                                   # (False: the steps being made are not kept)
        self._thumbs: OrderedDict = OrderedDict()           # ``tkey`` -> {mask key: thumbnail}
        self._side_thumbs: OrderedDict = OrderedDict()      # a master layer's kept starless image -> its masks' thumbnails
        self._own_seen: dict = {}                           # layer id -> (its own filters as last composed, ``_own_edited``)
        self.progress = None                                # called with each filter a composition applies (the banner)
        self.costs: dict = {}                               # filter id -> seconds it took at a screen level, last time
        self.applied: set = set()                           # the filters the last composition applied
        self._screen = None                                 # the level of the screen's picture (``screen_at``)
        self._step_thumbs: OrderedDict = OrderedDict()      # a filter's step -> its mask's thumbnail (``_own_jump``)
        self.skipped = 0                                    # rows the last composition did not compute
        self.plan_errors = 0                                # rows whose digest the plan did not foresee (tests: 0)
        self.stats: OrderedDict = OrderedDict()             # (filter input's key, measure) -> its whole-image values
        self._fstats: dict = {}                             # a mix of the layer tree -> its statistics (``_fusion_stats``)
        self._forms: dict = {}                              # a combination -> how its roles read its inputs (``_form``)
        self._stars_new = False                             # a search of stars ended since the last composition
        self._stars_missing = False                         # this composition made something without stars searched
        self._stars_wait = False                            # this composition waits for a search (said in its notes)
        self._stars_failed = False                          # a search of its stars failed (said in its notes)
        _COMPOSERS.add(self)

    # ------------------------------------------------------------ geometry
    def screen_at(self, level) -> None:
        """The screen's picture is now made at ``level`` (the view's size or zoom changed): what was kept at its former
        level goes. Kept, the first picture's stack (made at the half before the view knew its size) took the room
        of the full resolution's steps: every exact view made its twelve filters again (13 s on a 27 Mpx image, not
        1.2). A level a draft or a region uses is not a screen level and is left to the caches."""
        old, self._screen = self._screen, level
        if old is None or old == level or old == 0:
            return
        for lru in (self.images, self.steps, self.backdrops, self.parts):
            lru.drop_level(old)

    def canvas_of(self, layers: list[dict]) -> Canvas | None:
        pids = []
        for lay in reversed(layers):
            if is_pixel(lay):
                pids += [x for x in source_ids(lay) if self.provider.available(x) and x not in pids]
        if not pids:
            return None
        key = (tuple(pids), tuple(self.provider.version(p) for p in pids))
        if self._canvas is None or self._canvas[0] != key:
            self._canvas = (key, self.provider.canvas(pids))
        return self._canvas[1]

    def forget(self):
        """Drop every cached image (the results changed on disk, or an Undo)."""
        self.images.clear()
        self.steps.clear()
        self.backdrops.clear()
        self.parts.clear()
        self._brush.clear()
        self._thumbs.clear()
        self._side_thumbs.clear()
        self._own_seen.clear()
        self._step_thumbs.clear()
        self._fstats.clear()
        self._forms.clear()
        self._edits, self._focus, self._edited = [], None, None
        self._canvas = None

    def release(self):
        """Leaving Post: every cache goes (all of it is rebuilt by the next render)."""
        self.forget()

    # ------------------------------------------------------------ one workflow layer
    def _key(self, lay: dict, level: int, abox, canvas: Canvas, draft: bool = False) -> tuple:
        """What an image of a pixel layer depends on: the same for a copy of the layer. The canvas is part of it: it
        changes when another workflow's result comes in. A draft (a quick look) is kept apart from the exact image."""
        sid = source_id(lay)
        part = lay.get("part")
        over = lay.get("stars_over")          # the stars of a combination: stretched with the starless one's statistics
        key = (sid, part, lay.get("blend") if part == "combined" else None,
                json.dumps(part_sliders(lay.get("sliders"), "stars" if over is not None and part is None else part),
                           sort_keys=True),
                json.dumps(lay.get("base"), sort_keys=True, default=str), lay.get("linked"), self.provider.version(sid),
                level, None if level > 0 else abox, (canvas.W, canvas.H, canvas.origin))             + (("draft",) if draft else ()) + (("over", over, self.provider.version(over)) if over is not None else ())             + (("stretch", lay["stretch"]) if layer_stretch(lay) != "auto" and over is None and part not in ("stars", "synth")
               else ())                # the base stretch, when it is not the automatic one (the keys before it stay)
        return key + ((FUSION, fusion_core(lay, lay.get("node")), lay.get("node")) if part == FUSION else ())

    def cached(self, lay: dict, level: int, canvas: Canvas) -> np.ndarray | None:
        """The image of a pixel layer over the whole canvas at a level if it is in the cache (level > 0); a Stars and
        background layer's from its two images."""
        if is_group(lay):
            got = [self.images.get(self._key(sub_layer(lay, sub), level, None, canvas)) for sub in self._shown(lay)]
            return None if any(a is None for a in got) else self._together(lay, got, canvas.size(level)[::-1])
        return self.images.get(self._key(lay, level, None, canvas))

    def _shown(self, lay: dict) -> list[str]:
        """A Stars and background layer's images put together, the starless image first (a side whose result cannot
        be read is left out: the other one is shown)."""
        empty = lay.get("stars_pid", lay.get("pid")) is None            # its stars side has no workflow (#165)
        return [sub for sub in reversed(SUBS) if (lay.get("shown") or {}).get(sub, True) and not (empty and sub == "stars")
                and self.provider.readable(sub_layer(lay, sub))]

    def _together(self, lay: dict, got: list, hw) -> np.ndarray:
        """A Stars and background layer's images put together as its ``blend`` says (render/stars.blend: the same
        three modes as Post's)."""
        if not got:
            return np.zeros((*hw, 3), np.float32)
        return got[0] if len(got) == 1 else self.blend(lay["blend"], got[0], got[1])

    def layer_rgb(self, lay: dict, level: int, abox, canvas: Canvas, cache: bool = True, draft: bool = False,
                  stop=None) -> np.ndarray:
        """The image of a pixel layer over the box at the level; cached by content (a copy of a layer shares it). A
        draft takes the exact image when it is cached, else a quick look. A Stars and background layer's two images
        are rendered and cached apart, then put together (issue #19)."""
        if is_group(lay):
            got = [self.layer_rgb(sub_layer(lay, sub), level, abox, canvas, cache, draft, stop)
                   for sub in self._shown(lay)]
            return self._together(lay, got, (abox[3] - abox[1], abox[2] - abox[0]))
        if lay.get("part") == FUSION:
            return self._fusion_rgb(lay, level, abox, canvas, cache)
        W, H = canvas.size(level)
        whole = (0, 0, W, H)
        want = whole if level > 0 else abox
        key = self._key(lay, level, abox, canvas)
        a = self.images.get(key) if cache else None
        if a is None and draft and level > 0:
            key = self._key(lay, level, abox, canvas, True)
            a = self.images.get(key) if cache else None
        if a is None:
            a = self.provider.layer(lay, level, want, draft=draft and level > 0, stop=stop)
            self._drawn += 1
            if cache:
                self.images.put(key, _frozen(a), (lay["id"], lay.get("part"), level))
        elif cache:
            self.images.hold(key, (lay["id"], lay.get("part"), level))
        if level > 0 and abox != whole:
            x0, y0, x1, y1 = abox
            a = a[y0:y1, x0:x1]
        return a

    def _stars_of(self, lay, level, abox, canvas, cache, draft, check):
        """The function giving a pixel layer's stars image (its luminance over the box) for the Stars mask part: a
        Combine layers layer's Stars side (a combination's master layer: its shared stars), a workflow's Output
        layer: its Stars block's stars alone; None for a layer without (an image layer, a workflow without the
        block)."""
        if is_group(lay):
            side = sub_layer(lay, "stars")
            if lay.get("stars_pid", lay.get("pid")) is None or not self.provider.readable(side):
                return None
        elif lay.get("kind") == "workflow" and lay.get("part") is None:
            side = dict(lay, part="stars")
            if not self.provider.available(source_id(side)):
                return None
        else:
            return None
        return lambda: self.star_discs(side, canvas)

    def star_discs(self, side: dict, canvas: Canvas):
        """(key, stars) of a stars side (``find_stars``): found once per result in its stars as the block computed
        them, at full resolution over the whole canvas (``Provider.linear_stars``), kept for the session; Post's star
        settings are not in it, nor the level or the region shown."""
        sid = source_id(side)
        over = side.get("stars_over")
        rid = self.provider.stars_id(side)
        # a provider that cannot name the result (the tests'): its own token, never its id (an id is used again once
        # its object is gone: another provider's stars came back from the shared cache)
        token = self.provider.__dict__.setdefault("_stars_token", __import__("uuid").uuid4().hex)
        key = ("stars", rid if rid is not None else (token, sid, self.provider.version(sid), over,
                                                       self.provider.version(over) if over is not None else None),
               canvas.W, canvas.H, canvas.origin)
        with _STARS_LOCK:
            hit = _STARS.get(key)
            if hit is not None:
                _STARS.move_to_end(key)
                return key, hit
            if key in _SEARCHING:                         # being searched: the mask without its stars meanwhile
                self._stars_wait = True
                return None
            if key in _FAILED:                            # failed in this session: none, said
                self._stars_failed = True
                return None
        name = hashlib.blake2b(repr(key).encode(), digest_size=12).hexdigest() + ".npy"
        folder = getattr(self.provider, "stars_dir", None) if rid is not None else None
        if folder is not None:
            try:
                hit = np.load(Path(folder) / name)
            except (OSError, ValueError):
                hit = None
        if hit is None and not getattr(self.provider, "search_apart", False):   # (a provider of the tests: at once)
            hit = find_stars(lum(self.provider.linear_stars(side, 0, (0, 0, canvas.W, canvas.H))))
        if hit is None:                                   # searched in another process; the mask without them meanwhile
            L = lum(self.provider.linear_stars(side, 0, (0, 0, canvas.W, canvas.H)))
            with _STARS_LOCK:
                start = key not in _SEARCHING
                _SEARCHING[key] = True
            if start:
                import threading
                threading.Thread(target=_search_stars, args=(key, L, folder, name), name="stars search",
                                 daemon=True).start()
            self._stars_wait = True
            return None
        with _STARS_LOCK:
            _STARS[key] = hit
            while len(_STARS) > 16:
                _STARS.popitem(last=False)
        return key, hit

    # ------------------------------------------------------------ masks
    def part_raster(self, part, level, abox, below, canvas: Canvas, ctx: Ctx, seen=None) -> np.ndarray:
        """A mask part's plane. Kept by its settings (not its id or its operation, which only combine it), the level,
        the box and, for a part that reads the image (Luminance, Color range), the image's content: adding a part to
        a mask computes that part only. The plane is shared: never modified by the caller. ``seen``: the digest of
        ``below`` when the caller already made it."""
        if part["src"] == "stars":                      # kept by its settings and the stars it draws
            got = ctx.stars()
            if got is None:
                return self._part_plane(part, level, abox, below, canvas, ctx)
            key = (tuple(sorted((k, v) for k, v in part.items() if k not in ("id", "op"))), level, abox, canvas.W,
                   canvas.H, got[0])
            r = self.parts.get(key)
            if r is None:
                r = self._part_plane(part, level, abox, below, canvas, ctx)
                self.part_draws += 1
                self.parts.put(key, r)
            return r
        if part["src"] not in ("luminance", "color", "gradient"):
            return self._part_plane(part, level, abox, below, canvas, ctx)
        reads = part["src"] != "gradient"
        if reads:
            if seen is None:
                seen = {}
            if "d" not in seen:
                seen["d"] = hashlib.blake2b(np.ascontiguousarray(below).data, digest_size=16).digest()
        key = (tuple(sorted((k, v) for k, v in part.items() if k not in ("id", "op"))), level, abox, canvas.W,
               canvas.H, seen["d"] if reads else None)
        r = self.parts.get(key)
        if r is None:
            r = self._part_plane(part, level, abox, below, canvas, ctx)
            self.part_draws += 1
            self.parts.put(key, r)
        return r

    def _part_plane(self, part, level, abox, below, canvas: Canvas, ctx: Ctx) -> np.ndarray:
        src = part["src"]
        if src == "halpha":
            return ctx.halpha()
        if src == "luminance":
            L = lum(np.clip(below, 0, 1))
            lo, hi, soft = part["lo"] / 100, part["hi"] / 100, 0.06
            return np.clip((L - (lo - soft)) / (2 * soft), 0, 1) * np.clip(((hi + soft) - L) / (2 * soft), 0, 1)
        if src == "color":
            h, sat = hue(np.clip(below, 0, 1))
            return hue_weight(h, part["hue"], part["width"]) * np.clip(sat * 4, 0, 1)
        if src == "gradient":
            W, H = canvas.size(level)
            x0, y0, x1, y1 = abox
            ys = np.arange(y0, y1, dtype=np.float32)[:, None] / H
            xs = np.arange(x0, x1, dtype=np.float32)[None, :] / W
            ys, xs = np.broadcast_arrays(ys, xs)
            t = dict(zip(DIRECTIONS, (ys, 1 - ys, xs, 1 - xs)))[part["direction"]]
            pos, soft = part["position"] / 100, max(part["softness"] / 100, 0.01)
            return np.clip((t - (pos - soft / 2)) / soft, 0, 1).astype(np.float32)
        if src == "brush":
            return self.brush_raster(part, level, abox, canvas)
        if src == "stars":
            hw = (abox[3] - abox[1], abox[2] - abox[0])
            got = ctx.stars()
            if got is None:                                # a layer without stars: nothing (the part is greyed)
                return np.zeros(hw, np.float32)
            return draw_stars(got[1], level, abox, float(part["grow"]), float(part["feather"]) / 2 ** level)
        raise ValueError(src)

    def brush_raster(self, part, level, abox, canvas: Canvas) -> np.ndarray:
        """The strokes of a brush part at a level. The whole canvas, kept: only the new strokes are stamped (the exact
        view and the export compose the whole canvas at level 0: stamped again from the first stroke, each stroke made
        the next picture longer, +35 ms a stroke on a laptop). A region at level 0: only its box, stamped again."""
        strokes = part["strokes"]
        n, last = len(strokes), (strokes[-1] if strokes else None)
        f = 2 ** level
        W, H = canvas.size(level)
        if level == 0 and tuple(abox) != (0, 0, W, H):  # a region at full resolution: its box only, stamped again
            x0, y0, x1, y1 = abox
            r = np.zeros((y1 - y0, x1 - x0), np.float32)
            for x, y, rad, hard, paint in strokes:
                self.stamp(r, x - x0, y - y0, rad, hard, paint)
            return r
        key = (part["id"], level, canvas.W, canvas.H)
        hit = self._brush.get(key)
        if hit is not None and hit[0] == n and hit[1] == last:
            r = hit[2]
        else:
            if hit is not None and hit[0] < n and (hit[0] == 0 or strokes[hit[0] - 1] == hit[1]):
                r, start = hit[2], hit[0]
            else:
                r, start = np.zeros((H, W), np.float32), 0
            for x, y, rad, hard, paint in strokes[start:]:
                self.stamp(r, x / f, y / f, rad / f, hard, paint)
            self._brush[key] = (n, last, r)
        x0, y0, x1, y1 = abox
        return r[y0:y1, x0:x1]

    @staticmethod
    def stamp(r, x, y, rad, hard, paint):
        H, W = r.shape
        rad = max(rad, 0.8)
        xa, xb = int(max(0, x - rad - 1)), int(min(W, x + rad + 2))
        ya, yb = int(max(0, y - rad - 1)), int(min(H, y + rad + 2))
        if xa >= xb or ya >= yb:
            return
        yy, xx = np.mgrid[ya:yb, xa:xb]
        d = np.hypot(xx - x, yy - y) / rad
        core = hard / 100
        v = np.clip((1 - d) / max(1 - core, 0.05), 0, 1).astype(np.float32)
        if paint:
            r[ya:yb, xa:xb] = np.maximum(r[ya:yb, xa:xb], v)
        else:
            r[ya:yb, xa:xb] *= 1 - v

    def mask(self, lay, level, abox, below, canvas: Canvas, ctx: Ctx, sources=None, sigs: dict | None = None
             ) -> np.ndarray | None:
        """``sources``: the images a part that reads one (Luminance range, Color range) may read instead of
        ``below``, by its ``from`` (a combination's filter's mask: the picture, the filter, another filter; ``READS``);
        a part without it reads the first of them. ``sigs``: what each of those images depends on (``below``'s under
        None), the digests the composition makes: a part that reads one is kept by it, its pixels are not read to name
        it (``part_raster``)."""
        m = lay["mask"]
        if m is None:
            return None
        if not m["parts"]:                                 # a new mask shows the whole layer, as in the reference image editor
            return self._finish(np.ones(below.shape[:2], np.float32), m, level)
        out, seens = None, {}
        for part in m["parts"]:
            img, key = below, None
            if sources and part["src"] in READS_IMAGE:
                key = part.get("from") if part.get("from") in sources else next(iter(sources))
                img = sources[key]
            seen = seens.setdefault(key, {})
            if sigs is not None and key in sigs and "d" not in seen:
                seen["d"] = ("sig", sigs[key], bool(ctx.draft))   # (a draft's filters may take a quick look)
            r = self.part_raster(part, level, abox, img, canvas, ctx, seen)
            if out is None:
                out = np.zeros_like(r) if part["op"] == "add" else np.ones_like(r)
            if part["op"] == "add":
                out = np.maximum(out, r)
            elif part["op"] == "sub":
                out = out * (1 - r)
            else:
                out = out * r
        return self._finish(out, m, level)

    @staticmethod
    def _finish(out, m, level):
        if m["feather"]:
            out = gauss(out, m["feather"] / 2 ** level)
        if m["density"] < 100:
            out = 1 - (m["density"] / 100) * (1 - out)
        if m["invert"]:
            out = 1 - out
        return out

    # ------------------------------------------------------------ the whole image
    @staticmethod
    def margin_of(layers: list[dict]) -> int:
        """Full-resolution pixels around a region: REGION_MARGIN, or more when a visible filter reads farther (a spot
        of Spot healing and its source)."""
        need = REGION_MARGIN
        for lay in layers:
            if lay["kind"] == "filter" and lay["visible"] and lay["filter"] in FILTERS:
                fn = FILTERS[lay["filter"]].get("margin")
                if fn is not None:
                    need = max(need, int(math.ceil(fn(lay["params"]))) + 8)
        return need

    # ------------------------------------------------------------ what a composition reuses
    def _plan(self, stack, level, abox, canvas, want, hist_ids, draft) -> list[_Item]:
        """The rows ``compose`` walks, bottom first, with what the composite after each depends on (the same digests
        as the steps': no pixel is read). A hidden row is one only when its mask is asked for."""
        items, sig, edit = [], "empty", "empty"
        where = (level, abox, canvas.W, canvas.H, canvas.origin, bool(draft))
        for lay in stack:
            if lay["kind"] == "filter" and lay.get("target"):
                continue                                                  # applied with its own layer
            if not lay["visible"]:
                edit = _sig(edit, "hidden", lay["id"], _settings_of(lay))
                items.append(_Item(lay, "hidden", sig=sig, edit=edit, needs=lay["id"] in want,
                                   tkey=_sig(sig if _reads_image(lay.get("mask")) else None, _settings_of(lay),
                                             lay["id"], where)))
                continue
            if is_pixel(lay):
                sids = source_ids(lay)
                if not any(self.provider.readable(x) for _s, x in side_layers(lay)):      # (one side of a Combine layers layer
                    continue                                              # that cannot be read: the other one is composed)
                own = [fl for fl in stack if fl["kind"] == "filter" and fl.get("target") == lay["id"]]
                ss, mine = self._src_sig(lay, own, level, abox, canvas, draft), [(_settings_of(f), f["visible"]) for f in own]
                sig = _sig(sig, ss, _settings_of(lay))
                edit = _sig(edit, "pixel", _settings_of(lay), mine, [self.provider.version(x) for x in sids])
                ids = {lay["id"], *(f["id"] for f in own), *((lay["id"], s) for s in subs_of(lay))}
                it = _Item(lay, "pixel", own, sig=sig, edit=edit, needs=bool(ids & want or ids & hist_ids),
                           opaque=lay["mode"] == "Normal" and lay["opacity"] / 100 >= 0.999 and lay["mask"] is None
                           and (is_group(lay) or self.provider.cover(lay, level, abox) is None))
                # its masks read its own image (and its sides', its filters' inputs), never what is under it
                it.tkey = _sig(ss, _settings_of(lay), mine, sorted(map(str, ids)), where)
                it.src = ss
            else:
                before = sig
                sig = _sig(sig, _settings_of(lay))
                edit = _sig(edit, "filter", _settings_of(lay))
                it = _Item(lay, "filter", sig=sig, edit=edit, needs=lay["id"] in want or lay["id"] in hist_ids,
                           tkey=_sig(before if _reads_image(lay.get("mask")) else None, _settings_of(lay), lay["id"],
                                     where))
            items.append(it)
        return items

    @staticmethod
    def _stats_keys(items, at: int) -> list[tuple]:
        """The whole-image measures (``Ctx.whole_stats``, ``Ctx.whole_measure`` taken at level ``at``) the visible
        filters of a plan use: their input's key, the same at any level and region (made of the rows' ``edit``)."""
        out, before = [], "empty"

        def keys(base, fl):
            return [(base, w[0], at) if len(w) > 1 else (base, w[0]) for w in _wants_whole(fl)]
        for it in items:
            if it.kind == "filter":
                out += keys(_sig(before, it.lay["id"]), it.lay)
            elif it.kind == "pixel":
                for fl in it.own:
                    out += keys(_sig(it.edit, fl["id"]), fl) if fl["visible"] else []
            before = it.edit
        return out

    def _img_sig(self, lay, level, abox, canvas, draft=False) -> str:
        """What a layer's image depends on (``layer_rgb``): its key's digest; in a draft, its quick look's when its exact
        image is not kept (a draft's composition reads the quick look then, and keeps what it makes by that)."""
        key = self._key(lay, level, abox, canvas)
        if draft and level > 0 and key not in self.images.items:
            return _sig(self._key(lay, level, abox, canvas, True))
        return _sig(key)

    def _src_sig(self, lay, own, level, abox, canvas, draft=False) -> str:
        """What a pixel layer's image after its own filters depends on: the digest ``compose`` and ``group_rgb`` make
        while they compute it, made without computing it."""
        if is_group(lay):
            sigs = [_image_settings(lay)]
            for sub in self._shown(lay):
                if sub == "stars" and not (lay.get("shown") or {}).get("real", True) and \
                        any(is_synth(fl) and fl.get("sub") == "stars" for fl in own):
                    sig = _sig("no real stars", (abox[3] - abox[1], abox[2] - abox[0]))
                elif mix_at(lay, sub) is not None:
                    sig = self._mix_sig(lay, own, sub, level, abox, canvas, draft)
                else:
                    sig = self._img_sig(sub_layer(lay, sub), level, abox, canvas, draft)
                synth = sub_layer(lay, "synth")
                if sub == "stars" and self.provider.available(source_id(synth)):
                    sig = _sig(sig, self._img_sig(synth, level, abox, canvas, draft))
                for fl in own:
                    if fl.get("sub") == sub and fl["visible"]:
                        sig = _sig(sig, _settings_of(fl))
                sigs.append(sig)
            sig = _sig(*sigs)
        else:
            sig = self._img_sig(lay, level, abox, canvas, draft)
        for fl in own:
            if not fl.get("sub") and fl["visible"]:
                sig = _sig(sig, _settings_of(fl))
        return sig

    def _follow(self, items: list[_Item]):
        """Where the composition changed since the last one followed: the lowest row that differs is the one being
        edited; the composite under it (its backdrop) is what the next compositions keep."""
        edits = [it.edit for it in items]
        n = next((i for i, (a, b) in enumerate(zip(edits, self._edits)) if a != b), min(len(edits), len(self._edits)))
        if edits != self._edits:
            self._focus = edits[n - 1] if 0 < n <= len(edits) else None
            self._edited = edits[n] if n < len(edits) else None
        self._edits = edits

    def _start(self, items: list[_Item], where, thumbs: bool) -> tuple[int, np.ndarray | None]:
        """The first row to compute, and the composite under it (None: black). The rows under an opaque one (Normal,
        100 %, no mask) never show; a kept composite stands for every row under it. A row whose mask or histogram is
        asked for is computed, and so is one whose mask thumbnails are asked for and not kept."""
        def blocks(it):
            if it.needs:
                return True
            if not thumbs:
                return False
            lay = it.lay
            masked = lay.get("mask") is not None or bool(lay.get("side_masks")) or \
                any(f.get("mask") is not None for f in it.own)
            return masked and (where is None or it.tkey not in self._thumbs)

        top = next((i for i, it in enumerate(items) if blocks(it)), len(items))
        for s in range(top, 0, -1):
            if where is not None:
                back = self.backdrops.get((items[s - 1].sig, where))
                if back is not None:
                    return s, back
            if s < len(items) and items[s].opaque:
                return s, None
        return 0, None

    def _filtered(self, it: _Item, where, thumbs: bool) -> np.ndarray | None:
        """A pixel layer's image after its own filters when it is kept (by all that made it): its layer and its filters
        are then not computed. Not when one of its filters' masks or histograms is asked for, nor for a layer without
        filters (its image is the layer's own, ``layer_rgb`` keeps it)."""
        if where is None or it.needs:
            return None
        lay = it.lay
        if not (is_group(lay) or any(not f.get("sub") and f["visible"] for f in it.own)):
            return None
        masked = lay.get("side_masks") or any(f.get("mask") is not None for f in it.own)
        if thumbs and masked and it.tkey not in self._thumbs:
            return None
        return self.steps.get((it.src, where))

    def _keep_composite(self, key, comp, holder):
        if key in self.backdrops.items:
            self.backdrops.hold(key, holder)
        else:
            self.backdrops.put(key, _frozen(comp), holder)

    def compose(self, layers: list[dict], level: int, box=None, *, want=(), thumbs: bool = False, stop=None,
                cache: bool = True, draft: bool = False, hists=(), track: bool = False,
                canvas: Canvas | None = None) -> Composite | None:
        """The composite (float 0..1) at a resolution, for a box of the canvas at that level (default: all of it).
        ``want``: ids of layers whose mask is returned (also those hidden). ``thumbs``: also a thumbnail of every mask
        computed. ``stop``: called between the steps (and by the engine between its stages), True abandons the work
        (``Superseded``). ``draft``: a quick look of the pixel layers whose exact image is not cached, and of the filters
        that have one (the curves' tables). ``hists``: ids of filters whose input's histograms are returned (the
        Curves editor's and the stretches' graphs). ``track``: this is the composition being edited (Post's own
        renders, not the before image or another version's): where it changed says which row's backdrop to keep.
        ``canvas``: the canvas to compose on, instead of the layers' own (the other side of a compare: the picture shown's
        grid). None when no pixel layer has a result."""
        # a search ended: nothing to drop, what was made without its stars was not kept (``_stars_missing``); what
        # does not read them (the other layers, the steps under a Stars part) stays
        self._stars_new = False
        self._stars_wait = self._stars_failed = self._stars_missing = False
        self.applied = set()
        own = self.canvas_of(layers)
        if own is None:
            return None
        canvas = canvas or own
        W, H = canvas.size(level)
        if box is None:
            box = (0, 0, W, H)
        x0, y0, x1, y1 = box
        if not (0 <= x0 < x1 <= W and 0 <= y0 < y1 <= H):
            raise ValueError(f"box {box} outside the {W} x {H} canvas at level {level}")
        pad = self.margin_of(layers) if level == 0 and box != (0, 0, W, H) else 0
        abox = (max(0, x0 - pad), max(0, y0 - pad), min(W, x1 + pad), min(H, y1 + pad))
        ax0, ay0, ax1, ay1 = abox
        comp = np.zeros((ay1 - ay0, ax1 - ax0, 3), np.float32)
        out = Composite(comp, level, box)
        ctx = Ctx(self, level, abox, draft)
        ctx.hist_ids, ctx.hists = set(hists), out.hists
        ctx.canvas_wh = (W, H)                                            # the whole canvas at this level
        ctx.comp_canvas = canvas                                          # (a level between two powers of two)
        # the image after each step is kept: hiding a filter recomputes only what is above it (a region: not kept)
        # (a draft keeps what it makes apart from the exact images: its filters may take a quick look)
        where = ((level, (canvas.W, canvas.H, canvas.origin)) + (("draft",) if draft else ())
                 if cache and abox == (0, 0, W, H) else None)
        want = set(want)
        self._drawn = 0
        check = stop or (lambda: False)
        ctx.check = check                                                 # (a filter made in bands stops between them)
        for c in (self.images, self.steps, self.backdrops):
            c.begin(level)

        def checkpoint():
            if check():
                raise Superseded()

        made: dict = {}                                                   # the mask thumbnails of the row being made

        def keep_mask(lay, m, key=None, thumb=None):
            """``key``: (layer id, side) for a side mask of a Combine layers layer, else the layer's id. Returns the
            thumbnail made; ``thumb`` without ``m``: a thumbnail kept, given again."""
            key = lay["id"] if key is None else key
            if m is None:
                if thumb is not None and thumbs:
                    out.mask_thumbs[key] = made[key] = thumb
                return None
            if key in want:
                out.masks[key] = m[y0 - ay0:y0 - ay0 + (y1 - y0), x0 - ax0:x0 - ax0 + (x1 - x0)]
            if thumbs:
                out.mask_thumbs[key] = made[key] = thumb_of(m)
                return made[key]
            return None

        stack = list(reversed(layers))                                    # bottom first
        items = self._plan(stack, level, abox, canvas, want, ctx.hist_ids, draft)
        at = level + stats_level(W, H)
        if abox != (0, 0, W, H) and any(key not in self.stats for key in self._stats_keys(items, at)):
            # a region: the measures of the whole image its filters need, made on the whole canvas at the measure
            # level first (Ctx.whole_stats; review 10.1)
            self.compose(layers, at, stop=stop, cache=cache, draft=draft, canvas=canvas)
        if track:
            self._follow(items)
        start, back = self._start(items, where, thumbs)
        self.skipped = start
        for it in items[:start]:                                          # rows not computed: their mask thumbnails
            if thumbs and it.tkey in self._thumbs:
                self._thumbs.move_to_end(it.tkey)
                out.mask_thumbs.update(self._thumbs[it.tkey])
        if back is not None:
            comp = back                                                   # (kept read only: nothing writes into it)
        comp_sig = items[start - 1].sig if start else "empty"
        under = any(it.kind == "pixel" for it in items[:start])           # an image layer drawn below the next one
        edited_at = next((i for i, it in enumerate(items) if it.edit == self._edited), None)
        for i in range(start, len(items)):
            it = items[i]
            lay = it.lay
            checkpoint()
            made = {}
            ctx.fusion_halpha = None                                      # (set again for a master layer's row)
            ctx.stars_fn = None                                           # (and its stars, for a pixel layer)
            ctx._memo.pop("stars", None)
            if it.kind == "hidden":
                if lay["id"] in want or (thumbs and lay["mask"] is not None):
                    keep_mask(lay, self.mask(lay, level, abox, comp, canvas, ctx, sigs={None: comp_sig}))
            elif it.kind == "pixel":
                own = it.own
                edited = i == edited_at                                   # its steps are kept, not only its image
                tops = [fl for fl in own if not fl.get("sub") and fl["visible"]]
                ctx.fusion_halpha = self._fusion_halpha_of(lay, level, abox, canvas, cache)
                ctx.stars_fn = self._stars_of(lay, level, abox, canvas, cache, draft, check)
                ctx._memo.pop("fusion halpha", None)
                got = self._filtered(it, where, thumbs)
                if got is not None:                                       # its image after its filters, kept
                    src, sig = got, it.src
                    if thumbs and it.tkey in self._thumbs:
                        made.update(self._thumbs[it.tkey])
                        out.mask_thumbs.update(made)
                else:
                    ctx.row_edit = it.edit                                # (the key of its filters' measures)
                    # the highest of its own filters' steps kept: the layer's image and the filters under it are not
                    # made again (a release of its top filter makes that filter only)
                    jump = self._own_jump(lay, own, level, abox, canvas, ctx, where, want, thumbs, draft)
                    if jump is not None:
                        done, src, sig, replay = jump
                        for fid, t in replay:                             # the masks' thumbnails of the filters passed
                            keep_mask({"id": fid}, None, fid, thumb=t)
                    elif is_group(lay):
                        self._keep = edited
                        src, sig = self.group_rgb(lay, own, level, abox, canvas, ctx, cache, draft, check,
                                                  keep_mask, where, want, thumbs)
                        self._keep = True
                        if where is not None and not tops and not self._stars_missing:   # (with filters: the last step)
                            self.steps.put((sig, where), _frozen(src), (lay["id"], "src", level))
                    else:
                        src = self.layer_rgb(lay, level, abox, canvas, cache, draft, check)
                        sig = self._img_sig(lay, level, abox, canvas, draft)
                    # at full resolution its steps are kept only under the filter being edited, and its last: all of
                    # them (327 MB each on a 27 Mpx image) took the room of the screen's
                    under_edit = self._own_edited(lay, tops)
                    for k, fl in enumerate(own):                          # this layer's own filters, bottom first
                        if fl.get("sub") or (jump is not None and k <= done):
                            continue
                        checkpoint()
                        if not fl["visible"]:
                            if fl["id"] in want or (thumbs and fl["mask"] is not None):
                                keep_mask(fl, self.mask(fl, level, abox, src, canvas, ctx, sigs={None: sig}))
                            continue
                        ctx.stats_key = _sig(it.edit, fl["id"])
                        src, fm, sig = self._step(fl, src, sig, where, want, thumbs, level, abox, canvas, ctx,
                                                  keep=(edited or fl is tops[-1]) if level > 0 else
                                                  fl["id"] == under_edit or fl is tops[-1])
                        self._keep_step_thumb(keep_mask(fl, fm), sig, where)
                comp_sig = _sig(comp_sig, sig, _settings_of(lay))
                m = self.mask(lay, level, abox, src, canvas, ctx, sigs={None: sig})
                keep_mask(lay, m)
                if not is_group(lay):                                     # an image fitted inside the canvas: clear around it
                    cov = self.provider.cover(lay, level, abox)
                    if cov is not None:
                        m = cov if m is None else m * cov
                # nothing under it: no mode to apply (never blended over a black that is not there); the mode stays
                # stored on the layer and applies again as soon as an image layer is under it
                src = self.blend(lay["mode"] if under else "Normal", comp, src)
                under = True
                a = lay["opacity"] / 100
                if m is None and a >= 0.999:
                    comp = src
                else:
                    a = a if m is None else (a * m)[..., None]
                    comp = comp + (src - comp) * a
            else:                                     # above the row edited its input changes at each edit: not kept
                ctx.stats_key = _sig(items[i - 1].edit if i else "empty", lay["id"])
                comp, m, comp_sig = self._step(lay, comp, comp_sig, where, want, thumbs, level, abox, canvas, ctx,
                                               keep=edited_at is None or i <= edited_at)
                keep_mask(lay, m)
            if comp_sig != it.sig:                   # the plan missed what this row depends on: keep nothing more
                self.plan_errors += 1
                where = None
            if self._stars_missing:                  # made without the stars being searched: nothing more kept
                where = None
            if where is None:
                continue
            if thumbs:
                self._thumbs[it.tkey] = made
                self._thumbs.move_to_end(it.tkey)
                while len(self._thumbs) > THUMB_ITEMS:
                    self._thumbs.popitem(last=False)
            key = (it.sig, where)
            if it.edit == self._focus:                                    # the backdrop of the row edited
                self._keep_composite(key, comp, ("backdrop", level))
            elif i == len(items) - 1 and self.backdrops.bytes + comp.nbytes <= self.backdrops.cap:
                self._keep_composite(key, comp, ("last", level))         # never instead of a backdrop
        cut = (slice(y0 - ay0, y0 - ay0 + (y1 - y0)), slice(x0 - ax0, x0 - ax0 + (x1 - x0)))
        out.rgb = np.clip(comp[cut], 0, 1).astype(np.float32, copy=False)
        # (a picture made of images kept from before the search says it too: a search is running)
        out.notes = set(self.provider.notes()) | ({STARS_SEARCHING} if self._stars_wait or _SEARCHING else set()) \
            | ({STARS_FAILED} if self._stars_failed else set())
        out.drawn = self._drawn
        return out

    def group_rgb(self, lay, own, level, abox, canvas, ctx, cache, draft, check, keep_mask, where, want, thumbs):
        """A Stars and background layer whose sub-layers have filters (or one of them hidden): its two images, each
        through its own filters (bottom first), put together as its ``blend`` says (render/stars.blend: the same three
        modes as Post's); with what the result depends on. The stars are a group (#93): the real stars at its bottom,
        then its rows bottom first, the Synthetic stars layer among them (Normal through its mask: it covers what is
        under it); the real stars' eye off, the group starts from black."""
        got, sigs = [], [_image_settings(lay)]
        for sub in self._shown(lay):                                      # the starless image first, then the stars
            part = sub_layer(lay, sub)
            if sub == "stars" and not (lay.get("shown") or {}).get("real", True) and \
                    any(is_synth(fl) and fl.get("sub") == "stars" for fl in own):
                hw = (abox[3] - abox[1], abox[2] - abox[0])
                x, sig = np.zeros((*hw, 3), np.float32), _sig("no real stars", hw)
            elif mix_at(lay, sub) is not None:                      # the combination's filters, mixed here
                x = self._mix_side(lay, own, sub, level, abox, canvas, ctx, cache, draft, check, keep_mask, where, want,
                                   thumbs)
                sig = self._mix_sig(lay, own, sub, level, abox, canvas, draft)
            else:
                x = None                                                  # (made below, unless a step is kept)
                sig = self._img_sig(part, level, abox, canvas, draft)
            synth = sub_layer(lay, "synth")
            ctx.synth = None
            if sub == "stars" and self.provider.available(source_id(synth)):
                ctx.synth = lambda: self.layer_rgb(synth, level, abox, canvas, cache, draft, check)
                sig = _sig(sig, self._img_sig(synth, level, abox, canvas, draft))
            chain = [fl for fl in own if fl.get("sub") == sub]
            # the highest of the side's filters' steps kept: its image and the filters under it are not made again
            jump = self._chain_jump(chain, sig, where, want, thumbs, ctx, level) if x is None else None
            done = -1
            if jump is not None:
                done, x, sig, replay = jump
                for fid, t in replay:
                    keep_mask({"id": fid}, None, fid, thumb=t)
            elif x is None:
                x = self.layer_rgb(part, level, abox, canvas, cache, draft, check)
            shown = [fl for fl in chain if fl["visible"]]
            under_edit = self._own_edited((lay["id"], sub), shown)
            for k, fl in enumerate(chain):
                if k <= done:
                    continue
                if check():
                    raise Superseded()
                if not fl["visible"]:
                    continue
                ctx.stats_key = _sig(ctx.row_edit, fl["id"])
                x, fm, sig = self._step(fl, x, sig, where, want, thumbs, level, abox, canvas, ctx,
                                        keep=True if level > 0 else fl["id"] == under_edit or fl is shown[-1])
                self._keep_step_thumb(keep_mask(fl, fm), sig, where)
            side = (lay.get("side_masks") or {}).get(sub)
            if side is not None:                                          # that side's mask, before the blend
                srcs = self._side_sources(lay, sub, side, x, got, level, abox, canvas, cache) if has_mix(lay) else None
                m = self.mask({"mask": side}, level, abox, x, canvas, ctx, srcs,
                              {"self": sig} if srcs else {None: sig})     # (the image before this mask)
                keep_mask(lay, m, (lay["id"], sub))
                x = x * m[..., None]
            got.append(x)
            sigs.append(sig)
        for sub in SUBS:                                                  # a hidden side's mask is still shown
            side = (lay.get("side_masks") or {}).get(sub)
            if side is not None and sub not in self._shown(lay) and ((lay["id"], sub) in want or thumbs):
                hw = (abox[3] - abox[1], abox[2] - abox[0])
                keep_mask(lay, self.mask({"mask": side}, level, abox, np.zeros((*hw, 3), np.float32), canvas, ctx,
                                         sigs={None: _sig("off", hw)}), (lay["id"], sub))
        return self._together(lay, got, (abox[3] - abox[1], abox[2] - abox[0])), _sig(*sigs)

    def _own_jump(self, lay, own, level, abox, canvas, ctx, where, want, thumbs, draft=False):
        """(the place in ``own`` of the highest of a layer's own filters whose result is kept, that result (a copy),
        what it depends on), or None: the digests ``_step`` keeps them by, made without computing anything (the
        layer's image's, ``_src_sig`` without its own filters, then each visible filter's settings). None when a mask
        or a histogram of the layer's sides or of a filter under it is asked for (they are made on the way)."""
        if where is None or not any(not f.get("sub") and f["visible"] for f in own):
            return None
        if is_group(lay) and (thumbs and (lay.get("side_masks") or any(f.get("sub") and f.get("mask") for f in own))
                              or any((lay["id"], s) in want for s in subs_of(lay))
                              or any(f.get("sub") and (f["id"] in want or f["id"] in ctx.hist_ids) for f in own)):
            return None
        sig = self._src_sig(lay, [f for f in own if f.get("sub")], level, abox, canvas, draft)
        return self._chain_jump([f for f in own if not f.get("sub")], sig, where, want, thumbs, ctx, level,
                                positions=[k for k, f in enumerate(own) if not f.get("sub")])

    def _chain_jump(self, chain, sig, where, want, thumbs, ctx, level, positions=None):
        """A chain of filters (bottom first) on an image whose digest is ``sig``: (the place of the highest whose result
        is kept, that result (a copy), its digest, the mask thumbnails of the filters passed), or None (``_own_jump``).
        ``positions``: the places to say, in place of the chain's own."""
        if where is None:
            return None
        best, thumbs_on_way = None, []
        for k, fl in enumerate(chain):
            if fl["id"] in want or fl["id"] in ctx.hist_ids:
                break                                       # (its mask or histogram is made from what is under it)
            if not fl["visible"]:
                if thumbs and fl.get("mask") is not None:
                    break
                continue
            sig = _sig(sig, _settings_of(fl))
            if thumbs and fl.get("mask") is not None:       # its mask's thumbnail, kept with its step, else made
                t = self._step_thumbs.get((sig, where))
                if t is None:
                    break
                thumbs_on_way.append((fl["id"], t))
            if (sig, where) in self.steps.items:
                best = (k, sig, len(thumbs_on_way))
        if best is None:
            return None
        key = (best[1], where)
        got = self.steps.get(key)
        fl = chain[best[0]]
        self.steps.hold(key, (fl["id"], fl.get("sub"), level))
        return (best[0] if positions is None else positions[best[0]]), got, best[1], thumbs_on_way[:best[2]]

    def _keep_step_thumb(self, t, sig, where):
        """A filter's mask thumbnail, kept with its step (``_chain_jump`` gives it again when it passes the filter)."""
        if t is None or where is None or self._stars_missing:
            return
        self._step_thumbs[(sig, where)] = t
        self._step_thumbs.move_to_end((sig, where))
        while len(self._step_thumbs) > THUMB_ITEMS:
            self._step_thumbs.popitem(last=False)

    def _own_edited(self, lay, tops) -> int | None:
        """The id of the filter of a layer whose result the filter being edited reads (the one under it, among the
        layer's own visible filters), None when it reads the layer's image or nothing was edited: the first of them
        whose settings differ from the composition before; kept until another one changes."""
        now = [(fl["id"], _settings_of(fl)) for fl in tops]
        lid = lay["id"] if isinstance(lay, dict) else lay        # (a side of a Combine layers layer: (its id, side))
        was, under = self._own_seen.get(lid, (None, None))
        if was is not None and was != now:
            j = next((i for i, (a, b) in enumerate(zip(now, was)) if a != b), min(len(now), len(was)))
            under = now[j - 1][0] if 0 < j <= len(now) else None
        self._own_seen[lid] = (now, under)
        return under

    def _step(self, fl, under, sig, where, want, thumbs, level, abox, canvas, ctx, keep=True):
        """``_apply``, kept: (result, mask, what the result depends on). ``where``: None when it is not to be kept;
        ``keep``: False, the result is kept only where there is room (an image a change upstream may replace each
        time). The filter holds its result: when its settings or what is under it change, the one it held goes."""
        new = _sig(sig, _settings_of(fl))
        key = None if where is None else (new, where)
        got = None if key is None else self.steps.get(key)
        holder = (fl["id"], fl.get("sub"), level)
        if got is not None:
            self.steps.hold(key, holder)
        if got is None:
            was = self._stars_missing
            self._stars_missing = False
            ctx.input_sig = None if ctx.draft is None else _sig(sig, bool(ctx.draft))   # (its whole-image measures)
            try:
                out, m = self._apply(fl, under, level, abox, canvas, ctx, sig)
            finally:
                ctx.input_sig = None
            missing, self._stars_missing = self._stars_missing, was or self._stars_missing
            if key is not None and missing:                 # its mask's Stars part waits for its stars: not kept
                self.steps._unhold(holder, key)
                key = None
            if key is not None:
                self.steps._unhold(holder, key)                   # the one it held before goes (settings changed)
                # a filter may work in place on what it is given: a copy is kept. A step of lesser use (``keep``
                # False: under a layer's last filter, or above the row edited) is kept only where there is room,
                # never instead of another one: hiding the filter above it still finds it (#87). At full resolution only
                # a step kept on purpose (``keep``), and only where there is room (``ByteLRU``)
                if level == 0:
                    if keep and self._keep:
                        self.steps.put(key, _frozen(out), holder)
                elif (keep and self._keep) or self.steps.bytes + out.nbytes <= self.steps.cap:
                    self.steps.put(key, _frozen(out), holder)
            return out, m, new
        if fl["id"] in ctx.hist_ids:
            ctx.hists[fl["id"]] = pf_curves.histograms(under)
        wanted = fl["mask"] is not None and (fl["id"] in want or thumbs)
        return got, self.mask(fl, level, abox, under, canvas, ctx, sigs={None: sig}) if wanted else None, new

    def _apply(self, fl, under, level, abox, canvas, ctx, sig=None):
        """A filter layer on an image (its own layer's, or everything below): the filtered image, through the layer's
        mask, opacity and mode. Returns (result, mask)."""
        if self.progress is not None and not ctx.draft:
            try:
                self.progress(fl)
            except Exception:  # noqa: BLE001 - the banner's word is a courtesy: never a failure
                pass
        t0 = time.perf_counter()
        if fl["id"] in ctx.hist_ids:
            ctx.hists[fl["id"]] = pf_curves.histograms(under)
        out = run_filter(fl["filter"], under, fl["params"], ctx)
        m = self.mask(fl, level, abox, under, canvas, ctx, sigs=None if sig is None else {None: sig})
        out = self.blend(fl["mode"], under, out)
        a = fl["opacity"] / 100
        if m is None and a >= 0.999:
            res = out
        else:
            a = a if m is None else (a * m)[..., None]
            res = under + (out - under) * a
        self.applied.add(fl["id"])
        if level > 0 and not ctx.draft:                       # its cost on the screen (the row's hover says it)
            self.costs[fl["id"]] = time.perf_counter() - t0
        return res, m

    @staticmethod
    def blend(mode, a, b):
        if mode == "Lighten":
            return np.maximum(a, b)
        if mode == "Screen":
            t = np.clip(b, 0, 1)
            np.subtract(1.0, t, out=t)
            t *= 1.0 - a                                     # (1 - a)(1 - b)
            np.subtract(1.0, t, out=t)
            return t
        if mode == "Linear dodge (add)":
            return np.minimum(a + b, 1.0)
        if mode == "Multiply":
            return a * b
        if mode == "Soft light":
            b = np.clip(b, 0, 1)
            return (1 - 2 * b) * a * a + 2 * b * a
        return b


def _reads_image(m: dict | None) -> bool:
    """Whether a mask reads the image it is on (a Luminance or Color range part); the others read the settings."""
    return bool(m) and any(p.get("src") in ("luminance", "color") for p in m.get("parts") or ())


def _settings_of(lay: dict) -> str:
    """What a layer or a filter does to the image: all of it but its name and place in the list. A brush's strokes
    count by their number and a digest (written out whole, each request of a painted mask grew with its strokes); a
    mask part's id is no setting (a new version copies its layers with ids of their own: the same settings are the
    same images, its switch reuses what the other one computed)."""
    d = {k: v for k, v in lay.items() if k not in ("id", "name", "target", "visible")}
    for key in ("mask", "side_masks"):
        if d.get(key):
            d[key] = _light_masks(d[key], key == "side_masks")
    return json.dumps(d, sort_keys=True, default=str)


def _image_settings(lay: dict) -> str:
    """What a Stars and background layer's image (its two images put together, ``group_rgb``) depends on of its own
    settings: all of them but its mask, opacity and mode, which act on the composition (its row), never on its image:
    a part added to its mask reads its image as it was (``mask``'s ``sigs``)."""
    return _settings_of({k: v for k, v in lay.items() if k not in ("mask", "opacity", "mode")})


def _strokes_digest(strokes) -> str:
    # tuples of numbers and booleans hash the same in every process (no hash seed for them)
    return f"{len(strokes)}:{hash(tuple(map(tuple, strokes)))}"


def _light_masks(m, several: bool):
    if several:
        return {k: _light_masks(v, False) for k, v in m.items()}
    if not m.get("parts"):
        return m
    return dict(m, parts=[{k: (_strokes_digest(v) if k == "strokes" and v else v) for k, v in p.items() if k != "id"}
                          for p in m["parts"]])


def eyes_open(lay: dict, sub: str) -> bool:
    """Whether the eye of a Combine layers layer's side or node, and those of the nodes above it, are open: a node
    whose parent's eye is off is out of the picture (its own eye stays as it was)."""
    shown = lay.get("shown") or {}
    while True:
        if not shown.get(sub, True):
            return False
        if sub in SUBS or not pf.is_path(sub):
            return True
        sub = pf.parent(sub)


def visible_signature(layers: list[dict], want=()) -> str:
    """What the composite depends on in these layers: the rows that are shown, themselves and through their parents (a
    hidden layer, a hidden filter, a filter under a hidden layer or a hidden image of a Stars and background layer
    count only as hidden). ``want`` are the mask keys the screen shows (a hidden row's selected mask counts). Two
    lists with the same signature give the same picture: a change to a row that is not shown needs no render."""
    ids = {k[0] if isinstance(k, tuple) else k for k in want}
    by_id = {x["id"]: x for x in layers}
    out = []
    for lay in layers:
        up = by_id.get(lay.get("target")) if lay["kind"] == "filter" and lay.get("target") else None
        shown = lay["visible"]
        if up is not None:
            shown = shown and up["visible"] and (not lay.get("sub") or eyes_open(up, lay["sub"]))
        out.append((lay["id"], lay.get("target"), bool(shown), _settings_of(lay) if shown or lay["id"] in ids else None))
    return _sig(out)


def _sig(*what) -> str:
    return hashlib.sha1(repr(what).encode("utf-8")).hexdigest()


def only_layer(layers: list[dict], lid: int) -> list[dict]:
    """A pixel layer alone with its own filters (visible, opaque, Normal, no mask): what 'each layer as a separate
    file' writes."""
    base = next(lay for lay in layers if lay["id"] == lid)
    own = [dict(f, visible=f["visible"]) for f in layers if f["kind"] == "filter" and f.get("target") == lid]
    return snapshot(own + [dict(base, visible=True, opacity=100.0, mode="Normal", mask=None)])
