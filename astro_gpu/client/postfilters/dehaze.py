"""Dehaze for night skies: the veil (residual sky glow, haze, a gradient left over) measured on a large-scale background
map, then taken off with the haze model, the white kept (issue #22).

The dark channel prior (K. He, J. Sun, X. Tang, "Single Image Haze Removal Using Dark Channel Prior", CVPR 2009 /
IEEE TPAMI 2011), the reference of daylight dehazing, does not hold on a night sky: the veil there is the darkest level,
not the brightest, and its "airlight" would be taken from the nebula's cores or the stars. The veil is measured instead
as the background of the astronomical source extractors (E. Bertin, S. Arnouts, "SExtractor", A&AS 117, 1996; its
documentation, Background): the image cut in meshes, each mesh's level its mode after an iterative 3-sigma clipping
(2.5 median - 1.5 mean, the median when the two differ by more than 30 % of the scatter), the grid of levels filtered by
a 3 x 3 median and interpolated by bicubic splines. It is then taken off with He's haze model I = J t + A (1 - t),
the airlight A the white (1): J = (I - h) / (1 - h), the veil h = Amount x the map, so that the white stays white and
the sky keeps (1 - Amount) of its level, never crushed (He's "a little haze kept", omega <= 0.95).

Each colour apart (a veil has a colour). Faint nebulosity wider than the meshes counts as veil: Scale is the key
control.
"""
from __future__ import annotations

import hashlib
from collections import OrderedDict
from threading import Lock

import cv2
import numpy as np

#: the clipping of a mesh: iterations, sigmas; the share of the scatter beyond which the mode falls back to the median
CLIP_ITER, CLIP_SIGMA, MODE_TOLERANCE = 3, 3.0, 0.3
AMOUNT_MAX = 0.95


def _nanmedian(a: np.ndarray) -> np.ndarray:
    """``np.nanmedian(a, axis=2, keepdims=True)`` for meshes of under 600 pixels, the same values: numpy takes them
    through masked arrays there (a draft's meshes of 8 px: 35 ms of its 52 ms); here a sort, NaN last, and the middle
    one or the mean of the middle two of the values that are not NaN."""
    if a.shape[2] >= 600:
        return np.nanmedian(a, axis=2, keepdims=True)
    s = np.sort(a, axis=2)
    n = np.count_nonzero(~np.isnan(a), axis=2)[..., None]
    lo = np.take_along_axis(s, np.maximum((n - 1) // 2, 0), 2)
    hi = np.take_along_axis(s, np.minimum(n // 2, a.shape[2] - 1), 2)
    out = (lo + hi) / 2
    out[n == 0] = np.nan
    return out


def mesh_levels(c: np.ndarray, m: int) -> np.ndarray:
    """The background level of each m x m mesh of one channel (h, w), SExtractor's: (gh, gw)."""
    h, w = c.shape
    gh, gw = max(1, -(-h // m)), max(1, -(-w // m))
    pad = np.pad(c, ((0, gh * m - h), (0, gw * m - w)), mode="reflect" if min(h, w) > 1 else "edge")
    a = pad.reshape(gh, m, gw, m).transpose(0, 2, 1, 3).reshape(gh, gw, m * m).astype(np.float64)
    for _ in range(CLIP_ITER):
        med = _nanmedian(a)
        sd = np.nanstd(a, axis=2, keepdims=True)
        a = np.where(np.abs(a - med) <= CLIP_SIGMA * sd + 1e-12, a, np.nan)
    med, mean, sd = _nanmedian(a)[..., 0], np.nanmean(a, axis=2), np.nanstd(a, axis=2)
    mode = 2.5 * med - 1.5 * mean
    return np.where(np.abs(mean - med) > MODE_TOLERANCE * sd, med, mode)


def background_grid(x: np.ndarray, m: int) -> np.ndarray:
    """The mesh levels of an image (h, w, 3), each channel's grid filtered by a 3 x 3 median: (3, gh, gw)."""
    out = []
    for k in range(x.shape[2]):
        g = mesh_levels(x[..., k], m).astype(np.float32)
        out.append(cv2.medianBlur(g, 3) if min(g.shape) >= 3 else g)
    return np.stack(out)


def map_of_grid(g: np.ndarray, m: int, h: int, w: int) -> np.ndarray:
    """The pixels (h, w, 3) of a grid of mesh levels: a bicubic interpolation, each level at its mesh's centre."""
    _c, gh, gw = g.shape
    # the grid at the mesh centres, then the pixels: resize with half-mesh alignment, cut to the image
    return np.stack([cv2.resize(c, (gw * m, gh * m), interpolation=cv2.INTER_CUBIC)[:h, :w] for c in g], -1)


def background_map(x: np.ndarray, m: int) -> np.ndarray:
    """The large-scale background of an image (h, w, 3): per channel, the mesh levels, a 3 x 3 median over the grid,
    then a bicubic interpolation back to the pixels (each level at its mesh's centre)."""
    h, w = x.shape[:2]
    return map_of_grid(background_grid(x, m), m, h, w).astype(np.float32)


def map_in_region(g: np.ndarray, m: int, hw: tuple[int, int], f: int, box, shape) -> np.ndarray:
    """The background map of the whole image measured on it ``f`` times smaller (grid ``g`` of meshes ``m``, that image
    ``hw``), at this rendering's scale over its region ``box`` (x0, y0, ...): (h, w, 3) of ``shape``. Post's region at
    100 % then takes off the veil of the whole image (review 10.1)."""
    big = map_of_grid(g, m, *hw)
    x0, y0 = box[:2]
    M = np.float32([[1 / f, 0, (x0 + 0.5) / f - 0.5], [0, 1 / f, (y0 + 0.5) / f - 0.5]])
    return cv2.warpAffine(big, M, (shape[1], shape[0]), flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_REPLICATE)


def take_off(x: np.ndarray, amount: float, veil_map: np.ndarray) -> np.ndarray:
    """He's haze model with the white as airlight: (x - h) / (1 - h), h = ``amount`` x the map."""
    h = np.clip(float(np.clip(amount, 0.0, AMOUNT_MAX)) * veil_map, 0.0, AMOUNT_MAX)
    return np.clip((x - h) / (1.0 - h), 0.0, 1.0).astype(np.float32)


#: the last background maps by what they were measured on: the veil does not depend on Amount, so a drag of that slider
#: (the same image, the same Scale) measures it once (the mesh levels took 4 s of the 4.4 s of each change)
_MAPS: "OrderedDict[tuple, np.ndarray]" = OrderedDict()
_MAPS_KEPT = 3
_MAPS_LOCK = Lock()


def cached_background_map(x: np.ndarray, m: int) -> np.ndarray:
    """``background_map``, kept for the images it was last asked for (the key is the image's own bytes: nothing else
    can make a map stale)."""
    x = np.ascontiguousarray(x)
    key = (x.shape, m, hashlib.blake2b(x.view(np.uint8), digest_size=16).digest())
    with _MAPS_LOCK:
        hit = _MAPS.get(key)
        if hit is not None:
            _MAPS.move_to_end(key)
            return hit
    out = background_map(x, m)
    with _MAPS_LOCK:
        _MAPS[key] = out
        while len(_MAPS) > _MAPS_KEPT:
            _MAPS.popitem(last=False)
    return out


def cached_background_grid(x: np.ndarray, m: int) -> np.ndarray:
    """``background_grid``, kept as ``cached_background_map`` keeps its maps."""
    x = np.ascontiguousarray(x)
    key = ("grid", x.shape, m, hashlib.blake2b(x.view(np.uint8), digest_size=16).digest())
    with _MAPS_LOCK:
        hit = _MAPS.get(key)
        if hit is not None:
            _MAPS.move_to_end(key)
            return hit
    out = background_grid(x, m)
    with _MAPS_LOCK:
        _MAPS[key] = out
        while len(_MAPS) > _MAPS_KEPT:
            _MAPS.popitem(last=False)
    return out


def dehaze_between(x: np.ndarray, amount: float, scale_px: float, f: float, coarse_hw: tuple[int, int]) -> np.ndarray:
    """``dehaze`` of an image reduced by a factor between two powers of two (the screen's picture at the size it is
    shown): the meshes are measured on it reduced to the next power of two, ``f`` times smaller (``coarse_hw``), where
    they are whole pixels of the full resolution's meshes, then the map brought to this image as a region's is
    (``map_in_region``); meshes of a fraction of a pixel would not fall on the full resolution's."""
    a = float(np.clip(amount, 0.0, AMOUNT_MAX))
    if a <= 0:
        return x
    h, w = coarse_hw
    small = cv2.resize(np.ascontiguousarray(x, np.float32), (w, h), interpolation=cv2.INTER_AREA)
    m = mesh_of(scale_px / f)
    return take_off(x, a, map_in_region(cached_background_grid(small, m), m, (h, w), f, (0, 0), x.shape))


def dehaze(x: np.ndarray, amount: float, scale_px: float) -> np.ndarray:
    """The image (h, w, 3, 0..1) with ``amount`` (0..AMOUNT_MAX) of its veil taken off; meshes of ``scale_px``
    pixels of this image."""
    a = float(np.clip(amount, 0.0, AMOUNT_MAX))
    if a <= 0:
        return x
    return take_off(x, a, cached_background_map(x, mesh_of(scale_px)))


def mesh_of(scale_px: float) -> int:
    return max(8, int(round(scale_px)))
