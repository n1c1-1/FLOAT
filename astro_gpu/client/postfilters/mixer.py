"""The Colour Mixer: hue, saturation and luminance of each colour range (issue #23).

Eight bands around the CIE L*c*h* hue circle (red, orange, yellow, green, aqua, blue, purple, magenta), each with a
hue shift, a saturation and a luminance setting, as the colour mixers of photo editors (their "HSL" panel) and the
hue-dependent saturation of the astronomy software; computed in CIE L*c*h* (CIE 1976, D65) rather than HSV, which
mixes lightness and colour (the astronomy software's documentation warns that HSV hue changes are colorimetrically
unsound). Each pixel takes from each band a share that falls smoothly with its distance to the band's hue (a raised
cosine reaching zero at the next band), and nothing where it has no colour (its chroma under NEUTRAL_C): the grey
background and the white star cores are left alone. The uses in astrophotography: the weak teal of OIII against the
dominant red in a dual-band image, the magenta of stars and halos in a narrowband palette, an H-alpha too pink.
"""
from __future__ import annotations

import cv2
import numpy as np

from .colour import from_lab, to_lab

#: the bands: their names and their hue on the CIE L*c*h* circle (degrees; the hues of the sRGB primaries and of
#: their mixtures: red 40, yellow 102, green 136, cyan 196, blue 306, magenta 328)
BANDS = (("red", 30.0), ("orange", 65.0), ("yellow", 100.0), ("green", 136.0), ("aqua", 196.0), ("blue", 262.0),
         ("purple", 306.0), ("magenta", 335.0))
#: under this chroma a pixel has no colour of its own: the mixer leaves it (a smooth start up to twice it)
NEUTRAL_C = 4.0
#: the settings' ends: a hue shift in degrees, saturation and luminance in percent
HUE_MAX, SAT_MAX, LUM_MAX = 30.0, 100.0, 100.0
#: the hue circle of the table the settings are read from
_PER_DEGREE = 4
_HUES = np.arange(360 * _PER_DEGREE + 2, dtype=np.float32) / _PER_DEGREE


def keys() -> list[str]:
    """The settings, band by band: hue_<band>, sat_<band>, lum_<band>."""
    return [f"{what}_{name}" for name, _h in BANDS for what in ("hue", "sat", "lum")]


def is_identity(p: dict) -> bool:
    return all(abs(float(p.get(k, 0.0))) < 1e-9 for k in keys())


def _weights(h: np.ndarray) -> np.ndarray:
    """(n bands, ...) the share of each band in each pixel of hue ``h`` (degrees): a raised cosine from the band's
    hue to its neighbours' (the shares add up to 1 around the circle)."""
    centers = np.array([c for _n, c in BANDS])
    n = len(centers)
    out = np.zeros((n,) + h.shape, np.float32)
    for i in range(n):
        c, lo, hi = centers[i], centers[i - 1], centers[(i + 1) % n]
        d = (h - c + 180.0) % 360.0 - 180.0                       # signed distance to the band's hue
        left = (c - lo) % 360.0
        right = (hi - c) % 360.0
        span = np.where(d < 0, left, right)
        t = np.clip(np.abs(d) / span, 0, 1)
        out[i] = 0.5 * (1 + np.cos(np.pi * t))
    return out


def _along(what: str, top: float, w: np.ndarray, p: dict) -> np.ndarray | None:
    """One setting along the hue circle (``_HUES``); None when it is zero everywhere."""
    values = [float(np.clip(float(p.get(f"{what}_{name}", 0.0)), -top, top)) for name, _c in BANDS]
    return None if not any(values) else np.tensordot(np.array(values, np.float32), w, 1)


#: the colourful pixels are worked on in pieces of this many, small enough for their intermediate values to stay in
#: the processor's cache (one pass over memory instead of one per operation: 3x faster), cut at a multiple of the
#: vector width so that each value is the one a single call on the whole gives
PIECE = 65536


def _tables(p: dict) -> dict:
    """For hue, saturation and luminance: the setting along the hue circle and its step to the next entry, or None."""
    w = _weights(_HUES)
    out = {}
    for what, top in (("hue", HUE_MAX), ("sat", SAT_MAX), ("lum", LUM_MAX)):
        curve = _along(what, top, w, p)
        out[what] = None if curve is None else (curve, curve[1:] - curve[:-1])
    return out


def _work(lab: np.ndarray, C: np.ndarray, hue: np.ndarray, res: np.ndarray, tables: dict) -> None:
    """The colourful pixels ``lab`` (n, 3), of chroma ``C`` and hue ``hue`` (degrees), through the settings'
    ``tables``, written into ``res`` (n, 3)."""
    a, b = lab[:, 1], lab[:, 2]
    t = hue * _PER_DEGREE
    i = t.astype(np.int32)
    t -= i
    share = np.minimum((C - NEUTRAL_C) / NEUTRAL_C, 1.0)                  # a smooth start

    def read(table):
        curve, step = table
        return share * (curve[i] + t * step[i])

    L = lab[:, 0]
    if tables["sat"] is not None:
        scale = np.clip(1.0 + read(tables["sat"]) / 100.0, 0.0, 3.0)
        a, b = a * scale, b * scale
    if tables["hue"] is not None:
        turn = np.radians(read(tables["hue"]))
        c, s = np.cos(turn), np.sin(turn)
        a, b = a * c - b * s, a * s + b * c
    if tables["lum"] is not None:
        # the luminance of a colour: up or down by its share of the way to white or black (a colourful pixel only)
        d = read(tables["lum"]) / 100.0
        L = np.where(d >= 0, L + (100.0 - L) * d * 0.5, L * (1.0 + d * 0.5))
    res[:, 0], res[:, 1], res[:, 2] = L, a, b


def _in_threads(n: int, fn) -> None:
    import threading
    threads = [threading.Thread(target=fn, args=(k,), name="post-mixer", daemon=True) for k in range(1, n)]
    for th in threads:
        th.start()
    fn(0)
    for th in threads:
        th.join()


def mix(x: np.ndarray, p: dict, threads: int = 1) -> np.ndarray:
    """``x`` (h, w, 3, sRGB 0..1) through the mixer's settings ``p`` (``keys``); the identity at zero. Only the pixels
    with a colour are worked on, and the bands' sum is read from a table along the hue circle (12 Mpx: 5 s without).
    The colourful pixels are taken out once, worked on in pieces (``PIECE``) in up to ``threads`` threads, and put back:
    the same values as one pass on the whole (the colour conversions, the chroma and the hue stay one call each:
    their values follow how they are cut, or the thread)."""
    if is_identity(p):
        return x
    out = np.clip(x, 0, 1).astype(np.float32, copy=False)
    lab = to_lab(out)
    on = cv2.magnitude(lab[..., 1], lab[..., 2]) > NEUTRAL_C               # nothing on what has no colour
    where = np.flatnonzero(on)
    n = len(where)
    if not n:
        return out
    picked = lab.reshape(-1, 3).take(where, axis=0)
    # (OpenCV's magnitude gives other values in other threads: chroma and hue are one call each, in this thread)
    C = cv2.magnitude(picked[:, 1], picked[:, 2]).ravel()
    hue = cv2.phase(picked[:, 1], picked[:, 2], angleInDegrees=True).ravel()
    res = np.empty((n, 3), np.float32)
    tables = _tables(p)
    pieces = range(0, n, PIECE)
    k = max(1, min(int(threads), len(pieces), 8))                          # (memory-bound past 8)

    def work(j):
        for s in pieces[j::k]:
            _work(picked[s:s + PIECE], C[s:s + PIECE], hue[s:s + PIECE], res[s:s + PIECE], tables)
    _in_threads(k, work)
    back = from_lab(res[:, None])[:, 0]
    flat = out.reshape(-1, 3)

    def put(j):
        for s in pieces[j::k]:
            flat[where[s:s + PIECE]] = np.clip(back[s:s + PIECE], 0, 1)
    _in_threads(k, put)
    return out
