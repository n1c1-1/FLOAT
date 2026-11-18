"""The Stars mask part's search of a result's stars (postengine.find_stars) is made tile by tile: the owner's M27
(16 Mpx, 23 700 stars) took 48 s at once, and the message "Finding the stars of the Stars mask…" stayed up for
minutes while Post rendered beside it; in tiles, 2.7 s. The stars found are those of the whole image."""
from __future__ import annotations

import numpy as np
import sep

from astro_gpu.client import postengine as pe
from astro_gpu.stages.combine import STAR_SIGMA


def _field(H=1800, W=2400, n=2500, seed=4):
    rng = np.random.default_rng(seed)
    img = rng.normal(0, 0.002, (H, W)).astype(np.float32) + 0.01
    yy, xx = np.mgrid[0:H, 0:W]
    for _ in range(n):
        x, y, s, b = rng.uniform(0, W), rng.uniform(0, H), rng.uniform(0.8, 3.0), rng.uniform(0.02, 0.8)
        r = int(5 * s) + 2
        ys, xs = slice(max(0, int(y) - r), min(H, int(y) + r + 1)), slice(max(0, int(x) - r), min(W, int(x) + r + 1))
        img[ys, xs] += b * np.exp(-((xx[ys, xs] - x) ** 2 + (yy[ys, xs] - y) ** 2) / (2 * s * s))
    return img


def _sub(img):
    bkg = sep.Background(img, bw=64, bh=64)
    sub = img - bkg.back()
    v = sub[::4, ::4]
    return sub, 1.4826 * float(np.median(np.abs(v - np.median(v))))


def test_the_stars_found_in_tiles_are_those_of_the_whole_image(monkeypatch):
    sub, rms = _sub(_field())
    monkeypatch.setattr(pe, "STAR_TILE", 1 << 20)                   # one tile: the whole image at once
    whole = pe._tiled_extract(sub, STAR_SIGMA, rms, minarea=4, deblend_cont=0.005)
    monkeypatch.setattr(pe, "STAR_TILE", 256)                        # (a tile boundary every 256 px)
    tiled = pe._tiled_extract(sub, STAR_SIGMA, rms, minarea=4, deblend_cont=0.005)
    assert abs(len(tiled) - len(whole)) <= 0.002 * len(whole) + 1
    a = np.stack([whole["x"], whole["y"]], 1)
    b = np.stack([tiled["x"], tiled["y"]], 1)
    d = np.hypot(*(a[:, None, :] - b[None, :, :]).transpose(2, 0, 1)).min(1) if len(a) < 4000 else None
    assert d is not None and (d < 0.01).mean() > 0.995                  # each star at its place, found once
    assert len(np.unique(np.round(b, 2), axis=0)) == len(b)


def test_find_stars_on_a_large_image_goes_through_the_tiles(monkeypatch):
    calls = []
    real = pe._tiled_extract

    def seen(sub, *a, **k):
        calls.append(sub.shape)
        return real(sub, *a, **k)
    monkeypatch.setattr(pe, "_tiled_extract", seen)
    stars = pe.find_stars(_field(1200, 1600, 600))
    assert calls and len(stars) > 400 and stars.shape[1] == 3
