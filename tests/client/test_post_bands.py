"""Post's filters made in bands of rows in the processor's threads (client/postengine.run_filter): every filter that may
be is the whole image's result bit for bit, on images of odd sizes, at every level, with its settings at their ends; a
filter that is not in ``BAND_MARGINS`` is made on the whole; a composition abandoned stops between two bands."""
from __future__ import annotations

import cv2
import numpy as np
import pytest
import torch

from astro_gpu.client import postengine as pe


def image(h, w, seed=0):
    rng = np.random.default_rng(seed)
    a = cv2.GaussianBlur(rng.random((h, w, 3), dtype=np.float32), (0, 0), 2.5) * 0.7
    a += rng.random((h, w, 3), dtype=np.float32) * 0.15                     # noise, stars-like peaks
    a[rng.random((h, w)) > 0.999] = 0.98
    return np.clip(a, 0, 1).astype(np.float32)


def settings(key):
    """Each filter's settings to try: its defaults moved, its ends."""
    base = {q["key"]: q["default"] for q in pe.FILTERS[key]["params"]}
    tries = {"levels": [dict(black=0.05, white=0.9, gamma=1.4), dict(black=0.2, gamma=0.6)],
             "balance": [dict(temperature=1.5, tint=-1.0), dict(temperature=-3.0, tint=3.0)],
             "curves": [dict(curves=pe.curves_of_old(0.1, 0.2, -0.1))],
             "scnr": [dict(strength=0.6), dict(strength=1.0, method="Maximum mask", lightness=False),
                      dict(strength=0.4, method="Additive mask", lightness=True)],
             "denoise": [dict(luminance=0.6, color=0.4), dict(luminance=1.0, color=1.0)],
             "sharpen": [dict(strength=0.8, radius=1.2), dict(strength=2.0, radius=4.0)],
             "local_contrast": [dict(strength=0.4, scale=20), dict(strength=1.5, scale=40), dict(strength=0.4, scale=120)],
             "faint_smooth": [dict(amount=0.7, threshold=0.4, radius=24), dict(amount=1.0, threshold=0.8, radius=64)],
             "highlights": [dict(strength=0.5, radius=40), dict(strength=1.0, knee=0.2, detail=1.0, radius=120)],
             "ml_sharpen": [dict(l1=0.5, l3=0.4, l5=0.3), dict(l1=1.0, l2=-0.5, l4=1.0, target="RGB")]}
    return [dict(base, **t) for t in tries[key]]


@pytest.mark.parametrize("key", sorted(pe.BAND_MARGINS))
def test_a_filter_in_bands_is_the_whole_image_s_bit_for_bit(key, monkeypatch):
    monkeypatch.setattr(pe, "_THREADS", 6)
    for h, w, level in ((1031, 773, 0), (517, 1203, 1), (300, 451, 2), (130, 190, 3)):
        x = image(h, w, seed=h)
        for p in settings(key):
            ctx = pe.Ctx(pe.Composer(None), level, (0, 0, w, h))
            ctx.canvas_wh = (w, h)
            whole = pe.FILTERS[key]["fn"](x.copy(), p, ctx)
            got = pe.run_filter(key, x.copy(), p, ctx)
            assert got.shape == whole.shape and np.array_equal(got, np.asarray(whole, np.float32)), (key, p, level)


def test_a_filter_with_a_whole_image_measure_is_made_on_the_whole(monkeypatch):
    for key in ("clahe", "dehaze", "ml_denoise", "saturation", "mixer", "levels", "ml_sharpen"):
        assert key not in pe.BAND_MARGINS
    calls = []
    real = pe.FILTERS["clahe"]["fn"]
    monkeypatch.setitem(pe.FILTERS["clahe"], "fn", lambda x, p, ctx: (calls.append(x.shape), real(x, p, ctx))[1])
    x = image(1024, 700)
    p = {q["key"]: q["default"] for q in pe.FILTERS["clahe"]["params"]}
    p["amount"] = 0.5
    pe.run_filter("clahe", x, p, pe.Ctx(pe.Composer(None), 0, (0, 0, 700, 1024)))
    assert calls == [x.shape]


def test_a_composition_abandoned_stops_between_two_bands_and_the_threads_come_back(monkeypatch):
    monkeypatch.setattr(pe, "_THREADS", 4)
    made = []
    real = pe.FILTERS["highlights"]["fn"]
    monkeypatch.setitem(pe.FILTERS["highlights"], "fn", lambda x, p, ctx: (made.append(1), real(x, p, ctx))[1])
    x = image(2048, 300)
    p = {q["key"]: q["default"] for q in pe.FILTERS["highlights"]["params"]}
    p["strength"] = 0.6
    ctx = pe.Ctx(pe.Composer(None), 0, (0, 0, 300, 2048))
    ctx.check = lambda: len(made) >= 2
    threads = torch.get_num_threads(), cv2.getNumThreads()
    with pytest.raises(pe.Superseded):
        pe.run_filter("highlights", x, p, ctx)
    assert len(made) < 2048 // pe.BAND_ROWS
    assert (torch.get_num_threads(), cv2.getNumThreads()) == threads
