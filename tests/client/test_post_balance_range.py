"""Post's Color balance reaches a sky with a strong cast (one taken without flats): its two sliders go to +-3."""
import numpy as np

from astro_gpu.client import postengine


def _param(key):
    return next(p for p in postengine.FILTERS["balance"]["params"] if p["key"] == key)


def test_the_balance_sliders_reach_a_strong_cast():
    # a cyan-green sky as measured on an image without flats: 22 / 32 / 31 of 255
    sky = np.array([[[22, 32, 31]]], np.float32) / 255
    for key in ("temperature", "tint"):
        assert _param(key)["lo"] <= -3 and _param(key)["hi"] >= 3
    best = None
    for t in np.arange(-3, 3.01, 0.05):
        for g in np.arange(-3, 3.01, 0.05):
            out = postengine.f_balance(sky.copy(), {"temperature": float(t), "tint": float(g)}, None)[0, 0]
            spread = float(out.max() - out.min())
            best = spread if best is None else min(best, spread)
    assert best * 255 < 0.5                               # a neutral sky is within the sliders' reach


def test_a_value_within_the_former_range_gives_the_same_gains():
    x = np.ones((1, 1, 3), np.float32)
    out = postengine.f_balance(x, {"temperature": 1.0, "tint": -1.0}, None)[0, 0]
    assert np.allclose(out, [1.15, 0.9, 0.85], atol=1e-6)
