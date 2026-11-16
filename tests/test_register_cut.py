"""Registration cut (the frames taken as already aligned): a frame whose stars could not be matched to the reference's
(no star found in it: clouds, a lost frame) carries no measure, as a frame the alignment could not fit, so that the
selection never prefers it. It had a residual of 0 and a transparency of 1, the best marks there are."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from astro_gpu.stages import register as rg
from astro_gpu.stages.stack import quality_weights

W, H = 600, 400


def _cut(tmp_path, cats):
    n = len(cats)
    said = {}
    ctx = SimpleNamespace(log=lambda *a, **k: None, out=lambda name: tmp_path / name,
                          metric=lambda k, v: said.__setitem__(k, v), preview=lambda *a, **k: None)
    meta = [{"path": f"f{i}.fit", "time": ""} for i in range(n)]
    nst = np.array([len(c) for c in cats])
    rg.Register()._not_aligned(ctx, None, meta, cats, np.full(n, 3.0), np.ones(n), np.ones(n), np.ones(n), [None] * n,
                               nst, W, H)
    return dict(np.load(tmp_path / "reg.npz")), said


def _stars(seed=0, n=60):
    rng = np.random.default_rng(seed)
    return np.c_[rng.uniform([0, 0], [W, H], (n, 2)), rng.uniform(500, 5000, n)]


def test_a_frame_without_stars_carries_no_measure(tmp_path):
    ref = _stars()
    dim = ref.copy()
    dim[:, 2] *= 0.5                                         # half the light: measured, transparency 0.5
    empty = np.zeros((0, 3))
    reg, said = _cut(tmp_path, [ref, dim, empty, ref.copy()])
    assert np.isfinite(reg["rms"]).tolist() == [True, True, False, True]
    assert reg["transp"][2] == 0 and reg["nmatch"][2] == 0
    assert abs(reg["transp"][1] - 0.5) < 1e-6 and reg["transp"][0] == 1 and reg["rms"][0] == 0
    assert said["aligned"] == 3 and said["not_aligned"] == 1 and said["rms_px"] == 0.0
    # what the selection reads: no weight for it, the measured frames as before
    weight, _ = quality_weights(reg["transp"].astype(float), reg["fwhm"], reg["rms"], np.ones((4, 3, 4)))
    assert weight[2] == 0 and (weight[[0, 1, 3]] > 0).all() and weight[1] < weight[0]


def test_frames_all_measured_and_frames_all_without_stars_are_as_before(tmp_path):
    ref = _stars()
    reg, said = _cut(tmp_path, [ref, ref.copy(), ref.copy()])
    assert (reg["rms"] == 0).all() and (reg["transp"] == 1).all() and said["aligned"] == 3 and "not_aligned" not in said
    # no star in any frame: nothing can be measured, every frame stays in (a set without stars is stacked whole)
    empty = np.zeros((0, 3))
    reg, said = _cut(tmp_path, [empty, empty, empty])
    assert (reg["rms"] == 0).all() and (reg["transp"] == 1).all() and said["aligned"] == 3


def test_the_log_line_of_the_frames_left_out_has_its_french():
    from astro_gpu.client import i18n
    from astro_gpu.client.services.run import engine_text
    i18n.set_language("fr")
    try:
        assert engine_text("2 frame(s) have no star in common with the reference and will be ignored") == \
            "2 image(s) sans étoile commune avec la référence, écartée(s)"
    finally:
        i18n.set_language("en")
