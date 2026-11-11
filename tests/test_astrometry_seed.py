"""The same stack solved twice gives the same solution: the triangle match's RANSAC (astroalign) drew from an unseeded
generator, and a 16 x 11 deg field used 7770 then 7771 stars, its WCS a little different. Its draws are now seeded per
match, without touching numpy's global random state."""
from __future__ import annotations

from types import SimpleNamespace

import astroalign
import numpy as np

from astro_gpu.core.determinism import MATCH_SEED, seeded_match
from astro_gpu.stages import astrometry as am
from astro_gpu.stages.astrometry import AstrometryParams
from tests.test_astrometry_search import NDEC0, NH, NRA0, NW, _ngc7000

TIMINGS = {"first_solution_s"}      # seconds measured, not a result


def _field():
    """The NGC 7000 field with its stars 1.2 px off where the catalogue puts them (blurred, a wide field's corners): the
    stars near the fine fit's tolerances are many, and a RANSAC that starts from another triangle, unseeded, ended on
    128, 130, 131, 128, 128 then 125 stars used in six runs."""
    gs, gm, det = _ngc7000(seed=1, spur=0.3, nb=400)
    return gs, gm, det + np.random.default_rng(9).normal(0, 1.2, det.shape)


def _solve(tmp_path, monkeypatch, gs, gm, det):
    """The stage on the field, its scale known (the triangle match of 100 control points, a RANSAC): (the bytes of
    wcs.fits, the metrics but the timings)."""
    o = np.zeros(len(det), dtype=[("x", float), ("y", float), ("flux", float), ("flag", int)])
    o["x"], o["y"], o["flux"] = det[:, 0], det[:, 1], np.arange(len(det), 0, -1)
    monkeypatch.setattr(am, "detect", lambda g: o)
    monkeypatch.setattr(am, "gaia_bright", lambda *a, **k: (gs, gm))
    monkeypatch.setattr(am, "remember_instrument", lambda *a: None)
    monkeypatch.setattr(am, "mirror_hint", lambda *a: (True, False))
    monkeypatch.setattr(am, "focal_hints", lambda *a: [])
    monkeypatch.setattr(am, "_asterion_solution", lambda *a: None)     # FLOAT's own match, Asterion installed or not
    tmp_path.mkdir(exist_ok=True)
    metrics = {}
    ctx = SimpleNamespace(load_json=lambda stage, name: {"scale": 1.0},
                          load=lambda *a, **k: np.zeros((3, NH, NW), np.float32), log=lambda m, level="info": None,
                          progress=lambda *a, **k: None, metric=lambda k, v: metrics.__setitem__(k, v),
                          out=lambda n: tmp_path / n, metric_of=lambda *a, **k: None)
    am.Astrometry().run(ctx, AstrometryParams(ra=NRA0, dec=NDEC0, scale=2.32, blind=False))
    return (tmp_path / "wcs.fits").read_bytes(), {k: v for k, v in metrics.items() if k not in TIMINGS}


def test_the_same_stack_solved_twice_gives_the_same_solution(tmp_path, monkeypatch):
    gs, gm, det = _field()
    np.random.seed(12345)
    before = np.random.get_state()
    runs = [_solve(tmp_path / str(i), monkeypatch, gs, gm, det) for i in range(3)]   # unseeded, 3 alike: ~1 in 10
    after = np.random.get_state()
    assert runs[0][1]["stars_used"] > 100
    assert all(r[0] == runs[0][0] and r[1] == runs[0][1] for r in runs[1:])
    # numpy's global random state neither read nor moved: other code sees no difference
    assert before[0] == after[0] and np.array_equal(before[1], after[1]) and before[2:] == after[2:]


def test_the_match_draws_from_its_own_seeded_generator_and_nothing_else_does():
    """Inside ``seeded_match`` astroalign's unseeded draws are the block's, the same at every block; outside, and for an
    explicit seed, numpy as before; the stop-aware model is kept."""
    aa = am._hooked(astroalign)
    with seeded_match(aa):
        first = aa._np.random.default_rng().permutation(50), aa._np.random.permutation(50)
    with seeded_match(aa):
        again = aa._np.random.default_rng().permutation(50), aa._np.random.permutation(50)
    assert all(np.array_equal(x, y) for x, y in zip(first, again))
    assert np.array_equal(first[0], np.random.default_rng(MATCH_SEED).permutation(50))
    with seeded_match(aa):
        assert np.array_equal(aa._np.random.default_rng(3).permutation(50), np.random.default_rng(3).permutation(50))
    assert aa._np.random.default_rng is np.random.default_rng and aa._np.random.permutation is np.random.permutation
    assert aa._np.float64 is np.float64 and aa._MatchTransform.stop_aware
