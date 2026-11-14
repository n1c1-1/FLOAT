"""A monochrome sensor (the inventory's cfa "GGGG") through the Classic method and the method advice: no demosaicing,
the plane written in the three channels as the drizzle writes it, and the advice reads the monochrome rules."""
from __future__ import annotations

import json

import numpy as np
import pytest

from astro_gpu.core import coverage as cv
from astro_gpu.core.journal import Journal
from astro_gpu.core.stage import Context
from test_stacking import copy_project, run_stage, toy_night


@pytest.fixture(scope="module")
def mono_night(tmp_path_factory):
    root = tmp_path_factory.mktemp("mono")
    proj, truth = toy_night(root / "p", n=8, n_stars=140, noisy=2, hazy=4, trail=3, size=(192, 256))
    f = proj.stage_dir("calibrate") / "cache.json"
    f.write_text(json.dumps({**json.loads(f.read_text()), "cfa": "GGGG"}))
    return root, proj, truth


def _stack(night, name, **settings):
    from astro_gpu.stages.stack import Stack, StackParams
    root, proj, _ = night
    p = copy_project(proj.root, root / name)
    return p, run_stage(p, Stack(), StackParams(**{"scale": 1.0, "pixfrac": 1.0, "half_block": 2, **settings}))


def test_both_helpers_know_the_inventorys_monochrome_value():
    from astro_gpu.core import cfa
    for v in ("GGGG", "", None, "MONO"):
        assert cv.is_mono(v)
    assert cfa.is_mono("GGGG") and not cv.is_mono("RGGB") and not cv.is_mono("RGB")


def test_classic_runs_on_a_monochrome_sensor_and_writes_the_drizzles_layout(mono_night):
    from astro_gpu.stages import stack as st
    if st.classic_missing():
        pytest.skip("the Classic modules are not there")
    pc, _ = _stack(mono_night, "classic", method="classic")
    pd, _ = _stack(mono_night, "drizzle", method="drizzle")
    c, d = pc.stage_dir("stack"), pd.stage_dir("stack")
    for f in ("drz_full.npy", "drz_A.npy", "drz_B.npy", "drz_weight.npy"):
        a, b = np.load(c / f), np.load(d / f)
        assert a.shape == b.shape == (3, 192, 256) and np.isfinite(a).all()
        assert np.array_equal(a[0], a[1]) and np.array_equal(a[2], a[1])          # the plane in the three
    full = np.load(c / "drz_full.npy")[1]
    ref = np.load(d / "drz_full.npy")[1]
    assert np.corrcoef(full[16:-16, 16:-16].ravel(), ref[16:-16, 16:-16].ravel())[0, 1] > 0.95
    diag = json.loads((c / st.DIAG).read_text())
    assert diag["rejection"]["method"] == "classic"
    assert np.load(c / st.SATURATION).shape[1:] == (192, 256)


def test_the_advice_reads_the_monochrome_rules(mono_night):
    from astro_gpu.stages.stack import Stack
    root, proj, _ = mono_night
    p = copy_project(proj.root, root / "advice")
    m = Stack().auto(Context(p, Stack(), Journal(p.logs, echo=False), "cpu"))["method"]
    assert m.value == "classic" and m.template == cv.M_MONO_FEW and m.args == {"n": 8}      # 8 frames: not M_FEW's red and blue
