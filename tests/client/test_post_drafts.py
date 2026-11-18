"""A draft of a layer's setting under the owner's stack (NGC 7000, twelve filters): 0.36 s each in the window, where
the gate's took 0.13. Dehaze measured its meshes twice at a draft's level (between two powers of two: a measure kept
there serves nothing), and its meshes' medians went through numpy's masked arrays; Saturation made a draft's rows in
one thread."""
from __future__ import annotations

import warnings

import numpy as np
from test_post_screen_size import Sky, stack

from astro_gpu.client import postengine as pe
from astro_gpu.client.postengine import Composer
from astro_gpu.client.postfilters import dehaze as pf_dehaze


def test_dehaze_measures_its_meshes_once_at_a_level_between_two(monkeypatch):
    calls = []
    real = pf_dehaze.background_grid
    monkeypatch.setattr(pf_dehaze, "background_grid", lambda x, m: (calls.append(m), real(x, m))[1])
    pf_dehaze._MAPS.clear()
    pr = Sky(1200, 800)
    Composer(pr).compose(pe.snapshot(stack([("dehaze", {"amount": 0.5, "scale": 128.0})])), 3.5, draft=True)
    assert len(calls) == 1


def test_the_meshes_medians_are_numpy_s():
    rng = np.random.default_rng(2)
    for m in (8, 11, 24):
        c = rng.random((377, 563)).astype(np.float32) * 0.3 + 0.05
        c[rng.random(c.shape) < 0.01] = 0.9                               # stars, clipped out
        a = rng.normal(size=(9, 13, m * m))
        a[rng.random(a.shape) < 0.2] = np.nan
        a[0, 0] = np.nan                                                  # a mesh with nothing left
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            assert np.array_equal(pf_dehaze._nanmedian(a), np.nanmedian(a, axis=2, keepdims=True), equal_nan=True)
        assert pf_dehaze.mesh_levels(c, m).shape == (-(-377 // m), -(-563 // m))
