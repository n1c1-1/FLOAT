"""A master shared by sessions is on the disk once: the session's copy of it is a hard link to the one kept for the
other sessions (the same bytes), a different master stays a file of its own."""
from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np

from astro_gpu.stages.calibrate import link_shared_masters


def test_a_master_equal_to_the_shared_one_is_linked_not_stored_again(tmp_path):
    store = tmp_path / "store"
    (store / "_masters").mkdir(parents=True)
    dark = np.random.default_rng(1).normal(100, 3, (40, 60)).astype(np.float32)
    np.save(store / "_masters" / "median_darks_abc.npy", dark)
    np.save(store / "_masters" / "flat_def.npy", np.ones((40, 60), np.float32))
    out = tmp_path / "store" / "calibrate" / "k.tmp"
    out.mkdir(parents=True)
    np.save(out / "master_dark.npy", dark)                                   # the same master
    np.save(out / "master_flat.npy", np.full((40, 60), 0.5, np.float32))    # another one: stays
    ctx = SimpleNamespace(project=SimpleNamespace(store=SimpleNamespace(path=store)), out=lambda n: out / n)
    link_shared_masters(ctx)
    assert os.stat(out / "master_dark.npy").st_nlink == 2
    assert os.stat(out / "master_flat.npy").st_nlink == 1
    assert np.array_equal(np.load(out / "master_dark.npy"), dark)
    assert not list(out.glob("*.lnk"))


def test_without_kept_masters_nothing_happens(tmp_path):
    out = tmp_path / "o"
    out.mkdir()
    np.save(out / "master_dark.npy", np.zeros((4, 4), np.float32))
    ctx = SimpleNamespace(project=SimpleNamespace(store=SimpleNamespace(path=tmp_path / "none")), out=lambda n: out / n)
    link_shared_masters(ctx)
    assert os.stat(out / "master_dark.npy").st_nlink == 1
