"""A stopped Registration keeps the star catalogues of the frames it had done (the run window says it goes on from
them): saved at the stop, whatever the count, and every few seconds."""
from __future__ import annotations

import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from astro_gpu.core.stage import Interrupted
from astro_gpu.stages import register as rg


class _Pool:
    def __init__(self, results):
        self.results = results

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def imap_unordered(self, func, todo, chunksize=1):
        for i in todo:
            yield self.results[i]


def _ctx(tmp_path, n, stop_after):
    cache = tmp_path / "lights.f16"
    np.memmap(cache, dtype=np.float16, mode="w+", shape=(n, 8, 8)).flush()
    info = {"path": str(cache), "n": n, "height": 8, "width": 8, "cfa": "MONO"}
    ck = tmp_path / "ck"
    ck.mkdir(exist_ok=True)
    seen = []

    def progress(frac, msg="", force=False):
        seen.append(msg)
        if len(seen) > stop_after:
            raise Interrupted("stop requested")
    meta = {"lights": [{"path": f"f{i}.fits", "time": None} for i in range(n)]}
    return SimpleNamespace(ckpt=ck, load_json=lambda *a: info if a == ("calibrate", "cache.json") else meta,
                           progress=progress, log=lambda *a, **k: None, metric=lambda *a: None,
                           preview=lambda *a, **k: None, out=lambda name: tmp_path / name), seen


def test_a_stop_keeps_the_catalogues_done_and_the_next_run_detects_only_the_others(tmp_path, monkeypatch):
    n = 12
    results = {i: (i, np.c_[np.full(5, float(i)), np.arange(5.0), np.ones(5), np.ones(5)], 2.0, 1.0, 0.1, 3.0,
                   (2.0, 0.9, 3.0, 5)) for i in range(n)}
    monkeypatch.setattr(rg, "get_context", lambda *a: SimpleNamespace(Pool=lambda *a, **k: _Pool(results)))
    monkeypatch.setattr(rg, "pool_size", lambda *a: 1)
    monkeypatch.setattr(rg, "memory_short", lambda *a: 1.0)
    ctx, _ = _ctx(tmp_path, n, stop_after=4)           # stopped after 4 frames: far from the 50 of a batch
    with pytest.raises(Interrupted):
        rg.Register().run(ctx, rg.RegisterParams(workers=1))
    kept = pickle.loads((ctx.ckpt / "catalogs.pkl").read_bytes())
    assert sum(c is not None for c in kept[0]) == 4
    monkeypatch.setattr(rg, "get_context", lambda *a: SimpleNamespace(Pool=lambda *a, **k: _Pool(
        {i: results[i] for i in range(n)})))
    ctx2, seen = _ctx(tmp_path, n, stop_after=1)
    ctx2.ckpt = ctx.ckpt
    with pytest.raises(Interrupted):
        rg.Register().run(ctx2, rg.RegisterParams(workers=1))
    assert seen[0].startswith("detecting the stars of 8 frames")
