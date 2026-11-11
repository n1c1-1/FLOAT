"""The Reference block aligns a reference image the same way at every run: its triangle match (astroalign's RANSAC)
drew from an unseeded generator, and the same images gave 39, 38, 16, 39 then 37 stars matched in five runs."""
from __future__ import annotations

import json
from types import SimpleNamespace

import astroalign
import numpy as np
from PIL import Image

from astro_gpu.stages import reference as rf
from astro_gpu.stages.reference import Reference, ReferenceParams

H, W, RH, RW = 1400, 2000, 700, 1000


def _stars():
    """(the reference's stars, the stack's): 60 stars of the stack, the reference half its size, turned 3 deg and
    shifted, its stars 1 px off and 18 of them false (a processed image's halos and nebula)."""
    rng = np.random.default_rng(4)
    B = rng.uniform([0, 0], [W, H], (60, 2))
    th = np.radians(3.0)
    A = B @ (0.5 * np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])).T + [20, 15]
    A += rng.normal(0, 1.0, B.shape)
    A[:18] = rng.uniform([0, 0], [RW, RH], (18, 2))
    return A[rng.permutation(60)], B


def test_three_runs_give_the_same_files(tmp_path, monkeypatch):
    A, B = _stars()
    monkeypatch.setattr(rf, "_stars", lambda g, n: A if g.shape == (RH, RW) else B)
    find = astroalign.find_transform

    def first_only(*a, **k):       # the 7 other orientations, vain, answered at once (their exhausted matches: ~14 s)
        first_only.n += 1
        if first_only.n > 1:
            raise astroalign.MaxIterError("no match")
        return find(*a, **k)
    monkeypatch.setattr(astroalign, "find_transform", first_only)
    png = tmp_path / "reference.png"
    Image.new("RGB", (RW, RH)).save(png)
    np.random.seed(12345)
    before = np.random.get_state()
    runs = []
    for i in range(3):
        first_only.n = 0
        out, metrics = tmp_path / f"align_ref_{i}.json", {}
        ctx = SimpleNamespace(project=SimpleNamespace(config=SimpleNamespace(reference_image=str(png))),
                              load_json=lambda stage, name: {"scale": 1.0},
                              load=lambda *a, **k: np.zeros((3, H, W), np.float32), progress=lambda *a, **k: None,
                              log=lambda m, level="info": None, metric=lambda k, v: metrics.__setitem__(k, v),
                              save_json=lambda name, obj, out=out: out.write_text(json.dumps(obj)))
        Reference().run(ctx, ReferenceParams())
        runs.append((out.read_bytes(), metrics))
    after = np.random.get_state()
    assert runs[0][1]["stars_matched"] > 20
    assert all(r == runs[0] for r in runs[1:])
    assert before[0] == after[0] and np.array_equal(before[1], after[1]) and before[2:] == after[2:]
