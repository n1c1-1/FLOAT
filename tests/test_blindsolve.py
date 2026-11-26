"""All-sky blind solver: a synthetic field drawn from a synthetic catalogue must be found again. The index is built here
from random stars (``blindsolve.build_index``, how the two index files of the star data were made); the program itself
only reads the files."""
from __future__ import annotations

import numpy as np

from astro_gpu.core.blindsolve import SkyIndex, _tangent, _unit, build_index, solve


def test_blind_solve_synthetic_field():
    rng = np.random.default_rng(7)
    n = 30000                                               # about the density of Hipparcos to V 9.5
    ra = rng.uniform(0, 360, n)
    dec = np.degrees(np.arcsin(rng.uniform(-1, 1, n)))
    codes, quads = build_index(ra, dec, 8)
    idx = SkyIndex("hip", arrays={"ra": ra, "dec": dec, "codes": codes, "quads": quads})
    ra0, dec0, scale = 84.0, -2.0, 25.0 / 3600            # 25"/px
    W, H = 1200, 800
    c = _unit(np.array([ra0]), np.array([dec0]))[0]
    near = idx.star_tree.query_ball_point(c, np.radians(6))
    xy = _tangent(idx.xyz[near], c) / np.radians(scale) + [W / 2, H / 2]
    rot = np.radians(23.0)
    R = np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])
    xy = (xy - [W / 2, H / 2]) @ R.T + [W / 2, H / 2]
    inside = (xy[:, 0] > 0) & (xy[:, 0] < W) & (xy[:, 1] > 0) & (xy[:, 1] < H)
    xy = xy[inside] + np.random.default_rng(0).normal(0, 0.3, (inside.sum(), 2))
    r = solve(xy, W, H, idx)
    assert r is not None
    assert abs(r["ra"] - ra0) < 0.2 and abs(r["dec"] - dec0) < 0.2
    assert abs(r["scale"] - 25.0) < 0.5
