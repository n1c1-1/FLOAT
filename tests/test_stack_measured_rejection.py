"""The drizzle's own rejection judges each sample against the mean and standard deviation MEASURED at its output point
(sigma clipping, as Siril's), saturated samples included. A flat toy sky of 30 frames drifting by one pixel a frame
(CPU): a lone saturated pixel fixed on the sensor, a star whose core saturates in some frames only, a trail in one."""
from __future__ import annotations

import numpy as np
import pytest
from test_stacking import copy_project, run_stage, toy_night

N = 30
SAT_MARK = 65000.0                      # what the calibrated cache holds for a saturated pixel
HOT = (160, 224)                        # a red pixel (row even, column even)
STAR = (150, 300)                       # where the star is on the sky (output grid)


def night(root, hot: bool, star: bool):
    """The toy night without stars, the frames drifting by one pixel a frame along x; with ``hot``, a saturated pixel
    at HOT in every frame; with ``star``, a Gaussian star at STAR whose peak is clipped in every third frame."""
    proj, truth = toy_night(root, n=N, n_stars=0, trail=3)
    reg = dict(np.load(proj.stage_dir("register") / "reg.npz"))
    ref = int(reg["ref"])
    shift = (np.arange(N) - ref).astype(float)                     # the frame content at x - shift
    W = 448
    reg["coef"][:, 0, 0] = W / 2 - shift
    vs = (np.arange(N) % 2).astype(float)                          # one row up in every other frame: every colour lands
    reg["coef"][:, 1, 0] = 320 / 2 - vs
    np.savez(proj.stage_dir("register") / "reg.npz", **reg)
    mm = np.memmap(root / "lights.f16", dtype=np.float16, mode="r+", shape=(N, 320, W))
    yy, xx = np.mgrid[0:320, 0:W]
    for i in range(N):
        f = mm[i].astype(np.float32)
        if star:
            peak = 90000.0 if i % 3 == 0 else 45000.0
            f += peak * np.exp(-0.5 * ((xx - (STAR[1] + shift[i])) ** 2 + (yy - (STAR[0] + vs[i])) ** 2) / 1.5 ** 2)
            f = np.where(f > 60000.0, SAT_MARK, f)
        if hot:
            f[HOT] = SAT_MARK
        mm[i] = f
    mm.flush()
    del mm
    return proj, shift, vs


def stacked(tmp_path, name, **kw):
    from astro_gpu.stages.stack import Stack, StackParams
    root = tmp_path / name
    proj, shift, vs = night(root / "p", **kw)
    ctx = run_stage(proj, Stack(), StackParams(scale=1.0, pixfrac=1.0, half_block=2))
    return np.load(proj.stage_dir("stack") / "drz_full.npy"), (shift, vs), ctx


def test_a_saturated_lone_pixel_fixed_on_the_sensor_is_not_in_the_stack(tmp_path):
    ref, _, _ = stacked(tmp_path, "control", hot=False, star=False)
    out, (shift, vs), _ = stacked(tmp_path, "hot", hot=True, star=False)
    sites = [(HOT[0] - int(v), HOT[1] - int(s)) for s, v in zip(shift, vs) if 0 <= HOT[1] - s < 448]
    assert len(sites) >= 25
    diff = np.array([out[0, y, x] - ref[0, y, x] for y, x in sites])
    assert np.abs(diff).max() < 60, diff.max()                    # unrejected: thousands of ADU at each site


def test_a_star_with_a_core_saturated_in_some_frames_leaves_no_hole(tmp_path):
    out, _, ctx = stacked(tmp_path, "star", hot=False, star=True)
    w = np.load(ctx.project.stage_dir("stack") / "drz_weight.npy")
    y, x = STAR
    yy, xx = np.mgrid[0:320, 0:448]
    r = np.hypot(yy - y, xx - x)
    core, ring = r < 1.6, (r > 4) & (r < 7)
    for c in range(3):
        # the drift of whole pixels leaves each colour on a chessboard: the points it covers, and no other, count
        assert (w[c][core] > 0).mean() >= 0.8 * (w[c][ring] > 0).mean(), c             # no point left without a sample
        assert out[c][core & (w[c] > 0)].min() >= out[c][ring & (w[c] > 0)].mean(), c    # and no dip at the core


def test_a_bright_trail_in_one_frame_is_rejected(tmp_path):
    out, _, _ = stacked(tmp_path, "trail", hot=False, star=False)
    row = 320 // 2 + 9 - 1                                        # frame 3's two rows (+4000 ADU), one up (odd frame)
    for c in range(3):
        trail = out[c, row:row + 2, 40:400].mean()
        beside = out[c, row - 6:row - 3, 40:400].mean()
        assert abs(trail - beside) < 15, (c, trail, beside)       # unrejected: 4000 / 30 = 130 more
