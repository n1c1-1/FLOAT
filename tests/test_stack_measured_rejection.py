"""The drizzle's own rejection judges each sample against the mean and standard deviation MEASURED at its output point
(sigma clipping, as Siril's), saturated samples included; with few values per point that cannot reject (one outlier
among N values reaches (N - 1) / sqrt(N) sigmas at most), so the rejection by frame count keeps the values of every
frame per point and runs the Classic method's rejection on them, and the engine suggests which from the number of
values. A flat toy sky of frames drifting by one pixel a frame (CPU): a lone saturated pixel fixed on the sensor, a
star whose core saturates in some frames only, a trail in one frame, a lone outlier."""
from __future__ import annotations

import json

import numpy as np
import pytest
from test_stacking import run_stage, toy_night

N = 30
SAT_MARK = 65000.0                      # what the calibrated cache holds for a saturated pixel
HOT = (160, 224)                        # a red pixel (row even, column even)
STAR = (150, 300)                       # where the star is on the sky (output grid)
OUT = (100, 150)                        # a green pixel holding a lone outlier in one frame


def night(root, hot: bool, star: bool, n: int = N, frac: bool = False, outlier: bool = False):
    """The toy night without stars, the frames drifting by one pixel a frame along x; with ``hot``, a saturated pixel
    at HOT in every frame; with ``star``, a Gaussian star at STAR whose peak is clipped in every third frame; with
    ``outlier``, +5000 ADU on the pixel OUT of frame 2. ``frac``: the shifts are not whole pixels, so that every
    colour of every frame reaches every point (whole shifts leave each colour on a chessboard)."""
    proj, truth = toy_night(root, n=n, n_stars=0, trail=3 if n > 3 else -1, noisy=-1, hazy=-1)
    reg = dict(np.load(proj.stage_dir("register") / "reg.npz"))
    ref = int(reg["ref"])
    shift = (np.arange(n) - ref).astype(float) + (0.37 if frac else 0.0)       # the frame content at x - shift
    W = 448
    reg["coef"][:, 0, 0] = W / 2 - shift
    vs = (np.arange(n) % 2).astype(float) + (0.41 if frac else 0.0)           # every other frame a row up
    reg["coef"][:, 1, 0] = 320 / 2 - vs
    np.savez(proj.stage_dir("register") / "reg.npz", **reg)
    mm = np.memmap(root / "lights.f16", dtype=np.float16, mode="r+", shape=(n, 320, W))
    yy, xx = np.mgrid[0:320, 0:W]
    for i in range(n):
        f = mm[i].astype(np.float32)
        if star:
            peak = 90000.0 if i % 3 == 0 else 45000.0
            f += peak * np.exp(-0.5 * ((xx - (STAR[1] + shift[i])) ** 2 + (yy - (STAR[0] + vs[i])) ** 2) / 1.5 ** 2)
            f = np.where(f > 60000.0, SAT_MARK, f)
        if hot:
            f[HOT] = SAT_MARK
        if outlier and i == 2:
            f[OUT] += 5000.0
        mm[i] = f
    mm.flush()
    del mm
    return proj, shift, vs


def stacked(tmp_path, name, rejection="own", **kw):
    from astro_gpu.stages.stack import Stack, StackParams
    root = tmp_path / name
    proj, shift, vs = night(root / "p", **kw)
    ctx = run_stage(proj, Stack(), StackParams(scale=1.0, pixfrac=1.0, half_block=2, drizzle_rejection=rejection))
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


# ------------------------------------------------------------------------------------------ rejection by frame count

@pytest.mark.parametrize("n", [5, 8])
def test_by_count_rejects_a_lone_outlier_with_few_frames(tmp_path, n):
    ref, _, _ = stacked(tmp_path, "c", "by_count", hot=False, star=False, n=n, frac=True)
    out, _, _ = stacked(tmp_path, "o", "by_count", hot=False, star=False, n=n, frac=True, outlier=True)
    y, x = OUT
    near = (slice(None), slice(y - 3, y + 4), slice(x - 3, x + 4))
    assert np.abs(out[near] - ref[near]).max() < 80                # unrejected: about +1000 ADU at its points
    own, _, _ = stacked(tmp_path, "w", "own", hot=False, star=False, n=n, frac=True, outlier=True)
    assert np.abs(own[near] - ref[near]).max() > 200               # the sigma clipping cannot, with so few values


def test_by_count_removes_a_saturated_hot_pixel_walking_across_seven_frames(tmp_path):
    ref, _, _ = stacked(tmp_path, "c", "by_count", hot=False, star=False, n=7, frac=True)
    out, (shift, vs), _ = stacked(tmp_path, "h", "by_count", hot=True, star=False, n=7, frac=True)
    for s, v in zip(shift, vs):
        y, x = int(round(HOT[0] - v)), int(round(HOT[1] - s))
        box = (slice(None), slice(y - 2, y + 3), slice(x - 2, x + 3))
        assert np.abs(out[box] - ref[box]).max() < 80, (s, v)


def test_by_count_leaves_no_hole_in_a_star_saturated_in_some_frames(tmp_path):
    out, _, ctx = stacked(tmp_path, "s", "by_count", hot=False, star=True, n=9, frac=True)
    w = np.load(ctx.project.stage_dir("stack") / "drz_weight.npy")
    y, x = STAR
    yy, xx = np.mgrid[0:320, 0:448]
    r = np.hypot(yy - y, xx - x)
    core, ring = r < 1.6, (r > 4) & (r < 7)
    for c in range(3):
        assert (w[c][core] > 0).all(), c                           # every point of the core keeps samples
        assert out[c][core].min() >= out[c][ring].mean(), c        # and no dip at the core


def test_by_count_keeps_the_half_stacks_the_rejection_map_and_the_diagnostics(tmp_path):
    from astro_gpu.stages.stack import DIAG, REJECTION
    _, _, ctx = stacked(tmp_path, "d", "by_count", hot=False, star=False, n=8, frac=True, outlier=True)
    d = ctx.project.stage_dir("stack")
    for f in ("drz_A.npy", "drz_B.npy", "drz_full.npy", "drz_weight.npy", "saturation.npy", REJECTION):
        assert (d / f).is_file(), f
    diag = json.loads((d / DIAG).read_text())
    assert diag["rejection"]["method"] == "by_count" and diag["rejection"]["by_count"]["choices"]["G"]
    with np.load(d / REJECTION) as z:
        assert int(z["high"].sum()) == sum(f["rejected_high"] for f in diag["frames"]) > 0
        assert int(z["low"].sum()) == sum(f["rejected_low"] for f in diag["frames"])


def test_bands_of_rows_give_the_same_result_as_one_band(tmp_path, monkeypatch):
    """The band size is the memory's: a tiny budget cuts the output in many bands, the stack is the same."""
    from astro_gpu.core import devices
    from astro_gpu.stages.stack import DIAG
    one, _, _ = stacked(tmp_path, "one", "by_count", hot=True, star=False, n=7, frac=True)
    monkeypatch.setattr(devices, "memory_budget", lambda: (2 * (60 * 7 * 3 * 448 * 64) + 2 * 14 * 4 * 320 * 448, 1 << 40))
    many, _, ctx = stacked(tmp_path, "many", "by_count", hot=True, star=False, n=7, frac=True)
    diag = json.loads((ctx.project.stage_dir("stack") / DIAG).read_text())
    assert diag["rejection"]["by_count"]["band_rows"] < 320
    assert np.allclose(one, many, rtol=1e-5, atol=1e-3)


# ------------------------------------------------------------------------------------------------ the automatic choice

def _auto(tmp_path, name, n):
    from astro_gpu.core.journal import Journal
    from astro_gpu.core.stage import Context
    from astro_gpu.stages.stack import Stack
    proj, _, _ = night(tmp_path / name / "p", hot=False, star=False, n=n, frac=True)
    ctx = Context(proj, Stack(), Journal(proj.logs, echo=False), "cpu")
    return Stack().auto(ctx)["drizzle_rejection"]


def test_the_automatic_rejection_is_by_count_for_seven_frames_and_own_for_sixty(tmp_path):
    few, many = _auto(tmp_path, "few", 7), _auto(tmp_path, "many", 60)
    assert few.value == "by_count" and many.value == "own"
    assert "under the 11" in few.text("en") and "at least the 11" in many.text("en")
    assert few.args["values"] < 11 <= many.args["values"]
    assert few.args["values"] == pytest.approx(7 * few.args["share"] / 100)
    assert few.text("fr") != few.text("en") and many.text("fr") != many.text("en")


def test_the_limit_is_the_clipping_s_own():
    """The fewest values for one outlier to reach k sigmas: (N - 1) / sqrt(N) >= k, 11 for 3."""
    from astro_gpu.stages.stack import clipping_min_values
    assert clipping_min_values(3.0) == 11
    for k in (2.0, 3.0, 4.0, 5.0):
        n = clipping_min_values(k)
        assert (n - 1) / np.sqrt(n) >= k > (n - 2) / np.sqrt(n - 1)
