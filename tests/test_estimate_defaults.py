"""(grid 12.1 Mpx) The estimate without any history (a fresh machine) against what a CPU laptop measured on the "Dual-band x2" template
(13 November 2026: 12 Mpx raws, a grid of 12.1 Mpx). The history is always passed in: the user's file is never read."""
import pytest

from astro_gpu.core import estimate as est

GRID = est.Facts(mpx=12.1, grid_mpx=12.1)


def seconds(stage, f=GRID, gpu=False):
    return est.estimate(stage, f, gpu=gpu, history={}).seconds


@pytest.mark.parametrize("stage,actual", [
    ("render_simple", 5.0), ("sessions", 1.2), ("combine_sho", 7.0), ("combine_lrgb", 7.0),
    ("align_on", 4.0), ("star_removal", 162.0)])
def test_the_blocks_default_costs_match_the_measured_ones(stage, actual):
    assert seconds(stage) == pytest.approx(actual, rel=0.15)


def test_gradient_is_under_a_second():
    assert seconds("gradient") < 1.0


def test_the_astrometry_row_is_seconds_not_minutes():
    row = sum(seconds(s) for s in ("astrometry", "catalogs", "surveys"))
    assert row == pytest.approx(3.4, rel=0.15)


def test_the_processor_denoise_is_a_third_of_what_it_was():
    assert seconds("denoise") == pytest.approx(2.0 * 12.1 * 4)
    assert est.estimate("denoise", GRID, gpu=True, history={}).seconds == pytest.approx(0.12 * 12.1 * 4)    # 5.6 s measured


def test_stack_and_calibration_per_unit():
    stack = est.Facts(lights=20, frames=20, mpx=12.1, scale=1.0, grid_mpx=12.1)      # the run's 20 dithered frames
    assert seconds("stack", stack) == pytest.approx(87.0, rel=0.35)
    cal = est.Facts(lights=20, frames=130, mpx=12.1, scale=1.0, grid_mpx=12.1)      # 0.17 s per frame of 12 Mpx
    assert seconds("calibrate", cal) == pytest.approx(0.17 * 130, rel=0.2)


def test_an_unknown_stage_keeps_the_minute_fallback():
    assert seconds("toy_stage", est.HEART_FACTS, gpu=True) == 60.0
