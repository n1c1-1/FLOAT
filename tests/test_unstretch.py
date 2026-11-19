"""Issue #174: an image already stretched: how it is detected, and its stretch undone (from the header exactly, or
estimated). Each stretch kind makes a round trip: a linear image stretched with the forward formula as the reference
program writes it, the HISTORY line written as it writes it, the file loaded, the linear image back within float
precision (but for the pixels the stretch clipped)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from astro_gpu.stages import unstretch as U
from astro_gpu.stages.image import read_image

astropy_fits = pytest.importorskip("astropy.io.fits")

rng = np.random.default_rng(174)


def linear_image(mono=False, sigma=0.01, sky=0.05, size=(64, 80)):
    """A synthetic linear stack: a sky with noise and a few bright blobs (values 0-1)."""
    h, w = size
    a = sky + sigma * rng.standard_normal((1 if mono else 3, h, w))
    yy, xx = np.mgrid[:h, :w]
    for _ in range(6):
        y0, x0 = rng.integers(8, h - 8), rng.integers(8, w - 8)
        a += rng.uniform(0.1, 0.5) * np.exp(-((yy - y0) ** 2 + (xx - x0) ** 2) / 8.0)
    return np.clip(a, 0.0005, 0.95).astype(np.float32)


def write_fits(path, data, history=()):
    hdu = astropy_fits.PrimaryHDU(np.asarray(data, np.float32))
    for line in history:
        hdu.header["HISTORY"] = line
    hdu.writeto(path, overwrite=True)
    return path


def loaded(path):
    """The file's data as the Image block reads it, in 0-1, and its analysis."""
    a = read_image(path)
    return a / U.scale_of(a), U.analyse(path)


def back(path, check=True):
    """The file undone from its header. ``check`` False: the header's parameters are taken as exact (a line that writes
    its numbers with one digit would not pass the rounding check on a synthetic image with so little noise)."""
    a, an = loaded(path)
    if not check:
        lines, ch = U.read_history(path)
        pl = U.plan_of(lines, ch)
        assert pl.refusal is None, pl.refusal
        return U.undo(a, pl.ops, ch == 1)
    assert an.exact, an.refusal
    return U.undo(a, an.ops, an.mono)


def close_but_clipped(lin, got, stretched, tol=2e-5):
    lin = np.broadcast_to(lin, got.shape)                    # a monochrome image is read on three channels
    stretched = np.broadcast_to(stretched, got.shape)
    ok = ~np.any((stretched <= 0.0) | (stretched >= 1.0), axis=0)
    assert ok.mean() > 0.9
    np.testing.assert_allclose(got[:, ok], lin[:, ok], atol=tol)


# ---------------------------------------------------------------------------------------------- round trips

@pytest.mark.parametrize("colour", [True, False])
def test_roundtrip_midtones_transfer_1_2(tmp_path, colour):
    """``Midtones transfer (lo, mid, hi)``, the 1.2 command line; mid-tones transfer of the reference program."""
    lin = linear_image(mono=not colour)
    lo, mid, hi = 0.002, 0.25, 1.0
    y = U.mtf(lin, mid, lo, hi).astype(np.float32)
    f = write_fits(tmp_path / "a.fits", y if colour else y[0], [f"Midtones transfer ({lo:.3f}, {mid:.4f}, {hi:.3f})"])
    got = back(f)
    close_but_clipped(lin, got, y.astype(np.float64))


def test_roundtrip_histogram_dialog_and_new_forms(tmp_path):
    lin = linear_image()
    for line, (m, lo, hi) in (("Histogram Transf. (mid=0.250, lo=0.002, hi=1.000)", (0.25, 0.002, 1.0)),
                              ("Linked MTF stretch (mid=0.250000, lo=0.002000, hi=1.000000)", (0.25, 0.002, 1.0))):
        y = U.mtf(lin, m, lo, hi).astype(np.float32)
        f = write_fits(tmp_path / "b.fits", y, [line])
        close_but_clipped(lin, back(f), y.astype(np.float64))


def test_roundtrip_unlinked_mtf_per_channel(tmp_path):
    lin = linear_image()
    p = [(0.250, 0.002), (0.300, 0.004), (0.200, 0.001)]
    y = np.stack([U.mtf(lin[i], m, lo, 1.0) for i, (m, lo) in enumerate(p)]).astype(np.float32)
    line = "Unlinked MTF stretch (" + "; ".join(
        f"Ch {i}: lo={lo:.6f}, mid={m:.6f}, hi=1.000000" for i, (m, lo) in enumerate(p)) + ")"
    f = write_fits(tmp_path / "c.fits", y, [line])
    close_but_clipped(lin, back(f), y.astype(np.float64))


def test_roundtrip_inverse_mtf_line_is_undone_by_the_stretch(tmp_path):
    """An "Inverse midtones transfer" in the header was a stretch the other way: its undo is the transfer."""
    lin = linear_image(sigma=0.03, sky=0.4)
    y = U.mtf_inverse(lin, 0.25, 0.0, 1.0).astype(np.float32)
    f = write_fits(tmp_path / "d.fits", y, ["Inverse midtones transfer (0.000, 0.2500, 1.000)"])
    got = back(f)
    ok = (y > 0) & (y < 1)
    np.testing.assert_allclose(got[ok], lin[ok], atol=2e-5)


@pytest.mark.parametrize("B", [-2.0, -1.0, 0.0, 2.0])
@pytest.mark.parametrize("kind,word", [("normal", "GHS ("), ("asinh", "GHS asinh (")])
def test_roundtrip_ghs_mono(tmp_path, B, kind, word):
    """The generalised hyperbolic stretch (command form), its normal and arcsinh variants, each local-intensity case
    (b < -1.. b = -1, b = 0, b > 0)."""
    lin = linear_image(mono=True)
    D, SP, LP, HP = 3.0, 0.1, 0.0, 0.7
    y = U.ght(lin[0], B, D, LP, SP, HP, kind).astype(np.float32)[None]
    if kind == "normal":
        line = f"GHS (pivot: {SP:.3f}, amount: {D:.2f}, local: {B:.1f} [{LP:.2f}, {HP:.2f}])"
    else:
        line = f"GHS asinh (pivot: {SP:.3f}, amount: {D:.2f} [{LP:.2f}, {HP:.2f}])"
    f = write_fits(tmp_path / "e.fits", y[0], [line])
    close_but_clipped(lin, back(f), y.astype(np.float64), tol=1e-4)


def test_roundtrip_ghs_inverse_line_and_dialog_forms(tmp_path):
    lin = linear_image(mono=True, sky=0.3, sigma=0.04)
    y = U.ght(lin[0], 1.0, 2.0, 0.0, 0.3, 0.8, "inverse").astype(np.float32)
    f = write_fits(tmp_path / "f.fits", y, ["Inverse GHS (pivot: 0.300, amount: 2.00, local: 1.0 [0.00, 0.80])"])
    got = back(f)
    ok = (y > 0.001) & (y < 0.999)
    np.testing.assert_allclose(got[0][ok], lin[0][ok], atol=1e-4)
    y2 = U.ght(lin[0], 1.0, 2.0, 0.0, 0.3, 0.8, "normal").astype(np.float32)
    f = write_fits(tmp_path / "g.fits", y2, ["GHS pivot: 0.300, amount: 2.00, local: 1.00 [0.00 0.80]"])
    assert back(f).shape == (3,) + lin.shape[1:]


@pytest.mark.parametrize("human", ["yes", "no"])
@pytest.mark.parametrize("colour", [True, False])
def test_roundtrip_asinh(tmp_path, human, colour):
    """``Asinh stretch (amount, offset, human)``: out = p asinh(beta x) / (x asinh beta), x the luminance."""
    lin = linear_image(mono=not colour, sky=0.02, sigma=0.01)
    beta, off = 20.0, 0.0
    w = np.array([0.2126, 0.7152, 0.0722] if human == "yes" else [1 / 3] * 3)
    p = np.maximum(0.0, (lin - off) / (1 - off)).astype(np.float64)
    x = np.tensordot(w, p, axes=1) if colour else p[0]
    k = np.where(x > 0, np.arcsinh(beta * x) / (x * math.asinh(beta)), 0.0)
    y = np.clip(p * k, 0, 1).astype(np.float32)
    f = write_fits(tmp_path / "h.fits", y if colour else y[0],
                   [f"Asinh stretch (amount: {beta:.1f}, offset: {off:.1f}, human: {human})"])
    close_but_clipped(lin, back(f, check=False), y.astype(np.float64), tol=1e-4)


def test_roundtrip_black_point_shift_and_chain_of_two(tmp_path):
    """A chain: an arcsinh stretch, then a mid-tones transfer, then a black-point shift: undone last first."""
    lin = linear_image(mono=True, sky=0.02, sigma=0.01)
    beta = 10.0
    p = lin[0].astype(np.float64)
    y1 = np.where(p > 0, np.arcsinh(beta * p) / math.asinh(beta), 0.0)
    y2 = U.mtf(y1, 0.4, 0.0, 1.0)
    y3 = np.maximum(0.0, (y2 - 0.1) / 0.9)
    f = write_fits(tmp_path / "i.fits", y3.astype(np.float32), [
        "stack of 20 frames",
        "Asinh stretch (amount: 10.0, offset: 0.0, human: no)",
        "Midtones transfer (0.000, 0.4000, 1.000)",
        "GHS BP shift (new BP: 0.100)",
        "Crop (10, 10, 50, 40)"])
    lines, ch = U.read_history(f)
    pl = U.plan_of(lines, ch)
    assert pl.refusal is None and len(pl.ops) == 3
    got = U.undo(loaded(f)[0], pl.ops, True)
    ok = y3 > 0.001
    np.testing.assert_allclose(got[0][ok], lin[0][ok], atol=2e-4)


# ---------------------------------------------------------------------------------------------- detection

def test_detection_header_line_beats_sky_and_names_the_line(tmp_path):
    lin = linear_image()
    y = U.mtf(lin, 0.25, 0.002, 1.0).astype(np.float32)
    an = U.analyse(write_fits(tmp_path / "s.fits", y, ["Linked MTF stretch (mid=0.250000, lo=0.002000, hi=1.000000)"]))
    assert an.stretched and an.exact and "Linked MTF stretch" in an.why[1]["line"]


def test_detection_linear_image_no_warning(tmp_path):
    an = U.analyse(write_fits(tmp_path / "l.fits", linear_image(sky=0.02), ["stacked", "Background extraction"]))
    assert not an.stretched and not an.exact and an.why is None


def test_detection_sky_above_12_5_percent_without_history(tmp_path):
    lin = linear_image(sky=0.3, sigma=0.02)
    an = U.analyse(write_fits(tmp_path / "t.fits", lin))
    assert an.stretched and not an.exact and an.why[1]["pct"]
    # just under the limit: not called stretched
    assert U.SKY_LIMIT == 0.125
    low = U.analyse(write_fits(tmp_path / "u.fits", linear_image(sky=0.05, sigma=0.01)))
    assert not low.stretched


def test_detection_8_bits(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    p = tmp_path / "v.tif"
    tifffile.imwrite(p, (linear_image()[0] * 255).astype(np.uint8))
    an = U.analyse(p)
    assert an.stretched and not an.exact


def test_detection_autostretch_recipe_is_refused_with_its_line(tmp_path):
    y = U.mtf(linear_image(), 0.25, 0.002, 1.0).astype(np.float32)
    an = U.analyse(write_fits(tmp_path / "w.fits", y, ["Autostretch (shadows: -2.80, target bg: 0.25, linked)"]))
    assert an.stretched and not an.exact and "Autostretch" in an.refusal[1]["line"]


def test_refusals_unknown_step_after_stretch_colour_ghs_saturation(tmp_path):
    y = U.mtf(linear_image(), 0.25, 0.002, 1.0).astype(np.float32)
    mtf = "Linked MTF stretch (mid=0.250000, lo=0.002000, hi=1.000000)"
    an = U.analyse(write_fits(tmp_path / "x1.fits", y, [mtf, "SCNR (green, average neutral, amount 1.00)"]))
    assert not an.exact and "SCNR" in an.refusal[1]["line"]
    an = U.analyse(write_fits(tmp_path / "x2.fits", y, ["GHS (pivot: 0.100, amount: 3.00, local: 1.0 [0.00, 0.70])"]))
    assert not an.exact and "colour" in an.refusal[0]
    an = U.analyse(write_fits(tmp_path / "x3.fits", y[0], ["GHS SAT pivot: 0.100, amount: 3.00, local: 1.00 [0.00 0.70]"]))
    assert not an.exact and "saturation" in an.refusal[0]


def test_refusal_a_parameter_rounded_too_coarsely_moves_the_sky(tmp_path):
    """mid=0.003 written with three digits: half a last digit is 17 % of it, the sky level moves by more than its noise."""
    lin = linear_image(sky=0.002, sigma=0.0004)
    y = U.mtf(lin, 0.003, 0.0, 1.0).astype(np.float32)
    an = U.analyse(write_fits(tmp_path / "r.fits", y, ["Histogram Transf. (mid=0.003, lo=0.000, hi=1.000)"]))
    assert an.stretched and not an.exact and "too few digits" in an.refusal[0]


# ---------------------------------------------------------------------------------------------- the estimate

def test_estimate_autostretched_stack_is_linear_up_to_offset_and_scale():
    """A synthetic stack stretched as the reference program's automatic stretch does (black point 2.8 noise units under
    the sky median, the sky at 0.25): the estimate is the stack to an offset and a scale; what stays after the affine
    fit is under the noise of its sky."""
    lin = linear_image(mono=True, sigma=0.004, sky=0.03)[0].astype(np.float64)
    med = float(np.median(lin))
    sig = 1.4826 * float(np.median(np.abs(lin - med)))
    lo = med - 2.8 * sig
    m = float(U.mtf(np.array(med - lo), 0.25, 0.0, 1.0))
    y = U.mtf(lin, m, lo, 1.0).astype(np.float32)
    est, info = U.estimate(np.repeat(y[None], 3, axis=0).astype(np.float64), mono=True)
    e = est[0]
    A = np.stack([lin.ravel(), np.ones(lin.size)], axis=1)
    coef, *_ = np.linalg.lstsq(A, e.ravel(), rcond=None)
    resid = e.ravel() - A @ coef
    scale = coef[0]
    assert resid.std() / scale < sig
    assert abs(coef[0]) > 0


def test_estimate_refuses_what_it_cannot_measure():
    with pytest.raises(ValueError):
        U.estimate_channel(np.zeros((8, 8)))
