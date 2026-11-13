"""The compiled CPU kernels of the Classic method (core/classic_kernels.py) give what the PyTorch code gives on the CPU:
RCD to the bit, the bad-sample filling and the whole stack to float rounding. Small synthetic frames."""
from __future__ import annotations

import numpy as np
import pytest
import torch
from test_rejection_classic import NF, H, W, make_frames

from astro_gpu.core import classic_kernels as ck
from astro_gpu.core import debayer as db
from astro_gpu.core import resample as rs
from astro_gpu.core import stack_classic as sc


@pytest.fixture
def torch_only():
    """Run the callers' PyTorch code (the reference) inside the test."""
    class Off:
        def __enter__(self):
            ck.ENABLED = False

        def __exit__(self, *a):
            ck.ENABLED = True
    return Off()


def _cfa(h=330, w=410, seed=3):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    a = 1000 + 300 * np.sin(xx / 13.0) * np.cos(yy / 9.0) + rng.normal(0, 20, (h, w))
    for _ in range(40):
        y, x = rng.integers(10, h - 10), rng.integers(10, w - 10)
        a[y - 2:y + 3, x - 2:x + 3] += rng.uniform(500, 20000)
    return torch.from_numpy(a.astype(np.float32))


def test_the_tile_constants_are_the_debayer_modules():
    assert (ck.TILE, ck.TILE_BORDER, ck.STEP) == (db.TILE, db.TILE_BORDER, db.STEP)


@pytest.mark.parametrize("pattern", ["RGGB", "GBRG", "BGGR", "GRBG"])
def test_rcd_is_the_same_to_the_bit(pattern, torch_only):
    cfa = _cfa()
    bal = db.cfa_balance(cfa, pattern)
    got = db.rcd(cfa, pattern, balance=bal)
    with torch_only:
        ref = db.rcd(cfa, pattern, balance=bal)
    assert np.array_equal(got.numpy(), ref.numpy())


def test_bad_samples_are_filled_the_same(torch_only):
    cfa = _cfa()
    rng = np.random.default_rng(4)
    cfa[rng.integers(0, 330, 300), rng.integers(0, 410, 300)] = float("nan")
    cfa[100:112, 200:212] = float("nan")                       # a patch: needs the next ring
    cfa[:2, :] = float("nan")
    cfa[:, 405:] = float("nan")
    got, m = db.fill_bad(cfa, ch_of_k="RGGB")
    with torch_only:
        ref, m2 = db.fill_bad(cfa, ch_of_k="RGGB")
    assert torch.equal(m, m2) and not torch.isnan(got).any()
    assert np.allclose(got.numpy(), ref.numpy(), rtol=1e-5, atol=1e-3)


@pytest.mark.parametrize("clamping", ["bilinear_guide", "none"])
def test_the_resampling_and_the_normalisation_are_the_same(clamping):
    h, w = 200, 260
    cfa = _cfa(h, w)
    cfa[50:54, 60:64] += 30000                                 # a star: ringing, the clamping works
    rgb = db.debayer(cfa, "RGGB")
    coef = np.zeros((2, 10))
    th = 0.004
    coef[0, :3] = [w / 2 + 3.3, np.cos(th) * w / 2, -np.sin(th) * h / 2]
    coef[1, :3] = [h / 2 - 2.1, np.sin(th) * w / 2, np.cos(th) * h / 2]
    coef[0, 3], coef[1, 5], coef[0, 6] = 1e-3, -1e-3, 5e-4
    Q = np.array([[10, 5, -3, 2, 1, -1], [0, 1, 1, 0, 0, 0], [-7, 0, 2, 0, 3, 0.5]], np.float32)
    scale = np.array([0.9, 1.05, 1.1], np.float32)
    X = torch.arange(w, dtype=torch.float32)
    Y = torch.arange(h, dtype=torch.float32)
    a, b = (X / (w / 2) - 1.0).numpy(), (Y / (h / 2) - 1.0).numpy()
    fl = (np.random.default_rng(5).random((h, w)) < 0.01).astype(np.uint8) * 3
    out = np.empty((3, h, w), np.float32)
    fo = np.empty((h, w), np.uint8)
    rs.resample_normalised_cpu(rgb.numpy(), coef, fl, a, b, Q, scale, out, fo, clamping=clamping)
    x, y = rs.source_positions(coef, h, w, w, h)
    ref = rs.resample(rgb, x, y, method="lanczos4", clamping=clamping)
    for c in range(3):
        bg = sc._quad(Q[c], X, Y, w, h)
        ref[c] = (ref[c] - bg) / float(scale[c])
    xi, yi = torch.round(x).long(), torch.round(y).long()
    inside = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
    fref = torch.where(inside, torch.from_numpy(fl)[yi.clamp(0, h - 1), xi.clamp(0, w - 1)], torch.zeros(1, dtype=torch.uint8))
    assert np.array_equal(np.isnan(out), np.isnan(ref.numpy()))
    assert np.array_equal(fo, fref.numpy())
    assert np.nanmax(np.abs(out - ref.numpy())) < 2e-2 * max(1.0, float(np.nanmax(np.abs(ref.numpy()))) * 1e-3)
    assert np.nanmedian(np.abs(out - ref.numpy())) < 1e-3


def test_the_whole_stack_is_the_same(torch_only):
    coef, fr, norm = make_frames()
    fr = [f.copy() for f in fr]
    for f in fr:
        f[20:24, 30:34] = 65000.0                              # saturated
        f[60:63, 80:83] = 55000.0                              # near saturation
    norm.Q[:, :, 1:] = np.random.default_rng(6).uniform(-3, 3, norm.Q[:, :, 1:].shape)

    def run():
        return sc.stack_classic(lambda i: torch.from_numpy(fr[i]), coef, None, norm, np.ones(NF),
                                sc.ClassicParams(half_block=4), "cpu", used=range(NF), ch_of_k="RGGB", H=H, W=W,
                                sat_level=60000.0)
    got = run()
    with torch_only:
        ref = run()
    assert np.array_equal(got.saturation, ref.saturation)
    for name in ("full", "A", "B", "weight"):
        g, r = getattr(got, name), getattr(ref, name)
        assert np.abs(g - r).max() < 1e-2, name
        assert np.abs(g - r).mean() < 1e-4, name
    assert abs(got.rejected_pct - ref.rejected_pct) < 0.05
