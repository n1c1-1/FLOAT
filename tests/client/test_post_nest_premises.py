"""#181, step 0: the two premises nested special layers rest on, checked on synthetic data (nothing of a real project).

(a) A combination's Output is a linear mix of its filters: its balanced channels are linear maps of what it reads
(stages/combine.Combination.steps ``maps``) and the Output is the recipe's matrix on them, so a combination reading an
image that is itself a combination's Output is, in the filters of the inner one, one linear map (offset included):
``kappa . lin + constant``. Post can then stretch every leaf plane with one factor and add them up.

(b) The stars side of a Combine layers over a combination (the stars combination's Output, stretched with the starless
side's statistics, render/stars.CombinedStarsEngine) can be remade from the stars combination's balanced filters: the
linear image through the matrix is the Output, and the stretch of the neutral stars (a grey, the plain mean) is one
factor per pixel, so each filter can be stretched apart and mixed. The stretch of palette stars is a curve of each
channel, not one factor."""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from astro_gpu.render.simple import image_stats, mtf, mtf_midtone
from astro_gpu.render.stars import CombinedStarsEngine, StarSettings
from astro_gpu.stages import combine as C
from astro_gpu.stages import recipes
from tests.test_combine import _sky_field


class _Quiet:
    def log(self, *_a, **_k):
        pass

    def metric(self, *_a, **_k):
        pass

    def progress(self, *_a, **_k):
        pass


def _inputs(roles, image="output"):
    return [{"role": r, "full_scale": 65535.0, "image": image} for r in roles]


def _balanced(cls, params, raw, how="sky", image="output"):
    """(the maps, the Output (3, H, W) the run mixes, the balanced channels before their weights, the recipe) of a
    combination on channels read."""
    spec = cls().spec(params)
    ports = recipes.roles(spec)
    valid = np.ones(next(iter(raw.values())).shape, bool)
    chans = {r: raw[r].copy() for r in ports}
    maps: dict = {}
    tops = {r: 1.0 for r in ports}
    inputs = _inputs(ports, image)
    skies = cls().steps(_Quiet(), spec, inputs, chans, valid, tops, how, {}, maps)
    bal = {r: v.copy() for r, v in chans.items()}
    out = cls().mix_balanced(_Quiet(), params, spec, inputs, chans, skies, tops, valid)
    return maps, out, bal, spec


def _apply(maps, raw):
    return {r: sum(np.float64(c) * raw[k].astype(np.float64) for k, c in coef.items()) + off
            for r, (coef, off) in maps.items()}


@pytest.mark.parametrize("how", ["sky", "signal"])
def test_a_combination_s_output_is_its_matrix_on_its_balanced_weighted_filters(how):
    off = {"SII": 100.0, "Ha": 113.0, "OIII": 126.0}
    raw, _neb = _sky_field(off, {"SII": 2.0, "Ha": 3.0, "OIII": 4.0})
    p = C.SHOParams(weights={"SII": 1.4, "OIII": 0.7})
    maps, out, _bal, spec = _balanced(C.CombineSHO, p, raw, how)
    roles, mat = C.output_matrix(p, spec)
    lin = _apply(maps, raw)
    w = {r: float(p.weights.get(r, 1.0)) for r in roles}
    want = np.tensordot(np.asarray(mat, np.float64), np.stack([w[r] * lin[r] for r in roles]), 1)
    err = float(np.abs(out - want).max() / np.abs(out).max())
    print(f"premise a1 ({how}): relative error {err:.2e}")
    assert err < 1e-5


def test_a_combination_reading_a_combination_is_one_linear_map_of_the_inner_filters():
    """S-H-O (A), then H-O (B) reading A's luminance as its Hα and a filter of its own as its OIII."""
    offA = {"SII": 100.0, "Ha": 113.0, "OIII": 126.0}
    rawA, _ = _sky_field(offA, {"SII": 2.0, "Ha": 3.0, "OIII": 4.0})
    pA = C.SHOParams(weights={"Ha": 1.2})
    mapsA, outA, _balA, specA = _balanced(C.CombineSHO, pA, rawA)
    rolesA, matA = C.output_matrix(pA, specA)
    matA = np.asarray(matA, np.float64)
    wA = {r: float(pA.weights.get(r, 1.0)) for r in rolesA}
    leaf, _ = _sky_field({"OIII": 140.0}, {"OIII": 2.5}, seed=11)
    rawB = {"Ha": C.channel_of(outA, "L"), "OIII": leaf["OIII"]}
    pB = C.HOParams(weights={"OIII": 0.8})
    mapsB, outB, _balB, specB = _balanced(C.CombineHO, pB, rawB)
    rolesB, matB = C.output_matrix(pB, specB)
    matB = np.asarray(matB, np.float64)
    wB = {r: float(pB.weights.get(r, 1.0)) for r in rolesB}
    # B's Hα is LUM . (A's Output) = LUM . matA . (wA * balA): a row over A's balanced filters
    lum = np.asarray(C.LUM, np.float64)
    row = {j: float(lum @ matA[:, i]) * wA[j] for i, j in enumerate(rolesA)}
    balA = _apply(mapsA, rawA)
    got = np.zeros(outB.shape, np.float64)
    for c in range(3):
        for k, r in enumerate(rolesB):                  # B's filter r: its map over what B reads, then its weight
            coef, offset = mapsB[r]
            kk = matB[c, k] * wB[r]
            got[c] += kk * offset
            for ch, a in coef.items():
                if ch == "Ha":                          # what B reads as Ha is A's luminance: over A's filters
                    for j, v in row.items():
                        got[c] += kk * a * v * balA[j]
                else:                                   # a leaf of its own
                    got[c] += kk * a * rawB[ch]
    err = float(np.abs(outB - got).max() / np.abs(outB).max())
    print(f"premise a2 (combination inside a combination): relative error {err:.2e}")
    assert err < 1e-5


def _stars_folders(tmp_path, starless_lin, stars_lin):
    H, W = stars_lin.shape[-2:]
    valid = np.ones((H, W), bool)
    for name, lin in (("starless", starless_lin), ("stars", stars_lin)):
        d = tmp_path / name
        d.mkdir()
        np.save(d / "lin.npy", lin.astype(np.float32))
        np.save(d / "valid.npy", valid)
        (d / "prep_simple.json").write_text(json.dumps(
            {"M": np.eye(3).tolist(), "crop": [0, 0, W, H], "drizzle_scale": 1.0, "linked": True,
             "stats": image_stats(lin, valid)}), encoding="utf-8")
    return tmp_path / "stars", tmp_path / "starless"


def _stars_field(shape=(96, 128), seed=4):
    rng = np.random.default_rng(seed)
    H, W = shape
    yy, xx = np.mgrid[0:H, 0:W]
    out = {r: np.zeros(shape, np.float32) for r in ("SII", "Ha", "OIII")}
    for _ in range(40):
        x, y = rng.uniform(4, W - 4), rng.uniform(4, H - 4)
        a = rng.uniform(100, 6000)
        s = rng.uniform(1.0, 2.0)
        g = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * s * s)).astype(np.float32)
        for r, k in zip(out, rng.uniform(0.5, 1.5, 3)):
            out[r] += np.float32(a * k) * g
    return out


@pytest.mark.parametrize("stars", ["neutral", "palette"])
def test_the_stars_combination_s_stretch_can_be_remade_from_its_filters(tmp_path, stars):
    """The stars combination's Output stretched with the starless side's statistics (the stars side today), remade from
    its balanced filters."""
    stars_in = _stars_field()
    p = C.SHOParams(stars=stars)
    maps, lin_stars, bal, spec = _balanced(C.CombineSHO, p, stars_in, image="stars")
    raw_sl, _ = _sky_field({"SII": 100.0, "Ha": 113.0, "OIII": 126.0}, {"SII": 2.0, "Ha": 3.0, "OIII": 4.0},
                           H=96, W=128)
    _m, lin_sl, _b, _s = _balanced(C.CombineSHO, C.SHOParams(), raw_sl)
    sd, sl = _stars_folders(tmp_path, lin_sl, lin_stars)
    eng = CombinedStarsEngine(sd, sl, device="cpu")
    s = StarSettings(star_strength=1.0, star_size=0.0, star_small=0.0, star_saturation=1.0)
    y_eng = eng.render(s, 1.0, part="stars").final.numpy()
    # remade: the linear image from the filters through the layer's matrix, stretched as ``star_tone`` does
    roles, mat = C.output_matrix(p, spec)
    if stars == "neutral":
        mat = np.full((3, len(roles)), 1.0 / len(roles))
    x = np.tensordot(np.asarray(mat, np.float32), np.stack([bal[r] for r in roles]), 1).astype(np.float32)
    lin_err = float(np.abs(x - lin_stars).max() / np.abs(lin_stars).max())
    st = eng.prep["stats_full"]["lum"]
    c0 = st["median"] - 2.8 * st["mad"]
    hi = max(st["p99995"], c0 + 1e-6)
    xm = min(max((st["median"] - c0) / (hi - c0), 1e-6), 1 - 1e-6)
    m = min(max(mtf_midtone(xm, s.sky) / s.star_strength, 1e-6), 0.5)

    def tone(v):
        return mtf(torch.from_numpy(np.ascontiguousarray(v / (hi - c0))).clamp(0, 1), m).clamp(0, 1).numpy()
    whole = float(np.abs(tone(x) - y_eng).max())
    # one factor per pixel from the mix (as the starless side's filters are stretched), each filter stretched apart
    g = tone(x[1]) / np.maximum(x[1], 1e-9)
    parts = {r: g * bal[r] for r in roles}
    apart = np.tensordot(np.asarray(mat, np.float32), np.stack([parts[r] for r in roles]), 1)
    one = float(np.abs(apart - y_eng).max())
    print(f"premise b ({stars}): linear {lin_err:.2e}; whole stretch {whole:.2e}; one factor apart {one:.2e}")
    assert lin_err < 1e-5
    assert whole < 1e-5
    if stars == "neutral":
        assert one < 1e-4
    else:                                               # a curve of each channel: not one factor of the mix
        assert one > 1e-2
