"""Issue #163: the S-H-O block's default hues measured on reference images, and the narrow-band combinations composed
as lightness (the channels weighted by their noise) and colour. On synthetic channels: a filter's signal alone lands,
in Post's rendering with the colours kept, on its measured Oklch hue; a neutral input stays grey; a weak, noisy channel
no longer paints the sky; a colour camera's rendering is what it was; S-H-O and H-O are the narrow-band types."""
from __future__ import annotations

import numpy as np
import pytest

from astro_gpu.render import sliders
from astro_gpu.render.simple import SimpleRenderEngine, SimpleSettings
from astro_gpu.stages import combine as fusion
from astro_gpu.stages.combine import COMBINATIONS, HUBBLE_HUES, HUBBLE_OKLCH, CombineSHO, compose, hue_palette, mix

SHO = CombineSHO.ports
MAT = hue_palette(HUBBLE_HUES)                    # the S-H-O block's default hues


def oklch(rgb: np.ndarray) -> np.ndarray:
    """(..., 3) display sRGB 0..1 -> (..., 3) Oklch (L, C, hue in degrees; B. Ottosson 2020)."""
    a = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    m1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929], [0.2119034982, 0.6806995451, 0.1073969566],
                   [0.0883024619, 0.2817188376, 0.6299787005]])
    m2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468], [1.9779984951, -2.4285922050, 0.4505937099],
                   [0.0259040371, 0.7827717662, -0.8086757660]])
    lab = np.cbrt(a @ m1.T) @ m2.T
    return np.stack([lab[..., 0], np.hypot(lab[..., 1], lab[..., 2]),
                     np.degrees(np.arctan2(lab[..., 2], lab[..., 1])) % 360], -1)


def channels(n: int = 192, sigma: dict | None = None, seed: int = 0) -> tuple[dict, dict]:
    """Balanced S II, H-alpha and O III (sky at 0, noise ``sigma`` each) with four 32 px patches: each filter alone at
    0.6, and the three equal at 0.6 (a neutral source). Returns (the channels, the patches' slices)."""
    rng = np.random.default_rng(seed)
    sigma = sigma or {"SII": 0.002, "Ha": 0.002, "OIII": 0.002}
    ch = {r: rng.normal(0, sigma[r], (n, n)).astype(np.float32) for r in ("SII", "Ha", "OIII")}
    cut = {"SII": np.s_[16:48, 16:48], "Ha": np.s_[16:48, 80:112], "OIII": np.s_[80:112, 16:48],
           "grey": np.s_[80:112, 80:112]}
    for r in ch:
        ch[r][cut[r]] += 0.6
        ch[r][cut["grey"]] += 0.6
    return ch, cut


def test_each_filter_alone_lands_on_its_measured_hue_and_neutral_stays_grey():
    """Through the S-H-O composition and Post's rendering with the colours of the linear image kept (Keep the
    colours 1, saturation 1: the mapping the hues were placed on), S II alone, H-alpha alone and O III alone have the
    Oklch hues measured on the Hubble teams' images (the warm areas' lower quartile and median, the blue areas'
    median); equal channels stay grey, in the default rendering too."""
    ch, cut = channels()
    noise = {r: 0.002 for r in ch}
    rgb, _ = compose(ch, SHO, MAT, noise, sigma=0)
    eng = SimpleRenderEngine.from_array(rgb, "cpu")
    s = SimpleSettings(keep_colours=1.0, saturation=1.0)
    y = eng.render(s, 1.0).final.numpy().transpose(1, 2, 0)
    for role, target in zip(("SII", "Ha", "OIII"), HUBBLE_OKLCH):
        h = float(np.median(oklch(y[cut[role]].reshape(-1, 3).astype(np.float64))[:, 2]))
        assert abs((h - target + 180) % 360 - 180) < 4, (role, h, target)
    for img in (y, eng.render(SimpleSettings(), 1.0).final.numpy().transpose(1, 2, 0)):
        grey = img[cut["grey"]].reshape(-1, 3)
        assert np.abs(grey - grey.mean(1, keepdims=True)).max() < 0.01


def test_the_lightness_is_the_noise_weighted_mean_and_a_neutral_input_keeps_its_level():
    ch = {"SII": np.full((8, 8), 0.5, np.float32), "Ha": np.full((8, 8), 0.5, np.float32),
          "OIII": np.full((8, 8), 0.5, np.float32)}
    rgb, _ = compose(ch, SHO, MAT, {"SII": 0.02, "Ha": 0.001, "OIII": 0.003})
    assert np.allclose(rgb, 0.5, atol=1e-5)                           # grey, at the channels' level
    L, nl = fusion.lightness({"a": np.ones((2, 2), np.float32), "b": np.zeros((2, 2), np.float32)},
                             {"a": 1.0, "b": 2.0})
    assert np.allclose(L, 0.8) and nl == pytest.approx(1 / np.sqrt(1.25))      # weights 1 and 1/4
    L, nl = fusion.lightness({"a": np.ones((2, 2), np.float32), "b": np.zeros((2, 2), np.float32)},
                             {"a": 1.0, "b": 2.0}, equal=True)
    assert np.allclose(L, 0.5) and nl == pytest.approx(np.sqrt(5) / 2)          # the plain mean, its noise
    assert fusion.chroma_sigma({"a": 1.0, "b": 1.0}, 1.0) == 0.0               # no channel noisier: no smoothing
    s = fusion.chroma_sigma({"a": 15.0, "b": 1.0}, 1.0)                         # M27's S II against its H-alpha
    assert s == pytest.approx(15 / (2 * np.sqrt(np.pi)))


def test_a_weak_noisy_channel_no_longer_paints_the_sky():
    """M27's case: S II 15 times noisier than H-alpha and O III, raised by the balance. The plain palette scatters
    its noise over the sky as colour; composed, the sky is grey within a fraction of its noise and the lightness keeps
    the H-alpha's grain only."""
    sig = {"SII": 0.03, "Ha": 0.002, "OIII": 0.002}
    ch, cut = channels(sigma=sig, seed=1)
    sky = np.zeros((192, 192), bool)
    sky[130:, :] = True
    plain = mix(ch, SHO, MAT)
    comp, notes = compose(ch, SHO, MAT, sig, sky=sky)
    assert notes["sigma"] > 1

    def chroma(rgb):
        return np.abs(rgb - rgb.mean(0, keepdims=True)).max(0)[sky]
    assert chroma(comp).std() < 0.05 * chroma(plain).std()
    assert comp[:, sky].std() < 0.2 * plain[:, sky].std()
    fixed, notes = compose(ch, SHO, MAT, sig, sigma=2.5, sky=sky)               # a radius set by hand: that one
    assert notes["sigma"] == 2.5 and not np.array_equal(fixed, comp)


def test_a_colour_camera_keeps_its_rendering_and_the_palettes_agree():
    """A colour camera's settings and image are what they were; an RGB or LRGB combination is no narrow-band one (no
    composition); S-H-O and H-O are: Post greys their H-alpha slider."""
    for kind in sliders.KINDS:
        assert sliders.settings_at(kind, None, {}, True, "") == sliders.base_settings(kind)
    assert sliders.settings_at("simple", None, {}) == SimpleSettings()
    rng = np.random.default_rng(3)
    img = (rng.random((3, 64, 64)) * 0.1).astype(np.float32)
    eng = SimpleRenderEngine.from_array(img, "cpu")
    a = eng.render(SimpleSettings(), 1.0).final.numpy()
    b = eng.render(sliders.settings_at("simple", None, {}, True, ""), 1.0).final.numpy()
    assert np.array_equal(a, b)
    assert [c.kind for c in COMBINATIONS if c.narrowband] == ["sho", "ho"]
    assert sliders.availability("combined", True, "palette")["halpha"] == sliders.WHY_PALETTE


def test_a_split_layer_says_its_origin(tmp_path):
    """The Stars/background block after a narrow-band combination writes the Output's origin; Post reads it there (the
    recombined layer of an SHO fusion offered the H-alpha slider of a colour camera)."""
    import json

    from astro_gpu.render.layers import origin_of
    (tmp_path / "prep_stars.json").write_text(json.dumps({"origin": "palette"}), encoding="utf-8")
    assert origin_of(tmp_path) == "palette"
    (tmp_path / "prep_stars.json").write_text(json.dumps({}), encoding="utf-8")
    assert origin_of(tmp_path) == ""
