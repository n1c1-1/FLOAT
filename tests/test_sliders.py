"""Post's simplified sliders (render/sliders.py): tables, position 2 = the base, monotonic, greyed with reasons."""
from __future__ import annotations

import cv2
import numpy as np
import pytest
import torch

from astro_gpu.render import sliders as S
from astro_gpu.render.simple import SimpleRenderEngine, SimpleSettings
from astro_gpu.render.stars import StarSettings

DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
SIMPLE_BASES = [None, {"sky": 0.3, "saturation": 0.8, "noise_mix": 0.9}, SimpleSettings(contrast=0.2),
                {"saturation": 0.0}]
STAR_BASES = [None, {"sky": 0.3, "star_strength": 1.4, "blend": "Lighten"}, StarSettings(star_size=1.0),
              {"star_saturation": 0.0}]
POSITIONS = [i / 4 for i in range(17)]


def test_the_eleven_sliders_of_post():
    assert S.KEYS == ("sky", "contrast", "faint", "saturation", "colours", "halpha", "noise", "star_strength",
                      "star_share", "star_size", "star_color")
    # position 2 of every table is the suggested rendering (the defaults)
    s = StarSettings()
    for sl in S.SLIDERS:
        assert len(sl.ends) == 2 and sl.group in (S.NEBULA, S.STARS)
        assert (sl.group == S.STARS) == (sl.needs == "stars")
        for name, t in sl.table.items():
            assert len(t) == sl.hi + 1 and abs(t[2] - getattr(s, name)) < 1e-9, (sl.key, name)
        assert sl.hi == (6 if sl.key == "saturation" else 4) == S.top(sl.key)
    # Sky: the curve's target above 2, the black point below it (#186)
    assert S.BY_KEY["sky"].table == {"sky": S.SKY_TABLE, "black": S.BLACK_TABLE}
    assert S.SKY_TABLE == (0.25, 0.25, 0.25, 0.31, 0.37) and S.BLACK_TABLE == (0.84, 0.47, 0.0, 0.0, 0.0)
    assert S.BY_KEY["noise"].table["noise_mix"][:3] == (0.4, 0.7, 1.0)


@pytest.mark.parametrize("kind,bases", [("simple", SIMPLE_BASES), ("combined", STAR_BASES)])
def test_position_2_is_the_base(kind, bases):
    for base in bases:
        b = S.base_settings(kind, base)
        at2 = S.settings_at(kind, base, {k: 2.0 for k in S.KEYS})
        assert at2 == b
        assert S.settings_at(kind, base, {}) == b


def test_default_rendering_follows_the_tables():
    for pos in range(7):
        for sl in S.SLIDERS:
            if pos > sl.hi:
                continue
            kind = "simple" if sl.key == "noise" else "combined"
            got = S.settings_at(kind, None, {sl.key: pos})
            for name, t in sl.table.items():
                assert getattr(got, name) == pytest.approx(t[pos]), (sl.key, name, pos)


@pytest.mark.parametrize("kind,bases", [("simple", SIMPLE_BASES), ("combined", STAR_BASES)])
def test_each_slider_is_monotonic(kind, bases):
    offered = [sl for sl in S.SLIDERS if not S.availability(kind, True)[sl.key]]
    assert offered
    for base in bases:
        for sl in offered:
            vals = [S.settings_at(kind, base, {sl.key: p}) for p in POSITIONS]
            moved = False
            for name in sl.table:
                v = [getattr(x, name) for x in vals]
                up = all(b >= a - 1e-12 for a, b in zip(v, v[1:]))
                down = all(b <= a + 1e-12 for a, b in zip(v, v[1:]))      # Small stars: left = fewer stars (#7)
                assert up or down, (kind, base, sl.key, name, v)
                moved |= v[-1] != v[0]
            assert moved, (kind, base, sl.key)              # never frozen, even on a base of 0 (saturation)


def test_steps_and_ratios():
    b = S.settings_at("simple", {"sky": 0.3, "saturation": 1.0}, {"sky": 4, "saturation": 0})
    assert b.sky == pytest.approx(0.42)                                 # a step: +0.12
    assert b.saturation == pytest.approx(1.0 * 0.6 / 1.2)               # a ratio
    half = S.settings_at("simple", None, {"sky": 1.5, "contrast": 3.5})
    assert half.sky == pytest.approx(0.25) and half.black == pytest.approx(0.235)
    assert half.contrast == pytest.approx(0.375)
    assert S.settings_at("simple", {"sky": 0.55}, {"sky": 4}).sky == 0.6   # inside the setting's bounds (0.02-0.6)
    low = S.settings_at("simple", {"sky": 0.3}, {"sky": 0})        # the low end: the base's curve, its black raised (#186)
    assert low.sky == pytest.approx(0.3) and low.black == pytest.approx(0.84)
    assert S.settings_at("simple", None, {"sky": 9}).sky == pytest.approx(0.37)   # positions beyond 0-4 are clamped
    # Saturation goes on to +4 (positions 5, 6: the last step again), its law to +2 as it was, a range of its own
    assert S.BY_KEY["saturation"].table["saturation"][:5] == (0.6, 0.9, 1.2, 1.5, 1.85)
    assert S.settings_at("simple", None, {"saturation": 6}).saturation == pytest.approx(2.55)
    assert S.settings_at("simple", None, {"saturation": 5}).saturation == pytest.approx(2.2)
    assert S.settings_at("simple", None, {"saturation": 9}).saturation == pytest.approx(2.55)
    assert S.settings_at("simple", {"saturation": 1.5}, {"saturation": 6}).saturation == pytest.approx(3.0)  # bound
    assert S.settings_at("simple", None, {"unknown": 3}) == SimpleSettings()
    with pytest.raises(ValueError):
        S.settings_at("simple", "No such preset", {})
    with pytest.raises(ValueError):
        S.availability("other", True)
    # the Stars block's images have the star settings; the Output block's drop them
    assert isinstance(S.base_settings("combined"), StarSettings)
    assert S.base_settings("simple", StarSettings(sky=0.3, star_size=1.0)) == SimpleSettings(sky=0.3)
    assert S.settings_at("combined", None, {"star_size": 3}).star_size == pytest.approx(0.0)


def test_greyed_sliders_give_their_reasons():
    stars = ("star_strength", "star_share", "star_size", "star_color")
    simple = S.availability("simple", False)
    assert simple["sky"] == simple["contrast"] == simple["saturation"] == ""
    assert simple["noise"] == S.WHY_DENOISE and all(simple[k] == S.WHY_STARS for k in stars)
    assert S.availability("simple", True)["noise"] == ""
    exit_ = S.availability("stacking", True)
    assert exit_["noise"] == S.WHY_EXIT_DENOISE and all(exit_[k] == S.WHY_EXIT_STARS for k in stars)
    starless = S.availability("starless", True)
    assert starless["sky"] == "" and starless["noise"] == S.WHY_WHOLE and all(starless[k] == S.WHY_STARS for k in stars)
    alone = S.availability("stars", True)
    assert all(alone[k] == S.WHY_STARS_ONLY for k in ("sky", "contrast", "saturation", "noise"))
    assert all(alone[k] == "" for k in stars)
    combined = S.availability("combined", True)
    assert {k for k, v in combined.items() if v} == {"noise"} and combined["noise"] == S.WHY_WHOLE
    # a greyed slider's position is ignored
    assert S.settings_at("simple", None, {"noise": 0}, has_denoise=False) == SimpleSettings()
    assert S.settings_at("simple", None, {"noise": 0}, has_denoise=True).noise_mix == pytest.approx(0.4)
    got = S.settings_at("stacking", None, {"star_size": 4, "sky": 3})
    assert got.sky == pytest.approx(0.31) and got.model_copy(update={"sky": 0.31}) == SimpleSettings(sky=0.31)
    assert S.settings_at("stars", None, {"sky": 0, "star_size": 4}) == StarSettings(star_size=1.0)


def test_rendered_sliders_move_the_image():
    rng = np.random.default_rng(3)
    H, W = 160, 240
    neb = cv2.GaussianBlur(rng.random((H, W)).astype(np.float32), (0, 0), 12)
    neb = (neb - neb.min()) / (neb.max() - neb.min()) * 60
    lin = np.stack([neb, neb * .8, neb * .7]) + 20 + rng.normal(0, 2, (3, H, W)).astype(np.float32)
    eng = SimpleRenderEngine.from_array(lin, device=DEV)
    sky = [float(eng.render(S.settings_at("simple", None, {"sky": p}), 0.5).final.mean()) for p in (0, 1, 2, 3, 4)]
    assert all(b > a for a, b in zip(sky, sky[1:]))
    sat = [eng.render(S.settings_at("simple", None, {"saturation": p}), 0.5).final for p in (0, 4)]
    assert float(sat[0].std(0).mean()) < float(sat[1].std(0).mean())          # paler to more vivid
    base = eng.render(SimpleSettings(), 0.5).final
    assert torch.equal(eng.render(S.settings_at("simple", None, {k: 2 for k in S.KEYS}), 0.5).final, base)


def _sky_and_nebula():
    """A linear image: a sky of noise (20, deviation 2) and a nebula up to 600 above it, on a tenth of the field."""
    rng = np.random.default_rng(4)
    H, W = 160, 240
    yy, xx = np.mgrid[0:H, 0:W]
    neb = (600 * np.exp(-((xx - 120) ** 2 + (yy - 80) ** 2) / (2 * 15.0 ** 2))).astype(np.float32)
    lin = np.stack([neb, neb * .8, neb * .7]) + 20 + rng.normal(0, 2, (3, H, W)).astype(np.float32)
    return lin, neb


def _lum(y: torch.Tensor) -> np.ndarray:
    y = y.cpu().numpy()
    return 0.2126 * y[0] + 0.7152 * y[1] + 0.0722 * y[2]


def test_sky_below_its_default_raises_the_black_point():
    """Sky's left half moves the stretch's black point toward the median sky and keeps the curve of its default, as the
    black slider of a Levels tool (#186: a lower target of the curve hollowed the whole curve, the nebula darkened
    almost as much as the sky). The sky falls toward black, the nebula's core moves little."""
    lin, neb = _sky_and_nebula()
    eng = SimpleRenderEngine.from_array(lin, device=DEV)
    sky, core = neb < 0.5, neb > np.percentile(neb, 99)
    L = {p: _lum(eng.render(S.settings_at("simple", None, {"sky": p}), 1.0).final) for p in (0, 1, 2)}

    def fall(p, where):
        return 1 - float(np.median(L[p][where])) / float(np.median(L[2][where]))
    assert fall(0, sky) > 0.75 and fall(0, core) < 0.05, (fall(0, sky), fall(0, core))
    assert fall(0, sky) > 15 * fall(0, core)
    assert 0.25 < fall(1, sky) < fall(0, sky) and fall(1, core) < fall(0, core) + 1e-6
    # the image's median lands where the lower targets of the curve put it (0.05 at 0, 0.15 at 1)
    assert float(np.median(L[0])) == pytest.approx(0.05, abs=0.015)
    assert float(np.median(L[1])) == pytest.approx(0.15, abs=0.015)
    # the curve hollowed to the same median darkened the nebula several times more, its core and its middle tones
    hollow = _lum(eng.render(SimpleSettings(sky=0.05), 1.0).final)
    mid = (neb > 30) & (neb < 100)
    for where in (core, mid):
        assert 1 - float(np.median(hollow[where])) / float(np.median(L[2][where])) > 3 * fall(0, where)


def test_sky_at_its_default_is_the_stretch_of_before():
    """Position 2 and above: no black point raised; the picture is the screen transfer function as it was (black at
    median - 2.8 MAD, white at the 99.995th percentile, the median at Sky)."""
    from astro_gpu.render.simple import mtf, mtf_midtone
    lin, _neb = _sky_and_nebula()
    eng = SimpleRenderEngine.from_array(lin, device=DEV)
    st = eng.prep["stats"]["lum"]
    c0 = st["median"] - 2.8 * st["mad"]
    xm = (st["median"] - c0) / (st["p99995"] - c0)
    for pos, target in ((2, 0.25), (3, 0.31)):
        s = S.settings_at("simple", {"keep_colours": 0.0, "saturation": 1.0}, {"sky": pos})
        assert s.black == 0.0 and s.sky == pytest.approx(target)
        x = torch.from_numpy(lin)
        want = mtf(((x - c0) / (st["p99995"] - c0)).clamp(0, 1), mtf_midtone(xm, s.sky)).clamp(0, 1)
        got = eng.render(s, 1.0).final.cpu()
        assert torch.allclose(got, want.to(got.dtype), atol=1e-5)


def test_faint_tones_turn_around_the_sky_the_black_point_leaves():
    """Faint tones keeps the sky where the stretch put it: with the black point raised, that is under the Sky setting
    (render/simple.sky_level), and the median sky does not move."""
    lin, neb = _sky_and_nebula()
    eng = SimpleRenderEngine.from_array(lin, device=DEV)
    sky = neb < 0.5
    s = S.settings_at("simple", None, {"sky": 0})
    flat = float(np.median(_lum(eng.render(s, 1.0).final)[sky]))
    lifted = float(np.median(_lum(eng.render(s.model_copy(update={"faint": 1.5}), 1.0).final)[sky]))
    assert lifted == pytest.approx(flat, abs=1.5 / 255)


def test_the_halpha_slider_is_for_colour_cameras_only():
    """render/simple.emission adds the red above the green: on an SHO fusion red is S II and green is H-alpha, so the
    slider strengthened S II over H-alpha, the opposite of its name; a monochrome image has no such channels."""
    from types import SimpleNamespace

    from astro_gpu import api
    assert S.availability("simple", True)["halpha"] == ""
    assert S.availability("simple", True, "")["halpha"] == ""
    assert S.availability("simple", True, "palette")["halpha"] == S.WHY_PALETTE
    assert S.WHY_PALETTE == "Only for colour cameras: in a palette combination, Hα is its own channel"
    assert "monochrome" in S.availability("simple", True, "mono")["halpha"]
    assert not S.availability("simple", True, "palette")["sky"]                        # the others stay
    # a greyed slider is ignored by the settings, wherever its position is
    assert S.settings_at("simple", None, {"halpha": 4}, True, "palette").halpha == 0.0
    assert S.settings_at("simple", None, {"halpha": 4}, True, "").halpha > 0.0
    # where the project says it
    sho = SimpleNamespace(source={"fusion": {"type": "sho"}})
    rgb = SimpleNamespace(source={"fusion": {"type": "lrgb"}})
    mono = SimpleNamespace(source=None)
    p = SimpleNamespace(is_mono=lambda w: w is mono)
    assert api._origin(p, sho) == "palette" and api._origin(p, rgb) == ""
    assert api._origin(p, mono) == "mono" and api._origin(p, SimpleNamespace(source=None)) == ""


def test_small_stars_is_a_one_sided_share_slider():
    sl = S.BY_KEY["star_share"]
    assert sl.ends == ("None", "All") and S.slider_lo("star_share") == 2.0 and S.slider_lo("sky") == 0.0
    assert S.settings_at("combined", None, {"star_share": 2.0}).star_small == 0.0            # left end: none
    assert S.settings_at("combined", None, {"star_share": 3.0}).star_small == pytest.approx(1.0)   # half of the stars
    assert S.settings_at("combined", None, {"star_share": 4.0}).star_small == pytest.approx(2.0)   # all (the range's end)
