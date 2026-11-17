"""A combination's master layer in Post (client/postengine.py, client/postfusion.py): the Combine layers layer over a
combination of filters mixes them itself. Each filter is a sub-image under the layer's starless image, with its eye,
its mask and its filters, which act on that filter before the mix; the filters are stretched by one factor per pixel,
the same for all of them (the colour-preserving stretch of the mix: render/simple.SimpleRenderEngine.colour_factor),
so the mix of the stretched filters is the stretched mix."""
from __future__ import annotations

import copy
import time

import numpy as np
import pytest
import torch

from astro_gpu.client import postengine as pe
from astro_gpu.client.postengine import Canvas, Composer, Provider
from astro_gpu.render.simple import SimpleRenderEngine, SimpleSettings, image_stats
from astro_gpu.stages.combine import HUBBLE_HUES, hue_palette

H, W = 48, 64
ROLES = ("SII", "Ha", "OIII")


def planes():
    """Three balanced filters (their skies at 0, a little noise), a nebula of different strength in each."""
    rng = np.random.default_rng(5)
    yy, xx = np.mgrid[0:H, 0:W]
    neb = np.exp(-((xx - 30) ** 2 + (yy - 22) ** 2) / (2 * 9.0 ** 2)).astype(np.float32)
    ring = np.exp(-((np.hypot(xx - 30, yy - 22) - 12) ** 2) / (2 * 3.0 ** 2)).astype(np.float32)
    amp = {"SII": 0.2 * ring, "Ha": neb + 0.4 * ring, "OIII": 0.7 * neb}
    return {r: (amp[r] + rng.normal(0, 0.004, (H, W))).astype(np.float32) for r in ROLES}


class Filters(Provider):
    """The fake engine: the balanced filters at a level (block means), all covered; no stars side."""

    def __init__(self, p=None):
        self.p = p or planes()
        self.asked = 0

    def available(self, sid):
        return isinstance(sid, tuple) and sid[0] == pe.FUSION

    def canvas(self, pids):
        return Canvas(W, H)

    def layer(self, lay, level, box, draft=False, stop=None):
        raise AssertionError("a master layer's images are made by the composer")

    def fusion_planes(self, lay, level, box):
        self.asked += 1
        f = 2 ** level
        out = {}
        for r, a in self.p.items():
            a = a[:H // f * f, :W // f * f].reshape(H // f, f, W // f, f).mean(axis=(1, 3)) if f > 1 else a
            x0, y0, x1, y1 = box
            out[r] = a[y0:y1, x0:x1].astype(np.float32)
        return out, np.ones(next(iter(out.values())).shape, bool)


def master(**kw):
    g = pe.new_workflow_layer(5, "S-H-O · Combine layers", True, part="combined")
    g.update(starless="output", stars_pid=None)
    g["fusion"] = {"roles": list(ROLES), "names": {"SII": "S II", "Ha": "Hα", "OIII": "O III"},
                   "matrix": hue_palette(HUBBLE_HUES), "weights": {r: 1.0 for r in ROLES}, "balance": "sky"}
    g.update(kw)
    return g


def compose(layers, provider=None, want=(), level=0):
    return Composer(provider or Filters()).compose(pe.normalize(copy.deepcopy(layers)), level, want=want)


def mix(p, weights=None):
    mat = np.asarray(hue_palette(HUBBLE_HUES), np.float32)
    w = weights or {}
    return np.tensordot(mat, np.stack([p[r] * np.float32(w.get(r, 1.0)) for r in ROLES]), 1).astype(np.float32)


def test_the_rows_are_the_layer_its_stars_its_starless_image_and_under_it_one_row_per_filter():
    g = master()
    f = pe.new_filter_layer("levels", g["id"], pe.role_sub("OIII"))
    rows = pe.rows(pe.normalize([f, g]))
    shown = [(r.get("name"), d) for r, d in rows]
    assert shown == [("S-H-O · Combine layers", 0), ("Stars", 1), ("Starless image", 1), ("S II", 2), ("Hα", 2),
                     ("O III", 2), ("Levels", 3), ]
    assert rows[5][0]["sub"] == "f:OIII" and rows[5][0]["kind"] == "sub"


def test_a_layer_without_the_combination_s_description_is_the_combine_layers_layer_it_was():
    g = master()
    g.pop("fusion")
    assert [r.get("name") for r, _d in pe.rows([g])] == ["S-H-O · Combine layers", "Stars", "Starless image"]
    assert pe.sub_layer(g, "starless")["part"] is None                    # its Output, as before


def test_the_default_picture_is_the_mix_through_the_colour_preserving_stretch():
    p = planes()
    got = compose([master()]).rgb
    x = mix(p)
    st = image_stats(x, np.ones((H, W), bool))
    s = SimpleSettings(keep_colours=1.0)          # the stretch whose colours are the linear image's (Colours: true)
    want = SimpleRenderEngine.tone(torch.from_numpy(x), st, s, True).numpy()
    assert np.abs(got - np.moveaxis(want, 0, -1)).max() < 2e-5


def test_one_factor_stretches_every_filter_and_the_mix_of_the_stretched_filters_is_the_stretched_mix():
    p = planes()
    x = mix(p)
    st = image_stats(x, np.ones((H, W), bool))
    for s in (SimpleSettings(), SimpleSettings(stretch="arcsinh"), SimpleSettings(faint=1.0)):
        F, c0 = SimpleRenderEngine.colour_factor(torch.from_numpy(x), st, s)
        F = F.numpy()
        kids = {r: F * (p[r] - c0) for r in ROLES}
        for r in ROLES:                            # the same factor on each filter
            far = np.abs(p[r] - c0) > 1e-3
            assert np.allclose(kids[r][far] / (p[r] - c0)[far], F[far], rtol=1e-5)
        after = F[None] * np.maximum(x - c0, 0)    # the mix stretched
        before = np.maximum(mix(kids) - 0 * F[None], 0)       # the stretched filters mixed (balanced hues: rows sum 1)
        assert np.abs(before - after).max() < 1e-5


def test_the_filters_images_are_their_stretched_grey_and_the_starless_image_their_mix():
    comp = Composer(Filters())
    g = pe.normalize([master()])[0]
    cv = comp.canvas_of([g])
    whole = (0, 0, W, H)
    kids = {r: comp.layer_rgb(pe.sub_layer(g, pe.role_sub(r)), 0, whole, cv) for r in ROLES}
    for r in ROLES:
        assert np.array_equal(kids[r][..., 0], kids[r][..., 2])            # a grey image
    together = comp.layer_rgb(g, 0, whole, cv)
    assert np.abs(together - compose([g]).rgb).max() < 1e-6


def test_an_eye_off_leaves_that_filter_out_of_the_mix_and_the_others_as_they_were():
    p = planes()
    off = master(shown={"starless": True, "stars": True, "f:OIII": False})
    got = compose([off]).rgb
    without = dict(p, OIII=np.zeros_like(p["OIII"]))
    # the factor is the whole mix's: the other filters keep their look
    x = mix(p)
    st = image_stats(x, np.ones((H, W), bool))
    F, c0 = SimpleRenderEngine.colour_factor(torch.from_numpy(x), st, SimpleSettings())
    F = F.numpy()
    kids = {r: (F * (without[r] - c0) if r != "OIII" else np.zeros_like(F)) for r in ROLES}
    y = np.maximum(mix(kids), 0)
    want = SimpleRenderEngine.finish(torch.from_numpy(y), SimpleSettings()).numpy()
    assert np.abs(got - np.moveaxis(want, 0, -1)).max() < 2e-5
    assert np.abs(got - compose([master()]).rgb).max() > 0.05


def test_a_filter_s_own_filter_acts_on_it_before_the_mix():
    g = master()
    lv = pe.new_filter_layer("levels", g["id"], pe.role_sub("OIII"))
    lv["params"].update(black=0.1, gamma=0.7)
    got = compose([lv, g]).rgb
    comp = Composer(Filters())
    gg = pe.normalize([g])[0]
    cv = comp.canvas_of([gg])
    whole = (0, 0, W, H)
    parts = comp._fusion_parts(pe.sub_layer(gg, "starless"), 0, whole, cv, True)
    kids = dict(zip(ROLES, parts))
    ox = np.repeat(kids["OIII"][..., None], 3, -1)
    filtered = pe.FILTERS["levels"]["fn"](ox.copy(), lv["params"], pe.Ctx(comp, 0, whole))
    kids["OIII"] = pe.lum(filtered)
    want = comp._fusion_mix(gg, kids, parts[-2], parts[-1], cv)
    assert np.abs(got - want).max() < 1e-6
    assert np.abs(got - compose([g]).rgb).max() > 0.02


def test_a_filter_s_mask_keeps_it_where_it_is_white():
    disc = pe.new_mask()
    part = pe.new_part("brush")
    part["strokes"] = [(30.0, 22.0, 8.0, 100.0, True)]
    disc["parts"] = [part]
    g = master(side_masks={"f:OIII": disc})
    got = compose([g], want=[(g["id"], "f:OIII")])
    m = got.masks[(g["id"], "f:OIII")]
    plain = compose([master()]).rgb
    off = compose([master(shown={"starless": True, "stars": True, "f:OIII": False})]).rgb
    assert np.abs(got.rgb[m == 1] - plain[m == 1]).max() < 1e-6
    assert np.abs(got.rgb[m == 0] - off[m == 0]).max() < 1e-6


def test_an_intensity_scales_its_filter_before_the_stretch():
    p = planes()
    g = master()
    g["fusion"]["weights"]["OIII"] = 2.0
    got = compose([g]).rgb
    x = mix(p, {"OIII": 2.0})
    want = SimpleRenderEngine.tone(torch.from_numpy(x), image_stats(x, np.ones((H, W), bool)),
                                   SimpleSettings(keep_colours=1.0), True).numpy()
    assert np.abs(got - np.moveaxis(want, 0, -1)).max() < 2e-5


def test_a_hue_change_reads_the_filters_once_and_a_region_matches_the_whole_picture():
    prov = Filters()
    comp = Composer(prov)
    g = pe.normalize([master()])
    comp.compose(g, 1)
    n = prov.asked
    g2 = copy.deepcopy(g)
    g2[0]["fusion"]["matrix"] = hue_palette((0.0, 120.0, 240.0))
    comp.compose(g2, 1)
    assert prov.asked - n <= 2                  # the filters at the level and at the measure level, nothing more
    whole = comp.compose(g2, 0).rgb
    part = comp.compose(g2, 0, (8, 8, 40, 32)).rgb
    assert np.abs(part - whole[8:32, 8:40]).max() < 1e-6


def test_a_master_layer_saves_without_the_combination_s_description_and_loads_its_filters_rows():
    g = master(shown={"starless": True, "stars": True, "f:SII": False}, side_masks={"f:Ha": pe.new_mask()})
    f = pe.new_filter_layer("levels", g["id"], "f:OIII")
    saved = pe.save_layers([f, g])
    assert all("fusion" not in d for d in saved)
    back = pe.load_layers(saved, {5})
    lay = next(x for x in back if x["kind"] == "workflow")
    assert lay["shown"]["f:SII"] is False and "f:Ha" in lay["side_masks"]
    assert next(x for x in back if x["kind"] == "filter")["sub"] == "f:OIII"
    assert pe.selection_side("mask_f:Ha") == "f:Ha" and pe.selection_side("f:OIII") == "f:OIII"


def test_a_filter_of_a_role_the_recipe_no_longer_has_acts_on_the_mix():
    g = master()
    f = pe.new_filter_layer("levels", g["id"], "f:L")
    assert pe.normalize([f, g])[0]["sub"] == "starless"


@pytest.mark.parametrize("level", [1, 2])
def test_the_levels_agree(level):
    got = compose([master()], level=level).rgb
    assert got.shape[:2] == Canvas(W, H).size(level)[::-1]


def _lum_part(**kw):
    part = pe.new_part("luminance")
    part.update(lo=40.0, hi=100.0, **kw)
    m = pe.new_mask()
    m["parts"] = [part]
    return m


def test_a_filter_s_luminance_mask_reads_the_picture_by_default_or_the_filter_or_another_filter():
    """As a luminosity mask of the reference photo editors is taken from the composite picture: on a filter of the
    combination a Luminance range reads the picture (the filters mixed); it may read the filter itself or another."""
    comp = Composer(Filters())
    g = pe.normalize([master()])[0]
    cv = comp.canvas_of([g])
    whole = (0, 0, W, H)
    picture = comp.layer_rgb(g, 0, whole, cv)
    sii = comp.layer_rgb(pe.sub_layer(g, "f:SII"), 0, whole, cv)
    ha = comp.layer_rgb(pe.sub_layer(g, "f:Ha"), 0, whole, cv)
    ctx = pe.Ctx(comp, 0, whole)
    for reads, img in ((None, picture), ("picture", picture), ("self", sii), ("f:Ha", ha)):
        kw = {} if reads is None else {"from": reads}
        lay = master(side_masks={"f:SII": _lum_part(**kw)})
        got = compose([lay], want=[(lay["id"], "f:SII")]).masks[(lay["id"], "f:SII")]
        want = comp._finish(comp._part_plane(_lum_part()["parts"][0], 0, whole, img, cv, ctx), _lum_part(), 0)
        assert np.abs(got - want).max() < 1e-6, reads
    saved = pe.save_layers([master(side_masks={"f:SII": _lum_part(**{"from": "f:Ha"})})])
    assert pe.load_layers(saved, {5})[0]["side_masks"]["f:SII"]["parts"][0]["from"] == "f:Ha"


def test_the_h_alpha_part_of_a_master_layer_s_mask_is_its_h_alpha_filter():
    p = planes()
    m = pe.new_mask()
    m["parts"] = [pe.new_part("halpha")]
    lay = master(side_masks={"f:SII": m})
    got = compose([lay], want=[(lay["id"], "f:SII")]).masks[(lay["id"], "f:SII")]
    q0, q1 = np.percentile(p["Ha"], [50, 97])
    t = np.clip((p["Ha"] - q0) / (q1 - q0), 0, 1)
    assert np.abs(got - t * t * (3 - 2 * t)).max() < 1e-5


class FiltersAndStars(Filters):
    """The fake engine with a stars side: one bright star (and its halo) on a dark sky, the filters without it."""

    def available(self, sid):
        return super().available(sid) or sid == ("stars", 4)

    def layer(self, lay, level, box, draft=False, stop=None):
        x0, y0, x1, y1 = box
        f = 2 ** level
        yy, xx = np.mgrid[0:H // f, 0:W // f] * f
        star = np.exp(-((xx - 10) ** 2 + (yy - 40) ** 2) / (2 * 3.0 ** 2)).astype(np.float32)
        return np.repeat(star[y0:y1, x0:x1, None], 3, -1)


def test_a_filter_s_mask_reading_the_picture_reads_it_without_its_stars():
    lay = master(stars_pid=4, side_masks={"f:SII": _lum_part()})
    got = compose([lay], FiltersAndStars(), want=[(lay["id"], "f:SII")])
    m = got.masks[(lay["id"], "f:SII")]
    assert got.rgb[40, 10].min() > 0.9                       # the star is in the picture, put back on the mix
    assert m[40, 10] == 0.0                                  # and not in the mask: the picture read is the starless mix
    assert m[22, 30] > 0.5                                   # the nebula's bright core is


# ---------------------------------------------------------------------------------------------- the Stars mask part
class StarField(Filters):
    """The fake engine with a stars side of two stars on black: a bright one (wide where the display shows it) and a
    faint one; the filters without them. Also an ordinary workflow (1) whose Stars block has the same stars."""
    STARS = ((10, 40, 1.0, 3.0), (50, 12, 0.08, 1.2))       # x, y, peak, Gaussian width (px)

    def available(self, sid):
        return super().available(sid) or sid in (("stars", 4), ("stars", 1), 1)

    def stars_image(self, level, box):
        f = 2 ** level
        yy, xx = (np.mgrid[0:H // f, 0:W // f] * f).astype(np.float32)
        s = np.zeros(yy.shape, np.float32)
        for x, y, a, w in self.STARS:
            s += a * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * w ** 2))
        x0, y0, x1, y1 = box
        return np.repeat(np.clip(s, 0, 1)[y0:y1, x0:x1, None], 3, -1).astype(np.float32)

    def layer(self, lay, level, box, draft=False, stop=None):
        if lay.get("part") == "stars" or lay.get("pid") == 4:
            return self.stars_image(level, box)
        x0, y0, x1, y1 = box
        return np.full((y1 - y0, x1 - x0, 3), 0.3, np.float32)


def _stars_mask(grow=1.0, feather=0.0, op="add"):
    m = pe.new_mask()
    part = pe.new_part("stars")
    part.update(grow=grow, feather=feather, op=op)
    m["parts"] = [part]
    return m


def _radius(m, x, y):
    """How far from (x, y) the mask holds 1, along the row."""
    row = m[y]
    r = 0
    while x + r + 1 < len(row) and row[x + r + 1] >= 0.5:
        r += 1
    return r


def test_the_stars_part_covers_each_star_to_its_extent_and_grows_with_it():
    lay = master(stars_pid=4, mask=_stars_mask())
    m = compose([lay], StarField(), want=[lay["id"]]).masks[lay["id"]]
    big, small = _radius(m, 10, 40), _radius(m, 50, 12)
    assert big > small >= 0 and m[40, 10] == 1.0 and m[30, 30] == 0.0
    # Grow: each star's radius times the setting: the brighter star grows more, in pixels
    l2 = master(stars_pid=4, mask=_stars_mask(grow=2.0))
    m2 = compose([l2], StarField(), want=[l2["id"]]).masks[l2["id"]]
    assert abs(_radius(m2, 10, 40) - 2 * big) <= 2 and _radius(m2, 10, 40) - big > _radius(m2, 50, 12) - small
    # Feather: no hard edge
    l3 = master(stars_pid=4, mask=_stars_mask(grow=2.0, feather=6.0))
    m3 = compose([l3], StarField(), want=[l3["id"]]).masks[l3["id"]]
    edge = m3[40, 10 + _radius(m2, 10, 40)]
    assert 0.05 < edge < 0.95


def test_the_part_s_feather_softens_its_stars_and_the_whole_mask_s_feather_comes_on_top():
    """The part's Feather softens the edge of its discs; the whole mask's Feather blurs the mask the parts made, after
    them (the reference editor's mask Feather over a feathered selection): two Gaussians one after the other, whose
    widths add in quadrature, so a part's Feather much under the whole mask's changes little (the tester's M27: 29 px
    over the whole mask)."""
    def mask(part_f, whole_f):
        m = _stars_mask(grow=2.0, feather=part_f)
        m["feather"] = whole_f
        lay = master(stars_pid=4, mask=m)
        return compose([lay], StarField(), want=[lay["id"]]).masks[lay["id"]]
    hard, soft = mask(0.0, 0.0), mask(4.0, 0.0)
    assert np.abs(soft - hard).max() > 0.3                     # the part's Feather alone: its edge soft
    both = mask(4.0, 6.0)
    assert np.allclose(both, pe.gauss(soft, 6.0), atol=1e-5)    # the whole mask's on top of it
    assert np.allclose(mask(0.0, 6.0), pe.gauss(hard, 6.0), atol=1e-5)
    # a Gaussian of 4 then of 6 is one of sqrt(4² + 6²) = 7.2: the part's 4 px moves the result by that much only
    assert np.abs(both - pe.gauss(hard, np.hypot(4.0, 6.0))).max() < 0.05


def test_the_grow_reaches_ten_times_a_star_s_radius_and_a_stored_part_is_kept_in_its_ranges():
    """Grow goes to 10 (a star's wing may be wider than three times its radius at the detection's threshold, the
    tester's M27 at Grow 3); a part read from a project is brought into its settings' ranges."""
    assert dict((k, (lo, hi)) for k, _n, lo, hi, _u in pe.PART_SETTINGS["stars"])["grow"] == (1, 10)
    l5 = master(stars_pid=4, mask=_stars_mask(grow=5.0))
    l1 = master(stars_pid=4, mask=_stars_mask())
    m5 = compose([l5], StarField(), want=[l5["id"]]).masks[l5["id"]]
    m1 = compose([l1], StarField(), want=[l1["id"]]).masks[l1["id"]]
    assert abs(_radius(m5, 10, 40) - 5 * _radius(m1, 10, 40)) <= 5     # (the test's canvas is 64 px wide)
    assert pe._clean_mask(_stars_mask(grow=10.0))["parts"][0]["grow"] == 10.0
    got = pe._clean_mask(_stars_mask(grow=40.0, feather=-3.0))["parts"][0]
    assert got["grow"] == 10.0 and got["feather"] == 0.0


def test_a_filter_s_mask_minus_the_stars_leaves_the_stars_out_and_reads_the_shared_stars_side():
    m = _lum_part()
    m["parts"] = m["parts"] + _stars_mask(grow=2.0, op="sub")["parts"]
    m["parts"][0].update(lo=0.0)                        # (the whole picture, the stars taken out)
    lay = master(stars_pid=4, side_masks={"f:SII": m})
    got = compose([lay], StarField(), want=[(lay["id"], "f:SII")]).masks[(lay["id"], "f:SII")]
    assert got[40, 10] == 0.0 and got[40, 10 + 3] == 0.0 and got[22, 30] == 1.0


def test_the_stars_part_protects_the_stars_of_an_ordinary_layer_from_a_filter():
    lay = pe.new_workflow_layer(1, "Full", True)
    f = pe.new_filter_layer("levels", lay["id"])
    f["params"].update(black=0.25)
    f["mask"] = dict(_stars_mask(grow=2.0), invert=True)        # the filter everywhere but on the stars
    got = compose([f, lay], StarField(), want=[f["id"]])
    m = got.masks[f["id"]]
    assert m[40, 10] == 0.0 and m[30, 30] == 1.0
    plain = compose([lay], StarField()).rgb
    assert np.allclose(got.rgb[40, 10], plain[40, 10]) and not np.allclose(got.rgb[30, 30], plain[30, 30])


def test_the_stars_part_says_nothing_where_the_layer_has_no_stars():
    lay = master(mask=_stars_mask())                     # no stars side
    m = compose([lay], StarField(), want=[lay["id"]]).masks[lay["id"]]
    assert m.max() == 0.0


def test_the_stars_part_is_found_once_in_the_stars_result_whatever_the_star_settings(monkeypatch):
    """A star setting (Strength, Star size, Star color) moved the stars the Stars part read, and each change found the
    stars again (on M27, 10 s of a 16 s release): the part now finds them once in the stars as their block computed
    them (Provider.linear_stars, none of Post's star settings), at full resolution, whatever the level shown; a star
    setting changes nothing in the mask and costs no detection, and every level draws the same discs."""
    found = []
    real = pe.find_stars
    monkeypatch.setattr(pe, "find_stars", lambda L: found.append(L.shape) or real(L))
    prov = StarField()
    cp = Composer(prov)
    lay = master(stars_pid=4, mask=_stars_mask(grow=2.0))
    m = cp.compose(pe.normalize(copy.deepcopy([lay])), 0, want=[lay["id"]]).masks[lay["id"]]
    for sliders in ({"star_strength": 3.6}, {"star_size": 0.5, "star_color": 3.0}):
        moved = dict(copy.deepcopy(lay), sliders=sliders)
        got = cp.compose(pe.normalize([moved]), 0, want=[moved["id"]]).masks[moved["id"]]
        assert np.array_equal(got, m)
    half = cp.compose(pe.normalize(copy.deepcopy([lay])), 1, want=[lay["id"]]).masks[lay["id"]]
    assert found == [(H, W)]                                     # once, at full resolution
    assert np.abs(half - m[::2, ::2]).mean() < 0.05              # the same discs, drawn at the level shown


def test_a_search_of_stars_never_holds_the_window_the_mask_waits_without_them(monkeypatch):
    """The first search of a result's stars froze the window a minute (Combine layers' Adjust, the preview empty,
    nothing said): it runs in a process of its own, waited for in a thread; meanwhile the mask is drawn without the
    stars and the picture says what is being done; when they are found the views are told and the mask has them."""
    import threading

    from astro_gpu.client import starsearch
    gate, told = threading.Event(), []
    monkeypatch.setattr(starsearch, "search", lambda L, work: gate.wait(30) and pe.find_stars(L))
    monkeypatch.setattr(pe, "STARS_FOUND_HOOKS", [lambda: told.append(1)])

    class Apart(StarField):
        search_apart = True

        def stars_id(self, lay):
            return f"apart-{id(self)}"
    prov = Apart()
    cp = Composer(prov)
    lay = master(stars_pid=4, mask=_stars_mask(grow=2.0))
    first = cp.compose(pe.normalize(copy.deepcopy([lay])), 0, want=[lay["id"]])
    assert first.masks[lay["id"]].max() == 0.0 and pe.STARS_SEARCHING in first.notes    # at once, without the stars
    gate.set()
    end = time.monotonic() + 30
    while not told and time.monotonic() < end:
        time.sleep(0.01)
    assert told and not pe._SEARCHING
    then = cp.compose(pe.normalize(copy.deepcopy([lay])), 0, want=[lay["id"]])
    assert then.masks[lay["id"]][40, 10] == 1.0 and pe.STARS_SEARCHING not in then.notes


def test_the_search_s_own_process_finds_what_the_function_finds(tmp_path):
    from astro_gpu.client import starsearch
    L = pe.lum(StarField().stars_image(0, (0, 0, W, H)))
    assert np.array_equal(starsearch.search(L, tmp_path), pe.find_stars(L)) and not list(tmp_path.iterdir())


def test_a_failed_search_of_stars_is_never_kept_as_no_stars_and_is_said(monkeypatch, tmp_path):
    """A failed search was saved as "no stars" (an empty file in cache/stars) and never searched again: a failure is
    never kept on disk; the picture says it, the mask has no stars, the session does not search again."""
    import threading

    from astro_gpu.client import starsearch
    calls, told = [], []

    def fails(L, work):
        calls.append(1)
        raise RuntimeError("the search's process ended")
    monkeypatch.setattr(starsearch, "search", fails)
    monkeypatch.setattr(pe, "STARS_FOUND_HOOKS", [lambda: told.append(1)])

    class Apart(StarField):
        search_apart = True
        stars_dir = tmp_path / "stars"

        def stars_id(self, lay):
            return f"failing-{id(self)}"
    cp = Composer(Apart())
    lay = master(stars_pid=4, mask=_stars_mask(grow=2.0))
    cp.compose(pe.normalize(copy.deepcopy([lay])), 0, want=[lay["id"]])
    end = time.monotonic() + 30
    while not told and time.monotonic() < end:
        time.sleep(0.01)
    got = cp.compose(pe.normalize(copy.deepcopy([lay])), 0, want=[lay["id"]])
    assert pe.STARS_FAILED in got.notes and pe.STARS_SEARCHING not in got.notes
    assert got.masks[lay["id"]].max() == 0.0 and calls == [1] and threading.active_count() >= 1
    assert not (tmp_path / "stars").exists() or not list((tmp_path / "stars").iterdir())


def test_every_child_s_mask_has_the_same_reads_choice():
    """The mask's Reads choice was offered on a filter's mask, not on the Stars child's: every child of a master layer
    reads, as its part says, the image itself (the Stars or Starless image's default, as before), the picture, or a
    filter."""
    def side_mask(frm=None):
        m = _lum_part()
        m["parts"][0].update(lo=50.0, hi=100.0)
        if frm is not None:
            m["parts"][0]["from"] = frm
        return m
    got = {}
    for frm in (None, "self", "picture", "f:Ha"):
        lay = master(stars_pid=4, side_masks={"stars": side_mask(frm)})
        got[frm] = compose([lay], StarField(), want=[(lay["id"], "stars")]).masks[(lay["id"], "stars")]
    assert np.array_equal(got[None], got["self"])                 # what it read before the choice
    assert not np.array_equal(got["self"], got["picture"]) and not np.array_equal(got["picture"], got["f:Ha"])


def test_a_filter_moved_between_children_is_named_one_way_both_ways():
    """A filter moved from S II to Stars was undone as "Layers reordered", back as "“Local contrast” changed": one
    wording, from where to where."""
    from astro_gpu.client.post3 import step_words
    lay = master(stars_pid=4)
    lay["name"] = "S-H-O · Combine layers"
    f = pe.new_filter_layer("levels", lay["id"], "f:SII")
    f["name"] = "Local contrast"
    moved = dict(f, sub="stars")
    assert step_words([f, lay], [moved, lay]) == "“Local contrast” moved from S II to Stars"
    assert step_words([moved, lay], [lay, f]) == "“Local contrast” moved from Stars to S II"
