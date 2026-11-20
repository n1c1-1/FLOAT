"""#181, step 4: the composition renders the layer tree of a Combine layers layer at any depth (client/postmix.py), the
nodes with an eye, a mask and filters of their own. Synthetic data, CPU only: a combination A (S-H-O) read by a combination
B (H-O) as its Hα, and B's other filter and its raw plane; the stars side made of the stars combination.

The reference is always the linear mix the run makes (B's Output = its matrix on its balanced filters, one of which reads
A's Output through its luminance), stretched with one factor from its statistics: at the defaults the render IS that
picture, at every depth."""
from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from astro_gpu.client import postengine as pe
from astro_gpu.client import postfusion as pf
from astro_gpu.client.postengine import Canvas, Composer, Provider
from astro_gpu.render.simple import SimpleRenderEngine, SimpleSettings, image_stats
from astro_gpu.stages.combine import HUBBLE_HUES, hue_palette

H, W = 48, 64
ROW = pf.channel_row("L")
MB = np.array([[1.0, 0.0], [0.0, 0.5], [0.0, 0.5]])          # B, H-O: Hα to red, O III to green and blue
MAPS_B = {"Ha": ({"Ha": 1.3, "OIII": 0.1}, 0.02), "OIII": ({"OIII": 0.8}, 0.0)}
A_ROLES, B_ROLES = ("SII", "Ha", "OIII"), ("Ha", "OIII")


def _blob(cx, cy, s, rng, noise=0.004):
    yy, xx = np.mgrid[0:H, 0:W]
    return (np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * s ** 2)) + rng.normal(0, noise, (H, W))).astype(np.float32)


def planes_a():
    rng = np.random.default_rng(3)
    return {"SII": 0.3 * _blob(20, 20, 8, rng), "Ha": _blob(30, 24, 10, rng), "OIII": 0.7 * _blob(36, 18, 7, rng)}


def raw_r():
    return 0.5 * _blob(26, 28, 9, np.random.default_rng(4))


def fusion_a(weights=None, hues=None):
    return {"stage": "combine_sho", "roles": list(A_ROLES), "names": {"SII": "S II", "Ha": "Hα", "OIII": "O III"},
            "matrix": hue_palette(hues or HUBBLE_HUES), "weights": dict({r: 1.0 for r in A_ROLES}, **(weights or {})),
            "balance": "sky", "stars": False, "neutral": False, "palette": ""}


def fusion_b(weights=None):
    return {"stage": "combine_ho", "roles": list(B_ROLES), "names": {"Ha": "Hα", "OIII": "O III"},
            "matrix": MB.tolist(), "weights": dict({r: 1.0 for r in B_ROLES}, **(weights or {})), "balance": "sky",
            "stars": False, "neutral": False, "palette": ""}


def tree(a=None, b=None, stars=None):
    """B's mix at "starless" with A as its Hα (path "f:Ha")."""
    a = a or fusion_a()
    node_a = {"kind": "mix", "block": 1, "pid": 6, "title": "S-H-O", "fusion": a,
              "roles": {r: {"kind": "image"} for r in A_ROLES}}
    node_b = {"kind": "mix", "block": 2, "pid": 5, "title": "H-O", "fusion": b or fusion_b(),
              "roles": {"Ha": node_a, "OIII": {"kind": "image"}}}
    sides = {"starless": node_b}
    if stars is not None:
        sides["stars"] = stars
    return {"kind": "layers", "block": 3, "sides": sides}


def _down(a, level, box):
    f = 2 ** level
    if f > 1:
        a = a[:H // f * f, :W // f * f].reshape(H // f, f, W // f, f).mean(axis=(1, 3))
    x0, y0, x1, y1 = box
    return a[y0:y1, x0:x1].astype(np.float32)


class Nested(Provider):
    """A combination A (pid 6) read by B (pid 5), as the run balanced them: B's Hα is its map of A's luminance."""
    version_of: dict = {}

    def __init__(self):
        self.a, self.r = planes_a(), raw_r()
        self.asked = []

    def available(self, sid):
        return isinstance(sid, tuple) and sid[0] == pe.FUSION

    def canvas(self, pids):
        return Canvas(W, H)

    def layer(self, lay, level, box, draft=False, stop=None):
        raise AssertionError("a tree's images are made by the composer")

    def x_a(self, fa=None):
        fa = fa or fusion_a()
        return np.tensordot(np.asarray(fa["matrix"]), np.stack([fa["weights"][r] * self.a[r] for r in A_ROLES]), 1)

    def read_ha(self):
        return np.tensordot(ROW, self.x_a(), 1)

    def balanced_b(self):
        return {"Ha": 1.3 * self.read_ha() + 0.1 * self.r + 0.02, "OIII": 0.8 * self.r}

    def fusion_planes(self, lay, level, box):
        self.asked.append(lay["pid"])
        src = self.a if lay["pid"] == 6 else self.balanced_b()
        out = {r: _down(a, level, box) for r, a in src.items()}
        return out, np.ones(next(iter(out.values())).shape, bool)

    def fusion_form(self, lay):
        return {"maps": MAPS_B, "channels": {"Ha": "L", "OIII": "L"}}

    def fusion_reads(self, lay, level, box):
        return {"Ha": _down(self.read_ha(), level, box), "OIII": _down(self.r, level, box)}


def group(nest, **kw):
    g = pe.new_workflow_layer(5, "H-O · Combine layers", True, part="combined")
    g.update(starless="output", stars_pid=None)
    g["fusion"] = nest["sides"]["starless"]["fusion"]
    g["nest"] = nest
    g.update(kw)
    return g


def compose(layers, prov=None, want=(), level=0, box=None, thumbs=False):
    return Composer(prov or Nested()).compose(pe.normalize(copy.deepcopy(layers)), level, box, want=want, thumbs=thumbs)


def engine_picture(x):
    """The stretched picture of a linear (3, h, w): what the engine's Output rendering gives (the colours of the linear
    image kept)."""
    st = image_stats(x.astype(np.float32), np.ones(x.shape[1:], bool))
    return np.moveaxis(SimpleRenderEngine.tone(torch.from_numpy(x.astype(np.float32)), st, SimpleSettings(keep_colours=1.0), True).numpy(), 0, -1)


def output_b(prov, wb=None, fa=None, p_ha=None):
    """B's Output as the run makes it: its matrix on its balanced, weighted filters."""
    bal = prov.balanced_b()
    w = dict({"Ha": 1.0, "OIII": 1.0}, **(wb or {}))
    return np.tensordot(MB, np.stack([w["Ha"] * bal["Ha"], w["OIII"] * bal["OIII"]]), 1)


# ------------------------------------------------------------------------------------------------ defaults == engine
@pytest.mark.parametrize("weights", [({}, {}), ({"Ha": 1.25, "SII": 0.7}, {"OIII": 1.4})])
def test_the_default_render_is_the_engine_s_picture_at_every_depth(weights):
    wa, wb = weights
    prov = Nested()
    fa, fb = fusion_a(wa), fusion_b(wb)

    class Prov(Nested):                                       # (A's weights are in its Output, as the run made it)
        def x_a(self, fa_=None):
            return Nested.x_a(self, fa)
    prov = Prov()
    got = compose([group(tree(fa, fb))], prov).rgb
    want = engine_picture(output_b(prov, wb))
    assert np.abs(got - want).max() < 3e-5
    # the leaves of the whole tree: B's O III (its plane and its raw one) and A's three filters
    assert pf.leaf_plan(tree(fa, fb)["sides"]["starless"], "starless", lambda n: Nested().fusion_form({})) == [
        ("starless", "OIII", "bal"), ("starless", "OIII", "read"), ("f:Ha", "SII", "bal"), ("f:Ha", "Ha", "bal"),
        ("f:Ha", "OIII", "bal")]


def test_the_render_at_the_screen_s_level_is_the_engine_s_picture_too():
    prov = Nested()
    got = compose([group(tree())], prov, level=1).rgb
    x = output_b(prov).astype(np.float32)
    x1 = np.stack([_down(c, 1, (0, 0, W // 2, H // 2)) for c in x])
    st = image_stats(x, np.ones(x.shape[1:], bool))              # (the statistics are the whole canvas's, kept apart)
    want = SimpleRenderEngine.tone(torch.from_numpy(x1), st, SimpleSettings(keep_colours=1.0), True).numpy()
    assert np.abs(got - np.moveaxis(want, 0, -1)).max() < 3e-5


def test_a_region_is_the_whole_s_pixels():
    layers = [group(tree())]
    whole = compose(layers).rgb
    box = (10, 6, 50, 40)
    part = compose(layers, box=box).rgb
    assert np.abs(part - whole[6:40, 10:50]).max() < 1e-6


# ------------------------------------------------------------------------------------------------ a nested level's settings
def test_the_inner_combination_s_hues_and_intensities_reach_the_picture_as_the_linear_mix_would():
    prov = Nested()
    base = compose([group(tree())], prov).rgb
    fa = fusion_a({"Ha": 1.5}, hues=[20.0, 100.0, 210.0])

    class Prov(Nested):
        def x_a(self, fa_=None):
            return Nested.x_a(self, fa)
    p2 = Prov()
    got = compose([group(tree(fa))], p2).rgb
    assert np.abs(got - base).max() > 0.02
    assert np.abs(got - engine_picture(output_b(p2))).max() < 3e-5


def test_an_eye_off_takes_the_node_s_contribution_away_the_other_filters_as_they_were():
    prov = Nested()
    off = group(tree(), shown={"starless": True, "stars": True, "f:Ha": False})
    got = compose([off], prov).rgb
    # the stretch is the whole mix's: B's Hα is out, its O III stays as the default stretched it
    x = output_b(prov)
    st = image_stats(x.astype(np.float32), np.ones((H, W), bool))
    F, c0 = SimpleRenderEngine.colour_factor(torch.from_numpy(x.astype(np.float32)), st, SimpleSettings(keep_colours=1.0))
    F = F.numpy()
    k = F * (prov.balanced_b()["OIII"] - c0)
    lost = 1.0 - MB.sum(1)                                    # (a row that does not sum to 1 takes the black point back)
    y = np.maximum(MB[:, 1][:, None, None] * k[None] - F[None] * c0 * lost[:, None, None], 0)
    want = SimpleRenderEngine.finish(torch.from_numpy(y.astype(np.float32)), SimpleSettings()).numpy()
    assert np.abs(got - np.moveaxis(want, 0, -1)).max() < 3e-5
    assert np.abs(got - compose([group(tree())], prov).rgb).max() > 0.05


def test_an_eye_off_deep_in_the_tree_leaves_that_filter_out_of_the_inner_picture_only():
    prov = Nested()
    got = compose([group(tree(), shown={"starless": True, "stars": True, "f:Ha/f:SII": False})], prov).rgb
    x = output_b(prov)
    st = image_stats(x.astype(np.float32), np.ones((H, W), bool))
    F, c0 = SimpleRenderEngine.colour_factor(torch.from_numpy(x.astype(np.float32)), st, SimpleSettings(keep_colours=1.0))
    F = F.numpy()
    ma = np.asarray(hue_palette(HUBBLE_HUES))
    k_sii = F * (prov.a["SII"] - c0)                          # A's S II, stretched, then taken out of A's picture
    d_read = float(ROW @ ma[:, 0])                            # what A's luminance gets of it
    k_oiii = F * (prov.balanced_b()["OIII"] - c0)
    p_ha = F * (prov.balanced_b()["Ha"] - c0) - 1.3 * d_read * k_sii
    lost = 1.0 - MB.sum(1)
    y = np.maximum(np.tensordot(MB, np.stack([p_ha, k_oiii]), 1) - F[None] * c0 * lost[:, None, None], 0)
    want = SimpleRenderEngine.finish(torch.from_numpy(y.astype(np.float32)), SimpleSettings()).numpy()
    assert np.abs(got - np.moveaxis(want, 0, -1)).max() < 3e-5


def test_a_filter_on_a_node_acts_on_its_image_before_its_parent_mixes_it():
    prov = Nested()
    lv = pe.new_filter_layer("levels", None, "f:Ha")
    lv["params"].update(black=0.1, gamma=0.7)
    g = group(tree())
    lv["target"] = g["id"]
    got = compose([lv, g], prov).rgb
    comp = Composer(prov)
    gg = pe.normalize([g])[0]
    cv = comp.canvas_of([gg])
    whole = (0, 0, W, H)
    y, vf, rec = comp._default(gg, "starless", 0, whole, cv, True)
    ha = pe.grey3(rec["f:Ha"].numpy())
    filtered = pe.lum(pe.FILTERS["levels"]["fn"](ha.copy(), lv["params"], pe.Ctx(comp, 0, whole)))
    ox = rec["f:OIII"].numpy()
    c0 = comp._c0(gg, "starless", cv)
    F = comp._pass(gg, "starless", 0, whole, cv, True)[2].numpy()
    lost = 1.0 - MB.sum(1)
    yy = np.maximum(np.tensordot(MB, np.stack([filtered, ox]), 1) - F[None] * c0 * lost[:, None, None], 0)
    want = SimpleRenderEngine.finish(torch.from_numpy(yy.astype(np.float32)), SimpleSettings()).numpy()
    assert np.abs(got - np.moveaxis(want, 0, -1)).max() < 1e-6
    assert np.abs(got - compose([g], prov).rgb).max() > 0.02


def test_a_filter_on_a_leaf_of_the_inner_combination_acts_before_the_inner_mix_and_changes_the_outer_picture():
    prov = Nested()
    g = group(tree())
    lv = pe.new_filter_layer("levels", g["id"], "f:Ha/f:Ha")
    lv["params"].update(black=0.1, gamma=0.7)
    got = compose([lv, g], prov).rgb
    assert np.abs(got - compose([g], prov).rgb).max() > 0.01
    # a node that is not in the tree: the filter goes to the nearest node above it
    far = pe.new_filter_layer("levels", g["id"], "f:Ha/f:Ha/f:Zzz")
    assert pe.normalize([far, g])[0]["sub"] == "f:Ha/f:Ha" or pe.normalize([far, g])[0]["sub"] == "f:Ha"


def test_a_mask_on_a_node_keeps_it_where_it_is_white_and_reads_what_its_part_says():
    prov = Nested()
    disc = pe.new_mask()
    part = pe.new_part("brush")
    part["strokes"] = [(30.0, 22.0, 8.0, 100.0, True)]
    disc["parts"] = [part]
    g = group(tree(), side_masks={"f:Ha": disc})
    got = compose([g], prov, want=[(g["id"], "f:Ha")])
    m = got.masks[(g["id"], "f:Ha")]
    plain = compose([group(tree())], prov).rgb
    off = compose([group(tree(), shown={"starless": True, "stars": True, "f:Ha": False})], prov).rgb
    assert np.abs(got.rgb[m == 1] - plain[m == 1]).max() < 1e-6
    assert np.abs(got.rgb[m == 0] - off[m == 0]).max() < 1e-6
    for frm in ("self", "picture", "f:OIII", "f:Ha/f:SII"):            # a luminance mask reads each image of the choice
        lum = pe.new_part("luminance")
        lum.update(lo=30.0, hi=100.0, **{"from": frm})
        mk = pe.new_mask()
        mk["parts"] = [lum]
        gg = group(tree(), side_masks={"f:Ha": mk})
        r = compose([gg], prov, want=[(gg["id"], "f:Ha")])
        assert r.masks[(gg["id"], "f:Ha")].shape == (H, W), frm


def test_a_mask_on_a_leaf_of_the_inner_combination_reads_the_inner_picture_and_its_siblings():
    prov = Nested()
    reads = {}
    for frm in ("self", "picture", "f:Ha/f:Ha", "f:Ha/f:OIII", "f:OIII"):
        lum = pe.new_part("luminance")
        lum.update(lo=8.0, hi=20.0, **{"from": frm})
        mk = pe.new_mask()
        mk["parts"] = [lum]
        g = group(tree(), side_masks={"f:Ha/f:SII": mk})
        reads[frm] = compose([g], prov, want=[(g["id"], "f:Ha/f:SII")]).masks[(g["id"], "f:Ha/f:SII")]
    assert not np.array_equal(reads["self"], reads["picture"])
    assert not np.array_equal(reads["picture"], reads["f:Ha/f:Ha"]) and not np.array_equal(reads["self"], reads["f:Ha/f:OIII"])
    assert np.array_equal(reads["f:OIII"], reads["self"])          # (not a sibling: the node itself, the default)


# ------------------------------------------------------------------------------------------------ caches and digests
def test_the_plan_foresees_every_digest_at_every_depth_and_a_second_render_makes_nothing_again():
    prov = Nested()
    g = group(tree(), shown={"starless": True, "stars": True, "f:Ha/f:SII": False})
    lv = pe.new_filter_layer("levels", g["id"], "f:Ha/f:Ha")
    lv["params"].update(gamma=0.8)
    top = pe.new_filter_layer("saturation", g["id"])
    top["params"].update(strength=20.0)
    layers = pe.normalize([top, lv, g])
    cp = Composer(prov)
    for level in (2, 1, 0):
        a = cp.compose(copy.deepcopy(layers), level, thumbs=True, track=True)
        assert cp.plan_errors == 0, level
        cp.applied.clear()
        b = cp.compose(copy.deepcopy(layers), level, thumbs=True, track=True)
        assert cp.plan_errors == 0 and not cp.applied, level
        assert np.array_equal(a.rgb, b.rgb)
    asked = len(prov.asked)
    top["params"]["strength"] = 25.0
    cp.compose(copy.deepcopy(pe.normalize([top, lv, g])), 1, track=True)
    assert len(prov.asked) == asked                                  # a light gesture reads no file again


def test_a_hue_of_the_inner_combination_reads_no_file_again(monkeypatch):
    monkeypatch.setattr(pe, "STATS_SIDE", 30)                   # (the statistics are measured on a reduced image)
    prov = Nested()
    cp = Composer(prov)
    cp.compose(pe.normalize([group(tree())]), 1, track=True)
    asked = len(prov.asked)
    fa = fusion_a(hues=[10.0, 90.0, 200.0])
    cp.compose(pe.normalize([group(tree(fa))]), 1, track=True)
    assert len(prov.asked) == asked


# ------------------------------------------------------------------------------------------------ saved keys, versions
def test_the_tree_is_never_saved_and_the_nodes_keys_make_the_round_trip():
    g = group(tree(), shown={"starless": True, "stars": True, "f:Ha": False, "f:Ha/f:SII": False},
              side_masks={"f:Ha/f:Ha": pe.new_mask()})
    f = pe.new_filter_layer("levels", g["id"], "f:Ha/f:OIII")
    saved = pe.save_layers(pe.normalize([f, g]))
    assert all("nest" not in d and "fusion" not in d for d in saved)
    back = pe.load_layers(copy.deepcopy(saved), {5})
    lay = next(x for x in back if x["kind"] == "workflow")
    assert lay["shown"]["f:Ha"] is False and lay["shown"]["f:Ha/f:SII"] is False and "f:Ha/f:Ha" in lay["side_masks"]
    assert next(x for x in back if x["kind"] == "filter")["sub"] == "f:Ha/f:OIII"


def test_a_filter_on_a_node_the_tree_lost_goes_to_the_nearest_node_above_it():
    g = group(tree())
    deep = pe.new_filter_layer("levels", g["id"], "f:Ha/f:SII")
    gone = pe.new_filter_layer("levels", g["id"], "f:Ha/f:Zzz")
    far = pe.new_filter_layer("levels", g["id"], "f:Mm/f:Ha")
    out = pe.normalize([deep, gone, far, g])
    got = {x["id"]: x["sub"] for x in out if x["kind"] == "filter"}
    assert got == {deep["id"]: "f:Ha/f:SII", gone["id"]: "f:Ha", far["id"]: "starless"}


def test_a_filter_without_a_tree_waits_for_it_as_saved():
    g = group(tree())
    g.pop("nest")
    f = pe.new_filter_layer("levels", g["id"], "f:Ha/f:SII")
    assert pe.normalize([f, g])[0]["sub"] == "f:Ha/f:SII"


def test_the_rows_are_nested_by_their_depth_with_their_filters_one_step_further_in():
    g = group(tree())
    f = pe.new_filter_layer("levels", g["id"], "f:Ha/f:Ha")
    top = pe.new_filter_layer("levels", g["id"], "starless")
    rows = pe.rows(pe.normalize([top, f, g]))
    shown = [(r.get("name"), d) for r, d in rows]
    assert shown == [("H-O · Combine layers", 0), ("Stars", 1), ("Starless image", 1), ("Levels", 2), ("Hα", 2),
                     ("S II", 3), ("Hα", 3), ("Levels", 4), ("O III", 3), ("O III", 2)]
    assert [r["sub"] for r, _d in rows if r["kind"] == "sub"] == [
        "stars", "starless", "f:Ha", "f:Ha/f:SII", "f:Ha/f:Ha", "f:Ha/f:OIII", "f:OIII"]
    collapsed = pe.rows(pe.normalize([top, f, g]), {(g["id"], "f:Ha")})
    assert [(r.get("name"), d) for r, d in collapsed][3:] == [("Levels", 2), ("Hα", 2), ("O III", 2)]


# ------------------------------------------------------------------------------------------------ the stars side
class WithStars(Nested):
    """The same with a stars combination C (pid 7, two filters mixed as their mean) whose Output the engine stretches by a
    curve of each channel (as palette stars are): its picture is the provider's image of the stars side."""
    C = ("R", "G")

    def __init__(self):
        super().__init__()
        rng = np.random.default_rng(9)
        self.c = {"R": 0.8 * _blob(12, 30, 2, rng, 0.0).clip(0), "G": 0.5 * _blob(50, 14, 2, rng, 0.0).clip(0)}

    def available(self, sid):
        return super().available(sid) or sid in (("stars", 7), 7)

    def x_c(self, fc=None):
        m = np.asarray((fc or fusion_c())["matrix"])
        return np.tensordot(m, np.stack([self.c[r] for r in self.C]), 1)

    def layer(self, lay, level, box, draft=False, stop=None):
        assert lay.get("stars_over") is not None, lay
        x = np.stack([_down(c, level, (0, 0, W >> level, H >> level)) for c in self.x_c()])
        e = 1.0 - np.exp(-6.0 * np.clip(x, 0, None) * np.array([1.0, 0.8, 0.6])[:, None, None])     # a curve per channel
        x0, y0, x1, y1 = box
        return np.moveaxis(e, 0, -1)[y0:y1, x0:x1].astype(np.float32)

    def fusion_planes(self, lay, level, box):
        if lay["pid"] == 7:
            out = {r: _down(a, level, box) for r, a in self.c.items()}
            return out, np.ones(next(iter(out.values())).shape, bool)
        return super().fusion_planes(lay, level, box)


def fusion_c(neutral=False):
    m = hue_palette(HUBBLE_HUES)
    mat = [[0.7, 0.3], [0.1, 0.9], [0.4, 0.6]] if not neutral else [[0.5, 0.5]] * 3
    return {"stage": "combine_ho", "roles": list(WithStars.C), "names": {"R": "R", "G": "G"}, "matrix": mat,
            "weights": {"R": 1.0, "G": 1.0}, "balance": "sky", "stars": True, "neutral": neutral, "palette": ""}


def stars_group(**kw):
    stars = {"kind": "mix", "block": 4, "pid": 7, "title": "S-H-O · Stars", "fusion": fusion_c(),
             "roles": {r: {"kind": "image"} for r in WithStars.C}}
    g = group(tree(stars=stars), **kw)
    g.update(stars_pid=7, stars_image="output")
    return g


def test_the_stars_side_made_of_a_combination_is_the_engine_s_picture_at_the_defaults():
    prov = WithStars()
    g = stars_group()
    got = compose([g], prov).rgb
    sides = compose([group(tree())], prov).rgb                        # (the starless side alone)
    e = prov.layer({"stars_over": 5}, 0, (0, 0, W, H))
    want = 1.0 - (1.0 - sides) * (1.0 - e)                              # Screen, the layer's blend
    assert np.abs(got - want).max() < 3e-5


def test_what_is_changed_in_the_stars_mix_is_added_to_the_engine_s_picture():
    prov = WithStars()
    base = compose([stars_group()], prov).rgb
    off = compose([stars_group(shown={"starless": True, "stars": True, "stars/f:R": False})], prov).rgb
    assert np.abs(off - base).max() > 0.01
    # the picture is the engine's where the filter's plane is zero (the second star's pixels)
    assert np.abs(off[14, 50] - base[14, 50]).max() < 0.05
    f = pe.new_filter_layer("levels", None, "stars/f:G")
    g = stars_group()
    f["target"] = g["id"]
    f["params"].update(black=0.0, gamma=0.6)
    got = compose([f, g], prov)
    assert np.abs(got.rgb - base).max() > 0.01
    cp = Composer(prov)
    cp.compose(pe.normalize([f, g]), 1, thumbs=True, track=True)
    assert cp.plan_errors == 0
    assert [r["sub"] for r, _d in pe.rows(pe.normalize([g])) if r["kind"] == "sub"] == [
        "stars", "stars/f:R", "stars/f:G", "starless", "f:Ha", "f:Ha/f:SII", "f:Ha/f:Ha", "f:Ha/f:OIII", "f:OIII"]


def test_a_draft_takes_its_statistics_from_its_own_picture_and_never_changes_the_exact_picture(monkeypatch):
    monkeypatch.setattr(pe, "STATS_SIDE", 30)                    # (the statistics are measured on a reduced image)
    layers = pe.normalize([group(tree())])
    prov = Nested()
    cp = Composer(prov)
    draft = cp.compose(copy.deepcopy(layers), 2, draft=True, track=True).rgb
    exact = cp.compose(copy.deepcopy(layers), 2, track=True).rgb
    cold = Composer(Nested()).compose(copy.deepcopy(layers), 2).rgb
    assert np.array_equal(exact, cold)                           # the exact picture is what it was without the draft
    assert draft.shape == exact.shape and np.abs(draft - exact).max() < 0.1
