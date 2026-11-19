"""#181, step 2: Post's mixers and the settings per level (client/postfusion.py): the paths that name the layer tree's
nodes, the mixer every kind of the combination family has, the own settings of a nested mix (``tunes``), the graph's
settings copied node by node, the leaf planes' coefficients of a combination inside a combination, the stars'
combination described as a mix. No Qt."""
from __future__ import annotations

import numpy as np
import pytest

from astro_gpu.client import postfusion as pf
from astro_gpu.graphcore import post, rules
from astro_gpu.stages import combine as C
from astro_gpu.stages import recipes
from tests.client.test_post_nest_premises import _apply, _balanced
from tests.test_combine import _sky_field


def test_every_kind_of_the_combination_family_has_its_mixer_and_no_other_has_one():
    assert set(pf.MIXERS) == set(rules.GRAPH_SETTING_KINDS)
    assert {k for k in rules.ORDER if rules.nestable(k)} == set(pf.MIXERS)
    assert pf.MIXERS["combine_layers"].keys == pf.LAYERS_KEYS


def test_a_node_s_path_keeps_the_keys_of_the_starless_master_and_names_the_others():
    assert pf.child("", "Ha") == "f:Ha" and pf.child("starless", "Ha") == "f:Ha"
    assert pf.child("stars", "Ha") == "stars/f:Ha"
    assert pf.child("f:R", "Ha") == "f:R/f:Ha"
    assert pf.child("stars/f:R", "Ha") == "stars/f:R/f:Ha"
    for p in ("f:Ha", "stars/f:Ha", "f:R/f:Ha", "stars/f:R/f:Ha"):
        assert pf.child(pf.parent(p) if pf.parent(p) else "", p.rsplit(":", 1)[1]) == p
    assert pf.parent("f:Ha") == "starless" and pf.parent("stars") == "" and pf.parent("") == ""


def _tree():
    leaf = {"kind": "image"}
    inner = {"kind": "mix", "block": "i", "roles": {"Ha": leaf, "OIII": leaf}}
    return {"kind": "layers", "block": "cl", "sides": {
        "starless": {"kind": "mix", "block": "a", "roles": {"SII": leaf, "Ha": inner, "OIII": leaf}},
        "stars": {"kind": "mix", "block": "s", "roles": {"SII": leaf, "Ha": leaf, "OIII": leaf}}}}


def test_the_paths_of_a_tree_name_every_node_once():
    got = pf.node_paths(_tree())
    assert list(got) == ["", "starless", "f:SII", "f:Ha", "f:Ha/f:Ha", "f:Ha/f:OIII", "f:OIII", "stars", "stars/f:SII",
                         "stars/f:Ha", "stars/f:OIII"]
    assert got["f:Ha"]["block"] == "i"
    root = {"kind": "mix", "block": "m", "roles": {"Ha": {"kind": "image"}}}
    assert list(pf.node_paths(root)) == ["", "f:Ha"]


def test_the_own_settings_of_a_mix_are_the_layers_tune_for_its_own_mix_and_tunes_for_a_nested_one():
    lay: dict = {}
    t = {"hues": {"Ha": 40.0}, "weights": {"Ha": 1.5}}
    pf.put_tune(lay, "starless", t)
    pf.put_tune(lay, "f:Ha", {"hues": {"Ha": 10.0}, "weights": {}})
    pf.put_tune(lay, "stars", {"hues": {"Ha": 90.0}, "weights": {}})
    assert lay["tune"] == t and set(lay["tunes"]) == {"f:Ha", "stars"}
    assert pf.tune_at(lay, "") == t and pf.tune_at(lay, "stars")["hues"]["Ha"] == 90.0
    pf.put_tune(lay, "stars", None)
    pf.put_tune(lay, "f:Ha", None)
    assert "tunes" not in lay and lay["tune"] == t
    pf.put_tune(lay, "", None)
    assert lay == {}


def test_the_levels_the_graph_no_longer_has_lose_their_settings_and_nothing_is_moved():
    lay = {"tunes": {"f:Ha": {"hues": {}, "weights": {}}, "f:R": {"hues": {}, "weights": {}}}}
    assert not pf.drop_tunes(lay, ["f:Ha", "f:R"])
    assert pf.drop_tunes(lay, ["f:R"])
    assert list(lay["tunes"]) == ["f:R"]            # not moved to the level above
    assert pf.drop_tunes(lay, [])
    assert "tunes" not in lay


def test_the_graph_s_copy_is_put_node_by_node_and_what_it_does_not_name_is_left():
    lay = {"tunes": {"f:Ha": {"hues": {"Ha": 1.0}, "weights": {}}}}
    changed = pf.put_copy(lay, {"blend": "Lighten", "tune": {"hues": {"Ha": 5.0}, "weights": {}},
                                "tunes": {"stars": {"hues": {"Ha": 7.0}, "weights": {}}}})
    assert changed and lay["blend"] == "Lighten" and lay["tune"]["hues"]["Ha"] == 5.0
    assert set(lay["tunes"]) == {"f:Ha", "stars"}               # f:Ha was not named: left as it is
    assert not pf.put_copy(lay, {"tunes": {"stars": {"hues": {"Ha": 7.0}, "weights": {}}}})     # nothing new
    assert pf.put_copy(lay, {"tunes": {"f:Ha": None}}) and set(lay["tunes"]) == {"stars"}
    assert not pf.put_copy(lay, None)


def test_a_layer_saved_before_nesting_is_read_as_it_was():
    """The keys of today stay: a layer with a tune and no tunes has the starless master's settings and none nested."""
    lay = {"tune": {"hues": {"Ha": 5.0}, "weights": {}}}
    assert pf.tune_at(lay, "starless")["hues"]["Ha"] == 5.0 and pf.tune_at(lay, "f:Ha") is None


def test_only_a_recipe_composed_as_lightness_and_colour_is_plain_by_its_recipe():
    from astro_gpu.graphcore import combos
    assert pf.plain_reason(C.CombineSHO().spec(None)) == ""
    assert pf.plain_reason(C.CombineLRGB().spec(None)) == "lightness"
    assert pf.plain_reason(None) == "lightness"
    assert combos.spec is not None


def test_a_channel_read_is_a_row_of_the_images_colours():
    lum = np.asarray(C.LUM, np.float64)
    assert np.allclose(pf.channel_row("L"), lum) and np.allclose(pf.channel_row("G"), [0, 1, 0])
    assert np.allclose(pf.channel_row("GB"), [0, 0.5, 0.5]) and np.allclose(pf.channel_row("R"), [1, 0, 0])


def _node(cls, params, raw, children=None, channels=None):
    maps, out, bal, spec = _balanced(cls, params, raw)
    names = {r: r for r in recipes.roles(spec)}
    f = pf.describe(cls.name, spec, {"weights": dict(params.weights)}, names)
    return {"fusion": f, "maps": maps, "channels": channels or {}, "children": children or {}}, out, bal


def test_the_leaf_planes_of_a_combination_inside_a_combination_make_its_output_with_their_coefficients():
    """H-O (B) reads S-H-O (A)'s luminance as its Hα and a filter of its own as its OIII: B's Output is the sum of
    kappa * the leaf plane (A's three balanced filters, B's own) plus an offset; B's Hα is derived, its plane the
    same sum of A's filters."""
    rawA, _ = _sky_field({"SII": 100.0, "Ha": 113.0, "OIII": 126.0}, {"SII": 2.0, "Ha": 3.0, "OIII": 4.0})
    pA = C.SHOParams(weights={"Ha": 1.2})
    nodeA, outA, balA = _node(C.CombineSHO, pA, rawA)
    leaf, _ = _sky_field({"OIII": 140.0}, {"OIII": 2.5}, seed=11)
    rawB = {"Ha": C.channel_of(outA, "L"), "OIII": leaf["OIII"]}
    nodeB, outB, _balB = _node(C.CombineHO, C.HOParams(weights={"OIII": 0.8}), rawB, {"Ha": nodeA}, {"Ha": "L"})
    k, off = pf.kappas(nodeB)
    assert set(k) == {"f:Ha/f:SII", "f:Ha/f:Ha", "f:Ha/f:OIII", "f:OIII"}
    forms = pf.role_forms(nodeB)
    assert set(forms["Ha"][0]) == {"f:Ha/f:SII", "f:Ha/f:Ha", "f:Ha/f:OIII"} and forms["OIII"] == ({"f:OIII": 1.0}, 0.0)
    planes = {"f:Ha/" + pf.ROLE + r: _apply(nodeA["maps"], rawA)[r] for r in ("SII", "Ha", "OIII")}
    planes["f:OIII"] = _apply(nodeB["maps"], rawB)["OIII"]            # a leaf is its balanced plane
    got = sum(v[:, None, None] * planes[p][None] for p, v in k.items()) + off[:, None, None]
    assert np.abs(got - outB).max() / np.abs(outB).max() < 1e-5


def test_a_flat_combination_has_one_leaf_per_filter():
    raw, _ = _sky_field({"SII": 100.0, "Ha": 113.0, "OIII": 126.0}, {"SII": 2.0, "Ha": 3.0, "OIII": 4.0})
    node, out, _bal = _node(C.CombineSHO, C.SHOParams(weights={"SII": 1.4}), raw)
    k, off = pf.kappas(node)
    assert set(k) == {"f:SII", "f:Ha", "f:OIII"}
    lin = _apply(node["maps"], raw)
    got = sum(v[:, None, None] * lin[p[2:]][None] for p, v in k.items()) + off[:, None, None]
    assert np.abs(got - out).max() / np.abs(out).max() < 1e-5


@pytest.mark.parametrize("stars,neutral", [("neutral", True), ("palette", False)])
def test_the_stars_combination_is_described_as_a_mix_of_its_filters_balanced_on_the_sky(stars, neutral):
    spec = C.CombineSHO().spec(None)
    inputs = [{"role": r, "image": "stars"} for r in recipes.roles(spec)]
    f = pf.describe("combine_sho", spec, {"stars": stars, "balance": "signal"}, {}, inputs=inputs)
    assert f["stars"] is True and f["neutral"] is neutral and f["balance"] == "sky"
    if neutral:
        assert np.allclose(f["matrix"], 1 / 3)
    else:
        assert not np.allclose(f["matrix"], 1 / 3)
    own = pf.describe("combine_sho", spec, {"balance": "signal"}, {}, inputs=[{"role": "Ha", "image": "starless"}])
    assert own["stars"] is False and own["balance"] == "signal"
