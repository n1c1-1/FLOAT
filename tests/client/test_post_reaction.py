"""What Post says of its own reaction time: the banner names the filter being made and what it is on; a filter's row says
what it took on the screen, measured; a long release whose main cost is a heavy filter above the setting changed
gives a tip once, in the status line, with a link that remembers "never again"; a new filter goes just above the
chosen one, or at its family's place among the chosen layer's filters."""
from __future__ import annotations

import time

from astro_gpu.client import i18n, post3, prefs
from astro_gpu.client import postengine as pe
from astro_gpu.client.services.render import Result
from tests.client.test_post import post, preps, qapp  # noqa: F401  (fixtures by name)


def _layer(v):
    return next(x for x in v.layers if pe.is_pixel(x) and x.get("visible", True))


def test_the_banner_names_the_filter_being_made_and_its_layer_in_both_languages(post):  # noqa: F811
    v = post.view
    lay = _layer(v)
    v.select(lay["id"], "layer")
    fl = v.add_filter("denoise")
    v._on_working(v._main_seq, (fl["id"], lay["id"], None))
    assert v._working_text() == f"{post3.layer_name(fl)} · {post3.layer_name(lay)}"
    v._on_working(v._main_seq + 99, (lay["id"], None, None))     # another request's work: not said
    assert v._working_text().startswith(post3.layer_name(fl))
    try:
        i18n.set_language("fr")
        assert i18n._("Rendering: {what}") == "Rendu : {what}"
    finally:
        i18n.set_language("en")


def test_a_filter_s_row_says_its_measured_cost_on_the_screen(post):  # noqa: F811
    v = post.view
    lay = _layer(v)
    v.select(lay["id"], "layer")
    fl = v.layer(v.add_filter("saturation")["id"])
    fl["params"]["strength"] = 40.0
    v.request(final_only=True, force=True)
    assert post.settle()
    assert fl["id"] in v._costs
    row = next(r for r in v.rows if r.lay["id"] == fl["id"])
    assert " s on screen" in row.toolTip()
    try:
        i18n.set_language("fr")
        assert i18n._("{t} s on screen").format(t=i18n.number(0.3, 1)) == "0,3 s à l'écran"
    finally:
        i18n.set_language("en")


def test_a_long_release_mostly_a_heavy_filter_above_gives_one_tip_with_a_never_again(post, monkeypatch):  # noqa: F811
    v = post.view
    lay = _layer(v)
    v.select(lay["id"], "layer")
    heavy = v.add_filter("ml_denoise")
    light = v.add_filter("levels")
    said = []
    v.tipped.connect(lambda t, w: said.append((t, w)))
    monkeypatch.setattr(post3, "_HEAVY_TIP_SAID", False)
    monkeypatch.setattr(prefs, "get", lambda k, _g=prefs.get: False if k == "post_tip_heavy_off" else _g(k))
    res = Result(1, "half", 1, None, None, None)
    res.costs, res.applied = {heavy["id"]: 1.4, light["id"]: 0.1}, frozenset({heavy["id"], light["id"]})
    v._changed_lid = lay["id"]                                # a setting of the layer under them
    v._release_t0 = time.perf_counter() - 2.0
    v._screen_done(res)
    assert len(said) == 1 and post3.layer_name(heavy) in said[0][0] and said[0][1] == "Don't show again"
    v._release_t0 = time.perf_counter() - 2.0
    v._screen_done(res)
    assert len(said) == 1                                     # once a session
    monkeypatch.setattr(post3, "_HEAVY_TIP_SAID", False)
    v._release_t0, v._changed_lid = time.perf_counter() - 2.0, heavy["id"]   # the heavy filter's own setting: no tip
    v._screen_done(res)
    assert len(said) == 1
    stored = {}
    monkeypatch.setattr(prefs, "set", lambda k, val: stored.__setitem__(k, val))
    v.tip_off()
    assert stored == {"post_tip_heavy_off": True}


def test_a_new_filter_goes_above_the_chosen_one_or_at_its_family_s_place(post):  # noqa: F811
    v = post.view
    lay = _layer(v)

    def own():
        return [x.get("filter") for x in v.layers if x.get("target") == lay["id"]]      # top first
    v.select(lay["id"], "layer")
    sat = v.add_filter("saturation")
    v.select(lay["id"], "layer")
    v.add_filter("denoise")                                   # early: under the saturation
    v.select(lay["id"], "layer")
    v.add_filter("levels")                                    # tonal: between them
    assert own() == ["saturation", "levels", "denoise"]
    v.select(sat["id"], "layer")
    v.add_filter("denoise")                                   # a filter chosen: just above it, whatever its family
    assert own() == ["denoise", "saturation", "levels", "denoise"]
    saved = pe.save_layers(v.layers)                          # a composition saved keeps its order
    back = pe.load_layers(saved, {p.pid for p in post.model.paths})
    assert [x.get("filter") for x in back if x.get("target") == lay["id"]] == own()
