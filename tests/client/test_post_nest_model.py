"""#181, step 3: the model gives Post the layer tree of a layer (client/model.py, client/graphmodel.py): each node
described, the graph's settings copied node by node (once: what Inspect changes later never reaches the layer), and the
own settings of a level the graph loses dropped when the graph change is committed, given back by the graph's Undo.
A scratch project and the Filters full template, nothing computed."""
from __future__ import annotations

import copy

import pytest

from astro_gpu.client import postfusion as pf
from astro_gpu.graphcore.rules import GRAPH_SETTINGS
from tests.client.graphview.legacy import World
from tests.client.pagehelp import qapp  # noqa: F401

FULL = dict(key="filters_full", name="Filters full", kind="fusion", palette="S-H-O", filters=["SII", "Ha", "OIII"])
OPTS = dict(debruitage=True, etoiles="retirer", fond="simple")
HUES = {"hues": {"Ha": 20.0, "OIII": 200.0}, "weights": {"OIII": 1.5}}
HUES2 = {"hues": {"Ha": 60.0, "OIII": 180.0}, "weights": {}}


@pytest.fixture()
def world(qapp, tmp_path):  # noqa: F811
    """Filters full: the starless S-H-O and its "· Stars" twin under one Combine layers layer; then an H-O (Hα and O III
    of the same filters) wired as the Hα of the starless S-H-O: a combination inside a combination."""
    W = World(tmp_path / "w", workflows=())
    assert W.m.apply_template(FULL, "S-H-O", dict(OPTS)).ok
    W.settle()
    W.ids = _nest(W.m)
    W.settle()
    yield W
    W.close()


def _nest(m):
    g = m._g
    cl = next(b for b, x in g.blocks.items() if x.kind == "combine_layers")
    sho = next(w.src[0] for w in g.wires if w.dst == [cl, "Background"])
    ha = next(w.src for w in g.wires if w.dst == [sho, "Hα"])
    o3 = next(w.src for w in g.wires if w.dst == [sho, "O III"])
    assert m._change([("add_block", {"kind": "combine", "at": {"type": "comp"}})]).ok
    ho = [b for b, x in m._g.blocks.items() if x.kind == "combine"][-1]
    drop = next(w.id for w in m._g.wires if w.dst == [sho, "Hα"])
    assert m._change([("set_combination_type", {"block": ho, "type": "ho"}), ("unwire", {"wire": drop}),
                      ("wire", {"src": ha[0], "src_port": ha[1], "dst": ho, "dst_port": "Hα"}),
                      ("wire", {"src": o3[0], "src_port": "Starless", "dst": ho, "dst_port": "O III"}),
                      ("wire", {"src": ho, "src_port": "image", "dst": sho, "dst_port": "Hα"})]).ok
    return {"cl": cl, "sho": sho, "ho": ho, "ha": ha}


def layer(m):
    return next(x for x in m.post_layers() if x.get("part") == "combined")


def edit_layer(m, fn):
    lid = layer(m)["id"]

    def change(layers):
        fn(next(x for x in layers if x["id"] == lid))
    m._post_edit(change, None, scope="all")


def set_graph(m, block, value):
    r = m._change([("set_setting", {"block": block, "key": GRAPH_SETTINGS, "value": value})])
    assert r.ok, r.reason


def shape(n):
    k = n["kind"]
    if k == "layers":
        return ("layers", {s: shape(c) for s, c in n["sides"].items()})
    if k == "mix":
        return ("mix", n["title"], {r: shape(c) for r, c in n["roles"].items()})
    if k == "fixed":
        return ("fixed", n["combination"], n["by"], n["why"])
    return "image"


def test_the_layer_tree_has_the_nested_mix_the_stars_mix_and_their_descriptions(world):
    m = world.m
    t = m.post_tree(layer(m))
    assert shape(t) == ("layers", {
        "starless": ("mix", "S-H-O", {"SII": "image", "Ha": ("mix", "H-O", {"Ha": "image", "OIII": "image"}),
                                      "OIII": "image"}),
        "stars": ("mix", "S-H-O · Stars", {"SII": "image", "Ha": "image", "OIII": "image"})})
    assert list(pf.node_paths(t)) == ["", "starless", "f:SII", "f:Ha", "f:Ha/f:Ha", "f:Ha/f:OIII", "f:OIII", "stars",
                                      "stars/f:SII", "stars/f:Ha", "stars/f:OIII"]
    sl, st = t["sides"]["starless"], t["sides"]["stars"]
    assert sl["fusion"]["roles"] == ["SII", "Ha", "OIII"] and sl["fusion"]["stars"] is False
    assert st["fusion"]["stars"] is True and st["fusion"]["neutral"] is True and st["fusion"]["balance"] == "sky"
    inner = sl["roles"]["Ha"]
    assert inner["fusion"]["roles"] == ["Ha", "OIII"] and inner["pid"] not in (None, sl["pid"], st["pid"])


def test_a_combination_wired_after_a_stars_background_is_a_fixed_node_naming_the_block(world):
    m = world.m
    sho = world.ids["sho"]
    assert m._change([("add_block", {"kind": "stars", "at": {"type": "insert", "row": next(r.id for r in m._g.rows if sho in r.blocks),
                                                              "after": sho}})]).ok
    t = m.post_tree(dict(layer(m), starless="starless"))
    assert shape(t["sides"]["starless"]) == ("fixed", "S-H-O", "Stars/background", "after")


def test_the_copy_is_made_for_every_node_of_a_layer_made_now(world):
    m = world.m
    set_graph(m, world.ids["ho"], HUES)
    set_graph(m, world.ids["sho"], HUES2)
    lay = layer(m)
    got = m.graph_copy(lay)
    assert got[pf.TUNE] == HUES2 and got[pf.TUNES]["f:Ha"] == HUES and got[pf.TUNES]["stars"] is None
    assert got["blend"] == "Screen"


def test_a_nested_mix_copies_its_graph_settings_once_and_inspect_never_reaches_it_after(world):
    m = world.m
    lay = layer(m)
    assert "f:Ha" not in (lay.get(pf.TUNES) or {})              # nested after the layer was made: not copied yet
    set_graph(m, world.ids["ho"], HUES)
    assert m.node_copies(lay) == {pf.TUNES: {"f:Ha": HUES, "stars": None}}
    edit_layer(m, lambda x: m.with_node_copies(x))
    lay = layer(m)
    assert lay[pf.TUNES]["f:Ha"] == HUES
    assert m.node_copies(lay) is None and not m.with_node_copies(copy.deepcopy(lay))
    set_graph(m, world.ids["ho"], HUES2)                         # Inspect again: the graph's settings only
    lay = layer(m)
    assert lay[pf.TUNES]["f:Ha"] == HUES and m.node_copies(lay) is None
    assert m.graph_copy(lay)[pf.TUNES]["f:Ha"] == HUES2          # what Graph settings would copy now


def test_a_node_without_settings_of_its_own_is_copied_too_so_it_never_follows_the_graph_later(world):
    m = world.m
    edit_layer(m, lambda x: m.with_node_copies(x))
    lay = layer(m)
    assert lay[pf.TUNES]["f:Ha"] is None and lay[pf.TUNES]["stars"] is None
    set_graph(m, world.ids["ho"], HUES)                          # set after the copy: not followed
    lay = layer(m)
    assert lay[pf.TUNES]["f:Ha"] is None and m.node_copies(lay) is None


def test_the_layer_keeps_the_nested_settings_through_a_save_and_a_load(world):
    from astro_gpu.client import postengine as pe
    m = world.m
    set_graph(m, world.ids["ho"], HUES)
    edit_layer(m, lambda x: m.with_node_copies(x))
    lay = layer(m)
    back = pe.load_layers(pe.save_layers(pe.normalize(pe.load_layers([lay], {lay["pid"]}))), {lay["pid"]})[0]
    assert back[pf.TUNES] == lay[pf.TUNES]


def _unnest(m, ids):
    drop = next(w.id for w in m._g.wires if w.dst == [ids["sho"], "Hα"])
    assert m._change([("unwire", {"wire": drop}),
                      ("wire", {"src": ids["ha"][0], "src_port": ids["ha"][1], "dst": ids["sho"],
                                "dst_port": "Hα"})]).ok


def test_a_level_the_graph_loses_drops_its_settings_when_committed_and_the_graph_s_undo_gives_them_back(world):
    m = world.m
    set_graph(m, world.ids["ho"], HUES)
    edit_layer(m, lambda x: m.with_node_copies(x))
    mine = copy.deepcopy(layer(m)[pf.TUNES])
    assert mine["f:Ha"] == HUES
    _unnest(m, world.ids)                                        # the nested mix is no more
    after = layer(m).get(pf.TUNES) or {}
    assert "f:Ha" not in after and after == {k: v for k, v in mine.items() if k != "f:Ha"}   # dropped, not moved
    assert m.undo(graph=True)
    assert layer(m)[pf.TUNES] == mine                            # the graph's Undo: exactly as it was
    assert m.redo(graph=True)
    assert "f:Ha" not in (layer(m).get(pf.TUNES) or {})
    assert m.undo(graph=True) and layer(m)[pf.TUNES] == mine


def test_a_level_coming_back_starts_again_from_the_graph_s_settings(world):
    m = world.m
    set_graph(m, world.ids["ho"], HUES)
    edit_layer(m, lambda x: m.with_node_copies(x))
    _unnest(m, world.ids)
    set_graph(m, world.ids["ho"], HUES2)                         # while it is away, Inspect changes
    drop = next(w.id for w in m._g.wires if w.dst == [world.ids["sho"], "Hα"])
    ids = world.ids
    assert m._change([("unwire", {"wire": drop}), ("wire", {"src": ids["ho"], "src_port": "image",
                                                             "dst": ids["sho"], "dst_port": "Hα"})]).ok
    lay = layer(m)
    assert "f:Ha" not in (lay.get(pf.TUNES) or {})
    edit_layer(m, lambda x: m.with_node_copies(x))
    assert layer(m)[pf.TUNES]["f:Ha"] == HUES2                   # not the old HUES: it starts again from the graph
