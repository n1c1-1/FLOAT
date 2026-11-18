"""The graph's settings of a Combine layers block (client/postfusion.py): Inspect sets them on the block of the graph,
outside every result's key (nothing outdated, nothing to run, one step of the window's Undo); the layer of Post made for
the block takes a copy of them; a later change of them in Inspect leaves that layer as it is; Post's Graph settings
copies them into the layer again, one step of Post's Undo. A scratch project, nothing computed. The combination's
hues and intensities, the same rule on the owner's M27, are in test_post_fusion_window.py."""
from __future__ import annotations

import pytest

from tests.client.graphview.legacy import World
from tests.client.pagehelp import qapp  # noqa: F401

LOOK = {"blend": "Lighten", "stretch": "arcsinh", "sliders": {"saturation": 3.0, "stretch": 2.5}}
OTHER = {"blend": "Linear dodge (add)", "stretch": "auto", "sliders": {"sky": 1.0}}


@pytest.fixture()
def full(qapp, tmp_path):  # noqa: F811
    """A Full workflow with its Stars/background: the Combine layers Post shows by default, drawn on the graph, its
    layer not made yet (Post never opened)."""
    W = World(tmp_path / "w", workflows=("Full",))
    for q in W.m.paths:
        q.rendered = True
    W.settle()
    yield W
    W.close()


def _block(W):
    return next(b for b, x in W.graph.blocks.items() if x.kind == "combine_layers")


def _set(W, values):
    W.m.apply_change([W.m.paths[0]], {"kind": "layers_graph", "block": _block(W), "values": values})
    W.settle()


def test_the_graph_settings_are_the_block_s_one_step_and_never_a_run(full):
    W = full
    m, q = W.m, W.m.paths[0]
    bid = _block(W)
    assert m.layers_graph(bid) == {"blend": "Screen", "stretch": "auto", "sliders": {}}      # a new layer's
    keys = {st: m.stage_key(q, st) for st in m.all_stages(q)}
    layers = m.post_layers()
    n = len(m._undo)
    _set(W, LOOK)
    assert len(m._undo) == n + 1 and _block(W) == bid and m.layers_graph(bid) == LOOK
    assert {st: m.stage_key(q, st) for st in m.all_stages(q)} == keys                     # nothing outdated
    assert m.post_layers() == layers                                                        # Post's untouched
    c = m.change_cost({"kind": "layers_graph", "block": bid, "values": OTHER}, [q])
    assert not c.lost and not c.blocks                                                     # what Apply says: nothing
    m.undo()
    W.settle()
    assert m.layers_graph(bid)["blend"] == "Screen"


def test_the_layer_made_takes_a_copy_and_a_later_change_leaves_it(full):
    W = full
    m, q = W.m, W.m.paths[0]
    _set(W, LOOK)
    lid = m.default_post_layer(q.pid)                  # Post shows the block's layer for the first time: made
    lay = m.post_layer(lid)
    assert (lay["blend"], lay.get("stretch"), lay["sliders"]) == ("Lighten", "arcsinh", LOOK["sliders"])
    W.settle()
    _set(W, OTHER)                                     # Inspect again: the graph's settings only
    assert m.layers_graph(_block(W)) == OTHER
    lay = m.post_layer(lid)
    assert (lay["blend"], lay.get("stretch"), lay["sliders"]) == ("Lighten", "arcsinh", LOOK["sliders"])
    assert m.graph_copy(lay) == OTHER                  # what Graph settings would copy now


def test_post_s_graph_settings_copies_them_again_in_one_step(full):
    from astro_gpu.client.post3 import PostView
    W = full
    m, q = W.m, W.m.paths[0]
    lid = m.default_post_layer(q.pid)
    W.settle()
    _set(W, OTHER)
    pv = PostView(m, persist=False)
    try:
        lay = pv.layer(lid)
        assert lay["blend"] == "Screen" and lay["sliders"] == {}           # its own, made before
        n = len(pv.undo)
        assert pv.graph_settings(lid)
        lay = pv.layer(lid)
        assert (lay["blend"], lay.get("stretch"), lay["sliders"]) == ("Linear dodge (add)", None, {"sky": 1.0})
        assert len(pv.undo) == n + 1 and pv.next_words("undo").startswith("Graph settings on")
        assert not pv.graph_settings(lid) and len(pv.undo) == n + 1        # already them: no step
        pv.do_undo()
        lay = pv.layer(lid)
        assert lay["blend"] == "Screen" and lay["sliders"] == {} and len(pv.undo) == n
    finally:
        pv.shutdown()
        pv.deleteLater()


def test_the_button_says_what_it_copies(full):
    from PySide6.QtWidgets import QPushButton

    from astro_gpu.client import i18n
    from astro_gpu.client.post3 import PostView
    W = full
    m, q = W.m, W.m.paths[0]
    lid = m.default_post_layer(q.pid)
    W.settle()
    pv = PostView(m, persist=False)
    try:
        pv.sel = None
        pv.select(lid, "layer")
        b = next(x for x in pv.findChildren(QPushButton) if x.objectName() == "graph_settings")
        assert b.text() == "Graph settings" and "Ctrl+Z" in b.toolTip()
        i18n.set_language("fr")
        try:
            assert i18n._("Graph settings") == "Réglages du graphe" and i18n._(b.toolTip()) != b.toolTip()
        finally:
            i18n.set_language("en")
    finally:
        pv.shutdown()
        pv.deleteLater()
