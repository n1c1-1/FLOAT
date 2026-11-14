"""What the old graph page did and the new page over the project model lost, model side (the swap's list of defects):
each test is the old behaviour, on the project model and the new page (legacy.World: a scratch project, nothing
computed)."""
from __future__ import annotations

import pytest

from tests.client.graphview.legacy import World
from tests.client.pagehelp import qapp  # noqa: F401


@pytest.fixture()
def world(qapp, tmp_path):  # noqa: F811
    made = []

    def make(**kw):
        w = World(tmp_path / f"w{len(made)}", **kw)
        made.append(w)
        return w
    yield make
    for w in made:
        w.close()


POST_PAGE = type("PostView", (), {"is_post_page": True})()
CURVES = {"id": 50, "kind": "filter", "filter": "curves", "name": "Curves", "visible": True, "opacity": 100.0,
          "mode": "Normal", "params": {}, "mask": None}


def test_post_saving_its_layers_after_a_block_sent_to_post_keeps_the_workflows_not_computed_yet(world):
    """GUI exploration 4 (s6a-11), with a block's image sent to Post from the graph (To Post: its own workflow, a tap):
    Post saving a composition (a filter added, nothing computed) takes nobody out; graph and Post agree."""
    W = world(workflows=("Classic stack",))
    m = W.m
    res = m.apply([("send_to_post", {"block": W.block("astrometrie")})])
    assert res.ok, res.reason
    W.settle()
    before = set(W.bm.to_post)
    assert len(before) == 2
    wires = len([w for w in W.graph.wires if W.graph.blocks[w.dst[0]].kind == "post"])
    m.set_post_layers([CURVES], POST_PAGE, scope="all")
    W.settle()
    assert not m.post_removed() and set(W.bm.to_post) == before
    assert len([w for w in W.graph.wires if W.graph.blocks[w.dst[0]].kind == "post"]) == wires


def test_a_stars_workflow_has_its_combine_layers_block_and_post_counts_every_layer(world):
    """s6a-2 (test_composition, test_graph47_client::test_recombine_is_shown_greyed_before_the_lights): with Stars, a
    Combine layers block for the layer Post gives the workflow, wired from Stars/background, its image to Post; numbered
    when Post holds several; the Post block's line counts Post's layers, an image layer of its own included; greyed
    before the lights."""
    from astro_gpu.client import postengine
    W = world(workflows=("Full", "Classic stack"))
    m = W.m
    full = W.view("Full")
    post = W.one("post")

    def line():
        W.page.refresh()
        W.settle(2)
        return W.page.canvas.blocks[post].d["summary"]
    cl = W.kinds("combine_layers")
    assert len(cl) == 1
    stars = W.one("stars")
    assert {tuple(w.src) for w in W.graph.wires if w.dst[0] == cl[0]} == {(stars, "Starless"), (stars, "Stars only")}
    assert any(w.src == [cl[0], "image"] and w.dst[0] == post for w in W.graph.wires)
    assert line() == "2 layers"
    m.add_post_layer(full.pid, "combined")
    W.settle()
    assert len(W.kinds("combine_layers")) == 2 and line() == "3 layers"
    assert sorted(n.num for n in m.block_map.nodes if n.key == "composition") == [1, 2]
    m.set_post_layers([postengine.new_image_layer("stack/abc", "Stack")] + m.post_layers(), scope="active")
    W.settle()
    assert line() == "4 layers", (m.block_map.post_layers, [(x.get("kind"), x.get("pid"), x.get("part")) for x in m.post_version()["layers"]])


def test_combine_layers_is_greyed_before_the_lights(world):
    W = world(workflows=("Full",), lights=False)
    cl = W.one("combine_layers")
    assert W.m.block_map.state(W.m.block_map.node(cl)) in ("pending", "skipped") or W.page.canvas.blocks[cl].d.get(
        "greyed") or W.page.canvas.blocks[cl].d.get("waits")
