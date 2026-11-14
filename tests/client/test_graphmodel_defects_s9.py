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
