"""A workflow shown in Post by its plain layer (its Output image), set there, then given a Stars/background in the
graph: Post shows it by a Combine layers layer from then on. The tutorials' film: the new layer started with Sky and
Faint tones at zero, the plain layer's settings, filters and mask gone with it (the crop was kept). The plain layer
becomes the Combine layers layer: what was set on it is the new layer's."""
from __future__ import annotations

import copy

import pytest

from tests.client.graphview.legacy import World
from tests.client.pagehelp import qapp  # noqa: F401

CHAIN = ["calibration", "registration", "stacking", "astrometry", "gradient", "color"]
TONES = {"sky": 1.5, "faint": 2.5, "contrast": 2.2, "saturation": 3.0, "stretch": 1.0}


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


def _set_plain_layer(m, pid, name=None):
    """The workflow's plain layer made real with tone settings, a base stretch, a mask and a filter above it; a second
    version of Post holding its own settings of it. -> the layer's id, the filter's id."""
    from astro_gpu.client import postengine as pe
    m._post_edit(lambda layers: m._filled(layers), scope="all")     # as its first setting in Post makes it real
    lid = next(x["id"] for x in m.post_version()["layers"] if x.get("kind") == "workflow" and x.get("pid") == pid)
    m.set_post_layer(lid, sliders=dict(TONES), stretch="arcsinh", opacity=80.0,
                     mask={"parts": [{"strokes": [[0.1, 0.1, 0.2]]}]}, **({"name": name} if name else {}))
    fid = []

    def add(layers):
        f = pe.new_filter_layer(next(iter(pe.FILTERS)), target=lid)
        fid.append(f["id"])
        layers.insert(next(i for i, x in enumerate(layers) if x["id"] == lid), f)
    m._post_edit(add, scope="active")
    m.post_version()["crop"] = [0.1, 0.1, 0.8, 0.8]
    return lid, fid[0]


def _layers(m, version=None):
    return copy.deepcopy((version or m.post_version()).get("layers") or [])


@pytest.mark.parametrize("typed", [None, "My picture"])
def test_the_plain_layers_settings_filters_and_mask_go_to_the_combine_layers_layer(world, typed):
    W = world(workflows=(), chains={"A": CHAIN})
    m = W.m
    pid = m.paths[0].pid
    lid, fid = _set_plain_layer(m, pid, typed)
    W.settle()
    before = _layers(m)
    plain = next(x for x in before if x["id"] == lid)
    assert [x["id"] for x in before] == [fid, lid] and plain["part"] is None and plain["name"] == (typed or "A")
    steps = W.steps()
    W.choose(W.graph.rows[0].blocks[-1])
    for kind in ("denoise", "stars"):
        W.page.palette.double_click(kind)
        W.settle()
    after = _layers(m)
    assert [(x["id"], x["kind"]) for x in after] == [(fid, "filter"), (lid, "workflow")], after
    new = after[1]
    assert new["part"] == "combined" and new["pid"] == pid and new["stars_pid"] == pid and new["starless"] == "starless"
    # the tone settings are the starless image's (the same keys); the stars' own start at their defaults
    assert new["sliders"] == TONES and new["stretch"] == "arcsinh"
    assert new["opacity"] == 80.0 and new["mask"] == plain["mask"] and new["visible"] == plain["visible"]
    assert new["name"] == (typed or "A · Combine layers")
    assert after[0] == before[0]                           # its filter, still above it and its own
    assert m.post_version()["crop"] == [0.1, 0.1, 0.8, 0.8]
    assert (pid, None) not in m.post_removed() and m.post_defaults() == []
    # the graph draws one Combine layers for it, and its node is that layer
    g = W.graph
    heads = [r.blocks[0] for r in g.rows if r.blocks and g.blocks[r.blocks[0]].kind == "combine_layers"]
    assert len(heads) == 1 and W.bm.node(heads[0]).lid == lid
    # one Undo per block added: the plain layer is back as it was
    assert W.steps() == steps + 2
    W.b.undo()
    W.settle()
    assert _layers(m) == before


def test_a_workflow_without_a_layer_of_its_own_gets_its_combine_layers_at_its_defaults(world):
    """Nothing set in Post: the workflow's Combine layers is the only layer of it, at its default settings, as before."""
    W = world(workflows=(), chains={"A": CHAIN})
    m = W.m
    pid = m.paths[0].pid
    assert _layers(m) == []
    W.choose(W.graph.rows[0].blocks[-1])
    for kind in ("denoise", "stars"):
        W.page.palette.double_click(kind)
        W.settle()
    mine = [(x["part"], x["name"], x["sliders"]) for x in _layers(m) if x.get("kind") == "workflow" and x["pid"] == pid]
    assert mine in ([], [("combined", "A · Combine layers", {})]) and (m.post_defaults() == []) == bool(mine)
