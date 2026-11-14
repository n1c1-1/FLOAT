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


@pytest.mark.parametrize("key,filters,stars", [("filters_full", ("SII", "Ha", "OIII"), "S-H-O · Stars"),
                                               ("sho_rgb", ("SII", "Ha", "OIII"), "RGB"),
                                               ("dual_x2", (), "Stars")])
def test_a_template_with_combinations_draws_its_combine_layers_and_post_counts_it(world, key, filters, stars):
    """Issue #165 as on main: the rows, the combinations, each titled by its workflow's name (never "S-H-O 2"), ONE
    Combine layers block over the combination's Output, its Stars side from the "· Stars" combination's Output or the
    star row's stars alone, its image to Post; Post holds one "<name> · Combine layers" layer (the facades test compares
    Post's composition with the old model's), and the Post block says so."""
    W = world(workflows=(), filters=filters, nights=0 if filters else 1)
    W.template(key)
    m, g = W.m, W.graph
    combos = W.kinds("combine")
    for b in combos:
        r = next(r for r in g.rows if r.blocks and r.blocks[0] == b)
        assert W.page.canvas.blocks[b].d["title"] == r.name
    cl = W.one("combine_layers")
    ins = {w.dst[1]: w for w in g.wires if w.dst[0] == cl}
    sho = next(r for r in g.rows if r.name == "S-H-O")
    assert ins["Background"].src == [sho.blocks[0], "image"]
    src = ins["Stars"].src
    row = next(r for r in g.rows if src[0] in r.blocks)
    assert row.name == stars and src[1] == ("image" if stars.endswith("· Stars") else "Stars only")
    assert any(w.src == [cl, "image"] and g.blocks[w.dst[0]].kind == "post" for w in g.wires)
    layers = [x for x in m.post_version()["layers"] if x.get("kind") == "workflow"]
    assert [x["name"] for x in layers] == ["S-H-O · Combine layers"]
    W.page.refresh()
    W.settle(2)
    assert W.page.canvas.blocks[W.one("post")].d["summary"] == "1 layer"


def test_duplicate_copies_the_workflows_review_pauses(world):
    """s6a-22 (test_graph::test_the_pause_is_stored_per_workflow_undone_copied_and_leaves_with_its_block): Ctrl+D (the
    block menu's Duplicate) makes a copy that keeps the workflow's review pauses."""
    W = world(workflows=("Full",))
    m = W.m
    q = m.paths[0]
    m.set_breakpoint([q], "empilement", True)
    m.set_breakpoint([q], "fond", True)
    want = list(m.p.workflow(q.pid).breakpoints)
    assert len(want) == 2
    row = next(r for r in W.graph.rows if r.wid == q.pid)
    assert W.b.act("duplicate_row", row=row.id).ok
    W.settle()
    copy_ = m.paths[-1]
    assert copy_.pid != q.pid and sorted(m.p.workflow(copy_.pid).breakpoints) == sorted(want)


def test_a_variant_left_without_a_block_of_its_own_goes_with_it_in_one_undo_step(world):
    """s6a-20 (test_graph47_client::test_control_round_a_branch_left_without_a_block_of_its_own_goes_with_it)."""
    W = world(workflows=(), chains={"Main": ["calibration", "registration", "selection", "stacking", "astrometry"]})
    m = W.m
    assert W.b.act("add_variant", block=W.one("astrometry")).ok
    W.settle()
    var = next(r for r in W.graph.rows if r.fork)
    res = m.apply([("add_block", {"kind": "color", "at": {"type": "insert", "row": var.id, "after": var.blocks[-1]}})])
    assert res.ok, res.reason
    W.settle()
    names = [q.name for q in m.paths]
    assert len(names) == 2
    color = W.one("color")
    steps = len(m._undo)
    res = m.apply([("remove_block", {"block": color})])
    assert res.ok, res.reason
    W.settle()
    assert [q.name for q in m.paths] == names[:1] and not any(r.fork for r in W.graph.rows)
    assert "with its branch" in W.b.say(res.toast)
    assert len(m._undo) == steps + 1
    m.undo(graph=True)
    W.settle()
    assert [q.name for q in m.paths] == names and W.kinds("color")


def test_deleting_a_stack_takes_the_session_only_it_read_and_a_combinations_input_is_kept(world):
    """s6a-10 (test_fusion_sessions_client::test_removing_a_flow_or_its_session_says_what_goes,
    ::test_a_fusion_input_keeps_its_workflow): Delete workflow takes the Session only it drew (none left unwired), said,
    one Ctrl+Z brings both back; a workflow a combination reads is kept, said with the reason, nothing changes."""
    W = world(workflows=(), filters=("Ha", "OIII"))
    W.template("filters_min")
    m = W.m
    oiii = next(r for r in W.graph.rows if r.name == "OIII")
    res = W.b.act("delete_row", row=oiii.id)
    assert res.ok, res.reason
    W.settle()
    assert [s.id for s in m.p.sessions] == [1] and "OIII" in W.b.say(res.toast) and "files kept" in W.b.say(res.toast)
    m.undo(graph=True)
    W.settle()
    assert [s.id for s in m.p.sessions] == [1, 2] and len(m.paths) == 2
    W2 = world(workflows=(), filters=("Ha", "OIII"))
    W2.template("filters_full")
    m2 = W2.m
    ha = next(r for r in W2.graph.rows if r.name == "Ha")
    n, paths = len(m2._undo), [q.name for q in m2.paths]
    res = W2.b.act("delete_row", row=ha.id)
    assert not res.ok and "kept" in W2.b.say(res.reason) and "Ha" in W2.b.say(res.reason)
    assert len(m2._undo) == n and [q.name for q in m2.paths] == paths
