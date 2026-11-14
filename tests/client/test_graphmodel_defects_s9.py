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


def test_a_stack_without_selection_stacks_every_frame_with_the_same_weight_and_says_so(world):
    """The owner's decision: no Selection block = every frame kept, the same weight (the engine's "All frames"); the
    Stacking's line and the deletion's toast say it; a Selection block keeps its own rule."""
    from astro_gpu.graphcore import api as gapi
    W = world(workflows=("Full",))
    m = W.m
    q = m.paths[0]
    assert (m.p.workflow(q.pid).params.get("stack") or {}).get("selection") != "all"
    res = m.apply([("remove_block", {"block": W.block("selection")})])
    assert res.ok and "every frame is stacked, same weight" in W.b.say(res.toast)
    assert m.p.workflow(q.pid).params["stack"]["selection"] == "all"
    assert "every frame is stacked, same weight" in gapi.summary(m.stored, W.one("stacking"))
    m.undo(graph=True)
    assert (m.p.workflow(q.pid).params.get("stack") or {}).get("selection") != "all"


def test_a_review_pause_is_never_set_on_selection_and_an_old_one_is_a_pause_after_registration(world):
    from astro_gpu.client.graphview.menu import menu
    from astro_gpu.graphcore.convert import Converter
    W = world(workflows=("Full",))
    m = W.m
    sel = W.block("selection")
    res = W.b.act("set_pause", block=sel)
    assert not res.ok and "set the pause on Registration" in W.b.say(res.reason) and not W.graph.blocks[sel].pause
    item = next(i for i in menu(W.page.bridge.doc, sel) if i[1] == "set_pause")
    assert item[3] is False and "Registration" in item[4]
    # an old project's pause on Selection: the same stop, on Registration, said once
    p = m.p.__class__.from_dict(m.p.to_dict(), m.p.root)
    p.workflows[0].breakpoints = ["selection"]
    c = Converter(p)
    g = c.convert()
    reg = next(b for b, x in g.blocks.items() if x.kind == "registration")
    assert g.blocks[reg].pause and not any(x.pause for x in g.blocks.values() if x.kind == "selection")
    assert sum("pause after Selection" in x for x in c.log) == 1


def test_a_review_pause_on_a_calibration_is_kept_and_one_on_a_workflow_with_its_own_list_too(world):
    """s6a-5 (test_graph::test_every_block_has_pause_here_for_review_and_the_dot_shows): Review pause on a night's
    Calibration (shared by the workflows) is kept in each of them, and taken away again; a workflow holding its own
    list of pauses takes a pause set on another of its blocks."""
    W = world(workflows=("Full", "Classic stack"))
    m = W.m
    cal = W.one("calibration")
    res = W.b.act("set_pause", block=cal)
    assert res.ok
    assert all("calibration" in w.breakpoints for w in m.p.workflows)
    assert W.b.act("set_pause", block=cal).ok
    assert not any("calibration" in w.breakpoints for w in m.p.workflows)
    full = W.view("Full")
    row = next(r for r in W.graph.rows if r.wid == full.pid)
    assert m.apply([("set_flow_pauses", {"flow": row.id, "pauses": ["stacking"]})]).ok
    assert W.b.act("set_pause", block=W.block("fond", full)).ok
    assert set(m.p.workflow(full.pid).breakpoints) >= {"stacking", "gradient"}


def test_the_color_block_of_a_mono_workflow_is_greyed_with_its_reason_and_so_is_its_card(world):
    """s6a-7 (test_graph47_client::test_a_color_block_of_a_mono_workflow_is_greyed_with_its_reason)."""
    W = world(workflows=("Full",))
    m = W.m
    doc = W.page.bridge.doc
    color = W.one("color")
    assert doc.greyed(color) == ""
    W.choose(W.block("empilement"))
    assert doc.palette_state("color")[0] or "colour" not in doc.palette_state("color")[1]
    m.set_session_filter(1, "L")
    W.settle()
    assert "colours" in doc.greyed(color)
    W.choose(W.block("empilement"))
    ok, why = doc.palette_state("color")
    assert not ok and "colours" in why


def test_a_second_night_joining_a_filter_stack_is_named_and_filtered_after_it(world):
    """s6a-9 (test_fusion_sessions_client::test_a_filter_s_second_night_is_named_and_drawn_with_its_filter): Ha's
    Session chosen, a Session placed: "Ha 2", filter Ha, said once ("“Ha 2” joins “Ha”"), its Ctrl+Z once."""
    W = world(workflows=(), filters=("Ha", "OIII"))
    W.template("filters_min")
    m = W.m
    ha = next(r for r in W.graph.rows if r.name == "Ha")
    res = W.b.act("add_night", row=ha.id)
    assert res.ok, res.reason
    W.settle()
    s = m.p.sessions[-1]
    assert (s.name, s.filter) == ("Ha 2", "Ha")
    said = W.b.say(res.toast)
    assert said == "“Ha 2” joins “Ha”" and "Ctrl+Z" not in said


def test_rows_changed_in_the_definition_still_take_a_combination(world):
    """s6b (DEFECTS: rows whose blocks were changed in the definition, then read again, refused a combination)."""
    from astro_gpu.core import workflow as wfl
    W = world(workflows=(), filters=("Ha", "OIII"))
    m = W.m
    chain = ["calibration", "registration", "selection", "stacking"]
    for f, sid in (("Ha", 1), ("OIII", 2)):
        m.add_graph_workflow(f, [wfl.node(b) for b in chain], sessions=[sid])
    for q in m.paths:
        m.p.set_blocks(q.pid, [wfl.node(b) for b in chain + ["gradient", "denoise", "stars"]])
    m._sync()
    assert m.stored and m.graph_rebuilds == 1
    v = m.add_fusion("H-O", "ho", [{"pid": q.pid, "role": q.name, "channel": "L"} for q in m.paths])
    assert v is not None and (v.source or {}).get("fusion")


def test_to_post_after_the_wire_was_taken_away_gives_it_back(world):
    """s6a-4 (test_graph47_client::test_control_round_a_port_pulled_again_or_after_its_wire_went_says_so_or_gives_it_back)."""
    W = world(workflows=("Classic stack",))
    m = W.m
    q = m.paths[0]
    post = W.one("post")
    wire = next(w for w in W.graph.wires if w.dst[0] == post)
    assert m.apply([("unwire", {"wire": wire.id})]).ok
    W.settle()
    assert (q.pid, None) in m.post_removed() and q.pid not in W.bm.to_post
    last = wire.src[0]
    res = W.b.act("send_to_post", block=last)
    assert res.ok, res.reason
    W.settle()
    assert (q.pid, None) not in m.post_removed() and q.pid in W.bm.to_post
    assert any(w.src[0] == last and w.dst[0] == post for w in W.graph.wires)
    assert [x.pid for x in m.paths] == [q.pid]


def test_a_workflow_without_lights_keeps_its_wire_to_post_and_to_post_says_lights_first(world):
    """s6a-3 (test_waiting_wire): no lights: its wire to Post drawn (waiting), no layer of Post yet; To Post on its last
    block says "lights first"."""
    W = world(workflows=("Classic stack",), lights=False)
    m = W.m
    q = m.paths[0]
    post = W.one("post")
    wires = [w for w in W.graph.wires if w.dst[0] == post]
    assert len(wires) == 1 and [x.pid for x in m.post_waiting()] == [q.pid]
    assert not m.post_version().get("layers")
    res = W.b.act("send_to_post", block=wires[0].src[0])
    assert not res.ok and W.b.say(res.reason) == "lights first"
