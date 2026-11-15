"""Synthetic stars beside its workflow, not in its trunk (the owner's decision of 14 November): Combine layers has three
inputs, Background, Stars and Synthetic stars (optional, greyed while empty); Synthetic stars reads the real stars alone
and gives the synthetic ones, drawn under the trunk on the stars' wire, its output to the third input. Three wires,
none over a block. For the engine nothing moves: the block stays in its workflow's chain (same definition, same Post
row in the one Combine layers layer). A graph saved with Synthetic stars in the trunk opens drawn as now, same
definition. On a project model (legacy.World: a scratch project, nothing computed)."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

from astro_gpu.graphcore import model as gm
from astro_gpu.graphcore.compile import compile_into
from astro_gpu.graphcore.model import SYNTH_PORT
from tests.client.graphview.legacy import World

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "graphcore"))
import invariants as inv  # noqa: E402


@pytest.fixture()
def W(qapp, tmp_path):
    w = World(tmp_path / "w", workflows=("Full",))
    yield w
    w.close()


def ins(W, j):
    return {w.dst[1]: (W.graph.blocks[w.src[0]].kind, w.src[1]) for w in W.graph.wires if w.dst[0] == j and w.active}


def drawn_clean(W):
    d = W.b.drawing()
    assert not inv.i1(d) and not inv.i2(W.graph, d) and not inv.i3(W.graph, d), (inv.i1(d), inv.i2(W.graph, d))
    return d


def add_synth(W):
    W.choose(W.one("stars"))
    W.g.double_click_palette("synth")
    W.settle()
    return W.one("synth")


def test_combine_layers_has_three_inputs_the_third_greyed_and_optional_without_synthetic_stars(W):
    cl = W.one("combine_layers")
    assert ins(W, cl) == {"Background": ("stars", "Starless"), "Stars": ("stars", "Stars only")}
    it = W.page.canvas.blocks[cl]
    names = [(i[0], i[3]) for i in it.d["inputs"]]
    assert names == [("Background", True), ("Stars", True), (SYNTH_PORT, False)]
    third = next(i for i in it.d["inputs"] if i[0] == SYNTH_PORT)
    assert third[4] == "#5d6470"                                   # greyed while empty
    assert "Input Synthetic stars: optional" in it.d["tip"]
    drawn_clean(W)


def test_synthetic_stars_drawn_under_the_trunk_feeds_the_third_input_and_leaves_in_one_undo_step(W):
    m = W.m
    pid = m.paths[0].pid
    stars = W.one("stars")
    defn = copy.deepcopy(m.p.to_dict()["workflows"])
    before = W.steps()
    syn = add_synth(W)
    row = next(r for r in W.graph.rows if syn in r.blocks)
    assert row.blocks[-2:] == [stars, syn]                         # for the engine: the same chain, after the stars
    assert [q.pid for q in m.paths] == [pid] and pid in m.synth_wanted() and len(W.kinds("combine_layers")) == 1
    cl = W.one("combine_layers")
    assert ins(W, cl) == {"Background": ("stars", "Starless"), "Stars": ("stars", "Stars only"),
                          SYNTH_PORT: ("synth", SYNTH_PORT)}
    d = drawn_clean(W)
    S, Y = d.boxes[stars], d.boxes[syn]
    assert Y.y >= S.bottom and Y.x > S.right and d.cells[syn] == (d.cells[stars][0] + 1, d.cells[stars][1] + 1)
    down = next(w for w in d.wires if w.id == f"c:{stars}>{syn}")  # from the stars' port down to its input
    assert down.points[0] == d.ports[(stars, "Stars only", "out")] and down.points[-1] == d.ports[(syn, "Stars", "in")]
    assert [i[0] for i in W.page.canvas.blocks[syn].d["inputs"]] == ["Stars"]
    assert [o[0] for o in W.page.canvas.blocks[syn].d["outputs"]] == [SYNTH_PORT]
    assert W.steps() == before + 1
    m.undo(graph=True)
    W.settle()
    assert not W.kinds("synth") and pid not in m.synth_wanted()
    assert ins(W, W.one("combine_layers")) == {"Background": ("stars", "Starless"), "Stars": ("stars", "Stars only")}
    assert m.p.to_dict()["workflows"] == defn


def test_the_gestures_on_the_third_input_and_on_the_side_block(W):
    m = W.m
    syn = add_synth(W)
    cl, stars = W.one("combine_layers"), W.one("stars")
    # its own wires pulled again: already there
    assert W.b.check_wire(syn, SYNTH_PORT, cl, SYNTH_PORT) == "this wire already exists"
    assert W.b.check_wire(stars, "Stars only", syn, "Stars") == "this wire already exists"
    # the third input takes the synthetic stars only; Stars the real ones only
    assert W.b.check_wire(stars, "Stars only", cl, SYNTH_PORT) == \
        "the Synthetic stars port takes the output of a Synthetic stars block"
    # the synthetic stars on Background or Stars: refused, and the refusal says what to do instead (Post's layer names)
    elsewhere = "Synthetic stars go into the third input. To use them alone, hide Real stars in Post."
    assert W.b.check_wire(syn, SYNTH_PORT, cl, "Stars") == elsewhere
    assert W.b.check_wire(syn, SYNTH_PORT, cl, "Background") == elsewhere
    from astro_gpu.client import i18n
    lang = i18n.language()
    try:
        i18n.set_language("fr")
        assert i18n._(elsewhere) == ("Les étoiles synthétiques vont dans la troisième entrée. Pour les utiliser seules, "
                                     "masquez Étoiles réelles dans Post.")
        assert i18n._("Real stars") in i18n._(elsewhere)             # Post's own name of the layer, in French too
    finally:
        i18n.set_language(lang)
    assert W.b.check_wire(stars, "Starless", syn, "Stars") == "Synthetic stars read the stars only of a Stars/background"
    # its wire to the third input follows the block: removing the wire says how
    w = next(x for x in W.graph.wires if x.dst == [cl, SYNTH_PORT])
    res = m.apply([("unwire", {"wire": w.id})])
    assert not res.ok and "delete it or switch it off" in str(res.reason)
    # switched off: the third input empties (Post has no synthetic stars row), and back
    assert m.apply([("switch_off", {"block": syn})]).ok
    W.settle()
    assert SYNTH_PORT not in ins(W, cl) and not m.synth_wanted()
    m.undo(graph=True)
    W.settle()
    assert ins(W, cl)[SYNTH_PORT] == ("synth", SYNTH_PORT)
    # deleted (its cross): the third input empties, the Post row goes, one undo brings both
    steps = W.steps()
    assert m.apply([("remove_block", {"block": syn})]).ok
    W.settle()
    assert not W.kinds("synth") and SYNTH_PORT not in ins(W, W.one("combine_layers")) and not m.synth_wanted()
    assert W.steps() == steps + 1
    m.undo(graph=True)
    W.settle()
    assert W.kinds("synth") and ins(W, W.one("combine_layers"))[SYNTH_PORT] == ("synth", SYNTH_PORT)
    drawn_clean(W)


def test_a_wire_pulled_from_other_stars_to_a_side_block_moves_it_onto_them():
    """The stars of another workflow pulled to a Synthetic stars' input: the block goes on them (one gesture), under
    their row; the S-H-O's per-filter stars keep feeding the combination."""
    from fixtures import Doc
    from test_points import ok, row
    doc = Doc.from_template("filters_complete")
    sii, ha = row(doc, "SII"), row(doc, "Ha")
    ok(doc, "add_block", kind="synth", at={"type": "insert", "row": sii.id, "after": sii.blocks[-1]})
    syn = row(doc, "SII").blocks[-1]
    ha_stars = ha.blocks[-1]
    ok(doc, "wire", src=ha_stars, src_port="Stars only", dst=syn, dst_port="Stars")
    p = doc.project
    assert row(doc, "Ha").blocks[-2:] == [ha_stars, syn] and p.blocks[row(doc, "SII").blocks[-1]].kind == "stars"
    d = __import__("astro_gpu.graphcore.layout", fromlist=["layout"]).layout(p, doc.view)
    assert d.cells[syn][0] == d.cells[ha_stars][0] + 1 and not inv.static(doc, d)


def test_a_graph_saved_with_synthetic_stars_in_the_trunk_opens_drawn_as_now_with_the_same_definition(W):
    """The previous drawing: the synthetic stars' wire into Stars (beside the real stars', not taken), the starless
    image read from the Synthetic stars. Opened: its stars on the third input, the starless from the Stars/background,
    the same compiled definition."""
    syn = add_synth(W)
    g = W.graph
    cl, stars = W.one("combine_layers"), W.one("stars")
    d = gm.to_dict(g)
    for w in d["wires"]:                                            # back to the previous drawing
        if w["dst"] == [cl, SYNTH_PORT]:
            w["src"], w["dst"], w["active"] = [syn, "Stars only"], [cl, "Stars"], False
        elif w["dst"] == [cl, "Background"]:
            w["src"] = [syn, "Starless"]
    old = gm.from_dict(copy.deepcopy(d))
    gm.check(old)
    want = {(tuple(w.src), tuple(w.dst), w.active) for w in g.wires}
    assert {(tuple(w.src), tuple(w.dst), w.active) for w in old.wires} == want
    from astro_gpu.core.project2 import Project2
    p1, p2 = Project2(name="t"), Project2(name="t")
    compile_into(g, p1)
    compile_into(old, p2)
    assert p1.to_dict()["workflows"] == p2.to_dict()["workflows"]
    assert stars in gm.Index(old).row[gm.Index(old).row_of[syn]].blocks


def test_synthetic_stars_to_post_on_its_own_is_a_layer_saved_reopened_and_compiled(W, tmp_path):
    """Synthetic stars may go to Post as a layer of its own, beside feeding its Combine layers (the owner's design: an
    image may feed several places): To Post makes its wire and its layer, kept by a save and a reopening, compiled
    as the engine reads it."""
    from astro_gpu.core import project2
    from astro_gpu.client.model import ProjectModel
    m = W.m
    syn = add_synth(W)
    post = W.one("post")
    assert m.apply([("send_to_post", {"block": syn})]).ok
    W.settle()
    g = W.graph
    assert any(w.src[0] == syn and w.dst[0] == post for w in g.wires)
    assert ins(W, W.one("combine_layers"))[SYNTH_PORT] == ("synth", SYNTH_PORT)      # still the third input too
    layers = m.post_layers()
    assert len(layers) >= 2, layers
    p2 = project2.Project2(name="c")
    compile_into(g, p2)                                    # the compiled definition takes it
    m.flush()
    W.settle()
    m2 = ProjectModel(project2.load(m.p.root), W.w, readonly=True)
    W.w.wait(60)
    g2 = m2._g if hasattr(m2, "_g") else None
    assert g2 is not None and any(w.src[0] == syn and w.dst[0] == post for w in g2.wires)
    assert len(m2.post_layers()) == len(layers)
    m2.close(save=False)
