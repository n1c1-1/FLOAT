"""Synthetic stars on the stars side of a Combine layers read from another workflow (the owner's decision of 14
November): after the "<name> · Stars" combination (S-H-O family) or after a star row's Stars/background (RGB stars of
S-H-O + RGB, Dual-band x2). Behind model.SYNTH_ON_STARS_SIDE (on since the engine runs it after a combination's Output and Post puts its row in
the layer of the stars-side workflow): off, the star row's block is drawn, its image wired to nothing; on, the block feeds the third input of the Combine
layers that takes those stars, drawn under the stars' row (in the Composition band under the combination's row)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from fixtures import Doc  # noqa: E402
from test_points import ok, row  # noqa: E402
from astro_gpu.graphcore import layout, model  # noqa: E402
from astro_gpu.graphcore.model import Index, SYNTH_PORT  # noqa: E402
from astro_gpu.graphcore.operations import try_op  # noqa: E402
import invariants as inv  # noqa: E402


@pytest.fixture()
def on(monkeypatch):
    monkeypatch.setattr(model, "SYNTH_ON_STARS_SIDE", True)


@pytest.fixture()
def off(monkeypatch):
    monkeypatch.setattr(model, "SYNTH_ON_STARS_SIDE", False)


def cl_ins(doc):
    p = doc.project
    j = next(b for b, x in p.blocks.items() if x.kind == "combine_layers")
    return j, {w.dst[1]: (p.blocks[w.src[0]].kind, w.src[1]) for w in p.wires if w.dst[0] == j and w.active}


def star_row(doc):
    doc_rows = {r.name: r for r in doc.project.rows}
    return doc_rows["RGB"]


def drawn_under(doc, syn):
    d = layout(doc.project, doc.view)
    assert not inv.static(doc, d), inv.static(doc, d)
    ix = Index(doc.project)
    a = ix.pred(syn)
    assert d.boxes[syn].y >= d.boxes[a].bottom and d.boxes[syn].x >= d.boxes[a].right
    down = next(w for w in d.wires if w.id == f"c:{a}>{syn}")
    assert down.points[-1] == d.ports[(syn, "Stars", "in")]
    return d


def test_off_the_star_rows_block_is_drawn_and_wired_to_nothing(off):
    doc = Doc.from_template("sho_rgb")
    r = star_row(doc)
    ok(doc, "add_block", kind="synth", at={"type": "insert", "row": r.id, "after": r.blocks[-1]})
    syn = star_row(doc).blocks[-1]
    assert SYNTH_PORT not in cl_ins(doc)[1]
    drawn_under(doc, syn)


def test_on_the_star_rows_block_feeds_the_third_input(on):
    doc = Doc.from_template("sho_rgb")
    r = star_row(doc)
    ok(doc, "add_block", kind="synth", at={"type": "insert", "row": r.id, "after": r.blocks[-1]})
    syn = star_row(doc).blocks[-1]
    j, ins = cl_ins(doc)
    assert ins == {"Background": ("combine", "image"), "Stars": ("stars", "Stars only"), SYNTH_PORT: ("synth", SYNTH_PORT)}
    drawn_under(doc, syn)
    ok(doc, "remove_block", block=syn)
    assert SYNTH_PORT not in cl_ins(doc)[1]


def test_on_after_the_stars_combination(on):
    doc = Doc.from_template("filters_complete")
    j, _ins = cl_ins(doc)
    combo = next(w.src[0] for w in doc.project.wires if w.dst == [j, "Stars"])
    rid = Index(doc.project).row_of[combo]
    at = {"type": "insert", "row": rid, "after": combo}
    assert try_op(doc.project, "add_block", {"kind": "synth", "at": at}) == ""
    ok(doc, "add_block", kind="synth", at=at)
    syn = Index(doc.project).row[rid].blocks[1]
    assert cl_ins(doc)[1][SYNTH_PORT] == ("synth", SYNTH_PORT)
    for folded in (False, True):
        doc.view["folded"] = folded
        drawn_under(doc, syn)
    ok(doc, "remove_block", block=syn)
    assert SYNTH_PORT not in cl_ins(doc)[1]


def test_the_layout_of_synthetic_stars_after_the_stars_combination_whatever_the_engine_says(on):
    """The drawing alone (the block put in the combination's row by hand): under the combination in the Composition
    band, its Combine layers after it, three wires, I1-I3, unfolded and folded."""
    from astro_gpu.graphcore.model import Wire, clone, new_block, new_id
    doc = Doc.from_template("filters_complete")
    j, _ins = cl_ins(doc)
    q = clone(doc.project)
    combo = next(w.src[0] for w in q.wires if w.dst == [j, "Stars"])
    r = Index(q).row[Index(q).row_of[combo]]
    syn = new_block(q, "synth").id
    r.blocks.insert(1, syn)
    q.wires.append(Wire(new_id(q, "w"), [syn, SYNTH_PORT], [j, SYNTH_PORT]))
    model.check(q)
    doc.project = q
    for folded in (False, True):
        doc.view["folded"] = folded
        d = drawn_under(doc, syn)
        assert d.boxes[j].x > d.boxes[syn].right


@pytest.mark.parametrize("switch", [False, True])
def test_synthetic_stars_never_goes_to_post_by_its_menu_nor_by_a_copy_of_its_row(monkeypatch, switch):
    """Its image goes only to a Combine layers' third input: "To Post" on it is refused with the wire's reason, and a
    row ending in it duplicated sends its last other block's image to Post (or none), never the Synthetic stars' (the
    window fuzz: a To Post and a Ctrl+D on such a row wired it to Post's layers)."""
    from test_points import refused_inert
    monkeypatch.setattr(model, "SYNTH_ON_STARS_SIDE", switch)
    for start, kind in (("sho_rgb", None), ("filters_complete", "combine")):
        doc = Doc.from_template(start)
        if kind is None:
            r = star_row(doc)
            at = {"type": "insert", "row": r.id, "after": r.blocks[-1]}
        else:
            j, _ins = cl_ins(doc)
            combo = next(w.src[0] for w in doc.project.wires if w.dst == [j, "Stars"])
            r = Index(doc.project).row[Index(doc.project).row_of[combo]]
            at = {"type": "insert", "row": r.id, "after": combo}
            if try_op(doc.project, "add_block", {"kind": "synth", "at": at}):
                continue
        ok(doc, "add_block", kind="synth", at=at)
        r = Index(doc.project).row[r.id]
        syn = next(b for b in r.blocks if doc.project.blocks[b].kind == "synth")
        assert "Combine layers" in refused_inert(doc, "send_to_post", block=syn)
        ok(doc, "duplicate_row", row=r.id)
        p = doc.project
        post = Index(p).post
        assert not [w for w in p.wires if p.blocks[w.src[0]].kind == "synth" and w.dst[0] == post]
        copy = p.rows[-1] if p.rows[-1].name.endswith("copy") else next(x for x in p.rows if x.name == r.name + " copy")
        last = [b for b in copy.blocks if p.blocks[b].kind != "synth"][-1]
        sent = any(w.src[0] == last and w.dst[0] == post for w in p.wires)
        assert sent == (p.blocks[last].kind not in model.JOIN_KINDS)
