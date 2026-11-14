"""Synthetic stars in the layout (the owner's decision of 14 November): on the line under its row, after the block
whose stars it reads, its stars wire from that block's port; I1-I3 hold. The window fuzz's shape of 14 November: a
Synthetic stars switched off (absent for the engine) moved to the head of a variant: refused now (never a row's head);
a head left so by a deletion is still drawn on the line under its own row, with its fork wire."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from fixtures import Doc  # noqa: E402
from test_points import ok, row, refused_inert  # noqa: E402
from astro_gpu.graphcore import layout  # noqa: E402
from astro_gpu.graphcore.judge import NO_SYNTH_BRANCH  # noqa: E402
from astro_gpu.graphcore.model import Index  # noqa: E402
import invariants as inv  # noqa: E402


def under_its_row(doc, d):
    ix = Index(doc.project)
    for b, x in doc.project.blocks.items():
        if x.kind != "synth":
            continue
        own = next(i for i, (o, _y) in enumerate(d.lines) if o == ix.row_of[b])
        assert d.cells[b][0] == own + 1, (b, d.cells[b], own)
        a = ix.pred(b)
        if ix.row_of.get(a) == ix.row_of[b]:
            assert d.boxes[b].y >= d.boxes[a].bottom and d.boxes[b].x >= d.boxes[a].right
            down = next(w for w in d.wires if w.id == f"c:{a}>{b}")
            assert down.points[0] == d.ports[(a, "Stars only", "out")]
            assert down.points[-1] == d.ports[(b, "Stars", "in")]


def build():
    doc = Doc.from_template("colour_complete")
    r = row(doc, "Session 1")
    ok(doc, "add_block", kind="synth", at={"type": "insert", "row": r.id, "after": r.blocks[-1]})
    return doc


def test_synthetic_stars_and_its_variants_sit_under_their_rows():
    doc = build()
    r = row(doc, "Session 1")
    denoise = next(b for b in r.blocks if doc.project.blocks[b].kind == "denoise")
    ok(doc, "add_variant", block=denoise)                    # a variant copying Denoise, Stars/background, Synthetic
    for folded in (False, True):
        doc.view["folded"] = folded
        d = layout(doc.project, doc.view)
        assert not inv.static(doc, d), inv.static(doc, d)
        under_its_row(doc, d)


def test_switched_off_it_is_never_put_at_a_rows_head_and_a_head_left_so_is_drawn_under_its_row():
    doc = build()
    syn = row(doc, "Session 1").blocks[-1]
    stars = row(doc, "Session 1").blocks[-2]
    ok(doc, "switch_off", block=syn)
    ok(doc, "add_variant", block=stars)                      # a variant [Stars/background, Synthetic stars (off)]
    var = next(r for r in doc.project.rows if r.fork)
    assert refused_inert(doc, "move_block", block=syn, at={"type": "insert", "row": var.id, "after": None}) \
        == NO_SYNTH_BRANCH
    ok(doc, "remove_block", block=var.blocks[0])              # its Stars/background deleted: the switched-off head
    var = next(r for r in doc.project.rows if r.fork)
    assert doc.project.blocks[var.blocks[0]].kind == "synth"
    d = layout(doc.project, doc.view)
    assert not inv.static(doc, d), inv.static(doc, d)
    under_its_row(doc, d)
