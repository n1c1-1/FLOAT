"""Defects the window fuzz on the project model found in the graph page, each as the smallest scene that shows it:
a block made in a folded row is brought into view by its card, and a wire lit under the pointer (on top of the blocks)
never hides the output port it leaves from: a wire pulled from that port is pulled, accepted or refused with its
reason."""
from __future__ import annotations

from tests.client.graphview import test_gv_page as first

page, B = first.page, first.B                                       # the fixtures of the first part


def _out_wire(B, block):                                            # noqa: N803
    """The wire item leaving block's output, lit as under the pointer (it goes on top of the blocks)."""
    it = next(w for w in B.canvas.wires.values() if w.ends[0] == block)
    it.light(True)
    pt = B.g.port_point(block, "image", "out")
    assert B.canvas.item_at(pt) is it                                 # what the press meets first: the lit wire
    return it


def test_a_session_added_while_folded_shows_its_card_fully(B):
    B.tpl("colour_complete")
    B.g.display("folded")
    cv = B.canvas
    cv.zoom_to(2.5)
    cv.horizontalScrollBar().setValue(cv.horizontalScrollBar().maximum())
    cv.verticalScrollBar().setValue(cv.verticalScrollBar().minimum())
    first.pump()
    B.g.double_click_palette("session")
    first.pump()
    made = B.b.last_shown
    assert made and all(x not in cv.blocks for x in made)             # a block of a folded row: drawn as its card
    card = B.b.row_of(made[0]).id
    assert card in cv.blocks and B.on_screen(card)


def test_a_wire_pulled_from_a_port_under_a_lit_wire_is_made(B):
    B.tpl("filters_minimal")
    B.setup(("add_block", dict(kind="combine", at={"type": "comp"})))
    j = next(b for b in B.ids("combine") if not any(w.dst[0] == b for w in B.graph.wires))
    port = next(n for n, *_r in B.canvas.blocks[j].d["inputs"] if n.startswith("H"))
    end = B.row("Ha").blocks[-1]
    assert B.b.check_wire(end, "image", j, port) == ""
    _out_wire(B, end)
    before = B.js()
    B.g.pull_wire(end, "image", j, port)
    assert any(tuple(w.dst) == (j, port) and w.src[0] == end for w in B.graph.wires)
    assert B.canvas.selected_wire is None
    B.exact(before)


def test_a_refused_wire_from_a_port_under_a_lit_wire_says_why(B):
    B.tpl("colour_minimal")
    post = B.post()
    end = next(w.src[0] for w in B.post_wires())
    why = B.b.check_wire(end, "image", post, "layers")
    assert why
    _out_wire(B, end)
    before = B.js()
    B.g.pull_wire(end, "image", post, "layers")
    assert B.js() == before
    assert B.toast() == B.b.say(why)
