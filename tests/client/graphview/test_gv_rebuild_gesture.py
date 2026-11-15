"""The graph drawn again under a gesture (a change coming from the panel while the button is held on a block): the
gesture is given up, nothing reads a block item the new drawing deleted."""
from __future__ import annotations

import sys

from PySide6.QtCore import QPoint

from tests.client.graphview.test_gv_page import B, page, pump  # noqa: F401 - the page's fixtures


def _errors():
    seen = []
    old, sys.excepthook = sys.excepthook, lambda *a: seen.append(a)
    return seen, old


def _stack(B):                                                      # noqa: N803
    B.tpl("colour_minimal")
    r = B.b.graph.rows[0]
    return r, next(b for b in r.blocks if B.b.graph.blocks[b].kind == "stacking")


def test_a_block_held_while_the_graph_is_drawn_again_is_let_go(B):  # noqa: N803
    r, stk = _stack(B)
    c = B.page.canvas
    pt = B.g.block_point(stk)
    seen, old = _errors()
    try:
        B.g.press(pt)
        assert c.gesture is not None and c.gesture["kind"] == "press"
        B.act("switch_off", block=r.blocks[-1])                      # a change from the panel: drawn again
        pump()
        for d in (10, 40, 80):
            B.g.move(pt + QPoint(d, d))
        B.g.release(pt + QPoint(80, 80))
        pump()
    finally:
        sys.excepthook = old
    assert not seen, seen[0][1]
    assert c.gesture is None and c.floater is None

