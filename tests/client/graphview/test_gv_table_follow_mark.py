"""The Table's « ↑ » says a cell follows the Common row: never on a cell holding its own setting (a Stacking set to
Classic in its panel read « ↑ classic 3σ »), never before a cell with nothing to say (« ↑ — », a bare « ↑ »). A click
on a cell of the Common row never leaves the panel on another block (it showed « Session 1 »)."""
from PySide6.QtGui import QImage, QPainter

from astro_gpu.graphcore.operations import COMMON
from tests.client.graphview.test_gv_table import Bench


def _lines(b, ident):
    """The texts the cell tries when it is painted."""
    g = b.g
    c = g.cell(ident)
    assert c is not None, ident
    got = []
    real = g._first_fitting
    g._first_fitting = lambda p, x, y, texts, *a, **k: got.append(list(texts))
    img = QImage(400, 200, QImage.Format_ARGB32)
    p = QPainter(img)
    try:
        g._setting(p, c)
    finally:
        p.end()
        g._first_fitting = real
    return got[-1]


def _stacking(b):
    return next(x for x, blk in b.doc.graph.blocks.items() if blk.kind == "stacking")


def test_a_stacking_with_its_own_method_has_no_follow_mark(qapp):
    b = Bench(1920)
    rid = b.stacks()[0]["id"]
    ident = ("setting", rid, "stk")
    assert all(t.startswith("↑ ") for t in _lines(b, ident))            # it follows Common
    assert b.bridge.act("set_param", quiet=True, block=_stacking(b), stage="stack", name="method",
                        value="classic").ok
    b.g.after_model()
    lines = _lines(b, ident)
    assert lines[0].startswith("classic") and not any("↑" in t for t in lines)
    b.close()


def test_a_cell_with_nothing_to_say_shows_a_dash_alone(qapp):
    b = Bench(1920)
    rid = b.stacks()[0]["id"]
    assert b.bridge.act("remove_block", quiet=True, block=_stacking(b)).ok           # the stack has no Stacking
    b.g.after_model()
    assert _lines(b, ("setting", rid, "stk")) == ["—"]
    b.close()


def test_a_click_on_a_common_setting_cell_does_not_leave_another_block_in_the_panel(qapp):
    b = Bench(1920, host=True)
    sid = b.sessions()[0]
    b.click(("name", sid))
    assert b.bridge.selected == sid
    c = next(c for c in b.g.pinned + b.g.cells if c.kind == "setting" and c.id[1] == COMMON)
    b.click(c.id)
    assert b.bridge.selected != sid and b.bridge.file_cell is None
    b.close()
