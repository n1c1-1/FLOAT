"""Ctrl+Z with the cursor in a field: the field's own Undo first (what was typed in it), and when it has nothing to undo,
the window's Undo, as Edit > Undo. After Apply the notice suggests Ctrl+Z while the cursor is often still in a field:
the key did nothing there."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from tests.client.test_new_page_asks_to_save import _ready, qapp, win  # noqa: F401 - the fixtures
from tests.client.test_panel_typing import _target


def _pump(n=6):
    for _i in range(n):
        QApplication.processEvents()


def _focus_target(w):
    """The target field, the cursor in it, once the panel is built (it builds again after a change)."""
    for _i in range(20):
        w.worker.wait(2)
        _pump()
        ed = _target(w)
        if ed is not None:
            ed.setFocus()
            _pump()
            if QApplication.focusWidget() is ed:
                return ed
    raise AssertionError("the target field never took the focus")


def test_ctrl_z_in_a_field_with_nothing_to_undo_undoes_the_window(win):
    win.new_project()
    _ready(win)
    win.graph.set_layout("docked", "docked")
    _pump()
    rows = len(win.graph.bridge.graph.rows)
    assert win.graph.bridge.act("add_row", kind="session").ok
    _pump()
    assert len(win.graph.bridge.graph.rows) == rows + 1
    ed = _focus_target(win)
    before = ed.text()
    QTest.keyClicks(ed, "ab")                                  # the field's own Undo first
    QTest.keyClick(ed, Qt.Key_Z, Qt.ControlModifier)
    _pump()
    assert ed.text() != before + "ab" and len(win.graph.bridge.graph.rows) == rows + 1
    while ed.isUndoAvailable():
        QTest.keyClick(ed, Qt.Key_Z, Qt.ControlModifier)
    _pump()
    assert len(win.graph.bridge.graph.rows) == rows + 1
    ed = _focus_target(win)
    QTest.keyClick(ed, Qt.Key_Z, Qt.ControlModifier)    # nothing typed left: the window's
    _pump()
    assert len(win.graph.bridge.graph.rows) == rows
