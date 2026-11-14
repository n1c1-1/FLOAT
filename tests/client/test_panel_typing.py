"""The settings panel never builds itself again under the user's fingers: a project change arriving while a field is
being typed in (the model warming up, a background reading) waits until the field is applied, so a target typed fast
keeps every letter; Enter applies it, and the catalogues find the whole name."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QLineEdit

from tests.client.test_new_page_asks_to_save import _ready, qapp, win  # noqa: F401 - the fixtures


def _target(w):
    return next((e for e in w.graph.panel.findChildren(QLineEdit) if e.isVisible() and e.placeholderText() in (
        "NGC 7000", w.s.target_state().name)), None)


def test_a_target_typed_fast_while_the_project_changes_keeps_every_letter(win):
    win.new_project()
    _ready(win)
    win.graph.set_layout("docked", "docked")
    QApplication.processEvents()
    ed = _target(win)
    assert ed is not None
    ed.setFocus()
    so_far = ""
    for ch in "bubble":
        QTest.keyClick(QApplication.focusWidget(), ch)
        so_far += ch
        win.s.pathsChanged.emit()                       # a change of the project between two keys
        QApplication.processEvents()
        assert QApplication.focusWidget() is ed and ed.text() == so_far
    assert ed.text() == "bubble" and win.s.p.target.name == ""
    QTest.keyClick(ed, Qt.Key_Return)
    for _i in range(5):
        QApplication.processEvents()
    assert win.s.p.target.name == "bubble"
    assert win.s.target_state().dot == "known"          # the Bubble Nebula: a letter lost said "Not in the catalogues"
