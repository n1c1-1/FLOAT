"""A note being typed in the Comments field is part of the project's edits: Ctrl+S saves it, and quitting with a note
typed asks the save question as for any unsaved change. The field only sent its text when it lost the focus, and
the save took only one-line and numeric fields first: the note was dropped without a question."""
from __future__ import annotations

from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from astro_gpu.client.panel import CommentEdit
from tests.client.test_new_page_asks_to_save import _ready, qapp, win  # noqa: F401 - the fixtures


def _note(w):
    w.graph.set_layout("docked", "docked")
    QApplication.processEvents()
    return next(e for e in w.graph.panel.findChildren(CommentEdit) if e.isVisible())


def _type(ed, text):
    ed.setFocus()
    QApplication.processEvents()
    assert QApplication.focusWidget() is ed
    QTest.keyClicks(ed, text)


def test_ctrl_s_saves_the_note_being_typed(win, tmp_path):
    from astro_gpu.core import project2
    root = tmp_path / "ws" / "Noted"
    assert win.create_project(root.as_posix(), "Noted", "M 31")
    _ready(win)
    _type(_note(win), "clouds after 2 am")
    win.save()
    assert project2.load(root).description == "clouds after 2 am"


def test_leaving_with_a_note_typed_asks_to_save(win, tmp_path):
    root = tmp_path / "ws" / "Asked"
    assert win.create_project(root.as_posix(), "Asked", "M 31")
    _ready(win)
    assert not win.s.dirty
    _type(_note(win), "dew on the lens")
    win.close()                                          # the window's close box: asked, Cancel keeps the project
    QApplication.processEvents()
    assert win.asked == [win.project.root] and win.s.p.description == "dew on the lens"
