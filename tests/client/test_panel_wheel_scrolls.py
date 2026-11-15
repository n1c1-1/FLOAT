"""The mouse wheel over a block's settings scrolls the panel: a drop-down list, a spin box or a slider changes with the
wheel only when it has the keyboard focus (the wheel goes to what is under the cursor). A tester's Demosaicing went
from RCD to Bilinear, and a draft appeared, while he only scrolled the panel on a 1250 x 685 screen."""
from __future__ import annotations

import pytest
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtWidgets import QAbstractSpinBox, QApplication, QComboBox

from tests.client.graphview.legacy import World
from tests.client.pagehelp import qapp  # noqa: F401

ROW = ["calibration", "registration", "selection", "stacking", "astrometry", "gradient", "denoise", "stars"]


@pytest.fixture()
def world(qapp, tmp_path):  # noqa: F811
    W = World(tmp_path / "w", workflows=(), chains={"Session 1": ROW}, size=(1250, 685))
    yield W
    W.close()


def _wheel(w, steps=-1):
    ev = QWheelEvent(QPointF(5, 5), QPointF(w.mapToGlobal(QPoint(5, 5))), QPoint(0, 0), QPoint(0, 120 * steps),
                     Qt.NoButton, Qt.NoModifier, Qt.NoScrollPhase, False)
    QApplication.sendEvent(w, ev)
    QApplication.processEvents()


def test_the_wheel_scrolls_the_settings_panel_unless_the_editor_has_the_focus(world):
    W = world
    W.choose(W.block("calibration"))
    W.settle(6)
    pn = W.page.panel
    combos = [c for c in pn.findChildren(QComboBox) if c.isVisibleTo(pn) and c.isEnabled() and c.count() > 1]
    spins = [s for s in pn.findChildren(QAbstractSpinBox) if s.isVisibleTo(pn) and s.isEnabled()]
    assert combos and spins
    steps = len(W.m._undo)
    for ed in combos + spins:
        before = ed.currentIndex() if isinstance(ed, QComboBox) else ed.value()
        ed.clearFocus()
        _wheel(ed)
        _wheel(ed, 1)
        now = ed.currentIndex() if isinstance(ed, QComboBox) else ed.value()
        assert now == before, ed
        assert ed.focusPolicy() == Qt.StrongFocus                     # the wheel never gives it the focus
    W.settle(2)
    assert len(W.m._undo) == steps and not pn._draft_entries()       # no change, no draft
    c = next(c for c in combos if c.currentIndex() < c.count() - 1)
    c.setFocus()
    QApplication.processEvents()
    if c.hasFocus():                                                  # (a window without a screen may refuse it)
        i = c.currentIndex()
        _wheel(c)
        assert c.currentIndex() != i                                  # focused: the wheel changes it, as before
