"""The mouse wheel over Post's settings scrolls the panel: a slider or a list changes with the wheel only when it has
the keyboard focus (clicked). The pointer merely passing over a slider while the panel was scrolled changed its value,
by accident."""
from __future__ import annotations

from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtWidgets import QApplication, QComboBox, QSlider

from test_one_undo import _settle, qapp, win  # noqa: F401
from test_post_redo_all import _post

from astro_gpu.client.postengine import snapshot


def _wheel(w, steps=-1):
    ev = QWheelEvent(QPointF(5, 5), QPointF(w.mapToGlobal(QPoint(5, 5))), QPoint(0, 0), QPoint(0, 120 * steps),
                     Qt.NoButton, Qt.NoModifier, Qt.NoScrollPhase, False)
    QApplication.sendEvent(w, ev)
    QApplication.processEvents()


def _editors(view):
    pn = view.psc.widget()
    sliders = [s for s in pn.findChildren(QSlider) if s.isVisibleTo(pn) and s.isEnabled()]
    combos = [c for c in pn.findChildren(QComboBox) if c.isVisibleTo(pn) and c.isEnabled() and c.count() > 1]
    return sliders, combos


def _state(ed):
    return ed.currentIndex() if isinstance(ed, QComboBox) else ed.value()


def test_the_wheel_over_a_slider_or_a_list_of_post_scrolls_the_panel_unless_it_has_the_focus(win):
    view, _first, _values = _post(win)
    win.resize(1200, 420)                                            # a short window: the settings scroll
    win.activateWindow()
    _settle(win, 3)
    sliders, combos = _editors(view)
    assert len(sliders) >= 3 and combos
    bar = view.psc.verticalScrollBar()
    assert bar.maximum() > 0
    layers = snapshot(view.layers)
    steps = len(view.undo)
    for ed in sliders + combos:
        view.viewer.setFocus()
        QApplication.processEvents()
        bar.setValue(0)
        before = _state(ed)
        _wheel(ed)
        assert _state(ed) == before, ed
        assert bar.value() > 0, ed                                   # the panel scrolled
        _wheel(ed, 1)
        assert _state(ed) == before and ed.focusPolicy() == Qt.StrongFocus    # the wheel never gives it the focus
    _settle(win, 2)
    assert len(view.undo) == steps and view.layers == layers        # nothing changed, no step of Undo
    for ed in (sliders[-1], combos[0]):                              # clicked (the focus): the wheel is its own
        ed.setFocus()
        QApplication.processEvents()
        assert ed.hasFocus()
        before = _state(ed)
        _wheel(ed)
        if _state(ed) == before:                                     # (already at that end)
            _wheel(ed, 1)
        assert _state(ed) != before, ed
        sliders, combos = _editors(view)                             # (a list's choice may build the panel again)


def test_the_panel_built_again_keeps_the_rule(win):
    view, _first, _values = _post(win)
    view.add_filter("curves")                                        # another panel, with its own lists
    _settle(win, 3)
    view.select(next(x for x in view.layers if x.get("filter") == "dehaze")["id"], "layer")
    _settle(win, 3)
    sliders, _combos = _editors(view)
    view.viewer.setFocus()
    QApplication.processEvents()
    for ed in sliders:
        before = ed.value()
        _wheel(ed)
        assert ed.value() == before
