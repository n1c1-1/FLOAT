"""Redo in Post brings back every step undone, not the first only: after two Ctrl+Z, the second Ctrl+Y did nothing
("Nothing to redo"). A step redone goes back to the Undo list; the window read it there as a change made since the
Undo, over which nothing comes back. Through the window's shortcut, through Post's arrows, and the two mixed."""
from __future__ import annotations

import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from test_one_undo import _set_rbf, _settle, qapp, win  # noqa: F401

from astro_gpu.client.post3 import ContSlider

PLACES = (0.1, 0.45, 0.7)                # of the slider, as shares of its length


def _post(win):
    """Post shown on a layer with a Dehaze filter, its Amount moved three times with its slider: three steps.
    -> (the view, the value before, the three values)."""
    s = win.s
    s.add_post_layer(s.paths[0].pid, None)
    win.set_mode(2)
    _settle(win, 3)
    view = win.post.view
    view.add_filter("dehaze")
    _settle(win, 3)
    fl = next(x for x in view.layers if x.get("filter") == "dehaze")
    first = fl["params"]["amount"]
    values = []
    for at in PLACES:
        sl = next(c for c in view.psc.widget().findChildren(ContSlider) if c.name.text() == "Amount")
        sl.s.setValue(round(at * sl.s.maximum()))                     # a click on its rail: moved and let go
        _settle(win, 2)
        values.append(_amount(view))
    assert len({first, *values}) == 4 and len(view.undo) >= 3
    return view, first, values


def _amount(view):
    return next(x for x in view.layers if x.get("filter") == "dehaze")["params"]["amount"]


def _key(win, key):
    win.activateWindow()                                 # (the window's shortcuts are those of the active window)
    QApplication.processEvents()
    win.post.view.viewer.setFocus()
    QApplication.processEvents()
    QTest.keyClick(QApplication.focusWidget() or win, key, Qt.ControlModifier)
    _settle(win, 3)


WAYS = {"keys": (lambda w: _key(w, Qt.Key_Z), lambda w: _key(w, Qt.Key_Y)),
        "window": (lambda w: w.undo(), lambda w: w.redo()),
        "arrows": (lambda w: w.post.view.b_undo.click(), lambda w: w.post.view.b_redo.click())}


@pytest.mark.parametrize("undo, redo", [("window", "window"), ("keys", "keys"), ("arrows", "arrows"),
                                        ("arrows", "window"), ("window", "arrows")])
def test_three_undos_then_three_redos_bring_the_three_steps_back(win, undo, redo):
    view, first, VALUES = _post(win)
    for want in (VALUES[1], VALUES[0], first):
        WAYS[undo][0](win)
        _settle(win, 2)
        assert _amount(view) == want, (undo, want)
    for want in VALUES:
        WAYS[redo][1](win)
        _settle(win, 2)
        assert _amount(view) == want, (redo, want, win.status.text())
    assert not view.redo


def test_steps_of_the_graph_and_of_post_undone_together_all_come_back(win):
    """One order for the window: the Graph's step and Post's, undone, are redone one after the other."""
    _set_rbf(win)
    view, first, VALUES = _post(win)
    for _i in range(3):
        win.undo()
        _settle(win, 2)
    assert _amount(view) == first
    for want in VALUES:
        win.redo()
        _settle(win, 2)
        assert _amount(view) == want, win.status.text()


def test_redo_still_never_comes_back_over_a_change_made_since(win):
    view, first, VALUES = _post(win)
    win.undo()
    win.undo()
    _settle(win, 2)
    assert _amount(view) == VALUES[0]
    win.redo()
    _settle(win, 2)
    assert _amount(view) == VALUES[1]
    win.set_mode(0)
    _settle(win, 2)
    _set_rbf(win)                                                    # a change made since, in the Graph
    win.set_mode(2)
    _settle(win, 3)
    win.redo()
    assert win.status.text() == "Nothing to redo" and _amount(view) == VALUES[1]
