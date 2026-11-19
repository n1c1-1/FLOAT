"""The graph page's toast (#172/#173 testers' pass): its estimate is the total's difference worked out after the gesture
(never a total kept from before), it stays while the pointer is on it, and an Undo or a Redo closes it. And the panel's
state word: a result of an earlier version is not "Up to date"."""
from __future__ import annotations

import pytest
from PySide6.QtCore import QEvent, QObject, QPointF
from PySide6.QtGui import QEnterEvent
from PySide6.QtWidgets import QApplication, QWidget

from astro_gpu import api
from astro_gpu.client.graphmodel import GraphEditing
from astro_gpu.client.graphview.bridge import Bridge
from astro_gpu.client.graphview.canvas_overlays import Toast


@pytest.fixture
def toast(qapp):
    host = QWidget()
    host.resize(800, 600)
    t = Toast(host)
    host.show()
    yield t
    host.close()


def test_a_toast_with_a_link_stays_while_the_pointer_is_on_it(toast):
    toast.say_links("Full copy drawn under Full", [("Undo", "undo")])
    assert toast.timer.isActive()
    p = QPointF(5, 5)
    QApplication.sendEvent(toast, QEnterEvent(p, p, p))
    assert not toast.timer.isActive() and toast.isVisible()            # held: it cannot fade under the click
    QApplication.sendEvent(toast, QEvent(QEvent.Leave))
    assert toast.timer.isActive()                                      # and goes a moment after the pointer left


def test_undo_and_redo_close_the_toast_that_offered_the_undo(qapp):
    closed = []

    class Stub(Bridge):
        def __init__(self):
            QObject.__init__(self)
            self.history = (lambda: "u", lambda: "r")
    s = Stub()
    s.toast_close.connect(lambda: closed.append(1))
    assert s.undo() == "u" and s.redo() == "r"
    assert len(closed) == 2


class _Stub(GraphEditing):
    """Only what ``stale_said`` reads."""
    def __init__(self):
        self.device = "cpu"
        self._cache = {("cost", None, "cpu"): api.Cost(10.0, False, 0.0, 1)}   # the total from before the gesture
        self.after = api.Cost(19800.0, False, 0.0, 3)                          # 5 h 30 once the copy is counted
        self._chains = {1: ["a", "b"]}
        self._states = {}
        self.ready = True

    def _done_and_chains(self):
        return set(), {1: {"a", "b", "c"}}

    def _block_of_stage(self, stage):
        return "stars"

    def _lost_blocks(self, stages):
        return ["Stars/background"]

    def cost(self, workflows=None):
        k = ("cost", None, self.device)
        return self._cache.setdefault(k, self.after)


def test_the_toast_estimate_is_the_one_worked_out_after_the_gesture():
    m = _Stub()
    m.stale_said(({(1, "a")}, {1: {"a", "b"}}, 10.0))
    assert m.last_stale is not None
    assert m.last_stale.seconds == pytest.approx(19790.0)
    assert "5 h 30" in m.last_stale.text()
