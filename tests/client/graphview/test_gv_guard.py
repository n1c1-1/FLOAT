"""A gesture whose operation raises never reaches Qt: the document is taken back to the graph and view before it, a
toast says the gesture failed, and the error is logged through the client's logging."""
from __future__ import annotations

import logging
import sys

from astro_gpu.graphcore.operations import json_of


class Broken(Exception):
    pass


def _doc(fail="apply"):
    from astro_gpu.client.graphview.document import MemoryDocument

    class Raising(MemoryDocument):
        """A document whose apply makes its change, then raises (a core error after a half-made step)."""
        armed = False

        def apply(self, steps):
            if not self.armed:
                return super().apply(steps)
            if fail == "half":
                super().apply(steps)               # the step is made and pushed, then the error
            self.view["selected"] = "nowhere"
            raise Broken("the core failed")
    return Raising()


def _page(doc):
    from astro_gpu.client.graphview.gestures import pump
    from astro_gpu.client.graphview.page import GraphPage
    p = GraphPage(doc)
    p.resize(1200, 700)
    p.show()
    pump()
    return p


def _no_qt_exception():
    seen = []
    old, sys.excepthook = sys.excepthook, lambda *a: seen.append(a)
    return seen, old


def test_a_raising_operation_is_said_logged_and_changes_nothing(qapp, caplog):
    from astro_gpu.client.graphview.gestures import Gestures, pump
    for fail in ("apply", "half"):
        doc = _doc(fail)
        p = _page(doc)
        p.bridge.act("add_row", kind="session")
        pump()
        before, view = json_of(doc.graph), dict(doc.view)
        undo = len(doc._undo)
        doc.armed = True
        seen, old = _no_qt_exception()
        try:
            with caplog.at_level(logging.ERROR):
                Gestures(p).double_click_palette("image")
                pump()
                res = p.bridge.act("add_row", kind="image")
                pump()
        finally:
            sys.excepthook = old
        assert not seen, seen
        assert not res.ok
        assert json_of(doc.graph) == before, fail
        assert dict(doc.view) == view
        assert len(doc._undo) == undo, fail
        assert p.canvas.toast.isVisible() and "The gesture failed" in p.canvas.toast.plain
        assert any("raised" in r.getMessage() and r.exc_info for r in caplog.records), caplog.records
        caplog.clear()
        doc.armed = False
        assert p.bridge.act("add_row", kind="image").ok          # the page goes on
        p.close()


def test_an_undrawable_graph_is_taken_back(qapp, caplog, monkeypatch):
    from astro_gpu.client.graphview import document
    from astro_gpu.client.graphview.gestures import pump
    doc = document.MemoryDocument()
    p = _page(doc)
    before = json_of(doc.graph)
    real = document.Doc.layout

    def layout(self):
        if self.graph.rows:
            raise Broken("no drawing")
        return real(self)
    monkeypatch.setattr(document.Doc, "layout", layout)
    with caplog.at_level(logging.ERROR):
        res = p.bridge.act("add_row", kind="session")
        pump()
    assert not res.ok and json_of(doc.graph) == before
    assert "The gesture failed" in p.canvas.toast.plain
    assert any("cannot be drawn" in r.getMessage() for r in caplog.records)
    p.close()


def test_undo_that_raises_is_said(qapp, caplog):
    from astro_gpu.client.graphview.document import MemoryDocument
    from astro_gpu.client.graphview.gestures import pump

    class BadUndo(MemoryDocument):
        def undo(self):
            raise Broken("undo failed")
    doc = BadUndo()
    p = _page(doc)
    p.bridge.act("add_row", kind="session")
    with caplog.at_level(logging.ERROR):
        res = p.bridge.undo()
        pump()
    assert not res.ok and "The gesture failed" in p.canvas.toast.plain
    assert caplog.records
    p.close()
