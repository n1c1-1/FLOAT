"""Parity of the new graph page with today's (INTEGRATION §11 "C", the view's part): block badges and the reading veil
drawn as graph.py draws them, today's shortcuts, the greyed card's hint, the edge scroll of a drag, the blank page's
« From template » link opening the gallery, and the public signals and methods the window uses on today's page offered
with the same names (app.py read for them), on a graph held in memory."""
from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

from PySide6.QtCore import QPoint

from astro_gpu.graphcore.operations import json_of

APP = Path(__file__).resolve().parents[3] / "astro_gpu" / "client" / "app.py"


def _page(doc=None, size=(1400, 800)):
    from astro_gpu.client.graphview.document import MemoryDocument
    from astro_gpu.client.graphview.gestures import pump
    from astro_gpu.client.graphview.page import GraphPage
    p = GraphPage(doc if doc is not None else MemoryDocument())
    p.resize(*size)
    p.show()
    pump()
    return p


def _fixture(key):
    from astro_gpu.client.graphview.document import MemoryDocument
    from tests.graphcore import fixtures
    h = fixtures.Doc.from_template(key)
    return MemoryDocument(h.project, h.view)


# ======================================================================================================= badges
def test_badges_and_their_words_are_drawn_as_today(qapp):
    from astro_gpu.client.graphview.document import MemoryDocument
    from astro_gpu.client.graphview.gestures import pump

    class Badged(MemoryDocument):
        given = {}

        def badges(self, b):
            return self.given.get(self.graph.blocks[b].kind, {})
    h = _fixture("colour_minimal")
    doc = Badged(h.graph, h.view, states={b: "done" for b in h.graph.blocks})
    doc.given = {"session": {"broken": "a folder is not found", "warned": True, "nights": "8 nights · 28/06 to 21/07"},
                 "astrometry": {"missing": "the offline star catalogue is not installed"},
                 "stacking": {"fallback": ["drizzle off: too few frames"]}}
    p = _page(doc)
    by = {it.d["kind"]: it for it in p.canvas.blocks.values()}
    s, a, st = by["session"].d, by["astrometry"].d, by["stacking"].d
    assert s["broken"] and s["warned"] and "Cannot run: a folder is not found. Run is refused" in by["session"].toolTip()
    assert "8 nights · 28/06 to 21/07: their light frames" in by["session"].toolTip()
    assert a["missing"] and "Install it: Settings › Catalogs & astrometry" in by["astrometry"].toolTip()
    assert st["fallback"] and "Done with fallback" in by["stacking"].toolTip()
    for it in by.values():                             # painted with them (no exception, the scene draws)
        it.update()
    p.canvas.viewport().repaint()
    pump()
    p.close()


def test_the_reading_veil_and_an_unreadable_project(qapp):
    from astro_gpu.client.graphview.document import MemoryDocument

    class Reading(MemoryDocument):
        model = SimpleNamespace(ready=False, error="")
    doc = Reading()
    p = _page(doc)
    c = p.canvas
    assert c.wait.isVisible() and c.wait.text() == "Reading the project…" and c.veil.isVisible()
    doc.model.ready = True
    p.request_refresh()
    from astro_gpu.client.graphview.gestures import pump
    pump()
    assert not c.wait.isVisible() and not c.veil.isVisible()
    doc.model.error = "no project file"
    p.refresh()
    assert c.wait.text() == "Project unreadable: no project file" and not c.veil.isVisible()
    p.close()


# ======================================================================================================= keys
def test_todays_shortcuts(qapp):
    from astro_gpu.client.graphview.gestures import Gestures
    p = _page(_fixture("colour_minimal"))
    g, c, b = Gestures(p), p.canvas, p.bridge
    opened = []
    p.openBlock.connect(lambda: opened.append(1))
    k0 = c.zoom
    g.key("+", target=c)
    assert c.zoom > k0
    g.key("Ctrl+-", target=c)
    g.key("-", target=c)
    assert c.zoom < k0
    g.key("1", target=c)
    assert abs(c.zoom - 1.0) < 1e-6
    g.key("F", target=c)
    assert c.b_all.isChecked()
    g.key("Ctrl+0", target=c)
    stack = next(x for x, it in c.blocks.items() if it.d["kind"] == "gradient")
    g.click(stack)
    g.key("Return", target=c)
    assert opened
    before = json_of(b.graph)
    g.key("D", target=c)
    assert b.block(stack).off
    g.key("Ctrl+Z", target=c)
    assert json_of(b.graph) == before
    g.key("F2", target=c)
    assert c.name_edit is not None
    c.close_name()
    g.click(stack)
    rows = len(b.graph.rows)
    g.key("Ctrl+D", target=c)
    assert len(b.graph.rows) == rows + 1
    g.key("Esc", target=c)
    assert b.selected is None
    p.close()


# ======================================================================================================= palette
def test_a_card_refused_here_says_where_it_can_still_go(qapp):
    p = _page(_fixture("colour_minimal"))
    b = p.bridge
    r = b.graph.rows[0]
    b.choose(r.blocks[0])
    p.palette.refresh()
    p.palette.finish()
    card = p.palette.cards["stacking"]
    assert card.enabled and card.hint == "Drag it between blocks for a new branch"
    assert "One Stacking per workflow" in card.toolTip() and card.hint in card.toolTip()
    before = json_of(b.graph)
    p.palette.double_click("stacking")
    assert json_of(b.graph) == before
    assert p.canvas.toast.plain.endswith("· Drag it between blocks for a new branch")
    p.close()


# ======================================================================================================= edge scroll
def test_a_drag_held_near_the_edge_scrolls_the_view(qapp):
    from astro_gpu.client.graphview import canvas as cv
    from astro_gpu.client.graphview.gestures import pump
    p = _page(_fixture("sho_rgb"), size=(900, 600))
    c = p.canvas
    c.zoom_to(1.5)
    hb = c.horizontalScrollBar()
    hb.setValue(hb.minimum())
    c.gesture = dict(kind="palette", block="gradient")
    c._show_ghosts(p.bridge.places(kind="gradient"))
    edge = QPoint(c.viewport().width() - 4, c.viewport().height() // 2)
    c._edge_scroll(edge)
    pump(int(cv.EDGE_WAIT * 1000) // 20 + 30)
    import time
    t0 = time.perf_counter()
    while hb.value() == hb.minimum() and time.perf_counter() - t0 < 3:
        pump()
    assert hb.value() > hb.minimum()
    c.cancel_gesture()                                 # the drag ends: the scroll stops
    assert not c._edge_timer.isActive()
    v = hb.value()
    pump(20)
    assert hb.value() == v
    p.close()


# ======================================================================================================= templates
def test_the_blank_page_link_opens_the_gallery_and_a_template_makes_its_workflow(qapp):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    from astro_gpu.client.graphview.gestures import pump
    p = _page()
    asked = []
    p.templatesAsked.connect(lambda: asked.append(1))
    link = p.canvas.blank_link
    assert p.canvas.blank.isVisible() and link.text() == "From template"
    QTest.mouseClick(link, Qt.LeftButton)
    pump()
    assert asked and p.centre.currentWidget() is p.templates and p.templates.cards
    card = next(c for c in p.templates.cards if c.t["key"] == "colour_min")
    p.templates.chosen.emit(card.t)
    pump()
    assert p.centre.currentWidget() is p.canvas
    assert [r.name for r in p.bridge.graph.rows] == ["Colour minimal"]
    p.bridge.undo()
    assert not p.bridge.graph.rows                      # one step
    p.close()


# ======================================================================================================= the swap
def _app_uses():
    """The signals the window connects on the page it builds, and the members it calls on ``self.graph``."""
    src = APP.read_text(encoding="utf-8")
    built = src[src.index("    def graph_page("):]
    built = built[:built.index("\n    def ", 10)]                     # where the window builds the graph page
    signals = set(re.findall(r"\bpage\.(\w+)\.connect\(", built))
    members = set(re.findall(r"self\.graph\.(\w+)\b", src)) - {"graph"}
    inner = set(re.findall(r"self\.graph\.graph\.(\w+)", src)) | set(re.findall(
        r'getattr\(getattr\(self\.graph, "graph", None\), "(\w+)"', src))
    return signals, members, inner


def test_the_page_offers_what_the_window_uses_on_todays_page(qapp):
    from PySide6.QtCore import SignalInstance
    signals, members, inner = _app_uses()
    assert {"openBlock", "changed", "addSession"} <= signals and {"set_layout", "select"} <= members, (signals, members)
    p = _page(_fixture("colour_minimal"))
    for name in signals:
        assert isinstance(getattr(p, name, None), SignalInstance), name
    for name in members:
        assert hasattr(p, name), name
    for name in inner:
        assert hasattr(p.graph, name), name
    # without a project model they answer as an empty page does
    p.set_layout("docked", "hidden")
    assert not p.palette.isVisible()
    p.set_layout("docked", "docked")
    assert p.palette.isVisible()
    assert p.current_node() is None and not p.project_open()
    p.bridge.choose("project")
    assert p.project_open() and p.current_node().key == "project"
    assert p.place_images(["x.tif"]) is None and p.plus_session_pid is None
    changed = []
    p.changed.connect(lambda: changed.append(1))
    p.request_refresh()
    from astro_gpu.client.graphview.gestures import pump
    pump()
    assert changed
    blk = next(x for x, it in p.canvas.blocks.items() if it.d["kind"] == "gradient")
    p.bridge.choose(blk)
    p.remove_tool(None, None)
    assert blk not in p.bridge.graph.blocks
    p.rebuild()
    p.close()
