"""Parity of the new graph page with today's (PARITE 55-63, 76, 79, 103, 108, 121, 137, 139, 148, 150-171, 201): the
palette's three ways of showing and its arrow when hidden, today's settings panel hosted with its dock, the other
workflows dimmed, a Combine layers block opening its layer of Post, the run's progress over the blocks, the touchpad,
the wires' image names and the chosen block's side inputs, the zoom kept with the project, "+ N min to run".

A graph held in memory for the view; a project model (small FITS files on a temporary project, nothing written
elsewhere) for what reads the model: the panel, the layer of Post, the run overlay, the zoom in the settings, the
estimate."""
from __future__ import annotations

import time
from types import SimpleNamespace

import pytest
from PySide6.QtCore import QEvent, QObject, QPoint, QPointF, Qt, Signal
from PySide6.QtGui import QWheelEvent
from PySide6.QtWidgets import QApplication, QPushButton

from tests.client.graphview.test_gv_readings import M, _project


def _doc(key):
    from astro_gpu.client.graphview.document import MemoryDocument
    from tests.graphcore import fixtures
    h = fixtures.Doc.from_template(key)
    return MemoryDocument(h.project, h.view)


def _page(doc, size=(1400, 800)):
    from astro_gpu.client.graphview.gestures import pump
    from astro_gpu.client.graphview.page import GraphPage
    p = GraphPage(doc)
    p.resize(*size)
    p.show()
    pump()
    return p


def _wait(ms):
    from astro_gpu.client.graphview.gestures import pump
    end = time.monotonic() + ms / 1000
    while time.monotonic() < end:
        pump()
        time.sleep(0.01)
    pump()


# ======================================================================================================= palette
def test_the_palette_has_todays_three_ways_of_showing_and_an_arrow_when_hidden(qapp):
    from astro_gpu.client.graphview.style import PALETTE_FOLDED_W, PALETTE_W
    p = _page(_doc("colour_minimal"))
    pal = p.palette
    chosen, shown = [], []
    p.sideChosen.connect(lambda side, st: chosen.append((side, st)))
    p.sidesShown.connect(lambda: shown.append(1))
    # always open: today's width, the names and the group headers; the "⋯" button and its menu
    assert pal.width() == PALETTE_W and p.palette_slot.width() == PALETTE_W and pal.head.isVisible()
    assert pal.display_button.toolTip() == "Blocks display"
    m = pal.menu(run=False)
    assert [(a.text(), a.isChecked()) for a in m.actions()] == \
        [("Automatic", False), ("Always open", True), ("Always folded", False)]
    pal.choose("folded")
    assert chosen == [("left", "folded")]                      # the window keeps it, then gives it back
    # always folded: the icons only, the name heads the tooltip, no header
    p.set_layout("docked", "folded")
    _wait(0)
    card = pal.cards["gradient"]
    assert pal.width() == PALETTE_FOLDED_W and p.palette_slot.width() == PALETTE_FOLDED_W
    assert not card.title.isVisible() and not any(h.isVisible() for h in pal.headers) and not pal.head.isVisible()
    assert card.toolTip().startswith("Gradient\n")
    # automatic: the same strip, opened over the graph after the mouse rests on it, folded after it left
    p.set_layout("docked", "auto")
    assert pal.width() == PALETTE_FOLDED_W
    QApplication.sendEvent(pal, QEvent(QEvent.Enter))
    _wait(320)
    assert pal.width() == PALETTE_W and card.title.isVisible() and p.palette_slot.width() == PALETTE_FOLDED_W
    assert pal.geometry().right() > p.palette_slot.geometry().right()       # over the graph, which keeps its place
    QApplication.sendEvent(pal, QEvent(QEvent.Leave))
    _wait(420)
    assert pal.width() == PALETTE_FOLDED_W and not card.title.isVisible()
    # hidden (Tab): an arrow at the window's edge brings the sides back
    p.set_layout("docked", "hidden")
    assert not pal.isVisible() and not p.palette_slot.isVisible() and p.left_arrow.isVisible()
    assert p.left_arrow.toolTip() == "Show the blocks again"
    p.left_arrow.click()
    assert shown == [1]
    p.set_layout("docked", "docked")
    assert pal.isVisible() and pal.width() == PALETTE_W and not p.left_arrow.isVisible()
    p.set_display("table")                                     # the table has no palette
    assert not pal.isVisible() and not p.palette_slot.isVisible()
    p.close()


# ======================================================================================================= dimming
def test_a_chosen_block_dims_the_other_workflows(qapp):
    from astro_gpu.client.graphview.gestures import Gestures
    p = _page(_doc("sho_rgb"))
    c, g = p.canvas, Gestures(p)
    g.click("b10")                                             # the SII stack's Gradient
    dim = {b for b, it in c.blocks.items() if it.d.get("dim")}
    assert {"b14", "b16", "b20", "b33", "b39", "b49"} <= dim   # Ha, S-H-O, RGB, Combine layers
    assert not dim & {"b3", "b5", "b9", "b10", "b11", "b12"}   # SII's own chain
    assert not dim & {"b6", "b7", "b17", "b18", "b1"}          # the Sessions, their Calibrations, Post
    ha = next(w for w in c.wires.values() if w.ends and set(w.ends) <= {"b14", "b15", "b16", "b20", "b21", "b22"})
    assert ha.opacity() < 1
    p.bridge.choose(None)
    assert not any(it.d.get("dim") for it in c.blocks.values()) and ha.opacity() == 1
    p.close()


# ======================================================================================================= wires
def test_wires_carry_their_image_name_and_the_chosen_block_shows_its_side_inputs(qapp):
    p = _page(_doc("colour_complete"))
    c, b = p.canvas, p.bridge
    tips = [w.tipfn() for w in c.wires.values() if w.tipfn is not None and w.ends == ("b13", "b14")]
    assert sorted(t.split("\n")[0] for t in tips) == ["Starless image", "Stars alone"]
    assert all(len(t.split("\n")) == 2 for t in tips)          # then the type of image, as before
    b.choose("b12")                                            # Denoise reads the half-stacks of its Stacking
    texts = [it.text() for it in c.side_items if hasattr(it, "text")]
    assert texts == ["half-stacks"] and len(c.side_items) == 2
    b.choose("b11")                                            # Color reads the Gaia spectra Astrometry fetched
    assert [it.text() for it in c.side_items if hasattr(it, "text")] == ["Gaia spectra"]
    b.choose("b10")                                            # Gradient: none
    assert c.side_items == []
    assert b.act("switch_off", block="b11").ok
    b.choose("b11")                                            # switched off: none (graph.py)
    assert c.side_items == []
    p.close()


# ======================================================================================================= touchpad, zoom
def _wheel(c, angle=QPoint(0, 0), pixel=QPoint(0, 0), mods=Qt.NoModifier, phase=Qt.NoScrollPhase):
    vp = c.viewport()
    at = QPointF(vp.width() / 2, vp.height() / 2)
    QApplication.sendEvent(vp, QWheelEvent(at, QPointF(vp.mapToGlobal(at.toPoint())), pixel, angle, Qt.NoButton,
                                           mods, phase, False))


def test_touchpad_pans_and_pinch_zooms_at_the_pointer(qapp):
    from astro_gpu.client.graphview.gestures import pump
    p = _page(_doc("sho_rgb"), size=(900, 600))
    c = p.canvas
    c.zoom_to(1.5)
    pump()
    k = c.zoom
    _wheel(c, angle=QPoint(0, 120), mods=Qt.ControlModifier)  # a Windows pinch: Ctrl and the wheel
    assert c.zoom == pytest.approx(k * 1.15)
    k = c.zoom
    h, v = c.horizontalScrollBar().value(), c.verticalScrollBar().value()
    _wheel(c, pixel=QPoint(-30, -40), angle=QPoint(-90, -120), phase=Qt.ScrollUpdate)   # two fingers
    assert c.zoom == k and (c.horizontalScrollBar().value(), c.verticalScrollBar().value()) == (h + 30, v + 40)
    _wheel(c, angle=QPoint(0, -120))                           # a mouse wheel's notch zooms out
    assert c.zoom == pytest.approx(k / 1.15)
    p.close()


def test_the_zoom_is_kept_with_the_project_never_as_an_undo_step(qapp):
    from astro_gpu.client.graphview.gestures import pump
    doc = _doc("sho_rgb")
    p = _page(doc, size=(900, 600))
    c, b = p.canvas, p.bridge
    undo = doc.can_undo
    c.zoom_to(1.5)
    pump()
    centre = c.mapToScene(c.viewport().rect().center())
    assert b.view["zoomed"] is True and b.view["zoom"] == 1.5 and doc.can_undo == undo
    b.view_changed()                                           # drawn again: the zoom stays
    pump()
    assert c.zoom == 1.5
    p.close()
    q = _page(doc, size=(900, 600))                            # the same project shown again
    assert q.canvas.zoom == 1.5 and not q.canvas.auto
    back = q.canvas.mapToScene(q.canvas.viewport().rect().center())
    assert abs(back.x() - centre.x()) < 6 and abs(back.y() - centre.y()) < 6     # (a scroll bar's width)
    q.canvas.auto_view()                                       # the fit: the zoom forgotten
    assert q.bridge.view["zoomed"] is False
    q.close()


# ======================================================================================================= Combine layers
def test_combine_layers_opens_post_on_a_graph_in_memory(qapp):
    from astro_gpu.client.graphview import menu
    from astro_gpu.client.graphview.gestures import Gestures
    p = _page(_doc("colour_complete"))
    assert menu.menu(p.bridge.doc, "b14")[0][:2] == ("Open Post", "inspect")
    got = []
    p.openPost.connect(lambda: got.append("post"))
    p.openLayer.connect(lambda lid: got.append(lid))
    p.openBlock.connect(lambda: got.append("inspect"))
    Gestures(p).click("b14", double=True)                     # no project model: no layer id, Post itself
    assert got == ["post"]
    p.close()


def test_items_without_a_model_give_the_run_overlay_nothing_to_draw(qapp):
    p = _page(_doc("colour_minimal"))
    assert all(getattr(it, "node", None) is None for it in p.canvas.blocks.values())
    p.close()


# ======================================================================================================= on a model
@pytest.fixture(scope="module")
def model(qapp, tmp_path_factory):
    """One project model for the module, read-only: Colour full made on its two sessions."""
    from astro_gpu.client.services.worker import ApiWorker
    from astro_gpu.core import project2
    w = ApiWorker()
    m = M(project2.load(_project(tmp_path_factory.mktemp("sides"))), w, readonly=True)
    assert w.wait(120) and m.ready, m.error
    yield m
    m.close(save=False)
    w.stop()


def _model_page(model):
    from astro_gpu.client.gallery import GALLERY
    from astro_gpu.client.graphview.gestures import pump
    from astro_gpu.client.graphview.page import GraphPage
    page = GraphPage(model, None)
    page.resize(1600, 900)
    page.show()
    pump()
    if not page.bridge.graph.rows:
        page.template_chosen(next(t for t in GALLERY if t["key"] == "colour_full"))
        pump()
    page.set_display("blocks")
    pump()
    return page


def _kind(page, kind):
    return next(b for b, blk in page.bridge.graph.blocks.items() if blk.kind == kind)


def test_the_page_hosts_todays_settings_panel_and_its_dock(model):
    from astro_gpu.client.graphnode import Node as PageNode
    from astro_gpu.client.graphview.gestures import Gestures, pump
    from astro_gpu.client.panel import Panel
    page = _model_page(model)
    pn, dock = page.panel, page.dock
    assert isinstance(pn, Panel) and dock.state == "docked" and pn.isVisible() and not dock.strip.isVisible()
    stacking = _kind(page, "stacking")
    Gestures(page).click(stacking)
    _wait(0)
    assert isinstance(pn.node, PageNode) and pn.node is page.current_node() and pn.node.key == "empilement"
    # a block shared by two workflows: the scope row, all of them first, then one only
    gradient = _kind(page, "gradient")
    assert page.bridge.act("add_variant", block=gradient).ok
    pump()
    Gestures(page).click(stacking)
    _wait(0)
    scope = [b for b in pn.findChildren(QPushButton) if b.objectName() == "scope" and b.isVisibleTo(pn)]
    assert len(scope) == 3 and scope[0].isChecked() and scope[0].text() == "both workflows"
    pids = [q.pid for q in pn.node.paths]
    assert sorted(model.scope) == sorted(pids)                 # "apply to all the rows" until one is chosen
    scope[2].click()
    assert model.scope == [pids[1]]
    page.bridge.undo()
    pump()
    # the Project
    page.bridge.choose("project")
    _wait(0)
    assert pn.node is not None and pn.node.key == "project"
    # the dock: folded to icons, opened over the graph, hidden with its arrow; the corner button folds it
    chosen, shown = [], []
    page.sideChosen.connect(lambda side, st: chosen.append((side, st)))
    page.sidesShown.connect(lambda: shown.append(1))
    page.set_layout("icons", "docked")
    assert dock.state == "icons" and dock.strip.isVisible() and not pn.isVisible()
    dock.open_overlay()
    assert pn.isVisible() and dock.is_open()
    dock.close_overlay()
    page.set_layout("hidden", "docked")
    assert dock.arrow.isVisible() and dock.arrow.toolTip() == "Show the panel again"
    dock.arrow.click()
    assert shown == [1]
    page.set_layout("docked", "docked")
    assert pn.isVisible() and not dock.strip.isVisible()
    pn.corner.click()
    assert chosen == [("right", "icons")]
    page.set_layout("docked", "docked")
    # the panel's switch: the block chosen switched off as D does, one undo step
    Gestures(page).click(gradient)
    pn.cutToggled.emit()
    assert page.bridge.block(gradient).off
    page.bridge.undo()
    assert not page.bridge.block(gradient).off
    page.close()


def _combine_layers(page):
    """A Combine layers block fed by the Stars block (the palette's double-click on it, the Stars block chosen)."""
    from astro_gpu.client.graphview.gestures import pump
    stars = _kind(page, "stars")
    page.bridge.choose(stars)
    res = page.bridge.act("add_block", kind="combine_layers", row=page.bridge.row_of(stars).id)
    assert res.ok, res.reason
    pump()
    return next(b for b, blk in page.bridge.graph.blocks.items() if blk.kind == "combine_layers")


def test_a_chosen_wire_to_a_block_leaves_the_panel_empty_and_the_page_whole(model):
    """Only a wire to Post has a panel (its layer); a wire into Combine layers chosen empties the panel."""
    page = _model_page(model)
    cl = _combine_layers(page)
    w = next(x for x in page.bridge.graph.wires if x.dst[0] == cl)
    page.canvas.selected_wire = w.id
    page.bridge.choose(None)
    page.canvas.wire_chosen.emit(w.id)
    _wait(0)
    assert page.panel.node is None and not page.wire_panel(w.id)
    page.bridge.undo()
    page.close()


def test_combine_layers_opens_its_layer_of_post(model):
    from astro_gpu.client.graphview.gestures import Gestures
    page = _model_page(model)
    cl = _combine_layers(page)
    node = page.layer_node(cl)
    assert node is not None and node.key == "composition" and node.paths
    got = []
    page.openLayer.connect(lambda lid: got.append(lid))
    page.openPost.connect(lambda: got.append("post"))
    page.openBlock.connect(lambda: got.append("inspect"))
    Gestures(page).click(cl, double=True)
    assert len(got) == 1 and isinstance(got[0], int) and model.post_layer(got[0]) is not None
    assert model.sel_step == "composition"                     # chosen as today: Inspect opens it in Post
    page.bridge.undo()
    page.close()


class _Run(QObject):
    updated = Signal()


def test_the_run_overlay_draws_on_the_new_blocks(model):
    from astro_gpu.client.runs import RunOverlay
    from astro_gpu.client.services.run import RUNNING
    page = _model_page(model)
    stacking = _kind(page, "stacking")
    it = page.canvas.blocks[stacking]
    wid = it.node.paths[0].pid
    track = SimpleNamespace(jobs=[SimpleNamespace(key="j1", block="stacking", wf=[wid], state=RUNNING)], groups=[],
                            progress=lambda jobs: 0.4)
    ctl = _Run()
    ctl.track, ctl.running = track, True
    ov = RunOverlay(page.canvas, ctl)
    assert ov.fraction_of(it.node) == 0.4
    other = page.canvas.blocks[_kind(page, "gradient")]
    assert ov.fraction_of(other.node) is None
    ov.grab()                                                  # painted over the blocks without an error
    page.close()


def test_the_zoom_is_kept_per_project_in_the_settings(model):
    from astro_gpu.client import prefs
    from astro_gpu.client.graphview.canvas import VIEW_KEY
    from astro_gpu.client.graphview.gestures import pump
    page = _model_page(model)
    page.canvas.zoom_to(1.25)
    page.canvas._write_view()
    key = page.canvas._project_key()
    assert (prefs.get(VIEW_KEY) or {})[key]["k"] == 1.25
    page.close()
    model.graph_view["zoomed"] = False                         # a project opened again: only the settings have it
    again = _model_page(model)
    pump()
    assert again.canvas.zoom == 1.25
    again.canvas.auto_view()
    again.canvas._write_view()
    assert key not in (prefs.get(VIEW_KEY) or {})
    again.close()


def test_a_change_that_adds_time_says_it(model, monkeypatch):
    page = _model_page(model)
    said = []
    page.toasted.connect(said.append)
    monkeypatch.setattr(model, "run_problems", lambda *a, **k: [])
    monkeypatch.setattr(model, "has_data", lambda *a, **k: True)
    secs = [600.0]
    monkeypatch.setattr(model, "total_seconds", lambda: secs[0])
    page._said_at = -9.0
    page.refresh()
    assert not [t for t in said if "to run" in t]              # the first estimate is the reference
    secs[0] = 720.0
    page.refresh()
    assert [t for t in said if t.startswith("+ ") and t.endswith(" to run")], said
    said.clear()
    secs[0] = 900.0
    page.said()                                                # a gesture's own message just said: not over it
    page.refresh()
    assert not [t for t in said if "to run" in t]
    page.close()
