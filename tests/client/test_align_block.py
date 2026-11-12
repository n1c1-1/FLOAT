"""The Align on block (issue #165): placed right after Stacking or an Image, on a reference workflow wired on the graph or
picked in its panel's list; empty it is a hollow dot and the block is greyed with the reason; a loop is refused; one
Ctrl+Z per wire; Inspect reads through it."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if Path("C:/Windows/Fonts").is_dir():
    os.environ.setdefault("QT_QPA_FONTDIR", "C:/Windows/Fonts")
pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication, QComboBox, QMessageBox  # noqa: E402

from astro_gpu.client import app as appmod  # noqa: E402
from astro_gpu.client import prefs  # noqa: E402
from astro_gpu.client.blocks import Graph  # noqa: E402
from astro_gpu.client.graph import BlockItem  # noqa: E402
from astro_gpu.client.services.worker import Gpu, Machine  # noqa: E402

MACHINE = Machine([Gpu("cuda:0", "RTX", 16.0)], 16, 64.0)


@pytest.fixture(scope="session")
def qapp():
    from astro_gpu.client.widgets import QSS
    app = QApplication.instance() or QApplication([])
    app.setStyleSheet(QSS + appmod.EXTRA_QSS)
    return app


@pytest.fixture()
def world(tmp_path, monkeypatch, qapp):
    import cv2
    from astropy.io import fits
    monkeypatch.setenv("ASTRO_GPU_SETTINGS", str(tmp_path / "settings.json"))
    monkeypatch.setenv("ASTRO_GPU_HOME", str(tmp_path / "home"))
    prefs.load()
    prefs.set("home_sections", [])
    prefs.set("cache_dir", (tmp_path / "scratch").as_posix())
    appmod.apply_language("en")
    monkeypatch.setattr(appmod, "machine_info", lambda: MACHINE)
    for kind in ("information", "warning", "critical"):
        monkeypatch.setattr(QMessageBox, kind, lambda *a, **k: None)
    png, fit, fit2 = tmp_path / "a.png", tmp_path / "b.fits", tmp_path / "c.fits"
    cv2.imwrite(str(png), np.full((8, 8, 3), 128, np.uint8))
    fits.PrimaryHDU(np.full((3, 8, 8), 0.1, np.float32)).writeto(fit)
    fits.PrimaryHDU(np.full((3, 8, 8), 0.2, np.float32)).writeto(fit2)
    w = appmod.Main()
    w.resize(1500, 800)
    w.show()
    w.worker.wait()
    w.new_project()
    w.worker.wait()
    settle(w)
    w.graph.place_images([str(png), str(fit), str(fit2)])
    settle(w)
    w.said = []
    w.graph.toasted.connect(w.said.append)
    yield w
    w.close()
    w.deleteLater()
    QApplication.processEvents()


def settle(w, n=12):
    for _i in range(n):
        QApplication.processEvents()
        w.worker.wait(30)
    QApplication.processEvents()


def item_of(w, pid: int, key: str = "align_on") -> BlockItem:
    node = Graph(w.s).node_of(w.s.view(pid), key)
    return next(x for x in w.graph.graph.scene().items() if isinstance(x, BlockItem) and x.node.key == node.key
                and x.node.paths and x.node.paths[0].pid == pid)


def test_align_on_is_a_palette_block_placed_empty_and_greyed_with_its_reason(world):
    w = world
    s = w.s
    _a, b, _c = s.paths
    assert not s.palette_reason(b, "align_on") and "align_on" in w.graph.cards
    w.graph.place_block("align_on", b.pid)
    settle(w)
    assert s.display_keys(s.view(b.pid))[:2] == ["image", "align_on"] and s.align_reference(s.view(b.pid)) is None
    g = Graph(s)
    n = g.node_of(s.view(b.pid), "align_on")
    assert n.waits == "reference to wire" and g.summary(n).endswith("reference to wire")
    it = item_of(w, b.pid)
    assert [(d["name"], d["wired"]) for d in it.inputs] == [("Reference", False)]      # a hollow dot
    assert "empty" in it.toolTip() and "reference to wire" in it.toolTip()
    assert s.palette_reason(s.view(b.pid), "align_on")                                  # once per workflow


def test_a_wire_dropped_on_the_reference_input_wires_it_in_one_undo_step(world):
    w = world
    s = w.s
    a, b, _c = s.paths
    w.graph.place_block("align_on", b.pid)
    settle(w)
    n_undo = len(s._undo)
    gv = w.graph.graph
    src = Graph(s).node_of(s.view(a.pid), "image")
    dst = Graph(s).node_of(s.view(b.pid), "align_on")
    go, why = gv.plan_wire(src, None, dst, ("reference",))
    assert go is not None and why == ""
    w.graph.wire_to_input(src, None, dst, ("reference",))
    settle(w)
    assert s.align_reference(s.view(b.pid)) == a.pid and len(s._undo) == n_undo + 1       # one wire, one step
    assert Graph(s).node_of(s.view(b.pid), "align_on").waits == ""
    assert s.undo()
    settle(w)
    assert s.align_reference(s.view(b.pid)) is None


def test_a_real_drag_from_an_output_onto_the_reference_dot(world):
    from PySide6.QtCore import QPointF, Qt
    from PySide6.QtTest import QTest
    w = world
    s = w.s
    a, b, _c = s.paths
    w.graph.place_block("align_on", a.pid)               # the source image has blocks after it: its output ● is there
    w.graph.place_block("align_on", b.pid)
    settle(w)
    gv = w.graph.graph
    src_it = item_of(w, a.pid, "image")
    dst_it = item_of(w, b.pid)
    assert src_it.ports                                                           # the Image block has its output ●
    start = src_it.mapToScene(QPointF(src_it.w, src_it.ports[0][1]))
    dot = dst_it.mapToScene(QPointF(0, dst_it.inputs[0]["y"]))
    vp = gv.viewport()
    QTest.mousePress(vp, Qt.LeftButton, pos=gv.mapFromScene(start))
    QTest.mouseMove(vp, gv.mapFromScene(dot))
    QTest.mouseRelease(vp, Qt.LeftButton, pos=gv.mapFromScene(dot))
    settle(w)
    assert s.align_reference(s.view(b.pid)) == a.pid


def test_a_loop_is_refused_with_its_reason_and_nothing_changes(world):
    w = world
    s = w.s
    a, b, _c = s.paths
    w.graph.place_block("align_on", a.pid)
    w.graph.place_block("align_on", b.pid)
    settle(w)
    assert s.set_align_reference(s.view(b.pid), a.pid)
    n_undo = len(s._undo)
    g = Graph(s)
    src, dst = g.node_of(s.view(b.pid), "image"), g.node_of(s.view(a.pid), "align_on")
    go, why = w.graph.graph.plan_wire(src, None, dst, ("reference",))
    assert go is None and "loop" in why                                                  # b aligned on a, a on b
    with pytest.raises(ValueError, match="loop"):
        s.set_align_reference(s.view(a.pid), b.pid)
    w.graph.wire_to_input(src, None, dst, ("reference",))                                # a toast, no pop-up
    assert "loop" in w.said[-1]
    assert s.align_reference(s.view(a.pid)) is None and len(s._undo) == n_undo
    assert s.join_problem(a.pid, a.pid, True)                                            # itself


def test_the_panel_lists_every_workflow_for_the_reference_and_picks_it(world):
    w = world
    s = w.s
    a, b, c = s.paths
    w.graph.place_block("align_on", c.pid)
    settle(w)
    w.graph.select(index=s.paths.index(s.view(c.pid)), step="align_on")
    settle(w)
    cb = next(x for x in w.graph.panel.findChildren(QComboBox) if x.findData(a.pid) >= 0 and x.findData(b.pid) >= 0)
    assert cb.findData(c.pid) < 0 and cb.findData(None) >= 0                           # every other workflow, or none
    i = cb.findData(b.pid)
    cb.setCurrentIndex(i)
    cb.activated.emit(i)
    settle(w)
    assert s.align_reference(s.view(c.pid)) == b.pid
    assert s.p.workflow(c.pid).blocks[0]["block"] == "align_on" and s.p.workflow(c.pid).blocks[0]["method"] == "stars"
    assert s.channel_problem(s.view(b.pid)).startswith("reference of")                 # a reference cannot be deleted


def test_the_cross_removes_the_block_and_ctrl_z_brings_it_back(world):
    w = world
    s = w.s
    a, b, _c = s.paths
    w.graph.place_block("align_on", b.pid)
    settle(w)
    s.set_align_reference(s.view(b.pid), a.pid)
    node = Graph(s).node_of(s.view(b.pid), "align_on")
    w.graph.remove_tool(node, None)
    settle(w)
    assert s.align_node(s.view(b.pid)) is None and "align_on" not in s.display_keys(s.view(b.pid))
    assert s.undo()
    settle(w)
    assert s.align_reference(s.view(b.pid)) == a.pid


def test_align_images_is_gone_everywhere(world):
    w = world
    s = w.s
    assert not hasattr(s, "align_images") and not hasattr(s, "align_group") and not hasattr(s, "plan_align")
    assert "align_images" not in w.graph.cards and all("align" not in (q.source or {}) for q in s.paths)
    from astro_gpu.client import blocks
    assert "align_images" not in blocks.BY_KEY and "align_on" in blocks.BY_KEY


def test_inspect_reads_the_aligned_image_after_align_on(world):
    w = world
    s = w.s
    a, b, _c = s.paths
    w.graph.place_block("align_on", b.pid)
    w.graph.place_block("astrometry", b.pid)
    settle(w)
    from astro_gpu.client.steps import Steps
    st = Steps(s)
    v = s.view(b.pid)
    assert st.read_stage(v, "stack") == "align_on" and st.read_stage(s.view(a.pid), "stack") == "stack"
    assert [x.stage for x in st.visuals(v, "astrometrie") if x.kind == "array"] == ["align_on"]
    assert [x.stage for x in st.visuals(v, "align_on")] == ["align_on"]


def _method_combo(w):
    return next((x for x in w.graph.panel.findChildren(QComboBox) if x.findData("plate") >= 0), None)


def test_plate_is_greyed_for_a_stack_with_its_reason_and_open_for_an_image(world, monkeypatch):
    """Amendment A: no plate solution on a stack at that point (Astrometry comes after it); an imported image that
    carries one can use it."""
    from astro_gpu.client import settings
    w = world
    s = w.s
    c = s.paths[2]
    w.graph.place_block("align_on", c.pid)
    settle(w)
    for stack in (False, True):
        if stack:
            monkeypatch.setattr(settings, "_is_image", lambda _w: False)
        w.graph.select(index=s.paths.index(s.view(c.pid)), step="align_on")
        settle(w)
        cb = _method_combo(w)
        assert cb is not None
        item = cb.model().item(cb.findData("plate"))
        assert item.isEnabled() is not stack
        if stack:
            assert item.toolTip() == "the stack has no plate solution yet (Astrometry comes after)"
        assert cb.model().item(cb.findData("stars")).isEnabled()
