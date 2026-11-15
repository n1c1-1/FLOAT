"""One Undo for the window: Ctrl+Z (wherever the focus is), Edit > Undo and a notice's Undo link do the same thing (the
night test pass: Ctrl+Z did nothing in the Graph while the menu worked, a notice's Undo did nothing). Post and the graph
keep two histories by design: a page with nothing to undo says so and offers the other page's step, which its link
undoes there. What was undone is said in the interface's words, never the engine's raw names."""
from __future__ import annotations

import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from astro_gpu.client import app as appmod  # noqa: E402
from astro_gpu.client import prefs  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    from astro_gpu.client.widgets import QSS
    a = QApplication.instance() or QApplication([])
    a.setStyleSheet(QSS + appmod.EXTRA_QSS)
    return a


def _settle(w, n=8):
    for _i in range(n):
        w.worker.wait(30)
        QApplication.processEvents()
        time.sleep(0.01)


@pytest.fixture()
def win(qapp, tmp_path, monkeypatch):
    monkeypatch.setenv("ASTRO_GPU_SETTINGS", str(tmp_path / "settings.json"))
    prefs.load()
    appmod.apply_language("en")
    from astro_gpu.client.gallery import GALLERY
    w = appmod.Main()
    w.resize(1500, 900)
    w.show()
    w.activateWindow()
    w.new_project()
    end = time.monotonic() + 120
    while time.monotonic() < end and not w.s.ready:
        w.worker.wait(1)
        QApplication.processEvents()
    w.graph.show_templates()
    w.graph.templates.chosen.emit(next(t for t in GALLERY if t["key"] == "colour_full"))
    _settle(w)
    yield w
    w.close_project(discard=True)
    w.close()
    w.deleteLater()
    QApplication.processEvents()


def _gradient(w):
    g = w.graph.bridge.graph
    return next(k for k, x in g.blocks.items() if x.kind == "gradient")


def _method(w):
    return w.graph.bridge.graph.blocks[_gradient(w)].method


def _set_rbf(w):
    res = w.graph.bridge.act("set_node_method", block=_gradient(w), method="rbf")
    if not res.ok:
        res = w.graph.bridge.act("set_method", block=_gradient(w), method="rbf")
    assert res.ok
    _settle(w, 3)
    assert _method(w) == "rbf"


@pytest.mark.parametrize("how", ["ctrl+z on the graph", "edit menu", "notice link"])
def test_ctrl_z_the_menu_and_a_notice_link_undo_the_same_step(win, how):
    _set_rbf(win)
    if how == "ctrl+z on the graph":
        win.graph.canvas.setFocus()
        QApplication.processEvents()
        QTest.keyClick(QApplication.focusWidget() or win.graph.canvas, Qt.Key_Z, Qt.ControlModifier)
    elif how == "edit menu":
        next(a for m in win.menuBar().actions() for a in m.menu().actions() if a.text() == "Undo").trigger()
    else:
        win.graph.bridge.drops.link("undo")
    _settle(win, 3)
    assert _method(win) != "rbf", how
    said = win.status.text()
    assert said.startswith("Undone: ") and "RBF" in said and "rbf" not in said and "method:" not in said, said


def test_the_page_shortcut_never_competes_with_the_menu(win):
    from PySide6.QtGui import QKeySequence, QShortcut
    live = [sc for sc in win.graph.findChildren(QShortcut) if sc.isEnabled() and sc.key() == QKeySequence("Ctrl+Z")]
    assert live == []
    assert win.a_undo.shortcut() == QKeySequence("Ctrl+Z")


def test_post_with_nothing_of_its_own_offers_the_graphs_step(win):
    _set_rbf(win)
    win.set_mode(2)
    _settle(win, 3)
    assert win._post_shown()
    win.undo()
    said = win.status.text()
    assert "Nothing to undo in Post" in said and "Undo there" in said and _method(win) == "rbf"
    win._flash_linked("do")                                          # the link
    _settle(win, 3)
    assert win.mode == 0 and _method(win) != "rbf"


def test_the_graph_with_posts_step_last_offers_it(win):
    s = win.s
    win.set_mode(0)
    lid = s.add_post_layer(s.paths[0].pid, None)
    with s.from_post():
        s.set_post_layer(lid, visible=False)
    assert s.can_undo() and not s.can_undo(graph=True)
    win.undo()
    said = win.status.text()
    assert "Nothing to undo in the Graph" in said and "Undo there" in said
    win._flash_linked("do")
    _settle(win, 3)
    assert win.mode == 2
