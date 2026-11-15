"""On a small window (1250 px wide) the block's settings are a drawer shown on demand over the graph:
- answering Stay to the question asked when another block is chosen while the block holds a draft leaves the drawer
  open on the block (the click outside had closed it: the kept draft was out of sight);
- Tab never hides the side bars while a settings field could take it: with the focus in the settings, or their drawer
  open, it goes on to a field. Elsewhere, Tab hides the panels as before."""
from __future__ import annotations

from PySide6.QtWidgets import QApplication, QComboBox

from astro_gpu.client import layout as lay
from astro_gpu.client.blockdraft import DRAFT_ANSWER_ENV
from tests.client.test_new_page_asks_to_save import _ready, qapp, win  # noqa: F401 - the fixtures


def _pump(w, n=8):
    for _i in range(n):
        w.worker.wait(2)
        QApplication.processEvents()


def _small_with_blocks(win):
    from astro_gpu.client.gallery import GALLERY
    win.new_project()
    _ready(win)
    win.graph.show_templates()
    win.graph.templates.chosen.emit(dict(next(t for t in GALLERY if t["key"] == "colour_full"), file=""))
    _pump(win)
    win.size_range = lay.size_class(1250, 685)
    lay.remember(win.size_range, right="icons", hidden=False, right_hidden=False, max=False)
    win.apply_layout()
    _pump(win)
    page = win.graph
    assert page.dock.state == "icons"
    kinds = {x.kind: b for b, x in page.bridge.graph.blocks.items()}
    return page, kinds


def test_stay_leaves_the_drawer_open_on_the_block(win, monkeypatch):
    from PySide6.QtCore import Qt
    # the question waits for the mouse button to be up: a button another test left down never lets it come
    monkeypatch.setattr(QApplication, "mouseButtons", staticmethod(lambda: Qt.NoButton))
    page, kinds = _small_with_blocks(win)
    page.bridge.choose(kinds["calibration"])
    _pump(win)
    page.dock.open_overlay()
    assert page.dock.is_open()
    monkeypatch.setattr(win.s, "pending_change", object(), raising=False)     # the block holds a draft
    monkeypatch.setattr(page.panel, "draft_pending", lambda: ("Calibration", ["Bias: a → b"], ""))
    monkeypatch.setenv(DRAFT_ANSWER_ENV, "stay")
    page.dock.close_overlay()                       # the click on the graph outside the drawer closes it first...
    page.bridge.choose(kinds["gradient"])           # ...and chooses another block: asked, Stay
    _pump(win)
    assert page.bridge.selected == kinds["calibration"]
    assert page.dock.is_open() and page.panel.isVisible()


def test_tab_goes_to_the_settings_and_never_hides_the_bars_while_a_field_can_take_it(win):
    page, kinds = _small_with_blocks(win)
    page.bridge.choose(kinds["calibration"])
    _pump(win)
    cls = win.size_range
    page.dock.open_overlay()
    page.canvas.setFocus()                          # the focus left the drawer
    _pump(win)
    win._tab()
    _pump(win)
    assert not lay.state(cls)["hidden"] and page.dock.is_open()
    fw = QApplication.focusWidget()
    assert fw is not None and page.panel.isAncestorOf(fw)          # it went to a field of the settings
    combo = next((c for c in page.panel.findChildren(QComboBox) if c.isVisible() and c.isEnabled()), None)
    if combo is not None:                           # the focus on a list of the settings: Tab goes on
        combo.setFocus()
        _pump(win)
        win._tab()
        _pump(win)
        assert not lay.state(cls)["hidden"]
    page.dock.close_overlay()                       # no drawer open, the focus on the graph: Tab hides, as before
    page.canvas.setFocus()
    _pump(win)
    win._tab()
    _pump(win)
    assert lay.state(cls)["hidden"]
    win._tab()
    _pump(win)
