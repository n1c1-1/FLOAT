"""Every way into Inspect for a Combine layers block opens its editor in Inspect, never Post (the owner, on main: the
stars and background block sent him to Post). The ways: a double-click on its card, Enter, its menu's Inspect, its
panel's Adjust, the Inspect tab, Inspect's path bar, Post's "Open in Inspect". Its menu's Open Post still opens Post on
its layer. The owner's M27 (a scratch copy of its definition, its results read in place)."""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
pytest.importorskip("PySide6")
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication, QPushButton  # noqa: E402

from test_combination_family import M27, _whole, pump  # noqa: E402

WAYS = ("double_click", "enter", "menu_inspect", "panel_adjust", "inspect_tab", "path_bar", "post_link")


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def win(qapp, tmp_path_factory):
    w = _whole(tmp_path_factory.mktemp("inspect_cl"), M27)
    yield w
    w.close()


def _block(w):
    return next(k for k, b in w.graph.bridge.graph.blocks.items() if b.kind == "combine_layers")


def _on_graph_elsewhere(w):
    """The graph shown, another block chosen (a block of the chain): nothing of Combine layers chosen yet."""
    from astro_gpu.client.graphview.gestures import Gestures
    w.set_mode(0)
    pump(0.3)
    other = next(k for k, b in w.graph.bridge.graph.blocks.items() if b.kind == "gradient")
    Gestures(w.graph).click(other)
    pump(0.3)
    assert w.s.sel_step != "composition"


def _in_its_editor(w):
    pump(0.6)
    ed = w.tune.layers_editor
    return w.mode == 1 and w.s.sel_step == "composition" and ed.page_shown()


@pytest.mark.parametrize("way", WAYS)
def test_real_every_way_into_inspect_opens_combine_layers_editor(win, way):
    from astro_gpu.client.graphview.gestures import Gestures
    w = win
    _on_graph_elsewhere(w)
    cl = _block(w)
    g = Gestures(w.graph)
    if way == "double_click":
        g.click(cl, double=True)
    elif way == "enter":
        g.click(cl)
        from PySide6.QtTest import QTest
        QTest.keyClick(w.graph.canvas, Qt.Key_Return)
    elif way == "menu_inspect":
        m = w.graph.canvas.build_menu(w.graph.canvas.blocks[cl])
        names = [a.text() for a in m.actions() if a.text()]
        assert names[0] == "Inspect" and "Open Post" in names, names
        next(a for a in m.actions() if a.text() == "Inspect").trigger()
    elif way == "panel_adjust":
        g.click(cl)
        pump(0.3)
        b = next(b for b in w.graph.panel.findChildren(QPushButton) if b.text() == "Adjust" and
                 b.isVisibleTo(w.graph.panel))
        assert b.isEnabled()
        b.click()
    elif way == "inspect_tab":
        g.click(cl)
        pump(0.3)
        w.mode_btns[1].click()
    elif way == "path_bar":
        comb = next(k for k, b in w.graph.bridge.graph.blocks.items() if b.kind == "combine" and
                    not w.graph.bridge.row_of(k).name.endswith("Stars"))
        g.click(comb)                                    # Inspect on the combination, then Combine layers' step in its
        pump(0.3)                                        # path bar (S-H-O › Combine layers)
        w.set_mode(1)
        pump(0.5)
        assert w.s.sel_step != "composition"
        k = next(k for k, _t, _n in w.tune.strip.items(w.s, w.s.block_map) if k == "composition")
        w.tune._strip_pick(k)
    elif way == "post_link":
        w.set_mode(2)
        pump(1.0)
        pv = w._post_holder.page.view
        q = next(x for x in w.s.paths if w.s.master_layer(x.pid) is not None)
        lay = next(x for x in pv.layers if x.get("id") == w.s.master_layer(q.pid)["id"])
        pv.sel = None
        pv.select(lay["id"], "layer")
        pump(0.3)
        links = [b for b in pv.findChildren(QPushButton) if b.text() == "Open in Inspect  →" and
                 b.isVisibleTo(pv.right)]
        links[-1].click()                                # Combine layers' link (the last section's)
    assert _in_its_editor(w), (way, w.mode, w.s.sel_step)


def test_real_its_menu_s_open_post_opens_post_on_its_layer(win):
    w = win
    _on_graph_elsewhere(w)
    cl = _block(w)
    m = w.graph.canvas.build_menu(w.graph.canvas.blocks[cl])
    next(a for a in m.actions() if a.text() == "Open Post").trigger()
    pump(1.0)
    assert w.mode == 2
    lid = w.graph.layer_id(cl)
    assert lid is not None and w._post_holder.page.view.sel[0] == lid
