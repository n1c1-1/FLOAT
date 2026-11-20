"""#181, step 5: the layers panel shows the layer tree as a reference editor's layer groups: rows nested by their depth,
the triangle (Alt: it and what is nested in it, Ctrl: every row, its menu: Expand all, Collapse all), the eye at each
level (what is under a hidden node is dimmed, Alt shows one alone and again restores), a filter dropped on a node row (Shift:
at the bottom), the settings of the node chosen, the note "Mix fixed" naming the block that fixes a mix, in both
languages. The owner's M27 for the layers, its tree given three levels by hand (nothing is rendered here)."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtWidgets import QApplication

sys.path.insert(0, str(Path(__file__).parent))
from test_post_fusion_window import M27, m27, qapp  # noqa: E402,F401

from astro_gpu.client import i18n  # noqa: E402
from astro_gpu.client import postengine as pe  # noqa: E402
from astro_gpu.client import postfusion as pf  # noqa: E402
from astro_gpu.client.post3 import FIXED_NOTE, ContSlider, INDENT  # noqa: E402


def _inner(root):
    """An H-O of the same result under the Hα of the S-H-O (a combination inside a combination, as a graph can have)."""
    f = copy.deepcopy(root["fusion"])
    keep = [1, 2]
    f["roles"] = [f["roles"][i] for i in keep]
    f["names"] = {r: f["names"][r] for r in f["roles"]}
    f["matrix"] = [[row[i] for i in keep] for row in f["matrix"]]
    f["weights"] = {r: f["weights"][r] for r in f["roles"]}
    f["hue_params"] = {r: v for r, v in f["hue_params"].items() if r in f["roles"]}
    f["hues"] = {r: v for r, v in f["hues"].items() if r in f["roles"]}
    f["hue_defaults"] = {r: v for r, v in f["hue_defaults"].items() if r in f["roles"]}
    f["title"] = "H-O"
    return {"kind": "mix", "block": 99, "pid": root["pid"], "title": "H-O", "fusion": f,
            "roles": {r: {"kind": "image", "block": 98, "port": "image"} for r in f["roles"]}}


@pytest.fixture()
def pv(m27, monkeypatch):
    """Post of the M27 with three levels (Combine layers, S-H-O, an H-O as its Hα), nothing rendered."""
    w, view, _before = m27
    real = w.s.post_tree

    def tree(lay, live=None):
        t = real(lay, live) if live else real(lay)
        if t is not None and t["kind"] == "layers" and "starless" in t["sides"] and t["sides"]["starless"]["kind"] == "mix":
            root = t["sides"]["starless"]
            root["roles"]["Ha"] = _inner(root)
        return t
    monkeypatch.setattr(w.s, "post_tree", tree)
    monkeypatch.setattr(view, "request", lambda *a, **k: None)
    saved = pe.snapshot(view.layers), set(view._collapsed)
    view.sel = None
    view._collapsed = set()
    view.rebuild()
    yield view
    view.layers[:] = pe.load_layers(pe.save_layers(saved[0]), {p.pid for p in view.s.paths})
    view._collapsed = saved[1]
    view._solo = {}
    monkeypatch.undo()
    view.sel = None
    view.rebuild()


def _settle():
    """What the event loop does after a menu's signal: the deferred calls, then the widgets deleted later."""
    from PySide6.QtCore import QEvent
    for _ in range(3):
        QApplication.processEvents()
        QApplication.sendPostedEvents(None, QEvent.DeferredDelete)


def _g(pv):
    return next(x for x in pv.layers if pe.is_group(x) and x.get("starless") == "output")


def _names(pv):
    return [(r.lay.get("name"), r.depth) for r in pv.rows]


def _row(pv, sub):
    return next(r for r in pv.rows if r.lay.get("kind") == "sub" and r.lay["sub"] == sub)


def test_the_triangle_shuts_one_node_alt_it_and_what_is_nested_ctrl_every_row_and_its_menu_says_all(pv, monkeypatch):
    lay = _g(pv)
    key = (lay["id"], "f:Ha")
    pv.toggle_children(key)
    assert key in pv._collapsed and "f:Ha/f:Ha" not in [r.lay.get("sub") for r in pv.rows]
    pv.toggle_children(key)
    assert key not in pv._collapsed
    monkeypatch.setattr(QApplication, "keyboardModifiers", staticmethod(lambda: Qt.AltModifier))
    pv.toggle_children((lay["id"], "starless"))                      # Alt: the side and every node nested in it
    assert {(lay["id"], "starless"), (lay["id"], "f:Ha")} <= pv._collapsed
    pv.toggle_children((lay["id"], "starless"))
    assert not pv._collapsed
    monkeypatch.setattr(QApplication, "keyboardModifiers", staticmethod(lambda: Qt.ControlModifier))
    pv.toggle_children(lay["id"])                                    # Ctrl: every row of the panel
    assert lay["id"] in pv._collapsed and (lay["id"], "f:Ha") in pv._collapsed and [r.depth for r in pv.rows] == [0]
    pv.toggle_children(lay["id"])
    assert not pv._collapsed
    monkeypatch.undo()
    pv.rebuild()
    row = _row(pv, "starless")
    texts = [a.text() for a in row.toggle.actions()]
    assert texts == ["Expand all", "Collapse all"]
    row.toggle.actions()[1].trigger()
    assert [r.depth for r in pv.rows] != [0]                          # not inside the menu's own signal: its button lives on
    assert row.toggle.parent() is not None
    _settle()
    assert [r.depth for r in pv.rows] == [0]
    pv.rebuild()
    pv._collapsed = {(lay["id"], "f:Ha")}
    pv.rebuild()
    next(r for r in pv.rows if r.lay.get("kind") == "sub").toggle.actions()[0].trigger()
    _settle()
    assert pv._collapsed == set()
    i18n.set_language("fr")
    try:
        pv.rebuild()
        assert [a.text() for a in _row(pv, "starless").toggle.actions()] == ["Tout ouvrir", "Tout fermer"]
    finally:
        i18n.set_language("en")
        pv.rebuild()


def test_what_is_under_a_hidden_node_is_dimmed_and_keeps_its_eyes(pv):
    lay = _g(pv)
    pv.set_shown(lay["id"], "f:Ha", False)
    assert lay["shown"]["f:Ha"] is False
    below = _row(pv, "f:Ha/f:Ha")
    assert below.eye.dimmed and below.eye.isChecked()                  # its own eye is as it was
    assert not _row(pv, "f:Ha").eye.dimmed and not _row(pv, "f:OIII").eye.dimmed
    pv.do_undo()
    assert _g(pv)["shown"].get("f:Ha", True) is True and not _row(pv, "f:Ha/f:Ha").eye.dimmed


def test_alt_on_an_eye_shows_that_node_alone_among_its_siblings_and_again_gives_the_eyes_back(pv, monkeypatch):
    lay = _g(pv)
    pv.set_shown(lay["id"], "f:OIII", False)
    was = dict(lay["shown"])
    monkeypatch.setattr(QApplication, "keyboardModifiers", staticmethod(lambda: Qt.AltModifier))
    pv.set_shown(lay["id"], "f:Ha", True)
    assert [lay["shown"].get(x, True) for x in ("f:SII", "f:Ha", "f:OIII")] == [False, True, False]
    pv.set_shown(lay["id"], "f:Ha", True)
    assert {k: lay["shown"].get(k, True) for k in ("f:SII", "f:Ha", "f:OIII")} == {k: was.get(k, True) for k in (
        "f:SII", "f:Ha", "f:OIII")}
    pv.set_shown(lay["id"], "f:Ha/f:Ha", True)                         # among the two of the inner combination
    assert [lay["shown"].get(x, True) for x in ("f:Ha/f:Ha", "f:Ha/f:OIII")] == [True, False]


def test_a_node_chosen_shows_its_own_settings_and_the_filter_it_is_in_its_mix(pv):
    lay = _g(pv)
    pv.select(lay["id"], "f:Ha")
    names = [x.name.text() for x in pv.findChildren(ContSlider) if not x.isHidden()]
    assert {"Hα hue", "Hα intensity", "O III hue", "O III intensity"} <= set(names)
    assert names.count("Hα hue") == 2                                    # as a filter of the S-H-O and as the H-O's own
    pv.select(lay["id"], "f:Ha/f:Ha")
    names = [x.name.text() for x in pv.findChildren(ContSlider) if not x.isHidden()]
    assert names == ["Hα hue", "Hα intensity"]                          # a leaf: its place in the H-O, nothing of its own


def test_a_hue_of_the_inner_combination_is_that_node_s_own_one_step_of_undo(pv):
    lay = _g(pv)
    pv.select(lay["id"], "f:Ha")
    n = len(pv.undo)
    pv.set_fusion(lay, {"hue_oiii": 123.0}, "O III hue", path="f:Ha")
    assert lay["tunes"]["f:Ha"]["hues"]["OIII"] == 123.0 and "tune" not in lay and len(pv.undo) == n + 1
    pv.do_undo()
    assert not (_g(pv).get("tunes") or {}).get("f:Ha")


def mix_hue(lay, path, role):
    return pe.mix_at(lay, path)["fusion"]["hues"][role]


def test_the_tree_follows_a_version_switch_and_the_compared_side(pv):
    lay = _g(pv)
    saved = pe.save_layers(pv.layers)
    other = pe.load_layers(copy.deepcopy(saved), {p.pid for p in pv.s.paths})
    assert not pe.nodes_of(next(x for x in other if pe.is_group(x)))        # the tree is never saved
    pv._attach_tree(other)
    assert "f:Ha/f:Ha" in pe.nodes_of(next(x for x in other if pe.is_group(x)))
    assert "f:Ha/f:Ha" in pe.nodes_of(lay)


def test_a_mix_fixed_shows_its_note_on_the_layer_the_side_and_the_filter_row_naming_the_block(pv, monkeypatch):
    w = pv.s
    real = w.post_tree
    for place in ("side", "role"):
        def tree(lay, live=None, place=place):
            t = real(lay, live) if live else real(lay)
            if t is not None and t["kind"] == "layers" and t["sides"]["starless"]["kind"] == "mix":
                fixed = {"kind": "fixed", "block": 7, "combination": "H-O", "by": "Stars/background", "why": "after"}
                if place == "side":
                    t["sides"]["starless"] = dict(fixed, combination="S-H-O")
                else:
                    t["sides"]["starless"]["roles"]["OIII"] = fixed
            return t
        monkeypatch.setattr(w, "post_tree", tree)
        pv.rebuild()
        lay = _g(pv)
        sub = "starless" if place == "side" else "f:OIII"
        row = _row(pv, sub)
        assert row.note is not None and row.note.text() == FIXED_NOTE
        assert "Stars/background" in row.note.toolTip()
        i18n.set_language("fr")
        try:
            pv.rebuild()
            row = _row(pv, sub)
            assert row.note.text() == "Mélange figé" and "bloc Stars/background" in row.note.toolTip()
        finally:
            i18n.set_language("en")
        assert not any(r.note is not None for r in pv.rows if r.lay.get("sub") not in (sub,))
    assert lay is not None


def test_a_node_cannot_be_dragged_and_says_why_on_hover_and_in_its_menu(pv):
    row = _row(pv, "f:Ha/f:Ha")
    assert "cannot be dragged" in row.toolTip()
    menu = pv.layer_menu_of(row.lay)
    greyed = [a for a in menu.actions() if a.text() in ("Duplicate", "Rename", "Delete")]
    assert greyed and not any(a.isEnabled() for a in greyed) and all("graph" in a.toolTip() for a in greyed)
    pv.drag_start(row)
    pv.drag_cancel()


# (laid out last: showing the panel folds the properties column of a small window)
def test_the_rows_are_nested_by_their_depth_and_each_step_indents_by_one_step(pv):
    names = _names(pv)
    i = names.index(("Starless image", 1))
    assert names[i + 1:i + 7] == [("S II", 2), ("Hα", 2), ("Hα", 3), ("O III", 3), ("O III", 2)][:3] + [
        ("O III", 3), ("O III", 2)] or names[i + 1:i + 7] == [("S II", 2), ("Hα", 2), ("Hα", 3), ("O III", 3),
                                                              ("O III", 2)]
    deep = [(r.lay["sub"], r.depth) for r in pv.rows if r.lay.get("kind") == "sub" and r.lay["sub"].startswith("f:Ha")]
    assert deep == [("f:Ha", 2), ("f:Ha/f:Ha", 3), ("f:Ha/f:OIII", 3)]
    pv.show()
    pv.resize(1100, 800)
    QApplication.processEvents()
    x = {r.lay["sub"]: r.t.mapTo(r, QPoint(0, 0)).x() for r in pv.rows if r.lay.get("kind") == "sub"}
    step = x["f:Ha"] - x["starless"]                                  # one step in per level, the same at every depth
    assert INDENT - 6 <= step <= INDENT and INDENT - 6 <= x["f:Ha/f:Ha"] - x["f:Ha"] <= INDENT, x


def test_a_filter_dropped_on_a_node_row_goes_to_the_top_of_its_filters_shift_to_the_bottom(pv, monkeypatch):
    lay = _g(pv)
    a = pe.new_filter_layer("levels", lay["id"], "f:Ha/f:OIII")
    b = pe.new_filter_layer("saturation", None)
    pv.layers[:0] = [a, b]
    pv.layers[:] = pe.normalize(pv.layers)
    pv.rebuild()
    pv.show()
    pv.resize(900, 700)
    QApplication.processEvents()
    row = _row(pv, "f:Ha/f:OIII")
    g = row.geometry()
    holder = pv.lbox.parentWidget()
    cx, cy = g.left() + g.width() // 2, g.top() + g.height() // 2
    d = pv.drop_at(b, cx, cy)
    assert d["parent"] == lay["id"] and d["sub"] == "f:Ha/f:OIII" and d["index"] == 0 and "O III" in d["text"]
    monkeypatch.setattr(QApplication, "keyboardModifiers", staticmethod(lambda: Qt.ShiftModifier))
    d = pv.drop_at(b, cx, cy)
    assert d["sub"] == "f:Ha/f:OIII" and d["index"] == len([c for c in pe.children(pv.layers, lay["id"]) if c is not b])
    assert holder is not None


def test_the_real_context_menu_of_the_triangle_collapses_all_and_the_app_lives(pv):
    """The menu is the real popup (a nested loop), its action clicked as a user does: the panel is built again after
    its signal, the popup and its button are not deleted inside it."""
    from PySide6.QtCore import QTimer
    from PySide6.QtGui import QContextMenuEvent
    from PySide6.QtTest import QTest
    row = _row(pv, "starless")
    btn = row.toggle
    btn.show()
    seen = []

    def click():
        menu = QApplication.activePopupWidget()
        seen.append(menu is not None)
        if menu is None:
            return
        menu.setActiveAction(next(a for a in menu.actions() if a.text() == "Collapse all"))
        QTest.keyClick(menu, Qt.Key_Return)
    QTimer.singleShot(100, click)
    QTimer.singleShot(3000, lambda: QApplication.activePopupWidget() and QApplication.activePopupWidget().close())
    QApplication.sendEvent(btn, QContextMenuEvent(QContextMenuEvent.Mouse, QPoint(3, 3), btn.mapToGlobal(QPoint(3, 3))))
    _settle()
    assert seen == [True]
    assert [r.depth for r in pv.rows] == [0] and QApplication.activePopupWidget() is None
