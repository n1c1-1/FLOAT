"""The graph page with real events, second part: refusals and their messages, the palette's greyed cards, the
composition's wires, the global fold, the display control, the framing and the overview, the files dropped on a
Session's targets, the delete cross and the wire under the pointer (prototype scenarios s19-s42)."""
from __future__ import annotations

from PySide6.QtCore import QEvent, QMimeData, QPoint, Qt
from PySide6.QtGui import QDragEnterEvent, QDragLeaveEvent, QDragMoveEvent, QHelpEvent
from PySide6.QtWidgets import QApplication, QPushButton, QToolTip

from tests.client.graphview import test_gv_page as first
from tests.client.graphview.test_gv_page import fwd, make, pump, settle, template

page, B = first.page, first.B                                       # the fixtures of the first part

TEMPLATES = ("colour_minimal", "colour_complete", "filters_minimal", "filters_complete", "sho_rgb", "several_images",
             "blank")


# ======================================================================================================= refusals
def test_s19_drop_on_a_refused_place_changes_nothing_and_says_why(B):
    B.tpl("several_images")
    refused = [p for p in B.g.visible_places("color") if p["kind"] == "insert" and not p["lit"]]
    assert refused and all("Astrometry" in p["reason"] for p in refused)
    n, before = B.steps(), B.boxes()
    B.g.drag_palette("color", refused[0]["i"])
    assert B.boxes() == before and B.steps() == n
    assert "Astrometry" in B.toast() and not B.canvas.ghosts


def test_s20_d_switches_off_then_refuses_a_needed_block(B):
    B.tpl("colour_minimal")
    gr = B.one("gradient")
    B.g.select(gr)
    before = B.js()
    B.key("D")
    assert B.graph.blocks[gr].off and B.canvas.blocks[gr].d["off"]
    assert B.canvas.blocks[gr].d["summary"] == "off: image passed on"
    B.exact(before)
    a = B.one("astrometry")
    B.g.select(a)
    n = B.steps()
    B.key("D")
    assert not B.graph.blocks[a].off and B.steps() == n
    assert B.toast() == "needed by Color: switch Color off first"


def test_s21_del_on_a_needed_block_is_refused_with_its_reason(B):
    B.tpl("colour_minimal")
    a = B.one("astrometry")
    B.g.select(a)
    n, before = B.steps(), B.boxes()
    B.key("Del")
    assert a in B.graph.blocks and B.boxes() == before and B.steps() == n
    assert B.toast() == "needed by Color"


def test_s22_review_pause_from_the_menu(B):
    B.tpl("colour_minimal")
    gr = B.one("gradient")
    before = B.js()
    res = B.g.menu_pick(gr, "Review pause")
    assert res["ok"], res
    assert B.graph.blocks[gr].pause and B.canvas.blocks[gr].d["review"] == "pending"
    items = {t: (on, why) for t, on, why in B.g.menu_pick(gr, "", open_only=True)["items"]}
    assert items["Remove pause"][0] and items["Mark reviewed"] == (False, "after a stop on this block")
    B.exact(before)


# ======================================================================================================= the palette
def test_s24_palette_greys_a_card_and_says_why_on_hover(B):
    B.tpl("several_images")
    assert len(B.graph.rows) == 3 and len(B.post_wires()) == 3
    pal = B.page.palette
    pal.finish()
    card = pal.cards["denoise"]
    assert not card.enabled and "half-stacks" in card.reason
    QApplication.sendEvent(card, QHelpEvent(QEvent.ToolTip, QPoint(10, 10), card.mapToGlobal(QPoint(10, 10))))
    assert QToolTip.isVisible() and "Needs half-stacks" in QToolTip.text()     # the pointer resting on it
    QToolTip.hideText()
    assert pal.cards["gradient"].enabled and "half-stacks" not in pal.cards["gradient"].toolTip()
    assert any("half-stacks" in p["reason"] for p in B.g.visible_places("denoise") if not p["lit"])
    n = B.steps()
    B.g.double_click_palette("denoise")                               # a greyed card double-clicked: its reason
    assert B.steps() == n and "half-stacks" in B.toast()


# ======================================================================================================= composition
def test_s25_combination_to_the_composition_then_a_wire_into_its_port(B):
    B.tpl("filters_minimal")
    pl = next(p for p in B.g.visible_places("combine") if p["kind"] == "comp" and p["lit"])
    B.g.drag_palette("combine", pl["i"])
    joins = B.ids("combine")
    assert len(joins) == 2
    j = next(b for b in joins if not any(w.dst[0] == b for w in B.graph.wires))
    assert B.canvas.blocks[j].d["grey"] == "S II, Hα, O III to wire"
    ins = B.canvas.blocks[j].d["inputs"]
    assert len(ins) >= 2 and not any(wired for _n, _l, _y, wired, _c in ins)
    port = next(n for n, *_r in ins if n.startswith("H"))
    end = B.row("Ha").blocks[-1]
    before = B.js()
    B.g.pull_wire(end, "image", j, port)
    assert any(tuple(w.dst) == (j, port) and w.src[0] == end for w in B.graph.wires)
    assert dict((n, w) for n, _l, _y, w, _c in B.canvas.blocks[j].d["inputs"])[port]
    B.exact(before)


def test_s26_a_wire_closing_a_loop_turns_red_and_is_refused(B):
    B.tpl("filters_minimal")
    B.setup(("add_block", dict(kind="combine", at={"type": "comp"})))
    B.setup(("add_block", dict(kind="combine", at={"type": "comp"})))
    a, b = [x for x in B.ids("combine") if not any(w.dst[0] == x for w in B.graph.wires)][-2:]
    port_b = B.canvas.blocks[b].d["inputs"][0][0]
    B.setup(("wire", dict(src=a, src_port="image", dst=b, dst_port=port_b)))
    n, before = len(B.graph.wires), B.js()
    g = B.g
    port_a = B.canvas.blocks[a].d["inputs"][0][0]
    p0 = g.port_point(b, "image", "out")
    p1 = g.port_point(a, port_a, "in") + QPoint(6, 0)
    g.press(p0)
    for q in g._path(p0, p1):
        g.move(q)
    it = B.canvas.blocks[a]
    assert it.hot_input == (port_a, False)                            # the port under the wire is red
    assert "loop" in B.canvas.tip.text and B.canvas.tip.isVisible() and B.canvas.tip.refused
    g.release(p1)
    pump()
    assert len(B.graph.wires) == n and B.js() == before
    assert "loop" in B.toast() and B.canvas.rubber is None


# ======================================================================================================= the global fold
def test_s27_global_fold_keeps_the_order_and_is_never_an_undo_step(B):
    B.tpl("sho_rgb")
    n = B.steps()
    rows = [r.id for r in B.graph.rows]
    order = [x[0] for x in B.b.drawing().lines]
    B.g.display("folded")
    assert B.b.view.get("folded") is True
    d = B.b.drawing()
    assert d.cards and all(r.id in d.cards for r in B.graph.rows if r.nights)
    assert [x[0] for x in d.lines] == order and B.steps() == n
    B.key("Ctrl+Z")
    assert B.b.view.get("folded") is True                             # Ctrl+Z never unfolds
    B.g.click(rows[1])                                                # a click on a card selects it, never unfolds
    assert B.b.view.get("folded") is True and B.b.selected == rows[1] and B.canvas.blocks[rows[1]].d["selected"]
    B.g.display("blocks")
    assert B.b.view.get("folded") is False and not B.b.drawing().cards
    assert [x[0] for x in B.b.drawing().lines] == order and B.steps() == n and not B.overlaps()


def test_s28_merge_then_split_nights_each_undone_exactly(B):
    B.install().setup(("add_row", dict(kind="session")))
    B.setup(("add_night", dict(row=B.graph.rows[0].id)))
    s1, s2 = (x[0] for x in B.graph.rows[0].nights)
    B.setup(("add_files", dict(session=s1, kind="lights", paths=[f"E:/lights/L{i:03d}.fits" for i in range(40)])),
            ("add_files", dict(session=s2, kind="lights", paths=[f"E:/lights/M{i:03d}.fits" for i in range(30)])))
    before = B.js()
    assert B.b.act("merge_nights", a=s1, b=s2).ok and len(B.graph.rows[0].nights) == 1
    s = B.graph.rows[0].nights[0][0]
    assert "70 lights" in B.canvas.blocks[s].d["summary"]
    merged = B.js()
    lights = list(B.graph.blocks[s].files["lights"])
    assert B.b.act("split_night", night=s, lights=lights[-20:]).ok
    n1, n2 = (B.graph.blocks[x[0]].files["lights"] for x in B.graph.rows[0].nights)
    assert (len(n1), len(n2)) == (50, 20)
    split = B.js()
    B.key("Ctrl+Z")
    assert B.js() == merged
    B.key("Ctrl+Z")
    assert B.js() == before
    B.key("Ctrl+Y")
    B.key("Ctrl+Y")
    assert B.js() == split


# ======================================================================================================= the display control
def test_s39_one_display_control_and_ctrl_j_cycles(B):
    B.tpl("sho_rgb")
    p = B.page
    pos = {k: b.mapTo(p, b.rect().topLeft()) for k, b in p.positions.items()}
    assert [b.text() for b in p.positions.values()] == ["Blocks", "Folded", "Table"]
    assert all(b.isVisible() and not b.icon().isNull() for b in p.positions.values())
    assert p.position() == "blocks" and p.positions["blocks"].isChecked()
    assert B.b.view.get("folded") is not True and B.b.view.get("mode") == "graph"
    B.g.display("folded")
    assert B.b.view.get("folded") is True and p.centre.currentWidget() is p.canvas and p.positions["folded"].isChecked()
    B.g.display("table")
    assert B.b.view.get("mode") == "table" and p.centre.currentWidget() is p.table_host and p.table_host.isVisible()
    assert p.positions["table"].isChecked() and not p.palette.isVisible()
    assert all(b.mapTo(p, b.rect().topLeft()) == pos[k] for k, b in p.positions.items())      # never moves
    words = ("Table", "Graph", "Fold", "Unfold", "Folded", "Blocks", "Fold all", "Unfold all")
    assert {b for b in p.findChildren(QPushButton) if b.text() in words} == set(p.positions.values())
    B.g.display("blocks")
    assert p.position() == "blocks" and B.b.view.get("folded") is not True and p.centre.currentWidget() is p.canvas
    seen = []
    for _i in range(3):
        B.key("Ctrl+J", p.centre.currentWidget())
        seen.append(p.position())
    assert seen == ["folded", "table", "blocks"]
    assert [k for k, b in p.positions.items() if b.isChecked()] == ["blocks"]


def test_s29_display_control_on_blank_one_session_and_every_template(B):
    p = B.page
    for key in (None, "one") + TEMPLATES:
        B.install(template(key) if key not in (None, "one") else None)
        if key == "one":
            B.g.double_click_palette("session")
        B.g.display("table")
        assert p.centre.currentWidget() is p.table_host and p.table_host.isVisible(), key
        assert p.positions["table"].isChecked() and all(b.isVisible() and b.isEnabled() for b in p.positions.values())
        B.g.display("blocks")
        assert p.centre.currentWidget() is p.canvas and p.canvas.isVisible() and B.b.view.get("mode") == "graph", key


def test_s40_a_file_cell_of_the_table_in_the_right_pane(B, tmp_path):
    from astro_gpu.client.graphview.files_panel import FilesPanel
    from tests.client.graphview.test_gv_table import Bench as TableBench
    night = make(tmp_path / "night", 4, "ARW")
    B.install().setup(("add_row", dict(kind="session")))
    s = B.one("session")
    B.g.drop_files(s, "lights", [str(night)])
    settle(B)
    B.g.display("table")
    pump(0.05)
    tb = TableBench.__new__(TableBench)
    tb.g = B.page.table_host.table.grid
    tb.vp = tb.g.viewport()
    tb.click(("box", s, "lights"))
    assert B.b.file_cell == (s, "lights")
    fp = B.page.files_panel
    assert isinstance(fp, FilesPanel) and B.page.right.isVisible()
    settle(B)
    fp.refresh()
    assert fp.entries() == [fwd(night)]
    tb.click(("stack", B.graph.rows[0].id))                          # another cell: the pane goes
    assert B.b.file_cell is None and not B.page.right.isVisible() and B.page.files_panel is None
    B.g.display("blocks")


# ======================================================================================================= framing
def _big(B, n):  # noqa: N803
    """The complete colour template and n more Sessions, a row each."""
    B.install(template("colour_complete"))
    B.setup(*[("add_row", dict(kind="session"))] * n)


def test_s30_a_drop_far_below_shows_the_new_block_without_zooming(B):
    _big(B, 6)
    c = B.canvas
    k0 = c.zoom
    for kind in ("gradient", "denoise"):
        lit = [p["i"] for p in B.g.visible_places(kind) if p["kind"] == "branch" and p["lit"]]
        assert lit, kind
        old = set(B.graph.blocks)
        B.g.drag_palette(kind, lit[-1])                               # the lowest strip
        new = [b for b in set(B.graph.blocks) - old if B.graph.blocks[b].kind == kind]
        assert new and all(B.on_screen(b) for b in new), kind
        assert abs(c.zoom - k0) < 1e-9
    B.g.double_click_palette("session")
    assert B.b.selected in c.blocks and B.on_screen(B.b.selected) and abs(c.zoom - k0) < 1e-9


def test_s31_scene_rect_is_the_drawing_plus_a_margin(B):
    from astro_gpu.client.graphview.canvas import MARGIN
    _big(B, 8)
    c = B.canvas
    its = [i for i in c.scene().items() if i is not c.tip]
    bb = its[0].sceneBoundingRect()
    for i in its[1:]:
        bb = bb.united(i.sceneBoundingRect())
    sr = c.sceneRect()
    assert abs(sr.left() - (bb.left() - MARGIN)) < 1.5 and abs(sr.right() - (bb.right() + MARGIN)) < 1.5
    assert abs(sr.top() - (bb.top() - MARGIN)) < 1.5 and abs(sr.bottom() - (bb.bottom() + MARGIN)) < 1.5
    c.show_all()
    pump()
    view = c.mapToScene(c.viewport().rect()).boundingRect()
    tol = 2.5 / c.zoom + 1
    assert view.adjusted(-tol, -tol, tol, tol).contains(sr) or c.zoom <= 0.1001
    c.zoom_to(1.0)
    for sb in (c.horizontalScrollBar(), c.verticalScrollBar()):
        sb.setValue(sb.maximum())
    pump()
    view = c.mapToScene(c.viewport().rect()).boundingRect()
    assert sr.width() <= view.width() + 6 or abs(view.right() - sr.right()) < 6
    assert sr.height() <= view.height() + 6 or abs(view.bottom() - sr.bottom()) < 6
    for sb in (c.horizontalScrollBar(), c.verticalScrollBar()):
        sb.setValue(sb.minimum())
    view = c.mapToScene(c.viewport().rect()).boundingRect()
    assert abs(view.left() - sr.left()) < 6 and abs(view.top() - sr.top()) < 6
    B.install().setup(("add_row", dict(kind="session")))              # a small graph: nothing to scroll
    c.show_all()
    assert c.horizontalScrollBar().maximum() <= 2 and c.verticalScrollBar().maximum() <= 2


def test_s32_overview_only_when_needed_and_never_during_a_drag(B):
    from astro_gpu.client.graphview.canvas import MIME_KIND
    c = B.canvas
    B.install().setup(("add_row", dict(kind="session")))
    c.zoom_to(1.0)
    pump()
    assert not c.minimap.isVisible()
    _big(B, 14)
    c.zoom_to(1.0)
    pump()
    assert c.minimap.isVisible() and c.minimap.hover is False        # translucent while the pointer is elsewhere
    mime = QMimeData()
    mime.setData(MIME_KIND, b"gradient")
    vp = c.viewport()
    pt = QPoint(vp.width() // 2, 40)
    QApplication.sendEvent(vp, QDragEnterEvent(pt, Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier))
    QApplication.sendEvent(vp, QDragMoveEvent(pt, Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier))
    assert not c.minimap.isVisible()
    QApplication.sendEvent(vp, QDragLeaveEvent())
    pump()
    assert c.minimap.isVisible()
    lit = next(p["i"] for p in B.g.visible_places("gradient") if p["lit"])
    B.g.drag_palette("gradient", lit)
    assert c.minimap.isVisible()
    a = B.g.block_point(B.ids("gradient")[0])
    B.g.press(a)
    B.g.move(a + QPoint(40, 30))
    B.g.move(a + QPoint(80, 50))
    assert not c.minimap.isVisible()
    B.g.release(a + QPoint(80, 50))
    pump()
    assert c.minimap.isVisible()


# ======================================================================================================= files on the graph
def _session(B):  # noqa: N803
    B.install().setup(("add_row", dict(kind="session")))
    return B.one("session")


def test_s33_files_filtered_by_type_take_all_one_undo_step(B, tmp_path):
    mixed = make(tmp_path / "mixed", 5, "ARW")
    make(mixed, 2, "JPG")
    s = _session(B)
    n0 = B.steps()
    B.g.drop_files(s, "lights", [str(mixed)])
    settle(B)
    assert "5 ARW added" in B.toast() and "2 JPG left" in B.toast()
    assert {"take_all", "undo"} <= set(B.link_keys())
    assert B.steps() == n0 + 1 and B.b.doc.files_of(s, "lights")[1] == [fwd(mixed)]
    B.canvas.toast.link.emit("take_all")                              # the toast's « Take all » link
    settle(B)
    assert B.b.drops.count(B.b.doc.files_of(s, "lights")[1]) == (7, True)
    B.key("Ctrl+Z")
    assert B.b.drops.count(B.b.doc.files_of(s, "lights")[1]) == (5, True)
    s2 = _session(B)                                                  # single files: the filter applies at once
    B.g.drop_files(s2, "darks", [str(mixed / "IMG_0000.ARW"), str(mixed / "IMG_0000.JPG")])
    assert B.b.doc.files_of(s2, "darks")[1] == [fwd(mixed / "IMG_0000.ARW")]


def test_s34_sub_folders_not_read_include_and_no_duplicate(B, tmp_path):
    night = make(tmp_path / "night", 5, "ARW")
    make(night / "darks", 3, "ARW")
    make(night / "flats", 3, "ARW")
    s = _session(B)
    n0 = B.steps()
    B.g.drop_files(s, "lights", [str(night)])
    settle(B)
    assert B.b.drops.count(B.b.doc.files_of(s, "lights")[1]) == (5, True)       # the direct files only
    assert "5 ARW added" in B.toast() and "2 sub-folders not read" in B.toast() and "include" in B.link_keys()
    B.canvas.toast.link.emit("include")
    settle(B)
    lst = B.b.doc.files_of(s, "lights")[1]
    assert B.b.drops.count(lst) == (11, True) and B.steps() == n0 + 2
    assert lst == [fwd(night), fwd(night / "darks"), fwd(night / "flats")]   # each sub-folder an entry of its own
    before = B.js()
    B.g.drop_files(s, "lights", [str(night)])
    assert B.toast().startswith("Already there: nothing added")
    assert B.js() == before and B.steps() == n0 + 2


def test_s35_a_folder_without_direct_files_leaves_no_entry(B, tmp_path):
    only = tmp_path / "only"
    make(only / "a", 3, "ARW")
    make(only / "b", 3, "ARW")
    night = make(tmp_path / "night", 5, "ARW")
    s = _session(B)
    n0 = B.steps()
    B.g.drop_files(s, "lights", [str(only)])
    settle(B)
    assert not B.b.doc.files_of(s, "lights")[1] and B.steps() == n0
    assert "No direct file · 2 sub-folders" in B.toast() and B.link_keys() == ["include"]
    B.canvas.toast.link.emit("include")
    settle(B)
    assert B.b.drops.count(B.b.doc.files_of(s, "lights")[1]) == (6, True)
    s = _session(B)                                                   # one folder with files, one without
    n0 = B.steps()
    B.g.drop_files(s, "lights", [str(night), str(only)])
    settle(B)
    lst = B.b.doc.files_of(s, "lights")[1]
    assert lst == [fwd(night)] and B.steps() == n0 + 1


# ======================================================================================================= delete cross, wires
def test_s41_refused_delete_cross_frames_the_dependants_and_deletes_them_too(B):
    from astro_gpu.client.graphview.canvas import ALERT_MS
    B.tpl("colour_minimal")
    a, c = B.one("astrometry"), B.one("color")
    assert B.canvas.blocks[a].d["removable"]                          # the cross is there, refused or not
    n0, before = B.steps(), B.js()
    B.g.click_cross(a)
    assert a in B.graph.blocks and B.steps() == n0
    assert B.canvas.blocks[c].alert and not B.canvas.blocks[a].alert
    assert B.toast() == "needed by Color" and B.canvas.toast.links() == [("Delete too", "delete_too")]
    assert ALERT_MS == 1500
    pump(1.6)
    assert not B.canvas.blocks[c].alert
    B.canvas.toast.link.emit("delete_too")
    assert a not in B.graph.blocks and c not in B.graph.blocks and B.steps() == n0 + 1
    B.exact(before)
    cal = B.one("calibration")                                        # deleted like any block: the lights taken
    before = B.js()                                                   # as they are
    B.g.select(cal)
    B.key("Del")
    assert cal not in B.graph.blocks and B.steps() == n0 + 2
    assert "taken as they are" in B.toast()
    B.exact(before)


def test_s41b_a_dependant_fed_by_a_wire_is_framed_and_deleted_too(B):
    B.tpl("sho_rgb")
    st, cl = B.one("stars", "SII"), B.one("combine_layers")
    n0, before = B.steps(), B.js()
    B.g.click_cross(st)
    assert st in B.graph.blocks and B.steps() == n0
    assert B.toast() == "Combine layers needs it" and B.canvas.toast.links() == [("Delete too", "delete_too")]
    assert B.canvas.blocks[cl].alert and not B.canvas.blocks[st].alert
    B.canvas.toast.link.emit("delete_too")
    assert st not in B.graph.blocks and cl not in B.graph.blocks and B.steps() == n0 + 1
    B.exact(before)


def test_s42_a_wire_under_the_pointer_lights_its_ends_and_blocks(B):
    B.tpl("sho_rgb")
    c = B.canvas
    pt = it = None
    for cand in c.wires.values():                                     # a wire with a point nothing else covers
        if not (cand.ends and cand.ends[0] in c.blocks and cand.ends[1] in c.blocks):
            continue
        for pc in (0.5, 0.35, 0.65, 0.2, 0.8):
            q = c.mapFromScene(cand.path().pointAtPercent(pc))
            if c.viewport().rect().contains(q) and c.items(q) and c.items(q)[0] is cand:
                pt, it = q, cand
                break
        if it is not None:
            break
    assert it is not None
    B.g.move(pt + QPoint(1, 0), held=False)
    B.g.move(pt, held=False)
    pump()
    assert it.hot and it.zValue() > 2
    assert all(c.blocks[b].wire_hot for b in it.ends)
    assert sum(1 for x in c.blocks.values() if x.wire_hot) == 2
    B.g.move(QPoint(3, 3), held=False)
    pump()
    assert not it.hot and not any(x.wire_hot for x in c.blocks.values())

