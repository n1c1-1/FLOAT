"""The table alone (graphview/table.py) on the real core, with real Qt events (drag enter/move/drop with file URLs,
mouse, keys), offscreen, on real temporary folders. Ported from the validated mock-up's scenarios: widths, one undo
step per gesture, Common, the file-type filter, sub-folders not read by default, doubles, empty folders leaving no
entry, the keyboard, the structure gestures; nothing is ever guessed from a name."""
from __future__ import annotations

import time
from pathlib import Path

import pytest
from PySide6.QtCore import QEvent, QMimeData, QPoint, QPointF, Qt, QUrl
from PySide6.QtGui import QDragEnterEvent, QDragLeaveEvent, QDragMoveEvent, QDropEvent, QKeyEvent, QMouseEvent
from PySide6.QtWidgets import QApplication

from astro_gpu.graphcore.operations import COMMON

SIZES = {1366: (1046, 622), 1920: (1540, 934)}       # the table's widget in a 1366 × 768 / 1920 × 1080 window


def make(folder: Path, n: int, prefix="IMG", ext="fit", size=64) -> Path:
    """n small files <prefix>_0000.<ext>... in folder (made if needed)."""
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        (folder / f"{prefix}_{i:04d}.{ext}").write_bytes(bytes(size))
    return folder


def fwd(p) -> str:
    return str(p).replace("\\", "/")


class Bench:
    """A Table on a bridge held in memory, sized as in the window, driven with real events."""

    def __init__(self, width=1366, blank=False, sessions=1, delay=0.0, bridge=None, host=False):
        from astro_gpu.client.graphview.bridge import Bridge
        from astro_gpu.client.graphview.document import MemoryDocument
        from astro_gpu.graphcore import api
        if bridge is None:
            g = api.blank_project()
            if not blank:
                g, _r = api.apply(g, "add_row", kind="session")
                for _i in range(sessions - 1):
                    g, _r = api.apply(g, "add_night", row=g.rows[0].id)
            bridge = Bridge(MemoryDocument(g))
        self.bridge = bridge
        self.reader = bridge.drops.reader
        self.reader.delay = delay
        if host:
            from astro_gpu.client.graphview.table_host import TableHost
            self.host = TableHost(bridge)
            self.host.resize(*SIZES[width])
            self.host.show()
            self.wait(0.02)
            self.t = self.host.table
        else:
            from astro_gpu.client.graphview.table import Table
            self.host = None
            self.t = Table(bridge)
            self.t.resize(*SIZES[width])
            self.t.show()
        self.g = self.t.grid
        self.vp = self.g.viewport()
        self.g.choose_files = lambda title: []                 # never a real file dialog in tests
        self.wait(0.02)

    def close(self):
        (self.host or self.t).hide()

    # ---- the document
    @property
    def doc(self):
        return self.bridge.doc

    def steps(self):
        return len(self.doc.document._undo)

    def sessions(self):
        return [s["sid"] for pl in self.g.model.stacks() for s in pl["sessions"]]

    def stacks(self):
        return self.g.model.stacks()

    def box(self, owner, kind):
        self.g.model.invalidate()
        return self.g.model.box(owner, kind)

    def own(self, sid, kind):
        return self.g.model.own_list(sid, kind)

    # ---- time
    def wait(self, s=0.0):
        t0 = time.perf_counter()
        while True:
            QApplication.processEvents()
            if time.perf_counter() - t0 >= s:
                break
            time.sleep(0.005)
        QApplication.processEvents()

    def read(self, timeout=30):
        """Wait until the reader has finished and its signals have been delivered."""
        for _i in range(3):
            assert self.reader.wait(timeout)
            self.wait(0.05)

    # ---- positions
    def pos(self, ident, fx=0.5, fy=0.5):
        c = self.g.cell(ident)
        assert c is not None, f"no cell {ident}"
        self.g.ensure_visible(c)
        QApplication.processEvents()
        if c.pinned:
            return QPoint(int(c.x + fx * c.w), int(c.y + fy * c.h))
        from astro_gpu.client.graphview.table_paint import HEAD
        y = c.y - self.g.scroll_y() + HEAD + self.g.common_h
        return QPoint(int(c.x + fx * c.w), int(y + fy * min(c.h, 40)))

    # ---- drag and drop with real events
    def _mime(self, paths, internal):
        mime = QMimeData()
        if internal:
            from astro_gpu.client.graphview.table_gestures import INTERNAL_MIME
            mime.setData(INTERNAL_MIME, b"1")
            self.g.start_internal(internal)
        else:
            mime.setUrls([QUrl.fromLocalFile(str(p)) for p in paths])
        self.mime = mime
        return mime

    def drop(self, paths, ident, fx=0.25, fy=0.5, ctrl=False, internal=None):
        mime = self._mime(paths, internal)
        mods = Qt.ControlModifier if ctrl else Qt.NoModifier
        p = self.pos(ident, fx, fy)
        acts = Qt.CopyAction | Qt.MoveAction
        QApplication.sendEvent(self.vp, QDragEnterEvent(QPoint(p.x() + 40, p.y() + 20), acts, mime, Qt.LeftButton, mods))
        QApplication.sendEvent(self.vp, QDragMoveEvent(p, acts, mime, Qt.LeftButton, mods))
        d = QDropEvent(QPointF(p), acts, mime, Qt.LeftButton, mods)
        QApplication.sendEvent(self.vp, d)
        QApplication.processEvents()
        return d.isAccepted()

    def hover(self, paths, ident, fx=0.25, fy=0.5, ctrl=False, internal=None):
        """Enter and move over a cell without dropping: -> the hot target. Call end_hover() after."""
        mime = self._mime(paths, internal)
        mods = Qt.ControlModifier if ctrl else Qt.NoModifier
        p = self.pos(ident, fx, fy)
        acts = Qt.CopyAction | Qt.MoveAction
        QApplication.sendEvent(self.vp, QDragEnterEvent(p, acts, mime, Qt.LeftButton, mods))
        QApplication.sendEvent(self.vp, QDragMoveEvent(p, acts, mime, Qt.LeftButton, mods))
        QApplication.processEvents()
        return self.g.hot

    def end_hover(self):
        QApplication.sendEvent(self.vp, QDragLeaveEvent())

    # ---- mouse, keys
    def click(self, ident, fx=0.5, fy=0.5, zone=None, mods=Qt.NoModifier):
        from astro_gpu.client.graphview.table_paint import HEAD
        c = self.g.cell(ident)
        assert c is not None, f"no cell {ident}"
        if zone:
            self.g.ensure_visible(c)
            r = self.g.zones(c)[zone]
            y = r.center().y() - (0 if c.pinned else self.g.scroll_y() - HEAD - self.g.common_h)
            p = QPoint(int(r.center().x()), int(y))
        else:
            p = self.pos(ident, fx, fy)
        for typ in (QEvent.MouseButtonPress, QEvent.MouseButtonRelease):
            QApplication.sendEvent(self.vp, QMouseEvent(typ, QPointF(p), QPointF(self.vp.mapToGlobal(p)), Qt.LeftButton,
                                                        Qt.LeftButton if typ == QEvent.MouseButtonPress else Qt.NoButton,
                                                        mods))
        QApplication.processEvents()

    def key(self, key, mods=Qt.NoModifier, widget=None):
        for typ in (QEvent.KeyPress, QEvent.KeyRelease):
            QApplication.sendEvent(widget or self.g, QKeyEvent(typ, key, mods, ""))
        QApplication.processEvents()


@pytest.fixture()
def tree(tmp_path):
    """Real folders: a night (lights, darks, flats as sub-folders), raws and JPEGs together, sub-folders only."""
    f = {}
    night = tmp_path / "night"
    f["lights"] = make(night / "LIGHT", 12, "Light")
    f["darks"] = make(night / "DARK_300s", 5, "Dark")
    f["flats"] = make(night / "FLAT", 4, "Flat")
    f["night"] = night
    f["flats2"] = make(tmp_path / "flats_good", 4, "Flat")
    f["more"] = [make(tmp_path / f"night_{i}" / "LIGHT", 6, "Light") for i in range(3)]
    mixed = tmp_path / "arw_jpg"
    make(mixed, 5, "IMG", "ARW")
    make(mixed, 5, "IMG", "JPG")
    f["mixed"] = mixed
    sub = tmp_path / "with_sub"
    make(sub, 5, "IMG", "ARW")
    make(sub / "darks", 3, "D", "ARW")
    make(sub / "flats", 3, "F", "ARW")
    f["sub"] = sub
    only = tmp_path / "only_sub"
    make(only / "a", 3, "S", "ARW")
    make(only / "b", 3, "S", "ARW")
    f["only_sub"] = only
    f["lib"] = make(tmp_path / "lib", 6, "Dark")
    return f


# ----------------------------------------------------------------------------------------------- widths
def test_gv_table_widths_1366(qapp):
    b = Bench(1366)
    g = b.g
    assert g.viewport().width() == 1034
    assert sum(w for _x, w in g.xs.values()) == 1034
    assert list(g.xs).index("bias") == list(g.xs).index("flats") + 1      # Bias: always a column, beside Flats
    assert "flat_darks" not in g.xs and g.xs["bias"][1] == 96
    assert g.body_h() == 460                                         # Common compact with one session
    assert not g.horizontalScrollBar().isVisible()
    b.t.column("flat_darks", True)                                   # « + columns »
    assert sum(w for _x, w in g.xs.values()) == 1034 and g.xs["stk"][1] == 110 and g.xs["bias"][1] == 88
    b.t.column("flat_darks", False)
    assert "flat_darks" not in g.xs
    b.close()


def test_gv_table_widths_1920(qapp):
    b = Bench(1920)
    g = b.g
    assert g.viewport().width() + g.verticalScrollBar().width() == 1540
    assert g.xs["stack"][1] == 150 and g.xs["session"][1] == 140 and g.xs["lights"][1] == 150
    assert sum(w for _x, w in g.xs.values()) == g.viewport().width()
    b.close()


def test_gv_table_eleven_rows_at_1366(qapp):
    b = Bench(1366, sessions=11)
    g = b.g
    assert g.common_h == 48 and g.body_h() == 444                    # Common full with several sessions
    names = [c for c in g.cells if c.kind == "name"]
    assert len([c for c in names if c.y + 40 <= g.body_h()]) == 11
    b.close()


# ----------------------------------------------------------------------------------------------- one folder per cell
def test_gv_table_one_folder_per_cell(qapp, tree):
    b = Bench(1920)
    sid = b.sessions()[0]
    assert b.g.common_h == 32
    hot = b.hover([tree["lights"]], ("box", sid, "lights"))
    assert hot.ok and hot.label == "1 folder → Session 1 (lights)"
    b.end_hover()
    n0 = b.steps()
    b.drop([tree["lights"]], ("box", sid, "lights"))
    assert b.box(sid, "lights").mode == "own"                        # the box reacts at once
    b.drop([tree["darks"]], ("box", COMMON, "darks"))
    b.drop([tree["flats"]], ("box", sid, "flats"))
    b.read()
    assert b.steps() == n0 + 3                                       # one undo step per drop
    assert b.box(sid, "lights").n == 12 and b.box(COMMON, "darks").n == 5 and b.box(sid, "flats").n == 4
    assert b.box(sid, "darks").mode == "inherited" and b.box(sid, "darks").l1 == "↑ common · 5"
    for _i in range(3):
        b.key(Qt.Key_Z, Qt.ControlModifier)
    assert not b.doc.graph.blocks[sid].files and not any(b.doc.graph.common["files"].values())
    b.close()


def test_gv_table_whole_night_on_a_cell_reads_no_sub_folder(qapp, tree):
    """A night dropped on a cell goes there; its sub-folders are not read; nothing is sorted by their names."""
    b = Bench(1920)
    sid = b.sessions()[0]
    n0 = b.steps()
    b.drop([tree["night"]], ("box", sid, "lights"))
    b.read()
    assert not b.own(sid, "lights") and b.steps() == n0              # no direct file: no empty entry left
    assert "No direct file · 3 sub-folders" in b.t.toast_text
    assert "include" in b.t.toast_w.text() and "undo" not in b.t.toast_w.text()
    b.t.link("include")
    b.read()
    assert b.box(sid, "lights").n == 21 and b.steps() == n0 + 1       # every level read, still in the cell dropped on
    assert not any(b.doc.graph.common["files"].values()) and set(b.doc.graph.blocks[sid].files) == {"lights"}
    b.close()


def test_gv_table_new_sessions_and_common(qapp, tree):
    b = Bench(1920)
    sid = b.sessions()[0]
    rid = b.stacks()[0]["id"]
    b.drop([tree["more"][0]], ("box", sid, "lights"))
    n0 = b.steps()
    hot = b.hover(tree["more"][1:], ("ghost_s", rid))
    assert hot.label == "2 folders → 2 new sessions"
    b.end_hover()
    b.drop(tree["more"][1:], ("ghost_s", rid))
    ss = b.sessions()
    assert len(ss) == 3 and b.steps() == n0 + 1                      # one session per folder, one undo step
    b.drop([tree["lib"]], ("box", COMMON, "darks"))
    b.read()
    assert all(b.box(s, "lights").n == 6 for s in ss)
    assert b.box(COMMON, "darks").n == 6 and all(b.box(s, "darks").mode == "inherited" for s in ss)
    assert b.g.common_h == 48
    b.close()


def test_gv_table_new_stacks_and_own_setting(qapp, tree):
    b = Bench(1366)
    for folder in tree["more"]:
        b.drop([folder], ("ghost_p",))
    b.read()
    ps = b.stacks()
    assert len(ps) == 4
    g = b.g
    ha = ps[1]
    g.set_value(("setting", ha["id"], "stk"), "Rejection threshold", "8 σ")
    g.set_value(("setting", ha["id"], "stk"), "Method", "Standard stacking")
    t, own, _kind, _blk = g.cell_text(g.cell(("setting", ha["id"], "stk")))
    assert own and t == "classic · rejection at 8 σ"
    assert "Own setting for" in b.t.toast_text or g.is_own(g.cell(("setting", ha["id"], "stk")))
    b.close()


def test_gv_table_blank_body_and_no_flats_warning(qapp, tree):
    b = Bench(1920, blank=True)
    assert b.g.cell(("body",)) is not None                           # an empty project: the whole body is a target
    hot = b.hover(tree["more"][:2], ("body",))
    assert hot.kind == "ghost_p" and hot.label == "2 folders → new stack, 2 sessions"
    b.end_hover()
    n0 = b.steps()
    b.drop(tree["more"][:2], ("body",))
    assert len(b.stacks()) == 1 and len(b.sessions()) == 2 and b.steps() == n0 + 1
    s1, s2 = b.sessions()
    b.drop([tree["flats"]], ("box", s1, "flats"))
    b.read()
    bf = b.box(s2, "flats")
    assert bf.mode == "empty" and bf.warn is not None and bf.warn.text == "No flats: the other sessions have some"
    assert b.g.cell(("warn", s2, "flats")) is not None                # the amber line under the row
    b.click(("warn", s2, "flats"), zone="button")                     # « Leave »
    assert b.box(s2, "flats").warn is None and b.box(s2, "flats").corner
    b.close()


def test_gv_table_replace_half_and_rather_add(qapp, tree):
    b = Bench(1920)
    sid = b.sessions()[0]
    b.drop([tree["flats"]], ("box", sid, "flats"))
    b.read()
    hot = b.hover([tree["flats2"]], ("box", sid, "flats"), fx=0.8)
    assert hot.half == "replace" and "replaces the 4 of “Session 1”" in hot.label
    b.end_hover()
    n0 = b.steps()
    b.drop([tree["flats2"]], ("box", sid, "flats"), fx=0.8)
    b.read()
    assert b.own(sid, "flats") == [fwd(tree["flats2"])] and b.steps() == n0 + 1
    assert "replace" in b.t.toast_text and "rather_add" in b.t.toast_w.text()
    b.t.link("rather_add")
    b.read()
    assert b.box(sid, "flats").n == 8 and b.steps() == n0 + 1        # undone, then added: still one more step
    b.close()


# ----------------------------------------------------------------------------------------------- Common, states
def test_gv_table_common_states(qapp, tree):
    b = Bench(1366, sessions=5)
    g = b.g
    ss = b.sessions()
    b.drop([tree["lib"]], ("box", ss[0], "darks"))
    b.drop([tree["darks"]], ("box", ss[1], "darks"))
    b.read()
    hot = b.hover([tree["darks"]], ("box", COMMON, "darks"))
    assert "for 3 sessions of 5 (2 have their own)" in hot.label
    b.end_hover()
    b.drop([tree["darks"]], ("box", COMMON, "darks"))
    b.read()
    assert all(b.box(s, "darks").mode == "inherited" for s in ss[2:])
    assert b.own(ss[1], "darks") is None                             # a list equal to Common's follows it again
    assert b.box(ss[0], "darks").bar                                 # an own list beside Common's: the blue bar
    b.click(("box", ss[0], "darks"))
    g.hover = ("box", ss[0], "darks")
    b.click(("box", ss[0], "darks"), zone="reset")                   # ↺
    assert b.box(ss[0], "darks").mode == "inherited" and not b.box(ss[0], "darks").bar
    b.click(("box", COMMON, "darks"))
    b.key(Qt.Key_Delete)
    assert b.box(COMMON, "darks").mode == "empty" and all(b.box(s, "darks").mode == "empty" for s in ss)
    assert "Darks of Common emptied: 5 sessions have none left" in b.t.toast_text
    assert "undo" in b.t.toast_w.text()
    b.key(Qt.Key_Z, Qt.ControlModifier)
    assert b.box(COMMON, "darks").mode == "own"
    cell = ("setting", b.stacks()[0]["id"], "stk")
    g.set_value(cell, "Rejection threshold", "8 σ")
    assert g.is_own(g.cell(cell))
    g.set_value(cell, "Rejection threshold", "3 σ")                  # equal to Common: follows it again
    assert not g.is_own(g.cell(cell)) and "follows Common again" in b.t.toast_text
    g.set_value(cell, "Rejection threshold", "8 σ")
    b.click(cell)
    g.hover = cell
    b.click(cell, zone="reset")
    assert not g.is_own(g.cell(cell))
    n0 = b.steps()
    g.set_value(("setting", COMMON, "stk"), "Rejection threshold", "5 σ")
    assert b.steps() == n0 + 1 and "Common Stacking changed" in b.t.toast_text
    b.close()


def test_gv_table_batch_bar(qapp, tree):
    b = Bench(1366, sessions=3)
    ss = b.sessions()
    b.drop([tree["flats"]], ("box", ss[1], "flats"))
    b.drop([tree["flats"]], ("box", ss[2], "flats"))
    b.read()
    b.click(("name", ss[1]))
    b.click(("name", ss[2]), mods=Qt.ControlModifier)
    assert b.t.bar.currentIndex() == 1 and b.t.l_batch.text() == "2 sessions chosen"
    assert set(b.t.batch_buttons) == {"stack", "move", "common", "delete", "x"}
    b.t.batch_buttons["common"].click()
    assert b.own(ss[1], "flats") is None and b.own(ss[2], "flats") is None
    assert "2 items back to Common" in b.t.toast_text
    b.close()


def test_gv_table_merge_preview(qapp):
    b = Bench(1366)
    b.g.act([("add_row", dict(kind="session"))])
    ps = b.stacks()
    b.g.set_value(("setting", ps[1]["id"], "stk"), "Rejection threshold", "8 σ")
    hot = b.hover([], ("stack", ps[0]["id"]), internal=("stack", ps[1]["id"]), fx=0.3)
    assert hot.kind == "merge" and "1 session joins “Session 1”" in hot.label
    assert "own settings dropped: Rejection threshold 8 σ" in hot.label
    b.end_hover()
    n0 = b.steps()
    b.drop([], ("stack", ps[0]["id"]), internal=("stack", ps[1]["id"]), fx=0.3)
    assert len(b.stacks()) == 1 and len(b.stacks()[0]["sessions"]) == 2 and b.steps() == n0 + 1
    b.key(Qt.Key_Z, Qt.ControlModifier)
    assert len(b.stacks()) == 2
    b.close()


# ----------------------------------------------------------------------------------------------- drag feedback
def test_gv_table_drag_labels_and_refusals(qapp, tree):
    b = Bench(1366, sessions=2)
    g = b.g
    s1, s2 = b.sessions()
    b.drop([tree["lights"]], ("name", s1))
    b.read()
    hot = b.hover([tree["lib"]], ("box", s2, "darks"))
    assert hot.ok and hot.label == "1 folder → Darks of “Session 2”" and g.drag is not None
    b.end_hover()
    hot = b.hover([tree["lib"], tree["darks"]], ("box", s1, "lights"), fx=0.8)
    assert hot.half == "replace" and hot.label.startswith("2 folders → replaces the")
    b.end_hover()
    assert b.hover([tree["lib"], tree["darks"], tree["flats"]], ("name", s1)).label == "3 folders → Session 1 (lights)"
    b.end_hover()
    assert b.hover([tree["lib"]], ("ghost_s", b.stacks()[0]["id"])).label == "1 folder → new session"
    b.end_hover()
    assert b.hover([tree["lib"]], ("ghost_p",)).label == "1 folder → new stack"
    b.end_hover()
    n0 = b.steps()
    assert not b.hover([tree["lib"]], ("box", COMMON, "lights")).ok
    b.end_hover()
    b.drop([tree["lib"]], ("box", COMMON, "lights"))
    assert b.steps() == n0 and "Nothing dropped: lights go on a session, not on Common" in b.t.state.text()
    b.drop([tree["lib"]], ("setting", COMMON, "stk"))
    assert b.steps() == n0 and "Nothing dropped" in b.t.state.text()
    g.model.folded.add(b.stacks()[0]["id"])
    g.after_model()
    hot = b.hover([tree["lib"]], ("box", "folded:" + b.stacks()[0]["id"], "darks"))
    assert not hot.ok and "Unfold the stack" in hot.label
    b.end_hover()
    g.model.folded.clear()
    g.after_model()
    # a drop on a chosen box goes to every chosen box of the column: one undo step
    b.click(("box", s1, "darks"))
    b.click(("box", s2, "darks"), mods=Qt.ControlModifier)
    assert b.hover([tree["lib"]], ("box", s1, "darks")).label == "1 folder → 2 chosen boxes"
    b.end_hover()
    n0 = b.steps()
    b.drop([tree["lib"]], ("box", s1, "darks"))
    b.read()
    assert b.box(s1, "darks").n == 6 and b.box(s2, "darks").n == 6 and b.steps() == n0 + 1
    # a filled box dragged onto another moves its list; Ctrl copies
    b.drop([], ("box", s2, "flats"), internal=("box", s1, "darks"))
    assert b.own(s2, "flats") == [fwd(tree["lib"])] and not b.own(s1, "darks")
    b.key(Qt.Key_Z, Qt.ControlModifier)
    assert "Copy" in b.hover([], ("box", s2, "flats"), internal=("box", s1, "darks"), ctrl=True).label
    b.end_hover()
    b.close()


# ----------------------------------------------------------------------------------------------- reading
def test_gv_table_reading_and_stop(qapp, tree):
    b = Bench(1366, delay=0.6)
    sid = b.sessions()[0]
    b.drop([tree["lights"]], ("box", sid, "lights"))
    b.wait(0.15)
    bx = b.box(sid, "lights")
    assert bx.reading == "Reading…" and bx.progress is not None
    assert b.t.counter.text().startswith("Reading…") or b.g.model.readings()
    assert "stop" in b.g.zones(b.g.cell(("box", sid, "lights")))
    b.click(("box", sid, "lights"), zone="stop")
    b.reader.delay = 0.0
    b.read()
    assert not b.own(sid, "lights")                                  # × takes the drop back (one undo step)
    b.key(Qt.Key_Y, Qt.ControlModifier)
    assert b.own(sid, "lights")
    b.close()


def test_gv_table_drop_answers_at_once(qapp, tmp_path):
    loose = sorted(make(tmp_path / "many", 600, "L").glob("*.fit"))
    b = Bench(1366)
    sid = b.sessions()[0]
    t0 = time.perf_counter()
    b.drop(loose, ("box", sid, "lights"))
    assert (time.perf_counter() - t0) < 2.0
    assert b.box(sid, "lights").n == 600                             # counted at once (loose files)
    b.read()
    b.close()


# ----------------------------------------------------------------------------------------------- the type filter
def test_gv_table_type_filter(qapp, tree):
    from astro_gpu.client import prefs
    from astro_gpu.client.graphview import files
    files.set_index(2)
    b = Bench(1366)
    assert b.t.c_files.currentText() == "Files: Raws and FITS" and b.t.c_files.toolTip()
    sid = b.sessions()[0]
    n0 = b.steps()
    b.drop([tree["mixed"]], ("box", sid, "lights"))
    b.read()
    assert b.steps() == n0 + 1 and b.box(sid, "lights").n == 5
    assert all(f.path.endswith(".ARW") for f in b.reader.expand(tuple(b.own(sid, "lights"))))
    assert "5 ARW added · 5 JPG left" in b.t.toast_text and "take_all" in b.t.toast_w.text()
    b.t.link("take_all")
    b.read()
    assert b.steps() == n0 + 2 and b.box(sid, "lights").n == 10
    b.key(Qt.Key_Z, Qt.ControlModifier)
    assert b.box(sid, "lights").n == 5
    assert b.t.c_files.count() == 4                                  # no « All files »: the engine reads frames only
    b.t.c_files.setCurrentIndex(3)
    b.t.c_files.activated.emit(3)
    assert files.index() == 3 and prefs.get("graph_file_filter") == "All images"
    b.close()
    b = Bench(1366)
    assert b.t.c_files.currentIndex() == 3                           # remembered
    sid = b.sessions()[0]
    b.drop([tree["mixed"]], ("box", sid, "lights"))
    b.read()
    # JPEGs beside raws: the engine would set them aside, so the ten files are stored one by one
    assert b.box(sid, "lights").n == 10 and "left" not in b.t.toast_text
    assert fwd(tree["mixed"]) not in b.own(sid, "lights") and len(b.own(sid, "lights")) == 10
    b.close()
    files.set_index(2)


def test_gv_table_doubles_not_added(qapp, tree):
    b = Bench(1366)
    sid = b.sessions()[0]
    b.drop([tree["lights"]], ("box", sid, "lights"))
    b.read()
    n0 = b.steps()
    b.drop([tree["lights"]], ("box", sid, "lights"))
    assert b.steps() == n0 and b.t.toast_text == "Already there: nothing added"
    b.close()


# ----------------------------------------------------------------------------------------------- sub-folders
def test_gv_table_sub_folders_not_read(qapp, tree):
    from astro_gpu.client.graphview import files
    from astro_gpu.client.graphview.bridge import Bridge
    from astro_gpu.client.graphview.document import MemoryDocument
    files.set_index(2)
    b = Bench(1366)
    sid = b.sessions()[0]
    n0 = b.steps()
    b.drop([tree["sub"]], ("box", sid, "lights"))
    b.read()
    assert b.box(sid, "lights").n == 5 and b.steps() == n0 + 1
    assert "5 ARW added" in b.t.toast_text and "2 sub-folders not read" in b.t.toast_text
    assert "include" in b.t.toast_w.text()
    tip = b.g.tooltip_text(b.pos(("box", sid, "lights")))
    assert "folder · without sub-folders" in tip
    assert "file_options" not in b.doc.view                         # nothing about the filter is stored
    # the project reopens with the same files whatever the filter set then
    files.set_index(0)
    b2 = Bench(1366, bridge=Bridge(MemoryDocument(b.doc.graph, view=dict(b.doc.view))))
    b2.read()
    assert b2.box(sid, "lights").n == 5
    b2.close()
    files.set_index(2)
    b.t.link("include")
    b.read()
    assert b.box(sid, "lights").n == 11 and b.steps() == n0 + 2
    b.key(Qt.Key_Z, Qt.ControlModifier)
    b.read()
    assert b.box(sid, "lights").n == 5
    b.close()


def test_gv_table_empty_folder_leaves_no_entry(qapp, tree):
    b = Bench(1366)
    sid = b.sessions()[0]
    n0 = b.steps()
    b.drop([tree["sub"], tree["only_sub"]], ("box", sid, "lights"))     # one with direct files, one without
    b.read()
    assert b.own(sid, "lights") == [fwd(tree["sub"])] and b.box(sid, "lights").n == 5 and b.steps() == n0 + 1
    b.close()
    b = Bench(1366)
    n0, s0 = b.steps(), len(b.sessions())
    b.drop([tree["only_sub"]], ("ghost_p",))                          # no empty stack left behind
    b.read()
    assert len(b.sessions()) == s0 and b.steps() == n0 and "No direct file" in b.t.toast_text
    b.t.link("include")
    b.read()
    assert len(b.sessions()) == s0 + 1 and b.box(b.sessions()[-1], "lights").n == 6
    b.close()


def test_gv_table_nothing_guessed_from_names(qapp, tree):
    """A folder named like darks dropped on Lights stays in Lights; no header is read, no proposal is made."""
    b = Bench(1366)
    sid = b.sessions()[0]
    b.drop([tree["darks"]], ("box", sid, "lights"))
    b.read()
    assert b.own(sid, "lights") == [fwd(tree["darks"])] and b.box(sid, "lights").n == 5
    assert not any(b.doc.graph.common["files"].values()) and set(b.doc.graph.blocks[sid].files) == {"lights"}
    assert b.box(sid, "lights").l2 == "" and not b.g.model.all_warnings()
    assert not [c for c in b.g.cells if c.kind not in ("name", "box", "cal", "stack", "setting", "ghost_s", "ghost_p")]
    b.close()


def test_gv_table_unreadable_file(qapp, tmp_path):
    folder = make(tmp_path / "bad", 3, "L")
    (folder / "L_0001.fit").write_bytes(b"")
    b = Bench(1366)
    sid = b.sessions()[0]
    b.drop([folder], ("box", sid, "lights"))
    b.read()
    w = b.box(sid, "lights").warn
    assert w is not None and w.text == "1 file unreadable" and w.button == "Remove them"
    n0 = b.steps()
    b.click(("warn", sid, "lights"), zone="button")
    b.read()
    assert b.box(sid, "lights").n == 2 and b.steps() == n0 + 1
    b.close()


# ----------------------------------------------------------------------------------------------- keyboard
def test_gv_table_keyboard(qapp, tree):
    b = Bench(1366, sessions=2)
    g = b.g
    g.act([("add_row", dict(kind="session"))])
    s1, s2, s3 = b.sessions()
    g.setFocus()
    g.set_focus(g.cell(("name", s1)))
    b.key(Qt.Key_Right)
    assert g.focus == ("box", s1, "lights")
    b.key(Qt.Key_Down)
    assert g.focus == ("box", s2, "lights")
    for _i in range(4):
        b.key(Qt.Key_Right)
    assert g.focus == ("cal", s2)                                     # Darks, Flats, Bias, then Calibration
    b.key(Qt.Key_Right)
    assert g.focus[0] == "setting" and g.focus[2] == "reg"
    g.set_focus(g.cell(("name", s1)))
    b.key(Qt.Key_Tab)
    assert g.focus == ("box", s1, "lights")
    b.key(Qt.Key_Backtab, Qt.ShiftModifier)
    assert g.focus == ("name", s1)
    b.key(Qt.Key_Space)
    assert ("name", s1) in g.sel
    b.key(Qt.Key_Space)
    assert ("name", s1) not in g.sel
    asked = []
    g.choose_files = lambda title: asked.append(title) or [fwd(tree["lib"])]
    g.set_focus(g.cell(("box", s1, "darks")))
    b.key(Qt.Key_Return)                                              # Enter on an empty box: the picker
    b.read()
    assert asked == ["Darks of “Session 1”"] and b.box(s1, "darks").n == 6
    g.set_focus(g.cell(("name", s2)))
    b.key(Qt.Key_F2)
    assert g._editor is not None
    g._editor.setText("Night 13")
    b.key(Qt.Key_Return, widget=g._editor)
    b.wait(0.02)
    assert b.doc.graph.blocks[s2].name == "Night 13"
    b.key(Qt.Key_Z, Qt.ControlModifier)
    assert b.doc.graph.blocks[s2].name == "Session 2"
    g.set_focus(g.cell(("name", s2)))
    b.key(Qt.Key_F2)
    g._editor.setText("Cancelled")
    b.key(Qt.Key_Escape, widget=g._editor)
    assert b.doc.graph.blocks[s2].name == "Session 2"
    g.set_focus(g.cell(("name", s1)))
    b.key(Qt.Key_D, Qt.ControlModifier)                               # Ctrl+D: a session below, same darks, no lights
    ss = b.sessions()
    assert len(ss) == 4 and b.own(ss[1], "lights") is None and b.own(ss[1], "darks") == b.own(ss[0], "darks")
    g.set_focus(g.cell(("name", ss[1])))
    b.key(Qt.Key_Down, Qt.ControlModifier | Qt.ShiftModifier)
    assert b.sessions().index(ss[1]) == 2
    b.key(Qt.Key_Down, Qt.ControlModifier | Qt.ShiftModifier)
    assert b.g.model.stack_of(ss[1])["id"] != b.stacks()[0]["id"]     # crosses the stack border
    n = len(b.sessions())
    g.set_focus(g.cell(("name", ss[1])))
    b.key(Qt.Key_Delete)
    assert len(b.sessions()) == n - 1 and "files kept" in b.t.toast_text
    b.key(Qt.Key_Z, Qt.ControlModifier)
    assert len(b.sessions()) == n
    g.set_focus(g.cell(("box", s1, "darks")))
    b.key(Qt.Key_Delete)
    assert b.own(s1, "darks") is None and "emptied · files kept" in b.t.toast_text
    for i, s in enumerate(x for x in b.sessions() if x != s1):         # "No flats" is said of a session with lights
        r = b.doc.apply("add_files", session=s, kind="lights", paths=[str(tree["more"][i % len(tree["more"])])])
        assert r.ok, r.reason
    b.drop([tree["flats"]], ("box", s1, "flats"))
    b.read()
    assert b.g.model.all_warnings()
    b.key(Qt.Key_F8)
    assert "⚠" in b.t.state.text()
    g.sel = {("name", s1)}
    b.key(Qt.Key_Escape)
    assert not g.sel
    g.set_focus(g.cell(("box", s1, "flats")))
    b.key(Qt.Key_C, Qt.ControlModifier)
    g.set_focus(g.cell(("box", s3, "flats")))
    b.key(Qt.Key_V, Qt.ControlModifier)
    b.read()
    assert b.own(s3, "flats") == [fwd(tree["flats"])]
    QApplication.clipboard().clear()
    b.close()


# ----------------------------------------------------------------------------------------------- structure
def test_gv_table_structure(qapp, tree):
    b = Bench(1366)
    g = b.g
    rid = b.stacks()[0]["id"]
    n0 = b.steps()
    b.click(("ghost_s", rid))
    assert len(b.sessions()) == 2 and b.steps() == n0 + 1
    b.click(("ghost_p",))
    assert len(b.stacks()) == 2 and g._editor is not None             # « + Stack »: its name in edit
    g._editor.setText("Hα")
    g._editor.returnPressed.emit()
    b.wait(0.02)
    assert b.stacks()[1]["name"] == "Hα"
    s1, _s2, s3 = b.sessions()
    hot = b.hover([], ("name", s1), fy=0.9, internal=("session", s3))
    assert hot.kind == "insertion" and g.insertion is not None
    b.end_hover()
    b.drop([], ("name", s1), fy=0.9, internal=("session", s3))
    assert b.g.model.stack_of(s3)["id"] == rid and b.sessions()[1] == s3
    b.key(Qt.Key_Z, Qt.ControlModifier)
    assert b.g.model.stack_of(s3)["id"] != rid
    b.drop([], ("ghost_p",), internal=("session", b.sessions()[1]))
    assert len(b.stacks()) == 3
    b.key(Qt.Key_Z, Qt.ControlModifier)
    texts = [e[0] for e in g.menu_entries(g.cell(("stack", rid))) if e]
    assert "As reference" in texts and any(t.startswith("Merge into") for t in texts)
    g.take_reference(b.stacks()[1]["id"])
    assert b.stacks()[1]["ref"] and not b.stacks()[0]["ref"]
    g.set_focus(g.cell(("stack", b.stacks()[1]["id"])))
    g.delete()
    assert len(b.stacks()) == 1 and b.stacks()[0]["ref"] and "reference" in b.t.toast_text
    b.key(Qt.Key_Z, Qt.ControlModifier)
    g.set_value(("setting", b.stacks()[0]["id"], "stk"), "Rejection threshold", "8 σ")
    b.t.b_diff.setChecked(True)                                        # « Only differences »
    assert [c for c in g.cells if c.kind == "name"]
    b.t.b_diff.setChecked(False)
    g.click_zone(g.cell(("stack", rid)), "chevron")
    assert b.stacks()[0]["folded"] and g.cell(("folded_name", rid)).h == 40
    g.click_zone(g.cell(("stack", rid)), "chevron")
    assert not b.stacks()[0]["folded"]
    b.close()


def test_gv_table_variant_shown(qapp):
    b = Bench(1366)
    rid = b.stacks()[0]["id"]
    stk = b.g.model.stack_block(rid, "stacking")
    b.g.act([("add_variant", dict(block=stk))])
    vs = b.g.model.variants(rid)
    assert len(vs) == 1 and b.g.cell(("variant", vs[0]["id"])) is not None
    assert b.t.block_of_cell(("variant", vs[0]["id"])) == vs[0]["block"]
    b.close()
