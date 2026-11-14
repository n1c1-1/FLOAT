"""The Table: sessions as rows, files in boxes, a Common line, inheritance marks, variants (read only).

``Table(bridge)`` is the widget (toolbar, hint, grid, status line, toasts); ``Grid`` is the custom-painted grid. The
table reads and changes the project only through the core (the bridge's Doc: apply_many and the queries); it keeps
view state only (selection, focus, scroll, optional columns, stacks folded, warnings left). It guesses nothing: a drop
lands where it is dropped, files are taken by their extension only, nothing is sorted by name, folder or date.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

from PySide6.QtCore import QPoint, QPointF, Qt, QTimer, Signal
from PySide6.QtGui import QKeySequence, QPainter, QPalette, QShortcut
from PySide6.QtWidgets import (
    QAbstractScrollArea,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ...graphcore.operations import COMMON
from ..i18n import _, maybe, plural
from . import files
from .table_gestures import INTERNAL_MIME, Gestures, undo_link
from .table_logic import COLS_BLOCK, FOLDED, KIND_TITLE, KINDS, TableModel, count_text
from .table_paint import (
    ACCENT,
    BG,
    COMMON_COMPACT,
    COMMON_FULL,
    F11,
    GHOST,
    HEAD,
    ROW,
    VAR,
    WARNH,
    Cell,
    Painting,
    colour,
    width_of,
)

#: column widths at 1034 px (1366 wide window with the 320 px panel): without / with the optional Flat darks
W_NONE = {"stack": 124, "session": 92, "lights": 100, "darks": 96, "flats": 96, "bias": 96, "cal": 98,
          "reg": 96, "sel": 108, "stk": 128}
W_BOTH = {"stack": 118, "session": 88, "lights": 96, "darks": 88, "flats": 88, "bias": 88, "flat_darks": 88,
          "cal": 80, "reg": 90, "sel": 100, "stk": 110}
ORDER = ["stack", "session", "lights", "darks", "flats", "bias", "flat_darks", "cal", "reg", "sel", "stk"]
HINT = "One row per session. Drop each folder in its cell. The Common row: what serves every session."
FOOTNOTE = "Read from the files, for information: nothing is decided from it without your consent"


def widths(W, opts):
    """Column (x, width) for a table W px wide: Bias always, Flat darks only when in use or asked."""
    cols = [c for c in ORDER if c in W_NONE or c in opts]
    if W >= 1500:
        base = {"stack": 150, "session": 140, "lights": 150, "darks": 150, "flats": 150, "bias": 120,
                "flat_darks": 120, "cal": 150, "reg": 150, "sel": 150}
        w = {c: base.get(c, 150) for c in cols if c != "stk"}
        w["stk"] = W - sum(w.values())
    elif W == 1034:
        w = dict(W_BOTH if opts else W_NONE)
    else:
        ref = {c: (W_BOTH if opts else W_NONE).get(c, 90) for c in cols}
        tot = sum(ref.values())
        w = {c: int(ref[c] * W / tot) for c in cols}
        w[cols[-1]] += W - sum(w.values())
    xs, x = {}, 0
    for c in cols:
        xs[c] = (x, w[c])
        x += w[c]
    return xs


@dataclass
class Drag:
    paths: list = field(default_factory=list)
    n_dirs: int = 0
    n_files: int = 0
    internal: tuple | None = None
    copy: bool = False
    box_targets: bool = True

    def items_text(self):
        p = []
        if self.n_dirs:
            p.append(plural(self.n_dirs, "{n} folder", "{n} folders"))
        if self.n_files:
            p.append(plural(self.n_files, "{n} file", "{n} files"))
        return ", ".join(p)


@dataclass
class Target:
    kind: str                       # box | ghost_s | ghost_p | refused | insertion | merge
    ok: bool = True
    label: str = ""
    owner: str = ""
    file_kind: str = ""
    half: str = "add"
    rid: str = ""
    why: str = ""
    cells: set = field(default_factory=set)
    boxes: list = field(default_factory=list)
    before: object = None


class NameEditor(QLineEdit):
    accepted = Signal(str)
    cancelled = Signal()

    def __init__(self, parent, text):
        super().__init__(text, parent)
        self.done = False
        self.returnPressed.connect(self._ok)
        self.selectAll()

    def _ok(self):
        if not self.done:
            self.done = True
            self.accepted.emit(self.text())

    def keyPressEvent(self, e):                                       # noqa: N802
        if e.key() == Qt.Key_Escape:
            self.done = True
            self.cancelled.emit()
            e.accept()
            return
        super().keyPressEvent(e)
        if e.key() in (Qt.Key_Return, Qt.Key_Enter):
            e.accept()                                  # QLineEdit ignores Return: it must not reach the grid

    def focusOutEvent(self, e):                                       # noqa: N802
        if not self.done:
            self._ok()
        super().focusOutEvent(e)


def refused(why, label=""):
    return Target("refused", False, label, why=why)


class Grid(Gestures, Painting, QAbstractScrollArea):
    status = Signal(str, int)
    toast = Signal(object, object)         # (text or function, [(label, key)] or function)
    open_cell = Signal(object)             # a cell id whose block / files the panel should show (None: nothing)
    reading_done = Signal()
    toolbar_changed = Signal()             # selection, filter, reading counters: only the toolbar follows
    filter_changed = Signal()              # the file-type filter was chosen in a dialog
    _progress = Signal(object)             # the reader's thread -> this thread

    def __init__(self, bridge, reader=None, parent=None):
        super().__init__(parent)
        self.bridge = bridge
        self.reader = reader if reader is not None else bridge.drops.reader
        self.model = TableModel(bridge, self.reader)
        self._progress.connect(self._on_progress)
        self._listener = self._progress.emit
        self.reader.listen(self._listener)
        self.sel: set = set()
        self.focus = None
        self.focus_y = 0.0
        self.hover = None
        self.drag: Drag | None = None
        self.hot: Target | None = None
        self.drag_pos = QPoint(0, 0)
        self.insertion = None
        self.manual_columns: set = set()
        self.only_differences = False
        self.compact = True
        self.undo_delegated = False
        self.cells, self.pinned, self.lines = [], [], []
        self.xs = {}
        self.common_h = COMMON_COMPACT
        self.content_h = 0
        self.total_width = 0
        self._sigs: dict = {}
        self._press = None
        self._editor = None
        self._auto = 0
        self._internal = None
        self._warn_idx = -1
        self._watching = None
        self._rest = self._include = self._rather = None
        self._built_rev = None
        self.t_drop_ms = None
        self.t_last_paint = 0.0
        self.choose_files = self._choose_files
        self.setFrameShape(QFrame.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.verticalScrollBar().setStyleSheet(
            "QScrollBar:vertical{background:#15171c;width:12px;margin:0}"
            "QScrollBar::handle:vertical{background:#3a4150;border-radius:4px;min-height:24px;margin:2px}"
            "QScrollBar::add-line:vertical,QScrollBar::sub-line:vertical{height:0}")
        self.setAcceptDrops(True)
        self.viewport().setAcceptDrops(True)
        self.viewport().setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.verticalScrollBar().valueChanged.connect(lambda *_a: self.viewport().update())
        self._tick = QTimer(self)
        self._tick.setInterval(30)
        self._tick.timeout.connect(self._auto_scroll)
        self._bar_timer = QTimer(self)
        self._bar_timer.setSingleShot(True)
        self._bar_timer.setInterval(60)
        self._bar_timer.timeout.connect(self.toolbar_changed.emit)
        self.after_model(quiet=True)

    def detach(self):
        """No more news from the (shared) reader: the grid is about to go."""
        self.reader.unlisten(self._listener)
        self._watching = None

    # ------------------------------------------------------------------ small helpers
    @property
    def doc(self):
        return self.bridge.doc

    def scroll_y(self):
        return self.verticalScrollBar().value()

    def body_h(self):
        return self.viewport().height() - HEAD - self.common_h

    def setting_kind(self, c):
        return COLS_BLOCK.get(c.id[2]) if c.kind == "setting" else "calibration"

    def block_of(self, c):
        if c.kind == "cal":
            return self.model.calibration_of(c.id[1])
        return self.model.stack_block(c.id[1], COLS_BLOCK[c.id[2]])

    def row_of(self, ident):
        if ident and ident[0] in ("box", "name", "cal"):
            return ident[1]
        return None

    def focus_row(self):
        return self.row_of(self.focus)

    def multiple_selection(self):
        return len(self.sel) > 1

    def chosen_sessions(self):
        return [s["sid"] for pl in self.model.stacks() for s in pl["sessions"] if ("name", s["sid"]) in self.sel]

    def cell(self, ident):
        for c in self.pinned + self.cells:
            if c.id == ident:
                return c
        return None

    def session_name(self, sid):
        return self.model.session_name(sid)

    def variant_by_id(self, vid):
        return next((v for pl in self.model.stacks() for v in self.model.variants(pl["id"]) if v["id"] == vid), None)

    # ------------------------------------------------------------------ the model changed
    def after_model(self, quiet=False):
        """The document changed (a gesture, an undo, the graph): rebuild the rows, list what is new."""
        self.model.invalidate()
        self.rebuild()
        self.start_readings()
        self._built_rev = self.doc.revision
        if not quiet:
            self.toolbar_changed.emit()

    def act(self, steps, toast=None, links=(), error=None):
        """One gesture = one undo step: steps = [(name, kwargs | f(results) -> kwargs)]. -> the core's Result."""
        res = self.doc.apply_many(list(steps))
        if not res.ok:
            self.status.emit(self.bridge.say(res.reason) if res.reason else (error or _("Not possible")), 4000)
            return res
        self.bridge.view_changed()                    # the graph is drawn again; the host refreshes this table
        if self._built_rev != self.doc.revision:
            self.after_model()
        if toast:
            self.toast.emit(toast, links if callable(links) else list(links))
        return res

    def undo(self):
        res = self.bridge.undo()
        if res.ok and self._built_rev != self.doc.revision:
            self.after_model()
        elif not res.ok:
            self.status.emit(self.bridge.say(res.reason), 3000)
        return res

    def redo(self):
        res = self.bridge.redo()
        if res.ok and self._built_rev != self.doc.revision:
            self.after_model()
        elif not res.ok:
            self.status.emit(self.bridge.say(res.reason), 3000)
        return res

    # ------------------------------------------------------------------ layout
    def rebuild(self):
        mp = self.model
        W = self.viewport().width()
        self.total_width = W
        opts = set(mp.optional_in_use()) | self.manual_columns
        self.xs = xs = widths(W, opts)
        stacks = mp.stacks()
        self.compact = sum(len(p["sessions"]) for p in stacks) <= 1
        self.common_h = COMMON_COMPACT if self.compact else COMMON_FULL
        pinned, cells, lines = [], [], []
        y = HEAD
        pinned.append(Cell(("common_name",), "common_name", 0, y, xs["stack"][1] + xs["session"][1], self.common_h, True))
        for k in KINDS:
            if k in xs:
                pinned.append(Cell(("box", COMMON, k), "box", xs[k][0], y, xs[k][1], self.common_h, True))
        pinned.append(Cell(("cal", COMMON), "cal", xs["cal"][0], y, xs["cal"][1], self.common_h, True))
        for col in COLS_BLOCK:
            pinned.append(Cell(("setting", COMMON, col), "setting", xs[col][0], y, xs[col][1], self.common_h, True))
        y = 0
        xsess, wsess = xs["session"]
        wl = xs["cal"][0] + xs["cal"][1] - xsess
        wv = xs["stk"][0] + xs["stk"][1] - xsess
        for pl in stacks:
            y0 = y
            if pl["folded"]:
                cells.append(Cell(("folded_name", pl["id"]), "folded_name", xsess, y, wsess, ROW))
                for k in KINDS:
                    if k in xs:
                        cells.append(Cell(("box", FOLDED + pl["id"], k), "box", xs[k][0], y, xs[k][1], ROW))
                cells.append(Cell(("cal", FOLDED + pl["id"]), "folded_cal", xs["cal"][0], y, xs["cal"][1], ROW))
                lines.append({"kind": "folded", "y": y, "h": ROW, "rid": pl["id"]})
                y += ROW
                self._stack_cells(cells, pl, y0, y)
                continue
            shown = 0
            warn_cells = []                      # over the stack's spanning cells: painted and hit after them
            for s in pl["sessions"]:
                sid = s["sid"]
                if self.only_differences and not mp.has_difference(sid, pl):
                    continue
                shown += 1
                cells.append(Cell(("name", sid), "name", xsess, y, wsess, ROW))
                for k in KINDS:
                    if k in xs:
                        cells.append(Cell(("box", sid, k), "box", xs[k][0], y, xs[k][1], ROW))
                cells.append(Cell(("cal", sid), "cal", xs["cal"][0], y, xs["cal"][1], ROW))
                lines.append({"kind": "session", "y": y, "h": ROW, "sid": sid, "rid": pl["id"]})
                y += ROW
                for k in KINDS:                  # the amber line of a box: its whole text under the row, with its button
                    bx = mp.box(sid, k) if k in xs else None
                    if bx is not None and bx.warn is not None:
                        need = width_of("⚠ " + bx.warn.text, F11) + width_of(self.button_label(bx.warn), F11) + 70
                        ax = max(xsess, min(xs[k][0], W - need - 4))
                        warn_cells.append(Cell(("warn", sid, k), "warn", ax, y, W - ax - 2, WARNH))
                        lines.append({"kind": "warn", "y": y, "h": WARNH, "sid": sid, "rid": pl["id"]})
                        y += WARNH
            if shown == 0 and self.only_differences:
                continue
            self._stack_cells(cells, pl, y0, y)
            cells += warn_cells
            cells.append(Cell(("ghost_s", pl["id"]), "ghost_s", xsess, y, wl, GHOST))
            lines.append({"kind": "ghost_s", "y": y, "h": GHOST, "rid": pl["id"]})
            y += GHOST
            for v in mp.variants(pl["id"]):
                cells.append(Cell(("variant", v["id"]), "variant", xsess, y, wv, VAR))
                lines.append({"kind": "variant", "y": y, "h": VAR, "rid": pl["id"], "vid": v["id"]})
                y += VAR
        if not stacks:
            h = max(self.body_h(), 120)
            cells.append(Cell(("body",), "body", 0, 0, W, h - GHOST - 4))
            y = h - GHOST - 4
        cells.append(Cell(("ghost_p",), "ghost_p", 0, y, W, GHOST))
        lines.append({"kind": "ghost_p", "y": y, "h": GHOST})
        y += GHOST
        self.cells, self.pinned, self.lines, self.content_h = cells, pinned, lines, y
        vb = self.verticalScrollBar()
        vb.setRange(0, max(0, y - self.body_h()))
        vb.setPageStep(max(self.body_h(), 1))
        vb.setSingleStep(ROW)
        ids = {c.id for c in cells + pinned}
        self.sel = {i for i in self.sel if i in ids}
        if self.focus not in ids:
            self.focus = None
        self.viewport().update()

    def _stack_cells(self, cells, pl, y0, y1):
        xs = self.xs
        cells.append(Cell(("stack", pl["id"]), "stack", xs["stack"][0], y0, xs["stack"][1], y1 - y0))
        for col in COLS_BLOCK:
            cells.append(Cell(("setting", pl["id"], col), "setting", xs[col][0], y0, xs[col][1], y1 - y0))

    # ------------------------------------------------------------------ reading
    def start_readings(self):
        """Ask the reader for every list not listed yet (cheap when nothing changed)."""
        mp = self.model
        owners = [(COMMON, mp.common_list)]
        for pl in mp.stacks():
            for s in pl["sessions"]:
                owners.append((s["sid"], lambda k, sid=s["sid"]: mp.own_list(sid, k)))
        for owner, getter in owners:
            for k in KINDS:
                src = getter(k)
                if not src:
                    self._sigs.pop((owner, k), None)
                    continue
                files.restore_options(self.reader, self.doc.view, owner, k, src)
                sig = (tuple(src), tuple(self.reader.opts.get(x, files.DEFAULT) for x in src))
                if self._sigs.get((owner, k)) != sig:
                    self._sigs[(owner, k)] = sig
                    self.reader.request((owner, k), src)

    def _on_progress(self, job):
        self.model.invalidate()
        self._check_watch()
        if job.phase in ("done", "stopped"):
            self.rebuild()
            self.toolbar_changed.emit()
            self.reading_done.emit()
        else:
            self.viewport().update()
            if not self._bar_timer.isActive():
                self._bar_timer.start()

    # ------------------------------------------------------------------ painting and sizes
    def resizeEvent(self, e):                                         # noqa: N802
        super().resizeEvent(e)
        self.rebuild()

    def paintEvent(self, e):                                          # noqa: N802
        p = QPainter(self.viewport())
        p.fillRect(self.viewport().rect(), colour(BG))
        self.paint_all(p, self.viewport().width(), self.viewport().height())
        p.end()
        self.t_last_paint = time.perf_counter()

    def ensure_visible(self, c: Cell):
        if c.pinned:
            return
        vb = self.verticalScrollBar()
        h = self.body_h()
        if c.y < vb.value():
            vb.setValue(int(c.y))
        elif c.y + min(c.h, ROW) > vb.value() + h:
            vb.setValue(int(c.y + min(c.h, ROW) - h))

    # ------------------------------------------------------------------ hit testing
    def cell_at(self, pos):
        top = HEAD + self.common_h
        if pos.y() < HEAD:
            return None
        if pos.y() < top:
            return next((c for c in self.pinned if c.x <= pos.x() < c.x + c.w and c.y <= pos.y() < c.y + c.h), None)
        y = pos.y() - top + self.scroll_y()
        return next((c for c in reversed(self.cells) if c.x <= pos.x() < c.x + c.w and c.y <= y < c.y + c.h), None)

    def content_pos(self, c, pos):
        if c.pinned:
            return QPointF(pos.x(), pos.y())
        return QPointF(pos.x(), pos.y() - HEAD - self.common_h + self.scroll_y())

    def zone_at(self, c, pos):
        if c is None:
            return None
        q = self.content_pos(c, pos)
        return next((name for name, r in self.zones(c).items() if r.contains(q)), None)

    # ------------------------------------------------------------------ drag and drop, from outside
    @staticmethod
    def paths_of(mime):
        return [files.normalise(u.toLocalFile()) for u in mime.urls() if u.toLocalFile()]

    def dragEnterEvent(self, e):                                      # noqa: N802
        m = e.mimeData()
        if m.hasFormat(INTERNAL_MIME) and self._internal:
            self.drag = Drag(internal=self._internal, box_targets=self._internal[0] == "box")
        elif m.hasUrls():
            g = Drag(paths=self.paths_of(m))
            g.n_dirs, g.n_files = files.note_paths(self.reader, g.paths)
            self.drag = g
        else:
            e.ignore()
            return
        self.hot = None
        self.drag_pos = e.position().toPoint()
        self.viewport().update()
        self._follow(e)
        e.acceptProposedAction()

    def dragMoveEvent(self, e):                                       # noqa: N802
        if self.drag is None:
            return
        if self._follow(e):
            e.acceptProposedAction()
        else:
            e.ignore()

    def _follow(self, e):
        self.drag_pos = e.position().toPoint()
        self.drag.copy = bool(e.modifiers() & Qt.ControlModifier)
        self.hot = self.target_at(self.drag_pos)
        self.insertion = self.hot.before if self.hot and self.hot.kind == "insertion" else None
        h = self.viewport().height()
        y = self.drag_pos.y()
        top = HEAD + self.common_h
        if top < y < top + 40:
            self._auto = -int(4 + (top + 40 - y) * 0.7)
        elif y > h - 40:
            self._auto = int(4 + (y - (h - 40)) * 0.7)
        else:
            self._auto = 0
        if self._auto and not self._tick.isActive():
            self._tick.start()
        if not self._auto:
            self._tick.stop()
        self.viewport().update()
        return bool(self.hot and self.hot.ok)

    def _auto_scroll(self):
        if self.drag is None or not self._auto:
            self._tick.stop()
            return
        vb = self.verticalScrollBar()
        vb.setValue(vb.value() + self._auto)
        self.hot = self.target_at(self.drag_pos)
        self.viewport().update()

    def dragLeaveEvent(self, e):                                      # noqa: N802
        self.end_drag()

    def end_drag(self):
        self.drag, self.hot, self.insertion = None, None, None
        self._tick.stop()
        self._internal = None
        self.viewport().update()

    def dropEvent(self, e):                                           # noqa: N802
        t0 = time.perf_counter()
        if self.drag is None:
            return
        self.drag_pos = e.position().toPoint()
        self.drag.copy = bool(e.modifiers() & Qt.ControlModifier)
        target = self.target_at(self.drag_pos)
        g = self.drag
        self.end_drag()
        if not target.ok:
            self.status.emit(target.why, 4000)
            e.ignore()
            self.viewport().repaint()
            return
        e.acceptProposedAction()
        if g.internal:
            self._drop_internal(target, g)
        else:
            self.drop_files(target, g)
        self.viewport().repaint()                 # the visible reaction, at once; the listing goes on in the worker
        self.t_drop_ms = (time.perf_counter() - t0) * 1000

    # ---- targets
    def _box_label(self, g, owner, kind, replace):
        items = g.items_text()
        mp = self.model
        if owner == COMMON:
            if replace:
                return _("{items} → replaces the {n} of Common").format(items=items, n=mp.box(COMMON, kind).n)
            tot = own = 0
            for pl in mp.stacks():
                for s in pl["sessions"]:
                    tot += 1
                    own += mp.own_list(s["sid"], kind) is not None
            if own:
                return _("{items} → for {k} sessions of {n} ({j} have their own)").format(
                    items=items, k=tot - own, n=tot, j=own)
            return _("{items} → {kind} of Common").format(items=items, kind=_(KIND_TITLE[kind]))
        sn = self.session_name(owner)
        if replace:
            return _("{items} → replaces the {n} of “{s}”").format(items=items, n=mp.box(owner, kind).n, s=sn)
        if kind == "lights":
            return _("{items} → {s} (lights)").format(items=items, s=sn)
        return _("{items} → {kind} of “{s}”").format(items=items, kind=_(KIND_TITLE[kind]), s=sn)

    def target_at(self, pos) -> Target:
        return self.target_for(self.cell_at(pos), pos, self.drag)

    def target_for(self, c, pos, g) -> Target:
        if g.internal:
            return self._internal_target(pos, c, g)
        nothing = _("Nothing dropped: {why}")
        if c is None:
            return refused(nothing.format(why=_("outside the table's cells")))
        k = c.kind
        mp = self.model
        if k == "box":
            owner, kind = c.id[1], c.id[2]
            if owner.startswith(FOLDED):
                return refused(nothing.format(why=_("unfold the stack to choose the session")),
                               _("Unfold the stack to choose the session"))
            if owner == COMMON and kind == "lights":
                return refused(nothing.format(why=_("lights go on a session, not on Common")),
                               _("Lights go on a session"))
            b = mp.box(owner, kind)
            half = "add"
            if b.mode in ("own", "inherited", "none"):
                half = "replace" if pos.x() >= c.x + c.w / 2 else "add"
            group = [i for i in self.sel if i[0] == "box" and i[2] == kind and not i[1].startswith(FOLDED)]
            if c.id in self.sel and len(group) > 1:
                boxes = [(i[1], i[2]) for i in group if not (i[1] == COMMON and kind == "lights")]
                return Target("box", True, _("{items} → {n} chosen boxes").format(items=g.items_text(), n=len(boxes)),
                              owner, kind, half, cells=set(group), boxes=boxes)
            return Target("box", True, self._box_label(g, owner, kind, half == "replace"), owner, kind, half,
                          cells={c.id}, boxes=[(owner, kind)])
        if k == "name":
            sid = c.id[1]
            return Target("box", True, self._box_label(g, sid, "lights", False), sid, "lights", "add",
                          cells={c.id}, boxes=[(sid, "lights")])
        n = max(g.n_dirs, 1 if g.n_files else 0)
        if k == "ghost_s":
            lab = (_("{items} → new session").format(items=g.items_text()) if n <= 1 else
                   _("{items} → {n} new sessions").format(items=g.items_text(), n=n))
            return Target("ghost_s", True, lab, rid=c.id[1], cells={c.id})
        if k in ("ghost_p", "body"):
            lab = (_("{items} → new stack").format(items=g.items_text()) if n <= 1 else
                   _("{items} → new stack, {n} sessions").format(items=g.items_text(), n=n))
            return Target("ghost_p", True, lab, cells={c.id})
        if k in ("folded_name", "folded_cal"):
            return refused(nothing.format(why=_("unfold the stack to choose the session")),
                           _("Unfold the stack to choose the session"))
        why = {"setting": _("a settings cell takes no files"), "cal": _("the Calibration is set in its panel"),
               "stack": _("drop on a session or on “+ Session”"),
               "common_name": _("choose a cell of Common")}.get(k, _("this zone takes no files"))
        return refused(nothing.format(why=why), _("Nothing dropped"))

    def _internal_target(self, pos, c, g):
        kind_of = g.internal[0]
        mp = self.model
        nothing = _("Nothing dropped: {why}")
        if kind_of == "box":
            _k, so, sk = g.internal
            if c is None:
                return refused(nothing.format(why=_("outside the cells")))
            if c.kind == "name":
                owner, kind = c.id[1], "lights"
            elif c.kind == "box" and not c.id[1].startswith(FOLDED):
                owner, kind = c.id[1], c.id[2]
            else:
                return refused(nothing.format(why=_("drop on a file cell")))
            if owner == COMMON and kind == "lights":
                return refused(nothing.format(why=_("lights go on a session, not on Common")),
                               _("Lights go on a session"))
            if (owner, kind) == (so, sk):
                return refused(nothing.format(why=_("it is the same cell")))
            dest = self.box_title(owner, kind)
            lab = (_("Copy → {dest}").format(dest=dest) if g.copy else
                   _("Move → {dest} · Ctrl: copy").format(dest=dest))
            return Target("box", True, lab, owner, kind, cells={c.id}, boxes=[(owner, kind)])
        if kind_of == "session":
            sid = g.internal[1]
            if c is None:
                return refused(nothing.format(why=_("outside the sessions")))
            if c.kind in ("name", "box", "cal") and c.id[1] != COMMON and not str(c.id[1]).startswith(FOLDED):
                tsid = c.id[1]
                ln = next(x for x in self.lines if x.get("sid") == tsid and x["kind"] == "session")
                upper = self.content_pos(c, pos).y() < ln["y"] + ln["h"] / 2
                pl = mp.stack_of(tsid)
                names = [s["sid"] for s in pl["sessions"]]
                i = names.index(tsid)
                if upper:
                    before, y = tsid, ln["y"]
                else:
                    before = names[i + 1] if i + 1 < len(names) else ""
                    y = ln["y"] + ln["h"]
                if not self._moves(sid, pl["id"], before):
                    return refused(nothing.format(why=_("the session is already there")))
                return Target("insertion", True, _("“{s}” goes into “{p}”").format(s=self.session_name(sid), p=pl["name"]),
                              rid=pl["id"], before=y)
            if c.kind == "ghost_s":
                pl = mp.stack_by_id(c.id[1])
                return Target("insertion", True, _("“{s}” goes into “{p}”").format(s=self.session_name(sid), p=pl["name"]),
                              rid=pl["id"], before=c.y)
            if c.kind in ("ghost_p", "body"):
                return Target("ghost_p", True, _("“{s}” → new stack").format(s=self.session_name(sid)), cells={c.id})
            return refused(nothing.format(why=_("drop between two sessions")))
        if kind_of == "stack":
            rid = g.internal[1]
            if c is not None and c.kind == "stack" and c.id[1] != rid:
                src, dst = mp.stack_by_id(rid), mp.stack_by_id(c.id[1])
                lab = plural(len(src["sessions"]), "{n} session joins “{p}”", "{n} sessions join “{p}”", p=dst["name"])
                own = self.own_settings_text(rid)
                if own:
                    lab += " · " + _("own settings dropped: {list}").format(list=own)
                return Target("merge", True, lab, rid=c.id[1], cells={c.id})
            return refused(nothing.format(why=_("drop on another stack to merge them")))
        return refused(_("Nothing dropped"))

    def _moves(self, sid, rid, before):
        pl = self.model.stack_by_id(rid)
        names = [s["sid"] for s in pl["sessions"]]
        if sid in names:
            i = names.index(sid)
            nxt = names[i + 1] if i + 1 < len(names) else ""
            return before not in (sid, nxt)
        return True

    def own_settings_text(self, rid):
        parts = []
        for kind in COLS_BLOCK.values():
            b = self.model.stack_block(rid, kind)
            for k in (self.doc.own_keys(b) if b else []):
                parts.append(f"{_(k)} {maybe(str(self.doc.effective(b).get(k)))}")
        return ", ".join(parts)

    # ---- executing a drop of files
    @staticmethod
    def _groups(g):
        grp = [[p] for p in g.paths if os.path.isdir(p)]
        loose = [p for p in g.paths if not os.path.isdir(p)]
        if loose:
            grp.append(loose)
        return grp

    @staticmethod
    def _step(owner, kind, paths, mode):
        if owner == COMMON:
            return "set_common_files", dict(kind=kind, paths=list(paths), mode=mode)
        return "add_files", dict(session=owner, kind=kind, paths=list(paths), mode=mode)

    def _take(self, items, loose):
        """The files the filter left out, made known as single files. -> their paths."""
        left = list(self.reader.left_out(items))
        self.reader.adopt(left)
        for p in loose:
            self.reader.note(p, False)
        return [f.path for f in left] + list(loose)

    def _plan_include(self, items, boxes):
        """Steps that make the folders of a drop read all levels (their entries get the mark). -> (steps, changes)."""
        mp, steps, chg = self.model, [], []
        aims = [x for x in items if files.is_folder(self.reader, x) and not files.base(x)[1]]
        for o, k in boxes:
            own = None if o == COMMON else mp.own_list(o, k)
            cur = mp.common_list(k) if o == COMMON else list(own if own is not None else mp.common_list(k))
            if not any(x in aims for x in cur):
                continue
            steps.append(self._step(o, k, [x + files.MARK if x in aims else x for x in cur], "replace"))
            chg += [(o, k, x, x + files.MARK) for x in cur if x in aims]
        return steps, chg

    def _plan_include_sessions(self, sids, lots):
        steps, chg = [], []
        for sid, (x, _lg) in zip(sids, lots):
            cur = list(self.model.own_list(sid, "lights") or [])
            aims = [i for i in x if files.is_folder(self.reader, i) and not files.base(i)[1]]
            if aims:
                steps.append(self._step(sid, "lights", [i + files.MARK if i in aims else i for i in cur], "replace"))
                chg += [(sid, "lights", i, i + files.MARK) for i in cur if i in aims]
        return steps, chg

    def drop_files(self, target, g):
        self._rest = self._include = self._rather = None
        if target.kind == "box":
            self._drop_on_boxes(target, g)
        elif target.kind in ("ghost_s", "ghost_p"):
            self._drop_new_sessions(target, g)

    def _drop_on_boxes(self, target, g):
        mp = self.model
        mode = "replace" if target.half == "replace" else "add"
        boxes = list(target.boxes)
        items, loose = files.split_loose(self.reader, list(g.paths))
        ignored = 0
        if mode == "add":
            o0, k0 = boxes[0]
            own = None if o0 == COMMON else mp.own_list(o0, k0)
            exist = list(own if own is not None else mp.common_list(k0))
            new = [p for p in items if not files.already(p, exist)]
            ignored, items = len(items) - len(new), new
        self._rest = lambda: [self._step(o, k, self._take(items, loose), "add") for o, k in boxes]
        self._include = lambda: self._plan_include(items, boxes)
        if not items:
            self.toast.emit(*files.nothing_to_take(loose, ignored))
            return

        def build(paths, mode=mode):
            return [self._step(o, k, paths, mode) for o, k in boxes]
        steps = build(items)
        o0, k0 = boxes[0]
        base = files.drop_toast(self.reader, k0, None if o0 == COMMON else self.session_name(o0), items, mode, len(boxes))
        text = files.drop_sentence(self.reader, base, items, loose, mode, ignored)
        self._rather = build(items, "add")
        res = self.act(steps, text, files.drop_links(self.reader, items, loose, mode))
        if not res.ok:
            return
        for o, k in boxes:
            files.save_options(self.reader, self.doc.view, o, k, items)

        def include_empty(empties, remaining):
            st, chg = self._plan_include(remaining, boxes) if remaining else ([], [])
            marked = [x + files.MARK for x in empties]
            return list(st) + build(marked, "add"), list(chg) + [(o, k, x, x + files.MARK) for o, k in boxes for x in empties]
        self._watch(items, loose, mode,
                    lambda remaining: (lambda: self.act(build(remaining))) if remaining else None,
                    lambda empties, remaining: (lambda: include_empty(empties, remaining)))

    def _drop_new_sessions(self, target, g):
        groups, loose_of = [], []
        for grp in self._groups(g):
            items, loose = files.split_loose(self.reader, grp)
            groups.append(items or None)
            loose_of.append(loose)
        taken = [x for x in groups if x]
        loose_all = [p for lst in loose_of for p in lst]
        if not taken:
            if loose_all:
                self.toast.emit(_("Nothing taken · {left} left").format(left=files.summary(loose_all)), [])
            return
        lots_all = [(x, lg) for x, lg in zip(groups, loose_of) if x]
        new_session = target.kind == "ghost_s"

        def steps_for(taken):
            if new_session:
                return [("add_night", dict(row=target.rid, files={"lights": grp})) for grp in taken]
            out = [("add_row", dict(kind="session", files={"lights": taken[0]}))]
            out += [("add_night", lambda outs, grp=grp: dict(row=outs[0].created["row"], files={"lights": grp}))
                    for grp in taken[1:]]
            return out

        def launch(taken, lots, with_toast=True):
            n = len(taken)
            if new_session:
                txt = plural(n, "{n} session added", "{n} sessions added")
            else:
                txt = _("New stack added") if n == 1 else _("New stack added ({n} sessions)").format(n=n)
            every = [p for x in taken for p in x]
            if with_toast:
                res = self.act(steps_for(taken), files.drop_sentence(self.reader, txt, every, loose_all, "add"),
                               files.drop_links(self.reader, every, loose_all, "add"))
            else:
                res = self.act(steps_for(taken))
            if res.ok:
                made = [b for b in (res.created or {}).get("blocks", [])
                        if b in self.doc.graph.blocks and self.doc.graph.blocks[b].kind == "session"]

                def rest(made=made, lots=lots):
                    out = []
                    for sid, (x, lg) in zip(made, lots):
                        ps = self._take(x, lg)
                        if ps:
                            out.append(self._step(sid, "lights", ps, "add"))
                    return out
                self._rest = rest
                self._include = lambda: self._plan_include_sessions(made, lots)
                for sid, (x, _lg) in zip(made, lots):
                    files.save_options(self.reader, self.doc.view, sid, "lights", x)
                self.focus_last_session()
            return res

        res = launch(taken, lots_all)
        if not res.ok:
            return
        every = [p for x in taken for p in x]

        def rebuild(remaining):
            kept = [[x for x in grp if x in remaining] for grp in taken]
            kept = [x for x in kept if x]
            if not kept:
                return None
            lots = [([i for i in x if i in remaining], lg) for x, lg in lots_all if any(i in remaining for i in x)]
            return lambda: launch(kept, lots, with_toast=False)

        def include_empty(empties, remaining):
            """A new session (or stack) per emptied folder, read at all levels."""
            def do():
                marked = [[x + files.MARK for x in grp if x in empties] for grp in taken]
                marked = [x for x in marked if x]
                for grp in marked:
                    for xm in grp:
                        self.reader.note(xm, True)
                        self.reader.opts[xm] = self.reader.opts.get(xm[:-len(files.MARK)], files.DEFAULT)
                return steps_for(marked), [], (lambda res: self._note_included(res, marked))
            return do
        self._watch(every, loose_all, "add", rebuild, include_empty)

    def _note_included(self, res, marked):
        """After « Include » made new sessions of emptied folders: the all-levels choice is recorded for each."""
        made = [b for b in (res.created or {}).get("blocks", [])
                if b in self.doc.graph.blocks and self.doc.graph.blocks[b].kind == "session"]
        for sid, grp in zip(made, marked):
            files.save_options(self.reader, self.doc.view, sid, "lights", grp)

    # ---- a folder with no direct matching file leaves no empty entry behind
    def _watch(self, items, loose, mode, rebuild, include_empty):
        """Watch the drop just made: once every folder is listed, the entries holding no direct matching file are taken
        out again (the gesture is undone and, when some folders did hold files, made again without them: still one
        undo step), and the toast says why, with « Include » when sub-folders were not read."""
        self._watching = dict(items=list(items), loose=list(loose), mode=mode, rev=self.doc.revision,
                              rebuild=rebuild, include_empty=include_empty)
        self._check_watch()

    def _check_watch(self):
        w = self._watching
        if not w:
            return
        if self.doc.revision != w["rev"]:                       # something else was edited meanwhile: leave it
            self._watching = None
            return
        items = w["items"]
        if self.reader.expand(tuple(items)) is None:            # still listing
            return
        self._watching = None
        empties = files.empty_folders(self.reader, items)
        if not empties:
            return
        remaining = [x for x in items if x not in empties]
        redo = w["rebuild"](remaining)
        self.doc.undo()
        self.bridge.view_changed()
        if self._built_rev != self.doc.revision:
            self.after_model()
        if redo is not None:
            redo()
        text = files.drop_sentence(self.reader, lambda: _("Folder without a direct file: nothing added"), items,
                                   w["loose"], w["mode"])
        links0 = files.drop_links(self.reader, items, w["loose"], w["mode"])

        def links():
            return [(t, k) for t, k in links0() if k != "undo" or redo is not None]
        self._include = w["include_empty"](empties, remaining)
        self.toast.emit(text, links)

    def focus_last_session(self):
        sids = [s["sid"] for pl in self.model.stacks() for s in pl["sessions"]]
        if not sids:
            return
        sid = max(sids, key=lambda x: int(x[1:]) if x[1:].isdigit() else -1)
        c = self.cell(("name", sid))
        if c:
            self.focus = c.id
            self.focus_y = c.y + c.h / 2
            self.sel = {c.id}
            self.ensure_visible(c)
            self.toolbar_changed.emit()

    # ---- internal moves
    def start_internal(self, desc):
        self._internal = desc

    def _drop_internal(self, target, g):
        kind_of = g.internal[0]
        mp = self.model
        if kind_of == "box":
            _k, so, sk = g.internal
            owner, kind = target.owner, target.file_kind
            steps = [("move_files", dict(src_session=so, src_kind=sk, dst_session=owner, dst_kind=kind, copy=g.copy))]
            src = _("{kind} copied to {dest}") if g.copy else _("{kind} moved to {dest}")
            self.act(steps, src.format(kind=_(KIND_TITLE[sk]), dest=self.box_title(owner, kind)), undo_link())
            return
        if kind_of == "session":
            sid = g.internal[1]
            if target.kind == "ghost_p":
                self.act([("move_night", dict(night=sid, row=None))],
                         _("“{s}” in a new stack").format(s=self.session_name(sid)), undo_link())
                return
            pl = mp.stack_by_id(target.rid)
            args = dict(night=sid, row=target.rid)
            ln = next((x for x in self.lines if x["kind"] == "session" and x["rid"] == target.rid
                       and abs(x["y"] - target.before) < 1), None)
            if ln:
                args["before"] = ln["sid"]
            self.act([("move_night", args)], _("“{s}” goes into “{p}”").format(s=self.session_name(sid), p=pl["name"]),
                     undo_link())
            return
        if kind_of == "stack":
            self.merge(g.internal[1], target.rid)


class Table(QWidget):
    """The Table page: toolbar 34, hint 24, grid, status 24, toasts. ``bridge`` is the page's Bridge."""
    message = Signal(str)          # a note for the window (the table shows its own toasts and refusals)
    cell = Signal(object)          # a cell id whose block / files the panel should show (None: the selection is cleared)

    def __init__(self, bridge, parent=None, reader=None):
        super().__init__(parent)
        self.setObjectName("table")
        self.bridge = bridge
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self.bar = QStackedWidget()
        self.bar.setFixedHeight(34)
        lay.addWidget(self.bar)
        self.hint = QLabel("ⓘ  " + _(HINT))
        self.hint.setFixedHeight(24)
        self.hint.setStyleSheet("color:#8b93a1;font-size:12px;padding-left:10px;background:#181b21")
        self.hint.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)      # a long line never widens the table
        lay.addWidget(self.hint)
        self.grid = Grid(bridge, reader)
        lay.addWidget(self.grid, 1)
        self.state = QLabel()
        self.state.setFixedHeight(24)
        self.state.setTextFormat(Qt.RichText)
        self.state.setStyleSheet("color:#8b93a1;font-size:12px;padding-left:10px;background:#181b21")
        self.state.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.state.linkActivated.connect(self.link)
        lay.addWidget(self.state)
        self._normal_bar()
        self._batch_bar()
        self.toast_w = QLabel(self)
        self.toast_w.setTextFormat(Qt.RichText)
        self.toast_w.setStyleSheet("background:#2a2f39;border:1px solid #3a4150;border-radius:6px;padding:6px 12px;"
                                   "color:#e8ebf0;font-size:12px")
        for w in (self.toast_w, self.state):
            pal = w.palette()
            pal.setColor(QPalette.Link, colour(ACCENT))
            w.setPalette(pal)
        self.toast_w.linkActivated.connect(self.link)
        self.toast_w.hide()
        self.toast_text = ""
        self.status_text = ""
        self._toast_fn, self._toast_links = None, []
        self._toast_timer = QTimer(self)
        self._toast_timer.setSingleShot(True)
        self._toast_timer.setInterval(6000)
        self._toast_timer.timeout.connect(self._toast_end)
        self._status_timer = QTimer(self)
        self._status_timer.setSingleShot(True)
        self._status_timer.timeout.connect(self._status_default)
        g = self.grid
        g.status.connect(self._on_status)
        g.toast.connect(self._on_toast)
        g.toolbar_changed.connect(self._update_bars)
        g.reading_done.connect(self._refresh_toast)
        g.open_cell.connect(self.cell)
        g.filter_changed.connect(lambda: self.c_files.setCurrentIndex(files.index()))
        self.bridge.rich_toast_refresh.connect(self._refresh_toast)
        self._shortcuts = []
        for seq, fn in ((QKeySequence.Undo, g.undo), (QKeySequence.Redo, g.redo)):
            sc = QShortcut(seq, self)
            sc.setContext(Qt.WidgetWithChildrenShortcut)
            sc.activated.connect(fn)
            self._shortcuts.append(sc)
        self._status_default()
        self._update_bars()

    # ---- bars
    def _normal_bar(self):
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(8, 3, 10, 3)
        h.setSpacing(10)
        self.b_cols = QPushButton(_("+ columns"))
        self.b_cols.setFixedHeight(28)
        self.b_cols.setToolTip(_("Flat darks · optional"))
        self.b_cols.clicked.connect(self._columns_menu)
        h.addWidget(self.b_cols)
        self.c_files = QComboBox()
        self.c_files.setFixedHeight(28)
        self.c_files.setToolTip(_(files.TIP))
        for name, _e in files.TYPES:
            self.c_files.addItem(_("Files: {name}").format(name=_(name)))
        self.c_files.setCurrentIndex(files.index())
        self.c_files.activated.connect(files.set_index)
        h.addWidget(self.c_files)
        h.addStretch(1)
        self.counter = QLabel("")
        self.counter.setStyleSheet(f"color:{ACCENT};font-size:12px")
        h.addWidget(self.counter)
        self.b_warn = QPushButton("")
        self.b_warn.setObjectName("link")
        self.b_warn.setCursor(Qt.PointingHandCursor)
        self.b_warn.clicked.connect(lambda: self.grid.next_warning())
        h.addWidget(self.b_warn)
        self.b_diff = QPushButton(_("Only differences"))
        self.b_diff.setObjectName("toggle")
        self.b_diff.setCheckable(True)
        self.b_diff.setFixedHeight(28)
        self.b_diff.toggled.connect(self._only_differences)
        h.addWidget(self.b_diff)
        self.bar.addWidget(w)

    def _batch_bar(self):
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(10, 3, 10, 3)
        h.setSpacing(10)
        self.l_batch = QLabel("")
        self.l_batch.setStyleSheet("font-weight:600")
        h.addWidget(self.l_batch)
        g = self.grid
        b1 = QPushButton(_("New stack"))
        b1.clicked.connect(lambda: g.move_to(g.chosen_sessions(), None))
        self.b_move = QPushButton(_("Move to ▾"))
        self.b_move.clicked.connect(self._move_menu)
        b3 = QPushButton(_("Back to common"))
        b3.clicked.connect(lambda: g.back_to_common([g.cell(("name", s)) for s in g.chosen_sessions()]))
        b4 = QPushButton(_("Delete"))
        b4.clicked.connect(g.delete)
        b5 = QPushButton("×")
        b5.setFixedWidth(28)
        b5.setToolTip(_("Clear the selection"))
        b5.clicked.connect(g.clear_selection)
        for b in (b1, self.b_move, b3, b4):
            b.setFixedHeight(28)
            h.addWidget(b)
        h.addStretch(1)
        h.addWidget(b5)
        self.bar.addWidget(w)
        self.batch_buttons = {"stack": b1, "move": self.b_move, "common": b3, "delete": b4, "x": b5}

    def _columns_menu(self):
        m = QMenu(self)
        for k, lab in (("flat_darks", _("Flat darks · optional")),):
            a = m.addAction(lab)
            a.setCheckable(True)
            used = k in self.grid.model.optional_in_use()
            a.setChecked(used or k in self.grid.manual_columns)
            a.setEnabled(not used)
            if used:
                a.setToolTip(_("in use: the column stays"))
            a.toggled.connect(lambda on, k=k: self.column(k, on))
        m.exec(self.b_cols.mapToGlobal(QPoint(0, self.b_cols.height())))

    def column(self, k, on):
        (self.grid.manual_columns.add if on else self.grid.manual_columns.discard)(k)
        self.grid.rebuild()

    def _move_menu(self):
        m = QMenu(self)
        for pl in self.grid.model.stacks():
            m.addAction(f"“{pl['name']}”", lambda r=pl["id"]: self.grid.move_to(self.grid.chosen_sessions(), r))
        m.exec(self.b_move.mapToGlobal(QPoint(0, self.b_move.height())))

    def _only_differences(self, on):
        self.grid.only_differences = on
        self.grid.rebuild()

    def _update_bars(self):
        g = self.grid
        mp = g.model
        sids = g.chosen_sessions()
        if len(sids) >= 2:
            self.bar.setCurrentIndex(1)
            self.l_batch.setText(plural(len(sids), "{n} session chosen", "{n} sessions chosen"))
        else:
            self.bar.setCurrentIndex(0)
        n = len(mp.all_warnings())
        self.b_warn.setText(plural(n, "{n} warning ▸", "{n} warnings ▸") if n else "")
        self.b_warn.setVisible(bool(n))
        jobs = mp.readings()
        self.counter.setText(_("Reading… {n}").format(n=sum(t.listed for t in jobs)) if jobs else "")
        if not self._status_timer.isActive():
            self._status_default()

    # ---- status line and toasts
    def summary(self):
        mp = self.grid.model
        ns = nb = w = 0
        for pl in mp.stacks():
            a, b, v = mp.totals(pl)
            ns += a
            nb += b
            w += v
        t = plural(ns, "{n} session", "{n} sessions") + " · " + count_text(nb, "lights")
        return t + " · " + plural(w, "{n} warn.", "{n} warn.") if w else t

    def _status_default(self):
        self.state.setText(self.summary() + "   ·   " + _(FOOTNOTE))

    def _on_status(self, text, ms):
        self.state.setText(text)
        self._status_timer.start(ms)
        self.status_text = text                  # a refusal: this line only (the toast is not doubled)

    @staticmethod
    def _html(text, links):
        h = text
        for lab, key in (links() if callable(links) else links):
            h += f' · <a href="{key}"><span style="color:{ACCENT}">{lab}</span></a>'
        return h

    def _on_toast(self, text, links):
        self._toast_fn = text if callable(text) else None
        self._toast_links = links
        self._show_toast(text() if callable(text) else text)
        self._toast_timer.start()

    def _show_toast(self, t):
        self.toast_w.setText(self._html(t, self._toast_links))
        self.toast_w.adjustSize()
        self.toast_w.move(14, self.height() - 24 - self.toast_w.height() - 12)
        self.toast_w.show()
        self.toast_w.raise_()
        self.toast_text = t                      # shown once: the toast (with its links), not also in the line

    def _refresh_toast(self):
        if self._toast_fn is not None and self.toast_w.isVisible():
            self._show_toast(self._toast_fn())

    def _toast_end(self):
        if self.grid.model.readings():                 # still listing: the toast waits for the final count
            self._toast_timer.start()
            return
        self.toast_w.hide()
        self._toast_fn = None

    def link(self, key):
        """A link of the toast: "undo", "take_all", "include", "rather_add"."""
        g = self.grid
        self.toast_w.hide()
        self._toast_fn = None
        if key == "undo":
            g.undo()
        elif key == "take_all":
            steps = [s for s in (g._rest() if g._rest else []) if s[1].get("paths")]
            if steps:
                g.act(steps, _("All taken"), undo_link())
        elif key == "include":
            plan = g._include() if g._include else ([], [])
            steps, chg = plan[0], plan[1]
            after = plan[2] if len(plan) > 2 else None
            if steps:
                view = g.doc.view
                for o, k, old, new in chg:                       # the choice is recorded before the listing starts
                    name = files.option_name(g.reader, view, o, k, old)
                    g.reader.opts[new] = g.reader.opts.get(old, files.DEFAULT)
                    g.reader.note(new, True)
                    view.setdefault(files.OPTIONS, {})[f"{o}|{k}|{new}"] = [name, True]
                res = g.act(steps, _("All levels read"), undo_link())
                if after is not None and res.ok:
                    after(res)
        elif key == "rather_add":
            steps = g._rather
            r = g.doc.undo()
            if r.ok:
                g.bridge.view_changed()
                if g._built_rev != g.doc.revision:
                    g.after_model()
                if steps:
                    g.act(steps, _("Added rather than replaced"), undo_link())

    def resizeEvent(self, e):                                         # noqa: N802
        super().resizeEvent(e)
        if self.toast_w.isVisible():
            self.toast_w.move(14, self.height() - 24 - self.toast_w.height() - 12)

    # ---- public
    def delegate_undo(self):
        """In the page Ctrl+Z / Ctrl+Y belong to the page (one undo history for graph and table)."""
        for sc in self._shortcuts:
            sc.setEnabled(False)
        self.grid.undo_delegated = True

    def block_of_cell(self, ident):
        """The core id (block, or row id for a stack cell) the right panel shows for a cell id; None for Common
        cells (no block) and folded rows."""
        g = self.grid
        if not ident or ident[0] == "common_name" or len(ident) < 2 or ident[1] == COMMON \
                or str(ident[1]).startswith(FOLDED):
            return None
        if ident[0] in ("name", "box", "stack"):
            return ident[1]
        if ident[0] == "variant":
            return next((v["block"] for pl in g.model.stacks() for v in g.model.variants(pl["id"])
                         if v["id"] == ident[1]), None)
        c = g.cell(ident)
        if c is not None and ident[0] in ("cal", "setting"):
            return g.block_of(c)
        return None

    def refresh(self):
        """The document changed elsewhere (the graph, an undo from the page, another document)."""
        self.grid.after_model()
