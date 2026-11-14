"""The graph page's canvas: draws the core's Drawing and turns real gestures into operations.

It never computes a position, a legality or a text: positions come from the core's layout, drop places from its
places, texts from ``Doc``; every change goes through the bridge. During a drag nothing moves: the places are ghosts
over the unchanged drawing.
"""
from __future__ import annotations

import html
import time

from PySide6.QtCore import QEvent, QPoint, QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QKeyEvent, QPainter, QPen, QTransform
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QGraphicsScene,
    QGraphicsView,
    QLabel,
    QLineEdit,
    QMenu,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ...graphcore.model import FILE_KINDS, JOIN_KINDS, Index
from ...graphcore.rules import FILE_LABEL, title_msg
from ...graphcore.shapes import Result
from ..i18n import _
from ..widgets import Spinner
from . import style
from .bridge import PROJECT
from .items import (
    BandItem,
    BlockItem,
    CaptionItem,
    FileTarget,
    GhostItem,
    LinkItem,
    ProjectItem,
    RubberWire,
    StackItem,
    TipItem,
    WireItem,
)
from .menu import menu as block_menu
from .style import ACCENT, FALLBACK_WORD, IMAGE_COL, STATE_COL, kind_what, state_word

MIN_ZOOM, MAX_ZOOM, MIN_AUTO, MAX_AUTO = 0.1, 4.0, 0.75, 1.2
MARGIN = 48                 # scene margin around the drawing (also keeps blocks off the zoom buttons)
MIME_KIND = "application/x-float-graph-kind"
#: the blocks without an image out port
NO_OUT = ("session", "calibration", "registration", "selection", "crop", "post")
ZOOMS = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0)
#: a combination's and Combine layers' inputs: the colour of their dot
PORT_COL = {"Background": IMAGE_COL["starless"], "Stars": IMAGE_COL["stars_only"]}
OUT_COL = {"Starless": IMAGE_COL["starless"], "Stars only": IMAGE_COL["stars_only"]}
ALERT_MS = 1500
EDGE = 40                   # a drag this near the view's edge scrolls it (graph.py EDGE)
EDGE_WAIT = 0.35            # s held near the edge before it scrolls: a drag only crossing the edge moves nothing


def _fields(message) -> list:
    """The values a core message names (its fields, and theirs), as messages: what a refusal is about."""
    out = []
    if isinstance(message, tuple) and len(message) == 2 and isinstance(message[1], dict):
        for v in message[1].values():
            out.append(v)
            out += _fields(v)
    return out


class Toast(QLabel):
    """"… · Ctrl+Z", bottom-left over the canvas, gone after a few seconds; never a pop-up window. ``say_links`` gives
    it links ("Undo", "Take all", "Include", "Delete too"): the text may be a function (a count that follows a
    background listing) and the toast stays while ``keep()`` says something is still being read."""
    link = Signal(str)

    def __init__(self, parent):
        super().__init__(parent)
        self.setObjectName("toast")
        self.setStyleSheet("QLabel#toast { background: #262a33; color: #e8ebf0; border: 1px solid #3a4150; "
                           "border-radius: 6px; padding: 6px 12px; }")
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.timer = QTimer(self, singleShot=True, interval=4500)
        self.timer.timeout.connect(self._end)
        self._fn, self._links, self._keep = None, None, None
        self.plain = ""
        self.linkActivated.connect(self.link.emit)
        self.hide()

    def say(self, text):
        self._fn = self._links = self._keep = None
        self._show(text, False)

    def say_links(self, text, links=(), keep=None):
        self._fn, self._links, self._keep = text, links, keep
        self.refresh()

    def refresh(self):
        """The text of a function toast again (the listing moved on)."""
        if self._fn is None and self._links is None:
            return
        t = self._fn() if callable(self._fn) else self._fn
        ls = self._links() if callable(self._links) else list(self._links or [])
        h = html.escape(t)
        for lab, key in ls:
            h += f' · <a href="{key}"><span style="color:{ACCENT}">{html.escape(lab)}</span></a>'
        self._show(h, True, t)

    def links(self):
        """The (text, key) links shown now."""
        return self._links() if callable(self._links) else list(self._links or [])

    def _end(self):
        if self._keep is not None and self._keep():
            self.timer.start()
            return
        self.hide()

    def _show(self, text, rich, plain=None):
        self.setTextFormat(Qt.RichText if rich else Qt.PlainText)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, not rich)
        self.plain = plain if plain is not None else text
        self.setText(text)
        self.setMinimumWidth(0)
        self.setMaximumWidth(16777215)
        self.setWordWrap(False)
        self.adjustSize()
        if self.width() > self.parentWidget().width() - 40:
            self.setWordWrap(True)
            self.setFixedWidth(max(self.parentWidget().width() - 40, 200))
            self.adjustSize()
        else:
            self.setFixedWidth(self.width())
        self.move(16, self.parentWidget().height() - self.height() - 16)
        self.show()
        self.raise_()
        self.timer.start()


class Minimap(QWidget):
    """The overview in the corner, shown while part of the graph is out of view: click or drag moves the view."""
    W, H = 190.0, 130.0

    def __init__(self, view):
        super().__init__(view.viewport())
        self.view = view
        self.rects: list = []
        self.hover = False
        self.setCursor(Qt.OpenHandCursor)
        self.setToolTip(_("Overview: click or drag to move the view"))

    def _frame(self):
        r = self.view.scene().sceneRect()
        f = min(self.W / max(r.width(), 1.0), self.H / max(r.height(), 1.0))
        return QRectF(4, 4, r.width() * f, r.height() * f), f

    def place(self):
        m, _f = self._frame()
        vp = self.view.viewport().rect()
        self.setGeometry(vp.width() - int(m.width()) - 16, vp.height() - int(m.height()) - 16, int(m.width()) + 8,
                         int(m.height()) + 8)
        self.raise_()

    def paintEvent(self, e):                                         # noqa: N802
        m, f = self._frame()
        sr = self.view.scene().sceneRect()
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(15, 17, 21, 225 if self.hover else 120))     # translucent unless the pointer is on it
        p.drawRoundedRect(QRectF(self.rect()), 5, 5)

        def to(r):
            return QRectF(m.x() + (r.x() - sr.x()) * f, m.y() + (r.y() - sr.y()) * f, r.width() * f, r.height() * f)
        for r, col in self.rects:
            col = QColor(col)
            col.setAlpha(255 if self.hover else 150)
            p.setBrush(col)
            p.drawRoundedRect(to(r), 1.5, 1.5)
        v = to(self.view.mapToScene(self.view.viewport().rect()).boundingRect()).intersected(m)
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(QColor(ACCENT), 1.5))
        p.drawRect(v)
        p.end()

    def _go(self, pos):
        m, f = self._frame()
        sr = self.view.scene().sceneRect()
        self.view.centerOn(QPointF(sr.x() + (pos.x() - m.x()) / f, sr.y() + (pos.y() - m.y()) / f))
        self.view.auto = False

    def mousePressEvent(self, e):                                    # noqa: N802
        if e.button() == Qt.LeftButton:
            self._go(e.position())
        e.accept()

    def mouseMoveEvent(self, e):                                     # noqa: N802
        if e.buttons() & Qt.LeftButton:
            self._go(e.position())
        e.accept()

    def enterEvent(self, e):                                         # noqa: N802
        self.hover = True
        self.update()

    def leaveEvent(self, e):                                         # noqa: N802
        self.hover = False
        self.update()

    def wheelEvent(self, e):                                         # noqa: N802
        e.ignore()


class NameEdit(QLineEdit):
    """A name typed on its block itself: Enter or a click elsewhere confirms, Escape gives the old name back."""

    def __init__(self, parent, text, done):
        super().__init__(text, parent)
        self._done = done
        self.setStyleSheet(f"QLineEdit {{ background: #15171c; color: #e8ebf0; border: 1px solid {ACCENT}; "
                           "border-radius: 3px; padding: 0 3px; }")
        self.editingFinished.connect(self.finish)

    def event(self, e):                                               # noqa: N802
        if e.type() == QEvent.ShortcutOverride and e.key() in (Qt.Key_Escape, Qt.Key_Return, Qt.Key_Enter):
            e.accept()
            return True
        return super().event(e)

    def keyPressEvent(self, e):                                       # noqa: N802
        if e.key() == Qt.Key_Escape:
            self.finish(cancel=True)
            e.accept()
            return
        super().keyPressEvent(e)

    def finish(self, cancel=False):
        if self._done is None:
            return
        done, self._done = self._done, None
        text = self.text().strip()
        self.hide()
        self.deleteLater()
        if not cancel and text:
            done(text)


class GraphCanvas(QGraphicsView):
    inspect = Signal(object)             # Enter / double-click / menu "Inspect" on this id
    view_action = Signal(str, object)    # a menu action only the window can do: ("mark_reviewed", block)
    templates_asked = Signal()           # the blank page's link « From template »

    def __init__(self, bridge, over_palette=None, parent=None):
        super().__init__(parent)
        self.bridge = bridge
        self.over_palette = over_palette or (lambda gp: False)
        self.setScene(QGraphicsScene(self))
        self.setBackgroundBrush(QColor("#15171c"))
        self.setRenderHints(QPainter.Antialiasing | QPainter.TextAntialiasing | QPainter.SmoothPixmapTransform)
        self.setFrameShape(QFrame.NoFrame)
        self.setAcceptDrops(True)
        self.viewport().setAcceptDrops(True)
        self.setMouseTracking(True)
        self.viewport().setMouseTracking(True)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setTransformationAnchor(QGraphicsView.NoAnchor)
        self.setResizeAnchor(QGraphicsView.NoAnchor)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setAlignment(Qt.AlignLeft | Qt.AlignTop)     # the graph starts at the top left, as today's does
        self.blocks: dict[str, BlockItem] = {}
        self.wires: dict[str, WireItem] = {}
        self.ghosts: list[GhostItem] = []
        self.targets: list[FileTarget] = []
        self.project_item = None
        self.new_link = None
        self.tip = TipItem()
        self.rubber = None
        self.floater = None
        self._gesture = None             # the gesture under way: dict (see the property)
        self.auto = True                 # fits the graph (user zoom turns it off)
        self._refit = True               # the next build fits the view (new document); a gesture never re-zooms
        self._keep_in_view = ()          # ids a gesture made: kept on screen while the viewport settles
        self._size = None
        self.space = False
        self.name_edit = None
        self.selected_wire = None
        self._to_delete = None
        self._pressed = False            # a block pressed by the mouse is being selected (it is under the pointer)
        self._k = 1.0
        # a drag (palette card, placed block, wire, files) held near the view's edge scrolls the view (graph.py)
        self._edge_pos = None
        self._edge_since = 0.0
        self._edge_timer = QTimer(self)
        self._edge_timer.setInterval(30)
        self._edge_timer.timeout.connect(self._edge_tick)
        self.toast = Toast(self.viewport())
        self.minimap = Minimap(self)
        self.minimap.hide()
        # the controls over the canvas: zoom box + "Show all", then "Post →" while Post is out of view
        self.b_zoom = QPushButton("100 %", self.viewport())
        self.b_zoom.setMinimumWidth(62)
        self.b_zoom.setToolTip(_("Zoom (wheel, + and −)"))
        zm = QMenu(self.b_zoom)
        zm.addAction(_("Show all"), self.show_all)
        for z in ZOOMS:
            zm.addAction(f"{int(z * 100)} %", lambda zz=z: self.zoom_to(zz))
        self.b_zoom.setMenu(zm)
        self.b_all = QPushButton(_("Show all"), self.viewport())
        self.b_all.setObjectName("toggle")
        self.b_all.setCheckable(True)
        self.b_all.setToolTip(_("The whole graph in the window, smaller; else a readable size and scrolling"))
        self.b_all.clicked.connect(lambda on: self.show_all() if on else self.auto_view())
        self.b_post = QPushButton(_("Post →"), self.viewport())
        self.b_post.clicked.connect(self.go_post)
        self.b_post.hide()
        self.blank = QWidget(self.viewport())             # the blank page's words
        self.blank.hide()
        self._blank_built = False
        # while the project model reads the project: no graph, a veil, a spinner and its line (graph.py show_wait)
        self.veil = QFrame(self)
        self.veil.setStyleSheet("background: #15171c;")
        self.veil.hide()
        self.spinner = Spinner(self)
        self.spinner.hide()
        self.wait = QLabel(self)
        self.wait.setAlignment(Qt.AlignCenter)
        self.wait.setStyleSheet("color: #8b93a1; font-size: 10pt;")
        self.wait.hide()
        self._loading = False
        bridge.changed.connect(self.rebuild)
        bridge.states_changed.connect(self.update_states)
        bridge.selection.connect(self._selection_changed)
        bridge.appeared.connect(self._show_block)
        bridge.toast.connect(self.say)
        bridge.rich_toast.connect(lambda t, ls: self.toast.say_links(t, ls, self._reading))
        bridge.rich_toast_refresh.connect(lambda: self.toast.isVisible() and self.toast.refresh())
        self.toast.link.connect(self._toast_link)
        bridge.replaced.connect(self._replaced)
        self.rebuild()

    def _reading(self) -> bool:
        drops = self.bridge._drops
        return drops is not None and bool(drops.reader.active_jobs())

    @property
    def gesture(self):
        return self._gesture

    @gesture.setter
    def gesture(self, v):
        self._gesture = v
        if getattr(self, "minimap", None) is not None:
            self._update_minimap()

    @property
    def doc(self):
        return self.bridge.doc

    @property
    def plus_session_pid(self):
        """The workflow (id) a new session joins, as today's "+ Session" says it: the selected row's, through the
        project model's BlockMap; None without a selection or a model."""
        m, sel = self.doc.model, self.bridge.selected
        bm = getattr(m, "block_map", None) if m is not None else None
        r = (self.bridge.row(sel) or self.bridge.row_of(sel)) if sel and sel != PROJECT else None
        if r is None or bm is None:
            return None
        for b in [n[0] for n in r.nights] + list(r.blocks):
            a = bm.address(b)
            if a is not None:
                return a[0]
        return None

    def _wire_hover(self, ident, on):
        it = self.wires.get(ident)
        if it is None or it.ends is None:
            return
        for b in it.ends:
            bi = self.blocks.get(b)
            if bi is None:                               # a block folded into a card: the card
                r = self.bridge.row_of(b)
                bi = self.blocks.get(r.id) if r is not None else None
            if bi is not None:
                bi.wire_hot = on
                bi.update()

    def _toast_link(self, key):
        if key == "delete_too":
            self._delete_too()
        else:
            self.bridge.drops.link(key)

    # ------------------------------------------------------------------ a deletion that cannot be made
    def delete_block(self, ident):
        """The delete cross / Del on a block. When the core refuses, nothing is silent: the blocks that depend on it get
        a red frame for 1.5 s and the toast says why, with ONE action to delete them too when that is possible."""
        bridge = self.bridge
        res = bridge.act("remove_block", quiet="refusal", block=ident)
        if res.ok:
            return res
        reason = bridge.say(res.reason) or _("This block cannot be deleted")
        deps, possible = self.dependants(ident)
        framed = deps or [ident]
        for b in framed:
            it = self.blocks.get(b)
            if it is not None:
                it.alert = True
                it.update()
        QTimer.singleShot(ALERT_MS, self, lambda bs=tuple(framed): self._end_alert(bs))
        self._to_delete = [ident] + list(deps) if possible and deps else None
        links = []
        if possible and deps:
            links = [(_("Delete too"), "delete_too")]
        bridge.rich_toast.emit(reason, links)
        return res

    def _end_alert(self, ids):
        for b in ids:
            it = self.blocks.get(b)
            if it is not None:
                it.alert = False
                it.update()

    def _removals(self, ident, deps):
        return [("remove_block", {"block": b}) for b in list(reversed(deps)) + [ident]]

    def dependants(self, ident, limit=12):
        """(blocks that need `ident`, can they all go with it in one step). A block that needs it is the one the
        core's refusal names (its title message, compared as a message, never as text); the step is tried by a dry run
        each time, nothing changes."""
        doc = self.doc
        g = doc.graph
        ix = Index(g)
        rid = ix.row_of.get(ident)
        if rid is None:
            return [], False
        later = []
        for r in g.rows:                                  # blocks after it: its row's, then the variants' under it
            eff = ix.effective(r.id)
            if ident in eff:
                later += [b for b in eff[eff.index(ident) + 1:] if b not in later]
        later += [b for b in g.blocks if b != ident and b not in later]   # then the blocks its wires feed (composition)
        deps = []
        for _i in range(limit):
            res = doc.try_many(self._removals(ident, deps))
            if res.ok:
                return deps, bool(deps)
            named = _fields(res.reason)
            nxt = next((b for b in later if b not in deps and title_msg(g, b) in named), None)
            if nxt is None:
                return deps, False
            deps.append(nxt)
        return deps, False

    def _delete_too(self):
        ids, self._to_delete = self._to_delete, None
        if not ids:
            return
        ident, deps = ids[0], ids[1:]
        self.bridge.act_many(self._removals(ident, deps))

    # ================================================================== the scene from the Drawing
    def say(self, text):
        self.toast.say(text)

    def _replaced(self):
        self.auto = True
        self._refit = True
        self._keep_in_view = ()
        self.b_all.setChecked(False)

    def _ports(self, b, box, sense):
        d = self.bridge.drawing()
        return [(port, y - box.y) for (bb, port, s), (x, y) in d.ports.items() if bb == b and s == sense]

    def block_data(self, b, box):
        bridge, doc = self.bridge, self.doc
        blk = bridge.block(b)
        kind = blk.kind
        state = doc.state(b)
        grey = doc.greyed(b) or ""
        review = doc.review(b)
        row = bridge.row_of(b)
        badge = ""
        if kind == "session" and row is not None and row.nights and row.nights[0][0] == b and len(row.nights) == 1:
            badge = doc.row_badge(row.id)               # the block carries the row's name: and its badge
        wires = bridge.graph.wires
        inputs, outputs = [], []
        if kind in JOIN_KINDS:
            for port, y in self._ports(b, box, "in"):
                wired = any(tuple(w.dst) == (b, port) for w in wires)
                inputs.append((port, _(port), y, wired, PORT_COL.get(port, "#9aa3b2")))
            inputs.sort(key=lambda i: i[2])
        if kind not in NO_OUT:
            outs = self._ports(b, box, "out") or [("image", box.h / 2)]
            outs.sort(key=lambda o: o[1])
            for port, y in outs:
                wired = any(tuple(w.src) == (b, port) for w in wires)
                outputs.append((port, y, OUT_COL.get(port, "#9aa3b2"), wired))
        title = doc.title(b)
        summary = doc.summary(b)
        bd = doc.badges(b)
        fallback = list(bd.get("fallback") or []) if state == "done" and not grey else []
        broken, missing = bd.get("broken") or "", bd.get("missing") or ""
        word = _(FALLBACK_WORD) if fallback else state_word(state)
        tip = "\n".join(t for t in (title, kind_what(kind), summary, "" if grey else word,
                                    (_("Switched off: the image passes through.") if blk.off else "")) if t)
        if fallback:                                     # why, in the engine's words translated (graph.py)
            from ..services.run import engine_text
            tip += "".join("\n" + engine_text(w) for w in fallback)
        if broken:
            tip += "\n" + _("Cannot run: {why}. Run is refused until it is fixed.").format(why=broken.rstrip("."))
        if missing:
            tip += "\n" + _("Cannot run: {why}. Install it: Settings › Catalogs & astrometry. The runs leave this "
                            "block out, with the blocks after it.").format(why=missing.rstrip("."))
        if grey:
            tip += "\n" + _("Cannot run: {why}").format(why=grey)
        for _name, lab, _y, wired, _c in inputs:
            tip += "\n" + (_("Input {name}: wired") if wired else _("Input {name}: empty, pull a wire here")).format(
                name=lab)
        return dict(id=b, kind=kind, title=title, summary=summary, state=state, selected=bridge.selected == b,
                    off=bool(blk.off), grey=grey, badge=badge, review=review, own=doc.differs(b),
                    fallback=fallback, broken=broken, missing=missing, warned=bool(bd.get("warned")),
                    w=box.w, h=box.h, x=box.x, y=box.y, inputs=inputs, outputs=outputs,
                    removable=kind not in ("post", "calibration"), tip=tip)

    def card_data(self, row_id, box):
        bridge, doc = self.bridge, self.doc
        r = bridge.row(row_id)
        title, summary = doc.title(row_id), doc.summary(row_id)
        cal = r.nights[0][1] if r.nights else None
        kinds = {bridge.block(b).kind: b for b in r.blocks}
        dots = []
        for b in (cal, kinds.get("registration"), kinds.get("selection"), kinds.get("stacking")):
            if b is not None:
                dots.append((doc.state(b), doc.title(b) + ": " + doc.summary(b)))
        stack_state = doc.state(kinds["stacking"]) if "stacking" in kinds else "pending"
        return dict(id=row_id, kind="card", title=title, summary=summary, state=stack_state,
                    selected=bridge.selected == row_id, off=False, grey="", badge=doc.row_badge(row_id), review=None,
                    own=any(doc.differs(b) for b in r.blocks), w=box.w, h=box.h, x=box.x, y=box.y, inputs=[],
                    outputs=[("image", box.h / 2, "#9aa3b2", True)], removable=False, dots=dots,
                    tip="\n".join((title, summary, _("Blocks: show the blocks of every stack (Ctrl+J)"))))

    def rebuild(self):
        bridge, doc = self.bridge, self.doc
        self.close_name()
        self.ghosts, self.targets, self.rubber, self.floater = [], [], None, None
        self.blocks, self.wires = {}, {}
        sc = self.scene()
        sc.clear()
        self.tip = TipItem()
        drawing = bridge.drawing()
        for label, box in drawing.bands:
            sc.addItem(BandItem(QRectF(box.x, box.y, box.w, box.h), _(label)))
        for path in drawing.wires:
            sel = path.style in ("port", "port_inactive") and path.id.startswith("w")
            it = WireItem(path.id, path.points, path.style, path.colour, sel)
            it.tipfn = (lambda i=path.id: doc.wire_tip(i))
            it.ends = (path.src, path.dst)
            it.hover_cb = self._wire_hover
            sc.addItem(it)
            self.wires[path.id] = it
            if self.selected_wire == path.id:
                it.mark(True)
        rects = []
        for b, box in drawing.boxes.items():
            it = BlockItem(self.block_data(b, box))
            sc.addItem(it)
            self.blocks[b] = it
            rects.append((QRectF(box.x, box.y, box.w, box.h), QColor(STATE_COL.get(it.d["state"], "#5d6470"))))
        for r, box in drawing.cards.items():
            it = StackItem(self.card_data(r, box))
            sc.addItem(it)
            self.blocks[r] = it
            rects.append((QRectF(box.x, box.y, box.w, box.h), QColor(STATE_COL.get(it.d["state"], "#5d6470"))))
        for row_id, text, x, y in drawing.captions:
            sc.addItem(CaptionItem(x, y, text, badge=doc.row_badge(row_id)))
        self.minimap.rects = rects
        nw = drawing.extras.get("new_workflow")
        if nw is not None and (drawing.boxes or drawing.cards):
            self.new_link = LinkItem(QPointF(nw.x + nw.w / 2, nw.y + nw.h / 2), _("New workflow"))
            sc.addItem(self.new_link)
        else:
            self.new_link = None
        pb = drawing.extras.get("project")
        state, tip = doc.project_state()
        g = bridge.graph
        pr = QRectF(pb.x, pb.y, pb.w, pb.h) if pb is not None else QRectF(0, 0, 140, 34)
        self.project_item = ProjectItem(pr, g.name or _("Project"), g.target or _("target to choose"), state,
                                        bridge.selected == PROJECT, tip)
        sc.addItem(self.project_item)
        rect = sc.itemsBoundingRect()                    # the drawing's own box (blocks, bands, wires, captions)...
        if rect.width() < 10 or rect.height() < 10:
            rect = rect.united(QRectF(rect.x(), rect.y(), 10, 10))
        rect = rect.adjusted(-MARGIN, -MARGIN, MARGIN, MARGIN)   # ...plus a margin: the canvas never scrolls beyond it
        sc.addItem(self.tip)
        sc.setSceneRect(rect)
        self.minimap.hide()
        self._blank_page(not bridge.graph.rows)          # no row yet (the Post block is always drawn)
        if self._refit:                                  # a new document / first build: fit; a gesture never re-zooms
            self._refit = False
            self._apply_zoom()
        else:
            self._set_scale(self._k, None)
        show, bridge.to_show = bridge.to_show, ()
        if show:
            self._show(*show)
        self._update_buttons()
        self.show_wait()
        self.viewport().update()

    def show_wait(self):
        """The line over the graph while the project model reads the project, or why it could not (graph.py)."""
        r = self.doc.reading()
        text, busy = r if r is not None else ("", False)
        self.wait.setText(text)
        self.wait.setVisible(bool(text))
        self.veil.setVisible(busy)
        self.spinner.setVisible(busy)
        if busy != self._loading:                       # the system's busy cursor, as long as it lasts
            self._loading = busy
            if busy:
                self.viewport().setCursor(Qt.BusyCursor)
            else:
                self.viewport().unsetCursor()
        self._place_wait()

    def _place_wait(self):
        self.veil.setGeometry(self.rect())
        self.spinner.move((self.width() - self.spinner.width()) // 2, self.height() // 2 - 90)
        self.wait.setGeometry(0, self.height() // 2 - 40, self.width(), 30)
        for w in (self.veil, self.spinner, self.wait):
            if w.isVisible():
                w.raise_()

    def update_states(self):
        """The run states changed: the items are given them again."""
        drawing = self.bridge.drawing()
        for b, it in self.blocks.items():
            if it.d["kind"] == "card":
                it.d = self.card_data(b, drawing.cards[b])
            else:
                it.d = self.block_data(b, drawing.boxes[b])
            it.update()
        self.show_wait()
        self.viewport().update()
        self.minimap.update()

    def _show_block(self, ident):
        """A gesture made this block: it must be on screen (scroll to it if it is not)."""
        if ident not in self._keep_in_view:
            self._show(ident)

    def _show(self, *ids, follow=False):
        """The blocks of ids fully on screen: scroll the minimum, never zoom. The first id is the one that matters when
        they do not all fit together."""
        its = [self.blocks[i] for i in ids if i in self.blocks]
        if not its:
            return
        if not follow:
            self._keep_in_view = tuple(i for i in ids if i in self.blocks)
        vp = self.viewport().rect().adjusted(8, 8, -8, -8)
        union = its[0].sceneBoundingRect()
        for it in its[1:]:
            union = union.united(it.sceneBoundingRect())
        fits = union.width() * self._k <= vp.width() and union.height() * self._k <= vp.height()
        self.ensureVisible(union if fits else its[0].sceneBoundingRect(), 24, 24)

    def _selection_changed(self, ident):
        for b, it in self.blocks.items():
            if it.d["selected"] != (ident == b):
                it.d["selected"] = ident == b
                it.update()
        if self.project_item is not None:
            self.project_item.selected = ident == PROJECT
            self.project_item.update()
        if ident in self.blocks and not self._pressed:
            self._show(ident, follow=bool(self._keep_in_view))
        self._update_buttons()

    # ================================================================== the blank page
    def _blank_page(self, blank):
        self.blank.setVisible(False)
        if not self._blank_built:
            v = QVBoxLayout(self.blank)
            v.setContentsMargins(0, 0, 0, 0)
            v.setSpacing(8)
            self.blank_title = style.label(_("Blank page"), "h1")
            self.blank_text = style.label(_("Build the workflow from the palette on the left: drag or double-click "
                                            "a block."), "dim")
            # today's blank page link (graph.py LinkItem "Start from a template"): the gallery, in place of the graph
            self.blank_link = QPushButton(_("From template"))
            self.blank_link.setObjectName("link")
            self.blank_link.setFlat(True)
            self.blank_link.setCursor(Qt.PointingHandCursor)
            self.blank_link.setStyleSheet(f"QPushButton {{ color: {ACCENT}; border: none; font-size: 10pt; }} "
                                          "QPushButton:hover { text-decoration: underline; }")
            self.blank_link.setToolTip(_("Start the project from one of the gallery's templates"))
            self.blank_link.clicked.connect(self.templates_asked.emit)
            for w in (self.blank_title, self.blank_text, self.blank_link):
                v.addWidget(w, 0, Qt.AlignHCenter)
            self._blank_built = True
        if blank:
            self.blank.resize(560, 110)
            vp = self.viewport()
            self.blank.move((vp.width() - 560) // 2, vp.height() // 2 - 150)
            self.blank.show()
            self.blank.raise_()

    # ================================================================== zoom, view
    def _content(self) -> QRectF:
        return self.scene().sceneRect()

    def _apply_zoom(self, keep=False):
        if self.auto:
            r = self._content()
            vp = self.viewport()
            k = min(vp.width() / r.width(), vp.height() / r.height())
            # readable first (0.75); a little smaller rather than a scroll bar for the last few percent
            self._k = min(MAX_AUTO, k) if k >= 0.68 else MIN_AUTO
            self._set_scale(self._k, r.topLeft(), left_top=not keep)
        else:
            self._set_scale(self._k, None)

    def _set_scale(self, k, origin, left_top=False):
        self.setTransform(QTransform.fromScale(k, k))
        if left_top and origin is not None:
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().minimum())
            self.verticalScrollBar().setValue(self.verticalScrollBar().minimum())
        self.b_zoom.setText(f"{round(k * 100)} %")

    @property
    def zoom(self) -> float:
        return self._k

    def show_all(self):
        r = self._content()
        vp = self.viewport()
        self.auto = False
        self._keep_in_view = ()
        self._k = max(MIN_ZOOM, min(MAX_AUTO, min(vp.width() / r.width(), vp.height() / r.height())))
        self.setTransform(QTransform.fromScale(self._k, self._k))
        self.centerOn(r.center())
        self.b_zoom.setText(f"{round(self._k * 100)} %")
        self.b_all.setChecked(True)
        self._update_buttons()

    def auto_view(self):
        self.auto = True
        self.b_all.setChecked(False)
        self._apply_zoom()
        self._update_buttons()

    def zoom_to(self, k, centre=None):
        k = max(MIN_ZOOM, min(MAX_ZOOM, k))
        c = centre if centre is not None else self.mapToScene(self.viewport().rect().center())
        self.auto = False
        self._keep_in_view = ()
        self._k = k
        self.b_all.setChecked(False)
        self.setTransform(QTransform.fromScale(k, k))
        self.centerOn(c)
        self.b_zoom.setText(f"{round(k * 100)} %")
        self._update_buttons()

    def go_post(self):
        for it in self.blocks.values():
            if it.d["kind"] == "post":
                self.auto = False
                self.centerOn(it)
                self._update_buttons()

    def _update_buttons(self):
        vp = self.viewport()
        self.b_all.adjustSize()
        self.b_zoom.adjustSize()
        self.b_all.move(vp.width() - self.b_all.width() - 14, 10)
        self.b_zoom.move(self.b_all.x() - self.b_zoom.width() - 8, 10)
        view_rect = self.mapToScene(vp.rect()).boundingRect()
        self._update_minimap()
        post = next((it for it in self.blocks.values() if it.d["kind"] == "post"), None)
        show = post is not None and not view_rect.intersects(post.sceneBoundingRect())
        self.b_post.setVisible(show)
        if show:
            self.b_post.adjustSize()
            self.b_post.move(vp.width() - self.b_post.width() - 14, self.b_all.geometry().bottom() + 6)
            self.b_post.raise_()
        self.b_zoom.raise_()
        self.b_all.raise_()
        if self.toast.isVisible():
            self.toast.raise_()

    def _update_minimap(self):
        """The overview: only while the drawing exceeds the viewport, and never during a drag (it would hide the place
        being aimed at; it comes back at the drop)."""
        vp = self.viewport()
        view_rect = self.mapToScene(vp.rect()).boundingRect()
        drawing = self.scene().sceneRect().adjusted(MARGIN, MARGIN, -MARGIN, -MARGIN)
        outside = not view_rect.contains(drawing.adjusted(1, 1, -1, -1))
        dragging = self.gesture is not None and self.gesture.get("kind") in ("block", "wire", "palette", "files")
        show = outside and bool(self.blocks) and not dragging
        if self.minimap.isVisible() != show:
            self.minimap.setVisible(show)
        if show:
            self.minimap.place()

    def resizeEvent(self, e):                                         # noqa: N802
        super().resizeEvent(e)
        size = (self.width(), self.height())
        changed = size != self._size
        self._size = size
        if changed:
            self._keep_in_view = ()
            if self.auto:
                self._apply_zoom(keep=True)
        elif self._keep_in_view:                 # a scroll bar appeared and took room: keep what the gesture made
            self._show(*self._keep_in_view, follow=True)
        self._update_buttons()
        self._place_wait()
        if self.blank.isVisible():
            self.blank.move((self.viewport().width() - 560) // 2, self.viewport().height() // 2 - 150)

    def scrollContentsBy(self, dx, dy):                               # noqa: N802
        super().scrollContentsBy(dx, dy)
        self._update_buttons()

    def wheelEvent(self, e):                                          # noqa: N802
        d = e.angleDelta().y()
        if d == 0:
            return
        pos = self.mapToScene(e.position().toPoint())
        self._keep_in_view = ()
        k = max(MIN_ZOOM, min(MAX_ZOOM, self._k * (1.15 if d > 0 else 1 / 1.15)))
        self.auto = False
        self.b_all.setChecked(False)
        self._k = k
        before = e.position().toPoint()
        self.setTransform(QTransform.fromScale(k, k))
        delta = self.mapFromScene(pos) - before
        self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() + delta.x())
        self.verticalScrollBar().setValue(self.verticalScrollBar().value() + delta.y())
        self.b_zoom.setText(f"{round(k * 100)} %")
        self._update_buttons()
        e.accept()

    # ================================================================== hit testing
    def item_at(self, vp: QPoint):
        """The topmost interesting item under a viewport point."""
        for it in self.items(vp):
            if isinstance(it, (FileTarget, BlockItem, ProjectItem, WireItem, LinkItem)):
                return it
        return None

    def centre_of(self, ident) -> QPoint:
        """Viewport point of a block's centre (the gesture helpers use it)."""
        it = self.blocks[ident]
        return self.mapFromScene(it.sceneBoundingRect().adjusted(12, 12, -12, -12).center())

    # ================================================================== mouse
    def mousePressEvent(self, e):                                     # noqa: N802
        self.setFocus()
        self.close_name()
        self._keep_in_view = ()
        vp = e.position().toPoint()
        if e.button() == Qt.MiddleButton or (e.button() == Qt.LeftButton and self.space):
            self._start_pan(e)
            return
        if e.button() != Qt.LeftButton:
            return super().mousePressEvent(e)
        it = self.item_at(vp)
        sp = self.mapToScene(vp)
        if isinstance(it, BlockItem):
            loc = it.mapFromScene(sp)
            if it.d["kind"] == "card":
                self.bridge.choose(it.d["id"])
                return
            port = it.output_at(loc)
            if port is not None:
                self._start_wire(it, port)
                return
            if it.d["removable"] and it.hover and it.cross_rect().contains(loc):
                self.delete_block(it.id)
                return
            self.selected_wire = None
            self._pressed = True                         # the block is under the pointer: no scroll under it
            try:
                self.bridge.choose(it.id)
            finally:
                self._pressed = False
            self.gesture = dict(kind="press", id=it.id, press=e.position(), it=it)
            return
        if isinstance(it, WireItem):
            if it.style_name == "port_inactive":
                self.bridge.act("take_alternative", wire=it.ident)
                return
            if it.selectable:
                self.selected_wire = it.ident
                for f in self.wires.values():
                    f.mark(f is it)
                self.bridge.choose(None)
            else:
                self.bridge.toast.emit(_("A wire of the chain: it follows the blocks; switch off or delete a block "
                                         "to take it out"))
            return
        if isinstance(it, ProjectItem):
            self.bridge.choose(PROJECT)
            return
        if isinstance(it, LinkItem):
            self.new_workflow_menu(self.viewport().mapToGlobal(vp))
            return
        self.selected_wire = None
        for f in self.wires.values():
            f.mark(False)
        self.bridge.choose(None)
        self._start_pan(e)

    def _start_pan(self, e):
        self.gesture = dict(kind="pan", start=e.position(), h=self.horizontalScrollBar().value(),
                            v=self.verticalScrollBar().value())
        self.setCursor(Qt.ClosedHandCursor)

    def mouseMoveEvent(self, e):                                      # noqa: N802
        st = self.gesture
        if st is None:
            super().mouseMoveEvent(e)
            return
        if st["kind"] == "pan":
            d = e.position() - st["start"]
            self.horizontalScrollBar().setValue(int(st["h"] - d.x()))
            self.verticalScrollBar().setValue(int(st["v"] - d.y()))
            self.auto = False
            self._update_buttons()
        elif st["kind"] == "press":
            if (e.position() - st["press"]).manhattanLength() >= QApplication.startDragDistance() and \
                    st["it"].d["kind"] != "post":
                self._start_move(st)
        if self.gesture is not None and self.gesture["kind"] == "block":
            self._move_block(e.position())
            self._edge_scroll(e.position().toPoint())
        elif self.gesture is not None and self.gesture["kind"] == "wire":
            self._pull_wire(self.mapToScene(e.position().toPoint()))
            self._edge_scroll(e.position().toPoint())

    def mouseReleaseEvent(self, e):                                   # noqa: N802
        st, self.gesture = self.gesture, None
        self.unsetCursor()
        if st is None:
            return super().mouseReleaseEvent(e)
        if st["kind"] == "block":
            self._end_move(st, e)
        elif st["kind"] == "wire":
            self._end_wire(st, self.mapToScene(e.position().toPoint()))

    def mouseDoubleClickEvent(self, e):                               # noqa: N802
        it = self.item_at(e.position().toPoint())
        if isinstance(it, BlockItem):
            self.inspect.emit(it.id)
        elif isinstance(it, ProjectItem):
            self.inspect.emit(PROJECT)

    def new_workflow_menu(self, global_pos):
        """"New workflow": Session, Image, and "Copy of “X”" for each drawn row (never templates)."""
        bridge = self.bridge
        m = QMenu(self)
        m.addAction(_("Session"), lambda: bridge.act("add_row", kind="session"))
        m.addAction(_("Image"), lambda: bridge.act("add_row", kind="image"))
        rows = list(bridge.graph.rows)
        if rows:
            m.addSeparator()
        for r in rows:
            m.addAction(_("Copy of “{name}”").format(name=r.name),
                        lambda _c=False, rid=r.id: bridge.act("duplicate_row", row=rid))
        m.exec(global_pos)

    # ================================================================== keys
    def keyPressEvent(self, e: QKeyEvent):                            # noqa: N802
        k, mods = e.key(), e.modifiers()
        sel = self.bridge.selected
        if k == Qt.Key_Space and not e.isAutoRepeat():
            self.space = True
            self.setCursor(Qt.OpenHandCursor)
            return
        if k == Qt.Key_Z and mods & Qt.ControlModifier:        # the page's shortcut, here too when it was not active
            (self.bridge.redo() if mods & Qt.ShiftModifier else self.bridge.undo())
            return
        if k == Qt.Key_Y and mods & Qt.ControlModifier:
            self.bridge.redo()
            return
        if k == Qt.Key_Escape:
            if self.gesture is not None:
                self.cancel_gesture()
            else:
                self.selected_wire = None
                self.bridge.choose(None)
            return
        if k in (Qt.Key_Delete, Qt.Key_Backspace):
            self.delete_selection()
            return
        if k == Qt.Key_D and not mods & Qt.ControlModifier and sel in self.blocks and sel in self.bridge.graph.blocks:
            self.bridge.act("switch_off", block=sel)
            return
        if k == Qt.Key_D and mods & Qt.ControlModifier:
            r = self.bridge.row(sel) or self.bridge.row_of(sel) if sel else None
            if r is not None:
                self.bridge.act("duplicate_row", row=r.id)
            return
        if k == Qt.Key_F2 and sel in self.blocks:
            self.rename(sel)
            return
        if k in (Qt.Key_Return, Qt.Key_Enter) and sel is not None:
            self.inspect.emit(sel)
            return
        if k == Qt.Key_F or (k == Qt.Key_0 and mods & Qt.ControlModifier):
            self.show_all()
            return
        if k == Qt.Key_1:
            self.zoom_to(1.0)
            return
        if k in (Qt.Key_Plus, Qt.Key_Equal):
            self.zoom_to(self._k * 1.25)
            return
        if k == Qt.Key_Minus:
            self.zoom_to(self._k / 1.25)
            return
        super().keyPressEvent(e)

    def keyReleaseEvent(self, e):                                     # noqa: N802
        if e.key() == Qt.Key_Space and not e.isAutoRepeat():
            self.space = False
            self.unsetCursor()
            return
        super().keyReleaseEvent(e)

    def delete_selection(self):
        if self.selected_wire is not None:
            self.bridge.act("unwire", wire=self.selected_wire)
            self.selected_wire = None
            return
        sel = self.bridge.selected
        if sel in self.blocks:
            if self.blocks[sel].d["kind"] == "card":
                self.bridge.act("delete_row", row=sel)
            else:
                self.delete_block(sel)

    def cancel_gesture(self):
        self.gesture = None
        self.clean_gesture()
        self.unsetCursor()

    def clean_gesture(self):
        self._edge_scroll(None)
        for g in self.ghosts:
            self.scene().removeItem(g)
        for c in self.targets:
            self.scene().removeItem(c)
        self.ghosts, self.targets = [], []
        self.tip.hide()
        for it in self.blocks.values():
            it.dragged = False
            it.lit = {}
            it.hot_input = None
            it.target = it.target_refused = False
            it.update()
        if self.rubber is not None:
            self.scene().removeItem(self.rubber)
            self.rubber = None
        if self.floater is not None:
            self.scene().removeItem(self.floater)
            self.floater = None

    # ================================================================== edge scroll
    def _edge_scroll(self, pos):
        """A drag at ``pos`` (viewport): held near the view's edge after EDGE_WAIT, the view scrolls that way, faster
        nearer the edge (graph.py ``_edge_scroll``). None: the drag ended."""
        was = self._edge_pos
        self._edge_pos = pos
        if pos is None or self._edge_delta(pos) == (0, 0):
            self._edge_timer.stop()
            return
        if was is None or self._edge_delta(was) == (0, 0):
            self._edge_since = time.monotonic()
        if not self._edge_timer.isActive():
            self._edge_timer.start()

    def _edge_delta(self, pos) -> tuple[int, int]:
        vp = self.viewport().rect()

        def step(v, size):
            if v < EDGE:
                return -int(4 + 16 * min(1.0, (EDGE - v) / EDGE))
            if v > size - EDGE:
                return int(4 + 16 * min(1.0, (v - size + EDGE) / EDGE))
            return 0
        return step(pos.x(), vp.width()), step(pos.y(), vp.height())

    def _edge_tick(self):
        pos, st = self._edge_pos, self.gesture
        if pos is None or st is None or not self.isVisible():
            self._edge_scroll(None)
            return
        if time.monotonic() - self._edge_since < EDGE_WAIT:
            return
        dx, dy = self._edge_delta(pos)
        hb, vb = self.horizontalScrollBar(), self.verticalScrollBar()
        before = (hb.value(), vb.value())
        hb.setValue(hb.value() + dx)
        vb.setValue(vb.value() + dy)
        if (hb.value(), vb.value()) == before:
            return                                        # at the end: nothing more to show that way
        self.auto = False
        kind = st.get("kind")                             # what is under the pointer now, lit as a move would
        if kind == "block":
            self._move_block(QPointF(pos))
        elif kind == "wire":
            self._pull_wire(self.mapToScene(pos))
        elif kind == "palette":
            self._heat(self.mapToScene(pos))
        elif kind == "files":
            self._files_over(pos)

    def hideEvent(self, e):                                           # noqa: N802
        self._edge_scroll(None)
        super().hideEvent(e)

    # ================================================================== names
    def rename(self, ident, row=None):
        it = self.blocks.get(ident)
        if it is None:
            return
        bridge = self.bridge
        r = bridge.row_of(ident)
        blk = bridge.block(ident) if it.d["kind"] != "card" else None
        if row is not None:                              # the menu's "Workflow > Rename": the row's name
            r = bridge.row(row)
            name = r.name
            fn = lambda t: bridge.act("rename_row", row=row, name=t)               # noqa: E731
        elif it.d["kind"] == "card":
            r = bridge.row(ident)
            name = r.name if r else ""
            fn = lambda t: bridge.act("rename_row", row=ident, name=t)             # noqa: E731
        elif blk.kind == "session" and r is not None and len(r.nights) > 1:
            name = blk.name or it.d["title"]
            fn = lambda t: bridge.act("rename_night", night=ident, name=t)         # noqa: E731
        elif r is not None:
            name = r.name
            fn = lambda t: bridge.act("rename_row", row=r.id, name=t)              # noqa: E731
        else:
            return
        ed = NameEdit(self.viewport(), name, fn)
        tl = self.mapFromScene(it.mapToScene(QPointF(8, 4)))
        br = self.mapFromScene(it.mapToScene(QPointF(it.w - 8, 26)))
        ed.setGeometry(tl.x(), tl.y(), br.x() - tl.x(), br.y() - tl.y())
        ed.show()
        ed.setFocus()
        ed.selectAll()
        self.name_edit = ed

    def close_name(self):
        ed, self.name_edit = self.name_edit, None
        if ed is not None:
            try:
                ed.finish(cancel=True)
            except RuntimeError:
                pass

    # ================================================================== context menu
    def contextMenuEvent(self, e):                                    # noqa: N802
        it = self.item_at(e.pos())
        m = self.build_menu(it)
        if m is not None and m.actions():
            m.exec(e.globalPos())

    def build_menu(self, it):
        """The context menu of an item (block, card, wire, or the canvas when None)."""
        m = QMenu(self)
        m.setToolTipsVisible(True)
        if isinstance(it, BlockItem):
            self.bridge.choose(it.id)
            for entry in block_menu(self.doc, it.id):
                self._menu_entry(m, it.id, entry)
        elif isinstance(it, WireItem):
            if it.selectable and it.style_name == "port":
                self.selected_wire = it.ident
                a = m.addAction(_("Remove wire"))
                a.triggered.connect(lambda: self.bridge.act("unwire", wire=it.ident))
            elif it.style_name == "port_inactive":
                a = m.addAction(_("Take alternative"))
                a.triggered.connect(lambda: self.bridge.act("take_alternative", wire=it.ident))
            else:
                a = m.addAction(_("Chain wire"))
                a.setEnabled(False)
                a.setToolTip(_("A wire of the chain cannot be deleted: switch off or delete a block"))
        else:
            a = m.addAction(_("Show all"))
            a.triggered.connect(self.show_all)
        return m

    def _menu_entry(self, m, ident, entry):
        if entry is None:
            m.addSeparator()
            return
        text, op, args, enabled, why = entry
        if op == "submenu":                              # (text, "submenu", [entries], enabled, why)
            sub = m.addMenu(text)
            sub.setToolTipsVisible(True)
            sub.setEnabled(bool(enabled))
            for x in args:
                self._menu_entry(sub, ident, x)
            return
        a = m.addAction(text)
        a.setEnabled(bool(enabled))
        if why:
            a.setToolTip(why)
        if enabled:
            a.triggered.connect(lambda _c=False, o=op, ar=dict(args or {}): self.run_menu(ident, o, ar))

    def run_menu(self, ident, op, args):
        if op == "inspect":
            self.inspect.emit(ident)
        elif op == "rename":
            self.rename(ident, (args or {}).get("row"))
        elif op == "mark_reviewed":
            f = getattr(self.bridge.document, "mark_reviewed", None)
            if f is not None:
                res = self.doc.guarded(op, lambda: f(ident) or Result(True), take_back=False)
                if not res.ok:
                    self.say(self.bridge.say(res.reason))
                    return
            self.view_action.emit(op, ident)
            self.bridge.run_states_changed()
        else:
            self.bridge.act(op, **args)

    # ================================================================== a placed block dragged
    def _start_move(self, st):
        it = st["it"]
        places = self.bridge.places(block=it.id)
        self.gesture = dict(kind="block", id=it.id, it=it, press=st["press"])
        it.dragged = True
        it.update()
        self._show_ghosts(places)
        d = dict(it.d)                                   # the faint copy that follows the cursor
        d["selected"] = False
        fl = BlockItem(d)
        fl.setOpacity(0.85)
        fl.setZValue(8)
        self.scene().addItem(fl)
        self.floater = fl
        self.gesture["offset"] = QPointF(it.w / 2, it.h / 2)

    def _move_block(self, pos):
        sp = self.mapToScene(pos.toPoint())
        if self.floater is not None:
            self.floater.setPos(sp - self.gesture["offset"])
        self._heat(sp)

    def _end_move(self, st, e):
        sp = self.mapToScene(e.position().toPoint())
        hot = self._heat(sp)
        glob = e.globalPosition().toPoint()
        ident = st["id"]
        self.clean_gesture()
        if self.over_palette(glob):                      # dragged out onto the palette: deleted, as today
            self.delete_block(ident)
            return
        if hot is not None:
            self.bridge.drop(hot.place)

    # ================================================================== the places of a drag
    def _show_ghosts(self, places):
        for g in self.ghosts:
            self.scene().removeItem(g)
        self.ghosts = []
        for place in places:
            g = GhostItem(place)
            self.scene().addItem(g)
            self.ghosts.append(g)
        self.tip.set("", QRectF(), False)
        self.tip.hide()
        if self.tip.scene() is None:
            self.scene().addItem(self.tip)

    @staticmethod
    def _zone(g: GhostItem):
        """The area a pointer must be in to point at a ghost: a caret is a thin bar, given some play."""
        r = g.rect
        if g.place.kind == "insert":
            return r.adjusted(-12, -6, 12, 6)
        return r.adjusted(-3, -3, 3, 3)

    def _heat(self, sp: QPointF):
        """The ghost under a scene point lit (the lit first, the nearest centre); refused ones near the pointer shown."""
        best = None
        for g in self.ghosts:
            if self._zone(g).contains(sp):
                score = (0 if g.place.lit else 1, (g.rect.center() - sp).manhattanLength())
                if best is None or score < best[0]:
                    best = (score, g)
        hot = best[1] if best else None
        for g in self.ghosts:
            h = g is hot
            vis = g.place.lit or h or abs(g.rect.center().y() - sp.y()) < 36
            if g.isVisible() != vis:
                g.setVisible(vis)
            if g.hot != h:
                g.hot = h
                g.update()
        if hot is not None:
            pl = hot.place
            text = pl.tip_text(self.doc.tr) if pl.lit else (pl.reason_text(self.doc.tr) or pl.tip_text(self.doc.tr))
            self.tip.set(text, hot.rect, not pl.lit, below=pl.kind != "insert")
        else:
            self.tip.hide()
        return hot

    # ================================================================== drags coming from outside (palette, files)
    def dragEnterEvent(self, e):                                      # noqa: N802
        md = e.mimeData()
        if md.hasFormat(MIME_KIND):
            kind = bytes(md.data(MIME_KIND)).decode()
            self.gesture = dict(kind="palette", block=kind)
            self._show_ghosts(self.bridge.places(kind=kind))
            self._heat(self.mapToScene(e.position().toPoint()))
            e.acceptProposedAction()
        elif md.hasUrls():
            self.gesture = dict(kind="files", target=None, session=None)
            e.acceptProposedAction()
            self._files_over(e.position().toPoint())
        else:
            e.ignore()

    def dragMoveEvent(self, e):                                       # noqa: N802
        st = self.gesture
        if st is None:
            e.ignore()
            return
        if st["kind"] == "palette":
            self._heat(self.mapToScene(e.position().toPoint()))
        elif st["kind"] == "files":
            self._files_over(e.position().toPoint())
        self._edge_scroll(e.position().toPoint())
        e.acceptProposedAction()

    def dragLeaveEvent(self, e):                                      # noqa: N802
        if self.gesture is not None and self.gesture["kind"] in ("palette", "files"):
            self.gesture = None
            self.clean_gesture()

    def dropEvent(self, e):                                           # noqa: N802
        st, self.gesture = self.gesture, None
        if st is None:
            return
        sp = self.mapToScene(e.position().toPoint())
        if st["kind"] == "palette":
            hot = self._heat(sp)
            self.clean_gesture()
            e.acceptProposedAction()
            if hot is not None:
                self.bridge.drop(hot.place)
        elif st["kind"] == "files":
            self.gesture = st
            self._files_over(e.position().toPoint())
            self.gesture = None
            target, session = st.get("target"), st.get("session")
            if target is None and session is not None:
                target = "lights"
            paths = [u.toLocalFile() for u in e.mimeData().urls() if u.isLocalFile()]
            self.clean_gesture()
            e.acceptProposedAction()
            if session is not None and target is not None and paths:
                self.bridge.drops.drop(session, target, paths)       # the table's own path (files.py)

    def _session_under(self, vp: QPoint):
        """The Session block (id) a file drag is over: a Session block, or its night's Calibration."""
        it = self.item_at(vp)
        if isinstance(it, FileTarget):
            return self.gesture.get("session")
        if not isinstance(it, BlockItem) or it.d["kind"] == "card":
            return None
        if it.d["kind"] == "session":
            return it.id
        if it.d["kind"] == "calibration":
            for r in self.bridge.graph.rows:
                for s, c in r.nights:
                    if c == it.id:
                        return s
        return None

    def _files_over(self, vp: QPoint):
        st = self.gesture
        session = self._session_under(vp)
        if session != st.get("session"):
            for c in self.targets:
                self.scene().removeItem(c)
            self.targets = []
            st["session"] = session
            if session is not None:
                it = self.blocks[session]
                n = len(FILE_KINDS)
                hh, gap = 24, 4
                total = n * hh + (n - 1) * gap
                x = it.pos().x() + it.w + 10
                y0 = it.pos().y() + it.h / 2 - total / 2
                for i, k in enumerate(FILE_KINDS):
                    c = FileTarget(QRectF(x, y0 + i * (hh + gap), 110, hh), _(FILE_LABEL[k]), k)
                    self.scene().addItem(c)
                    self.targets.append(c)
        sp = self.mapToScene(vp)
        st["target"] = None
        for c in self.targets:
            h = c.rect.adjusted(-3, -2, 3, 2).contains(sp)
            if h:
                st["target"] = c.kind
            if c.hot != h:
                c.hot = h
                c.update()
        if session is not None and st["target"] is None:
            self.tip.set(_("Drop here: Lights; or on one of the targets"), self.blocks[session].sceneBoundingRect(),
                         False)
        else:
            self.tip.hide()

    # ================================================================== pulling a wire
    def _start_wire(self, it, port):
        bridge = self.bridge
        drawing = bridge.drawing()
        src = it.id
        cands = []                                       # (block id, port name)
        for (bb, p, sense) in drawing.ports:
            if sense == "in" and bb in self.blocks and self.blocks[bb].d["kind"] in JOIN_KINDS:
                cands.append((bb, p))
        for b2, it2 in self.blocks.items():
            if it2.d["kind"] == "post":
                cands.append((b2, "layers"))
        st = dict(kind="wire", src=src, port=port, cands={}, hot=None, it=it)
        for dst, dport in cands:
            if dst == src:
                continue
            why = bridge.check_wire(src, port, dst, dport)
            st["cands"][(dst, dport)] = why
            tgt = self.blocks[dst]
            if tgt.d["kind"] == "post":
                continue
            tgt.lit[dport] = not why
            tgt.update()
        it.hot_output = True
        self.gesture = st
        self.rubber = RubberWire()
        self.scene().addItem(self.rubber)
        a = it.mapToScene(QPointF(it.w, dict((p, y) for p, y, _c, _w in it.d["outputs"]).get(port, it.h / 2)))
        st["a"] = a
        self.rubber.stretch(a, a)
        if self.tip.scene() is None:
            self.scene().addItem(self.tip)

    def _wire_target(self, sp, st=None):
        for (dst, dport), why in (st or self.gesture)["cands"].items():
            tgt = self.blocks[dst]
            loc = tgt.mapFromScene(sp)
            if tgt.d["kind"] == "post":
                if QRectF(0, 0, tgt.w, tgt.h).adjusted(-6, -6, 6, 6).contains(loc):
                    return dst, dport, why
            elif tgt.input_at(loc) == dport:
                return dst, dport, why
        return None

    def _pull_wire(self, sp):
        st = self.gesture
        self.rubber.stretch(st["a"], sp)
        hit = self._wire_target(sp)
        for it in self.blocks.values():
            it.hot_input = None
            if it.target or it.target_refused:
                it.target = it.target_refused = False
        if hit is not None:
            dst, dport, why = hit
            tgt = self.blocks[dst]
            if tgt.d["kind"] == "post":
                tgt.target, tgt.target_refused = not why, bool(why)
            else:
                tgt.hot_input = (dport, not why)
            tgt.update()
            self.tip.set(why or _("Wire to {port}").format(port=_(dport)), tgt.sceneBoundingRect(), bool(why),
                         below=True)
        else:
            self.tip.hide()
        st["hot"] = hit

    def _end_wire(self, st, sp):
        hit = self._wire_target(sp, st)
        src, port = st["src"], st["port"]
        self.clean_gesture()
        if hit is None:
            return
        dst, dport, why = hit
        if why:
            self.bridge.toast.emit(why)
            return
        self.bridge.act("wire", src=src, src_port=port, dst=dst, dst_port=dport)
