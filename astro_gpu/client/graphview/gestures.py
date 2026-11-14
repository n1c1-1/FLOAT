"""Scripted gestures that perform REAL Qt events on the page (mouse press / move / release, drag-and-drop enter / move
/ drop, keys, a context-menu pick), at scene items found by id or by place. Used by the tests and the window fuzz.

    visible_places("gradient")                       -> [dict(i, kind, lit, tip, reason, row, line, col, op, rect)]
    drag_palette("gradient", place)                  place: an index, a Place, a dict of attributes, a tip fragment
    double_click_palette("session")
    move_block("b4", place)                          drag a placed block by the mouse
    pull_wire("b7", "Starless", "b9", "Background")  pull a wire from an output port to an input port
    menu_pick("b5", "Switch off")                    context menu of a block, pick the item whose text contains this
    key("Ctrl+Z")                                    a key (the focused widget gets it)
    drop_files("b1", "darks", [paths])               files dragged from the Explorer over a Session, onto a target

Every gesture returns a dict {"ok": bool, "toast": str, ...} describing what the page showed afterwards.
"""
from __future__ import annotations

from PySide6.QtCore import QEvent, QMimeData, QPoint, QPointF, Qt, QTimer, QUrl
from PySide6.QtGui import (
    QContextMenuEvent,
    QDragEnterEvent,
    QDragLeaveEvent,
    QDragMoveEvent,
    QDropEvent,
    QKeySequence,
    QMouseEvent,
)
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMenu

from .canvas import MIME_KIND


def pump(n=3):
    for _i in range(n):
        QApplication.processEvents()


class Gestures:
    def __init__(self, page):
        self.page = page
        self.bridge = page.bridge
        self.last_toast = ""
        self.toasts: list[str] = []
        self.bridge.toast.connect(self._toast)
        self.bridge.rich_toast.connect(lambda t, _l: self._toast(t() if callable(t) else t))

    def _toast(self, t):
        self.last_toast = t
        self.toasts.append(t)

    # ------------------------------------------------------------------ geometry
    @property
    def canvas(self):
        return self.page.canvas

    @property
    def vp(self):
        return self.page.canvas.viewport()

    def _vp_pt(self, sp: QPointF) -> QPoint:
        return self.canvas.mapFromScene(sp)

    def _rect_pt(self, rect) -> QPoint:
        return self._vp_pt(QPointF(rect.x + rect.w / 2, rect.y + rect.h / 2))

    def block_point(self, ident) -> QPoint:
        """Viewport point on a block (its centre, nudged off the delete cross)."""
        d = self.bridge.drawing()
        box = d.boxes.get(ident) or d.cards[ident]
        return self._vp_pt(QPointF(box.x + box.w * 0.45, box.y + box.h * 0.45))

    def port_point(self, ident, port, sense) -> QPoint:
        d = self.bridge.drawing()
        box = d.boxes[ident]
        xy = d.ports.get((ident, port, sense))
        if xy is None:
            xy = (box.x + (box.w if sense == "out" else 0), box.y + box.h / 2)
        return self._vp_pt(QPointF(*xy))

    def _see(self, pt: QPoint):
        """Scroll so a viewport point is inside (a gesture aims at what is visible)."""
        r = self.vp.rect().adjusted(20, 20, -20, -20)
        if not r.contains(pt):
            sp = self.canvas.mapToScene(pt)
            self.canvas.ensureVisible(sp.x(), sp.y(), 40, 40, 60, 60)

    # ------------------------------------------------------------------ low-level events
    def _mouse(self, type_, pt: QPoint, button=Qt.LeftButton, buttons=Qt.LeftButton, mods=Qt.NoModifier):
        g = self.vp.mapToGlobal(pt)
        QApplication.sendEvent(self.vp, QMouseEvent(type_, QPointF(pt), QPointF(g), button, buttons, mods))

    def press(self, pt):
        self._mouse(QEvent.MouseButtonPress, pt)

    def move(self, pt, held=True):
        self._mouse(QEvent.MouseMove, pt, Qt.NoButton, Qt.LeftButton if held else Qt.NoButton)

    def release(self, pt, global_pt=None):
        g = global_pt if global_pt is not None else self.vp.mapToGlobal(pt)
        QApplication.sendEvent(self.vp, QMouseEvent(QEvent.MouseButtonRelease, QPointF(pt), QPointF(g), Qt.LeftButton,
                                                    Qt.NoButton, Qt.NoModifier))

    @staticmethod
    def _path(a: QPoint, b: QPoint, n=6):
        return [QPoint(int(a.x() + (b.x() - a.x()) * i / n), int(a.y() + (b.y() - a.y()) * i / n))
                for i in range(1, n + 1)]

    def _mouse_drag(self, a: QPoint, b: QPoint, global_end=None):
        self.press(a)
        for p in self._path(a, b):
            self.move(p)
        self.release(b, global_end)
        pump()

    # ------------------------------------------------------------------ places
    def places(self, kind=None, block=None):
        return self.bridge.places(kind=kind, block=block)

    def visible_places(self, kind=None, block=None):
        """The places of a drag as the user would see them: dicts, `i` is the index to give back."""
        out = []
        d = self.bridge.drawing()
        say = self.bridge.say
        for i, pl in enumerate(self.places(kind, block)):
            line = d.lines[pl.line][0] if isinstance(pl.line, int) and 0 <= pl.line < len(d.lines) else None
            out.append(dict(i=i, kind=pl.kind, lit=pl.lit, tip=say(pl.tip), reason=say(pl.reason), row=line,
                            line=pl.line, col=pl.col, op=pl.op, rect=pl.rect))
        return out

    def find_place(self, places, place):
        if isinstance(place, int):
            return places[place]
        if hasattr(place, "rect") and hasattr(place, "op"):
            return place
        say = self.bridge.say
        if isinstance(place, str):
            for pl in places:
                if place in say(pl.tip):
                    return pl
            raise LookupError(f"no place whose tip contains {place!r}")
        if isinstance(place, dict):
            d = self.bridge.drawing()
            for i, pl in enumerate(places):
                ok = True
                for k, v in place.items():
                    if k == "i":
                        ok &= i == v
                    elif k == "tip":
                        ok &= v in say(pl.tip)
                    elif k == "op":
                        ok &= pl.op[0] == v
                    elif k == "row":
                        ok &= isinstance(pl.line, int) and 0 <= pl.line < len(d.lines) and d.lines[pl.line][0] == v
                    else:
                        ok &= getattr(pl, k, None) == v
                if ok:
                    return pl
            raise LookupError(f"no place matching {place!r}")
        raise TypeError(place)

    def _place_centre(self, pl) -> QPoint:
        return self._rect_pt(pl.rect)

    # ------------------------------------------------------------------ the gestures
    def _report(self, **more):
        pump()
        d = dict(ok=True, toast=self.last_toast)
        d.update(more)
        return d

    def _drag_in(self, mime, a: QPoint, target: QPoint):
        """Drag events into the canvas: enter, a few moves, the move on the target, then drop."""
        acts = Qt.CopyAction
        QApplication.sendEvent(self.vp, QDragEnterEvent(a, acts, mime, Qt.LeftButton, Qt.NoModifier))
        for p in self._path(a, target):
            QApplication.sendEvent(self.vp, QDragMoveEvent(p, acts, mime, Qt.LeftButton, Qt.NoModifier))
        QApplication.sendEvent(self.vp, QDropEvent(QPointF(target), acts, mime, Qt.LeftButton, Qt.NoModifier))
        pump()

    def drag_palette(self, kind, place, give_up=False):
        """Drag the palette card of `kind` onto a place and drop (give_up: leave the canvas instead of dropping)."""
        pl = self.find_place(self.places(kind=kind), place)
        self._see(self._place_centre(pl))
        target = self._place_centre(pl)
        mime = QMimeData()
        mime.setData(MIME_KIND, kind.encode())
        start = QPoint(self.vp.width() // 2, 4)
        self.last_toast = ""
        if give_up:
            QApplication.sendEvent(self.vp, QDragEnterEvent(start, Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier))
            for p in self._path(start, target):
                QApplication.sendEvent(self.vp, QDragMoveEvent(p, Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier))
            QApplication.sendEvent(self.vp, QDragLeaveEvent())
            return self._report(place=pl)
        self._drag_in(mime, start, target)
        return self._report(place=pl)

    def double_click_palette(self, kind):
        self.last_toast = ""
        self.page.palette.finish()                    # the cards' states are brought up to date one per event turn
        QTest.mouseDClick(self.page.palette.cards[kind], Qt.LeftButton)
        return self._report()

    def move_block(self, block, place, to_palette=False):
        """Drag a placed block by the mouse onto a place of its kind (or onto the palette: delete)."""
        self.last_toast = ""
        self._see(self.block_point(block))
        a = self.block_point(block)
        if to_palette:
            pal = self.page.palette
            gpt = pal.mapToGlobal(QPoint(pal.width() // 2, pal.height() // 2))
            self._mouse_drag(a, self.vp.mapFromGlobal(gpt), gpt)
            return self._report()
        pl = self.find_place(self.places(block=block), place)
        self._see(self._place_centre(pl))
        a = self.block_point(block)
        self._mouse_drag(a, self._place_centre(pl))
        return self._report(place=pl)

    def _in_point(self, dst, port):
        if self.bridge.graph.blocks[dst].kind == "post":
            return self.block_point(dst)
        return self.port_point(dst, port, "in") + QPoint(6, 0)

    def pull_wire(self, src, src_port, dst, dst_port):
        """Pull a wire from an output port to an input port (Post: port "layers")."""
        self.last_toast = ""
        self._see(self.port_point(src, src_port, "out"))
        self._see(self._in_point(dst, dst_port))
        self._mouse_drag(self.port_point(src, src_port, "out"), self._in_point(dst, dst_port))
        return self._report()

    def click(self, ident, double=False):
        self._see(self.block_point(ident))
        a = self.block_point(ident)
        self.press(a)
        self.release(a)
        if double:
            QApplication.sendEvent(self.vp, QMouseEvent(QEvent.MouseButtonDblClick, QPointF(a),
                                                        QPointF(self.vp.mapToGlobal(a)), Qt.LeftButton,
                                                        Qt.LeftButton, Qt.NoModifier))
            self.release(a)
        pump()
        return self._report()

    def click_cross(self, ident):
        """Hover a block, then click its delete cross."""
        self._see(self.block_point(ident))
        it = self.canvas.blocks[ident]
        it.hover = True
        pt = self._vp_pt(it.mapToScene(it.cross_rect().center()))
        self.last_toast = ""
        self.press(pt)
        self.release(pt)
        pump()
        return self._report()

    def display(self, key):
        """Click a position of the PREPROCESSING display control ("blocks", "folded", "table")."""
        self.last_toast = ""
        QTest.mouseClick(self.page.positions[key], Qt.LeftButton)
        pump()
        return self._report()

    def click_wire(self, wire_id):
        it = self.canvas.wires[wire_id]
        pt = self._vp_pt(it.path().pointAtPercent(0.5))
        self.press(pt)
        self.release(pt)
        pump()
        return self._report()

    def menu_pick(self, block, text, open_only=False):
        """Open the context menu of a block with a real context-menu event and pick the item whose text contains
        `text` (in a submenu too). The result says whether the item was enabled, and lists the items."""
        self.last_toast = ""
        self._see(self.block_point(block))
        pt = self.block_point(block)
        res = {"ok": False, "items": []}

        def actions(m):
            out = []
            for a in m.actions():
                if a.isSeparator():
                    continue
                out.append(a)
                if a.menu() is not None:
                    out += actions(a.menu())
            return out

        def pick():
            m = QApplication.activePopupWidget()
            if not isinstance(m, QMenu):
                return
            acts = actions(m)
            res["items"] = [(a.text(), a.isEnabled(), a.toolTip()) for a in acts]
            target = next((a for a in acts if a.menu() is None and text and text in a.text()), None)
            if target is not None and not open_only:
                res["ok"] = target.isEnabled()
                res["why"] = target.toolTip()
                m.close()
                if target.isEnabled():
                    target.trigger()
            else:
                m.close()
        QTimer.singleShot(60, pick)
        QApplication.sendEvent(self.vp, QContextMenuEvent(QContextMenuEvent.Mouse, pt, self.vp.mapToGlobal(pt)))
        pump()
        out = self._report(**{k: v for k, v in res.items() if k != "ok"})
        out["ok"] = res["ok"]
        return out

    def key(self, keys, target=None):
        """A key given as text: "Ctrl+Z", "Del", "D", "F2", "Return", "Esc", "+" ..."""
        self.last_toast = ""
        alias = {"Del": "Delete", "Esc": "Escape", "Enter": "Return"}
        txt = "+".join(alias.get(p, p) for p in keys.split("+")) if keys != "+" else "+"
        seq = QKeySequence(txt)
        combo = seq[0].toCombined() if hasattr(seq[0], "toCombined") else int(seq[0])
        mods = Qt.KeyboardModifier(combo & 0xFE000000)
        k = Qt.Key(combo & 0x01FFFFFF)
        QTest.keyClick(target or QApplication.focusWidget() or self.canvas, k, mods)
        pump()
        return self._report()

    def drop_files(self, session, kind, paths):
        """Files dragged from the Explorer over a Session block: its five targets appear, drop on `kind`."""
        self.last_toast = ""
        mime = QMimeData()
        mime.setUrls([QUrl.fromLocalFile(p) for p in paths])
        self._see(self.block_point(session))
        a = self.block_point(session)
        QApplication.sendEvent(self.vp, QDragEnterEvent(a, Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier))
        QApplication.sendEvent(self.vp, QDragMoveEvent(a, Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier))
        target = next((c for c in self.canvas.targets if c.kind == kind), None)
        if target is None:
            QApplication.sendEvent(self.vp, QDropEvent(QPointF(a), Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier))
            return self._report(ok=False, targets=[])
        labels = [c.text for c in self.canvas.targets]
        b = self._vp_pt(target.rect.center())
        for p in self._path(a, b):
            QApplication.sendEvent(self.vp, QDragMoveEvent(p, Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier))
        QApplication.sendEvent(self.vp, QDropEvent(QPointF(b), Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier))
        pump()
        return self._report(targets=labels)

    # ------------------------------------------------------------------ reading what the user sees
    def dump(self):
        """A plain-text dump of what is drawn: for the scripts and the fuzz."""
        d, doc = self.bridge.drawing(), self.bridge.doc
        return "\n".join(f"{b:>5} {doc.title(b):<22} x={box.x:.0f} y={box.y:.0f}  {doc.summary(b)}"
                         for b, box in sorted(d.boxes.items(), key=lambda kv: (kv[1].y, kv[1].x)))

    def select(self, ident):
        self.bridge.choose(ident)
        pump()

    def wait(self, ms=0):
        QTest.qWait(ms)

    def shot(self, path):
        pump()
        self.page.grab().save(str(path))
        return path
