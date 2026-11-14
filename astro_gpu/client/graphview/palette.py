"""The palette on the left: the client's groups and icon cards. A card is dragged onto the graph (ghost places over
the unchanged drawing) or placed at its natural place by a double-click; greyed, it says why on hover."""
from __future__ import annotations

import textwrap

from PySide6.QtCore import QMimeData, QPoint, Qt, QTimer
from PySide6.QtGui import QDrag
from PySide6.QtWidgets import QApplication, QFrame, QHBoxLayout, QLabel, QScrollArea, QVBoxLayout

from ..i18n import _
from . import style
from .canvas import MIME_KIND
from .style import ACCENT, GROUPS, PALETTE_W, ROLE, kind_title, kind_what


def _capital(text: str) -> str:
    return text[:1].upper() + text[1:]


class Card(QFrame):
    """A block of the palette."""

    def __init__(self, palette, kind):
        super().__init__()
        self.palette, self.kind = palette, kind
        self.enabled, self.reason = True, ""
        self.hint = ""                   # refused for the current workflow: where it can still go (graph.py drag_only)
        self._press = None
        self.setObjectName("tool")
        self.setMaximumWidth(PALETTE_W - 26)
        h = QHBoxLayout(self)
        h.setContentsMargins(7, 5, 7, 5)
        h.setSpacing(5)
        self.icon = style.IconLabel(ROLE[kind], 16, "normal")
        h.addWidget(self.icon, 0, Qt.AlignTop)
        title = kind_title(kind)
        self.title = QLabel(title)
        self.title.setWordWrap(True)
        self.title.ensurePolished()
        small = max(self.title.fontMetrics().horizontalAdvance(w) for w in title.split()) > PALETTE_W - 64
        self.base = "font-weight: 600; font-size: %s;" % ("8pt" if small else "9pt")
        h.addWidget(self.title, 1)
        self.set_state(True, "")

    def set_state(self, enabled, reason, hint=""):
        """Enabled or greyed with its reason; ``hint``: the card's place in the current workflow is refused (its reason
        in ``reason``) but it can still be dragged elsewhere, said in words."""
        self.enabled, self.reason, self.hint = bool(enabled), reason or "", hint or ""
        on = self.enabled
        self.setCursor(Qt.OpenHandCursor if on else Qt.ForbiddenCursor)
        self.setStyleSheet(f"QFrame#tool {{ background: {'#232831' if on else '#1e2128'}; border: 1px solid "
                           f"{'#353b47' if on else '#262a31'}; border-radius: 6px; }}"
                           f"QFrame#tool:hover {{ border-color: {ACCENT if on else '#3a4150'}; }}"
                           " QLabel { background: transparent; }")
        self.icon.state = "normal" if on else "disabled"
        self.icon.update()
        self.title.setStyleSheet(self.base + ("" if on else " color: #555b66;"))
        what = textwrap.fill(kind_what(self.kind), 70)
        if on and self.hint:
            self.setToolTip(what + "\n" + _capital(self.reason) + "\n" + self.hint)
        else:
            self.setToolTip(what if on or not self.reason else kind_title(self.kind) + "\n" + _capital(self.reason))

    def mousePressEvent(self, e):                                     # noqa: N802
        self._press = e.position().toPoint()

    def mouseMoveEvent(self, e):                                      # noqa: N802
        if self._press is None or not (e.buttons() & Qt.LeftButton) or not self.enabled:
            return
        if (e.position().toPoint() - self._press).manhattanLength() < QApplication.startDragDistance():
            return
        self._press = None
        self.drag()

    def drag(self):
        drag = QDrag(self)
        mime = QMimeData()
        mime.setData(MIME_KIND, self.kind.encode())
        drag.setMimeData(mime)
        drag.setPixmap(self.grab())
        drag.setHotSpot(QPoint(20, 12))
        drag.exec(Qt.CopyAction)

    def mouseReleaseEvent(self, e):                                   # noqa: N802
        self._press = None

    def mouseDoubleClickEvent(self, e):                               # noqa: N802
        if self.enabled:
            self.palette.double_click(self.kind)
        elif self.reason:
            self.palette.bridge.toast.emit(_capital(self.reason))


class Palette(QScrollArea):
    def __init__(self, bridge):
        super().__init__()
        self.bridge = bridge
        self.cards: dict[str, Card] = {}
        self.setFixedWidth(PALETTE_W)
        self.setFrameShape(QFrame.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setStyleSheet("QScrollArea { background: #1c1f26; border-right: 1px solid #2a2f38; } "
                           "QScrollBar:vertical { width: 8px; background: #1c1f26; } QScrollBar::handle:vertical "
                           "{ background: #3a4150; border-radius: 4px; min-height: 40px; } QScrollBar::add-line, "
                           "QScrollBar::sub-line { height: 0; }")
        body = QFrame()
        body.setObjectName("tools")
        body.setStyleSheet("QFrame#tools { background: #1c1f26; }")
        v = QVBoxLayout(body)
        v.setContentsMargins(10, 10, 14, 10)
        v.setSpacing(4)
        head = QLabel(_("Blocks"))
        head.setStyleSheet("font-weight: 600; font-size: 10.5pt; color: #e8ebf0;")
        self.head_row = QHBoxLayout()
        self.head_row.setContentsMargins(0, 0, 0, 0)
        self.head_row.addWidget(head, 1)
        v.addLayout(self.head_row)
        self.for_row = style.label("", "small", wrap=True)
        v.addWidget(self.for_row)
        for title, kinds in GROUPS:
            h = QLabel(_(title).upper())
            h.setStyleSheet("color: #8b93a1; font-size: 8pt; font-weight: 600; margin-top: 8px;")
            v.addWidget(h)
            for k in kinds:
                c = Card(self, k)
                self.cards[k] = c
                v.addWidget(c)
        v.addStretch(1)
        self.setWidget(body)
        self.setWidgetResizable(True)
        self._gen = 0
        self._queue: list[str] = []
        bridge.changed.connect(self.refresh)
        bridge.selection.connect(lambda *_a: self.refresh())
        self.refresh()
        self.finish()                    # the first time: all at once, nothing is waiting for the window yet

    def current_row(self):
        """The row the palette works for: the selected block's row (or the selected card), else None (first row)."""
        bridge = self.bridge
        sel = bridge.selected
        if sel is None:
            return None
        return bridge.row(sel) or bridge.row_of(sel)

    def refresh(self):
        """The cards' greyed states: the core needs a few tens of ms a card, so they are brought up to date one card
        per turn of the event loop, after the drop's repaint (a drop must feel instant), and again from the start if
        a change comes."""
        r = self.current_row()
        self.for_row.setText(_("for “{name}”").format(name=r.name) if r is not None else "")
        self.for_row.setVisible(r is not None)
        self._gen += 1
        self._queue = list(self.cards)
        QTimer.singleShot(0, self, lambda g=self._gen: self._step(g))     # (cancelled if the palette is deleted)

    def _step(self, gen):
        if gen != self._gen or not self._queue:
            return
        k = self._queue.pop(0)
        ok, why = self.bridge.doc.palette_state(k)
        hint = ""
        if ok:
            why, hint = self.hint(k)
        self.cards[k].set_state(ok, why, hint)
        if self._queue:
            QTimer.singleShot(0, self, lambda g=gen: self._step(g))

    def finish(self):
        """Bring every card up to date now (tests, first display)."""
        while self._queue:
            self._step(self._gen)

    def hint(self, kind):
        """(reason, words) when the current workflow refuses ``kind`` at its natural place while the card can still be
        dragged elsewhere ("Drag it onto another workflow", "Drag it between blocks for a new branch", as today's
        palette says); ("", "") else."""
        r = self.current_row()
        if r is None or kind in ("session", "image", "combine", "combine_layers"):
            return "", ""
        doc = self.bridge.doc
        res = doc.try_many([("add_block", {"kind": kind, "row": r.id})])
        if res.ok:
            return "", ""
        lit = [p for p in self.bridge.places(kind=kind) if p.lit]
        if any(p.kind != "branch" for p in lit):
            return doc.say(res.reason), _("Drag it onto another workflow")
        if lit:
            return doc.say(res.reason), _("Drag it between blocks for a new branch")
        return "", ""

    def double_click(self, kind):
        """A card double-clicked: it places and wires itself at its natural place; refused there, the toast also says
        where it can still be dragged."""
        card = self.cards.get(kind)
        r = self.current_row()
        if card is not None and card.hint and kind not in ("session", "image"):
            res = self.bridge.act("add_block", quiet="refusal", kind=kind, row=r.id if r is not None else None)
            if not res.ok:
                self.bridge.toast.emit(_capital(self.bridge.say(res.reason)) + " · " + card.hint)
            return
        bridge = self.bridge
        r = self.current_row()
        if kind == "session":
            if r is not None and r.nights:
                bridge.act("add_night", row=r.id)
            else:
                bridge.act("add_row", kind="session")
        elif kind == "image":
            bridge.act("add_row", kind="image")
        else:
            bridge.act("add_block", kind=kind, row=r.id if r is not None else None)

    def contains(self, global_pos) -> bool:
        return self.rect().contains(self.mapFromGlobal(global_pos))
