"""The palette on the left: the client's groups and icon cards. A card is dragged onto the graph (ghost places over
the unchanged drawing) or placed at its natural place by a double-click; greyed, it says why on hover.

Its three ways of showing are today's (graph.py ``palette_menu``, its "⋯" button "Blocks display"): Always open, Always
folded (the blocks' icons only, their names in the tooltips, the headers hidden) and Automatic (the same strip of icons,
opened over the graph after the mouse rests 200 ms on it, folded 300 ms after it left). Hidden (Tab), the page leaves a
small arrow at the window's edge. The window keeps the choice (``chosen``, then ``set_state`` through the page's
``set_layout``)."""
from __future__ import annotations

import textwrap

from PySide6.QtCore import QEvent, QMimeData, QPoint, Qt, QTimer, Signal
from PySide6.QtGui import QDrag
from PySide6.QtWidgets import QApplication, QFrame, QHBoxLayout, QLabel, QMenu, QScrollArea, QToolButton, QVBoxLayout

from ..i18n import _
from ..names import shown
from ..layout import FOLD_MS, UNFOLD_MS
from . import style
from .canvas import MIME_KIND
from .style import ACCENT, GROUPS, PALETTE_FOLDED_W, PALETTE_W, ROLE, kind_title, kind_what

#: the palette's three ways of showing, as today's menu words them (graph.py palette_menu)
SHOWN = (("auto", "Automatic"), ("docked", "Always open"), ("folded", "Always folded"))


def _capital(text: str) -> str:
    return text[:1].upper() + text[1:]


def recipe_tip(spec: dict | None) -> str:
    """A combination recipe's tooltip: its whole name, what it does, its inputs (translated)."""
    if not spec:
        return ""
    words = {"mono": _("monochrome"), "colour": _("colour"), "any": _("any image")}
    ins = ", ".join(f"{_(i.get('label') or i['role'])} ({words.get(i.get('type'), '')})" for i in spec["inputs"])
    out = [_(spec["name"])]
    if spec.get("description"):
        out.append(textwrap.fill(_(spec["description"]), 70))
    out.append(_("Inputs: {names}").format(names=ins))
    return "\n".join(out)


class Card(QFrame):
    """A block of the palette; narrow (folded or automatic): its icon alone, its name in the tooltip."""

    def __init__(self, palette, kind):
        super().__init__()
        self.palette, self.kind = palette, kind
        self.enabled, self.reason = True, ""
        self.hint = ""                   # refused for the current workflow: where it can still go (graph.py drag_only)
        self.narrow = False
        self._press = None
        self.setObjectName("tool")
        self.setMaximumWidth(PALETTE_W - 26)
        h = QHBoxLayout(self)
        h.setContentsMargins(7, 5, 7, 5)
        h.setSpacing(5)
        self.box = h
        self.icon = style.IconLabel(ROLE[kind], 16, "normal")
        h.addWidget(self.icon, 0, Qt.AlignTop)
        title = kind_title(kind)
        self.title = QLabel(title)
        self.title.setWordWrap(True)
        self.title.ensurePolished()
        small = max(self.title.fontMetrics().horizontalAdvance(w) for w in title.split()) > PALETTE_W - 64
        self.base = "font-weight: 600; font-size: %s;" % ("8pt" if small else "9pt")
        h.addWidget(self.title, 1)
        self.choose = None
        if kind == "combine":            # the recipe it places, on the card itself, and its list one click away
            from ..graphnode import combine_title
            self.title.setText(combine_title(palette.bridge.combo_type))
            self.choose = QToolButton()
            self.choose.setText("▾")
            self.choose.setAutoRaise(True)
            self.choose.setObjectName("recipe_choice")
            self.choose.setToolTip(_("Recipe: the combination this card places (S-H-O, H-O, RGB, LRGB, HOO, HaRGB…); "
                                     "its block's panel changes it later"))
            self.choose.setStyleSheet("QToolButton { border: none; padding: 0 2px; color: #8b93a1; } "
                                      "QToolButton:hover { color: #e8ebf0; }")
            self.choose.clicked.connect(lambda: palette.combination_types(self))
            h.addWidget(self.choose, 0, Qt.AlignTop)
        self.set_state(True, "")

    # the hint asks the core whether the current workflow takes the card at its natural place: a dry run of the
    # gesture, so it is asked when it is read (the tooltip shown, a double-click, a test), never for every card at
    # every change; the palette sets the card again at each change and each choice of a block (``_due``)
    @property
    def hint(self):
        self._resolve()
        return self._hint

    @hint.setter
    def hint(self, value):
        self._hint, self._due = value or "", False

    @property
    def reason(self):
        self._resolve()
        return self._reason

    @reason.setter
    def reason(self, value):
        self._reason = value or ""

    def _resolve(self):
        if getattr(self, "_due", False):
            self._due = False
            why, self._hint = self.palette.hint(self.kind)
            self._reason = why or ""
            self._tip()

    def toolTip(self):                                                # noqa: N802
        self._resolve()
        return super().toolTip()

    def event(self, e):
        if e.type() == QEvent.ToolTip:
            self._resolve()
        return super().event(e)

    def set_state(self, enabled, reason, hint="", due=False):
        """Enabled or greyed with its reason; ``hint``: the card's place in the current workflow is refused (its reason
        in ``reason``) but it can still be dragged elsewhere, said in words; ``due``: enabled, that hint is asked of
        the core when first read."""
        same = getattr(self, "_shown_state", None) == (bool(enabled), reason or "", hint or "", bool(due))
        self.enabled, self._reason, self._hint = bool(enabled), reason or "", hint or ""
        self._due = bool(due) and self.enabled
        if same:                     # restyled again while a run goes, the card under the mouse never showed its reason
            return
        self._shown_state = (self.enabled, self._reason, self._hint, bool(due))
        on = self.enabled
        self.setCursor(Qt.OpenHandCursor if on else Qt.ForbiddenCursor)
        self.setStyleSheet(f"QFrame#tool {{ background: {'#232831' if on else '#1e2128'}; border: 1px solid "
                           f"{'#353b47' if on else '#262a31'}; border-radius: 6px; }}"
                           f"QFrame#tool:hover {{ border-color: {ACCENT if on else '#3a4150'}; }}"
                           " QLabel { background: transparent; }")
        self.icon.state = "normal" if on else "disabled"
        self.icon.update()
        self.title.setStyleSheet(self.base + ("" if on else " color: #555b66;"))
        self._tip()

    def _tip(self):
        on, reason, hint = self.enabled, self._reason, self._hint
        what = textwrap.fill(kind_what(self.kind), 70)
        if self.kind == "combine":                       # the recipe it places now (its list: a click on the card)
            from ...graphcore import combos
            what += "\n\n" + recipe_tip(combos.recipe(self.palette.bridge.combo_type))
        if on and hint:
            tip = what + "\n" + _capital(reason) + "\n" + hint
        elif on or not reason:
            tip = what
        else:
            tip = _capital(reason)
        # folded (graph.py PaletteCard): the name heads the tooltip, the card shows only its icon
        name = kind_title(self.kind)
        self.setToolTip(name + "\n" + tip if (self.narrow or not on and reason and not hint) else tip)

    def set_narrow(self, narrow: bool):
        """Folded or automatic: the icon alone, centred in the strip; the name comes back with the tooltip's."""
        if narrow == self.narrow:
            return
        self.narrow = narrow
        self.title.setVisible(not narrow)
        if self.choose is not None:
            self.choose.setVisible(not narrow)
        self.setMaximumWidth(PALETTE_FOLDED_W - 12 if narrow else PALETTE_W - 26)
        self.box.setContentsMargins(*((8, 6, 8, 6) if narrow else (7, 5, 7, 5)))
        self._tip()

    def keep_height(self, on: bool):
        """Automatic palette: the card as tall folded as opened (its name on one line or two), so that opening it
        under the mouse moves no card; otherwise its own height."""
        if on:
            room = PALETTE_W - 26 - 7 - 7 - 16 - 5                # the card's width less its margins and icon
            h = max(self.title.heightForWidth(room), self.title.fontMetrics().height(), 16) + 10
            self.setFixedHeight(max(h, 28))
        elif self.minimumHeight() == self.maximumHeight():
            self.setMinimumHeight(0)
            self.setMaximumHeight(16777215)

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
        self.palette.unfold(False)               # automatic: the places it covered are the first ones (graph.py)
        drag.exec(Qt.CopyAction)

    def mouseReleaseEvent(self, e):                                   # noqa: N802
        pressed, self._press = self._press, None
        if self.kind == "combine" and pressed is not None and e.button() == Qt.LeftButton and self.enabled:
            if getattr(self, "_flyout", None) is None:     # today's flyout of the combination's types, once the
                self._flyout = QTimer(self, singleShot=True, interval=QApplication.doubleClickInterval())
                self._flyout.timeout.connect(lambda: self.palette.combination_types(self))     # double-click is
            self._flyout.start()                                                                 # not coming

    def mouseDoubleClickEvent(self, e):                               # noqa: N802
        if getattr(self, "_flyout", None) is not None:
            self._flyout.stop()
        if self.enabled:
            self.palette.double_click(self.kind)
        elif self.reason:
            self.palette.bridge.toast.emit(_capital(self.reason))


class Palette(QScrollArea):
    chosen = Signal(str)                 # the user's choice in the "⋯" menu: "auto" | "docked" | "folded"

    def __init__(self, bridge):
        super().__init__()
        self.bridge = bridge
        self.cards: dict[str, Card] = {}
        self.state = "docked"            # docked | auto | folded | hidden (the window's, through set_state)
        self._unfolded = False           # automatic: opened over the graph now
        self.on_fit = None               # the page places the palette again when its width changes
        self._unfold_timer = QTimer(self, singleShot=True, interval=UNFOLD_MS)
        self._unfold_timer.timeout.connect(lambda: self.unfold(True))
        self._fold_timer = QTimer(self, singleShot=True, interval=FOLD_MS)
        self._fold_timer.timeout.connect(lambda: self.unfold(False))
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
        self.body_box = v
        self.head = QLabel(_("Blocks"))
        self.head.setStyleSheet("font-weight: 600; font-size: 10.5pt; color: #e8ebf0;")
        self.head_row = QHBoxLayout()
        self.head_row.setContentsMargins(0, 0, 0, 0)
        self.head_row.addWidget(self.head, 1)
        # the display button: a click folds or opens the palette at once; a right click gives its three ways of showing
        self.display_button = QToolButton()
        style.icon_only(self.display_button, "more", _("Fold or open the blocks · right click: Automatic, Always "
                                                       "open, Always folded"), 16, "dim")
        self.display_button.setAutoRaise(True)
        self.display_button.setFixedSize(20, 18)
        self.display_button.setStyleSheet("QToolButton { border: none; padding: 0; } "
                                          "QToolButton:hover { background: #2b3446; border-radius: 3px; }")
        self.display_button.clicked.connect(self.toggle)
        self.display_button.setContextMenuPolicy(Qt.CustomContextMenu)
        self.display_button.customContextMenuRequested.connect(lambda _p: self.menu())
        self.head_row.addWidget(self.display_button, 0, Qt.AlignTop | Qt.AlignRight)
        v.addLayout(self.head_row)
        # its line is there from the start, empty until a block is chosen, and always one line: appearing with the
        # first block added, or wrapping on a long name, it pushed every entry down under the mouse
        self.for_row = style.label("", "small")
        self.for_row.setFixedHeight(self.for_row.fontMetrics().height() + 2)
        v.addWidget(self.for_row)
        self.headers: list[QLabel] = []
        for title, kinds in GROUPS:
            h = QLabel(_(title).upper())
            h.setStyleSheet("color: #8b93a1; font-size: 8pt; font-weight: 600; margin-top: 8px;")
            v.addWidget(h)
            self.headers.append(h)
            for k in kinds:
                c = Card(self, k)
                self.cards[k] = c
                v.addWidget(c)
        v.addStretch(1)
        self.setWidget(body)
        self.setWidgetResizable(True)
        self._gen = 0
        self._queue: list[str] = []
        self._for_text = ""
        bridge.changed.connect(self.refresh)
        bridge.selection.connect(lambda *_a: self.refresh())
        # the cards are judged once the palette shows, one per turn of the event loop as after a change: a project
        # opens without waiting for them (``finish`` brings them all up to date at once)
        self.refresh()

    # ------------------------------------------------------------------ its ways of showing (graph.py)
    def set_state(self, st: str):
        """docked | auto | folded | hidden, as the window's layout says (``set_layout``)."""
        if st not in ("docked", "auto", "folded", "hidden"):
            st = "docked"
        self._unfold_timer.stop()
        self._fold_timer.stop()
        self._unfolded = False
        self.state = st
        self._fit()

    def wide(self) -> bool:
        """Its names shown: always open, or automatic while it is opened over the graph."""
        return self.state == "docked" or (self.state == "auto" and self._unfolded)

    def slot_width(self) -> int:
        """The room it takes in the page: the strip of icons when folded or automatic."""
        return PALETTE_FOLDED_W if self.state in ("folded", "auto") else PALETTE_W

    def _fit(self):
        wide = self.wide()
        auto = self.state == "auto"      # opened over the graph under the mouse: no card may move under it
        self.setFixedWidth(PALETTE_W if wide else PALETTE_FOLDED_W)
        top = 10 if wide or auto else 6
        self.body_box.setContentsMargins(*((10, top, 14, 10) if wide else (4, top, 4, 6)))
        for w in [self.head, self.for_row, *self.headers]:   # automatic: a hidden heading keeps its room
            sp = w.sizePolicy()
            if sp.retainSizeWhenHidden() != auto:
                sp.setRetainSizeWhenHidden(auto)
                w.setSizePolicy(sp)
        self.head.setVisible(wide)
        self.for_row.setVisible(wide)
        for h in self.headers:           # folded: only the blocks' icons (graph.py: the headers hidden)
            h.setVisible(wide)
        for c in self.cards.values():
            c.set_narrow(not wide)
            c.keep_height(auto)
        self._show_for()
        if self.on_fit is not None:
            self.on_fit()

    def unfold(self, on: bool):
        """Automatic: opened over the graph, or back to its icons (a menu open keeps it open)."""
        if not on and QApplication.activePopupWidget() is not None and self.state == "auto":
            self._fold_timer.start()
            return
        self._unfold_timer.stop()
        self._fold_timer.stop()
        on = on and self.state == "auto"
        if on != self._unfolded:
            self._unfolded = on
            self._fit()
            self.raise_()

    def enterEvent(self, e):                                          # noqa: N802
        if self.state == "auto":
            self._fold_timer.stop()
            if not self._unfolded:
                self._unfold_timer.start()
        super().enterEvent(e)

    def leaveEvent(self, e):                                          # noqa: N802
        self._unfold_timer.stop()
        if self._unfolded:
            self._fold_timer.start()
        super().leaveEvent(e)

    def hideEvent(self, e):                                           # noqa: N802
        self.unfold(False)               # another page, Tab
        super().hideEvent(e)

    def menu(self, run=True):
        """The "⋯" menu: Automatic, Always open, Always folded; the choice goes to the window (``chosen``), which keeps
        it for this size of window and gives it back by ``set_layout``. ``run`` False: the menu built, not shown
        (tests)."""
        m = QMenu(self)
        for st, word in SHOWN:
            a = m.addAction(_(word))
            a.setCheckable(True)
            a.setChecked(st == self.state)
            a.setData(st)
        if not run:
            return m
        b = self.display_button
        a = m.exec(b.mapToGlobal(b.rect().bottomLeft()))
        st = a.data() if a is not None else None
        m.deleteLater()
        self.choose(st)
        return None

    def choose(self, st):
        if st and st != self.state:
            self.chosen.emit(st)

    def toggle(self):
        """A click on the display button: open (its names shown) -> Always folded; the icons only -> Always open."""
        self.choose("folded" if self.wide() else "docked")

    # ------------------------------------------------------------------ the cards
    def current_row(self):
        """The row the palette works for: the selected block's row (or the selected card), else None (first row)."""
        bridge = self.bridge
        sel = bridge.selected
        if sel is None:
            return None
        return bridge.row(sel) or bridge.row_of(sel)

    def _show_for(self):
        """For “name”: one line, cut when too long (whole in its tooltip): the blocks under it never move."""
        text = self._for_text
        self.for_row.setToolTip(text)
        self.for_row.setText(self.for_row.fontMetrics().elidedText(text, Qt.ElideRight, PALETTE_W - 44))

    def refresh(self):
        """The cards' greyed states: the core needs a few tens of ms a card, so they are brought up to date one card
        per turn of the event loop, after the drop's repaint (a drop must feel instant), and again from the start if
        a change comes."""
        self._gen += 1
        if not self.isVisible():         # hidden (the Table, Tab): brought up to date when it shows again
            self._queue, self._hidden_change = [], True
            return
        self._hidden_change = False
        self._for_current()
        self._queue = list(self.cards)
        QTimer.singleShot(0, self, lambda g=self._gen: self._step(g))     # (cancelled if the palette is deleted)

    def _for_current(self):
        r = self.current_row()
        self._for_text = _("for “{name}”").format(name=shown(r.name)) if r is not None else ""
        self._show_for()
        self.for_row.setVisible(self.wide())

    def showEvent(self, e):                                           # noqa: N802
        super().showEvent(e)
        if getattr(self, "_hidden_change", False):
            self.refresh()

    def _step(self, gen):
        if gen != self._gen or not self._queue:
            return
        k = self._queue.pop(0)
        ok, why = self.bridge.doc.palette_state(k)
        self.cards[k].set_state(ok, "" if ok else why, due=ok)
        if self._queue:
            QTimer.singleShot(0, self, lambda g=gen: self._step(g))

    def finish(self):
        """Bring every card up to date now (tests, first display)."""
        if getattr(self, "_hidden_change", False):
            self._hidden_change = False
            self._gen += 1
            self._for_current()
            self._queue = list(self.cards)
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
        elif kind == "combine":
            bridge.add_combination(row=r.id if r is not None else None)
        else:
            bridge.act("add_block", kind=kind, row=r.id if r is not None else None)

    def combination_types(self, card, run=True):
        """The Combination card's flyout (graph.py GroupFlyout): the application's combination recipes (S-H-O, HOO,
        HaRGB...: graphcore.combos, in the folder's order), the one the card then places (dragged or double-clicked);
        its name on the card, its whole name and what it does in the tooltip. -> the menu (``run`` False: not shown,
        tests)."""
        from PySide6.QtWidgets import QMenu

        from ...graphcore import combos
        from ..graphnode import combine_title
        m = QMenu(self)
        m.setToolTipsVisible(True)
        for t in combos.offered():
            a = m.addAction(combine_title(t))
            a.setCheckable(True)
            a.setChecked(t == self.bridge.combo_type)
            a.setToolTip(recipe_tip(combos.recipe(t)))
            a.triggered.connect(lambda _c=False, t=t: self.set_combination_type(t))
        if run:
            m.exec(card.mapToGlobal(card.rect().bottomLeft()))
        return m

    def set_combination_type(self, t):
        from ..graphnode import combine_title
        self.bridge.combo_type = t
        card = self.cards.get("combine")
        if card is not None:
            card.title.setText(combine_title(t))
            card._tip()

    def contains(self, global_pos) -> bool:
        return self.isVisible() and self.rect().contains(self.mapFromGlobal(global_pos))
