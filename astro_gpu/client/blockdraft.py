"""A block's settings changed and not applied yet: one way for every block, the Gradient view's.

On a block with computed results a change of setting is a DRAFT of that block (``Draft``): the panel shows it at once
(the radio checked, the slider moved), nothing computed is outdated, the graph and its states are unchanged. While the
draft differs from the block's applied settings, ONE bar (``DraftBar``) is pinned at the foot of the panel (in the
Gradient view, where its draft buttons are): what changed (old → new), what Apply would compute again and about how
long, and Apply, Try variant, Reset. Leaving the block while it holds a draft (another block, another page, Home, Close,
Run, Save) asks the question every way out of a project asks (``ask_leave``, the form of ``Main.ask_save``): Apply, Try
variant, Discard or Stay; a draft never survives leaving its block.

The draft's edits are steps of the window's one Undo order (journal.py) while it exists: Ctrl+Z takes the latest one
back; Apply is one step of the workflows' Undo. With the preference "Confirm changes" off, a change applies at once (no
draft, no bar), as before."""
from __future__ import annotations

import copy
import html
import os

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QFrame, QHBoxLayout, QMessageBox, QPushButton, QSizePolicy, QVBoxLayout

from . import journal
from .i18n import _, in_context
from .widgets import WARN, label

#: an unattended run (the test suite) answers the leave question beforehand: "apply", "variant", "discard" or "stay"
DRAFT_ANSWER_ENV = "FLOAT_DRAFT_ANSWER"
ANSWERS = ("apply", "variant", "discard", "stay")


def word(key: str) -> str:
    """The draft's words, the same everywhere: Apply, Try variant, Reset, Discard, Stay."""
    return in_context("draft", {"apply": "Apply", "variant": "Try variant", "reset": "Reset", "discard": "Discard",
                                "stay": "Stay"}[key])


def owned(what: str, owner: str, change: str = "") -> str:
    """A step of a setting, as every block of the combination family names it in Undo, Redo and the notices (Post's
    own way: 'Strength of "S-H-O · Combine layers"'): the setting, then its owner, then what it became when said:
    'S II hue of "S-H-O": 16° → 40°'."""
    if change:
        return _("{what} of “{name}”: {change}").format(what=what, name=owner, change=change)
    return _("{what} of “{name}”").format(what=what, name=owner)


class Draft:
    """The settings of one block changed and not applied: ``vals`` {name: value} (a setting of the Gradient view, or a
    change of the panel by its identity), each edit one step of its own Undo (``undo`` / ``redo``, the window's stamps in
    ``useq`` / ``rseq``). ``step`` and ``pids``: the block it belongs to (the panel's); ``cost``: what Apply computes
    again (a ChangeCost, the panel's). Empty, it is no draft (``bool``)."""

    def __init__(self, step: str | None = None, pids=()):
        self.step, self.pids = step, list(pids)
        self.vals: dict = {}
        self.undo: list[dict] = []
        self.redo: list[dict] = []
        self.useq: list[int] = []
        self.rseq: list[int] = []
        self.cost = None

    def __bool__(self) -> bool:
        return bool(self.vals)

    def __len__(self) -> int:
        return len(self.vals)

    def put(self, updates: dict, same=lambda _k, _v: False, step: bool = True) -> bool:
        """Settings of the draft: ``same(name, value)`` True when the value is the applied one (it leaves the draft);
        one step of its Undo (``step``). False when nothing changed."""
        before = copy.deepcopy(self.vals)
        for k, v in updates.items():
            if same(k, v):
                self.vals.pop(k, None)
            else:
                self.vals[k] = copy.deepcopy(v)
        if self.vals == before:
            return False
        if step:
            self.undo.append(before)
            self.redo.clear()
            self.useq.append(journal.tick())          # the window's one order of Undo (journal.py)
            self.rseq = []
        return True

    def move(self, src: str) -> set | None:
        """The last step of ``src`` ("undo" or "redo") taken: the names it changed, None when there is none."""
        dst = "redo" if src == "undo" else "undo"
        steps, other = getattr(self, src), getattr(self, dst)
        if not steps:
            return None
        names = set(self.vals) | set(steps[-1])
        other.append(copy.deepcopy(self.vals))
        self.vals = steps.pop()
        sq, dq = getattr(self, src[0] + "seq"), getattr(self, dst[0] + "seq")
        del sq[len(steps):]
        dq.append(journal.tick())
        del dq[:max(0, len(dq) - len(other))]
        return names

    def top(self, how: str = "undo") -> int | None:
        """The stamp of the step the window's Undo (``how`` "undo") or Redo would take next, None without one."""
        steps, seqs = getattr(self, how), getattr(self, how[0] + "seq")
        if not steps:
            return None
        return seqs[-1] if len(seqs) == len(steps) else 0

    def drop_pending(self, keep: set) -> None:
        """What the applied settings now hold leaves the draft (``keep``: the names still differing)."""
        for k in list(self.vals):
            if k not in keep:
                self.vals.pop(k)


class DraftBar(QFrame):
    """The draft's bar: what changed (old → new, one line each), what Apply computes again and how long, and Apply,
    Try variant, Reset. ``apply``, ``variant``, ``reset`` signals; ``show_draft(lines, cost)``."""

    apply = Signal()
    variant = Signal()
    reset = Signal()

    def __init__(self, parent=None, narrow: bool = False):
        super().__init__(parent)
        self.setObjectName("draft_bar")
        self.setStyleSheet(f"QFrame#draft_bar {{ background: #2a2316; border: 1px solid {WARN}; border-radius: 6px; }}"
                           " QLabel { background: transparent; }")
        v = QVBoxLayout(self)
        v.setContentsMargins(10, 8, 10, 8)
        v.setSpacing(4)
        self.what = label("", wrap=True)
        self.what.setObjectName("draft_what")
        self.what.setTextFormat(Qt.RichText)
        v.addWidget(self.what)
        self.cost = label("", "small", wrap=True)
        self.cost.setObjectName("draft_cost")
        v.addWidget(self.cost)
        row = QHBoxLayout()
        row.setSpacing(6)
        self.b_apply = QPushButton(word("apply"))
        self.b_apply.setObjectName("primary")
        self.b_apply.setAccessibleName("draft_apply")
        self.b_apply.setToolTip(_("The draft becomes the block's settings; what it changes is computed by the next run "
                                  "(Ctrl+Z takes it back)"))
        self.b_variant = QPushButton(word("variant"))
        self.b_variant.setAccessibleName("draft_variant")
        self.b_variant.setToolTip(_("The draft on a variant of the workflow: what is computed stays as it is, the "
                                    "variant shares everything up to this block"))
        self.b_reset = QPushButton(word("reset"))
        self.b_reset.setAccessibleName("draft_reset")
        self.b_reset.setToolTip(_("Back to the applied settings: the draft goes"))
        self.b_apply.clicked.connect(self.apply.emit)
        self.b_variant.clicked.connect(self.variant.emit)
        self.b_reset.clicked.connect(self.reset.emit)
        #: a narrow bar (an editor's column: Gradient's, a combination's) keeps one height whatever the draft holds: the
        #: state, the changes and the cost on one line each, cut, the whole text on hover; growing, it pushed the
        #: controls under it down between a press and a click (a reviewer's click on "On sky" landed on "On signal")
        self.fixed = narrow
        self._lines: list[str] = []
        self._cost = ""
        if narrow:                                      # a narrow column (the Gradient view's): Apply under the two
            for lb, n in ((self.what, 2), (self.cost, 1)):
                lb.setWordWrap(False)
                lb.setTextFormat(Qt.PlainText)
                lb.setFixedHeight(n * lb.fontMetrics().height() + 2)
                lb.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
            row.addWidget(self.b_reset)
            row.addWidget(self.b_variant)
            v.addLayout(row)
            v.addWidget(self.b_apply)
            return
        for b in (self.b_apply, self.b_variant, self.b_reset):
            row.addWidget(b)
        row.addStretch(1)
        v.addLayout(row)

    def show_draft(self, lines, cost: str = "") -> None:
        """The draft's changes, one line each (bold), and what Apply computes again. A narrow bar (``fixed``): the
        state on its first line, the changes on the second, the cost on the third, each cut to the bar's width, the
        whole on hover; its height never changes."""
        lines = [t for t in lines if t]
        if self.fixed:
            self._lines, self._cost = lines, cost
            self._elide()
            tip = "\n".join(lines + ([cost] if cost else []))
            self.what.setToolTip(tip)
            self.cost.setToolTip(tip)
            return
        self.what.setText("<br>".join(f"<b>{html.escape(t)}</b>" for t in lines))
        self.what.setVisible(any(lines))
        self.cost.setText(html.escape(cost))
        self.cost.setVisible(bool(cost))

    def _elide(self):
        w = max(40, self.what.width() if self.what.width() > 50 else self.width() - 20)
        fm = self.what.fontMetrics()
        first = self._lines[0] if self._lines else ""
        rest = " · ".join(self._lines[1:])
        self.what.setText("\n".join(fm.elidedText(t, Qt.ElideRight, w) for t in (first, rest)))
        self.cost.setText(self.cost.fontMetrics().elidedText(self._cost, Qt.ElideRight, w))

    def resizeEvent(self, e):                                         # noqa: N802 - Qt's name
        super().resizeEvent(e)
        if self.fixed:
            self._elide()


def ask_leave(parent, block: str, lines=(), cost: str = "") -> str:
    """The block left while it holds a draft: "apply", "variant", "discard" or "stay", as the user answers, in the form
    of the unsaved-changes question (``Main.ask_save``). Only an unattended run (the test suite) gives its answer
    beforehand, in ``DRAFT_ANSWER_ENV``."""
    preset = os.environ.get(DRAFT_ANSWER_ENV, "")
    if preset in ANSWERS:
        return preset
    box = QMessageBox(QMessageBox.Warning, _("Settings not applied"),
                      _("{block} · settings not applied").format(block=block), parent=parent)
    info = [t for t in lines if t] + ([cost] if cost else [])
    if info:
        box.setInformativeText("\n".join(info))
    b_apply = box.addButton(word("apply"), QMessageBox.AcceptRole)
    b_var = box.addButton(word("variant"), QMessageBox.AcceptRole)     # (beside Apply: the same order as the bar)
    b_drop = box.addButton(word("discard"), QMessageBox.DestructiveRole)
    b_stay = box.addButton(word("stay"), QMessageBox.RejectRole)
    b_apply.setToolTip(_("The draft becomes the block's settings"))
    b_var.setToolTip(_("The draft on a variant of the workflow; what is computed stays as it is"))
    b_drop.setToolTip(_("The draft goes: the block keeps its applied settings"))
    b_stay.setToolTip(_("Back to the block, its draft as it is"))
    box.setDefaultButton(b_apply)
    box.setEscapeButton(b_stay)
    box.exec()
    c = box.clickedButton()
    return "apply" if c is b_apply else "variant" if c is b_var else "discard" if c is b_drop else "stay"


def settle(holder, parent) -> bool:
    """The draft of ``holder`` (a panel or the Gradient view: ``draft_pending()`` -> (block, lines, cost) or None, and
    ``draft_apply()``, ``draft_variant()``, ``draft_reset()``) answered before its block is left: True when the way out
    may go on (no draft, or applied, made a variant, discarded), False to stay."""
    if holder is None:
        return True
    try:
        pending = holder.draft_pending()
    except RuntimeError:                                 # its widget is gone
        return True
    if not pending:
        return True
    block, lines, cost = pending
    answer = ask_leave(parent, block, lines, cost)
    if answer == "stay":
        dock = getattr(parent, "dock", None)             # a small window's drawer, closed by the click that left:
        if dock is not None and getattr(dock, "state", "") in ("icons", "auto"):
            dock.open_overlay()                          # open again on the block, its draft in sight
        return False
    if answer == "apply":
        return bool(holder.draft_apply())
    if answer == "variant":
        return bool(holder.draft_variant())
    holder.draft_reset()
    return True
