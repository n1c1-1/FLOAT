"""The window's question before an Undo or a Redo that would compute again results computed and up to date: the same
card as a change of setting on a computed project (panel.choice_box), over the foot of the window: what the step is,
what it computes again and the time, Undo (or Redo, Enter) and Cancel (Esc), and "Do not ask" (the preference
``confirm_changes`` that turns every such question off)."""
from __future__ import annotations

import html

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout

from . import prefs
from .i18n import _
from .widgets import WARN, label


class HistoryCard(QFrame):
    """``how`` "undo" or "redo"; ``what``: the step's words; ``cost``: a ChangeCost; ``accept`` / ``cancel``: what the
    answers do (each closes the card first); ``said``: the window's status line (for "Do not ask")."""

    def __init__(self, win, how: str, what: str, cost, accept, cancel=None, said=None):
        super().__init__(win)
        self.setObjectName("historycard")
        self.setStyleSheet(f"QFrame#historycard {{ background: #2a2316; border: 1px solid {WARN}; border-radius: 6px; }}"
                           " QLabel { background: transparent; }")
        self._accept, self._cancel, self._said = accept, cancel, said
        v = QVBoxLayout(self)
        v.setContentsMargins(10, 8, 10, 8)
        v.setSpacing(6)
        head = _("Undo: {what}") if how == "undo" else _("Redo: {what}")
        v.addWidget(label("<b>" + html.escape(head.format(what=what or _("a change"))) + "</b>", wrap=True))
        c = label(html.escape(cost.text()), "small", wrap=True)
        c.setObjectName("history_cost")
        v.addWidget(c)
        row = QHBoxLayout()
        row.setSpacing(6)
        self.ok = QPushButton(_("Undo") if how == "undo" else _("Redo"))
        self.ok.setObjectName("primary")
        self.ok.setToolTip(_("Their results are computed again at the next run (Enter)"))
        self.ok.clicked.connect(self.accept)
        self.no = QPushButton(_("Cancel"))
        self.no.setObjectName("history_cancel")
        self.no.setToolTip(_("Nothing changes (Esc)"))
        self.no.clicked.connect(self.cancel)
        row.addWidget(self.ok)
        row.addWidget(self.no)
        row.addStretch(1)
        v.addLayout(row)
        never = QLabel(f'<a href="never"><span style="color:#8b93a1">{html.escape(_("Do not ask"))}</span></a>')
        never.setObjectName("history_never")
        never.setToolTip(_("Changes apply at once from now on; Edit › Settings › General turns the question on again"))
        never.linkActivated.connect(lambda _k: self._never_ask())
        v.addWidget(never, 0, Qt.AlignLeft)
        for keys, fn in (("Return", self.accept), ("Enter", self.accept), ("Escape", self.cancel)):
            sc = QShortcut(QKeySequence(keys), self)
            sc.setContext(Qt.WidgetWithChildrenShortcut)
            sc.activated.connect(fn)
        self.place()
        self.show()
        self.raise_()
        self.ok.setFocus()

    def place(self):
        win = self.parentWidget()
        w = max(320, min(620, win.width() - 32))
        self.setFixedWidth(w)
        h = self.sizeHint().height()
        sb = win.statusBar().height() if hasattr(win, "statusBar") else 0
        self.setGeometry((win.width() - w) // 2, win.height() - sb - h - 10, w, h)

    def _close(self):
        self.hide()
        self.deleteLater()

    def accept(self):
        fn, self._accept = self._accept, None
        self._close()
        if fn is not None:
            fn()

    def cancel(self):
        fn, self._accept = self._cancel, None
        self._close()
        if fn is not None:
            fn()

    def _never_ask(self):
        """"Do not ask": the preference is turned off (the step waiting still waits for its answer)."""
        prefs.set("confirm_changes", False)
        if self._said is not None:
            self._said(_("Changes apply at once from now on · Edit › Settings › General turns the question on again"))
