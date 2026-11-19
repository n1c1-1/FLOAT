"""The widgets over the graph page's canvas: the toast (a message, its links), the minimap, the name being edited."""
from __future__ import annotations

import html

from PySide6.QtCore import QEvent, QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QLabel, QLineEdit, QWidget

from ..i18n import _
from .style import ACCENT

#: a toast with a link (Undo) stays at least this long (ms), and while the mouse is over it
LINK_MS = 10000


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
        self.timer = QTimer(self, singleShot=True, interval=4500)       # a toast with links: LINK_MS, and under the mouse
        self.timer.timeout.connect(self._end)
        self._fn, self._links, self._keep = None, None, None
        self.plain = ""
        self.linkActivated.connect(self.link.emit)
        self.hide()

    def enterEvent(self, e):                                         # noqa: N802 - Qt's name
        if self._links and self.isVisible():
            self.timer.stop()                            # the pointer is on it: it stays (a snackbar's way)
        super().enterEvent(e)

    def leaveEvent(self, e):                                         # noqa: N802
        if self._links and self.isVisible():
            self.timer.start(2500)                       # and goes a moment after the pointer leaves
        super().leaveEvent(e)

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
        if (self._keep is not None and self._keep()) or (self._links and self.underMouse()):
            self.timer.start(1000 if self._links and self.underMouse() else self.timer.interval())
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
        self.timer.start(LINK_MS if rich and self._links else 4500)   # a link stays long enough to be reached


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
        if e.key() in (Qt.Key_Return, Qt.Key_Enter):   # confirmed here: the field leaves Enter to its parent, the
            self.finish()                              # graph, where it opens the chosen block in Inspect
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
