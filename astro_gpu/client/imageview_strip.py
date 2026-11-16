"""The viewer's sequence: the information band, the side scroll and the film strip."""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen
from PySide6.QtWidgets import (QFrame, QHBoxLayout, QLabel, QPushButton, QScrollArea, QScrollBar, QSizePolicy,
                               QVBoxLayout, QWidget)

from .i18n import _, language, number
from .icons import IconText, draw
from .widgets import ACCENT, BAD, OK, WARN, ElidedLabel, label
from .imageview_images import ViewImage
from .imageview_loading import to_qimage


# ---------------------------------------------------------------------------------------------- a sequence of frames
def time_text(t: str) -> str:
    """The time of day of an ISO date and time ("2026-08-16T23:41:05" -> "23:41:05")."""
    if not t:
        return ""
    return t[11:19] if len(t) >= 19 and t[10] in "T " else t


def exposure_text(t) -> str:
    """An exposure time: 1/1000 s under a second, else 15 s (a decimal comma in French); "" when unknown."""
    if t is None:
        return ""
    t = float(t)
    if 0 < t < 1 and round(1 / t) >= 2 and abs(round(1 / t) * t - 1) < 0.05:
        return f"1/{round(1 / t)} s"
    if t <= 0:
        return "0 s"
    txt = f"{t:g}"
    return (txt.replace(".", ",") if language() == "fr" else txt) + " s"


class SideScroll(QScrollArea):
    """A row that scrolls sideways when the window is too narrow for it, instead of making the window wider (the
    frame's facts and the viewer's tools are wider than a small screen). A scroll bar under the row only when it is
    needed."""

    def __init__(self, inner: QWidget, parent=None, min_width: int = 240):
        super().__init__(parent)
        self.setWidget(inner)
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.NoFrame)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setMinimumWidth(min_width)
        self.setStyleSheet("QScrollArea { background: transparent; border: none; }")
        self.viewport().setAutoFillBackground(False)
        inner.setAutoFillBackground(False)
        self._fit()

    def _fit(self):
        inner = self.widget()
        need = inner.minimumSizeHint().width() > self.viewport().width()
        h = inner.sizeHint().height() + (self.horizontalScrollBar().sizeHint().height() if need else 0)
        if self.height() != h or self.minimumHeight() != h:
            self.setFixedHeight(h)

    def resizeEvent(self, e):  # noqa: N802 - Qt's name
        super().resizeEvent(e)
        self._fit()

    def sizeHint(self):  # noqa: N802 - Qt's name
        return self.widget().sizeHint()                  # as wide as the row wants when there is room


class InfoBand(QFrame):
    """What the caller knows of the frame shown: name, time, exposure, ISO / gain, temperature, FWHM, roundness,
    transparency, kept or excluded, and Exclude / Include."""

    excludeClicked = Signal()

    CELLS = ("time", "exposure", "iso", "temperature", "fwhm", "roundness", "transparency", "weight")

    def __init__(self, parent=None):
        super().__init__(parent)
        h = QHBoxLayout(self)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(18)
        col = QVBoxLayout()
        col.setSpacing(0)
        self.pos_l = label("", "ivcap")
        col.addWidget(self.pos_l)
        self.name_l = ElidedLabel("", "ivname")
        self.name_l.setMinimumWidth(120)
        col.addWidget(self.name_l)
        h.addLayout(col, 2)
        self.cells: dict[str, tuple[QLabel, QLabel]] = {}
        caps = {"time": _("Time"), "exposure": _("Exposure"), "iso": _("ISO"), "temperature": _("Temperature"),
                "fwhm": "FWHM", "roundness": _("Roundness"), "transparency": _("Transparency"),
                "weight": _("Weight")}
        for k in self.CELLS:
            c = QVBoxLayout()
            c.setSpacing(0)
            cap = label(caps[k], "ivcap")
            val = label("–", "ivval")
            c.addWidget(cap)
            c.addWidget(val)
            h.addLayout(c)
            self.cells[k] = (cap, val)
        self.extra = QHBoxLayout()
        self.extra.setSpacing(18)
        h.addLayout(self.extra)
        h.addStretch(1)
        self.state = QLabel("")
        self.state.setAlignment(Qt.AlignCenter)
        h.addWidget(self.state)
        self.toggle = QPushButton(_("Exclude"))
        self.toggle.setObjectName("ivstate")
        self.toggle.setToolTip(_("Leave this frame out of the stack, or take it back (X)"))
        self.toggle.clicked.connect(self.excludeClicked.emit)
        h.addWidget(self.toggle)
        self.mode = "on"                          # on | off (no toggle, no state) | spec (◇: shown, disabled)
        self.spec_tip = ""
        # why the engine flags the frame shown, written out (not only in a tooltip): a line under the numbers
        self.why = IconText("warning", "", "warning")
        self.why.label.setTextFormat(Qt.PlainText)
        self.why.hide()

    def set_mode(self, mode: str, tip: str = ""):
        """What Exclude / Include does here: ``on``; ``off`` (masters: neither the toggle nor the state); ``spec``
        (◇ the engine cannot do it yet: the toggle disabled, ``tip`` says why)."""
        self.mode = mode if mode in ("on", "off", "spec") else "on"
        self.spec_tip = tip
        self.toggle.setVisible(self.mode != "off")
        self.state.setVisible(self.mode != "off")

    def set_frame(self, vi: ViewImage, index: int, total: int):
        info = vi.info or {}
        self.pos_l.setText(_("Frame {i} of {n}").format(i=index + 1, n=total))
        self.name_l.setText(vi.name)
        vals = {
            "time": time_text(info.get("time") or ""),
            "exposure": exposure_text(info.get("exposure")),
            "iso": (number(info["iso"], 0) if info.get("iso") is not None else
                    number(info["gain"], 0) if info.get("gain") is not None else ""),
            "temperature": number(info["temperature"], 1) + " °C" if info.get("temperature") is not None else "",
            "fwhm": (number(info["fwhm"], 2) + " px" + (f"  ·  {number(info['fwhm_arcsec'], 1)}″"
                                                        if info.get("fwhm_arcsec") is not None else ""))
            if info.get("fwhm") is not None else "",
            "roundness": number(info["roundness"], 2) if info.get("roundness") is not None else "",
            "transparency": number(100 * info["transparency"], 0) + " %" if info.get("transparency") is not None
            else "",
            "weight": number(info["weight"], 2) if info.get("weight") is not None else "",
        }
        self.cells["iso"][0].setText(_("Gain") if info.get("is_gain") or (info.get("iso") is None
                                                                          and info.get("gain") is not None) else _("ISO"))
        compact = bool(info.get("compact"))      # only the cells with a value (a master, a calibration frame)
        for k, (cap, val) in self.cells.items():
            val.setText(vals[k] or "–")
            val.setStyleSheet("" if vals[k] else "color: #5d6470;")
            show = bool(vals[k]) or (k != "weight" and not compact)
            cap.setVisible(show)
            val.setVisible(show)
        while self.extra.count():
            it = self.extra.takeAt(0)
            lay = it.layout()
            if lay is not None:
                while lay.count():
                    w = lay.takeAt(0).widget()
                    if w is not None:
                        w.hide()                  # not drawn again under the next frame's numbers before it goes
                        w.deleteLater()
        for cap_t, text in info.get("extra") or []:
            c = QVBoxLayout()
            c.setSpacing(0)
            c.addWidget(label(cap_t, "ivcap"))
            c.addWidget(label(text, "ivval"))
            self.extra.addLayout(c)
        excluded = bool(info.get("excluded"))
        reason = info.get("reason") or ""
        if excluded or reason:
            self.state.setText(_("Excluded") if excluded else _("Left out"))
            self.state.setToolTip(reason)
            self.state.setStyleSheet(f"color: {BAD}; border: 1px solid {BAD}; border-radius: 9px; padding: 2px 10px;")
        else:
            self.state.setText(_("Kept"))
            self.state.setToolTip("")
            self.state.setStyleSheet(f"color: {OK}; border: 1px solid {OK}; border-radius: 9px; padding: 2px 10px;")
        flag = info.get("flag")                  # a quality flag of the engine: red (bad) or amber, the reason below
        col = BAD if info.get("flag_level") == "bad" else WARN
        if flag and not (excluded or reason):
            self.state.setText(str(flag))
            self.state.setToolTip(info.get("flag_reason") or "")
            self.state.setStyleSheet(f"color: {col}; border: 1px solid {col}; border-radius: 9px; "
                                     "padding: 2px 10px;")
        why = str(info.get("flag_reason") or "") if flag else ""
        self.why.setText(why)
        self.why.icon.set_role("warning", "error" if info.get("flag_level") == "bad" else "warning")
        self.why.setStyleSheet(f"color: {col}; font-weight: 600;")
        self.why.setVisible(bool(why))
        self.toggle.setText(_("Include") if excluded else _("Exclude"))
        if self.mode == "spec":
            self.toggle.setText(self.toggle.text() + "  ◇")
            self.toggle.setEnabled(False)
            self.toggle.setToolTip("◇ " + (self.spec_tip or _("Not in the engine yet.")))
        else:
            self.toggle.setEnabled(True)
            self.toggle.setToolTip(_("Leave this frame out of the stack, or take it back (X)"))


class FilmStrip(QWidget):
    """The frames as thumbnails, the one shown framed, the excluded ones marked; click to show one."""

    picked = Signal(int)
    visibleChanged = Signal(int, int)

    CELL_W, CELL_H, GAP = 104, 70, 6

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(self.CELL_H + 22)
        self.setMouseTracking(True)
        self.n = 0
        self.current = 0
        self.thumbs: dict[int, QImage] = {}
        self.excluded: set[int] = set()
        self.flagged: dict[int, str] = {}           # frame -> the engine's flag: "bad" (red) or "warn" (amber)
        self.offset = 0.0
        self.bar = QScrollBar(Qt.Horizontal, self)
        self.bar.valueChanged.connect(self._scrolled)
        self._hover = -1

    def set_count(self, n: int):
        self.n = n
        self.thumbs.clear()
        self.excluded.clear()
        self.flagged.clear()
        self._layout_bar()
        self.update()

    def set_thumb(self, i: int, arr: np.ndarray):
        if 0 <= i < self.n:
            a = np.ascontiguousarray(arr)
            self.thumbs[i] = to_qimage(a).copy()
            self.update()

    def set_excluded(self, i: int, on: bool):
        (self.excluded.add if on else self.excluded.discard)(i)
        self.update()

    def set_flagged(self, i: int, level: str | None):
        """Mark a frame the engine flags: "bad" (a red frame and corner), "warn" (amber), None (no mark)."""
        if level in ("bad", "warn"):
            self.flagged[i] = level
        else:
            self.flagged.pop(i, None)
        self.update()

    def set_current(self, i: int):
        self.current = i
        x = i * (self.CELL_W + self.GAP)
        if x < self.offset or x + self.CELL_W > self.offset + self.width():
            self.bar.setValue(int(max(0, x - (self.width() - self.CELL_W) / 2)))
        self.update()

    def visible_range(self) -> tuple[int, int]:
        step = self.CELL_W + self.GAP
        i0 = int(self.offset // step)
        return max(0, i0), min(self.n, i0 + int(self.width() // step) + 2)

    def _layout_bar(self):
        total = self.n * (self.CELL_W + self.GAP)
        self.bar.setGeometry(0, self.height() - 10, self.width(), 10)
        self.bar.setRange(0, max(0, int(total - self.width())))
        self.bar.setPageStep(max(1, self.width()))
        self.bar.setSingleStep(self.CELL_W + self.GAP)
        self.bar.setVisible(total > self.width())

    def _scrolled(self, v):
        self.offset = float(v)
        self.visibleChanged.emit(*self.visible_range())
        self.update()

    def resizeEvent(self, e):  # noqa: N802
        self._layout_bar()
        self.visibleChanged.emit(*self.visible_range())
        super().resizeEvent(e)

    def wheelEvent(self, e):  # noqa: N802
        d = e.angleDelta().y() or e.angleDelta().x()
        self.bar.setValue(self.bar.value() - int(d / 120 * (self.CELL_W + self.GAP)))

    def _at(self, x: float) -> int:
        step = self.CELL_W + self.GAP
        i = int((x + self.offset) // step)
        return i if 0 <= i < self.n and (x + self.offset) % step <= self.CELL_W else -1

    def mousePressEvent(self, e):  # noqa: N802
        i = self._at(e.position().x())
        if i >= 0 and e.position().y() < self.CELL_H + 8:
            self.picked.emit(i)

    def mouseMoveEvent(self, e):  # noqa: N802
        i = self._at(e.position().x())
        if i != self._hover:
            self._hover = i
            self.update()

    def leaveEvent(self, e):  # noqa: N802
        self._hover = -1
        self.update()

    def paintEvent(self, e):  # noqa: N802
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#16181e"))
        f = QFont(self.font())
        f.setPointSizeF(7.5)
        p.setFont(f)
        step = self.CELL_W + self.GAP
        i0, i1 = self.visible_range()
        for i in range(i0, i1):
            x = i * step - self.offset + 3
            cell = QRectF(x, 4, self.CELL_W, self.CELL_H)
            p.fillRect(cell, QColor("#0f1115"))
            qi = self.thumbs.get(i)
            if qi is not None:
                fw, fh = qi.width(), qi.height()
                k = min(cell.width() / fw, cell.height() / fh)
                tr = QRectF(cell.x() + (cell.width() - fw * k) / 2, cell.y() + (cell.height() - fh * k) / 2,
                            fw * k, fh * k)
                p.setRenderHint(QPainter.SmoothPixmapTransform)
                p.drawImage(tr, qi)
            ex = i in self.excluded
            fl = self.flagged.get(i)
            if fl is not None:                    # flagged by the engine: a coloured frame and a corner mark
                fc = QColor(BAD if fl == "bad" else WARN)
                p.setPen(QPen(fc, 2))
                p.setBrush(Qt.NoBrush)
                p.drawRect(cell.adjusted(1, 1, -1, -1))
                p.setPen(Qt.NoPen)
                p.setBrush(QColor(15, 17, 21, 200))
                p.drawRoundedRect(QRectF(cell.x() + 2, cell.y() + 2, 20, 20), 3, 3)
                draw(p, QRectF(cell.x() + 4, cell.y() + 4, 16, 16), "warning", "error" if fl == "bad" else "warning")
            if ex:
                p.fillRect(cell, QColor(15, 17, 21, 150))
                draw(p, QRectF(cell.right() - 18, cell.y() + 3, 16, 16), "remove", "error")
            if i == self.current:
                p.setPen(QPen(QColor(ACCENT), 2))
                p.setBrush(Qt.NoBrush)
                p.drawRect(cell.adjusted(-1, -1, 1, 1))
            elif i == self._hover:
                p.setPen(QPen(QColor("#8b93a1"), 1))
                p.setBrush(Qt.NoBrush)
                p.drawRect(cell)
            tag_r = QRectF(cell.x() + 3, cell.bottom() - 16, 30, 13)
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(0, 0, 0, 160))
            p.drawRoundedRect(tag_r, 3, 3)
            p.setPen(QColor(BAD) if ex else QColor("#d8dce4"))
            p.drawText(tag_r, Qt.AlignCenter, str(i + 1))
        p.end()
