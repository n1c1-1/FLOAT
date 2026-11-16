"""The image viewer, the same everywhere (Frames, Calibration, Stacking, Denoise, Gradient, Output).

One widget, ``ImageView``, shows an image of any size (up to 6384 x 4272 RGB float, mono, or a raw Bayer mosaic) with:

* zoom (wheel, Fit, 1:1, presets), pan, a minimap, and a pixel readout (x, y, the values in ADU or 0-1, RA / Dec
  when the image has an astrometric solution);
* a display stretch as in the reference program: automatic (channels linked or not), linear, midtones (MTF), asinh or histogram
  equalisation, black / mid / white handles on a live histogram, per channel or linked, "Neutral sky" for an image
  still in the camera's colours, Reset. The stretch is display only: the data are never changed;
* compare with the previous steps of the workflow: the caller gives named images, the view a selector; split slider
  (before / after) or side by side, with the same stretch on both if wanted. The compared image is drawn on the grid
  of the one shown, so that a star sits at the same place in both (``alignment``: by the placements the caller gives,
  ``ViewImage.place`` - drizzle scale, crop, half size -, else by the astrometric solutions), resampled for display
  only, and a line above the images says how;
* a sequence of frames: previous / next (arrow keys), a frame number, a film strip, an info band (the caller's
  numbers: time, exposure, ISO / gain, temperature, FWHM, roundness, transparency, kept or excluded) and Exclude /
  Include; frames load in worker threads, the neighbours ahead of time;
* Export… (FITS 32-bit float with the WCS and a header; TIFF 16-bit, PNG, JPEG: the data or the stretched view),
  Send to Post (a signal) and Copy image (the picture as shown, to paste elsewhere).

How it stays fast on the CPU: an image is prepared once in a worker thread (``prepare``: a pyramid of halves and a
histogram of a sample, about 0.15 s for the 6384 x 4272 stack). The view draws from the pyramid level nearest to the
screen's resolution; full resolution is touched only for the tiles in view at about 1:1 and above. Each 512 x 512
tile is quantized once to 16 bits and shown through the look-up tables of ``stretch.py``: moving a handle rebuilds
three tables and re-indexes the tiles in view, in parallel threads. The visible part is resampled to the screen's
pixels in one OpenCV call.

The views call (see ``ImageView`` for the whole list)::

    v = ImageView(post=True)                                  # Send to Post shown
    v.set_image(ViewImage(arr, "Stack", key=("stack", k), wcs=wcs, camera=True))
    v.set_compare_targets([ViewImage(arr0, "Calibrated"), ...])        # the previous steps
    v.set_sequence([ViewImage(path, name, info={...}, thumb=fn), ...], index=0)
    v.excludeToggled.connect(...); v.sendToPost.connect(...)
    SyncGroup([v1, v2], stretch=True)                         # two viewers share zoom, pan (and the stretch)

Classes: ``ImageView`` (the widget) draws on an ``ImageCanvas``, with a ``StretchPanel`` (its ``HistogramView``), an
``InfoBand`` and a ``FilmStrip``; a ``Loader`` turns each ``ViewImage`` into a ``Prepared`` image in worker threads and a
``Display`` shows it through a stretch; ``ImageExportDialog`` is Export…; ``SyncGroup`` links viewers. Post has its own
viewer (``viewer.Viewer``).
"""
from __future__ import annotations

import heapq
import itertools
import math
import os
import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PySide6.QtCore import QEvent, QObject, QPointF, QRectF, QStandardPaths, Qt, QTimer, Signal
from PySide6.QtGui import (
    QColor,
    QFont,
    QFontMetricsF,
    QGuiApplication,
    QImage,
    QKeySequence,
    QPainter,
    QPainterPath,
    QPen,
    QPolygonF,
    QShortcut,
)
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QScrollBar,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from . import APP_NAME
from . import stretch as sm
from .i18n import _, language, log_detail, number, plain_error
from .icons import IconLabel, IconText, draw, icon_only, menu_button, set_icon
from .layout import pinch_factor, wheel_gesture
from .widgets import ACCENT, BAD, OK, WARN, ElidedLabel, cap_dialog, label

# The viewer's parts live in the modules below by subject; this module keeps the view (ImageView), the default
# loader, the synchronised views and the export, and gives every name it had, so its importers read them from here.
# ruff: noqa: F401
from .imageview_images import (TILE, CANVAS_BG, CANVAS_RGB, CHANNEL_COLORS, MONO_COLOR, ZOOM_STEP, ZOOM_PRESETS,
                               MONO_FONT, QSS, ViewImage, Placement, Alignment, _identity, SKY_GRID, _sky_map,
                               alignment, RAW_SUFFIXES, read_file, _wcs_of, _UID, _Lru, QTILES, RTILES, Prepared,
                               planar, _read, _to_side, quick_thumbnail, prepare, _place_of, thumbnail_of,
                               default_stretch, kind_of)
from .imageview_loading import Unavailable, Loader, Display, to_qimage, TAG_BG, tag
from .imageview_canvas import ImageCanvas, _state_of, _mirror_state
from .imageview_stretch import nice_ticks, short_number, spaced_ticks, HistogramView, value_text, StretchPanel


_default_loader: Loader | None = None


def memory_budget() -> int:
    """The prepared images kept in memory: a fifth of the machine's memory, from 3 to 24 GB (a drizzled stack takes
    several hundred MB once prepared; with too small a budget a tab shown again is read from the disk again)."""
    try:
        import psutil
        total = int(psutil.virtual_memory().total)
    except Exception:  # noqa: BLE001 - unknown machine: the smallest budget
        return 3 * 2 ** 30
    return int(min(max(total // 5, 3 * 2 ** 30), 24 * 2 ** 30))


def default_loader() -> Loader:
    global _default_loader
    if _default_loader is None:
        _default_loader = Loader(budget=memory_budget(), threads=max(2, min(4, (os.cpu_count() or 2) // 4)),
                                 parent=QApplication.instance())
    return _default_loader


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


# ---------------------------------------------------------------------------------------------- the viewer
def _sep() -> QFrame:
    f = QFrame()
    f.setObjectName("ivsep")
    f.setFixedSize(1, 20)
    return f


class ImageView(QWidget):
    """The common image viewer. See the module's docstring; the API the views call:

    Images: ``set_image(ViewImage | array, keep_view=False, keep_stretch=False)``, ``image()``, ``prepared()``,
    ``clear()``. Compare: ``set_compare_targets([ViewImage, ...], current=None)``, ``set_compare(index | None)``,
    ``set_compare_mode("split" | "side")``, ``set_same_stretch(bool)``, ``compare_alignment()`` and
    ``alignment_note()`` (how the compared image was put on this one's grid). Sequence: ``set_sequence([ViewImage, ...],
    index=0)``, ``set_index(i)``, ``index()``, ``next_frame()``, ``previous_frame()``, ``set_frame_info(i, info)``,
    ``set_excluded(i, bool)``, ``set_lock_stretch(bool)``. Stretch: ``stretch()``, ``set_stretch(Stretch)``,
    ``auto_stretch()``, ``show_histogram(bool)``. View: ``fit()``, ``zoom_to(scale, cx, cy)``,
    ``set_zoom_percent(p)``, ``view_state()``, ``set_view_state(state)``. Actions: ``export_dialog()``,
    ``export(path, fmt, content, area)``, ``copy_view()``. Readout: ``pixel_text(x, y)``. Overlays: ``add_overlay(fn)``
    (fn(painter, canvas): ``canvas.img_to_w(x, y)``, ``canvas.scale``). Tests: ``wait(timeout)``.

    Signals: ``imageShown(key)``, ``viewChanged()``, ``stretchChanged(Stretch)``, ``hovered(x, y)``,
    ``frameChanged(i)``, ``excludeToggled(i, excluded)``, ``compareChanged(key | None)``, ``sendToPost(key)``,
    ``toasted(text)``, ``rectDone(x0, y0, x1, y1)``.
    """

    imageShown = Signal(object)
    viewChanged = Signal()
    stretchChanged = Signal(object)
    hovered = Signal(float, float)
    frameChanged = Signal(int)
    excludeToggled = Signal(int, bool)
    compareChanged = Signal(object)
    sendToPost = Signal(object)
    toasted = Signal(str)
    rectDone = Signal(float, float, float, float)

    def __init__(self, parent=None, *, loader: Loader | None = None, histogram: bool = True, post: bool = False,
                 export: bool = True, copy: bool = True, compare: bool = True):
        super().__init__(parent)
        self.setStyleSheet(QSS)
        self.loader = loader or default_loader()
        self.export_dir: str = ""                 # where Export… proposes to write: the last folder written to,
        self.default_dir = None                   # else this (a callable: the project's folder), else Pictures
        self._vi: ViewImage | None = None
        self._prep: Prepared | None = None
        self._main: Display | None = None
        self._st: sm.Stretch | None = None
        self._want = None                         # the key of the image asked for
        self._failed_key = None                   # the key of the last image that could not be read
        self._keep = (False, False)               # keep_view, keep_stretch of the image asked for
        self._targets: list[ViewImage] = []
        self._cmp_i: int | None = None
        self._cmp: Display | None = None
        self._cmp_mode = "split"
        self._same = False
        self._frames: list[ViewImage] = []
        self._index = 0
        self._lock = False                        # "Same stretch for all frames": off, each frame its own stretch
        self._locked: dict[str, sm.Stretch] = {}  # with it on: kind -> the stretch its frames are shown with
        self._st_kind: str | None = None          # the kind of the image the stretch was made for
        self._units = "adu"
        self._syncing = False
        self._shown_at: float | None = None
        self.times: dict[str, float] = {}         # measurements: open (s)
        self.loader.ready.connect(self._ready)
        self.loader.failed.connect(self._failed)
        self.loader.thumbReady.connect(self._thumb)
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        # the tools in a row that scrolls sideways in a narrow window (they would widen it otherwise)
        v.addWidget(SideScroll(self._toolbar(post, export, copy, compare)))
        # the tools of an editor on the image (Inspect's Gradient: its draft, samples, excluded areas), above the image
        self.tool_row = QWidget()
        trv = QVBoxLayout(self.tool_row)
        trv.setContentsMargins(0, 0, 0, 0)
        trv.setSpacing(0)
        self.tool_row.hide()
        v.addWidget(self.tool_row)
        self.align_bar = QFrame()                  # one line: how the compared image was put on this one's grid
        self.align_bar.setObjectName("ivalign")
        ab = QHBoxLayout(self.align_bar)
        ab.setContentsMargins(12, 2, 12, 3)
        self.align_l = ElidedLabel("", "ivalign", Qt.ElideRight)
        ab.addWidget(IconLabel("link", 16, "dim"))
        ab.addWidget(self.align_l, 1)
        self.align_bar.hide()
        v.addWidget(self.align_bar)
        self._al: Alignment | None = None
        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        self.area = QWidget()
        ah = QHBoxLayout(self.area)
        ah.setContentsMargins(0, 0, 0, 0)
        ah.setSpacing(2)
        self.canvas2 = ImageCanvas()               # the compared image, side by side
        self.canvas2.hide()
        self.canvas = ImageCanvas()
        ah.addWidget(self.canvas2, 1)
        ah.addWidget(self.canvas, 1)
        body.addWidget(self.area, 1)
        self.panel = StretchPanel()
        self.panel.changed.connect(self._panel_changed)
        # in a column that scrolls when the viewer is short (squeezed, its rows would run into each other)
        self.panel_box = QScrollArea()
        self.panel_box.setWidget(self.panel)
        self.panel_box.setWidgetResizable(True)
        self.panel_box.setFrameShape(QFrame.NoFrame)
        self.panel_box.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.panel_box.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.panel_box.setFixedWidth(self.panel.width() + self.panel_box.verticalScrollBar().sizeHint().width())
        self.panel_box.setStyleSheet("QScrollArea { background: transparent; border: none; }")
        self.panel_box.viewport().setAutoFillBackground(False)
        self.panel.setVisible(histogram)
        self.panel_box.setVisible(histogram)
        body.addWidget(self.panel_box)
        v.addLayout(body, 1)
        v.addWidget(self._readout_bar())
        v.addWidget(self._sequence_bar())
        self.hist_b.setChecked(histogram)
        for c in (self.canvas, self.canvas2):
            c.hovered.connect(lambda x, y, which, cv=c: self._hover(cv, x, y, which))
            c.left.connect(self._left)
            c.viewChanged.connect(lambda cv=c: self._canvas_moved(cv))
            c.zoomed.connect(self._zoom_label)
        self.canvas.rectDone.connect(self.rectDone.emit)
        self._shortcuts()
        self._blink = QTimer(self)
        self._blink.setInterval(200)
        self._blink.timeout.connect(self._blink_tick)
        self._update_actions()

    # ------------------------------------------------------------ building
    def _toolbar(self, post, export, copy, compare) -> QFrame:
        bar = QFrame()
        bar.setObjectName("ivbar")
        h = QHBoxLayout(bar)
        h.setContentsMargins(10, 6, 10, 6)
        h.setSpacing(6)

        def tool(text, tip, slot, checkable=False):
            b = QPushButton(text)
            b.setObjectName("toggle" if checkable else "ivtool")
            b.setCheckable(checkable)
            b.setToolTip(tip)
            (b.toggled if checkable else b.clicked).connect(slot)
            h.addWidget(b)
            return b
        self.fit_b = icon_only(tool("", "", self.fit), "fit", _("Fit: the whole image (F)"), 16)
        self.one_b = icon_only(tool("", "", lambda: self.set_zoom_percent(100)), "actual_size",
                               _("Actual size: one image pixel per screen pixel (1)"), 16)
        self.zoom_b = QPushButton("100 %")
        self.zoom_b.setObjectName("ivtool")
        self.zoom_b.setToolTip(_("Zoom (wheel, + and −)"))
        self.zoom_b.setMinimumWidth(66)
        menu = QMenu(self.zoom_b)
        menu.addAction(_("Fit"), self.fit)
        for z in ZOOM_PRESETS:
            menu.addAction(f"{number(z * 100, 1 if z * 100 % 1 else 0)} %",
                           lambda zz=z: self.set_zoom_percent(zz * 100))
        self.zoom_b.setMenu(menu)
        h.addWidget(self.zoom_b)
        h.addSpacing(6)
        h.addWidget(_sep())
        h.addSpacing(6)
        self.cmp_box = QWidget()
        ch = QHBoxLayout(self.cmp_box)
        ch.setContentsMargins(0, 0, 0, 0)
        ch.setSpacing(6)
        ch.addWidget(label(_("Compare with"), "dim"))
        self.cmp_cb = QComboBox()
        self.cmp_cb.setMinimumWidth(150)
        self.cmp_cb.setMaximumWidth(330)
        # as wide as 26 characters: a long name (long in French) would make the toolbar, so the window, wider than
        # the screen; the list that opens shows the names whole
        self.cmp_cb.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.cmp_cb.setMinimumContentsLength(26)
        self.cmp_cb.currentIndexChanged.connect(self._cmp_picked)
        ch.addWidget(self.cmp_cb)
        self.cmp_opts = QWidget()                  # shown once an image is chosen
        oh = QHBoxLayout(self.cmp_opts)
        oh.setContentsMargins(0, 0, 0, 0)
        oh.setSpacing(6)
        seg = QHBoxLayout()
        seg.setSpacing(0)
        self.split_b = QPushButton(_("Split"))
        self.side_b = QPushButton(_("Side by side"))
        grp = QButtonGroup(self)
        for b, mode, tip in ((self.split_b, "split", _("Before on the left of the divider, after on the right")),
                             (self.side_b, "side", _("The two images next to each other, zoom and pan linked"))):
            b.setObjectName("ivseg")
            b.setCheckable(True)
            b.setToolTip(tip)
            b.clicked.connect(lambda _x=False, m=mode: self.set_compare_mode(m))
            grp.addButton(b)
            seg.addWidget(b)
        self.split_b.setChecked(True)
        oh.addLayout(seg)
        self.same_c = QCheckBox(_("Same stretch"))
        self.same_c.setToolTip(_("Show the compared image with the same stretch as this one"))
        self.same_c.toggled.connect(self.set_same_stretch)
        oh.addWidget(self.same_c)
        ch.addWidget(self.cmp_opts)
        self.cmp_opts.hide()
        ch.addSpacing(6)
        ch.addWidget(_sep())
        h.addWidget(self.cmp_box)
        self.cmp_box.setVisible(False)
        self._compare_allowed = compare
        h.addSpacing(6)
        self.auto_b = tool(_("Auto-stretch"), _("Display only: the data stay linear (A)"), self._auto_toggled, True)
        self.hist_b = tool(_("Histogram"), _("The stretch and the histogram (H)"), self.show_histogram, True)
        # the toggles of a tool on the image (Inspect's Gradient: the survey, the marks), after the viewer's own
        self.tool_box = QWidget()
        th = QHBoxLayout(self.tool_box)
        th.setContentsMargins(6, 0, 0, 0)
        th.setSpacing(6)
        h.addWidget(self.tool_box)
        self.tool_box.hide()
        h.addStretch(1)
        # what leaves the viewer, in one menu (a file, the clipboard, Post): one button, whatever the language
        self.out_b = QPushButton(_("Export"))
        self.out_b.setObjectName("ivtool")
        self.out_b.setToolTip(_("Export the image: to a file, to the clipboard or to Post"))
        menu = QMenu(self.out_b)
        menu.setToolTipsVisible(True)

        def item(text, tip, slot, shown):
            a = menu.addAction(text)
            a.setToolTip(tip)
            a.triggered.connect(lambda _c=False: slot())
            a.setVisible(shown)
            return a
        self.export_b = item(_("File…"), _("Write the image to a file: FITS, TIFF, PNG or JPEG (Ctrl+E)"),
                             self.export_dialog, export)
        self.copy_b = item(_("Copy image"), _("Copies the picture as it is shown here (its zoom, stretch and overlays) "
                                              "to the clipboard, to paste it into a message or another program "
                                              "(Ctrl+C)"), self.copy_view, copy)
        self.post_b = item(_("Send to Post"), _("Use this image in Post"), self._send_post, post)
        self.out_b.setMenu(menu)
        menu_button(self.out_b)
        self.out_b.setVisible(export or copy or post)
        h.addWidget(self.out_b)
        return bar

    def _readout_bar(self) -> QFrame:
        bar = QFrame()
        bar.setObjectName("ivread")
        h = QHBoxLayout(bar)
        h.setContentsMargins(10, 3, 10, 3)
        h.setSpacing(10)
        self.readout = QLabel("")
        self.readout.setObjectName("ivread")
        self.readout.setTextFormat(Qt.PlainText)
        h.addWidget(self.readout, 1)
        #: what the active tool does (its gestures and keys), as Post's line under its image
        self.tool_hint = label("", "small")
        self.tool_hint.hide()
        h.addWidget(self.tool_hint, 2)
        self.size_l = label("", "small")
        h.addWidget(self.size_l)
        self.units_cb = QComboBox()
        self.units_cb.addItem("ADU", "adu")
        self.units_cb.addItem("0–1", "norm")
        self.units_cb.setToolTip(_("The values in the readout: ADU, or 0 to 1 (the saturation level is 1)"))
        self.units_cb.currentIndexChanged.connect(lambda _i: self._set_units(self.units_cb.currentData()))
        h.addWidget(self.units_cb)
        return bar

    def _sequence_bar(self) -> QFrame:
        self.seq = QFrame()
        self.seq.setObjectName("ivseq")
        v = QVBoxLayout(self.seq)
        v.setContentsMargins(10, 6, 10, 4)
        v.setSpacing(6)
        row = QHBoxLayout()
        row.setSpacing(6)
        self.prev_b = icon_only(QPushButton(), "previous", _("Previous frame (←)"), 16)
        self.prev_b.setObjectName("ivtool")
        self.prev_b.clicked.connect(self.previous_frame)
        row.addWidget(self.prev_b)
        self.num = QSpinBox()                     # with its arrows (see StretchPanel)
        self.num.setMinimum(1)
        self.num.setKeyboardTracking(False)
        self.num.setAlignment(Qt.AlignRight)
        self.num.setFixedWidth(72)
        self.num.setToolTip(_("Frame number"))
        self.num.valueChanged.connect(lambda n: self.set_index(n - 1))
        row.addWidget(self.num)
        self.total_l = label("", "dim")
        row.addWidget(self.total_l)
        self.next_b = icon_only(QPushButton(), "next", _("Next frame (→)"), 16)
        self.next_b.setObjectName("ivtool")
        self.next_b.clicked.connect(self.next_frame)
        row.addWidget(self.next_b)
        self.blink_b = set_icon(QPushButton(_("Blink")), "play", 16)
        self.blink_b.setObjectName("toggle")
        self.blink_b.setCheckable(True)
        self.blink_b.setToolTip(_("Plays the frames at 5 frames/s (Space)"))
        self.blink_b.toggled.connect(self._blink_toggled)
        row.addWidget(self.blink_b)
        self.lock_c = QCheckBox(_("Same stretch"))
        self.lock_c.setToolTip(_("Keeps the stretch of the frame shown for the next frames of the same kind, so that "
                                 "blinking shows what changes. Off: each frame gets its own automatic stretch."))
        self.lock_c.setChecked(False)
        self.lock_c.toggled.connect(self.set_lock_stretch)
        row.addWidget(self.lock_c)
        row.addSpacing(14)
        self.band = InfoBand()
        self.band.excludeClicked.connect(self.toggle_excluded)
        # never wider than the window: it scrolls sideways instead; its room is what the row's other controls leave
        # (the bar's smallest width is under a 1366 x 768 window's Inspect pane, so that nothing is drawn over another)
        row.addWidget(SideScroll(self.band, min_width=110), 1)
        for w in (self.band.state, self.band.toggle):  # Included / Exclude stay in view, out of the scrolled row
            self.band.layout().removeWidget(w)
            row.addWidget(w)
        v.addLayout(row)
        v.addWidget(self.band.why)
        self.strip = FilmStrip()
        self.strip.picked.connect(self.set_index)
        self.strip.visibleChanged.connect(self._want_thumbs)
        v.addWidget(self.strip)
        self.seq.hide()
        return self.seq

    def _shortcuts(self):
        def sc(keys, fn):
            for k in keys:
                s = QShortcut(QKeySequence(k), self)
                s.setContext(Qt.WidgetWithChildrenShortcut)
                s.activated.connect(fn)
        sc(("F", "Ctrl+0"), self.fit)
        sc(("1", "Ctrl+1"), lambda: self.set_zoom_percent(100))
        sc(("2",), lambda: self.set_zoom_percent(200))
        sc(("+", "=", "Ctrl++"), lambda: self.canvas.zoom_by(ZOOM_STEP))
        sc(("-", "Ctrl+-"), lambda: self.canvas.zoom_by(1 / ZOOM_STEP))
        sc(("Right", "PgDown"), self.next_frame)
        sc(("Left", "PgUp"), self.previous_frame)
        sc(("Home",), lambda: self.set_index(0))
        sc(("End",), lambda: self.set_index(len(self._frames) - 1))
        sc(("X",), self.toggle_excluded)
        sc(("Space",), lambda: self.blink_b.toggle() if self._frames else None)
        sc(("A",), lambda: self.auto_b.toggle())
        sc(("H",), lambda: self.hist_b.toggle())
        sc(("Ctrl+C",), self.copy_view)
        sc(("Ctrl+E",), self.export_dialog)

    # ------------------------------------------------------------ images
    def set_image(self, image: ViewImage | np.ndarray, keep_view: bool = False, keep_stretch: bool = False):
        """Show an image (prepared in a worker thread; at once when the loader has it)."""
        vi = image if isinstance(image, ViewImage) else ViewImage(image)
        self._vi = vi
        self._want = vi.identity()
        self._keep = (keep_view, keep_stretch)
        self._shown_at = time.perf_counter()
        hit = self.loader.cached(self._want)
        if hit is not None:
            self._show(hit)
            return
        self.canvas.busy = self.canvas.display is not None
        self.canvas.message = "" if self.canvas.display is not None else _("Loading the image…")
        self.canvas.update()
        self._update_actions()                      # the picture still shown is the previous one: not written as this
        self.loader.request(vi, 0)

    def image(self) -> ViewImage | None:
        return self._vi

    def prepared(self) -> Prepared | None:
        return self._prep

    def clear(self):
        self._vi = self._prep = self._main = None
        self._want = None
        self.canvas.set_display(None)
        self.canvas.message = ""
        self.panel.set_image(None, None)
        self.readout.setText("")
        self.size_l.setText("")
        self._update_actions()

    def _ready(self, key, prep: Prepared):
        if key == self._want:
            self._show(prep)
        if self._cmp_i is not None and self._cmp_i < len(self._targets) and \
                self._targets[self._cmp_i].identity() == key:
            self._show_compare(prep)
        if self._frames:
            self._prefetch()

    def _failed(self, key, why: str):
        if key == self._want:
            self._failed_key = key
            self.canvas.busy = False
            self.canvas.message = why if key in self.loader.unavailable else \
                _("This image cannot be shown: {why}").format(why=plain_error(why))
            self._failed_text = self.canvas.message
            log_detail(getattr(self.window(), "s", None), why)      # the program's error: in the Log only
            if not self._frames:
                self.canvas.set_display(None)
            self.canvas.update()
            self._update_actions()
        if self._cmp_i is not None and self._cmp_i < len(self._targets) and \
                self._targets[self._cmp_i].identity() == key:
            self._cmp_failed = key                  # the compared image cannot be read: said, and not waited for
            text = why if key in self.loader.unavailable else \
                _("This image cannot be shown: {why}").format(why=plain_error(why))
            self.canvas2.message = text
            self.canvas2.update()
            self.toasted.emit(text)

    _cmp_failed = None
    _failed_text = ""

    def _current(self) -> Prepared | None:
        """The prepared image when it is the one asked for: None while another is read, or could not be."""
        p = self._prep
        return p if p is not None and p.key == self._want else None

    def _show(self, prep: Prepared):
        """Show a prepared image. Its stretch: the automatic one of its kind, computed for it (the mode and the linked
        channels the user chose for the image before, when it was of the same kind); with "Same stretch for all
        frames", the stretch the frames of its kind were shown with (each kind its own: a dark's stretch never shows a
        flat)."""
        keep_view, keep_stretch = self._keep
        self._prep = prep
        old = self._st
        kind = kind_of(self._vi)
        same = old is not None and self._st_kind == kind
        locked = self._locked.get(kind) if self._frames and self._lock else None
        if locked is not None:
            st = locked.adapted(prep.channels)
        elif same and keep_stretch and not self._frames:
            st = old.adapted(prep.channels)
        elif same and old.auto:
            st = default_stretch(prep, old.mode, old.linked if prep.channels > 1 and old.channels > 1 else None, None,
                                 self._vi)
        else:
            st = default_stretch(prep, vi=self._vi)
        self._st = st
        self._st_kind = kind
        if self._frames and self._lock and locked is None:
            self._locked[kind] = st.copy()
        if self._main is not None:
            self._main.release()
        self._main = Display(prep, st)
        self.canvas.busy = False
        self.canvas.message = ""
        self.canvas.set_display(self._main, keep_view=keep_view or bool(self._frames))
        if self._cmp is not None:
            self._align()                           # the compared image on the grid of this one
        if self._cmp is not None and self._cmp_mode == "split":
            self.canvas.set_compare(self._cmp)
        self.panel.set_image(prep, st, self._vi)
        self._sync_auto_button()
        self._apply_same()
        self.size_l.setText(self._size_text(prep))
        self.units_cb.blockSignals(True)
        self._units = self._vi.units if self._vi.units in ("adu", "norm") else "adu"
        self.units_cb.setCurrentIndex(0 if self._units == "adu" else 1)
        self.units_cb.blockSignals(False)
        self._update_actions()
        self._names()
        self.canvas.repaint()
        if self._shown_at is not None:
            self.times["open"] = time.perf_counter() - self._shown_at
            self._shown_at = None
        self.imageShown.emit(prep.key)

    def _size_text(self, prep: Prepared) -> str:
        d = prep.data
        kind = (_("X-Trans") if len(prep.cfa or "") == 36 else _("Bayer {cfa}").format(cfa=prep.cfa) if prep.cfa else
                _("mono") if prep.channels == 1 else _("RGB"))
        depth = (_("{n}-bit float").format(n=d.dtype.itemsize * 8) if np.issubdtype(d.dtype, np.floating)
                 else _("{n}-bit").format(n=d.dtype.itemsize * 8))
        return f"{prep.width} × {prep.height}  ·  {kind}  ·  {depth}"

    def _update_actions(self):
        has = self._prep is not None
        for b in (self.export_b, self.post_b, self.copy_b, self.out_b, self.fit_b, self.one_b, self.zoom_b, self.auto_b):
            b.setEnabled(has)
        # the image asked for is not the one prepared (still read, or unreadable): never written or sent as it
        why = "" if self._want is None or self._current() is not None else \
            self._failed_text if self._failed_key == self._want else _("Loading the image…")
        if why:
            self.export_b.setEnabled(False)
            self.post_b.setEnabled(False)
        self.export_b.setToolTip(why or _("Write the image to a file: FITS, TIFF, PNG or JPEG (Ctrl+E)"))
        if not self._post_ok:
            self.post_b.setEnabled(False)
        self.post_b.setToolTip(self._post_tip if not self._post_ok else why or _("Use this image in Post"))
        self.cmp_box.setVisible(self._compare_allowed and bool(self._targets))

    # ------------------------------------------------------------ the stretch
    def stretch(self) -> sm.Stretch | None:
        return self._st.copy() if self._st is not None else None

    def set_stretch(self, st: sm.Stretch, emit: bool = False):
        """Show the image through ``st`` (a ``stretch.Stretch``, e.g. another view's)."""
        if self._main is None:
            self._st = st.copy()
            return
        self._st = st.adapted(self._main.prep.channels)
        self._main.set_stretch(self._st)
        self.panel.set_stretch(self._st)
        self._sync_auto_button()
        self._keep_locked()
        self._apply_same()
        self.canvas.update()
        self.canvas2.update()
        if emit:
            self.stretchChanged.emit(self._st.copy())

    def _keep_locked(self):
        """With "Same stretch for all frames", the stretch now shown is the one of the frames of its kind."""
        if self._frames and self._lock and self._st is not None and self._prep is not None:
            self._locked[kind_of(self._vi)] = self._st.copy()

    def _kind_auto(self, mode: str | None, linked: bool | None, neutral: bool | None) -> sm.Stretch:
        return sm.auto_for(self._prep.stats, self._vi.kind if self._vi is not None else "", mode, linked, neutral)

    def auto_stretch(self):
        """The automatic stretch of the image's kind again, with the Neutral sky choice the image had (after
        Auto-stretch off, which shows the data linear and plain, the choice made before it: once, turning Auto-stretch
        off and on again switched a stack shown with a neutral sky to its camera colours)."""
        if self._prep is not None:
            st = self._st
            neutral = st.neutral
            off = st.mode == "linear" and not st.auto           # after Auto-stretch off: the kind's own stretch
            if off and self._neutral_before is not None:
                neutral = self._neutral_before
            self.set_stretch(self._kind_auto(None if off else st.mode, None if off else st.linked, neutral), emit=True)

    _neutral_before: bool | None = None

    def set_neutral(self, on: bool):
        """Neutral sky on or off (display only): the automatic stretch with or without it."""
        if self._prep is None or self._st is None or bool(self._st.neutral) == bool(on):
            return
        st = self._st
        self.set_stretch(self._kind_auto(st.mode if st.mode != "linear" or st.auto else None, st.linked, bool(on)),
                         emit=True)

    def _panel_changed(self, st: sm.Stretch):
        self._st = st.adapted(self._main.prep.channels) if self._main else st
        if self._main is not None:
            self._main.set_stretch(self._st)
        self._sync_auto_button()
        self._keep_locked()
        self._apply_same()
        self.canvas.update()
        self.canvas2.update()
        self.stretchChanged.emit(self._st.copy())

    def _auto_toggled(self, on: bool):
        if self._prep is None or self._busy_auto:
            return
        if on:
            self.auto_stretch()
            return
        self._neutral_before = bool(self._st.neutral) if self._st is not None else None
        stats = self._prep.stats
        c = self._prep.channels
        st = sm.Stretch("linear", np.full(c, float(stats.vmin.min())), np.full(c, 0.5 * (stats.lo + stats.hi)),
                        np.full(c, float(stats.vmax.max())), np.ones(c), np.zeros(c), True, False, False)
        self.set_stretch(st, emit=True)

    _busy_auto = False

    def _sync_auto_button(self):
        self._busy_auto = True
        try:
            st = self._st
            self.auto_b.setChecked(bool(st is not None and (st.auto or st.mode != "linear")))
        finally:
            self._busy_auto = False

    def show_histogram(self, on: bool):
        self.panel.setVisible(bool(on))
        self.panel_box.setVisible(bool(on))
        if self.hist_b.isChecked() != bool(on):
            self.hist_b.setChecked(bool(on))

    # ------------------------------------------------------------ the view
    def fit(self):
        self.canvas.fit()
        if self.canvas2.isVisible():
            self.canvas2.fit()

    def zoom_to(self, scale: float, cx: float | None = None, cy: float | None = None):
        self.canvas.zoom_to(scale, cx, cy)

    def set_zoom_percent(self, pct: float):
        self.canvas.set_zoom_percent(pct)

    def view_state(self) -> dict:
        """Where the view is, independent of the image size: zoom relative to the image width, centre in fractions."""
        c = self.canvas
        if c.display is None:
            return {"fitted": True}
        w, h = c.display.size()
        out = {"fitted": c.fitted, "zoom": c.scale * w, "fx": c.cx / w, "fy": c.cy / h}
        pl = getattr(c.display, "place", None)
        if pl is not None:
            # the same point of the common grid (a final image is a crop of the stack at another scale)
            m = pl.m
            out["grid"] = (pl.frame, m[0] * c.cx + m[1] * c.cy + m[2], m[3] * c.cx + m[4] * c.cy + m[5],
                           c.scale / max(abs(m[0]), 1e-12))
        return out

    def set_view_state(self, state: dict):
        c = self.canvas
        pl = getattr(c.display, "place", None) if c.display is not None else None
        g = state.get("grid")
        if pl is not None and g is not None and g[0] == pl.frame and not state.get("fitted", True):
            # back from the common grid to this image's pixels
            inv = np.linalg.inv(pl.matrix())
            x, y, _one = inv @ np.array([g[1], g[2], 1.0])
            c.zoom_to(g[3] * abs(pl.m[0]), float(x), float(y))
            return
        _mirror_state(c, state)

    def _canvas_moved(self, cv: ImageCanvas):
        self._zoom_label()
        if self._syncing:
            return
        self._syncing = True
        try:
            if self.canvas2.isVisible():
                other = self.canvas2 if cv is self.canvas else self.canvas
                _mirror_state(other, _state_of(cv))
        finally:
            self._syncing = False
        if cv is self.canvas or self.canvas2.isVisible():
            self.viewChanged.emit()

    def _zoom_label(self):
        self.zoom_b.setText(number(self.canvas.zoom_percent(), 0 if self.canvas.zoom_percent() >= 10 else 1) + " %")

    # ------------------------------------------------------------ readout
    def _set_units(self, u: str):
        self._units = u

    def pixel_text(self, x: float, y: float, prep: Prepared | None = None, vi: ViewImage | None = None) -> str:
        """The readout of the image point (x, y): position, values (ADU or 0-1), RA / Dec."""
        if prep is None:
            prep, vi = self._prep, self._vi
        if prep is None:
            return ""
        xi, yi = int(math.floor(x)), int(math.floor(y))
        got = prep.value(xi, yi)
        if got is None:
            return ""
        vals, letter = got
        vi = vi or prep.vi

        def whole(v) -> str:                       # a pixel without a value (outside a stack's cover): a dash
            return str(int(round(v))) if math.isfinite(v) else "–"
        if vi.stretched:                           # a picture for the eye: its levels, without a unit
            text = "   ".join(f"{c} {whole(v)}" for c, v in zip("RGB", vals)) if len(vals) == 3 else whole(vals[0])
            return f"x {xi:>5}  y {yi:>5}    {text}" + self._sky_part(prep, x, y)
        if vi.units == "count":                    # a count per pixel (a rejection map): whole numbers
            text = whole(vals[0]) if len(vals) == 1 else "   ".join(f"{c} {whole(v)}" for c, v in zip("RGB", vals))
            return f"x {xi:>5}  y {yi:>5}    {text}" + self._sky_part(prep, x, y)
        if vi.units not in ("adu", "norm"):        # a number without a unit (a coverage map)
            fmt = [value_text(v, prep.stats) for v in vals]
            text = "   ".join(f"{c} {v}" for c, v in zip("RGB", fmt)) if len(vals) == 3 else fmt[0]
            return f"x {xi:>5}  y {yi:>5}    {text}" + self._sky_part(prep, x, y)
        white = vi.white_level()
        if vi.units == "norm" and self._units == "adu":
            vals, unit = [v * white for v in vals], "ADU"
        elif vi.units != "norm" and self._units == "norm":
            vals, unit = [v / white for v in vals], ""
        else:
            unit = "ADU" if self._units == "adu" else ""
        integer = prep.integer and self._units == "adu" and vi.units != "norm"
        if self._units == "norm":
            fmt = [number(v, 5) for v in vals]
        else:
            fmt = [value_text(v, None if integer else prep.stats, integer) for v in vals]
        if len(vals) == 3:
            text = "   ".join(f"{c} {v}" for c, v in zip("RGB", fmt))
        elif letter:
            text = f"{letter} {fmt[0]}"
        else:
            text = fmt[0]
        return f"x {xi:>5}  y {yi:>5}    {text}" + (f" {unit}" if unit else "") + self._sky_part(prep, x, y)

    @staticmethod
    def _sky_part(prep: Prepared, x: float, y: float) -> str:
        if prep.wcs is None:
            return ""
        try:
            ra, dec = prep.wcs.pixel_to_world_values(x - 0.5, y - 0.5)
            if np.isfinite(ra) and np.isfinite(dec):
                return "    " + sky_text(float(ra), float(dec))
        except Exception:  # noqa: BLE001 - outside the solution
            pass
        return ""

    def _hover(self, cv: ImageCanvas, x: float, y: float, which: int):
        if cv is self.canvas2 or which == 1:
            prep = self._cmp.prep if self._cmp is not None else None
            vi = self._targets[self._cmp_i] if self._cmp_i is not None else None
            name = vi.name if vi is not None else ""
            if self._cmp is not None and self._cmp.align is not None:
                x, y = self._cmp.align.apply(x, y)  # the point of this grid -> the compared image's own pixel
        else:
            prep, vi = self._prep, self._vi
            name = ""
            self.hovered.emit(x, y)
        t = self.pixel_text(x, y, prep, vi) if prep is not None else ""
        self.readout.setText((name + "  ·  " if t and name else "") + t)

    def _left(self):
        self.readout.setText("")

    # ------------------------------------------------------------ compare
    def set_compare_targets(self, targets: list[ViewImage], current: int | None = None):
        """The images this one can be compared with (the previous steps of the workflow), by name."""
        self._targets = list(targets)
        self.cmp_cb.blockSignals(True)
        self.cmp_cb.clear()
        self.cmp_cb.addItem("—", None)
        for i, t in enumerate(self._targets):
            self.cmp_cb.addItem(t.name or _("Image {n}").format(n=i + 1), i)
        view = self.cmp_cb.view()
        view.setMinimumWidth(view.sizeHintForColumn(0) + 24)          # the list shows the names whole
        self.cmp_cb.blockSignals(False)
        self._update_actions()
        self.set_compare(current)

    def set_compare(self, index: int | None):
        if index is not None and not (0 <= index < len(self._targets)):
            index = None
        self._cmp_i = index
        self.cmp_cb.blockSignals(True)
        self.cmp_cb.setCurrentIndex(0 if index is None else index + 1)
        self.cmp_cb.setToolTip(self.cmp_cb.currentText() if index is not None else "")   # a name cut short, whole
        self.cmp_cb.blockSignals(False)
        if self._cmp is not None:
            self._cmp.release()
        self._cmp = None
        self._align()
        self.canvas.set_compare(None)
        self.canvas2.set_display(None)
        self.canvas2.setVisible(index is not None and self._cmp_mode == "side")
        self.cmp_opts.setVisible(index is not None)
        if index is not None:
            vi = self._targets[index]
            hit = self.loader.cached(vi.identity())
            if hit is not None:
                self._show_compare(hit)
            else:
                self.canvas2.message = _("Loading the image…")
                self.loader.request(vi, 0)
        self._names()
        self.compareChanged.emit(None if index is None else self._targets[index].identity())

    def _cmp_picked(self, i):
        self.set_compare(self.cmp_cb.itemData(i))

    def set_compare_mode(self, mode: str):
        self._cmp_mode = "side" if mode == "side" else "split"
        self.split_b.setChecked(self._cmp_mode == "split")
        self.side_b.setChecked(self._cmp_mode == "side")
        side = self._cmp_mode == "side" and self._cmp_i is not None
        self.canvas2.setVisible(side)
        if self._cmp is not None:
            if side:
                self.canvas.set_compare(None)
                self.canvas2.set_display(self._cmp)
                _mirror_state(self.canvas2, _state_of(self.canvas))
                QTimer.singleShot(0, lambda: _mirror_state(self.canvas2, _state_of(self.canvas)))
            else:
                self.canvas2.set_display(None)
                self.canvas.set_compare(self._cmp)
        self._names()

    def compare_mode(self) -> str:
        return self._cmp_mode

    def set_same_stretch(self, on: bool):
        self._same = bool(on)
        if self.same_c.isChecked() != self._same:
            self.same_c.setChecked(self._same)
        self._apply_same()

    def _same_possible(self) -> bool:
        """Same stretch only between images of one kind of values: a picture already stretched for the eye (the final
        image, 0-255) and linear data (a stack) never share one (the data would show black)."""
        vi = self._targets[self._cmp_i] if self._cmp_i is not None and self._cmp_i < len(self._targets) else None
        return vi is None or self._vi is None or bool(vi.stretched) == bool(self._vi.stretched)

    def _cmp_stretch(self, prep: Prepared) -> sm.Stretch:
        if self._same and self._st is not None and self._same_possible():
            return self._st.adapted(prep.channels)
        st = self._st
        vi = self._targets[self._cmp_i] if self._cmp_i is not None and self._cmp_i < len(self._targets) else None
        if st is None or kind_of(vi or prep.vi) != kind_of(self._vi):
            return default_stretch(prep, vi=vi)               # another kind of image: its own automatic stretch
        mode = st.mode if st.mode != "linear" else None
        # Neutral sky is the user's choice for both sides: "before" in the camera's colours stays as it is unless the
        # main picture is shown with a neutral sky (a mosaic and a mono picture keep their own default)
        neutral = None
        if prep.channels == 3 and self._prep is not None and self._prep.channels == 3 and \
                not (vi is not None and vi.stretched):
            neutral = bool(st.neutral) if (st.auto or st.mode != "linear") else bool(self._neutral_before)
            if self._cmp_neutral is not None:
                neutral = bool(self._cmp_neutral)
        return default_stretch(prep, mode, st.linked if prep.channels > 1 else True, neutral, vi)

    _cmp_neutral: bool | None = None

    def set_compare_neutral(self, on: bool | None):
        """Neutral sky for the compared image only (display only): True or False, or None to follow this one's."""
        self._cmp_neutral = None if on is None else bool(on)
        self._apply_same()

    def _apply_same(self):
        if self._cmp is not None:
            self._cmp.set_stretch(self._cmp_stretch(self._cmp.prep))
            self.canvas.update()
            self.canvas2.update()

    def _show_compare(self, prep: Prepared):
        if self._cmp is not None:
            self._cmp.release()
        self._cmp = Display(prep, self._cmp_stretch(prep))
        self._align()
        self.canvas2.message = ""
        if self._cmp_mode == "side":
            self.canvas2.set_display(self._cmp)
            self.canvas2.show()
            _mirror_state(self.canvas2, _state_of(self.canvas))
        else:
            self.canvas.set_compare(self._cmp)
        self._names()

    def _align(self):
        """Put the compared image on the grid of the one shown, so that a star sits at the same place in both (split
        and side by side; ``alignment``: the results' geometry, else the astrometric solutions), and say how in one line
        above the images. Display only: the compared image is resampled as it is drawn."""
        d = self._cmp
        al = None
        if d is not None and self._prep is not None:
            try:
                al = alignment(self._prep, d.prep)
            except Exception:  # noqa: BLE001 - a geometry that cannot be read: drawn as before, matched by width
                al = None
            d.set_alignment(al, self._prep.size())
        self._al = al
        text = self.alignment_note()
        self.align_l.setText(text)
        self.align_bar.setVisible(bool(text))
        if d is not None and self.canvas2.isVisible() and self.canvas2.display is d:
            self.canvas2.set_display(d)
            _mirror_state(self.canvas2, _state_of(self.canvas))
        self.canvas.update()
        self.canvas2.update()

    def compare_alignment(self) -> Alignment | None:
        """How the compared image is drawn on this one (None: the same grid, or no compare)."""
        return self._al if self._cmp is not None else None

    def alignment_note(self) -> str:
        """The line that says how the compared image was put on this one's grid ("" when nothing was done)."""
        al = self.compare_alignment()
        if al is None or self._cmp_i is None or self._cmp_i >= len(self._targets):
            return ""
        name = self._targets[self._cmp_i].name or _("the compared image")
        if al.how == "width":
            return _("Not aligned: no common grid is known for “{name}”; it is matched to this image by its "
                     "width").format(name=name)
        k = 1.0 / al.scale
        ox, oy = al.origin()
        bits = [_("× {k}").format(k=number(k, 3 if k < 0.1 else 2))] if abs(k - 1.0) > 5e-4 else []
        if abs(al.angle) >= 0.05:
            bits.append(_("turned {a}°").format(a=number(-al.angle, 2)))
        if abs(ox) >= 0.05 or abs(oy) >= 0.05:
            bits.append(_("shifted {dx}, {dy} px").format(dx=number(ox, 1), dy=number(oy, 1)))
        if al.field is not None:
            bits.append(_("their distortions followed, up to {d} px").format(d=number(al.residual, 1)))
        what = ", ".join(bits) or _("on the same pixels")
        if al.how == "sky":
            return _("Aligned for display by the astrometric solutions: “{name}” resampled onto this image "
                     "({what})").format(name=name, what=what)
        return _("Aligned for display by the geometry of the results (drizzle scale, crop, size): “{name}” "
                 "resampled onto this image ({what})").format(name=name, what=what)

    def _names(self):
        ok = self._same_possible()
        self.same_c.setEnabled(ok)
        self.same_c.setToolTip(_("Show the compared image with the same stretch as this one") if ok else _(
            "Not possible here: one image is already stretched for the eye (the final image, rendered by Post), the "
            "other is linear data. Each keeps its own stretch."))
        main = self._vi.name if self._vi is not None else ""
        other = self._targets[self._cmp_i].name if self._cmp_i is not None else ""
        self.canvas.split_names = (other, main)
        side = self._cmp_mode == "side" and self._cmp_i is not None
        self.canvas.title = main if side else ""
        self.canvas2.title = other if side else ""
        self.canvas.update()
        self.canvas2.update()

    # ------------------------------------------------------------ sequence
    def set_sequence(self, frames: list[ViewImage], index: int = 0):
        """Show a sequence of frames (paths or loaders, with the caller's ``info``)."""
        self._frames = list(frames)
        self._locked = {}
        self.seq.setVisible(bool(self._frames))
        if not self._frames:
            self._blink.stop()
            return
        self.strip.set_count(len(self._frames))
        for i, f in enumerate(self._frames):
            self.strip.set_excluded(i, bool((f.info or {}).get("excluded")))
            self.strip.set_flagged(i, (f.info or {}).get("flag_level") if (f.info or {}).get("flag") else None)
            th = self.loader.thumbs.get(f.identity())
            if th is not None:
                self.strip.set_thumb(i, th)
        self.num.blockSignals(True)
        self.num.setMaximum(len(self._frames))
        self.num.blockSignals(False)
        self.total_l.setText("/ " + str(len(self._frames)))
        self._index = -1
        self._st = None                           # the first frame shown gets its own automatic stretch
        self.set_index(index)
        QTimer.singleShot(0, lambda: self._want_thumbs(*self.strip.visible_range()))

    def frames(self) -> list[ViewImage]:
        return list(self._frames)

    def index(self) -> int:
        return self._index

    def set_index(self, i: int):
        if not self._frames:
            return
        i = int(np.clip(i, 0, len(self._frames) - 1))
        if i == self._index:
            return
        self._index = i
        f = self._frames[i]
        self.num.blockSignals(True)
        self.num.setValue(i + 1)
        self.num.blockSignals(False)
        self.band.set_frame(f, i, len(self._frames))
        self.strip.set_current(i)
        self.prev_b.setEnabled(i > 0)
        self.next_b.setEnabled(i < len(self._frames) - 1)
        self.set_image(f, keep_view=True, keep_stretch=self._lock)
        self._prefetch()
        self.frameChanged.emit(i)

    def next_frame(self):
        if self._frames:
            self.set_index(self._index + 1)

    def previous_frame(self):
        if self._frames:
            self.set_index(self._index - 1)

    def _prefetch(self):
        """The neighbours of the frame shown, ahead of time (the next ones first)."""
        n = len(self._frames)
        keys = []
        for pr, d in ((1, 1), (1, -1), (2, 2), (3, -2), (3, 3)):
            j = self._index + d
            if 0 <= j < n:
                vi = self._frames[j]
                keys.append(vi.identity())
                if self.loader.cached(vi.identity()) is None:
                    self.loader.request(vi, pr)
        self.loader.forget_requests(keep=keys)

    def set_frame_info(self, i: int, info: dict):
        if 0 <= i < len(self._frames):
            self._frames[i].info = dict(info)
            self.strip.set_excluded(i, bool(info.get("excluded")))
            self.strip.set_flagged(i, info.get("flag_level") if info.get("flag") else None)
            if i == self._index:
                self.band.set_frame(self._frames[i], i, len(self._frames))

    def set_excluded(self, i: int, excluded: bool):
        """Mark a frame kept or excluded (no signal: the caller's own change)."""
        if 0 <= i < len(self._frames):
            info = dict(self._frames[i].info or {})
            info["excluded"] = bool(excluded)
            self.set_frame_info(i, info)

    def set_exclude_mode(self, mode: str, tip: str = ""):
        """``on`` (default), ``off`` (no Exclude / Include, no state: masters) or ``spec`` (◇: the engine cannot
        leave such a frame out yet; the toggle is shown disabled with ``tip``)."""
        self.band.set_mode(mode, tip)
        if self._frames and 0 <= self._index < len(self._frames):
            self.band.set_frame(self._frames[self._index], self._index, len(self._frames))

    def set_post_enabled(self, on: bool, tip: str = ""):
        """Send to Post can be offered (the Post page takes images) or not: disabled, ``tip`` says why."""
        self._post_ok = bool(on)
        self._post_tip = tip
        self._update_actions()

    _post_ok = True
    _post_tip = ""

    def toggle_excluded(self):
        """Exclude / Include the frame shown: ``excludeToggled(index, excluded)``."""
        if not self._frames or self.band.mode != "on":
            return
        i = self._index
        now = not bool((self._frames[i].info or {}).get("excluded"))
        self.set_excluded(i, now)
        self.excludeToggled.emit(i, now)

    def set_lock_stretch(self, on: bool):
        """"Same stretch for all frames": on, the stretch shown is kept for the next frames of the same kind (each
        kind its own); off, each frame gets its own automatic stretch, the one shown now at once."""
        on = bool(on)
        changed = on != self._lock
        self._lock = on
        if self.lock_c.isChecked() != self._lock:
            self.lock_c.setChecked(self._lock)
        if not changed:
            return
        self._locked = {}
        if on:
            self._keep_locked()
        elif self._prep is not None and self._st is not None and self._st.auto:
            st = self._st
            self.set_stretch(default_stretch(self._prep, st.mode, st.linked if st.channels > 1 else None, None,
                                             self._vi), emit=True)

    def _want_thumbs(self, i0: int, i1: int):
        for i in range(max(0, i0 - 4), min(len(self._frames), i1 + 4)):
            f = self._frames[i]
            if f.thumb is not None and f.identity() not in self.loader.thumbs:
                self.loader.request_thumb(f, 10 + abs(i - self._index))

    def _thumb(self, key, arr):
        for i, f in enumerate(self._frames):
            if f.identity() == key:
                self.strip.set_thumb(i, arr)

    def _blink_toggled(self, on: bool):
        set_icon(self.blink_b, "pause" if on else "play", 16)
        if on:
            self._blink.start()
        else:
            self._blink.stop()

    def _blink_tick(self):
        if not self._frames:
            self._blink.stop()
            return
        j = (self._index + 1) % len(self._frames)
        if self.loader.cached(self._frames[j].identity()) is None:
            self.loader.request(self._frames[j], 0)
            return
        if j == 0:
            self._index = -1
        self.set_index(j)

    # ------------------------------------------------------------ actions
    def _send_post(self):
        if self._current() is not None:
            self.sendToPost.emit(self._prep.key)

    def grab_view(self) -> QImage:
        return self.area.grab().toImage()

    def copy_view(self):
        if self._prep is None:
            return
        QGuiApplication.clipboard().setImage(self.grab_view())
        self.toasted.emit(_("Image copied to the clipboard, as shown: paste it where you want"))

    def visible_area(self) -> tuple[int, int, int, int]:
        x0, y0, x1, y1 = self.canvas.visible()
        return int(math.floor(x0)), int(math.floor(y0)), int(math.ceil(x1)), int(math.ceil(y1))

    def export(self, path: str | os.PathLike, fmt: str | None = None, content: str = "data",
               area: tuple[int, int, int, int] | None = None, wcs: bool = True) -> Path:
        """Write the image shown (in the GUI thread: the dialog runs it in a worker)."""
        if self._current() is None:
            raise RuntimeError("no image")
        return export_image(path, self._prep, self._st, fmt=fmt, content=content, area=area, with_wcs=wcs,
                            vi=self._vi)

    def export_dialog(self):
        if self._current() is None:
            return None
        dlg = ImageExportDialog(self, self.window())
        dlg.setAttribute(Qt.WA_DeleteOnClose)       # its copy of the image goes with it once closed
        dlg.open()
        return dlg

    def add_overlay(self, fn):
        self.canvas.overlays.append(fn)
        self.canvas.update()

    def set_tool_widgets(self, widgets):
        """The toggles of the tool on the image, in the bar after Histogram (what it draws over the image, as Post's
        Mask view sits in its bar); [] takes them away."""
        lay = self.tool_box.layout()
        while lay.count():
            w = lay.takeAt(0).widget()
            if w is not None:
                w.setParent(None)
        for w in widgets:
            lay.addWidget(w)
        self.tool_box.setVisible(bool(widgets))

    def set_tool_row(self, widget):
        """The bar of an editor on the image, on a row above it (the tools that act on the image sit next to it, as in
        the reference photo editors); None takes it away."""
        lay = self.tool_row.layout()
        while lay.count():
            w = lay.takeAt(0).widget()
            if w is not None:
                w.setParent(None)
        if widget is not None:
            lay.addWidget(widget)
        self.tool_row.setVisible(widget is not None)

    #: the annotations the overlays draw, for Export… ((sky, show), or None): a callable the caller sets
    annotation_source = None

    def annotations(self):
        """(sky, show) the overlays draw (the sky grid, the named objects), or None: Export… offers to burn them in."""
        fn = self.annotation_source
        try:
            return fn() if callable(fn) else None
        except RuntimeError:
            return None

    # ------------------------------------------------------------ tests and scripts
    def wait(self, timeout: float = 60.0) -> bool:
        """Process events until the image asked for (and the compared one) are shown."""
        end = time.monotonic() + timeout
        app = QApplication.instance()
        while time.monotonic() < end:
            app.processEvents()
            done = self._want is None or (self._prep is not None and self._prep.key == self._want) or \
                self._failed_key == self._want
            cmp_done = self._cmp_i is None or self._cmp is not None or (
                self._cmp_i < len(self._targets) and self._cmp_failed == self._targets[self._cmp_i].identity())
            if done and cmp_done:
                app.processEvents()
                return True
            time.sleep(0.003)
        return False


class SyncGroup(QObject):
    """Viewers that move together: zoom and pan, and the stretch too when ``stretch`` ("same stretch on both")."""

    def __init__(self, views=(), stretch: bool = False, parent=None):
        super().__init__(parent)
        self.views: list[ImageView] = []
        self.stretch = bool(stretch)
        self._busy = False
        for v in views:
            self.add(v)

    def add(self, v: ImageView):
        if v in self.views:
            return
        self.views.append(v)
        v.viewChanged.connect(lambda vv=v: self._moved(vv))
        v.stretchChanged.connect(lambda st, vv=v: self._stretched(vv, st))

    def remove(self, v: ImageView):
        if v in self.views:
            self.views.remove(v)

    def set_stretch_shared(self, on: bool):
        self.stretch = bool(on)
        if on and self.views and self.views[0].stretch() is not None:
            self._stretched(self.views[0], self.views[0].stretch())

    def _moved(self, src: ImageView):
        if self._busy:
            return
        self._busy = True
        try:
            st = src.view_state()
            for v in self.views:
                if v is not src:
                    v.set_view_state(st)
        finally:
            self._busy = False

    def _stretched(self, src: ImageView, st):
        if self._busy or not self.stretch:
            return
        self._busy = True
        try:
            for v in self.views:
                if v is not src:
                    v.set_stretch(st)
        finally:
            self._busy = False


# ---------------------------------------------------------------------------------------------- export
FORMATS = {"fits": ("FITS 32-bit float", ".fits"), "tiff16": ("TIFF 16-bit", ".tif"),
           "png16": ("PNG 16-bit", ".png"), "png8": ("PNG 8-bit", ".png"), "jpeg": ("JPEG", ".jpg")}


def format_of(path: str | os.PathLike) -> str:
    suf = Path(path).suffix.lower()
    return {".fits": "fits", ".fit": "fits", ".fts": "fits", ".tif": "tiff16", ".tiff": "tiff16", ".png": "png16",
            ".jpg": "jpeg", ".jpeg": "jpeg"}.get(suf, "fits")


def ascii_text(text: str) -> str:
    """The text without its accents ("Aperçu" -> "Apercu"): what a FITS header can hold (the night test pass: a French
    view's title made the FITS export fail)."""
    import unicodedata
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


def slug(name: str) -> str:
    """A file name from a name: its accents dropped, every character but letters, digits, "_", "." and "-" made "_"."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", ascii_text(name)).strip("_") or "image"


def export_image(path: str | os.PathLike, prep: Prepared, st: sm.Stretch | None = None, fmt: str | None = None,
                 content: str = "data", area: tuple[int, int, int, int] | None = None, with_wcs: bool = True,
                 header: dict | None = None, vi: ViewImage | None = None, annotations=None) -> Path:
    """Write a prepared image. FITS: the data as they are (32-bit float), with the WCS (shifted for a crop) and a
    header. TIFF 16-bit, PNG, JPEG: ``content`` "data" (ADU as 16-bit integers, 0-1 data scaled to the full range) or
    "view" (the stretch shown, at full precision). ``area``: (x0, y0, x1, y1) image pixels, else the whole image.
    ``annotations``: (sky, show) burned into the stretched view of a picture (``annotate.burn``: the sky grid, the
    named objects); never into a FITS file nor into the linear data."""
    path = Path(path)
    fmt = fmt or format_of(path)
    if fmt not in FORMATS:
        raise ValueError(f"unknown format {fmt!r}")
    path.parent.mkdir(parents=True, exist_ok=True)
    vi = vi or prep.vi
    x0, y0, x1, y1 = area if area is not None else (0, 0, prep.width, prep.height)
    if prep.cfa:                                    # a mosaic is cut on whole cells (2 x 2, 6 x 6): the pattern stays
        P = 6 if len(prep.cfa) == 36 else 2
        x0, y0 = x0 // P * P, y0 // P * P
        x1, y1 = min(prep.width, -(-x1 // P) * P), min(prep.height, -(-y1 // P) * P)
    x0, y0 = max(0, int(x0)), max(0, int(y0))
    x1, y1 = min(prep.width, int(x1)), min(prep.height, int(y1))
    if x1 <= x0 or y1 <= y0:
        raise ValueError("empty area")
    if fmt == "fits":
        content = "data"
    if content == "view":
        st = st or default_stretch(prep, vi=vi)
        f = prep.factor
        lv = prep.levels[0][:, y0 // f:-(-y1 // f), x0 // f:-(-x1 // f)]
        img = sm.apply(lv, st.adapted(prep.channels), prep.stats)        # (C, h, w) 0..1
    else:
        d = prep.data
        img = d[y0:y1, x0:x1] if d.ndim == 2 else d[:, y0:y1, x0:x1]
    if fmt == "fits":
        _write_fits(path, img, prep, (x0, y0), with_wcs, header, vi)
        return path
    a = np.asarray(img, np.float32)
    if a.ndim == 3:
        a = a[0] if a.shape[0] == 1 else a.transpose(1, 2, 0)
    if content == "view":
        unit = a
        if annotations is not None and a.ndim == 3:
            from .annotate import burn, shifted
            sky, show = annotations
            unit = burn(np.clip(np.nan_to_num(a, nan=0.0), 0.0, 1.0), shifted(sky, x0, y0), show)
    elif vi.units == "norm":
        unit = a
    else:
        unit = a / 65535.0                          # ADU as they are: 16-bit integers
    unit = np.clip(np.nan_to_num(unit, nan=0.0), 0.0, 1.0)
    if fmt in ("tiff16", "png16"):
        out = (unit * 65535.0 + 0.5).astype(np.uint16)
    else:
        out = (unit * 255.0 + 0.5).astype(np.uint8)
    out = np.ascontiguousarray(out)
    if fmt == "tiff16":
        import tifffile
        tifffile.imwrite(path, out, photometric="rgb" if out.ndim == 3 else "minisblack", compression="zlib",
                         metadata=None)
    elif fmt == "png16":
        from ..io import cvfile                           # any path (review 13.1)
        cvfile.write(path, out[:, :, ::-1] if out.ndim == 3 else out)
    else:
        from PIL import Image
        im = Image.fromarray(out)
        if fmt == "jpeg":
            im.save(path, quality=95)
        else:
            im.save(path)
    return path


def _write_fits(path: Path, img: np.ndarray, prep: Prepared, origin, with_wcs: bool, extra: dict | None,
                vi: ViewImage):
    import datetime as dt
    import warnings

    from astropy.io import fits
    a = np.asarray(img, np.float32)
    if a.ndim == 3 and a.shape[0] == 1:
        a = a[0]
    hdr = fits.Header()
    if with_wcs and prep.wcs is not None and hasattr(prep.wcs, "to_header"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            w = prep.wcs.deepcopy()
            w.wcs.crpix = [w.wcs.crpix[0] - origin[0], w.wcs.crpix[1] - origin[1]]
            hdr.update(w.to_header(relax=True))
    if vi.name:
        hdr["OBJECT"] = (ascii_text(vi.name)[:68], f"the image's name in {APP_NAME}")
    hdr["BUNIT"] = ("ADU" if vi.units == "adu" else "", "data units")
    if prep.cfa:
        hdr["BAYERPAT"] = (prep.cfa, "colour pattern of the mosaic (36 letters: X-Trans)")
        hdr["XBAYROFF"] = 0
        hdr["YBAYROFF"] = 0
    if origin != (0, 0):
        hdr["XORIGIN"] = (int(origin[0]), "x of the first column in the full image")
        hdr["YORIGIN"] = (int(origin[1]), "y of the first row in the full image")
    hdr["CREATOR"] = APP_NAME
    hdr["DATE"] = (dt.datetime.now().isoformat(timespec="seconds"), "file written")
    for k, v in {**(vi.header or {}), **(extra or {})}.items():
        key = str(k).upper()[:8]
        if key in ("HISTORY", "COMMENT"):
            for line in (v if isinstance(v, list | tuple) else [v]):
                hdr.add_history(str(line)) if key == "HISTORY" else hdr.add_comment(str(line))
        elif isinstance(v, str | int | float | bool):
            hdr[key] = v
    hdr.add_history(f"Written by the {APP_NAME} image viewer: the data as they are (no display stretch).")
    fits.PrimaryHDU(a, header=hdr).writeto(path, overwrite=True)


class ImageExportDialog(QDialog):
    """Export…: the format, what is written (the data or the stretched view), the area, the file."""

    finished_export = Signal(str, str)              # path, error ("" when written)

    def __init__(self, view: ImageView, parent=None):
        super().__init__(parent)
        self.view = view
        self.prep = view.prepared()
        self.vi = view.image() or self.prep.vi
        self.st = view.stretch()
        self.result_path: Path | None = None
        self.error = ""
        self.setWindowTitle(_("Export image"))
        self.setMinimumWidth(600)
        cap_dialog(self)
        self.setStyleSheet(QSS)
        v = QVBoxLayout(self)
        v.setContentsMargins(18, 16, 18, 14)
        v.setSpacing(10)
        v.addWidget(label(_("Export image"), "h1"))
        sub = label(f"{self.vi.name or _('Image')}  ·  {view._size_text(self.prep)}", "dim")
        v.addWidget(sub)
        grid = QGridLayout()
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(8)
        grid.setColumnMinimumWidth(0, 110)
        r = 0
        grid.addWidget(label(_("Format")), r, 0)
        self.fmt = QComboBox()
        for key, (name, _suf) in FORMATS.items():
            self.fmt.addItem(_(name), key)
        grid.addWidget(self.fmt, r, 1)
        r += 1
        grid.addWidget(label(_("Content")), r, 0, Qt.AlignTop)
        cb = QVBoxLayout()
        cb.setSpacing(4)
        self.c_data = QRadioButton(_("Data as is"))
        self.c_view = QRadioButton(_("Stretched view"))
        self.c_view.setChecked(True)
        cb.addWidget(self.c_data)
        cb.addWidget(self.c_view)
        grid.addLayout(cb, r, 1)
        r += 1
        grid.addWidget(label(_("Area")), r, 0, Qt.AlignTop)
        ab = QVBoxLayout()
        ab.setSpacing(4)
        self.a_all = QRadioButton(_("Whole image ({w} × {h})").format(w=self.prep.width, h=self.prep.height))
        x0, y0, x1, y1 = view.visible_area()
        self.area = (x0, y0, x1, y1)
        self.a_vis = QRadioButton(_("Visible part ({w} × {h})").format(w=x1 - x0, h=y1 - y0))
        self.a_all.setChecked(True)
        self.a_vis.setEnabled(not view.canvas.whole_visible())
        ab.addWidget(self.a_all)
        ab.addWidget(self.a_vis)
        grid.addLayout(ab, r, 1)
        r += 1
        self.c_wcs = QCheckBox(_("Astrometry (WCS) in header"))
        self.c_wcs.setChecked(hasattr(self.prep.wcs, "to_header"))
        grid.addWidget(self.c_wcs, r, 1)
        r += 1
        # the annotations the view draws (the sky grid, the named objects), burned into a picture; never into a FITS
        self.ann = view.annotations() if callable(getattr(view, "annotations", None)) else None
        self.c_grid = QCheckBox(_("Sky grid"))
        self.c_names = QCheckBox(_("Object names"))
        self.ann_row = QWidget()
        ah = QHBoxLayout(self.ann_row)
        ah.setContentsMargins(0, 0, 0, 0)
        ah.addWidget(label(_("Include annotations:")))
        ah.addWidget(self.c_grid)
        ah.addWidget(self.c_names)
        ah.addStretch(1)
        if self.ann is not None:
            show = self.ann[1]
            self.c_grid.setChecked(bool(show.get("grid")) and bool(getattr(self.ann[0], "lines", None)))
            self.c_names.setChecked(bool(show.get("objects")) and bool(getattr(self.ann[0], "objects", None)))
            self.c_grid.setEnabled(bool(getattr(self.ann[0], "lines", None)))
            self.c_names.setEnabled(bool(getattr(self.ann[0], "objects", None)))
        self.ann_row.setVisible(self.ann is not None)
        grid.addWidget(self.ann_row, r, 1)
        r += 1
        grid.addWidget(label(_("File")), r, 0)
        fh = QHBoxLayout()
        self.dest = QLineEdit()
        self.browse = icon_only(QPushButton(), "open", _("Choose the folder"), 16)
        self.browse.setFixedWidth(34)
        self.browse.setToolTip(_("Choose the folder. The file name stays in the field: type it there."))
        self.browse.clicked.connect(self._browse)
        fh.addWidget(self.dest, 1)
        fh.addWidget(self.browse)
        grid.addLayout(fh, r, 1)
        v.addLayout(grid)
        self.note = label("", "small")
        self.note.setWordWrap(True)
        v.addWidget(self.note)
        self.status = label("", "small")
        self.status.setWordWrap(True)
        v.addWidget(self.status)
        v.addStretch(1)
        bb = QHBoxLayout()
        bb.addStretch(1)
        self.cancel_b = QPushButton(_("Cancel"))
        self.cancel_b.clicked.connect(self.reject)
        self.go = QPushButton(_("Export"))
        self.go.setObjectName("primary")
        self.go.setDefault(True)
        self.go.clicked.connect(self._go)
        bb.addWidget(self.cancel_b)
        bb.addWidget(self.go)
        v.addLayout(bb)
        folder = view.export_dir or (view.default_dir() if callable(view.default_dir) else "") or \
            QStandardPaths.writableLocation(QStandardPaths.PicturesLocation) or str(Path.home())
        self.dest.setText((Path(folder) / (slug(self.vi.name or "image") + ".fits")).as_posix())
        self.fmt.currentIndexChanged.connect(self._changed)
        self.c_data.toggled.connect(self._changed)
        self.dest.textChanged.connect(self._ready)
        self.finished_export.connect(self._done)
        self._changed()

    def format_key(self) -> str:
        return self.fmt.currentData()

    def _ready(self, *_a):
        """After an export, any change of the format, the content or the file is the start of the next one: the window
        goes back to ready (without it _go would just close the window and nothing else could be written)."""
        if self.result_path is None:
            return
        self.result_path = None
        self.go.setText(_("Export"))
        self.cancel_b.setEnabled(True)
        self.status.setStyleSheet("")
        self.status.setText("")

    def _changed(self, *_a):
        self._ready()
        key = self.format_key()
        fits_ = key == "fits"
        self.c_view.setEnabled(not fits_)
        if fits_:
            self.c_data.setChecked(True)
        self.c_wcs.setEnabled(fits_ and hasattr(self.prep.wcs, "to_header"))
        self.c_wcs.setToolTip("" if fits_ else _("Only the FITS file has a header."))
        burn = not fits_ and not self.c_data.isChecked()
        self.ann_row.setEnabled(burn)
        self.ann_row.setToolTip("" if burn else _("Drawn into a picture of the stretched view only: a FITS file and "
                                                  "the linear data stay clean."))
        text = self.dest.text().strip()
        if text and Path(text).name:
            self.dest.setText(Path(text).with_suffix(FORMATS[key][1]).as_posix())
        if fits_:
            self.note.setText(_("FITS keeps the data as they are, 32-bit float: the display stretch is not applied."))
        elif self.c_data.isChecked():
            self.note.setText(_("The linear data: ADU as 16-bit integers (0–1 data over the whole range). They look "
                                "dark without a stretch.") if key != "jpeg" and key != "png8" else
                              _("The linear data in 8 bits: dark, and the faint levels are lost. For a picture, "
                                "export the stretched view."))
        else:
            self.note.setText(_("The image as the stretch shows it, computed at full resolution."))

    def _browse(self):
        """Picks the folder only: nothing is written before Export, so no "Save" dialog; the name stays in the field."""
        text = self.dest.text().strip()
        cur = Path(text) if text else None
        start = cur.parent.as_posix() if cur is not None and cur.name else (self.view.export_dir or str(Path.home()))
        d = QFileDialog.getExistingDirectory(self, _("Choose the folder"), start)
        if d:
            name = cur.name if cur is not None and cur.name else slug(self.vi.name or "image") + FORMATS[self.format_key()][1]
            self.dest.setText((Path(d) / name).as_posix())
            self.view.export_dir = Path(d).as_posix()                   # remembered like the folder of an export

    def options(self) -> dict:
        show = {"grid": self.c_grid.isChecked() and self.c_grid.isEnabled(),
                "objects": self.c_names.isChecked() and self.c_names.isEnabled()}
        burn = self.ann is not None and self.ann_row.isEnabled() and (show["grid"] or show["objects"])
        return {"path": self.dest.text().strip(), "fmt": self.format_key(),
                "content": "data" if self.c_data.isChecked() else "view",
                "area": self.area if self.a_vis.isChecked() else None,
                "with_wcs": self.c_wcs.isEnabled() and self.c_wcs.isChecked(),
                "annotations": (self.ann[0], show) if burn else None}

    def _go(self):
        if self.result_path is not None:
            self.accept()
            return
        o = self.options()
        if not o["path"]:
            self.status.setText(_("Choose a file."))
            return
        self.go.setEnabled(False)
        self.status.setStyleSheet("")
        self.status.setText(_("Writing…"))
        prep, st, vi = self.prep, self.st, self.vi

        def job():
            try:
                p = export_image(o["path"], prep, st, fmt=o["fmt"], content=o["content"], area=o["area"],
                                 with_wcs=o["with_wcs"], vi=vi, annotations=o["annotations"])
                self.finished_export.emit(str(p), "")
            except Exception as e:  # noqa: BLE001 - shown in the dialog
                try:
                    self.finished_export.emit(o["path"], f"{type(e).__name__}: {e}")
                except RuntimeError:
                    pass
        self._thread = threading.Thread(target=job, daemon=True)
        self._thread.start()

    def _done(self, path: str, error: str):
        self.go.setEnabled(True)
        if error:
            self.error = error
            self.status.setText(_("The export failed: {why}").format(why=plain_error(error)))
            log_detail(getattr(self.parent(), "s", None) or getattr(self.window(), "s", None), error)
            self.status.setStyleSheet(f"color: {WARN};")
            return
        self.result_path = Path(path)
        self.view.export_dir = self.result_path.parent.as_posix()      # the next export proposes the same folder
        self.status.setText(_("Written: {path}").format(path=path))
        self.go.setText(_("Close"))
        self.cancel_b.setEnabled(False)
        self.view.toasted.emit(_("Exported to {path}").format(path=path))

    def wait(self, timeout: float = 60.0) -> bool:
        end = time.monotonic() + timeout
        app = QApplication.instance()
        while time.monotonic() < end:
            app.processEvents()
            t = getattr(self, "_thread", None)
            if t is not None and not t.is_alive():
                app.processEvents()
                return True
            time.sleep(0.01)
        return False


# ---------------------------------------------------------------------------------------------- helpers
def sky_text(ra: float, dec: float) -> str:
    """RA and Dec as the readout writes them: RA 02h42m06.6s   Dec +61°18′01″ (a decimal comma in French)."""
    tenths = round((ra % 360.0) / 15.0 * 36000) % (24 * 36000)     # tenths of a second of time, rounded once
    h, rest = divmod(tenths, 36000)
    m, t = divmod(rest, 600)
    sec = t / 10
    arcsec = round(abs(dec) * 3600)
    dd, rest = divmod(arcsec, 3600)
    dm, ds = divmod(rest, 60)
    sign = "+" if dec >= 0 else "−"
    sec_txt = number(sec, 1).rjust(4, "0")
    return f"RA {h:02d}h{m:02d}m{sec_txt}s   Dec {sign}{dd:02d}°{dm:02d}′{ds:02d}″"


__all__ = ["ViewImage", "ImageView", "SyncGroup", "Loader", "default_loader", "Prepared", "prepare", "Display",
           "ImageCanvas", "HistogramView", "StretchPanel", "InfoBand", "FilmStrip", "ImageExportDialog",
           "export_image", "sky_text", "read_file", "FORMATS"]
