"""The viewer's stretch controls: the histogram and the stretch panel."""
from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QFontMetricsF, QPainter, QPainterPath, QPen, QPolygonF
from PySide6.QtWidgets import (QButtonGroup, QCheckBox, QComboBox, QDoubleSpinBox, QFrame, QGridLayout, QHBoxLayout,
                               QLabel, QPushButton, QVBoxLayout, QWidget)

from . import stretch as sm
from .i18n import _, number
from .widgets import ACCENT, label
from .imageview_images import CHANNEL_COLORS, MONO_COLOR, MONO_FONT, ViewImage, Prepared, default_stretch


# ---------------------------------------------------------------------------------------------- the histogram
def nice_ticks(a0: float, a1: float) -> list[float]:
    """Round values (1, 2, 5 x 10^n, and 0) between a0 and a1."""
    span = max(abs(a0), abs(a1), 1e-12)
    top = int(math.floor(math.log10(span)))
    out = {0.0}
    for e in range(top - 6, top + 1):
        for m in (1, 2, 5):
            v = m * 10.0 ** e
            out.update((v, -v))
    return sorted(v for v in out if a0 <= v <= a1)


def short_number(v: float) -> str:
    """A tick label: 0, 0.002, 50, 1000, 20k (a decimal comma in French)."""
    a = abs(v)
    if a >= 10000:
        return number(v / 1000, 0) + "k"
    if a >= 1 or a == 0:
        return number(v, 0)
    d = max(0, -int(math.floor(math.log10(a))))
    return number(v, d)


def spaced_ticks(ticks: list[tuple[float, str]], width, left: float, right: float, gap: float = 8.0,
                 least: float = 34.0) -> list[tuple[float, str]]:
    """The tick labels (x, text) kept on an axis: inside [left, right], each at least ``least`` px from the previous
    one and never touching it (half widths plus ``gap``); ``width(text)``: its width in px. A fixed spacing let a dark's
    narrow range put "-100" against "100"."""
    out, last, last_half = [], -1e9, 0.0
    for x, text in ticks:
        half = width(text) / 2
        if x < left or x > right or x - last < max(least, last_half + half + gap):
            continue
        out.append((x, text))
        last, last_half = x, half
    return out


class HistogramView(QWidget):
    """The histogram of the data, with the black, mid and white handles and the curve of the stretch. The axis is a
    soft logarithm of the data (asinh): the sky, the faint nebula and the stars all get room, and the handles never
    pile up at the left as on a linear axis."""

    edited = Signal(str, float)                 # black | mid | white, value (after the neutral pre-scaling)
    released = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(220, 150)
        self.setMouseTracking(True)
        self.stats: sm.Stats | None = None
        self.st: sm.Stretch | None = None
        self.channel = -1
        self.units = ""
        self._axis = None
        self._bins = None
        self._drag: str | None = None
        self._hover: float | None = None

    def set_state(self, stats: sm.Stats | None, st: sm.Stretch | None, channel: int = -1, units: str = ""):
        if stats is not self.stats or st is None or self.st is None or not (
                np.array_equal(st.gain, self.st.gain) and np.array_equal(st.shift, self.st.shift)):
            self._axis = self._bins = None
        self.stats, self.st, self.channel, self.units = stats, st, channel, units
        self.update()

    # ------------------------------------------------------------ geometry
    def plot_rect(self) -> QRectF:
        return QRectF(8, 6, self.width() - 16, self.height() - 6 - 36)

    def axis(self):
        """(a0, a1, s, A): u = asinh((y - a0) / s) / A, y in the units of the pre-scaled data."""
        if self._axis is None:
            t = sm.transformed(self.stats, self.st.gain, self.st.shift)
            a0 = max(float(t["lo"].min()), float((t["median"] - 10 * t["sigma"]).min()))
            a1 = float(t["hi"].max())
            if not a1 > a0:
                a1 = a0 + 1e-6
            s = max(float(t["sigma"].min()) * 2, (a1 - a0) * 1e-6)
            self._axis = (a0, a1, s, math.asinh((a1 - a0) / s))
        return self._axis

    def u_of(self, y):
        a0, _a1, s, A = self.axis()
        return np.arcsinh((np.asarray(y, np.float64) - a0) / s) / A

    def y_of(self, u):
        a0, _a1, s, A = self.axis()
        return a0 + s * np.sinh(np.asarray(u, np.float64) * A)

    def px_of(self, y) -> float:
        r = self.plot_rect()
        return r.x() + float(self.u_of(y)) * r.width()

    def bins(self, n: int) -> np.ndarray:
        """(C, n) counts of the data along the axis."""
        if self._bins is None or self._bins.shape[1] != n:
            st, vals = self.st, self.stats.values
            out = np.zeros((self.stats.channels, n))
            for k in range(self.stats.channels):
                u = self.u_of(vals * st.gain[k] + st.shift[k])
                ok = (u >= 0) & (u < 1)
                idx = (u[ok] * n).astype(np.int64)
                out[k] = np.bincount(idx, weights=self.stats.hist[k][ok], minlength=n)[:n]
            self._bins = out
        return self._bins

    # ------------------------------------------------------------ drawing
    def paintEvent(self, e):  # noqa: N802
        p = QPainter(self)
        try:
            self._paint(p)
        finally:
            p.end()

    def _paint(self, p: QPainter):
        p.fillRect(self.rect(), QColor("#1c1f26"))
        r = self.plot_rect()
        p.fillRect(r, QColor("#12141a"))
        p.setPen(QPen(QColor("#2a2f38"), 1))
        p.drawRect(r)
        if self.stats is None or self.st is None:
            return
        p.setRenderHint(QPainter.Antialiasing)
        n = max(32, int(r.width()))
        b = self.bins(n)
        a0, a1, _s, _A = self.axis()
        # the ticks
        f = QFont(self.font())
        f.setPointSizeF(7.5)
        p.setFont(f)
        fm = QFontMetricsF(f)
        for x, text in spaced_ticks([(self.px_of(v), short_number(v)) for v in nice_ticks(a0, a1)],
                                    fm.horizontalAdvance, r.x() + 2, r.right() - 2):
            p.setPen(QPen(QColor(255, 255, 255, 18), 1))
            p.drawLine(QPointF(x, r.y()), QPointF(x, r.bottom()))
            p.setPen(QColor("#6b7280"))
            p.drawText(QRectF(x - 30, r.bottom() + 20, 60, 14), Qt.AlignHCenter | Qt.AlignTop, text)
        # the counts (logarithmic)
        top = float(np.log1p(b.max())) or 1.0
        chans = range(b.shape[0])
        cols = [MONO_COLOR] if b.shape[0] == 1 else list(CHANNEL_COLORS)
        order = [k for k in chans if k != self.channel] + ([self.channel] if self.channel >= 0 else [])
        for k in order:
            hk = np.log1p(b[k]) / top
            path = QPainterPath(QPointF(r.x(), r.bottom()))
            xs = r.x() + (np.arange(n) + 0.5) * r.width() / n
            ys = r.bottom() - hk * (r.height() - 4)
            for x, y in zip(xs, ys):
                path.lineTo(QPointF(float(x), float(y)))
            path.lineTo(QPointF(r.right(), r.bottom()))
            c = QColor(cols[k])
            faint = self.channel >= 0 and k != self.channel
            c.setAlpha(28 if faint else 70)
            p.setBrush(c)
            c2 = QColor(cols[k])
            c2.setAlpha(90 if faint else 230)
            p.setPen(QPen(c2, 1))
            p.drawPath(path)
        # the curve of the stretch
        k = self.channel if self.channel >= 0 else (1 if self.st.channels == 3 else 0)
        m = 160
        us = np.linspace(0, 1, m)
        ys = self.y_of(us)
        t = np.clip((ys - self.st.black[k]) / max(self.st.white[k] - self.st.black[k], 1e-30), 0, 1)
        if self.st.mode == "histogram":
            vals = self.stats.values
            curve = sm.curves(self.st, self.stats)[k]
            yv = vals * self.st.gain[k] + self.st.shift[k]
            out = np.interp(ys, yv, curve)
        else:
            out = np.clip(sm._curve(t, self.st, k), 0, 1)
        path = QPainterPath()
        for i, (u, o) in enumerate(zip(us, out)):
            pt = QPointF(r.x() + u * r.width(), r.bottom() - o * (r.height() - 4))
            path.moveTo(pt) if i == 0 else path.lineTo(pt)
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(QColor(255, 255, 255, 170), 1.3))
        p.drawPath(path)
        # the handles
        for which, col, style in (("black", "#0b0c0f", Qt.DashLine), ("mid", "#9aa3b2", Qt.DotLine),
                                  ("white", "#f0f2f5", Qt.DashLine)):
            if which == "mid" and self.st.mode in ("linear", "histogram"):
                continue
            x = self.px_of(self._value(which))
            if r.x() - 1 <= x <= r.right() + 1:
                p.setPen(QPen(QColor(255, 255, 255, 70), 1, style))
                p.drawLine(QPointF(x, r.y()), QPointF(x, r.bottom()))
            x = float(np.clip(x, r.x(), r.right()))
            y = r.bottom() + 3
            tri = QPolygonF([QPointF(x, y), QPointF(x - 6, y + 11), QPointF(x + 6, y + 11)])
            p.setBrush(QColor(col))
            p.setPen(QPen(QColor(ACCENT) if self._drag == which else QColor("#8b93a1"), 1.2))
            p.drawPolygon(tri)
        if self._hover is not None:
            f.setPointSizeF(8)
            p.setFont(f)
            v = float(self.y_of((self._hover - r.x()) / r.width()))
            txt = value_text(v, self.stats) + (" " + self.units if self.units else "")
            p.setPen(QColor("#c9ced8"))
            p.drawText(QRectF(r.x() + 4, r.y() + 2, r.width() - 8, 14), Qt.AlignRight | Qt.AlignTop, txt)

    def _value(self, which: str) -> float:
        arr = getattr(self.st, which)
        return float(arr[self.channel] if self.channel >= 0 else np.mean(arr))

    # ------------------------------------------------------------ interaction
    def _pick(self, x: float) -> str | None:
        handles = ["black", "white"] + ([] if self.st.mode in ("linear", "histogram") else ["mid"])
        r = self.plot_rect()
        pos = {h: float(np.clip(self.px_of(self._value(h)), r.x(), r.right())) for h in handles}
        best = min(handles, key=lambda h: (abs(pos[h] - x), 0 if h == "mid" else 1))
        if abs(pos[best] - x) > 12:
            return None
        if "mid" in pos and abs(pos["mid"] - pos["black"]) < 6 and best in ("mid", "black"):
            best = "black" if x < pos["black"] else "mid"
        return best

    def mousePressEvent(self, e):  # noqa: N802
        if self.st is None or e.button() != Qt.LeftButton:
            return
        self._drag = self._pick(e.position().x())
        self.update()

    def mouseMoveEvent(self, e):  # noqa: N802
        if self.st is None:
            return
        r = self.plot_rect()
        x = float(np.clip(e.position().x(), r.x(), r.right()))
        self._hover = x if r.adjusted(0, 0, 0, 20).contains(e.position()) else None
        if self._drag:
            self.edited.emit(self._drag, float(self.y_of((x - r.x()) / r.width())))
        else:
            self.setCursor(Qt.SizeHorCursor if self._pick(e.position().x()) else Qt.ArrowCursor)
        self.update()

    def mouseReleaseEvent(self, e):  # noqa: N802
        if self._drag:
            self._drag = None
            self.released.emit()
        self.update()

    def leaveEvent(self, e):  # noqa: N802
        self._hover = None
        self.update()
        super().leaveEvent(e)


def value_text(v: float, stats: sm.Stats | None = None, integer: bool = False) -> str:
    """A data value with as many decimals as its range needs (a decimal comma in French)."""
    if integer:
        return number(v, 0)
    span = (stats.hi - stats.lo) if stats is not None else max(abs(v), 1.0)
    d = int(np.clip(4 - math.floor(math.log10(max(span, 1e-12))), 0 if span >= 1e5 else 1, 6))
    return number(v, d)


# ---------------------------------------------------------------------------------------------- the stretch panel
class StretchPanel(QFrame):
    """The display stretch: Auto, the mode, linked channels, neutral sky, the channel edited, the histogram with its
    handles, the three points as numbers, Reset. Display only: ``changed`` gives the new ``Stretch``."""

    changed = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("ivpanel")
        self.setFixedWidth(312)
        self.prep: Prepared | None = None
        self.vi: ViewImage | None = None
        self.units = "adu"
        self.st: sm.Stretch | None = None
        self.channel = -1
        self._busy = False
        v = QVBoxLayout(self)
        v.setContentsMargins(12, 10, 12, 10)
        v.setSpacing(8)
        head = QHBoxLayout()
        head.addWidget(label(_("Stretch"), "h2"))
        head.addStretch(1)
        self.reset_b = QPushButton(_("Reset"))
        self.reset_b.setObjectName("link")
        self.reset_b.setToolTip(_("Back to the automatic stretch"))
        self.reset_b.clicked.connect(self.reset)
        head.addWidget(self.reset_b)
        v.addLayout(head)
        row = QHBoxLayout()
        self.auto_b = QPushButton(_("Auto"))
        self.auto_b.setCheckable(True)
        self.auto_b.setObjectName("toggle")
        self.auto_b.setToolTip(_("Black {k} σ under the sky, white at the 99.99th percentile, the sky at a quarter "
                                 "(A)").format(k=number(sm.SHADOWS, 1)))
        self.auto_b.toggled.connect(self._auto)
        row.addWidget(self.auto_b)
        self.mode_cb = QComboBox()
        for key, text in (("linear", _("Linear")), ("mtf", _("Midtones")), ("asinh", _("Asinh")),
                          ("histogram", _("Histogram equalisation"))):
            self.mode_cb.addItem(text, key)
        self.mode_cb.currentIndexChanged.connect(self._mode)
        row.addWidget(self.mode_cb, 1)
        v.addLayout(row)
        row = QHBoxLayout()
        self.linked_c = QCheckBox(_("Linked channels"))
        self.linked_c.setToolTip(_("The same black, mid and white for the three channels"))
        self.linked_c.toggled.connect(self._linked)
        row.addWidget(self.linked_c)
        self.neutral_c = QCheckBox(_("Neutral sky"))
        self.neutral_c.setToolTip(_("For an image still in the camera's colours: each channel scaled so that the sky "
                                    "is grey and the stars white (display only)"))
        self.neutral_c.toggled.connect(self._neutral)
        row.addWidget(self.neutral_c)
        row.addStretch(1)
        v.addLayout(row)
        self.chan_row = QWidget()
        cr = QHBoxLayout(self.chan_row)
        cr.setContentsMargins(0, 0, 0, 0)
        cr.setSpacing(0)
        self.chan_btns = []
        grp = QButtonGroup(self)
        for i, t in enumerate((_("RGB"), "R", "G", "B")):
            b = QPushButton(t)
            b.setCheckable(True)
            b.setObjectName("ivseg")
            b.clicked.connect(lambda _x=False, c=i - 1: self.set_channel(c))
            grp.addButton(b)
            cr.addWidget(b)
            self.chan_btns.append(b)
        cr.addStretch(1)
        v.addWidget(self.chan_row)
        self.hist = HistogramView()
        self.hist.edited.connect(self._edited)
        v.addWidget(self.hist)
        grid = QGridLayout()
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(4)
        self.fields = {}
        for i, (which, text) in enumerate((("black", _("Black")), ("mid", _("Mid")), ("white", _("White")))):
            grid.addWidget(label(text, "ivcap"), 0, i)
            f = QDoubleSpinBox()                 # with its arrows: without them the client's style sheet hides the text
            f.setObjectName("ivfield")
            f.setLocale(self.locale())
            f.setKeyboardTracking(False)
            f.valueChanged.connect(lambda val, w=which: self._field(w, val))
            grid.addWidget(f, 1, i)
            self.fields[which] = f
        v.addLayout(grid)
        self.clip_l = label("", "small")
        v.addWidget(self.clip_l)
        v.addSpacing(4)
        v.addWidget(label(_("Statistics"), "h2"))
        self.table = QGridLayout()
        self.table.setHorizontalSpacing(10)
        self.table.setVerticalSpacing(2)
        self.cells: dict[tuple[int, int], QLabel] = {}
        rows = (_("Minimum"), _("Median"), _("Noise σ"), _("99.99 %"), _("Maximum"))
        for r, t in enumerate(rows):
            self.table.addWidget(label(t, "small"), r + 1, 0)
        for c, (name, col) in enumerate(zip(("R", "G", "B"), CHANNEL_COLORS)):
            hd = QLabel(name)
            hd.setAlignment(Qt.AlignRight)
            hd.setStyleSheet(f"color: {col}; font-weight: 600; font-size: 9pt;")
            self.table.addWidget(hd, 0, c + 1)
            self.cells[(-1, c)] = hd
            for r in range(len(rows)):
                cell = QLabel("")
                cell.setAlignment(Qt.AlignRight)
                cell.setStyleSheet(f"color: #c9ced8; font-family: '{MONO_FONT}'; font-size: 8.5pt;")
                self.table.addWidget(cell, r + 1, c + 1)
                self.cells[(r, c)] = cell
        self.table.setColumnStretch(0, 1)
        v.addLayout(self.table)
        self.stats_l = label("", "small")
        self.stats_l.setWordWrap(True)
        v.addWidget(self.stats_l)
        note = label(_("Display only: the data are not changed."), "reason")
        note.setWordWrap(True)
        v.addWidget(note)
        v.addStretch(1)
        self.setEnabled(False)

    # ------------------------------------------------------------ state
    def set_image(self, prep: Prepared | None, st: sm.Stretch | None, vi: ViewImage | None = None):
        self.prep = prep
        self.vi = vi or (prep.vi if prep is not None else None)
        self.units = self.vi.units if self.vi is not None else "adu"
        if prep is None or st is None:
            self.st = None
            self.setEnabled(False)
            self.hist.set_state(None, None)
            return
        self.setEnabled(True)
        if prep.channels == 1:
            self.channel = -1
        self.auto_b.setToolTip(_(sm.auto_text(self.vi.kind if self.vi is not None else "")).format(
            k=number(sm.SHADOWS, 1)))
        self.set_stretch(st)

    def set_stretch(self, st: sm.Stretch):
        """Show a stretch (no signal)."""
        self.st = st.copy()
        self._busy = True
        try:
            colour = self.prep is not None and self.prep.channels == 3
            self.auto_b.setChecked(st.auto)
            self.mode_cb.setCurrentIndex(max(0, self.mode_cb.findData(st.mode)))
            self.linked_c.setChecked(st.linked)
            self.linked_c.setVisible(colour)
            self.neutral_c.setChecked(st.neutral)
            self.neutral_c.setVisible(colour)
            if st.linked or not colour:
                self.channel = -1
            self.chan_row.setVisible(colour and not st.linked)
            for i, b in enumerate(self.chan_btns):
                b.setChecked(i - 1 == self.channel)
            self._fill_fields()
        finally:
            self._busy = False
        self.hist.set_state(self.prep.stats if self.prep else None, self.st, self.channel, self._units())
        self._numbers()

    def _units(self) -> str:
        if self.prep is None:
            return ""
        return "ADU" if self.units == "adu" else ""

    def _fill_fields(self):
        stats = self.prep.stats
        span = stats.hi - stats.lo
        dec = int(np.clip(4 - math.floor(math.log10(max(span, 1e-12))), 1, 6))
        lo = min(stats.lo - span, float(np.min(self.st.black)))        # a model's black at zero shows as zero
        hi = max(stats.hi + span, float(np.max(self.st.white)))
        for which, f in self.fields.items():
            f.blockSignals(True)
            f.setDecimals(dec)
            f.setRange(lo, hi)
            f.setSingleStep(10 ** -dec * 10)
            arr = getattr(self.st, which)
            f.setValue(float(arr[self.channel] if self.channel >= 0 else np.mean(arr)))
            f.setEnabled(not (which == "mid" and self.st.mode in ("linear", "histogram")))
            f.blockSignals(False)

    def _numbers(self):
        if self.prep is None or self.st is None:
            return
        stats = self.prep.stats
        u = self._units()
        integer = self.prep.integer
        for c in range(3):
            shown = c < stats.channels
            hd = self.cells[(-1, c)]
            hd.setVisible(shown)
            if stats.channels == 1:
                hd.setText(_("Value"))
                hd.setStyleSheet(f"color: {MONO_COLOR}; font-weight: 600; font-size: 9pt;")
            else:                                         # a colour image after a mono one in the same viewer
                hd.setText("RGB"[c])
                hd.setStyleSheet(f"color: {CHANNEL_COLORS[c]}; font-weight: 600; font-size: 9pt;")
            for r, arr in enumerate((stats.vmin, stats.median, stats.sigma, stats.high, stats.vmax)):
                cell = self.cells[(r, c)]
                cell.setVisible(shown)
                if shown:
                    exact = integer and r in (0, 4)
                    cell.setText(value_text(float(arr[c]), None if exact else stats, exact))
        self.stats_l.setText(_("Data units: {u}").format(u=u) if u else
                             _("Data units: 0 to 1") if self.units == "norm" else "")
        lo_f, hi_f = sm.clipped(self.st, stats)
        self.clip_l.setText(_("Clipped: {b} % black · {w} % white").format(b=number(100 * lo_f, 2),
                                                                            w=number(100 * hi_f, 2)))

    def set_channel(self, c: int):
        self.channel = c
        for i, b in enumerate(self.chan_btns):
            b.setChecked(i - 1 == c)
        self._busy = True
        try:
            self._fill_fields()
        finally:
            self._busy = False
        self.hist.set_state(self.prep.stats, self.st, c, self._units())
        self._numbers()

    def _apply(self, st: sm.Stretch):
        self.set_stretch(st)
        self.changed.emit(st.copy())

    def _auto_for(self, mode=None, linked=None, neutral=None) -> sm.Stretch:
        """The automatic stretch of the image's kind, with the choices shown (or the ones given)."""
        st = self.st
        return sm.auto_for(self.prep.stats, self.vi.kind if self.vi is not None else "", mode or st.mode,
                           st.linked if linked is None else linked, st.neutral if neutral is None else neutral)

    # ------------------------------------------------------------ the controls
    def reset(self):
        if self.prep is None:
            return
        self.channel = -1
        self._apply(default_stretch(self.prep, vi=self.vi))

    def _auto(self, on: bool):
        if self._busy or self.st is None:
            return
        if on:
            self._apply(self._auto_for())
        else:
            st = self.st.copy()
            st.auto = False
            self._apply(st)

    def _mode(self, _i):
        if self._busy or self.st is None:
            return
        mode = self.mode_cb.currentData()
        if self.st.auto:
            self._apply(self._auto_for(mode=mode))
            return
        st = self.st.copy()
        st.mode = mode
        if mode in ("mtf", "asinh"):                      # a mid point that fits the new curve
            ref = self._auto_for(mode=mode)
            for k in range(st.channels):
                t = (st.mid[k] - st.black[k]) / max(st.white[k] - st.black[k], 1e-30)
                if not (0 < t < (0.5 if mode == "asinh" else 1)):
                    st.mid[k] = st.black[k] + (ref.mid[k] - ref.black[k]) / max(ref.white[k] - ref.black[k], 1e-30) \
                        * (st.white[k] - st.black[k])
        self._apply(st)

    def _linked(self, on: bool):
        if self._busy or self.st is None:
            return
        if self.st.auto:
            self._apply(self._auto_for(linked=on))
            return
        st = self.st.copy()
        st.linked = on
        if on:
            st.black[:] = st.black.mean()
            st.mid[:] = st.mid.mean()
            st.white[:] = st.white.max()
        self.channel = -1
        self._apply(st)

    def _neutral(self, on: bool):
        if self._busy or self.st is None:
            return
        self._apply(self._auto_for(neutral=on))

    def _edited(self, which: str, value: float):
        if self.st is None:
            return
        self._apply(sm.manual(self.st, which, value, self.channel, self.prep.stats))

    def _field(self, which: str, value: float):
        if self._busy or self.st is None:
            return
        self._apply(sm.manual(self.st, which, value, self.channel, self.prep.stats))
