"""The viewer's canvas: the zoomable, pannable image (ImageCanvas) and the state two canvases share."""
from __future__ import annotations

import math
import time

import numpy as np
from PySide6.QtCore import QEvent, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen, QPolygonF
from PySide6.QtWidgets import QSizePolicy, QWidget

from .i18n import _
from .layout import pinch_factor, wheel_gesture
from .widgets import ACCENT, WARN
from .imageview_images import CANVAS_BG, ZOOM_STEP, Prepared
from .imageview_loading import Display, to_qimage, tag


def draw_split(p: QPainter, xs: float, height: float, left_name: str = "", right_name: str = ""):
    """The divider of a version compare, shared by the block editors' Split and Post's Compare: a shadowed line at ``xs``,
    its handle with two chevrons, and the two names as tags."""
    p.setRenderHint(QPainter.Antialiasing)
    p.setPen(QPen(QColor(0, 0, 0, 120), 4))
    p.drawLine(QPointF(xs, 0), QPointF(xs, height))
    p.setPen(QPen(QColor("#ffffff"), 1.5))
    p.drawLine(QPointF(xs, 0), QPointF(xs, height))
    c = QPointF(xs, height / 2)
    p.setBrush(QColor("#ffffff"))
    p.setPen(QPen(QColor(0, 0, 0, 140), 1))
    p.drawEllipse(c, 11, 11)
    p.setPen(QPen(QColor("#15171c"), 1.6))
    for sgn in (-1, 1):
        tri = QPolygonF([QPointF(xs + sgn * 3, c.y() - 4), QPointF(xs + sgn * 7, c.y()),
                         QPointF(xs + sgn * 3, c.y() + 4)])
        p.drawPolyline(tri)
    if left_name:
        tag(p, QPointF(xs - 10, 10), left_name, right=True, role="chevron_left")
    if right_name:
        tag(p, QPointF(xs + 10, 10), right_name, role="chevron_right")
    p.setRenderHint(QPainter.Antialiasing, False)


# ---------------------------------------------------------------------------------------------- the canvas
class ImageCanvas(QWidget):
    """The drawing surface: one image (and, in split mode, the image it is compared with), zoom and pan, the minimap,
    overlays. ``scale`` is in logical pixels per image pixel; (cx, cy) is the image point at the centre."""

    viewChanged = Signal()
    hovered = Signal(float, float, int)        # image coordinates, 0: the image, 1: the compared one (split)
    left = Signal()
    rectDone = Signal(float, float, float, float)
    splitMoved = Signal(float)
    zoomed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMinimumSize(240, 160)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.display: Display | None = None
        self.compare: Display | None = None        # shown left of the split line
        self.split = 0.5
        self.split_names = ("", "")
        self.title = ""                             # the name tag (side by side)
        self.message = ""                           # centred: loading, failure
        self.busy = False                           # a small "Loading…" badge over the image shown
        self.scale = 1.0
        self.cx = self.cy = 0.0
        self.fitted = True                          # follows the size of the widget
        self.overlays: list = []                    # callables (painter, canvas)
        self.minimap_on = True
        self.mode = "pan"                           # pan | rect
        #: an editing tool (the Gradient block's samples and areas): press(canvas, x, y, button) -> bool (taken),
        #: move(canvas, x, y), release(canvas), hover(canvas, x, y), cursor (a Qt cursor), in image coordinates
        self.tool = None
        self._space = False                         # Space held: the hand, whatever the tool (as in the reference image editor)
        self.frame_ms: list[float] = []             # paint times (the measurements)
        self._drag = None
        self._rect = None
        self._mouse: QPointF | None = None

    # ------------------------------------------------------------ what the overlays read
    @property
    def source(self) -> Prepared | None:
        return self.display.prep if self.display is not None else None

    def _tag(self, p, pt, text, right=False):
        tag(p, pt, text, right)

    # ------------------------------------------------------------ images
    def set_display(self, d: Display | None, keep_view: bool = False):
        same = (self.display is not None and d is not None and self.display.size() == d.size())
        self.display = d
        if d is not None and not (keep_view and same):
            self.fit()
        self.update()

    def set_compare(self, d: Display | None):
        self.compare = d
        self.update()

    def ratio(self) -> float:
        """Compared image pixels per image pixel (images of different sizes are matched by their width)."""
        if self.display is None or self.compare is None:
            return 1.0
        return self.compare.size()[0] / max(self.display.size()[0], 1)

    # ------------------------------------------------------------ the view
    def fit_scale(self) -> float:
        if self.display is None:
            return 1.0
        w, h = self.display.size()
        return max(min((self.width() - 16) / w, (self.height() - 16) / h), 1e-4)

    def min_scale(self) -> float:
        return min(self.fit_scale() * 0.5, 0.05)

    def max_scale(self) -> float:
        return 32.0 / self.devicePixelRatioF()

    def fit(self):
        if self.display is None:
            return
        w, h = self.display.size()
        self.scale = self.fit_scale()
        self.cx, self.cy = w / 2, h / 2
        self.fitted = True
        self.viewChanged.emit()
        self.zoomed.emit()
        self.update()

    def zoom_to(self, scale: float, cx: float | None = None, cy: float | None = None):
        self.scale = float(np.clip(scale, self.min_scale(), self.max_scale()))
        if cx is not None:
            self.cx, self.cy = float(cx), float(cy)
        self.fitted = False
        self._clamp()
        self.viewChanged.emit()
        self.zoomed.emit()
        self.update()

    def zoom_by(self, k: float, at: QPointF | None = None):
        """Zoom by ``k`` keeping the image point under ``at`` (default: the centre) in place."""
        at = at or QPointF(self.width() / 2, self.height() / 2)
        ix, iy = self.w_to_img(at.x(), at.y())
        s = float(np.clip(self.scale * k, self.min_scale(), self.max_scale()))
        self.scale = s
        self.cx = ix - (at.x() - self.width() / 2) / s
        self.cy = iy - (at.y() - self.height() / 2) / s
        self.fitted = False
        self._clamp()
        self.viewChanged.emit()
        self.zoomed.emit()
        self.update()

    def zoom_percent(self) -> float:
        """The zoom as an image editor says it: 100 % is one image pixel per screen pixel."""
        return 100.0 * self.scale * self.devicePixelRatioF()

    def set_zoom_percent(self, pct: float, at: QPointF | None = None):
        self.zoom_by(pct / 100.0 / self.devicePixelRatioF() / self.scale, at)

    def pan_by(self, dx: float, dy: float):
        """Move the image by (dx, dy) logical pixels."""
        self.cx -= dx / self.scale
        self.cy -= dy / self.scale
        self.fitted = False
        self._clamp()
        self.viewChanged.emit()
        self.update()

    def _clamp(self):
        if self.display is None:
            return
        w, h = self.display.size()
        mx, my = self.width() / 2 / self.scale, self.height() / 2 / self.scale
        # wider than the view: the edges stop at the borders; narrower: centred
        self.cx = float(np.clip(self.cx, mx, w - mx)) if w > 2 * mx else w / 2
        self.cy = float(np.clip(self.cy, my, h - my)) if h > 2 * my else h / 2

    def img_to_w(self, x, y) -> QPointF:
        return QPointF((x - self.cx) * self.scale + self.width() / 2, (y - self.cy) * self.scale + self.height() / 2)

    def w_to_img(self, px, py) -> tuple[float, float]:
        return (px - self.width() / 2) / self.scale + self.cx, (py - self.height() / 2) / self.scale + self.cy

    def visible(self) -> tuple[float, float, float, float]:
        """The visible part of the image (x0, y0, x1, y1), image pixels."""
        if self.display is None:
            return 0, 0, 0, 0
        w, h = self.display.size()
        x0, y0 = self.w_to_img(0, 0)
        x1, y1 = self.w_to_img(self.width(), self.height())
        return max(0.0, x0), max(0.0, y0), min(float(w), x1), min(float(h), y1)

    def whole_visible(self) -> bool:
        if self.display is None:
            return True
        w, h = self.display.size()
        x0, y0, x1, y1 = self.visible()
        return x0 <= 0.5 and y0 <= 0.5 and x1 >= w - 0.5 and y1 >= h - 0.5

    def resizeEvent(self, e):  # noqa: N802 - Qt's name
        if self.fitted:
            self.fit()
        super().resizeEvent(e)

    # ------------------------------------------------------------ drawing
    def paintEvent(self, e):  # noqa: N802
        t0 = time.perf_counter()
        p = QPainter(self)
        try:
            self._paint(p)
        finally:
            p.end()
        self.frame_ms.append(1000 * (time.perf_counter() - t0))
        if len(self.frame_ms) > 400:
            del self.frame_ms[:200]

    def _draw(self, p: QPainter, d: Display, s_dev: float, cx: float, cy: float, clip: QRectF | None = None):
        """One image over the whole area (the same resampling whatever the divider), drawn inside ``clip``."""
        dpr = self.devicePixelRatioF()
        W, H = max(1, round(self.width() * dpr)), max(1, round(self.height() * dpr))
        res = d.view(s_dev, cx, cy, W, H)
        if res is None:
            return
        buf, u0, v0 = res
        qi = to_qimage(buf)
        qi.setDevicePixelRatio(dpr)
        if clip is not None:
            p.save()
            p.setClipRect(clip)
        p.drawImage(QPointF(u0 / dpr, v0 / dpr), qi)
        if clip is not None:
            p.restore()

    def _paint(self, p: QPainter):
        p.fillRect(self.rect(), CANVAS_BG)
        d = self.display
        if d is not None:
            dpr = self.devicePixelRatioF()
            s_dev = self.scale * dpr
            if self.compare is not None:
                r = self.ratio()
                xs = round(self.width() * self.split * dpr) / dpr
                self._draw(p, self.compare, s_dev / r, self.cx * r, self.cy * r, QRectF(0, 0, xs, self.height()))
                self._draw(p, d, s_dev, self.cx, self.cy, QRectF(xs, 0, self.width() - xs, self.height()))
            else:
                self._draw(p, d, s_dev, self.cx, self.cy)
            if s_dev >= 16:
                self._pixel_grid(p)
            for ov in self.overlays:
                ov(p, self)
            if self.compare is not None:
                self._split_line(p)
            if self.minimap_on and not self.whole_visible():
                self._minimap(p)
            if self.title:
                tag(p, QPointF(10, 10), self.title)
            if self.busy:
                tag(p, QPointF(self.width() - 10, 10), _("Loading…"), right=True, color="#ffd28a")
        if self._rect:
            a, b = self._rect
            p.setPen(QPen(QColor(WARN), 1.5, Qt.DashLine))
            p.setBrush(QColor(230, 162, 60, 40))
            p.drawRect(QRectF(self.img_to_w(*a), self.img_to_w(*b)).normalized())
        if self.message:
            f = QFont(self.font())
            f.setPointSizeF(11)
            p.setFont(f)
            m = p.fontMetrics()
            w = min(m.horizontalAdvance(self.message) + 40, self.width() - 20)
            r = QRectF((self.width() - w) / 2, (self.height() - m.height() * 3) / 2, w, m.height() * 3)
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(28, 31, 38, 230))
            p.drawRoundedRect(r, 6, 6)
            p.setPen(QColor("#c9ced8"))
            p.drawText(r, Qt.AlignCenter | Qt.TextWordWrap, self.message)

    def _pixel_grid(self, p: QPainter):
        x0, y0, x1, y1 = self.visible()
        p.setPen(QPen(QColor(255, 255, 255, 28), 1))
        for x in range(int(x0), int(math.ceil(x1)) + 1):
            px = self.img_to_w(x, 0).x()
            p.drawLine(QPointF(px, self.img_to_w(0, y0).y()), QPointF(px, self.img_to_w(0, y1).y()))
        for y in range(int(y0), int(math.ceil(y1)) + 1):
            py = self.img_to_w(0, y).y()
            p.drawLine(QPointF(self.img_to_w(x0, 0).x(), py), QPointF(self.img_to_w(x1, 0).x(), py))

    def _split_line(self, p: QPainter):
        draw_split(p, self.width() * self.split, self.height(), *self.split_names)

    def minimap_rect(self) -> QRectF:
        w, h = self.display.size()
        mw, mh = 190.0, 130.0
        f = min(mw / w, mh / h)
        rw, rh = w * f, h * f
        return QRectF(self.width() - rw - 12, self.height() - rh - 12, rw, rh)

    def _minimap(self, p: QPainter):
        r = self.minimap_rect()
        dpr = self.devicePixelRatioF()
        th = self.display.thumbnail(int(max(r.width(), r.height()) * dpr))
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(15, 17, 21, 220))
        p.drawRoundedRect(r.adjusted(-4, -4, 4, 4), 5, 5)
        qi = to_qimage(th)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        p.drawImage(r, qi)
        w, h = self.display.size()
        x0, y0, x1, y1 = self.visible()
        f = r.width() / w
        v = QRectF(r.x() + x0 * f, r.y() + y0 * f, (x1 - x0) * f, (y1 - y0) * f)
        p.save()
        p.setClipRect(r)
        outside = QPainterPath()
        outside.addRect(r)
        inner = QPainterPath()
        inner.addRect(v)
        p.setBrush(QColor(0, 0, 0, 110))
        p.drawPath(outside.subtracted(inner))
        p.restore()
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(QColor(ACCENT), 1.5))
        p.drawRect(v)

    # ------------------------------------------------------------ interaction
    def _near_split(self, x: float) -> bool:
        return self.compare is not None and abs(x - self.width() * self.split) < 12

    def _in_minimap(self, pos: QPointF) -> bool:
        return (self.display is not None and self.minimap_on and not self.whole_visible()
                and self.minimap_rect().adjusted(-4, -4, 4, 4).contains(pos))

    def _minimap_goto(self, pos: QPointF):
        r = self.minimap_rect()
        w, _h = self.display.size()
        f = r.width() / w
        self.cx = (pos.x() - r.x()) / f
        self.cy = (pos.y() - r.y()) / f
        self.fitted = False
        self._clamp()
        self.viewChanged.emit()
        self.update()

    def wheelEvent(self, e):  # noqa: N802
        if self.display is None:
            return
        kind, a, b = wheel_gesture(e)                      # a touchpad's two fingers pan, its pinch (Ctrl) zooms (#44)
        if kind == "pan":
            self.pan_by(a, b)
        elif a:
            self.zoom_by(ZOOM_STEP ** a, e.position())
        e.accept()

    def mousePressEvent(self, e):  # noqa: N802
        pos = e.position()
        ix, iy = self.w_to_img(pos.x(), pos.y())
        if e.button() == Qt.LeftButton and self._near_split(pos.x()):
            self._drag = ("split",)
        elif e.button() == Qt.LeftButton and self._in_minimap(pos):
            self._drag = ("minimap",)
            self._minimap_goto(pos)
        elif e.button() == Qt.LeftButton and self.mode == "rect":
            self._rect = [(ix, iy), (ix, iy)]
            self._drag = ("rect",)
        elif (self.tool is not None and e.button() != Qt.MiddleButton and not self._space
              and not (e.button() == Qt.RightButton and getattr(self.tool, "mode", None) == "brush")
              and self.tool.press(self, ix, iy, e.button())):
            self._drag = ("tool",)
        elif e.button() in (Qt.LeftButton, Qt.MiddleButton) or (self.tool is not None and e.button() == Qt.RightButton):
            self._drag = ("pan", pos, self.cx, self.cy)
            self.setCursor(Qt.ClosedHandCursor)
        self.update()

    def mouseMoveEvent(self, e):  # noqa: N802
        pos = e.position()
        self._mouse = pos
        ix, iy = self.w_to_img(pos.x(), pos.y())
        d = self._drag
        if d is None:
            if self.compare is not None and pos.x() < self.width() * self.split:
                r = self.ratio()
                self.hovered.emit(ix * r, iy * r, 1)
            else:
                self.hovered.emit(ix, iy, 0)
            if (self.tool is not None and not self._space and not self._near_split(pos.x())
                    and not self._in_minimap(pos)):
                self.tool.hover(self, ix, iy)
                self.setCursor(self.tool.cursor)
                self.update()
                return
            self.setCursor(Qt.SplitHCursor if self._near_split(pos.x()) else
                           Qt.PointingHandCursor if self._in_minimap(pos) else
                           Qt.CrossCursor if self.mode == "rect" else Qt.CrossCursor)
            return
        if d[0] == "split":
            self.split = float(np.clip(pos.x() / max(self.width(), 1), 0.02, 0.98))
            self.splitMoved.emit(self.split)
        elif d[0] == "minimap":
            self._minimap_goto(pos)
        elif d[0] == "rect":
            self._rect[1] = (ix, iy)
        elif d[0] == "tool":
            self.tool.move(self, ix, iy)
        elif d[0] == "pan":
            self.cx = d[2] - (pos.x() - d[1].x()) / self.scale
            self.cy = d[3] - (pos.y() - d[1].y()) / self.scale
            self.fitted = False
            self._clamp()
            self.viewChanged.emit()
        self.update()

    def mouseReleaseEvent(self, e):  # noqa: N802
        d = self._drag
        self._drag = None
        self.setCursor(Qt.CrossCursor if self.tool is None else self.tool.cursor)
        if d and d[0] == "tool" and self.tool is not None:
            self.tool.release(self)
        if d and d[0] == "rect" and self._rect:
            (ax, ay), (bx, by) = self._rect
            self._rect = None
            if abs(bx - ax) * self.scale > 6 and abs(by - ay) * self.scale > 6:
                self.rectDone.emit(min(ax, bx), min(ay, by), max(ax, bx), max(ay, by))
        self.update()

    def event(self, e):
        # the keys of the tool on (Escape, Delete, M…) before Inspect's and the window's shortcuts
        if (e.type() == QEvent.ShortcutOverride and self.tool is not None and hasattr(self.tool, "wants")
                and self.tool.wants(e)):
            e.accept()
            return True
        # Ctrl+1 is the window's Graph page (its page keys win); 100 % is the key 1, in the View menu with its key
        f = pinch_factor(e)
        if f is not None and self.display is not None:   # a native pinch (macOS, Wayland)
            self.zoom_by(f, e.position())
            return True
        return super().event(e)

    def keyPressEvent(self, e):  # noqa: N802
        if self.tool is not None and e.key() != Qt.Key_Space and self.tool.key(e):
            return
        if e.key() == Qt.Key_Space and self.tool is not None and not e.isAutoRepeat():
            self._space = True
            self.setCursor(Qt.OpenHandCursor)
            return
        super().keyPressEvent(e)

    def keyReleaseEvent(self, e):  # noqa: N802
        if e.key() == Qt.Key_Space and not e.isAutoRepeat():
            self._space = False
            self.setCursor(self.tool.cursor if self.tool is not None else Qt.CrossCursor)
            return
        super().keyReleaseEvent(e)

    def mouseDoubleClickEvent(self, e):  # noqa: N802
        if self.display is None or self.mode != "pan" or self.tool is not None or self._near_split(e.position().x()):
            return
        if self.fitted or self.scale <= self.fit_scale() * 1.01:
            self.set_zoom_percent(100.0, e.position())
        else:
            self.fit()

    def leaveEvent(self, e):  # noqa: N802
        self._mouse = None
        self.left.emit()
        super().leaveEvent(e)


def _state_of(c: ImageCanvas) -> dict:
    if c.display is None:
        return {"fitted": True}
    w, h = c.display.size()
    return {"fitted": c.fitted, "zoom": c.scale * w, "fx": c.cx / w, "fy": c.cy / h}


def _mirror_state(c: ImageCanvas, state: dict):
    """Put a canvas where another one is (images of other sizes: the same part of the field)."""
    if c.display is None:
        return
    if state.get("fitted", True) or "zoom" not in state:
        c.fit()
        return
    w, h = c.display.size()
    c.zoom_to(state["zoom"] / w, state["fx"] * w, state["fy"] * h)
