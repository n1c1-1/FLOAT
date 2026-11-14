"""The scene's items, painted as today's graph page paints its blocks (block, folded stack card, wire, drop place,
band, Project block), from plain data. They hold no logic: the canvas tells them what to show."""
from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QFontMetrics, QFontMetricsF, QPainter, QPainterPath, QPainterPathStroker, QPen
from PySide6.QtWidgets import QGraphicsItem, QGraphicsPathItem, QStyleOptionGraphicsItem

from ..i18n import _
from .style import (
    ACCENT,
    BAD,
    BREAK_COL,
    BROKEN_COL,
    CUT_COL,
    GREY_COL,
    IMAGE_COL,
    OK,
    PAUSE_COL,
    REVIEWED_COL,
    STATE_COL,
    WARN,
    WIRE_COL,
    draw,
)

MIN_SCALE = 0.75
DIM = 0.55                     # the opacity of a block of another workflow while one is chosen
#: the mark of a block holding a setting unlike the same block of the other rows
OWN_MARK = "own"


def _font(size, bold=False):
    f = QFont("Segoe UI", size)
    f.setBold(bold)
    return f


def elide(text, font, width):
    return QFontMetrics(font).elidedText(text, Qt.ElideRight, max(int(width), 1))


def _rounded(r: QRectF) -> QPainterPath:
    p = QPainterPath()
    p.addRoundedRect(r, 7, 7)
    return p


def _band(p: QPainter, h: float, col: str):
    """The 4 px state band on the block's left edge (a rounded rectangle cut to its first 4 px)."""
    p.setPen(Qt.NoPen)
    p.setBrush(QColor(col))
    path = QPainterPath()
    path.addRoundedRect(QRectF(0, 0, 8, h), 4, 4)
    clip = QPainterPath()
    clip.addRect(QRectF(0, 0, 4, h))
    p.drawPath(path.intersected(clip))


BADGE_MAX = 150


def badge_text(text, maxw=BADGE_MAX):
    return QFontMetricsF(_font(7.5, True)).elidedText(text, Qt.ElideRight, maxw - 12)


def badge_width(text, maxw=BADGE_MAX) -> float:
    return QFontMetricsF(_font(7.5, True)).horizontalAdvance(badge_text(text, maxw)) + 12


def paint_badge(p: QPainter, x: float, y: float, text, maxw=BADGE_MAX):
    bw = badge_width(text, maxw)
    box = QRectF(x, y, bw, 15)
    p.setPen(QPen(QColor(ACCENT), 1.0))
    p.setBrush(QColor("#1f2a3f"))
    p.drawRoundedRect(box, 7, 7)
    p.setFont(_font(7.5, True))
    p.setPen(QColor(ACCENT))
    p.drawText(box, Qt.AlignCenter, badge_text(text, maxw))


def paint_review(p: QPainter, state: str):
    """The review dot on the top-left corner: red = the run stops here, yellow + bars = stopped, green tick = checked."""
    c = QPointF(2, 2)
    col = QColor({"pending": BREAK_COL, "paused": PAUSE_COL, "reviewed": REVIEWED_COL}[state])
    r = 9.5 if state == "paused" else 8.0
    disc = state != "reviewed"
    dark = QColor("#15171c")
    p.save()
    p.setPen(QPen(dark, 2.4))
    p.setBrush(col if disc else dark)
    p.drawEllipse(c, r, r)
    if not disc:
        p.setPen(QPen(col, 2.0))
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(c, r - 1.4, r - 1.4)
    if state == "paused":
        p.setPen(Qt.NoPen)
        p.setBrush(dark)
        p.drawRoundedRect(QRectF(c.x() - 4.4, c.y() - 4.6, 3.2, 9.2), 1.0, 1.0)
        p.drawRoundedRect(QRectF(c.x() + 1.2, c.y() - 4.6, 3.2, 9.2), 1.0, 1.0)
    elif state == "reviewed":
        p.setPen(QPen(col, 2.0, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        p.drawPolyline([QPointF(c.x() - 3.4, c.y() + 0.2), QPointF(c.x() - 1.0, c.y() + 2.8),
                        QPointF(c.x() + 3.6, c.y() - 2.6)])
    p.restore()


def _own_chip_width() -> float:
    return QFontMetricsF(_font(7.5)).horizontalAdvance(_(OWN_MARK)) + 12


def _paint_own_chip(p: QPainter, box: QRectF):
    p.setPen(QPen(QColor(WARN), 1.0))
    p.setBrush(Qt.NoBrush)
    p.drawRoundedRect(box, 7, 7)
    p.setFont(_font(7.5))
    p.setPen(QColor(WARN))
    p.drawText(box, Qt.AlignCenter, _(OWN_MARK))


class BlockItem(QGraphicsItem):
    """A block, as the client paints it. ``d`` is a dict filled by the canvas from the core: id, kind, title, summary,
    state, selected, off, grey, badge, review (None | pending | paused | reviewed), own (a setting unlike the other
    rows), w, h, x, y, inputs [(name, label, y, wired, colour)], outputs [(port, y, colour, wired)], removable, tip."""

    def __init__(self, d: dict):
        super().__init__()
        self.d = d
        self.w, self.h = d["w"], d["h"]
        self.id = d["id"]
        self.hover = False
        self.hot_input = None          # (port name, acceptable): the input under a pulled wire
        self.lit = {}                  # port name -> True (acceptable) / False (refused): lit while a wire is pulled
        self.hot_output = False
        self.target = False            # the Post block while a wire is pulled over it (acceptable)
        self.target_refused = False
        self.dragged = False           # being dragged: shown faint where it stands
        self.alert = False             # a deletion that cannot be made: red frame for 1.5 s on what depends on it
        self.wire_hot = False          # the pointer is on a wire that starts or ends here
        self.name_w = max((QFontMetricsF(_font(8.5)).horizontalAdvance(i[1]) for i in d["inputs"]), default=0) + \
            (14 if d["inputs"] else 0)
        self.setAcceptHoverEvents(True)
        self.setPos(d["x"], d["y"])
        self.setZValue(2)
        self.setToolTip(d.get("tip", ""))

    # -- geometry used by the canvas' hit tests (item coordinates)
    def boundingRect(self):                                           # noqa: N802
        return QRectF(-12, -12, self.w + 24, self.h + 24)

    def shape(self):                                                  # noqa: D102 - what a click or a hover finds
        """The body, its ports, its named inputs and its review dot: the margin around it is not the block's, so a
        chain wire in the gap between two blocks is found there (clicked, hovered: it says it cannot be removed)."""
        path = QPainterPath()
        path.addRect(QRectF(0, 0, self.w, self.h))
        for _port, y, _c, _wired in self.d["outputs"]:
            path.addEllipse(QPointF(self.w, y), 12, 12)
        for _name, _lab, y, _wired, _c in self.d["inputs"]:
            path.addRect(QRectF(-9, y - 9, self.name_w + 18, 18))
        if self.d.get("review"):
            path.addEllipse(QPointF(2, 2), 10, 10)
        return path

    def cross_rect(self):
        return QRectF(self.w - 20, 4, 16, 16)

    def input_at(self, pos):
        for name, _lab, y, _w, _c in self.d["inputs"]:
            if QRectF(-9, y - 9, self.name_w + 18, 18).contains(pos):
                return name
        return None

    def output_at(self, pos):
        for port, y, _c, _w in self.d["outputs"]:
            if (pos - QPointF(self.w, y)).manhattanLength() < (10 if len(self.d["outputs"]) > 1 else 12):
                return port
        return None

    def paint(self, p: QPainter, opt, widget=None):
        d = self.d
        p.setRenderHint(QPainter.Antialiasing)
        off, grey, sel = d["off"], d["grey"], d["selected"]
        # a block of a workflow not through the chosen block: dimmed (graph.py ``dim``)
        p.setOpacity((0.25 if self.dragged else 0.55 if off else 1.0) * (DIM if d.get("dim") else 1.0))
        w, h = self.w, self.h
        r = QRectF(0, 0, w, h)
        hot = sel or self.target
        fill = "#2b3446" if hot else ("#272c36" if self.hover else "#22262e")
        paused = d["review"] == "paused"
        p.setPen(QPen(QColor(ACCENT if hot else (PAUSE_COL if paused else "#3a4150")), 2 if hot or paused else 1))
        p.setBrush(QColor(fill))
        p.drawRoundedRect(r, 7, 7)
        if off:                                          # switched off: hatched, the wire drawn across it
            p.save()
            p.setClipPath(_rounded(r))
            p.setPen(QPen(QColor("#3a4150"), 1.0))
            for x in range(-int(h), int(w) + 1, 9):
                p.drawLine(QPointF(x, h), QPointF(x + h, 0))
            p.restore()
            p.setPen(QPen(QColor("#9aa3b2"), 1.8))
            p.drawLine(QPointF(0, h / 2), QPointF(w, h / 2))
        broken, missing = d.get("broken"), d.get("missing")
        fallback, warned = bool(d.get("fallback")), bool(d.get("warned"))
        if grey or missing:                              # greyed, or its sky data not on this computer: amber dashed
            p.setPen(QPen(QColor(GREY_COL), 1.4, Qt.DashLine))
            p.setBrush(Qt.NoBrush)
            p.drawRoundedRect(r.adjusted(1, 1, -1, -1), 7, 7)
        if broken:                                       # it cannot run at all: circled in red, Run refused
            p.setPen(QPen(QColor(BROKEN_COL), 2.2))
            p.setBrush(Qt.NoBrush)
            p.drawRoundedRect(r.adjusted(-2, -2, 2, 2), 8, 8)
        if self.target_refused:
            p.setPen(QPen(QColor(BAD), 2.0))
            p.setBrush(Qt.NoBrush)
            p.drawRoundedRect(r.adjusted(-1, -1, 1, 1), 8, 8)
        if self.wire_hot:
            p.setPen(QPen(QColor(ACCENT), 2.0))
            p.setBrush(Qt.NoBrush)
            p.drawRoundedRect(r.adjusted(-2, -2, 2, 2), 9, 9)
        if self.alert:
            p.setPen(QPen(QColor(BAD), 3.0))
            p.setBrush(Qt.NoBrush)
            p.drawRoundedRect(r.adjusted(-3, -3, 3, 3), 10, 10)
        state = d["state"]
        if d["kind"] not in ("session", "post") or state != "pending":
            _band(p, h, CUT_COL if off else GREY_COL if grey else
                  STATE_COL["fallback"] if fallback else STATE_COL.get(state, "#5d6470"))
        f = _font(10, True)
        p.setFont(f)
        p.setPen(QColor("#e8ebf0"))
        mark = ("failed", "error") if broken else ("off", "dim") if off else \
            ("not_computable", "warning") if grey or missing else ("warning", "warning") if fallback else None
        x0 = 11 + (20 if mark else 0) + self.name_w
        tw = w - 7 - x0 - (18 if d["removable"] and self.hover else 0)
        readable = QStyleOptionGraphicsItem.levelOfDetailFromTransform(p.worldTransform()) >= MIN_SCALE - 1e-6
        bw = badge_width(d["badge"]) if d["badge"] else 0
        fm = QFontMetrics(f)
        inline = bool(bw) and fm.horizontalAdvance(d["title"]) + 6 + bw <= tw       # the badge fits beside the name
        below = bool(bw) and not inline and readable                                 # else it takes the 2nd line
        shown = fm.elidedText(d["title"], Qt.ElideRight, int(max(tw - (bw + 6 if inline else 0), 8)))
        p.drawText(QRectF(x0, 6, tw, 20) if readable else QRectF(x0, 0, tw, h), Qt.AlignLeft | Qt.AlignVCenter, shown)
        if inline:
            paint_badge(p, x0 + fm.horizontalAdvance(shown) + 6, 9 if readable else (h - 15) / 2, d["badge"])
        elif below:
            paint_badge(p, x0, 27, d["badge"], maxw=w - 6 - x0)
        if mark:
            draw(p, QRectF(11 + self.name_w, 8 if readable else (h - 16) / 2, 16, 16), mark[0], mark[1])
        if readable and not below:
            f2 = _font(8.5)
            p.setFont(f2)
            p.setPen(QColor("#8b93a1"))
            sx = 11 + self.name_w + (20 if warned else 0)  # a warning on its files: its icon heads the summary
            room = w - 4 - sx
            p.drawText(QRectF(sx, 28, room, 18), Qt.AlignLeft | Qt.AlignVCenter, elide(d["summary"], f2, room))
            if warned:
                draw(p, QRectF(11 + self.name_w, 29, 16, 16), "warning", "warning")
        if d["own"]:                                     # a setting unlike the other rows
            cw = _own_chip_width()
            _paint_own_chip(p, QRectF(w - 8 - cw - (20 if d["removable"] and self.hover else 0), 8, cw, 15))
        if d["removable"] and self.hover:
            draw(p, self.cross_rect(), "remove", "error")
        if d["inputs"]:                                  # the named inputs: a hollow dot while empty, the name whole
            f3 = _font(8.5)
            for name, lab, y, wired, colour in d["inputs"]:
                col = QColor(colour)
                hi = self.hot_input is not None and self.hot_input[0] == name
                st = self.lit.get(name)
                ok = hi and self.hot_input[1]
                red = (hi and not ok) or (st is False and not hi)
                p.setPen(QPen(QColor(BAD if red else ACCENT if (hi or st) else col), 1.6 if not st else 2.0))
                p.setBrush(QColor(ACCENT if ok else col) if wired or ok else QColor("#15171c"))
                rr = 6.5 if hi or st is not None else 4.5
                p.drawEllipse(QPointF(0, y), rr, rr)
                p.setFont(f3)
                p.setPen(QColor("#c9d1dc" if wired else "#8b93a1"))
                p.drawText(QRectF(9, y - 9, self.name_w, 18), Qt.AlignLeft | Qt.AlignVCenter, lab)
        for port, y, colour, wired in d["outputs"]:
            big = self.hover or sel or self.hot_output
            filled = big or (wired and port != "image")      # an image exit is hollow, the stars' two ports coloured
            p.setPen(QPen(QColor(colour if filled else "#4a5263"), 1.5))
            p.setBrush(QColor(colour if filled else "#15171c"))
            rr = 6.5 if big else 4.5
            p.drawEllipse(QPointF(w, y), rr, rr)
        if d["review"]:
            paint_review(p, d["review"])

    def hoverEnterEvent(self, e):                                     # noqa: N802
        self.hover = True
        self._grab_cursor(e.pos())
        self.update()

    def hoverMoveEvent(self, e):                                      # noqa: N802
        self._grab_cursor(e.pos())

    def hoverLeaveEvent(self, e):                                     # noqa: N802
        self.hover = False
        self.unsetCursor()
        self.update()

    def _grab_cursor(self, pos):
        """The open hand over an output port (the wire is grabbed there), the arrow on the body (today's page)."""
        if self.d["kind"] != "card" and self.output_at(pos) is not None:     # a card's port is not pulled
            if self.cursor().shape() != Qt.OpenHandCursor:
                self.setCursor(Qt.OpenHandCursor)
        elif self.hasCursor():
            self.unsetCursor()


class StackItem(BlockItem):
    """A row folded into one card: the row's name and badge, "3 sessions · 78 lights · drizzle auto", four dots
    (Calibration, Registration, Selection, Stacking) in their states. Folding is global (the PREPROCESSING control).
    ``d`` adds: dots [(state, tip)]."""
    R, GAP = 4.5, 6.5

    def centres(self):
        n = len(self.d["dots"])
        width = n * 2 * self.R + max(0, n - 1) * self.GAP
        x = self.w - 12 - width + self.R
        return [QPointF(x + i * (2 * self.R + self.GAP), 37.5) for i in range(n)]

    def dot_at(self, pos):
        for i, c in enumerate(self.centres()):
            if (pos - c).manhattanLength() <= 2 * self.R + 3:
                return i
        return None

    def paint(self, p, opt, widget=None):
        d = self.d
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.w, self.h
        r = QRectF(0, 0, w, h)
        sel = d["selected"]
        p.setOpacity(DIM if d.get("dim") else 1.0)
        fill = "#2b3446" if sel else ("#272c36" if self.hover else "#22262e")
        p.setPen(QPen(QColor(ACCENT if sel else "#3a4150"), 2 if sel else 1))
        p.setBrush(QColor("#1b1e25"))                    # the stack: two sheets behind the card
        p.drawRoundedRect(QRectF(5, 5, w - 2, h - 2), 7, 7)
        p.setBrush(QColor("#1f232b"))
        p.drawRoundedRect(QRectF(2.5, 2.5, w - 1, h - 1), 7, 7)
        p.setBrush(QColor(fill))
        p.drawRoundedRect(r, 7, 7)
        if self.alert:
            p.setPen(QPen(QColor(BAD), 3.0))
            p.setBrush(Qt.NoBrush)
            p.drawRoundedRect(r.adjusted(-3, -3, 3, 3), 10, 10)
        if self.wire_hot:
            p.setPen(QPen(QColor(ACCENT), 2.0))
            p.setBrush(Qt.NoBrush)
            p.drawRoundedRect(r.adjusted(-2, -2, 2, 2), 9, 9)
        _band(p, h, STATE_COL.get(d["state"], "#5d6470"))
        f = _font(10, True)
        p.setFont(f)
        p.setPen(QColor("#e8ebf0"))
        chip = _own_chip_width() if d["own"] else 0
        bw = badge_width(d["badge"]) if d["badge"] else 0
        tw = w - 7 - 11 - 8 - (chip + 4 if chip else 0)
        readable = QStyleOptionGraphicsItem.levelOfDetailFromTransform(p.worldTransform()) >= MIN_SCALE - 1e-6
        shown = QFontMetrics(f).elidedText(d["title"], Qt.ElideRight, int(tw - (bw + 6 if bw else 0)))
        p.drawText(QRectF(11, 6, tw, 20) if readable else QRectF(11, 0, tw, h), Qt.AlignLeft | Qt.AlignVCenter, shown)
        if bw:
            paint_badge(p, 11 + QFontMetrics(f).horizontalAdvance(shown) + 6, 9 if readable else (h - 15) / 2,
                        d["badge"])
        if chip:
            _paint_own_chip(p, QRectF(w - 8 - chip, 8, chip, 15))
        cs = self.centres()
        dots_w = (w - 8 - (cs[0].x() - self.R)) if cs else 0
        if readable:
            f2 = _font(8.5)
            p.setFont(f2)
            p.setPen(QColor("#8b93a1"))
            room = w - 11 - dots_w - 8 - 4
            p.drawText(QRectF(11, 28, room, 18), Qt.AlignLeft | Qt.AlignVCenter, elide(d["summary"], f2, room))
        for c, (state, _tip) in zip(cs, d["dots"]):
            col = QColor(STATE_COL.get(state, "#5d6470"))
            if state == "running":
                ring = QColor(ACCENT)
                ring.setAlphaF(0.7)
                p.setPen(QPen(ring, 1.8))
                p.setBrush(Qt.NoBrush)
                p.drawEllipse(c, self.R + 3.0, self.R + 3.0)
            p.setPen(QPen(QColor("#15171c"), 1.0))
            p.setBrush(col)
            p.drawEllipse(c, self.R, self.R)
        for _port, y, colour, _wired in d["outputs"]:
            big = self.hover or sel or self.hot_output
            p.setPen(QPen(QColor(colour), 1.5))
            p.setBrush(QColor(colour))
            rr = 6.5 if big else 4.5
            p.drawEllipse(QPointF(w, y), rr, rr)
        if d["review"]:
            paint_review(p, d["review"])


def _dist(a, b):
    return ((a.x() - b.x()) ** 2 + (a.y() - b.y()) ** 2) ** 0.5


class WireItem(QGraphicsPathItem):
    """A wire from ``Path.points``, corners rounded at 8 px. Port wires are selectable (a click, Del removes them); the
    others are the chain itself and say so. Under the pointer a wire lights itself whole, on top, with its two ends and
    the two blocks it joins (``hover_cb``)."""

    def __init__(self, ident, points, style_name, colour, selectable, tip=""):
        super().__init__()
        self.ident, self.style_name, self.selectable = ident, style_name, selectable
        path = QPainterPath(QPointF(*points[0]))
        n = len(points)
        for i in range(1, n):
            cur = QPointF(*points[i])
            if i < n - 1:
                prev, nxt = QPointF(*points[i - 1]), QPointF(*points[i + 1])
                a = min(8.0, _dist(prev, cur) / 2, _dist(cur, nxt) / 2)
                if a > 0.5 and _dist(prev, cur) > 0 and _dist(cur, nxt) > 0:
                    before = cur + (prev - cur) * (a / _dist(prev, cur))
                    after = cur + (nxt - cur) * (a / _dist(cur, nxt))
                    path.lineTo(before)
                    path.quadTo(cur, after)
                    continue
            path.lineTo(cur)
        self.setPath(path)
        col = QColor(IMAGE_COL.get(colour) or WIRE_COL)
        width = 3.0 if style_name == "bar" else 2.2
        pen = QPen(col, width, Qt.DashLine if style_name == "port_inactive" else Qt.SolidLine)
        pen.setJoinStyle(Qt.RoundJoin)
        pen.setCapStyle(Qt.FlatCap)
        self.base_pen = QPen(pen)
        self.setPen(pen)
        self.selected = False
        self.setAcceptHoverEvents(True)
        self.setZValue(1)
        self.setToolTip(tip)
        self.tipfn = None
        self.hover_cb = None           # (wire id, on): the canvas lights the two blocks the wire joins
        self.ends = None
        self.hot = False
        self.end_points = (QPointF(*points[0]), QPointF(*points[-1]))

    def boundingRect(self):                                           # noqa: N802
        return super().boundingRect().adjusted(-8, -8, 8, 8)

    def paint(self, p, opt, widget=None):
        super().paint(p, opt, widget)
        if self.hot:                                     # the two ends of the wire under the pointer
            p.setPen(QPen(QColor("#15171c"), 1.5))
            p.setBrush(QColor(ACCENT))
            for q in self.end_points:
                p.drawEllipse(q, 5.5, 5.5)

    def shape(self):                                                  # noqa: D102 - a wide band to hit
        st = QPainterPathStroker()
        st.setWidth(14)
        return st.createStroke(self.path())

    def _look(self, hot):
        pen = QPen(self.base_pen)
        if hot or self.selected:
            pen.setColor(QColor(ACCENT))
            pen.setWidthF(self.base_pen.widthF() + 1.2)
        self.setPen(pen)

    def hoverEnterEvent(self, e):                                     # noqa: N802
        if self.tipfn is not None:
            self.setToolTip(self.tipfn())
            self.tipfn = None
        self.light(True)

    def light(self, on):
        """A wire under the pointer shows itself whole, on top, with its two ends and the two blocks it joins."""
        self.hot = on
        self.setZValue(5 if on else 1)
        self._look(on)
        self.update()
        if self.hover_cb is not None:
            self.hover_cb(self.ident, on)

    def hoverLeaveEvent(self, e):                                     # noqa: N802
        self.light(False)

    def mark(self, sel):
        self.selected = sel
        self._look(False)


class RubberWire(QGraphicsPathItem):
    """The wire being pulled from an output port to the cursor."""

    def __init__(self):
        super().__init__()
        self.setPen(QPen(QColor(ACCENT), 2.0, Qt.DashLine))
        self.setZValue(8)

    def stretch(self, a: QPointF, b: QPointF):
        path = QPainterPath(a)
        mx = (a.x() + b.x()) / 2
        path.cubicTo(QPointF(mx, a.y()), QPointF(mx, b.y()), b)
        self.setPath(path)


class GhostItem(QGraphicsItem):
    """A place of a drag, drawn over the unchanged drawing. ``place`` is the core's Place (or any object with kind,
    rect, lit, reason, tip). The hot one is outlined in blue and says its tip; a refused one is dotted and says why."""
    TEXT = {"row": "Workflow: Session/Image", "night": "Session"}

    def __init__(self, place, text=None):
        super().__init__()
        self.place = place
        b = place.rect
        self.rect = QRectF(b.x, b.y, b.w, b.h)
        self.hot = False
        self.text = text if text is not None else _(self.TEXT.get(place.kind, "")) if place.kind in self.TEXT else ""
        self.setZValue(5)
        self.setAcceptHoverEvents(False)

    def boundingRect(self):                                           # noqa: N802
        return self.rect.adjusted(-4, -4, 4, 4)

    def paint(self, p, opt, widget=None):
        p.setRenderHint(QPainter.Antialiasing)
        pl, r = self.place, self.rect
        caret = pl.kind == "insert"
        if not pl.lit:
            p.setOpacity(1.0 if self.hot else 0.55)
            p.setPen(QPen(QColor("#6b7383" if not self.hot else "#8a93a3"), 1.2, Qt.DotLine))
            p.setBrush(Qt.NoBrush)
            p.drawRoundedRect(r, 3 if caret else 7, 3 if caret else 7)
            if not caret and r.height() > 14 and not self.hot:
                p.setFont(_font(9))
                p.setPen(QColor("#6b7383"))
                p.drawText(r, Qt.AlignCenter, self.text)
            return
        if caret:
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(ACCENT if self.hot else "#4a5263"))
            rr = r.adjusted(-1.5, -2, 1.5, 2) if self.hot else r
            p.drawRoundedRect(rr, 3, 3)
            return
        p.setPen(QPen(QColor(ACCENT if self.hot else "#4a5263"), 1.6 if self.hot else 1.2, Qt.DashLine))
        p.setBrush(QColor("#1d2633") if self.hot else Qt.NoBrush)
        p.drawRoundedRect(r, 7, 7)
        f = _font(9)
        p.setFont(f)
        p.setPen(QColor(ACCENT if self.hot else "#8b93a1"))
        small = r.height() <= 24
        role = {"branch": "branch", "night": "add", "row": "add", "port": "add"}.get(pl.kind)
        txt = self.text
        if small and pl.kind == "branch":                # the strip: its icon (the tip says the rest when hot)
            draw(p, QRectF(r.center().x() - 8, r.center().y() - 8, 16, 16), role, "selected" if self.hot else "dim")
            return
        tw = QFontMetricsF(f).horizontalAdvance(txt)
        if role:
            x = r.center().x() - (tw + 20) / 2
            draw(p, QRectF(x, r.center().y() - 8, 16, 16), role, "selected" if self.hot else "dim")
            p.drawText(QRectF(x + 20, r.y(), tw + 2, r.height()), Qt.AlignVCenter | Qt.AlignLeft, txt)
        else:
            p.drawText(r, Qt.AlignCenter, elide(txt, f, r.width() - 8))


class TipItem(QGraphicsItem):
    """The bubble that tells what the hot place does (its tip) or why it is refused (its reason)."""

    def __init__(self):
        super().__init__()
        self.text = ""
        self.anchor = QRectF()
        self.refused = False
        self.rect = QRectF()
        self.setZValue(9)
        self.hide()

    def set(self, text, anchor: QRectF, refused: bool, below=False):
        self.prepareGeometryChange()
        self.text, self.anchor, self.refused = text, anchor, refused
        fm = QFontMetricsF(_font(9))
        w = min(fm.horizontalAdvance(text) + 18, 420)
        lines = max(1, int(fm.horizontalAdvance(text) // 400) + 1)
        h = 12 + lines * fm.height()
        x = anchor.center().x() - w / 2
        y = anchor.top() - h - 8
        if below or y < -40:
            y = anchor.bottom() + 8
        self.rect = QRectF(x, y, w, h)
        self.show()
        self.update()

    def boundingRect(self):                                           # noqa: N802
        return self.rect.adjusted(-2, -2, 2, 2)

    def paint(self, p, opt, widget=None):
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(QPen(QColor(BAD if self.refused else ACCENT), 1.2))
        p.setBrush(QColor("#262a33"))
        p.drawRoundedRect(self.rect, 5, 5)
        p.setFont(_font(9))
        p.setPen(QColor("#e8ebf0"))
        p.drawText(self.rect.adjusted(9, 4, -9, -4), Qt.AlignCenter | Qt.TextWordWrap, self.text)


class FileTarget(QGraphicsItem):
    """One of the five labelled targets of a file drag over a Session block."""

    def __init__(self, rect: QRectF, text: str, kind: str):
        super().__init__()
        self.rect, self.text, self.kind = rect, text, kind
        self.hot = False
        self.setZValue(7)

    def boundingRect(self):                                           # noqa: N802
        return self.rect.adjusted(-2, -2, 2, 2)

    def paint(self, p, opt, widget=None):
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(QPen(QColor(ACCENT if self.hot else "#4a5263"), 1.6 if self.hot else 1.2, Qt.DashLine))
        p.setBrush(QColor("#1d2633") if self.hot else QColor("#191c22"))
        p.drawRoundedRect(self.rect, 6, 6)
        p.setFont(_font(9, self.hot))
        p.setPen(QColor(ACCENT if self.hot else "#c9cfd8"))
        p.drawText(self.rect, Qt.AlignCenter, self.text)


class BandItem(QGraphicsItem):
    """A section of the page (PREPROCESSING, LINEAR PROCESSING, COMPOSITION), its title high in the band."""

    def __init__(self, rect: QRectF, text):
        super().__init__()
        self.rect, self.text = rect, text
        self.setZValue(-10)

    def boundingRect(self):                                           # noqa: N802
        return self.rect

    def _title(self, rect=None):
        f = _font(8.5, True)
        r = (rect or self.rect).adjusted(12, 2, -8, 0)
        fm = QFontMetricsF(f)
        words = [w if fm.horizontalAdvance(w) <= r.width() else fm.elidedText(w, Qt.ElideRight, r.width())
                 for w in self.text.upper().split()]
        return f, r, " ".join(words)

    def title_rect(self, rect=None) -> QRectF:
        """Where its title is written (on one line or two, as the band's width wraps it), in scene coordinates."""
        f, r, text = self._title(rect)
        used = QFontMetricsF(f).boundingRect(QRectF(r.x(), r.y(), r.width(), 1e4),
                                             int(Qt.AlignLeft | Qt.AlignTop | Qt.TextWordWrap), text)
        return QRectF(r.x(), r.y(), used.width(), used.height())

    def paint(self, p, opt, widget=None):
        p.setPen(Qt.NoPen)
        p.setBrush(QColor("#191c22"))
        p.drawRoundedRect(self.rect, 10, 10)
        f, r, text = self._title()                        # high in the band: the first row's caption sits right under
        p.setFont(f)
        p.setPen(QColor("#6b7280"))
        p.drawText(r, Qt.AlignLeft | Qt.AlignTop | Qt.TextWordWrap, text)


class CaptionItem(QGraphicsItem):
    """A row caption (its name above its first block, with its role badge) or any small text."""

    def __init__(self, x, y, text, color="#8b93a1", size=8.5, badge=""):
        super().__init__()
        self.pt, self.text, self.color, self.size, self.badge = QPointF(x, y), text, color, size, badge
        self.setPos(x, y)
        self.setZValue(1.5)

    def boundingRect(self):                                           # noqa: N802
        return QRectF(0, -16, 260, 20)

    def text_rect(self) -> QRectF:
        """Where its words and badge are written, in scene coordinates."""
        f = _font(self.size)
        room = 258 - (badge_width(self.badge) + 6 if self.badge else 0)
        w = QFontMetricsF(f).horizontalAdvance(elide(self.text, f, room)) + (
            badge_width(self.badge) + 6 if self.badge else 0)
        return QRectF(self.pos().x(), self.pos().y() - 15, w, 17)

    def paint(self, p, opt, widget=None):
        p.setFont(_font(self.size))
        p.setPen(QColor(self.color))
        room = 258 - (badge_width(self.badge) + 6 if self.badge else 0)
        shown = elide(self.text, _font(self.size), room)
        p.drawText(QRectF(0, -14, 260, 16), Qt.AlignLeft | Qt.AlignVCenter, shown)
        if self.badge:
            paint_badge(p, QFontMetricsF(_font(self.size)).horizontalAdvance(shown) + 6, -15, self.badge)


class ProjectItem(QGraphicsItem):
    """The Project block above the first row: outlined in its colour (red no target, amber partial, green complete)."""
    COLOURS = {"none": BAD, "partial": WARN, "complete": OK}
    #: today's sizes of its two lines (graph.py ProjectItem), in points
    TITLE_PT, SUMMARY_PT = 10, 8.5

    def __init__(self, rect: QRectF, title, summary, state, selected, tip):
        super().__init__()
        self.rect, self.title, self.summary, self.state, self.selected = rect, title, summary, state, selected
        self.setToolTip(tip)
        self.setZValue(6)

    def boundingRect(self):                                           # noqa: N802
        return self.rect.adjusted(-2, -2, 2, 2)

    def paint(self, p, opt, widget=None):
        p.setRenderHint(QPainter.Antialiasing)
        col = QColor(self.COLOURS.get(self.state, WARN))
        p.setPen(QPen(QColor(ACCENT) if self.selected else col, 2 if self.selected else 1.4))
        p.setBrush(QColor("#2b3446" if self.selected else "#22262e"))
        p.drawRoundedRect(self.rect, 7, 7)
        p.save()
        p.translate(self.rect.topLeft())
        _band(p, self.rect.height(), self.COLOURS.get(self.state, WARN))
        p.restore()
        # today's fonts (10 pt bold, 8.5 pt) whatever the box's height: the core's box is shorter, the lines closer
        compact = self.rect.height() < 46
        f = _font(self.TITLE_PT, True)
        p.setFont(f)
        p.setPen(QColor("#e8ebf0"))
        w = self.rect.width() - 20
        p.drawText(QRectF(self.rect.x() + 11, self.rect.y() + (0 if compact else 6), w, 19 if compact else 20),
                   Qt.AlignLeft | Qt.AlignVCenter, elide(self.title, f, w))
        f2 = _font(self.SUMMARY_PT)
        p.setFont(f2)
        p.setPen(QColor("#8b93a1"))
        p.drawText(QRectF(self.rect.x() + 11, self.rect.y() + (17 if compact else 28), self.rect.width() - 15,
                          17 if compact else 18),
                   Qt.AlignLeft | Qt.AlignVCenter, elide(self.summary, f2, self.rect.width() - 15))


class LinkItem(QGraphicsItem):
    """A link: words alone, underlined under the mouse ("New workflow")."""

    def __init__(self, centre: QPointF, text: str):
        super().__init__()
        self.text = text
        self.font = _font(10)
        w = QFontMetricsF(self.font).horizontalAdvance(text) + 8
        self.rect = QRectF(centre.x() - w / 2, centre.y() - 12, w, 24)
        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.PointingHandCursor)
        self.setZValue(7)

    def boundingRect(self):                                           # noqa: N802
        return self.rect

    def paint(self, p, opt, widget=None):
        f = QFont(self.font)
        f.setUnderline(self.isUnderMouse())
        p.setFont(f)
        p.setPen(QColor(ACCENT))
        p.drawText(self.rect, Qt.AlignCenter, self.text)

    def hoverEnterEvent(self, e):                                     # noqa: N802
        self.update()

    def hoverLeaveEvent(self, e):                                     # noqa: N802
        self.update()
