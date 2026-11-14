"""Painting of the table (a mixin of table.Grid): everything is custom-painted from plain data, like the graph's items."""
from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QFontMetrics, QPen, QPolygonF

from ...graphcore.operations import COMMON
from ..i18n import _, plural
from . import style
from .table_logic import FOLDED, KIND_TITLE, count_text, duration, settings_text, short_count

ROW, WARNH, VAR, GHOST, HEAD, COMMON_FULL, COMMON_COMPACT = 40, 20, 26, 32, 48, 48, 32
BG, CELL, CELL2, COMBG, LINE = "#15171c", "#1b1e25", "#20242d", "#222733", "#2a2f39"
TXT, DIM, FAINT = "#e8ebf0", "#8b93a1", "#5d6470"
ACCENT, WARN, OK = style.ACCENT, style.WARN, style.OK
DASH = "#4a5263"
STATE = style.STATE_COL

#: the column tooltips (English; shown through _)
KIND_TIP = {"lights": "Lights = the images of the target", "darks": "Darks = closed-cap frames, same exposure",
            "flats": "Flats = frames of an even light", "bias": "Bias = shortest frames",
            "flat_darks": "Flat darks = closed-cap frames at the flats' exposure"}
#: the column headers (English; shown through _)
HEADER = {"stack": "Stacks", "session": "Session", "lights": "Lights", "darks": "Darks", "flats": "Flats",
          "bias": "Bias", "flat_darks": "Flat darks", "cal": "Calibration", "reg": "Registration",
          "sel": "Selection", "stk": "Stacking"}


def font(px, bold=False, italic=False):
    f = QFont("Segoe UI")
    f.setPixelSize(px)
    f.setBold(bold)
    f.setItalic(italic)
    return f


F13, F13B, F11, F12B, F10B, F11I, F12 = (font(13), font(13, True), font(11), font(12, True), font(9, True),
                                         font(11, italic=True), font(12))


@dataclass
class Cell:
    id: tuple
    kind: str
    x: float
    y: float
    w: float
    h: float
    pinned: bool = False

    @property
    def rect(self):
        return QRectF(self.x, self.y, self.w, self.h)


def colour(c, a=255):
    q = QColor(c)
    q.setAlpha(a)
    return q


def elide(text, f, w):
    return QFontMetrics(f).elidedText(text, Qt.ElideRight, max(int(w), 1))


def width_of(text, f):
    return QFontMetrics(f).horizontalAdvance(text)


class Painting:
    # ------------------------------------------------------------------ small tools
    def _text(self, p, x, y, text, f, col, w):
        p.setFont(f)
        p.setPen(colour(col))
        p.drawText(QPointF(x, y), elide(text, f, w))

    def _first_fitting(self, p, x, y, candidates, f, col, w, last=True):
        """Paint the first candidate that fits whole in `w` (a text is never cut when a shorter form exists). None
        fits: the last one is cut when `last`, else nothing is painted (an optional line)."""
        for t in candidates:
            if t and width_of(t, f) <= w:
                p.setFont(f)
                p.setPen(colour(col))
                p.drawText(QPointF(x, y), t)
                return True
        if last and candidates:
            self._text(p, x, y, candidates[-1], f, col, w)
        return False

    def _pill(self, p, rect, text, bg, col, f=F11):
        p.setPen(Qt.NoPen)
        p.setBrush(colour(bg))
        p.drawRoundedRect(rect, rect.height() / 2, rect.height() / 2)
        p.setFont(f)
        p.setPen(colour(col))
        p.drawText(rect, Qt.AlignCenter, text)

    def _dashes(self, p, rect, col, a=255, thick=1.0, r=4):
        pen = QPen(colour(col, a), thick, Qt.DashLine)
        pen.setDashPattern([3, 3])
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        p.drawRoundedRect(rect, r, r)

    def _link_glyph(self, p, rect, col):
        """The link glyph of the Common line: two overlapping rounded rings."""
        p.setPen(QPen(colour(col), 1.6))
        p.setBrush(Qt.NoBrush)
        p.drawRoundedRect(QRectF(rect.x(), rect.y() + 3, 10, 7), 3.5, 3.5)
        p.drawRoundedRect(QRectF(rect.x() + 6, rect.y() + 3, 10, 7), 3.5, 3.5)

    @staticmethod
    def button_label(w):
        return _(w.button or "Leave")

    # ------------------------------------------------------------------ zones (hit-testing of sub-elements)
    def zones(self, c: Cell) -> dict:
        k = c.kind
        z = {}
        r = QRectF(c.x, c.y, c.w, c.h)
        if k == "box":
            b = self.model.box(c.id[1], c.id[2])
            R = r.adjusted(3, 3, -3, -3)
            if b.mode in ("own", "none", "inherited") and b.bar and not self.multiple_selection():
                z["reset"] = QRectF(R.right() - 18, R.top() + 1, 16, 16)
            if (self.focus_row() == self.row_of(c.id) or self.hover == c.id or c.id in self.sel) \
                    and b.mode not in ("greyed", "folded"):
                z["picker"] = QRectF(R.right() - 18, R.bottom() - 18, 16, 16)
            if b.reading:
                z["stop"] = QRectF(R.right() - 18, R.top() + 1, 16, 16)
        elif k == "warn":
            w = self.model.box(c.id[1], c.id[2]).warn
            if w is not None:
                lab = self.button_label(w)
                x0 = c.x + 6 + width_of("⚠ " + w.text, F11) + 14
                z["button"] = QRectF(x0, c.y + 2, width_of(lab, F11) + 14, 16)
                if w.button:
                    z["leave"] = QRectF(x0 + width_of(lab, F11) + 20, c.y + 2, 16, 16)
        elif k in ("setting", "cal"):
            if self.is_own(c):
                z["reset"] = QRectF(c.x + c.w - 20, self.y_top(c) + 2, 16, 16)
        elif k == "stack":
            z["chevron"] = QRectF(c.x + 2, self.y_top(c) + 4, 16, 16)
        elif k == "name":
            z["grip"] = QRectF(c.x + 1, c.y + 8, 10, 24)
        return z

    def y_top(self, c: Cell):
        """Top of the visible part of a spanning cell (its text is sticky)."""
        if c.pinned:
            return c.y
        top = self.scroll_y()
        return min(max(c.y, top), max(c.y, c.y + c.h - ROW))

    # ------------------------------------------------------------------ header
    def paint_header(self, p, W):
        xs = self.xs
        p.fillRect(0, 0, W, HEAD, colour("#181b21"))

        def band(a, b, title):
            x0, x1 = xs[a][0], xs[b][0] + xs[b][1]
            r = QRectF(x0 + 2, 2, x1 - x0 - 4, 16)
            p.setPen(Qt.NoPen)
            p.setBrush(colour("#191c22"))
            p.drawRoundedRect(r, 6, 6)
            p.setFont(F10B)
            p.setPen(colour("#6b7280"))
            p.drawText(r, Qt.AlignCenter, title)
        band("lights", "cal", _("SESSION"))
        band("reg", "stk", _("STACK"))
        for col, (x, w) in xs.items():
            t = _(HEADER[col])
            if col == "flat_darks":
                full = _("Flat darks · optional")
                t = full if width_of(full, F12B) <= w - 12 else t
            p.setFont(F12B)
            p.setPen(colour("#c9ced8"))
            p.drawText(QPointF(x + 8, 38), elide(t, F12B, w - 12))
        p.setPen(colour(LINE))
        p.drawLine(0, HEAD - 1, W, HEAD - 1)

    # ------------------------------------------------------------------ cells
    def paint_cell(self, p, c: Cell):
        f = getattr(self, "p_" + c.kind, None)
        if f:
            f(p, c)
        if c.id in self.sel:
            p.setPen(QPen(colour(ACCENT), 2))
            p.setBrush(Qt.NoBrush)
            p.drawRoundedRect(QRectF(c.x + 1.5, c.y + 1.5, c.w - 3, c.h - 3), 4, 4)
        if c.id == self.focus and self.hasFocus():
            p.setPen(QPen(colour("#c9ced8"), 1, Qt.DotLine))
            p.setBrush(Qt.NoBrush)
            p.drawRoundedRect(QRectF(c.x + 3.5, c.y + 3.5, c.w - 7, c.h - 7), 3, 3)

    def _is_hot(self, c):
        return self.hot is not None and c.id in self.hot.cells

    def p_box(self, p, c: Cell):
        owner, kind = c.id[1], c.id[2]
        b = self.model.box(owner, kind)
        R = QRectF(c.x + 3, c.y + 3, c.w - 6, c.h - 6)
        common = owner == COMMON
        compact = common and self.compact
        dragging = self.drag is not None
        zs = self.zones(c)
        if b.mode == "greyed":
            self._text(p, R.x() + 8, R.y() + R.height() / 2 + 4, "—", F13, FAINT, R.width())
            return
        if b.mode == "folded":
            n = b.n
            self._first_fitting(p, R.x() + 8, R.y() + 18, [count_text(n, kind), short_count(n, kind)] if n else ["—"],
                                F13, DIM, R.width() - 12)
            return
        hot = self._is_hot(c)
        target = dragging and self.drag.box_targets
        tint = 150 if not common else 90
        if b.mode == "empty":
            if compact and not dragging:
                self._text(p, R.x() + 8, R.y() + R.height() / 2 + 4, "—", F13, FAINT, R.width())
            else:
                short = "+ " + _(KIND_TITLE[kind])
                lab = _("+ Flat darks · optional") if kind == "flat_darks" else short
                if target:
                    p.setPen(Qt.NoPen)
                    p.setBrush(colour(ACCENT, 70 if hot else 14))
                    p.drawRoundedRect(R, 4, 4)
                    self._dashes(p, R, ACCENT, 255 if hot else tint, 1.5 if hot else 1.0)
                else:
                    self._dashes(p, R, DASH)
                self._first_fitting(p, R.x() + 10, R.y() + R.height() / 2 + 4, [lab, short], F11,
                                    TXT if hot else DIM, R.width() - 14 - (12 if b.warn is not None else 0))
                if b.warn is not None:                  # the warning itself is the amber line under the row
                    p.setFont(F11)
                    p.setPen(colour(WARN))
                    p.drawText(QPointF(R.right() - 14 - (20 if "picker" in zs else 0), R.y() + R.height() / 2 + 4), "⚠")
        else:
            if b.mode == "own":
                p.setPen(Qt.NoPen)
                p.setBrush(colour(COMBG if common else CELL2))
                p.drawRoundedRect(R, 4, 4)
            if b.bar:
                p.setPen(Qt.NoPen)
                p.setBrush(colour(ACCENT))
                p.drawRoundedRect(QRectF(R.x(), R.y() + 2, 3, R.height() - 4), 1.5, 1.5)
            col = TXT if b.mode == "own" else (DIM if b.mode == "none" else "#6f7787")
            y1 = R.y() + (14 if not common else 17)
            if compact:
                y1 = R.y() + R.height() / 2 + 4
            if b.mode == "own":
                p.setPen(Qt.NoPen)
                p.setBrush(colour(OK if b.job is None or not b.job.active else ACCENT))
                p.drawEllipse(QPointF(R.x() + 9, y1 - 4), 3, 3)
                self._first_fitting(p, R.x() + 17, y1, [b.l1, b.l1s], F13, col, R.width() - 22)
            else:
                self._first_fitting(p, R.x() + 8, y1, [b.l1, b.l1s, b.l1.split(" · ")[0]],
                                    F13 if b.mode == "inherited" else F11I, col, R.width() - 12)
            y2 = y1 + 14
            if b.reading and not compact:
                self._first_fitting(p, R.x() + 17, y2, [b.reading], F11, ACCENT, R.width() - 40)
            elif b.l2 and not compact:
                self._first_fitting(p, R.x() + 17 if b.mode == "own" else R.x() + 8, y2, [b.l2, b.l2s], F11, DIM,
                                    R.width() - 40, last=False)
            if "stop" in zs:
                p.setPen(colour(DIM))
                p.setFont(F13)
                p.drawText(zs["stop"], Qt.AlignCenter, "×")
            if b.progress is not None:
                p.setPen(Qt.NoPen)
                p.setBrush(colour(ACCENT))
                p.drawRect(QRectF(R.x(), R.bottom() - 2, R.width() * max(b.progress, 0.04), 2))
            if (b.corner or b.warn is not None) and "stop" not in zs:
                p.setFont(F11)
                p.setPen(colour(WARN))
                p.drawText(QPointF(R.right() - 14, R.y() + 12), "⚠")
            if "reset" in zs and self.hover == c.id and not dragging and "stop" not in zs:
                p.setFont(F13)
                p.setPen(colour(ACCENT))
                p.drawText(zs["reset"], Qt.AlignCenter, "↺")
            if target:
                self._halves(p, R, hot, common)
        if "picker" in zs and not dragging:
            p.setPen(QPen(colour("#4a5263"), 1))
            p.setBrush(colour("#2a2f39"))
            p.drawRoundedRect(zs["picker"], 3, 3)
            p.setPen(colour(TXT))
            p.setFont(F11)
            p.drawText(zs["picker"], Qt.AlignCenter, "▾")

    def _halves(self, p, R, hot, common):
        """A filled or inherited box under a drag: two halves Add | Replace."""
        mid = R.x() + R.width() / 2
        left = QRectF(R.x(), R.y(), R.width() / 2, R.height())
        right = QRectF(mid, R.y(), R.width() / 2, R.height())
        half = self.hot.half if hot and self.hot is not None else None
        for rect, name, lab, short in ((left, "add", _("Add"), _("Add")), (right, "replace", _("Replace"), _("Repl."))):
            if width_of(lab, F11) + 6 > rect.width():
                lab = short
            on = hot and half == name
            p.setPen(Qt.NoPen)
            p.setBrush(colour("#1d2633" if not on else "#2b4468"))
            p.drawRect(rect.adjusted(0.5, 0.5, -0.5, -0.5))
            p.setFont(F11)
            p.setPen(colour(TXT if on else "#9db6d8"))
            p.drawText(rect, Qt.AlignCenter, lab)
        self._dashes(p, R, ACCENT, 255 if hot else (150 if not common else 90), 1.5 if hot else 1.0)
        p.setPen(colour(ACCENT, 140))
        p.drawLine(QPointF(mid, R.y() + 3), QPointF(mid, R.bottom() - 3))

    def p_name(self, p, c: Cell):
        sid = c.id[1]
        R = QRectF(c.x + 3, c.y + 3, c.w - 6, c.h - 6)
        hot = self._is_hot(c)
        if self.drag is not None and self.drag.box_targets:
            p.setPen(Qt.NoPen)
            p.setBrush(colour(ACCENT, 70 if hot else 14))
            p.drawRoundedRect(R, 4, 4)
            self._dashes(p, R, ACCENT, 255 if hot else 150, 1.5 if hot else 1.0)
        if self.hover == c.id and self.drag is None:
            p.setPen(Qt.NoPen)
            p.setBrush(colour("#2a2f39"))
            p.drawRoundedRect(R, 4, 4)
            p.setBrush(colour(DIM))
            for i in range(3):
                for j in range(2):
                    p.drawEllipse(QPointF(R.x() + 4 + j * 4, R.y() + R.height() / 2 - 6 + i * 6), 1.2, 1.2)
        date = self.model.date_text(sid)              # its night(s) and the time its lights add up to (headers)
        if date:
            self._text(p, R.x() + 14, R.y() + 14, self.model.session_name(sid), F13, TXT, R.width() - 16)
            self._first_fitting(p, R.x() + 14, R.y() + 28, [date, date.split(" · ")[0]], F11, DIM, R.width() - 16)
        else:
            self._text(p, R.x() + 14, R.y() + R.height() / 2 + 5, self.model.session_name(sid), F13, TXT,
                       R.width() - 16)

    def p_folded_name(self, p, c: Cell):
        pl = self.model.stack_by_id(c.id[1])
        n = len(pl["sessions"]) if pl else 0
        self._text(p, c.x + 10, c.y + 24, plural(n, "{n} session", "{n} sessions"), F13, DIM, c.w - 14)

    def p_stack(self, p, c: Cell):
        pl = self.model.stack_by_id(c.id[1])
        if pl is None:
            return
        yh = self.y_top(c)
        x, w = c.x + 20, c.w - 26
        R = QRectF(c.x + 2, c.y + 2, c.w - 4, c.h - 4)
        p.setPen(Qt.NoPen)
        p.setBrush(colour(CELL))
        p.drawRoundedRect(R, 5, 5)
        if self._is_hot(c) and self.drag is not None:
            p.setBrush(colour(ACCENT, 60))
            p.drawRoundedRect(R, 5, 5)
        p.setBrush(colour(DIM))
        cx, cy = c.x + 10, yh + 14
        if pl["folded"]:
            p.drawPolygon(QPolygonF([QPointF(cx - 2, cy - 4), QPointF(cx + 3, cy), QPointF(cx - 2, cy + 4)]))
        else:
            p.drawPolygon(QPolygonF([QPointF(cx - 4, cy - 2), QPointF(cx + 4, cy - 2), QPointF(cx, cy + 3)]))
        self._text(p, x, yh + 17, pl["name"], F13B, TXT, w - 16)
        ref_word = _("reference")
        wb = width_of(ref_word, F10B) + 12

        align = pl["row"].align or {}
        on_grid = bool(align.get("ref")) and not align.get("off")

        def mark(px, py):
            if pl["badge"] == ref_word or (pl["ref"] and not pl["badge"]):
                self._pill(p, QRectF(px, py - 12, wb, 15), ref_word, ACCENT, "#0f1115", F10B)
            elif pl["badge"]:
                self._first_fitting(p, px, py, [pl["badge"], _("on ref. grid")] if on_grid else [pl["badge"]],
                                    F11, DIM, c.x + c.w - 4 - px)
        ns, nb, wv = self.model.totals(pl)
        if ns >= 2:
            parts = [plural(ns, "{n} session", "{n} sessions"), count_text(nb, "lights")]
            sec = self.model.seconds(pl)                 # the time its lights add up to (headers)
            if sec:
                parts.append(duration(sec))
            full = " · ".join(parts + ([plural(wv, "{n} warn.", "{n} warn.")] if wv else []))
            self._first_fitting(p, c.x + 10, yh + 33, [full, " · ".join(parts), parts[0]], F11, DIM, c.w - 14)
            mark(c.x + 10, yh + 52)
        else:
            mark(c.x + 10, yh + 33)
            if wv:
                p.setFont(F11)
                p.setPen(colour(WARN))
                p.drawText(QPointF(c.x + c.w - 18, yh + 17), "⚠")

    def cell_text(self, c):
        """(text, own?, block kind, block) of a settings cell."""
        kind = self.setting_kind(c)
        if c.id[1] == COMMON:
            return settings_text(kind, self.doc.effective(kind)), False, kind, None
        b = self.block_of(c)
        if b is None:                                    # the stack has no such block (switched out): nothing to show
            return "—", False, kind, None
        own = self.doc.own_keys(b)
        return settings_text(kind, self.doc.effective(b)), bool(own), kind, b

    def is_own(self, c):
        if c.id[1] == COMMON or str(c.id[1]).startswith(FOLDED):
            return False
        return self.cell_text(c)[1]

    def p_setting(self, p, c: Cell):
        self._setting(p, c)

    def p_cal(self, p, c: Cell):
        self._setting(p, c)

    def p_folded_cal(self, p, c: Cell):
        pass

    def _setting(self, p, c):
        t, own, kind, b = self.cell_text(c)
        yh = self.y_top(c)
        common = c.id[1] == COMMON
        R = QRectF(c.x + 3, c.y + 3, c.w - 6, c.h - 6)
        p.setPen(Qt.NoPen)
        p.setBrush(colour(COMBG if common else (CELL2 if own else CELL)))
        p.drawRoundedRect(R, 4, 4)
        if own:
            p.setBrush(colour(ACCENT))
            p.drawRoundedRect(QRectF(R.x(), R.y() + 2, 3, R.height() - 4), 1.5, 1.5)
        y = yh + (17 if not common else 20)
        if common and c.h <= COMMON_COMPACT + 1:
            y = c.y + c.h / 2 + 4
        x = R.x() + 8
        if b is not None:
            p.setPen(Qt.NoPen)
            p.setBrush(colour(STATE.get(self.doc.state(b), "#5d6470")))
            p.drawEllipse(QPointF(x + 2, y - 4), 3, 3)
            x += 10
        pre = "" if (own or common) else "↑ "
        ts = settings_text(kind, self.doc.effective(b if b is not None else kind), short=True)
        self._first_fitting(p, x, y, [pre + t, pre + ts], F12, TXT if (own or common) else "#6f7787", R.right() - x - 6)
        zs = self.zones(c)
        if "reset" in zs and self.hover == c.id and self.drag is None:
            p.setFont(F13)
            p.setPen(colour(ACCENT))
            p.drawText(zs["reset"], Qt.AlignCenter, "↺")

    def _ghost(self, p, c, text):
        R = QRectF(c.x + 3, c.y + 3, c.w - 6, c.h - 6)
        hot = self._is_hot(c)
        if self.drag is not None:
            p.setPen(Qt.NoPen)
            p.setBrush(colour(ACCENT, 70 if hot else 14))
            p.drawRoundedRect(R, 4, 4)
            self._dashes(p, R, ACCENT, 255 if hot else 150, 1.5 if hot else 1.0)
        else:
            self._dashes(p, R, "#3c4350" if self.hover != c.id else DASH)
        self._text(p, R.x() + 12, R.y() + R.height() / 2 + 4, text, F11, TXT if hot else DIM, R.width() - 16)

    def p_ghost_s(self, p, c: Cell):
        self._ghost(p, c, _("+ Session · drop a session's folder"))

    def p_ghost_p(self, p, c: Cell):
        self._ghost(p, c, _("+ Stack · drop a session's folder"))

    def p_warn(self, p, c: Cell):
        """The amber warning of a box: its whole text on its own line under the row, with its button."""
        w = self.model.box(c.id[1], c.id[2]).warn
        if w is None:
            return
        zs = self.zones(c)
        p.setFont(F11)
        p.setPen(colour(WARN))
        p.drawText(QPointF(c.x + 6, c.y + 14), "⚠ " + w.text)
        self._pill(p, zs["button"], self.button_label(w), "#4a3a1d", "#ffd28a")
        if "leave" in zs:
            p.setFont(F11)
            p.setPen(colour(WARN))
            p.drawText(zs["leave"], Qt.AlignCenter, "×")

    def p_variant(self, p, c: Cell):
        v = self.variant_by_id(c.id[1])
        R = QRectF(c.x + 3, c.y + 2, c.w - 6, c.h - 4)
        p.setPen(Qt.NoPen)
        p.setBrush(colour("#1a1d24"))
        p.drawRoundedRect(R, 4, 4)
        if v:
            name = _("↳ Variant “{name}”").format(name=v["name"])
            self._first_fitting(p, R.x() + 12, R.y() + R.height() / 2 + 4, [f"{name} · {v['text']}", name], F11, DIM,
                                R.width() - 16)

    def p_body(self, p, c: Cell):
        R = QRectF(c.x + 8, c.y + 8, c.w - 16, c.h - 16)
        hot = self._is_hot(c)
        if self.drag is not None:
            p.setPen(Qt.NoPen)
            p.setBrush(colour(ACCENT, 60 if hot else 14))
            p.drawRoundedRect(R, 8, 8)
            self._dashes(p, R, ACCENT, 255 if hot else 150, 1.5, 8)
        else:
            self._dashes(p, R, DASH, 255, 1.0, 8)
        p.setFont(font(15))
        p.setPen(colour(TXT if hot else DIM))
        p.drawText(R, Qt.AlignCenter, _("No session. Drop a session's folder here, or several."))

    def p_common_name(self, p, c: Cell):
        R = QRectF(c.x + 3, c.y + 3, c.w - 6, c.h - 6)
        p.setPen(Qt.NoPen)
        p.setBrush(colour(COMBG))
        p.drawRoundedRect(R, 4, 4)
        self._link_glyph(p, QRectF(R.x() + 8, R.y() + (R.height() - 13) / 2 if self.compact else R.y() + 5, 16, 13),
                         ACCENT)
        if self.compact:
            self._first_fitting(p, R.x() + 30, R.y() + R.height() / 2 + 4,
                                [_("Common · used when there are several sessions"), _("Common · several sessions"),
                                 _("Common")], F11, DIM, R.width() - 34)
            return
        self._text(p, R.x() + 30, R.y() + 17, _("Common"), F13B, TXT, R.width() - 34)
        self._text(p, R.x() + 30, R.y() + 32, _("every session"), F11, DIM, R.width() - 34)

    # ------------------------------------------------------------------ whole paint
    def paint_all(self, p, W, H):
        top = HEAD + self.common_h
        p.save()
        p.setClipRect(QRectF(0, top, W, H - top))
        p.translate(0, top - self.scroll_y())
        vis0, vis1 = self.scroll_y() - 60, self.scroll_y() + (H - top) + 60
        for c in self.cells:
            if c.y + c.h < vis0 or c.y > vis1:
                continue
            self.paint_cell(p, c)
        self.paint_body_overlay(p)
        p.restore()
        p.fillRect(0, 0, W, top, colour(BG))
        self.paint_header(p, W)
        for c in self.pinned:
            self.paint_cell(p, c)
        p.setPen(colour("#323846"))
        p.drawLine(0, top - 1, W, top - 1)
        self.paint_drag(p, W, H)

    def paint_body_overlay(self, p):
        if self.insertion is not None:
            p.setPen(Qt.NoPen)
            p.setBrush(colour(ACCENT))
            p.drawRect(QRectF(self.xs["session"][0] + 2, self.insertion - 1, self.total_width - self.xs["session"][0] - 4, 2))

    def paint_drag(self, p, W, H):
        if self.drag is None or self.hot is None or not self.hot.label:
            return
        t = self.hot.label
        w = width_of(t, F11) + 18
        pos = self.drag_pos
        x = min(max(pos.x() + 14, 4), max(W - w - 4, 4))
        y = min(max(pos.y() + 18, 4), H - 26)
        r = QRectF(x, y, w, 22)
        p.setPen(QPen(colour(ACCENT if self.hot.ok else "#e06c6c"), 1))
        p.setBrush(colour("#0f1115", 235))
        p.drawRoundedRect(r, 5, 5)
        p.setFont(F11)
        p.setPen(colour(TXT))
        p.drawText(r, Qt.AlignCenter, t)
