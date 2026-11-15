"""What the application shows, collected from a running window: every visible text (widgets' texts, items drawn on
the graph, tooltips, the status bar, toasts) with the rectangle it takes in the window, for the two checks the tests
make on it: no English word in the French interface (and no French word in the English one), and no two visible texts
drawn over each other.

``texts(root)`` gives ``Shown`` records ``(text, kind, rect, where)``: ``rect`` is the text's own box in the top-level
window's coordinates (None for a tooltip or a text with no place), ``kind`` one of "label", "button", "item", "tip",
"painted"... The graph is a scene: its items are painted into a recording painter (``Spy``), which gives the text each
one really draws (elided as drawn) and where."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from PySide6.QtCore import QPoint, QPointF, QRect, QRectF, Qt
from PySide6.QtGui import QFontMetricsF, QPainter
from PySide6.QtWidgets import (
    QAbstractButton,
    QAbstractItemView,
    QComboBox,
    QGraphicsItem,
    QGraphicsView,
    QGroupBox,
    QLabel,
    QLineEdit,
    QListWidget,
    QMenu,
    QPlainTextEdit,
    QTabBar,
    QTableWidget,
    QTextEdit,
    QTreeWidget,
    QWidget,
)

from astro_gpu.client import i18n


@dataclass
class Shown:
    text: str
    kind: str
    rect: QRect | None = None          # in the top-level window's coordinates
    where: str = ""

    def __repr__(self):
        return f"{self.kind}:{self.text!r}" + (f"@{self.where}" if self.where else "")


# ---------------------------------------------------------------------------------------------- the painted texts
def _read(painter: QPainter, a: tuple):
    """(text, box in the painter's logical coordinates) of the arguments of one ``drawText`` call, or None."""
    fm = QFontMetricsF(painter.font())
    if isinstance(a[0], (QRect, QRectF)) and len(a) >= 3 and isinstance(a[2], str):
        return a[2], fm.boundingRect(QRectF(a[0]), int(a[1]), a[2])
    if isinstance(a[0], (QPoint, QPointF)) and len(a) >= 2 and isinstance(a[1], str):
        return a[1], fm.boundingRect(a[1]).translated(QPointF(a[0]))
    if isinstance(a[0], (QRect, QRectF)) and len(a) == 2 and isinstance(a[1], str):
        return a[1], fm.boundingRect(QRectF(a[0]), 0, a[1])
    if len(a) >= 3 and isinstance(a[2], str) and isinstance(a[0], (int, float)):
        return a[2], fm.boundingRect(a[2]).translated(a[0], a[1])
    return None


def painted(root: QWidget) -> list[tuple[str, QRect]]:
    """The texts the window's own paint code draws (the graph's items, the table, the charts: everything that is not a
    label or a button), with each one's box in the window's coordinates, clipped to what shows. The window is grabbed
    once with ``QPainter.drawText`` recording its calls."""
    got: list[tuple[str, QRectF]] = []
    orig = QPainter.drawText

    def spy(self, *a):
        try:
            r = _read(self, a)
            if r is not None and r[0].strip():
                box = self.deviceTransform().mapRect(r[1])
                clip = self.clipBoundingRect()
                if not clip.isEmpty():
                    box = box.intersected(self.deviceTransform().mapRect(clip))
                got.append((r[0], box))
        except Exception:  # noqa: BLE001 - a call this recorder does not read is drawn all the same
            pass
        return orig(self, *a)

    QPainter.drawText = spy
    try:
        root.grab()
    finally:
        QPainter.drawText = orig
    return [(t, b.toAlignedRect()) for t, b in got if not b.isEmpty()]


def _tips(item: QGraphicsItem, out: list):
    if item.toolTip():
        out.append(item.toolTip())
    fn = getattr(item, "tipfn", None)
    if callable(fn):
        try:
            t = fn()
        except Exception:  # noqa: BLE001 - a tip that cannot be built yet
            t = ""
        if t:
            out.append(t)


# ---------------------------------------------------------------------------------------------- widgets
def _plain(s: str) -> str:
    return re.sub(r"<[^>]+>", "", s or "").replace("&&", "\0").replace("&", "").replace("\0", "&").strip()


def _visible(w: QWidget, root: QWidget) -> bool:
    return w is root or w.isVisibleTo(root)


def _box(w: QWidget, root: QWidget, text: str, flags: int) -> QRect | None:
    """The box a widget's text takes, in the window's coordinates and clipped to what shows of the widget."""
    vis = w.visibleRegion().boundingRect()
    if vis.isEmpty():
        return None
    inner = w.contentsRect() if isinstance(w, QLabel) else w.rect().adjusted(6, 0, -6, 0)
    fm = w.fontMetrics()
    r = fm.boundingRect(inner, flags, text)
    r = r.intersected(vis)
    if r.isEmpty():
        return None
    return QRect(w.mapTo(root, r.topLeft()), r.size())


def texts(root: QWidget, scenes: bool = True) -> list[Shown]:
    """Every visible text of a window or dialog."""
    out: list[Shown] = []
    seen_views = []
    for w in [root] + root.findChildren(QWidget):
        if not _visible(w, root):
            continue
        name = w.objectName() or type(w).__name__
        if isinstance(w, QLabel):
            t = _plain(w.text())
            if t and w.pixmap().isNull():
                fl = int(w.alignment()) | (int(Qt.TextWordWrap) if w.wordWrap() else 0)
                out.append(Shown(t, "label", _box(w, root, t, fl), name))
        elif isinstance(w, QAbstractButton):
            t = _plain(w.text())
            if t:
                out.append(Shown(t, "button", _box(w, root, t, int(Qt.AlignCenter)), name))
        elif isinstance(w, QGroupBox) and w.title():
            out.append(Shown(_plain(w.title()), "label", None, name))
        elif isinstance(w, QComboBox):
            out += [Shown(w.itemText(i), "item", None, name) for i in range(w.count())]
            if w.currentText():
                out.append(Shown(w.currentText(), "label", _box(w, root, w.currentText(), int(Qt.AlignLeft)), name))
        elif isinstance(w, QTabBar):
            for i in range(w.count()):
                out.append(Shown(_plain(w.tabText(i)), "item", None, name))
                if w.tabToolTip(i):
                    out.append(Shown(w.tabToolTip(i), "tip", None, name))
        elif isinstance(w, QListWidget):
            out += [Shown(w.item(i).text(), "item", None, name) for i in range(w.count())]
        elif isinstance(w, QTableWidget):
            out += [Shown(w.horizontalHeaderItem(i).text(), "item", None, name) for i in range(w.columnCount())
                    if w.horizontalHeaderItem(i) is not None]
        elif isinstance(w, QTreeWidget):
            hi = w.headerItem()
            if hi is not None:
                out += [Shown(hi.text(i), "item", None, name) for i in range(w.columnCount())]
        elif isinstance(w, (QTextEdit, QPlainTextEdit)):
            t = w.toPlainText().strip()
            if t:
                out.append(Shown(t, "label", None, name))
        if isinstance(w, QLineEdit) and w.placeholderText():
            out.append(Shown(w.placeholderText(), "label", None, name))
        if w.toolTip():
            out.append(Shown(_plain(w.toolTip()), "tip", None, name))
        if isinstance(w, QGraphicsView) and scenes and w.scene() is not None:
            seen_views.append(w)
        if isinstance(w, QAbstractItemView) and not isinstance(w, (QListWidget, QTableWidget, QTreeWidget)):
            pass
    for t, box in painted(root) if scenes else []:
        out.append(Shown(t, "painted", QRect(box), "paint"))
    for v in seen_views:
        tips: list = []
        for it in v.scene().items():
            _tips(it, tips)
        out += [Shown(_plain(t), "tip", None, "scene") for t in tips]
    return [s for s in out if s.text]


def menus(win) -> list[Shown]:
    """The menus of a main window: titles, actions, tooltips, submenus."""
    out = []
    todo = [a.menu() for a in win.menuBar().actions()]
    while todo:
        m: QMenu = todo.pop()
        out.append(Shown(_plain(m.title()), "item", None, "menu"))
        for a in m.actions():
            if a.text():
                out.append(Shown(_plain(a.text()), "item", None, "menu"))
            if a.toolTip() and a.toolTip() != a.text():
                out.append(Shown(_plain(a.toolTip()), "tip", None, "menu"))
            if a.menu() is not None:
                todo.append(a.menu())
    return out


# ---------------------------------------------------------------------------------------------- the words
WORD = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]{3,}")
#: proper names, units, abbreviations and the words the two languages share: never flagged
ALLOWED = {
    "float", "gaia", "fits", "tiff", "tif", "jpeg", "jpg", "png", "raw", "adu", "rgb", "lrgb", "sho", "hoo", "cfa", "gpu",
    "cpu", "fwhm", "iso", "ccd", "cmos", "fwhm", "ngc", "ssd", "hdd", "bayer", "grbg", "rggb", "gbrg", "bggr", "xtrans",
    "drizzle", "windows", "zwo", "asi", "sony", "canon", "nikon", "seestar", "iris", "heart", "mono", "dec", "ascom",
    "vram", "mhz", "cuda", "openngc", "steinicke", "esa", "dpac", "licence", "apache", "mit", "bsd", "html", "url",
    "log", "pdf", "csv", "json", "xisf", "dng", "cr2", "cr3", "arw", "nef", "fit", "fts", "sfp", "sftemplate",
    "untitled", "session", "post", "image", "stack", "tsv", "gz", "ngcic", "sharpless", "vizier", "openngc",
}


def vocabulary(lang: str) -> tuple[set[str], set[str]]:
    """(the words only English has, the words only French has), from the French catalogue: its keys are the English
    source texts, its values the French ones. A word both sides use ("session", "image") is in neither."""
    cat: dict[str, str] = {}
    from astro_gpu.client.i18n import LOCALE
    for f in sorted((LOCALE / "fr").glob("*.json")):
        cat.update(json.loads(f.read_text(encoding="utf-8")))
    en, fr = set(), set()
    for k, v in cat.items():
        en |= {w.lower() for w in WORD.findall(k)}
        fr |= {w.lower() for w in WORD.findall(v)}
    return en - fr - ALLOWED, fr - en - ALLOWED


def foreign(text: str, lang: str, mine: set[str] = frozenset(), vocab=None) -> list[str]:
    """The words of ``text`` that belong to the other language (the interface is in ``lang``); ``mine``: names the user
    typed (never flagged)."""
    en_only, fr_only = vocab or vocabulary(lang)
    bad = fr_only if lang == "en" else en_only
    keep = {w.lower() for m in mine for w in WORD.findall(m)}
    text = re.sub(r"[A-Za-z]:[\\/][^ )\"]*", " ", text)            # a file's path is the user's
    return [w for w in WORD.findall(text) if w.lower() in bad and w.lower() not in keep]


# ---------------------------------------------------------------------------------------------- the boxes
def overlaps(shown: list[Shown], slack: int = 1) -> list[tuple[Shown, Shown]]:
    """The pairs of visible texts whose boxes cross (by more than ``slack`` pixels each way)."""
    boxes = [s for s in shown if s.rect is not None and s.kind != "tip"]
    out = []
    for i, a in enumerate(boxes):
        for b in boxes[i + 1:]:
            if a.text == b.text and a.rect == b.rect:
                continue
            x = a.rect.intersected(b.rect)
            if x.width() > slack and x.height() > slack:
                out.append((a, b))
    return out


def set_language(code: str):
    i18n.set_language(code)
