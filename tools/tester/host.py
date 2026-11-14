"""The process that holds the application's window for a usability session (started by session.py, never by hand).

It starts the application's real main window from the checkout given on the command line, listens on 127.0.0.1 (the
port is written to <session dir>/port) and executes one command at a time on the Qt thread, as a user's mouse and
keyboard would: input events go to the window under the point, through Qt's own input path. Each command is a list of
timed steps; the step after one is scheduled before it runs, so a step that opens a modal dialog, a menu or a drag
(which run an event loop of their own) is followed by the next one, as a user's hand would follow. The answer is sent
once the window has stopped changing (or after a timeout): the path of the screenshot and what a person reads on screen.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import traceback
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=DeprecationWarning)

ap = argparse.ArgumentParser()
ap.add_argument("--app", required=True)
ap.add_argument("--dir", required=True)
ap.add_argument("--size", default="1920x1080")
ap.add_argument("--project", default="")
ap.add_argument("--onscreen", action="store_true")
ap.add_argument("--idle-minutes", type=float, default=30.0)
ARGS = ap.parse_args()

APP_DIR = Path(ARGS.app).resolve()
SDIR = Path(ARGS.dir).resolve()
SHOTS = SDIR / "shots"
SHOTS.mkdir(parents=True, exist_ok=True)
JOURNAL = SDIR / "journal.jsonl"
W, H = (int(v) for v in ARGS.size.lower().split("x"))
sys.path.insert(0, str(APP_DIR))

from PySide6 import QtGui, QtWidgets  # noqa: E402
from PySide6.QtCore import QByteArray, QMimeData, QPoint, QPointF, QRect, Qt, QTimer, QUrl  # noqa: E402
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QWheelEvent  # noqa: E402
from PySide6.QtNetwork import QHostAddress, QTcpServer  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402

QAbstractButton = QtWidgets.QAbstractButton
QApplication = QtWidgets.QApplication
QDialog = QtWidgets.QDialog
QFileDialog = QtWidgets.QFileDialog
QLabel = QtWidgets.QLabel
QMainWindow = QtWidgets.QMainWindow
QMenu = QtWidgets.QMenu
QMessageBox = QtWidgets.QMessageBox
QToolTip = QtWidgets.QToolTip
QWidget = QtWidgets.QWidget
QDragEnterEvent = QtGui.QDragEnterEvent
QDragMoveEvent = QtGui.QDragMoveEvent
QDropEvent = QtGui.QDropEvent
QDragLeaveEvent = QtGui.QDragLeaveEvent

# ---------------------------------------------------------------- file dialogs: answered by choose_file / choose_folder
ANSWERS: list[tuple[str, list[str]]] = []          # queued answers: ("file" | "folder", paths)
NOTES: list[str] = []                              # what happened since the last answer, said with it


def _take(kind_wanted: str, caption: str):
    if not ANSWERS:
        NOTES.append(f'A file dialog "{caption}" was opened, with no answer prepared: it was cancelled. Send '
                     f'choose_file or choose_folder first, then do the same gesture again.')
        return None
    kind, paths = ANSWERS.pop(0)
    NOTES.append(f'The file dialog "{caption}" was answered with: ' + "; ".join(paths))
    return paths


def _get_open_file_name(parent=None, caption="", dir="", filter="", *a, **k):  # noqa: A002
    got = _take("file", caption)
    return (got[0], "") if got else ("", "")


def _get_open_file_names(parent=None, caption="", dir="", filter="", *a, **k):  # noqa: A002
    got = _take("file", caption)
    return (list(got), "") if got else ([], "")


def _get_save_file_name(parent=None, caption="", dir="", filter="", *a, **k):  # noqa: A002
    got = _take("file", caption)
    return (got[0], "") if got else ("", "")


def _get_existing_directory(parent=None, caption="", dir="", *a, **k):  # noqa: A002
    got = _take("folder", caption)
    return got[0] if got else ""


QFileDialog.getOpenFileName = staticmethod(_get_open_file_name)
QFileDialog.getOpenFileNames = staticmethod(_get_open_file_names)
QFileDialog.getSaveFileName = staticmethod(_get_save_file_name)
QFileDialog.getExistingDirectory = staticmethod(_get_existing_directory)

# ---------------------------------------------------------------- the application
from astro_gpu.client import app as float_app  # noqa: E402

qapp = QApplication.instance() or QApplication(sys.argv[:1])
MAIN: QMainWindow | None = None
STATE = {"last": time.time(), "busy": False, "quitting": False, "start": time.time()}
PENDING: list = []                                 # commands waiting (one at a time)


def journal(rec: dict):
    with open(JOURNAL, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def next_shot() -> Path:
    n = 1 + max([int(p.stem) for p in SHOTS.glob("[0-9][0-9][0-9][0-9].png")] or [0])
    return SHOTS / f"{n:04d}.png"


# ---------------------------------------------------------------- what is on screen
def tops() -> list[QWidget]:
    """The visible top-level windows other than the main one, bottom to top (dialogs, then popups, then tooltips)."""
    out = []
    for w in QApplication.topLevelWidgets():
        if w is MAIN or not w.isVisible() or w.width() <= 1 or w.height() <= 1:
            continue
        if type(w).__name__ in ("QShapedPixmapWindow",):
            continue
        t = w.windowType()
        rank = 2 if t == Qt.ToolTip else 1 if t == Qt.Popup else 0
        if rank == 0 and w is QApplication.activeModalWidget():
            rank = 0.5
        out.append((rank, w))
    out.sort(key=lambda x: x[0])
    return [w for _r, w in out]


def origin() -> QPoint:
    return MAIN.mapToGlobal(QPoint(0, 0))


def compose() -> QImage:
    size = MAIN.size()
    img = QImage(size, QImage.Format_RGB32)
    img.fill(QColor("#101216"))
    p = QPainter(img)
    p.drawPixmap(QRect(QPoint(0, 0), size), MAIN.grab())
    o = origin()
    for w in tops():
        g = w.geometry()
        pos = w.mapToGlobal(QPoint(0, 0)) - o
        rect = QRect(pos, g.size())
        is_dialog = w.windowType() not in (Qt.Popup, Qt.ToolTip)
        if is_dialog:                                   # a title bar, as the window's frame would show it
            bar = QRect(rect.x(), max(rect.y() - 26, 0), rect.width(), 26)
            p.fillRect(bar, QColor("#2b3446"))
            p.setPen(QColor("#e6e6e6"))
            f = QFont(p.font())
            f.setBold(True)
            p.setFont(f)
            p.drawText(bar.adjusted(10, 0, -10, 0), Qt.AlignVCenter | Qt.AlignLeft, w.windowTitle() or "")
            p.drawText(bar.adjusted(10, 0, -12, 0), Qt.AlignVCenter | Qt.AlignRight, "x")
        p.drawPixmap(rect, w.grab())
        if is_dialog:
            p.setPen(QPen(QColor("#5a6478"), 1))
            p.drawRect(rect.adjusted(0, 0, -1, -1))
    pm = DRAG.get("pixmap")
    if DRAG.get("loop") is not None and pm is not None and not pm.isNull():     # what is being dragged, under the mouse
        p.setOpacity(0.75)
        p.drawPixmap(MOUSE["pos"] - DRAG.get("hot", QPoint(0, 0)), pm)
        p.setOpacity(1.0)
    p.end()
    return img


def plain(t: str) -> str:
    t = re.sub(r"<br\s*/?>", " ", t or "", flags=re.I)
    t = re.sub(r"<[^>]+>", "", t)
    for a, b in (("&nbsp;", " "), ("&lt;", "<"), ("&gt;", ">"), ("&amp;", "&"), ("&quot;", '"'), ("&#39;", "'")):
        t = t.replace(a, b)
    return re.sub(r"\s+", " ", t).strip()


def visible_texts() -> dict:
    out: dict = {"title": MAIN.windowTitle()}
    if QToolTip.isVisible() and QToolTip.text():
        out["tooltip"] = plain(QToolTip.text())
    notes = []
    for w in MAIN.findChildren(QWidget):
        if not w.isVisible():
            continue
        cls = type(w).__name__.lower()
        name = (w.objectName() or "").lower()
        if any(k in cls or k in name for k in ("toast", "banner", "notice", "strip_note", "flash")):
            texts = []
            if isinstance(w, QLabel):
                texts.append(plain(w.text()))
            else:
                texts += [plain(c.text()) for c in w.findChildren(QLabel) if c.isVisible() and c.text()]
                texts += [plain(c.text()) + " [button]" for c in w.findChildren(QAbstractButton)
                          if c.isVisible() and c.text()]
            t = " | ".join(x for x in texts if x)
            if t and t not in notes:
                notes.append(t)
    if notes:
        out["notices"] = notes
    sb = MAIN.statusBar()
    st = []
    if sb.isVisible():
        if sb.currentMessage():
            st.append(sb.currentMessage())
        for c in sb.findChildren(QLabel):
            if c.isVisible() and c.text() and plain(c.text()) not in st:
                st.append(plain(c.text()))
        for c in sb.findChildren(QAbstractButton):
            if c.isVisible() and c.text():
                st.append(plain(c.text()) + " [button]")
    if st:
        out["status"] = st
    dialogs = []
    for w in tops():
        t = w.windowType()
        if t == Qt.ToolTip:
            continue
        if t == Qt.Popup:
            kind = "menu" if isinstance(w, QMenu) else \
                "list" if w.findChild(QtWidgets.QAbstractItemView) is not None else "pop-up panel"
            dialogs.append({"popup": kind})
            continue
        d = {"dialog": w.windowTitle()}
        if isinstance(w, QMessageBox):
            d["text"] = plain(w.text())
            if w.informativeText():
                d["more"] = plain(w.informativeText())
        else:                                          # its first lines, as read at a glance
            seen = []
            for c in w.findChildren(QLabel):
                t = plain(c.text()) if c.isVisible() else ""
                if t and t not in seen and t != w.windowTitle():
                    seen.append(t)
            if seen:
                d["text"] = " / ".join(seen[:6])[:300]
        dialogs.append(d)
    if dialogs:
        out["on_top"] = dialogs
    return out


# ---------------------------------------------------------------- input
def window_at(gp: QPoint):
    """The top-level window a press at this screen point reaches (an open popup takes every press, as in Qt)."""
    popup = QApplication.activePopupWidget()
    if popup is not None:
        return popup
    for w in reversed(tops()):
        if w.windowType() == Qt.ToolTip:
            continue
        if QRect(w.mapToGlobal(QPoint(0, 0)), w.size()).contains(gp):
            return w
    modal = QApplication.activeModalWidget()
    return modal if modal is not None else MAIN


def local(w: QWidget, x: int, y: int) -> QPoint:
    return w.mapFromGlobal(origin() + QPoint(int(x), int(y)))


BUTTONS = {"left": Qt.LeftButton, "right": Qt.RightButton, "middle": Qt.MiddleButton}
MOUSE = {"pos": QPoint(0, 0)}


DRAG: dict = {}                                    # the drag in progress (a block from the palette, a card…)


def _drag_exec(self, *a, **k):
    """QDrag.exec, done by the session: the platform without a screen has no drag and drop of its own, and a real one
    would follow the system's cursor, not the session's. The drag goes where the session's mouse goes; the release
    drops it on the widget under it, which accepts or refuses it as it would a real drop."""
    from PySide6.QtCore import QEventLoop
    acts = a[0] if a else k.get("supportedActions", Qt.MoveAction)
    loop = QEventLoop()
    DRAG.clear()
    DRAG.update(drag=self, mime=self.mimeData(), actions=acts, loop=loop, target=None, accepted=False,
                action=Qt.IgnoreAction, pixmap=self.pixmap(), hot=self.hotSpot())
    print("drag started", type(self.source()).__name__, flush=True)
    loop.exec()
    result = DRAG.get("result", Qt.IgnoreAction)
    print("drag ended", result, type(DRAG.get("target")).__name__, flush=True)
    DRAG.clear()
    return result


QtGui.QDrag.exec = _drag_exec
if hasattr(QtGui.QDrag, "exec_"):
    QtGui.QDrag.exec_ = _drag_exec


def drop_target(gp: QPoint):
    top = window_at(gp)
    w = top.childAt(top.mapFromGlobal(gp)) or top
    while w is not None and not w.acceptDrops():
        w = w.parentWidget()
    return w


def drag_to(gp: QPoint, release: bool = False):
    mime, acts = DRAG["mime"], DRAG["actions"]
    w = drop_target(gp)
    if w is not DRAG.get("target"):
        old = DRAG.get("target")
        if old is not None:
            QApplication.sendEvent(old, QDragLeaveEvent())
        DRAG["target"], DRAG["accepted"] = w, False
        if w is not None:
            ev = QDragEnterEvent(w.mapFromGlobal(gp), acts, mime, Qt.LeftButton, Qt.NoModifier)
            QApplication.sendEvent(w, ev)
            DRAG["accepted"] = ev.isAccepted()
    if w is not None and DRAG["accepted"]:
        ev = QDragMoveEvent(w.mapFromGlobal(gp), acts, mime, Qt.LeftButton, Qt.NoModifier)
        QApplication.sendEvent(w, ev)
        DRAG["move_ok"] = ev.isAccepted()
        DRAG["action"] = ev.dropAction()
    if release:
        result = Qt.IgnoreAction
        if w is not None and DRAG["accepted"] and DRAG.get("move_ok", True):
            ev = QDropEvent(QPointF(w.mapFromGlobal(gp)), acts, mime, Qt.LeftButton, Qt.NoModifier)
            QApplication.sendEvent(w, ev)
            if ev.isAccepted():
                result = ev.dropAction()
        elif w is not None:
            QApplication.sendEvent(w, QDragLeaveEvent())
        if result == Qt.IgnoreAction:
            NOTES.append("The drag was dropped where nothing takes it.")
        DRAG["result"] = result
        DRAG["loop"].quit()


def _release_after_drag(gp, button, mods):
    w = window_at(gp)
    QTest.mouseRelease(w.windowHandle(), button, mods, w.mapFromGlobal(gp))


def mouse(kind: str, x, y, button=Qt.LeftButton, mods=Qt.NoModifier):
    gp = origin() + QPoint(int(x), int(y))
    if os.environ.get("TESTER_DEBUG") == "1":
        print("mouse", kind, x, y, bool(DRAG), flush=True)
    if DRAG.get("loop") is not None and "result" not in DRAG and kind in ("move", "release"):
        MOUSE["pos"] = QPoint(int(x), int(y))
        if kind == "release":                         # the button comes up for Qt too (its own state of the mouse)
            drag_to(gp, release=True)
            QTimer.singleShot(0, lambda: _release_after_drag(gp, button, mods))
        else:
            drag_to(gp)
        return
    w = window_at(gp)
    win = w.windowHandle()
    lp = w.mapFromGlobal(gp)
    MOUSE["pos"] = QPoint(int(x), int(y))
    if kind == "move":
        QTest.mouseMove(win, lp)
    elif kind == "press":
        activate(w)
        QTest.mousePress(win, button, mods, lp)
    elif kind == "release":
        QTest.mouseRelease(win, button, mods, lp)
    elif kind == "dclick":
        activate(w)
        QTest.mouseDClick(win, button, mods, lp)


KEYMAP = {"enter": "Return", "return": "Return", "esc": "Escape", "escape": "Escape", "del": "Delete",
          "delete": "Delete", "backspace": "Backspace", "space": "Space", "tab": "Tab", "pgup": "PgUp",
          "pgdown": "PgDown", "pageup": "PgUp", "pagedown": "PgDown", "ins": "Ins", "insert": "Ins",
          "ctrl": "Ctrl", "control": "Ctrl", "shift": "Shift", "alt": "Alt", "up": "Up", "down": "Down",
          "left": "Left", "right": "Right", "home": "Home", "end": "End", "plus": "+", "minus": "-"}


def activate(w):
    """A press on a window, or a key with no window active, makes it the active one, as the system would (without a
    screen, a closed dialog leaves no window active and the menus' shortcuts would not answer)."""
    if w is None or w.windowType() in (Qt.Popup, Qt.ToolTip):
        return
    if QApplication.activeWindow() is not w:
        w.activateWindow()
        QApplication.setActiveWindow(w) if hasattr(QApplication, "setActiveWindow") else None
        QApplication.processEvents()


def key_target():
    for w in (QApplication.activePopupWidget(), QApplication.activeModalWidget(), QApplication.activeWindow()):
        if w is not None:
            return w
    activate(MAIN)
    return MAIN


def press_key(combo: str):
    parts = [p for p in re.split(r"\+(?!$)", combo.strip()) if p]
    names = []
    for p in parts:
        k = KEYMAP.get(p.lower())
        names.append(k if k else (p.upper() if len(p) > 1 and p.lower().startswith("f") and p[1:].isdigit()
                                  else p.upper() if len(p) == 1 else p.capitalize()))
    seq = QtGui.QKeySequence("+".join(names))
    if seq.count() == 0:
        raise ValueError(f"unknown key: {combo}")
    kc = seq[0]
    w = key_target()
    QTest.keyClick(w.windowHandle(), kc.key(), kc.keyboardModifiers())


def type_text(text: str):
    """Each character as a key press and release with its text, to the widget that has the keyboard focus at that
    moment (a completer's list may open meanwhile)."""
    from PySide6.QtCore import QEvent
    from PySide6.QtGui import QKeyEvent
    for ch in text:
        w = QApplication.focusWidget() or key_target()
        if ch == " ":
            key = Qt.Key_Space
        elif ch.isascii() and ch.isalnum():
            key = ord(ch.upper())
        else:
            key = 0
        QApplication.sendEvent(w, QKeyEvent(QEvent.KeyPress, key, Qt.NoModifier, ch))
        w = QApplication.focusWidget() or w
        QApplication.sendEvent(w, QKeyEvent(QEvent.KeyRelease, key, Qt.NoModifier, ch))
        QApplication.processEvents()


def wheel(x, y, delta, ctrl=False):
    gp = origin() + QPoint(int(x), int(y))
    top = window_at(gp)
    target = top.childAt(top.mapFromGlobal(gp)) or top
    w = target
    while w is not None:                              # as the system's wheel: up the parents until one takes it
        lp = w.mapFromGlobal(gp)
        ev = QWheelEvent(QPointF(lp), QPointF(gp), QPoint(0, 0), QPoint(0, int(delta)), Qt.NoButton,
                         Qt.ControlModifier if ctrl else Qt.NoModifier, Qt.NoScrollPhase, False)
        QApplication.sendEvent(w, ev)
        if ev.isAccepted() or w.isWindow():
            break
        w = w.parentWidget()


def drop_files(x, y, paths):
    gp = origin() + QPoint(int(x), int(y))
    w = drop_target(gp)
    if w is None:
        NOTES.append("Nothing at that point takes dropped files.")
        return
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(p) for p in paths])
    lp = w.mapFromGlobal(gp)
    acts = Qt.CopyAction | Qt.MoveAction | Qt.LinkAction
    enter = QDragEnterEvent(lp, acts, mime, Qt.LeftButton, Qt.NoModifier)
    QApplication.sendEvent(w, enter)
    if not enter.isAccepted():
        QApplication.sendEvent(w, QDragLeaveEvent())
        NOTES.append("The files were refused at that point (the place does not take them).")
        return
    mv = QDragMoveEvent(lp, acts, mime, Qt.LeftButton, Qt.NoModifier)
    QApplication.sendEvent(w, mv)
    if not mv.isAccepted():
        QApplication.sendEvent(w, QDragLeaveEvent())
        NOTES.append("The files were refused at that point (the place does not take them).")
        return
    QApplication.sendEvent(w, QDropEvent(QPointF(lp), mv.dropAction() or Qt.CopyAction, mime, Qt.LeftButton,
                                         Qt.NoModifier))


# ---------------------------------------------------------------- commands
def steps_for(cmd: str, a: list[str]):
    """[(function, ms to wait after it)] and the settle timeout (s)."""
    if cmd in ("shot", "mark"):
        return [], 3
    if cmd == "move":
        x, y = a[0], a[1]
        return [(lambda: mouse("move", int(x) - 3, int(y)), 30), (lambda: mouse("move", x, y), 1200)], 4
    if cmd == "click":
        x, y = int(a[0]), int(a[1])
        btn = BUTTONS.get(a[2] if len(a) > 2 and a[2] in BUTTONS else "left")
        double = "double" in a[2:]
        if double:
            return [(lambda: mouse("move", x, y), 40), (lambda: mouse("dclick", x, y, btn), 250)], 6
        return [(lambda: mouse("move", x, y), 40), (lambda: mouse("press", x, y, btn), 80),
                (lambda: mouse("release", x, y, btn), 250)], 6
    if cmd in ("press", "release"):                  # a drag in several commands: press, move…, release
        x, y = int(a[0]), int(a[1])
        btn = BUTTONS.get(a[2] if len(a) > 2 and a[2] in BUTTONS else "left")
        return [(lambda: mouse("move", x, y), 60), (lambda: mouse(cmd, x, y, btn), 400)], 6
    if cmd == "drag":
        x1, y1, x2, y2 = (int(v) for v in a[:4])
        n = max(int(a[4]) if len(a) > 4 else 15, 2)
        st = [(lambda: mouse("move", x1, y1), 60), (lambda: mouse("press", x1, y1), 150)]
        for i in range(1, n + 1):
            xi = round(x1 + (x2 - x1) * i / n)
            yi = round(y1 + (y2 - y1) * i / n)
            st.append((lambda xi=xi, yi=yi: mouse("move", xi, yi), 40))
        st.append((lambda: mouse("move", x2, y2), 150))
        st.append((lambda: mouse("release", x2, y2), 300))
        return st, 6
    if cmd == "wheel":
        x, y, d = int(a[0]), int(a[1]), int(a[2])
        ctrl = "ctrl" in [v.lower() for v in a[3:]]
        return [(lambda: mouse("move", x, y), 40), (lambda: wheel(x, y, d, ctrl), 300)], 5
    if cmd == "key":
        return [(lambda: press_key(a[0]), 300)], 6
    if cmd == "type":
        return [(lambda: type_text(" ".join(a)), 300)], 5
    if cmd in ("choose_file", "choose_folder"):
        if ANSWERS:
            NOTES.append("The answer prepared before, never used, is replaced.")
        ANSWERS[:] = [("file" if cmd == "choose_file" else "folder", list(a))]
        NOTES.append("The next file dialog will be answered with: " + "; ".join(a))
        return [], 2
    if cmd == "drop_files":
        x, y, paths = int(a[0]), int(a[1]), list(a[2:])
        return [(lambda: mouse("move", x, y), 40), (lambda: drop_files(x, y, paths), 400)], 8
    if cmd == "resize":
        w, h = (int(v) for v in a[0].lower().split("x"))
        return [(lambda: (MAIN.showNormal(), MAIN.resize(w, h)), 500)], 5
    if cmd == "open":
        return [(lambda: MAIN.open_project(a[0]), 1500)], 30
    if cmd == "_eval" and os.environ.get("TESTER_DEBUG") == "1":      # the tool's own checks only
        def ev():
            NOTES.append(repr(eval(" ".join(a), {**globals(), "W": MAIN})))
        return [(ev, 300)], 3
    if cmd == "_find" and os.environ.get("TESTER_DEBUG") == "1":      # the tool's own checks: a button's place
        def find():
            want = " ".join(a).strip().rstrip(".…")
            for top in [*reversed([t for t in tops() if t.windowType() != Qt.ToolTip]), MAIN]:
                for b in top.findChildren(QAbstractButton):
                    if b.isVisible() and plain(b.text()).strip().rstrip(".…") == want:
                        c = b.mapToGlobal(b.rect().center()) - origin()
                        NOTES.append(f"at {c.x()} {c.y()}")
                        return
            NOTES.append("not found")
        return [(find, 100)], 2
    if cmd == "wait":
        s = min(max(float(a[0]) if a else 1.0, 0.0), 120.0)
        return [(lambda: None, int(s * 1000))], 3
    raise ValueError(f"unknown command: {cmd}")


class Job:
    """One command: its steps on timers (each next step scheduled before the step runs), then the settle, then the
    answer."""

    def __init__(self, sock, req):
        self.sock, self.req = sock, req
        self.cmd = req.get("cmd", "")
        self.args = [str(x) for x in req.get("args", [])]
        self.t0 = time.time()
        self.error = ""
        try:
            self.steps, self.settle_max = steps_for(self.cmd, self.args)
        except Exception as e:  # noqa: BLE001 - said back to the tester
            self.steps, self.settle_max, self.error = [], 1, str(e)
        self.i = 0
        QTimer.singleShot(0, self.tick)

    def tick(self):
        if self.i >= len(self.steps):
            self.last_sig, self.same, self.settle_t0 = None, 0, time.time()
            QTimer.singleShot(150, self.settle)
            return
        fn, after = self.steps[self.i]
        self.i += 1
        QTimer.singleShot(after, self.tick)          # before the step: a step that runs a loop of its own is followed
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            self.error = f"{type(e).__name__}: {e}"
            traceback.print_exc()

    def signature(self):
        img = compose()
        return hashlib.md5(bytes(img.constBits())).hexdigest(), img

    def settle(self):
        try:
            self._settle()
        except Exception:  # noqa: BLE001 - never a session left without an answer
            traceback.print_exc()
            self.error = self.error or "the screenshot failed"
            img = QImage(MAIN.size(), QImage.Format_RGB32)
            img.fill(QColor("#000000"))
            self.answer(img)

    def _settle(self):
        sig, img = self.signature()
        if sig == self.last_sig:
            self.same += 1
        else:
            self.same, self.last_sig = 0, sig
        if self.same >= 2 or time.time() - self.settle_t0 > self.settle_max:
            self.answer(img)
        else:
            QTimer.singleShot(200, self.settle)

    def answer(self, img):
        shot = next_shot()
        img.save(str(shot))
        rec = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "t": round(time.time() - STATE["start"], 1),
               "cmd": self.cmd, "args": self.args, "shot": shot.as_posix()}
        texts = visible_texts()
        notes = list(NOTES)
        NOTES.clear()
        if self.error:
            notes.append("Error: " + self.error)
        if notes:
            texts["notes"] = notes
        rec["seconds"] = round(time.time() - self.t0, 1)
        journal(rec)
        reply = {"shot": shot.as_posix(), **texts}
        send(self.sock, reply)
        STATE["busy"] = False
        STATE["last"] = time.time()
        QTimer.singleShot(0, run_pending)


def send(sock, obj):
    try:
        sock.write(QByteArray((json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")))
        sock.flush()
        sock.disconnectFromHost()
    except RuntimeError:
        pass


def descendants():
    try:
        import psutil
        return psutil.Process(os.getpid()).children(recursive=True)
    except Exception:  # noqa: BLE001
        return []


def quit_session(sock=None, why="quit"):
    if STATE["quitting"]:
        return
    STATE["quitting"] = True
    kids = descendants()
    journal({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "t": round(time.time() - STATE["start"], 1),
             "cmd": "quit", "args": [why], "shot": ""})
    try:
        for w in tops():
            if isinstance(w, QDialog):
                w.reject()
        MAIN.close()
    except Exception:  # noqa: BLE001
        traceback.print_exc()
    kids += [k for k in descendants() if k.pid not in {x.pid for x in kids}]
    for k in kids:                                   # the run workers this session started, and their children
        try:
            k.kill()
        except Exception:  # noqa: BLE001
            pass
    if sock is not None:
        send(sock, {"shot": "", "title": "session ended", "notes": [f"{len(kids)} worker process(es) stopped"]})
    for f in ("port",):
        try:
            (SDIR / f).write_text("", encoding="utf-8")
        except OSError:
            pass
    QTimer.singleShot(300, qapp.quit)
    QTimer.singleShot(5000, lambda: os._exit(0))


def run_pending():
    if STATE["busy"] or not PENDING or STATE["quitting"]:
        return
    sock, req = PENDING.pop(0)
    STATE["busy"] = True
    STATE["last"] = time.time()
    if req.get("cmd") == "quit":
        quit_session(sock)
        return
    STATE["job"] = Job(sock, req)                   # held: the timers do not keep it alive


SERVER = QTcpServer()
BUF: dict = {}


def on_conn():
    while SERVER.hasPendingConnections():
        s = SERVER.nextPendingConnection()
        BUF[id(s)] = b""

        def read(s=s):
            BUF[id(s)] += bytes(s.readAll())
            if b"\n" in BUF[id(s)]:
                line = BUF.pop(id(s)).split(b"\n", 1)[0]
                try:
                    req = json.loads(line.decode("utf-8"))
                except ValueError:
                    send(s, {"shot": "", "notes": ["Error: unreadable command"]})
                    return
                PENDING.append((s, req))
                run_pending()
        s.readyRead.connect(read)


def watchdog():
    if not STATE["busy"] and time.time() - STATE["last"] > ARGS.idle_minutes * 60:
        quit_session(None, "no command for %d minutes" % ARGS.idle_minutes)


def setup():
    global MAIN
    for w in QApplication.topLevelWidgets():
        if isinstance(w, float_app.Main):
            MAIN = w
    if MAIN is None:
        QTimer.singleShot(100, setup)
        return
    if ARGS.onscreen:
        MAIN.showNormal()
        MAIN.resize(W, H)
        MAIN.move(40, 40)
    else:
        MAIN.showMaximized()                          # the screen is the session's size: maximised, as on a laptop
    MAIN.activateWindow()
    SERVER.newConnection.connect(on_conn)
    if not SERVER.listen(QHostAddress.LocalHost, 0):
        print("cannot listen", file=sys.stderr)
        os._exit(2)
    (SDIR / "pid").write_text(str(os.getpid()), encoding="utf-8")
    (SDIR / "port").write_text(str(SERVER.serverPort()), encoding="utf-8")
    t = QTimer(qapp)
    t.timeout.connect(watchdog)
    t.start(30000)
    qapp._tester_watchdog = t
    journal({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "t": 0.0, "cmd": "start",
             "args": [ARGS.size, ARGS.project], "shot": ""})


QTimer.singleShot(0, setup)
code = float_app.main([ARGS.project] if ARGS.project else [])
for k in descendants():
    try:
        k.kill()
    except Exception:  # noqa: BLE001
        pass
os._exit(code or 0)
