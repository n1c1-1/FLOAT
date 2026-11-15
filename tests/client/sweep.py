"""The sweep's bench (tests/client/test_sweep_controls.py, test_sweep_post.py): every control a hand can reach, driven
one by one, each named by a readable path ("graph/Denoise/Method/Pretrained + adapted"), under one oracle: no uncaught
exception (sys.excepthook, Qt's critical messages, a ``NotPorted``), no message box, the window still answering, and
after Reset or Undo the project's definition as it was. What cannot be driven is listed with its reason, never skipped
silently."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import traceback

from PySide6.QtCore import QtMsgType, qInstallMessageHandler
from PySide6.QtWidgets import (
    QAbstractButton,
    QAbstractSlider,
    QAbstractSpinBox,
    QApplication,
    QComboBox,
    QDialog,
    QFileDialog,
    QLineEdit,
    QMenu,
    QMessageBox,
    QScrollBar,
    QWidget,
)

#: the window must come back to its event loop within this many seconds after a gesture
ANSWER_S = 20.0
#: a gesture not back after this many seconds is a hang: the stacks are written and the run ends
HANG_S = 90.0
#: the file that names the gesture being driven (read after a hang)
HANG_NOTE = os.environ.get("FLOAT_SWEEP_NOTE") or os.path.join(tempfile.gettempdir(), "float_sweep_current.txt")


class Oracle:
    """Records what must never reach the user while it is installed: uncaught exceptions, Qt's critical messages,
    message boxes. File dialogs answer "cancelled", modal dialogs are closed at once (listed, not failures)."""

    def __init__(self, monkeypatch):
        self.errors: list[str] = []
        self.dialogs: list[str] = []
        self._old_hook = sys.excepthook
        sys.excepthook = self._hook
        self._old_qt = qInstallMessageHandler(self._qt)
        mp = monkeypatch

        def box(kind):
            def f(*a, **k):
                self.errors.append(f"a message box ({kind}): {' | '.join(str(x) for x in a[1:3])}")
                return QMessageBox.Cancel if kind == "question" else QMessageBox.Ok
            return f
        for kind in ("information", "warning", "critical", "question", "about"):
            mp.setattr(QMessageBox, kind, box(kind))

        def box_exec(mb, *a, **k):
            self.errors.append(f"a message box: {mb.text()[:120]}")
            return 0
        mp.setattr(QMessageBox, "exec", box_exec)
        mp.setattr(QMessageBox, "open", lambda mb, *a, **k: box_exec(mb))

        def dlg_exec(d, *a, **k):
            self.dialogs.append(type(d).__name__)
            QApplication.processEvents()
            d.reject()
            return 0
        mp.setattr(QDialog, "exec", dlg_exec)
        # a menu's or a dialog's own loop (a class attribute set from Python does not reach a call of the class' own
        # method): every popup and modal window closed by a timer of that loop, a message box being a failure
        from PySide6.QtCore import QTimer
        self._closer = QTimer()
        self._closer.timeout.connect(self._close_popups)
        self._closer.start(40)
        from PySide6.QtWidgets import QColorDialog, QInputDialog

        def asked(what, back):
            def f(*a, **k):
                self.dialogs.append(what)
                return back
            return staticmethod(f)
        for f, back in (("getText", ("", False)), ("getInt", (0, False)), ("getDouble", (0.0, False)),
                        ("getItem", ("", False)), ("getMultiLineText", ("", False))):
            mp.setattr(QInputDialog, f, asked(f"QInputDialog.{f}", back))
        from PySide6.QtGui import QColor
        mp.setattr(QColorDialog, "getColor", asked("QColorDialog.getColor", QColor()))
        for f in ("getOpenFileName", "getSaveFileName"):
            mp.setattr(QFileDialog, f, staticmethod(lambda *a, **k: ("", "")))
        mp.setattr(QFileDialog, "getOpenFileNames", staticmethod(lambda *a, **k: ([], "")))
        mp.setattr(QFileDialog, "getExistingDirectory", staticmethod(lambda *a, **k: ""))

    def _close_popups(self):
        w = QApplication.activePopupWidget()
        if w is not None:
            w.close()
        d = QApplication.activeModalWidget()
        if d is not None:
            if isinstance(d, QMessageBox):
                self.errors.append(f"a message box: {d.text()[:120]}")
            else:
                self.dialogs.append(type(d).__name__)
            if isinstance(d, QDialog):
                d.reject()
            else:
                d.close()

    def _hook(self, t, v, tb):
        self.errors.append("".join(traceback.format_exception(t, v, tb)))

    def _qt(self, mode, ctx, msg):
        if mode in (QtMsgType.QtCriticalMsg, QtMsgType.QtFatalMsg):
            self.errors.append(f"Qt: {msg}")

    def close(self):
        self._closer.stop()
        sys.excepthook = self._old_hook
        qInstallMessageHandler(self._old_qt)

    def take(self) -> list[str]:
        out, self.errors = self.errors, []
        return out


def first_line(err: str) -> str:
    """The line of an error that names it (the exception's last line)."""
    lines = [x for x in err.strip().splitlines() if x.strip()]
    return lines[-1].strip() if lines else err


def text_of(w: QWidget) -> str:
    """A control's name as a reader sees it: its text, else its accessible name, its tooltip's start, its class."""
    for f in ("text", "currentText"):
        t = getattr(w, f, None)
        if callable(t):
            try:
                t = t()
            except (TypeError, RuntimeError):
                t = ""
            if isinstance(t, str) and t.strip():
                return " ".join(t.split())[:40]
    for t in (w.accessibleName(), w.toolTip(), w.objectName()):
        if t and t.strip():
            return " ".join(t.split())[:40]
    return type(w).__name__


def row_name(w: QWidget, root: QWidget) -> str:
    """The setting row a control sits in (its name), "" when none."""
    from astro_gpu.client.panel import CompactRow
    p = w.parentWidget()
    while p is not None and p is not root:
        if isinstance(p, CompactRow):
            r = p.r
            return str(r.get("label") or r.get("title") or r.get("name") or r.get("key") or "")
        p = p.parentWidget()
    return ""


def inside(w: QWidget, others) -> bool:
    p = w
    while p is not None:
        if any(p is o for o in others):
            return True
        p = p.parentWidget()
    return False


def controls(root: QWidget, skip=()) -> list[tuple[str, QWidget]]:
    """Every control of ``root`` a hand can reach (shown, enabled), by readable name (unique: "#2" when repeated)."""
    from astro_gpu.client.panel import MethodCard, StackMethodCard
    out, seen = [], {}
    for w in root.findChildren(QWidget):
        try:
            if not w.isVisibleTo(root) or not w.isEnabled() or inside(w, skip):
                continue
        except RuntimeError:
            continue
        if isinstance(w, (MethodCard, StackMethodCard)):
            pass
        elif isinstance(w, QAbstractButton):
            if isinstance(w.parentWidget(), (QAbstractSpinBox, QComboBox)):
                continue
        elif isinstance(w, (QComboBox, QAbstractSpinBox)):
            pass
        elif isinstance(w, QAbstractSlider):
            if isinstance(w, QScrollBar):
                continue
        elif isinstance(w, QLineEdit):
            if isinstance(w.parentWidget(), (QAbstractSpinBox, QComboBox)) or w.isReadOnly():
                continue
        else:
            continue
        rn = row_name(w, root)
        name = (rn + "/" if rn else "") + text_of(w)
        n = seen.get(name, 0) + 1
        seen[name] = n
        out.append((name if n == 1 else f"{name}#{n}", w))
    return out


def gestures(w: QWidget) -> list[tuple[str, object]]:
    """The gestures of one control: (name, do) — a click, each other entry, a step up and down, a move, a value."""
    from astro_gpu.client.panel import MethodCard, StackMethodCard
    if isinstance(w, (MethodCard, StackMethodCard)):
        return [("click", lambda: w.mousePressEvent(None))]
    if isinstance(w, QAbstractButton):
        return [("click", w.click)]
    if isinstance(w, QComboBox):
        cur = w.currentIndex()
        out = []
        for i in range(w.count()):
            if i == cur:
                continue
            it = w.model().item(i) if hasattr(w.model(), "item") else None
            if it is not None and not it.isEnabled():
                continue
            out.append((w.itemText(i)[:30] or f"entry {i}", lambda i=i: w.setCurrentIndex(i)))
        return out

    if isinstance(w, QAbstractSpinBox):
        def step(n):
            def go():
                w.stepBy(n)
                w.editingFinished.emit()
            return go
        return [("up", step(1)), ("down", step(-1))]
    if isinstance(w, QAbstractSlider):
        def move():
            v = w.value() + max(w.pageStep(), 1)
            w.setValue(v if v <= w.maximum() else w.value() - max(w.pageStep(), 1))
            w.sliderReleased.emit()
        return [("move", move)]
    if isinstance(w, QLineEdit):
        def typed():
            w.setText(w.text())
            w.returnPressed.emit()
            w.editingFinished.emit()
        return [("value", typed)]
    return []


def definition(m) -> str:
    """The project's definition as the model holds it (the graph), as text."""
    from astro_gpu.graphcore import model as gm
    g = getattr(m, "_g", None)
    if g is None:
        return ""
    d = gm.to_dict(g)
    return json.dumps(d, sort_keys=True, default=str)


class Sweep:
    """One area's sweep: ``drive(path, do)`` runs a gesture under the oracle; failures, driven count, not driven."""

    def __init__(self, oracle: Oracle, settle, area: str):
        self.o, self.settle, self.area = oracle, settle, area
        self.driven = 0
        self.failures: list[tuple[str, str]] = []
        self.not_driven: list[tuple[str, str]] = []

    def drive(self, path: str, do) -> bool:
        import faulthandler
        self.driven += 1
        t0 = time.monotonic()
        # a gesture that never comes back (a modal loop): every stack written to stderr, the gesture named first
        sys.stderr.write("")
        faulthandler.dump_traceback_later(HANG_S, exit=True)
        self.current = f"{self.area}/{path}"
        with open(HANG_NOTE, "w", encoding="utf-8") as fh:
            fh.write(self.current)
        try:
            do()
        except RuntimeError as e:
            last = traceback.extract_tb(e.__traceback__)[-1].filename.replace("\\", "/")
            if "already deleted" in str(e) and "/tests/" in last:  # the bench held a control built again: not driven
                faulthandler.cancel_dump_traceback_later()
                self.driven -= 1
                self.skip(path, "the control was built again before its gesture")
                return True
            self.o.errors.append("".join(traceback.format_exception(type(e), e, e.__traceback__)))
        except Exception as e:                       # noqa: BLE001 - raised to the gesture: the user's crash
            self.o.errors.append("".join(traceback.format_exception(type(e), e, e.__traceback__)))
        try:
            self.settle()
        except Exception as e:                       # noqa: BLE001
            self.o.errors.append("".join(traceback.format_exception(type(e), e, e.__traceback__)))
        faulthandler.cancel_dump_traceback_later()
        if time.monotonic() - t0 > ANSWER_S:
            self.o.errors.append(f"the window did not answer for {time.monotonic() - t0:.0f} s")
        errs = self.o.take()
        for e in errs:
            self.failures.append((f"{self.area}/{path}", e))
        return not errs

    def fail(self, path: str, what: str):
        self.failures.append((f"{self.area}/{path}", what))

    def skip(self, path: str, why: str):
        self.not_driven.append((f"{self.area}/{path}", why))

    def report(self) -> str:
        lines = [f"== {self.area}: {self.driven} gestures driven, {len(self.failures)} failures, "
                 f"{len(self.not_driven)} not driven"]
        lines += [f"  NOT DRIVEN {p}: {why}" for p, why in self.not_driven]
        seen = set()
        for p, e in self.failures:
            key = (p, first_line(e))
            if key in seen:
                continue
            seen.add(key)
            lines.append(f"  FAIL {p}: {first_line(e)}")
        return "\n".join(lines)

    def full(self) -> str:
        """Each distinct failure's whole text (the first one per error line)."""
        out, seen = [], set()
        for p, e in self.failures:
            if first_line(e) in seen:
                continue
            seen.add(first_line(e))
            out.append(f"--- {p}\n{e}")
        return "\n".join(out)
