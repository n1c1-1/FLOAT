"""The editor of a block of the combination family in Inspect, the Gradient editor's way (gradientedit.py): the shared
part of a combination's editor (combineedit.CombineEditor) and of Combine layers' (layersedit.LayersEditor), so that a
user who knows one finds his way in the other and a block of the family to come is built the same.

What every such editor is, from this base:

* a DRAFT of the block's settings per workflow while the project is open (blockdraft.Draft): the project does not
  change, the graph stays up to date; each edit one step of the window's one Undo order (a slider dragged: one step,
  let go); a setting back to its applied value leaves the draft; a block without a result takes a change at once;
* the page in Inspect's centre: the title row (the block's name, "Previous settings" when its result is outdated, the
  comparison Draft / Applied / Split), the editor's column at the left (its settings, then the draft bar pinned at the
  foot: blockdraft.DraftBar, Apply / Try variant / Reset), the image at the right with what it shows in a tag, a cover
  saying why there is nothing to show (an input to wire, not computed yet, with its Run to here), and the settings
  panel at the right of the image (layout.Dock, as the Gradient's: the draft's Undo and Redo, the measurements, the
  display);
* the preview: computed in a thread of its own, one at a time, the last edit only, the previous image kept until the
  next is ready (``_work`` makes it, on a reduced copy: about gradientedit.PREVIEW_PIXELS);
* leaving the block with a draft asks the question every block asks (blockdraft.settle: ``draft_pending``,
  ``draft_apply``, ``draft_variant``, ``draft_reset``); ``history_top``, ``undo``, ``redo`` for the window's Undo.

A subclass says what its block is: ``KEY`` (the view key of its block, "fusion", "composition"), ``SIDE_PREF`` (where
the settings panel's state is kept), ``wants(node, p)``, ``applied(p)``, ``parts(pid)`` (what the draft changes: one
(name, setting, change) each), ``draft_change(pid)``, ``title(q)``, ``heading(q)``, ``has_result(p)``,
``state_of(p)``, ``cost_text(pid)``, ``norm(key, value)``, ``_fill_column(cv)`` (its settings), ``_sync_controls()``,
``_measurements(dv)``, ``_request()`` and ``_work(req)`` (the preview), ``_absorb(req, out)`` (what a preview brings
besides its image), ``missing_text()`` (why there is nothing to preview), ``applied_label()``.
"""
from __future__ import annotations

import copy
import threading
import time

from PySide6.QtCore import QEvent, QObject, Qt, QTimer, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from . import journal
from .blockdraft import Draft, DraftBar, owned
from .i18n import _
from .widgets import WARN, label

#: the column of the editor at the left of the image: its width
PANEL_W = 380
#: the image area's comparisons
COMPARE = (("draft", "Draft", "The draft's preview"), ("applied", "Applied", "The block's applied result"),
           ("split", "Split", "The draft on the right of the divider, the applied result on the left"))


class PressToRelease(QObject):
    """A slider's press-to-release gesture is one drag, wherever it is pressed: the slider is down from the press on,
    before Qt moves its handle there (the Windows style takes a press on the rail to that place: the hand pass's two
    Undo steps, 205 then 267, for one drag), and up again at the release, as Post's sliders (post3.ContSlider)."""

    def __init__(self, slider):
        super().__init__(slider)
        self.sl = slider

    def eventFilter(self, obj, e):  # noqa: N802 - Qt's name
        if obj is self.sl and e.type() == QEvent.MouseButtonPress and e.button() == Qt.LeftButton:
            self.sl.setSliderDown(True)                  # (sliderPressed: the drag starts)
        elif obj is self.sl and e.type() == QEvent.MouseButtonRelease and e.button() == Qt.LeftButton:
            # after Qt's own release, which lets go of a handle but not of a press on the rail's page step
            QTimer.singleShot(0, self, lambda: self.sl.setSliderDown(False) if self.sl.isSliderDown() else None)
        return False


class Clickable(QFrame):
    """A frame that is a choice: a click on it (a method card, a thumbnail)."""

    clicked = Signal()

    def mousePressEvent(self, e):                                     # noqa: N802 - Qt's name
        if e.button() == Qt.LeftButton and self.isEnabled():
            self.clicked.emit()
        super().mousePressEvent(e)


def choice_card(name: str, line: str, tip: str = "") -> Clickable:
    """A method card as Gradient's: its name, one plain sentence under it; lit when chosen (``lit``)."""
    c = Clickable()
    c.setObjectName("combchoice")
    v = QVBoxLayout(c)
    v.setContentsMargins(10, 6, 10, 6)
    v.setSpacing(2)
    n = label(f"<b>{name}</b>")
    n.setTextFormat(Qt.RichText)
    v.addWidget(n)
    ln = label(line, "small", wrap=True)
    ln.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
    v.addWidget(ln)
    c.setToolTip(tip or line)
    c.setCursor(Qt.PointingHandCursor)
    return c


def lit(card: QFrame, on: bool):
    card.setProperty("on", bool(on))
    card.setStyleSheet("QFrame#combchoice { border: 1px solid %s; border-radius: 6px; background: %s; }"
                       % (("#5ea1ff", "#2b3446") if on else ("#3a4150", "transparent")))


class DraftEditor(QObject):
    """A block of the combination family tuned as a draft in Inspect's centre (module docstring). ``center(tv, node,
    p)`` from TuneView.refresh; ``set_draft``, ``apply()``, ``try_variant()``, ``reset()``."""

    KEY = ""                                   # the view key of the block it edits
    SIDE_PREF = ""                             # the preference keeping its settings panel's state
    VARIANT = True                             # Try variant offered (a Post layer's block has none: greyed, why)

    _done = Signal(int, object)
    historyChanged = Signal()

    def __init__(self, tune):
        super().__init__(tune)
        self.tv, self.s = tune, tune.s
        self.p = None
        self.node = None
        self.drafts: dict[int, Draft] = {}
        self.page = self.gv = None
        self._ui: dict = {}
        self.cmp = "draft"
        self.result = None                     # the draft's preview (3, h, w) in 0-1
        self.result_key = None
        self.result_ms = None
        self.error = ""
        self.busy = False
        self._want = None                      # the request waiting for the one running
        self.gen = 0
        self.computed = self.painted = 0       # previews started, previews shown (the tests count them)
        self._drag = None                      # the draft's settings when a slider was pressed
        self._stashed: dict[int, tuple] = {}   # variant -> (original, the draft it took)
        self._syncing = False
        self.timings: list[float] = []         # seconds from an edit to its image shown (the measure of the brief)
        self._t_edit = None
        self._lang = None                      # the language the page was built in (built again in another)
        self._shape = None                     # what the page was built for
        self._cost = None                      # ((workflow, the draft), what Apply computes again)
        self.dock = self.drawer = None         # the settings panel at the right of the image (as the Gradient's)
        self._win = None
        self.applied_cache = None              # the applied picture {"pid", "key", "img", ...}
        self._done.connect(self._arrived)

    # ------------------------------------------------------------ what a subclass says
    def wants(self, node, p) -> bool:
        raise NotImplementedError

    def applied(self, p=None) -> dict:
        raise NotImplementedError

    def parts(self, pid: int | None = None) -> list[tuple[str, str, str]]:
        raise NotImplementedError

    def draft_change(self, pid: int | None = None) -> dict | None:
        raise NotImplementedError

    def title(self, q=None) -> str:
        raise NotImplementedError

    def heading(self, q=None) -> str:
        return self.title(q)

    def has_result(self, p=None) -> bool:
        return self.state_of(p) in ("done", "stale")

    def state_of(self, p=None) -> str:
        raise NotImplementedError

    def shape(self, p) -> tuple:
        """What the page is built for: built again when it changes."""
        return ()

    def norm(self, key: str, value):
        return value

    def cost_text(self, pid: int | None = None) -> str:
        return ""

    def missing_text(self) -> str:
        return ""

    def applied_label(self) -> str:
        return _("Applied result") if self.state_of() == "done" else _("Previous settings")

    def variant_why(self) -> str:
        return ""

    def _fill_column(self, cv):
        pass

    def _sync_controls(self):
        pass

    def _measurements(self, dv):
        pass

    def _drawer_sig(self) -> tuple:
        return ()

    def _request(self) -> dict:
        raise NotImplementedError

    def _work(self, req: dict) -> dict:
        raise NotImplementedError

    def _absorb(self, req: dict, out: dict):
        pass

    def _needs_run(self, q, ch) -> bool:
        return True

    def _paths(self, q):
        return [q]

    # ------------------------------------------------------------ the draft
    def _draft(self, pid: int | None = None) -> Draft | None:
        if pid is None:
            if self.p is None:
                return None
            pid = self.p.pid
        return self.drafts.get(pid)

    def has_draft(self, pid: int | None = None) -> bool:
        return bool(self._draft(pid))

    def value(self, name: str):
        d = self._draft()
        if d is not None and name in d.vals:
            return copy.deepcopy(d.vals[name])
        return copy.deepcopy(self.applied().get(name))

    def values(self) -> dict:
        """The settings as the draft has them (the applied ones where it changes nothing)."""
        out = self.applied()
        d = self._draft()
        if d is not None:
            out.update(copy.deepcopy(d.vals))
        return out

    def set_draft(self, updates: dict, undo: bool = True) -> bool:
        """Settings of the draft: one step of its Undo (``undo``); a setting back to its applied value leaves it."""
        if self.p is None:
            return False
        d = self.drafts.setdefault(self.p.pid, Draft(self.KEY, [self.p.pid]))
        ap = self.applied()
        norm = {k: self.norm(k, v) for k, v in updates.items()}
        if not d.put(norm, same=lambda k, v: v == ap.get(k), step=undo):
            return False
        if undo and not self.has_result():
            # a block without a result (never computed, or waiting for an input): nothing to keep up to date, the
            # change is made at once, as every control of it and every block's (the reviewer's round 2, point 5)
            self.apply(quiet=True)
            return True
        self._changed()
        if undo:
            self.historyChanged.emit()
        return True

    # the slider dragged: the draft follows, one step of Undo when it is let go
    def begin_drag(self):
        d = self._draft()
        self._drag = copy.deepcopy(d.vals) if d is not None else {}

    def end_drag(self):
        before, self._drag = self._drag, None
        d = self._draft()
        if before is None or d is None or d.vals == before:
            return
        d.undo.append(before)
        d.redo.clear()
        d.useq.append(journal.tick())
        d.rseq = []
        if not self.has_result():                   # made at once, the slider let go (``set_draft``)
            self.apply(quiet=True)
            return
        self.historyChanged.emit()
        self._sync()                                # (the panel's Undo, now that the drag is a step)

    def _changed(self):
        self._sync()
        self.preview()

    def undo(self, pid: int | None = None) -> bool:
        return self._move(pid, "undo")

    def redo(self, pid: int | None = None) -> bool:
        return self._move(pid, "redo")

    def _move(self, pid, how: str) -> bool:
        d = self._draft(pid)
        if d is None or d.move(how) is None:
            return False
        if pid is None or (self.p is not None and pid == self.p.pid):
            self._changed()
        self.historyChanged.emit()
        return True

    def history_top(self, how: str = "undo") -> tuple[int, int] | None:
        best = None
        for pid, d in self.drafts.items():
            seq = d.top(how)
            if seq is not None and (best is None or seq > best[0]):
                best = (seq, pid)
        return best

    def _view_of(self, pid):
        if pid is None or (self.p is not None and pid == self.p.pid):
            return self.p
        return self.s.view(pid)

    def changes(self, pid: int | None = None) -> list[tuple[str, str]]:
        """What the draft changes, one (name, words) per setting, as the draft bar lists them ("Hα hue 31° → 50°")."""
        return [(k, f"{what} {change}") for k, what, change in self.parts(pid)]

    def step_words(self, q) -> tuple[str, str]:
        """(the step's words, named as every step of the family: blockdraft.owned; a short form for a variant's name)."""
        parts = self.parts(q.pid)
        words = [f"{what} {change}" for _k, what, change in parts]
        short = ", ".join(words) if len(words) <= 2 else _("{n} changes").format(n=len(words))
        what = owned(parts[0][1], self.title(q), parts[0][2]) if len(parts) == 1 else \
            owned(_("{n} settings").format(n=len(parts)), self.title(q), ", ".join(words))
        return what, short

    def apply(self, pid: int | None = None, quiet: bool = False):
        """The draft becomes the block's settings: one step of the window's Undo. ``quiet``: a change made at once on
        a block without a result (no draft was shown: nothing said; refused, the change goes)."""
        q = self._view_of(pid)
        ch = self.draft_change(q.pid) if q is not None else None
        if ch is None:
            return
        from ..core import workflow as wfl
        from .model import NotInEngine
        n = self.s.steps_count() if hasattr(self.s, "steps_count") else None
        try:
            self._apply_change(q, ch)
        except (ValueError, NotInEngine, wfl.OptionRefused) as e:
            self.tv.toasted.emit(_(str(e)))
            if quiet:
                self.drafts.pop(q.pid, None)
                self._changed()
            return
        if hasattr(self.s, "log"):
            self.s.log("{what}", what=ch["what"])
        if n is not None:                               # the step's words, whatever the graph's own said first
            self.s.name_steps_since(n, ch["what"])
        run = self._needs_run(q, ch)
        self.drafts.pop(q.pid, None)
        if quiet:
            self._changed()
        else:                                   # what was done, named as every step of the family; Run when it needs one
            self.tv.toasted.emit(" · ".join([ch["what"]] + ([_("Run to here computes it")] if run else []) +
                                            ["Ctrl+Z"]))
        self.historyChanged.emit()
        self.tv.request_refresh()

    def _apply_change(self, q, ch):
        if hasattr(self.s, "apply_as"):
            self.s.apply_as(ch, "replace", self._paths(q))
        else:
            self.s.apply_change(self._paths(q), ch)

    def try_variant(self, pid: int | None = None):
        """The draft on a variant of the workflow (what is computed stays as it is); the variant is shown."""
        q = self._view_of(pid)
        ch = self.draft_change(q.pid) if q is not None else None
        if ch is None or self.variant_why():
            return
        from ..core import workflow as wfl
        from .model import NotInEngine
        title = f"{q.name} · {ch['short']}"
        try:
            v = (self.s.apply_as(ch, "variant", [q], title) if hasattr(self.s, "apply_as")
                 else self.s.add_variant(q, ch, title))
        except (ValueError, NotInEngine, wfl.OptionRefused) as e:
            self.tv.toasted.emit(_(str(e)))
            return
        d = self.drafts.pop(q.pid, None)
        self._stashed[v.pid] = (q.pid, copy.deepcopy(d))       # an Undo of the variant gives the draft back
        self.historyChanged.emit()
        self.tv._variant_made(v.pid, q.pid)

    def reset(self, pid: int | None = None):
        q = self._view_of(pid)
        if q is None or q.pid not in self.drafts:
            return
        self.drafts.pop(q.pid, None)
        self.historyChanged.emit()
        if self.p is not None and q.pid == self.p.pid:
            self._changed()

    # the question every block asks before it is left (blockdraft.settle)
    def _pending_pid(self):
        if self.p is not None and self.has_draft(self.p.pid):
            return self.p.pid
        return next((pid for pid, d in self.drafts.items() if d and self.s.view(pid) is not None), None)

    def draft_pending(self):
        pid = self._pending_pid()
        if pid is None:
            return None
        ch = self.draft_change(pid)
        q = self.s.view(pid)
        return (self.heading(q) if q is not None else self.title(), [ch["what"]] if ch else [],
                self.cost_text(pid))

    def draft_apply(self) -> bool:
        pid = self._pending_pid()
        if pid is not None:
            self.apply(pid)
        return pid is None or not self.has_draft(pid)

    def draft_variant(self) -> bool:
        pid = self._pending_pid()
        if pid is not None:
            if self.variant_why():                     # no variant of this block: the draft is applied instead
                self.apply(pid)
            else:
                self.try_variant(pid)
        return pid is None or not self.has_draft(pid)

    def draft_reset(self) -> bool:
        pid = self._pending_pid()
        if pid is not None:
            self.reset(pid)
        return True

    # ------------------------------------------------------------ the page: Inspect's centre for the block
    def _page_ok(self) -> bool:
        if self.page is None:
            return False
        from shiboken6 import isValid
        if not isValid(self.page) or not isValid(self.gv):
            self.page = self.gv = None
            self._ui = {}
            return False
        return True

    def detach(self):
        if self._page_ok() and self.tv.center.indexOf(self.page) >= 0:
            self._side(True)

    def center(self, tv, node, p) -> bool:
        """From TuneView.refresh: the centre is this page when the block shown is this editor's (True)."""
        want = node is not None and p is not None and not getattr(node, "cut", False) and self.wants(node, p)
        if not want:
            self._leave_page()
            return False
        self.node = node
        self._enter_page(p)
        return True

    def _side(self, show: bool):
        """TuneView's panel and header: away while the page shows (it has its own), back after."""
        tv = self.tv
        d = getattr(tv, "dock", None)
        try:
            hb = getattr(tv, "head_box", None)
            if hb is not None:
                hb.setVisible(show)
            if show:
                tv.panel.setVisible(d is None or d.state == "docked")
                if d is not None:
                    d.strip.setVisible(d.state in ("icons", "auto"))
                    d.arrow.setVisible(d.state == "hidden")
            else:
                tv.panel.hide()
                if d is not None:
                    d.strip.hide()
                    d.arrow.hide()
        except (RuntimeError, AttributeError):
            pass

    def _load(self, p):
        for vpid, (opid, d) in list(self._stashed.items()):
            if self.s.view(vpid) is None:      # the variant undone: the draft it took is the original's again
                self._stashed.pop(vpid)
                if d and not self.has_draft(opid):
                    self.drafts[opid] = d
        if self.p is None or p.pid != self.p.pid:
            self._new_block()
        self.p = p
        d = self._draft()
        if d is not None:                      # what the block now holds leaves the draft
            ap = self.applied()
            d.drop_pending({k for k, v in d.vals.items() if v != ap.get(k)})

    def _new_block(self):
        """Another block shown: what the view showed of the one before goes."""
        self.result = None

    def _enter_page(self, p):
        tv = self.tv
        from .i18n import language
        new = self.p is None or self.p.pid != p.pid or not self._page_ok() or self._lang != language() or \
            self._shape != self.shape(p)
        self._lang = language()
        self._shape = self.shape(p)
        self._load(p)
        if not self._page_ok() or new:
            self._build_page()
        if tv.center.indexOf(self.page) < 0:
            tv._clear_center()
            tv.center.addWidget(self.page, 1)
            self.page.show()
        self._side(False)
        for x in self._sides():                         # TuneView's panel stays away while the page shows
            x.installEventFilter(self)
        win = tv.window()
        if win is not self._win:
            self._win = win
            win.installEventFilter(self)
        if self.dock is not None:
            self.dock.set_state(self._side_state())
        tv.view, tv.views = self.gv, []
        tv._center_sig = (self.KEY, p.pid)
        self._sync()
        self._present()
        self.preview()

    # ------------------------------------------------------------ the settings panel at the right (the Gradient's)
    def _sides(self) -> list:
        tv = self.tv
        d = getattr(tv, "dock", None)
        out = [getattr(tv, "panel", None)] + ([d.strip, d.arrow] if d is not None else [])
        return [x for x in out if x is not None]

    def _side_state(self) -> str:
        """The settings panel's state: the user's choice when he made one, else by the window's width (kept open on a
        large window, automatic on a small one), as the Gradient's."""
        from .gradientedit import SMALL_W
        try:
            from . import prefs
            st = prefs.get(self.SIDE_PREF)
        except Exception:  # noqa: BLE001 - no preference readable: by the width
            st = None
        if st in ("docked", "icons", "auto"):
            return st
        try:
            return "docked" if self.tv.window().width() >= SMALL_W else "auto"
        except RuntimeError:
            return "docked"

    def _side_chosen(self, st: str):
        try:
            from . import prefs
            prefs.set(self.SIDE_PREF, st)
        except Exception:  # noqa: BLE001 - not kept: the next time by the width again
            pass

    def eventFilter(self, obj, e):                                   # noqa: N802 - Qt's name
        t = e.type()
        if t == QEvent.Show and obj in self._sides() and self.page_shown():
            QTimer.singleShot(0, lambda: self._side(False) if self.page_shown() else None)
        elif t == QEvent.Resize and obj is self._win and self.page_shown():
            if self.dock is not None:
                want = self._side_state()          # no choice of the user yet: the window's width chooses
                if want != self.dock.state:
                    self.dock.set_state(want)
            QTimer.singleShot(0, lambda: self._side(False) if self.page_shown() else None)
        return False

    def _fill_drawer(self):
        """The panel's groups: the draft's Undo and Redo; the measurements (the subclass's); the display (the stretch
        column, fixed under them)."""
        if self.drawer is None or self.p is None:
            return
        from .icons import icon_only
        from .widgets import clear
        d = self._draft()
        sig = (self.p.pid, bool(d and d.undo), bool(d and d.redo)) + tuple(self._drawer_sig())
        if sig == getattr(self, "_drawer_sig_now", None):
            return
        self._drawer_sig_now = sig
        dv = self.drawer.rows
        clear(dv)
        self.title_of_group(dv, "Draft")
        row = QHBoxLayout()
        for key, tip, fn, on in (("undo", "Undo the draft's last step (Ctrl+Z)", self.undo, bool(d and d.undo)),
                                 ("redo", "Redo the draft's step (Ctrl+Y)", self.redo, bool(d and d.redo))):
            bt = icon_only(QPushButton(), key, _(tip), 16)
            bt.setEnabled(on)
            bt.clicked.connect(lambda _c=False, f=fn: f())
            row.addWidget(bt)
            self._ui[key] = bt
        row.addStretch(1)
        dv.addLayout(row)
        self._measurements(dv)

    @staticmethod
    def title_of_group(dv, text):
        t = label(_(text), "h2")
        t.setContentsMargins(0, 8, 0, 2)
        dv.addWidget(t)

    def _leave_page(self):
        if not self._page_ok() or self.tv.center.indexOf(self.page) < 0:
            return
        self.tv.center.removeWidget(self.page)
        self.page.hide()
        self.page.setParent(None)
        self._side(True)
        if self.tv.view is self.gv:
            self.tv.view = None

    def page_shown(self) -> bool:
        try:
            return self._page_ok() and self.page.isVisibleTo(self.tv)
        except RuntimeError:
            return False

    def _build_page(self):
        """Built for the block shown: at the left the editor's column (its settings, the draft bar at its foot), the
        image at the right, its comparison in the title row, the settings panel at the right of the image."""
        from .imageview import ImageView
        if self._page_ok():
            try:
                self.page.setParent(None)
                self.page.deleteLater()
            except RuntimeError:
                pass
        u = self._ui = {}
        self._drawer_sig_now = None
        page = QWidget()
        page.setObjectName("combpage")
        page.setStyleSheet(
            "QPushButton#ivseg { padding: 3px 12px; border-radius: 0; }"
            "QPushButton#ivseg:checked { background: #2b3446; border-color: #5ea1ff; color: #ffffff; }"
            "QFrame#combcard { border: 1px solid #333a46; border-radius: 6px; }"
            "QLabel#thumb { border: 1px solid #333a46; }")
        v = QVBoxLayout(page)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(6)
        v.addWidget(self._title_row())
        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(10)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setFixedWidth(PANEL_W)
        col = QWidget()
        cv = QVBoxLayout(col)
        cv.setContentsMargins(0, 0, 8, 0)
        cv.setSpacing(8)
        bar = DraftBar(narrow=True)
        bar.what.setMinimumHeight(2 * bar.what.fontMetrics().height() + 4)
        bar.apply.connect(lambda: self.apply())
        bar.variant.connect(lambda: self.try_variant())
        bar.reset.connect(lambda: self.reset())
        u["bar"], u["state"] = bar, bar.what
        u["apply"], u["variant"], u["reset"] = bar.b_apply, bar.b_variant, bar.b_reset
        self._fill_column(cv)
        cv.addStretch(1)
        scroll.setWidget(col)
        left = QWidget()                               # the settings, then the draft bar at the column's bottom
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.setSpacing(8)
        lv.addWidget(scroll, 1)
        bar.setFixedWidth(PANEL_W - 8)
        lv.addWidget(bar)
        body.addWidget(left)
        box = QWidget()
        bl = QVBoxLayout(box)
        bl.setContentsMargins(0, 0, 0, 0)
        gv = ImageView(post=False, compare=True)
        gv.default_dir = lambda: str(getattr(self.s, "root", None) or "")
        gv.toasted.connect(self.tv.toasted.emit)
        gv.layout().itemAt(0).widget().hide()          # its own toolbar: the page's comparison in its title row
        gv.imageShown.connect(self._shown)
        gv.add_overlay(self._paint_tag)
        bl.addWidget(gv, 1)
        cover = QFrame()                               # under the image while there is nothing to show: why, and Run
        cl = QVBoxLayout(cover)
        cl.setContentsMargins(0, 6, 0, 6)
        head = label("", "dim")
        head.setAlignment(Qt.AlignCenter)
        cl.addWidget(head)
        text = label("", "dim", wrap=True)
        text.setAlignment(Qt.AlignCenter)
        cl.addWidget(text)
        runs = QHBoxLayout()
        cl.addLayout(runs)
        cover.hide()
        bl.addWidget(cover)
        u["cover"], u["cover_head"], u["cover_text"], u["cover_runs"] = cover, head, text, runs
        body.addWidget(box, 1)
        bw = QWidget()
        bw.setLayout(body)
        v.addWidget(bw, 1)
        self.page, self.gv = page, gv
        from .gradientedit import SettingsPanel
        from .layout import Dock
        self.drawer = SettingsPanel(self)
        self.dock = Dock(bw, self.drawer, lambda: body.addWidget(self.drawer), lambda: body.removeWidget(self.drawer),
                         title=_("Settings"), role="adjust")
        body.insertWidget(body.indexOf(box) + 1, self.dock.strip)
        body.addWidget(self.drawer)
        self.dock.set_state(self._side_state())
        self.dock.chosen.connect(self._side_chosen)
        gv.panel_box.setParent(None)                   # the stretch column: in the panel, never beside the image
        self.drawer.fixed.addWidget(label(_("Display"), "h2"))
        self.drawer.fixed.addWidget(gv.panel_box)
        gv.show_histogram(True)

    def _title_row(self) -> QWidget:
        u = self._ui
        w = QWidget()
        top = QHBoxLayout(w)
        top.setContentsMargins(0, 0, 0, 0)
        top.setSpacing(10)
        u["title"] = label("", "h1")
        top.addWidget(u["title"])
        u["tag"] = label("", "small")
        u["tag"].setStyleSheet(f"color: {WARN};")
        top.addWidget(u["tag"])
        top.addStretch(1)
        seg = QHBoxLayout()
        seg.setSpacing(0)
        for mode, text, tip in COMPARE:
            b = QPushButton(_(text))
            b.setObjectName("ivseg")
            b.setCheckable(True)
            b.setToolTip(_(tip))
            b.clicked.connect(lambda _c=False, m=mode: self.set_compare(m))
            seg.addWidget(b)
            u["cmp_" + mode] = b
        top.addLayout(seg)
        w.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        return w

    def _wire_slider(self, sl, set_value):
        """A slider's moves go to the draft as it is dragged (the preview follows), one step of Undo when it is let
        go; a key or the wheel, one step each."""
        sl.sliderPressed.connect(self.begin_drag)
        sl.sliderReleased.connect(self.end_drag)
        sl.valueChanged.connect(lambda i, s=sl: None if self._syncing else self._mark() or set_value(
            i, undo=not s.isSliderDown()))
        sl.installEventFilter(PressToRelease(sl))

    def _mark(self):
        self._t_edit = time.perf_counter()

    def set_compare(self, mode: str):
        self.cmp = mode
        self._present()
        self._sync()

    # ------------------------------------------------------------ the controls as the draft is
    def _status(self) -> str:
        n = len(self.changes())
        if n:
            return _("Draft · 1 change") if n == 1 else _("Draft · {n} changes").format(n=n)
        if self.state_of() == "stale":
            return _("Applied · not computed yet")
        return _("Applied settings")

    @property
    def applied_img(self):
        a = self.applied_cache
        return a if a is not None and self.p is not None and a.get("pid") == self.p.pid else None

    def _sync(self):
        """The page's controls as the draft is."""
        if self.p is None or not self._ui:
            return
        self._syncing = True
        try:
            u = self._ui
            draft = self.has_draft()
            ch = self.draft_change() if draft else None
            u["bar"].show_draft([self._status()] + ([t for _k, t in self.changes()] if ch else []),
                                self.cost_text() if draft else "")
            u["state"].setStyleSheet(f"color: {WARN};" if draft else "color: #8b95a5;")
            why = _("No draft: change a setting first")
            for k in ("reset", "variant", "apply"):
                u[k].setEnabled(draft)
            u["apply"].setToolTip((_("The draft becomes the block's settings (Ctrl+Z takes it back)") +
                                   ("\n" + self.cost_text() if self.cost_text() else "")) if draft else why)
            u["variant"].setToolTip(_("The draft on a variant of the workflow: what is computed stays as it is, the "
                                      "variant is computed by Run") if draft else why)
            if self.variant_why():                     # no variant of this block: greyed, why on hover
                u["variant"].setEnabled(False)
                u["variant"].setToolTip(_("No variant: {why}").format(why=self.variant_why()))
            u["reset"].setToolTip(_("Back to the applied settings: the draft goes") if draft else why)
            u["title"].setText(self.heading())
            self._fill_drawer()
            stale = self.state_of() == "stale" and not draft
            u["tag"].setText(_("Previous settings") if stale else "")
            for mode, _t, _tip in COMPARE:
                b = u.get("cmp_" + mode)
                if b is not None:
                    b.setChecked(self.cmp == mode)
            self._sync_controls()
            self._update_cover()
        except RuntimeError:
            pass
        finally:
            self._syncing = False

    def _update_cover(self):
        """Under the image: why there is nothing to show (an input not computed, an input to wire); a block never
        computed says "Not computed yet" with its Run to here, as every block's page."""
        u = self._ui
        if "cover" not in u:
            return
        never = self.state_of() not in ("done", "stale")
        u["cover_head"].setText(_("Not computed yet") if never else "")
        u["cover_head"].setVisible(never)
        u["cover_text"].setText(self.error)
        u["cover_text"].setVisible(bool(self.error))
        box = u["cover_runs"]
        if never and box.count() == 0 and hasattr(self.tv, "_run_button"):
            try:
                box.addStretch(1)
                box.addWidget(self.tv._run_button(self.p, self.KEY))
                box.addStretch(1)
            except Exception:  # noqa: BLE001 - no Run button here (a stand-in window): the words alone
                pass
        u["cover"].setVisible(never or bool(self.error))

    # ------------------------------------------------------------ the preview
    def preview(self):
        """The draft's preview (or the applied settings'): one computation at a time, the last edit only; the
        previous image stays until the next is ready."""
        if self.p is None:
            return
        miss = self.missing_text()
        if miss:
            self.error = miss
            self._sync()
            return
        self.gen += 1
        req = self._request()
        if self.busy:
            self._want = req
            return
        self._start(req)

    def _start(self, req: dict):
        self.busy = True
        self.computed += 1

        def work():
            try:
                out = self._work(req)
            except Exception as e:  # noqa: BLE001 - said over the image, the editor goes on
                out = {"req": req, "error": str(e)}
            out.setdefault("req", req)
            try:
                self._done.emit(req["gen"], out)
            except RuntimeError:                       # the window closed meanwhile: nobody to tell
                pass
        # a Python thread, not Qt's pool: PyTorch (the renders) never returned in a thread of the pool
        threading.Thread(target=work, name=f"{self.KEY} preview", daemon=True).start()

    def _arrived(self, gen: int, out: dict):
        self.busy = False
        req = out["req"]
        self._absorb(req, out)
        if "applied" in out:
            self.applied_cache = out["applied"]
        mine = self.p is not None and req["pid"] == self.p.pid
        if "error" in out:
            if mine:
                self.error = _(out["error"])
                if "applied" in out:
                    self._present()             # the block's own result, though the draft cannot be made now
        elif mine:
            self.error = ""
            if self._want is None:             # the latest edit: shown (an older one is never painted)
                self.painted += 1
                self.result, self.result_ms = out["result"], out.get("ms")
                self.result_key = (f"{self.KEY}-preview", req["pid"], gen)
                self._present()
        want, self._want = self._want, None
        if want is not None and self.p is not None and want["pid"] == self.p.pid:
            want["gen"] = self.gen
            self._start(want)
        self._sync()

    def _present(self):
        """The image area: the draft's preview, the applied result, or both split (the draft on the right)."""
        from .imageview import ViewImage
        if not self._page_ok() or self.p is None:
            return
        pid = self.p.pid
        drafting = self.has_draft() or self._view_only()
        tag = (_("Draft preview · reduced copy") if self.has_draft() else _("Preview · reduced copy")) + \
            (f" · {self.result_ms:.0f} ms" if self.result_ms is not None else "") + self.tag_extra()
        self.preview_tag = tag
        pv = ViewImage(self.result, tag, key=self.result_key,
                       stretched=True, units="norm", white=1.0, camera=False, neutral=False) \
            if self.result is not None else None
        ap = self.applied_img
        av = ViewImage(ap["img"], self.applied_label(), key=(f"{self.KEY}-applied", pid, ap["key"]), stretched=True,
                       units="norm", white=1.0, camera=False, neutral=False) \
            if ap is not None and ap.get("img") is not None else None
        if self.cmp == "applied":
            main, other = av or pv, None
        elif self.cmp == "split":
            main, other = pv or av, av if pv is not None else None
        else:
            main, other = (pv if (drafting or av is None) else av) or pv, None
        if main is None:
            return
        self.shown_tag = main.name or ""
        gv = self.gv
        try:
            if gv._want != main.identity():
                gv.set_image(main, keep_view=gv.canvas.display is not None, keep_stretch=True)
            if other is None:
                gv.set_compare_targets([], None)
            else:
                gv.set_compare_mode("split")
                gv.set_compare_targets([other], 0)
                gv.set_same_stretch(True)
        except RuntimeError:
            pass

    def tag_extra(self) -> str:
        """What the image's tag says besides what it is (a search under way: " · " and its words)."""
        return ""

    def _view_only(self) -> bool:
        """What the view alone changes (a filter taken out of the mix): the preview shown rather than the applied."""
        return False

    def _paint_tag(self, painter, canvas):
        """The single image's tag, at its top (Split names its two sides itself), as the Gradient's: what it is, the
        reduced copy, the time it took."""
        if self.cmp == "split" or not self._page_ok():
            return
        from PySide6.QtCore import QPointF

        from .gradientedit import DRAFT_BG
        from .imageview import tag
        text = getattr(self, "shown_tag", "")
        if text:
            tag(painter, QPointF(10, 8), text, bg=DRAFT_BG if self.has_draft() else QColor(0, 0, 0, 170))

    def _shown(self, key):
        """An image of the editor on screen: the time from the edit that asked for it (``timings``)."""
        if self._t_edit is not None and isinstance(key, tuple) and key and key[0] == f"{self.KEY}-preview":
            self.timings.append(time.perf_counter() - self._t_edit)
            self._t_edit = None
