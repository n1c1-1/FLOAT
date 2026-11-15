"""The Gradient block tuned in Inspect as a draft (``GradientEditor``): try, look, then Apply or Try variant.

Every adjustment of the view (the method and its settings, the samples placed, moved and removed on the image, the
areas left out with the brush or ellipses, the generated grid, Away from emission) changes a DRAFT of the block's
settings, kept per workflow while the project is open: the project does not change, the graph stays up to date, Run
is unchanged. The draft is seen at once on a reduced copy of the stack (about a million pixels, whatever the sensor),
computed with the stage's own functions (``sky_model_of``, ``read_grid``) in a thread of its own, the last edit only,
the previous image kept on screen until the next one is ready; it is compared with the applied result on the same
grid (read from the block's own sky model, never computed again) or with the stack, Split or Side by side.

When it looks right: **Apply** (the draft becomes the block's settings, one step of the window's Undo; what it
computes again and how long is said on the draft bar), **Try variant** (the draft goes on a variant of the workflow:
the computed chain stays as it is) or **Reset** (back to the applied settings): the draft bar every block's panel has
(blockdraft.py), where its buttons were. Leaving the block with a draft asks the question every block asks (Apply, Try
variant, Discard, Stay): a draft never survives leaving its block.

The tools are on a bar above the image, as the reference photo editors have them: Samples (click adds, drag moves,
Remove or a right click removes the one selected), Brush (paint / erase the areas left out, X, [ ]), Ellipse (drag,
handles, Remove), Generate grid, the draft's Undo / Redo. The samples are outlined in cyan when the model uses them,
in red when it leaves them out, in white while not computed; the areas left out under a red veil at 50 %. Space held,
the middle button or (with the brush) a right drag move the image; M or Marks hides what is drawn over it.

The H-alpha survey (R, or its toggle in the image's bar, the one action that may download it): the professional
H-alpha map of the field, drawn under the samples in amber where no sample should sit. Away from emission leaves those
cells out of Generate grid and of the samples the block chooses; the preview and Generate grid read the survey from
this computer only. A guide only: the sky model never reads the survey.
"""
from __future__ import annotations

import copy
import math
import os
import time

import numpy as np
from PySide6.QtCore import QEvent, QObject, QPointF, QRectF, QRunnable, Qt, QThreadPool, QTimer, Signal
from PySide6.QtGui import QColor, QImage, QPen
from PySide6.QtWidgets import (
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .blockdraft import Draft, DraftBar
from .i18n import _
from .icons import icon_only
from .widgets import WARN, label

#: the preview's copy of the stack holds about this many pixels
PREVIEW_PIXELS = 1_000_000
#: the cell the RBF preview's sky is evaluated on when the block chooses its samples, as a share of the reduced copy's
#: shorter side (never finer than the run's 8 px): 32 px on a 1064 x 712 copy, the preview of 16 per side in 0.14 s of
#: CPU instead of 0.5 s, within 0.5 ADU (0.2 % of the sky) of the run's cell
PREVIEW_CELL_SHARE = 22
#: the preview starts this long after the last edit (the automatic model, slower, waits a little more)
#: the preview starts once the draft has not changed for this long: short when this method's last preview was fast
#: (FAST_MS or less), so that the polynomial follows the settings live
DEBOUNCE_MS, FAST_DEBOUNCE_MS, FAST_MS = 200, 80, 150
#: a preview still running after this long is said on the image ("Computing preview…" and a bar along its top)
BUSY_SHOWN_MS = 250
#: what an action did stays in the image's tag this long
SAID_MS = 4000
#: an area just painted or drawn shows filled, then fades away over this long (the samples it leaves out stay red)
FADE_MS = 700
#: the cost of Apply is asked of the model this long after the last edit
COST_MS = 500
#: below this window width, the stretch column of this view is folded (the image keeps the room)
SMALL_W = 1500
#: Inspect waits at most this long for its layout and its image before it paints (``GradientEditor.hold``)
HOLD_MS = 800
#: a click this close to a sample (screen px) takes it
PICK_PX = 9
USED, UNUSED, OUTLINE, PICKED = QColor("#56c8f5"), QColor("#ff4d55"), QColor(255, 255, 255, 220), QColor("#ffd24a")
VEIL = (229, 72, 77, 128)                      # the excluded areas: red, 50 %
SURVEY_TINT, SURVEY_ALPHA = (255, 176, 32), 128   # the H-alpha survey's emission: amber, 18 to 50 %
DRAFT_BG = QColor(110, 74, 12, 225)            # the draft preview's tag: amber, apart from the views' grey tags
FIELDS = ("sample_points", "exclude_ellipses", "exclude_brush")
#: the methods, in the order of the list, and their names (2 words at most; what they do goes to the tooltip)
METHODS = ("polynomial", "rbf", "automatic")
METHOD_NAME = {"polynomial": "Polynomial", "rbf": "RBF", "automatic": "Automatic"}
METHOD_LINE = {"polynomial": "A smooth surface fitted on the sky between the objects",
               "rbf": "A surface through the samples: follows a gradient of any shape",
               "automatic": "A flexible surface on every pixel that does not look like an object"}
METHOD_TIP = {"polynomial": "A smooth surface fitted on the sky between the objects. Keeps faint, very wide structures "
                            "best.",
              "rbf": "A surface through the samples (thin-plate splines): follows a gradient of any shape; best with "
                     "samples placed by hand.",
              "automatic": "A flexible surface fitted on every pixel that does not look like an object: removes "
                           "complex gradients best, and with them part of the faintest wide structures. Its preview "
                           "takes a second or two: the previous one stays on screen meanwhile."}
#: each method's main setting, at the image: (setting, name, lowest, highest, step, decimals, tooltip)
MAIN = {"polynomial": ("degree", "Degree", 1, 5, 1, 0, "Degree of the polynomial surface: 2 for most fields, 3-4 for "
                                                        "wide fields with complex gradients"),
        "rbf": ("rbf_smooth", "Smoothing", 0.0, 1.0, 0.05, 2, "0: the surface passes through every sample; higher, it "
                                                               "passes near them, smoother"),
        "automatic": ("scale", "Scale", 1.0, 10.0, 0.5, 1, "Below this scale, relative to the field, a variation is "
                                                           "taken as an object rather than gradient")}
#: an ellipse's handles: their size, and how far above the ellipse the rotation one is (screen px)
HANDLE_PX, ROT_PX = 5, 26
#: the keys of the tools: the image takes them before the window's and Inspect's shortcuts (Escape goes back to the
#: graph otherwise)
TOOL_KEYS = (Qt.Key_Escape, Qt.Key_Delete, Qt.Key_Backspace, Qt.Key_M, Qt.Key_R, Qt.Key_B, Qt.Key_BracketLeft,
             Qt.Key_BracketRight)
#: the tools of the strip at the left of the image: (mode, icon, name, what it does); Grid is an action
TOOLS = (("points", "add", "Sample", "Click adds a sample of sky, a drag moves it"),
         ("brush", "mask", "Brush", "Paint an area left out of the sky model ([ ] size)"),
         ("ellipse", "missing", "Ellipse", "Drag an ellipse around an object to leave it out; its handles size and turn "
                                           "it"),
         ("eraser", "remove", "Eraser", "Removes what is under the cursor: a sample, a painted area, an ellipse"))
#: the actions on the samples (each one step of the draft's Undo): (key, name, what it does)
ACTIONS = (("generate", "Generate grid", "Replaces the samples by a regular grid of this pitch, looking at nothing else"),
           ("objects", "Avoid objects", "Removes, among the samples present, those on objects: the stage's own rule "
                                        "(brighter than the sky around them by Object rejection)"),
           ("nebula", "Avoid nebulosity", "Removes, among the samples present, those where the professional H-alpha "
                                          "survey of the field sees nebulosity"),
           ("clearall", "Clear all", "Removes every sample: none is chosen for you afterwards"))
#: the marks each method reads (stages/gradient.py, sky_model_of): the samples (placed, or made by Grid) feed the
#: polynomial and the RBF; the excluded areas (brush, ellipses) are left out of all three models, the automatic one
#: included; the automatic model takes no sample
MARK_KINDS = {"polynomial": ("samples", "areas"), "rbf": ("samples", "areas"), "automatic": ("areas",)}
#: the kind of marks each tool makes ("" for the Eraser: any kind the method reads)
TOOL_KIND = {"points": "samples", "brush": "areas", "ellipse": "areas", "eraser": ""}
#: why a kind of marks is greyed, per method
NO_KIND = {"samples": "{method} uses no samples: it fits every pixel that does not look like an object",
           "areas": "{method} uses no excluded areas",
           "": "{method} uses no marks"}
#: the settings panel at the right of the image: its width
PANEL_W = 380
#: M kept down longer than this hides the marks only while it is down (shorter: a tap shows or hides them)
MARKS_HOLD_S = 0.35
def _ell(e) -> tuple[float, float, float, float, float]:
    """An ellipse of exclude_ellipses: centre, half axes, angle (degrees; an earlier one has none)."""
    return float(e[0]), float(e[1]), float(e[2]), float(e[3]), float(e[4]) if len(e) > 4 else 0.0


def _local(e, x, y) -> tuple[float, float]:
    """(x, y) in the frame of the ellipse: along its first axis, along its second."""
    cx, cy, _rx, _ry, a = _ell(e)
    c, s = math.cos(math.radians(a)), math.sin(math.radians(a))
    dx, dy = x - cx, y - cy
    return dx * c + dy * s, -dx * s + dy * c


def _world(e, u, v) -> tuple[float, float]:
    cx, cy, _rx, _ry, a = _ell(e)
    c, s = math.cos(math.radians(a)), math.sin(math.radians(a))
    return cx + u * c - v * s, cy + u * s + v * c


def _inside(e, x, y) -> bool:
    _cx, _cy, rx, ry, _a = _ell(e)
    u, v = _local(e, x, y)
    return (u / max(rx, 1e-6)) ** 2 + (v / max(ry, 1e-6)) ** 2 <= 1


def _norm(name: str, v):
    """A setting of the block as the draft compares it (lists of floats, the defaults of a setting not stored)."""
    if name in FIELDS:
        return [[float(x) for x in e] for e in (v or [])]
    if name == "method":
        return str(v or "polynomial")
    if name == "avoid_emission":
        return True if v is None else bool(v)
    if name == "samples":
        return int(v or 16)
    return v


class _Job(QRunnable):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def run(self):
        self.fn()


class GradientEditor(QObject):
    """The Gradient block of the workflow shown, tuned as a draft on the main viewer of Inspect. ``attach(view, p)``
    when the block's picture is shown, ``detach()`` otherwise; ``controls(p)`` the panel's method list; ``drafting(p)``
    whether the panel's settings go to the draft; ``apply()``, ``try_variant()``, ``reset()``. It is the canvas's
    ``tool`` while a tool is on. ``gesture``: the mouse is down on the image (Inspect is not built again under it)."""

    _done = Signal(int, object)                # (request number, result of the thread)
    #: a step of a draft's Undo made, taken back or made again: the window's Edit › Undo / Redo follow it (a beta
    #: tester's Ctrl+Y did nothing after a Ctrl+Z on the draft, Edit › Redo greyed, while the panel's arrow worked)
    historyChanged = Signal()
    _step = Signal(int, str)                   # (request number, how far the automatic preview is)

    def __init__(self, tune):
        super().__init__(tune)
        self.tv, self.s = tune, tune.s
        self.view = None
        self.p = None
        self.mode: str | None = None           # None (looking), "points", "brush", "ellipse", "eraser"
        self.paint = True
        self.size = 60.0                       # brush diameter, stack pixels
        self.shown = True                      # what is drawn over the image (M, Marks)
        self.drafts: dict[int, Draft] = {}     # workflow -> its draft (blockdraft.Draft: {setting: value}, its Undo)
        self.work: dict[str, list] = {f: [] for f in FIELDS}
        self.done = False                      # the block is computed with its applied settings
        self.gesture = False
        self.gen = 0
        self.small = None                      # {"pid", "arr", "valid", "f", "W", "H"}
        self.applied = None                    # {"key", "arr" or None}: the applied result on the reduced copy
        self.samples = None                    # {"x", "y", "used", "manual", "box"} in stack pixels
        self.result = None                     # the last preview (3, h, w)
        self.result_ms = None                  # what it took (CPU of the thread, ms)
        self.busy = False
        self.step_text = ""                    # how far the automatic preview is
        self.page = self.gv = self.drawer = None   # the page, its viewer, its Settings drawer (built once)
        self.before = False                    # Before / After: the stack shown
        self.cmp = "toggle"                    # toggle | split | side
        self.result_key = None
        self._m_down = None                    # (when M went down, the marks shown then)
        self.defaults: dict[int, list] = {}    # workflow -> the samples the block places by itself (stack px)
        self._said = None                      # (what an action did, when): said in the image's tag a moment
        self._ops = None                       # the action waiting for its computation (ops, its points)
        self.dock = None
        self.carried: dict[int, dict] = {}     # variant -> the preview its draft had, and the original's result
        self._stashed: dict[int, tuple] = {}   # variant -> (original, the draft it took)
        self.hist_shown = None
        self._syncing = False
        self._cost = None                      # (workflow, settings, what Apply computes again)
        self._applied_at = None
        self._win = None                       # the window whose resizing the editor follows
        self._held = None                      # Inspect held from painting until its layout settles (hold)
        self._restore = None                   # (image key, view state) to give back when the preview shows
        self._cmp_name = None                 # the image the preview is compared with, by name
        self._in_show = False
        self._dims = None                      # (pid, W, H) of the stack
        self._pending = None                   # the request waiting for the one running
        self._shown_gen = 0                    # the request number when the workflow shown last changed
        self._hover = None
        self._grab = None
        self._ell = None                       # the ellipse being drawn (two corners, stack px)
        self._sel = None                       # the ellipse selected (its handles shown)
        self._selpt = None                     # the sample selected (Remove takes it)
        self._edit_ell = None                  # (what is dragged: move / rx / ry / rot, the start)
        self._last = None
        self._stroke0 = None                   # where the brush stroke in progress starts in exclude_brush
        self._extra = None                     # the click that asked for the grid: a sample added to it
        self._veil = None                      # (key, QImage)
        self.survey = None                     # {"pid", "level" (h, w) on the reduced copy, "title"} or {"pid", "error"}
        self.survey_shown = False              # the survey drawn under the samples (R)
        self._survey_wanted = None             # asked while a computation ran (its fetch flag)
        self._grid_after_survey = False        # Generate waits for the survey (Away from emission)
        self._survey_img = None                # (key, QImage)
        self._ui: dict = {}
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(DEBOUNCE_MS)
        self._timer.timeout.connect(self.preview)
        self.computed = self.painted = 0        # previews started, previews painted (the tests count them)
        self._ms_of: dict[str, float] = {}      # method -> the last preview's time (ms)
        self._busy_shown = False                # the computation said on the image (after BUSY_SHOWN_MS)
        self._busy_timer = QTimer(self)
        self._busy_timer.setSingleShot(True)
        self._busy_timer.setInterval(BUSY_SHOWN_MS)
        self._busy_timer.timeout.connect(self._show_busy)
        self._anim = QTimer(self)               # the bar along the image's top while it computes
        self._anim.setInterval(40)
        self._anim.timeout.connect(self._tick)
        self._phase = 0.0
        self._from_row = False                  # the change came from a settings row: its rows are not built again
        self._preview_due = False               # a preview asked while another computation ran
        self._fade_t0 = None                    # when the last area was painted or drawn (its fade)
        self._fade = QTimer(self)
        self._fade.setInterval(40)
        self._fade.timeout.connect(self._fade_tick)
        self._excl = None                       # (key, which samples sit in an excluded area)
        self._cost_timer = QTimer(self)
        self._cost_timer.setSingleShot(True)
        self._cost_timer.setInterval(COST_MS)
        self._cost_timer.timeout.connect(self._update_cost)
        self._done.connect(self._arrived)
        self._step.connect(self._stepped)

    # ------------------------------------------------------------ the draft
    def _applied_value(self, name: str, q=None):
        q = q or self.p
        return _norm(name, self.s.param_value(q, "gradient", name))

    def _draft(self, pid: int | None = None) -> dict | None:
        if pid is None:
            if self.p is None:
                return None
            pid = self.p.pid
        return self.drafts.get(pid)

    def has_draft(self, pid: int | None = None) -> bool:
        d = self._draft(pid)
        return bool(d)

    def value(self, name: str):
        """A setting as the draft has it (the applied one when the draft does not change it)."""
        d = self._draft()
        if d is not None and name in d.vals:
            return copy.deepcopy(d.vals[name])
        return self._applied_value(name)

    def _explicit_empty(self) -> bool:
        """No sample, by the user's choice (Clear all): the block must not choose its own then."""
        d = self._draft()
        return bool(d and "sample_points" in d.vals and d.vals["sample_points"] == [])

    def _fields(self) -> dict:
        """The marks on screen: the draft's; while the samples were never touched, the ones the block places by itself
        (``defaults``), drawn as any sample: the first change makes them the draft's own."""
        w = {f: self.value(f) for f in FIELDS}
        if not w["sample_points"] and not self._explicit_empty() and "samples" in self.kinds() and \
                self.p is not None:
            w["sample_points"] = copy.deepcopy(self.defaults.get(self.p.pid) or [])
        return w

    def n_min(self) -> int:
        """The fewest samples the stage fits with (sky_model_of's own minimum)."""
        if self._method() == "rbf":
            return 3
        d = int(self.value("degree") or 2)
        return (d + 1) * (d + 2) // 2 + 1

    def too_few(self) -> bool:
        """Samples cleared, or fewer than the method's minimum: nothing to apply, nothing removed in the preview."""
        if "samples" not in self.kinds():
            return False
        pts = self.value("sample_points")
        return self._explicit_empty() or 0 < len(pts) < self.n_min()

    def say(self, text: str):
        """What an action did, in the image's tag for a few seconds."""
        self._said = (text, time.monotonic())
        self._repaint()
        QTimer.singleShot(SAID_MS + 50, self._repaint)

    def drafting(self, p) -> bool:
        """Whether the block of ``p`` is tuned as a draft here: it has a result on screen (computed, or Outdated with
        its previous one). Never computed, its settings are the panel's as any block's."""
        st = self.s.state(p, "gradient")
        return st == "done" or (st == "stale" and self.tv._has_previous(p, "fond"))

    def set_draft(self, updates: dict, undo: bool = True) -> bool:
        """Settings of the draft: one step of its Undo; a setting back to its applied value leaves the draft."""
        if self.p is None:
            return False
        d = self.drafts.setdefault(self.p.pid, Draft("fond", [self.p.pid]))
        before = set(d.vals)
        norm = {k: _norm(k, v) for k, v in updates.items()}
        # back to its applied value, a setting leaves the draft; no sample by choice (Clear all) stays one, never the
        # block's own choice
        if not d.put(norm, same=lambda k, v: v == self._applied_value(k) and not (k == "sample_points" and v == []),
                     step=undo):
            self.work = self._fields()
            self._sync()
            return False
        self._draft_changed(before | set(d.vals) | set(updates))
        if undo:
            self.historyChanged.emit()
        return True

    def _draft_changed(self, names: set):
        self.work = self._fields()
        self._veil = None
        if self._sel is not None and self._sel >= len(self.work["exclude_ellipses"]):
            self._sel = None
        if self._selpt is not None and self._selpt >= len(self.work["sample_points"]):
            self._selpt = None
        self._cost = None
        if self.has_draft():
            self._schedule()
            self._cost_timer.start()
        else:
            self._to_applied()
        if self.mode and TOOL_KIND.get(self.mode) and TOOL_KIND[self.mode] not in self.kinds():
            self.set_mode(None)                # a method that reads none of its marks: the tool goes off
        if names - set(FIELDS) and not self._from_row:     # the method, an Undo: the panel's rows show it
            self._fill_drawer()
        self._sync()
        if self.view is not None:
            try:
                self.view.canvas.update()
            except RuntimeError:
                self.view = None

    def _to_applied(self):
        """No draft any more: the block's own result on screen again."""
        self.result, self.result_ms = None, None
        self.samples = self._computed_samples()
        self._present()

    def undo(self, pid: int | None = None) -> bool:
        """The draft's last step taken back (the panel's arrow; the window's Undo when the draft's step is the latest
        change of the window: ``history_top``). ``pid``: the draft of another workflow than the one shown (its step is
        taken back, nothing on screen changes)."""
        return self._move(pid, "undo", "redo")

    def redo(self, pid: int | None = None) -> bool:
        return self._move(pid, "redo", "undo")

    def _move(self, pid, src: str, dst: str) -> bool:
        d = self._draft(pid)
        names = d.move(src) if d is not None else None
        if names is None:
            return False
        if pid is None or (self.p is not None and pid == self.p.pid):
            self._draft_changed(names)
        self.historyChanged.emit()
        return True

    def history_top(self, how: str = "undo") -> tuple[int, int] | None:
        """(stamp, workflow) of the draft step the window's Undo (``how`` "undo") or Redo would take next, the latest of
        every workflow's draft; None without one."""
        best = None
        for pid, d in self.drafts.items():
            seq = d.top(how)
            if seq is not None and (best is None or seq > best[0]):
                best = (seq, pid)
        return best

    def _paths(self, q):
        paths = self.s.scope_paths()
        return paths if q.pid in [x.pid for x in paths] else [q]

    def _view_of(self, pid):
        if pid is None or (self.p is not None and pid == self.p.pid):
            return self.p
        return self.s.view(pid)

    def draft_change(self, pid: int | None = None) -> dict | None:
        """The draft as one change of the panel's kind (a method: ``opt``; settings: ``params``; both: ``many``), with
        its words (``what``, ``short``); None without a draft."""
        q = self._view_of(pid)
        d = self.drafts.get(q.pid) if q is not None else None
        if not d:
            return None
        from .model import GRADIENT_VIEW
        vals = dict(d.vals)
        chs = []
        m = vals.pop("method", None)
        if m is not None:
            chs.append({"kind": "opt", "key": "fond", "value": GRADIENT_VIEW.get(m, "simple")})
        if vals:
            chs.append({"kind": "params", "items": [("gradient", k, None if v == [] else v)
                                                    for k, v in sorted(vals.items())]})
        ch = dict(chs[0]) if len(chs) == 1 else {"kind": "many", "changes": chs}
        bits = []
        if m is not None:
            bits.append(_(METHOD_NAME[m]))
        from .graphnode import value_words
        if d.vals.get("sample_points"):
            bits.append(value_words("sample_points", d.vals["sample_points"]))
        if any(k in d.vals for k in ("exclude_ellipses", "exclude_brush")):
            bits.append(_("exclusions"))
        short = ", ".join(bits) or _("gradient draft")
        ch.update(what=_("Gradient draft: {what}").format(what=short), short=short)
        return ch

    def _update_cost(self):
        """What Apply computes again, asked of the model (the general confirmation's words when it has them)."""
        if self.p is None:
            return
        ch = self.draft_change()
        if ch is None:
            self._cost = None
            self._sync()
            return
        paths = self._paths(self.p)
        text = ""
        try:
            if hasattr(self.s, "change_cost"):
                c = self.s.change_cost(ch, paths)
                text = c.text() if c.lost else _("Nothing up to date is computed again")
            else:
                from .i18n import hm
                c = self.s.cost_of_change(paths, ch)
                text = _("Computes again · about {t}").format(t=hm(c.seconds))
        except Exception:  # noqa: BLE001 - the cost is a hint: Apply works without it
            text = ""
        self._cost = (self.p.pid, repr(sorted(self._draft().vals.items())), text)
        self._sync()

    def cost_text(self) -> str:
        d = self._draft()
        if self._cost is None or d is None or self._cost[0] != self.p.pid or \
                self._cost[1] != repr(sorted(d.vals.items())):
            return ""
        return self._cost[2]

    def _refused(self, e):
        self.tv.toasted.emit(_(str(e)))

    def apply(self, pid: int | None = None):
        """The draft becomes the block's settings, in the workflows the panel's changes go to: one step of the
        window's Undo. The preview stays on screen until Run to here computes it."""
        q = self._view_of(pid)
        if q is not None and self.p is not None and q.pid == self.p.pid and self.too_few():
            self.say(_("Too few samples for {method}: place samples or generate a grid").format(
                method=_(METHOD_NAME.get(self._method(), self._method()))))
            return                             # the stage would choose its own: never applied behind the user
        ch = self.draft_change(q.pid) if q is not None else None
        if ch is None:
            return
        from ..core import workflow as wfl
        from .model import NotInEngine
        paths = self._paths(q)
        try:
            if hasattr(self.s, "apply_as"):
                self.s.apply_as(ch, "replace", paths)
            else:
                for c in (ch["changes"] if ch["kind"] == "many" else [ch]):
                    self.s.apply_change(paths, c)
        except (ValueError, NotInEngine, wfl.OptionRefused) as e:
            self._refused(e)
            return
        self.s.log("{what} · {names}", what=ch["what"], names=", ".join(x.name for x in paths))
        self.drafts.pop(q.pid, None)
        self.tv.toasted.emit(_("Gradient: draft applied · Run to here computes it · Ctrl+Z"))
        if self.p is not None and q.pid == self.p.pid:
            self.work = self._fields()
            self.done = self.s.state(self.p, "gradient") == "done"
        self.tv.request_refresh()

    def try_variant(self, pid: int | None = None):
        """The draft on a variant of the workflow (the chain computed stays as it is); the variant is shown."""
        q = self._view_of(pid)
        if q is not None and self.p is not None and q.pid == self.p.pid and self.too_few():
            self.say(_("Too few samples for {method}: place samples or generate a grid").format(
                method=_(METHOD_NAME.get(self._method(), self._method()))))
            return                             # the stage would choose its own: never applied behind the user
        ch = self.draft_change(q.pid) if q is not None else None
        if ch is None:
            return
        from ..core import workflow as wfl
        from .model import NotInEngine
        title = f"{q.name} · {ch['short']}"
        try:
            if hasattr(self.s, "apply_as"):
                v = self.s.apply_as(ch, "variant", [q], title)
            else:
                chs = ch["changes"] if ch["kind"] == "many" else [ch]
                v = self.s.add_variant(q, chs[0], title)
                for c in chs[1:]:
                    self.s.apply_change([v], c)
        except (ValueError, NotInEngine, wfl.OptionRefused) as e:
            self._refused(e)
            return
        self.s.log('Variant "{name}" created', name=v.name)
        d = self.drafts.pop(q.pid, None)
        self._stashed[v.pid] = (q.pid, copy.deepcopy(d))       # an Undo of the variant gives the draft back
        if self.p is not None and q.pid == self.p.pid:
            # what was tuned goes with the variant: its preview, and the original's result to compare it with
            if self.result is not None and self.small is not None and self.small["pid"] == q.pid:
                ap = self.applied.get("arr") if self.applied is not None and self.applied["key"][0] == q.pid else None
                self.carried[v.pid] = {"result": self.result, "key": ("gradient-variant", v.pid, self.gen), "ms": self.result_ms,
                                       "samples": self.samples, "small": self.small, "orig": ap, "of": q.name}
            self.result, self.result_ms, self.result_key = None, None, None
        self.tv._variant_made(v.pid, q.pid)

    def reset(self, pid: int | None = None):
        """Back to the applied settings: the draft goes."""
        q = self._view_of(pid)
        if q is None or q.pid not in self.drafts:
            return
        self.drafts.pop(q.pid, None)
        if self.p is not None and q.pid == self.p.pid:
            self._draft_changed({"method"})

    # ------------------------------------------------------------ leaving the block with a draft
    def eventFilter(self, obj, e):                                   # noqa: N802 - Qt's name
        t = e.type()
        if t in (QEvent.KeyPress, QEvent.KeyRelease) and self._page_ok() and obj in (self.gv.canvas, self.page) and \
                e.key() == Qt.Key_M and not e.modifiers() & (Qt.ControlModifier | Qt.AltModifier):
            if not e.isAutoRepeat():
                self._marks_key(t == QEvent.KeyPress)
            return True
        if t == QEvent.Resize and self._ui and obj is self.gv:
            cv = self._ui.get("cover")
            if cv is not None and cv.isVisible():
                cv.setGeometry(self.gv.rect())
            return False
        if t == QEvent.Show and obj in self._sides() and self._page_shown():
            self.hold()                        # the window docked TuneView's panel again: away while the page shows
            QTimer.singleShot(0, lambda: self._side(False) if self._page_shown() else None)
        elif t == QEvent.Resize and obj is self._win:
            if self.view is not None and self.dock is not None:
                want = self._side_state()      # no choice of the user yet: the window's width chooses
                if want != self.dock.state:
                    self.dock.set_state(want)
            if self.view is not None:
                # the window's own layout of Inspect (its panel docked or folded) is taken away again while the page shows
                QTimer.singleShot(0, lambda: self._side(False) if self.view is not None else None)
        return False

    # ------------------------------------------------------------ one paint per change of view
    def hold(self):
        """Inspect paints nothing until its layout has settled: the rows of the tools, the stretch column, the panel's
        rows, the header and the image asked for all in place; then the image is fitted once on its final size and
        shown in one paint (the previous picture stays on the screen meanwhile). At most HOLD_MS."""
        tv = self.tv
        try:
            if not tv.isVisible():
                return
            if self._held is None:
                tv.setUpdatesEnabled(False)
            self._held = {"t0": time.monotonic(), "sig": None, "same": 0}
        except RuntimeError:
            return
        QTimer.singleShot(0, self._hold_check)

    def _hold_check(self):
        h = self._held
        if h is None:
            return
        try:
            v = self.tv.view
            ready = v is None or v._want is None or (v._prep is not None and v._prep.key == v._want) or \
                v._failed_key == v._want
            sig = None if v is None else (v.canvas.width(), v.canvas.height(), v.canvas2.isVisible(), ready)
        except (RuntimeError, AttributeError):
            sig, ready = None, True
        h["same"] = h["same"] + 1 if sig == h["sig"] else 0
        h["sig"] = sig
        if (ready and h["same"] >= 2) or time.monotonic() - h["t0"] > HOLD_MS / 1000:
            self._release()
        else:
            QTimer.singleShot(10, self._hold_check)

    def _release(self):
        self._held = None
        try:
            v = self.tv.view
            if v is not None and v.canvas.fitted:
                v.canvas.fit()                 # once, on the final size
            self.tv.setUpdatesEnabled(True)
        except RuntimeError:
            pass

    def _shown(self, key):
        """The preview shown in place of an image of another size: the same part of the sky at the same size on the
        screen (the viewer would fit it again)."""
        want, self._restore = self._restore, None
        if want is None or want[0] != key or self.view is None:
            return
        try:
            if not want[1].get("fitted", True):
                self.view.set_view_state(want[1])
        except RuntimeError:
            pass

    def _here(self, pid) -> bool:
        """The Gradient block of this workflow is the one Inspect shows."""
        try:
            s = self.s
            return (self.tv.isVisible() and s.path is not None and s.path.pid == pid
                    and str(s.sel_step).partition("~")[0] == "fond")
        except RuntimeError:
            return False

    # the question every block asks before it is left (blockdraft.settle): this view's draft
    def _pending_pid(self):
        """The workflow whose Gradient draft is not applied (the one shown first), None."""
        if self.p is not None and self.has_draft(self.p.pid):
            return self.p.pid
        return next((pid for pid, d in self.drafts.items() if d and self.s.view(pid) is not None), None)

    def draft_pending(self):
        pid = self._pending_pid()
        if pid is None:
            return None
        ch = self.draft_change(pid)
        q = self.s.view(pid)
        cost = self.cost_text() if self.p is not None and pid == self.p.pid else ""
        return (_("Gradient") + (f" · {q.name}" if q is not None else ""), [ch["what"]] if ch else [], cost)

    def draft_apply(self) -> bool:
        pid = self._pending_pid()
        if pid is not None:
            self.apply(pid)
        return pid is None or not self.has_draft(pid)

    def draft_variant(self) -> bool:
        pid = self._pending_pid()
        if pid is not None:
            self.try_variant(pid)
        return pid is None or not self.has_draft(pid)

    def draft_reset(self) -> bool:
        pid = self._pending_pid()
        if pid is not None:
            self.reset(pid)
        return True

    def _watch_window(self):
        win = self.tv.window()
        if win is not self._win:
            if self._win is not None:
                try:
                    self._win.removeEventFilter(self)
                except RuntimeError:
                    pass
            self._win = win
            win.installEventFilter(self)

    # ------------------------------------------------------------ the workflow and the viewer
    def _method(self) -> str:
        return self.value("method") if self.p is not None else "polynomial"

    def _applied_params(self, q=None):
        from ..stages.gradient import GradientParams
        q = q or self.p
        vals = {}
        for name in GradientParams.model_fields:
            v = self.s.param_value(q, "gradient", name)
            if v is not None:
                vals[name] = v
        return GradientParams(**vals)

    def _applied_key(self):
        st = self.s.state_info(self.p, "gradient") if hasattr(self.s, "state_info") else None
        return (self.p.pid, getattr(st, "key", None), getattr(st, "previous_key", None),
                self.s.state(self.p, "gradient"))

    def _load(self, p):
        """The workflow shown (another one, an Undo, a run, the project opened again): its draft, the samples drawn
        (the preview's, else the ones the last run used)."""
        for vpid, (opid, d) in list(self._stashed.items()):
            if self.s.view(vpid) is None:      # the variant undone: the draft it took is the original's again
                self._stashed.pop(vpid)
                self.carried.pop(vpid, None)
                if d and not self.has_draft(opid):
                    self.drafts[opid] = d
        if self.p is None or p.pid != self.p.pid:
            if self.p is not None:
                self.set_mode(None)
            self.p = p
            self._shown_gen = self.gen         # the answers of the requests made before are another workflow's
            self.result, self.result_ms, self.samples = None, None, None
            self._sel, self._selpt, self._veil = None, None, None
            if self.small is not None and self.small["pid"] != p.pid:
                self.small = None
            if self.survey is not None and self.survey["pid"] != p.pid:
                self.survey, self._survey_img = None, None
                if self.survey_shown:
                    self._run("survey")     # shown: the survey of the workflow now shown, from this computer only
        self.p = p
        try:
            sig = (p.pid, self._applied_params().model_dump_json(), self._applied_key())
        except Exception:  # noqa: BLE001 - settings not read yet: as they were
            sig = self._applied_at
        if sig != self._applied_at:            # the applied settings moved (Undo, Apply, a run)
            first = self._applied_at is None or self._applied_at[0] != p.pid
            self._applied_at = sig
            self.applied = None
            d = self._draft()
            if d is not None:                  # what the draft now holds as the applied value leaves it
                d.drop_pending({k for k, v in d.vals.items() if v != self._applied_value(k)})
            if not first and self.has_draft():
                self._schedule()
        self.work = self._fields()
        self.done = self.s.state(p, "gradient") == "done"
        c = self.carried.get(p.pid)
        if c is not None and self.done:        # the variant computed: its own result from now on
            self.carried.pop(p.pid)
        elif c is not None and self.result is None:
            self.result, self.result_key, self.result_ms, self.samples = c["result"], c["key"], c["ms"], c["samples"]
            self.small = {**c["small"], "pid": p.pid}
        if self.done and not self.has_draft():
            self.result, self.result_ms = None, None   # the computed result is what is shown
        if self.result is None:
            self.samples = self._computed_samples()

    def _computed_samples(self):
        from ..api import stage_outputs
        try:
            out = stage_outputs(self.s.p, self.p.pid, "gradient")
            z = np.load(f"{out.dir}/samples.npz") if out is not None else None
        except (OSError, ValueError, KeyError, AttributeError):
            return None
        if z is None or "box" not in z.files:
            return None
        return {"x": z["x"], "y": z["y"], "used": z["used"].astype(bool), "manual": bool(z["manual"]),
                "box": float(z["box"])}

    # ------------------------------------------------------------ the page: Inspect's centre for the block
    def attach(self, view, p):
        """Kept for Inspect's generic image tab: the Gradient block is shown on its own page (``center``)."""
        return

    def detach(self):
        """Inspect is built again (another block, the same one again, the Project block): a gesture cut, TuneView's
        panel given back (the page takes it away again when it stays)."""
        self.gesture = False
        self._grab = self._stroke0 = self._ell = self._edit_ell = self._last = None
        if self._page_ok() and self.tv.center.indexOf(self.page) >= 0:
            self.hold()
            self._side(True)

    def _page_ok(self) -> bool:
        if self.page is None:
            return False
        from shiboken6 import isValid
        if not isValid(self.page) or not isValid(self.gv):     # taken with the centre (the Project block's form)
            self.page = self.gv = self.drawer = None
            self.view = None
            self._ui = {}
            return False
        return True

    def page_for(self, p, tab) -> bool:
        """The page shows the block's result: computed, or Outdated with its previous one; its Sky model tab and a
        block never computed are Inspect's own."""
        return tab in (None, "result") and (self.drafting(p) or p.pid in self.carried or self.no_input(p))

    def no_input(self, p=None) -> bool:
        """Whether the stack the block reads does not exist yet (not computed, no earlier result either) and the block
        has no result of its own: the page then shows the method only, the rest greyed."""
        p = p or self.p
        if p is None or self.drafting(p) or p.pid in self.carried:
            return False
        try:
            st = self.s.state(p, "stack")
            if st == "done":
                return False
            return not (st == "stale" and self.tv._has_previous(p, "empilement"))
        except Exception:  # noqa: BLE001 - no state readable: as a block with its input
            return False

    def _update_cover(self):
        """The image area while the stack does not exist: the block's 'Not computed yet' and its Run to here, over the
        image's own rectangle (which does not change)."""
        cover = self._ui.get("cover")
        if cover is None:
            return
        try:
            want = self.no_input()
            if want and not cover.isVisible():
                from .widgets import clear
                clear(cover.layout())
                box = cover.layout()
                box.addStretch(1)
                box.addWidget(label(_("Not computed: run to see the picture"), "dim"), 0, Qt.AlignHCenter)
                box.addSpacing(8)
                rb = self.tv._run_button(self.p, "fond")
                rb.setObjectName("primary")
                box.addWidget(rb, 0, Qt.AlignHCenter)
                if not self.s.has_data(self.p):
                    box.addWidget(label(self.tv._no_data(self.p), "warn"), 0, Qt.AlignHCenter)
                box.addStretch(2)
            cover.setGeometry(self.gv.rect())
            cover.setVisible(want)
            if want:
                cover.raise_()
        except RuntimeError:
            pass

    def center(self, tv, node, p) -> bool:
        """From TuneView.refresh: the centre is this page when the block shown is the Gradient with a result (True);
        otherwise the page leaves the centre, kept for the next time."""
        want = (node is not None and p is not None and str(tv.s.sel_step) == "fond" and not node.cut
                and self.page_for(p, tv.tab_of.get("fond")))
        if not want:
            self._leave_page()
            return False
        self._enter_page(p)
        return True

    def _sides(self) -> list:
        tv = self.tv
        d = getattr(tv, "dock", None)
        out = [getattr(tv, "panel", None)] + ([d.strip, d.arrow] if d is not None else [])
        return [x for x in out if x is not None]

    def _page_shown(self) -> bool:
        try:
            return self._page_ok() and self.tv.center.indexOf(self.page) >= 0 and self.page.isVisibleTo(self.tv)
        except RuntimeError:
            return False

    def _side(self, show: bool):
        """TuneView's settings panel: away while the page shows (its Settings drawer holds them), back after."""
        tv = self.tv
        d = getattr(tv, "dock", None)
        try:
            hb = getattr(tv, "head_box", None)  # Inspect's header: the page has its own (title, Before / After, Settings)
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

    def _enter_page(self, p):
        tv = self.tv
        new = self.p is None or self.p.pid != p.pid
        if not self._page_ok():
            self._load(p)                              # the page is made from the block's settings
            self._build_page()
            for x in self._sides():
                x.installEventFilter(self)
        if tv.center.indexOf(self.page) < 0:
            tv._clear_center()
            tv.center.addWidget(self.page, 1)
            self.page.show()
        self._side(False)
        spec = next((sp for sp in tv.visuals(p, "fond") if sp.key == "result"), None)
        tv.view, tv.views = self.gv, []
        tv._cur = (p, "fond", spec) if spec is not None else None      # the tab Inspect's walkthrough and tests read
        tv._center_sig = ("gradient", p.pid)
        self._load(p)
        self.view = self.gv
        self._watch_window()
        nostack = self.no_input(p)
        if self._ui.get("cover") is not None:
            self._ui["cover"].hide()                   # built again below with the block's present state
        if "samples" in self.kinds() and not nostack:
            if p.pid not in self.defaults and not self._applied_value("sample_points"):
                # the block places its own samples: the same grid, made here as the stage makes it
                self._run_points(["generate", "objects", "nebula_disk"], [], default=True)
            if self.mode is None:
                self.set_mode("points")        # the samples are edited at once: a click adds one
        self._fill_drawer()
        self._sync()
        self._present(fresh=new)
        if self.has_draft() and self.result is None:
            self._schedule()

    def _leave_page(self):
        if not self._page_ok() or self.tv.center.indexOf(self.page) < 0:
            return
        self.hold()
        self.tv.center.removeWidget(self.page)
        self.page.hide()
        self.page.setParent(None)
        self._side(True)
        if self.tv.view is self.gv:
            self.tv.view = None
        self.view = None

    def _build_page(self):
        """Built once, whole: nothing is added to or taken from it afterwards (a control that does not apply is greyed
        with its reason), so the image's rectangle never moves while the page is shown. Read in order: at the left the
        method (the block's main decision), the tools that act on the image, Marks, the draft and its actions; the image
        in the middle; at the right the method's settings and the rest, a side panel kept open, automatic or folded as
        the window's own (``layout.Dock``), folded or automatic over the image."""
        from .imageview import ImageView
        from .layout import Dock
        page = QWidget()
        page.setObjectName("gradpage")
        page.setStyleSheet(
            "QPushButton#tool { text-align: left; padding: 4px 10px; }"
            "QPushButton#tool:checked { background: #2b3446; border-color: #5ea1ff; color: #ffffff; }"
            "QPushButton#ivseg { padding: 3px 9px; border-radius: 0; }"
            "QPushButton#ivseg:checked { background: #2b3446; border-color: #5ea1ff; color: #ffffff; }"
            "QPushButton#marks { padding: 6px 10px; font-weight: 600; }"
            f"QPushButton#marks:checked {{ background: #3a2f12; border-color: {WARN}; color: #ffd27a; }}")
        self._ui = {}
        v = QVBoxLayout(page)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(6)
        v.addWidget(self._title_row())
        body = QWidget()
        hb = QHBoxLayout(body)
        hb.setContentsMargins(0, 0, 0, 0)
        hb.setSpacing(0)
        col = QFrame()
        col.setObjectName("gradcol")
        cv = QVBoxLayout(col)
        cv.setContentsMargins(0, 0, 10, 0)
        cv.setSpacing(10)
        cv.addWidget(self._method_part())
        cv.addWidget(self._tools_part())
        cv.addStretch(1)
        cv.addWidget(self._actions_part())
        col.ensurePolished()
        col.setFixedWidth(max(240, col.sizeHint().width()))   # once, from what it holds: the image never moves
        hb.addWidget(col)
        box = QWidget()
        bl = QVBoxLayout(box)
        bl.setContentsMargins(0, 0, 0, 0)
        gv = ImageView(post=False, compare=True)
        gv.default_dir = lambda: str(getattr(self.s, "root", None) or "")
        gv.toasted.connect(self.tv.toasted.emit)
        gv.layout().itemAt(0).widget().hide()          # its own toolbar: the page's Before / After and the panel
        gv.readout.parentWidget().hide()               # the readout line: none under the image
        bl.addWidget(gv)
        cover = QFrame(gv)                             # shown over the image while the stack does not exist
        cover.setObjectName("gradcover")
        cover.setStyleSheet("QFrame#gradcover { background: #15171c; }")     # the dark empty state of the other views
        QVBoxLayout(cover).setContentsMargins(0, 0, 0, 0)
        cover.hide()
        self._ui["cover"] = cover
        gv.installEventFilter(self)
        hb.addWidget(box, 1)
        v.addWidget(body, 1)
        self.page, self.gv = page, gv
        self.drawer = SettingsPanel(self)
        self.dock = Dock(body, self.drawer, lambda: hb.addWidget(self.drawer), lambda: hb.removeWidget(self.drawer),
                         title=_("Settings"), role="adjust")
        hb.insertWidget(hb.indexOf(box) + 1, self.dock.strip)
        hb.addWidget(self.drawer)                      # docked to begin with; then the state the user chose, or the
        self.dock.set_state(self._side_state())        # window's size says
        self.dock.chosen.connect(self._side_chosen)
        gv.panel_box.setParent(None)                   # the stretch column: in the panel, never beside the image
        self.drawer.fixed.addWidget(gv.panel_box)
        gv.show_histogram(True)
        gv.align_bar.setParent(None)
        self.drawer.fixed.addWidget(gv.align_bar)
        gv.add_overlay(self._paint)
        gv.imageShown.connect(self._shown)
        for x in (gv.canvas, page):                    # M held hides the marks while it is down
            x.installEventFilter(self)

    def _side_state(self) -> str:
        """The settings panel's state: the user's choice when he made one, else by the window's width (kept open on a
        large window, automatic on a small one)."""
        try:
            from . import prefs
            st = prefs.get("gradient_settings_side")
        except Exception:  # noqa: BLE001 - no preference readable: by the width
            st = None
        if st in ("docked", "icons", "auto"):
            return st
        return "docked" if self.tv.window().width() >= SMALL_W else "auto"

    def _side_chosen(self, st: str):
        try:
            from . import prefs
            prefs.set("gradient_settings_side", st)
        except Exception:  # noqa: BLE001 - not kept: the next time by the width again
            pass

    @staticmethod
    def _fixed_rows(w: QWidget) -> QWidget:
        """A row above the image: as high as what it holds (which never changes), never given the image's room."""
        w.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        return w

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
        ba = QPushButton(_("Before / After"))
        ba.setObjectName("toggle")
        ba.setCheckable(True)
        ba.toggled.connect(self._set_before)
        top.addWidget(ba)
        u["before"] = ba
        return self._fixed_rows(w)

    def _method_part(self) -> QWidget:
        """The method: a list of the three, named in full (each one's line in the list's tooltips), and one line under
        it saying what the chosen one does."""
        from PySide6.QtWidgets import QComboBox
        u = self._ui
        w = QWidget()
        col = QVBoxLayout(w)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(4)
        col.addWidget(label(_("Method"), "h2"))
        cb = QComboBox()
        cb.setObjectName("method")
        for i, m in enumerate(METHODS):
            cb.addItem(_(METHOD_NAME[m]), m)
            cb.setItemData(i, _(METHOD_LINE[m]), Qt.ToolTipRole)
        cb.activated.connect(lambda i: self.set_draft({"method": METHODS[i]}))
        col.addWidget(cb)
        u["method"] = u["methods"] = cb
        line = label("", "small", wrap=True)
        line.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)     # the column's width, never its own
        fm = line.fontMetrics()
        line.setMinimumHeight(2 * fm.height() + 2)     # two lines kept: the longest; the column never moves
        col.addWidget(line)
        u["m_line"] = line
        return w

    def _tools_part(self) -> QWidget:
        """The tools that act on the image, each with its icon AND its name (its sentence on hover), in two columns;
        under them Marks, wide and lit while the marks are shown."""
        from PySide6.QtWidgets import QGridLayout

        from .icons import set_icon
        u = self._ui
        w = QFrame()
        w.setObjectName("gradtools")
        lay = QGridLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        lay.addWidget(label(_("Tools"), "h2"), 0, 0, 1, 2)
        for n, (key, role, name, tip) in enumerate(TOOLS):
            b = QPushButton(_(name))
            set_icon(b, role, 16)
            b.setObjectName("tool")
            b.setCheckable(key != "grid")
            b.setToolTip(_(tip))
            if key == "grid":
                b.clicked.connect(self.generate)
            else:
                b.clicked.connect(lambda _c=False, k=key: self.set_mode(None if self.mode == k else k))
            lay.addWidget(b, 1 + n // 2, n % 2)
            u[key] = b
        row = 1 + (len(TOOLS) + 1) // 2
        size = QWidget()                       # the Brush's and the Eraser's size, under them
        sh = QHBoxLayout(size)
        sh.setContentsMargins(0, 0, 0, 0)
        sh.setSpacing(6)
        sh.addWidget(label(_("Size"), "dim"))
        sz = QSlider(Qt.Horizontal)
        sz.setRange(4, 1200)
        sz.setValue(int(self.size))
        sz.setToolTip(_("Brush and Eraser size, in pixels of the stack ([ ])"))
        sz.valueChanged.connect(self._set_size)
        sh.addWidget(sz, 1)
        szl = label(f"{int(self.size)} px", "small")
        szl.setMinimumWidth(szl.fontMetrics().horizontalAdvance("1200 px") + 2)
        sh.addWidget(szl)
        lay.addWidget(size, row, 0, 1, 2)
        u["size"], u["size_l"] = sz, szl
        row += 1
        lay.addWidget(label(_("Samples"), "h2"), row, 0, 1, 2)
        for n, (key, name, tip) in enumerate(ACTIONS):
            b = QPushButton(_(name))
            b.setToolTip(_(tip))
            b.clicked.connect(lambda _c=False, k=key: self.act(k))
            if key == "generate":
                lay.addWidget(b, row + 1, 0)
                pitch = QSpinBox()
                pitch.setRange(4, 64)
                pitch.setValue(self._density())
                pitch.setSuffix(" " + _("per side"))
                pitch.setKeyboardTracking(False)
                pitch.setToolTip(_("The grid's pitch: samples along the shorter side of the image"))
                pitch.valueChanged.connect(self._set_density)
                lay.addWidget(pitch, row + 1, 1)
                u["pitch"] = pitch
            else:
                lay.addWidget(b, row + 2 + (n - 1) // 2, (n - 1) % 2)
            u[key] = b
        row += 2 + (len(ACTIONS) - 1 + 1) // 2
        mk = QPushButton(_("Marks"))
        set_icon(mk, "eye", 16)
        mk.setObjectName("marks")
        mk.setCheckable(True)
        mk.setChecked(self.shown)
        mk.setToolTip(_("Show or hide the samples and the excluded areas over the image, to judge the result without "
                        "them (M; M held: hidden while it is down). They come back as soon as a tool is used."))
        mk.toggled.connect(self._set_shown)
        lay.addWidget(mk, row, 0, 1, 2)
        u["marks"] = mk
        return w

    def _actions_part(self) -> QWidget:
        """The draft bar every block's panel has (blockdraft.DraftBar), in the column's width: the draft's state or its
        changes, what Apply computes again, Reset and Try variant side by side, Apply under them. Always there (greyed
        without a draft): the image never moves."""
        u = self._ui
        bar = DraftBar(narrow=True)
        bar.what.setMinimumHeight(2 * bar.what.fontMetrics().height() + 4)
        bar.apply.connect(lambda: self.apply())
        bar.variant.connect(lambda: self.try_variant())
        bar.reset.connect(lambda: self.reset())
        u["bar"], u["state"] = bar, bar.what
        u["apply"], u["variant"], u["reset"] = bar.b_apply, bar.b_variant, bar.b_reset
        return bar

    def kinds(self) -> tuple:
        """The kinds of marks the draft's method reads (MARK_KINDS)."""
        return MARK_KINDS.get(self._method(), ())

    def _marks_key(self, down: bool):
        """M: tapped, the marks shown or hidden; held, hidden while it is down, back as they were on release; nothing
        when the method reads no mark (Marks greyed)."""
        if not self.kinds():
            self._m_down = None
            return
        if down:
            self._m_down = (time.monotonic(), self.shown)
            if self.shown:
                self.shown = False
                self.gv.canvas.update()
            return
        if self._m_down is None:
            return
        t0, was = self._m_down
        self._m_down = None
        self._set_shown(was if time.monotonic() - t0 > MARKS_HOLD_S else not was)

    def _set_before(self, on: bool):
        self.before = bool(on)
        self._present()

    def _set_compare(self, mode: str):
        self.cmp = mode
        self._present()
        self._sync()

    def _main_spec(self):
        return MAIN[self._method() if self._method() in MAIN else "polynomial"]

    def _main_moved(self, i: int):
        name, text, lo, hi, step, dec, tip = self._main_spec()
        val = lo + i * step
        if "main_v" in self._ui:
            self._ui["main_v"].setText(f"{val:.{dec}f}")
        sl = self._ui.get("main")
        if sl is not None and not sl.isSliderDown() and not self._syncing:
            self._main_set(i)

    def _main_set(self, i: int):
        name, text, lo, hi, step, dec, tip = self._main_spec()
        val = lo + i * step
        self.set_draft({name: int(round(val)) if dec == 0 else round(val, dec)})

    def _fill_drawer(self):
        """The settings panel's groups, as the draft has them, the method's settings first (built again when the method
        or a setting changes: the panel's width is fixed, the image does not move)."""
        if self.drawer is None or self.p is None:
            return
        from .icons import set_icon
        from .panel import CompactRow, kv, unit_column
        from .widgets import clear
        dv = self.drawer.rows
        clear(dv)
        p = self.p
        u = self._ui

        def title(text):
            t = label(_(text), "h2")
            t.setContentsMargins(0, 8, 0, 2)
            dv.addWidget(t)

        groups = self.groups(p)
        rows = [r for _t, rs in groups for r in rs if r.get("level") != "system"]
        if rows:
            title("Method settings")
            uw = unit_column([r.get("unit") for r in rows])
            for r in rows:
                cr = CompactRow(self.draft_row(r), uw)
                cr.changed.connect(self.draft_setting)
                dv.addWidget(cr)
        title("Draft")
        row = QHBoxLayout()
        for key, tip, fn in (("undo", "Undo the draft's last step (Ctrl+Z)", self.undo),
                             ("redo", "Redo the draft's step (Ctrl+Y)", self.redo)):
            b = icon_only(QPushButton(), key, _(tip), 16)
            b.clicked.connect(lambda _c=False, f=fn: f())
            row.addWidget(b)
            u[key] = b
        row.addStretch(1)
        dv.addLayout(row)
        if not self.done and self.s.state(p, "gradient") != "running":
            dv.addWidget(self.tv._run_button(p, "fond"), 0, Qt.AlignLeft)
        title("H-alpha survey")
        here = self.survey_here()
        r4 = QHBoxLayout()
        r4.addWidget(label(_("On this computer") if here else _("Not on this computer"), "small"))
        if not here:
            fb = QPushButton(_("Fetch"))
            fb.setToolTip(_("Fetches the professional H-alpha survey of this field once, kept on this computer "
                            "(never during a run)"))
            fb.clicked.connect(lambda: self._fetch_survey(then=None))
            r4.addWidget(fb)
            u["fetch"] = fb
        r4.addStretch(1)
        dv.addLayout(r4)
        for b in self._view_toggles():
            b.setEnabled(here)
            if not here:
                b.setToolTip(_("The survey is not on this computer: Fetch it first"))
            dv.addWidget(b, 0, Qt.AlignLeft)
        box = QVBoxLayout()
        dv.addLayout(box)
        try:                                            # the Gradient block's survey row, as the panel draws it
            from types import SimpleNamespace

            from .panel import Panel
            Panel.survey_row(SimpleNamespace(hv=box, s=self.s, changed=self.tv.panel.changed,
                                             toast=self.tv.panel.toast), p)
        except Exception:  # noqa: BLE001 - the row is the panel's: without it, the toggle above still shows the survey
            pass
        title("Compare")
        seg = QHBoxLayout()
        seg.setSpacing(0)
        for mode, text, tip in (("toggle", "Before / After", "Before / After at the top switches the image"),
                                ("split", "Split", "Before on the left of the divider, after on the right"),
                                ("side", "Side by side", "The two images next to each other, zoom and pan linked")):
            b = QPushButton(_(text))
            b.setObjectName("ivseg")
            b.setCheckable(True)
            b.setChecked(self.cmp == mode)
            b.setToolTip(_(tip))
            b.clicked.connect(lambda _c=False, m=mode: self._set_compare(m))
            seg.addWidget(b)
        seg.addStretch(1)
        dv.addLayout(seg)
        neu = QCheckBox(_("Neutral sky (display)"))
        st = self.gv.stretch() if self.gv is not None else None
        neu.setChecked(bool(getattr(st, "neutral", True)))
        neu.setToolTip(_("Each channel scaled on the screen so that the sky is grey and the stars white. The data stay "
                         "in the camera's colours."))
        neu.toggled.connect(lambda on: self.gv.set_neutral(bool(on)))
        dv.addWidget(neu)
        model = QPushButton(_("Sky model"))
        set_icon(model, "samples", 16)
        model.setToolTip(_("The sky the block removes, on Inspect's own tab"))
        model.clicked.connect(self._sky_model)
        dv.addWidget(model, 0, Qt.AlignLeft)
        facts = self.tv.steps.facts(p, "fond") if getattr(self.tv, "steps", None) is not None else None
        if facts:
            title("Measurements")
            dv.addWidget(kv(facts))
        title("Display")

    def _sky_model(self):
        """The Sky model tab of the block: Inspect's own view of it (its tabs lead back)."""
        tv = self.tv
        keys = [sp.key for sp in tv.visuals(self.p, "fond")]
        other = next((k for k in keys if k != "result"), None)
        if other is not None:
            tv.tab_of["fond"] = other
            tv.request_refresh()

    def _view_toggles(self) -> list:
        """The H-alpha survey drawn under the samples (R), in the settings panel (Marks is at the left, by the tools)."""
        out = []
        for key, text, tip, on, slot in (
                ("survey", "Show survey", "Shows in amber, over the image, where the professional H-alpha survey of "
                 "the field sees nebulosity (R): why Avoid nebulosity removes a sample",
                 self.survey_shown, self._set_survey_shown),):
            b = QPushButton(_(text))
            b.setObjectName("toggle")
            b.setCheckable(True)
            b.setChecked(on)
            b.setToolTip(_(tip))
            b.toggled.connect(slot)
            self._ui[key] = b
            out.append(b)
        return out

    def _present(self, fresh: bool = False):
        """The image of the page: After (the draft's preview, else the applied result) or Before (the stack), or both
        with Split / Side by side. The part of the sky shown and its size on the screen stay as they were."""
        from .imageview import ViewImage
        if not self._page_ok() or self.p is None:
            return
        gv, p = self.gv, self.p
        pid = p.pid
        if self.has_draft() and self.too_few() and self.small is not None and self.small["pid"] == pid:
            after = ViewImage(self.small["arr"], _("No gradient removed"), key=("gradient-none", pid),
                              camera=True, neutral=True)
            before = after
        elif self.result is not None and self.small is not None and self.small["pid"] == pid:
            after = ViewImage(self.result, _("Draft preview") if self.has_draft() else _("Preview"),
                              key=self.result_key, camera=True, neutral=True)
            before = ViewImage(self.small["arr"], _("Stack"), key=("gradient-small", pid), camera=True, neutral=True)
            c = self.carried.get(pid)
            if c is not None and c.get("orig") is not None:
                before = ViewImage(c["orig"], _("Result of {name}").format(name=c["of"]), key=("gradient-orig", pid),
                                   camera=True, neutral=True)
        else:
            after, before = self._applied_vi(p), self._stack_vi(p)
        main, other = (before, after) if self.before and self.cmp == "toggle" else (after, before)
        if main is None:
            main, other = after or before, None
        if main is None:
            return
        try:
            if gv._want != main.identity():
                if gv.canvas.display is not None and not fresh:
                    self._restore = (main.identity(), gv.view_state())
                gv.set_image(main, keep_view=not fresh, keep_stretch=not fresh)
            if self.cmp == "toggle" or other is None:
                gv.set_compare_targets([], None)
            else:
                gv.set_compare_mode(self.cmp)
                gv.set_compare_targets([other], 0)
                gv.set_same_stretch(True)
            gv.canvas.tool = self if self.mode else None
            gv.canvas.update()
        except RuntimeError:
            pass
        self._sync()

    def _applied_vi(self, p):
        tv = self.tv
        try:
            spec = next(sp for sp in tv.visuals(p, "fond") if sp.key == "result")
            vi = tv.steps.view_image(p, spec)
        except (StopIteration, Exception):  # noqa: BLE001 - nothing to show: the page says why on its line
            return None
        if vi is not None:
            vi.name = _("Previous settings") if self.s.state(p, "gradient") == "stale" else _("Applied result")
        return vi

    def _stack_vi(self, p):
        tv = self.tv
        try:
            spec = next(sp for sp in tv.visuals(p, "fond") if sp.key == "result")
            targets = tv._targets(p, "fond", spec)
        except (StopIteration, Exception):  # noqa: BLE001 - no stack to compare with: Before greyed
            return None
        vi = targets[0] if targets else None
        if vi is not None:
            vi.name = _("Stack")
        return vi

    # ------------------------------------------------------------ the panel's section (Inspect's page has the drawer)
    def controls(self, p) -> QWidget:
        """What TuneView's panel shows of the block while the page holds its settings (the panel is away then)."""
        self._load(p)
        return label(_("The settings are in the page's Settings."), "small", wrap=True)

    def groups(self, p) -> list:
        """The settings groups of the draft's method (the applied method's are the block's own)."""
        from .model import GRADIENT_VIEW
        from .settings import CATALOG, catalog_key, resolve, step_settings
        key = "fond:" + GRADIENT_VIEW.get(self._method(), "simple")
        if key == catalog_key(p, "fond") or key not in CATALOG:
            return list(step_settings(self.s, p, "fond"))
        out = []
        for title, items in CATALOG[key]:
            rs = [r for r in (resolve(self.s, p, "fond", it) for it in items if it["kind"] == "engine") if r is not None]
            if rs:
                out.append((_(title), rs))
        return out

    def draft_row(self, r: dict) -> dict:
        """A settings row as the draft has it."""
        d = self._draft()
        if d is None or r.get("name") not in d.vals:
            return r
        val = d.vals[r["name"]]
        return {**r, "value": val, "changed": val != r.get("suggested")}

    def draft_setting(self, r: dict, value):
        """A settings row edited: into the draft (None: back to the suggestion)."""
        self._from_row = True                  # the row shows its value already: never built again under the mouse
        try:
            self.set_draft({r["name"]: r.get("suggested") if value is None else value})
        finally:
            self._from_row = False

    def _status(self) -> str:
        if self.has_draft():
            n = len(self._draft())
            text = _("Draft · 1 change") if n == 1 else _("Draft · {n} changes").format(n=n)
        elif self.result is not None and not self.done:
            text = _("Preview · not computed yet")
        elif self.s.state(self.p, "gradient") == "stale":
            text = _("Applied · not computed yet")
        else:
            text = _("Applied settings")
        if self.busy:
            text += " · " + (self.step_text or _("computing the preview…"))
        return text

    def _hint(self):
        """No help line under the image: each tool says what it does in its tooltip."""
        return

    def _sync(self):
        """The page's controls as the editor is (never built again, never shown or hidden: greyed with a reason)."""
        if self.p is None or not self._ui:
            return
        self._syncing = True
        try:
            u = self._ui
            method = self._method()
            kinds = self.kinds()
            for key, _role, name, tip in TOOLS:
                b = u.get(key)
                if b is None:
                    continue
                if b.isCheckable():
                    b.setChecked(self.mode == key)
                kind = TOOL_KIND[key]
                on = kind in kinds if kind else bool(kinds)
                b.setEnabled(on)
                why = _(NO_KIND[kind]).format(method=_(METHOD_NAME.get(method, method)))
                b.setToolTip(f"<b>{_(name)}</b><br>{_(tip) if on else why}")
            if "marks" in u:                   # nothing to show: greyed, its reason on hover
                u["marks"].setEnabled(bool(kinds))
                if not kinds:
                    u["marks"].setToolTip(_(NO_KIND[""]).format(method=_(METHOD_NAME.get(method, method))))
            for m in METHODS:
                b = u.get("m_" + m)
                if b is not None:
                    b.setChecked(m == method)
            if "main" in u:
                name, text, lo, hi, step, dec, tip = self._main_spec()
                cur = self.value(name)
                cur = lo if cur is None else float(cur)
                sl = u["main"]
                sl.blockSignals(True)
                sl.setRange(0, int(round((hi - lo) / step)))
                sl.setValue(int(round((cur - lo) / step)))
                sl.blockSignals(False)
                sl.setToolTip(_(tip))
                u["main_l"].setText(_(text))
                u["main_v"].setText(f"{cur:.{dec}f}")
            draft = self.has_draft()
            why = _("No draft: change a setting, a sample or an area first")
            few = self.too_few()
            for k in ("reset", "variant", "apply"):
                if k in u:
                    u[k].setEnabled(draft and (k == "reset" or not few))
            for key, _name, tip in ACTIONS:
                if key in u:
                    on = "samples" in kinds and (key in ("generate",) or bool(self.work["sample_points"]))
                    u[key].setEnabled(on)
                    u[key].setToolTip(_(tip) if on else _(NO_KIND["samples"]).format(
                        method=_(METHOD_NAME.get(method, method))) if "samples" not in kinds else
                        _("No sample to act on"))
            if "pitch" in u:
                u["pitch"].setEnabled("samples" in kinds)
            if "apply" in u and draft and few:
                fw = _("Too few samples for {method}: place samples or generate a grid").format(
                    method=_(METHOD_NAME.get(method, method)))
                u["apply"].setToolTip(fw)
                u["variant"].setToolTip(fw)
            elif "apply" in u:
                c = self.cost_text()
                u["apply"].setToolTip((_("The draft becomes the block's settings (Ctrl+Z takes it back)") +
                                       ("\n" + c if c else "")) if draft else why)
                u["variant"].setToolTip(_("The draft on a variant of the workflow: what is computed stays as it is, "
                                          "the variant is computed by Run") if draft else why)
                u["reset"].setToolTip(_("Back to the applied settings: the draft goes") if draft else why)
            if "m_line" in u:
                u["m_line"].setText(_(METHOD_LINE.get(method, "")))
            if "method" in u:
                u["method"].setCurrentIndex(METHODS.index(method) if method in METHODS else 0)
                u["method"].setToolTip(_(METHOD_TIP.get(method, "")))
            if "bar" in u:                     # the shared draft bar: its changes and what Apply computes again
                ch = self.draft_change() if draft else None
                u["bar"].show_draft([self._status()] + ([ch["what"]] if ch else []), self.cost_text() if draft else "")
                u["state"].setStyleSheet(f"color: {WARN};" if draft else "color: #8b95a5;")
            if "title" in u:
                u["title"].setText(_("Gradient") + "  ·  " + self.p.name)
                u["title"].setToolTip(_("The stack with its sky gradient removed, in the camera's colours, as every "
                                        "later block reads it. The sky keeps its level."))
                stale = self.s.state(self.p, "gradient") == "stale" and not draft
                u["tag"].setText(_("Previous settings") if stale else "")
                from .tune import previous_tip
                u["tag"].setToolTip(previous_tip(self.s, self.p, "fond") if stale else "")
            if "before" in u:
                b = u["before"]
                b.blockSignals(True)
                b.setChecked(self.before and self.cmp == "toggle")
                b.blockSignals(False)
                b.setEnabled(self.cmp == "toggle")
                b.setToolTip(_("Switch the image between the gradient removed and the stack before it (B)")
                             if self.cmp == "toggle" else _("Split and Side by side show both: Settings › Compare"))
            for k, on in (("marks", self.shown), ("survey", self.survey_shown), ("avoid", self.avoid)):
                if k in u:
                    b = u[k]
                    b.blockSignals(True)
                    b.setChecked(on)
                    b.blockSignals(False)
            d = self._draft()
            if "undo" in u:
                u["undo"].setEnabled(bool(d is not None and d.undo))
                u["redo"].setEnabled(bool(d is not None and d.redo))
            if "size_l" in u:
                u["size_l"].setText(f"{int(self.size)} px")
            if self.no_input():                # no stack: only the method is live; the rest says why
                why = _("The stack is not computed yet")
                for k in [t[0] for t in TOOLS] + [a[0] for a in ACTIONS] + ["pitch", "size", "marks", "before"]:
                    b = u.get(k)
                    if b is not None:
                        b.setEnabled(False)
                        b.setToolTip(why)
                if "points" in u:
                    for k in [t[0] for t in TOOLS]:
                        if u[k].isCheckable():
                            u[k].setChecked(False)
            self._update_cover()
        except RuntimeError:                   # a drawer row built again meanwhile
            pass
        finally:
            self._syncing = False

    # ------------------------------------------------------------ modes
    def set_mode(self, mode: str | None):
        if mode == "grid":
            mode = None
        if mode and not self.shown:            # a marks tool: never edited blind
            self._set_shown(True)
        self.mode = mode
        self._sel = None
        if mode != "points":
            self._selpt = None
        if self.view is not None:
            try:
                c = self.view.canvas
                c.tool = self if mode else None
                if mode:
                    c.setFocus()               # the keys of the tool go to the image
                c.update()
            except RuntimeError:
                self.view = None
        if mode and self.small is None:
            self._run("load")
        self._hint()
        self._sync()

    def _density(self) -> int:
        """The grid's points along the shorter side: the block's setting (issue #15: one density, set here only)."""
        return int(self.value("samples") or 16)

    def _set_density(self, n: int):
        """The grid's pitch (the block's samples per side): Generate grid uses it; the samples stay as they are."""
        if n != self._density():
            self.set_draft({"samples": int(n)})

    def _set_paint(self, on: bool):
        self.paint = on
        self._sync()

    def _set_size(self, val):
        self.size = float(val)
        if "size_l" in self._ui:
            try:
                self._ui["size_l"].setText(f"{int(self.size)} px")
            except RuntimeError:
                pass
        if self.view is not None:
            self.view.canvas.update()

    def _set_shown(self, on: bool):
        self.shown = bool(on)
        self._sync()
        if self.view is not None:
            self.view.canvas.update()

    def survey_here(self) -> bool:
        """Whether the H-alpha survey of the field is on this computer (the chain's, or fetched since)."""
        if self.survey is not None and self.survey.get("pid") == getattr(self.p, "pid", None) and "level" in self.survey:
            return True
        try:
            from .. import api
            st = api.survey_state(self.s.p, self.p.pid)
        except Exception:  # noqa: BLE001 - nothing readable: not here
            return False
        return bool(st) and (st.get("survey", "none") != "none" or bool(st.get("available")))

    def _set_survey_shown(self, on: bool):
        """Show survey (R): the survey of the field over the image, read from this computer (Fetch is its own click)."""
        if on and not self.survey_here():
            on = False
        self.survey_shown = bool(on)
        if on and (self.survey is None or self.survey["pid"] != self.p.pid or "error" in self.survey):
            self._run("survey")
        self._sync()
        if self.view is not None:
            self.view.canvas.update()

    @property
    def avoid(self) -> bool:
        """Away from emission, as the draft has it (the samples the block chooses, and Generate grid's)."""
        return bool(self.value("avoid_emission")) if self.p is not None else True

    def _set_avoid(self, on: bool):
        if bool(on) != self.avoid:
            self.set_draft({"avoid_emission": bool(on)})

    def wants(self, e) -> bool:
        """Whether the image takes this key before the shortcuts (the canvas asks, ShortcutOverride): the tools' keys.
        Ctrl+Z / Ctrl+Y are the window's: its one order of Undo takes the draft's step when it is the latest change."""
        if e.modifiers() & Qt.ControlModifier:
            return False
        return e.key() in TOOL_KEYS and not e.modifiers() & (Qt.ControlModifier | Qt.AltModifier)

    def key(self, e) -> bool:
        """The keys of the tool, from the canvas (it has the focus while a tool is on)."""
        k = e.key()
        if e.modifiers() & (Qt.ControlModifier | Qt.AltModifier):
            return False
        if k == Qt.Key_BracketLeft and self.mode in ("brush", "eraser"):
            self._resize(1 / 1.25)
        elif k == Qt.Key_BracketRight and self.mode in ("brush", "eraser"):
            self._resize(1.25)
        elif k == Qt.Key_B and self.cmp == "toggle":
            self._set_before(not self.before)
        elif k == Qt.Key_M:
            self._set_shown(not self.shown)
        elif k == Qt.Key_R:
            self._set_survey_shown(not self.survey_shown)
        elif k in (Qt.Key_Delete, Qt.Key_Backspace) and (self._sel is not None or self._selpt is not None):
            self.remove_selected()
        elif k == Qt.Key_Escape:
            self.set_mode(None)
        else:
            return False
        return True

    def _resize(self, k: float):
        self.size = float(np.clip(self.size * k, 4, 1200))
        if "size" in self._ui:
            try:
                self._ui["size"].blockSignals(True)
                self._ui["size"].setValue(int(self.size))
                self._ui["size"].blockSignals(False)
            except (RuntimeError, AttributeError):
                pass
        self._set_size(self.size)

    # ------------------------------------------------------------ the edits: into the draft
    def _commit(self):
        """The edits of the image into the draft: one step of its Undo; the preview follows."""
        self._veil = None
        vals = {f: self.work[f] for f in FIELDS}
        vals["exclude_brush"] = [[round(x, 1), round(y, 1), round(r, 1), float(d)] for x, y, r, d in vals["exclude_brush"]]
        vals["sample_points"] = [[round(x, 1), round(y, 1)] for x, y in vals["sample_points"]]
        self.set_draft(vals)
        self._sync()
        if self.view is not None:
            self.view.canvas.update()

    def clear(self):
        self.work = {f: [] for f in FIELDS}
        self._sel = self._selpt = None
        self._commit()

    def remove_selected(self):
        """The sample or the ellipse selected goes (the visible way, besides the right click and Delete)."""
        if self.mode == "ellipse" and self._sel is not None and self._sel < len(self.work["exclude_ellipses"]):
            self.work["exclude_ellipses"].pop(self._sel)
            self._sel = None
        elif self._selpt is not None and self._selpt < len(self.work["sample_points"]):
            self.work["sample_points"].pop(self._selpt)
            self._selpt = None
        else:
            return
        self._commit()

    def act(self, key: str):
        """An action on the samples: one step of the draft's Undo, what it did said in the image's tag."""
        if not self.shown:
            self._set_shown(True)
        if "samples" not in self.kinds():
            return
        n0 = len(self.work["sample_points"])
        if key == "clearall":
            self.set_draft({"sample_points": []})
            self.say(_("{n} samples removed").format(n=n0))
        elif key == "generate":
            pts = self.grid_points_of()
            self.set_draft({"sample_points": pts})
            self.say(_("{n} samples on a grid").format(n=len(pts)))
        elif key in ("objects", "nebula"):
            self._run_points([key], [list(x) for x in self.work["sample_points"]])

    def grid_points_of(self) -> list:
        """A regular grid of ``samples`` per shorter side (the stage's generated grid's places), in stack pixels."""
        W, H = self._stack_size()
        n = self._density()
        step = min(H, W) / n
        return [[round(float(x), 1), round(float(y), 1)] for y in np.arange(step / 2, H, step)
                for x in np.arange(step / 2, W, step)]

    def _run_points(self, ops: list, pts: list, default: bool = False):
        self._ops = (list(ops), pts, default, self.p.pid)
        self._run("points")

    def _fetch_survey(self, then: str | None = "nebula"):
        """The survey fetched once, on the user's click (never during a run): from Avoid nebulosity (then applies it)
        or the panel's Fetch."""
        from .services.survey import job, where
        j = job()
        pid, root = self.p.pid, where(self.s.p)

        def ended(eroot, epid, why):
            try:
                j.finished.disconnect(ended)
            except (RuntimeError, TypeError):
                pass
            if epid != pid or eroot != root:
                return
            if why:
                self.say(_("H-alpha survey: {why}").format(why=_(why)))
                return
            self.survey = None                 # read again from this computer, then the action
            self._fill_drawer()
            if then == "nebula":
                self._run_points(["nebula"], [list(x) for x in self.work["sample_points"]])
        j.finished.connect(ended, Qt.QueuedConnection)
        if not j.start(self.s.p, pid):
            j.finished.disconnect(ended)
            self.say(_("H-alpha survey: a fetch is already under way"))
        else:
            self.say(_("H-alpha survey: fetching…"))

    def generate(self):
        """Generate grid (kept for the keyboard and the tests): the action."""
        self.act("generate")

    def _generate(self):
        """Generate grid: a grid of samples, the brighter areas left out (and, Away from emission, the survey's
        emission, read from this computer first); they become placed samples of the draft."""
        if self.avoid and (self.survey is None or self.survey["pid"] != self.p.pid):
            self._grid_after_survey = True
            self._run("survey")
            self._sync()
            return
        self._run("grid")

    # ------------------------------------------------------------ the tool (ImageCanvas.tool)
    @property
    def cursor(self):
        return Qt.BlankCursor if self.mode in ("brush", "eraser") else Qt.CrossCursor

    def _stack_size(self) -> tuple[int, int]:
        """(W, H) of the stack: the reduced copy's, else read from the stack (the block may not be computed)."""
        if self.small is not None and self.small["pid"] == self.p.pid:
            return self.small["W"], self.small["H"]
        if self._dims is None or self._dims[0] != self.p.pid:
            from ..api import stage_outputs
            try:
                out = stage_outputs(self.s.p, self.p.pid, "stack")
                H, W = np.load(f"{out.dir}/drz_full.npy", mmap_mode="r").shape[1:]
            except (OSError, ValueError, AttributeError):
                return 1, 1
            self._dims = (self.p.pid, int(W), int(H))
        return self._dims[1], self._dims[2]

    def _f(self, canvas) -> float:
        """Displayed image pixels per stack pixel."""
        if canvas.display is None:
            return 1.0
        return canvas.display.size()[0] / max(self._stack_size()[0], 1)

    def _pick(self, canvas, sx, sy) -> int | None:
        pts = self.work["sample_points"]
        if not pts:
            return None
        f = self._f(canvas)
        d = [math.hypot((x - sx) * f * canvas.scale, (y - sy) * f * canvas.scale) for x, y in pts]
        i = int(np.argmin(d))
        return i if d[i] <= PICK_PX else None

    def press(self, canvas, ix, iy, button) -> bool:
        took = self._press(canvas, ix, iy, button)
        self.gesture = took and button == Qt.LeftButton
        return took

    def _press(self, canvas, ix, iy, button) -> bool:
        f = self._f(canvas)
        sx, sy = ix / f, iy / f
        if self.mode == "points":
            i = self._pick(canvas, sx, sy)
            if button == Qt.RightButton:
                if i is not None:
                    self.work["sample_points"].pop(i)
                    self._selpt = None
                    self._commit()
                return True
            if button == Qt.LeftButton:
                if i is None:
                    self.work["sample_points"].append([sx, sy])
                    i = len(self.work["sample_points"]) - 1
                self._grab = self._selpt = i
                self._sync()
                return True
        elif self.mode == "brush" and button == Qt.LeftButton:
            self._last = None
            self._stroke0 = len(self.work["exclude_brush"])
            self._dab(sx, sy)
            return True
        elif self.mode == "eraser" and button == Qt.LeftButton:
            # what is under the cursor: a sample, else an ellipse, else the painted areas (erased as a stroke); only
            # the kinds the method reads, the ones drawn (the others stay in the draft, unseen and untouched)
            kinds = self.kinds()
            i = self._pick(canvas, sx, sy) if "samples" in kinds else None
            if i is not None:
                self.work["sample_points"].pop(i)
                self._selpt = None
                self._commit()
                return True
            ells = self.work["exclude_ellipses"]
            hit = next((j for j in reversed(range(len(ells))) if _inside(ells[j], sx, sy)), None) \
                if "areas" in kinds else None
            if hit is not None:
                ells.pop(hit)
                self._sel = None
                self._commit()
                return True
            if "areas" not in kinds:
                return True
            self._last = None
            self._stroke0 = len(self.work["exclude_brush"])
            self._dab(sx, sy, paint=False)
            return True
        elif self.mode == "ellipse":
            ells = self.work["exclude_ellipses"]
            if button == Qt.RightButton:
                for j in reversed(range(len(ells))):
                    if _inside(ells[j], sx, sy):
                        ells.pop(j)
                        self._sel = None
                        self._commit()
                        break
                return True
            if button == Qt.LeftButton:
                # the handles of the selected ellipse, then inside an ellipse (select and move), else a new one
                h = self._handle(canvas, ix, iy)
                if h is not None:
                    self._edit_ell = (h, list(ells[self._sel]), (sx, sy))
                    return True
                hit = next((j for j in reversed(range(len(ells))) if _inside(ells[j], sx, sy)), None)
                if hit is not None:
                    self._sel = hit
                    self._edit_ell = ("move", list(ells[hit]), (sx, sy))
                    self._sync()
                    canvas.update()
                    return True
                self._sel = None
                self._ell = [(sx, sy), (sx, sy)]
                self._sync()
                return True
        return False

    def _handles(self, canvas, f) -> dict[str, tuple[float, float]]:
        """The selected ellipse's handles, in screen px: its four axis ends and the rotation one above it."""
        if self._sel is None or self._sel >= len(self.work["exclude_ellipses"]):
            return {}
        e = self.work["exclude_ellipses"][self._sel]
        _cx, _cy, rx, ry, _a = _ell(e)
        k = f * canvas.scale
        out = {}
        for name, (u, v) in (("rx", (rx, 0)), ("-rx", (-rx, 0)), ("ry", (0, ry)), ("-ry", (0, -ry)),
                             ("rot", (0, -ry - ROT_PX / max(k, 1e-9)))):
            x, y = _world(e, u, v)
            p = canvas.img_to_w(x * f, y * f)
            out[name] = (p.x(), p.y())
        return out

    def _handle(self, canvas, ix, iy) -> str | None:
        f = self._f(canvas)
        p = canvas.img_to_w(ix, iy)
        for name, (hx, hy) in self._handles(canvas, f).items():
            if math.hypot(p.x() - hx, p.y() - hy) <= HANDLE_PX + 4:
                return name
        return None

    def move(self, canvas, ix, iy):
        f = self._f(canvas)
        sx, sy = ix / f, iy / f
        self._hover = (ix, iy)
        if self.mode == "points" and self._grab is not None:
            self.work["sample_points"][self._grab] = [sx, sy]
        elif self.mode == "brush" and self._stroke0 is not None:
            self._dab(sx, sy)
        elif self.mode == "eraser" and self._stroke0 is not None:
            self._dab(sx, sy, paint=False)
        elif self.mode == "ellipse" and self._edit_ell is not None:
            what, e0, (x0, y0) = self._edit_ell
            cx, cy, rx, ry, a = _ell(e0)
            if what == "move":
                e = [cx + sx - x0, cy + sy - y0, rx, ry, a]
            elif what == "rot":                  # the handle above the ellipse follows the mouse around its centre
                e = [cx, cy, rx, ry, math.degrees(math.atan2(sy - cy, sx - cx)) + 90.0]
            else:
                u, v = _local(e0, sx, sy)
                e = [cx, cy, max(abs(u), 2.0), ry, a] if what.endswith("rx") else [cx, cy, rx, max(abs(v), 2.0), a]
            self.work["exclude_ellipses"][self._sel] = e
            self._veil = None
        elif self.mode == "ellipse" and self._ell:
            self._ell[1] = (sx, sy)
        canvas.update()

    def release(self, canvas):
        self.gesture = False
        edited = False
        if self.mode == "points" and self._grab is not None:
            self._grab = None
            edited = True
        elif self.mode in ("brush", "eraser") and self._stroke0 is not None:
            self._last = None
            if self.mode == "eraser" and not any(d[3] for d in self.work["exclude_brush"][:self._stroke0]):
                del self.work["exclude_brush"][self._stroke0:]       # nothing painted to erase: no step
            else:
                edited = True
            self._stroke0 = None
        elif self.mode == "ellipse" and self._edit_ell is not None:
            self._edit_ell = None
            edited = True
        elif self.mode == "ellipse" and self._ell:
            (ax, ay), (bx, by) = self._ell
            self._ell = None
            k = self._f(canvas) * canvas.scale
            if abs(bx - ax) * k > 6 and abs(by - ay) * k > 6:
                self.work["exclude_ellipses"].append([(ax + bx) / 2, (ay + by) / 2, abs(bx - ax) / 2, abs(by - ay) / 2,
                                                      0.0])
                self._sel = len(self.work["exclude_ellipses"]) - 1     # selected: its handles at once
                edited = True
        if edited:
            if self.mode in ("brush", "ellipse"):
                self._fade_t0 = time.monotonic()           # shown filled a moment, then fading
                self._fade.start()
            self._commit()
        if self.tv._dirty:                     # a refresh held back while the mouse was down
            self.tv.request_refresh()
        canvas.update()

    def hover(self, canvas, ix, iy):
        self._hover = (ix, iy)
        if self.mode in (None, "points") and "samples" in self.kinds():
            f = self._f(canvas)
            i = self._pick(canvas, ix / f, iy / f)
            ex = self._excluded()
            if i is not None and ex is not None and i < len(ex) and ex[i]:
                from PySide6.QtGui import QCursor
                from PySide6.QtWidgets import QToolTip
                QToolTip.showText(QCursor.pos(), _("Excluded area: this sample is not used"))

    def _fade_tick(self):
        if self._fade_t0 is None or time.monotonic() - self._fade_t0 > FADE_MS / 1000:
            self._fade_t0 = None
            self._fade.stop()
        self._repaint()

    def _fade_alpha(self) -> float:
        if self._fade_t0 is None:
            return 0.0
        return max(0.0, 1.0 - (time.monotonic() - self._fade_t0) / (FADE_MS / 1000))

    def _excluded(self):
        """Which samples sit in an excluded area (ellipses, then the brush's dabs in their order), as the stage reads
        them: they are drawn red and not used."""
        pts = self.work["sample_points"]
        key = (len(pts), str(pts[:3]), str(pts[-3:]), len(self.work["exclude_brush"]),
               str(self.work["exclude_ellipses"]))
        if self._excl is not None and self._excl[0] == key:
            return self._excl[1]
        if not pts:
            out = None
        else:
            P = np.asarray(pts, float).reshape(-1, 2)
            out = np.zeros(len(P), bool)
            for e in self.work["exclude_ellipses"]:
                out |= np.array([_inside(e, x, y) for x, y in P], bool)
            for x, y, r, paint in self.work["exclude_brush"]:
                near = np.hypot(P[:, 0] - x, P[:, 1] - y) <= r
                out[near] = bool(paint)
        self._excl = (key, out)
        return out

    def _dab(self, x, y, paint: bool | None = None):
        paint = self.paint if paint is None else paint
        r = self.size / 2
        pts = [(x, y)]
        if self._last is not None:                       # fill the gaps of a fast stroke (as Post's brush)
            lx, ly = self._last
            n = int(math.hypot(x - lx, y - ly) / max(r / 3, 1))
            pts = [(lx + (x - lx) * k / (n + 1), ly + (y - ly) * k / (n + 1)) for k in range(1, n + 1)] + pts
        for px, py in pts:
            self.work["exclude_brush"].append([float(px), float(py), float(r), 1.0 if paint else 0.0])
        self._last = (x, y)

    # ------------------------------------------------------------ what is drawn over the image
    def _params(self, work=None):
        """The block's settings as the draft has them (``work``: the marks being edited)."""
        from ..stages.gradient import GradientParams
        vals = {}
        for name in GradientParams.model_fields:
            v = self.s.param_value(self.p, "gradient", name)
            if v is not None:
                vals[name] = v
        d = self._draft()
        if d is not None:
            vals.update(copy.deepcopy(d.vals))
        vals.update(work if work is not None else self.work)
        if not self.value("sample_points"):    # the block's own placement (or none): the stage chooses as it will
            vals["sample_points"] = []
        return GradientParams(**vals)

    def _paint(self, painter, canvas):
        """The aids drawn over the page's image (the survey, the samples, the areas, the tag): on the page's own canvas
        only, never on another viewer, a compared side, Post or an export."""
        if self.p is None or canvas.display is None or self.gv is None or canvas is not self.gv.canvas:
            return
        f = self._f(canvas)
        dw, dh = canvas.display.size()
        if self.survey_shown:
            self._paint_survey(painter, canvas, dw, dh)
        kinds = self.kinds()
        if self.shown and kinds and not (self.before and self.cmp == "toggle"):  # Before: the stack, nothing on it
            if "areas" in kinds:
                self._paint_veil(painter, canvas, f, dw, dh)
            self._paint_samples(painter, canvas, f)
            show_areas = "areas" in kinds and (self.mode in ("brush", "ellipse", "eraser") or "samples" not in kinds)
            for j, e in enumerate(self.work["exclude_ellipses"] if show_areas else []):
                cx, cy, rx, ry, a = _ell(e)
                sel = j == self._sel and self.mode == "ellipse"
                painter.save()
                painter.translate(canvas.img_to_w(cx * f, cy * f))
                painter.rotate(a)
                painter.setBrush(Qt.NoBrush)
                painter.setPen(QPen(PICKED if sel else OUTLINE, 1.6 if sel else 1.2, Qt.SolidLine if sel else Qt.DashLine))
                painter.drawEllipse(QPointF(0, 0), rx * f * canvas.scale, ry * f * canvas.scale)
                painter.restore()
            hs = self._handles(canvas, f) if self.mode == "ellipse" else {}
            if hs:
                painter.setPen(QPen(OUTLINE, 1.2))
                painter.drawLine(QPointF(*hs["-ry"]), QPointF(*hs["rot"]))
                painter.setBrush(QColor("#15171c"))
                for name, (hx, hy) in hs.items():
                    if name == "rot":
                        painter.drawEllipse(QPointF(hx, hy), HANDLE_PX + 1, HANDLE_PX + 1)
                    else:
                        painter.drawRect(QRectF(hx - HANDLE_PX, hy - HANDLE_PX, 2 * HANDLE_PX, 2 * HANDLE_PX))
        if self._ell:
            (ax, ay), (bx, by) = self._ell
            painter.setPen(QPen(OUTLINE, 1.5, Qt.DashLine))
            painter.setBrush(QColor(*VEIL))
            painter.drawEllipse(QRectF(canvas.img_to_w(ax * f, ay * f), canvas.img_to_w(bx * f, by * f)).normalized())
        if self.mode in ("brush", "eraser") and self._hover is not None:
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(QColor(255, 255, 255, 200), 1.2))
            r = self.size / 2 * f * canvas.scale
            painter.drawEllipse(canvas.img_to_w(*self._hover), r, r)
        from .imageview import tag
        text = self.tag_text()
        if text:                               # at the top of the image, in view at every size
            tag(painter, QPointF(10, 8), text, bg=DRAFT_BG if self.has_draft() else QColor(0, 0, 0, 170))
        if self.busy and self._busy_shown:     # a thin bar running along the image's top edge while it computes
            w = canvas.width()
            seg = max(60.0, w * 0.25)
            x = (w + seg) * self._phase - seg
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(WARN))
            painter.drawRect(QRectF(max(0.0, x), 0, min(seg, w - max(0.0, x)), 3))

    def tag_text(self) -> str:
        """What the image shows, said on it: the draft's preview is never taken for the result."""
        if self.before and self.cmp == "toggle":
            c = self.carried.get(self.p.pid) if self.p is not None else None
            if c is not None and c.get("orig") is not None:
                return _("Before · the result of {name}").format(name=c["of"])
            return _("Before · the stack, its gradient not removed")
        if not self.has_draft() and self.result is None:
            stale = self.p is not None and self.s.state(self.p, "gradient") == "stale"
            return _("Previous settings") if stale else _("After · applied result")
        if self.busy and self._busy_shown:
            return _("Computing preview…") + (" · " + self.step_text if self.step_text else "")
        if self._said is not None and time.monotonic() - self._said[1] < SAID_MS / 1000:
            return self._said[0]
        if self.has_draft() and self.too_few():
            return _("Too few samples: no gradient removed · place samples or generate a grid")
        if self.has_draft():
            if self.result is None:
                return _("Applied result · the draft preview follows")
            ms = f" · {self.result_ms:.0f} ms" if self.result_ms is not None else ""
            return _("Draft preview · reduced copy") + ms
        if self.result is not None:
            return _("Preview of the settings · not computed yet: Run to here")
        return self.step_text if self.busy else ""

    def _paint_veil(self, painter, canvas, f, dw, dh):
        """The excluded areas: a thin outline while Brush, Ellipse or Eraser is on (always, for a method without
        samples); filled only a moment after they are painted, then fading: never a red block over the image."""
        brush = self.work["exclude_brush"]
        done = brush if self._stroke0 is None else brush[:self._stroke0]
        outline = self.mode in ("brush", "ellipse", "eraser") or "samples" not in self.kinds()
        fade = self._fade_alpha()
        if (self.work["exclude_ellipses"] or done) and (outline or fade > 0):
            import cv2

            from ..stages.gradient import exclusion_mask
            k = min(1.0, 1500 / max(dw, 1))
            key = (dw, dh, len(done), str(self.work["exclude_ellipses"]), id(self.work))
            if self._veil is None or self._veil[0] != key:
                work = {**self.work, "exclude_brush": done}
                m = exclusion_mask(self._params(work), int(dh * k), int(dw * k), f * k)
                filled = np.zeros((*m.shape, 4), np.uint8)
                filled[m] = VEIL
                edge = m & ~cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
                line = np.zeros((*m.shape, 4), np.uint8)
                line[edge] = (255, 255, 255, 210)
                imgs = [QImage(a.data, m.shape[1], m.shape[0], 4 * m.shape[1], QImage.Format_RGBA8888).copy()
                        for a in (filled, line)]
                self._veil = (key, imgs)
            rect = QRectF(canvas.img_to_w(0, 0), canvas.img_to_w(dw, dh))
            if fade > 0:
                painter.setOpacity(fade)
                painter.drawImage(rect, self._veil[1][0])
                painter.setOpacity(1.0)
            if outline:
                painter.drawImage(rect, self._veil[1][1])
        if self._stroke0 is not None:          # the stroke in progress, drawn as it goes (the veil follows on release)
            painter.setPen(Qt.NoPen)
            for x, y, r, paint in brush[self._stroke0:]:
                painter.setBrush(QColor(*VEIL) if paint else QColor(0, 0, 0, 90))
                painter.drawEllipse(canvas.img_to_w(x * f, y * f), r * f * canvas.scale, r * f * canvas.scale)

    def _paint_survey(self, painter, canvas, dw, dh):
        """The survey's emission in amber where a sample should not sit (above ``EMISSION_FREE``), stronger where it is
        brighter (its square root, as a stretched image); its sky left clear."""
        from ..stages.gradient import EMISSION_FREE
        sv = self.survey
        if sv is None or "level" not in sv or sv["pid"] != self.p.pid:
            return
        if self._survey_img is None or self._survey_img[0] != id(sv):
            lv = np.nan_to_num(sv["level"])
            a = np.where(lv > EMISSION_FREE, 0.35 + 0.65 * np.sqrt(np.clip(lv, 0, 1)), 0.0)
            rgba = np.zeros((*a.shape, 4), np.uint8)
            rgba[..., :3] = SURVEY_TINT
            rgba[..., 3] = (a * SURVEY_ALPHA).astype(np.uint8)
            img = QImage(rgba.data, a.shape[1], a.shape[0], 4 * a.shape[1], QImage.Format_RGBA8888).copy()
            self._survey_img = (id(sv), img)
        painter.drawImage(QRectF(canvas.img_to_w(0, 0), canvas.img_to_w(dw, dh)), self._survey_img[1])

    def _paint_samples(self, painter, canvas, f):
        """The placed samples, coloured as the last computation (run or preview) used them, white while not computed,
        the one selected in yellow; none placed: the ones the block chose at its last run."""
        if "samples" not in self.kinds():      # a method that reads no sample: none drawn (the draft keeps them)
            return
        s, pts = self.samples, self.work["sample_points"]
        if pts:
            P = np.asarray(pts, float).reshape(-1, 2)
            same = (s is not None and s.get("manual") and len(s["x"]) == len(P)
                    and np.allclose(np.c_[s["x"], s["y"]], P, atol=0.6))
            used = list(s["used"]) if same else [None] * len(P)
            W, H = self._stack_size()
            box = s["box"] if s is not None else max(4, int(min(W, H) / self._density() / 2))
            ex = self._excluded()
            if ex is not None:                 # in an excluded area: not used, whatever the last preview said
                used = [False if e else u for u, e in zip(used, ex)]
            rows = zip(P[:, 0], P[:, 1], used)
        else:
            return
        half = max(3.0, box / 2 * f * canvas.scale)
        for i, (x, y, u) in enumerate(rows):
            c = canvas.img_to_w(x * f, y * f)
            picked = pts and i == self._selpt and self.mode == "points"
            col = PICKED if picked else OUTLINE if u is None else USED if u else UNUSED
            painter.setPen(QPen(col, 2.4 if picked or u is False else 1.6))
            painter.setBrush(QColor(255, 77, 85, 70) if u is False else Qt.NoBrush)
            painter.drawRect(QRectF(c.x() - half, c.y() - half, 2 * half, 2 * half))

    # ------------------------------------------------------------ the preview (one thread at a time)
    def _schedule(self):
        """The draft's preview, once the draft has not changed for a moment (shorter when this method is fast): steps of
        a setting in quick succession make one preview, of the last one."""
        if self.p is None or self.no_input():
            return
        fast = self._ms_of.get(self._method(), 1e9) <= FAST_MS
        self._timer.setInterval(FAST_DEBOUNCE_MS if fast else DEBOUNCE_MS)
        self._timer.start()

    def preview(self, force: bool = False):
        if self.p is None or not (self.has_draft() or force):
            return
        if self.too_few():                     # nothing removed: the stack as it is, said on it
            self.result, self.result_ms = None, None
            if self.small is None or self.small["pid"] != self.p.pid:
                self._run("load")
            self._present()
            return
        self._run("preview")

    def _run(self, what: str, fetch: bool = False):
        """One computation at a time: a request made while one runs waits, replaced by the next (only the last
        counts); the answer of an older edit is dropped. ``fetch``: the survey may be downloaded (its toggle only)."""
        if self.p is None or self.no_input():          # no stack: nothing to read, nothing to preview
            return
        self.gen += 1
        if self.busy:
            if what == "survey":
                self._survey_wanted = bool(fetch) or bool(self._survey_wanted)
            elif what != "preview":             # a preview asked while one runs: the latest is made on arrival
                self._pending = what if self._pending != "grid" else "grid"
            else:
                self._preview_due = True
            return
        gen, params, pid, n = self.gen, self._params(), self.p.pid, self._density()
        sig = params.model_dump_json()
        if what == "preview":
            self.computed += 1
            self._busy_timer.start()
        small = self.small if self.small and self.small["pid"] == pid else None
        sv = self.survey if (self.avoid or what == "points") and self.survey and self.survey["pid"] == pid else None
        level = sv.get("level") if sv else None
        # the block's own samples avoid the survey's emission: the preview too (the survey read from the disk with it)
        read = (what == "preview" and self.avoid and params.method != "automatic" and not params.sample_points
                and (self.survey is None or self.survey["pid"] != pid))
        akey = self._applied_key()
        want_applied = what in ("preview", "load") and (self.applied is None or self.applied["key"] != akey)
        project = self.s.p
        ops = self._ops if what == "points" else None
        self.busy = True
        self.step_text = ""
        self._sync()
        if self.view is not None:
            self.view.canvas.update()
        step = self._step
        ed = self

        def still(c):
            """The stage's progress, between its colours: a preview the draft has left is stopped there."""
            if ed._wanted_sig() != sig:
                raise _Abandoned()
            step.emit(gen, _("automatic model: colour {n} of 3").format(n=c + 1))

        def work():
            try:
                sm = small or _reduced_stack(project, pid)
                out = {"small": sm, "what": what, "pid": pid}
                if want_applied:
                    out["applied"] = {"key": akey, "arr": _applied_small(project, pid, sm)}
                if what == "points":
                    out.update(_points_job(sm, params, ops, level, project, pid))
                elif what == "grid":
                    from ..stages.gradient import exclusion_mask, grid_points
                    ex = exclusion_mask(params, *sm["valid"].shape, sm["f"])
                    ok = sm["valid"] & ~ex if ex is not None else sm["valid"]
                    pts, kept, box = grid_points(sm["arr"], ok, n, params.object_clip)
                    if level is not None:
                        from ..stages.gradient import away_from_emission
                        kept, out["avoided"] = away_from_emission(pts, kept, level, box)
                    out["grid"] = [[float(x) / sm["f"], float(y) / sm["f"]] for x, y in pts[kept]]
                elif what == "preview":
                    if read:
                        try:
                            out["survey"] = {"pid": pid, **_survey_small(project, pid, sm)}
                        except Exception as e:  # noqa: BLE001 - no survey: the preview without it
                            out["survey"] = {"pid": pid, "error": _survey_why(e)}
                    lv = level if level is not None else (out.get("survey") or {}).get("level")
                    t0 = time.thread_time()
                    try:
                        out.update(_preview(sm, params, lv, progress=still))
                    except _Abandoned:
                        out["abandoned"] = True
                    out["ms"] = 1000 * (time.thread_time() - t0)
                    out["sig"] = sig
                    out["method"] = params.method
                elif what == "survey":
                    try:
                        out["survey"] = {"pid": pid, **_survey_small(project, pid, sm, fetch=fetch)}
                    except Exception as e:  # noqa: BLE001 - no survey for this field: said, the editor goes on
                        out["survey"] = {"pid": pid, "error": _survey_why(e)}
                self._done.emit(gen, out)
            except Exception as e:  # noqa: BLE001 - said in the bar, the editor goes on
                self._done.emit(gen, {"error": str(e), "what": what, "pid": pid})
        QThreadPool.globalInstance().start(_Job(work))

    def _stepped(self, gen: int, text: str):
        if gen == self.gen:
            self.step_text = text
            self._sync()
            if self.view is not None:
                try:
                    self.view.canvas.update()
                except RuntimeError:
                    self.view = None

    def _points_arrived(self, out: dict):
        ops, _pts, default, pid = out["ops"]
        if "survey" in out and out["survey"] is not None:
            self.survey, self._survey_img = out["survey"], None
        if pid != (self.p.pid if self.p is not None else None):
            return
        if default:                            # the block's own placement: drawn, not a change of the draft
            self.defaults[pid] = out["points"]
            self.work = self._fields()
            self._veil = None
            self._sync()
            self._repaint()
            return
        if out.get("need_survey"):
            Notice(self.tv.window(), _("Avoid nebulosity needs the H-alpha survey of this field, not on this computer."),
                   [(_("Fetch"), self._fetch_survey), (_("Cancel"), lambda: None)])
            return
        removed = sum(out["removed"].values())
        self.set_draft({"sample_points": out["points"]})
        if "nebula" in ops and out.get("no_emission"):
            self.say(_("The H-alpha survey shows no clear nebulosity here: no sample removed"))
        elif "nebula" in ops:
            self.say(_("{n} samples on nebulosity removed").format(n=removed))
        else:
            self.say(_("{n} samples on objects removed").format(n=removed))

    def _wanted_sig(self) -> str | None:
        """The settings the preview should show now (the draft's), as the job compares them (any thread)."""
        try:
            return self._params().model_dump_json() if self.p is not None else None
        except Exception:  # noqa: BLE001 - settings being changed: the job goes on, its result is checked on arrival
            return None

    def _show_busy(self):
        if self.busy:
            self._busy_shown = True
            self._anim.start()
            self._repaint()

    def _tick(self):
        self._phase = (self._phase + 0.025) % 1.0
        self._repaint()

    def _repaint(self):
        if self.view is not None:
            try:
                self.view.canvas.update()
            except RuntimeError:
                self.view = None

    def _arrived(self, gen: int, out: dict):
        self.busy = False
        self.step_text = ""
        # for the workflow shown, asked since it is shown: a preview of another one (the same settings) is never shown
        mine = self.p is not None and out.get("pid") == self.p.pid and gen > self._shown_gen
        if out.get("what") == "preview" and "ms" in out and not out.get("abandoned"):
            self._ms_of[out.get("method", "")] = out["ms"]
        if out.get("small") is not None:
            self.small = out["small"]
        if "applied" in out:
            self.applied = out["applied"]
        if "survey" in out:                    # asked for itself, or read with a preview
            self.survey, self._survey_img = out["survey"], None
            if "error" in out["survey"] and out.get("what") == "survey":
                self.tv.toasted.emit(_("H-alpha survey: {why}").format(why=out["survey"]["error"]))
            if self._grid_after_survey:        # Generate asked for it: the grid now, with or without the survey
                self._grid_after_survey = False
                self._pending = "grid"
        if "error" in out:
            self.tv.toasted.emit(_("Gradient preview: {why}").format(why=_(out["error"])))
        elif "points" in out:
            self._points_arrived(out)
        elif "grid" in out:
            self.work["sample_points"] = out["grid"] + ([self._extra] if self._extra else [])
            self._extra = None
            if out.get("avoided") is False:
                self.tv.toasted.emit(_("The H-alpha survey shows no clear nebulosity here: no sample left out"))
            self._commit()
        elif "result" in out and mine and out.get("sig") == self._wanted_sig() and \
                (self.has_draft() or not self.done):  # the draft as it is now: a preview it has left is never shown
            self.painted += 1
            self.result, self.result_ms = out["result"], out.get("ms")
            self.result_key = ("gradient-preview", self.p.pid, gen)
            self.samples = out["samples"]
            self._present()
        if self._survey_wanted is not None:
            fetch, self._survey_wanted = self._survey_wanted, None
            if self.survey is None or self.survey["pid"] != self.p.pid or (fetch and "error" in self.survey):
                self._run("survey", fetch=fetch)
                return
        if self._pending is not None:
            what, self._pending = self._pending, None
            self._run(what)
        elif (out.get("what") == "preview" and (out.get("sig") != self._wanted_sig() or not mine) or
              self._preview_due) and \
                self.has_draft() and not self._timer.isActive():
            self._preview_due = False
            self.preview()                     # the draft changed meanwhile: one preview, of the latest
        if out.get("what") == "load" and self.has_draft() and self.too_few():
            self._present()                    # the stack, nothing removed, now that it is read
        if not self.busy:
            self._busy_timer.stop()
            self._anim.stop()
            self._busy_shown = False
        self._sync()
        if self.view is not None:
            try:
                self.view.canvas.update()
            except RuntimeError:
                self.view = None


class _Abandoned(Exception):
    """A preview the draft has left, stopped at the stage's next progress call."""


class SettingsPanel(QFrame):
    """The settings at the right of the image: the method's settings first, then the samples, the draft, the H-alpha
    survey, the compare modes, the measurements and the stretch, in titled groups. A side panel of the window's kind
    (``layout.Dock``): kept open, automatic or kept folded, chosen with its header's button or the strip's pin.
    ``rows``: the editor's settings (built again when they change); ``fixed``: the viewer's stretch column and
    alignment line, kept."""

    def __init__(self, ed):
        super().__init__()
        self.ed = ed
        self.setObjectName("panel")
        self.setFixedWidth(PANEL_W)
        col = QVBoxLayout(self)
        col.setContentsMargins(10, 6, 6, 6)
        col.setSpacing(4)
        head = QHBoxLayout()
        head.addWidget(label(_("Settings"), "h2"))
        head.addStretch(1)
        self.mode_b = icon_only(QPushButton(), "fold_right", _("Kept open, automatic or kept folded"), 16)
        self.mode_b.clicked.connect(lambda: ed.dock.menu(self.mode_b))
        head.addWidget(self.mode_b)
        col.addLayout(head)
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        sc.setFrameShape(QFrame.NoFrame)
        sc.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        inner = QWidget()
        iv = QVBoxLayout(inner)
        iv.setContentsMargins(0, 0, 4, 0)
        iv.setSpacing(4)
        self.rows = QVBoxLayout()
        self.rows.setSpacing(4)
        iv.addLayout(self.rows)
        self.fixed = QVBoxLayout()
        iv.addLayout(self.fixed)
        iv.addStretch(1)
        sc.setWidget(inner)
        col.addWidget(sc, 1)


class Notice(QFrame):
    """A notice with its actions over the foot of the window (drawn over the pages, out of every layout: nothing
    under it moves when it comes or goes); an action hides it."""

    def __init__(self, win, text: str, actions):
        super().__init__(win)
        self.setObjectName("gradnotice")
        self.setStyleSheet("QFrame#gradnotice { background: #3a2f1c; border: 1px solid #5a4a26; border-radius: 6px; }"
                           " QLabel { background: transparent; }")
        h = QHBoxLayout(self)
        h.setContentsMargins(12, 6, 8, 6)
        h.setSpacing(6)
        h.addWidget(label(text, "", wrap=True), 1)
        self.buttons = []
        for words, fn in actions:
            b = QPushButton(words)
            b.clicked.connect(lambda _c=False, f=fn: (self.hide(), f()))
            h.addWidget(b)
            self.buttons.append(b)
        self.place()
        self.show()
        self.raise_()

    def place(self):
        win = self.parentWidget()
        w = max(320, min(900, win.width() - 32))
        h = max(self.heightForWidth(w), self.sizeHint().height()) if self.hasHeightForWidth() else \
            self.sizeHint().height()
        sb = win.statusBar().height() if hasattr(win, "statusBar") else 0
        self.setGeometry((win.width() - w) // 2, win.height() - sb - h - 10, w, h)


def _points_job(sm: dict, params, req, level, project, pid) -> dict:
    """The actions on the samples, on the reduced copy, with the stage's own rules: ``generate`` (the regular grid's
    places), ``objects`` (``sky_blocks`` with Object rejection on the points' values), ``nebula`` (``away_from_emission``
    on the survey's level; ``nebula_disk``: only when the survey is on this computer, never asked)."""
    from ..core.imgutil import poly_terms
    from ..stages.gradient import LUM, away_from_emission, exclusion_mask, point_values, sky_blocks
    ops, pts, default, _pid = req
    h, w = sm["valid"].shape
    f = sm["f"]
    n = params.samples
    if "generate" in ops:
        step = min(sm["H"], sm["W"]) / n
        pts = [[float(x), float(y)] for y in np.arange(step / 2, sm["H"], step) for x in np.arange(step / 2, sm["W"], step)]
    P = np.asarray(pts, float).reshape(-1, 2)
    box = max(4, int(min(h, w) / n / 2))
    out = {"removed": {}, "ops": req}
    if "objects" in ops and len(P):
        ex = exclusion_mask(params, h, w, f)
        ok = sm["valid"] & ~ex if ex is not None else sm["valid"]
        ps = P * f
        vals, use = point_values(sm["arr"], ok, ps, box)
        L = LUM @ np.nan_to_num(vals)
        A = poly_terms(ps[:, 0], ps[:, 1], w, h, 2)
        kept = sky_blocks(L, use, A, params.object_clip, (1, len(ps))) if use.sum() >= 3 * A.shape[1] else use
        out["removed"]["objects"] = int((~kept).sum())
        P = P[kept]
    if ("nebula" in ops or "nebula_disk" in ops) and len(P):
        if level is None:
            try:
                sv = {"pid": pid, **_survey_small(project, pid, sm, fetch=False)}
                level = sv["level"]
                out["survey"] = sv
            except Exception:  # noqa: BLE001 - not on this computer: asked of the user (an explicit action only)
                if "nebula" in ops:
                    out["need_survey"] = True
        if level is not None:
            kept, used = away_from_emission(P * f, np.ones(len(P), bool), level, box)
            out["no_emission"] = not used
            out["removed"]["nebula"] = int((~kept).sum()) if used else 0
            P = P[kept]
    out["points"] = [[round(float(x), 1), round(float(y), 1)] for x, y in P]
    return out


def _reduced_stack(project, pid: int) -> dict:
    """The stack of the workflow reduced to about PREVIEW_PIXELS (block means), its covered pixels."""
    from ..api import stage_outputs
    # an Outdated stack: its earlier result, still kept, as the page shows it ("Previous settings"; the night test pass:
    # the page showed the earlier stack and the preview said "The stack is not computed")
    out = stage_outputs(project, pid, "stack") or stage_outputs(project, pid, "stack", which="previous")
    if out is None:
        raise ValueError(_("The stack is not computed"))
    arr = np.load(f"{out.dir}/drz_full.npy", mmap_mode="r")
    wt = np.asarray(np.load(f"{out.dir}/drz_weight.npy", mmap_mode="r")[1], np.float32)
    H, W = arr.shape[1:]
    k = max(1, int(math.ceil(math.sqrt(H * W / PREVIEW_PIXELS))))
    h, w = H // k, W // k
    small = np.stack([np.asarray(arr[c, :h * k, :w * k], np.float32).reshape(h, k, w, k).mean(axis=(1, 3))
                      for c in range(3)])
    pos = wt[wt > 0]
    valid = (wt > 0.6 * float(np.median(pos))) if pos.size else wt > 0
    valid = valid[:h * k, :w * k].reshape(h, k, w, k).mean(axis=(1, 3)) > 0.99
    return {"pid": pid, "arr": small, "valid": valid, "f": 1.0 / k, "W": W, "H": H}


def _applied_small(project, pid: int, sm: dict) -> np.ndarray | None:
    """The applied result on the reduced copy: the stack less the sky variation the block's last run kept (its
    sky.npy and gradient.json, read back as the stage reads them, at the copy's scale); None without a result."""
    import json

    import cv2

    from ..api import stage_outputs
    from ..stages.gradient import INFO_FILE, SKY_FILE, enlarge
    out = stage_outputs(project, pid, "gradient")
    if out is None or not os.path.isfile(f"{out.dir}/{SKY_FILE}"):
        return None
    grid = np.load(f"{out.dir}/{SKY_FILE}")
    info = json.loads(open(f"{out.dir}/{INFO_FILE}", encoding="utf-8").read())
    if (info.get("width"), info.get("height")) != (sm["W"], sm["H"]):
        return None
    h, w = sm["arr"].shape[1:]
    k = 1.0 / sm["f"]
    if info.get("grid") == "corners":
        # corners aligned on the whole stack: the copy covers its first h * k x w * k pixels
        full_h, full_w = sm["H"] / k, sm["W"] / k
        var = np.stack([enlarge(g, int(round(full_h)), int(round(full_w)))[:h, :w] for g in grid])
    else:
        fac = float(info.get("factor", 8)) / k
        gh, gw = grid.shape[1:]
        size = (max(w, int(round(gw * fac))), max(h, int(round(gh * fac))))
        var = np.stack([cv2.resize(g.astype(np.float32), size, interpolation=cv2.INTER_CUBIC)[:h, :w] for g in grid])
    return (sm["arr"] - var).astype(np.float32)


def _survey_small(project, pid: int, sm: dict, fetch: bool = False) -> dict:
    """The H-alpha survey of the workflow's solved field on the reduced copy's grid, as its emission level
    (surveys.emission_level: 0 its sky, 1 its bright emission), and its name. From this computer only, unless
    ``fetch`` (the survey's toggle, an explicit action of the user)."""
    import json

    from astropy.io import fits
    from astropy.wcs import WCS
    from astropy.wcs.utils import proj_plane_pixel_scales

    from ..api import stage_outputs
    from ..stages.surveys import SURVEYS, emission_level, survey_on_grid
    done = stage_outputs(project, pid, "surveys")          # the chain's survey, when it ran
    js = json.loads(open(f"{done.dir}/survey.json", encoding="utf-8").read()) if done is not None else None
    if js is not None and js.get("survey", "none") != "none":
        img, name = np.load(f"{done.dir}/ref_ha.npy", mmap_mode="r"), js["survey"]
    else:
        out = stage_outputs(project, pid, "astrometry")
        if out is None:
            raise ValueError(_("the field is not solved yet (Astrometry)"))
        if not os.path.isfile(f"{out.dir}/wcs.fits"):        # Astrometry continued without a solution (its fallback)
            raise ValueError(_("the field was not identified: no plate solution"))
        wd = WCS(fits.getheader(f"{out.dir}/wcs.fits"))
        scale = float(np.mean(proj_plane_pixel_scales(wd.celestial))) * 3600.0
        img, name = survey_on_grid(wd, sm["W"], sm["H"], scale, log=lambda *a: None, fetch=fetch)
    k = int(round(1 / sm["f"]))
    h, w = sm["arr"].shape[1:]
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)       # cells outside the survey: NaN
            small = np.nanmean(img[:h * k, :w * k].reshape(h, k, w, k), axis=(1, 3))
    return {"level": emission_level(small), "title": SURVEYS[name][1]}


def _survey_why(e: Exception) -> str:
    from ..core import online
    from ..core.stage import StageError
    from ..stages.surveys import SurveyMissing
    if isinstance(e, SurveyMissing):
        return _("not on this computer (the H-alpha survey toggle fetches it)")
    if isinstance(e, StageError):
        return _("no H-alpha survey covers this field")
    if isinstance(e, ValueError):
        return str(e)
    if isinstance(e, online.ServiceDown):
        return _("the survey's servers do not answer: try again later")
    return _("not available ({why})").format(why=str(e)[:120])


def preview_cell(params, shape: tuple[int, int]) -> int | None:
    """The cell of the grid the preview's sky is evaluated on: coarser than the run's for the RBF on the samples the
    block chooses (evaluating the splines is most of the preview's time there; the sky is smooth at that scale), the
    run's otherwise (None). Apply and the run keep the stage's own cell."""
    import inspect

    from ..stages.gradient import sky_model_of
    if params.sample_points or params.method != "rbf" or "cell" not in inspect.signature(sky_model_of).parameters:
        return None
    return max(8, int(min(shape) / PREVIEW_CELL_SHARE))


def _preview(sm: dict, params, emission=None, progress=None) -> dict:
    """The stack with the model's gradient removed (its level kept), on the reduced copy, and its samples (``emission``:
    the survey's level on the copy, which the block's own samples avoid, as the run's): the stage's own functions, its
    sky evaluated on ``preview_cell``."""
    from ..stages.gradient import read_grid, sky_model_of
    cell = preview_cell(params, sm["arr"].shape[1:])
    kw = {"cell": cell} if cell is not None else {}
    m = sky_model_of(sm["arr"], sm["valid"], params, f=sm["f"], progress=progress, emission=emission, **kw)
    h, w = sm["arr"].shape[1:]
    g = read_grid(m["grid"], m["info"], h, w)
    lvl = np.array([np.median(g[c][sm["valid"]]) for c in range(3)], np.float32)[:, None, None]
    return {"result": (sm["arr"] - g + lvl).astype(np.float32), "samples": m["samples"]}
