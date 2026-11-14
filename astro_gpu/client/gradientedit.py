"""The Gradient block tuned in Inspect as a draft (``GradientEditor``): try, look, then Apply or Try variant.

Every adjustment of the view (the method and its settings, the samples placed, moved and removed on the image, the
areas left out with the brush or ellipses, the generated grid, Away from emission) changes a DRAFT of the block's
settings, kept per workflow while the project is open: the project does not change, the graph stays up to date, Run
is unchanged. The draft is seen at once on a reduced copy of the stack (about a million pixels, whatever the sensor),
computed with the stage's own functions (``sky_model_of``, ``read_grid``) in a thread of its own, the last edit only,
the previous image kept on screen until the next one is ready; it is compared with the applied result on the same
grid (read from the block's own sky model, never computed again) or with the stack, Split or Side by side.

When it looks right: **Apply** (the draft becomes the block's settings, one step of the window's Undo; what it
computes again and how long is said first, on the button and on the bar), **Try variant** (the draft goes on a variant
of the workflow: the computed chain stays as it is) or **Reset** (back to the applied settings). Leaving the block with
a draft asks the same in the window's notice strip; Later keeps it, and coming back shows it again.

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
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .i18n import _
from .icons import icon_only
from .widgets import WARN, ElidedLabel, label

#: the preview's copy of the stack holds about this many pixels
PREVIEW_PIXELS = 1_000_000
#: the preview starts this long after the last edit (the automatic model, slower, waits a little more)
DEBOUNCE_MS, AUTO_DEBOUNCE_MS = 120, 600
#: the cost of Apply is asked of the model this long after the last edit
COST_MS = 500
#: below this window width, the stretch column of this view starts folded (the image keeps the room)
SMALL_W = 1500
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
METHOD_LINE = {"polynomial": "Smooth surface under the sky",
               "rbf": "Surface through the samples",
               "automatic": "No samples, every sky pixel"}
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
TOOL_KEYS = (Qt.Key_Escape, Qt.Key_Delete, Qt.Key_Backspace, Qt.Key_M, Qt.Key_R, Qt.Key_X, Qt.Key_BracketLeft,
             Qt.Key_BracketRight)


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
    _step = Signal(int, str)                   # (request number, how far the automatic preview is)

    def __init__(self, tune):
        super().__init__(tune)
        self.tv, self.s = tune, tune.s
        self.view = None
        self.p = None
        self.mode: str | None = None           # None (looking), "points", "brush", "ellipse"
        self.paint = True
        self.size = 60.0                       # brush diameter, stack pixels
        self.shown = True                      # what is drawn over the image (M, Marks)
        self.drafts: dict[int, dict] = {}      # workflow -> {"vals": {setting: value}, "undo", "redo", "told"}
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
        self.hist_shown: bool | None = None    # the stretch column of this view (None: by the window's width)
        self._hist_saved = None                # Inspect's own choice, given back on leaving
        self._cost = None                      # (workflow, settings, what Apply computes again)
        self._applied_at = None
        self._banner_for = None
        self._cmp_name = None                  # the image the preview is compared with, by name
        self._in_show = False
        self._dims = None                      # (pid, W, H) of the stack
        self._pending = None                   # the request waiting for the one running
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
        return bool(d and d["vals"])

    def value(self, name: str):
        """A setting as the draft has it (the applied one when the draft does not change it)."""
        d = self._draft()
        if d is not None and name in d["vals"]:
            return copy.deepcopy(d["vals"][name])
        return self._applied_value(name)

    def drafting(self, p) -> bool:
        """Whether the block of ``p`` is tuned as a draft here: it has a result on screen (computed, or Outdated with
        its previous one). Never computed, its settings are the panel's as any block's."""
        st = self.s.state(p, "gradient")
        return st == "done" or (st == "stale" and self.tv._has_previous(p, "fond"))

    def set_draft(self, updates: dict, undo: bool = True) -> bool:
        """Settings of the draft: one step of its Undo; a setting back to its applied value leaves the draft."""
        if self.p is None:
            return False
        d = self.drafts.setdefault(self.p.pid, {"vals": {}, "undo": [], "redo": [], "told": False})
        before = copy.deepcopy(d["vals"])
        for k, v in updates.items():
            v = _norm(k, v)
            if v == self._applied_value(k):
                d["vals"].pop(k, None)
            else:
                d["vals"][k] = copy.deepcopy(v)
        if d["vals"] == before:
            self.work = {f: self.value(f) for f in FIELDS}
            self._sync()
            return False
        if undo:
            d["undo"].append(before)
            d["redo"].clear()
        d["told"] = False
        self._draft_changed(set(before) | set(d["vals"]) | set(updates))
        return True

    def _draft_changed(self, names: set):
        self.work = {f: self.value(f) for f in FIELDS}
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
        if names - set(FIELDS):                # the method or a setting: the panel's rows and the bar show it
            self.tv.request_refresh()
        else:
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
        if self.view is not None:
            self.tv._center_sig = None         # the viewer built again on the block's picture
            self.tv.request_refresh()

    def undo(self):
        """The draft's last step taken back (the window's Undo is the project's)."""
        d = self._draft()
        if not d or not d["undo"]:
            return
        names = set(d["vals"]) | set(d["undo"][-1])
        d["redo"].append(copy.deepcopy(d["vals"]))
        d["vals"] = d["undo"].pop()
        self._draft_changed(names)

    def redo(self):
        d = self._draft()
        if not d or not d["redo"]:
            return
        names = set(d["vals"]) | set(d["redo"][-1])
        d["undo"].append(copy.deepcopy(d["vals"]))
        d["vals"] = d["redo"].pop()
        self._draft_changed(names)

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
        if not d or not d["vals"]:
            return None
        from .model import GRADIENT_VIEW
        vals = dict(d["vals"])
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
        if d["vals"].get("sample_points"):
            bits.append(_("{n} samples").format(n=len(d["vals"]["sample_points"])))
        if any(k in d["vals"] for k in ("exclude_ellipses", "exclude_brush")):
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
        self._cost = (self.p.pid, repr(sorted(self._draft()["vals"].items())), text)
        self._sync()

    def cost_text(self) -> str:
        d = self._draft()
        if self._cost is None or d is None or self._cost[0] != self.p.pid or \
                self._cost[1] != repr(sorted(d["vals"].items())):
            return ""
        return self._cost[2]

    def _refused(self, e):
        self.tv.toasted.emit(_(str(e)))

    def apply(self, pid: int | None = None):
        """The draft becomes the block's settings, in the workflows the panel's changes go to: one step of the
        window's Undo. The preview stays on screen until Run to here computes it."""
        q = self._view_of(pid)
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
        self._hide_banner(q.pid)
        self.tv.toasted.emit(_("Gradient: draft applied · Run to here computes it · Ctrl+Z"))
        if self.p is not None and q.pid == self.p.pid:
            self.work = {f: self.value(f) for f in FIELDS}
            self.done = self.s.state(self.p, "gradient") == "done"
        self.tv.request_refresh()

    def try_variant(self, pid: int | None = None):
        """The draft on a variant of the workflow (the chain computed stays as it is); the variant is shown."""
        q = self._view_of(pid)
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
        self.drafts.pop(q.pid, None)
        self._hide_banner(q.pid)
        if self.p is not None and q.pid == self.p.pid:
            self.result, self.result_ms = None, None
        self.tv._variant_made(v.pid, q.pid)

    def reset(self, pid: int | None = None):
        """Back to the applied settings: the draft goes."""
        q = self._view_of(pid)
        if q is None or q.pid not in self.drafts:
            return
        self.drafts.pop(q.pid, None)
        self._hide_banner(q.pid)
        if self.p is not None and q.pid == self.p.pid:
            self._draft_changed({"method"})

    # ------------------------------------------------------------ leaving the block with a draft
    def eventFilter(self, obj, e):                                   # noqa: N802 - Qt's name
        if e.type() == QEvent.Hide and obj is self.view:
            QTimer.singleShot(0, self._check_left)
        return False

    def _here(self, pid) -> bool:
        """The Gradient block of this workflow is the one Inspect shows."""
        try:
            s = self.s
            return (self.tv.isVisible() and s.path is not None and s.path.pid == pid
                    and str(s.sel_step).partition("~")[0] == "fond")
        except RuntimeError:
            return False

    def _check_left(self):
        """The block left with a draft: said once in the window's notice strip, with Apply, Try variant, Reset and
        Later (the draft kept); never dropped nor applied silently."""
        for pid, d in list(self.drafts.items()):
            if not d["vals"] or d["told"] or self._here(pid):
                continue
            q = self.s.view(pid)
            if q is None:
                continue
            d["told"] = True
            text = _("Gradient · {name}: a draft is not applied. It is kept while the project is open.").format(
                name=q.name)
            banner = getattr(self.tv.window(), "banner", None)
            if banner is None:
                self.tv.toasted.emit(text)
                return
            self._banner_for = pid
            banner.ask(text, (_("Apply"), lambda pp=pid: self.apply(pp)),
                       (_("Try variant"), lambda pp=pid: self.try_variant(pp)),
                       (_("Reset"), lambda pp=pid: self.reset(pp)), (_("Later"), lambda: None))
            return

    def _hide_banner(self, pid):
        if self._banner_for == pid:
            self._banner_for = None
            banner = getattr(self.tv.window(), "banner", None)
            if banner is not None:
                banner.hide()

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
        if self.p is None or p.pid != self.p.pid:
            if self.p is not None:
                self.set_mode(None)
            self.p = p
            self.result, self.result_ms, self.samples = None, None, None
            self._sel, self._selpt, self._veil = None, None, None
            if self.small is not None and self.small["pid"] != p.pid:
                self.small = None
            if self.survey is not None and self.survey["pid"] != p.pid:
                self.survey, self._survey_img = None, None
                if self.survey_shown:
                    self._run("survey", fetch=True)     # shown: the survey of the workflow now shown
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
                for k in list(d["vals"]):
                    if d["vals"][k] == self._applied_value(k):
                        d["vals"].pop(k)
            if not first and self.has_draft():
                self._schedule()
        self.work = {f: self.value(f) for f in FIELDS}
        self.done = self.s.state(p, "gradient") == "done"
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

    def attach(self, view, p):
        """The Gradient block's picture is shown in ``view`` (its tab built again: the tool and the draft stay)."""
        self._load(p)
        self.view = view
        view.add_overlay(self._paint)
        view.canvas.tool = self if self.mode else None
        view.set_tool_widgets([])
        if hasattr(view, "set_tool_row"):
            view.set_tool_row(self._bar())
        view.installEventFilter(self)
        d = self._draft()
        if d is not None:
            d["told"] = False
        self._hide_banner(p.pid)
        if self._hist_saved is None:
            self._hist_saved = self.tv.show_hist
        want = self.hist_shown
        if want is None:
            want = self.tv.window().width() >= SMALL_W
        try:
            if view.panel_box.isVisibleTo(view) != bool(want):
                view.show_histogram(bool(want))
            view.hist_b.toggled.connect(self._hist_toggled)
            view.compareChanged.connect(self._compare_changed)
        except (RuntimeError, AttributeError):
            pass
        self._hint()
        self._sync()
        if self.result is not None:
            self._show(self.result)
        elif self.has_draft():
            self._schedule()

    def detach(self):
        # a gesture cut by a rebuild (a refresh asked before the mouse went down): its release will not come
        self.gesture = False
        self._grab = self._stroke0 = self._ell = self._edit_ell = self._last = None
        if self.view is not None:
            v = self.view
            try:
                v.hist_b.toggled.disconnect(self._hist_toggled)
                v.compareChanged.disconnect(self._compare_changed)
            except (RuntimeError, AttributeError, TypeError):
                pass
            try:
                v.removeEventFilter(self)
                v.canvas.tool = None
                v.tool_hint.hide()
                v.set_tool_widgets([])
                if hasattr(v, "set_tool_row"):
                    v.set_tool_row(None)
            except RuntimeError:
                pass
        if self._hist_saved is not None:       # Inspect's other views keep their own choice of the column
            self.tv.show_hist = self._hist_saved
            self._hist_saved = None
        self.view = None
        self._ui = {}
        QTimer.singleShot(0, self._check_left)

    def _hist_toggled(self, on):
        self.hist_shown = bool(on)

    def _compare_changed(self, _key):
        if self._in_show or self.view is None:
            return
        try:
            i = self.view._cmp_i
            self._cmp_name = self.view._targets[i].name if i is not None else ""
        except (RuntimeError, AttributeError, IndexError):
            pass

    def _view_toggles(self) -> list:
        """What the editor draws over the image, as toggles on the tools' row (as Post's Mask view): the H-alpha
        survey (R) and the marks (M)."""
        out = []
        for key, text, tip, on, slot in (
                ("survey", "H-alpha survey", "Show in amber where the professional H-alpha survey of the field sees "
                 "nebulosity (R): no sample should sit there. Downloaded once; the field must be solved",
                 self.survey_shown, self._set_survey_shown),
                ("marks", "Marks", "Show the samples and the excluded areas over the image (M)", self.shown,
                 self._set_shown)):
            b = QPushButton(_(text))
            b.setObjectName("toggle")
            b.setCheckable(True)
            b.setChecked(on)
            b.setToolTip(_(tip))
            b.toggled.connect(slot)
            self._ui[key] = b
            out.append(b)
        return out

    # ------------------------------------------------------------ the bar at the image
    def _bar(self) -> QWidget:
        """Two rows above the image: the draft (method, its setting, Away from emission, its state, Reset, Try variant,
        Apply), then the tools (Samples, Brush, Ellipse, the active tool's options, Remove, Remove all, Undo, Redo).
        Built whole: a tool switched on or off shows or hides its options, the rows never move."""
        method = self._method()
        sampled = method in ("polynomial", "rbf")
        bar = QFrame()
        bar.setObjectName("gradbar")
        bar.setStyleSheet("QFrame#gradbar { background: #1b1e25; border-bottom: 1px solid #2a2f38; }")
        v = QVBoxLayout(bar)
        v.setContentsMargins(10, 4, 10, 4)
        v.setSpacing(4)
        r1 = QHBoxLayout()
        r1.setSpacing(6)
        cb = QComboBox()
        for i, m in enumerate(METHODS):
            cb.addItem(_(METHOD_NAME[m]), m)
            cb.setItemData(i, _(METHOD_TIP[m]), Qt.ToolTipRole)
        cb.setCurrentIndex(METHODS.index(method) if method in METHODS else 0)
        cb.setToolTip(_("How the sky gradient is modelled") + "\n" + _(METHOD_TIP.get(method, "")))
        cb.currentIndexChanged.connect(lambda i: self.set_draft({"method": METHODS[i]}))
        r1.addWidget(cb)
        self._ui["method"] = cb
        name, text, lo, hi, step, dec, tip = MAIN[method if method in MAIN else "polynomial"]
        r1.addWidget(label(_(text), "dim"))
        sp = QDoubleSpinBox() if dec else QSpinBox()
        if dec:
            sp.setDecimals(dec)
        sp.setRange(lo, hi)
        sp.setSingleStep(step)
        cur = self.value(name)
        sp.setValue(cur if cur is not None else lo)
        sp.setKeyboardTracking(False)
        sp.setToolTip(_(tip))
        sp.valueChanged.connect(lambda val, nm=name: self.set_draft({nm: val}))
        r1.addWidget(sp)
        self._ui["main"] = sp
        av = QCheckBox(_("Away from emission"))
        av.setChecked(self.avoid)
        av.setEnabled(sampled)
        av.setToolTip(_("Generate grid and the samples the block chooses leave out the cells where the H-alpha survey "
                        "shows nebulosity (shown with H-alpha survey)") if sampled else
                      _("The automatic method uses no samples: it fits every pixel that does not look like an object"))
        av.toggled.connect(self._set_avoid)
        r1.addWidget(av)
        self._ui["avoid"] = av
        st = ElidedLabel("", "gradstate", Qt.ElideRight)
        st.setMinimumWidth(40)
        st.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        r1.addWidget(st, 1)
        self._ui["state"] = st
        for key, tip, fn in (("undo", "Undo the draft's last step (Ctrl+Z on the image)", self.undo),
                             ("redo", "Redo (Ctrl+Y on the image)", self.redo)):
            b = icon_only(QPushButton(), key, _(tip), 16)
            b.clicked.connect(fn)
            r1.addWidget(b)
            self._ui[key] = b
        for key, text, tip, fn in (("reset", "Reset", "Back to the applied settings: the draft goes", self.reset),
                                   ("variant", "Try variant", "The draft on a variant of the workflow: what is "
                                    "computed stays as it is, the variant is computed by Run", self.try_variant),
                                   ("apply", "Apply", "The draft becomes the block's settings (Ctrl+Z takes it back)",
                                    self.apply)):
            b = QPushButton(_(text))
            if key == "apply":
                b.setObjectName("primary")
            b.setToolTip(_(tip))
            b.clicked.connect(lambda _c=False, f=fn: f())
            r1.addWidget(b)
            self._ui[key] = b
        v.addLayout(r1)
        r2 = QHBoxLayout()
        r2.setSpacing(6)
        for key, text, tip in (
                ("points", "Samples", "Samples of sky: click adds one, drag moves it, Remove or a right click removes "
                 "the one selected"),
                ("brush", "Brush", "Paint the areas left out of the sky model (X: paint / erase, [ ]: size)"),
                ("ellipse", "Ellipse", "Drag an ellipse around an object to leave it out; its handles size and turn "
                 "it")):
            b = QPushButton(_(text))
            b.setObjectName("face")
            b.setCheckable(True)
            on = key != "points" or sampled
            b.setEnabled(on)
            b.setToolTip(_(tip) if on else _("The automatic method uses no samples: it fits every pixel that does "
                                            "not look like an object"))
            b.clicked.connect(lambda _c=False, k=key: self.set_mode(None if self.mode == k else k))
            r2.addWidget(b)
            self._ui[key] = b
        r2.addSpacing(6)
        g = QPushButton(_("Generate grid"))
        g.setToolTip(_("Editable samples on a regular grid, the brighter areas left out (Object rejection), in place "
                       "of the ones the block chooses at each run; then move or remove them"))
        g.clicked.connect(self.generate)
        dens = QSpinBox()
        dens.setRange(4, 64)
        dens.setValue(self._density())
        dens.setSuffix(" " + _("per side"))
        dens.setKeyboardTracking(False)
        dens.setToolTip(_("Samples of the grid along the shorter side of the image"))
        dens.valueChanged.connect(self._set_density)
        for w, k in ((g, "grid"), (dens, "density")):
            w.setVisible(sampled)
            r2.addWidget(w)
            self._ui[k] = w
        for text, val, tip in (("Paint", True, "The brush adds to the areas left out (X)"),
                               ("Erase", False, "The brush takes the areas left out away (X)")):
            b = QPushButton(_(text))
            b.setObjectName("face")
            b.setCheckable(True)
            b.setToolTip(_(tip))
            b.clicked.connect(lambda _c=False, vv=val: self._set_paint(vv))
            r2.addWidget(b)
            self._ui["paint" if val else "erase"] = b
        sz = QSlider(Qt.Horizontal)
        sz.setRange(4, 1200)
        sz.setValue(int(self.size))
        sz.setFixedWidth(90)
        sz.setToolTip(_("Brush size, in pixels of the stack ([ ])"))
        sz.valueChanged.connect(self._set_size)
        r2.addWidget(sz)
        self._ui["size"] = sz
        szl = label("", "small")
        r2.addWidget(szl)
        self._ui["size_l"] = szl
        r2.addSpacing(6)
        for b in self._view_toggles():         # what is drawn over the image: on this row, never scrolled away
            r2.addWidget(b)
        r2.addStretch(1)
        rm = QPushButton(_("Remove"))
        rm.setToolTip(_("Remove the sample or the ellipse selected (Delete; a right click on it does the same)"))
        rm.clicked.connect(self.remove_selected)
        r2.addWidget(rm)
        self._ui["remove"] = rm
        clr = QPushButton(_("Remove all"))
        clr.setToolTip(_("Remove the placed samples and the excluded areas from the draft: the block chooses its "
                         "samples again (Undo brings them back)"))
        clr.clicked.connect(self.clear)
        r2.addWidget(clr)
        self._ui["clear"] = clr
        v.addLayout(r2)
        for w in bar.findChildren(QPushButton):        # never squeezed: a name cut short reads as another word
            w.setMinimumWidth(w.sizeHint().width())
        return bar

    # ------------------------------------------------------------ the panel's section
    def controls(self, p) -> QWidget:
        """The panel's Method: one line per method (what it does and its durations in the tooltip), chosen into the
        draft; under it, where the tools are, and Run to here when the block waits for it."""
        self._load(p)
        from .model import methods_of
        from .panel import durations_text
        box = QFrame()
        v = QVBoxLayout(box)
        v.setContentsMargins(0, 2, 0, 6)
        v.setSpacing(3)
        cur = self._method()
        applied = self._applied_value("method")
        for m in methods_of("fond"):
            key = m["engine_key"]
            if key not in METHODS or m.get("new"):
                continue
            on = key == cur
            text = ("●  " if on else "○  ") + _(METHOD_NAME[key]) + "  ·  " + _(METHOD_LINE[key])
            if key == applied and cur != applied:
                text += "  ·  " + _("applied")
            b = QPushButton(text)
            b.setObjectName("methodline")
            b.setFlat(True)
            b.setCursor(Qt.PointingHandCursor)
            b.setStyleSheet("QPushButton#methodline { text-align: left; padding: 3px 6px; border: 1px solid "
                            + ("#5ea1ff" if on else "transparent") + "; border-radius: 4px; }"
                            "QPushButton#methodline:hover { background: #20242c; }")
            import re
            times = self.s.method_times(p, "fond", m["key"]) if hasattr(self.s, "method_times") else None
            dur = re.sub(r"<[^>]+>", "", durations_text(times, "gpu" if getattr(self.s, "gpu", False) else "cpu"))
            b.setToolTip(_(METHOD_TIP[key]) + "\n" + dur)
            b.clicked.connect(lambda _c=False, k=key: self.set_draft({"method": k}))
            v.addWidget(b)
        v.addWidget(label(_("Samples, exclusions, Apply and Try variant: on the bar above the image. Every change is "
                            "a draft until it is applied."), "small", wrap=True))
        if not self.done and self.s.state(p, "gradient") != "running" and not self.tv._has_previous(p, "fond"):
            v.addWidget(self.tv._run_button(p, "fond"), 0, Qt.AlignLeft)
        return box

    def groups(self, p) -> list:
        """The panel's settings groups of the draft's method (the applied method's are the block's own)."""
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
        """A settings row of the panel as the draft has it."""
        d = self._draft()
        if d is None or r.get("name") not in d["vals"]:
            return r
        val = d["vals"][r["name"]]
        return {**r, "value": val, "changed": val != r.get("suggested")}

    def draft_setting(self, r: dict, value):
        """A settings row of the panel edited: into the draft (None: back to the suggestion)."""
        self.set_draft({r["name"]: r.get("suggested") if value is None else value})

    def _status(self) -> str:
        bits = []
        if self.has_draft():
            n = len(self._draft()["vals"])
            bits.append(_("Draft: 1 change") if n == 1 else _("Draft: {n} changes").format(n=n))
            c = self.cost_text()
            if c:
                bits.append(c)
        else:
            bits.append(_("Applied settings") if self.done else _("Applied settings · not computed yet"))
        if self.busy:
            bits.append(self.step_text or _("computing the preview…"))
        pts, s = self.work["sample_points"], self.samples
        if self._method() != "automatic":
            if pts:
                bits.append(_("{n} samples placed").format(n=len(pts)))
            elif s is not None and not s.get("manual"):
                bits.append(_("samples chosen at each run ({n} of sky the last time)").format(n=int(np.sum(s["used"]))))
        sv = self.survey
        if (self.survey_shown or self._grid_after_survey) and sv is None:
            bits.append(_("H-alpha survey: reading…"))
        elif sv is not None and "error" in sv:
            bits.append(_("H-alpha survey: {why}").format(why=sv["error"]))
        return " · ".join(bits)

    def _hint(self):
        """The gestures of the tool on, on the line under the image (as Post's)."""
        if self.view is None:
            return
        t = {"points": _("Samples: click adds (none placed: the grid first) · drag moves · Remove or right click "
                         "removes · Space or middle button: move the image · R H-alpha survey · M hide"),
             "brush": _("Brush: drag to paint · X paint / erase · [ ] size · right drag: move the image · M hide"),
             "ellipse": _("Ellipse: drag to draw · click inside: select and move · handles: size and rotation · "
                          "Remove, Delete or right click inside: remove · Space: move the image")}.get(self.mode, "")
        try:
            self.view.tool_hint.setText(t)
            self.view.tool_hint.setVisible(bool(t))
        except RuntimeError:
            self.view = None

    def _sync(self):
        """The bar's widgets as the editor is (without building them again)."""
        if self.p is None:
            return
        try:
            u = self._ui
            for k in ("points", "brush", "ellipse"):
                if k in u:
                    u[k].setChecked(self.mode == k)
            brush = self.mode == "brush"
            for k, on in (("paint", self.paint), ("erase", not self.paint)):
                if k in u:
                    u[k].setChecked(on)
                    u[k].setVisible(brush)
            if "size" in u:
                u["size"].setVisible(brush)
                u["size_l"].setVisible(brush)
                u["size_l"].setText(f"{int(self.size)} px")
            sampled = self._method() in ("polynomial", "rbf")
            for k in ("grid", "density"):
                if k in u:
                    u[k].setVisible(sampled and not brush)
            for k, on in (("marks", self.shown), ("survey", self.survey_shown), ("avoid", self.avoid)):
                if k in u:
                    b = u[k]
                    b.blockSignals(True)
                    b.setChecked(on)
                    b.blockSignals(False)
            draft = self.has_draft()
            why = _("No draft: change a setting, a sample or an area first")
            for k in ("reset", "variant", "apply"):
                if k in u:
                    u[k].setEnabled(draft)
            if "apply" in u:
                c = self.cost_text()
                u["apply"].setToolTip(_("The draft becomes the block's settings (Ctrl+Z takes it back)") +
                                      ("\n" + c if c else "") if draft else why)
                u["variant"].setToolTip(_("The draft on a variant of the workflow: what is computed stays as it is, "
                                          "the variant is computed by Run") if draft else why)
                u["reset"].setToolTip(_("Back to the applied settings: the draft goes") if draft else why)
            if "state" in u:
                u["state"].setText(self._status())
                u["state"].setStyleSheet(f"color: {WARN};" if draft else "color: #8b95a5;")
            if "remove" in u:
                sel = (self.mode == "ellipse" and self._sel is not None) or \
                    (self.mode == "points" and self._selpt is not None)
                u["remove"].setEnabled(sel)
                if not sel:
                    u["remove"].setToolTip(_("Select a sample (Samples) or an ellipse (Ellipse) first"))
                else:
                    u["remove"].setToolTip(_("Remove the sample or the ellipse selected (Delete; a right click on it "
                                             "does the same)"))
            d = self._draft()
            if "undo" in u:
                u["undo"].setEnabled(bool(d and d["undo"]))
                u["redo"].setEnabled(bool(d and d["redo"]))
            if "clear" in u:
                u["clear"].setEnabled(any(self.work[f] for f in FIELDS))
        except RuntimeError:                   # the bar was built again
            self._ui = {}

    # ------------------------------------------------------------ modes
    def set_mode(self, mode: str | None):
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
        """The grid's points along the shorter side, then a new grid (two steps of the draft's Undo)."""
        if n == self._density():
            return
        self.set_draft({"samples": int(n)})
        self.generate()

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

    def _set_survey_shown(self, on: bool):
        """The H-alpha survey under the samples (R): fetched the first time for this workflow (the one action of the
        view that may download it)."""
        self.survey_shown = bool(on)
        if on and (self.survey is None or self.survey["pid"] != self.p.pid or "error" in self.survey):
            self._run("survey", fetch=True)
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
        """Whether the image takes this key before the shortcuts (the canvas asks, ShortcutOverride): the tools' keys,
        and Ctrl+Z / Ctrl+Y while the draft has steps."""
        if e.modifiers() & Qt.ControlModifier and e.key() in (Qt.Key_Z, Qt.Key_Y):
            d = self._draft()
            return bool(d and (d["undo"] or d["redo"]))
        return e.key() in TOOL_KEYS and not e.modifiers() & (Qt.ControlModifier | Qt.AltModifier)

    def key(self, e) -> bool:
        """The keys of the tool, from the canvas (it has the focus while a tool is on)."""
        k = e.key()
        if e.modifiers() & Qt.ControlModifier and k in (Qt.Key_Z, Qt.Key_Y):
            d = self._draft()
            if not d or not (d["undo"] or d["redo"]):
                return False
            redo = k == Qt.Key_Y or bool(e.modifiers() & Qt.ShiftModifier)
            (self.redo if redo else self.undo)()
            return True
        if e.modifiers() & (Qt.ControlModifier | Qt.AltModifier):
            return False
        if k == Qt.Key_BracketLeft and self.mode == "brush":
            self._resize(1 / 1.25)
        elif k == Qt.Key_BracketRight and self.mode == "brush":
            self._resize(1.25)
        elif k == Qt.Key_X and self.mode == "brush":
            self._set_paint(not self.paint)
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

    def generate(self):
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
        return Qt.BlankCursor if self.mode == "brush" else Qt.CrossCursor

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
                if not self.work["sample_points"]:
                    self._extra = [sx, sy]     # none placed: the grid first, this sample with it
                    self.generate()
                    return True
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
        elif self.mode == "brush" and self._stroke0 is not None:
            self._last = None
            self._stroke0 = None
            edited = True
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
            self._commit()
        if self.tv._dirty:                     # a refresh held back while the mouse was down
            self.tv.request_refresh()
        canvas.update()

    def hover(self, canvas, ix, iy):
        self._hover = (ix, iy)

    def _dab(self, x, y):
        r = self.size / 2
        pts = [(x, y)]
        if self._last is not None:                       # fill the gaps of a fast stroke (as Post's brush)
            lx, ly = self._last
            n = int(math.hypot(x - lx, y - ly) / max(r / 3, 1))
            pts = [(lx + (x - lx) * k / (n + 1), ly + (y - ly) * k / (n + 1)) for k in range(1, n + 1)] + pts
        for px, py in pts:
            self.work["exclude_brush"].append([float(px), float(py), float(r), 1.0 if self.paint else 0.0])
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
            vals.update(copy.deepcopy(d["vals"]))
        vals.update(work if work is not None else self.work)
        return GradientParams(**vals)

    def _paint(self, painter, canvas):
        if self.p is None or canvas.display is None:
            return
        f = self._f(canvas)
        dw, dh = canvas.display.size()
        if self.survey_shown:
            self._paint_survey(painter, canvas, dw, dh)
        if self.shown:
            self._paint_veil(painter, canvas, f, dw, dh)
            self._paint_samples(painter, canvas, f)
            for j, e in enumerate(self.work["exclude_ellipses"]):
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
        if self.mode == "brush" and self._hover is not None:
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(QColor(255, 255, 255, 200), 1.2))
            r = self.size / 2 * f * canvas.scale
            painter.drawEllipse(canvas.img_to_w(*self._hover), r, r)
        from .imageview import tag
        text = self.tag_text()
        if text:                               # bottom left: the curtain's names are on top
            tag(painter, QPointF(10, canvas.height() - 28), text, bg=DRAFT_BG if self.has_draft() else QColor(0, 0, 0, 170))

    def tag_text(self) -> str:
        """What the image shows, said on it: the draft's preview is never taken for the result."""
        if self.has_draft():
            if self.result is None:
                return self.step_text or _("Applied result · the draft preview follows")
            ms = f" · {self.result_ms:.0f} ms" if self.result_ms is not None else ""
            busy = " · " + (self.step_text or _("computing…")) if self.busy else ""
            return _("Draft preview · reduced copy") + ms + busy
        if self.result is not None:
            return _("Preview of the settings · not computed yet: Run to here")
        return self.step_text if self.busy else ""

    def _paint_veil(self, painter, canvas, f, dw, dh):
        brush = self.work["exclude_brush"]
        done = brush if self._stroke0 is None else brush[:self._stroke0]
        if self.work["exclude_ellipses"] or done:
            from ..stages.gradient import exclusion_mask
            k = min(1.0, 1500 / max(dw, 1))
            key = (dw, dh, len(done), str(self.work["exclude_ellipses"]), id(self.work))
            if self._veil is None or self._veil[0] != key:
                work = {**self.work, "exclude_brush": done}
                m = exclusion_mask(self._params(work), int(dh * k), int(dw * k), f * k)
                rgba = np.zeros((*m.shape, 4), np.uint8)
                rgba[m] = VEIL
                img = QImage(rgba.data, m.shape[1], m.shape[0], 4 * m.shape[1], QImage.Format_RGBA8888).copy()
                self._veil = (key, img)
            painter.drawImage(QRectF(canvas.img_to_w(0, 0), canvas.img_to_w(dw, dh)), self._veil[1])
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
        if self._method() == "automatic":
            return
        s, pts = self.samples, self.work["sample_points"]
        if pts:
            P = np.asarray(pts, float).reshape(-1, 2)
            same = (s is not None and s.get("manual") and len(s["x"]) == len(P)
                    and np.allclose(np.c_[s["x"], s["y"]], P, atol=0.6))
            used = list(s["used"]) if same else [None] * len(P)
            W, H = self._stack_size()
            box = s["box"] if s is not None else max(4, int(min(W, H) / self._density() / 2))
            rows = zip(P[:, 0], P[:, 1], used)
        elif s is not None and not s.get("manual"):
            box, rows = s["box"], zip(s["x"], s["y"], s["used"])
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
        """The draft's preview, a moment after the last edit (the automatic model waits a little more)."""
        if self.p is None:
            return
        self._timer.setInterval(AUTO_DEBOUNCE_MS if self._method() == "automatic" else DEBOUNCE_MS)
        self._timer.start()

    def preview(self, force: bool = False):
        if self.p is None or not (self.has_draft() or force):
            return
        self._run("preview")

    def _run(self, what: str, fetch: bool = False):
        """One computation at a time: a request made while one runs waits, replaced by the next (only the last
        counts); the answer of an older edit is dropped. ``fetch``: the survey may be downloaded (its toggle only)."""
        if self.p is None:
            return
        self.gen += 1
        if self.busy:
            if what == "survey":
                self._survey_wanted = bool(fetch) or bool(self._survey_wanted)
            else:
                self._pending = what if what != "preview" or self._pending != "grid" else "grid"
            return
        gen, params, pid, n = self.gen, self._params(), self.p.pid, self._density()
        small = self.small if self.small and self.small["pid"] == pid else None
        sv = self.survey if self.avoid and self.survey and self.survey["pid"] == pid else None
        level = sv.get("level") if sv else None
        # the block's own samples avoid the survey's emission: the preview too (the survey read from the disk with it)
        read = (what == "preview" and self.avoid and params.method != "automatic" and not params.sample_points
                and (self.survey is None or self.survey["pid"] != pid))
        akey = self._applied_key()
        want_applied = what in ("preview", "load") and (self.applied is None or self.applied["key"] != akey)
        project = self.s.p
        self.busy = True
        self.step_text = ""
        self._sync()
        if self.view is not None:
            self.view.canvas.update()
        step = self._step

        def work():
            try:
                sm = small or _reduced_stack(project, pid)
                out = {"small": sm, "what": what}
                if want_applied:
                    out["applied"] = {"key": akey, "arr": _applied_small(project, pid, sm)}
                if what == "grid":
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
                    out.update(_preview(sm, params, lv, progress=lambda c: step.emit(gen, _(
                        "automatic model: colour {n} of 3").format(n=c + 1))))
                    out["ms"] = 1000 * (time.thread_time() - t0)
                    out["sig"] = params.model_dump_json()
                elif what == "survey":
                    try:
                        out["survey"] = {"pid": pid, **_survey_small(project, pid, sm, fetch=fetch)}
                    except Exception as e:  # noqa: BLE001 - no survey for this field: said, the editor goes on
                        out["survey"] = {"pid": pid, "error": _survey_why(e)}
                self._done.emit(gen, out)
            except Exception as e:  # noqa: BLE001 - said in the bar, the editor goes on
                self._done.emit(gen, {"error": str(e), "what": what})
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

    def _arrived(self, gen: int, out: dict):
        self.busy = False
        self.step_text = ""
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
        elif "grid" in out:
            self.work["sample_points"] = out["grid"] + ([self._extra] if self._extra else [])
            self._extra = None
            if out.get("avoided") is False:
                self.tv.toasted.emit(_("The H-alpha survey shows no clear nebulosity here: no sample left out"))
            self._commit()
        elif "result" in out and gen == self.gen and self.p is not None and (self.has_draft() or not self.done):
            self.result, self.result_ms = out["result"], out.get("ms")
            self.samples = out["samples"]
            self._show(self.result)
        if self._survey_wanted is not None:
            fetch, self._survey_wanted = self._survey_wanted, None
            if self.survey is None or self.survey["pid"] != self.p.pid or (fetch and "error" in self.survey):
                self._run("survey", fetch=fetch)
                return
        if self._pending is not None:
            what, self._pending = self._pending, None
            self._run(what)
        elif gen != self.gen and out.get("what") == "preview" and self.has_draft():
            self.preview()                     # edits came during the computation: the last one
        self._sync()
        if self.view is not None:
            try:
                self.view.canvas.update()
            except RuntimeError:
                self.view = None

    def _show(self, res):
        """The preview as the main image; the applied result on the same grid and the stack to compare it with."""
        from .imageview import ViewImage
        if self.view is None or self.small is None or self.small["pid"] != self.p.pid:
            return
        try:
            v = self.view
            pid = self.p.pid
            # Neutral sky at first (display only, as the block's view): before the colour calibration, the green of
            # the camera hides what the edits do to the sky
            draft = self.has_draft()
            vi = ViewImage(res, _("Draft preview") if draft else _("Preview"), key=("gradient-preview", pid, self.gen),
                           camera=True, neutral=True)
            targets = []
            ap = self.applied
            if ap is not None and ap.get("arr") is not None and ap["key"][0] == pid:
                targets.append(ViewImage(ap["arr"], _("Applied") if draft else _("Previous settings"),
                                         key=("gradient-applied", ap["key"]), camera=True, neutral=True))
            targets.append(ViewImage(self.small["arr"], _("Stack"), key=("gradient-small", pid), camera=True,
                                     neutral=True))
            names = [t.name for t in targets]
            cur = names.index(self._cmp_name) if self._cmp_name in names else (None if self._cmp_name == "" else 0)
            self._in_show = True
            try:
                v.set_image(vi, keep_view=True, keep_stretch=True)
                v.set_compare_targets(targets, cur)
                v.set_same_stretch(True)
            finally:
                self._in_show = False
            v.canvas.tool = self if self.mode else None
        except RuntimeError:
            self.view = None


def _reduced_stack(project, pid: int) -> dict:
    """The stack of the workflow reduced to about PREVIEW_PIXELS (block means), its covered pixels."""
    from ..api import stage_outputs
    out = stage_outputs(project, pid, "stack")
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


def _preview(sm: dict, params, emission=None, progress=None) -> dict:
    """The stack with the model's gradient removed (its level kept), on the reduced copy, and its samples (``emission``:
    the survey's level on the copy, which the block's own samples avoid, as the run's): the stage's own functions."""
    from ..stages.gradient import read_grid, sky_model_of
    m = sky_model_of(sm["arr"], sm["valid"], params, f=sm["f"], progress=progress, emission=emission)
    h, w = sm["arr"].shape[1:]
    g = read_grid(m["grid"], m["info"], h, w)
    lvl = np.array([np.median(g[c][sm["valid"]]) for c in range(3)], np.float32)[:, None, None]
    return {"result": (sm["arr"] - g + lvl).astype(np.float32), "samples": m["samples"]}
