"""The combination block (S-H-O, H-O and the other recipes of stages/combine.py) tuned in Inspect as a draft
(``CombineEditor``), built as the Gradient block's editor (gradientedit.py) is.

Every adjustment (a palette, a filter's hue or intensity, the balance, the stars) changes a DRAFT of the block's
settings, kept per workflow while the project is open (blockdraft.Draft): the project does not change, the graph stays
up to date, Run is unchanged. The draft is seen at once on a reduced copy of the inputs (about PREVIEW_PIXELS, whatever
the sensor), computed with the stage's own functions (``Combination.steps``, the balance, measured once per balance in
the editor's thread; ``Combination.mix_balanced``, the weights and the recipe's output, at every edit: milliseconds) and
shown through the default render of the block's Output (``output_render``: render_simple's statistics and
SimpleRenderEngine with its default settings), the last edit only, the previous image kept until the next is ready. It
is compared with the applied result on the same grid (the block's own image, reduced the same way): Draft, Applied,
Split.

When it looks right: **Apply** (the draft becomes the block's settings, one step of the window's Undo), **Try variant**
(the draft goes on a variant of the workflow) or **Reset**: the draft bar every block has (blockdraft.DraftBar), at the
top of the editor. Leaving the block with a draft asks the question every block asks (Apply, Try variant, Discard,
Stay).

The editor, from the top: the draft bar; Palette, starting hues in one click (only the palettes the recipe's filters
fit); one card per filter, in the recipe's order: its thumbnail (balanced, stretched; a click shows that filter alone
in the image), Hue (a rail of the colours, a swatch, the degrees), Intensity (the weight after the balance, 1 = as
balanced), an eye that takes the filter out of the preview's mix (the view only, never applied), and the gain the
balance measured and why; Balance, On signal or On sky. The combination of the stars alone has only its Stars setting
(Neutral, Palette). The gains of a preview are the reduced copy's: when the draft keeps the balance, the applied run's
are shown (its drz_meta.json), the copy's said in the tooltip when they differ.
"""
from __future__ import annotations

import colorsys
import copy
import math
import threading
import time
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, QRunnable, QSize, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from . import journal
from .blockdraft import Draft, DraftBar
from .gradientedit import PREVIEW_PIXELS
from .i18n import _, number
from .icons import set_icon
from .widgets import WARN, label

#: the column of the editor at the left of the image: its width
PANEL_W = 380
#: a filter's thumbnail: its width (px)
THUMB_W = 84
#: Intensity: its range and its step (1 = as balanced)
INTENSITY_MAX, INTENSITY_STEP = 3.0, 0.05
#: the starting palettes: (key, name, the filters they colour, {filter: hue}); a palette is offered when the recipe's
#: hued filters are exactly its own. Hubble: the S-H-O hues measured on the Hubble teams' images (stages/combine.py,
#: HUBBLE_HUES, the S-H-O recipe's defaults); H-O-O: H-alpha red, O III cyan.
PALETTES = (("hubble", "Hubble (S-H-O)", ("SII", "Ha", "OIII"), None),
            ("hoo", "H-O-O", ("Ha", "OIII"), {"Ha": 0.0, "OIII": 180.0}))
BALANCES = ("signal", "sky")
BALANCE_NAME = {"signal": "On signal", "sky": "On sky"}
BALANCE_LINE = {"signal": "Each filter's nebula fitted onto the H-alpha's, never raised into its own noise.",
                "sky": "Each filter's sky brought to the H-alpha's (its median and its spread), whatever its signal."}
STARS = ("neutral", "palette")
STARS_NAME = {"neutral": "Neutral", "palette": "Palette"}
STARS_LINE = {"neutral": "White stars: the filters' plain mean, no hue.",
              "palette": "The stars through the filters' hues, coloured like the nebula."}
#: why a filter has its gain (the balance's notes, stages/combine.Combination.steps)
REASONS = {"signal": "signal", "noise": "limited by noise", "sky": "on sky", "match": "linear match",
           "reference": "reference"}
#: what each reason means (the gain line's tooltip)
REASON_TIPS = {"signal": "Signal: the gain that fits this filter's nebula onto the H-alpha's.",
               "noise": "Limited by noise: its signal asked for more, the gain stops where this filter's sky noise "
                        "equals the H-alpha's, so a weak filter is not raised into its noise.",
               "sky": "On sky: the gain that brings this filter's sky median and spread to the H-alpha's.",
               "match": "Linear match: the least squares fit of this filter onto the reference.",
               "reference": "Reference: the other filters are brought to this one's scale."}
#: the image area's comparisons
COMPARE = (("draft", "Draft", "The draft's preview"), ("applied", "Applied", "The block's applied result"),
           ("split", "Split", "The draft on the right of the divider, the applied result on the left"))
#: the gains of the copy and of the run differ when they are this far apart (relative)
GAIN_APART = 0.03
HUE_RAIL = ("qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #ff0000, stop:0.1667 #ffff00, stop:0.3333 #00ff00, "
            "stop:0.5 #00ffff, stop:0.6667 #0000ff, stop:0.8333 #ff00ff, stop:1 #ff0000)")


# ---------------------------------------------------------------------------------------------- the stage's pieces
def stage_class(stage: str):
    """The combination block of a stage name (stages/combine.ALL)."""
    from ..stages.combine import ALL
    return next(c for c in ALL if c.name == stage)


def hue_params(spec: dict) -> dict[str, str]:
    """{filter: the parameter of its hue} of a recipe whose output gives each filter a hue ("$hue_ha"); {} for a
    recipe mixed through a matrix (RGB, LRGB, HaRGB)."""
    o = spec.get("output") or {}
    hues = o.get("hues") or []
    return {r: h[1:] for r, h in zip(o.get("channels") or [], hues) if isinstance(h, str) and h.startswith("$")}


def stars_only(inputs) -> bool:
    """Every wired input reads its stars alone: the stars' combination (its only setting is Stars)."""
    return bool(inputs) and all(i.get("image") == "stars" for i in inputs)


def has_balance(spec: dict, inputs) -> bool:
    """Whether the setting Balance acts on this combination: its recipe equalises, and it is not the stars'."""
    return any(s.get("op") == "equalise" for s in spec.get("steps") or []) and not stars_only(inputs)


def applied_values(cls, spec: dict, stored: dict) -> dict:
    """The settings the editor tunes, as the block has them: each hue (by its parameter), "weights" ({filter: weight},
    1 left out), "balance" and "stars" when the block has them."""
    fields = cls.Params.model_fields
    out = {}
    vals = stored.get("values") or {}
    for _r, name in hue_params(spec).items():
        if name in fields and name not in ("values",):
            out[name] = float(stored.get(name, fields[name].default))
        else:
            out[name] = float(vals.get(name, spec["parameters"][name]["default"]))
    out["weights"] = norm_weights(stored.get("weights") or {})
    for name in ("balance", "stars"):
        if name in fields:
            out[name] = stored.get(name, fields[name].default)
    return out


def fitting_palettes(spec: dict) -> list[tuple[str, str]]:
    """(key, name) of the palettes a recipe's hued filters fit."""
    have = set(hue_params(spec or {}))
    return [(k, n) for k, n, roles, _h in PALETTES if have and set(roles) == have]


def palette_hues(key: str) -> dict[str, float]:
    from ..stages.combine import HUBBLE_HUES
    for k, _n, roles, hues in PALETTES:
        if k == key:
            return dict(zip(roles, HUBBLE_HUES)) if hues is None else dict(hues)
    return {}


def palette_of(spec: dict, values: dict) -> str:
    """The palette (its key) the hues of ``values`` are, "" when they are the user's own."""
    names = hue_params(spec or {})
    for k, _n in fitting_palettes(spec):
        if all(abs(float(values.get(names[r], -1)) - h) < 0.05 for r, h in palette_hues(k).items()):
            return k
    return ""


def norm_weights(w: dict) -> dict:
    return {r: round(float(v), 3) for r, v in sorted(w.items()) if abs(float(v) - 1.0) > 1e-6}


def params_of(cls, spec: dict, stored: dict, values: dict):
    """The block's parameters with ``values`` (the editor's settings) over the stored ones: what the run would take."""
    d = {k: copy.deepcopy(v) for k, v in stored.items() if k not in ("inputs", "solution")}
    fields = cls.Params.model_fields
    for name, v in values.items():
        if name == "weights":
            d["weights"] = dict(v)
        elif name in ("balance", "stars") or (name in fields and name != "values"):
            d[name] = v
        else:
            d.setdefault("values", {})
            d["values"] = {**d["values"], name: v}
    d["recipe"] = spec
    return cls.Params.model_validate(d)


class _Quiet:
    """What the stage's functions ask of a run's context, for the preview: nothing is written, nothing said."""

    def log(self, *_a, **_k):
        pass

    def metric(self, *_a, **_k):
        pass

    def progress(self, *_a, **_k):
        pass


def reduce_mean(a: np.ndarray, k: int) -> np.ndarray:
    """An image (H, W) reduced by k x k block means (the incomplete border dropped)."""
    if k <= 1:
        return np.asarray(a, np.float32)
    h, w = a.shape[0] // k, a.shape[1] // k
    return np.asarray(a[:h * k, :w * k], np.float32).reshape(h, k, w, k).mean(axis=(1, 3))


def reduced_inputs(inputs: list[dict], spec: dict, pixels: int = PREVIEW_PIXELS) -> dict:
    """The combination's inputs reduced to about ``pixels`` (block means), each its channel (stages/combine.channel_of),
    and their covered pixels: {"chans", "valid", "k", "H", "W", "inputs"}. ValueError naming an input not computed."""
    from ..stages import recipes
    from ..stages.combine import IMAGES, channel_of
    chans, valid, k, shape = {}, None, None, None
    for inp in inputs:
        d = Path(inp["dir"])
        name = IMAGES[inp.get("image", "output")][1]
        if not (d / name).is_file():
            raise ValueError(_("{name} is not computed").format(name=_port(inp["role"])))
        arr = np.load(d / name, mmap_mode="r")
        H, W = arr.shape[-2:]
        if k is None:
            k = max(1, int(math.ceil(math.sqrt(H * W / pixels))))
            shape = (H, W)
        ch = channel_of(np.asarray(arr), inp.get("channel", recipes.default_channel(spec, inp["role"])))
        chans[inp["role"]] = reduce_mean(ch, k)
        v = np.load(d / "valid.npy")
        v = reduce_mean(v.astype(np.float32), k) > 0.999
        valid = v if valid is None else (valid & v)
    if not chans:
        raise ValueError(_("No input to combine"))
    return {"chans": chans, "valid": valid, "k": k, "H": shape[0], "W": shape[1], "inputs": inputs}


def balance(small: dict, stage: str, spec: dict, how: str) -> dict:
    """The recipe's steps on the reduced inputs (the stage's ``steps``): the balanced channels, their skies, scales and
    each filter's gain and why; the thumbnails of the balanced filters."""
    cls = stage_class(stage)
    chans = {r: c.copy() for r, c in small["chans"].items()}
    tops = {r: 1.0 for r in chans}
    notes: dict = {}
    skies = cls().steps(_Quiet(), spec, small["inputs"], chans, small["valid"], tops, how, notes)
    return {"chans": chans, "skies": skies, "tops": tops, "notes": notes,
            "thumbs": {r: thumbnail(c, small["valid"]) for r, c in chans.items()}}


def mixed(small: dict, bal: dict, stage: str, spec: dict, p, off=(), solo=None) -> np.ndarray:
    """(3, h, w): the stage's own end (``mix_balanced``: the weights, the recipe's output) on the balanced channels,
    the filters in ``off`` (or all but ``solo``) at 0: the preview's mix."""
    cls = stage_class(stage)
    chans = {r: (np.zeros_like(c) if r in off or (solo is not None and r != solo) else c.copy())
             for r, c in bal["chans"].items()}
    return cls().mix_balanced(_Quiet(), p, spec, small["inputs"], chans, dict(bal["skies"]), dict(bal["tops"]),
                              small["valid"])


def output_render(rgb: np.ndarray, settings: dict | None = None) -> np.ndarray:
    """(3, h, w) in 0-1: a combination's image as its Output shows it: render_simple keeps a combination's sky at 0
    (``sky_neutral``), measures its stretch's statistics on the covered pixels (``image_stats``) and Post renders it
    with SimpleRenderEngine: with the settings of the workflow's layer of Post (``settings``, ``layer_settings``: its
    sliders, and the scale of its Sky slider: a layer saved before 2026-11-16 keeps its sky at 0.25), else the
    default ones (render_simple's preview)."""
    from ..render.simple import SimpleRenderEngine, SimpleSettings
    eng = SimpleRenderEngine.from_array(rgb, device="cpu", linked=True, sample=1)
    s = SimpleSettings(**settings) if settings else SimpleSettings()
    return np.asarray(eng.render(s, 1.0).final.numpy(), np.float32)


def layer_settings(layers: list[dict], pid: int) -> dict | None:
    """The rendering settings Post gives the workflow's own layer (its Output image, else its image recombined with
    its stars), as Post's render worker computes them (render/sliders.settings_at); None without a layer."""
    from ..render.sliders import settings_at
    from .postengine import sky_scale
    mine = [x for x in layers or [] if x.get("kind") == "workflow" and x.get("pid") == pid
            and x.get("part") in (None, "combined", "starless")]
    if not mine:
        return None
    lay = min(mine, key=lambda x: (None, "combined", "starless").index(x.get("part")))
    try:
        return settings_at("simple", lay.get("base"), lay.get("sliders"), has_denoise=False, origin="palette",
                           stretch=lay.get("stretch"), sky_scale=sky_scale(lay)).model_dump()
    except Exception:  # noqa: BLE001 - a layer of an earlier format: the default render
        return None


def thumbnail(ch: np.ndarray, valid: np.ndarray, width: int = THUMB_W) -> np.ndarray:
    """(h, w) uint8: a balanced filter stretched for its card (black just under the sky, its median at a quarter)."""
    import cv2
    v = ch[valid] if valid.any() else ch.ravel()
    med = float(np.median(v))
    mad = 1.4826 * float(np.median(np.abs(v - med)))
    lo, hi = med - 2.8 * mad, float(np.percentile(v, 99.9))
    x = np.clip((ch - lo) / max(hi - lo, 1e-12), 0, 1)
    xm = min(max((med - lo) / max(hi - lo, 1e-12), 1e-6), 1 - 1e-6)
    m = ((0.25 - 1) * xm) / ((2 * 0.25 - 1) * xm - 0.25)       # the midtone that shows the median at 0.25
    y = (m - 1) * x / ((2 * m - 1) * x - m)
    h = max(1, int(round(ch.shape[0] * width / ch.shape[1])))
    return (np.clip(cv2.resize(y.astype(np.float32), (width, h), interpolation=cv2.INTER_AREA), 0, 1) * 255
            ).astype(np.uint8)


def applied_small(result_dir: str, k: int, settings: dict | None = None) -> dict | None:
    """The block's own result (drz_full) reduced as the inputs were, rendered as its Output shows it, and the gains its
    run measured ({filter: gain}, drz_meta.json's scales); None without a result."""
    import json
    d = Path(result_dir) if result_dir else None
    if d is None or not (d / "drz_full.npy").is_file():
        return None
    arr = np.load(d / "drz_full.npy", mmap_mode="r")
    if k is None:                                       # the inputs not read: the same rule on the result's size
        k = max(1, int(math.ceil(math.sqrt(arr.shape[-2] * arr.shape[-1] / PREVIEW_PIXELS))))
    rgb = np.stack([reduce_mean(arr[c], k) for c in range(3)])
    meta = json.loads((d / "drz_meta.json").read_text(encoding="utf-8")) if (d / "drz_meta.json").is_file() else {}
    return {"img": output_render(rgb, settings), "tops": meta.get("channels") or {}}


def _port(role: str) -> str:
    from .graphnode import port_name
    return port_name(role)


def hue_colour(h: float) -> QColor:
    r, g, b = colorsys.hsv_to_rgb((float(h) % 360) / 360, 1.0, 1.0)
    return QColor.fromRgbF(r, g, b)


def _qimage(a: np.ndarray) -> QImage:
    a = np.ascontiguousarray(a)
    return QImage(a.data, a.shape[1], a.shape[0], a.strides[0], QImage.Format_Grayscale8).copy()


class _Job(QRunnable):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def run(self):
        self.fn()


class _Clickable(QFrame):
    """A frame that is a choice: a click on it (a Balance card, a thumbnail)."""

    clicked = Signal()

    def mousePressEvent(self, e):                                     # noqa: N802 - Qt's name
        if e.button() == Qt.LeftButton and self.isEnabled():
            self.clicked.emit()
        super().mousePressEvent(e)


def _choice_card(name: str, line: str, tip: str = "") -> _Clickable:
    """A method card as Gradient's: its name, one plain sentence under it; lit when chosen (``set_on``)."""
    c = _Clickable()
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


def _lit(card: QFrame, on: bool):
    card.setProperty("on", bool(on))
    card.setStyleSheet("QFrame#combchoice { border: 1px solid %s; border-radius: 6px; background: %s; }"
                       % (("#5ea1ff", "#2b3446") if on else ("#3a4150", "transparent")))


class CombineEditor(QObject):
    """The combination block of the workflow shown, tuned as a draft in Inspect's centre. ``center(tv, node, p)`` from
    TuneView.refresh; ``set_draft``, ``apply()``, ``try_variant()``, ``reset()``; ``draft_pending`` / ``draft_apply``
    / ``draft_variant`` / ``draft_reset`` for the leave question (blockdraft.settle); ``history_top``, ``undo``, ``redo``
    for the window's one order of Undo."""

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
        self.off: set = set()                  # the filters taken out of the preview (the eyes): the view only
        self.solo = None                       # the filter shown alone (its thumbnail clicked)
        self.small = None                      # the reduced inputs {"pid", "sig", "chans", "valid", "k", ...}
        self.bal: dict = {}                    # (pid, sig, balance) -> the balanced channels (``balance``)
        self.result = None                     # the draft's preview (3, h, w) in 0-1
        self.result_key = None
        self.result_ms = None
        self.notes: dict = {}                  # the gains of the balance shown, {filter: (gain, why)}
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
        self._shape = None                     # what the page was built for (its filters, the stars' setting alone)
        self._done.connect(self._arrived)

    # ------------------------------------------------------------ the block shown
    def _stage(self, p=None) -> str | None:
        from ..core.workflow import combine_stage
        p = p or self.p
        try:
            return combine_stage(self.s.p.workflow(p.pid))
        except (KeyError, AttributeError, ValueError):
            return None

    def _spec(self, p=None) -> dict | None:
        from .graphnode import combine_spec
        return combine_spec(p or self.p)

    def _stored(self, p=None) -> dict:
        p = p or self.p
        st = self._stage(p)
        try:
            return dict(self.s.p.workflow(p.pid).params.get(st) or {}) if st else {}
        except (KeyError, AttributeError):
            return {}

    def _wired(self, p=None) -> list[dict]:
        p = p or self.p
        return list(((getattr(p, "source", None) or {}).get("fusion") or {}).get("inputs") or [])

    def roles(self, p=None) -> list[str]:
        """The recipe's inputs, in its order."""
        spec = self._spec(p)
        return [i["role"] for i in (spec or {}).get("inputs") or []]

    def applied(self, p=None) -> dict:
        p = p or self.p
        st, spec = self._stage(p), self._spec(p)
        if not st or spec is None:
            return {}
        return applied_values(stage_class(st), spec, self._stored(p))

    def stars_mode(self, p=None) -> bool:
        spec = self._spec(p)
        return bool(spec and spec.get("narrowband")) and stars_only(self._wired(p))

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

    def weight(self, role: str) -> float:
        return float((self.value("weights") or {}).get(role, 1.0))

    def set_draft(self, updates: dict, undo: bool = True) -> bool:
        """Settings of the draft: one step of its Undo (``undo``); a setting back to its applied value leaves it."""
        if self.p is None:
            return False
        d = self.drafts.setdefault(self.p.pid, Draft("fusion", [self.p.pid]))
        ap = self.applied()
        norm = {k: (norm_weights(v) if k == "weights" else round(float(v), 2) if k not in ("balance", "stars") else v)
                for k, v in updates.items()}
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

    def has_result(self, p=None) -> bool:
        """The block has a result to show, computed or Outdated: its settings are a draft (Apply); else at once."""
        p = p or self.p
        st = self._stage(p)
        return bool(st) and self.s.state(p, st) in ("done", "stale")

    def set_hue(self, role: str, deg: float, undo: bool = True) -> bool:
        name = hue_params(self._spec()).get(role)
        return bool(name) and self.set_draft({name: float(deg)}, undo)

    def set_intensity(self, role: str, w: float, undo: bool = True) -> bool:
        ws = dict(self.value("weights") or {})
        ws[role] = round(float(w), 3)
        return self.set_draft({"weights": ws}, undo)

    def set_palette(self, key: str) -> bool:
        hues = self.palette_hues(key)
        if not hues:
            return False
        names = hue_params(self._spec())
        return self.set_draft({names[r]: h for r, h in hues.items()})

    def palettes(self) -> list[tuple[str, str]]:
        """(key, name) of the palettes the recipe's hued filters fit."""
        return fitting_palettes(self._spec() or {})

    @staticmethod
    def palette_hues(key: str) -> dict[str, float]:
        return palette_hues(key)

    def palette_now(self) -> str:
        """The palette the hues are (its key), "" when they are the user's own."""
        return palette_of(self._spec() or {}, self.values())

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
        """What the draft changes, one (name, words) per setting (a filter's hue, its intensity, the balance)."""
        q = self._view_of(pid)
        d = self.drafts.get(q.pid) if q is not None else None
        if not d:
            return []
        names = {v: r for r, v in hue_params(self._spec(q) or {}).items()}
        ap = self.applied(q)
        out = []
        for k, v in d.vals.items():
            if k in names:
                out.append((k, _("{filter} hue {old}° → {new}°").format(filter=_port(names[k]), old=f"{ap.get(k, 0):.0f}",
                                                                        new=f"{float(v):.0f}")))
            elif k == "weights":
                for r in sorted(set(v) | set(ap.get("weights") or {})):
                    a, b = float((ap.get("weights") or {}).get(r, 1.0)), float(v.get(r, 1.0))
                    if abs(a - b) > 1e-6:
                        out.append((f"weights.{r}", _("{filter} intensity {old} → {new}").format(
                            filter=_port(r), old=number(a, 2), new=number(b, 2))))
            elif k == "balance":
                out.append((k, _("Balance: {old} → {new}").format(old=_(BALANCE_NAME.get(ap.get(k), "")),
                                                                  new=_(BALANCE_NAME.get(v, v)))))
            elif k == "stars":
                out.append((k, _("Stars: {old} → {new}").format(old=_(STARS_NAME.get(ap.get(k), "")),
                                                                new=_(STARS_NAME.get(v, v)))))
        return out

    def draft_change(self, pid: int | None = None) -> dict | None:
        """The draft as one change of the model (kind "combine": graphmodel.GraphEditing._combine_steps), with its
        words; None without a draft."""
        q = self._view_of(pid)
        d = self.drafts.get(q.pid) if q is not None else None
        if not d:
            return None
        ap = self.applied(q)
        vals = {k: v for k, v in d.vals.items() if k != "weights"}
        if "weights" in d.vals:
            w, aw = d.vals["weights"], ap.get("weights") or {}
            vals["weights"] = {r: float(w.get(r, 1.0)) for r in sorted(set(w) | set(aw))
                               if abs(float(w.get(r, 1.0)) - float(aw.get(r, 1.0))) > 1e-6}
        words = [t for _k, t in self.changes(q.pid)]
        short = ", ".join(words) if len(words) <= 2 else _("{n} changes").format(n=len(words))
        return {"kind": "combine", "values": vals, "short": short,
                "what": _("{block} draft: {what}").format(block=self.title(q), what=", ".join(words))}

    def title(self, q=None) -> str:
        from .graphnode import combine_kind, combine_title
        q = q or self.p
        return combine_title(combine_kind(q), q) if q is not None else _("Combination")

    def heading(self, q=None) -> str:
        """The page's title: the workflow's name, its recipe's title in it the block's own (graphcore combos.retitled:
        a block whose recipe became H-O is never titled S-H-O), the recipe's title before a name the user gave."""
        from ..graphcore import combos
        q = q or self.p
        title = self.title(q)
        name = combos.retitled(getattr(q, "name", "") or "", combos.short((self._spec(q) or {}).get("name") or title))
        if not name:
            return title
        return name if combos.title_stem(name) else f"{title}  ·  {name}"

    def _paths(self, q):
        return [q]

    def apply(self, pid: int | None = None, quiet: bool = False):
        """The draft becomes the block's settings: one step of the window's Undo. ``quiet``: a change made at once on
        a block without a result (no draft was shown: nothing said; refused, the change goes)."""
        q = self._view_of(pid)
        ch = self.draft_change(q.pid) if q is not None else None
        if ch is None:
            return
        from ..core import workflow as wfl
        from .model import NotInEngine
        try:
            if hasattr(self.s, "apply_as"):
                self.s.apply_as(ch, "replace", self._paths(q))
            else:
                self.s.apply_change(self._paths(q), ch)
        except (ValueError, NotInEngine, wfl.OptionRefused) as e:
            self.tv.toasted.emit(_(str(e)))
            if quiet:
                self.drafts.pop(q.pid, None)
                self._changed()
            return
        if hasattr(self.s, "log"):
            self.s.log("{what} · {names}", what=ch["what"], names=q.name)
        self.drafts.pop(q.pid, None)
        if quiet:
            self._changed()
        else:
            self.tv.toasted.emit(_("{block}: draft applied · Run to here computes it · Ctrl+Z").format(
                block=self.title(q)))
        self.historyChanged.emit()
        self.tv.request_refresh()

    def try_variant(self, pid: int | None = None):
        """The draft on a variant of the workflow (what is computed stays as it is); the variant is shown."""
        q = self._view_of(pid)
        ch = self.draft_change(q.pid) if q is not None else None
        if ch is None:
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
        return (self.heading(q) if q is not None else self.title(), [ch["what"]] if ch else [], "")

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
        """From TuneView.refresh: the centre is this page when the block shown is a combination (True)."""
        want = (node is not None and p is not None and str(tv.s.sel_step) == "fusion" and not node.cut
                and self._stage(p) is not None and self._spec(p) is not None)
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
            self.off, self.solo = set(), None
            self.result = None
            self.notes = {}
        self.p = p
        d = self._draft()
        if d is not None:                      # what the block now holds leaves the draft
            ap = self.applied()
            d.drop_pending({k for k, v in d.vals.items() if v != ap.get(k)})

    def _enter_page(self, p):
        tv = self.tv
        from .i18n import language
        new = self.p is None or self.p.pid != p.pid or not self._page_ok() or self._lang != language() or \
            self._shape != (self.stars_mode(p), tuple(self.roles(p)), self._stage(p))
        self._lang = language()
        self._shape = (self.stars_mode(p), tuple(self.roles(p)), self._stage(p))
        self._load(p)
        if not self._page_ok() or new:
            self._build_page()
        if tv.center.indexOf(self.page) < 0:
            tv._clear_center()
            tv.center.addWidget(self.page, 1)
            self.page.show()
        self._side(False)
        tv.view, tv.views = self.gv, []
        tv._center_sig = ("combination", p.pid)
        self._sync()
        self._present()
        self.preview()

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
        """Built for the block shown: at the left the editor's column (the draft bar, Palette, the filters' cards,
        Balance; the stars' combination: Stars only), the image at the right, its comparison in the title row."""
        from .imageview import ImageView
        if self._page_ok():
            try:
                self.page.setParent(None)
                self.page.deleteLater()
            except RuntimeError:
                pass
        u = self._ui = {}
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
        bar.apply.connect(lambda: self.apply())
        bar.variant.connect(lambda: self.try_variant())
        bar.reset.connect(lambda: self.reset())
        u["bar"], u["state"] = bar, bar.what
        u["apply"], u["variant"], u["reset"] = bar.b_apply, bar.b_variant, bar.b_reset
        cv.addWidget(bar)
        if self.stars_mode():
            cv.addWidget(label(_("Stars"), "h2"))
            for key in STARS:
                c = _choice_card(_(STARS_NAME[key]), _(STARS_LINE[key]))
                c.clicked.connect(lambda k=key: self.set_draft({"stars": k}))
                cv.addWidget(c)
                u["stars_" + key] = c
        else:
            if hue_params(self._spec()):
                cv.addWidget(label(_("Palette"), "h2"))
                pal = QComboBox()
                pal.setObjectName("palette")
                for k, n in self.palettes():
                    pal.addItem(_(n), k)
                pal.addItem(_("Custom"), "")
                pal.setToolTip(_("Starting hues for the filters in one click; then adjust each filter below"))
                pal.activated.connect(lambda i, b=pal: b.itemData(i) and self.set_palette(b.itemData(i)))
                cv.addWidget(pal)
                u["palette"] = pal
            cv.addWidget(label(_("Filters"), "h2"))
            for role in self.roles():
                cv.addWidget(self._card(role))
            if has_balance(self._spec(), self._wired()) and "balance" in self.applied():
                cv.addWidget(label(_("Balance"), "h2"))
                for key in BALANCES:
                    c = _choice_card(_(BALANCE_NAME[key]), _(BALANCE_LINE[key]))
                    c.clicked.connect(lambda k=key: self.set_draft({"balance": k}))
                    cv.addWidget(c)
                    u["balance_" + key] = c
        cv.addStretch(1)
        scroll.setWidget(col)
        body.addWidget(scroll)
        box = QWidget()
        bl = QVBoxLayout(box)
        bl.setContentsMargins(0, 0, 0, 0)
        gv = ImageView(post=False, compare=True, histogram=False)
        gv.default_dir = lambda: str(getattr(self.s, "root", None) or "")
        gv.toasted.connect(self.tv.toasted.emit)
        gv.layout().itemAt(0).widget().hide()          # its own toolbar: the page's comparison in its title row
        gv.imageShown.connect(self._shown)
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
        v.addLayout(body, 1)
        self.page, self.gv = page, gv

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

    def _card(self, role: str) -> QFrame:
        """A filter's card: its thumbnail, its name and eye, Hue (the rail of the colours, a swatch, the degrees),
        Intensity, and the gain the balance measured."""
        u = self._ui
        spec = self._spec()
        name = hue_params(spec).get(role)
        card = QFrame()
        card.setObjectName("combcard")
        h = QHBoxLayout(card)
        h.setContentsMargins(8, 8, 8, 8)
        h.setSpacing(8)
        th = _Clickable()
        tl = QVBoxLayout(th)
        tl.setContentsMargins(0, 0, 0, 0)
        img = QLabel()
        img.setObjectName("thumb")
        img.setFixedSize(QSize(THUMB_W, THUMB_W * 2 // 3))
        img.setAlignment(Qt.AlignCenter)
        tl.addWidget(img)
        tl.addStretch(1)
        th.setCursor(Qt.PointingHandCursor)
        th.setToolTip(_("{filter} alone in the image (a click again: every filter)").format(filter=_port(role)))
        th.clicked.connect(lambda r=role: self.set_solo(None if self.solo == r else r))
        h.addWidget(th, 0, Qt.AlignTop)
        col = QVBoxLayout()
        col.setSpacing(3)
        head = QHBoxLayout()
        nm = label(f"<b>{_port(role)}</b>")
        nm.setTextFormat(Qt.RichText)
        head.addWidget(nm)
        head.addStretch(1)
        eye = QPushButton()
        eye.setObjectName("eye")
        eye.setCheckable(True)
        eye.setChecked(True)
        eye.setFlat(True)
        set_icon(eye, "eye", 16)
        eye.setToolTip(_("Takes {filter} out of the preview's mix, to see what it brings (the view only: never "
                         "applied)").format(filter=_port(role)))
        eye.toggled.connect(lambda on, r=role: self.set_eye(r, on))
        head.addWidget(eye)
        col.addLayout(head)
        lw = max(label(_("Hue")).sizeHint().width(), label(_("Intensity")).sizeHint().width()) + 4
        if name is not None:
            q = (spec.get("parameters") or {}).get(name, {})
            row = QHBoxLayout()
            row.setSpacing(6)
            hl = label(_("Hue"), "dim")
            hl.setFixedWidth(lw)
            row.addWidget(hl)
            sl = QSlider(Qt.Horizontal)
            sl.setObjectName("hue")
            sl.setRange(int(q.get("min", 0)), int(q.get("max", 360)))
            sl.setStyleSheet(f"QSlider::groove:horizontal {{ height: 8px; border-radius: 4px; background: {HUE_RAIL}; }}"
                             "QSlider::handle:horizontal { background: #f2f2f2; border: 1px solid #202020; width: 10px;"
                             " margin: -4px 0; border-radius: 5px; }"
                             # the whole rail of colours whatever the handle's place (the window's style filled the
                             # part left of the handle: O III's 0-202° showed one blue)
                             "QSlider::sub-page:horizontal, QSlider::add-page:horizontal { background: transparent; }")
            sl.setToolTip(_(q.get("help") or "") or _("The colour of the filter on the colour wheel"))
            self._wire_slider(sl, lambda i, r=role, **k: self.set_hue(r, float(i), **k))
            row.addWidget(sl, 1)
            sw = QLabel()
            sw.setFixedSize(14, 14)
            row.addWidget(sw)
            deg = label("", "small")
            deg.setMinimumWidth(deg.fontMetrics().horizontalAdvance("360°") + 2)
            row.addWidget(deg)
            col.addLayout(row)
            u[f"hue_{role}"], u[f"swatch_{role}"], u[f"deg_{role}"] = sl, sw, deg
        row = QHBoxLayout()
        row.setSpacing(6)
        il = label(_("Intensity"), "dim")
        il.setFixedWidth(lw)
        row.addWidget(il)
        it = QSlider(Qt.Horizontal)
        it.setObjectName("intensity")
        it.setRange(0, int(round(INTENSITY_MAX / INTENSITY_STEP)))
        it.setToolTip(_("The filter's weight after the balance (1: as balanced). Lower a faint filter whose noise the "
                        "balance raised."))
        self._wire_slider(it, lambda i, r=role, **k: self.set_intensity(r, i * INTENSITY_STEP, **k))
        row.addWidget(it, 1)
        iv = label("", "small")
        iv.setMinimumWidth(iv.fontMetrics().horizontalAdvance("3.00") + 2)
        row.addWidget(iv)
        col.addLayout(row)
        gain = label("", "small")
        gain.setStyleSheet("color: #8b95a5;")
        col.addWidget(gain)
        h.addLayout(col, 1)
        u[f"thumb_{role}"], u[f"eye_{role}"] = img, eye
        u[f"int_{role}"], u[f"intv_{role}"], u[f"gain_{role}"] = it, iv, gain
        return card

    def _wire_slider(self, sl: QSlider, set_value):
        """A slider's moves go to the draft as it is dragged (the preview follows), one step of Undo when it is let
        go; a key or the wheel, one step each."""
        sl.sliderPressed.connect(self.begin_drag)
        sl.sliderReleased.connect(self.end_drag)
        sl.valueChanged.connect(lambda i, s=sl: None if self._syncing else self._mark() or set_value(
            i, undo=not s.isSliderDown()))

    def _mark(self):
        self._t_edit = time.perf_counter()

    def set_eye(self, role: str, on: bool):
        if on:
            self.off.discard(role)
        else:
            self.off.add(role)
        self._mark()
        self._changed()

    def set_solo(self, role):
        self.solo = role
        self._mark()
        self._changed()

    def set_compare(self, mode: str):
        self.cmp = mode
        self._present()
        self._sync()

    # ------------------------------------------------------------ the controls as the draft is
    def _status(self) -> str:
        n = len(self.changes())
        if n:
            return _("Draft · 1 change") if n == 1 else _("Draft · {n} changes").format(n=n)
        st = self.s.state(self.p, self._stage()) if self._stage() else ""
        if st == "stale":
            return _("Applied · not computed yet")
        return _("Applied settings")

    def gain_text(self, role: str) -> tuple[str, str]:
        """(the line under a filter, its tooltip): the gain its balance measured and why."""
        notes = self.notes
        if role not in notes:
            refs = {s.get("reference") for s in (self._spec() or {}).get("steps") or [] if s.get("reference")}
            if role in refs and self._bal_now() is not None:
                return _("reference"), _("The other filters are brought to this one's scale")
            return "", ""
        a, why = notes[role]
        reason = _(REASONS.get(why, why))
        about = _(REASON_TIPS.get(why, ""))
        run = self._run_gain(role)
        st = self.s.state(self.p, self._stage()) if self._stage() else ""
        # the run's value only for the settings it was computed with (the block up to date, the balance not in the
        # draft); else the reduced copy's, an estimate, said so (the reviewer read ×7.40 before the run, ×7.35 after)
        if st == "done" and "balance" not in (self._draft().vals if self._draft() else {}) and run is not None:
            tip = about
            if abs(run - a) > GAIN_APART * max(run, 1e-12):
                tip += "\n" + _("The run measured ×{run} on the whole image; the preview ×{copy} on its reduced "
                                "copy").format(run=number(run, 2), copy=number(a, 2))
            return f"×{number(run, 2)} · {reason}", tip
        tip = about + "\n" + _("An estimate on the preview's reduced copy: the run measures it on the whole image")
        return _("≈ ×{gain}, from the preview · {why}").format(gain=number(a, 2), why=reason), tip

    def _run_gain(self, role: str) -> float | None:
        """The gain the applied run gave a filter (its scale in drz_meta.json, the weight taken out)."""
        ap = self.applied_img if isinstance(self.applied_img, dict) else None
        if ap is None or role not in (ap.get("tops") or {}):
            return None
        t = float(ap["tops"][role])
        w = float((self.applied().get("weights") or {}).get(role, 1.0))
        return 1.0 / (t * w) if t > 0 and w > 0 else None

    @property
    def applied_img(self):
        a = self.applied_cache
        return a if a is not None and self.p is not None and a.get("pid") == self.p.pid else None

    applied_cache = None

    def _sync(self):
        """The page's controls as the draft is."""
        if self.p is None or not self._ui:
            return
        self._syncing = True
        try:
            u = self._ui
            draft = self.has_draft()
            ch = self.draft_change() if draft else None
            u["bar"].show_draft([self._status()] + ([t for _k, t in self.changes()] if ch else []))
            u["state"].setStyleSheet(f"color: {WARN};" if draft else "color: #8b95a5;")
            why = _("No draft: change a setting first")
            for k in ("reset", "variant", "apply"):
                u[k].setEnabled(draft)
            u["apply"].setToolTip(_("The draft becomes the block's settings (Ctrl+Z takes it back)") if draft else why)
            u["variant"].setToolTip(_("The draft on a variant of the workflow: what is computed stays as it is, the "
                                      "variant is computed by Run") if draft else why)
            u["reset"].setToolTip(_("Back to the applied settings: the draft goes") if draft else why)
            u["title"].setText(self.heading())
            stale = self._stage() and self.s.state(self.p, self._stage()) == "stale" and not draft
            u["tag"].setText(_("Previous settings") if stale else "")
            for mode, _t, _tip in COMPARE:
                b = u.get("cmp_" + mode)
                if b is not None:
                    b.setChecked(self.cmp == mode)
            if "palette" in u:
                pal = u["palette"]
                pal.setCurrentIndex(max(0, pal.findData(self.palette_now())))
            names = hue_params(self._spec() or {})
            for role in self.roles():
                if f"hue_{role}" in u:
                    h = float(self.value(names[role]))
                    u[f"hue_{role}"].setValue(int(round(h)))
                    c = hue_colour(h)
                    u[f"swatch_{role}"].setStyleSheet(f"background: {c.name()}; border: 1px solid #202020; "
                                                      "border-radius: 3px;")
                    u[f"deg_{role}"].setText(f"{h:.0f}°")
                if f"int_{role}" in u:
                    w = self.weight(role)
                    u[f"int_{role}"].setValue(int(round(w / INTENSITY_STEP)))
                    u[f"intv_{role}"].setText(number(w, 2))
                    e = u[f"eye_{role}"]
                    e.setChecked(role not in self.off)
                    set_icon(e, "eye" if role not in self.off else "eye_off", 16)
                    t, tip = self.gain_text(role)
                    u[f"gain_{role}"].setText(t)
                    u[f"gain_{role}"].setToolTip(tip)
                    thumb = (self._bal_now() or {}).get("thumbs", {}).get(role)
                    if thumb is not None:
                        pm = QPixmap.fromImage(_qimage(thumb)).scaled(THUMB_W, THUMB_W * 2 // 3, Qt.KeepAspectRatio,
                                                                      Qt.SmoothTransformation)
                        u[f"thumb_{role}"].setPixmap(pm)
                    u[f"thumb_{role}"].setStyleSheet("border: 2px solid #5ea1ff;" if self.solo == role else "")
            bal = self.value("balance")
            for key in BALANCES:
                if "balance_" + key in u:
                    _lit(u["balance_" + key], bal == key)
            st = self.value("stars")
            for key in STARS:
                if "stars_" + key in u:
                    _lit(u["stars_" + key], st == key)
            self._update_cover()
        except RuntimeError:
            pass
        finally:
            self._syncing = False

    def _update_cover(self):
        """Under the image: why there is nothing to show (an input not computed, an input to wire, hues the stage
        refuses); a block never computed says "Not computed yet" with its Run to here, as every block's page."""
        u = self._ui
        if "cover" not in u:
            return
        st = self.s.state(self.p, self._stage()) if self._stage() else ""
        never = st not in ("done", "stale")
        u["cover_head"].setText(_("Not computed yet") if never else "")
        u["cover_head"].setVisible(never)
        u["cover_text"].setText(self.error)
        u["cover_text"].setVisible(bool(self.error))
        box = u["cover_runs"]
        if never and box.count() == 0 and hasattr(self.tv, "_run_button"):
            try:
                box.addStretch(1)
                box.addWidget(self.tv._run_button(self.p, "fusion"))
                box.addStretch(1)
            except Exception:  # noqa: BLE001 - no Run button here (a stand-in window): the words alone
                pass
        u["cover"].setVisible(never or bool(self.error))

    # ------------------------------------------------------------ the preview
    def _sig(self):
        """What the reduced copy is read from: the inputs wired (their workflow, image, channel) and their results."""
        st = self._stage()
        keys = []
        for i in self._wired():
            q = self.s.view(i["pid"])
            try:
                k = self.s.stage_key(q, self.s.all_stages(q)[-1]) if q is not None else None
            except Exception:  # noqa: BLE001 - a key not readable now: the copy is read again when it is
                k = None
            keys.append((i["role"], i["pid"], i.get("image", "output"), i.get("channel", ""), k))
        return (self.p.pid, st, tuple(keys))

    def _bal_now(self):
        if self.small is None or self.p is None or self.small.get("pid") != self.p.pid:
            return None
        return self.bal.get((self.p.pid, self.small["sig"], self._balance_used()))

    def _balance_used(self) -> str:
        from ..stages.combine import Combination
        st = self._stage()
        if not st:
            return "sky"
        p = type("P", (), {"balance": self.values().get("balance", "sky")})()
        return Combination.balance_of(p, self._wired())

    def missing(self) -> list[str]:
        have = {i["role"] for i in self._wired()}
        return [r for r in self.roles() if r not in have]

    def preview(self):
        """The draft's preview (or the applied settings', with the eyes): one computation at a time, the last edit
        only; the previous image stays until the next is ready."""
        if self.p is None:
            return
        miss = self.missing()
        if miss:
            self.error = _("Waiting: {names} to wire").format(names=", ".join(_port(r) for r in miss))
            self._sync()
            return
        self.gen += 1
        req = self._request()
        if self.busy:
            self._want = req
            return
        self._start(req)

    def _request(self) -> dict:
        st, spec = self._stage(), self._spec()
        vals = self.values()
        return {"gen": self.gen, "pid": self.p.pid, "stage": st, "spec": copy.deepcopy(spec),
                "stored": self._stored(), "values": vals, "balance": self._balance_used(),
                "off": set(self.off), "solo": self.solo, "sig": self._sig(), "project": self.s.p,
                "render": self._render_settings(), "result": self._result_key()}

    def _render_settings(self) -> dict | None:
        """The settings of the workflow's layer of Post (``layer_settings``): the preview as Post shows that layer."""
        try:
            return layer_settings(self.s.post_layers(), self.p.pid) if hasattr(self.s, "post_layers") else None
        except Exception:  # noqa: BLE001 - Post's layers not readable now: the default render
            return None

    def _result_key(self):
        st = self._stage()
        info = self.s.state_info(self.p, st) if hasattr(self.s, "state_info") and st else None
        return (self.p.pid, getattr(info, "key", None), getattr(info, "previous_key", None),
                self.s.state(self.p, st) if st else "", repr(self._render_settings()))

    def _start(self, req: dict):
        self.busy = True
        self.computed += 1
        small = self.small if self.small is not None and self.small.get("sig") == req["sig"] else None
        bal = self.bal.get((req["pid"], req["sig"], req["balance"])) if small is not None else None
        want_applied = self.applied_cache is None or self.applied_cache.get("key") != req["result"]

        def work():
            out = {"req": req}
            sm = small
            try:
                t0 = time.perf_counter()
                if sm is None:
                    sm = {**_reduced_for(req["project"], req["pid"], req["stage"], req["spec"]),
                          "pid": req["pid"], "sig": req["sig"]}
                    out["small"] = sm
                b = bal
                if b is None:
                    b = balance(sm, req["stage"], req["spec"], req["balance"])
                    out["bal"] = b
                cls = stage_class(req["stage"])
                p = params_of(cls, req["spec"], req["stored"], req["values"])
                rgb = mixed(sm, b, req["stage"], req["spec"], p, req["off"], req["solo"])
                out["result"] = output_render(rgb, req["render"])
                out["ms"] = 1000 * (time.perf_counter() - t0)
            except Exception as e:  # noqa: BLE001 - said over the image, the editor goes on
                out["error"] = str(e)
            if want_applied:          # the block's own result: shown even when its inputs cannot be read now
                try:
                    out["applied"] = {"pid": req["pid"], "key": req["result"],
                                      **(_applied_for(req["project"], req["pid"], req["stage"],
                                                      sm["k"] if sm is not None else None, req["render"]) or
                                         {"img": None, "tops": {}})}
                except Exception:  # noqa: BLE001 - no result to read: Applied greyed
                    pass
            try:
                self._done.emit(req["gen"], out)
            except RuntimeError:                       # the window closed meanwhile: nobody to tell
                pass
        # a Python thread, not Qt's pool: PyTorch (the Output's render) never returned in a thread of the pool
        threading.Thread(target=work, name="combination preview", daemon=True).start()

    def _arrived(self, gen: int, out: dict):
        self.busy = False
        req = out["req"]
        if "small" in out:
            self.small = out["small"]
        if "bal" in out:
            self.bal = {k: v for k, v in self.bal.items() if k[0] == req["pid"] and k[1] == req["sig"]}
            self.bal[(req["pid"], req["sig"], req["balance"])] = out["bal"]
        if "applied" in out:
            self.applied_cache = out["applied"]
        mine = self.p is not None and req["pid"] == self.p.pid
        if "error" in out:
            if mine:
                self.error = _(out["error"])
                if "applied" in out:
                    self._present()             # the block's own result, though its inputs cannot be read now
        elif mine:
            self.error = ""
            b = out.get("bal") or self.bal.get((req["pid"], req["sig"], req["balance"]))
            self.notes = dict((b or {}).get("notes") or {})
            if self._want is None:             # the latest edit: shown (an older one is never painted)
                self.painted += 1
                self.result, self.result_ms = out["result"], out["ms"]
                self.result_key = ("combination-preview", req["pid"], gen)
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
        drafting = self.has_draft() or self.off or self.solo is not None
        pv = ViewImage(self.result, _("Draft preview") if self.has_draft() else _("Preview"), key=self.result_key,
                       stretched=True, units="norm", white=1.0, camera=False, neutral=False) \
            if self.result is not None else None
        ap = self.applied_img
        av = ViewImage(ap["img"], _("Applied result") if self.s.state(self.p, self._stage()) == "done"
                       else _("Previous settings"), key=("combination-applied", pid, ap["key"]), stretched=True,
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

    def _shown(self, key):
        """An image of the editor on screen: the time from the edit that asked for it (``timings``)."""
        if self._t_edit is not None and isinstance(key, tuple) and key and key[0] == "combination-preview":
            self.timings.append(time.perf_counter() - self._t_edit)
            self._t_edit = None


def _reduced_for(project, pid: int, stage: str, spec: dict) -> dict:
    """The reduced inputs of a combination: the inputs its next run reads (their results now), else the ones its
    applied (or previous) result read."""
    from .. import api
    tried = []
    try:
        keys = api._keys(project)
        ins = keys.overrides(keys.project.workflow(pid), stage).get("inputs") or []
        tried.append(ins)
    except Exception:  # noqa: BLE001 - the inputs as the run would take them are not known: the result's
        pass
    for which in ("current", "previous"):
        try:
            out = api.stage_outputs(project, pid, stage, which=which)
        except Exception:  # noqa: BLE001 - no result to read
            out = None
        if out is not None:
            tried.append(out.params.get("inputs") or [])
    err = None
    for ins in tried:
        if not ins:
            continue
        try:
            return reduced_inputs(ins, spec)
        except (ValueError, OSError) as e:
            err = e
    raise ValueError(str(err) if err else _("The inputs are not computed: run the workflows the combination reads"))


def _applied_for(project, pid: int, stage: str, k: int, settings: dict | None = None) -> dict | None:
    from .. import api
    for which in ("current", "previous"):
        try:
            out = api.stage_outputs(project, pid, stage, which=which)
        except Exception:  # noqa: BLE001 - no result to read
            out = None
        if out is not None:
            return applied_small(out.dir, k, settings)
    return None
