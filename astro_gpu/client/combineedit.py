"""The combination block (S-H-O, H-O and the other recipes of stages/combine.py) tuned in Inspect as a draft
(``CombineEditor``), built as the Gradient block's editor (gradientedit.py) is, on the base every block of the
combination family's editor shares (blockeditor.DraftEditor: the draft, the page, the preview's thread; Combine layers'
editor, layersedit.py, is the other).

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
balance measured and why; Balance, On sky or On signal. The combination of the stars alone has only its Stars setting
(Neutral, Palette). The gains of a preview are the reduced copy's: when the draft keeps the balance, the applied run's
are shown (its drz_meta.json), the copy's said in the tooltip when they differ.
"""
from __future__ import annotations

import colorsys
import copy
import math
import time
from pathlib import Path

import numpy as np
from PySide6.QtCore import QEvent, QObject, QRunnable, QSize, Qt
from PySide6.QtGui import QColor, QImage, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
)

from .blockeditor import COMPARE, PANEL_W, DoubleClickDefault, DraftEditor  # noqa: F401 - (the family's, kept importable here)
from .blockeditor import Clickable as _Clickable  # noqa: F401
from .blockeditor import PressToRelease as _PressToRelease  # noqa: F401
from .blockeditor import choice_card as _choice_card
from .blockeditor import lit as _lit
from .gradientedit import PREVIEW_PIXELS
from .i18n import _, number
from .icons import set_icon
from .widgets import label

#: a filter's thumbnail: its width (px)
THUMB_W = 84
#: Intensity: its range and its step (1 = as balanced)
INTENSITY_MAX, INTENSITY_STEP = 10.0, 0.01             # (the tester's recipe: S II 14 times fainter than H-alpha;
#                                                        a hundredth: the fine setting near 1 the testers asked for)
#: Intensity's rail: even by ratio (the brush size's kind of law), 0.1 to INTENSITY_MAX, 1 at its middle, 0 (the
#: filter off) at its left end; INTENSITY_POSITIONS positions: about half a percent of the value a position near 1 (a
#: linear 0-10 rail took a 28 px drag from 1.00 to 2.95)
INTENSITY_LOW, INTENSITY_POSITIONS = 0.1, 1000


def intensity_at(u: float) -> float:
    """An intensity at a place of its rail (``u`` 0 to 1): 0 at the left end, else INTENSITY_LOW times the ratio to
    INTENSITY_MAX to the power u, rounded to the hundredth."""
    if u <= 0:
        return 0.0
    return round(INTENSITY_LOW * (INTENSITY_MAX / INTENSITY_LOW) ** min(float(u), 1.0), 2)


def intensity_u(w: float) -> float:
    """The place of an intensity on its rail (``intensity_at``'s inverse; 0 for 0)."""
    w = float(w)
    if w <= 0:
        return 0.0
    return min(1.0, max(0.0, math.log(max(w, INTENSITY_LOW) / INTENSITY_LOW) / math.log(INTENSITY_MAX / INTENSITY_LOW)))


#: the starting palettes: (key, name, the filters they colour, {filter: hue}); a palette is offered when the recipe's
#: hued filters are exactly its own. Hubble: the S-H-O hues measured on the Hubble teams' images (stages/combine.py,
#: HUBBLE_HUES, the S-H-O recipe's defaults); H-O-O: H-alpha red, O III cyan.
PALETTES = (("hubble", "Hubble (S-H-O)", ("SII", "Ha", "OIII"), None),
            ("hoo", "H-O-O", ("Ha", "OIII"), {"Ha": 0.0, "OIII": 180.0}))
BALANCES = ("sky", "signal")
#: the balance is the combination's one setting its run measures: what its change asks
BALANCE_RUN = "Needs a run: the combination is computed again by Run"
BALANCE_NAME = {"sky": "On sky", "signal": "On signal"}
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


def balance(small: dict, stage: str, spec: dict, how: str, maps: dict | None = None) -> dict:
    """The balanced channels of the reduced inputs, their skies, scales and each filter's gain and why, the thumbnails
    of the balanced filters. ``maps``: the balance as the run makes it (client/postfusion.balance_maps: the run's own,
    or measured once on the whole inputs), applied to the reduced copy (a map is linear: the block means of the
    balanced filters are the map of the block means): the run's gains, the same as Post's. Without it (the whole
    inputs cannot be read), the stage's ``steps`` on the reduced copy, an estimate."""
    if maps is not None:
        from .postfusion import apply_maps
        chans = apply_maps(maps["maps"], small["chans"])
        for c in chans.values():
            c[~small["valid"]] = 0
        notes = dict(maps["gains"])
        skies = {r: (0.0, float((maps.get("noise") or {}).get(r, 0.0))) for r in chans}
        tops = {r: 1.0 / notes[r][0] if r in notes and notes[r][0] > 0 else 1.0 for r in chans}
        return {"chans": chans, "skies": skies, "tops": tops, "notes": notes, "noise": dict(maps.get("noise") or {}),
                "from": maps.get("from", "run"), "thumbs": {r: thumbnail(c, small["valid"]) for r, c in chans.items()}}
    cls = stage_class(stage)
    chans = {r: c.copy() for r, c in small["chans"].items()}
    tops = {r: 1.0 for r in chans}
    notes: dict = {}
    skies = cls().steps(_Quiet(), spec, small["inputs"], chans, small["valid"], tops, how, notes)
    return {"chans": chans, "skies": skies, "tops": tops, "notes": notes, "from": "copy",
            "noise": {r: float(n) for r, (_lv, n) in skies.items()},
            "thumbs": {r: thumbnail(c, small["valid"]) for r, c in chans.items()}}


def run_balance(small: dict, stage: str, spec: dict, how: str, result_dir=None) -> dict:
    """``balance`` with the run's maps (client/postfusion.balance_maps) when the whole inputs can be read."""
    from .postfusion import balance_maps
    try:
        maps = balance_maps(stage, spec, small["inputs"], how, result_dir)
    except (KeyError, OSError, ValueError):            # the whole inputs not readable: the reduced copy's estimate
        maps = None
    return balance(small, stage, spec, how, maps)


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
    sliders), else the default ones (render_simple's preview)."""
    from ..render.simple import SimpleRenderEngine, SimpleSettings
    eng = SimpleRenderEngine.from_array(rgb, device="cpu", linked=True, sample=1)
    s = SimpleSettings(**settings) if settings else SimpleSettings()
    return np.asarray(eng.render(s, 1.0).final.numpy(), np.float32)


def layer_settings(layers: list[dict], pid: int) -> dict | None:
    """The rendering settings Post gives the workflow's own layer (its Output image, else its image recombined with
    its stars), as Post's render worker computes them (render/sliders.settings_at); None without a layer."""
    from ..render.sliders import settings_at
    mine = [x for x in layers or [] if x.get("kind") == "workflow" and x.get("pid") == pid
            and x.get("part") in (None, "combined", "starless")]
    if not mine:
        return None
    lay = min(mine, key=lambda x: (None, "combined", "starless").index(x.get("part")))
    try:
        return settings_at("simple", lay.get("base"), lay.get("sliders"), has_denoise=False, origin="palette",
                           stretch=lay.get("stretch")).model_dump()
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


class FineDrag(QObject):
    """Shift held while a slider is dragged: its handle follows the pointer ``FACTOR`` times slower (the fine step of
    the reference editors' sliders), from where it was when Shift was pressed; the arrow keys move it by one step (one
    degree on a hue rail). Without Shift the drag is the slider's own."""

    FACTOR = 10

    def __init__(self, slider):
        super().__init__(slider)
        self.sl = slider
        self._from = None                                # (pointer x, value) where the fine drag started
        slider.installEventFilter(self)

    def _per_px(self) -> float:
        from PySide6.QtWidgets import QStyle, QStyleOptionSlider
        opt = QStyleOptionSlider()
        self.sl.initStyleOption(opt)
        st = self.sl.style()
        rail = st.subControlRect(QStyle.CC_Slider, opt, QStyle.SC_SliderGroove, self.sl)
        handle = st.subControlRect(QStyle.CC_Slider, opt, QStyle.SC_SliderHandle, self.sl)
        span = max(1, rail.width() - handle.width())
        return (self.sl.maximum() - self.sl.minimum()) / span

    def eventFilter(self, obj, e):  # noqa: N802 - Qt's name
        t = e.type()
        if obj is self.sl and t == QEvent.MouseMove and e.buttons() & Qt.LeftButton and self.sl.isSliderDown():
            if not e.modifiers() & Qt.ShiftModifier:
                self._from = None
                return False
            x = e.position().x()
            if self._from is None:
                self._from = (x, self.sl.value())
            x0, v0 = self._from
            self.sl.setValue(int(round(v0 + (x - x0) * self._per_px() / self.FACTOR)))
            return True
        if obj is self.sl and t == QEvent.MouseButtonRelease:
            self._from = None
        return False



class CombineEditor(DraftEditor):
    """The combination block of the workflow shown, tuned as a draft in Inspect's centre (blockeditor.DraftEditor, the
    family's editor: the draft, the page, the preview's thread). ``set_hue``, ``set_intensity``, ``set_palette``,
    ``set_eye``, ``set_solo``; the balance and the stars by ``set_draft``."""

    KEY = "fusion"
    SIDE_PREF = "combine_settings_side"

    def __init__(self, tune):
        super().__init__(tune)
        self.off: set = set()                  # the filters taken out of the preview (the eyes): the view only
        self.solo = None                       # the filter shown alone (its thumbnail clicked)
        self.small = None                      # the reduced inputs {"pid", "sig", "chans", "valid", "k", ...}
        self.bal: dict = {}                    # (pid, sig, balance) -> the balanced channels (``balance``)
        self.notes: dict = {}                  # the gains of the balance shown, {filter: (gain, why)}
        tuned = getattr(self.s, "postTuned", None)
        if tuned is not None:                  # a setting changed in Post's master layer: the block's draft goes
            tuned.connect(self._post_tuned)

    # ------------------------------------------------------------ the block shown
    def wants(self, node, p) -> bool:
        return str(self.tv.s.sel_step) == "fusion" and self._stage(p) is not None and self._spec(p) is not None

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
        """The combination's settings: the block's, its master layer's hues and intensities in their place when Post
        shows it by one (client/postfusion.py: one source, the layer)."""
        p = p or self.p
        st = self._stage(p)
        if st and hasattr(self.s, "combine_stored"):
            try:
                return dict(self.s.combine_stored(p.pid))
            except Exception:  # noqa: BLE001 - the model cannot say now: the block's own
                pass
        try:
            return dict(self.s.p.workflow(p.pid).params.get(st) or {}) if st else {}
        except (KeyError, AttributeError):
            return {}

    def master(self, p=None) -> bool:
        """Post shows this combination by its master layer: its hues and intensities are that layer's (they never
        ask for a run), its balance the block's."""
        p = p or self.p
        try:
            return p is not None and hasattr(self.s, "master_layer") and self.s.master_layer(p.pid) is not None
        except Exception:  # noqa: BLE001 - the model cannot say now: the block's settings
            return False

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

    def shape(self, p) -> tuple:
        return (self.stars_mode(p), tuple(self.roles(p)), self._stage(p))

    def state_of(self, p=None) -> str:
        p = p or self.p
        st = self._stage(p) if p is not None else None
        return self.s.state(p, st) if st else ""

    def norm(self, key: str, value):
        if key == "weights":
            return norm_weights(value)
        return value if key in ("balance", "stars") else round(float(value), 2)

    # ------------------------------------------------------------ the draft
    def weight(self, role: str) -> float:
        return float((self.value("weights") or {}).get(role, 1.0))

    def _post_tuned(self, pid: int):
        """The combination's settings changed in Post (its master layer: one source with this editor): a draft of the
        block goes, said in the status line, and the page shows the settings as they are now."""
        if self.drafts.pop(pid, None) is not None:
            self.tv.toasted.emit(_("Draft dropped: changed in Post"))
            self.historyChanged.emit()
        if self.p is not None and self.p.pid == pid and self._page_ok():
            self.applied_cache = None
            if self.page_shown():
                self._changed()
            else:                              # the page hidden (Post shown): its preview is made when it shows again,
                self.result = None             # never in the background, beside Post's own render (the same processor)

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

    def parts(self, pid: int | None = None) -> list[tuple[str, str, str]]:
        """What the draft changes, one (name, the setting, its change "old → new") per setting (a filter's hue, its
        intensity, the balance, the stars)."""
        q = self._view_of(pid)
        d = self.drafts.get(q.pid) if q is not None else None
        if not d:
            return []
        names = {v: r for r, v in hue_params(self._spec(q) or {}).items()}
        ap = self.applied(q)
        out = []
        for k, v in d.vals.items():
            if k in names:
                out.append((k, _("{filter} hue").format(filter=_port(names[k])),
                            f"{ap.get(k, 0):.0f}° → {float(v):.0f}°"))
            elif k == "weights":
                for r in sorted(set(v) | set(ap.get("weights") or {})):
                    a, b = float((ap.get("weights") or {}).get(r, 1.0)), float(v.get(r, 1.0))
                    if abs(a - b) > 1e-6:
                        out.append((f"weights.{r}", _("{filter} intensity").format(filter=_port(r)),
                                    f"{number(a, 2)} → {number(b, 2)}"))
            elif k == "balance":
                out.append((k, _("Balance"), f"{_(BALANCE_NAME.get(ap.get(k), ''))} → {_(BALANCE_NAME.get(v, v))}"))
            elif k == "stars":
                out.append((k, _("Stars"), f"{_(STARS_NAME.get(ap.get(k), ''))} → {_(STARS_NAME.get(v, v))}"))
        return out

    def draft_change(self, pid: int | None = None) -> dict | None:
        """The draft as one change of the model (kind "combine": graphmodel.GraphEditing._combine_steps), with its
        words (named as every step of the family: blockdraft.owned); None without a draft."""
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
        what, short = self.step_words(q)
        return {"kind": "combine", "values": vals, "short": short, "what": what}

    def title(self, q=None) -> str:
        """The block's name as the graph shows it (graphcore rules.title: its recipe's title, numbered among the
        blocks of its type, or the name the user gave), the same in the path bar, the page's title, the draft's words,
        the leave question and Post's panel."""
        from .graphnode import combine_kind, combine_title
        q = q or self.p
        if q is None:
            return _("Combination")
        try:
            from ..graphcore import rules
            return rules.title(self.s._g, self.s._combine_block(q), tr=_)
        except Exception:  # noqa: BLE001 - no graph here (a stand-in model): its type's name
            return combine_title(combine_kind(q), q)

    def heading(self, q=None) -> str:
        """The page's title: the block's name (``title``); without a graph, the workflow's name, its recipe's title in
        it the block's own (graphcore combos.retitled: a block whose recipe became H-O is never titled S-H-O)."""
        from ..graphcore import combos
        q = q or self.p
        if q is not None and getattr(self.s, "_g", None) is not None:
            return self.title(q)
        title = self.title(q)
        name = combos.retitled(getattr(q, "name", "") or "", combos.short((self._spec(q) or {}).get("name") or title))
        if not name:
            return title
        return name if combos.title_stem(name) else f"{title}  ·  {name}"

    def _needs_run(self, q, ch) -> bool:
        """Whether an applied change asks for a run: the block's settings do; a master layer's hues and intensities
        are the layer's (client/postfusion.py), nothing to compute."""
        if not self.master(q):
            return True
        from .postfusion import split_change
        return bool(split_change(self._spec(q) or {}, dict(ch["values"]))[1])

    def cost_text(self, pid: int | None = None) -> str:
        """What Apply computes again, as the model says it ("Computes again … about N min"); for a master layer's hues
        and intensities alone, nothing (they are the layer's)."""
        q = self._view_of(pid)
        ch = self.draft_change(q.pid) if q is not None else None
        if ch is None:
            return ""
        key = (q.pid, repr(sorted((self.drafts[q.pid].vals or {}).items(), key=str)))
        if self._cost is not None and self._cost[0] == key:
            return self._cost[1]
        vals = dict(ch["values"])
        if self.master(q):
            from .postfusion import split_change
            _layer, vals = split_change(self._spec(q) or {}, vals)
        text = _("Post's layer only: nothing is computed again")
        if vals:
            try:
                if hasattr(self.s, "change_cost"):
                    c = self.s.change_cost(dict(ch, values=vals), [q])
                    text = c.text() if (c.lost or c.blocks) else _(BALANCE_RUN)
                else:
                    text = _(BALANCE_RUN)
            except Exception:  # noqa: BLE001 - the cost is a hint: Apply works without it
                text = ""
        self._cost = (key, text)
        return text

    # ------------------------------------------------------------ the page
    def _new_block(self):
        self.off, self.solo = set(), None
        self.result = None
        self.notes = {}

    def _measurements(self, dv):
        """Each filter's gain and why, its balanced sky noise, where the gains come from."""
        b = self._bal_now() or {}
        self.title_of_group(dv, "Measurements")
        dv.addWidget(label(_("Balance: {how}").format(how=_(BALANCE_NAME.get(self._balance_used(), ""))), "small"))
        notes, noise = b.get("notes") or {}, b.get("noise") or {}
        for role in self.roles():
            g = notes.get(role)
            text = f"{_port(role)}  ·  " + (f"×{number(g[0], 2)} · {_(REASONS.get(g[1], g[1]))}" if g else
                                             _(REASONS["reference"]))
            if role in noise:
                text += "  ·  " + _("sky noise {v}").format(v=_digits(noise[role], 3))
            dv.addWidget(label(text, "small"))
        src = {"run": "The gains of the run (its result)",
               "measured": "The gains measured on the whole image as the run measures them",
               "copy": "An estimate on the preview's reduced copy"}.get(b.get("from"))
        if src:
            dv.addWidget(label(_(src), "small", wrap=True))

    def _drawer_sig(self) -> tuple:
        b = self._bal_now() or {}
        return (repr(sorted((b.get("notes") or {}).items())), repr(sorted((b.get("noise") or {}).items())),
                b.get("from"), self._balance_used())

    def _fill_column(self, cv):
        """Palette, the filters' cards, Balance; the stars' combination: Stars only."""
        u = self._ui
        if self.stars_mode():
            cv.addWidget(label(_("Stars"), "h2"))
            for key in STARS:
                c = _choice_card(_(STARS_NAME[key]), _(STARS_LINE[key]))
                c.clicked.connect(lambda k=key: self.set_draft({"stars": k}))
                cv.addWidget(c)
                u["stars_" + key] = c
            return
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
            FineDrag(sl)
            DoubleClickDefault(sl, lambda r=role, d=float(q.get("default", 0.0)): self.set_hue(r, d))   # its recipe's
            sl.setToolTip(sl.toolTip() + "\n" + _("Shift while dragging: ten times finer · arrow keys: one degree"))
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
        it.setRange(0, INTENSITY_POSITIONS)             # (even by ratio: intensity_at)
        it.setToolTip(_("The filter's weight after the balance (1: as balanced). Lower a faint filter whose noise the "
                        "balance raised.") + "\n" + _("Shift while dragging: ten times finer · arrow keys: one step"))
        self._wire_slider(it, lambda i, r=role, **k: self.set_intensity(r, intensity_at(i / INTENSITY_POSITIONS), **k))
        FineDrag(it)
        DoubleClickDefault(it, lambda r=role: self.set_intensity(r, 1.0))      # (1: as balanced)
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

    def _view_only(self) -> bool:
        return bool(self.off) or self.solo is not None

    def applied_label(self) -> str:
        return _("Applied settings · reduced copy") if self.master() else super().applied_label()

    # ------------------------------------------------------------ the controls as the draft is
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
        got = (self._bal_now() or {}).get("from")
        if got == "run":                                 # the run's own gains: Post and the run show the same
            return f"×{number(a, 2)} · {reason}", about
        if got == "measured":                            # measured on the whole inputs as the run does, once
            return f"×{number(a, 2)} · {reason}", about + "\n" + _(
                "Measured on the whole image as the run measures it (the run's own result does not hold it)")
        tip = about + "\n" + _("An estimate on the preview's reduced copy: the run measures it on the whole image")
        return _("≈ ×{gain}, from the preview · {why}").format(gain=number(a, 2), why=reason), tip

    def _sync_controls(self):
        u = self._ui
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
                u[f"int_{role}"].setValue(int(round(intensity_u(w) * INTENSITY_POSITIONS)))
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

    def missing_text(self) -> str:
        miss = self.missing()
        return _("Waiting: {names} to wire").format(names=", ".join(_port(r) for r in miss)) if miss else ""

    def _request(self) -> dict:
        st, spec = self._stage(), self._spec()
        vals = self.values()
        ap = self.applied()
        from ..stages.combine import Combination
        return {"gen": self.gen, "pid": self.p.pid, "stage": st, "spec": copy.deepcopy(spec),
                "stored": self._stored(), "values": vals, "balance": self._balance_used(),
                "applied": ap, "applied_balance": Combination.balance_of(
                    type("P", (), {"balance": ap.get("balance", "sky")})(), self._wired()),
                "master": self.master(), "off": set(self.off), "solo": self.solo, "sig": self._sig(),
                "project": self.s.p, "render": self._render_settings(), "result": self._result_key(),
                "small": self.small if self.small is not None and self.small.get("sig") == self._sig() else None,
                "want_applied": None}

    def _render_settings(self) -> dict | None:
        """The settings of the workflow's layer of Post (``layer_settings``): the preview as Post shows that layer. A
        combination Post shows by its master layer (client/postfusion.py) is stretched there by one factor per pixel,
        the colour-preserving stretch (render/simple.SimpleRenderEngine.colour_factor): the same here, Colours at 1,
        the stretch that factor is."""
        try:
            layers = self.s.post_layers() if hasattr(self.s, "post_layers") else []
            got = layer_settings(layers, self.p.pid) if hasattr(self.s, "post_layers") else None
        except Exception:  # noqa: BLE001 - Post's layers not readable now: the default render
            layers, got = [], None
        fusion_of = getattr(self.s, "fusion_of", None)
        try:
            master = fusion_of is not None and any(
                x.get("part") == "combined" and x.get("starless") == "output" and x.get("pid") == self.p.pid
                for x in layers or []) and fusion_of(self.p.pid)[0] is not None
        except Exception:  # noqa: BLE001 - the combination not readable now: as its Output
            master = False
        if master:
            from ..render.simple import SimpleSettings
            got = dict(got or SimpleSettings().model_dump(), keep_colours=1.0)
        return got

    def _result_key(self):
        st = self._stage()
        info = self.s.state_info(self.p, st) if hasattr(self.s, "state_info") and st else None
        return (self.p.pid, getattr(info, "key", None), getattr(info, "previous_key", None),
                self.s.state(self.p, st) if st else "", repr(self._render_settings()),
                repr(sorted(self.applied().items(), key=str)) if self.master() else "",
                tuple(sorted(self.off)), self.solo if self.master() else None)

    def _start(self, req: dict):
        req["bal"] = self.bal.get((req["pid"], req["sig"], req["balance"])) if req.get("small") is not None else None
        req["want_applied"] = self.applied_cache is None or self.applied_cache.get("key") != req["result"]
        req["result_dir"] = _result_dir_for(req["project"], req["pid"], req["stage"])
        super()._start(req)

    def _work(self, req: dict) -> dict:
        """The preview, in the editor's thread: the reduced inputs (read once per set of inputs), their balance (once
        per balance), the stage's own mix, the Output's render; the applied picture when it changed."""
        out = {"req": req}
        sm = req.get("small")
        result_dir = req["result_dir"]
        cls = stage_class(req["stage"])
        b = None
        try:
            t0 = time.perf_counter()
            if sm is None:
                sm = {**_reduced_for(req["project"], req["pid"], req["stage"], req["spec"]),
                      "pid": req["pid"], "sig": req["sig"]}
                out["small"] = sm
            b = req.get("bal")
            if b is None:
                b = run_balance(sm, req["stage"], req["spec"], req["balance"], result_dir)
                out["bal"] = b
            p = params_of(cls, req["spec"], req["stored"], req["values"])
            rgb = mixed(sm, b, req["stage"], req["spec"], p, req["off"], req["solo"])
            out["result"] = output_render(rgb, req["render"])
            out["ms"] = 1000 * (time.perf_counter() - t0)
        except Exception as e:  # noqa: BLE001 - said over the image, the editor goes on
            out["error"] = str(e)
        if req["want_applied"] and req.get("master") and sm is not None and "result" in out:
            # a master layer: its applied picture is the layer's, made as the draft's (the run's balance, the
            # layer's hues and intensities): the same picture at the same settings
            try:
                ab = b if req["balance"] == req["applied_balance"] else run_balance(
                    sm, req["stage"], req["spec"], req["applied_balance"], result_dir)
                pa = params_of(cls, req["spec"], req["stored"], req["applied"])
                out["applied"] = {"pid": req["pid"], "key": req["result"], "tops": {},
                                  "img": output_render(mixed(sm, ab, req["stage"], req["spec"], pa, req["off"],
                                                             req["solo"]), req["render"])}
            except Exception:  # noqa: BLE001 - not made: Applied greyed
                pass
        elif req["want_applied"]:        # the block's own result: shown even when its inputs cannot be read now
            try:
                out["applied"] = {"pid": req["pid"], "key": req["result"],
                                  **(_applied_for(req["project"], req["pid"], req["stage"],
                                                  sm["k"] if sm is not None else None, req["render"]) or
                                     {"img": None, "tops": {}})}
            except Exception:  # noqa: BLE001 - no result to read: Applied greyed
                pass
        return out

    def _absorb(self, req: dict, out: dict):
        """The reduced inputs and the balance a preview read, kept for the next; the gains it shows."""
        if "small" in out:
            self.small = out["small"]
        if "bal" in out:
            self.bal = {k: v for k, v in self.bal.items() if k[0] == req["pid"] and k[1] == req["sig"]}
            self.bal[(req["pid"], req["sig"], req["balance"])] = out["bal"]
        if "error" not in out and self.p is not None and req["pid"] == self.p.pid:
            b = out.get("bal") or self.bal.get((req["pid"], req["sig"], req["balance"]))
            self.notes = dict((b or {}).get("notes") or {})


def _digits(v: float, n: int) -> str:
    """A number with ``n`` significant digits, written the interface's way (i18n.number: a decimal comma in
    French)."""
    import math as _m
    if not v or not _m.isfinite(v):
        return number(float(v or 0.0), 0)
    return number(float(v), max(0, n - 1 - int(_m.floor(_m.log10(abs(v))))))


def _result_dir_for(project, pid: int, stage: str):
    """The folder of the combination's result (the current one, else the one an Outdated block had), None without."""
    from .. import api
    for which in ("current", "previous"):
        try:
            out = api.stage_outputs(project, pid, stage, which=which)
        except Exception:  # noqa: BLE001 - no result to read
            out = None
        if out is not None:
            return out.dir
    return None


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
