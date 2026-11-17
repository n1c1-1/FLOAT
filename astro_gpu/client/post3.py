"""The Post of the client: the image at the centre; on the right the layers, as in the reference image editor, and the properties of
the selected layer or of its mask.

The layers panel follows the reference image editor's model. A pixel layer (a workflow's output, one of its Stars block's images,
or a stage's image sent from Inspect) shows its own filters as children, indented under it: they apply to it alone and
move with it. A filter that applies to everything below is a row of its own, like a layer, above what it affects. Drag a
filter onto a layer to nest it, between two rows to make it apply to everything below (an indented line: it joins that
layer's children); the line, the frame and a short label say where it goes. Read from the bottom up, the panel is the
order of the composition.

A row: eye, the ▾ of a layer's children, thumbnail, mask thumbnail, name, blend mode and opacity. A click on the
thumbnail shows the layer's settings; a click on the mask thumbnail shows its mask, made of parts as in the reference photo editor, with a
red overlay on the image.

Nothing here renders. Every image comes from the render worker (services/render.py): a draft while a slider or the
brush moves (an eighth of the image on the CPU, a quarter on a GPU), on release the exact quarter first on the CPU then
the half resolution (the screen), and the visible region at full resolution when zoomed in. A newer request replaces an
older one. The layers are saved in the project (the active one of ``project.post["versions"]``) through the model, which the graph edits
too (the Stars block's images sent to Post, the Stars and background blocks and their settings): a change made there
comes back here (``postChanged``).

A workflow with the Stars block comes in as a Stars and background layer (its starless image and its stars, each
stretched, the stars put back in Screen: render/stars.py), with the star sliders; the others as their Output block's
image. Its stars are a group (#93): the real stars and, with the Synthetic stars block, the synthetic ones beside them,
each with its eye; the synthetic stars are added and removed here too, their block with them in the graph.

    page = PostPage3(model)              # model: the ProjectModel
    page.goWorkflow                      # "Go to graph →"
    page.view.goResults(pid)             # '"X": not computed yet →' and "Show workflow →": Inspect on that Output
    page.export(parent)                  # File > Export image… and the Export… button
    page.add_image_layer(name, source, key)   # Inspect: a stage's image as a layer

Classes: ``PostPage3`` holds the ``PostView`` (the viewer, the layers panel of ``LayerRow`` rows, the properties with
their ``ContSlider`` sliders and the Curves editor); ``PostSource`` and ``BeforeSource`` feed the viewer.
"""
from __future__ import annotations

import copy
import json
import math
import re

import numpy as np
from PySide6.QtCore import QEvent, QPoint, QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QImage, QKeySequence, QPainter, QPen, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFrame,
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QPushButton,
    QScrollArea,
    QSlider,
    QSplitter,
    QStackedWidget,
    QStyle,
    QStyleOptionFrame,
    QStyleOptionSlider,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..core.project2 import active_version
from ..render.sliders import ARCSINH_WHY, STRENGTH_STEP
from ..render.sliders import STARS as STARS_GROUP
from ..render.sliders import group_title
from . import prefs
from .combineedit import HUE_RAIL, INTENSITY_MAX, INTENSITY_STEP, FineDrag
from .curveseditor import CurvesEditor, FunctionGraph, caption
from .i18n import _, language, log_detail, maybe, number, plain_error, plural
from .icons import draw as draw_icon
from .icons import icon_only, pixmap, set_icon
from .imageview import ZOOM_PRESETS, ZOOM_STEP
from .layout import Dock, Sections
from .model import POST_PART_NAMES
from .postengine import (
    visible_signature,
    LAYER_STRETCHES,
    STRETCH_KEY,
    STRETCH_NAMES,
    layer_stretch,
    set_layer_stretch,
    BLENDS,
    DIRECTIONS,
    FAMILIES,
    FILTERS,
    IMAGE_SLIDERS,
    MASK_SOURCES,
    MAX_LEVEL,
    MODES,
    OLD_NAMES,
    OPS,
    PART_SETTINGS,
    ROLE_SUB,
    SUB_NAMES,
    SUB_ROWS,
    SideLayer,
    READS_IMAGE,
    fusion_roles,
    is_fusion,
    role_sub,
    is_side,
    mask_selection,
    selection_side,
    sub_name,
    sub_role,
    subs_of,
    SUBS,
    THUMB,
    children,
    crop_within,
    has_synth,
    is_group,
    is_pixel,
    is_synth,
    load_layers,
    move,
    new_filter_layer,
    new_id,
    new_image_layer,
    new_mask,
    new_part,
    new_workflow_layer,
    normalize,
    rows,
    save_layers,
    snapshot,
    tree,
    stars_side,
    with_synth_rows,
)
from .services.render import CPU_DRAFT_LEVEL, DRAFT_LEVEL, HALF_LEVEL, RenderWorker, Request, default_device_pref
from .viewer import Viewer, to_qimage
from .widgets import ACCENT, WARN, ElidedLabel, clear, label, single_choice, wheel_scrolls

NL = "\n"                                                   # a line break in a tooltip

FAMILY_COL = {"Stretch": "#c9a35e", "Contrast": "#d98a4e", "Color": "#c46fa8", "Detail": "#5ea1ff",
              "Noise": "#62c2b1", "Combine": "#e06c6c"}
TW, TH = THUMB                                           # thumbnail size
ROW_H, SMALL_W, SMALL_H = 24, 30, 20                     # a row of the list and its thumbnails (#88)
#: the mask brush, px across: its largest is the image's long side, this at least (``PostView.brush_max``)
BRUSH_MIN, BRUSH_MAX = 4.0, 1200.0


def brush_pos(px: float, top: float = BRUSH_MAX) -> float:
    """The slider's position of a brush size, ``top`` its largest: even by ratio (the same share of the rail doubles
    the size, wherever it is), so that the small sizes stay easy to set however large the image is."""
    return float(np.log(np.clip(px, BRUSH_MIN, top) / BRUSH_MIN) / np.log(top / BRUSH_MIN))


def brush_px(t: float, top: float = BRUSH_MAX) -> int:
    return int(round(BRUSH_MIN * (top / BRUSH_MIN) ** t))
COPY = " (copy)"                                         # a piece of the layer names, translated where shown
#: what the Stars block's images are, under their name in the properties
PART_NOTES = {"starless": "The Stars/background block's starless image, stretched like the whole image.",
              "stars": "The Stars/background block's stars alone, on black: in Screen mode above the background.",
              "combined": "The starless image and the stars, recombined. The same settings as its block in the "
                          "graph."}
#: a combination's Hue slider: the rail of the colours of Inspect's editor, its handle
HUE_QSS = (f"QSlider::groove:horizontal {{ height: 8px; border-radius: 4px; background: {HUE_RAIL}; }}"
           "QSlider::handle:horizontal { background: #f2f2f2; border: 1px solid #202020; width: 10px; margin: -4px 0; "
           "border-radius: 5px; } QSlider::sub-page:horizontal, QSlider::add-page:horizontal { background: transparent; }")
REGION_MAX = 12_000_000                                  # pixels of the region rendered at full resolution
REGION_WAIT_MS = 150                                     # after the view stops moving
INDENT = 24                                              # a child row's indent (px)
HOLD_MS = 600                                            # the exact image within this: the half one is not shown
QUARTER_SCALE = 0.26                                     # zoomed out to this, the quarter-resolution image fills the screen
TOGGLE_W = 16                                            # the ▾ / ▸ of a layer's children


def tr_name(n: str, image: bool = False) -> str:
    """A layer's name as shown: the pieces the program built it from (a filter's name, the Stars block's images,
    "(copy)") translated; the name of a workflow, or a name typed by the user, as it is (the graph shows it so). ``image``: an
    image layer, named by Inspect in English (``tune.TuneView._send``): every piece the catalogue has is translated
    (a file name stays as it is)."""
    k = 0
    while n.endswith(COPY):
        n, k = n[:-len(COPY)], k + 1
    if image:
        return _pieces(n, maybe) + _(COPY) * k
    known = {f["name"] for f in FILTERS.values()} | set(OLD_NAMES) | set(POST_PART_NAMES.values()) | set(SUB_NAMES.values())

    def piece(b):                                 # "Combine layers 2": a numbered copy of a Split image's name
        m = re.fullmatch(r"(.+) (\d+)", b)
        if b not in known and m and m.group(1) in POST_PART_NAMES.values():
            return _(m.group(1)) + " " + m.group(2)
        return _(b) if b in known else b
    return " · ".join(piece(b) for b in n.split(" · ")) + _(COPY) * k


def _sub_selected(what) -> bool:
    """A selection kind that is one of a Combine layers layer's images (its stars, its starless image, the real
    stars, a combination's filter)."""
    return what in SUB_ROWS or sub_role(what) is not None


def layer_name(lay: dict) -> str:
    """The name of a layer as shown (``tr_name``)."""
    from . import names                               # default names follow the interface's language
    return names.shown(tr_name(lay["name"], lay.get("kind") == "image"))


def setting_name(lay: dict, field: str, key: str) -> str:
    """A setting of a layer as its panel names it, in the language of the screen: a filter's setting by its label, a
    slider of a workflow's layer or of an image layer by its slider's name (the Undo / Redo words named it by its key:
    "Star strength" for the slider the panel calls Strength, in English whatever the language)."""
    if field == "params" and lay.get("kind") == "filter":
        label = next((q["label"] for q in FILTERS.get(lay.get("filter"), {}).get("params", ()) if q["key"] == key), None)
    elif lay.get("kind") == "image":
        label = next((name for k, name, _ends in IMAGE_SLIDERS if k == key), None)
    elif field == "sliders" and key == STRETCH_KEY:      # the Strength of the layer's Arcsinh stretch
        label = "Stretch strength"
    else:
        from ..render.sliders import BY_KEY
        label = BY_KEY[key].name if key in BY_KEY else None
    return _(label if label is not None else str(key).replace("_", " ").capitalize())


def step_words(old: list[dict], new: list[dict]) -> str:
    """What changed between two states of the layers, in a few words ("Saturation of "Filter" changed"), for the
    status line of an Undo / Redo. ``old``: the earlier state."""
    a, b = {x["id"]: x for x in old}, {x["id"]: x for x in new}
    kind = lambda lay: _("Filter") if lay.get("kind") == "filter" else _("Layer")  # noqa: E731
    # a layer added or deleted with its own filters (a duplicate, a delete) is named, not its first filter
    for these, other, word in ((b, a, "{kind} “{name}” added"), (a, b, "{kind} “{name}” deleted")):
        moved = [lay for i, lay in these.items() if i not in other]
        if moved:
            lay = next((x for x in moved if x.get("kind") != "filter"), moved[0])
            return _(word).format(kind=kind(lay), name=layer_name(lay))
    if [x["id"] for x in old] != [x["id"] for x in new]:
        return _("Layers reordered")
    for i, lay in b.items():
        was = a[i]
        if was == lay:
            continue
        name = layer_name(lay)
        if was.get("name") != lay.get("name"):
            return _("“{old}” renamed “{name}”").format(old=layer_name(was), name=name)
        if was.get("visible") != lay.get("visible"):
            return (_("“{name}” shown") if lay.get("visible") else _("“{name}” hidden")).format(name=name)
        for sub, v in (lay.get("shown") or {}).items():             # the eye of one of a group's images
            if (was.get("shown") or {}).get(sub) != v:
                side = _(_side_name(lay, sub))
                return (_("“{name}” shown") if v else _("“{name}” hidden")).format(name=side)
        if was.get("tune") != lay.get("tune"):                    # a combination's own hues and intensities
            return _tune_words(was, lay)
        if layer_stretch(was) != layer_stretch(lay):               # Automatic / Arcsinh: named with its choice
            return _("Stretch of “{name}”: {choice}").format(name=name, choice=_(STRETCH_NAMES[layer_stretch(lay)]))
        for key, word in (("opacity", _("opacity")), ("mode", _("blend mode")), ("blend", _("blend mode"))):
            if was.get(key) != lay.get(key):
                return _("{what} of “{name}”").format(what=word.capitalize(), name=name)
        if was.get("mask") != lay.get("mask"):
            return mask_words(was.get("mask"), lay.get("mask"), name)
        sa, sb = was.get("side_masks") or {}, lay.get("side_masks") or {}
        for sub in sorted(set(sa) | set(sb)):
            if sa.get(sub) != sb.get(sub):
                return mask_words(sa.get(sub), sb.get(sub), _(_side_name(lay, sub)))
        for field in ("sliders", "params"):
            ka, kb = was.get(field) or {}, lay.get(field) or {}
            for k in list(kb) + [k for k in ka if k not in kb]:        # (a slider back at its middle has no key)
                if ka.get(k) != kb.get(k):
                    return _("{what} of “{name}”").format(what=setting_name(lay, field, k), name=name)
        return _("“{name}” changed").format(name=name)
    return _("Crop changed") if getattr(old, "crop", None) != getattr(new, "crop", None) else ""


def _side_name(lay: dict, sub: str) -> str:
    """The name of one of a Combine layers layer's images as its row says it ("Stars", "O III"), never its key."""
    role = sub_role(sub)
    if role is None:
        return SUB_NAMES.get(sub, sub)
    names = (lay.get("fusion") or {}).get("names") or {}
    if role in names:
        return names[role]
    from .graphnode import port_name
    return port_name(role)


def _tune_words(was: dict, lay: dict) -> str:
    """A combination's hue or intensity changed in its master layer: "O III hue", "Hα intensity"."""
    a, b = was.get("tune") or {}, lay.get("tune") or {}
    for key, word in (("hues", "{filter} hue"), ("weights", "{filter} intensity")):
        for role in sorted(set(a.get(key) or {}) | set(b.get(key) or {})):
            if (a.get(key) or {}).get(role) != (b.get(key) or {}).get(role):
                return _(word).format(filter=_side_name(lay, ROLE_SUB + role))
    return _("“{name}” changed").format(name=layer_name(lay))


def mask_words(old: dict | None, new: dict | None, name: str) -> str:
    """What changed in the mask of the layer named ``name`` (``step_words``): added, deleted, a part added or
    removed, a stroke, a part's setting, a setting of the whole mask."""
    if old is None:
        return _("Mask added to “{name}”").format(name=name)
    if new is None:
        return _("Mask of “{name}” deleted").format(name=name)
    pa, pb = {p["id"]: p for p in old["parts"]}, {p["id"]: p for p in new["parts"]}
    for i, p in pb.items():
        if i not in pa:
            return _("{part} added to the mask of “{name}”").format(part=_(MASK_SOURCES[p["src"]]), name=name)
    for i, p in pa.items():
        if i not in pb:
            return _("{part} removed from the mask of “{name}”").format(part=_(MASK_SOURCES[p["src"]]), name=name)
    for i, p in pb.items():
        if pa[i] != p:
            if pa[i].get("strokes") != p.get("strokes"):
                return _("Brush stroke on the mask of “{name}”").format(name=name)
            return _("{part} of the mask of “{name}”").format(part=_(MASK_SOURCES[p["src"]]), name=name)
    for key, word in (("feather", "Feather"), ("density", "Density"), ("invert", "Invert")):
        if old.get(key) != new.get(key):
            return _("{what} of the mask of “{name}”").format(what=_(word), name=name)
    return _("{what} of “{name}”").format(what=_("mask").capitalize(), name=name)


def _pieces(n: str, fn) -> str:
    """``fn`` applied to each piece of a name made of pieces joined by "·" (the separators kept as they are)."""
    parts = re.split(r"(\s*·\s*)", n)
    return "".join(p if i % 2 else fn(p) for i, p in enumerate(parts))


def dec(s: str) -> str:
    """A decimal comma in French."""
    return s.replace(".", ",") if language() == "fr" else s


def offset_text(x: float) -> str:
    """A rendering slider's value, as the reference photo editor shows it: its offset from the suggested setting (position 2), 0 there,
    with a true minus sign."""
    return "0" if round(x - 2, 1) == 0 else dec(f"{x - 2:+.1f}").replace("-", "−")


def strength_text(x: float) -> str:
    """The Strength of a layer's Arcsinh stretch, as ``offset_text`` to its step (two decimals)."""
    return "0" if round(x - 2, 2) == 0 else dec(f"{x - 2:+.2f}").replace("-", "−")


def with_unit(v: str, unit: str) -> str:
    if not unit:
        return v
    return v + ("" if unit == "°" or unit == "%" and language() != "fr" else " ") + unit


def pix(a: np.ndarray, w=TW, h=TH) -> QPixmap:
    """A pixmap of a uint8 image (h, w) or (h, w, 3), at the size of a thumbnail."""
    a = np.ascontiguousarray(a)
    if a.ndim == 2:
        a = np.repeat(a[..., None], 3, -1)
    if a.shape[:2] != (h, w):
        import cv2
        a = cv2.resize(a, (w, h), interpolation=cv2.INTER_AREA)
    return QPixmap.fromImage(to_qimage(a))


def decimals(step) -> int:
    """How many decimals a slider's step has (0.005: 3; 1 or none: 0)."""
    return next((d for d in range(9) if abs(step * 10 ** d - round(step * 10 ** d)) < 1e-9), 9) if step else 0


def fmt(v, step):
    if step is not None and step >= 1:
        return f"{v:.0f}"
    return dec(f"{v:.{max(2, decimals(step))}f}".rstrip("0").rstrip("."))   # (a step of 0.005 shows 0.205)


# ======================================================================== a continuous slider
def small(pm: QPixmap) -> QPixmap:
    """A thumbnail at the size of a row."""
    return pm.scaled(SMALL_W, SMALL_H, Qt.KeepAspectRatio, Qt.SmoothTransformation)


class _NoContext:
    """A ``with`` that does nothing (a model without ``from_post``)."""

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


#: the tips of a mask part's sliders whose short name leaves something out: (source, setting) → its one line
PART_TIPS = {
    ("stars", "grow"): "Each star's disc, in multiples of its radius: the brightest stars' wings reach further and may "
                       "need a larger Grow.",
    ("stars", "feather"): "Softens the edge of the stars' discs; the whole mask's feather adds to it.",
}


class Step(list):
    """A step of Post's Undo: the layers, and the crop of that moment. ``words``: what the step did, when the two
    states compared do not say it (a mask pasted is a mask added, or its parts changed)."""
    crop = None
    words = ""


class ContSlider(QWidget):
    """A slider that glides: `moved` while dragging (draft render), `done` when let go (full render). Double-click
    on the slider or on its name: back to the default (the reference photo editor's, issue #16).

    With a ``step`` it counts in steps: every value it takes is ``lo`` plus a whole number of steps, an arrow key moves
    it by one step (so a range with more values than the rail has pixels is still set exactly), and the value it is let
    go on is the one of its last move. Without one it counts in thousandths of its range."""
    moved = Signal(float)
    done = Signal(float)
    GLIDE = 1000                                          # the positions of a slider without a step

    def __init__(self, name, lo, hi, value, default, ends=None, unit="", step=None, show=None, parent=None):
        super().__init__(parent)
        self.lo, self.hi, self.default, self.unit, self.step, self.show = lo, hi, default, unit, step, show
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 2, 0, 4)
        v.setSpacing(0)
        top = QHBoxLayout()
        self.name = QLabel(name)
        self.name.setToolTip(_("Double-click: default value"))
        self.setToolTip(_("Double-click: default value"))
        self.val = label("", "small")
        top.addWidget(self.name)
        top.addStretch(1)
        top.addWidget(self.val)
        v.addLayout(top)
        row = QHBoxLayout()
        row.setSpacing(6)
        self.s = QSlider(Qt.Horizontal)
        # (a step that does not divide the range: the last whole step under ``hi`` is the end)
        self._n = max(1, math.floor((hi - lo) / step + 1e-6)) if step else self.GLIDE
        self.s.setRange(0, self._n)
        if ends:
            a, b = label(ends[0], "small"), label(ends[1], "small")
            row.addWidget(a)
            row.addWidget(self.s, 1)
            row.addWidget(b)
        else:
            row.addWidget(self.s, 1)
        v.addLayout(row)
        self.why = label("", "warn")
        self.why.setWordWrap(True)
        self.why.hide()
        v.addWidget(self.why)
        self.set_value(value)
        self.s.valueChanged.connect(self._changed)
        self.s.sliderReleased.connect(lambda: self.done.emit(self.value()))
        self.name.mouseDoubleClickEvent = lambda e: self.reset()
        self.s.installEventFilter(self)

    def reset(self):
        """Back to the default value (a double-click)."""
        if not self.s.isEnabled():
            return
        self.set_value(self.default)
        self.moved.emit(self.default)
        self.done.emit(self.default)

    def eventFilter(self, obj, e):                                   # noqa: N802 - Qt's name
        if obj is self.s and e.type() == QEvent.MouseButtonDblClick and e.button() == Qt.LeftButton:
            self.reset()
            return True
        if obj is self.s and e.type() == QEvent.MouseButtonPress and e.button() == Qt.LeftButton:
            self._to_click(e.position().x())                 # then Qt's own press takes the handle now under the mouse
        return super().eventFilter(obj, e)

    def _to_click(self, x: float):
        """A click on the rail: the slider goes where it is clicked (issue #72); on the handle, nothing. The rail and
        the handle are measured as the slider's own drag measures them (its style's): the handle comes under the
        pointer, and the drag that follows starts from the value of the click (measured otherwise, the first pixel of
        that drag moved the value by a notch)."""
        opt = QStyleOptionSlider()
        self.s.initStyleOption(opt)
        style = self.s.style()
        handle = style.subControlRect(QStyle.CC_Slider, opt, QStyle.SC_SliderHandle, self.s)
        if handle.left() <= x <= handle.right():
            return
        rail = style.subControlRect(QStyle.CC_Slider, opt, QStyle.SC_SliderGroove, self.s)
        span = max(1, rail.right() - handle.width() + 1 - rail.x())
        at = int(round(x - handle.width() / 2 - rail.x()))
        self.s.setSliderDown(True)                           # one drag: `done` when let go
        self.s.setValue(QStyle.sliderValueFromPosition(0, self._n, min(max(at, 0), span), span, opt.upsideDown))

    def value(self) -> float:
        if self.step:
            return round(self.lo + self.s.value() * self.step, decimals(self.step) + decimals(self.lo))
        return self.lo + (self.hi - self.lo) * self.s.value() / self._n

    def set_value(self, x):
        self.s.blockSignals(True)
        self.s.setValue(int(round((x - self.lo) / self.step if self.step else
                                  self._n * (x - self.lo) / ((self.hi - self.lo) or 1))))
        self.s.blockSignals(False)
        self._label()

    def _label(self):
        x = self.value()
        if self.show:
            self.val.setText(self.show(x))
        else:
            self.val.setText(with_unit(fmt(x, self.step), self.unit))
        changed = abs(x - self.default) > 1e-6 * max(1, abs(self.hi - self.lo))
        self.val.setStyleSheet(f"color: {WARN};" if changed else "")

    def _changed(self, _):
        self._label()
        self.moved.emit(self.value())
        if not self.s.isSliderDown():                       # keyboard or click on the groove
            self.done.emit(self.value())

    def set_available(self, ok, why=""):
        self.s.setEnabled(ok)
        self.name.setStyleSheet("" if ok else "color: #6b7280;")
        self.why.setText(why)
        self.why.setVisible(not ok and bool(why))


# ======================================================================== the layers panel
#: the sides' widths: by default, and the limits of the user's drag
PANEL_W, PANEL_MIN, PANEL_MAX = 400, 320, 760          # the settings, on the right
LAYERS_W, LAYERS_MIN, LAYERS_MAX = 300, 230, 520       # the layers, on the left


def layers_default(window_w: int) -> int:
    """The layers' width before the user sets one: a fifth of a wide window, LAYERS_W at least (the laptop's D6: 300 px
    on a 1920 px screen cut the names of a Combine layers layer's rows)."""
    return min(LAYERS_MAX, max(LAYERS_W, round(0.18 * window_w)))
SUB_MAX = 120                                          # a layer row's blend, at most (its name keeps the rest)
CENTRE_MIN = 480                                       # the image: its tools go into "…" rather than push the sides out
VERSION_W = ((120, 200), (95, 120))                    # the version list's least and most width, then narrowed (#146)


class EyeBox(QCheckBox):
    """A layer's visibility, drawn as an eye: a check box underneath (its clicks, keys and signals as any check box's)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(20, 20)
        self.setCursor(Qt.PointingHandCursor)

    def hitButton(self, pos):                             # noqa: N802 - Qt's name
        return self.rect().contains(pos)

    def paintEvent(self, e):                              # noqa: N802
        p = QPainter(self)
        on = self.isChecked()
        p.drawPixmap(2, 2, pixmap("eye" if on else "eye_off", 16, "normal" if on else "dim", self.devicePixelRatioF()))
        p.end()


class NameEdit(QLineEdit):
    """A name typed in place that shows its start when it is not typed in, cut at its end with "…" when it does not fit,
    as the reference image editors' lists (#146: a name set or typed left the cursor at its end, which kept the end in
    view, "oft background" for "Strong stars, soft background", after a click elsewhere and a resize too); while it is
    typed in it scrolls as any field, Enter and the focus gone show its start again."""

    def __init__(self, text: str = "", parent=None):
        super().__init__(text, parent)
        self.setCursorPosition(0)
        self.textChanged.connect(lambda _t: None if self.hasFocus() else self.setCursorPosition(0))
        self.editingFinished.connect(lambda: self.setCursorPosition(0))

    def focusOutEvent(self, e):                                      # noqa: N802 - Qt's name
        super().focusOutEvent(e)
        self.setCursorPosition(0)

    def _text_rect(self, opt: QStyleOptionFrame):
        """Where the field draws its text: its contents, less its text margins and the 2 px each side it keeps."""
        m = self.textMargins()
        r = self.style().subElementRect(QStyle.SE_LineEditContents, opt, self)
        return r.adjusted(m.left() + 2, m.top(), -m.right() - 2, -m.bottom())

    def shown(self) -> str:
        """The text as it is drawn: whole while typed in or when it fits, else its start and "…"."""
        opt = QStyleOptionFrame()
        self.initStyleOption(opt)
        if self.hasFocus():
            return self.text()
        return self.fontMetrics().elidedText(self.text(), Qt.ElideRight, max(self._text_rect(opt).width(), 1))

    def paintEvent(self, e):                                         # noqa: N802 - Qt's name
        shown = self.shown()
        if shown == self.text():
            super().paintEvent(e)
            return
        opt = QStyleOptionFrame()
        self.initStyleOption(opt)
        p = QPainter(self)
        self.style().drawPrimitive(QStyle.PE_PanelLineEdit, opt, p, self)          # as the field draws its own
        p.setPen(self.palette().color(self.foregroundRole()))
        p.drawText(self._text_rect(opt), int(Qt.AlignLeft | Qt.AlignVCenter), shown)
        p.end()


class LayerRow(QFrame):
    """A row of the layers panel. ``depth`` 1: a filter nested in the pixel layer above it (indented, with a guide
    line), or one of a Stars and background layer's two images (``kind`` "sub", issue #3); 2: a filter of one of those
    images; ``kids``: how many children a pixel layer has (its ▾ / ▸)."""

    def __init__(self, panel, lay, thumb, mthumb, sel_layer, sel_mask, depth=0, kids=0, collapsed=False, parent=None):
        super().__init__(parent)
        self.panel, self.lay, self.depth = panel, lay, depth
        self._press = None
        self._dragging = False
        self.name_edit = None                           # its name typed in place (Rename, a double-click)
        self.setObjectName("lrow")
        on = sel_layer or sel_mask
        self.setStyleSheet(f"QFrame#lrow {{ background: {'#2b3446' if on else 'transparent'}; border-radius: 4px; "
                           "border: 2px solid transparent; } QFrame#lrow:hover { background: #262b35; } "
                           f"QFrame#lrow[drop=\"true\"] {{ border: 2px solid {ACCENT}; background: #22314a; }} "
                           "QLabel { background: transparent; }")
        h = QHBoxLayout(self)
        self.setFixedHeight(ROW_H)                      # one short line each, whatever it holds (#88)
        h.setContentsMargins(4, 0, 6, 0)
        h.setSpacing(6)
        eye = EyeBox()
        eye.setChecked(lay["visible"])
        eye.setToolTip(_("Show / hide layer"))
        if lay["kind"] == "sub":
            eye.setToolTip(_("Show / hide the real stars: hidden, the synthetic stars show alone") if lay["sub"] == "real"
                           else _("Show / hide this image in its Combine layers layer"))
            eye.toggled.connect(lambda v: panel.set_shown(lay["id"], lay["sub"], v))
        else:
            eye.toggled.connect(lambda v: panel.set_visible(lay["id"], v))
        h.addWidget(eye)
        self.toggle = None
        if depth:                                       # a child: indented, a guide line down from its layer
            h.addSpacing(INDENT * (depth - 1))
            guide = QFrame()
            guide.setFixedWidth(2)
            guide.setStyleSheet("background: #4a5366;")
            h.addSpacing(TOGGLE_W // 2 - 1)
            h.addWidget(guide)
            h.addSpacing(INDENT - TOGGLE_W // 2 - 1)
        if kids:                                        # issue #21: down expanded, right collapsed; none without
            key = (lay["id"], lay["sub"]) if lay["kind"] == "sub" else lay["id"]
            self.toggle = QToolButton()
            icon_only(self.toggle, "chevron_right" if collapsed else "chevron_down", _("Show / hide what is under it"), 16, "dim")
            self.toggle.setAutoRaise(True)
            self.toggle.setFixedSize(TOGGLE_W, TOGGLE_W)
            self.toggle.setStyleSheet("QToolButton { border: none; padding: 0; }")
            self.toggle.clicked.connect(lambda: panel.toggle_children(key))
            h.addWidget(self.toggle)
        else:
            h.addSpacing(TOGGLE_W)
        self.t = QLabel()
        self.t.setFixedSize(SMALL_W + 2, SMALL_H + 2)
        self.t.setPixmap(small(thumb))
        self.t.setStyleSheet(f"border: 1px solid {ACCENT if sel_layer else 'transparent'};")
        if is_pixel(lay) or lay["kind"] == "sub" or is_synth(lay):       # its image, larger, under the mouse
            self.t.installEventFilter(self)
        else:
            self.t.setToolTip(_("Layer settings"))
        h.addWidget(self.t)
        self.m = None
        if mthumb is not None:
            self.m = QLabel()
            self.m.setPixmap(small(mthumb))
            self.m.setFixedSize(SMALL_W + 2, SMALL_H + 2)
            self.m.setStyleSheet(f"border: 1px solid {ACCENT if sel_mask else '#3a4150'};")
            self.m.setToolTip(_("Layer mask: white shows the layer, black hides it") + NL +
                              _("Drag it onto another layer to copy it there"))
            h.addWidget(self.m)
        v = QVBoxLayout()
        v.setSpacing(0)
        self.n = ElidedLabel(layer_name(lay), mode=Qt.ElideRight)     # a long name never widens the row
        self.n.setMinimumWidth(60)
        self.n.setStyleSheet("font-weight: 600;" + ("" if lay["visible"] else " color: #8b93a1;"))
        v.addWidget(self.n)
        self._names = v
        # its blend, shortened before the name is (the night test pass: "Cou…" for Curves beside "on all below · Normal")
        self.sub = ElidedLabel(panel.sub_text(lay), "small", mode=Qt.ElideRight)
        self.sub.setMaximumWidth(SUB_MAX)
        h.addLayout(v, 1)
        h.addWidget(self.sub)
        # at its usual blend (Normal, 100 %) a row says nothing more
        usual = f"{_('Normal')} · {with_unit('100', '%')}"
        self.sub.setVisible(lay["kind"] != "sub" and self.sub.full_text() != usual)
        if lay["kind"] == "sub" and lay["sub"] == "real":
            self.setToolTip(_("The stars of the Stars/background block, under the synthetic stars: they show where the "
                              "synthetic stars' mask is black"))
        elif lay["kind"] == "sub" and sub_role(lay["sub"]) is not None:
            self.setToolTip(_("One filter of the combination: its own filters, mask and spot healing act on it "
                              "before the filters are mixed"))
        elif lay["kind"] == "sub":
            self.setToolTip(_("The filters nested here apply to this image only, before the stars are put back on "
                              "the starless image"))
        elif is_synth(lay):
            self.setToolTip(_("The stars of the Synthetic stars block: where the mask is white they replace the real "
                              "stars, where it is painted black the real stars show"))
        elif lay.get("sub"):
            grp = panel.layer(lay.get("target"))
            self.setToolTip(_('Applies to "{name}" only').format(
                name=_(sub_name(grp, lay["sub"]) if grp is not None else SUB_NAMES.get(lay["sub"], lay["sub"]))))
        elif depth:
            tgt = panel.layer(lay.get("target"))
            self.setToolTip(_('Applies to "{name}" only').format(name=layer_name(tgt)) if tgt else "")
        elif lay["kind"] == "filter":
            self.setToolTip(_("On all below"))

    def edit_name(self):
        """Its name typed in place, in its row, as the reference image editors rename a layer (a window over the work
        stopped everything): Enter or a click elsewhere keeps it, Escape gives it up; an empty or unchanged name changes
        nothing (``rename``), a new one is one step of Undo."""
        if self.name_edit is not None:
            return
        ed = self.name_edit = NameEdit(layer_name(self.lay), self)
        ed.setStyleSheet("QLineEdit { font-weight: 600; background: transparent; border: none; padding: 0; "
                         f"border-bottom: 1px solid {ACCENT}; border-radius: 0; }}")
        self._names.insertWidget(self._names.indexOf(self.n), ed)
        self.n.hide()
        # the note under way while it is typed: beside it the field had about 50 px, only the name's end shown (#153)
        self._sub_shown = not self.sub.isHidden()
        self.sub.hide()
        ed.show()                                        # (a child made after its row was shown)
        ed.editingFinished.connect(lambda: self.end_name(True))
        ed.installEventFilter(self)
        QApplication.instance().installEventFilter(self)          # a click anywhere else keeps the name
        n = len(ed.text())
        ed.setSelection(n, -n)                           # all selected, the cursor at its start: the name's start shown
        ed.setFocus(Qt.OtherFocusReason)

    def end_name(self, keep: bool):
        """The name typed kept (``keep``) or given up, the row's name shown again."""
        ed, self.name_edit = self.name_edit, None
        if ed is None:                                   # (Enter, then the focus it loses: once)
            return
        QApplication.instance().removeEventFilter(self)
        text = ed.text().strip()
        ed.blockSignals(True)
        ed.hide()
        ed.deleteLater()
        self.n.show()
        self.sub.setVisible(self._sub_shown)
        if keep:
            lay = self.panel.layer(self.lay["id"])       # (the layers may have been built again meanwhile)
            if lay is not None:
                self.panel.rename(lay, text)

    def eventFilter(self, obj, e):                                   # noqa: N802 - Qt's name
        ed = self.name_edit
        if ed is not None and obj is ed and e.type() in (QEvent.ShortcutOverride, QEvent.KeyPress) \
                and e.key() == Qt.Key_Escape:
            e.accept()                                   # Escape is the field's (it leaves Post otherwise)
            if e.type() == QEvent.KeyPress:
                self.end_name(False)
            return True
        if ed is not None and e.type() == QEvent.MouseButtonPress and isinstance(obj, QWidget) and obj is not ed \
                and not ed.rect().contains(ed.mapFromGlobal(e.globalPosition().toPoint())):
            QTimer.singleShot(0, lambda: self.end_name(True))     # after the click, which may rebuild the rows
        if obj is self.t and e.type() == QEvent.Enter:
            self.panel.show_big(self.lay, self.t)
        elif obj is self.t and e.type() == QEvent.Leave:
            self.panel.hide_big()
        return super().eventFilter(obj, e)

    def set_drop(self, on: bool, text: str = ""):
        """The frame of a drop that nests the dragged filter in this layer; ``text`` (what the drop does) shows in place of
        the row's mode and opacity while it lasts."""
        if bool(self.property("drop")) == on and (not on or self.sub.full_text() == text):
            return
        self.setProperty("drop", on)
        if on:
            self.sub.setVisible(True)                   # a compact row says what the drop does too
        self.style().unpolish(self)
        self.style().polish(self)
        self.sub.setText(text if on else self.panel.sub_text(self.lay))
        self.sub.setStyleSheet(f"color: {ACCENT}; font-weight: 600;" if on else "")

    def _target(self, pos):
        return "mask" if self.m is not None and self.m.geometry().contains(pos) else "layer"

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._press = e.position().toPoint()
            self._dragging = False

    def contextMenuEvent(self, e):                                # noqa: N802 - Qt's name
        # a sub-layer too: its filters (#120); opened on the context menu event (at the release, see app.py
        # context_menus_on_release), never at the press: the release then chose the entry under the pointer
        self.panel.layer_menu(self.lay, e.globalPos())
        e.accept()

    def mouseMoveEvent(self, e):
        if self._press is not None and self.m is not None and self.m.geometry().contains(self._press):
            # its mask's thumbnail dragged: the mask goes on the row it is let go on, the rows stay in place
            if self._dragging or (e.position().toPoint() - self._press).manhattanLength() > 8:
                self._dragging = True
                self.panel.mask_drag_move(self, e.globalPosition().toPoint())
            return
        if self.lay["kind"] == "sub" or is_synth(self.lay):     # a sub-layer, the synthetic stars stay in their group
            return
        if self._press is not None and (e.position().toPoint() - self._press).manhattanLength() > 8:
            if not self._dragging:
                self._dragging = True
                self.panel.drag_start(self)
            self.panel.drag_move(self, e.globalPosition().toPoint())

    def mouseReleaseEvent(self, e):
        if self._press is None:
            return
        press, self._press = self._press, None
        if self._dragging and self.m is not None and self.m.geometry().contains(press):
            self._dragging = False
            self.panel.mask_drag_end(self, e.globalPosition().toPoint())
        elif self._dragging:
            self._dragging = False
            self.panel.drag_end(self, e.globalPosition().toPoint())
        else:                                            # a click: select (after the event, the rows are rebuilt)
            what = self._target(press) if self.lay["kind"] != "sub" else \
                "mask_" + self.lay["sub"] if self.m is not None and self.m.geometry().contains(press) else self.lay["sub"]
            QTimer.singleShot(0, lambda: self.panel.select(self.lay["id"], what))

    def mouseDoubleClickEvent(self, e):
        if self.lay["kind"] != "sub":
            self.panel.rename(self.lay)


def _upsample(a: np.ndarray, k: int = 2) -> np.ndarray:
    return np.repeat(np.repeat(a, k, 0), k, 1)


class PostSource:
    """What the viewer shows: the composite (half resolution; the quarter, enlarged, from a draft that is newer), the
    region at full resolution when zoomed in (from the worker; the half image enlarged until it arrives), or the mask in
    grey. Everything is already rendered: the viewer asks in the GUI thread and nothing is computed here."""

    def __init__(self, view):
        self.view = view
        self.version = 0
        self.img1 = None                                  # uint8 (H/2, W/2, 3) and the request it answers
        self.seq1 = 0
        self.img2 = None                                  # the draft, (H/4, W/4, 3)
        self.seq2 = 0
        self.level2 = 2                                   # its resolution: 2 a quarter, 3 an eighth
        self.full = None                                  # (box, uint8 array) at full resolution
        self.canvas = None                                # (W, H)

    def size(self):
        return self.canvas or (2, 2)

    def _half(self) -> np.ndarray | None:
        """The image at half resolution to show: the newest of the half render and the (enlarged) draft."""
        if self.img2 is not None and self.seq2 > self.seq1:
            return _upsample(self.img2, 2 ** (self.level2 - 1))[:self.canvas[1] // 2, :self.canvas[0] // 2]
        return self.img1

    def region(self, level, x0, y0, x1, y1):
        v = self.view
        if v.show_mask and v.mask1 is not None:
            m = (np.clip(v.mask1, 0, 1) * 255).astype(np.uint8)
            if level == 0:
                m = _upsample(m)
            return np.repeat(m[y0:y1, x0:x1, None], 3, -1)
        a = self._half()
        if a is None:
            return np.zeros((y1 - y0, x1 - x0, 3), np.uint8)
        if level == 1:
            return a[y0:y1, x0:x1]
        b = _upsample(a[y0 // 2: -(-y1 // 2), x0 // 2: -(-x1 // 2)])
        out = b[y0 % 2: y0 % 2 + (y1 - y0), x0 % 2: x0 % 2 + (x1 - x0)]
        if self.full is not None:
            (bx0, by0, bx1, by1), arr = self.full
            ix0, iy0, ix1, iy1 = max(x0, bx0), max(y0, by0), min(x1, bx1), min(y1, by1)
            if ix1 > ix0 and iy1 > iy0:
                out = np.array(out)
                out[iy0 - y0:iy1 - y0, ix0 - x0:ix1 - x0] = arr[iy0 - by0:iy1 - by0, ix0 - bx0:ix1 - bx0]
        return out


class BeforeSource:
    """The before of the before/after: the bottom workflow layer, all sliders at the suggestion (rendered by the
    worker, at half resolution; enlarged when zoomed in)."""

    def __init__(self, view):
        self.view = view
        self.version = 0
        self.img1 = None

    def size(self):
        return self.view.src.size()

    def region(self, level, x0, y0, x1, y1):
        a = self.img1
        if a is None:
            return np.zeros((y1 - y0, x1 - x0, 3), np.uint8)
        if level == 1:
            return a[y0:y1, x0:x1]
        b = _upsample(a[y0 // 2: -(-y1 // 2), x0 // 2: -(-x1 // 2)])
        return b[y0 % 2: y0 % 2 + (y1 - y0), x0 % 2: x0 % 2 + (x1 - x0)]


# ======================================================================== the view
#: a step of Post's Undo that is a change of a workflow (its Output's channels): the workflows' Undo takes it back
class ModelStep:
    """A step of the workflows' Undo seen from a version's Undo list: the model's own step it stands for, so that the
    version takes it back only while it is the model's latest one (a step another version made is never reached)."""

    def __init__(self, step=None):
        self.step = step


MODEL_STEP = ModelStep()


class PostView(QWidget):
    is_post_page = True                                   # its changes go to its own Undo, not the graph's (2.1)
    goProcessing = Signal(int, str)
    goResults = Signal(int)
    panelChosen = Signal(str, str)                        # a side ("left", "right") shown another way by the user
    rendered = Signal(object)                             # a Result was shown (tests, the status line)
    undoChanged = Signal()                                # Post's Undo / Redo lists changed (the window's Edit menu follows)
    toasted = Signal(str)                                 # what a change did beyond Post (the window's status line)
    layerAsked = Signal(int)                              # Open in Post of a layer (the page lists it when not computed)
    openImages = Signal()                                 # images asked as layers
    runAsked = Signal()                                   # an image's kind changed: it is computed again

    def __init__(self, session, parent=None, *, worker: RenderWorker | None = None, persist: bool = True):
        super().__init__(parent)
        self.setObjectName("root")
        self.s = session
        self.persist = persist
        cpu = getattr(session, "device", None) == "cpu" or getattr(session, "gpu", True) is False
        self.worker = worker or RenderWorker(getattr(session, "p", None),
                                             device="cpu" if cpu else default_device_pref())
        self.layers: list[dict] = []
        self.sel = None                                 # (layer id, "layer" | "mask")
        self.part_sel = None                            # the mask part being edited (its id)
        self._mask_clip = None                          # the mask copied (Copy mask), for Paste mask
        self.heal_size = 80.0                           # Spot healing: the next spot's diameter (px)
        self._heal_stroke = None                        # the stroke being painted: (its points, the layers before)
        self._heal_hand: dict = {}                      # a version's source set by hand, while another is shown
        self._heal_src = None                           # Alt+click: the source set by hand (x, y), then its offset
        self._heal_off = None                           # (dx, dy) of the strokes since, as the reference image editor's Aligned
        self.show_mask = False
        self.show_tint = True                            # the red over the picture while a mask is edited
        self.crop = None                                 # [x0, y0, x1, y1] of the canvas, or None: a setting of the composition
        self._big = None                                 # the larger thumbnail under the mouse
        self.mask1 = None                               # the selected layer's mask at half resolution (float)
        self.undo, self.redo = [], []
        self._stacks: dict = {}                          # the other versions' (undo, redo)
        self._version = None                             # the id of the version shown
        self._mthumbs: dict[int, QPixmap] = {}
        self._lthumbs: dict[int, tuple] = {}            # layer id -> (signature, pixmap)
        self._overlay_img = None
        self._stroke_before = None
        self._last_pt = None
        self._notes: list[str] = []
        self._device = ""
        self._last_seq = {"draft": 0, "half": 0, "region": 0, "before": 0, "refine": 0}
        self._saved = ""
        self._clean = None                               # (version, layers, crop, the model's edits) as saved: ``_save``
        self.draft_ms: list[float] = []
        self.draft_level = CPU_DRAFT_LEVEL if cpu else DRAFT_LEVEL
        self._collapsed: set[int] = set()                # the pixel layers whose children are hidden in the panel
        self._half_due = False                           # (CPU) the exact quarter is shown, the half not rendered
        self._missing_images: set[str] = set()           # image layers' keys whose pixels the worker did not find
        self._drop = None                                # the drop the drag shows (see drop_at)
        self._restoring = False                          # a step of the workflows' Undo being undone/redone here
        self._hists: dict[int, dict] = {}                # filter id -> the histograms of its input (the last render)
        self.curves_editor: CurvesEditor | None = None   # the Curves panel shown, or the graph of a stretch
        self._curve_channel: dict[int, str] = {}         # filter id -> the channel its Curves panel showed last
        self.function_graph: FunctionGraph | None = None
        self.first_image = False
        self._live: dict[int, dict] = {}                 # a combination's settings being moved here, not yet its block's
        self._plain_why: dict[int, str] = {}             # a combination's layer that stays plain: why (its tooltip)
        self.sync_paths(initial=True)
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        # the layers on the left, the settings of the selection on the right, as the usual image editors: their widths
        # are the user's (their inner edge dragged, kept in the settings). Each is docked, folded to a strip of icons (a
        # click opens it over the image), automatic (the strip opens under the mouse) or hidden, as the graph's palette:
        # the menu at its top right corner says which, remembered per size range
        self.split = QSplitter(Qt.Horizontal)
        self.split.setChildrenCollapsible(False)
        self.left = self._build_layers()
        self.centre = self._build_centre()
        self.right = self._build_settings()
        for w in (self.left, self.centre, self.right):
            self.split.addWidget(w)
        self.split.splitterMoved.connect(self._panel_moved)
        self.ldock = Dock(self, self.left, lambda: self._dock_side(self.left), lambda: self.left.setParent(self),
                          title=_("Layers"), role="layers", side="left")
        self.dock = Dock(self, self.right, lambda: self._dock_side(self.right), lambda: self.right.setParent(self),
                         lambda: self.folds.items(), _("Properties"), role="adjust")
        root.addWidget(self.ldock.strip)
        root.addWidget(self.split)
        root.addWidget(self.dock.strip)
        self.ldock.chosen.connect(lambda st: self.panelChosen.emit("left", st))
        self.dock.chosen.connect(lambda st: self.panelChosen.emit("right", st))
        self.lmenu_b.clicked.connect(lambda: self.ldock.menu(self.lmenu_b))
        self.rmenu_b.clicked.connect(lambda: self.dock.menu(self.rmenu_b))
        self._fit_split()
        self._connect(own_worker=worker is None)
        if self.layers:
            self.sel = self._choice_of(self._version_shown()) or (self.layers[-1]["id"], "layer")
        self.build_filter_menu()
        self.rebuild()
        self._send_final()

    def _build_centre(self) -> QWidget:
        """The centre: the bar of view tools, the viewer and the hint line under it."""
        mid = QWidget()
        mv = QVBoxLayout(mid)
        mv.setContentsMargins(0, 0, 0, 0)
        mv.setSpacing(0)
        bar = QHBoxLayout()
        bar.setContentsMargins(10, 6, 10, 6)
        self.top_bar = bar
        self._probes = {}                                # hidden copies of its buttons, measured for its fit (_width_as)
        self.b_fit = icon_only(QPushButton(), "fit", _("Fit (F)"), 16)
        self.b_100 = icon_only(QPushButton(), "actual_size", _("Actual size (1)"), 16)
        self.zoom_lab = QPushButton(with_unit("100", "%"))   # the zoom, and its presets (issue #44: as Inspect's viewer)
        self.zoom_lab.setToolTip(_("Zoom (wheel, + and −)"))
        self.zoom_menu = QMenu(self.zoom_lab)
        self.zoom_menu.addAction(_("Fit"), lambda: self.viewer.fit())
        for z in ZOOM_PRESETS:
            self.zoom_menu.addAction(with_unit(f"{z * 100:g}", "%"), lambda zz=z: self.viewer.zoom_to(zz))
        self.zoom_lab.setMenu(self.zoom_menu)
        self.b_cmp = QPushButton(_("Before / after"))
        self.b_cmp.setCheckable(True)
        self.b_cmp.setObjectName("toggle")
        self.b_cmp.setToolTip(_("Before: the bottom layer, suggested settings (C)"))
        self.b_crop = QPushButton(_("Crop"))
        self.b_crop.setCheckable(True)
        self.b_crop.setObjectName("toggle")
        self.b_crop.setToolTip(_("Crop the picture: nothing is lost, it can be changed at any time (R)"))
        self.b_mask = QPushButton(_("Show mask"))
        self.b_mask.setCheckable(True)
        self.b_mask.setObjectName("toggle")
        self.b_mask.setToolTip(_("Mask of the selected layer, in gray (M)"))
        # the annotations of the astrometric solution over the image (and, if wanted, burned into an export)
        self.ann_show = {"grid": False, "objects": False}
        self._ann = None                                 # annotate.Annotations of the canvas shown
        self._ann_key = None
        self._ann_token = None
        self._canvas = None                              # the last result's canvas (size and origin on the grid)
        self.b_grid = QPushButton(_("Sky grid"))
        self.b_grid.setToolTip(_("The RA / Dec grid of the astrometric solution"))
        self.b_names = QPushButton(_("Object names"))
        self.b_names.setToolTip(_("The known deep-sky objects of the field (Messier, NGC, IC, Sharpless and their "
                                  "common names), outlined at their size. Export… can burn them into the picture."))
        for b in (self.b_grid, self.b_names):
            b.setCheckable(True)
            b.setObjectName("toggle")
        self.b_undo = icon_only(QPushButton(), "undo", _("Undo (Ctrl+Z)"), 16)
        self.b_redo = icon_only(QPushButton(), "redo", _("Redo (Ctrl+Y)"), 16)
        for b in (self.b_undo, self.b_redo):
            b.installEventFilter(self)                   # their tip names the step (_undo_tips)
        self.b_export = set_icon(QPushButton(_("Export…")), "export", 16)
        for w in (self.b_fit, self.b_100, self.zoom_lab):
            bar.addWidget(w)
        bar.addSpacing(12)
        # the versions of the processing (#123: a "+" and a "−" beside a list were not understood), as the reference
        # editors' snapshots: their name, a named button that makes one, a bin that deletes the one shown
        self.version_lab = label(_("Version"), "dim")
        self.cb_version = QComboBox()
        self.cb_version.setEditable(True)
        # its name's start shown, "…" at its end (#146); made in the list, the list's own field: made apart, it was styled
        # as a field of its own, its padding and border inside the list's, its text 13 px high, a g's tail cut (#152)
        self.cb_version.setLineEdit(NameEdit("", self.cb_version))
        self.cb_version.setToolTip(_("The version shown: a click opens the list, a double-click renames it"))
        # one width whatever its names, today's (about "Version 1"), within bounds: a long name never pushes the bar, it
        # is cut (#146; as wide as its names, the list resized itself after the bar's layout and covered New version)
        self.cb_version.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.cb_version.setMinimumContentsLength(9)
        self.cb_version.setMinimumWidth(VERSION_W[0][0])
        self.cb_version.setMaximumWidth(VERSION_W[0][1])
        self.b_version_new = set_icon(QPushButton(_("New version")), "add", 16)
        self.b_version_new.setToolTip(_("New version: a copy of this one, to try other settings without losing these"))
        self.b_version_del = icon_only(QPushButton(), "delete", _("Delete this version"), 16)
        for w in (self.version_lab, self.cb_version, self.b_version_new, self.b_version_del):
            bar.addWidget(w)
        self._entering = False
        self._version_click = QTimer(self, singleShot=True)
        self._version_click.timeout.connect(self.cb_version.showPopup)
        self.cb_version.lineEdit().installEventFilter(self)
        self.cb_version.installEventFilter(self)         # its focus, given by a click (eventFilter)
        self.cb_version.activated.connect(self._version_picked)
        self.cb_version.lineEdit().editingFinished.connect(self._version_renamed)
        self.b_version_new.clicked.connect(self._version_new)
        self.b_version_del.clicked.connect(self._version_delete)
        bar.addStretch(1)
        self.cb_cmp = QComboBox()
        self.cb_cmp.setToolTip(_("What the picture is compared with; V shows one then the other whole"))
        self.cb_cmp.setVisible(False)
        self.cb_cmp.currentIndexChanged.connect(lambda *_a: self.b_cmp.isChecked() and self._send_before())
        bar.addWidget(self.cb_cmp)
        self._fill_versions()
        # a narrow centre (a laptop, Windows' scale at 125 or 150 %): the tools least used go into "…" one by one,
        # rather than the bar pushing the sides out of the window (the night test pass)
        self.b_more = icon_only(QPushButton(), "more", _("More tools"), 16)     # as the bar's other buttons
        self.more_menu = QMenu(self.b_more)
        self.b_more.clicked.connect(self._show_more)
        self.b_more.setVisible(False)
        # the crop is confirmed or given up with named buttons, shown while it is drawn (#121), as the reference
        # editors' options bar; Spot healing is a tool of the bar (#122: it was found only in the filters' menu)
        self.b_crop_ok = set_icon(QPushButton(_("Apply")), "ok", 16, tip=_("Keep this crop (Enter)"))
        self.b_crop_ok.setObjectName("primary")
        self.b_crop_cancel = QPushButton(_("Cancel"))
        self.b_crop_cancel.setToolTip(_("Leave the crop as it was (Esc)"))
        for b in (self.b_crop_ok, self.b_crop_cancel):
            b.setVisible(False)
        self.b_heal = QPushButton(_("Spot healing"))
        self.b_heal.setCheckable(True)
        self.b_heal.setObjectName("toggle")
        self.b_heal.setToolTip(_("Heal spots, dust and satellite trails: click or paint over them (Alt+click: the "
                                 "source by hand)"))
        for w in (self.b_crop, self.b_crop_ok, self.b_crop_cancel, self.b_heal, self.b_grid, self.b_names, self.b_cmp,
                  self.b_mask, self.b_more, self.b_undo, self.b_redo, self.b_export):
            bar.addWidget(w)
        mv.addLayout(bar)
        mid.setMinimumWidth(CENTRE_MIN)
        mid.installEventFilter(self)
        self.src = PostSource(self)
        self.before = BeforeSource(self)
        self.viewer = Viewer()
        self.viewer.overlays.append(self._overlay)
        self.viewer.overlays.append(self._ann_overlay)
        self.viewer.overlays.append(self._heal_overlay)
        self.viewer.overlays.append(self._behind_tag)
        self.viewer.banner = _("Rendering…")
        self.viewer.crop_box = tuple(self.crop) if self.crop else None
        # a picture from outdated results: said over it, with the Run that brings it up to date (not a pop-up)
        self.stale = QFrame()
        self.stale.setStyleSheet(f"QFrame {{ border-bottom: 2px solid {WARN}; }} QLabel {{ border: none; }}")
        sl = QHBoxLayout(self.stale)
        sl.setContentsMargins(10, 4, 10, 4)
        self.stale_text = label(_("Outdated: run to update"), "dim")
        self.stale_run = QPushButton(_("Run"))
        self.stale_run.setToolTip(_("Run the workflows that are out of date"))
        self.stale_run.clicked.connect(lambda: self.runAsked.emit())
        sl.addWidget(self.stale_text, 1)
        sl.addWidget(self.stale_run)
        self.stale.hide()
        mv.addWidget(self.stale)
        mv.addWidget(self.viewer, 1)
        # one line, cut with "…" when the window is narrow and whole on hover (the laptop's D3: cut at 1250 px)
        self.hint = ElidedLabel("", "small", Qt.ElideRight)
        self.hint.setContentsMargins(10, 3, 10, 5)
        mv.addWidget(self.hint)
        return mid

    def _overflow(self) -> tuple:
        """The tools that go into "…" when the bar has no room, the first to go first."""
        return (self.b_names, self.b_grid, self.b_mask, self.b_cmp, self.b_heal, self.b_crop, self.b_undo, self.b_redo,
                self.b_version_del, self.b_version_new)

    def _bar_steps(self) -> tuple:
        """The bar narrowed one step at a time when the centre has no room (a laptop, Windows' scale at 125 or 150 %),
        the least used first. Export and New version give up their text before any tool goes into "…" (their icon
        alone, their name on hover: in French the bar is about 113 px wider, and Spot healing went into "…" where the
        English bar still showed it); then the view's aids go into "…", the word "Version", a narrower list, 100 % (the
        zoom's menu has it); the tools one works with, Spot healing and Crop, only then; the bin and New
        version into "…", Redo and Undo (the Edit menu has them too), Fit the last (the laptop's D1: the most used, it
        went first with Before / after on a laptop's window): at the centre's narrowest, the crop's
        Apply and Cancel shown, the bar still fits (#142). The names stay in the tooltips and in "…"."""
        return (("icon", self.b_export), ("icon", self.b_version_new),
                ("hide", self.b_names), ("hide", self.b_grid), ("hide", self.b_mask), ("hide", self.b_cmp),
                ("hide", self.version_lab), ("narrow", self.cb_version), ("hide", self.b_100),
                ("hide", self.b_heal), ("hide", self.b_crop),
                ("hide", self.b_version_del), ("hide", self.b_version_new), ("hide", self.b_redo), ("hide", self.b_undo),
                ("hide", self.b_fit))

    def _bar_texts(self) -> dict:
        """The buttons of the bar that go to their icon, and their name."""
        return {self.b_export: _("Export…"), self.b_version_new: _("New version")}

    @staticmethod
    def _item_width(w: QWidget, min_w: int | None = None, max_w: int | None = None) -> int:
        """The width the bar's layout gives a widget, as Qt's widget item works it out: its size hint, at least its
        minimum hint and its minimum width (``min_w`` in its place), at most its maximum width (``max_w`` in its place)."""
        hint = min(max(w.sizeHint().width(), w.minimumSizeHint().width()), w.maximumWidth() if max_w is None else max_w)
        return max(hint, w.minimumWidth() if min_w is None else min_w)

    def _width_as(self, b: QPushButton, text: str) -> int:
        """The width the bar gives button ``b`` with this text (its icon alone for ""): a hidden copy measures it, as
        working the fit out must change nothing in the bar (a change asks the layout again)."""
        key = (b.property("iconRole"), b.iconSize().width(), text)
        probe = self._probes.get(key)
        if probe is None:
            probe = self._probes[key] = set_icon(QPushButton(text, self.centre), key[0], key[1])
            probe.hide()
        return self._item_width(probe)

    def _bar_need(self, k: int) -> int:
        """The bar's width with the first ``k`` steps of ``_bar_steps`` taken, as its layout works it out (its widgets'
        widths, its spacings, its margins), the bar left as it is."""
        steps = self._bar_steps()
        hidden = [w for kind, w in steps[:k] if kind == "hide"]
        icons = [w for kind, w in steps[:k] if kind == "icon"]
        narrow = [w for kind, w in steps[:k] if kind == "narrow"]
        steered = [w for _kind, w in steps]
        more = any(b in hidden for b in self._overflow())
        texts = self._bar_texts()
        bar = self.top_bar
        m = bar.contentsMargins()
        total, n = m.left() + m.right(), 0
        for i in range(bar.count()):
            item = bar.itemAt(i)
            w = item.widget()
            if w is None:                                # the space after the zoom, the stretch
                total += item.sizeHint().width()
                continue
            if not (w not in hidden if w in steered else more if w is self.b_more else not w.isHidden()):
                continue
            n += 1
            if w in texts:
                total += self._width_as(w, "" if w in icons else texts[w])
            elif w is self.cb_version:
                total += self._item_width(w, *VERSION_W[w in narrow])
            else:
                total += self._item_width(w)
        return total + max(bar.spacing(), 0) * max(n - 1, 0)

    def _fit_bar(self):
        """As many tools in the bar as the centre has room for, the others in "…": the bar's width worked out for each
        step, without trying them on the bar, and the first that fits taken (#142: tried on the bar, a width that grew
        afterwards, the zoom's "Zoom 27 %" or the list's longer name, was not seen: New version covered the list)."""
        if not hasattr(self, "centre"):
            return
        steps = self._bar_steps()
        room = self.centre.width()
        k = next((k for k in range(len(steps) + 1) if self._bar_need(k) <= room), len(steps))
        hidden = [w for kind, w in steps[:k] if kind == "hide"]
        icons = [w for kind, w in steps[:k] if kind == "icon"]
        more = any(b in hidden for b in self._overflow())
        # only what changes is set: an unchanged bar asks nothing of the layout (this runs at each of its requests)
        for w in [w for _kind, w in steps] + [self.b_more]:
            show = more if w is self.b_more else w not in hidden
            if w.isHidden() == show:
                w.setVisible(show)
        for b, text in self._bar_texts().items():
            want = "" if b in icons else text
            if b.text() != want:
                b.setText(want)
        self.b_export.setToolTip(_("Export the picture to a file (TIFF, PNG, JPEG, FITS) · Ctrl+E"))
        lo, hi = VERSION_W[any(kind == "narrow" for kind, _w in steps[:k])]
        if (self.cb_version.minimumWidth(), self.cb_version.maximumWidth()) != (lo, hi):
            self.cb_version.setMinimumWidth(lo)
            self.cb_version.setMaximumWidth(hi)

    def _show_more(self):
        self._fill_more()
        self.more_menu.exec(self.b_more.mapToGlobal(self.b_more.rect().bottomLeft()))

    def _fill_more(self):
        """The "…" menu: the tools the bar has no room for, as their buttons (checked, greyed, their tip)."""
        self.more_menu.clear()
        self._undo_tips()
        names ={self.b_undo: _("Undo"), self.b_redo: _("Redo"), self.b_version_del: _("Delete this version"),
                 self.b_version_new: _("New version")}                   # (shown by their icons in the bar)
        for b in reversed(self._overflow()):
            if not b.isHidden():
                continue
            a = self.more_menu.addAction(names.get(b) or b.text())
            a.setCheckable(b.isCheckable())
            a.setChecked(b.isChecked())
            a.setEnabled(b.isEnabled())
            a.setToolTip(b.toolTip())
            a.triggered.connect(lambda _c=False, bb=b: bb.click())
        self.more_menu.setToolTipsVisible(True)

    def eventFilter(self, obj, e):                                   # noqa: N802 - Qt's name
        # resized, or a widget of the bar asks for another width (the zoom's text, a version's name, the crop's
        # buttons, the compact look's margins): the fit again (#142)
        if obj is getattr(self, "centre", None) and e.type() in (QEvent.Resize, QEvent.LayoutRequest):
            self._fit_bar()
        if e.type() == QEvent.ToolTip and obj in (getattr(self, "b_undo", None), getattr(self, "b_redo", None)):
            self._undo_tips()                            # the step's name, as it is when the tip shows
        if e.type() in (QEvent.ShortcutOverride, QEvent.KeyPress) and e.key() == Qt.Key_Escape                 and hasattr(self, "cb_version") and obj is self.cb_version.lineEdit():
            e.accept()                                   # Escape is the field's, as a layer's name: it gives the typing
            if e.type() == QEvent.KeyPress:              # up (the window's page-leaving shortcut never sees it)
                self._fill_versions()                    # the name as it was
                self.viewer.setFocus(Qt.OtherFocusReason)
            return True
        if hasattr(self, "cb_version") and obj is self.cb_version and e.type() == QEvent.FocusIn:
            # a real click gives the field the keyboard before this filter sees the press (the tutorials' film: B5
            # again, a text cursor): that is no name being typed; the double-click's own focus is another reason. (The
            # list has the focus for its field, which hands it on without a FocusIn of its own.)
            self._focus_by_click = e.reason() == Qt.MouseFocusReason
        if hasattr(self, "cb_version") and obj is self.cb_version.lineEdit() and (
                not obj.hasFocus() or getattr(self, "_focus_by_click", False)):
            # the name not being typed in: a click opens the list (the laptop's B5: it put a text cursor), a double-click
            # types in it to rename, as a layer's name; the click waits a double-click's time to know which
            if e.type() == QEvent.MouseButtonPress and e.button() == Qt.LeftButton:
                self._focus_by_click = False
                if obj.hasFocus():
                    self.viewer.setFocus(Qt.OtherFocusReason)   # the keyboard back to the page: no cursor in the name
                self._version_click.start(QApplication.doubleClickInterval())
                return True
            if e.type() == QEvent.MouseButtonDblClick and e.button() == Qt.LeftButton:
                self._version_click.stop()
                self._focus_by_click = False
                obj.setFocus(Qt.OtherFocusReason)
                obj.selectAll()
                return True
            if e.type() == QEvent.MouseButtonRelease:
                return True
        if e.type() == QEvent.KeyPress and e.key() in (Qt.Key_Return, Qt.Key_Enter) and                 obj is self.cb_version.lineEdit():
            self._entering = True                       # Enter typed in the version's name: a renaming, not a pick
            QTimer.singleShot(0, lambda: setattr(self, "_entering", False))
        return super().eventFilter(obj, e)

    @staticmethod
    def _menu_button() -> QToolButton:
        """A side's menu: how it shows (see ``Dock.menu``)."""
        b = QToolButton()
        icon_only(b, "more", _("Panel display"), 16, "dim")
        b.setAutoRaise(True)
        b.setFixedSize(20, 18)
        b.setStyleSheet("QToolButton { border: none; padding: 0; } "
                        "QToolButton:hover { background: #2b3446; border-radius: 3px; }")
        return b

    def _build_layers(self) -> QWidget:
        """The left side: the layers and their buttons."""
        left = QWidget()
        left.setObjectName("panel")
        left.setMinimumWidth(LAYERS_MIN)
        left.setMaximumWidth(LAYERS_MAX)
        lv = QVBoxLayout(left)
        lv.setContentsMargins(10, 10, 10, 8)
        lv.setSpacing(6)
        head = QHBoxLayout()
        head.addWidget(label(_("Layers"), "h2"))
        head.addStretch(1)
        # while a row is dragged between two others: where it lands and what the drop does, over the drop's line in the
        # list (the laptop's D4: in the panel's title it was cut and said nothing of where); a drop onto a layer says it
        # in that layer's row
        self.drop_text = QLabel()
        self.drop_text.setStyleSheet(f"color: {ACCENT}; font-weight: 600; background: #1b1f27; padding: 0 4px; "
                                     "border-radius: 3px;")
        self.drop_text.hide()
        self.lmenu_b = self._menu_button()
        head.addWidget(self.lmenu_b, 0, Qt.AlignTop)
        lv.addLayout(head)
        self.lsc = QScrollArea()
        self.lsc.setWidgetResizable(True)
        self.lsc.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        lh = QWidget()
        self.lbox = QVBoxLayout(lh)
        self.lbox.setContentsMargins(0, 0, 4, 12)          # (room for a drop line after the last row)
        self.lbox.setSpacing(2)
        self.lsc.setWidget(lh)
        lv.addWidget(self.lsc, 1)
        self.drop_line = QFrame(lh)
        self.drop_line.setFixedHeight(3)
        self.drop_line.setStyleSheet(f"background: {ACCENT}; border-radius: 1px;")
        self.drop_line.hide()
        self.drop_dot = QFrame(lh)                       # the start of the line, as in the reference image editor
        self.drop_dot.setFixedSize(9, 9)
        self.drop_dot.setStyleSheet(f"background: #1b1f27; border: 2px solid {ACCENT}; border-radius: 4px;")
        self.drop_dot.hide()
        btns = QHBoxLayout()
        btns.setSpacing(4)
        # the layers' buttons, as the reference editors': icons with their full names in the tooltips
        self.b_open = icon_only(QPushButton(), "block_image", _("Open images as layers"), 16, box=28)
        self.b_open.clicked.connect(self.openImages.emit)
        # #124: a "+" alone read as adding a layer: the button says "Filter", its menu the filters by family
        self.b_filter = set_icon(QPushButton(_("Filter")), "filter_layer", 16,
                                 tip=_("Add a filter layer above the selected layer"))
        self.filter_menu = QMenu(self.b_filter)
        self.b_filter.setMenu(self.filter_menu)
        self.b_dup = icon_only(QPushButton(), "duplicate", _("Duplicate (Ctrl+J)"), 16, box=28)
        self.b_maskadd = icon_only(QPushButton(), "mask", _("Add a mask to the selected layer"), 16, box=28)
        self.b_del = icon_only(QPushButton(), "delete", _("Delete the selected layer"), 16, box=28)
        for b in (self.b_open, self.b_filter, self.b_dup, self.b_maskadd, self.b_del):
            btns.addWidget(b)
        btns.addStretch(1)
        lv.addLayout(btns)
        return left

    def _build_settings(self) -> QWidget:
        """The right side: the settings of the selection (a layer, its mask or one of its parts)."""
        right = QWidget()
        right.setObjectName("panel")
        right.setMinimumWidth(PANEL_MIN)
        right.setMaximumWidth(PANEL_MAX)
        rv = QVBoxLayout(right)
        rv.setContentsMargins(10, 8, 10, 8)
        rv.setSpacing(2)
        head = QHBoxLayout()
        head.addWidget(label(_("Properties"), "dim"))
        head.addStretch(1)
        self.rmenu_b = self._menu_button()
        head.addWidget(self.rmenu_b)
        rv.addLayout(head)
        psc = QScrollArea()
        psc.setWidgetResizable(True)
        psc.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        ph = QWidget()
        self.pbox = QVBoxLayout(ph)
        self.pbox.setContentsMargins(0, 0, 14, 0)
        self.pbox.setSpacing(4)
        psc.setWidget(ph)
        rv.addWidget(psc, 1)
        self.psc = psc
        self.folds = Sections(self.pbox, psc)            # its groups fold, as the block panel's sections (issue #44)
        return right

    def _connect(self, own_worker: bool):
        """The buttons, the shortcuts, the timers and the signals of the worker and the model."""
        self.b_fit.clicked.connect(self.viewer.fit)
        self.b_100.clicked.connect(lambda: self.viewer.zoom_to(1.0))
        self.b_cmp.toggled.connect(self._compare)
        self.b_mask.toggled.connect(self._mask_view)
        self.b_crop.toggled.connect(self._crop_tool)
        self.b_crop_ok.clicked.connect(self.crop_apply)
        self.b_crop_cancel.clicked.connect(self.crop_cancel)
        self.b_heal.toggled.connect(self.heal_tool)
        self.viewer.cropChanged.connect(self._crop_drawn)
        self.b_grid.toggled.connect(lambda on: self.set_annotations(grid=on))
        self.b_names.toggled.connect(lambda on: self.set_annotations(objects=on))
        self.b_undo.clicked.connect(self.do_undo)
        self.b_redo.clicked.connect(self.do_redo)
        self.b_export.clicked.connect(lambda: self.export_dialog())
        self.b_dup.clicked.connect(self.duplicate)
        self.b_maskadd.clicked.connect(self.mask_button)
        self.b_del.clicked.connect(self.delete)
        self.viewer.viewChanged.connect(self._view_changed)
        self.viewer.stroke.connect(self._stroke)
        self.viewer.strokeEnd.connect(self._stroke_end)
        self.viewer.toolPress.connect(self._heal_press)
        self.viewer.toolMove.connect(self._heal_move)
        self.viewer.toolRelease.connect(self._heal_release)
        # Ctrl+Z / Ctrl+Y are the window's Edit > Undo / Redo, which come here while Post is shown (two shortcuts on
        # one key would cancel each other)
        for seqs, _label, fn in POST_KEYS:
            for seq in seqs if fn is not None else ():
                sc = QShortcut(QKeySequence(seq), self)
                sc.setContext(Qt.WidgetWithChildrenShortcut)
                sc.activated.connect(lambda fn=fn: fn(self))
        # Enter keeps the crop, only while it is drawn: always on, these two took the key from every field of Post
        # (#141: a version's name, a Curves value were no longer applied by Enter)
        self._crop_keys = []
        for seq in ("Return", "Enter"):
            sc = QShortcut(QKeySequence(seq), self)
            sc.setContext(Qt.WidgetWithChildrenShortcut)
            sc.activated.connect(self.crop_apply)
            sc.setEnabled(False)
            self._crop_keys.append(sc)
        self._rendered_sig = None                       # the visible signature of the last final render asked
        self._draft_dirty = False                       # a draft is on the screen after the last final render
        self._draft_t = QTimer(self)
        self._draft_t.setSingleShot(True)
        self._draft_t.timeout.connect(self._send_draft)
        self._final_t = QTimer(self)
        self._final_t.setSingleShot(True)
        self._final_t.timeout.connect(self._send_final)
        self._held = None                                # the half image waiting for the exact one (see HOLD_MS)
        self._hold_t = QTimer(self)
        self._hold_t.setSingleShot(True)
        self._hold_t.timeout.connect(self._show_held)
        self._main_seq = 0                               # the newest draft or release request (the exact view's, Q1)
        self._failed_seq = 0                             # the newest request that failed (no "Updating…" for it)
        self._exact_wait = None                          # the release whose exact view is on its way (still updating)
        self._final_kind = "half"                        # the answer of the newest request that is its picture
        self._done_seq = 0                               # the newest request whose picture is on the screen
        self._region_t = QTimer(self)
        self._region_t.setSingleShot(True)
        self._region_t.timeout.connect(self._send_region)
        self._save_t = QTimer(self)                      # the composition into the project, just after an edit (push)
        self._save_t.setSingleShot(True)
        self._save_t.timeout.connect(self._save)
        self.worker.rendered.connect(self._on_result)
        self.worker.failed.connect(self._on_failed)
        self.worker.deviceChanged.connect(self._on_device)
        if self.worker.device:                           # the worker may have said it before we listened
            self._on_device(self.worker.device)
        self.s.pathsChanged.connect(self.sync_paths)
        if hasattr(self.s, "postChanged"):
            self.s.postChanged.connect(self._post_changed)
            self.s.post_shown = lambda: save_layers(self.layers)
            self.destroyed.connect(lambda *_a, s=self.s: setattr(s, "post_shown", None))
        self.destroyed.connect(self.worker.stop if own_worker else (lambda *_a: None))

    # ------------------------------------------------------------ the model
    def path(self, pid):
        """The model's workflow of a layer (a WorkflowView), or None."""
        return self.s.view(pid)

    def shutdown(self):
        self.flush()
        self.worker.stop()

    def flush(self):
        """The last edit into the project now (before the project is saved and closed), and the layer chosen."""
        self._keep_choice()
        if self._save_t.isActive():
            self._save_t.stop()
            self._save()

    # ------------------------------------------------------------ layers from the workflows
    def sync_paths(self, initial=False):
        paths = list(self.s.paths)
        alive = {p.pid for p in paths}
        if initial:
            version = self._version_shown()
            self.layers = self._with_synth(load_layers(version.get("layers"), alive))
            self._show_crop(version.get("crop"))
            self._saved = json.dumps(save_layers(self.layers), sort_keys=True)
            self._version = version.get("id")
            self._fill_versions()
        known = {lay["pid"] for lay in self.layers if lay["kind"] == "workflow"}
        fresh = not known
        ids_before = {x["id"] for x in self.layers}
        removed = getattr(self.s, "post_removed", lambda: set())() | getattr(self.s, "version_removed", lambda: set())()
        for p in paths:
            # every workflow's image comes, except one the user took out (issues #10, #52: it does not come back)
            stars = p.opts.get("etoiles") == "retirer"
            out = (p.pid, "combined" if stars else None) in removed
            channel = getattr(self.s, "feeds_fusion", None) is not None and self.s.feeds_fusion(p)   # #47: the fusion
            channel = channel or (getattr(self.s, "gives_post", None) is not None and not self.s.gives_post(p))
            if not out and p.rendered and p.pid not in known and not channel:   # with Stars: its Combine
                # a block's image sent to Post (a branch made for it) is shown at once, as the Split images are
                sent = getattr(p, "end", None) is not None or getattr(self.s, "is_sent_branch", lambda q: False)(p) or \
                    getattr(self.s, "is_plain_image", lambda q: False)(p)
                # a variant is made to be looked at: its layer arrives visible
                sent = sent or getattr(p, "variant_of", None) is not None
                self.layers.insert(self._place_of(p.pid, paths), new_workflow_layer(
                    p.pid, p.name + (" · " + POST_PART_NAMES["combined"] if stars else ""), bool(sent),
                    part="combined" if stars else None))
        self.layers = [lay for lay in self.layers if lay["kind"] != "workflow" or lay["pid"] in alive]
        self.layers = normalize(self._with_synth(normalize(self.layers)))
        wl = [lay for lay in self.layers if lay["kind"] == "workflow"]
        if wl and fresh:                                # the first layers: "Full" (else the bottom one) is the base
            names = {lay["id"]: self.path(lay["pid"]).name for lay in wl}
            base = next((lay for lay in reversed(wl) if names[lay["id"]] == "Full"),
                        next((lay for lay in reversed(wl) if names[lay["id"]].startswith("Full")), wl[-1]))
            self.layers.remove(base)
            self.layers.append(base)
            base["visible"] = True
        elif wl and {x["id"] for x in self.layers} != ids_before and not any(lay["visible"] for lay in wl):
            wl[-1]["visible"] = True                       # (only when layers came or went: one hidden by the eye stays so)
        if not initial:
            # a layer gone, or none chosen yet (the night test pass: Post opened by Open images showed an empty panel):
            # the bottom one, as on reopening
            if (self.sel is None or not self.layer(self.sel[0])) and self.layers:
                self.sel = (self.layers[-1]["id"], "layer")
            elif self.sel and not self.layer(self.sel[0]):
                self.sel = None
            self.rebuild()
            self.worker.refresh()
            self.request(final_only=True, force=True)                  # (nothing is saved here: the layers a workflow brings are
                                                           # worked out again from the workflows, only edits are kept)

    def _version_shown(self) -> dict:
        p = getattr(self.s, "p", None)
        return active_version(p.post) if p is not None and hasattr(p, "post") else {}

    def _with_synth(self, layers):
        """The layers with the Synthetic stars row of the workflows that have the block (issue #77)."""
        return with_synth_rows(layers, self.s.synth_wanted()) if hasattr(self.s, "synth_wanted") else layers

    def _fill_versions(self):
        """The Version list and the choices of the comparison, as the project has them."""
        if not hasattr(self, "cb_version"):
            return
        p = getattr(self.s, "p", None)
        versions = p.post["versions"] if p is not None and hasattr(p, "post") else []
        for cb in (self.cb_version, self.cb_cmp):
            cb.blockSignals(True)
        kept = self.cb_cmp.currentData()
        self.cb_version.clear()
        self.cb_cmp.clear()
        self.cb_cmp.addItem(_("Suggested"), None)
        for v in versions:
            self.cb_version.addItem(v["name"], v["id"])
            if v["id"] != self._version:
                self.cb_cmp.addItem(v["name"], v["id"])
        i = self.cb_version.findData(self._version)
        self.cb_version.setCurrentIndex(max(i, 0))
        self.cb_version.lineEdit().setCursorPosition(0)  # its start, Enter's renaming too (#146)
        tip = _("The version shown: a click opens the list, a double-click renames it")
        self.cb_version.setToolTip(f"{self.cb_version.currentText()}\n{tip}" if self.cb_version.count() else tip)
        j = self.cb_cmp.findData(kept)
        self.cb_cmp.setCurrentIndex(max(j, 0))
        single_choice(self.cb_cmp)                       # Suggested alone: its text (D5)
        self.b_version_del.setEnabled(len(versions) > 1)
        self.b_version_del.setToolTip(_("Delete this version") if len(versions) > 1 else
                                      _("The only version cannot be deleted"))
        for cb in (self.cb_version, self.cb_cmp):
            cb.blockSignals(False)

    def _switch_version(self, version: dict):
        """Another version in the panel: its layers, its crop and its own Undo, and the layer chosen in it when it was
        left."""
        self._keep_choice()
        self._stacks[self._version] = (self.undo, self.redo)
        self._heal_hand[self._version] = (self._heal_src, self._heal_off)    # the source set by hand is the version's own
        self._heal_src, self._heal_off = self._heal_hand.get(version.get("id"), (None, None))
        self.undo, self.redo = self._stacks.get(version.get("id"), ([], []))
        self._version = version.get("id")
        self.layers = self._with_synth(load_layers(version.get("layers"), {p.pid for p in self.s.paths}))
        self._saved = json.dumps(save_layers(self.layers), sort_keys=True)
        self._show_crop(version.get("crop"))
        self._fill_versions()
        if self.sel and not self.layer(self.sel[0]):
            self.sel = self._choice_of(version) or ((self.layers[0]["id"], "layer") if self.layers else None)
        self.sync_paths()
        self.rebuild()
        self._undo_buttons()
        self.request(final_only=True, force=True)
        if self.b_cmp.isChecked():
            self._send_before()

    def _from_post(self):
        """The changes of the model made inside are Post's steps: the graph's Undo leaves them to Post."""
        return self.s.from_post() if hasattr(self.s, "from_post") else _NoContext()

    def _version_picked(self, i: int):
        vid = self.cb_version.itemData(i)
        if self._entering:                              # a name typed that another version has: refused, not a pick
            if vid is not None and vid != self._version:
                self._version_renamed()
            return
        if vid is not None and vid != self._version:
            self._save()
            self.s.set_post_version(vid)
        self.viewer.setFocus(Qt.OtherFocusReason)       # the keys back to the page: Ctrl+Z undoes the version's edits (B4)

    def _version_new(self):
        self._save()
        old = self.cb_version.currentText()
        with self._from_post():
            self.s.new_post_version()
        self.redo.clear()                                # (making a version is no step of Undo: the new one starts empty)
        self._undo_buttons()
        self.toasted.emit(_("“{name}”, a copy of “{old}”: its own Undo starts here").format(
            name=self.cb_version.currentText(), old=old))

    def _version_renamed(self):
        """The name typed for the version shown: refused with its reason when empty or already another version's, the
        field back to the current name."""
        name = self.cb_version.currentText().strip()
        versions = getattr(getattr(self.s, "p", None), "post", {}).get("versions", [])
        mine = next((v["name"] for v in versions if v["id"] == self._version), None)
        if not name:
            self.toasted.emit(_("Name refused: a version needs a name"))
        elif name != mine and any(v["name"] == name and v["id"] != self._version for v in versions):
            self.toasted.emit(_("Name refused: another version is already named “{name}”").format(name=name))
        elif name != mine:
            with self._from_post():
                self.s.rename_post_version(self._version, name)
            self.undo.append(self._model_mark("_undo"))
            self.redo.clear()
            self._undo_buttons()
        self._fill_versions()

    def _version_delete(self):
        """The version shown taken out, as one step of Undo (#145: the bin took it for good): the model keeps the step
        (its snapshot holds the composition, every version with its layers and its crop), and the version shown after
        it holds it on top of its Undo, so that Ctrl+Z there brings the version back before any older edit of its own."""
        p = getattr(self.s, "p", None)
        if p is None or len(p.post["versions"]) < 2:
            return
        self._save()
        gone = self.cb_version.currentText()
        with self._from_post():
            self.s.delete_post_version(self._version)
        self.toasted.emit(_("“{name}” deleted").format(name=gone) + " · Ctrl+Z")      # (the model's words for its step)
        self.undo.append(self._model_mark("_undo"))                     # (the delete showed the other version: its own Undo now)
        self.redo.clear()
        self._undo_buttons()

    def _flip_compare(self):
        """The whole picture from one of the two (the comparison on): the other one at the next press."""
        if self.b_cmp.isChecked():
            self.viewer.curtain = 0.0 if self.viewer.curtain > 0.5 else 1.0
            self.viewer.update()

    def _compared(self):
        """The version the comparison shows, None: the suggested settings."""
        vid = self.cb_cmp.currentData() if hasattr(self, "cb_cmp") else None
        p = getattr(self.s, "p", None)
        return None if vid is None or p is None else next((v for v in p.post["versions"] if v["id"] == vid), None)

    def _post_changed(self, source):
        """The graph changed the composition (a layer sent to Post, a Stars and background block, its settings):
        taken in as one undo step."""
        if source is self:
            return
        version = self._version_shown()
        if version.get("id") != self._version:
            self._switch_version(version)
            return
        self._fill_versions()
        text = json.dumps(version.get("layers") or [], sort_keys=True)
        # what is shown, not what was saved last: a branch removed takes its layer from the panel without a save, and
        # its Ctrl+Z puts back the very composition saved before (the layer with its settings, not a new one)
        if text == self._saved and text == json.dumps(save_layers(self.layers), sort_keys=True):
            return
        old = {lay["id"] for lay in self.layers}
        before = self.layers
        self.layers = self._with_synth(load_layers(version.get("layers"), {p.pid for p in self.s.paths}))
        if getattr(self, "_restoring", False):          # (a step of the workflows' Undo undone here: no step of its own)
            pass
        elif getattr(self.s, "_post_depth", 0) == 0:
            # made in the Graph: its step is the Graph's to undo, never Post's. Post's steps keep working around it: the
            # layers the Graph put in or took out are put in or taken out of the steps of this version's lists too
            self._rebase_steps(before, self.layers)
        else:
            self.undo.append(snapshot(before))
            self.redo.clear()
        self._saved = json.dumps(save_layers(self.layers), sort_keys=True)
        added = [lay["id"] for lay in self.layers if lay["id"] not in old]
        if added:                                   # a layer sent from the graph: selected, as a pasted layer is
            self.sel = (added[0], "layer")
        elif self.sel and not self.layer(self.sel[0]):
            self.sel = (self.layers[0]["id"], "layer") if self.layers else None
        self.sync_paths()                           # the layers the workflows give and the composition lacks yet

    def _rebase_steps(self, before: list, after: list):
        """The Graph changed the layers (``before`` to ``after``): the layers it added are put into the steps of Post's Undo
        and Redo (at their place, as they are now), the ones it took out are taken out of them, so that going back through
        Post's own steps never undoes the Graph's."""
        was = {x["id"] for x in before}
        now = {x["id"] for x in after}
        added = [(i, x) for i, x in enumerate(after) if x["id"] not in was]
        gone = was - now
        if not added and not gone:
            return
        for steps in (self.undo, self.redo):
            for k, step in enumerate(steps):
                if isinstance(step, ModelStep):
                    continue
                have = {x["id"] for x in step}
                fresh = Step(x for x in step if x["id"] not in gone)
                fresh.crop = getattr(step, "crop", None)
                for i, lay in added:
                    if lay["id"] not in have:
                        fresh.insert(min(i, len(fresh)), snapshot([lay])[0])
                steps[k] = fresh

    def select_layer(self, lid):
        """Show this layer's settings (the graph's Open in Post); not computed yet, the page's list shows it."""
        if self.layer(lid) is not None:
            self.select(lid, "layer")
        if lid is not None:
            self.layerAsked.emit(lid)

    def layer(self, lid):
        return next((lay for lay in self.layers if lay["id"] == lid), None)

    def index(self, lid):
        return next(i for i, lay in enumerate(self.layers) if lay["id"] == lid)

    def _save(self):
        """The composition goes into the project (its definition file is written by the model, as for any edit)."""
        p = getattr(self.s, "p", None)
        if not self.persist or p is None or not hasattr(p, "post"):
            return
        version = active_version(p.post)
        # what is saved, kept when this is the first edit since the project was saved or opened: an Undo back to it
        # with no other edit made meanwhile leaves no edit (``_back_to_saved``)
        clean = getattr(self.s, "dirty", True) is False
        was = (self._version, self._saved, copy.deepcopy(version.get("crop")))
        edits = getattr(self.s, "edits", None)
        self._write(version)
        if clean:
            self._clean = (*was, self.s.edits) if self.s.dirty and edits is not None else None
        elif self._clean is not None:                    # another edit since, or another version's: not Post's to say
            own = self._clean[3] == edits and self._clean[0] == self._version
            self._clean = (*self._clean[:3], self.s.edits) if own else None

    def _back_to_saved(self):
        """After an Undo / Redo of Post's own: the composition is as the project is saved and nothing else was edited
        since: the project has no edit left (the laptop's pass: every change undone, the title kept its "*")."""
        c = self._clean
        if c is not None and getattr(self.s, "dirty", False) and c == (self._version, self._saved, self.crop, self.s.edits):
            self._clean = None
            self.s.back_to_saved()

    def _write(self, version):
        if version.get("crop") != self.crop:
            version["crop"] = self.crop
            if hasattr(self.s, "edited"):
                self.s.edited()
        data = save_layers(self.layers)
        text = json.dumps(data, sort_keys=True)
        if text == self._saved:
            return
        self._saved = text
        if hasattr(self.s, "set_post_layers"):
            self.s.set_post_layers(data, self, scope=self.s.post_scope(data))
            return
        version["layers"] = data
        edited = getattr(self.s, "edited", None)
        if edited is not None:
            edited()

    def _keep_choice(self):
        """The chosen layer is kept with the version when the project is saved (``flush``), and chosen again when it is
        opened (the tutorials' film: the bottom one was chosen). Choosing is no edit: the project is not marked, and
        a project opened and left is unchanged."""
        p = getattr(self.s, "p", None)
        if not self.persist or p is None or not isinstance(getattr(p, "post", None), dict) or not self.sel:
            return
        version = next((v for v in p.post.get("versions") or [] if v.get("id") == self._version), None)
        lid = self.sel[0]
        if version is None or self.layer(lid) is None or version.get("selected") == lid:
            return
        if "selected" in version or lid != self.layers[-1]["id"]:     # the bottom one, chosen by opening: nothing to keep
            version["selected"] = lid

    def _choice_of(self, version: dict | None):
        """The layer chosen when this version was left (``_keep_choice``), if it is still there; else None."""
        lid = (version or {}).get("selected")
        return (lid, "layer") if lid is not None and self.layer(lid) is not None else None

    # ------------------------------------------------------------ rendering (in the worker)
    def refresh(self):
        self.request(final_only=True, force=True)

    def _visible_sig(self) -> str:
        want, overlay = self._selected_wants()
        return visible_signature(self.layers, want if overlay else ())

    def request(self, final_only=False, force=False, settings_only=False):
        """Something changed: a draft now (while it moves), the screen's image 220 ms after the last change. A change
        that leaves the shown rows as they are (a hidden filter's slider, a hidden layer's settings) renders nothing and
        shows no draft: the picture on the screen is the one the layers give; the settings are kept all the same.
        Only a settings change says so (``settings_only``: a slider, the Curves): every other request renders, as the
        signature knows the rows alone (a crop, a version switch, the bases are not in it)."""
        if (settings_only and not force and not self._draft_dirty and not self._final_t.isActive()
                and self._rendered_sig == self._visible_sig()):
            need = tuple(self._stale_thumbs())
            if need:
                self.worker.request(Request(kind="thumbs", layers=snapshot(self.layers), need_thumbs=need))
            return
        self.src.full = None
        self._draft_dirty = self._draft_dirty or not final_only
        if not final_only:
            self._draft_t.start(0)
        self._final_t.start(0 if final_only else 220)
        self.viewer.update()                              # its "Updating…" at once

    def _selected_wants(self):
        want = (self._mask_key(),) if self.sel else ()
        return want, bool(self.sel and mask_selection(self.sel[1]))

    def _mask_key(self):
        """What the render calls the selected mask: its layer's id, or (id, side) for a side mask."""
        if not self.sel:
            return None
        return (self.sel[0], self.sel[1][5:]) if self.sel[1].startswith("mask_") else self.sel[0]

    def mask_owner(self):
        """What the mask tools act on: the selected layer, or a view of the chosen side of a Combine layers layer."""
        lay = self.layer(self.sel[0]) if self.sel else None
        side = selection_side(self.sel[1]) if lay is not None else None
        return lay if side is None or not is_group(lay) else SideLayer(lay, side)

    def _hist_ids(self) -> tuple:
        """The selected filter, when its panel shows the histogram of its input (Curves, the stretches)."""
        lay = self.layer(self.sel[0]) if self.sel and self.sel[1] == "layer" else None
        if lay is None or lay["kind"] != "filter" or not FILTERS[lay["filter"]].get("hist"):
            return ()
        return (lay["id"],)

    def _attach_fusion(self, layers=None):
        """The combinations' master layers get their description from the model (client/postfusion.py: the block's
        settings, those being moved here on top), the others none; never saved."""
        fusion_of = getattr(self.s, "fusion_of", None)
        for lay in self.layers if layers is None else layers:
            if lay.get("kind") == "workflow" and lay.get("part") is None and fusion_of is not None:
                try:                                       # a combination's Output as it is: why, on its row
                    f, why = fusion_of(lay["pid"])
                except Exception:  # noqa: BLE001 - not readable now: nothing said
                    f, why = None, ""
                if f is not None or why in ("lightness", "after"):
                    self._plain_why[lay["pid"]] = why or "output"
                else:
                    self._plain_why.pop(lay["pid"], None)
            if not (is_group(lay) and lay.get("starless") == "output") or fusion_of is None:
                lay.pop("fusion", None)
                continue
            try:
                f, why = fusion_of(lay["pid"], self._live.get(lay["pid"]), tune=lay.get("tune"))
            except Exception:  # noqa: BLE001 - a combination that cannot be read now: its plain layer
                f, why = None, ""
            if f is None:
                lay.pop("fusion", None)
                if why in ("lightness", "after"):
                    self._plain_why[lay["pid"]] = why
            else:
                lay["fusion"] = f
                self._plain_why.pop(lay["pid"], None)

    def _send_draft(self):
        self._attach_fusion()
        self._draft_dirty = True
        want, overlay = self._selected_wants()
        self._final_kind = "draft"
        self._main_seq = self.worker.request(Request(kind="draft", layers=snapshot(self.layers),
                                                     level=self.draft_level, want=want, overlay=overlay))

    def _show_held(self):
        """The exact image is long to come: the half one in the meantime."""
        res, self._held = self._held, None
        if res is not None and res.seq == self._main_seq:
            self._done_seq = res.seq
            self.src.img1, self.src.seq1 = res.rgb, res.seq
            self.src.version += 1
            self.viewer.refresh()

    def _exact_wanted(self, res) -> bool:
        """After the release's image (GUI exploration 4, Q1 and Q4): the fitted view made again from the full
        resolution, when the screen shows the image reduced (zoomed in, the region is exact). The reduced image is not
        the file reduced: a filter sees its noise already averaged (a strong Saturation, the 1 px detail sliders), and
        so does the stretch."""
        if res.seq != self._main_seq or self.viewer.scale >= 0.5:
            return False
        return res.kind == "half" or (res.kind == "refine" and self._half_due)

    def _quarter_enough(self) -> bool:
        """On the CPU, when the screen shows the image at a quarter or less: the exact quarter is the picture."""
        cpu = (self._device or self.worker.device) == "cpu"
        return cpu and self.src.canvas is not None and self.viewer.scale <= QUARTER_SCALE

    def _send_final(self):
        self._attach_fusion()
        self._draft_t.stop()
        self._draft_dirty = False
        self._rendered_sig = self._visible_sig()
        want, overlay = self._selected_wants()
        cpu = (self._device or self.worker.device) == "cpu"
        # the CPU's exact quarter before the half is not the picture yet ("Updating…" until the half)
        self._final_kind = "refine" if self._quarter_enough() else "half"
        if self._quarter_enough():
            # the CPU, zoomed out: the exact quarter (about a second) is what the screen shows; the half comes when the
            # view is zoomed in past a quarter
            self._half_due = True
            self._main_seq = self.worker.request(Request(kind="refine", layers=snapshot(self.layers),
                                                         level=HALF_LEVEL + 1, want=want, thumbs=True, overlay=overlay,
                                                         hists=self._hist_ids()))
        else:
            # on the CPU the exact quarter comes first, then the half
            self._half_due = False
            self._main_seq = self.worker.request(Request(kind="half", layers=snapshot(self.layers), level=HALF_LEVEL,
                                                         want=want, thumbs=True, overlay=overlay, refine=cpu,
                                                         hists=self._hist_ids()))
        if self.b_cmp.isChecked():
            self._send_before()
        need = tuple(self._stale_thumbs())
        if need:
            self.worker.request(Request(kind="thumbs", layers=snapshot(self.layers), need_thumbs=need))

    def _send_before(self):
        other = self._compared()
        if other is None:
            self.worker.request(Request(kind="before", layers=snapshot(self.layers), level=HALF_LEVEL))
            return
        layers = load_layers(copy.deepcopy(other.get("layers")), {p.pid for p in self.s.paths})
        self._attach_fusion(layers)
        self.worker.request(Request(kind="before", layers=snapshot(layers), level=HALF_LEVEL, plain=True))

    def _stale_thumbs(self):
        for lay in self.layers:
            if is_pixel(lay) and (self._lthumbs.get(lay["id"], (None,))[0] != self._thumb_sig(lay) or
                                  (is_group(lay) and any((lay["id"], s) not in self._lthumbs for s in subs_of(lay)))):
                yield lay["id"]

    @staticmethod
    def _thumb_sig(lay):
        return (lay.get("pid"), lay.get("key"), lay.get("part"), json.dumps(lay["sliders"], sort_keys=True),
                json.dumps(lay.get("base"), sort_keys=True, default=str), lay.get("linked"), lay.get("stretch"),
                json.dumps(lay.get("fusion"), sort_keys=True, default=str))

    def _view_changed(self):
        self.zoom_lab.setText(_("Zoom {p}").format(p=with_unit(f"{self.viewer.scale * 100:.0f}", "%")))
        if self._half_due and self.viewer.scale > QUARTER_SCALE:     # zoomed in past the quarter: the half now
            self._half_due = False
            self._final_t.start(0)
        self._region_t.start(REGION_WAIT_MS)

    def visible_box(self):
        """The part of the image to render at full resolution when zoomed in: what the viewer shows with a quarter more
        on each side, or None (zoomed out: the half image is enough)."""
        v = self.viewer
        W, H = self.src.size()
        if self.src.canvas is None or v.scale < 0.5:
            return None
        ix0, iy0 = v.w_to_img(0, 0)
        ix1, iy1 = v.w_to_img(v.width(), v.height())
        ix0, iy0, ix1, iy1 = max(0.0, ix0), max(0.0, iy0), min(float(W), ix1), min(float(H), iy1)
        if ix1 - ix0 < 2 or iy1 - iy0 < 2:
            return None
        mx, my = (ix1 - ix0) / 4, (iy1 - iy0) / 4
        box = [max(0, ix0 - mx), max(0, iy0 - my), min(W, ix1 + mx), min(H, iy1 + my)]
        if (box[2] - box[0]) * (box[3] - box[1]) > REGION_MAX:
            box = [ix0, iy0, ix1, iy1]
        return int(box[0]), int(box[1]), min(W, int(math.ceil(box[2]))), min(H, int(math.ceil(box[3])))

    def _send_region(self):
        box = self.visible_box()
        if box is None:
            return
        f = self.src.full
        if f is not None and f[0][0] <= box[0] and f[0][1] <= box[1] and f[0][2] >= box[2] and f[0][3] >= box[3]:
            return
        if self.src.img1 is None:
            return
        self.worker.request(Request(kind="region", layers=snapshot(self.layers), level=0, box=box))

    def _empty_banner(self) -> str:
        """Why the canvas is empty: nothing to show of the layers there are (hidden), or no layer at all."""
        pixel = [x for x in self.layers if is_pixel(x)]
        if not pixel:
            return _("No layers yet")
        for lay in pixel:
            if is_group(lay) and lay.get("visible", True) and not any((lay.get("shown") or {}).values()):
                return _("Both images of “{name}” are hidden").format(name=layer_name(lay))
        if any(x.get("visible", True) and not (is_group(x) and not any((x.get("shown") or {}).values()))
               for x in pixel):
            return _("Waiting for the layers' results")        # visible layers whose image does not exist yet
        return _("Every image layer is hidden")

    def _on_result(self, res):
        if res.kind == "inspect":
            return                                              # Inspect's final image, rendered on the same worker
        if res.kind == "exact":                                 # the release's image from the full resolution (Q1)
            if res.tag == self._exact_wait:
                self._exact_wait = None
                self.viewer.update()
            if res.tag == self._main_seq and res.rgb is not None:     # nothing asked since its release
                self._done_seq = res.tag
                self._held = None
                self._hold_t.stop()
                self.src.img1, self.src.seq1 = res.rgb, res.seq
                self.src.version += 1
                self.viewer.refresh()
                self.rendered.emit(res)
            return
        if res.seq < self._last_seq.get(res.kind, 0):
            return                                              # an answer older than one already shown
        if res.kind != "thumbs":
            self._last_seq[res.kind] = res.seq
        if res.canvas is not None:
            self._canvas = res.canvas
            size = (res.canvas.W, res.canvas.H)
            if self.src.canvas != size:
                self.src.canvas = size
                self.viewer.set_source(self.src)
                if self.crop and not self.b_crop.isChecked():       # the crop kept, inside the image as it is now
                    self.viewer.crop_box = self._crop_shown()
            if self.ann_show["grid"] or self.ann_show["objects"]:
                self._want_annotations()
        if res.kind == "thumbs":
            self._take_layer_thumbs(res)
            return
        if res.kind in ("half", "refine") and set(res.missing) != self._missing_images:
            self._missing_images = set(res.missing)
            self._refresh_rows()
        if res.rgb is None or (self.layers and not any(is_pixel(x) and x.get("visible", True) for x in self.layers)):
            self.viewer.banner = self._empty_banner()
            self.src.img1 = self.src.img2 = self.src.full = None       # nothing to show: the neutral canvas, not the old picture
            self.src.version += 1
            self.viewer.refresh()
            self.viewer.update()
            return
        lost = []
        for x in self.layers:
            if is_pixel(x) and x.get("visible", True):
                if f"layer:{x['id']}" in self._missing_images:
                    lost.append(layer_name(x))
                lost += [f"{layer_name(x)} ({_(SUB_NAMES[sub]).lower()})" for sub in SUBS
                         if f"side:{x['id']}:{sub}" in self._missing_images]
        self.viewer.banner = (_("Not shown, its result cannot be read: {names}")
                              .format(names=", ".join(lost)))[:160] if lost else ""
        self.first_image = True
        if res.hists:
            self._hists.update(res.hists)
            self._show_hist()
        if res.kind in ("draft", "half", "refine"):
            if res.kind in ("draft", "refine"):
                self.src.img2, self.src.seq2, self.src.level2 = res.rgb, res.seq, res.level
                if res.kind == "draft":
                    self.draft_ms = (self.draft_ms + [res.ms])[-4:]
                    self._adapt_draft()
            elif self._exact_wanted(res) and self.src.img1 is not None:  # one change on screen per gesture (#87)
                self._held = res
                self._hold_t.start(HOLD_MS)
            else:
                self.src.img1, self.src.seq1 = res.rgb, res.seq
            if res.seq == self._main_seq and res.kind == self._final_kind and self._held is None:
                self._done_seq = res.seq
            if res.kind == "refine" and res.mask_thumbs:
                self._take_mask_thumbs(res)
            if res.kind == "refine" and res.layer_thumbs:
                self._take_layer_thumbs(res)
            sel = self._mask_key()
            m = res.masks.get(sel) if sel is not None else None
            self.mask1 = None if m is None else (_upsample(m, 2 ** (res.level - 1)) if res.level > 1 else m)
            self._set_overlay(res.overlay if self.sel and mask_selection(self.sel[1]) else None)
            self.src.version += 1
            if res.kind == "half":
                self._take_mask_thumbs(res)
                self._take_layer_thumbs(res)
                if self.viewer.scale >= 0.5:
                    self._region_t.start(60)
            self.viewer.refresh()
            if self._exact_wanted(res):
                self.worker.request(Request(kind="exact", layers=snapshot(self.layers), level=HALF_LEVEL,
                                            tag=res.seq))
                self._exact_wait = res.seq                     # the picture changes once more: still "Updating…"
        elif res.kind == "region":
            self.src.full = (res.box, res.rgb)
            self.src.version += 1
            self.viewer.refresh()
        elif res.kind == "before":
            self.before.img1 = res.rgb
            self.before.version += 1
            self.viewer.update()
        if res.notes:                                           # the engine fell back to the CPU: said until it changes
            self._notes = sorted(res.notes)
        self._hint()
        self.rendered.emit(res)

    def _show_hist(self):
        """The histogram of the selected filter's input into its panel."""
        h = self._hists.get(self.sel[0]) if self.sel else None
        if h is None:
            return
        if self.curves_editor is not None:
            self.curves_editor.set_hist(h)
        if self.function_graph is not None:
            self.function_graph.hist = h
            self.function_graph.update()

    def _adapt_draft(self):
        """The draft must stay within its budget (150 ms of the engine): on a slow machine it drops a level (to an
        eighth of the image at most), and goes back up to its device's level when that is fast again."""
        from .services.render import BUDGET
        if len(self.draft_ms) < 2:
            return
        avg = sum(self.draft_ms[-2:]) / 2
        base = CPU_DRAFT_LEVEL if self._device == "cpu" else DRAFT_LEVEL
        if self.draft_level < MAX_LEVEL and avg > 2.5 * 1000 * BUDGET["draft"]:
            self.draft_level, self.draft_ms = self.draft_level + 1, []
        elif self.draft_level > base and avg < 0.3 * 1000 * BUDGET["draft"]:
            self.draft_level, self.draft_ms = self.draft_level - 1, []

    def _on_failed(self, seq, kind, text):
        if kind in ("draft", "half", "refine"):
            self._failed_seq = max(self._failed_seq, seq)
            self.viewer.update()
        elif kind == "exact":                            # the half resolution stays: no more updating to wait for
            self._exact_wait = None
            self.viewer.update()
        if kind in ("draft", "half", "region", "before", "start"):
            first = _("Image not computed: {why}").format(why=plain_error(text))
            if first != self.viewer.banner:                  # said once; the program's error in the Log only
                log_detail(self.s, text)
            self.viewer.banner = first[:160]
            self.viewer.update()

    def _on_device(self, dev):
        self._device, self._notes = dev, []
        if dev == "cpu":
            self.draft_level = max(self.draft_level, CPU_DRAFT_LEVEL)

    def _take_mask_thumbs(self, res):
        for lid, t in res.mask_thumbs.items():
            self._mthumbs[lid] = pix(t)
        self._update_mask_thumb()

    def _take_layer_thumbs(self, res):
        changed = False
        for lid, t in res.layer_thumbs.items():
            lay = self.layer(lid[0] if isinstance(lid, tuple) else lid)      # (its id, one of its two images)
            if lay is not None:
                self._lthumbs[lid] = (self._thumb_sig(lay), pix(t))
                changed = True
        if changed:
            self._update_layer_thumbs()

    def _set_overlay(self, a):
        if a is None:
            self._overlay_img = None
            return
        a = np.ascontiguousarray(a)
        h, w = a.shape[:2]
        self._overlay_img = QImage(a.data, w, h, 4 * w, QImage.Format_RGBA8888).copy()

    # ------------------------------------------------------------ the annotations of the solution
    def set_annotations(self, grid: bool | None = None, objects: bool | None = None):
        """The sky grid and the named objects over the image, on or off (Export… proposes the same)."""
        if grid is not None:
            self.ann_show["grid"] = bool(grid)
        if objects is not None:
            self.ann_show["objects"] = bool(objects)
        for b, k in ((self.b_grid, "grid"), (self.b_names, "objects")):
            if b.isChecked() != self.ann_show[k]:
                b.blockSignals(True)
                b.setChecked(self.ann_show[k])
                b.blockSignals(False)
        if self.ann_show["grid"] or self.ann_show["objects"]:
            self._want_annotations()
        self.viewer.update()

    def _ann_pid(self):
        """The workflow whose solution the canvas uses: the bottom workflow layer (as the export does)."""
        return next((lay["pid"] for lay in reversed(self.layers) if lay["kind"] == "workflow"), None)

    def ann_problem(self) -> str:
        """Why the sky grid and the names cannot be drawn, '' when they can: the bottom workflow has no Astrometry, or it
        is not computed (the night test pass: they could be ticked and showed nothing, on screen and in the export)."""
        pid = self._ann_pid()
        p = self.path(pid) if pid is not None else None
        if p is None or not hasattr(self.s, "state_info"):
            return ""
        st = self.s.state_info(p, "astrometry")
        if st is None:
            # no Astrometry block of its own: a combination, like Align on, hands the plate solution of its reference
            # row on with its image; it is there when that stage is computed and the solution file came with it
            st = self._handed_on_solution(p, pid)
        if st is None:
            return _("The image at the bottom has no Astrometry: place the block in its workflow, or put a layer that "
                     "has one at the bottom")
        if st.state not in ("done", "stale"):
            return _("Astrometry is not computed yet for the image at the bottom")
        if st.state == "done" and getattr(st, "fallback", ""):      # it continued without a solution (its fallback)
            return _("The field of the image at the bottom was not identified: no plate solution to draw")
        return ""

    def _handed_on_solution(self, p, pid):
        """The state of the combination (or Align on) stage that outputs the bottom workflow's plate solution, None
        when there is none (no such stage, or it handed none on)."""
        from pathlib import Path

        from .. import api
        from ..core.workflow import COMBINE
        if not hasattr(self.s, "all_stages"):
            return None
        st = next((i for s in self.s.all_stages(p) if s == "align_on" or s in COMBINE
                   for i in [self.s.state_info(p, s)] if i is not None), None)
        if st is None or st.state not in ("done", "stale"):
            return st
        for which in ("current", "previous"):
            out = api.stage_outputs(self.s.p, pid, "astrometry", which)
            if out is not None and (Path(out.dir) / "wcs.fits").is_file():
                return st
        return None

    def _ann_buttons(self):
        """Sky grid and Object names greyed, the reason on hover, while there is no solution to draw."""
        why = self.ann_problem()
        if not why and not getattr(self, "_ann_warmed", False):
            self._ann_warmed = True                       # the catalogue is read now, not at the first tick (1 to 3 s)
            try:
                from .services import inspect as si
                si.service_of(self.s).request_job("post.warm", si.warm_annotations)
            except Exception:  # noqa: BLE001 - a session without the inspect service (tests)
                pass
        for b, k in ((self.b_grid, "grid"), (self.b_names, "objects")):
            b.setEnabled(not why)
            if why:
                if not b.property("why"):
                    b.setProperty("tip", b.toolTip())         # its own tip, back once there is a solution
                b.setToolTip(why)
                if b.isChecked():
                    self.set_annotations(**{k: False})
            elif b.property("why"):
                b.setToolTip(b.property("tip") or "")
            b.setProperty("why", why)

    def _want_annotations(self):
        c, pid = self._canvas, self._ann_pid()
        if c is None or pid is None:
            return
        key = (pid, c.W, c.H, tuple(c.origin))
        if key == self._ann_key:
            return
        try:
            from .services import inspect as si
            sv = si.service_of(self.s)
        except Exception:  # noqa: BLE001 - a session without the inspect service (tests): no annotations
            return
        if not getattr(self, "_ann_wired", False):
            sv.jobReady.connect(self._ann_ready)
            self._ann_wired = True
        self._ann_key = key
        self._ann_token = sv.request_job("post.sky", si.post_annotations, pid, c.W, c.H, tuple(c.origin))

    def _ann_ready(self, name: str, token: int, result):
        if name != "post.sky" or token != self._ann_token:
            return
        self._ann = result
        why = getattr(result, "why", "")
        n = len(getattr(result, "objects", None) or [])
        self.b_names.setToolTip(_("The known deep-sky objects of the field (Messier, NGC, IC, Sharpless and their "
                                  "common names), outlined at their size. Export… can burn them into the picture.")
                                + (NL + plural(n, "{n} named object", "{n} named objects") if n else "")
                                + (NL + _(why) if why else ""))
        self.viewer.update()

    def _ann_overlay(self, p: QPainter, v: Viewer):
        if self._ann is None or self.show_mask or not (self.ann_show["grid"] or self.ann_show["objects"]):
            return
        from .annotate import paint
        W, H = self.src.size()
        x0, y0, x1, y1 = v.crop_box if v.crop_box and not v.crop_edit else (0, 0, W, H)
        paint(p, self._ann, v, self.ann_show, QRectF(v.img_to_w(x0, y0), v.img_to_w(x1, y1)))

    def behind(self) -> bool:
        """The picture on the screen is behind the settings: a render is asked, or on its way, and not shown yet (one
        that failed says so in the banner instead)."""
        if self.src.img1 is None and self.src.img2 is None:
            return False                                         # nothing shown: the banner speaks
        if self._draft_t.isActive() or self._final_t.isActive() or self._held is not None:
            return True
        if self._failed_seq >= self._main_seq:
            return False
        if self._exact_wait is not None and self._exact_wait == self._main_seq:
            return True
        return self._done_seq < self._main_seq

    def _behind_tag(self, p: QPainter, v: Viewer):
        """As the Gradient view: "Updating…" over the top of the picture while it is behind the settings."""
        if self.behind():
            from .imageview import tag
            corner = v.img_to_w(0, 0) if self.src.canvas is not None else QPointF(0, 0)
            # on the picture's top left corner (not above it when it is letterboxed), inside the view when zoomed in
            tag(p, QPointF(min(max(corner.x(), 0.0), max(v.width() - 160.0, 0.0)) + 10,
                           min(max(corner.y(), 0.0), max(v.height() - 40.0, 0.0)) + 10), _("Updating…"))

    # ------------------------------------------------------------ the red overlay of the mask being edited
    def _overlay(self, p: QPainter, v: Viewer):
        if not self.sel or not mask_selection(self.sel[1]) or self._overlay_img is None or self.show_mask or not self.show_tint or \
                self.b_cmp.isChecked() or self.src.canvas is None:
            return
        W, H = self.src.size()
        x0, y0, x1, y1 = v.extent()                      # a cropped picture: the red inside the crop, as the picture
        p.save()
        p.setClipRect(QRectF(v.img_to_w(x0, y0), v.img_to_w(x1, y1)))
        p.drawImage(QRectF(v.img_to_w(0, 0), v.img_to_w(W, H)), self._overlay_img)
        p.restore()

    # ------------------------------------------------------------ undo
    def push(self, before=None, words: str = ""):
        """One undo step: the composition before a change. The change itself follows in the same call (or came just
        before it, for a slider let go), so the composition is saved when the event loop comes back, with the change.
        ``words``: the step's name, when the change itself does not say it (``Step``)."""
        self.undo.append(self._step(before if before is not None else snapshot(self.layers)))
        self.undo[-1].words = words
        self.redo.clear()
        self._undo_buttons()
        self._save_t.start(0)

    def _undo_buttons(self):
        self.b_undo.setEnabled(bool(self.undo))          # Post's own steps only: the graph's are undone in the graph
        self.b_redo.setEnabled(bool(self.redo))
        self.undoChanged.emit()

    def next_words(self, how: str = "undo") -> str:
        """The words of the step Post's next Undo (``how`` "undo") or Redo takes, nothing moved; "" when not said."""
        steps = list(self.undo if how == "undo" else self.redo)
        self._drop_empty(steps)                          # (on a copy: a name asked never changes the lists)
        if not steps:
            return ""
        top = steps[-1]
        if isinstance(top, ModelStep):                   # a step of the workflows made here: the model's words
            words = getattr(self.s, "next_words", None)
            return words(how) if words is not None and self._model_mine(top, "_" + how) else ""
        if getattr(top, "words", ""):
            return top.words
        now = self._step(snapshot(self.layers))
        return step_words(top, now) if how == "undo" else step_words(now, top)

    def _undo_tips(self):
        """The arrows' tips name their step ("Undo: Saturation of “Filter” (Ctrl+Z)"), as Edit > Undo does."""
        for b, how, plain, named in ((self.b_undo, "undo", _("Undo (Ctrl+Z)"), _("Undo: {what} (Ctrl+Z)")),
                                     (self.b_redo, "redo", _("Redo (Ctrl+Y)"), _("Redo: {what} (Ctrl+Y)"))):
            what = self.next_words(how) if b.isEnabled() else ""
            b.setToolTip(named.format(what=what) if what else plain)

    def _set_workflow_param(self, p, stage, name, value, shown=None):
        """A setting of the workflow shown here (its Output's channels): a step of Post's Undo too (2.4), undone by the
        workflows' Undo."""
        with self._from_post():
            self.s.set_param(p, stage, name, value)
            self._name_step(_("Setting “{name}” changed").format(name=_(shown or name)))
        self.undo.append(self._model_mark("_undo"))
        self.redo.clear()
        self._undo_buttons()

    def _delete_key(self):
        """The Delete key: the selected layer taken out, as the bin does (Undo brings it back); fields keep the key for
        their own text."""
        if self.sel and self.sel[1] == "layer" and self.b_del.isEnabled():
            self.delete()

    def _name_step(self, text: str):
        """The words of the workflows' step just made, for the Undo / Redo message ("Undone: <step>")."""
        name = getattr(self.s, "name_step", None)
        if name is not None:
            name(text)

    def _model_mark(self, on: str) -> ModelStep:
        """The mark of the model's latest step of its list ``on`` ("_undo" or "_redo"), just made."""
        stack = getattr(self.s, on, None)
        return ModelStep(stack[-1] if stack else None)

    def _model_mine(self, mark, on: str) -> bool:
        """The step this mark stands for is still the model's latest of its list: it is this version's to take back."""
        if mark.step is None:
            return True
        stack = getattr(self.s, on, None)
        return bool(stack) and stack[-1] is mark.step

    def do_undo(self) -> bool:
        """Post's last change undone, said in the status. Only Post's own steps (a "To Post" made in the graph is the
        graph's, undone there); with none, it says so. True when it was a step of the workflows Post made."""
        self._drop_empty(self.undo)
        if self.undo and not isinstance(self.undo[-1], ModelStep):
            before = self._step(snapshot(self.layers))
            before.words = getattr(self.undo[-1], "words", "")       # the step keeps its name in the Redo list
            self.redo.append(before)
            self.layers = self.undo.pop()
            self._restored()
            self._say_step("Undone", "Undone: {what}", before.words or step_words(self.layers, before))
            return False
        if not self.undo or not self._model_mine(self.undo[-1], "_undo"):
            can = getattr(self.s, "can_undo", None)
            if can is None or can(graph=True):            # what is left is the Graph's: said there
                self.toasted.emit(_("Nothing to undo here: the Graph's changes are undone in the Graph"))
            else:
                self.toasted.emit(_("Nothing more to undo in this version"))
            return False
        self.undo.pop()
        return self._model_step(getattr(self.s, "undo", None), "redo", "Undone", "Undone: {what}")

    def _drop_empty(self, steps: list):
        """The steps on top of an Undo / Redo list that would change nothing go (the laptop's last pass: Ctrl+Z said
        "Undone" and nothing changed): a step is kept before a gesture that may then change nothing (a slider let go
        where it was, a drop at its own place)."""
        now = None
        while steps and not isinstance(steps[-1], ModelStep):
            now = snapshot(self.layers) if now is None else now
            try:
                same = list(steps[-1]) == now and getattr(steps[-1], "crop", None) == self.crop
            except (ValueError, TypeError):              # an array in a layer: not compared, kept
                same = False
            if not same:
                return
            steps.pop()

    def do_redo(self) -> bool:
        self._drop_empty(self.redo)
        if self.redo and not isinstance(self.redo[-1], ModelStep):
            before = self._step(snapshot(self.layers))
            before.words = getattr(self.redo[-1], "words", "")
            self.undo.append(before)
            self.layers = self.redo.pop()
            self._restored()
            self._say_step("Redone", "Redone: {what}", before.words or step_words(before, self.layers))
            return False
        if not self.redo or not self._model_mine(self.redo[-1], "_redo"):
            self.toasted.emit(_("Nothing to redo here: the Graph's changes are redone in the Graph"))
            return False
        self.redo.pop()
        return self._model_step(getattr(self.s, "redo", None), "undo", "Redone", "Redone: {what}")

    def _say_step(self, plain: str, said: str, what: str):
        """What an Undo / Redo of Post did, in the window's status line (and the Log of the workflows' step)."""
        if what and not what.isspace():
            self.toasted.emit(_(said).format(what=what))
        else:
            self.toasted.emit(_(plain))

    def _model_step(self, fn, other: str, plain: str = "Undone", said: str = "Undone: {what}") -> bool:
        """A step of the workflows' Undo taken back or redone from here: the composition it puts back is not a step
        of Post's. ``other``: "undo" or "redo", the list that takes it back, the one of the version shown after the
        step (a deleted version brought back shows itself, with its own lists, #145)."""
        if fn is None:
            return False
        self._restoring = True
        try:
            done = fn()
            if done:
                getattr(self, other).append(self._model_mark("_undo" if other == "undo" else "_redo"))
        finally:
            self._restoring = False
        self.rebuild()
        self._undo_buttons()
        if done:
            what = getattr(self.s, "restored", "")
            if hasattr(self.s, "log"):
                self.s.log(said, what=what) if what else self.s.log(plain)
            self._say_step(plain, said, what)
        return done

    def _step(self, layers) -> Step:
        s = Step(layers)
        s.crop = self.crop
        return s

    def _restored(self):
        if not any(x.get("kind") == "filter" and x.get("filter") == "heal" and x["params"].get("spots")
                   for x in self.layers):
            self._heal_src = self._heal_off = None          # every spot undone: the hand-set source goes with them
        self._show_crop(getattr(self.layers, "crop", self.crop))
        if self.sel and not self.layer(self.sel[0]):
            self.sel = (self.layers[-1]["id"], "layer") if self.layers else None
        elif self.sel and mask_selection(self.sel[1]) and (own := self.mask_owner()) is not None \
                and own["mask"] is None:                    # the mask undone: its layer (or side) stays chosen
            self.sel = (self.sel[0], selection_side(self.sel[1]) or "layer")
        self.rebuild()
        self.request(final_only=True)
        self._save()
        self._back_to_saved()

    # ------------------------------------------------------------ the layers panel
    def thumb(self, lay) -> QPixmap:
        if is_synth(lay) or lay["kind"] == "sub":       # an image of a Combine layers layer (the real stars: its stars)
            hit = self._lthumbs.get((lay["target"], "synth") if is_synth(lay) else
                                    (lay["id"], "stars" if lay["sub"] == "real" else lay["sub"]))
            if hit is not None:
                return hit[1]
        if lay["kind"] == "filter" and not is_synth(lay):
            pm = QPixmap(TW, TH)
            pm.fill(QColor("#1d2129"))
            qp = QPainter(pm)
            draw_icon(qp, QRectF((TW - 20) / 2, (TH - 20) / 2, 20, 20), "filter_layer", "dim", 20)
            qp.end()
            return pm
        hit = None if lay["kind"] in ("sub", "filter") else self._lthumbs.get(lay["id"])
        if hit is not None:
            return hit[1]
        pm = QPixmap(TW, TH)
        pm.fill(QColor("#1d2129"))
        return pm

    def mask_thumb(self, lay):
        if lay["mask"] is None:
            return None
        pm = self._mthumbs.get(lay.get("mkey", lay["id"]))
        if pm is None:
            pm = pix(np.full((TH, TW), 255, np.uint8))             # an empty mask shows the layer everywhere
            self._mthumbs[lay.get("mkey", lay["id"])] = pm
        return pm

    def _update_mask_thumb(self):
        for row in self.rows:
            key = row.lay.get("mkey", row.lay["id"])
            if row.m is not None and key in self._mthumbs:
                row.m.setPixmap(small(self._mthumbs[key]))

    def _update_layer_thumbs(self):
        for row in self.rows:
            if is_pixel(row.lay) and row.lay["id"] in self._lthumbs:
                row.t.setPixmap(small(self._lthumbs[row.lay["id"]][1]))
            elif row.lay["kind"] == "sub" or is_synth(row.lay):
                row.t.setPixmap(small(self.thumb(row.lay)))

    def show_big(self, lay, at: QWidget):
        """A row's image, larger, beside the list while the mouse is on its thumbnail (#88)."""
        if self._big is None:
            self._big = QLabel(self, Qt.ToolTip)
            self._big.setStyleSheet("border: 1px solid #3a4150; background: #14161b;")
        self._big.setPixmap(self.thumb(lay).scaled(TW * 3, TH * 3, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self._big.adjustSize()
        pos = at.mapToGlobal(QPoint(0, 0))
        self._big.move(*self._big_place(pos, at.width(), self._big.width(), self._big.height(),
                                        (at.screen() or QApplication.primaryScreen()).availableGeometry()))
        self._big.show()

    @staticmethod
    def _big_place(pos, at_w: int, w: int, h: int, area) -> tuple[int, int]:
        """Where the enlarged thumbnail goes: left of the list, else right of the thumbnail when the screen has no room
        on the left; never off the screen."""
        x = pos.x() - w - 8
        if x < area.left():
            x = pos.x() + at_w + 8
        x = max(area.left(), min(x, area.right() - w))
        y = max(area.top(), min(pos.y(), area.bottom() - h))
        return x, y

    def hide_big(self):
        if self._big is not None:
            self._big.hide()

    def build_filter_menu(self):
        """The Filter button's menu: the filters by family."""
        self.filter_menu.clear()
        self._fill_filters(self.filter_menu)

    def _fill_filters(self, menu: QMenu, lay=None):
        """The filters by family into ``menu``; ``lay``: the layer they go on (chosen first: a layer's right-click)."""
        for fam in FAMILIES:
            sub = menu.addMenu(_(fam))
            sub.setToolTipsVisible(True)
            for k, f in FILTERS.items():
                if f["family"] == fam and not f.get("hidden"):
                    a = sub.addAction(_(f["name"]))
                    a.setToolTip(_(f["source"]))
                    a.triggered.connect(lambda _c=False, kk=k: (
                        lay is not None and self.select(lay["id"], lay["sub"] if lay["kind"] == "sub" else "layer"),
                        self.add_filter(kk)))

    def rebuild(self, props: bool = True):
        """The layers' rows built again; ``props``: the properties panel too (not for a visibility toggle, issue #24:
        its scroll and an open curves editor were lost)."""
        self._attach_fusion()
        self._update_stale()
        self._mask_button()
        self._ann_buttons()
        self.b_heal.blockSignals(True)                   # the tool is on while its layer is the one chosen
        self.b_heal.setChecked(self._heal_layer() is not None)
        self.b_heal.blockSignals(False)
        self.hide_big()
        for r in getattr(self, "rows", []):
            if r.name_edit is not None:                  # a name being typed is kept, as a click elsewhere keeps it
                r.end_name(True)
        clear(self.lbox)
        self.rows = []
        ids = {lay["id"] for lay in self.layers}
        self._collapsed = {c for c in self._collapsed if (c[0] if isinstance(c, tuple) else c) in ids}
        if self.sel and (lay := self.layer(self.sel[0])) is not None and lay.get("target"):
            self._collapsed -= {lay["target"], (lay["target"], lay.get("sub"))}     # the selected filter is shown
        for lay, depth in rows(self.layers, self._collapsed):
            sl = self.sel is not None and self.sel[0] == lay["id"]
            if lay["kind"] == "sub" or (sl and _sub_selected(self.sel[1])):   # a sub-layer: (its layer, it)
                sl = sl and self.sel[1] == lay.get("sub", "layer")
                if lay["kind"] == "sub":
                    n = sum(1 for f in children(self.layers, lay["id"]) if f.get("sub") == lay["sub"]) + \
                        (len(fusion_roles(self.layer(lay["id"]) or {})) if lay["sub"] == "starless" else 0)
                    row = LayerRow(self, lay, self.thumb(lay), self.mask_thumb(lay), sl and self.sel[1] == lay["sub"],
                                   self.sel == (lay["id"], "mask_" + lay["sub"]), depth, n,
                                   (lay["id"], lay["sub"]) in self._collapsed)
                else:
                    row = LayerRow(self, lay, self.thumb(lay), None, sl, False, depth)
                    self._plain_tip(row, lay)
                self.lbox.addWidget(row)
                self.rows.append(row)
                continue
            row = LayerRow(self, lay, self.thumb(lay), self.mask_thumb(lay), sl and self.sel[1] == "layer",
                           sl and self.sel[1] == "mask", depth,
                           len(children(self.layers, lay["id"])) + 2 * is_group(lay) if is_pixel(lay) else 0,
                           lay["id"] in self._collapsed)
            self._plain_tip(row, lay)
            self.lbox.addWidget(row)
            self.rows.append(row)
        waiting = [p for p in self.s.paths if not p.rendered]
        for p in waiting:
            b = QPushButton(_('"{name}": to compute  →').format(name=tr_name(p.name)))
            b.setObjectName("link")
            b.clicked.connect(lambda _=False, pid=p.pid: self.goResults.emit(pid))
            self.lbox.addWidget(b)
        self.lbox.addStretch(1)
        sel_row = next((r for r in self.rows if self.sel and r.lay["id"] == self.sel[0]), None)
        if sel_row is not None:
            # scrolled to the selected row up and down only: the list never stays scrolled sideways (it has no
            # horizontal bar, so its left column, the visibility boxes, could not be brought back)
            QTimer.singleShot(0, lambda: (self.lsc.ensureWidgetVisible(sel_row, 0, 10),
                                          self.lsc.horizontalScrollBar().setValue(0)))
        lay = self.layer(self.sel[0]) if self.sel else None
        image = self._image_of_group()                    # one of a group's images: it goes with its layer (B2, B3)
        self.b_dup.setEnabled(lay is not None and not is_synth(lay) and not image)  # one Synthetic stars layer per group
        # it names what it copies: the chosen filter is copied, not its layer (the tutorials' film expected the layer)
        self.b_dup.setToolTip(_("Duplicate “{name}” (Ctrl+J)").format(name=layer_name(lay)) if self.b_dup.isEnabled() else
                              _("Choose a layer first") if lay is None else image or
                              _("One Synthetic stars layer per Stars group: it cannot be duplicated"))
        self.b_del.setEnabled(lay is not None and not image and (not is_pixel(lay) or sum(map(is_pixel, self.layers)) > 1))
        # greyed, it says why (control round): the last image layer stays, or nothing chosen
        self.b_del.setToolTip(_("Delete the selected layer") if self.b_del.isEnabled() else
                              _("Choose a layer first") if lay is None else image or
                              _("The last image layer stays: Post needs an image (hide it with its eye)"))
        own = self.mask_owner()
        real = lay is not None and self.sel[1] == "real"           # the real stars have no mask of their own
        self.b_maskadd.setEnabled(own is not None and not real)
        masked = own is not None and own["mask"] is not None      # the accent: the selected layer has its mask
        side = own is not lay and own is not None
        icon_only(self.b_maskadd, "mask", (_("Edit the mask of the selected image") if side else
                                           _("Edit the mask of the selected layer")) if masked else
                  _("Add a mask to the selected image") if side else _("Add a mask to the selected layer"),
                  16, "selected" if masked else "normal", box=28)
        if real:                                                  # greyed, it says why
            self.b_maskadd.setToolTip(_("The Real stars row has no mask of its own: choose the layer or one of its "
                                        "two images"))
        self._undo_buttons()
        self.build_filter_menu()
        if props:
            self.build_props()

    def _plain_tip(self, row, lay):
        """A combination's Combine layers layer Post cannot make again from its filters: why, in its row's tooltip."""
        plain = (is_group(lay) and lay.get("starless") == "output" and not is_fusion(lay)) or \
            (lay.get("kind") == "workflow" and lay.get("part") is None)
        why = self._plain_why.get(lay.get("pid")) if plain else None
        if why:
            row.setToolTip({
                "lightness": _("A combination composed as lightness and colour: Post shows its Output as it is"),
                "after": _("Blocks follow the combination in its workflow: Post shows its Output as it is"),
                "output": _("The combination's Output as its run mixed it: its Combine layers layer shows each filter "
                            "and mixes them here")}[why])

    def select(self, lid, what):
        if self.b_crop.isChecked():
            self.b_crop.setChecked(False)
        if self.sel == (lid, what):
            return
        self.sel = (lid, what)
        lay = self.mask_owner() if mask_selection(what) else None
        if lay is not None and lay["mask"] is not None:
            brush = next((p for p in lay["mask"]["parts"] if p["src"] == "brush"), None)
            self.part_sel = brush["id"] if brush else None
        else:
            self.part_sel = None
        self.rebuild()
        self.request(final_only=True)

    def set_visible(self, lid, v):
        self.push()
        self.layer(lid)["visible"] = v
        self.rebuild(props=False)
        self.request(final_only=True)                    # no draft: the steps below it are kept, it is quick (#87)

    def set_shown(self, lid, sub, v):
        """One of a Stars and background layer's two images shown or not in it (issue #3)."""
        self.push()
        self.layer(lid)["shown"][sub] = v
        self.rebuild(props=False)
        self.request(final_only=True)

    def toggle_children(self, key):
        """The arrow: show or hide the rows under a layer (``key`` its id), or under one of a Stars and background
        layer's two images (``key`` (its id, the image)); the image does not change. A hidden filter that was selected
        leaves the selection to the row it is under."""
        self._collapsed ^= {key}
        if key in self._collapsed and isinstance(key, tuple) and key[1] == "stars" and self.sel == (key[0], "real"):
            self.sel = key                               # the real stars folded in their Stars group: the group (#93)
        cur = self.layer(self.sel[0]) if self.sel else None
        if key in self._collapsed and cur is not None and cur["kind"] == "filter":
            lid, sub = key if isinstance(key, tuple) else (key, None)
            if cur.get("target") == lid and (sub is None or cur.get("sub") == sub):
                self.sel, self.part_sel = (lid, sub or "layer"), None
        self.rebuild()

    # ---- drag and drop (the reference image editor's): onto a layer nests, between rows places; the line says at which depth
    def drag_start(self, row):
        eff = QGraphicsOpacityEffect(row)
        eff.setOpacity(0.45)
        row.setGraphicsEffect(eff)
        for r in self.rows:                              # a layer is dragged with its children
            if r.lay.get("target") == row.lay["id"] and r.depth:
                e = QGraphicsOpacityEffect(r)
                e.setOpacity(0.45)
                r.setGraphicsEffect(e)
        self.lsc.viewport().setCursor(Qt.ClosedHandCursor)

    def drop_at(self, lay, x: float, y: float) -> dict | None:
        """Where a row dropped at (x, y) of the panel goes: {parent, index, row (nest) | y, x0 (line), text}. A filter
        over the middle of a pixel layer's row nests in it (on top of its children); between two rows it goes there -
        among a layer's children when the line is indented (the gap sits in a group, or the pointer is to the right
        after a group), else as a row of its own that applies to everything below. A pixel layer goes between top-level
        rows only."""
        rs = self.rows
        if not rs:
            return None
        is_f = lay["kind"] == "filter"
        mine = {lay["id"]} | ({c["id"] for c in children(self.layers, lay["id"])} if is_pixel(lay) else set())
        gap = len(rs) if y > rs[-1].geometry().bottom() else 0
        for i, r in enumerate(rs):
            g = r.geometry()
            if g.top() <= y <= g.bottom():
                rel = (y - g.top()) / max(1, g.height())
                if is_f and is_pixel(r.lay) and r.lay["id"] != lay.get("target", -1) and 0.28 <= rel <= 0.72:
                    return {"parent": r.lay["id"], "index": 0, "row": i,
                            "text": _('Applies to "{name}" only').format(name=layer_name(r.lay))}
                if is_f and is_pixel(r.lay) and 0.28 <= rel <= 0.72:
                    return None                         # onto its own layer: where it is
                if is_f and r.lay["kind"] == "sub" and 0.28 <= rel <= 0.72:
                    # onto the real stars: the bottom of the Stars group, under the synthetic stars (#93)
                    real = r.lay["sub"] == "real"
                    index = len([c for c in children(self.layers, r.lay["target"]) if c["id"] not in mine]) if real else 0
                    return {"parent": r.lay["target"], "index": index, "row": i, "sub": "stars" if real else r.lay["sub"],
                            "text": _('Applies to "{name}" only').format(
                                name=_(sub_name(self.layer(r.lay["id"]) or r.lay, r.lay["sub"])))}
                gap = i if rel < 0.5 else i + 1
                break
        above = rs[gap - 1] if gap > 0 else None
        below = rs[gap] if gap < len(rs) else None
        indent_x = self.rows[0].geometry().left() + 60
        parent = None
        if below is not None and below.depth:           # inside a group
            parent = below.lay["target"]
        elif above is not None and is_f and x >= indent_x and (above.depth or is_pixel(above.lay)) and \
                (above.depth or above.lay["id"] not in self._collapsed):
            parent = above.lay["target"] if above.depth else above.lay["id"]
        if parent is not None and (not is_f or parent in mine):
            # a pixel layer is never nested: the line goes after the group
            while gap < len(rs) and rs[gap].depth:
                gap += 1
            above = rs[gap - 1] if gap > 0 else None
            below = rs[gap] if gap < len(rs) else None
            parent = None
        if parent is not None:
            kids = [c["id"] for c in children(self.layers, parent) if c["id"] not in mine]
            before = [r.lay["id"] for r in rs[:gap] if r.depth and r.lay.get("target") == parent and
                      r.lay["id"] not in mine]
            index = len([k for k in kids if k in before])
            text = _('Applies to "{name}" only').format(name=layer_name(self.layer(parent)))
            depth = 1
            if is_group(self.layer(parent)):              # between the filters of one of its two images: in it
                near = below if below is not None and below.depth in (2, 3) else \
                    above if above is not None and (above.depth in (2, 3) or above.lay["kind"] == "sub") else None
                sub = near.lay.get("sub") if near is not None else None
                if sub:
                    text = _('Applies to "{name}" only').format(name=_(sub_name(self.layer(parent), sub)))
                    depth = 3 if sub_role(sub) is not None else 2
                    sub = "stars" if sub == "real" else sub        # in the Stars group, above the real stars
                return {"parent": parent, "index": index, "sub": sub, "text": text,
                        "y": below.geometry().top() - 2 if below is not None else rs[-1].geometry().bottom() + 1,
                        "x0": rs[0].geometry().left() + 30 + 4 + INDENT * depth}
        else:
            index = len([r for r in rs[:gap] if not r.depth and r.lay["id"] not in mine])
            text = _("On all below") if is_f else ""
            depth = 0
        line_y = below.geometry().top() - 2 if below is not None else rs[-1].geometry().bottom() + 1
        x0 = rs[0].geometry().left() + (34 + INDENT if depth else 30)
        return {"parent": parent, "index": index, "y": line_y, "x0": x0, "text": text}

    def drag_move(self, row, gpos):
        holder = self.lbox.parentWidget()
        vp = self.lsc.viewport()
        y_vp = vp.mapFromGlobal(gpos).y()               # near an edge of the list: it scrolls
        sb = self.lsc.verticalScrollBar()
        if y_vp < 24:
            sb.setValue(sb.value() - 14)
        elif y_vp > vp.height() - 24:
            sb.setValue(sb.value() + 14)
        pt = holder.mapFromGlobal(gpos)
        d = self.drop_at(row.lay, pt.x(), pt.y())
        self._drop = d
        for r in self.rows:
            nest = d is not None and d.get("row") is not None and r is self.rows[d["row"]]
            r.set_drop(nest, d["text"] if nest else "")
        self.drop_line.hide()
        self.drop_dot.hide()
        self.drop_text.hide()
        if d is None:
            self.lsc.viewport().setCursor(Qt.ForbiddenCursor)
            return
        self.lsc.viewport().setCursor(Qt.ClosedHandCursor)
        w = holder.width()
        if d.get("row") is None:
            self.drop_line.setGeometry(d["x0"], d["y"] - 1, w - d["x0"] - 4, 3)
            self.drop_dot.move(d["x0"] - 7, d["y"] - 4)
            self.drop_line.show()
            self.drop_dot.show()
            self.drop_line.raise_()
            self.drop_dot.raise_()
            below = next((r for r in self.rows if r.geometry().top() - 2 == d["y"]), None)
            where = _("Above “{name}”").format(name=below.n.full_text()) if below is not None else _("At the bottom")
            text = where + (" · " + d["text"] if d["text"] else "")
            if self.drop_text.parent() is not holder:
                self.drop_text.setParent(holder)
            room = max(w - d["x0"] - 8, 40)
            self.drop_text.setText(self.drop_text.fontMetrics().elidedText(text, Qt.ElideRight, room - 10))
            self.drop_text.setToolTip(text)
            self.drop_text.adjustSize()
            self.drop_text.move(d["x0"] + 4, max(d["y"] - self.drop_text.height() - 1, 0))
            self.drop_text.show()
            self.drop_text.raise_()

    def drag_cancel(self):
        """The drag ends without a drop: the hints go, nothing moves."""
        self._drop = None
        self.drop_line.hide()
        self.drop_dot.hide()
        self.drop_text.hide()
        self.lsc.viewport().unsetCursor()
        for r in self.rows:
            r.set_drop(False)
            r.setGraphicsEffect(None)

    def drag_end(self, row, gpos):
        d = self._drop
        self.drag_cancel()
        if d is not None:
            if d["parent"] is not None:
                self._collapsed.discard(d["parent"])
            moved = self.move(row.lay["id"], d["parent"], d["index"])
            if row.lay["kind"] == "filter" and d["parent"] is not None and is_group(self.layer(d["parent"])):
                self._set_sub(self.layer(row.lay["id"]), d.get("sub"), push=not moved)

    def _set_sub(self, fl, sub, push=True):
        """A filter of a Stars and background layer put on one of its two images (``sub``), or on the two together."""
        if fl.get("sub") == sub:
            return
        if push:
            self.push()
        fl["sub"] = sub
        self.rebuild()
        self.request(final_only=True)

    def move(self, lid, parent=None, index=0) -> bool:
        """Move a layer as a drop does (postengine.move): one undo step, nothing when it lands where it was."""
        before = snapshot(self.layers)
        new = move(self.layers, lid, parent, index)
        if [(x["id"], x.get("target")) for x in new] == [(x["id"], x.get("target")) for x in before]:
            return False
        self.push(before)
        self.layers = new
        self.rebuild()
        self.request(final_only=True)
        return True

    def scope_choices(self, fl) -> list[tuple[str, int | None]]:
        """What a filter can apply to: (text, target): the layer it is nested in, else the pixel layer just under its
        row; and everything below."""
        tgt = self.layer(fl.get("target")) if fl.get("target") else None
        if tgt is None:
            t = tree(self.layers)
            i = next(k for k, (x, _kids) in enumerate(t) if x["id"] == fl["id"])
            tgt = next((x for x, _kids in t[i + 1:] if is_pixel(x)), None)
        out = [(_('"{name}" only').format(name=layer_name(tgt)), tgt["id"])] if tgt else []
        if tgt is not None and is_group(tgt):           # or one of its two images (issue #3)
            out += [(_('"{name}" only').format(name=_(sub_name(tgt, sub))), (tgt["id"], sub)) for sub in subs_of(tgt)]
        return out + [(_("All below"), None)]

    @staticmethod
    def scope_of(fl):
        """What a filter applies to, as ``scope_choices`` says it."""
        return (fl["target"], fl["sub"]) if fl.get("sub") else fl.get("target")

    def layer_menu(self, lay, gpos):
        self.layer_menu_of(lay).exec(gpos)

    def layer_menu_of(self, lay) -> QMenu:
        """A layer's right-click menu."""
        m = QMenu(self)
        if is_synth(lay):                                # its mask, and out with its block (#93)
            if lay["mask"] is None:
                m.addAction(_("Add mask")).triggered.connect(lambda: self.add_mask(lay))
            else:
                m.addAction(_("Delete mask")).triggered.connect(lambda: self.remove_mask(lay))
            self._mask_clip_actions(m, lay)
            m.addSeparator()
            a = m.addAction(_("Delete"))
            a.setToolTip(_("Takes them out, their block in the graph too · Ctrl+Z"))
            m.setToolTipsVisible(True)
            a.triggered.connect(lambda: self.remove_synth(self.layer(lay["target"])))
            return m
        self._fill_filters(m.addMenu(_("Add a filter")), lay)     # #120: where one looks for them first
        if lay["kind"] == "sub":                         # one of a layer's images: its filters only; the Stars group,
            if lay["sub"] == "stars":                    # its synthetic stars too (#93)
                m.addSeparator()
                self._synth_action(m, self.layer(lay["id"]))
            m.addSeparator()
            self._mask_clip_actions(m, lay)
            m.addSeparator()                             # what it cannot do, greyed with why (B2, B3)
            why = _("One of the images of “{name}”: it stays in its layer. Duplicate, rename, move or delete the "
                    "layer itself").format(name=layer_name(self.layer(lay["id"]) or lay))
            for txt in ("Duplicate", "Rename", "Delete"):
                a = m.addAction(_(txt))
                a.setEnabled(False)
                a.setToolTip(why)
            m.setToolTipsVisible(True)
            return m
        m.addSeparator()
        if is_group(lay):
            self._synth_action(m, lay)
            m.addSeparator()
        m.addAction(_("Duplicate")).triggered.connect(lambda: (self.select(lay["id"], "layer"), self.duplicate()))
        if lay["mask"] is None:
            m.addAction(_("Add mask")).triggered.connect(lambda: self.add_mask(lay))
        else:
            m.addAction(_("Delete mask")).triggered.connect(lambda: self.remove_mask(lay))
        self._mask_clip_actions(m, lay)
        if lay["kind"] == "filter":
            m.addSeparator()
            for _txt, val in self.scope_choices(lay):
                a = m.addAction(_("On all below") if val is None else _("Applies to {what}").format(
                    what=_txt))
                a.setCheckable(True)
                a.setChecked(self.scope_of(lay) == val)
                a.triggered.connect(lambda _c=False, t=val: self.set_target(lay, t) if self.scope_of(lay) != t else None)
            m.addSeparator()
        if lay["kind"] == "workflow":
            m.addSeparator()
            m.addAction(_("Show workflow  →")).triggered.connect(lambda: self.goResults.emit(lay["pid"]))
            q = self.path(lay["pid"])
            if getattr(self.s, "is_plain_image", lambda x: False)(q):      # told from its file when it can be
                a = m.addAction(_("Already stretched"))
                a.setCheckable(True)
                a.setChecked(bool(q.source.get("stretched")))
                a.toggled.connect(lambda on, x=q: self.set_stretched(x, on))
        m.addAction(_("Rename")).triggered.connect(lambda: self.rename(lay))
        if not is_pixel(lay) or sum(map(is_pixel, self.layers)) > 1:
            m.addSeparator()
            m.addAction(_("Delete")).triggered.connect(lambda: (self.select(lay["id"], "layer"), self.delete()))
        return m

    def set_stretched(self, q, stretched: bool):
        """An image's kind said by the user: computed again."""
        self.s.set_image_file(q, stretched=stretched)
        self.runAsked.emit()

    # ---- the Synthetic stars of a Stars group (#93): added and removed here, their block with them in the graph
    def _synth_action(self, menu: QMenu, group):
        """"Synthetic stars", ticked when the Stars group of ``group`` holds them: a click places their block in the
        graph or takes it out; greyed with its reason on hover when the block cannot be placed."""
        if group is None or not is_group(group):
            return
        on = has_synth(self.layers, group["id"])
        why = "" if on else self.synth_problem(group)
        a = menu.addAction(_("Synthetic stars"))
        a.setCheckable(True)
        a.setChecked(on)
        a.setEnabled(not why)
        a.setToolTip(why or (_("Takes them out, their block in the graph too · Ctrl+Z") if on else
                             _("Places the Synthetic stars block after Stars/background in the graph: they replace "
                               "the real stars, paint their mask where the real ones stay · Ctrl+Z")))
        menu.setToolTipsVisible(True)
        a.triggered.connect(lambda _c=False: self.remove_synth(group) if on else self.add_synth(group))

    def synth_problem(self, group) -> str:
        """Why the Synthetic stars block cannot be placed in the workflow of a Combine layers layer, "" when it can."""
        p = self.path(stars_side(group))             # the block lives where the stars come from (a stars row, a combination)
        if p is None or not hasattr(self.s, "insert_block"):
            return _("not in workflow")
        return self.s.palette_reason(p, "synth") if hasattr(self.s, "palette_reason") else ""

    def _model_change(self, fn, toast: str) -> bool:
        """A change of the graph made from here (``fn``): one step of the workflows' Undo, which Post's Undo takes back
        (MODEL_STEP), the composition it puts back being none of Post's own steps. False when the model refused it."""
        self.flush()
        self._restoring = True
        try:
            with self._from_post():
                fn()
        except ValueError as e:
            self.toasted.emit(str(e))
            return False
        finally:
            self._restoring = False
        self._name_step(toast)
        self.undo.append(self._model_mark("_undo"))
        self.redo.clear()
        self._undo_buttons()
        self.toasted.emit(toast)
        return True

    def add_synth(self, group):
        """Synthetic stars in the Stars group of a Combine layers layer: their block placed in the graph after
        Stars/background, as the palette places it; the layer comes with it (``with_synth_rows``, its mask open),
        selected."""
        p = self.path(stars_side(group))
        if p is None or self.synth_problem(group):
            return

        def place():
            with self.s.batch():
                self.s.insert_block(p, "synth")
                if hasattr(self.s, "log"):
                    self.s.log('{block} added to "{name}"', block=_("Synthetic stars"), name=p.name)
        if self._model_change(place, _('Synthetic stars added to "{name}": its block is in the graph · Ctrl+Z').format(
                name=p.name)):
            row = next((x for x in self.layers if is_synth(x) and x.get("target") == group["id"]), None)
            if row is not None:
                self._collapsed -= {group["id"], (group["id"], "stars")}
                self.sel, self.part_sel = (row["id"], "layer"), None
                self.rebuild()

    def remove_synth(self, group):
        """The Synthetic stars of a Stars group taken out, their block with them from the graph: one step of Undo,
        which brings both back, the layer with its mask (the composition without it is saved in the same step)."""
        from .model import node_key
        p = self.path(stars_side(group))
        key = next((node_key(n) for n in self.s.graph(p) if n["block"] == "synth"), None) \
            if p is not None and hasattr(self.s, "graph") else None
        why = self.s.removal_problem(p, key) if key is not None else _("not in workflow")
        if why:
            self.toasted.emit(_("{block} kept: {why}").format(block=_("Synthetic stars"), why=why))
            return

        def take_out():
            with self.s.batch():
                if hasattr(self.s, "log"):
                    self.s.log('{block} removed from "{name}"', block=_("Synthetic stars"), name=p.name)
                self.s.remove_node(p, key)
                self.layers = [x for x in self.layers if not (is_synth(x) and x.get("target") == group["id"])]
                self._save()
        if self._model_change(take_out, _('Synthetic stars removed from "{name}": its block too · Ctrl+Z').format(
                name=p.name)):
            if self.sel and self.layer(self.sel[0]) is None:
                self.sel = (group["id"], "stars")
            self.rebuild()
            self.request(final_only=True)

    def _dock_side(self, w: QWidget):
        """A side docked again, at its place in the splitter."""
        if w is self.left:
            self.split.insertWidget(0, w)
        else:
            self.split.addWidget(w)
        self._fit_split()

    def resizeEvent(self, e):                                       # noqa: N802 - Qt's name
        super().resizeEvent(e)
        # no width chosen for the layers: their default follows the window's (D6)
        if not prefs.get("post_layers_width") and self.left.parent() is self.split and                 layers_default(e.oldSize().width()) != layers_default(self.width()):
            self._fit_split()

    def showEvent(self, e):                                         # noqa: N802 - Qt's name
        """Shown (the Post tab clicked): the keyboard comes to the image, so that R, C, V… answer at once (the night
        test pass: it stayed on the tab's button)."""
        super().showEvent(e)
        fw = QApplication.focusWidget()
        if fw is None or not self.isAncestorOf(fw):
            QTimer.singleShot(0, lambda: self.isVisible() and self.viewer.setFocus(Qt.OtherFocusReason))

    def _tool_on(self) -> str | None:
        """The tool Escape ends before it leaves Post: the crop being drawn, Spot healing, a mask being painted."""
        if self.b_crop.isChecked():
            return "crop"
        if self.b_heal.isChecked():
            return "heal"
        return "brush" if self.viewer.mode == "brush" else None

    def event(self, e):                                             # noqa: N802 - Qt's name
        # Escape is the tool's first, as in the reference image editors (#143: it left Post for the graph, through
        # Inspect's shortcut, the crop drawn so far kept and the tool still on); with no tool on, it leaves Post
        if e.type() == QEvent.ShortcutOverride and e.key() == Qt.Key_Escape and e.modifiers() == Qt.NoModifier and \
                self._tool_on():
            e.accept()
            return True
        return super().event(e)

    def keyPressEvent(self, e):                                     # noqa: N802 - Qt's name
        tool = self._tool_on() if e.key() == Qt.Key_Escape and e.modifiers() == Qt.NoModifier else None
        if tool == "crop":
            self.crop_cancel()                           # as its Cancel button
        elif tool == "heal":
            self.b_heal.setChecked(False)
        elif tool == "brush":
            self.part_sel = None
            self.build_props()
        else:
            super().keyPressEvent(e)
            return
        e.accept()

    def _fit_split(self):
        """The docked sides at their widths (the user's, else the default), the image the rest."""
        wide = {id(self.left): int(prefs.get("post_layers_width") or layers_default(self.width())),
                id(self.right): int(prefs.get("post_panel_width") or PANEL_W)}
        ws = [self.split.widget(i) for i in range(self.split.count())]
        for i, w in enumerate(ws):
            self.split.setStretchFactor(i, 1 if w is self.centre else 0)
        self.split.setSizes([wide.get(id(w), 10_000) for w in ws])

    def set_layout(self, left: str, right: str):
        """The two sides as the window chose for this size range (each docked | icons | auto | hidden)."""
        self.ldock.set_state(left)
        self.dock.set_state(right)

    def set_compact(self, on: bool):
        self.top_bar.setContentsMargins(*((8, 2, 8, 2) if on else (10, 6, 10, 6)))

    def _panel_moved(self, *_a):
        for w, key in ((self.left, "post_layers_width"), (self.right, "post_panel_width")):
            if w.parent() is self.split and w.isVisible():
                prefs.set(key, int(w.width()))

    def rename(self, lay, new: str | None = None):
        """The layer named ``new``; without it, its name typed in place in its row (``LayerRow.edit_name``), after the
        menu that asked it has closed and given the focus back."""
        old = layer_name(lay)
        if new is None:
            QTimer.singleShot(0, lambda: next((r.edit_name() for r in self.rows
                                               if r.lay["kind"] != "sub" and r.lay["id"] == lay["id"]), None))
            return
        if not new or new == old:
            return
        p = self.path(lay["pid"]) if lay["kind"] == "workflow" and lay.get("part") is None else None
        if p is not None and lay.get("name") == p.name and hasattr(self.s, "name_problem"):
            # the layer named after its workflow: the workflow (and its wire in the graph) take the name too, its
            # layers follow (control round); one step of the workflows' Undo, as a branch removed with its layer
            why = self.s.name_problem(new, pid=p.pid)
            if why:
                self.toasted.emit(_("Name refused: {why}").format(why=why))
                return
            self._save_t.stop()
            self._save()                                  # the layers shown into the project, then renamed there
            self._restoring = True
            try:
                with self._from_post():
                    self.s.rename(p.pid, new)
                self._post_changed(None)                  # and taken back here with their new names
            finally:
                self._restoring = False
            self._name_step(_("Workflow renamed “{name}”").format(name=new))
            self.undo.append(self._model_mark("_undo"))
            self.redo.clear()
            self._undo_buttons()
            self.toasted.emit(_("“{old}” renamed “{name}”: its workflow too · Ctrl+Z").format(old=old, name=new))
            return
        self.push()
        lay["name"] = new
        self.rebuild()

    # ------------------------------------------------------------ layer actions
    def add_image_layer(self, name: str, source, key: str) -> dict:
        """A stage's image as a layer, on top: stretched automatically, blendable, maskable (see PostPage3). Its name
        is shown translated where the catalogue has it (Inspect gives it in English); an image Inspect showed with a
        neutral sky (``source.neutral``: still in the camera's colours) has its channels stretched apart, so that its
        sky is neutral here too."""
        self.worker.set_image(key, source)
        self._missing_images.discard(key)
        self.push()
        lay = new_image_layer(key, name)
        if getattr(source, "neutral", False):
            lay["linked"] = False
        self.layers.insert(0, lay)
        self.layers = normalize(self.layers)
        self.sel = (lay["id"], "layer")
        self.part_sel = None
        self.rebuild()
        self.request(final_only=True)
        return lay

    def _insert_above_sel(self, lay):
        i = self.index(self.sel[0]) if self.sel else 0
        self.layers.insert(i, lay)
        self.layers = normalize(self.layers)
        self.sel = (lay["id"], "layer")
        self.part_sel = None

    def add_filter(self, key):
        """A new filter is nested in the chosen pixel layer, on top of its children (or goes where the chosen filter
        is: in its layer, or above it on everything below). On the real stars of a Stars group: at the group's bottom,
        under the synthetic stars, so that it acts on the real stars alone (#93)."""
        cur = self.layer(self.sel[0]) if self.sel else None
        target = sub = None
        real = cur is not None and is_pixel(cur) and self.sel[1] == "real"
        if cur is not None:
            target = cur["id"] if is_pixel(cur) else cur.get("target")
            sub = "stars" if real else selection_side(self.sel[1]) or (None if is_pixel(cur) else cur.get("sub"))
        self.push()
        fl = new_filter_layer(key, target, sub)
        same = sum(1 for lay in self.layers if lay["kind"] == "filter" and lay.get("filter") == key)
        if same:                                        # a second Curves is "Curves · 2" (issue #15: no twin names)
            fl["name"] = f"{fl['name']} · {same + 1}"
        if target is not None:                          # on top of that layer's own filters (real stars: under them)
            group = [lay for lay in self.layers if lay.get("target") == target]
            self.sel = ((group[0] if group and not real else self.layer(target))["id"], "layer")
        self._insert_above_sel(fl)
        self.rebuild()
        self.request(final_only=True)
        return fl

    def set_target(self, fl, target):
        """Nest a filter in a pixel layer (on top of its children), or in one of a Stars and background layer's two
        images (``target`` (its id, the image)), or take it out: then it is the row just above that layer and applies
        to everything below."""
        sub = None
        if isinstance(target, tuple):
            target, sub = target
        if target is not None and target == fl.get("target"):
            self._set_sub(fl, sub)
            return
        if target is None:
            old = fl.get("target")
            t = [x["id"] for x, _kids in tree(self.layers)]
            idx = t.index(old) if old in t else t.index(fl["id"])
        else:
            idx = 0
        self._collapsed.discard(target)
        moved = self.move(fl["id"], target, idx)
        if sub:
            self._set_sub(self.layer(fl["id"]), sub, push=not moved)

    @staticmethod
    def _copy(lay):
        d = snapshot([lay])[0]
        d["id"] = new_id()
        for m in [d["mask"] or {}, *(d.get("side_masks") or {}).values()]:
            for part in m.get("parts", []):
                part["id"] = new_id()
        return d

    def _image_of_group(self) -> str:
        """When the selection is one of a Combine layers layer's images (or its mask): why it cannot be duplicated,
        renamed, moved or deleted apart (the laptop's B2, B3: Duplicate with Stars chosen copied the whole layer, and
        nothing said the images stay in place); else ""."""
        if not self.sel or not _sub_selected(self.sel[1]) and selection_side(self.sel[1]) is None:
            return ""
        lay = self.layer(self.sel[0])
        return _("One of the images of “{name}”: it stays in its layer. Duplicate, rename, move or delete the layer "
                 "itself").format(name=layer_name(lay)) if lay is not None else ""

    def duplicate(self):
        """A copy above; a pixel layer comes with its own filters."""
        lay = self.layer(self.sel[0]) if self.sel else None
        if lay is None or self._image_of_group():
            return
        self.push()
        d = self._copy(lay)
        d["name"] = lay["name"] + COPY
        d["visible"] = True
        top = self.index(lay["id"])
        if is_pixel(lay):
            own = [x for x in self.layers if x.get("target") == lay["id"]]
            copies = []
            for f in own:
                c = self._copy(f)
                c["target"] = d["id"]
                copies.append(c)
            if own:
                top = self.index(own[0]["id"])
            self.layers[top:top] = copies + [d]
        else:
            self.layers.insert(top, d)
        self.layers = normalize(self.layers)
        self.sel = (d["id"], "layer")
        self.part_sel = None
        self.rebuild()
        self.request(final_only=True)

    def delete(self):
        """A pixel layer goes with its own filters; the last pixel layer stays."""
        lay = self.layer(self.sel[0]) if self.sel else None
        if self._image_of_group():
            return
        others = [x for x in self.layers if x["id"] != lay["id"] and x.get("target") != lay["id"]] if lay else []
        if lay is None or not any(is_pixel(x) for x in others) and is_pixel(lay):
            return
        if is_synth(lay):                                # the Synthetic stars: their block goes with them (#93)
            self.remove_synth(self.layer(lay["target"]))
            return
        self.push()
        i = self.index(lay["id"])
        self.layers = normalize(others)
        L = self.layers
        self.sel = (L[min(i, len(L) - 1)]["id"], "layer") if L else None
        self.rebuild()
        self.request(final_only=True)

    def mask_button(self):
        lay = self.mask_owner()
        if lay is None or self.sel[1] == "real":
            return
        if lay["mask"] is None:
            self.add_mask(lay)
        else:
            self.select(lay["id"], self._mask_kind(lay))

    @staticmethod
    def _mask_kind(lay):
        """The selection of a mask: the layer's own, or one side's of a Combine layers layer."""
        return "mask_" + lay["sub"] if isinstance(lay, SideLayer) else "mask"

    def add_mask(self, lay):
        self.push()
        lay["mask"] = new_mask()
        self._mthumbs.pop((lay["id"], lay["sub"]) if isinstance(lay, SideLayer) else lay["id"], None)
        self.sel = None
        self.select(lay["id"], self._mask_kind(lay))

    def remove_mask(self, lay):
        self.push()
        lay["mask"] = None
        self._mthumbs.pop((lay["id"], lay["sub"]) if isinstance(lay, SideLayer) else lay["id"], None)
        self.sel = (lay["id"], lay["sub"] if isinstance(lay, SideLayer) else "layer")
        self.rebuild()
        self.request(final_only=True)

    # ---- a mask put on another layer (the reference image editors' Copy mask / Paste mask, and its thumbnail dragged)
    def _mask_clip_actions(self, menu: QMenu, lay):
        """A row's "Copy mask" (when it has one) and "Paste mask" (greyed, the reason on hover, when no mask was
        copied or the row has no mask of its own)."""
        if lay.get("mask") is not None:
            menu.addAction(_("Copy mask")).triggered.connect(lambda _c=False: self.copy_mask(lay))
        a = menu.addAction(_("Paste mask"))
        if self._mask_place(lay) is None:
            why = _("The Real stars row has no mask of its own: choose the layer or one of its two images")
        else:
            why = "" if self._mask_clip is not None else _("No mask copied: use Copy mask on a layer that has one")
        a.setEnabled(not why)
        a.setToolTip(why or (_("Replaces the mask of this layer with the one copied · Ctrl+Z")
                             if lay.get("mask") is not None else _("Puts the mask copied on this layer · Ctrl+Z")))
        menu.setToolTipsVisible(True)
        a.triggered.connect(lambda _c=False: self.paste_mask(lay))

    def _mask_place(self, lay):
        """Where a row's mask is written: its layer, a side of a Combine layers layer (``SideLayer``) for one of its two
        images; None for a row without a mask of its own (the real stars)."""
        if lay.get("kind") != "sub":
            return self.layer(lay["id"])
        own = self.layer(lay["id"])
        return SideLayer(own, lay["sub"]) if own is not None and is_side(lay["sub"]) else None

    def copy_mask(self, lay):
        """Copy mask: the mask of a row kept for Paste mask, as it is now (its layer's later changes are not in it)."""
        if lay.get("mask") is None:
            return
        self._mask_clip = copy.deepcopy(lay["mask"])
        self.toasted.emit(_("Mask of “{name}” copied").format(name=layer_name(lay)))

    def paste_mask(self, lay, mask=None) -> bool:
        """Paste mask: the mask copied (or ``mask``, a thumbnail dragged onto the row) put on a row, in place of the one
        it had: a copy of its own (its parts renumbered), one step of Undo named after it. Every kind of mask: its
        parts (brush, luminance or color range, gradient, H-alpha) and its feather, density and inversion; on a layer, a
        filter, the synthetic stars or one of a Combine layers layer's two images."""
        mask = self._mask_clip if mask is None else mask
        own = self._mask_place(lay)
        if mask is None or own is None:
            return False
        new = copy.deepcopy(mask)
        for part in new["parts"]:
            part["id"] = new_id()
        name = layer_name(lay)
        self.push(words=_("Mask pasted on “{name}”").format(name=name))
        own["mask"] = new
        self._mthumbs.pop(lay.get("mkey", lay["id"]), None)
        if self.sel and self.sel[0] == lay["id"] and mask_selection(self.sel[1]):
            self.part_sel = None                         # the mask shown is another one: none of its parts is chosen
        self.rebuild()
        self.request(final_only=True)
        self.toasted.emit(_("Mask pasted on “{name}” · Ctrl+Z").format(name=name))
        return True

    def _mask_drop_row(self, row, gpos):
        """The row a mask thumbnail dragged from ``row`` would be copied on: another row, with a mask of its own."""
        pt = self.lbox.parentWidget().mapFromGlobal(gpos)
        return next((r for r in self.rows if r is not row and r.geometry().contains(pt)
                     and self._mask_place(r.lay) is not None), None)

    def mask_drag_move(self, row, gpos):
        over = self._mask_drop_row(row, gpos)
        for r in self.rows:
            r.set_drop(r is over, _("Copy mask here") if r is over else "")
        self.lsc.viewport().setCursor(Qt.DragCopyCursor if over is not None else Qt.ForbiddenCursor)

    def mask_drag_end(self, row, gpos):
        """A mask's thumbnail let go on another row: copied there (the row dragged keeps its own)."""
        over = self._mask_drop_row(row, gpos)
        self.drag_cancel()
        if over is not None and row.lay.get("mask") is not None:
            self.paste_mask(over.lay, row.lay["mask"])

    # ------------------------------------------------------------ properties of the selection
    def build_props(self):
        """The properties panel made again. Of the same selection (a mask edited, an Undo), it stays where it was
        scrolled (the laptop's B7: on a small window it went back to the top, or jumped as the rows came back)."""
        key = (self.sel, self.b_crop.isChecked())
        sb = self.psc.verticalScrollBar() if hasattr(self, "psc") else None
        keep = sb.value() if sb is not None and key == getattr(self, "_props_key", None) else 0
        self._props_key = key
        self._build_props()
        wheel_scrolls(self.psc.widget(), self.psc)       # the wheel scrolls the panel, not the slider under the cursor
        if keep:
            self.psc.widget().adjustSize()
            sb.setValue(keep)
            QTimer.singleShot(0, lambda: sb.setValue(min(keep, sb.maximum())))

    def _build_props(self):
        clear(self.pbox)
        self.folds.reset()
        self.curves_editor = self.function_graph = None
        self.viewer.mode = "pan"
        lay = self.layer(self.sel[0]) if self.sel else None
        if self.b_crop.isChecked():
            self.viewer.mode = "crop"
            self._crop_props()
            self.pbox.addStretch(1)
            return
        if lay is None:
            self.pbox.addStretch(1)
            return
        if mask_selection(self.sel[1]) and (own := self.mask_owner()) is not None and own["mask"] is not None:
            self.mask_props(own)
        elif sub_role(self.sel[1]) is not None and is_fusion(lay):
            self.fusion_child_props(lay, sub_role(self.sel[1]))
        elif self.sel[1] in SUB_ROWS:
            self.pbox.addWidget(label(_(SUB_NAMES[self.sel[1]]), "h2"))
            if self.sel[1] == "real":                    # the bottom of the Stars group (#93)
                self.pbox.addWidget(label(_("The stars of the Stars/background block, under the synthetic stars: "
                                            "they show where the synthetic stars' mask is black. The filters added "
                                            "while they are selected apply to them only."), "small", wrap=True))
            else:
                self.pbox.addWidget(label(_('Image of "{name}". The filters added while it is selected apply to it '
                                            "only, before the stars are put back on the starless image.").format(
                                                name=layer_name(lay)), "small", wrap=True))
            self._part_props(lay, "stars" if self.sel[1] == "real" else self.sel[1])
        else:
            self.layer_props(lay)
        self.pbox.addStretch(1)
        self.folds.finish(f"{lay['kind']}:{lay.get('filter', '')}:{self.sel[1]}")     # issue #44: as the block panel's
        self.dock.refresh_icons()
        self._hint()

    def _group(self, text):
        self.pbox.addWidget(self.folds.title(text))      # its look is the section title's own (layout.SectionTitle)

    def layer_props(self, lay):
        self.pbox.addWidget(label(layer_name(lay), "h2"))
        row = QHBoxLayout()
        row.addWidget(label(_("Blend mode"), "dim"))
        cb = QComboBox()
        for md in MODES:
            cb.addItem(_(md), md)
        cb.setCurrentIndex(MODES.index(lay["mode"]))
        if is_pixel(lay) and lay in self.layers and not any(
                is_pixel(x) and x.get("visible", True) for x in self.layers[self.layers.index(lay) + 1:]):
            # nothing under it to blend with (the bottom layer, or the ones under it hidden): its mode is kept but not
            # applied, the layer is shown as Normal
            cb.setEnabled(False)
            cb.setToolTip(_("Nothing under this layer to blend with: shown as Normal"))
        cb.currentIndexChanged.connect(lambda i: self._set(lay, "mode", cb.itemData(i)))
        row.addWidget(cb, 1)
        self.pbox.addLayout(row)
        op = ContSlider(_("Opacity"), 0, 100, lay["opacity"], 100, unit="%", step=1)
        self._wire(op, lambda v: lay.__setitem__("opacity", v))
        self.pbox.addWidget(op)
        if lay["kind"] == "filter":
            f = FILTERS[lay["filter"]]
            # what it applies to is its place in the layers (issue #26): moved there, or with its right-click menu
            self._group(_(f["name"]))
            self.pbox.addWidget(caption(_(f["source"])))
            refresh = lambda: None                                          # noqa: E731
            if f.get("editor") == "spots":
                self.heal_props(lay)
            if f.get("editor") == "curves":
                self.curves_editor = ed = CurvesEditor(lay["params"]["curves"], self._hists.get(lay["id"]),
                                                       channel=self._curve_channel.get(lay["id"], "RGB/K"))
                ed.channelChanged.connect(lambda ch, i=lay["id"]: self._curve_channel.__setitem__(i, ch))
                self._wire_editor(ed)
                self.pbox.addWidget(ed)
            elif f.get("graph"):
                self.function_graph = g = FunctionGraph(lambda xs, fn=f["graph"]: fn(lay["params"], xs),
                                                        self._hists.get(lay["id"]))
                self.pbox.addWidget(g)
                refresh = g.update
            for q in f["params"]:
                if q["type"] in ("curve", "spots"):
                    continue
                if q["type"] == "bool":
                    cb = QCheckBox(_(q["label"]))
                    cb.setChecked(bool(lay["params"][q["key"]]))
                    cb.toggled.connect(lambda v, k=q["key"]: self._set_param(lay, k, bool(v)))
                    self.pbox.addWidget(cb)
                elif q["type"] == "choice":
                    r = QHBoxLayout()
                    r.addWidget(QLabel(_(q["label"])))
                    c = QComboBox()
                    for ch in q["choices"]:
                        c.addItem(_(ch), ch)
                    c.setCurrentIndex(q["choices"].index(lay["params"][q["key"]]))
                    c.currentIndexChanged.connect(lambda i, k=q["key"], c=c: (self._set_param(lay, k, c.itemData(i)),
                                                                              refresh()))
                    r.addWidget(c, 1)
                    self.pbox.addLayout(r)
                else:
                    sl = ContSlider(_(q["label"]), q["lo"], q["hi"], lay["params"][q["key"]], q["default"],
                                    unit=q["unit"], step=q["step"])
                    if q.get("help"):                  # what the short name leaves out
                        tip = _(q["help"]) + NL + _("Double-click: default value")
                        sl.setToolTip(tip)
                        sl.name.setToolTip(tip)
                    self._wire(sl, lambda v, k=q["key"]: (lay["params"].__setitem__(k, v), refresh()))
                    self.pbox.addWidget(sl)
            return
        if lay["kind"] == "image":
            self.image_props(lay)
            return
        p = self.path(lay["pid"])
        part = lay.get("part")
        if part:
            self.pbox.addWidget(label(_(PART_NOTES[part]), "small", wrap=True))
        if is_fusion(lay):
            self.fusion_props(lay)
        elif p is not None and hasattr(self.s, "param_value") and hasattr(self.s, "set_param"):
            # issue #49: the Output block's setting, here (the block is in the wire): its channels stretched linked
            # the same control, the same name as in the Output's panel of the graph (one name, one kind of control)
            c = QCheckBox(_("Linked channels"))
            c.setToolTip(_("Needs a run. Unticked: every color stretched apart, which neutralizes the background "
                           "(the workflow's output is prepared again)"))
            try:
                linked = self.s.param_value(p, "render_simple", "linked")
            except Exception:  # noqa: BLE001 - a value the worker has not read yet
                linked = True
            c.setChecked(linked in (None, True))
            c.toggled.connect(lambda on, q=p: (self._set_workflow_param(q, "render_simple", "linked", bool(on),
                                                                        "Linked channels"),
                                               self._update_stale(),
                                               self.toasted.emit(_("Linked channels is a setting of the output: "
                                                                   "run to see it"))))
            self.pbox.addWidget(c)
            self.linked_box = c
            self.linked_note = label(_("Linked channels only: a setting of the output, applied by a run. The sliders below act at once."), "small", wrap=True)
            self.pbox.addWidget(self.linked_note)
        bases = self.bases_of(p)
        if bases:
            r = QHBoxLayout()
            r.addWidget(label(_("Rendering"), "dim"))
            c = QComboBox()
            c.addItem(_("Suggested"), None)
            for name in bases:
                c.addItem(name, name)
            c.setCurrentIndex(max(0, c.findData(lay.get("base"))))
            c.currentIndexChanged.connect(lambda i, c=c: self._set(lay, "base", c.itemData(i)))
            r.addWidget(c, 1)
            self.pbox.addLayout(r)
        group = None
        infos = self.s.slider_infos(p, part) if p is not None else []
        if part:            # the Stars block's images: the sliders they have, the stars first (what they are for)
            infos = sorted((x for x in infos if x.available), key=lambda x: x.group != STARS_GROUP)
        for sd in infos:
            if sd.group != group:
                group = sd.group
                self._group(_(group_title(group, part)))
                if group != STARS_GROUP and part != "stars":
                    self._stretch_props(lay)             # the base stretch: first of the tone settings
                if group == STARS_GROUP and part == "combined":
                    r = QHBoxLayout()
                    r.addWidget(label(_("Star blend mode"), "dim"))
                    c = QComboBox()
                    for b in BLENDS:
                        c.addItem(_(b), b)
                    c.setToolTip(_("How the stars go back on the starless image: Screen (the usual way), Linear "
                                   "dodge (add) or Lighten"))
                    c.setCurrentIndex(max(0, BLENDS.index(lay.get("blend") or BLENDS[0])))
                    c.currentIndexChanged.connect(lambda i, c=c: self._set(lay, "blend", c.itemData(i)))
                    r.addWidget(c, 1)
                    self.pbox.addLayout(r)
            self._add_slider(lay, p, sd, part)

    def _add_slider(self, lay, p, sd, part):
        pos = lay["sliders"].get(sd.key, 2.0)
        sl = ContSlider(_(sd.name), 0, 4, pos, 2, ends=[_(e) for e in sd.ends] if sd.ends else None,
                        show=offset_text)
        ok, why, _need = self.s.slider_state(p, sd.key, part)   # the model answers in the language of the screen
        if ok and layer_stretch(lay) == "arcsinh" and sd.key in ARCSINH_WHY:
            ok, why = False, _(ARCSINH_WHY[sd.key])      # a setting the Arcsinh stretch does not read
        elif ok and is_fusion(lay) and sd.key == "colours":
            # its stretch is one factor per pixel on every filter: the colours of the mix are kept by it
            ok, why = False, _("The combination keeps its colours: one stretch factor for every filter")
        sl.set_available(ok, why)
        self._wire(sl, lambda v, k=sd.key: self._slider(lay, k, v))
        self.pbox.addWidget(sl)

    def _stretch_props(self, lay):
        """The base stretch of a workflow layer, on its linear image: Automatic (the screen transfer) or Arcsinh (the
        colour-preserving arcsinh stretch, Lupton et al. 2004), and the Arcsinh's Strength right under it."""
        r = QHBoxLayout()
        r.addWidget(label(_("Stretch"), "dim"))
        c = QComboBox()
        for k in LAYER_STRETCHES:
            c.addItem(_(STRETCH_NAMES[k]), k)
        c.setToolTip(_("How the linear image is brought to the screen. Automatic: fine for most images. Arcsinh: for "
                       "an object far brighter than the sky (a planetary nebula, a galaxy core), which Automatic "
                       "flattens near white; its structure and its colours are kept"))
        c.setCurrentIndex(LAYER_STRETCHES.index(layer_stretch(lay)))
        # (the panel is built again by the choice: once the list has closed)
        c.activated.connect(lambda i, c=c: QTimer.singleShot(0, lambda k=c.itemData(i): self.set_stretch(lay, k)))
        r.addWidget(c, 1)
        self.pbox.addLayout(r)
        self.stretch_box = c
        self.stretch_strength = None
        if layer_stretch(lay) != "arcsinh":
            return
        # in steps (an arrow key: one step, a sky level or two), its value to the step
        sl = ContSlider(_("Strength"), 0, 4, lay["sliders"].get(STRETCH_KEY, 2.0), 2,
                        ends=[_("gentler"), _("stronger")], step=STRENGTH_STEP, show=strength_text)
        tip = _("The level of the sky under the Arcsinh. In the middle the sky is where the Sky setting puts it within "
                "the range of the luminance. To the left the sky darkens and the bright areas spread apart; to the right "
                "the faint areas rise") + NL +            _("Double-click: default value")
        sl.setToolTip(tip)
        sl.name.setToolTip(tip)
        self._wire(sl, lambda v: self._slider(lay, STRETCH_KEY, v))
        self.pbox.addWidget(sl)
        self.stretch_strength = sl

    def set_stretch(self, lay, kind: str):
        """The base stretch of a workflow layer chosen: one undo step, the panel built again (the Strength shows only
        for Arcsinh)."""
        if not any(x is lay for x in self.layers) or layer_stretch(lay) == kind:
            return
        self.push()
        set_layer_stretch(lay, kind)
        self._refresh_rows()
        self.request(final_only=True)
        self.build_props()

    # ------------------------------------------------------------ a combination's master layer
    def _fusion_title(self, lay) -> str:
        """The combination's name as its block shows it everywhere (the graph, Inspect's path and title)."""
        from .graphnode import combine_kind, combine_title, node_title
        q = self.path(lay["pid"])
        if q is None:
            return _("Combination")
        try:
            node = self.s.block_map.node_of(q, "fusion")
            if node is not None:
                return node_title(self.s, node)
        except Exception:  # noqa: BLE001 - no map of the blocks now: its type's name
            pass
        return combine_title(combine_kind(q), q)

    def _fusion_slider(self, lay, role, what: str):
        """A filter's Hue (``what`` "hue") or Intensity slider: the combination block's setting, moved here (the picture
        follows), one step of the workflows' Undo when it is let go."""
        f = lay["fusion"]
        name = f["names"].get(role, role)

        def make(x):
            if what == "hue":
                return {f["hue_params"][role]: round(float(x), 2)}
            return {"weights": {role: round(float(x), 3)}}
        if what == "hue":
            sl = ContSlider(_("{filter} hue").format(filter=name), 0, 360, float(f["hues"][role]),
                            float(f["hue_defaults"].get(role, f["hues"][role])), unit="°", step=1)
            sl.s.setStyleSheet(HUE_QSS)
            FineDrag(sl.s)
            sl.setToolTip(_("The colour of {filter} on the colour wheel: the combination's setting, the same in "
                            "Inspect").format(filter=name) + NL +
                          _("Shift while dragging: ten times finer · arrow keys: one degree"))
            words = _("{filter} hue").format(filter=name)
        else:
            sl = ContSlider(_("{filter} intensity").format(filter=name), 0, INTENSITY_MAX, f["weights"].get(role, 1.0),
                            1.0, step=INTENSITY_STEP)
            sl.setToolTip(_("{filter}'s weight on top of the gain the balance measured (1: as balanced): the "
                            "combination's setting, the same in Inspect").format(filter=name))
            words = _("{filter} intensity").format(filter=name)
        pid = lay["pid"]

        def moved(x):
            self._live[pid] = {**self._live.get(pid, {}), **make(x)}
            self._attach_fusion()
            self.request()

        now = float(f["hues"][role]) if what == "hue" else float(f["weights"].get(role, 1.0))
        half = (1.0 if what == "hue" else INTENSITY_STEP) / 2

        def done(x):
            self._live.pop(pid, None)
            if abs(float(x) - now) <= half + 1e-9:       # let go where it was (its rail's step): no change of the block
                self._attach_fusion()
                self.request(final_only=True)
                return
            self.set_fusion(lay, make(x), words, rebuild=False)
        sl.moved.connect(moved)
        sl.done.connect(done)
        return sl

    def fusion_props(self, lay):
        """A combination's master layer: the combination's settings (the block's own, the same as Inspect's editor):
        its palette, each filter's hue and intensity, its balance; the link to the editor."""
        from .combineedit import BALANCE_NAME, BALANCES, fitting_palettes, has_balance, palette_hues
        from .graphnode import combine_spec
        f = lay["fusion"]
        q = self.path(lay["pid"])
        spec = combine_spec(q) if q is not None else None
        self._group(_("Combination ({type})").format(type=self._fusion_title(lay)))
        pals = fitting_palettes(spec or {}) if f["hue_params"] else []
        if pals:
            r = QHBoxLayout()
            r.addWidget(label(_("Palette"), "dim"))
            c = QComboBox()
            for k, n in pals:
                c.addItem(_(n), k)
            c.addItem(_("Custom"), "")
            c.setCurrentIndex(max(0, c.findData(f.get("palette", ""))))
            c.setToolTip(_("Starting hues for the filters in one click; then adjust each filter below"))
            c.activated.connect(lambda i, c=c: c.itemData(i) and self.set_fusion(
                lay, {f["hue_params"][r_]: h for r_, h in palette_hues(c.itemData(i)).items()},
                _("Palette: {name}").format(name=c.currentText())))
            r.addWidget(c, 1)
            self.pbox.addLayout(r)
            self.palette_box = (lay["id"], c)            # (said again in place when a hue is let go)
        for role in f["roles"]:
            if role in f["hue_params"]:
                self.pbox.addWidget(self._fusion_slider(lay, role, "hue"))
            self.pbox.addWidget(self._fusion_slider(lay, role, "intensity"))
        inputs = list(((getattr(q, "source", None) or {}).get("fusion") or {}).get("inputs") or []) if q else []
        if spec is not None and has_balance(spec, inputs):
            r = QHBoxLayout()
            r.addWidget(label(_("Balance"), "dim"))
            c = QComboBox()
            for k in BALANCES:
                c.addItem(_(BALANCE_NAME[k]), k)
            c.setCurrentIndex(max(0, c.findData(f.get("balance"))))
            c.setToolTip(_("How each filter is brought to the H-alpha before the hues: On sky, its sky's median and "
                           "spread; On signal, its nebula fitted onto the H-alpha's"))
            c.activated.connect(lambda i, c=c: self.set_fusion(lay, {"balance": c.itemData(i)},
                                                               _("Balance: {name}").format(name=c.currentText())))
            r.addWidget(c, 1)
            self.pbox.addLayout(r)
            # the one setting of the combination its run measures (the gains): changing it needs a run, said in its
            # tip; once it is not the balance the result was computed with, said under it
            c.setToolTip(c.toolTip() + NL + _("Changing it needs a run: the combination is computed again by Run"))
            # said under it only when the engine itself has the block to compute (Outdated or not computed: the
            # state Run reads) and its result was computed with another balance; a result the engine accepts (an
            # earlier version of the method, computed under the former default) is up to date: nothing said
            ran = getattr(self.s, "combine_result_balance", lambda pid: None)(lay["pid"])
            try:
                state = self.s.state(q, f["stage"]) if q is not None and hasattr(self.s, "state") else ""
            except Exception:  # noqa: BLE001 - no state now: nothing said
                state = ""
            if state in ("stale", "pending") and ran is not None and ran != f.get("balance"):
                self.pbox.addWidget(label(_("Needs a run: the balance is not the one computed"), "small", wrap=True))
        b = QPushButton(_("Open in Inspect  →"))
        b.setObjectName("link")
        b.setToolTip(_("The combination's editor: a draft to compare, Apply, Try variant"))
        b.clicked.connect(lambda: self.goProcessing.emit(lay["pid"], "fusion"))
        self.pbox.addWidget(b)

    def _palette_said(self, lay):
        """The palette field of the master panel shown, said again in place (the hues it names, or Custom), the panel
        not built again."""
        box = getattr(self, "palette_box", None)
        if not box or box[0] != lay["id"]:
            return
        try:
            c = box[1]
            c.blockSignals(True)
            c.setCurrentIndex(max(0, c.findData((lay.get("fusion") or {}).get("palette", ""))))
            c.blockSignals(False)
        except RuntimeError:                              # (a panel built again meanwhile)
            self.palette_box = None

    def fusion_child_props(self, lay, role):
        """One filter of a combination's master layer: its gain, its hue and intensity (the combination's settings),
        what acts on it."""
        from .combineedit import REASONS
        f = lay["fusion"]
        name = f["names"].get(role, role)
        self.pbox.addWidget(label(_("{filter} · in {combination}").format(filter=name,
                                                                         combination=self._fusion_title(lay)), "h2"))
        gains = getattr(self.s, "fusion_gains", lambda pid: {})(lay["pid"])
        tip = ""
        if role in gains:
            g, why, src = gains[role]
            text = _("Balanced: gain × {gain}").format(gain=number(g, 2)) + \
                (f" ({_(REASONS[why])})" if why in REASONS else "")
            if src == "measured":
                tip = _("Measured on the whole image as the run measures it (the run's own result does not hold it)")
        elif gains:
            text = _("The reference: the other filters are brought to its scale")
        else:
            text = _("Its gain is measured by the combination's run")
        gl = label(text, "small", wrap=True)
        gl.setToolTip(tip)
        self.pbox.addWidget(gl)
        if role in f["hue_params"]:
            self.pbox.addWidget(self._fusion_slider(lay, role, "hue"))
        self.pbox.addWidget(self._fusion_slider(lay, role, "intensity"))
        self.pbox.addWidget(label(_("No stretch of its own: the layer's one stretch for every filter. No stars of its "
                                    "own: the layer's Stars. Its filters, mask and spot healing act on {filter} before "
                                    "the mix.").format(filter=name), "small", wrap=True))

    def set_fusion(self, lay, values: dict, words: str, rebuild: bool = True):
        """A combination's settings changed here. Its hues and intensities are the master layer's own (client/
        postfusion.py): one step of Post's Undo, nothing to compute. The balance is the block's (the gains the run
        measures): one step of the workflows' Undo listed in Post's, the block to run. Either way a draft of the
        block in Inspect is dropped (``postTuned``)."""
        from . import postfusion
        c = getattr(self.s, "_combination", lambda pid: None)(lay["pid"])
        part, rest = postfusion.split_change(c[1], values) if c is not None and is_fusion(lay) else ({}, values)
        if part and c is not None:
            self.push(words=words)
            lay["tune"] = postfusion.new_tune(c[0], c[1], c[2], lay.get("tune"), part)
            self._attach_fusion()
            tuned = getattr(self.s, "postTuned", None)
            if tuned is not None:
                tuned.emit(lay["pid"])
            self._refresh_rows()
            self.request(final_only=True)
            if rebuild:            # (a slider let go: the panel stays as it is, the slider under the pointer)
                QTimer.singleShot(0, self.build_props)
            else:
                self._palette_said(lay)
        if not rest:
            return
        values = rest
        q = self.s.view(lay["pid"]) if hasattr(self.s, "view") else None
        if q is None or not hasattr(self.s, "apply_as"):
            return
        try:
            with self._from_post():
                self.s.apply_as({"kind": "combine", "values": values, "what": words, "short": words}, "replace", [q])
                self._name_step(words)
        except ValueError as e:
            self.toasted.emit(_(str(e)))
            return
        self.undo.append(self._model_mark("_undo"))
        self.redo.clear()
        self._undo_buttons()
        tuned = getattr(self.s, "postTuned", None)
        if tuned is not None:
            tuned.emit(lay["pid"])
        self._attach_fusion()
        self.request(final_only=True)
        QTimer.singleShot(0, self.build_props)

    def _part_props(self, lay, sub):
        """The sliders of one of a Stars and background layer's two images: those of the layer, the same settings
        (issue #73)."""
        p = self.path(lay["pid"])
        if p is None:
            return
        part = lay.get("part")
        if sub == "starless":
            self._stretch_props(lay)                     # the base stretch is the starless image's
        for sd in self.s.slider_infos(p, part):
            if sd.available and (sd.group == STARS_GROUP) == (sub == "stars"):
                self._add_slider(lay, p, sd, part)

    # ------------------------------------------------------------ the crop (issue #75): a setting, never a layer
    def _crop_tool(self, on):
        v = self.viewer
        if on:
            self._crop_before = list(self.crop) if self.crop else None
        for b in (self.b_crop_ok, self.b_crop_cancel):
            b.setVisible(on)
        for sc in self._crop_keys:
            sc.setEnabled(on)
        self._fit_bar()                                  # its two buttons take room in the bar
        v.crop_edit = on
        if on and v.crop_box is None and v.source:
            W, H = v.source.size()
            v.crop_box = (0, 0, W, H)
        if not on:
            v.crop_ratio = None
            v.end_crop_drag()                            # (Escape while the box is dragged: its release changes nothing)
            v.crop_box = self._crop_shown()
            v.mode = "pan"
        v.fit()
        self.build_props()
        self.request(final_only=True)

    def crop_apply(self):
        """Apply: the crop drawn is kept (each change of the box was one already), the tool is left."""
        if self.b_crop.isChecked():
            self.b_crop.setChecked(False)

    def crop_cancel(self):
        """Cancel: the crop as it was when the tool was taken, the tool is left."""
        if not self.b_crop.isChecked():
            return
        before = getattr(self, "_crop_before", None)
        if (list(self.crop) if self.crop else None) != before:
            self._set_crop(before)
        self.b_crop.setChecked(False)

    def heal_tool(self, on: bool = True):
        """The Spot healing tool: its layer on the selected layer (the one there is, else a new one), its panel shown;
        off, back to moving the picture."""
        if not on:
            if self._heal_layer() is not None:
                self.viewer.mode = "pan"
                bottom = next((x for x in reversed(self.layers) if is_pixel(x)), None)
                self._drop_empty_heal()
                if bottom is not None:
                    self.select(bottom["id"], "layer")
            return
        lay = next((x for x in self.layers if x["kind"] == "filter" and x.get("filter") == "heal"), None)
        if lay is not None:
            self.select(lay["id"], "layer")
        else:
            self._heal_made = self.add_filter("heal")["id"]

    def _show_crop(self, crop):
        self.crop = list(crop) if crop else None
        if not hasattr(self, "viewer"):                  # the page is being built: it reads it from there
            return
        if self.b_crop.isChecked():
            self.b_crop.setChecked(False)
        self.viewer.crop_box = self._crop_shown()
        self.viewer.fit()

    def _crop_shown(self):
        """The crop as it is shown, said and exported: its part inside Post's image (``crop_within``: a box kept from a
        larger image goes past its edge). The box kept (``self.crop``) is not rewritten: the image may grow back."""
        if not self.crop or self.src.canvas is None:
            return tuple(self.crop) if self.crop else None
        return crop_within(self.crop, *self.src.canvas)

    def _crop_drawn(self):
        """The box as drawn becomes the crop (one undo step); the whole picture is no crop."""
        W, H = self.src.size()
        box = [int(round(c)) for c in self.viewer.crop_box]
        self._set_crop(None if box == [0, 0, W, H] else box)

    def _set_crop(self, box):
        self.push()
        self.crop = box
        self._crop_info()

    def _crop_info(self):
        if getattr(self, "crop_size", None) is not None:
            W, H = self.src.size()
            x0, y0, x1, y1 = self._crop_shown() or (0, 0, W, H)          # the size the export will have
            self.crop_size.setText(f"{x1 - x0} × {y1 - y0} px")
            self.crop_off.setEnabled(self.crop is not None)

    def _crop_props(self):
        self.pbox.addWidget(label(_("Crop"), "h2"))
        self.pbox.addWidget(label(_("Drag the handles or the box, or draw a new box. Nothing is lost: the rendering "
                                    "of the whole image does not change."), "small", wrap=True))
        r = QHBoxLayout()
        r.addWidget(label(_("Proportions"), "dim"))
        W, H = self.src.size()
        cb = QComboBox()
        cb.addItem(_("Free"), None)
        cb.addItem(_("Original"), W / max(H, 1))
        for name, q in (("3:2", 1.5), ("2:3", 2 / 3), ("16:9", 16 / 9), ("9:16", 9 / 16), ("1:1", 1.0)):
            cb.addItem(name, q)
        cb.currentIndexChanged.connect(lambda i: self._crop_ratio(cb.itemData(i)))
        r.addWidget(cb, 1)
        self.pbox.addLayout(r)
        self.crop_ratio_cb = cb
        self.crop_size = label("", "small")
        self.pbox.addWidget(self.crop_size)
        self.crop_off = QPushButton(_("Remove crop"))
        self.crop_off.clicked.connect(self._crop_remove)
        self.pbox.addWidget(self.crop_off, 0, Qt.AlignLeft)
        self._crop_info()

    def _crop_ratio(self, q):
        """A fixed proportion: the largest box of it inside the image, centred on the present box's centre (as far as the
        image allows), as the reference photo editors do."""
        v = self.viewer
        v.crop_ratio = q
        if q and v.crop_box:
            x0, y0, x1, y1 = v.crop_box
            W, H = self.src.size()
            w, h = (W, W / q) if W / q <= H else (H * q, H)
            cx = min(max((x0 + x1) / 2, w / 2), W - w / 2)
            cy = min(max((y0 + y1) / 2, h / 2), H - h / 2)
            v.crop_box = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
            v.update()
            self._crop_drawn()

    def _crop_remove(self):
        W, H = self.src.size()
        self.viewer.crop_box = (0, 0, W, H)
        self.viewer.update()
        self._set_crop(None)

    def image_props(self, lay):
        """An image layer: where it comes from, its automatic stretch (linked or each channel) and its three
        sliders."""
        ok = self.worker_has_image(lay)
        self.pbox.addWidget(label(_("A stage's image, stretched automatically.") if ok else
                                  _("This image is no longer in the project's results."), "small" if ok else "warn"))
        # the same control, the same name as the Output block's in the graph (one name, one kind of control)
        c = QCheckBox(_("Linked channels"))
        c.setToolTip(_("Unticked: every color stretched apart, which neutralizes the background"))
        c.setChecked(bool(lay.get("linked", True)))
        c.toggled.connect(lambda on: self._set(lay, "linked", bool(on)))
        self.pbox.addWidget(c)
        self.linked_box = c
        self._group(_("Stretch"))
        for key, name, ends in IMAGE_SLIDERS:
            pos = lay["sliders"].get(key, 2.0)
            sl = ContSlider(_(name), 0, 4, pos, 2, ends=[_(e) for e in ends],
                            show=offset_text)
            self._wire(sl, lambda v, k=key: self._slider(lay, k, v))
            self.pbox.addWidget(sl)

    def worker_has_image(self, lay) -> bool:
        """Whether an image layer's pixels are there (the last image the worker drew had it; until then, yes)."""
        return lay["key"] not in self._missing_images

    def bases_of(self, p) -> list[str]:
        """The named renderings of the project (faithful, artistic…), when it has some."""
        rb = getattr(getattr(self.s, "p", None), "render_bases", None) or {}
        return list(rb)

    def _slider(self, lay, k, v):
        if abs(v - 2) < 0.02:
            lay["sliders"].pop(k, None)
        else:
            lay["sliders"][k] = v

    def _wire(self, sl: ContSlider, apply):
        """A slider: draft render while it moves, full render and one undo step when it is let go."""
        state = {"before": None}

        def moved(v):
            if state["before"] is None:
                state["before"] = snapshot(self.layers)
            apply(v)
            self.request(settings_only=True)

        def done(v):
            if state["before"] is None:
                state["before"] = snapshot(self.layers)
            apply(v)
            self.push(state["before"])
            state["before"] = None
            self.request(final_only=True, settings_only=True)
            QTimer.singleShot(0, self._refresh_rows)
        sl.moved.connect(moved)
        sl.done.connect(done)

    def _wire_editor(self, ed: CurvesEditor):
        """The Curves panel: a draft while a point moves; on release the exact render and one undo step (the state
        before is taken when the edit starts, before the points change)."""
        state = {"before": None}

        def starting():
            if state["before"] is None:
                state["before"] = snapshot(self.layers)

        def committed():
            self.push(state["before"] if state["before"] is not None else snapshot(self.layers))
            state["before"] = None
            self.request(final_only=True, settings_only=True)
        ed.starting.connect(starting)
        ed.changed.connect(lambda *a: self.request(settings_only=True))
        ed.committed.connect(committed)

    def sub_text(self, lay) -> str:
        """Under a layer's name: mode and opacity; for a filter of its own row, that it applies to everything below (a
        nested one says it by its place)."""
        bits = []
        # the marks first: the end of a long row is cut (the label is at most SUB_MAX wide)
        if (is_pixel(lay) and lay["kind"] != "image" and f"layer:{lay['id']}" in self._missing_images) or (
                lay["kind"] == "sub" and f"side:{lay['id']}:{lay['sub']}" in self._missing_images):
            bits.append(_("not computed"))
        if lay["kind"] == "workflow" and self.is_preview(lay["pid"]):
            bits.append(_("preview"))
        if lay["kind"] == "filter" and not lay.get("target"):
            bits.append(_("on everything below"))
        bits.append(f"{_(lay['mode'])} · {with_unit(format(lay['opacity'], '.0f'), '%')}")
        if lay["kind"] == "image":
            bits.append(_("image") if self.worker_has_image(lay) else _("not found"))
        if is_pixel(lay) and lay["id"] in self._collapsed:
            n = len(children(self.layers, lay["id"]))
            bits.append(plural(n, "{n} filter", "{n} filters"))
        return " · ".join(bits)

    def is_preview(self, pid) -> bool:
        """The workflow's image is still an approximation: its output is Outdated and the layer shows the earlier one."""
        p = self.path(pid)
        return p is not None and (bool(getattr(p, "provisional", False)) or (p.rendered and not p.computed))

    def _place_of(self, pid, paths) -> int:
        """Where a workflow's layer comes among the others: the later in the order the workflows have (the order the
        images were opened in), the higher, whichever was computed first."""
        order = {q.pid: i for i, q in enumerate(paths)}
        mine = order.get(pid, len(order))
        flows = [(j, order.get(x["pid"], -1)) for j, x in enumerate(self.layers)
                 if x["kind"] == "workflow" and x.get("target") is None]
        below = next((j for j, o in flows if o < mine), None)
        if below is not None:
            return below
        return flows[-1][0] + 1 if flows else 0

    def _update_stale(self):
        """The strip over the picture: shown while a layer on screen comes from outdated results."""
        old = [x for x in self.layers if x["kind"] == "workflow" and x.get("visible") and self.is_preview(x["pid"])]
        self.stale.setVisible(bool(old))
        if old:
            self.stale_text.setText(_("Outdated: run to update") + "  ·  " + ", ".join(layer_name(x) for x in old[:3]))

    def _refresh_rows(self):
        """The row texts (mode, opacity) without rebuilding the properties under the mouse."""
        self._update_stale()
        usual = f"{_('Normal')} · {with_unit('100', '%')}"
        for row in self.rows:
            row.sub.setText(self.sub_text(row.lay))
            if row.n.isHidden():                                  # (being renamed)
                continue
            text = row.sub.full_text()
            # a row says more than the usual blend once a mark (not computed) is on it, and is silent again without
            row.sub.setVisible(_("not computed") in text if row.lay["kind"] == "sub" else text != usual)

    def _set(self, lay, key, value):
        self.push()
        lay[key] = value
        self._refresh_rows()
        self.request(final_only=True)

    def _set_param(self, lay, key, value):
        self.push()
        lay["params"][key] = value
        self.request(final_only=True)

    # ------------------------------------------------------------ the mask
    def mask_props(self, lay):
        m = lay["mask"]
        top = QHBoxLayout()
        top.addWidget(label(_('Mask of "{name}"').format(name=layer_name(lay)), "h2"), 1)
        rm = QPushButton(_("Delete"))
        rm.clicked.connect(lambda: self.remove_mask(lay))
        top.addWidget(rm)
        self.pbox.addLayout(top)
        tint = QCheckBox(_("Red tint"))
        tint.setToolTip(_("The red over the picture, where the mask hides the layer. Off: the picture as it is, the "
                          "brush still paints the mask."))
        tint.setChecked(self.show_tint)
        tint.toggled.connect(self._tint_view)
        self.pbox.addWidget(tint)
        if not m["parts"]:
            self.pbox.addWidget(label(_("Empty: the layer shows everywhere."), "small"))
        for j, part in enumerate(m["parts"]):
            self.pbox.addWidget(self.part_box(lay, part, j))
        add = set_icon(QPushButton(_("Add to mask")), "add", 16)
        am = QMenu(add)
        owner = self._fusion_owner(lay)
        for src, name in MASK_SOURCES.items():
            a = am.addAction(_(name))
            a.triggered.connect(lambda _c=False, s=src: self.add_part(lay, s))
            if src == "halpha" and owner is not None and "Ha" not in fusion_roles(owner):
                a.setEnabled(False)                      # greyed, with why
                a.setToolTip(_("This combination has no H-alpha filter"))
            if src == "stars" and not self._has_stars(lay):
                a.setEnabled(False)
                a.setToolTip(_("This layer has no stars image: its workflow has no Stars/background block"))
        am.setToolTipsVisible(True)
        add.setMenu(am)
        self.pbox.addWidget(add, 0, Qt.AlignLeft)
        self._group(_("Whole mask"))
        fe = ContSlider(_("Feather"), 0, 60, m["feather"], 0, unit="px", step=1)
        self._wire(fe, lambda v: m.__setitem__("feather", v))
        self.pbox.addWidget(fe)
        de = ContSlider(_("Density"), 0, 100, m["density"], 100, unit="%", step=1)
        self._wire(de, lambda v: m.__setitem__("density", v))
        self.pbox.addWidget(de)
        inv = QCheckBox(_("Invert"))
        inv.setChecked(m["invert"])
        inv.toggled.connect(lambda v: self._invert_mask(m, v))
        self.pbox.addWidget(inv)
        if self.part_sel and any(p["id"] == self.part_sel and p["src"] == "brush" for p in m["parts"]):
            self.viewer.mode = "brush"
            part = next(p for p in m["parts"] if p["id"] == self.part_sel)
            self.viewer.brush_r = part["size"] / 2

    def _invert_mask(self, m, v):
        """Invert: the brushes keep the effect chosen (Hide stays Hide), so the next stroke does what it said; the
        strokes already painted follow the mask, as the rest of it."""
        self.push()
        m["invert"] = v
        for p in m["parts"]:
            if p["src"] == "brush":
                p["paint"] = not p["paint"]
        self.request(final_only=True)
        QTimer.singleShot(0, self.build_props)

    def part_box(self, lay, part, j):
        on = part["id"] == self.part_sel
        f = QFrame()
        f.setObjectName("part")
        f.setStyleSheet(f"QFrame#part {{ background: #20242c; border: 1px solid {ACCENT if on else '#353b47'}; "
                        "border-radius: 6px; } QLabel { background: transparent; }")
        v = QVBoxLayout(f)
        v.setContentsMargins(8, 6, 8, 6)
        v.setSpacing(2)
        top = QHBoxLayout()
        cb = QComboBox()
        for k, t in OPS.items():
            cb.addItem(_(t), k)
        cb.setCurrentIndex(list(OPS).index(part["op"]))
        cb.setFixedWidth(110)
        cb.currentIndexChanged.connect(lambda i: self._part_set(part, "op", cb.itemData(i)))
        top.addWidget(cb)
        t = QLabel(_(MASK_SOURCES[part["src"]]))
        t.setStyleSheet("font-weight: 600;")
        top.addWidget(t, 1)
        x = icon_only(QPushButton(), "remove", _("Remove {part} from the mask, with what it holds (Ctrl+Z brings it "
                                                 "back)").format(part=_(MASK_SOURCES[part["src"]])), 16, box=22)
        x.clicked.connect(lambda: self.remove_part(lay, part))
        top.addWidget(x)
        v.addLayout(top)
        src = part["src"]
        default = new_part(src)
        owner = self._fusion_owner(lay)
        if src == "halpha" and owner is not None:        # a combination's: its H-alpha filter
            v.addWidget(label(_("The combination's H-alpha filter: nothing at its middle value, all at its brightest "
                                "3 %. Feather, Density and Invert act on it."), "small", wrap=True))
        elif src == "halpha":                            # no setting of its own: what it is made of (the laptop's B9)
            v.addWidget(label(_("Where the image is redder than green, the hydrogen glow of the workflow at the bottom "
                                "(its starless image when it has one): nothing at its middle value, all at its "
                                "brightest 3 %. Feather, Density and Invert act on it."), "small", wrap=True))
        if src in READS_IMAGE and isinstance(lay, SideLayer) and sub_role(lay.side) is not None and owner is not None:
            # a filter's mask reads the combination's picture by default, as luminosity masks are taken from the
            # composite picture in the reference photo editors; or the filter itself, or another filter
            r = QHBoxLayout()
            r.addWidget(label(_("Reads"), "dim"))
            c = QComboBox()
            c.addItem(_("Picture"), "picture")
            c.setItemData(0, _("The combined picture without its stars: the filters (starless images) mixed, before the stars are put "
                "back"), Qt.ToolTipRole)
            c.addItem(_("This filter"), "self")
            for o in fusion_roles(owner):
                if role_sub(o) != lay.side:
                    c.addItem(_side_name(owner, role_sub(o)), role_sub(o))
            c.setCurrentIndex(max(0, c.findData(part.get("from", "picture"))))
            c.setToolTip(_("The image this part reads its brightness or its colour from"))
            c.activated.connect(lambda i, c=c: self._part_set(part, "from", c.itemData(i)))
            r.addWidget(c, 1)
            v.addLayout(r)

        if src == "stars":                               # what it is made of, its source
            v.addWidget(label(_("The stars of the layer's stars image (its Stars/background; a combination's: its "
                                "Stars side), each as the screen shows it, grown by Grow times its own radius: a "
                                "brighter star grows more. Subtract it from another part to leave the stars and "
                                "their wings out."), "small", wrap=True))

        def sl(key, name, lo, hi, unit=""):
            s = ContSlider(_(name), lo, hi, part[key], default[key], unit=unit, step=0.1 if key == "grow" else 1)
            say = PART_TIPS.get((src, key))
            if say:                                      # what the short name leaves out
                tip = _(say) + NL + _("Double-click: default value")
                s.setToolTip(tip)
                s.name.setToolTip(tip)
            self._wire(s, lambda val: part.__setitem__(key, val))
            v.addWidget(s)

        for key, name, lo, hi, unit in PART_SETTINGS.get(src, []):
            if src == "gradient" and key == "position":                 # the direction comes first
                r = QHBoxLayout()
                r.addWidget(QLabel(_("Direction")))
                c = QComboBox()
                for d in DIRECTIONS:
                    c.addItem(_(d), d)
                c.setCurrentIndex(DIRECTIONS.index(part["direction"]))
                c.currentIndexChanged.connect(lambda i, c=c: self._part_set(part, "direction", c.itemData(i)))
                r.addWidget(c, 1)
                v.addLayout(r)
            sl(key, name, lo, hi, unit)
        if src == "brush":
            r = QHBoxLayout()
            # named by their effect on the layer (the owner's rule): a brush of a part that subtracts, or of an
            # inverted mask, hides where it paints
            shows = (part["op"] != "sub") != bool((lay["mask"] or {}).get("invert"))
            for txt, tip, val in (("Show", "Painting here shows the layer", shows),
                                  ("Hide", "Painting here hides the layer", not shows)):
                b = QPushButton(_(txt))
                b.setToolTip(_(tip))
                b.setCheckable(True)
                b.setChecked(part["paint"] == val)
                b.setObjectName("face")
                b.clicked.connect(lambda _=False, vv=val: self._brush_mode(part, vv))
                r.addWidget(b)
            r.addStretch(1)
            cl = QPushButton(_("Clear all"))
            cl.setToolTip(_("Removes every stroke of this brush. One stroke: paint the other effect over it, or "
                            "Ctrl+Z for the last one"))
            cl.clicked.connect(lambda: self._part_set(part, "strokes", []))
            r.addWidget(cl)
            v.addLayout(r)
            top = self.brush_max()                       # as large as the image: one stroke covers its field
            sz = ContSlider(_("Size"), 0.0, 1.0, brush_pos(part["size"], top), brush_pos(60.0, top),
                            show=lambda t: with_unit(str(brush_px(t, top)), "px"))
            for sig in (sz.moved, sz.done):
                sig.connect(lambda t: (part.__setitem__("size", float(brush_px(t, top))),
                                       setattr(self.viewer, "brush_r", brush_px(t, top) / 2)))
            v.addWidget(sz)
            hd = ContSlider(_("Hardness"), 0, 100, part["hardness"], 50, unit="%", step=1)
            hd.done.connect(lambda val: part.__setitem__("hardness", val))
            v.addWidget(hd)
            if not on:
                go = QPushButton(_("Paint on image"))
                go.clicked.connect(lambda: self._edit_part(part))
                v.addWidget(go, 0, Qt.AlignLeft)
        f.mousePressEvent = lambda e: self._edit_part(part) if part["src"] == "brush" else None
        return f

    def _has_stars(self, lay) -> bool:
        """Whether the layer a mask is on has a stars image for the Stars part (``postengine.Composer._stars_of``):
        a Combine layers layer with its Stars side (and its filters), a workflow's Output whose workflow has the Stars
        block; a filter's, that of the layer it is nested in."""
        own = lay.lay if isinstance(lay, SideLayer) else lay
        if own is not None and own.get("kind") == "filter":
            own = self.layer(own.get("target")) if own.get("target") else None
        if own is None:
            return False
        if is_group(own):
            return own.get("stars_pid", own.get("pid")) is not None
        if own.get("kind") == "workflow" and own.get("part") is None:
            chains = getattr(self.s, "_chains", {}) or {}
            return "star_removal" in (chains.get(own["pid"]) or [])
        return False

    def _fusion_owner(self, lay):
        """The master layer a mask belongs to (its own, or one of its filters'), None for any other layer."""
        own = lay.lay if isinstance(lay, SideLayer) else lay
        return own if own is not None and is_fusion(own) else None

    def _edit_part(self, part):
        self.part_sel = part["id"]
        self.build_props()

    def _brush_mode(self, part, paint):
        part["paint"] = paint
        self.part_sel = part["id"]
        self.build_props()

    def toggle_paint(self):
        lay = self.mask_owner()
        if not lay or not lay["mask"]:
            return
        part = next((p for p in lay["mask"]["parts"] if p["id"] == self.part_sel and p["src"] == "brush"), None)
        if part:
            self._brush_mode(part, not part["paint"])

    def brush_max(self) -> float:
        """The mask brush's largest size: the long side of Post's image, BRUSH_MAX at least (a small image, or none
        yet)."""
        return float(max(BRUSH_MAX, *(self.src.canvas or ())))

    def brush_size(self, k):
        if self.viewer.mode == "heal":
            self.heal_size = float(np.clip(self.heal_size * k, 6, 1200))
            self.viewer.brush_r = self.heal_size / 2
            self.build_props()
            return
        lay = self.mask_owner()
        part = next((p for p in (lay["mask"]["parts"] if lay and lay["mask"] else [])
                     if p["id"] == self.part_sel and p["src"] == "brush"), None)
        if part:
            part["size"] = float(np.clip(part["size"] * k, BRUSH_MIN, self.brush_max()))
            self.viewer.brush_r = part["size"] / 2
            self.build_props()

    def add_part(self, lay, src):
        self.push()
        part = new_part(src)
        if src == "brush" and not lay["mask"]["parts"]:
            # an empty mask shows the whole layer: its first brush keeps it so (it takes away from a full mask, as the
            # Synthetic stars' brush) and paints the effect that changes it, Hide (Show when the mask is inverted)
            part["op"] = "sub"
        lay["mask"]["parts"].append(part)
        self.part_sel = part["id"] if src == "brush" else None
        self.build_props()
        self.request(final_only=True)
        return part

    def remove_part(self, lay, part):
        self.push()
        lay["mask"]["parts"] = [p for p in lay["mask"]["parts"] if p["id"] != part["id"]]
        if self.part_sel == part["id"]:
            self.part_sel = None
        self.build_props()
        self.request(final_only=True)
        # said, as any removal (its strokes went with it): Undo brings it back
        self.toasted.emit(_("{part} removed from the mask · Ctrl+Z").format(part=_(MASK_SOURCES[part["src"]])))

    def _part_set(self, part, key, value):
        self.push()
        part[key] = value
        self.request(final_only=True)

    def _stroke(self, x, y, first):
        lay = self.mask_owner()
        part = next((p for p in (lay["mask"]["parts"] if lay and lay["mask"] else []) if p["id"] == self.part_sel),
                    None)
        if part is None or part["src"] != "brush":
            return
        if first:
            self._stroke_before = snapshot(self.layers)
            self._last_pt = None
        rad = part["size"] / 2
        pts = [(x, y)]
        if self._last_pt is not None:                    # fill the gaps of a fast stroke
            lx, ly = self._last_pt
            n = int(math.hypot(x - lx, y - ly) / max(rad / 3, 1))
            pts = [(lx + (x - lx) * k / (n + 1), ly + (y - ly) * k / (n + 1)) for k in range(1, n + 1)] + pts
        for px, py in pts:
            part["strokes"].append((px, py, rad, part["hardness"], part["paint"]))
        self._last_pt = (x, y)
        self.request()

    def _stroke_end(self):
        if self._stroke_before is not None:
            self.push(self._stroke_before)
            self._stroke_before = None
        self.request(final_only=True)

    # ------------------------------------------------------------ spot healing
    def heal_props(self, lay):
        """The Spot healing filter's panel (the reference image editor's Spot Healing Brush, issue #20): the brush size, what is
        healed, and the source set by hand (Alt+click) when there is one."""
        self.viewer.mode = "heal"
        self.viewer.brush_r = self.heal_size / 2
        sz = ContSlider(_("Size"), 6, 1200, self.heal_size, 80, unit="px", step=1)
        sz.moved.connect(lambda v: (setattr(self, "heal_size", float(v)), setattr(self.viewer, "brush_r", v / 2)))
        sz.done.connect(lambda v: (setattr(self, "heal_size", float(v)), setattr(self.viewer, "brush_r", v / 2)))
        self.pbox.addWidget(sz)
        r = QHBoxLayout()
        n = len(lay["params"]["spots"])
        r.addWidget(label(_("{n} spots healed").format(n=n) if n != 1 else _("1 spot healed"), "small"))
        r.addStretch(1)
        cl = QPushButton(_("Remove all"))
        cl.setEnabled(n > 0)
        cl.clicked.connect(lambda: self._heal_set(lay, []))
        r.addWidget(cl)
        self.pbox.addLayout(r)
        src = QHBoxLayout()
        manual = self._heal_src is not None or self._heal_off is not None
        src.addWidget(label(_("Source: set by hand (Alt+click)") if manual else
                            _("Source: found around each stroke · Alt+click: set it by hand"), "small", wrap=True), 1)
        if manual:
            au = QPushButton(_("Automatic"))
            au.setToolTip(_("The source found around each stroke again"))
            au.clicked.connect(lambda: (setattr(self, "_heal_src", None), setattr(self, "_heal_off", None),
                                        self.build_props()))
            src.addWidget(au)
        self.pbox.addLayout(src)

    def _drop_empty_heal(self):
        """The tool left with no spot healed: its layer goes (nothing is left in the list), and with it the Undo step
        its making was."""
        empty = [x for x in self.layers if x["kind"] == "filter" and x.get("filter") == "heal"
                 and not x["params"].get("spots")]
        if not empty:
            return
        gone = {x["id"] for x in empty}
        self.layers = normalize([x for x in self.layers if x["id"] not in gone])
        made = getattr(self, "_heal_made", None)
        if made in gone and self.undo and not isinstance(self.undo[-1], ModelStep) and                 not any(x["id"] == made for x in self.undo[-1]):
            self.undo.pop()                                  # (the step before its making: it was not a change)
        self._heal_made = None
        self.sel = None
        self.rebuild()
        self.request(final_only=True)
        self._undo_buttons()

    def _heal_layer(self):
        lay = self.layer(self.sel[0]) if self.sel and self.sel[1] == "layer" else None
        return lay if lay is not None and lay["kind"] == "filter" and lay["filter"] == "heal" else None

    def _heal_set(self, lay, spots):
        self.push()
        lay["params"]["spots"] = spots
        self.request(final_only=True)
        self.build_props()

    def _heal_source(self, x, y, r, avoid=()):
        """The source of a spot: the best match around it on the image shown (its half resolution), as the reference image editor's
        Spot Healing Brush; ``avoid``: the stroke's other spots (x, y, r), no source on them (a trail is not healed
        from itself); beside the spot when the image is not there yet."""
        from .postfilters.heal import find_source
        W, H = self.src.size()
        R = int(math.ceil(r * 4.8)) + 4
        x0, y0 = max(0, int(x - R) // 2), max(0, int(y - R) // 2)
        x1, y1 = min(W // 2, int(x + R) // 2 + 1), min(H // 2, int(y + R) // 2 + 1)
        a = self.src.region(1, x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None
        if a is not None and a.size and a.max() > 0:
            dx, dy = find_source(a.astype(np.float32) / 255.0, x / 2 - x0, y / 2 - y0, r / 2,
                                 avoid=[(ax / 2 - x0, ay / 2 - y0, ar / 2) for ax, ay, ar in avoid])
            if dx or dy:
                return 2 * dx, 2 * dy
        return 2.5 * r, 0.0

    def _heal_press(self, x, y, button, alt):
        """the reference image editor's Spot Healing Brush: a press starts a stroke (healed when the button is let go); Alt+click
        sets the source by hand (the reference image editor's Healing Brush) for the strokes that follow."""
        lay = self._heal_layer()
        if lay is None or button != int(Qt.LeftButton.value):
            return
        if alt:
            self._heal_src, self._heal_off = (x, y), None
            self.build_props()
            self.viewer.update()
            return
        if self._heal_src is not None:                          # the first stroke after Alt+click fixes the offset
            self._heal_off = (self._heal_src[0] - x, self._heal_src[1] - y)
            self._heal_src = None
        self._heal_stroke = ([(x, y)], snapshot(self.layers))
        self.viewer.update()

    def _heal_move(self, x, y):
        if self._heal_layer() is None or self._heal_stroke is None:
            return
        pts = self._heal_stroke[0]
        step = max(1.0, self.heal_size / 2 * 0.5)              # a spot every half radius along the stroke
        lx, ly = pts[-1]
        d = math.hypot(x - lx, y - ly)
        for k in range(1, int(d // step) + 1):
            pts.append((lx + (x - lx) * k * step / d, ly + (y - ly) * k * step / d))
        self.viewer.update()

    def _heal_release(self):
        """The stroke healed: a spot every half radius along it, each from a source found around it (not on the
        stroke) or from the source set by hand."""
        lay = self._heal_layer()
        st, self._heal_stroke = self._heal_stroke, None
        if lay is None or st is None:
            self.viewer.update()
            return
        pts, before = st
        r = self.heal_size / 2
        avoid = [(px, py, r) for px, py in pts]
        new = []
        for x, y in pts:
            dx, dy = self._heal_off if self._heal_off is not None else self._heal_source(x, y, r, avoid)
            new.append([round(x, 1), round(y, 1), round(r, 1), round(dx, 1), round(dy, 1)])
        self.push(before)
        lay["params"]["spots"] = lay["params"]["spots"] + new
        self.request(final_only=True)
        self.build_props()
        self.viewer.update()

    def _heal_overlay(self, p: QPainter, v: Viewer):
        """While painting, the stroke in light grey (as the reference image editor's), and the source set by hand as a small cross;
        nothing is left on the image afterwards."""
        lay = self._heal_layer()
        if lay is None or self.b_cmp.isChecked():
            return
        if self._heal_stroke is not None:
            rr = self.heal_size / 2 * v.scale
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(255, 255, 255, 70))
            for x, y in self._heal_stroke[0]:
                p.drawEllipse(v.img_to_w(x, y), rr, rr)
        cross = None
        if self._heal_src is not None:
            cross = v.img_to_w(*self._heal_src)
        elif self._heal_off is not None and self._heal_stroke is not None:
            x, y = self._heal_stroke[0][-1]
            cross = v.img_to_w(x + self._heal_off[0], y + self._heal_off[1])
        if cross is not None:
            p.setPen(QPen(QColor("#ffffff"), 1.5))
            p.drawLine(QPointF(cross.x() - 7, cross.y()), QPointF(cross.x() + 7, cross.y()))
            p.drawLine(QPointF(cross.x(), cross.y() - 7), QPointF(cross.x(), cross.y() + 7))

    # ------------------------------------------------------------ view toggles
    def _compare(self, on):
        self.cb_cmp.setVisible(on)
        self.viewer.compare = self.before if on else None
        if on:
            self._send_before()
        self.viewer.update()
        self._hint()

    def _toggle_mask(self):
        if self.b_mask.isEnabled():
            self.b_mask.toggle()

    def _mask_button(self):
        """Show mask: greyed, the reason on hover, when the selected layer has none (issue #71)."""
        lay = self.mask_owner()
        has = lay is not None and lay.get("mask") is not None
        self.b_mask.setEnabled(has)
        self.b_mask.setToolTip(_("Mask of the selected layer, in gray (M)") if has else
                               _("The selected layer has no mask"))
        if not has and self.b_mask.isChecked():
            self.b_mask.setChecked(False)

    def _tint_view(self, on):
        """The red of the mask being edited, shown or not: a choice of the view, kept while the page lives."""
        self.show_tint = on
        self.viewer.update()
        self._hint()

    def _mask_view(self, on):
        self.show_mask = on
        self.src.version += 1
        self.viewer.refresh()
        self._hint()

    def _hint(self):
        if self.viewer.mode == "brush":
            t = _("Brush: drag to paint · X show / hide · [ ] size · M show mask")
        elif self.viewer.mode == "heal":
            t = _("Spot healing: click or paint over a defect, healed when released · [ ] size · Alt+click: the "
                  "source by hand · Ctrl+Z: undo · middle button: move the image")
        elif self.sel and mask_selection(self.sel[1]):
            t = _("Red: where the layer is hidden · M: mask in gray") if self.show_tint else _("M: mask in gray")
        elif self.b_cmp.isChecked():
            t = _("Drag the divider · Before: the bottom layer, suggested settings")
        else:
            t = ""
        if self._notes:
            note = "; ".join(_(n) for n in self._notes)
            t = note + (" · " + t if t else "")
        self.hint.setText(t)

    # ------------------------------------------------------------ export
    def export_dialog(self, parent=None):
        """The Export dialog on the composition as it is now (File > Export image…, or Export… in the bar)."""
        from .exportdlg import ExportDialog
        d = ExportDialog(self, parent or self.window())
        d.exec()
        d.deleteLater()


#: Post's own keys: (the key sequences, what they do (English, translated where shown), what they run on the page, None
#: for a key the page handles apart). The one table binds the keys and gives the Help menu's Shortcuts their list.
POST_KEYS = (
    (("R",), "Crop", lambda p: p.b_crop.toggle()),
    (("C",), "Compare", lambda p: p.b_cmp.toggle()),
    (("V",), "Flip compare", lambda p: p._flip_compare()),
    (("M",), "Show mask", lambda p: p._toggle_mask()),
    (("F", "Ctrl+0"), "Fit", lambda p: p.viewer.fit()),
    (("1",), "Zoom 100 %", lambda p: p.viewer.zoom_to(1.0)),
    (("+", "=", "Ctrl++"), "Zoom in", lambda p: p.viewer.zoom_by(ZOOM_STEP)),
    (("-", "Ctrl+-"), "Zoom out", lambda p: p.viewer.zoom_by(1 / ZOOM_STEP)),
    (("X",), "Brush: show / hide", lambda p: p.toggle_paint()),
    (("[",), "Smaller brush", lambda p: p.brush_size(1 / 1.25)),
    (("]",), "Bigger brush", lambda p: p.brush_size(1.25)),
    (("Ctrl+J",), "Duplicate", lambda p: p.duplicate()),
    (("Delete",), "Delete", lambda p: p._delete_key()),
    (("Return", "Enter"), "Keep crop", None),
    (("Escape",), "Leave tool", None),
)


class PostPage3(QWidget):
    """The Post, or what is missing for it."""
    goWorkflow = Signal()
    openImages = Signal()

    def __init__(self, session, parent=None, *, worker: RenderWorker | None = None, persist: bool = True):
        super().__init__(parent)
        self.setObjectName("root")
        self.s = session
        self.pages = QStackedWidget()
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.addWidget(self.pages)
        self.empty = QWidget()
        ev = QVBoxLayout(self.empty)
        ev.setAlignment(Qt.AlignCenter)
        # nothing computed yet: the layers the graph counts (its Post block's "N layers"), named, each "not computed
        # yet", so that the page and the block say the same thing
        self.empty_title = label(_("No layers yet"), "h1")
        ev.addWidget(self.empty_title, 0, Qt.AlignHCenter)
        self.empty_note = label(_("Each computed workflow becomes a layer here."), "dim")
        ev.addWidget(self.empty_note, 0, Qt.AlignHCenter)
        # the project's last picture (its cache's thumbnail, as Home and the Report show it) when its results are no
        # longer kept: the page says so and shows it, rather than an empty page
        self.empty_pic = QLabel()
        self.empty_pic.hide()
        ev.addWidget(self.empty_pic, 0, Qt.AlignHCenter)
        self.empty_list = QVBoxLayout()
        self.empty_list.setSpacing(2)
        ev.addLayout(self.empty_list)
        o = set_icon(QPushButton(_("Open images…")), "block_image", 16, "on_accent")
        o.setObjectName("primary")
        o.setToolTip(_("Each image becomes a layer, and a block in the graph"))
        o.clicked.connect(self.openImages.emit)
        ev.addWidget(o, 0, Qt.AlignHCenter)
        b = QPushButton(_("Graph  →"))
        b.setObjectName("link")
        b.clicked.connect(self.goWorkflow.emit)
        ev.addWidget(b, 0, Qt.AlignHCenter)
        self.pages.addWidget(self.empty)
        self.view = PostView(session, worker=worker, persist=persist)
        self.pages.addWidget(self.view)
        self._asked = None                               # the layer Open in Post asked for, shown in the list
        self.view.layerAsked.connect(self._ask)
        # bound to this page's own methods, never a lambda: Qt drops them with the page (control round: a page
        # rebuilt by a language switch still answered the project's changes, a Python error at the next Ctrl+Z)
        session.pathsChanged.connect(self.update_page)
        if hasattr(session, "postChanged"):
            session.postChanged.connect(self._post_changed)
        self.update_page()

    def _post_changed(self, _source):
        self.update_page()

    def _ask(self, lid):
        self._asked = lid
        self.update_page()

    def _last_picture(self):
        """The project's last picture (cache/thumb.jpg), scaled for the page; None when it has none."""
        from pathlib import Path

        from PySide6.QtGui import QPixmap
        root = getattr(getattr(self.s, "p", None), "root", None)
        if root is None:
            return None
        f = Path(root) / "cache" / "thumb.jpg"
        if not f.is_file():
            return None
        pm = QPixmap(str(f))
        return None if pm.isNull() else pm.scaled(480, 320, Qt.KeepAspectRatio, Qt.SmoothTransformation)

    def coming_layers(self) -> list[tuple[str, int | None]]:
        """The layers Post will have, top first, as the graph counts them: (name as saved, workflow id)."""
        try:
            from .graphnode import blocks_of as Graph
            return list(Graph(self.s).post_layers)
        except Exception:  # noqa: BLE001 - a model without the graph's surface: its workflows' images
            return [(p.name, p.pid) for p in self.s.paths]

    def update_page(self):
        shown = any(p.rendered for p in self.s.paths) or any(lay["kind"] == "image" for lay in self.view.layers)
        self.pages.setCurrentIndex(1 if shown else 0)
        if shown:
            return
        clear(self.empty_list)
        coming = self.coming_layers()
        last = self._last_picture() if coming else None
        self.empty_title.setText(_("Results no longer kept") if last is not None else
                                 _("Nothing computed yet") if coming else _("No layers yet"))
        self.empty_note.setText(
            _("The project's last picture; its workflows' results are no longer kept (removed or outdated): compute "
              "them again to work on it here. The layers it will have:") if last is not None else
            _("The layers Post will have once their workflows are computed:") if coming else
            _("Each computed workflow becomes a layer here."))
        self.empty_pic.setVisible(last is not None)
        if last is not None:
            self.empty_pic.setPixmap(last)
        asked = self.view.layer(self._asked) if self._asked is not None else None
        if asked is None and self._asked is not None and hasattr(self.s, "post_layer"):
            asked = self.s.post_layer(self._asked)
        from . import names                               # default names follow the interface's language
        for name, _pid in coming:
            lb = label(_("{name}  ·  not computed yet").format(name=names.shown(tr_name(name))), "dim")
            if asked is not None and asked.get("name") == name:     # Open in Post of this layer's block
                lb.setStyleSheet(f"color: {ACCENT}; font-weight: 600;")
            self.empty_list.addWidget(lb, 0, Qt.AlignHCenter)

    def add_image_layer(self, name: str, source, key: str) -> dict:
        """Inspect sends a stage's image to Post: a stack, a gradient-removed image, a denoised image. It becomes a
        normal layer on top of the others: stretched automatically (Sky, Contrast, Saturation; channels linked or each
        apart), with a blend mode, an opacity, a mask and filters of its own.

        ``source``: the linear image - an array (H, W), (H, W, 3) or (3, H, W), a file (.npy, .fits, .tif, .png) or a
        function returning one (called in the render worker, never in the GUI thread); a function with ``neutral =
        True`` (``services/inspect.post_source``) is an image still in the camera's colours: its channels are
        stretched apart. ``key``: the stage result it
        is, saved in the project instead of the pixels; when the project is opened again the image is found from it:
        "<stage>/<store key>[/<array>]" is read from the project's store (``store/<stage>/<store key>/``, the array's
        name with or without .npy; without it, the stage's main image), any other key through ``image_resolver``.
        Returns the new layer (selected)."""
        lay = self.view.add_image_layer(name, source, key)
        self.update_page()
        return lay

    def set_image_resolver(self, fn):
        """key -> source (as ``add_image_layer``'s) for the image layers of a composition opened again whose key is not
        a store path; None when it is not there. Called in the render worker."""
        self.view.worker.set_image_resolver(fn)
        if any(lay["kind"] == "image" for lay in self.view.layers):
            self.view.request(final_only=True, force=True)           # (a first image may have been drawn without them)

    def refresh(self):
        self.update_page()
        if self.pages.currentIndex() == 1:
            self.view.worker.refresh()
            self.view.refresh()

    def release_memory(self):
        """Leaving Post: the render engines and the caches (layers' images, steps, masks' planes) go, with the freed
        pages; the next render, on coming back, opens them again."""
        self.view.worker.release()
        self.view._rendered_sig = None                  # the next request renders again, whatever changed

    #: below this much memory the engine may still fill (the machine's free memory, or the Memory limit of the
    #: Settings less what FLOAT holds), leaving Post hands its engines and caches back
    LEAVE_ROOM_BYTES = 2 << 30

    def leave(self):
        """Leaving Post for another page: its engines and caches are kept, so that coming back shows the picture at once
        (releasing them cost 2 to 5 s on each return, on a laptop), unless memory is short: then they are released."""
        from ..core import devices
        mb = devices.memory_budget()
        if mb is not None and mb[0] < self.LEAVE_ROOM_BYTES:
            self.release_memory()

    def export(self, parent=None):
        """File > Export image…"""
        if self.pages.currentIndex() == 1:
            self.view.export_dialog(parent)

    def shutdown(self):
        self.view.shutdown()
