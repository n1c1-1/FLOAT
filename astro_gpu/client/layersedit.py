"""Combine layers tuned in Inspect as a draft (``LayersEditor``), the family's editor (blockeditor.DraftEditor) as a
combination's is (combineedit.py), both on the Gradient editor's pattern (gradientedit.py).

Combine layers is a layer of Post: the stars put back on the starless image (a Stars/background block's two images,
or a combination's Output and the "· Stars" combination's). Its settings are that LAYER's, one source for Post's
panel, the block's panel in the graph and this editor: how the stars go back (the blend), the stars' sliders, the
starless image's stretch and sliders (render/sliders.py); for stars made by a combination of the stars alone, how its
filters' stars mix (Neutral or Palette), the setting of that block (its run makes the stars).

Every adjustment changes a DRAFT of the layer's settings: the project does not change until Apply (one step of the
window's Undo: the layer's settings, and the stars' combination's when it changed), Reset, or the question asked when
the block is left with a draft. Try variant is greyed: a layer of Post has no variant of the workflow (its versions are
Post's). Which settings are available, and why one is greyed, is Post's rule (post3.layer_slider_state).

The draft is seen at once on a reduced copy (about gradientedit.PREVIEW_PIXELS): the layer rendered by Post's own
renderer (services/render.Renderer: the same composition, its masks and its own filters), at the pyramid level that
copy is, in the editor's thread; Draft, Applied (the layer at its applied settings), Split.
"""
from __future__ import annotations

import copy
import math
import time
from pathlib import Path

import numpy as np
from PySide6.QtWidgets import QComboBox, QHBoxLayout

from .blockeditor import DraftEditor, choice_card, lit
from .i18n import _
from .widgets import label

#: the draft's keys of a layer's sliders: "slider.<key>" (render/sliders.py), the Arcsinh's Strength among them
SLIDER = "slider."
#: the one-line reason Try variant is greyed
NO_VARIANT = "Try variant takes the layer's settings alone (a new version of Post) or the stars' setting alone"
#: the stars' combination's setting changed: what it asks
STARS_RUN = "Needs a run: the stars' combination is computed again by Run"


def _texts():
    from ..render.sliders import STRETCH_KEY
    from .post3 import offset_text, strength_text
    return lambda k: strength_text if k == STRETCH_KEY else offset_text


def on_graph(s, node) -> bool:
    """Whether a Combine layers node is a block of the graph (the map's), not one made up for a workflow without."""
    try:
        return s.block_map.block_of(node) is not None
    except AttributeError:                             # a stand-in model without a map: as it is given
        return True


class LayersEditor(DraftEditor):
    """The Combine layers block of the workflow shown, tuned as a draft in Inspect's centre (module docstring)."""

    KEY = "composition"
    SIDE_PREF = "layers_settings_side"

    _starsFound = __import__("PySide6.QtCore", fromlist=["Signal"]).Signal()   # (another thread: a search ended)

    def __init__(self, tune):
        super().__init__(tune)
        from .postengine import STARS_FOUND_HOOKS  # a search of the Stars part's stars ended: shown again
        STARS_FOUND_HOOKS.append(self._starsFound.emit)
        self._starsFound.connect(self._stars_came)
        self.lid = None                        # the layer of Post it tunes
        self._renderer = None                  # Post's renderer, made in the preview's thread (module docstring)
        self._renderer_of = None
        self.level = None                      # the pyramid level of the reduced copy
        self.size = None                       # the reduced copy's size (w, h)
        self.drawn_ms = None

    # ------------------------------------------------------------ the block shown
    def wants(self, node, p) -> bool:
        """A Combine layers block of the graph over the workflow shown: never one a choice of workflow would make (a
        workflow without the block shows its own view, as a combination's editor does on a workflow without one)."""
        return str(self.tv.s.sel_step) == "composition" and getattr(node, "key", "") == "composition" and \
            bool(getattr(node, "paths", None)) and on_graph(self.s, node)

    def _enter_page(self, p):
        lid = None
        for cand in (getattr(self.s, "sel_layer", None), getattr(self.node, "lid", None)):   # the one chosen, else
            lay = self.s.post_layer(cand) if cand is not None and hasattr(self.s, "post_layer") else None   # the node's
            if lay and lay.get("part") == "combined" and lay.get("pid") == p.pid:
                lid = cand
                break
        if lid is None and hasattr(self.s, "default_post_layer"):
            # the layer Post shows by default for the workflow, made a layer of its own (no step: opening is never one)
            lid = self.s.default_post_layer(p.pid)
        if lid != self.lid:
            self.lid = lid
            self.applied_cache = None
            self.result = None
        super()._enter_page(p)

    def layer(self) -> dict:
        """The layer of Post as the project stores it."""
        try:
            return dict(self.s.post_layer(self.lid) or {}) if self.lid is not None else {}
        except Exception:  # noqa: BLE001 - the model cannot say now: nothing
            return {}

    def slider_infos(self):
        """Post's sliders of this layer, the stars first (what the block is for), as its panel shows them."""
        from ..render.sliders import STARS as STARS_GROUP
        try:
            infos = [x for x in self.s.slider_infos(self.p, "combined") if x.available]
        except Exception:  # noqa: BLE001 - not read yet: none
            infos = []
        return sorted(infos, key=lambda x: x.group != STARS_GROUP)

    def stars_combination(self):
        """(the workflow, its stage, its recipe, its stored settings) of the combination of the stars alone the layer's
        Stars side reads, None when its stars are another image (post3.PostView.stars_combination's rule)."""
        from ..core.workflow import combine_stage
        from .combineedit import stars_only
        from .graphnode import combine_spec
        lay = self.layer()
        pid = lay.get("stars_pid", lay.get("pid")) if lay else None
        q = self.s.view(pid) if pid is not None else None
        if q is None or not stars_only(((getattr(q, "source", None) or {}).get("fusion") or {}).get("inputs") or []):
            return None
        try:
            stage = combine_stage(self.s.p.workflow(q.pid))
        except (KeyError, AttributeError, ValueError):
            return None
        spec = combine_spec(q)
        if not stage or not (spec or {}).get("narrowband"):
            return None
        stored = self.s.combine_stored(q.pid) if hasattr(self.s, "combine_stored") else \
            dict(self.s.p.workflow(q.pid).params.get(stage) or {})
        return q, stage, spec, stored

    def applied(self, p=None) -> dict:
        """The layer's settings: its blend, its stretch, each of its sliders' positions (2: the suggestion), and the
        stars' combination's Stars when its stars are one."""
        from ..render.sliders import STRETCH_KEY
        from .post3 import BLENDS
        from .postengine import layer_stretch
        lay = self.layer()
        sliders = lay.get("sliders") or {}
        out = {"blend": lay.get("blend") or BLENDS[0], "stretch": layer_stretch(lay)}
        for sd in self.slider_infos():
            out[SLIDER + sd.key] = float(sliders.get(sd.key, 2.0))
        out[SLIDER + STRETCH_KEY] = float(sliders.get(STRETCH_KEY, 2.0))
        got = self.stars_combination()
        if got is not None:
            from .combineedit import applied_values, stage_class
            q, stage, spec, stored = got
            st = applied_values(stage_class(stage), spec, stored).get("stars")
            if st is not None:
                out["stars"] = st
        return out

    def norm(self, key: str, value):
        return float(value) if key.startswith(SLIDER) else value

    def shape(self, p) -> tuple:
        from .i18n import language
        return (self.lid, tuple(sd.key for sd in self.slider_infos()), self.stars_combination() is not None,
                self.values().get("stretch"), language())

    def state_of(self, p=None) -> str:
        """The block's state as the graph shows it (the Stars/background's images it combines)."""
        p = p or self.p
        try:
            g = self.s.block_map
            n = self.node if self.node is not None and p is self.p else g.node_of(p, "composition")
            return g.state(n) if n is not None else "pending"
        except Exception:  # noqa: BLE001 - no map now: computed
            return "done"

    def title(self, q=None) -> str:
        """The layer's name as Post's row shows it (post3.layer_name, the name the graph shows its workflow by in it):
        the owner every step of it names."""
        from .post3 import layer_name, retitle
        lay = self.layer()
        q = q or self.p
        if not lay and q is not None:
            from .model import POST_PART_NAMES
            from .names import neutral
            lay = {"kind": "workflow", "pid": q.pid, "name": neutral(q.name) + " · " + POST_PART_NAMES["combined"]}
        if not lay:
            return _("Combine layers")
        retitle(self.s, lay)
        return layer_name(lay)

    def variant_why(self) -> str:
        """Try variant takes Post's layer alone (a new version of Post) or the stars' combination's setting alone (a
        variant of that workflow): not both at once."""
        d = self._draft()
        return _(NO_VARIANT) if d and "stars" in d.vals and len(d.vals) > 1 else ""

    def layer_only(self, q, ch) -> bool:
        return ch is not None and "stars" not in ch["values"]

    def _apply_in(self, q, ch, twin):
        if self.lid is not None:
            self.lid = twin(self.lid)
            self.s.sel_layer = self.lid
        self._apply_change(q, ch)

    def _variant_target(self, q, ch):
        got = self.stars_combination()
        what, short = ch["what"], ch["short"]
        return got[0], {"kind": "combine", "values": {"stars": ch["values"]["stars"]}, "what": what, "short": short}

    def cost_text(self, pid: int | None = None) -> str:
        d = self._draft(pid)
        if not d:
            return ""
        return _(STARS_RUN) if "stars" in d.vals else _("Post's layer only: nothing is computed again")

    def _needs_run(self, q, ch) -> bool:
        return "stars" in ch["values"]

    # ------------------------------------------------------------ the draft
    def set_slider(self, key: str, pos: float, undo: bool = True) -> bool:
        return self.set_draft({SLIDER + key: float(pos)}, undo)

    def name_of(self, key: str) -> str:
        """A setting of the draft as Post names it."""
        from ..render.sliders import BY_KEY, STRETCH_KEY
        if key == "blend":
            return _("Star blend mode")
        if key == "stretch":
            return _("Stretch")
        if key == "stars":
            return _("Stars")
        k = key[len(SLIDER):]
        if k == STRETCH_KEY:
            return _("Stretch strength")
        from .post3 import slider_name
        return slider_name(k, BY_KEY[k].name) if k in BY_KEY else k

    def word_of(self, key: str, v) -> str:
        from .combineedit import STARS_NAME
        from .postengine import STRETCH_NAMES
        if key == "blend":
            return _(v)
        if key == "stretch":
            return _(STRETCH_NAMES.get(v, v))
        if key == "stars":
            return _(STARS_NAME.get(v, v))
        return _texts()(key[len(SLIDER):])(float(v))

    def parts(self, pid: int | None = None) -> list[tuple[str, str, str]]:
        q = self._view_of(pid)
        d = self.drafts.get(q.pid) if q is not None else None
        if not d:
            return []
        ap = self.applied(q)
        return [(k, self.name_of(k), f"{self.word_of(k, ap.get(k, v))} → {self.word_of(k, v)}") for k, v in d.vals.items()]

    def draft_change(self, pid: int | None = None) -> dict | None:
        """The draft as the layer's changes (each of them the model's kind "layer", graphmodel.GraphEditing), the stars'
        combination's Stars a change of that block (kind "combine"), with its words; None without a draft."""
        q = self._view_of(pid)
        d = self.drafts.get(q.pid) if q is not None else None
        if not d:
            return None
        what, short = self.step_words(q)
        return {"kind": "layers", "values": copy.deepcopy(d.vals), "short": short, "what": what}

    def _apply_change(self, q, ch):
        """One step of the window's Undo: the layer's settings (its layer made real by the first), the stars'
        combination's Stars."""
        s = self.s
        vals = dict(ch["values"])
        with s.batch():
            lid = self.lid if self.lid is not None else s.add_post_layer(q.pid, "combined", None, keep_default=False)
            self.lid = lid
            for k, v in vals.items():
                if k == "stars":
                    got = self.stars_combination()
                    if got is not None:
                        s.apply_as({"kind": "combine", "values": {"stars": v}, "what": ch["what"],
                                    "short": ch["short"]}, "replace", [got[0]])
                    continue
                if k == "blend":
                    one = {"blend": v}
                elif k == "stretch":
                    one = {"stretch": v}
                else:
                    one = {"slider": [k[len(SLIDER):], float(v)]}
                s.apply_change([q], {"kind": "layer", "lid": lid, "values": one})

    # ------------------------------------------------------------ the page
    def _fill_column(self, cv):
        """The stars (how they go back, their mix when a combination makes them, their sliders), then the starless
        image (its stretch, its sliders): Post's sections, Post's names, Post's ranges."""
        from ..render.sliders import STARS as STARS_GROUP
        from ..render.sliders import STRENGTH_STEP, STRETCH_KEY, group_title
        from ..render.sliders import top as slider_top
        from .combineedit import STARS, STARS_LINE, STARS_NAME
        from .post3 import BLENDS, ContSlider, layer_slider_state, slider_name
        from .postengine import LAYER_STRETCHES, STRETCH_NAMES
        u = self._ui
        lay = self.layer()
        group = None
        stretch_now = self.values().get("stretch")
        for sd in self.slider_infos():
            if sd.group != group:
                group = sd.group
                cv.addWidget(label(_(group_title(group, "combined")), "h2"))
                if group == STARS_GROUP:
                    row = QHBoxLayout()
                    row.addWidget(label(_("Star blend mode"), "dim"))
                    cb = QComboBox()
                    cb.setObjectName("blend")
                    for b in BLENDS:
                        cb.addItem(_(b), b)
                    cb.setToolTip(_("How the stars go back on the starless image: Screen (the usual way), Linear "
                                    "dodge (add) or Lighten"))
                    cb.activated.connect(lambda i, b=cb: self.set_draft({"blend": b.itemData(i)}))
                    row.addWidget(cb, 1)
                    cv.addLayout(row)
                    u["blend"] = cb
                    if self.stars_combination() is not None:
                        for key in STARS:          # the stars' combination: how its filters' stars mix
                            c = choice_card(_(STARS_NAME[key]), _(STARS_LINE[key]),
                                            _(STARS_LINE[key]) + "\n" + _(STARS_RUN))
                            c.clicked.connect(lambda k=key: self.set_draft({"stars": k}))
                            cv.addWidget(c)
                            u["stars_" + key] = c
                else:
                    row = QHBoxLayout()
                    row.addWidget(label(_("Stretch"), "dim"))
                    cb = QComboBox()
                    cb.setObjectName("stretch")
                    for k in LAYER_STRETCHES:
                        cb.addItem(_(STRETCH_NAMES[k]), k)
                    cb.setToolTip(_("How the linear image is brought to the screen. Automatic: fine for most images. "
                                    "Arcsinh: for an object far brighter than the sky (a planetary nebula, a galaxy "
                                    "core), which Automatic flattens near white; its structure and its colours are kept"))
                    cb.activated.connect(lambda i, b=cb: self.set_draft({"stretch": b.itemData(i)}))
                    row.addWidget(cb, 1)
                    cv.addLayout(row)
                    u["stretch"] = cb
                    if stretch_now == "arcsinh":
                        sl = ContSlider(_("Strength"), 0, 4, 2.0, 2, ends=[_("gentler"), _("stronger")],
                                        step=STRENGTH_STEP, show=_texts()(STRETCH_KEY))
                        self._wire_cont(sl, STRETCH_KEY)
                        cv.addWidget(sl)
                        u[SLIDER + STRETCH_KEY] = sl
            sl = ContSlider(slider_name(sd.key, sd.name), 0, slider_top(sd.key), 2.0, 2, ends=[_(e) for e in sd.ends],
                            show=_texts()(sd.key))
            sl.setObjectName(sd.key)
            ok, why = layer_slider_state(self.s, self.p, lay, sd.key, "combined", stretch_now)   # Post's rule
            sl.set_available(ok, why)
            self._wire_cont(sl, sd.key)
            cv.addWidget(sl)
            u[SLIDER + sd.key] = sl
        if not self.slider_infos():
            cv.addWidget(label(_("Its settings show once the Stars/background block's images are read."), "dim",
                               wrap=True))

    def _wire_cont(self, sl, key: str):
        """A slider of Post's kind (post3.ContSlider): the draft follows it as it moves, one step of Undo when it is
        let go (a key, a click on the rail, a double-click: one step each)."""
        def moved(v):
            if self._syncing:
                return
            self._mark()
            if self._drag is None:
                self.begin_drag()
            self.set_slider(key, v, undo=False)

        def done(v):
            if self._syncing:
                return
            if self._drag is None:
                self.begin_drag()
            self.set_slider(key, v, undo=False)
            self.end_drag()
        sl.moved.connect(moved)
        sl.done.connect(done)

    def _sync_controls(self):
        from .combineedit import STARS
        u = self._ui
        vals = self.values()
        for k, w in u.items():
            if k.startswith(SLIDER) and k[len(SLIDER):] and hasattr(w, "set_value"):
                w.set_value(float(vals.get(k, 2.0)))
        for k in ("blend", "stretch"):
            if k in u:
                u[k].setCurrentIndex(max(0, u[k].findData(vals.get(k))))
        for key in STARS:
            if "stars_" + key in u:
                lit(u["stars_" + key], vals.get("stars") == key)
        if self._shape is not None and self.p is not None and self._shape != self.shape(self.p):
            from PySide6.QtCore import QTimer  # the Arcsinh's Strength comes or goes: the column again
            QTimer.singleShot(0, lambda: self._enter_page(self.p) if self.p is not None else None)

    def tag_extra(self) -> str:
        from .postengine import STARS_FAILED, STARS_SEARCHING
        notes = getattr(self, "_notes", ())
        return "".join(" · " + _(n) for n in (STARS_SEARCHING, STARS_FAILED) if n in notes)

    def _stars_came(self):
        if self.p is not None and self.page_shown():
            self.applied_cache = None
            self._changed()

    def _measurements(self, dv):
        """What the layer combines (its two images' workflows) and the reduced copy the preview is made on."""
        self.title_of_group(dv, "Measurements")
        lay = self.layer()
        for words, pid in ((_("Starless image"), lay.get("pid")), (_("Stars"), lay.get("stars_pid", lay.get("pid")))):
            q = self.s.view(pid) if pid is not None else None
            name = (self.s.shown_title(q.pid) if hasattr(self.s, "shown_title") else q.name) if q is not None \
                else _("not wired")
            dv.addWidget(label(f"{words}  ·  {name}", "small"))
        if self.size is not None:
            dv.addWidget(label(_("Reduced copy: {w} × {h} px (1/{f} of the image)").format(
                w=self.size[0], h=self.size[1], f=2 ** int(self.level or 0)), "small"))

    def _drawer_sig(self) -> tuple:
        lay = self.layer()
        return (lay.get("pid"), lay.get("stars_pid"), self.size, self.level)

    def missing_text(self) -> str:
        if not self.slider_infos():
            return _("Waiting: the Stars/background block's images to read")
        return ""

    def applied_label(self) -> str:
        return _("Applied settings · reduced copy")

    # ------------------------------------------------------------ the preview: Post's renderer on this layer
    def _layers_for(self, vals: dict) -> list[dict]:
        """The layer (with ``vals``) and its own filters, as Post composes them: what the preview renders."""
        from . import postengine as pe
        try:
            alive = {q.pid for q in self.s.paths}
            layers = pe.load_layers(self.s.post_layers(), alive)
        except Exception:  # noqa: BLE001 - not readable now: none
            return []
        lay = next((x for x in layers if x.get("id") == self.lid), None)
        if lay is None:
            return []
        own = [x for x in layers if x.get("kind") == "filter" and x.get("target") == self.lid]
        lay = copy.deepcopy(lay)
        lay["visible"] = True
        lay["opacity"] = 100
        if "blend" in vals:
            lay["blend"] = vals["blend"]
        if "stretch" in vals:
            pe.set_layer_stretch(lay, vals["stretch"])
        sliders = dict(lay.get("sliders") or {})
        for k, v in vals.items():
            if k.startswith(SLIDER):
                if abs(float(v) - 2.0) < 1e-9:
                    sliders.pop(k[len(SLIDER):], None)
                else:
                    sliders[k[len(SLIDER):]] = float(v)
        lay["sliders"] = sliders
        fusion_of = getattr(self.s, "fusion_of", None)
        if fusion_of is not None and lay.get("starless") == "output":
            try:
                f, _why = fusion_of(lay["pid"], None, tune=lay.get("tune"))
            except Exception:  # noqa: BLE001 - not readable now: its plain layer
                f = None
            if f is not None:
                lay["fusion"] = f
        return [copy.deepcopy(x) for x in own] + [lay]

    def _request(self) -> dict:
        vals, ap = self.values(), self.applied()
        return {"gen": self.gen, "pid": self.p.pid, "project": self.s.p, "layers": self._layers_for(vals),
                "applied_layers": self._layers_for(ap) if (self.applied_cache is None or
                                                           self.applied_cache.get("key") != repr(sorted(ap.items()))
                                                           or self.applied_cache.get("pid") != self.p.pid) else None,
                "applied_key": repr(sorted(ap.items()))}

    def _render(self, layers, project):
        """(3, h, w) in 0-1: the layers by Post's renderer at the reduced copy's level (made once per project, in this
        thread)."""
        from .gradientedit import PREVIEW_PIXELS
        from .services.render import LayerSource, Renderer, Request, api_resolver, bases_of, choose_device
        if self._renderer is None or self._renderer_of is not project:
            from ..render.layers import LayerService
            dev = choose_device("auto")
            self._renderer = Renderer(LayerSource(api_resolver(project), LayerService(dev), bases_of(project)),
                                      device=dev)
            root = getattr(project, "root", None)            # the stars the Stars mask part finds: kept as Post's
            self._renderer.source.stars_dir = Path(root) / "cache" / "stars" if root else None
            self._renderer_of = project
        r = self._renderer
        canvas = r.composer.canvas_of(layers)
        if canvas is None:
            raise ValueError(_("The Stars/background block's images are not computed"))
        W, H = canvas.size(0)
        self.level = max(0, int(math.ceil(math.log2(max(1.0, math.sqrt(W * H / PREVIEW_PIXELS))))))
        res = r.run(Request(kind="half", layers=layers, level=self.level))
        if res.rgb is None:
            raise ValueError(_("The Stars/background block's images are not computed"))
        self._notes = set(res.notes or ())
        self.size = (int(res.rgb.shape[1]), int(res.rgb.shape[0]))
        return np.ascontiguousarray(np.moveaxis(res.rgb.astype(np.float32) / 255.0, -1, 0))

    def _work(self, req: dict) -> dict:
        out = {"req": req}
        if not req["layers"]:
            out["error"] = _("The layer is not in Post yet")
            return out
        t0 = time.perf_counter()
        out["result"] = self._render(req["layers"], req["project"])
        out["ms"] = 1000 * (time.perf_counter() - t0)
        if req["applied_layers"]:
            try:
                out["applied"] = {"pid": req["pid"], "key": req["applied_key"],
                                  "img": self._render(req["applied_layers"], req["project"])}
            except Exception:  # noqa: BLE001 - not made: Applied greyed
                pass
        return out
