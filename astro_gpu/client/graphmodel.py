"""GraphEditing: the project model's editing of the workflows, with the stored graph (graphcore) as the truth.

Built beside ``ProjectModel`` (INTEGRATION §1, §12 step 4), mixed in ahead of it: ``class M(GraphEditing,
ProjectModel)``. The definition the engine runs (``self.p``, a ``Project2``) is compiled from the graph after every
change, into the SAME object (``id(model.p)`` never changes: the engine's per-project context is kept); the readers of
the model (states, keys, costs, Post, the views' ``graph(p)``...) keep reading it.

* ``apply(steps)``: graph core operations as ONE undo step (``operations.apply_many``), compiled into ``self.p``; the
  wires to Post a gesture removes take the image out of every version of Post (engine review 3, ``graphcore.post``).
* ``document``: what the new graph page edits (graphview/document.py): its ``graph``, ``view``, ``apply``, ``undo``...
  A separate object, because the page's names (``graph``, ``view``, ``state``, ``cost``, ``can_undo``) are names the
  model already serves with other meanings, which Post and the panel call.
* Every facade the panel, Post, Inspect, Sources, the calibration panel, the run and the services call keeps today's
  signature; a writer is re-expressed on core operations (or, where the core has no operation for it, on a copy of
  the stored graph edited in place, checked, then committed the same way). A writer not ported yet raises
  ``NotImplementedError`` with its name.
* What the kept parts of the model still write straight into the definition (the frames excluded by hand, a
  calibration frame left out, files of a session) is taken into the graph before the next step ("absorbed"). Any other
  change found in the definition that the graph did not make rebuilds the graph from the definition (INTEGRATION §7:
  "changed outside FLOAT"), counted in ``graph_rebuilds``.

Undo (§9): the snapshot keeps today's 9-tuple; index 0 is the graph's JSON (the frames excluded by hand and the view
left out: an exclusion is never a step), index 3 None, index 4 Post's JSON.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

from ..graphcore import model as gm
from ..graphcore.compile import compile_into
from ..graphcore.convert import convert
from ..graphcore.msg import M, say
from ..graphcore.operations import apply_many, new_view, normalise, owned_stages
from ..graphcore.shapes import Result
from .graphnode import BlockMap
from .i18n import _

#: the model's own name of a graph block kind that is not a node of its own engine name
_VIEW_OF_NAME = {"combine": "fusion", "session": "session", "calibration": "calibration", "image": "image",
                 "combine_layers": "composition"}
#: the Selection settings of the stack stage (owned by the Selection block, operations.SELECTION_KEYS)
_SELECTION = ("min_transparency", "max_fwhm_ratio", "selection", "selection_transparency")
#: the stages a stack holds for all its nights (calibrated together), its Output's, its Align on's
_ROW_STAGES = ("inventory", "calibrate", "render_simple", "align_on")
_PORT_OF_ROLE = {"SII": "S II", "Ha": "Hα", "OIII": "O III", "L": "L", "R": "R", "G": "G", "B": "B"}
_PORT_OF_IMAGE = {"starless": "Starless", "stars": "Stars only"}
_TITLE_OF_KIND = {"sho": "S-H-O", "ho": "H-O", "rgb": "RGB", "lrgb": "LRGB"}


def _mm():
    """The client's model module (imported when first needed: model.py will import this module at the swap)."""
    from . import model
    return model


class NotPorted(NotImplementedError):
    """A facade of the model not re-expressed on the graph yet."""


def _not_ported(name):
    raise NotPorted(f"GraphEditing.{name} is not ported to the graph yet")


class GraphRefused(ValueError):
    """A gesture the graph core refused: its reason, translated."""


# ---------------------------------------------------------------------------------------------- the page's document
class GraphDocument:
    """The document interface of the graph page (graphview/document.py) on a project model with ``GraphEditing``."""

    def __init__(self, model):
        self.model = model

    @property
    def graph(self):
        return self.model._g                  # read on every paint: what the model absorbed at its last step

    @property
    def view(self):
        return self.model.graph_view

    @property
    def tr(self):
        return _

    @property
    def revision(self):
        return self.model.revision

    def apply(self, steps) -> Result:
        return self.model.apply(steps)

    def undo(self) -> Result:
        if not self.model.undo(graph=True):
            return Result(False, reason="Nothing to undo")
        return Result(True, toast=M("Undone: {label}", label=self.model.restored or _("a change")))

    def redo(self) -> Result:
        if not self.model.redo(graph=True):
            return Result(False, reason="Nothing to redo")
        return Result(True, toast=M("Redone: {label}", label=self.model.restored or _("a change")))

    @property
    def can_undo(self) -> bool:
        return self.model.can_undo(graph=True)

    @property
    def can_redo(self) -> bool:
        return self.model.can_redo(graph=True)

    def state(self, block) -> str:
        m = self.model
        n = m.block_map.node(block)
        return m.block_map.state(n) if n is not None else "pending"

    def review(self, block):
        m = self.model
        n = m.block_map.node(block)
        if n is None or not n.paths:
            return None
        got = {m.review_state(q, n.ref) for q in n.paths}
        return next((x for x in ("paused", "pending", "reviewed") if x in got), None)

    def mark_reviewed(self, block):
        m = self.model
        n = m.block_map.node(block)
        pause = m.pause_point(only=n.paths[0], upto=n.ref) if n is not None and n.paths else None
        if pause is not None:
            m.mark_reviewed(pause)

    def cost(self) -> float:
        try:
            return float(self.model.cost().seconds)
        except Exception:  # noqa: BLE001 - a project that cannot be planned yet costs nothing to show
            return 0.0


# ---------------------------------------------------------------------------------------------- the mixin
class GraphEditing:
    """The graph as the truth of a project model (see the module's text). Mixed in ahead of ``ProjectModel``."""

    #: the client's block keys the converter reads when the project opens (False: the drawing's structure only)
    GRAPH_KEYS = True

    def __init__(self, project, *args, **kwargs):
        # the graph first, compiled into the definition before the worker reads it (no race with the warm job)
        g = self._convert(project)
        c = compile_into(g, project)
        super().__init__(project, *args, **kwargs)
        self._g = g
        self.compiled = c
        self.graph_view = new_view()
        self.revision = 0
        self.graph_rebuilds = 0                    # the times the definition was changed outside the graph
        self._bm = None
        self._absorbing = False
        self._g = self._with_post_wires(self._g)
        self._sig = self._def_sig()

    # ------------------------------------------------------------------ the graph
    def _convert(self, p):
        if self.GRAPH_KEYS:
            try:
                return convert(p)
            except Exception:  # noqa: BLE001 - keys cannot be read: the drawing's structure alone decides
                pass
        return convert(p, block_keys=lambda wid: [])

    @property
    def stored(self):
        """The stored graph (graphcore.model.Project): a new object after every accepted change."""
        self._absorb()
        return self._g

    @property
    def document(self) -> GraphDocument:
        """What the graph page edits (graphview/document.py)."""
        return GraphDocument(self)

    @property
    def block_map(self) -> BlockMap:
        """The blocks of the graph as Inspect, the run and the panel know them (one Node per block id)."""
        if self._bm is None or self._bm[0] != self.revision:
            self._bm = (self.revision, BlockMap(self.compiled, self, self._g))
        return self._bm[1]

    def _def_sig(self) -> str:
        """The compiled sections of the definition the graph decides, but the sessions (absorbed one by one)."""
        from dataclasses import asdict
        p = self.p
        return json.dumps([[asdict(w) for w in p.workflows], p.stack_mode, p.next_workflow_id], sort_keys=True,
                          default=str)

    def _commit(self, g):
        """The graph ``g`` becomes the truth: compiled into the same definition, the views read again."""
        self._g = g
        self.compiled = compile_into(g, self.p)
        self._rebuild_views()
        self._sig = self._def_sig()
        self.revision += 1

    def _absorb(self):
        """What the kept parts of the model wrote into the definition goes into the graph (the frames excluded by hand,
        the sessions' names, filters, files and frames left out, a session added); anything else the graph did not
        make rebuilds it from the definition."""
        if getattr(self, "_g", None) is None or self._absorbing:
            return
        self._absorbing = True
        try:
            if self._def_sig() != self._sig:
                self._rebuild_from_definition()
                return
            q = gm.clone(self._g)
            if not self._absorb_sessions(q):
                self._rebuild_from_definition()
                return
            q = self._with_post_wires(q)
            if gm.model_json(q) != gm.model_json(self._g):
                self._g = q
                self.compiled = compile_into(q, self.p)
                self._sig = self._def_sig()
                self.revision += 1
        finally:
            self._absorbing = False

    def _rebuild_from_definition(self):
        self.graph_rebuilds += 1
        self.log("Changed outside the graph: graph rebuilt from the definition")
        g = self._convert(self.p)
        self._g = g
        self.compiled = compile_into(g, self.p)
        self._rebuild_views()
        self._g = self._with_post_wires(self._g)
        self._sig = self._def_sig()
        self.revision += 1

    def _absorb_sessions(self, q) -> bool:
        """The sessions of the definition into the graph ``q`` (in place); False when a session went or the order changed
        (the graph is then rebuilt)."""
        p = self.p
        exc = [[int(a), str(b)] for a, b in p.excluded]
        if [list(x) for x in q.excluded] != exc:
            q.excluded = exc
        have = [n["sid"] for n in q.nights]
        ids = [s.id for s in p.sessions]
        if ids[:len(have)] != have:
            return False
        by = {s.id: s for s in p.sessions}
        for sid in ids[len(have):]:                       # a session added to the definition: a night no stack draws yet
            s = by[sid]
            q.nights.append({"sid": sid, "name": s.name, "filter": s.filter,
                             "files": {k: list(getattr(s, k)) for k in gm.FILE_KINDS},
                             "left_out": copy.deepcopy(s.left_out)})
        for n in q.nights:
            s = by[n["sid"]]
            if n.get("name") != s.name:
                n["name"] = s.name
            if n.get("filter", "") != s.filter:
                n["filter"] = s.filter
            if (n.get("left_out") or {}) != (s.left_out or {}):
                n["left_out"] = copy.deepcopy(s.left_out)
            blocks = [b for b in q.blocks.values() if b.kind == "session" and b.sid == s.id]
            for b in blocks:
                if b.name != s.name:
                    b.name = s.name
            for k in gm.FILE_KINDS:
                want = list(getattr(s, k))
                if blocks:
                    for b in blocks:
                        eff = b.files[k] if k in b.files else q.common["files"].get(k, [])
                        if list(eff) != want:
                            b.files[k] = list(want)
                elif list((n.get("files") or {}).get(k) or []) != want:
                    n.setdefault("files", {})[k] = want
        return True

    # ------------------------------------------------------------------ Post's wires (INTEGRATION 4.7)
    def shared_post_images(self) -> set:
        """The images some version of Post shows, and those Post shows by default: (workflow id, part)."""
        from ..graphcore.post import shared_images
        try:
            defaults = {(q.pid, self.post_default_part(q)) for q in self.post_defaults()}
        except Exception:  # noqa: BLE001 - views not built yet
            defaults = set()
        return shared_images(getattr(self.p, "post", None) or {}, defaults)

    def _with_post_wires(self, g):
        """The graph whose wires to Post are those of the shared images (nothing else changes)."""
        from ..graphcore.post import with_post_wires
        try:
            q = with_post_wires(g, self.shared_post_images())
        except Exception:  # noqa: BLE001 - a graph the projection cannot read keeps its wires
            return g
        if [(w.src, w.dst, w.wid) for w in q.wires] == [(w.src, w.dst, w.wid) for w in g.wires]:
            return g
        return q

    def _post_follow(self, before, after, wired=False):
        """A wire to Post removed in the graph takes its image out of every version; one added for an image Post does
        not show yet puts it in every version. An image whose block went is Post's own pruning's."""
        from ..graphcore.post import place_of, wired_images
        a, b = wired_images(before), wired_images(after)
        alive = {q.pid for q in self.paths}
        shared = self.shared_post_images()
        # a wire taken away while its image still leaves the same block (not a block gone or replaced: the projection
        # draws that image's wire from its new place)
        acts = [("remove", wid, part) for wid, part in sorted(a - b, key=str)
                if wid in alive and place_of(after, wid, part) is not None
                and place_of(after, wid, part) == place_of(before, wid, part)]
        # an image wired to Post by the user's own wire (not the wire a new row's gesture draws by itself) reaches every
        # version when Post does not show it yet
        acts += [("add", wid, part) for wid, part in sorted(b - a, key=str)
                 if wired and wid in alive and (wid, part) not in shared]
        if not acts:
            return
        active = self.p.post["active"]
        for what, wid, part in acts:
            if what == "add":
                self.add_post_layer(wid, part, None)
                continue
            for v in list(self.p.post["versions"]):
                self.set_post_version(v["id"])
                self.remove_post_part(wid, part, None)
            self.set_post_version(active)

    # ------------------------------------------------------------------ applying
    def apply(self, steps, view=None) -> Result:
        """Graph core operations as ONE undo step: ``steps`` = [(operation, kwargs | f(results so far) -> kwargs)].
        Refused: nothing changes, the Result says why."""
        return self._change(list(steps), view=view)

    def _change(self, steps=(), raw=None, view=None, name=None) -> Result:
        """Operations, then ``raw(graph)`` (an edit the core has no operation for, on a copy), as one step."""
        self._absorb()
        before = self._g
        g, res = (apply_many(before, list(steps), self.graph_view if view is None else view, _) if steps
                  else (before, Result(True)))
        if not res.ok:
            return res
        if raw is not None:
            q = gm.clone(g)
            out = raw(q)
            if out is False:
                return Result(False, reason="nothing changes")
            normalise(q)
            try:
                gm.check(q)
            except gm.Invalid as e:                       # an edit the stored model cannot hold: nothing changes
                return Result(False, reason=str(e))
            g = q if gm.model_json(q) != gm.model_json(g) else g
        if g is before:
            return res
        with self._edit():
            self._commit(g)
            post = gm.Index(before).post
            self._post_follow(before, g, wired=any(s[0] == "wire" and isinstance(s[1], dict) and s[1].get("dst") == post
                                                   for s in steps))
            g2 = self._with_post_wires(self._g)
            if g2 is not self._g:
                self._g = g2
        text = name or (say(res.toast, _) if res.toast else "")
        if text:
            self.name_step(text)
        return res

    def _do(self, steps=(), raw=None, name=None) -> Result:
        """``_change``, a refusal raised as ValueError with its reason (translated): the facades' way."""
        res = self._change(steps, raw=raw, name=name)
        if not res.ok:
            raise GraphRefused(say(res.reason, _) or _("refused"))
        return res

    # ------------------------------------------------------------------ undo (INTEGRATION §9)
    def _graph_json(self) -> str:
        d = gm.to_dict(self._g)
        d.pop("excluded", None)
        return json.dumps(d, sort_keys=True, ensure_ascii=False, default=str)

    def _snapshot(self):
        mm = _mm()
        from dataclasses import asdict
        self._absorb()
        post = json.dumps(getattr(self.p, "post", None) or {}, sort_keys=True)
        return mm._Snap((self._graph_json(), self.p.current, self.p.stack_mode, None, post, asdict(self.p.target),
                         self.p.description, self.p.name, dict(self.p.acquisition)))

    def _restore(self, src: list, dst: list) -> bool:
        """The graph and Post put back as an earlier snapshot had them, compiled into the same definition."""
        from ..core.project import Target
        if not src:
            return False
        now = self._snapshot()
        now.from_post = src[-1].from_post
        dst.append(now)
        had = {v.pid: self.block_keys(v) for v in self.paths}
        old_keys = self.block_keys(self.path) if self.path is not None else []
        sel = self.graph_view.get("selected")
        old_g = self._g
        pred = None
        if sel in old_g.blocks:
            try:
                pred = gm.Index(old_g).pred(sel)
            except (KeyError, IndexError):
                pred = None
        files_before = [(s.id, *(tuple(getattr(s, k)) for k in gm.FILE_KINDS)) for s in self.p.sessions]
        gjson, current, _mode, _none, post, target, description, name, acquisition = src.pop()
        self.p.target, self.p.description, self.p.name, self.p.acquisition = Target(**target), description, name, \
            dict(acquisition)
        g = gm.from_dict(json.loads(gjson))
        g.excluded = [[int(a), str(b)] for a, b in self.p.excluded]          # the hand exclusions are not undone
        g.excluded = [x for x in g.excluded if any(n["sid"] == x[0] for n in g.nights)]
        self._g = g
        self.compiled = compile_into(g, self.p)
        if current is not None and any(w.id == current for w in self.p.workflows):
            self.p.current = current
        self._sig = self._def_sig()
        self.revision += 1
        post_back = json.loads(post) != (getattr(self.p, "post", None) or {})
        in_post = self.sel_step in ("final", "composition")
        if post_back:
            here = getattr(self.p, "post", None) or {}
            back = json.loads(post)
            if in_post and here.get("active") is not None and {v["id"] for v in back.get("versions", [])} == \
                    {v["id"] for v in here.get("versions", [])} and any(
                    v["id"] == here["active"] for v in back.get("versions", [])):
                back["active"] = here["active"]
            self.p.post = back
            post_back = back != here
        files_after = [(s.id, *(tuple(getattr(s, k)) for k in gm.FILE_KINDS)) for s in self.p.sessions]
        if files_after != files_before:
            self.reload_sources()
        self.pending_change = None
        self.edited()
        self._sync()
        self._unpatch_rows()
        # the selection on block ids: a block that came back is selected; one gone leaves its predecessor selected
        if sel is not None and sel not in g.blocks and not any(r.id == sel for r in g.rows):
            self.graph_view["selected"] = pred if pred in g.blocks else None
        if self.paths:
            self.current = next((i for i, v in enumerate(self.paths) if v.pid == self.p.current), 0)
            back = [(i, k) for i, v in sorted(enumerate(self.paths), key=lambda iv: iv[0] != self.current)
                    for k in self.block_keys(v) if k != "post" and k not in had.get(v.pid, [])]
            keys = self.block_keys(self.path)
            if in_post:
                pass
            elif back:
                self.current, self.sel_step = back[0]
            elif self.sel_step and self.sel_step not in ("sources", "final") and self.sel_step not in keys:
                j = old_keys.index(self.sel_step) if self.sel_step in old_keys else len(old_keys)
                self.sel_step = next((k for k in reversed(old_keys[:j]) if k in keys), keys[-1] if keys else "sources")
        self.pathsChanged.emit()
        self.currentChanged.emit()
        if post_back:
            self.postChanged.emit(None)
        return True

    # the hooks of ProjectModel._edit that edited the workflows after a change: the graph does it, or nothing is needed
    def _sync_graphs(self):
        """Today the graphs follow options set by code that speaks options: here the blocks are the truth."""

    def _prune_breakpoints(self):
        """A block that leaves a workflow takes its pause with it: the pause is the block's (compiled)."""

    def _follow_branch_names(self):
        """A block's image sent to Post keeps a name saying what it is ("Ha · Gradient" becomes "Ha · Denoise" once its
        last block changes); a name the user gave stays. On the rows and taps of the graph."""
        if getattr(self, "_g", None) is None:
            return
        mm = _mm()
        from .graphnode import BY_KEY
        titles = {mm.ProjectModel.branch_name("", k)[3:] for k in BY_KEY}
        names = {w.name for w in self.p.workflows}
        renames = {}
        for w in self.p.workflows:
            flow, sep, title = w.name.partition(" · ")
            if not sep or w.blocks is None or flow not in names - {w.name}:
                continue
            base = title.rsplit(" ", 1)[0] if title not in titles and title.rsplit(" ", 1)[-1].isdigit() else title
            on = [n for n in w.blocks if not n.get("off") and n.get("block") != "output"]
            if base not in titles or not on:
                continue
            want = mm.ProjectModel.branch_name(flow, mm.node_key(on[-1]))
            if want != f"{flow} · {base}":
                renames[w.id] = self.unique_name(want)
        if not renames:
            return
        olds = {w.id: w.name for w in self.p.workflows if w.id in renames}
        q = gm.clone(self._g)
        for f in list(q.rows) + list(q.taps):
            if f.wid in renames:
                f.name = renames[f.wid]
        self._g = q
        self.compiled = compile_into(q, self.p)
        self._sig = self._def_sig()
        self.revision += 1
        for wid, name in renames.items():
            self._rename_layers(wid, name, olds.get(wid))

    def reload_sources(self):
        self._absorb()
        super().reload_sources()

    # ------------------------------------------------------------------ finding things in the graph
    def _flow(self, p, g=None):
        """The row or tap of a workflow (a view, or its id), None."""
        g = g or self._g
        wid = int(getattr(p, "pid", p))
        return next((f for f in list(g.rows) + list(g.taps) if f.wid == wid), None)

    def _row(self, p, g=None):
        f = self._flow(p, g)
        if f is None or not isinstance(f, gm.Row):
            raise GraphRefused(_("not in workflow"))
        return f

    def _chain(self, p, g=None):
        """The block ids of a workflow's chain, in order (its nights excluded)."""
        g = g or self._g
        f = self._flow(p, g)
        if f is None:
            return []
        ix = gm.Index(g)
        if isinstance(f, gm.Row):
            return ix.effective(f.id)
        if f.block not in ix.row_of:
            return []
        eff = ix.effective(ix.row_of[f.block])
        return eff[:eff.index(f.block) + 1]

    def _bid(self, p, key, g=None):
        """The graph block of a view key ("fond~2", "calibration@3", "fusion", "session@2") in a workflow, None."""
        g = g or self._g
        wid = int(getattr(p, "pid", p))
        base, at, sid = key.partition("@")
        mm = _mm()
        for bid, pairs in (self.compiled.map or {}).items():
            blk = g.blocks.get(bid)
            if blk is None:
                continue
            for w, n in pairs:
                if w != wid:
                    continue
                k = _VIEW_OF_NAME.get(n) or mm.view_key(n)
                if k == base and (not at or blk.sid == int(sid)):
                    return bid
        if base == "sources":
            f = self._flow(p, g)
            root = self._root(f, g) if f is not None else None
            if root is not None and root.nights:
                return root.nights[0][0]
        return None

    def _root(self, f, g=None):
        g = g or self._g
        ix = gm.Index(g)
        rid = f.id if isinstance(f, gm.Row) else ix.row_of.get(f.block)
        return ix.row[ix.root(rid)] if rid in ix.row else None

    def _view_of_wid(self, wid):
        self._rebuild_views()
        return self.view(wid)

    def _made_wid(self, res, kind="rows"):
        """The workflow id of the first row (or tap) a gesture made."""
        ids = (res.created or {}).get(kind) or []
        for rid in ids:
            w = self.compiled.wids.get(rid)
            if w is not None:
                return w
        return None

    def _wids(self) -> set:
        return {f.wid for f in list(self._g.rows) + list(self._g.taps) if f.wid is not None}

    def _new_view(self, before: set):
        """The view of the workflow a gesture made (its id not among ``before``)."""
        wid = next((w.id for w in self.p.workflows if w.id not in before), None)
        return self._view_of_wid(wid) if wid is not None else None

    def _sid_block(self, sid, g=None):
        """The Session block drawing a session (the first of them), None for a night no stack draws."""
        g = g or self._g
        return next((b.id for b in g.blocks.values() if b.kind == "session" and b.sid == sid), None)

    # ------------------------------------------------------------------ the blocks of a workflow (Post, the panel)
    def insert_block(self, p, block: str, index: int | None = None, method: str | None = None) -> str:
        mm = _mm()
        wfl = mm.wfl
        g, src = self.graph(p), self.source_kind(p)
        if block == "color" and self.is_mono(p):
            raise ValueError(_(wfl.MONO_NO_COLOUR))
        if index is None:
            index = wfl.insert_place(g, src, block)
            if index is None:
                raise ValueError(self.palette_reason(p, block))
        e = wfl.place_problem(g, src, block, index)
        if e is not None:
            raise ValueError(mm.refused_text(e))
        if block == "align_on":
            self.set_align_on(p, None, method or "stars")
            return "align_on"
        row = self._row(p)
        after = None
        for n in reversed(g[:index]):                     # the block it follows in the graph (the nights' Calibration
            if n["block"] == "calibration":               # and the row's Align on are not blocks of the chain)
                break
            k = mm.node_key(n) if n["block"] != "align_on" else None
            if k is None:
                continue
            after = self._bid(p, k)
            if after is not None:
                break
        ix = gm.Index(self._g)
        if after is not None and after not in row.blocks:
            if row.fork and after == row.fork[1]:
                at = {"type": "insert", "row": row.id, "after": None}
            else:                                         # a block of the part the row shares: in the row holding it
                at = {"type": "insert", "row": ix.row_of[after], "after": after}
        else:
            at = {"type": "insert", "row": row.id, "after": after}
        steps = [("add_block", {"kind": block, "at": at})]
        if method is not None:
            steps.append(("set_method", lambda res: {"block": res[0].created["block"], "method": method}))
        res = self._do(steps)
        new = res.created["block"]
        name = next((n for w, n in self.compiled.map.get(new, []) if w == p.pid), block)
        return mm.view_key(name) if block in wfl.IMAGE_OPS else mm.view_key(block)

    def remove_node(self, p, key: str) -> None:
        why = self.removal_problem(p, key)
        if why:
            raise ValueError(why)
        if key == "align_on":
            self._do([("own_grid", {"row": self._row(p).id})])
            return
        bid = self._bid(p, key)
        if bid is None:
            raise ValueError(_("not in workflow"))
        self._do([("remove_block", {"block": bid})])

    def set_cut(self, paths, key: str, on: bool) -> bool:
        paths = [q for q in paths if self.is_cut(q, key) != on]
        if not paths:
            return False
        for q in paths:
            why = self.can_switch_off(q, key) if on else ""
            if why:
                raise ValueError(why)
        steps, seen = [], set()
        for q in paths:
            if key == "align_on":
                r = self._row(q)
                if r.id not in seen:
                    seen.add(r.id)
                    steps.append(("set_align_off", {"row": r.id}))
                continue
            bid = self._bid(q, key)
            if bid is None:
                raise ValueError(_("not in workflow"))
            if bid not in seen and self._g.blocks[bid].off != on:
                seen.add(bid)
                steps.append(("switch_off", {"block": bid}))
        if not steps:
            return False
        self._do(steps)
        return True

    def set_node_method(self, paths, key: str, method: str) -> bool:
        steps, seen = [], set()
        for q in paths:
            bid = self._bid(q, key)
            if bid is None or bid in seen or self._g.blocks[bid].method == method:
                continue
            seen.add(bid)
            steps.append(("set_method", {"block": bid, "method": method}))
        if not steps:
            return False
        self._do(steps)
        return True

    def set_breakpoint(self, paths, key: str, on: bool = True):
        """Pause after this block in these workflows (or no longer): on the block (its pause), or, for a block the
        graph holds per workflow (a night's Calibration, an Align on), in the workflow's own list."""
        mm = _mm()
        wfl = mm.wfl
        eb = mm.engine_name(key)
        if wfl.base(eb) not in wfl.BLOCKS:
            raise ValueError(f"no breakpoint on {key!r}: it is not a block of the engine")
        want = {}
        for q in paths:
            eb = mm.engine_name(self._bp_key(q, key))
            have = set(self.p.workflow(q.pid).breakpoints)
            new = sorted(have | {eb} if on else have - {eb}, key=wfl.block_order)
            if new != sorted(have, key=wfl.block_order):
                want[q.pid] = (eb, new)
        if not want:
            return

        def raw(g):
            ix = gm.Index(g)
            for wid, (eb, new) in want.items():
                f = self._flow(wid, g)
                bid = self._bid(wid, mm.view_key(eb) if "@" not in eb else eb, g)
                kind = g.blocks[bid].kind if bid in g.blocks else None
                pausable = kind in ("registration", "selection", "crop", "stacking", "astrometry", "gradient", "deconv",
                                    "denoise", "color", "stars", "synth", "image", "combine")
                if f.pauses is None and pausable:
                    for other in list(g.rows) + list(g.taps):   # the other workflows of a shared block keep theirs
                        if other.wid not in want and other.wid is not None and bid in self._chain(other.wid, g) \
                                and other.pauses is None and any(w.id == other.wid for w in self.p.workflows):
                            other.pauses = list(self.p.workflow(other.wid).breakpoints)
                    g.blocks[bid].pause = on
                else:
                    f.pauses = list(new)
            del ix
        self._do(raw=raw)

    # ------------------------------------------------------------------ settings (they are the blocks')
    def _owner(self, g, wid, stage):
        """Where a workflow's setting of ``stage`` is stored: ("block", id, stage) | ("row", row id, stage) |
        ("extra", None, stage)."""
        base, at, sid = stage.partition("@")
        f = self._flow(wid, g)
        if base in ("inventory", "calibrate") and at:
            root = self._root(f, g)
            c = next((c for s, c in (root.nights if root else []) if g.blocks[c].sid == int(sid)), None)
            return ("block", c, base) if c is not None else ("extra", None, stage)
        if stage in _ROW_STAGES:
            return ("row", f.id, stage) if isinstance(f, gm.Row) else ("extra", None, stage)
        q = gm.clone(g)
        for b in self._chain(wid, g):
            if stage in owned_stages(q, b):
                return "block", b, stage
        return "extra", None, stage

    def _put_setting(self, g, wid, stage, name, value):
        """A setting of a workflow stored where it belongs in the graph ``g`` (in place); the workflows' leftovers
        that would hide it are taken out."""
        from ..graphcore.operations import Refusal, _put, check_param
        try:
            v = None if value is None else check_param(stage, name, value, internal=name == "zone")
        except Refusal as e:
            raise ValueError(say(e.msg, _)) from None
        kind, where, st = self._owner(g, wid, stage)
        if stage == "stack" and kind == "block":
            own = "selection" if name in _SELECTION else "crop" if name == "zone" else "stacking"
            b = next((b for b in self._chain(wid, g) if g.blocks[b].kind == own), None)
            kind, where = ("block", b) if b is not None else ("extra", None)
        f = self._flow(wid, g)
        if kind == "block":
            blk = g.blocks[where]
            if blk.kind == "calibration":
                st = f"{st}@{blk.sid}"
            _put(blk.params, st, name, v)
            through = [x for x in list(g.rows) + list(g.taps) if x.wid is not None and where in self._chain(x.wid, g)]
            through += [x for x in g.rows if any(c == where for _s, c in x.nights)]
        elif kind == "row":
            ix = gm.Index(g)
            if stage in ("inventory", "calibrate"):
                root = ix.row[ix.root(where)]
                _put(root.calib, stage, name, v)
                through = [x for x in g.rows if ix.root(x.id) == root.id] + \
                    [t for t in g.taps if t.block in ix.row_of and ix.root(ix.row_of[t.block]) == root.id]
            elif stage == "render_simple":
                store = {"render_simple": f.output}
                _put(store, stage, name, v)
                f.output = store.get("render_simple", {})
                through = [f]
            else:
                r = ix.row[where]
                if not r.align:
                    raise ValueError(_("not in workflow"))
                _put(r.align.setdefault("params", {}), stage, name, v)
                through = [f]
        else:
            through = []
            _put(f.extra, stage, name, v)
            f.unset = [x for x in f.unset or [] if list(x) != [stage, name]]
            return
        for x in through:
            if x.extra and name in (x.extra.get(st) or {}):
                d = dict(x.extra[st])
                d.pop(name, None)
                x.extra = {k: vv for k, vv in x.extra.items() if k != st} | ({st: d} if d else {})
            x.unset = [u for u in x.unset or [] if list(u) != [st, name]]

    def set_param(self, p, stage: str, name: str, value, suggested=None):
        ids = [q.pid for q in (self.scope_paths() if p is None else [p])]
        if value is not None and suggested is not None and value == suggested:
            value = None
        self._set_params(ids, [(stage, name, value)])
        self._patch_rows(ids, stage, name, value)

    def _set_params(self, ids, items):
        """Settings stored in these workflows, as the engine stores them (a value equal to a known suggestion is not
        kept: api.set_param's rule, read through api.with_change; an invalid value raises ValueError naming it)."""
        from .. import api
        want = []
        for wid in ids:
            for stage, name, value in items:
                c = api.with_change(self.p, wid, {"kind": "param", "stage": stage, "name": name, "value": value})
                want.append((wid, stage, name, (c.params.get(stage) or {}).get(name)))

        def raw(g):
            for wid, stage, name, v in want:
                self._put_setting(g, wid, stage, name, v)
        self._do(raw=raw)

    def set_trunk_param(self, stage: str, name: str, value) -> bool:
        try:
            self._set_params([q.pid for q in self.paths], [(stage, name, value)])
        except ValueError:
            return False
        return True

    def apply_change(self, paths, ch) -> list[str]:
        """Replace: the change goes to these workflows (settings on the blocks holding them, a method on its block)."""
        mm = _mm()
        ids = [q.pid for q in paths]
        notes: list[str] = []
        if ch["kind"] in ("param", "params"):
            items = [(c["stage"], c["name"], c["value"]) for c in mm.ProjectModel._api_changes(ch)]
            self._set_params(ids, items)
            for q in paths:
                if ch["kind"] == "param":
                    self._patch_rows([q.pid], ch["stage"], ch["name"], ch.get("value"))
            return notes
        steps, seen = [], set()
        for q in paths:
            for bid, method in self._method_change(q, ch):
                if bid not in seen:
                    seen.add(bid)
                    steps.append(("set_method", {"block": bid, "method": method}))
        if steps:
            self._do(steps)
        return notes

    def _method_change(self, q, ch):
        """A change of an option that is a block's method: [(block, method)]; ◇ raises NotInEngine; an option that
        adds or takes a block is not ported."""
        mm = _mm()
        key, v = ch["key"], ch["value"]
        if key == "dn":
            if v not in mm.DN_ENGINE:
                raise mm.NotInEngine(mm.NOT_IN_ENGINE)
            kind, method = "denoise", mm.DN_ENGINE[v]
        elif key == "fond":
            kind, method = "gradient", mm.GRADIENT_ENGINE.get(v, "polynomial")
        elif key in ("deconv_m",):
            kind, method = "deconv", mm.DECONV_ENGINE.get(v, "rl")
        else:
            _not_ported(f"apply_change({key})")
        b = next((b for b in self._chain(q.pid) if self._g.blocks[b].kind == kind), None)
        return [(b, method)] if b is not None else []

    def add_variant(self, p, ch, title: str):
        """Try as variant: a copy of the workflow through the changed block, the change made on the copy."""
        mm = _mm()
        if ch["kind"] in ("param", "params"):
            items = [(c["stage"], c["name"], c["value"]) for c in mm.ProjectModel._api_changes(ch)]
            kind, bid, _st = self._owner(self._g, p.pid, items[0][0])
            if items[0][0] == "stack" and kind == "block":
                own = "selection" if items[0][1] in _SELECTION else "crop" if items[0][1] == "zone" else "stacking"
                bid = next((b for b in self._chain(p.pid) if self._g.blocks[b].kind == own), bid)
            if kind != "block":
                _not_ported("add_variant(a setting no block holds)")
            stage = items[0][0].partition("@")[0]
            steps = [("try_variant", {"block": bid, "change": {"param": [stage, items[0][1], items[0][2]]},
                                      "name": self.unique_name(title)})]
            for st, nm, v in items[1:]:
                steps.append(("set_param", lambda res, st=st, nm=nm, v=v: {"block": res[0].focus, "stage": st,
                                                                            "name": nm, "value": v}))
        else:
            pairs = self._method_change(p, ch)
            if not pairs:
                raise ValueError(_("not in workflow"))
            bid, method = pairs[0]
            steps = [("try_variant", {"block": bid, "change": {"method": method}, "name": self.unique_name(title)})]
        res = self._do(steps)
        return self._view_of_wid(self._made_wid(res))

    def keep_variant(self, q):
        self._do([("keep_variant", {"row": self._row(q).id})])
        self.current = self.index_of(q.pid)
        self.currentChanged.emit()

    def drop_variant(self, q):
        self._do([("discard_variant", {"row": self._row(q).id})])
        v = self.view(self.p.current) if self.p.current is not None else None
        self.current = self.paths.index(v) if v in self.paths else min(self.current, len(self.paths) - 1)
        self.currentChanged.emit()

    # ------------------------------------------------------------------ workflows: names, copies, deletion
    def rename(self, pid: int, name: str) -> bool:
        name = name.strip()
        if name == self.p.workflow(pid).name or self.name_problem(name, pid=pid):
            return False
        self._rename(pid, name)
        return True

    def _rename(self, pid: int, name: str):
        old = self.p.workflow(pid).name

        def raw(g):
            f = self._flow(pid, g)
            if f is None:
                raise GraphRefused(_("not in workflow"))
            f.name = name
        with self._edit():                                # its Post layers' names in the same step
            self._do(raw=raw)
            self._rename_layers(pid, name, old)

    def _rename_layers(self, pid, name, old=None):
        """Post's layers named after a workflow follow its new name (GUI exploration 3.4)."""
        if old is None:
            return
        for v in self.p.post["versions"]:
            v["layers"] = [{**x, "name": name + x["name"][len(old):]}
                           if x.get("kind") == "workflow" and x.get("pid") == pid and str(x.get("name", "")).startswith(old)
                           else x for x in v.get("layers") or []]

    def duplicate(self, src, name: str):
        """A copy of a workflow: the same blocks drawn once, a workflow of its own ending where src ends (a tap)."""
        new_name = self.unique_name(name)

        def raw(g):
            f = self._flow(src.pid, g)
            chain = self._chain(src.pid, g)
            if f is None or not chain:
                raise GraphRefused(_("not in workflow"))
            t = gm.Tap(gm.new_id(g, "t"), chain[-1], new_name)
            from ..graphcore.operations import _copy_flow
            _copy_flow(t, f)
            t.pauses = list(self.p.workflow(src.pid).breakpoints)
            t.wid, g.next_wid = g.next_wid, g.next_wid + 1
            g.order.append(t.wid)
            g.taps.append(t)
        before = self._wids()
        self._do(raw=raw)
        return self._new_view(before)

    def remove_path(self, i: int):
        pid = self.paths[i].pid
        self._delete_flow(pid)
        self.current = min(self.current, len(self.paths) - 1)
        self.currentChanged.emit()

    def _delete_flow(self, pid):
        f = self._flow(pid)
        if f is None:
            raise GraphRefused(_("not in workflow"))
        if isinstance(f, gm.Tap):
            self._do([("remove_tap", {"tap": f.id})])
            return
        ix = gm.Index(self._g)
        kids = [c for c in ix.children[f.id]]
        if kids:                                          # its variants become workflows of their own: not ported
            _not_ported("remove_path(a workflow its variants start from)")
        self._do([("delete_row", {"row": f.id})])

    # ------------------------------------------------------------------ a block's image to Post
    def send_to_post(self, p, key: str, name: str):
        if not self.can_send_to_post(p, key):
            raise ValueError(_("no Post from here"))
        bid = self._bid(p, key)
        if bid is None:
            raise ValueError(_("no Post from here"))
        before = self._wids()
        self._do([("send_to_post", {"block": bid, "name": self.unique_name(name)})])
        return self._new_view(before)

    # ------------------------------------------------------------------ an Image block
    def set_image_file(self, p, path: str | None = None, stretched: bool | None = None):
        from ..stages.image import stretched_by_nature
        src = dict(self.p.workflow(p.pid).source or {})
        before = stretched_by_nature(src.get("path") or "")
        if path is not None:
            src["path"] = Path(path).as_posix() if path else ""
        if stretched is not None:
            src["stretched"] = bool(stretched)
        elif path and stretched_by_nature(path) and not self.graph(p):
            src["stretched"] = True
        elif path and before and not self.graph(p):
            src["stretched"] = False

        def raw(g):
            f = self._flow(p.pid, g)
            root = self._root(f, g) if f is not None else None
            head = g.blocks[root.blocks[0]] if root is not None and root.blocks else None
            if head is None or head.kind != "image":
                raise GraphRefused(_("not in workflow"))
            head.source = src
        self._do(raw=raw)

    # ------------------------------------------------------------------ combinations
    def _source_port(self, pid, image):
        """(block, out port) a combination's input reading a workflow's image is wired from."""
        chain = self._chain(pid)
        if image in ("starless", "stars"):
            b = next((x for x in chain if self._g.blocks[x].kind == "stars"), None)
            return b, _PORT_OF_IMAGE[image]
        b = next((x for x in reversed(chain) if "image" in gm.out_ports(self._g.blocks[x])), None)
        return b, "image"

    def _combine_block(self, p):
        r = self._row(p)
        head = r.blocks[0] if r.blocks else None
        if head is None or self._g.blocks[head].kind != "combine":
            raise GraphRefused(_("not in workflow"))
        return head

    def _wire_steps(self, j, inputs):
        steps, fix = [], []
        for i in inputs:
            image = i.get("image", "output") or "output"
            src, port = self._source_port(int(i["pid"]), image)
            if src is None:
                raise ValueError(_("no workflow"))
            dport = _PORT_OF_ROLE.get(str(i["role"]), str(i["role"]))
            steps.append(("wire", (lambda res, src=src, port=port, dport=dport, ch=str(i.get("channel", "L")):
                                   {"src": src, "src_port": port, "dst": j(res), "dst_port": dport, "channel": ch})))
            fix.append((dport, int(i["pid"]), None if image == "output" else image))
        return steps, fix

    @staticmethod
    def _fix_wires(g, j, fix):
        """The input wires of combination j carry their workflow's id and image as the engine stores them."""
        for dport, pid, image in fix:
            w = next((w for w in g.wires if w.dst == [j, dport] and w.active), None)
            if w is not None:
                w.wid = pid
                w.image = image

    def add_fusion(self, name: str, kind: str, inputs: list[dict] | None = None):
        def j(res):
            return res[0].created["block"]
        steps = [("add_combination", {"type": kind, "name": self.unique_name(name)})]
        more, fix = self._wire_steps(j, inputs or [])
        steps += more

        def raw(g):
            jb = next(r.blocks[0] for r in reversed(g.rows) if r.blocks and g.blocks[r.blocks[0]].kind == "combine")
            self._fix_wires(g, jb, fix)
        before = self._wids()
        self._do(steps, raw=raw)
        return self._new_view(before)

    def remove_fusion(self, p):
        self._do([("delete_row", {"row": self._row(p).id})])
        self.current = min(self.current, len(self.paths) - 1)
        self.currentChanged.emit()

    def set_fusion_input(self, p, role: str, pid: int, image: str = "output") -> bool:
        f = self.p.workflow(p.pid).source["fusion"]
        cur = next((i for i in f["inputs"] if i["role"] == role), None)
        if cur is not None and cur["pid"] == int(pid) and cur.get("image", "output") == image:
            return False
        why = self.input_problem(p, role, int(pid), image)
        if why:
            raise ValueError(why)
        j = self._combine_block(p)
        ch = (cur or {}).get("channel") or self._default_channel(p, role, self.view(int(pid)))
        steps, fix = self._wire_steps(lambda res: j, [{"pid": pid, "role": role, "channel": ch, "image": image}])
        self._do(steps, raw=lambda g: self._fix_wires(g, j, fix))
        return True

    def set_fusion_channel(self, p, role: str, channel: str) -> bool:
        from ..stages.combine import CHANNELS
        f = self.p.workflow(p.pid).source["fusion"]
        cur = next((i for i in f["inputs"] if i["role"] == role), None)
        if cur is None or channel not in CHANNELS or cur.get("channel", "L") == channel:
            return False
        j = self._combine_block(p)
        self._do([("set_input", {"block": j, "port": _PORT_OF_ROLE.get(role, role), "channel": channel})])
        return True

    def set_fusion_image(self, p, role: str, image: str) -> bool:
        cur = next((i for i in self.p.workflow(p.pid).source["fusion"]["inputs"] if i["role"] == role), None)
        return cur is not None and self.set_fusion_input(p, role, cur["pid"], image)

    def set_fusion_weight(self, p, role: str, weight: float) -> bool:
        w = self.p.workflow(p.pid)
        st = _mm().wfl.combine_stage(w)
        cur = dict((w.params.get(st) or {}).get("weights") or {})
        weight = round(float(weight), 3)
        if cur.get(role, 1.0) == weight:
            return False
        j = self._combine_block(p)
        self._do([("set_input", {"block": j, "port": _PORT_OF_ROLE.get(role, role), "weight": weight})])
        return True

    def clear_fusion_input(self, p, role: str) -> bool:
        f = self.p.workflow(p.pid).source["fusion"]
        if not any(i["role"] == role for i in f["inputs"]):
            return False
        j = self._combine_block(p)
        port = _PORT_OF_ROLE.get(role, role)
        w = next((w for w in self._g.wires if w.dst == [j, port] and w.active), None)
        if w is None:
            return False
        self._do([("unwire", {"wire": w.id})])
        return True

    # ------------------------------------------------------------------ alignment (#165, INTEGRATION 4.3)
    def _align_from(self, ref_pid, block):
        """The block of the target workflow an Align on leaves from: ``block`` an engine node name, None."""
        if ref_pid is None or block is None:
            return None
        return next((b for b, pairs in self.compiled.map.items() for w, n in pairs if w == int(ref_pid) and n == block),
                    None)

    def set_align_reference(self, p, pid: int | None, block: str | None = "auto") -> bool:
        mm = _mm()
        n = self.align_node(p)
        if block == "auto":
            block = self.default_align_source(pid)
        elif block is not None:
            block = mm.engine_name(block)
        if n is None or (n.get("reference") == pid and n.get("reference_block") == block):
            return False
        if pid is not None:
            why = self.join_problem(p.pid, int(pid), True)
            if why:
                raise ValueError(why)
        ref = self._flow(int(pid)) if pid is not None else None
        self._do([("set_align", {"row": self._row(p).id, "ref": ref.id if ref is not None else None,
                                 "block": self._align_from(pid, block), "method": n.get("method")})])
        return True

    def set_align_on(self, p, reference: int | None, method: str = "stars", block: str | None = "auto") -> bool:
        mm = _mm()
        wfl = mm.wfl
        if self.align_node(p) is not None:
            return self.set_align_reference(p, reference, block)
        if block == "auto":
            block = self.default_align_source(reference)
        elif block is not None:
            block = mm.engine_name(block)
        if reference is not None:
            why = self.join_problem(p.pid, int(reference), True)
            if why:
                raise ValueError(why)
        g, src = self.graph(p), self.source_kind(p)
        index = wfl.insert_place(g, src, "align_on")
        if index is None:
            raise ValueError(self.palette_reason(p, "align_on"))
        e = wfl.place_problem(g, src, "align_on", index)
        if e is not None:
            raise ValueError(mm.refused_text(e))
        ref = self._flow(int(reference)) if reference is not None else None
        self._do([("set_align", {"row": self._row(p).id, "ref": ref.id if ref is not None else None,
                                 "block": self._align_from(reference, block) if reference is not None else None,
                                 "method": method})])
        return True

    def align_without_astrometry(self, p, reference: int) -> None:
        self.set_align_on(p, reference)
        astro = [b for b in self._chain(p.pid) if self._g.blocks[b].kind == "astrometry"]
        if astro:
            self._do([("remove_block", {"block": b}) for b in astro])

    # ------------------------------------------------------------------ templates (graphcore.templates)
    def _run_plan(self, plan):
        """A template's plan as one step: its operations, then Post's actions by the model's own Post functions."""
        with self._edit():
            res = self._change(plan.steps)
            if not res.ok:
                raise GraphRefused(say(res.reason, _))
            wids = self.compiled.wids
            rows = res.created["rows"]
            keys = {}
            wid_of = {r: wids.get(r) for r in rows} | dict(wids)
            for a in plan.post(rows):
                if a[0] == "add_layer":
                    keys[a[3]] = self.add_post_layer(wid_of[a[1]], a[2], None)
                elif a[0] == "side":
                    self.set_layer_side(keys[a[1]], a[2], wid_of[a[3]], a[4])
                elif a[0] == "remove_part":
                    self.remove_post_part(wid_of[a[1]], a[2], None)
                elif a[0] == "removed":
                    gone = {(int(x), y) for x, y in self.p.post.get("removed") or []} | \
                        {(wid_of[r], part) for r, part in a[1]}
                    self.p.post = {**self.p.post, "removed": sorted([list(x) for x in gone], key=str)}
        return res

    def _template(self, t, name, opts):
        from ..graphcore import templates
        return templates.build(self._g, t, name, opts, tr=_)

    def add_sho_rgb(self, name: str, kind: str, filters: list[str], opts: dict):
        self._absorb()
        t = {"kind": "fusion", "rgb_stars": True, "filters": list(filters), "palette": _TITLE_OF_KIND.get(kind, kind)}
        res = self._run_plan(self._template(t, name, opts))
        return self._view_of_wid(self._fusion_made(res))

    def add_dual_x2(self, name: str, opts: dict):
        self._absorb()
        t = {"kind": "fusion", "dual_x2": True, "palette": "S-H-O"}
        res = self._run_plan(self._template(t, name, opts))
        return self._view_of_wid(self._fusion_made(res))

    def _fusion_made(self, res):
        """The last combination a gesture made reading starless images (or the last one)."""
        out = None
        for rid in res.created["rows"]:
            r = next((x for x in self._g.rows if x.id == rid), None)
            if r is not None and r.blocks and self._g.blocks[r.blocks[0]].kind == "combine":
                out = r.wid if out is None or " · " not in r.name else out
        return out

    def add_filters_combination(self, name: str, kind: str, made: list, stars: bool):
        from ..graphcore import templates
        self._absorb()
        g = self._g
        plan = templates.Plan(g)
        rows = []
        for f, q in made:
            row = self._row(q)
            b = {}
            for x in self._chain(q.pid):
                b[g.blocks[x].kind] = x
            rows.append((f, row.id, b))
        if not stars:
            templates._combination(plan, name, kind, rows, "output")
        else:
            from ..graphcore.msg import name_of
            from ..graphcore.msg import TR
            token = TR.set(_)
            try:
                star_fus = templates._combination(plan, f"{name} · " + name_of("Stars"), kind, rows, "stars")
                fus = templates._combination(plan, name, kind, rows, "starless")
            finally:
                TR.reset(token)
            plan.actions += [("add_layer", fus, "combined", "lay"), ("side", "lay", "starless", fus, "output"),
                             ("side", "lay", "stars", star_fus, "output"), ("remove_part", fus, None),
                             ("remove_part", star_fus, None),
                             ("removed", [(fus, None), (star_fus, None)] + [(r, "combined") for _f, r, _b in rows])]
        res = self._run_plan(plan)
        return self._view_of_wid(self._fusion_made(res))

    def mono_blocks(self, opts):
        return _mm().ProjectModel.mono_blocks(opts)

    # ------------------------------------------------------------------ new workflows
    def _nodes_of(self, opts, src="frames", end=None, colour=True):
        mm = _mm()
        wfl = mm.wfl
        e = mm.KEY_ENGINE_BLOCK.get(end, end) if end else None
        return [dict(n) for n in wfl.from_options(mm.to_engine(dict(opts)), e, src)
                if not n["off"] and (colour or n["block"] != "color")]

    def _new_stack(self, name, sids, nodes, params=None, all_nights=False, end=None, options=None):
        """A stack drawing these sessions with these nodes' blocks, its settings stored where they belong.
        ``options``: a workflow made from options (today's ``add_workflow``): a row made before the free graph, kept
        as its options until the first change of its chain (as the converter keeps one)."""
        from ..graphcore.templates import chain_of
        chain = chain_of(nodes)
        if sids:
            steps = [("add_stack", {"name": name, "nights": list(sids), "blocks": chain, "align": None,
                                    "all_nights": all_nights})]
        else:
            steps = []

        def raw(g):
            if not sids:                                  # a project without a session yet: a row being built
                r = gm.Row(gm.new_id(g, "r"), name, [gm.new_block(g, c["kind"], method=c.get("method")).id
                                                     for c in chain], [])
                r.all_nights = all_nights
                g.rows.append(r)
                r.wid, g.next_wid = g.next_wid, g.next_wid + 1
                g.order.append(r.wid)
            r = g.rows[-1]
            if options is not None:
                r.legacy = {"options": copy.deepcopy(options), "end": end, "frozen": False}
            else:
                r.end = end
            for stage, kv in (params or {}).items():
                for k, v in kv.items():
                    self._put_setting_new(g, r, stage, k, v)
        before = self._wids()
        self._do(steps, raw=raw)
        return self._new_view(before)

    def _legacy_nodes(self, options, end, src, mono):
        """The blocks a workflow made from options draws (convert._legacy_nodes: a mono one has no Color)."""
        wfl = _mm().wfl
        return [dict(n) for n in wfl.from_options(options, end, src)
                if not n["off"] and n["block"] not in ("image", "output", "align_on", "calibration")
                and not (mono and n["block"] == "color")]

    def _put_setting_new(self, g, r, stage, name, value):
        """A setting of a row just made (no workflow id compiled yet): on its block, its stack, else its leftovers."""
        from ..graphcore.operations import _put
        ix = gm.Index(g)
        chain = ix.effective(r.id)
        base, at, sid = stage.partition("@")
        if base in ("inventory", "calibrate"):
            c = next((c for _s, c in r.nights if at and g.blocks[c].sid == int(sid)), None)
            _put(g.blocks[c].params if c is not None else r.calib, stage, name, value)
            return
        if stage == "render_simple":
            store = {"render_simple": r.output}
            _put(store, stage, name, value)
            r.output = store.get("render_simple", {})
            return
        if stage == "align_on" and r.align:
            _put(r.align.setdefault("params", {}), stage, name, value)
            return
        if stage == "stack":
            own = "selection" if name in _SELECTION else "crop" if name == "zone" else "stacking"
            b = next((b for b in chain if g.blocks[b].kind == own), None)
        else:
            q = gm.clone(g)
            b = next((b for b in chain if stage in owned_stages(q, b)), None)
        if b is not None:
            _put(g.blocks[b].params, stage, name, value)
        else:
            _put(r.extra, stage, name, value)

    def _path_params(self, params):
        """today's add_path: a new workflow's settings, the current one's pre-processing when it has none of its own,
        the camera white balance as the project's first workflow has it."""
        mm = _mm()
        params = copy.deepcopy(params or {})
        if not params and self.path is not None:
            params = {s: dict(v) for s, v in self.path.params.items() if s in mm.TRUNK}
        have = [w.params.get("stack", {}).get("white_balance", False) for w in self.p.workflows]
        if (bool(have[0]) if have else True):
            params.setdefault("stack", {})["white_balance"] = True
        return params

    def add_path(self, name: str, opts: dict, params: dict | None = None, end: str | None = None):
        mm = _mm()
        params = self._path_params(params)
        e = mm.wfl.check_end(mm.KEY_ENGINE_BLOCK.get(end, end)) if end else None
        options = mm.wfl.check_options(mm.to_engine(opts))
        sids = [s.id for s in self.p.sessions]
        mono = bool(sids) and all(s.filter for s in self.p.sessions)
        return self._new_stack(name, sids, self._legacy_nodes(options, e, "frames", mono), params, all_nights=True,
                               end=e, options=options)

    def add_graph_workflow(self, name: str, blocks: list[dict], source: dict | None = None,
                           sessions: list[int] | None = None):
        if source is not None:
            _not_ported("add_graph_workflow(from an image)")
        sids = list(sessions) if sessions is not None else [s.id for s in self.p.sessions]
        params = self._path_params(None)
        with self._edit():
            q = self._new_stack(self.unique_name(name), sids, [n for n in blocks if n.get("block") != "align_on"],
                                params, all_nights=sessions is None)
            cal = any(n.get("block") == "calibration" for n in blocks)
            r = self._row(q.pid)
            if r.calib_node != cal:
                def raw(g):
                    self._row(q.pid, g).calib_node = cal
                self._do(raw=raw)
        return self._view_of_wid(q.pid)

    def ensure_frames_workflow(self):
        if not self.p.sessions or any(q.source is None for q in self.paths):
            return None
        return self.add_graph_workflow(_("Workflow"), [])

    def filter_workflow(self, flt: str, opts: dict | None = None, name: str | None = None):
        sids = self.filters().get(flt)
        if not sids:
            return None
        for q in self.paths:
            if q.source is None and q.sessions is not None and sorted(q.sessions) == sorted(sids) and not q.end:
                return q
        mm = _mm()
        o = dict(opts or {"debruitage": True, "etoiles": "garder", "fond": "simple"})
        options = mm.wfl.check_options(mm.to_engine(o))
        with self._edit():                                # one step, as today's
            if self.p.stack_mode == "together":
                self._do([("set_calibrate_together", {"on": False})])
            return self._new_stack(self.unique_name(name or flt), sids, self._legacy_nodes(options, None, "frames", True),
                                   options=options)

    def add_image_workflow(self, name: str, path: str, opts: dict | None = None, stretched: bool = False):
        mm = _mm()
        o = dict(opts or {"debruitage": False, "etoiles": "garder", "fond": "simple"})
        if stretched and o.get("etoiles") == "retirer":
            o["debruitage"] = True
        options = mm.wfl.check_options(mm.to_engine(o))
        src = "stretched" if stretched else "image"
        nodes = self._legacy_nodes(options, None, src, False)
        source = {"path": Path(path).as_posix() if path else "", "stretched": bool(stretched)}
        name = self.unique_name(name)

        def raw(g):
            head = gm.new_block(g, "image", engine="image", source=source)
            r = gm.Row(gm.new_id(g, "r"), name, [head.id] + [gm.new_block(g, n["block"], method=n.get("method")).id
                                                              for n in nodes])
            r.legacy = {"options": copy.deepcopy(options), "end": None, "frozen": False}
            r.wid, g.next_wid = g.next_wid, g.next_wid + 1
            g.order.append(r.wid)
            g.rows.append(r)
        before = self._wids()
        self._do(raw=raw)
        return self._new_view(before)

    def set_graph(self, p, blocks: list[dict]) -> None:
        """A workflow given a whole graph (a template's): its row's chain replaced by these nodes' blocks, its settings
        kept where the new blocks hold them (the others in its leftovers); a row kept as its options is one of blocks
        from then on."""
        self._replace_chain(p, blocks)

    def set_opts(self, p, opts: dict):
        """The options of a workflow replaced as a whole: a workflow of blocks takes the blocks they give (today's
        ``_sync_graphs``); one kept as its options keeps them."""
        mm = _mm()
        wfl = mm.wfl
        w = self.p.workflow(p.pid)
        o = wfl.check_options(mm.to_engine(opts))
        if w.blocks is None:
            src = wfl.source_of(w)
            self._replace_chain(p, self._legacy_nodes(o, w.end, src, self.is_mono(p)), options=o)
            return
        src = wfl.source_of(w)
        try:
            same = wfl.options_of(w.blocks, src) == o
        except ValueError:
            same = False
        if not same:
            self._replace_chain(p, wfl.sync_blocks(copy.deepcopy(w.blocks), src, o))

    def add_exit(self, src, end: str, name: str):
        """A workflow that copies another up to Stacking, then goes to its own Output (today's ``api.add_exit``: made
        from options, the settings of the stages it runs)."""
        mm = _mm()
        wfl = mm.wfl
        block = mm.KEY_ENGINE_BLOCK.get(end, end)
        if block not in wfl.EXITS:
            raise ValueError(f"no early output at {block!r} (allowed: {list(wfl.EXITS)})")
        w = self.p.workflow(src.pid)
        chain = wfl.chain(w.options, block, reference=bool(self.p.reference_image))
        params = {s: copy.deepcopy(kv) for s, kv in w.params.items() if s in chain}
        sids = [s.id for s in self.p.sessions]
        mono = bool(sids) and all(s.filter for s in self.p.sessions)
        return self._new_stack(self.unique_name(name), sids, self._legacy_nodes(w.options, block, "frames", mono),
                               params, all_nights=True, end=block, options=dict(w.options))

    def _replace_chain(self, p, blocks, options=None):
        """The row of workflow ``p``: its chain made of these nodes (``options``: kept as these options, made before
        the free graph), its workflow's settings stored again where the new blocks hold them."""
        from ..graphcore.compile import NODE_KINDS
        from ..graphcore.operations import new_align
        wfl = _mm().wfl
        w = self.p.workflow(p.pid)
        params = copy.deepcopy(w.params)

        def raw(g):
            r = self._row(p.pid, g)
            ix = gm.Index(g)
            if r.fork or ix.children[r.id] or any(t.block in r.blocks for t in g.taps):
                _not_ported("set_graph / set_opts(a row variants or taps start from)")
            head = [b for b in r.blocks[:1] if g.blocks[b].kind in ("image", "combine")]
            gone = set(r.blocks) - set(head)
            new, align = [], None
            for n in blocks:
                k = n["block"]
                if k in ("calibration", "output", "image"):
                    continue
                if k == "align_on":
                    align = n
                    continue
                if k not in NODE_KINDS:
                    continue
                b = gm.new_block(g, k, method=n.get("method"), off=bool(n.get("off")), fork=n.get("fork"),
                                 engine=n.get("name") if k in wfl.IMAGE_OPS and n.get("name") else None)
                new.append(b.id)
            for b in gone:
                g.blocks.pop(b, None)
            g.wires = [x for x in g.wires if x.src[0] not in gone and x.dst[0] not in gone]
            r.blocks = head + new
            if r.nights:
                cal = next((n for n in blocks if n["block"] == "calibration"), None)
                r.calib_node = cal is not None
                if cal is not None and bool(cal.get("off")) != any(g.blocks[c].off for _s, c in r.nights):
                    for _s, c in r.nights:
                        g.blocks[c].off = bool(cal.get("off"))
            if align is not None:
                ref = self._flow(align["reference"], g) if align.get("reference") is not None else None
                r.align = {**new_align(ref.id if ref is not None else None), "method": align.get("method"),
                           "off": bool(align.get("off"))}
            elif r.align and not any(g.blocks[b].kind in ("stacking", "image") for b in r.blocks):
                r.align = None
            r.legacy = None if options is None else {**(r.legacy or {"end": w.end, "frozen": False}),
                                                     "options": copy.deepcopy(options)}
            r.extra, r.unset = {}, []
            for stage, kv in params.items():
                for k, v in kv.items():
                    self._put_setting_new(g, r, stage, k, v)
        self._do(raw=raw)

    def add_branch(self, p, key: str, block: str):
        _not_ported("add_branch")

    def image_name(self, base: str) -> str:
        if not any(isinstance(q.source, dict) and "path" in q.source for q in self.paths):
            return self.unique_name(base)

        def free() -> str:
            names = {x.name for x in self.paths}
            n = 1
            while f"{base} {n}" in names:
                n += 1
            return f"{base} {n}"
        for q in list(self.paths):
            if q.name == base and isinstance(q.source, dict) and "path" in q.source:
                self._rename(q.pid, free())               # the Image named after its kind takes its number
                self._rebuild_views()
        return free()

    # ------------------------------------------------------------------ the calibration (INTEGRATION 4.4)
    def set_stack_mode(self, mode: str) -> bool:
        from ..core.project2 import STACK_MODES
        if mode not in STACK_MODES:
            raise ValueError(f"not a stack mode: {mode!r}")
        if mode == self.p.stack_mode:
            return False
        self._do([("set_calibrate_together", {"on": mode == "together"})])
        return True

    def _night_calibration(self, p, sid):
        """The Calibration block of a night in a workflow's stack."""
        f = self._flow(p)
        root = self._root(f) if f is not None else None
        c = next((c for s, c in (root.nights if root else []) if self._g.blocks[c].sid == int(sid)), None)
        if c is None:
            raise GraphRefused(_("not in workflow"))
        return c

    def _set_calibration_files(self, paths, sid: int, kind: str, files) -> None:
        steps, seen = [], set()
        for q in paths:
            c = self._night_calibration(q, sid)
            if c in seen:
                continue
            seen.add(c)
            if files is None:
                if kind in (self._g.blocks[c].own or {}):
                    steps.append(("follow_session", {"block": c, "kind": kind}))
            else:
                steps.append(("set_own_calibration", {"block": c, "kind": kind, "paths": list(files),
                                                      "mode": "replace"}))
        if steps:
            self._do(steps)

    def remove_own_calibration(self, p, sid: int) -> None:
        c = self._night_calibration(p, sid)
        if self._g.blocks[c].own:
            self._do([("remove_own_calibration", {"block": c})])

    # ------------------------------------------------------------------ sessions and their files
    def _night_edit(self, sid, files=None, name=None, flt=None, left_out=None, toast=""):
        """A session's lists of files (``files``: {kind: list}), name, filter or frames left out, as one step: on the
        Session blocks drawing it, else on the night no stack draws."""
        def raw(g):
            n = next((x for x in g.nights if x["sid"] == sid), None)
            if n is None:
                raise GraphRefused(_("session not found"))
            blocks = [b for b in g.blocks.values() if b.kind == "session" and b.sid == sid]
            for k, lst in (files or {}).items():
                for b in blocks:
                    b.files[k] = list(lst)
                n.setdefault("files", {})[k] = list(lst)
            if name is not None:
                n["name"] = name
                for b in blocks:
                    b.name = name
            if flt is not None:
                n["filter"] = flt
            if left_out is not None:
                n["left_out"] = copy.deepcopy(left_out)
        return self._do(raw=raw, name=toast or None)

    def rename_session(self, sid: int, name: str) -> bool:
        name = name.strip()
        s = self.p.session(sid)
        if name == s.name or self.name_problem(name, sid=sid):
            return False
        self._night_edit(sid, name=name)
        return True

    def set_session_filter(self, sid: int, flt: str) -> bool:
        s = self.p.session(sid)
        if flt == s.filter:
            return False
        self._night_edit(sid, flt=flt)
        return True

    def add_filter_session(self, flt: str, name: str = "") -> int:
        from .. import api
        sid = max([s.id for s in self.p.sessions] + [n["sid"] for n in self._g.nights], default=0) + 1
        name = api.unique_session_name(self.p, flt or name) if flt or name else self.new_session_name(sid)
        steps = [("new_night", {"name": name, "filter": flt})]
        if len(self.p.sessions) + 1 >= 2 and self.p.stack_mode == "together":
            steps.append(("set_calibrate_together", {"on": False}))
        self._do(steps)
        return self._g.nights[-1]["sid"]

    def add_session(self, name: str = "", lights=(), **calibration) -> int | None:
        from .. import api
        mm = _mm()
        bad = set(calibration) - set(mm.CALIBRATION_KINDS)
        if bad:
            raise ValueError(f"not kinds of calibration frames: {sorted(bad)}")
        sid = max((s.id for s in self.p.sessions), default=0) + 1
        nm = api.unique_session_name(self.p, name) if name else self.new_session_name(sid)
        files = {"lights": [Path(x).as_posix() for x in lights]} | \
            {k: [Path(x).as_posix() for x in v] for k, v in calibration.items()}

        def raw(g):
            from ..graphcore.operations import _night
            g.nights.append({"sid": sid, "name": nm, "filter": "",
                             "files": {k: list(files.get(k, [])) for k in gm.FILE_KINDS}, "left_out": {}})
            for r in g.rows:                            # a stack of every night draws it too
                if r.all_nights and r.nights and not r.fork:
                    pair = _night(g, nm, {k: v for k, v in files.items() if v})
                    g.blocks[pair[0]].sid = g.blocks[pair[1]].sid = sid
                    g.blocks[pair[1]].off = g.blocks[r.nights[-1][1]].off if g.calibrate_together else False
                    r.nights.append(pair)
        self._do(raw=raw)
        self.reload_sources()
        return sid

    def add_session_entries(self, sid: int, kind: str, paths, notes: list | None = None) -> int:
        from ..core.project import frame_kind, list_frames
        from .services.worker import norm_path
        mm = _mm()
        if kind not in ("lights",) + tuple(mm.CALIBRATION_KINDS):
            raise ValueError(f"not a kind of frames: {kind!r}")
        s = self.p.session(sid)
        notes = [] if notes is None else notes
        cur = list(getattr(s, kind))
        have = {norm_path(self.p.abspath(e)) for e in cur}
        folders = {norm_path(self.p.abspath(e)) for e in cur if Path(self.p.abspath(e)).is_dir()}

        def known(f) -> bool:
            return norm_path(f) in have or (bool(frame_kind(Path(f))) and norm_path(str(Path(f).parent)) in folders)
        new: list[str] = []
        for x in paths:
            px = Path(x)
            files = [str(f) for f in list_frames([str(px)])] if px.is_dir() else [str(px)]
            if not files:
                notes.append(("“{name}” holds no frames (camera raw, FITS, JPEG, PNG or TIFF): not added.", px.name))
            elif all(known(f) for f in files):
                notes.append(("“{name}” is already in the session.", px.name))
            for f in files:
                if not known(f):
                    new.append(Path(f).as_posix())
                    have.add(norm_path(f))
        if not new:
            return 0
        flt = ""
        if kind == "lights" and (not s.filter or not cur):
            from ..io.frames import frame_filter
            first = next((f for f in list_frames(new) if frame_kind(f) == "fits"), None)
            flt = frame_filter(first) if first is not None else ""
        moved = bool(flt) and bool(s.filter) and flt != s.filter
        left = copy.deepcopy(s.left_out)
        probe = type("S", (), {"left_out": left})()
        mm.ProjectModel._note_left_out(probe, kind, new, False)
        with self._edit():
            self._night_edit(sid, files={kind: cur + new}, flt=flt or None, left_out=probe.left_out)
            if moved:
                self._refile_session(sid)
        self.reload_sources()
        return len(new)

    def remove_session_files(self, sid: int, kind: str, files: list[str]) -> bool:
        from .services.worker import norm_path
        s = self.p.session(sid)
        gone = {norm_path(self.p.abspath(f)) for f in files}
        keep: list[str] = []
        for e in getattr(s, kind):
            pth = self.p.abspath(e)
            if norm_path(pth) in gone:
                continue
            if not Path(e).suffix:
                inside = [x for x in self.session_files(sid, kind) if norm_path(str(Path(x).parent)) == norm_path(pth)]
                if any(norm_path(x) in gone for x in inside):
                    keep += [x for x in inside if norm_path(x) not in gone]
                    continue
            keep.append(e)
        once: dict[str, str] = {}
        for e in keep:
            once.setdefault(norm_path(self.p.abspath(e)), e)
        keep = list(once.values())
        if keep == list(getattr(s, kind)):
            return False
        self._night_edit(sid, files={kind: keep})
        self.reload_sources()
        return True

    def remove_session_entry(self, sid: int, kind: str, entry: str) -> bool:
        from .services.worker import norm_path
        s = self.p.session(sid)
        lst = list(getattr(s, kind))
        keep = [e for e in lst if norm_path(self.p.abspath(e)) != norm_path(self.p.abspath(entry))]
        if len(keep) == len(lst):
            return False
        self._night_edit(sid, files={kind: keep})
        self.reload_sources()
        return True

    def use_from_session(self, src: int, dst: int, kind: str) -> bool:
        mm = _mm()
        if kind not in mm.CALIBRATION_KINDS:
            raise ValueError(f"not a kind of calibration frames: {kind!r}")
        frm, to = self.p.session(src), self.p.session(dst)
        want = [self.p.abspath(e) for e in getattr(frm, kind)]
        if [self.p.abspath(e) for e in getattr(to, kind)] == want:
            return False
        self._night_edit(dst, files={kind: [Path(x).as_posix() for x in want]})
        self.reload_sources()
        return True

    def remove_session(self, sid: int) -> bool:
        """A session out of the project (its files are never touched): its night out of every stack drawing it (a
        stack of that night alone goes with it), its frames excluded by hand, its own settings."""
        if self.session_removal_problem(sid):
            return False
        steps = [("remove_night", {"night": b.id}) for b in self._g.blocks.values()
                 if b.kind == "session" and b.sid == sid]

        def raw(g):
            g.nights = [n for n in g.nights if n["sid"] != sid]
            g.excluded = [x for x in g.excluded if x[0] != sid]
            for f in list(g.rows) + list(g.taps):
                if f.extra:
                    f.extra = {k: v for k, v in f.extra.items() if not k.endswith(f"@{sid}")}
                if f.sessions is not None:
                    f.sessions = [x for x in f.sessions if x != sid]
            for r in g.rows:
                if r.calib:
                    r.calib = {k: v for k, v in r.calib.items() if not k.endswith(f"@{sid}")}
        self._do(steps, raw=raw)
        self.current = max(0, min(self.current, len(self.paths) - 1))
        self.reload_sources()
        return True

    def join_session(self, p, sid: int):
        """A new session joins the chosen flow of frames ``p`` (today's rules: a filter's flow, else a flow of its own;
        see ProjectModel.join_session), on the graph: its night drawn in the stacks of that flow."""
        frames = [w for w in self.p.workflows if w.source is None]
        ses = self.p.session(sid)

        def ids(w):
            return None if w.sessions is None else tuple(sorted(w.sessions))
        flows = {ids(w) for w in frames if ids(w) is None or sid not in ids(w)}
        w = next((x for x in frames if p is not None and x.id == p.pid), None)
        if w is None and p is None and len(flows) == 1:
            w = next(x for x in frames if ids(x) in flows)
        if w is not None and sid in (w.sessions or ()):
            return self._view_of(w.id)
        flt = {x.filter for x in self.p.sessions_of(w.id) if x.id != sid} if w is not None else set()
        like = None
        if w is not None and ses.filter and flt and flt != {ses.filter}:
            like = w
            w = next((x for x in frames if x.sessions and sid not in x.sessions
                      and {s.filter for s in self.p.sessions_of(x.id)} == {ses.filter}), None)
            if w is not None:
                flt = {ses.filter}
        with self._edit():
            if w is None:                                 # a flow of its own: the stacks of every night keep theirs
                mine = [x.id for x in self.p.sessions if x.id != sid]
                every = {x.id for x in frames if x.sessions is None}

                def raw(g):
                    for f in list(g.rows) + list(g.taps):
                        if f.wid in every and f.all_nights:
                            f.all_nights = False
                            f.sessions = list(mine)
                    for r in g.rows:
                        gone = [pr for pr in r.nights if g.blocks[pr[0]].sid == sid]
                        if gone and len(gone) < len(r.nights) and not r.fork:
                            for pr in gone:
                                r.nights.remove(pr)
                                for b in pr:
                                    g.blocks.pop(b, None)
                self._do(raw=raw)
                blocks = copy.deepcopy(self.p.workflow(like.id).blocks) if like is not None and \
                    self.p.workflow(like.id).blocks else []
                return self.add_graph_workflow(ses.filter or _("Flow"), blocks, sessions=[sid])
            if len(flt) == 1 and "" not in flt and not ses.filter:
                new_flt = next(iter(flt))
                mates = [x.name for x in self.p.sessions_of(w.id) if x.id != sid]
                default = ses.name.startswith(_("Session {n}").format(n="")) and ses.name.split()[-1].isdigit()
                name = None
                if default and mates and all(new_flt in x for x in mates):
                    k = len(mates) + 1
                    while f"{new_flt} {k}" in {x.name for x in self.p.sessions}:
                        k += 1
                    name = f"{new_flt} {k}"
                self._night_edit(sid, flt=new_flt, name=name)
            if w.sessions is not None:
                flow = ids(w)
                wids = {x.id for x in frames if ids(x) == flow}
                self._share_night(sid, wids)
        return self._view_of(w.id)

    def _share_night(self, sid, wids):
        """The night ``sid`` drawn in the stacks of these workflows (their session lists take it too)."""
        def raw(g):
            from ..graphcore.operations import _night
            ix = gm.Index(g)
            src = next((b for b in g.blocks.values() if b.kind == "session" and b.sid == sid), None)
            entry = next(n for n in g.nights if n["sid"] == sid)
            done = set()
            for wid in wids:
                f = self._flow(wid, g)
                if f is None:
                    continue
                if f.sessions is not None and sid not in f.sessions:
                    f.sessions = list(f.sessions) + [sid]
                rid = f.id if isinstance(f, gm.Row) else ix.row_of.get(f.block)
                root = ix.row[ix.root(rid)] if rid in ix.row else None
                if root is None or root.id in done or any(g.blocks[s].sid == sid for s, _c in root.nights):
                    continue
                done.add(root.id)
                files = {k: list(v) for k, v in src.files.items()} if src is not None else \
                    {k: list(v) for k, v in (entry.get("files") or {}).items() if v}
                pair = _night(g, src.name if src is not None else entry.get("name", ""), files)
                g.blocks[pair[0]].sid = g.blocks[pair[1]].sid = sid
                if root.nights and g.calibrate_together:
                    g.blocks[pair[1]].off = g.blocks[root.nights[-1][1]].off
                root.nights.append(pair)
        self._do(raw=raw)

    def add_parallel_flow(self):
        """A second flow of frames: its own empty Session; the stacks that took every session keep the ones they had."""
        with self._edit():
            mine = [s.id for s in self.p.sessions]
            every = {w.id for w in self.p.workflows if w.source is None and w.sessions is None}
            if every and not mine:
                mine = [self.add_filter_session("")]
                self._share_night(mine[0], every)

            def raw(g):
                for f in list(g.rows) + list(g.taps):
                    if f.wid in every:
                        f.all_nights = False
                        f.sessions = list(mine)
            if every:
                self._do(raw=raw)
            sid = self.add_filter_session("")
            q = self.add_graph_workflow(_("Flow"), [], sessions=[sid])
        return q

    def _refile_session(self, sid: int) -> None:
        """A session whose filter changed leaves the flow it was in and joins its filter's (``join_session``)."""
        frames = [w for w in self.p.workflows if w.source is None]
        w = next((x for x in frames if x.sessions is not None and sid in x.sessions), None) \
            or next((x for x in frames if x.sessions is None), None)
        if w is None:
            return
        listed = {x.id for x in frames if x.sessions is not None and sid in x.sessions}

        def raw(g):
            ix = gm.Index(g)
            for f in list(g.rows) + list(g.taps):
                if f.wid not in listed:
                    continue
                if f.sessions is not None:
                    f.sessions = [k for k in f.sessions if k != sid]
                rid = f.id if isinstance(f, gm.Row) else ix.row_of.get(f.block)
                root = ix.row[ix.root(rid)] if rid in ix.row else None
                gone = [pr for pr in (root.nights if root else []) if g.blocks[pr[0]].sid == sid]
                if root is not None and gone and len(gone) < len(root.nights):
                    for pr in gone:
                        root.nights.remove(pr)
                        for b in pr:
                            g.blocks.pop(b, None)
        with self._edit():
            if listed:
                self._do(raw=raw)
            self.join_session(self._view_of(w.id), sid)


#: the writers of today's model the mixin re-expresses on the graph, and those not ported yet (NotImplementedError)
PORTED = (
    "_snapshot", "_restore", "_sync_graphs", "_prune_breakpoints", "_follow_branch_names", "reload_sources",
    "insert_block", "remove_node", "set_cut", "set_node_method", "set_breakpoint", "set_param", "set_trunk_param",
    "apply_change", "add_variant", "keep_variant", "drop_variant", "rename", "_rename", "duplicate", "remove_path",
    "delete_workflow", "send_to_post", "set_image_file", "add_fusion", "remove_fusion", "set_fusion_input",
    "set_fusion_channel", "set_fusion_image", "set_fusion_weight", "clear_fusion_input", "set_align_reference",
    "set_align_on", "align_without_astrometry", "add_sho_rgb", "add_dual_x2", "add_filters_combination", "add_path",
    "add_graph_workflow", "ensure_frames_workflow", "filter_workflow", "add_image_workflow", "image_name", "set_graph",
    "set_opts", "add_exit", "remove_session", "join_session", "add_parallel_flow", "_refile_session",
    # today's code, unchanged, on ported facades: new_session, delete_workflow
    "set_stack_mode", "_set_calibration_files", "remove_own_calibration", "rename_session", "set_session_filter",
    "add_filter_session", "add_session", "add_session_entries", "remove_session_files", "remove_session_entry",
    "use_from_session",
)
NOT_PORTED = ("add_branch",)
#: kept as today's: they read the compiled definition, or edit Post / the parts of the definition the graph absorbs
READERS_KEPT = ("graph", "gives_post", "feeds_fusion", "is_sent_branch", "synth_wanted", "removal_problem",
                "palette_reason", "name_problem", "throws_away", "cost_with", "cost_of_change", "scope_paths",
                "set_layer_side", "calibration_files", "add_calibration_files", "remove_calibration_files",
                "follow_session", "set_excluded", "set_calibration_excluded", "leave_out_calibration")
