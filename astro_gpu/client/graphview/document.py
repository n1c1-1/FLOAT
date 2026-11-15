"""What the graph page edits: a small "document" interface, so that the page runs on the project model of the client
(bound later) as well as alone on a graph held in memory (tests, the dev launcher).

A document has:

    graph                   the stored graph (graphcore.model.Project); a new object after every accepted change
    view                    the view state (graphcore.operations.new_view() and what the page keeps there: the file
                            read options of each dropped folder, the files left out by hand)
    apply(steps) -> Result  steps = [(operation name, kwargs | f(results so far) -> kwargs)], ONE undo step
    undo() / redo() -> Result,  can_undo / can_redo
    state(block) -> str     the run state of a block: a key of graphnode.STATE_WORD ("done", "pending", ...)
    review(block) -> None | "pending" | "paused" | "reviewed"    its review point
    cost() -> float         seconds the next run is expected to take
    revision                a number that changes with every accepted change, undo and redo
    tr                      the translation the core's messages are written with

``Doc`` adds the core's queries bound to the current graph (titles, summaries, greyed reasons, effective settings,
file lists...), with the client's translation: the page reads everything through it. A gesture that raises never
reaches Qt: ``Doc`` takes the document back to the graph and view before it, logs the error and answers a refusal
whose reason is ``FAILED``.
"""
from __future__ import annotations

import copy
import json
import logging
from pathlib import Path

from ...graphcore import api
from ...graphcore.model import blank_project
from ...graphcore.msg import M
from ...graphcore.operations import new_view
from ...graphcore.shapes import Result
from ..i18n import _
from ..names import shown

#: seconds a block not up to date is counted for by the document held in memory
BLOCK_SECONDS = 0.4
#: the toast of a gesture that raised (English; shown through the translation)
FAILED = "The gesture failed: nothing changed"
#: the reason of a gesture the user took back (a block's draft kept by Stay): nothing to say
STAYED = ""
log = logging.getLogger(__name__)


class MemoryDocument:
    """A graph held in memory with its own undo: the page alone (tests, the dev launcher). Nothing is ever written
    to disk. ``states`` gives run states by block id (default "pending"), ``reviewed`` the blocks checked."""

    def __init__(self, graph=None, view=None, tr=None, states=None):
        self.graph = graph if graph is not None else blank_project()
        self.view = view if view is not None else new_view()
        self.tr = tr or _
        self.states = dict(states or {})
        self.reviewed: set[str] = set()
        self.paused: set[str] = set()
        self.revision = 0
        self._undo: list[tuple[object, object]] = []
        self._redo: list[tuple[object, object]] = []

    # ------------------------------------------------------------------ editing
    def apply(self, steps) -> Result:
        before = self.graph
        graph, res = api.apply_many(before, list(steps), self.view, self.tr)
        if res.ok and graph is not before:
            self._undo.append((res.toast or steps[0][0], before))
            self._redo.clear()
            self.graph = graph
            self.revision += 1
        return res

    def undo(self) -> Result:
        if not self._undo:
            return Result(False, reason="Nothing to undo")
        label, g = self._undo.pop()
        self._redo.append((label, self.graph))
        self.graph = g
        self.revision += 1
        return Result(True, toast=M("Undone: {label}", label=label))

    def redo(self) -> Result:
        if not self._redo:
            return Result(False, reason="Nothing to redo")
        label, g = self._redo.pop()
        self._undo.append((label, self.graph))
        self.graph = g
        self.revision += 1
        return Result(True, toast=M("Redone: {label}", label=label))

    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)

    # ------------------------------------------------------------------ the run (none here)
    def state(self, block) -> str:
        return self.states.get(block, "pending")

    def review(self, block):
        b = self.graph.blocks.get(block)
        if block in self.paused:
            return "paused"
        if block in self.reviewed:
            return "reviewed"
        return "pending" if b is not None and b.pause else None

    def mark_reviewed(self, block):
        self.paused.discard(block)
        self.reviewed.add(block)

    def cost(self) -> float:
        n = sum(1 for k, b in self.graph.blocks.items() if not b.off and self.state(k) != "done")
        return round(BLOCK_SECONDS * n, 1)


def model_badges(m, b, kind, state) -> dict:
    """The badges of block ``b`` from a project model ``m`` (blocks.py Graph's rules, on BlockMap's node)."""
    from ..graphnode import block_fallback
    n = m.block_map.node(b)
    if n is None:
        return {}
    out = {"fallback": [], "broken": "", "missing": "", "warned": False}
    paths = list(n.paths)

    def first(f, *args):
        return next((w for q in paths if (w := f(q, *args))), "") if f is not None else ""
    if kind == "session" and n.sid is not None:
        why = getattr(m, "session_problem", None)
        out["broken"] = why(n.sid) if why is not None else ""
        fw = getattr(m, "file_warnings", None)
        out["warned"] = bool(fw(n.sid)) if fw is not None else False
        nt = getattr(m, "nights_text", None)            # "8 nights · 28/06 to 21/07" from the lights' own dates
        out["nights"] = nt(n.sid) if nt is not None else ""
    elif kind == "image":
        out["broken"] = first(getattr(m, "image_problem", None))
    elif kind == "combine":
        out["broken"] = first(getattr(m, "fusion_problem", None))
    if kind not in ("session", "post", "combine_layers") and not n.cut:
        out["missing"] = first(getattr(m, "sky_missing", None), n.key)
    if state == "done" and kind not in ("session", "post", "combine_layers", "calibration") and not n.cut:
        for q in paths:
            out["fallback"] += [w for w in block_fallback(m, q, n.key) if w not in out["fallback"]]
    return out


def touched_blocks(before, g) -> set:
    """The blocks of graph ``g`` whose run state a change from graph ``before`` can change: in each row's chain (its
    mother's part included), from the first block that changed, moved or went on; a row whose nights changed, wholly;
    then what reads one of those through a wire (combinations, taps, Align on, Combine layers, Post), from it on.
    Every block when ``before`` is None."""
    from ...graphcore.model import Index
    if before is None:
        return set(g.blocks)
    ia, ib = Index(g), Index(before)
    seeds = {b for b, blk in g.blocks.items() if before.blocks.get(b) != blk}
    old_w = {(tuple(w.src), tuple(w.dst)) for w in before.wires}
    new_w = {(tuple(w.src), tuple(w.dst)) for w in g.wires}
    seeds |= {d[0] for _s, d in old_w ^ new_w if d[0] in g.blocks}
    out: set = set()
    chains = {}
    for r in g.rows:
        eff = ia.effective(r.id)
        old = ib.effective(r.id) if r.id in ib.row else []
        root = ia.row[ia.root(r.id)]
        nights = [x for s, c in root.nights for x in (s, c) if x is not None]
        was = ib.row.get(ib.root(r.id)) if r.id in ib.row else None
        moved = was is None or [list(n) for n in was.nights] != [list(n) for n in root.nights]
        chains[r.id] = (eff, old, nights, moved)
    for _round in range(len(g.rows) + 2):
        n = len(out)
        hot = seeds | out
        for eff, old, nights, moved in chains.values():
            if moved or any(x in hot for x in nights):
                out.update(nights)
                out.update(eff)
                continue
            i = next((k for k, b in enumerate(eff) if b in hot or k >= len(old) or old[k] != b), len(eff))
            out.update(eff[i:])
        out |= {d[0] for s, d in new_w if s[0] in out and d[0] in g.blocks}
        if len(out) == n and n:
            break
    return (out | seeds) & set(g.blocks)


def open_project(path) -> MemoryDocument:
    """A FLOAT project file (format 2) opened read-only: its definition converted to a graph, in memory. The file and
    its folder are never written."""
    from ...core import project2
    from ...graphcore.convert import convert
    path = Path(path)
    if path.is_dir():
        path = path / project2.FILE
    d = json.loads(path.read_text(encoding="utf-8"))
    p = project2.Project2.from_dict(d, path.parent)
    return MemoryDocument(convert(p))


class Doc:
    """The core's queries on a document's current graph, written with its translation. Everything the page shows
    comes from here; every change goes through ``apply``."""

    def __init__(self, document):
        self.document = document

    # --- the document
    @property
    def graph(self):
        return self.document.graph

    project = graph

    @property
    def view(self):
        return self.document.view

    @property
    def tr(self):
        return getattr(self.document, "tr", None) or _

    @property
    def revision(self):
        return self.document.revision

    def apply(self, name, /, **kwargs) -> Result:
        return self.apply_many([(name, kwargs)])

    def apply_many(self, steps) -> Result:
        """Steps as one undo step. Frames are filed by where they were dropped or added, never by their headers: a
        session's filter is the name the user or the template gives."""
        steps = list(steps)
        go = self.guard(steps) if self.guard is not None else True
        if go is False:                                   # a block's draft kept: its question answered Stay
            return Result(False, reason=STAYED)
        try:
            return self.guarded(steps[0][0] if steps else "apply", lambda: self.document.apply(steps))
        finally:
            if callable(go):                              # what the guard does once the gesture is made
                go()

    #: ``guard(steps)`` before a gesture: False stops it, True lets it go, a function lets it go and is called after it
    #: (the page's: a block holding a draft is asked before a gesture changes it)
    guard = None

    def try_many(self, steps) -> Result:
        """Whether these steps would be accepted, nothing changed: -> Result."""
        def dry():
            _g, res = api.apply_many(self.graph, list(steps), dict(self.view), self.tr)
            return res
        return self.guarded("try", dry, take_back=False)

    def undo(self) -> Result:
        return self.guarded("undo", self.document.undo, take_back=False)

    def redo(self) -> Result:
        return self.guarded("redo", self.document.redo, take_back=False)

    # --- a gesture that raises
    def guarded(self, what, fn, take_back=True) -> Result:
        """``fn()`` (-> Result); when it raises: the error logged, the document taken back to the graph and view before
        (``take_back``), a refusal whose reason is FAILED."""
        d = self.document
        graph, rev = d.graph, getattr(d, "revision", None)
        try:
            view = copy.deepcopy(d.view)
        except Exception:  # noqa: BLE001 - a view that cannot be copied is kept as a shallow copy
            view = dict(d.view)
        try:
            return fn()
        except Exception:  # noqa: BLE001 - the page never lets the core's error reach Qt
            log.exception("graph page: the gesture %r raised", what)
            if take_back:
                self.take_back(graph, rev, view)
            return Result(False, reason=FAILED)

    def take_back(self, graph, rev, view):
        """The document as it was before a gesture that raised: its own undo when it made a step, else its graph set
        back (a document held in memory); the view dict given back its keys."""
        d = self.document
        try:
            if d.graph is not graph or getattr(d, "revision", None) != rev:
                if getattr(d, "can_undo", False):
                    d.undo()
                if d.graph is not graph and isinstance(d, MemoryDocument):
                    d.graph = graph
        except Exception:  # noqa: BLE001 - taken back as far as the document allows
            log.exception("graph page: the graph could not be taken back")
        try:
            v = d.view
            v.clear()
            v.update(view)
        except Exception:  # noqa: BLE001
            log.exception("graph page: the view could not be taken back")

    @property
    def can_undo(self) -> bool:
        return self.document.can_undo

    @property
    def can_redo(self) -> bool:
        return self.document.can_redo

    def state(self, b) -> str:
        return self.document.state(b)

    def why(self, b) -> str:
        """Why the block is Outdated, in a few words (the model's ``BlockMap.why``); "" otherwise or without a model."""
        m = self.model
        g = getattr(m, "block_map", None) if m is not None else None
        if g is None:
            return ""
        try:
            return g.why(g.node(b))
        except Exception:  # noqa: BLE001 - a reason the model cannot give now: the state word alone
            log.debug("graph page: no reason for %s", b, exc_info=True)
            return ""

    def run_word(self, b) -> str:
        """The block's state word as the run says it ("Running · 40 %", "Stopped"); "" without a model."""
        g = getattr(self.model, "block_map", None) if self.model is not None else None
        if g is None:
            return ""
        try:
            return g.word(g.node(b))
        except Exception:  # noqa: BLE001 - the plain state word then
            log.debug("graph page: no run word for %s", b, exc_info=True)
            return ""

    def run_detail(self, b) -> str:
        """What the next run does with a stopped block, in the run window's words; "" otherwise or without a model."""
        g = getattr(self.model, "block_map", None) if self.model is not None else None
        if g is None:
            return ""
        try:
            n = g.node(b)
            return g.stopped(n) if g.state(n) == "interrupted" else ""
        except Exception:  # noqa: BLE001
            log.debug("graph page: no stop words for %s", b, exc_info=True)
            return ""

    def review(self, b):
        f = getattr(self.document, "review", None)
        return f(b) if f is not None else None

    def cost(self) -> float:
        return self.document.cost()

    # --- what the run and the project model say of a block (nothing for a graph held in memory)
    @property
    def model(self):
        """The client's project model behind the document (``GraphDocument.model``), or None."""
        return getattr(self.document, "model", None)

    def badges(self, b) -> dict:
        """What today's graph page draws on a block beside its state (graph.py BlockItem, blocks.py Graph), from the
        project model through ``BlockMap``: {"fallback": [the engine's reasons], "broken": why it cannot run at all,
        "missing": the sky data it needs and this computer has not, "warned": its files have a warning}; {} without a
        model. A document may give its own (``document.badges``)."""
        own = getattr(self.document, "badges", None)
        if own is not None:
            return own(b) or {}
        m = self.model
        blk = self.graph.blocks.get(b)
        if m is None or blk is None or not hasattr(m, "block_map"):
            return {}
        try:
            return model_badges(m, b, blk.kind, self.state(b))
        except Exception:  # noqa: BLE001 - a badge the model cannot answer yet is not drawn
            log.debug("graph page: no badges for %s", b, exc_info=True)
            return {}

    def states_pending(self) -> bool:
        """The project model's worker reads the states of its last edit (a graph held in memory: never)."""
        f = getattr(self.model, "states_pending", None)
        return bool(f()) if f is not None else False

    def reading(self):
        """(text, busy) while the project model reads the project ("Reading the project…", True) or could not
        ("Project unreadable: why", False); None when it is ready, or without a model."""
        m = self.model
        if m is None:
            return None
        err = getattr(m, "error", None)
        if err:
            return _("Project unreadable: {why}").format(why=err), False
        if not getattr(m, "ready", True):
            return _("Reading the project…"), True
        return None

    # --- texts of the core
    def say(self, message) -> str:
        return api.say(message, self.tr)

    def title(self, b) -> str:
        return shown(api.title(self.graph, b, self.tr))          # a row's default name, in the interface's language

    def summary(self, b, model=True) -> str:
        """The line under a block's name; ``model`` False: the graph's own (never what the project model reads)."""
        m = self.model if model else None
        bm = getattr(m, "block_map", None) if m is not None else None
        blk = self.graph.blocks.get(b)
        if bm is not None and blk is not None and blk.kind == "post" and hasattr(bm, "post_line"):
            return bm.post_line()                    # Post's layers as Post lists them (an image's own layer too)
        if bm is not None and blk is not None and blk.kind in ("session", "calibration", "image"):
            n = bm.node(b)                           # a folder not found, an image's file: the model's
            if n is not None:
                return bm.summary(n, self.tr) or api.summary(self.graph, b, self.tr)
        line = api.summary(self.graph, b, self.tr)
        if self.model is not None and blk is not None:   # a variant's first block: what differs from its original
            from ..graphnode import variant_diff
            diff = variant_diff(self.graph, b)
            line = f"{line} · {diff}" if diff and diff not in line else line
        return line

    def greyed(self, b, model=True) -> str:
        """Why a block cannot run; ``model`` False: the graph's own reason only."""
        why = api.greyed(self.graph, b, self.tr)
        if why or self.model is None or not model:
            return why
        blk = self.graph.blocks.get(b)
        return self._mono_colour(b) if blk is not None and blk.kind == "color" else ""

    def _mono_colour(self, b) -> str:
        """A Color block of a monochrome workflow (its sessions' filter L, Ha...): no colours to calibrate (the model's
        reason, today's page greyed it)."""
        m = self.model
        bm = getattr(m, "block_map", None)
        n = bm.node(b) if bm is not None else None
        if n is None or not n.paths or not hasattr(m, "greyed_reason"):
            return ""
        return next((w for w in (m.greyed_reason(q, "couleur") for q in n.paths) if w), "")

    def wire_tip(self, w) -> str:
        """The core's words on a wire."""
        return api.wire_tip(self.graph, w, self.tr)

    def port_source(self, b, port) -> str:
        """What a Combination input reads: the name of the workflow wired into it (``files.port_source``); ""."""
        from .files import port_source
        return port_source(self.graph, b, port)

    def row_badge(self, r) -> str:
        return api.row_badge(self.graph, r, self.tr)

    def wire_check(self, src, src_port, dst, dst_port) -> str:
        """The core's verdict, then the project model's (a Stars input of a Combine layers that Post cannot take)."""
        why = api.wire_check(self.graph, self.view, src, src_port, dst, dst_port, self.tr)
        m = self.model
        if not why and m is not None and hasattr(m, "wire_refusal"):
            try:
                why = m.wire_refusal(src, src_port, dst, dst_port)
            except Exception:  # noqa: BLE001 - the model cannot judge now: the gesture judges it
                log.debug("graph page: the model's wire check failed", exc_info=True)
        return why

    def project_state(self):
        return api.project_state(self.graph, self.tr)

    def project_block(self):
        """(title, line under it, state, tooltip) of the Project block. On a project model, issue #89's rule as today's
        page drew it: red with no target, orange with shooting details missing, green complete; its line the target
        (typed, or what Astrometry found), its tooltip what is missing. Without a model, the core's readiness."""
        m = self.model
        if m is not None and hasattr(m, "project_state") and hasattr(m, "target_state"):
            try:
                st, ps = m.target_state(), m.project_state()
                tip = [_("The project: target, comments, capture data"), ps.line] + ([st.note] if st.name and st.note
                                                                                    else []) + \
                    [_("Double-click: its form in Inspect")]
                name = getattr(getattr(m, "p", None), "name", "") or self.graph.name
                return name or _("Project"), st.name or _("target to choose"), ps.level, "\n".join(t for t in tip if t)
            except Exception:  # noqa: BLE001 - a project the model cannot judge now: the core's readiness
                log.debug("graph page: the Project block's state from the model failed", exc_info=True)
        state, tip = self.project_state()
        g = self.graph
        return g.name or _("Project"), g.target or _("target to choose"), state, tip

    def palette_state(self, kind):
        ok, why = api.palette_state(self.graph, self.view, kind, self.tr)
        if ok and kind == "color" and self.model is not None and hasattr(self.model, "is_mono"):
            sel = self.view.get("selected")              # the workflow the palette places in: a mono one has no colours
            bm = getattr(self.model, "block_map", None)
            n = bm.node(sel) if bm is not None and sel in self.graph.blocks else None
            if n is not None and n.paths and all(self.model.is_mono(q) and q.source is None for q in n.paths):
                from ..i18n import _
                from ...core import workflow as wfl
                return False, _(wfl.MONO_NO_COLOUR)
        return ok, why

    def warnings(self, scope=None):
        return api.warnings(self.graph, scope, self.tr)

    # --- reading the graph
    def layout(self):
        return api.layout(self.graph, self.view, self.tr)

    def places(self, kind=None, block=None):
        return api.places(self.graph, self.view, kind=kind, block=block)

    def stacks(self):
        return api.stacks(self.graph)

    def effective(self, b):
        return api.effective(self.graph, b)

    def own_keys(self, b):
        return api.own_keys(self.graph, b)

    def followers(self, kind):
        return api.followers(self.graph, kind)

    def files_of(self, session, kind):
        return api.files_of(self.graph, session, kind)

    def differs(self, b) -> bool:
        return bool(api.differs(self.graph, b))
