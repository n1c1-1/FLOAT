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
file lists...), with the client's translation: the page reads everything through it.
"""
from __future__ import annotations

import json
from pathlib import Path

from ...graphcore import api
from ...graphcore.model import blank_project
from ...graphcore.msg import M
from ...graphcore.operations import new_view
from ...graphcore.shapes import Result
from ..i18n import _

#: seconds a block not up to date is counted for by the document held in memory
BLOCK_SECONDS = 0.4


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
        return self.document.apply([(name, kwargs)])

    def apply_many(self, steps) -> Result:
        return self.document.apply(steps)

    def try_many(self, steps) -> Result:
        """Whether these steps would be accepted, nothing changed: -> Result."""
        _g, res = api.apply_many(self.graph, list(steps), dict(self.view), self.tr)
        return res

    def undo(self) -> Result:
        return self.document.undo()

    def redo(self) -> Result:
        return self.document.redo()

    @property
    def can_undo(self) -> bool:
        return self.document.can_undo

    @property
    def can_redo(self) -> bool:
        return self.document.can_redo

    def state(self, b) -> str:
        return self.document.state(b)

    def review(self, b):
        f = getattr(self.document, "review", None)
        return f(b) if f is not None else None

    def cost(self) -> float:
        return self.document.cost()

    # --- texts of the core
    def say(self, message) -> str:
        return api.say(message, self.tr)

    def title(self, b) -> str:
        return api.title(self.graph, b, self.tr)

    def summary(self, b) -> str:
        return api.summary(self.graph, b, self.tr)

    def greyed(self, b) -> str:
        return api.greyed(self.graph, b, self.tr)

    def wire_tip(self, w) -> str:
        return api.wire_tip(self.graph, w, self.tr)

    def row_badge(self, r) -> str:
        return api.row_badge(self.graph, r, self.tr)

    def wire_check(self, src, src_port, dst, dst_port) -> str:
        return api.wire_check(self.graph, self.view, src, src_port, dst, dst_port, self.tr)

    def project_state(self):
        return api.project_state(self.graph, self.tr)

    def palette_state(self, kind):
        return api.palette_state(self.graph, self.view, kind, self.tr)

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
