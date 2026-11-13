"""A small undo helper around the pure operations, for the tests and the fuzz only. The client keeps its own undo
(its snapshots); the core has none. A History holds the current graph, the view state and two stacks of graphs."""
import json

from . import api
from .model import blank_project, model_json, to_dict, from_dict, check
from .msg import M
from .operations import apply_many as _apply_many, new_view, json_of
from .shapes import Result


class History:
    """graph + view + undo/redo stacks. apply/apply_many/undo/redo behave as the Doc of the first mockup did."""

    def __init__(self, graph=None, view=None, tr=None):
        self.project = graph or blank_project()
        self.view = view if view is not None else new_view()
        self.tr = tr
        self._undo, self._redo = [], []

    # --- creation and files (the model as plain JSON, nothing else)
    @classmethod
    def loads(cls, text):
        d = json.loads(text)
        h = cls(from_dict(d["model"]))
        h.view = d["view"]
        check(h.project)
        return h

    def dumps(self):
        return json.dumps({"model": to_dict(self.project), "view": self.view}, ensure_ascii=False, indent=1)

    # --- editing
    def apply(self, name, /, **kwargs):
        return self.apply_many([(name, kwargs)])

    def apply_many(self, steps):
        before = self.project
        graph, res = _apply_many(before, steps, self.view, self.tr)
        if res.ok and graph is not before:
            self._undo.append((res.toast or steps[0][0], before))
            self._redo.clear()
            self.project = graph
        return res

    def drop(self, place):
        if not place.lit:
            return Result(False, reason=place.reason)
        return self.apply(place.op[0], **place.op[1])

    def undo(self):
        if not self._undo:
            return Result(False, reason="nothing to undo")
        label, g = self._undo.pop()
        self._redo.append((label, self.project))
        self.project = g
        return Result(True, toast=M("Undone: {label}", label=label))

    def redo(self):
        if not self._redo:
            return Result(False, reason="nothing to redo")
        label, g = self._redo.pop()
        self._undo.append((label, self.project))
        self.project = g
        return Result(True, toast=M("Redone: {label}", label=label))

    @property
    def can_undo(self):
        return bool(self._undo)

    @property
    def can_redo(self):
        return bool(self._redo)

    @property
    def json(self):
        return json_of(self.project)

    # --- queries (the api functions bound to the current graph)
    def title(self, b):
        return api.title(self.project, b, self.tr)

    def summary(self, b):
        return api.summary(self.project, b, self.tr)

    def greyed(self, b):
        return api.greyed(self.project, b, self.tr)

    def wire_tip(self, w):
        return api.wire_tip(self.project, w, self.tr)

    def row_badge(self, r):
        return api.row_badge(self.project, r, self.tr)

    def wire_check(self, src, src_port, dst, dst_port):
        return api.wire_check(self.project, self.view, src, src_port, dst, dst_port, self.tr)

    def project_state(self):
        return api.project_state(self.project, self.tr)

    def palette_state(self, kind):
        return api.palette_state(self.project, self.view, kind, self.tr)

    def warnings(self, scope=None):
        return api.warnings(self.project, scope, self.tr)

    def stacks(self):
        return api.stacks(self.project)

    def effective(self, b):
        return api.effective(self.project, b)

    def own_keys(self, b):
        return api.own_keys(self.project, b)

    def followers(self, kind):
        return api.followers(self.project, kind)

    def files_of(self, session, kind):
        return api.files_of(self.project, session, kind)

    def differs(self, b):
        return api.differs(self.project, b)

    def layout(self):
        return api.layout(self.project, self.view, self.tr)

    def places(self, kind=None, block=None):
        return api.places(self.project, self.view, kind=kind, block=block)
