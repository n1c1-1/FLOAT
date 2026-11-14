"""The only place where the page's widgets touch the document: it keeps the current document, caches the Drawing
between two changes (a repaint never lays out again) and turns the Result of an operation into signals (a toast, a
selection, a block to show).

Nothing is computed here: the drawing, the drop places, the texts and the legality all come from the core through
``Doc``. If the core changes a signature, only ``document.Doc`` and this file follow.
"""
from __future__ import annotations

import logging

from PySide6.QtCore import QObject, Signal

from ...graphcore.shapes import Result
from ..i18n import _
from .document import FAILED, Doc, MemoryDocument

log = logging.getLogger(__name__)

#: the selection id of the Project block
PROJECT = "project"


class Bridge(QObject):
    changed = Signal()            # the graph (or its view state) changed: everything is drawn again
    states_changed = Signal()     # only the run states changed
    toast = Signal(str)           # a sentence for the toast (and the window's status line)
    selection = Signal(object)    # the selected id (block, row) or None
    replaced = Signal()           # another document was installed
    rich_toast = Signal(object, object)       # (text or function, links or function): a toast with links (file drops)
    rich_toast_refresh = Signal()             # the text of a function toast changed (a background listing moved on)
    files_chosen = Signal()                   # a FILE cell of the table was selected: the panel shows what it holds
    appeared = Signal(object)     # the block a gesture made (the one of the kind dropped): shown on screen
    acted = Signal(str, object)   # an operation the bridge applied was accepted: (its name, its Result)

    def __init__(self, document=None, parent=None):
        super().__init__(parent)                          # the page's: it goes with the page (no cycle kept by Qt)
        self.doc = Doc(document if document is not None else MemoryDocument())
        self._drawing = None
        self._drawn_for = None        # (graph object, revision) the cached Drawing was laid out for
        self._drops = None
        self.file_cell = None         # (owner, kind) of the selected file cell of the table
        self.go_common = None         # set by the table host: selects the Common cell of a kind
        self.last_shown = ()          # ids the last accepted gesture made (kept for the checks of the window fuzz)
        self.to_show = ()             # ids the last accepted gesture made (focus, created): the canvas shows them
        self.open_entries = {}        # (owner, kind) -> entries unfolded in the files panel (kept over a redraw)

    @property
    def drops(self):
        """The files dropped on Session targets (graph_drops.py), made when first needed."""
        if self._drops is None:
            from .graph_drops import GraphDrops
            self._drops = GraphDrops(self)
        return self._drops

    # ------------------------------------------------------------------ the document
    @property
    def document(self):
        return self.doc.document

    def install(self, document):
        self.doc = Doc(document)
        self.file_cell = None
        self._drawing = None
        self.replaced.emit()
        self.changed.emit()
        self.selection.emit(self.selected)

    @property
    def graph(self):
        return self.doc.graph

    @property
    def view(self):
        return self.doc.view

    # ------------------------------------------------------------------ reading
    def _key(self):
        return id(self.doc.graph), getattr(self.doc.document, "revision", None)

    def stale(self) -> bool:
        """The document changed outside the page (the project model's other views: Post, the panel, the sources): the
        cached Drawing is not the graph's."""
        return self._drawing is None or self._drawn_for != self._key()

    def drawing(self):
        if self._drawing is not None and self._drawn_for != self._key():
            self._invalidate()                # changed outside the page: laid out again, a selection gone dropped
        if self._drawing is None:
            self._drawing = self.doc.layout()
            self._drawn_for = self._key()
        return self._drawing

    def places(self, kind=None, block=None):
        return self.doc.places(kind=kind, block=block)

    @property
    def selected(self):
        return self.doc.view.get("selected")

    def block(self, b):
        return self.doc.graph.blocks.get(b)

    def row_of(self, b):
        """The row (Row object) a block belongs to, or None (Post, unknown)."""
        for r in self.doc.graph.rows:
            if b in r.blocks or any(b in n for n in r.nights):
                return r
        return None

    def row(self, row_id):
        for r in self.doc.graph.rows:
            if r.id == row_id:
                return r
        return None

    def say(self, message) -> str:
        return self.doc.say(message)

    # ------------------------------------------------------------------ writing
    def _invalidate(self):
        self._drawing = None
        fc = self.file_cell
        if fc is not None and fc[0] != "common" and fc[0] not in self.doc.graph.blocks:
            self.file_cell = None             # the session of the selected file cell is gone (undo, delete)
        sel = self.doc.view.get("selected")   # what was selected may be gone (undo, delete): nothing is selected
        if sel and sel != PROJECT and sel not in self.doc.graph.blocks and self.row(sel) is None:
            self.doc.view["selected"] = None

    def act(self, name, /, quiet=False, **kw):
        """One operation, then the signals. quiet: True = no toast at all; "refusal" = no toast for a refusal.
        The operation's name is positional only: an operation's own ``name=`` (rename_row, rename_night) goes into
        ``kw``. -> the Result."""
        before = self.doc.graph
        res = self.after(self.doc.apply(name, **kw), quiet, kw.get("kind"), before=before)
        if res.ok:
            self.acted.emit(name, res)
        return res

    def act_many(self, steps, quiet=False):
        """Several operations as one undo step, then the signals. -> the Result."""
        before = self.doc.graph
        res = self.doc.apply_many(steps)
        return self.after(res, quiet, before=before)

    #: the type a Combination card makes (the palette's flyout of types: S-H-O, H-O, RGB, LRGB)
    combo_type = "sho"

    def _steps(self, name, kw):
        """The operation of a place or a card; a Combination of the type chosen in the palette's flyout."""
        steps = [(name, dict(kw or {}))]
        if (kw or {}).get("kind") == "combine" and self.combo_type != "sho":
            t = self.combo_type
            steps.append(("set_combination_type", lambda res: {"block": res[0].created["block"], "type": t}))
        return steps

    def drop(self, place):
        """Drop on a Place: its operation (a refused place gives its reason)."""
        if not place.lit:
            res = Result(False, reason=place.reason)
            self.after(res, False)
            return res
        name, kw = place.op
        before = self.doc.graph
        steps = self._steps(name, kw)
        res = self.doc.apply(name, **(kw or {})) if len(steps) == 1 else self.doc.apply_many(steps)
        return self.after(res, False, (kw or {}).get("kind"), before=before)

    def add_combination(self, **kw):
        """The Combination card double-clicked: placed at its natural place, of the type chosen in the flyout."""
        steps = self._steps("add_block", {"kind": "combine", **kw})
        if len(steps) == 1:
            return self.act("add_block", kind="combine", **kw)
        res = self.act_many(steps)
        if res.ok:
            self.acted.emit("add_block", res)
        return res

    def _drawable(self, back=None) -> bool:
        """The graph a change gave can be drawn; when its drawing raises, the error is logged and ``back()`` takes the
        change back (an undo, or the redo of an undo): nothing of it reaches Qt."""
        self._invalidate()
        try:
            self.drawing()
            return True
        except Exception:  # noqa: BLE001 - the page never lets the core's error reach Qt
            log.exception("graph page: the graph a gesture gave cannot be drawn")
            self._drawing = None
            if back is not None:
                back()
            self._invalidate()
            return False

    def after(self, res, quiet=False, kind=None, before=None):
        if res.ok and not self._drawable(self.doc.undo if before is not None and self.doc.graph is not before else None):
            res = Result(False, reason=FAILED)
            quiet = False
        if res.ok:
            made = res.created or {}
            new = next((b for b in made.get("blocks", []) if kind and self.block(b) is not None
                        and self.block(b).kind == kind), made.get("block"))
            if res.focus is not None and (res.focus in self.doc.graph.blocks or self.row(res.focus) is not None):
                self.doc.view["selected"] = res.focus         # a wire id is not a selection of the panel
            self.to_show = tuple(x for x in (res.focus, new) if x is not None and (
                x in self.doc.graph.blocks or self.row(x) is not None))
            self.last_shown = self.to_show
            self.changed.emit()
            self.selection.emit(self.selected)
            if new is not None:
                self.appeared.emit(new)
            if res.toast and quiet is not True:
                self.toast.emit(self.say(res.toast) + " · Ctrl+Z")
        elif res.reason and not quiet:
            self.toast.emit(self.say(res.reason))
        return res

    def undo(self):
        return self._after_history(self.doc.undo(), "undo")

    def redo(self):
        return self._after_history(self.doc.redo(), "redo")

    def _after_history(self, res, what="undo"):
        if res.ok and not self._drawable(self.doc.redo if what == "undo" else self.doc.undo):
            res = Result(False, reason=FAILED)
        if res.ok:
            self._invalidate()
            self.changed.emit()
            self.selection.emit(self.selected)
            if res.toast:
                self.toast.emit(self.say(res.toast))
        elif res.reason:
            self.toast.emit(self.say(res.reason))
        return res

    def choose_files(self, owner, kind):
        """A file cell of the table: the right panel shows its list (no block is selected)."""
        self.file_cell = (owner, kind)
        self.doc.view["selected"] = None
        self.selection.emit(None)
        self.files_chosen.emit()

    def choose(self, ident):
        """Select an id (view state, never undone)."""
        if self.file_cell is not None:
            self.file_cell = None
            self.doc.view["selected"] = ident
            self.selection.emit(ident)
            return
        if self.doc.view.get("selected") != ident:
            self.doc.view["selected"] = ident
            self.selection.emit(ident)

    def view_changed(self):
        """The view dict was changed by a view-only operation (fold, mode): draw again."""
        self._invalidate()
        self.changed.emit()

    def check_wire(self, src, src_port, dst, dst_port) -> str:
        """"" if a wire src.src_port -> dst.dst_port would be accepted, else the reason (nothing changes)."""
        return self.doc.wire_check(src, src_port, dst, dst_port) or ""

    def run_states_changed(self):
        self.states_changed.emit()

    def tr(self, s: str) -> str:
        return _(s)
