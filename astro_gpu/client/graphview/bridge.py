"""The only place where the page's widgets touch the document: it keeps the current document, caches the Drawing
between two changes (a repaint never lays out again) and turns the Result of an operation into signals (a toast, a
selection, a block to show).

Nothing is computed here: the drawing, the drop places, the texts and the legality all come from the core through
``Doc``. If the core changes a signature, only ``document.Doc`` and this file follow.
"""
from __future__ import annotations

from PySide6.QtCore import QObject, Signal

from ...graphcore.shapes import Result
from ..i18n import _
from .document import Doc, MemoryDocument

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

    def __init__(self, document=None):
        super().__init__()
        self.doc = Doc(document if document is not None else MemoryDocument())
        self._drawing = None
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
    def drawing(self):
        if self._drawing is None:
            self._drawing = self.doc.layout()
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

    def act(self, name, quiet=False, **kw):
        """One operation, then the signals. quiet: True = no toast at all; "refusal" = no toast for a refusal.
        -> the Result."""
        res = self.doc.apply(name, **kw)
        self.after(res, quiet, kw.get("kind"))
        return res

    def act_many(self, steps, quiet=False):
        """Several operations as one undo step, then the signals. -> the Result."""
        res = self.doc.apply_many(steps)
        self.after(res, quiet)
        return res

    def drop(self, place):
        """Drop on a Place: its operation (a refused place gives its reason)."""
        if not place.lit:
            res = Result(False, reason=place.reason)
            self.after(res, False)
            return res
        name, kw = place.op
        res = self.doc.apply(name, **(kw or {}))
        self.after(res, False, (kw or {}).get("kind"))
        return res

    def after(self, res, quiet=False, kind=None):
        if res.ok:
            self._invalidate()
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

    def undo(self):
        res = self.doc.undo()
        self._after_history(res)
        return res

    def redo(self):
        res = self.doc.redo()
        self._after_history(res)
        return res

    def _after_history(self, res):
        if res.ok:
            self._invalidate()
            self.changed.emit()
            self.selection.emit(self.selected)
            if res.toast:
                self.toast.emit(self.say(res.toast))
        elif res.reason:
            self.toast.emit(self.say(res.reason))

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
