"""The new graph page: the PREPROCESSING header with its one display control (Blocks | Folded | Table, Ctrl+J
cycles), the palette, the canvas or the table, and the right pane that shows a selected file cell's content.

It edits a document (document.py: graph, apply(steps), undo/redo, state(block), cost): ``GraphPage(document)`` runs
on a graph held in memory as well as on the client's project model: ``GraphPage(model, steps)`` takes a model with
``GraphEditing`` (graphmodel.py) as today's page takes it, and offers today's page's public signals and methods with
the same names (``openBlock``, ``openPost``, ``openLayer``, ``addSession``, ``variantMade``, ``changed``, ``toasted``,
``sideChosen``, ``sidesShown``; ``set_layout``, ``project_open``, ``select``, ``rebuild``, ``current_node``,
``place_images``, ``plus_session_pid``, ``request_refresh``, ``remove_tool``, ``graph``), so that the window swaps one
line. On a project model it hosts today's settings panel and its dock as today's page does (host.py), the palette's
ways of showing and the arrows of hidden sides, and says the time a change adds to the run. The window around it
(menus, Run) stays the window's: the page only emits what it cannot do itself (``inspect``, ``view_action``).
"""
from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import QEvent, Qt, QTimer, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import QButtonGroup, QFrame, QHBoxLayout, QLabel, QPushButton, QStackedWidget, QVBoxLayout, QWidget

from ..i18n import _
from . import style
from .bridge import PROJECT, Bridge
from .canvas import GraphCanvas, Toast
from .host import SidesHost
from .palette import Palette

log = logging.getLogger(__name__)
#: the project model's signals after which the page is drawn again (today's page connects the same; postChanged
#: through ``_post_changed``: not for a slider of the panel)
MODEL_SIGNALS = ("pathsChanged", "currentChanged", "costChanged", "sourcesChanged", "readyChanged")


def _is_model(x) -> bool:
    """A project model with GraphEditing (it holds the graph's view and gives the page its document)."""
    return x is not None and hasattr(x, "graph_view") and hasattr(x, "block_map")

#: the three positions of the display control: key, word, icon, tooltip
POSITIONS = (("blocks", "Blocks", "graph", "The blocks unfolded: every stack with all its blocks"),
             ("folded", "Folded", "layers", "Every stack as one card: the preprocessing fits in one column"),
             ("table", "Table", "samples", "The same preprocessing as a table, one line per session"))
RIGHT_W = 360


class GraphPage(SidesHost, QWidget):
    inspect = Signal(object)             # Enter / double-click / menu "Inspect" on this id
    view_action = Signal(str, object)    # a menu action only the window can do: ("mark_reviewed", block)
    display_changed = Signal(str)        # "blocks" | "folded" | "table"
    # today's page's signals (graph.py GraphPage), the window connects them where it builds the page
    openBlock = Signal()                 # double-click or Enter on a block: Inspect (Post when sel_step is "final")
    openPost = Signal()                  # the Post block opened
    openLayer = Signal(int)              # Open in Post: Post, with this layer selected
    addSession = Signal()                # (today's "+ Session" link; the new page's "+ Session" is a drop place)
    variantMade = Signal(int, int)       # a variant (workflow id) was made from this workflow (id)
    changed = Signal()                   # the page was drawn again: the window updates its Run button and status
    toasted = Signal(str)                # a toast was shown
    sideChosen = Signal(str, str)        # a side folded or pinned: "right" | "left", its state
    sidesShown = Signal()                # the arrow of a hidden side: the panels come back
    templatesAsked = Signal()            # the blank page's « From template »: the gallery is shown in its place

    def __init__(self, document=None, steps=None, parent=None):
        if isinstance(steps, QWidget) and parent is None:   # GraphPage(document, parent)
            steps, parent = None, steps
        super().__init__(parent)
        self.setObjectName("root")
        self.s = document if _is_model(document) else None   # the project model (today's page's ``s``), or None
        self.steps = steps
        self.left_state, self.right_state = "docked", "docked"
        self._pending = False
        self._mirroring = False
        self._last = None                # (seconds to run, count of workflows) at the last refresh: "+ N min to run"
        self.panel = self.dock = None    # today's settings panel and its dock: on a project model only
        self.templates = None            # the gallery's TemplatePage, made when first asked
        self.bridge = Bridge(document.document if self.s is not None else document, self)
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        v.addWidget(self._header())
        body = QWidget()
        self.body = body
        h = QHBoxLayout(body)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(0)
        self.palette = Palette(self.bridge)
        self._build_palette_room(body, h)
        self.centre = QStackedWidget()
        self.canvas = GraphCanvas(self.bridge, over_palette=self.palette.contains)
        self.table_host = None           # built when the table is first shown (it lists files in the background)
        self.centre.addWidget(self.canvas)
        h.addWidget(self.centre, 1)
        self.right = QFrame()
        self.right.setObjectName("panel")
        self.right.setFixedWidth(RIGHT_W)
        self.right_box = QVBoxLayout(self.right)
        self.right_box.setContentsMargins(0, 0, 0, 0)
        self.right.hide()
        self.files_panel = None
        h.addWidget(self.right)
        if self.s is not None:
            self._build_panel(body, h)
        v.addWidget(body, 1)
        body.installEventFilter(self)
        self.graph = self.canvas         # today's page's ``graph`` (its GraphView): rebuild(), plus_session_pid
        self.canvas.inspect.connect(self.inspect.emit)
        self.canvas.inspect.connect(self._opened)
        self.canvas.view_action.connect(self.view_action.emit)
        self.canvas.templates_asked.connect(self.show_templates)
        self.canvas.wire_chosen.connect(self.panel_later)
        self.bridge.changed.connect(self._update_header)
        self.bridge.changed.connect(self.changed.emit)
        self.bridge.states_changed.connect(self.changed.emit)
        self.bridge.toast.connect(self.toasted.emit)
        self.bridge.toast.connect(self.said)
        self.bridge.rich_toast.connect(self.said)
        self.bridge.files_chosen.connect(self._show_files)
        self.bridge.selection.connect(self._selection)
        self.bridge.selection.connect(self._mirror)
        self.bridge.selection.connect(self.panel_later)
        self.bridge.replaced.connect(self._replaced)
        self.bridge.acted.connect(self._acted)
        if self.s is not None:
            for name in MODEL_SIGNALS:
                sig = getattr(self.s, name, None)
                if sig is not None:
                    sig.connect(self.request_refresh)
            for name in ("pathsChanged", "currentChanged"):
                sig = getattr(self.s, name, None)
                if sig is not None:
                    sig.connect(self._project_choice)
            if getattr(self.s, "postChanged", None) is not None:
                self.s.postChanged.connect(self._post_changed)
        self._shortcuts()
        self._update_header()
        self.follow_model_selection()
        self.refresh_panel()

    # ------------------------------------------------------------------ the document
    @property
    def document(self):
        return self.bridge.document

    def set_document(self, document):
        self.bridge.install(document)

    # ------------------------------------------------------------------ the PREPROCESSING header
    def _header(self):
        band = QFrame()
        band.setObjectName("graph_header")
        band.setStyleSheet("QFrame#graph_header { background: #15171c; border-bottom: 1px solid #2a2f38; }")
        h = QHBoxLayout(band)
        h.setContentsMargins(12, 5, 12, 5)
        h.setSpacing(0)
        t = QLabel(_("PREPROCESSING"))
        t.setStyleSheet("color: #8b93a1; font-size: 8.5pt; font-weight: 700; letter-spacing: 1px; margin-right: 14px;")
        h.addWidget(t)
        self.positions: dict[str, QPushButton] = {}
        group = QButtonGroup(band)
        for key, word, role, tip in POSITIONS:
            b = style.set_icon(QPushButton(_(word)), role, 20)
            b.setObjectName("seg")
            b.setCheckable(True)
            b.setToolTip(_(tip) + "   " + _("(Ctrl+J: the next one)"))
            b.setMinimumWidth(104)
            b.setFocusPolicy(Qt.NoFocus)
            b.clicked.connect(lambda _c=False, k=key: self.set_display(k))
            group.addButton(b)
            h.addWidget(b)
            self.positions[key] = b
        h.addStretch(1)
        return band

    def position(self) -> str:
        v = self.bridge.view
        if v.get("mode") == "table":
            return "table"
        return "folded" if v.get("folded") is True else "blocks"

    def set_display(self, key):
        """Blocks | Folded | Table: the view state of the page (never an undo step)."""
        bridge = self.bridge
        if key == "table":
            if bridge.view.get("mode") != "table":
                self._set_mode("table")
        else:
            if bridge.view.get("mode") == "table":
                self._set_mode("graph")
            if (key == "folded") != (bridge.view.get("folded") is True):
                bridge.act("fold_all" if key == "folded" else "unfold_all")
        self._update_header()
        self.display_changed.emit(self.position())

    def cycle_display(self):
        """Ctrl+J: the next position; one the core refuses (Folded, when a variant starts from a stack's frames) is
        passed over, its reason said, so that Table is always reached."""
        order = [k for k, *_r in POSITIONS]
        key = order[(order.index(self.position()) + 1) % len(order)]
        if key == "folded":
            res = self.bridge.doc.try_many([("fold_all", {})])
            if not res.ok:
                self.set_display("table")
                self.bridge.toast.emit(self.bridge.say(res.reason))
                return
        self.set_display(key)

    def _set_mode(self, mode):
        self.bridge.doc.apply("set_mode", mode=mode)
        self.bridge.view_changed()

    def _update_header(self):
        pos = self.position()
        for key, b in self.positions.items():
            b.setChecked(key == pos)
        table = pos == "table"
        if table and self.table_host is None:
            self._build_table()
        self.centre.setCurrentWidget(self.table_host if table else self.canvas)
        self._show_palette(not table)
        if not table and self.bridge.file_cell is None:
            self._right_pane(False)

    def _build_table(self):
        from .table_host import TableHost
        self.table_host = TableHost(self.bridge)
        self.table_toast = Toast(self.table_host)       # the canvas' toast is hidden behind the table
        self.bridge.toast.connect(self._table_say)
        self.bridge.rich_toast.connect(lambda t, ls: self._table_say(t() if callable(t) else t, ls))
        self.centre.addWidget(self.table_host)

    def _table_say(self, text, links=None):
        if self.position() != "table":
            return
        if links:
            self.table_toast.say_links(text, links)
        else:
            self.table_toast.say(text)

    # ------------------------------------------------------------------ the right pane: a file cell's content
    def _show_files(self):
        from .files_panel import FilesPanel
        cell = self.bridge.file_cell
        if cell is None:
            return
        if self.files_panel is not None:
            self.right_box.removeWidget(self.files_panel)
            self.files_panel.deleteLater()
        self.files_panel = FilesPanel(self.bridge, cell[0], cell[1], go_common=self.bridge.go_common)
        self.right_box.addWidget(self.files_panel)
        self._right_pane(True)

    def _right_pane(self, on: bool):
        """A file cell's content on the right: in the settings panel's place while it is shown (one right side)."""
        self.right.setVisible(on)
        if self.panel is not None and self.dock is not None and self.dock.state == "docked":
            self.panel.setVisible(not on)

    def _project_choice(self, *_a):
        """The Project block chosen follows the project model at once (today's page read its choice from the model): a
        block chosen elsewhere (Inspect's strip) takes the choice from it; an Undo of what its form changed, which
        leaves the model's choice on the project, keeps it chosen."""
        s, b = self.s, self.bridge
        if s is None or self._mirroring:
            return
        step = getattr(s, "sel_step", "")
        if b.selected == PROJECT and step:
            ident = self._block_of_selection()
            if ident not in (None, PROJECT):
                self._mirroring = True
                try:
                    b.choose(ident)
                finally:
                    self._mirroring = False
        elif b.selected is None and not step and getattr(self, "_chosen", None) == PROJECT:
            b.doc.view["selected"] = PROJECT              # silently: the model's step dropped it, the model kept it
            if self.canvas.project_item is not None:
                self.canvas.project_item.selected = True
                self.canvas.project_item.update()

    def _selection(self, ident):
        self._chosen = ident
        if self.bridge.file_cell is None and self.files_panel is not None:
            self.right_box.removeWidget(self.files_panel)
            self.files_panel.deleteLater()
            self.files_panel = None
            self._right_pane(False)

    def _replaced(self):
        self._selection(None)
        self._update_header()

    # ------------------------------------------------------------------ keys
    def _shortcuts(self):
        for seq, fn in (("Ctrl+Z", self.bridge.undo), ("Ctrl+Y", self.bridge.redo),
                        ("Ctrl+Shift+Z", self.bridge.redo), ("Ctrl+J", self.cycle_display)):
            s = QShortcut(QKeySequence(seq), self)
            s.setContext(Qt.WidgetWithChildrenShortcut)
            s.activated.connect(fn)

    # ================================================================== today's page's public methods (graph.py)
    def set_layout(self, right: str, left: str):
        """The window's layout of the sides (layout.state, as today's page takes it): the settings panel (docked |
        icons | auto | hidden, its Dock) and the palette (docked | folded | auto | hidden, its arrow then)."""
        self.right_state, self.left_state = right, left
        if left != self.palette.state:
            self.palette.set_state(left)
        if self.dock is not None:
            self.dock.set_state(right)
        if right == "hidden":
            self.right.hide()
        shown = self.templates is None or self.centre.currentWidget() is not self.templates
        self._show_palette(shown and self.position() != "table")

    def eventFilter(self, obj, e):                                    # noqa: N802
        if obj is getattr(self, "body", None) and e.type() == QEvent.Resize:
            self._place_palette()
        return super().eventFilter(obj, e)

    def project_open(self) -> bool:
        """Whether the Project block is the one chosen: Inspect shows its form then."""
        return self.bridge.selected == PROJECT or (self.is_blank() and self.bridge.selected is None)

    def is_blank(self) -> bool:
        return not self.bridge.graph.rows

    def current_node(self, g=None):
        """The selected block as the panel, Inspect and the run know it: its ``Node`` (graphnode.BlockMap), the
        Project block's node, or None (nothing selected, or a page without a project model)."""
        sel = self.bridge.selected
        if sel == PROJECT or (self.is_blank() and sel is None):
            return self._project_node()
        bm = self._block_map()
        if bm is None or sel is None:
            return None
        n = bm.node(sel)
        if n is None and self.bridge.row(sel) is not None:   # a stack card: its Stacking block
            n = bm.node(self._card_block(sel))
        return n

    def _project_node(self):
        if getattr(self, "_pnode", None) is None:
            from ..graphnode import Node
            self._pnode = Node("project")
        return self._pnode

    def _block_map(self):
        return getattr(self.s, "block_map", None) if self.s is not None else None

    def _card_block(self, row_id):
        r = self.bridge.row(row_id)
        if r is None:
            return None
        kinds = {self.bridge.block(b).kind: b for b in r.blocks if self.bridge.block(b) is not None}
        return kinds.get("stacking") or (r.blocks[-1] if r.blocks else None)

    def select(self, index=None, step=None):
        """Select a workflow and a block as the window asks it (today's ``GraphPage.select``): the model's selection,
        then the block it names chosen on the page."""
        s = self.s
        if s is None:
            return
        s.pending_change = None
        s.select(index, step)
        ident = self._block_of_selection()
        bm = self._block_map()
        node = bm.node(ident) if bm is not None and ident not in (None, PROJECT) else None
        s.scope = [q.pid for q in node.paths] if node is not None and node.key not in ("sources", "final") else None
        self._mirroring = True
        try:
            self.bridge.choose(ident)
        finally:
            self._mirroring = False

    def _block_of_selection(self):
        """The block id of the model's selection (``path`` and ``sel_step``), PROJECT for none, or None."""
        s, bm = self.s, self._block_map()
        if s is None or bm is None:
            return None
        k = getattr(s, "sel_step", "")
        if not k:
            return PROJECT
        if k == "final":
            return next((b for b, blk in self.bridge.graph.blocks.items() if blk.kind == "post"), None)
        if k == "sources":
            return self._frames_block()
        n = bm.node_of(s.path, k) if s.path is not None else None
        return bm.block_of(n) if n is not None else None

    def _frames_block(self):
        """The block that stands for the frames on the page (today's Frames block): the first Session of the current
        workflow's stack; in a workflow without frames its Image (or Combination), never the frames' import."""
        s, g = self.s, self.bridge.graph
        q = getattr(s, "path", None)
        wids = getattr(getattr(s, "compiled", None), "wids", None) or {}
        mine = [r for r in g.rows if q is not None and wids.get(r.id) == q.pid]
        for r in mine + [r for r in g.rows if r not in mine]:
            for night in r.nights or []:
                if night and night[0] in g.blocks:
                    return night[0]
            for b in r.blocks:
                blk = g.blocks.get(b)
                if blk is not None and blk.kind in ("image", "combine"):
                    return b
        return None

    def follow_model_selection(self):
        """Nothing chosen on the page while the model has a selection (a project just opened: its frames): the page
        chooses the block it names, so that the panel shows it as today's page did. Once only: a block the user
        deselected stays deselected."""
        if self.s is None or getattr(self, "_followed", False) or self.is_blank():
            return
        self._followed = True
        if self.bridge.selected is not None:
            return
        ident = self._block_of_selection()
        if ident is None or ident == PROJECT:
            return
        self._mirroring = True
        try:
            self.bridge.choose(ident)
        finally:
            self._mirroring = False

    def _mirror(self, ident):
        """The page's selection told to the project model (BlockMap.address: the workflow and the view key), so that
        Inspect, the panel and the run window show the block chosen here (INTEGRATION §8)."""
        s = self.s
        if s is None or self._mirroring:
            return
        self._mirroring = True
        try:
            s.pending_change = None
            if ident == PROJECT:
                s.scope = None
                s.sel_step = ""
                s.select()
                return
            blk = self.bridge.block(ident) if ident is not None else None
            if blk is not None and blk.kind == "post":
                s.scope = None
                s.select(step="final")
                return
            if blk is not None and blk.kind == "combine_layers":   # its layer of Post (graph.py select_node)
                n = self.layer_node(ident)
                pid = n.paths[0].pid if n is not None and n.paths else None
                index = next((i for i, q in enumerate(s.paths) if q.pid == pid), None)
                if index is not None:
                    s.sel_layer = n.lid
                    s.select(index, "composition")
                    s.scope = [pid]
                return
            if ident is not None and self.bridge.row(ident) is not None:
                ident = self._card_block(ident)
            bm = self._block_map()
            addr = bm.address(ident) if bm is not None and ident is not None else None
            if addr is None:
                return
            wid, key = addr
            index = next((i for i, q in enumerate(s.paths) if q.pid == wid), None)
            if index is None:
                return
            s.select(index, key)
            n = bm.node(ident)
            s.scope = [q.pid for q in n.paths] if n is not None and n.key not in ("sources", "final") else None
        except Exception:  # noqa: BLE001 - the model's selection is a mirror: the page's stays
            log.exception("graph page: the selection could not be told to the project model")
        finally:
            self._mirroring = False

    def rebuild(self):
        """Draw the graph again (today's ``GraphView.rebuild``, called by the window after Post's changes)."""
        self.canvas.rebuild()

    @property
    def plus_session_pid(self):
        """The workflow a new session joins (today's "+ Session"): the selected block's, else None."""
        return self.canvas.plus_session_pid

    def request_refresh(self, *_a):
        """The project model changed (a run, Post, the panel, the sources): the page follows once the event loop
        turns."""
        if not self._pending:
            self._pending = True
            QTimer.singleShot(0, self, self.refresh)

    def refresh(self):
        self._pending = False
        b = self.bridge
        if b.stale():
            b.view_changed()                  # the graph changed outside the page: laid out and drawn again
        else:
            b.run_states_changed()            # states, badges and the reading line only
        if not self.is_blank():
            self.hide_templates()
        self.follow_model_selection()
        self.refresh_panel()
        self.cost_toast()

    def toast(self, text: str):
        self.canvas.say(text)
        self.toasted.emit(text)
        if "Ctrl+Z" in text and hasattr(self.s, "name_step"):
            self.s.name_step(text)                        # what Ctrl+Z / Ctrl+Y will say they undid or redid
            self.said()

    def remove_tool(self, node, pids=None):
        """Delete a block (today's ``remove_tool``, the window's Delete): the block of ``node``, else the selected
        one; refused, the toast says why."""
        bm = self._block_map()
        ident = bm.block_of(node) if bm is not None and node is not None else None
        ident = ident or self.bridge.selected
        if ident is None or ident == PROJECT:
            return
        if self.bridge.row(ident) is not None:
            self.bridge.act("delete_row", row=ident)
        elif ident in self.bridge.graph.blocks:
            self.canvas.delete_block(ident)

    def place_images(self, files: list):
        """One Image workflow per file, each a layer of Post (today's ``place_images``, File > Open images…): the first
        workflow made, or None (a file that cannot be read: said, nothing made; a page without a project model)."""
        s = self.s
        if s is None or not files:
            return None
        from ...stages.image import image_problem, stretched_by_nature
        for f in files:
            why = image_problem(f)
            if why:
                self.toast(_(why).format(name=Path(f).name))
                return None
        made = []
        try:
            with s.batch():
                for f in files:
                    q = s.add_image_workflow(s.unique_name(Path(f).stem), f, stretched=stretched_by_nature(f))
                    s.set_graph(q, [])
                    made.append(s.view(q.pid))
        except Exception:  # noqa: BLE001 - never reaches Qt: said and logged
            log.exception("graph page: the images could not be placed")
            self.toast(_("The gesture failed: nothing changed"))
            return None
        self.request_refresh()
        return made[0] if made else None

    # ------------------------------------------------------------------ what the window is told
    def _opened(self, ident):
        """Enter / double-click on a block: today's signals (Inspect; Post for the Post block; Post on its layer for a
        Combine layers block, graph.py ``open_layer``)."""
        blk = self.bridge.block(ident) if ident != PROJECT else None
        if blk is not None and blk.kind == "post":
            self.openPost.emit()
        elif blk is not None and blk.kind == "combine_layers":
            lid = self.layer_id(ident)
            if lid is not None:
                self.openLayer.emit(lid)
            else:
                self.openPost.emit()
        else:
            self.openBlock.emit()

    def _acted(self, name, res):
        """A variant made on the page: the window compares it with its original (today's ``variantMade``)."""
        if name != "add_variant" or self.s is None:
            return
        rows = (res.created or {}).get("rows") or []
        r = self.bridge.row(rows[0]) if rows else None
        wids = getattr(getattr(self.s, "compiled", None), "wids", None) or {}
        if r is not None and r.fork and wids.get(r.id) is not None and wids.get(r.fork[0]) is not None:
            self.variantMade.emit(int(wids[r.id]), int(wids[r.fork[0]]))

    # ------------------------------------------------------------------ the gallery, in place of the blank graph
    def show_templates(self):
        """« From template »: the gallery's templates in place of the blank graph (today's page shows the same)."""
        from ..gallery import TemplatePage
        if self.templates is None:
            self.templates = TemplatePage()
            self.templates.back.connect(self.hide_templates)
            self.templates.chosen.connect(self.template_chosen)
            self.centre.addWidget(self.templates)
        self.templates.build()
        self.centre.setCurrentWidget(self.templates)
        self._show_palette(False)
        self.templatesAsked.emit()

    def hide_templates(self):
        if self.templates is not None and self.centre.currentWidget() is self.templates:
            self._update_header()

    def template_chosen(self, t: dict):
        """A template of the gallery: its workflow made in one step by the core's operations (graphcore.templates), its
        Post actions by the project model when there is one."""
        from ...graphcore import templates as gt
        from ...graphcore.shapes import Result
        from ..gallery import _view_opts
        doc = self.bridge.doc
        if t.get("copy") is not None:                    # a copy is Ctrl+D on its workflow, not a template
            self.hide_templates()
            return
        base = _(t["palette"]) if t.get("kind") == "fusion" else _(t["name"])
        name = gt.Plan(doc.graph).unique_name(base)
        m = self.s
        if m is not None and hasattr(m, "_run_plan") and hasattr(m, "_template"):
            def run():
                from ..graphmodel import GraphRefused
                try:
                    return m._run_plan(m._template(t, name, _view_opts(t)))
                except GraphRefused as e:
                    return Result(False, reason=str(e))
            res = doc.guarded("template", run)
            if res.ok:
                self.bridge.view_changed()
            else:
                self.toast(self.bridge.say(res.reason))
        else:
            def build():
                return Result(True, created={"plan": gt.build(doc.graph, t, name, _view_opts(t), tr=_)})
            res = doc.guarded("template", build, take_back=False)
            if res.ok:
                self.bridge.act_many(res.created["plan"].steps)
            else:
                self.toast(self.bridge.say(res.reason))
        self.hide_templates()
