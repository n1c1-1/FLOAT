"""The new graph page: the PREPROCESSING header with its one display control (Blocks | Folded | Table, Ctrl+J
cycles), the palette, the canvas or the table, and the right pane that shows a selected file cell's content.

It edits a document (document.py: graph, apply(steps), undo/redo, state(block), cost): ``GraphPage(document)`` runs
on a graph held in memory as well as, later, on the client's project model. The window around it (menus, side bars,
the settings panel, Run) stays the window's: the page only emits what it cannot do itself (``inspect``,
``view_action``).
"""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import QButtonGroup, QFrame, QHBoxLayout, QLabel, QPushButton, QStackedWidget, QVBoxLayout, QWidget

from ..i18n import _
from . import style
from .bridge import Bridge
from .canvas import GraphCanvas, Toast
from .palette import Palette

#: the three positions of the display control: key, word, icon, tooltip
POSITIONS = (("blocks", "Blocks", "graph", "The blocks unfolded: every stack with all its blocks"),
             ("folded", "Folded", "layers", "Every stack as one card: the preprocessing fits in one column"),
             ("table", "Table", "samples", "The same preprocessing as a table, one line per session"))
RIGHT_W = 360


class GraphPage(QWidget):
    inspect = Signal(object)             # Enter / double-click / menu "Inspect" on this id
    view_action = Signal(str, object)    # a menu action only the window can do: ("mark_reviewed", block)
    display_changed = Signal(str)        # "blocks" | "folded" | "table"

    def __init__(self, document=None, parent=None):
        super().__init__(parent)
        self.setObjectName("root")
        self.bridge = Bridge(document)
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        v.addWidget(self._header())
        body = QWidget()
        h = QHBoxLayout(body)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(0)
        self.palette = Palette(self.bridge)
        h.addWidget(self.palette)
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
        v.addWidget(body, 1)
        self.canvas.inspect.connect(self.inspect.emit)
        self.canvas.view_action.connect(self.view_action.emit)
        self.bridge.changed.connect(self._update_header)
        self.bridge.files_chosen.connect(self._show_files)
        self.bridge.selection.connect(self._selection)
        self.bridge.replaced.connect(self._replaced)
        self._shortcuts()
        self._update_header()

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
        self.palette.setVisible(not table)
        if not table and self.bridge.file_cell is None:
            self.right.hide()

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
        self.right.show()

    def _selection(self, ident):
        if self.bridge.file_cell is None and self.files_panel is not None:
            self.right_box.removeWidget(self.files_panel)
            self.files_panel.deleteLater()
            self.files_panel = None
            self.right.hide()

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
