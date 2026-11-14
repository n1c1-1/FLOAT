"""The host of the table view in the graph page: ``TableHost(bridge)``.

The table (table.Table) is built the first time the host is shown, and again after another document is installed
(``bridge.replaced``). The host keeps it in step with the document: ``refresh()`` on ``bridge.changed`` (a graph
gesture, an undo, the table's own gestures); a cell chosen in the table goes to the panel (``bridge.choose`` for a
block or a stack, ``bridge.choose_files(owner, kind)`` for a file cell); the table's notes go to ``bridge.toast``.
Ctrl+Z / Ctrl+Y belong to the page: the table's own shortcuts are switched off.
"""
from __future__ import annotations

from PySide6.QtWidgets import QStackedLayout, QWidget

from ...graphcore.operations import COMMON
from .table_logic import FOLDED


class TableHost(QWidget):
    def __init__(self, bridge, parent=None):
        super().__init__(parent)
        self.bridge = bridge
        self.setObjectName("root")
        self.stack = QStackedLayout(self)
        self.empty = QWidget()
        self.stack.addWidget(self.empty)
        self.table = None
        self.built_for = None
        self._busy = False
        bridge.changed.connect(self.refresh)
        bridge.replaced.connect(self._replaced)

    def showEvent(self, e):                                           # noqa: N802
        super().showEvent(e)
        if self.table is None:
            self.build()

    def build(self):
        from .table import Table
        bridge = self.bridge
        if self.table is not None:
            self.table.grid.detach()
            self.stack.removeWidget(self.table)
            self.table.deleteLater()
        self.table = t = Table(bridge)
        self.built_for = bridge.doc
        self.stack.addWidget(t)
        self.stack.setCurrentWidget(t)
        t.delegate_undo()
        t.cell.connect(self._cell)
        t.message.connect(bridge.toast.emit)
        bridge.go_common = self.go_common
        return t

    def _replaced(self):
        if self.table is not None:
            self.build()

    def refresh(self):
        """The document changed (graph, undo, the table's own gesture): the table follows."""
        if self.table is None or self._busy:
            return
        self._busy = True
        try:
            if self.built_for is not self.bridge.doc:            # another document: build the table again
                self.build()
            else:
                self.table.refresh()
        finally:
            self._busy = False

    def _cell(self, ident):
        """A table cell was chosen: the right panel shows the block behind it, or the files of a file cell."""
        if ident is None:                                  # the table's selection was cleared: the panel shows nothing
            self.bridge.choose(None)
            return
        if ident[0] == "box":
            if not str(ident[1]).startswith(FOLDED):
                self.bridge.go_common = self.go_common
                self.bridge.choose_files(ident[1], ident[2])
            return
        block = self.table.block_of_cell(ident)
        if block:
            self.bridge.choose(block)

    def go_common(self, kind):
        """« See the Common cell »: select the Common cell of this kind and show its files."""
        ident = ("box", COMMON, kind)
        if self.table is not None:
            g = self.table.grid
            g.focus_cell(ident)
            g.sel = {ident}
            g.viewport().update()
        self.bridge.choose_files(COMMON, kind)
