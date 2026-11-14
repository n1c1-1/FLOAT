"""The right panel of a FILE cell of the table: what was dropped there (not a repeat of the row).

The entries (folders or files) with, for each, its type filter, whether sub-folders are read, and its count; an entry
unfolds to its files grouped by folder (a flat model read by a QListView: only the visible rows are asked for, so
thousands of files stay light). Entries or files are chosen and « Remove » takes them out: an entry goes through the
core (``remove_files`` / ``set_common_files``, ONE undo step); a single file inside a dropped folder is kept as an
EXCLUSION in ``view["exclusions"]`` (view side, like the filter; the counts elsewhere do not follow it). A cell that
follows Common says so, with a link that selects the Common cell.
"""
from __future__ import annotations

import os

import shiboken6
from PySide6.QtCore import QAbstractListModel, Qt
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import QAbstractItemView, QHBoxLayout, QListView, QPushButton, QVBoxLayout, QWidget

from ...graphcore.operations import COMMON
from ..i18n import _, number, plural
from . import files
from .style import label
from .table_logic import KIND_TITLE

ENTRY, FOLDER, FILE = "entry", "folder", "file"
EXCLUSIONS = "exclusions"          # view key: {"owner|kind": [file paths left out by hand]}


def size_text(n):
    for unit in ("{v} B", "{v} KB", "{v} MB", "{v} GB"):
        if n < 1024 or unit == "{v} GB":
            return _(unit).format(v=f"{n:.0f}" if unit == "{v} B" else number(n, 1))
        n /= 1024
    return ""


class FileRows(QAbstractListModel):
    """The rows: an entry, then (when unfolded) its folders and files. A row is (kind, entry, text, file path)."""

    def __init__(self, panel):
        super().__init__()
        self.panel = panel
        self.rows = []
        self.open = set()

    def rebuild(self):
        self.beginResetModel()
        p = self.panel
        self.rows = []
        for e in p.entries():
            opened = e in self.open
            self.rows.append((ENTRY, e, p.entry_text(e, opened), None))
            if not opened:
                continue
            fs = p.reader.expand((e,))
            if fs is None:
                self.rows.append((FOLDER, e, _("reading…"), None))
                continue
            groups = {}
            for f in fs:
                groups.setdefault(os.path.dirname(f.path), []).append(f)
            root = e.rstrip("/")
            for d in sorted(groups):
                gl = groups[d]
                rel = d[len(root):].lstrip("/") if d.startswith(root) else d
                self.rows.append((FOLDER, e, f"{rel or '.'}  ·  {len(gl)}", None))
                for f in sorted(gl, key=lambda x: x.path):
                    self.rows.append((FILE, e, f"{os.path.basename(f.path)}   {size_text(f.size)}", f.path))
        self.endResetModel()

    def rowCount(self, parent=None):                                   # noqa: N802
        return 0 if parent is not None and parent.isValid() else len(self.rows)

    def data(self, index, role=Qt.DisplayRole):
        k, e, text, path = self.rows[index.row()]
        if role == Qt.DisplayRole:
            return {ENTRY: "", FOLDER: "      ", FILE: "            "}[k] + text
        if role == Qt.ForegroundRole:
            if k == FILE and path in self.panel.excluded():
                return QColor("#5d6470")
            return QColor("#e8ebf0" if k == ENTRY else "#c9d1dc" if k == FILE else "#8b93a1")
        if role == Qt.FontRole:
            f = QFont()
            f.setBold(k == ENTRY)
            f.setStrikeOut(k == FILE and path in self.panel.excluded())
            return f
        if role == Qt.ToolTipRole:
            return path or e
        return None

    def flags(self, index):
        k = self.rows[index.row()][0]
        return Qt.ItemIsEnabled | (Qt.ItemIsSelectable if k != FOLDER else Qt.NoItemFlags)


class FilesPanel(QWidget):
    """``FilesPanel(bridge, owner, kind, go_common=None)``: owner a session id or COMMON; go_common(kind) selects the
    Common cell of that kind (the « See the Common cell » link)."""

    def __init__(self, bridge, owner, kind, go_common=None):
        super().__init__()
        self.bridge, self.owner, self.kind = bridge, owner, kind
        self.go_common = go_common
        self.reader = bridge.drops.reader
        v = QVBoxLayout(self)
        v.setContentsMargins(14, 10, 14, 10)
        v.setSpacing(6)
        name = _("Common") if owner == COMMON else bridge.doc.title(owner)
        v.addWidget(label(f"{_(KIND_TITLE[kind])} · {name}", "h1"))
        self.rows = FileRows(self)
        self.rows.open = bridge.open_entries.setdefault((owner, kind), set())     # kept over a redraw
        blk = bridge.block(owner) if owner != COMMON else None
        self.inherited = blk is not None and kind not in (blk.files or {})
        self.link = None
        if self.inherited:
            v.addWidget(label(_("from Common: these files are those of the Common row."), "small", wrap=True))
            self.link = QPushButton(_("See the Common cell"))
            self.link.setObjectName("link")
            self.link.setCursor(Qt.PointingHandCursor)
            self.link.clicked.connect(self._to_common)
            v.addWidget(self.link, 0, Qt.AlignLeft)
        self.summary = label("", "small", wrap=True)
        v.addWidget(self.summary)
        self.view = QListView()
        self.view.setModel(self.rows)
        self.view.setUniformItemSizes(True)
        self.view.setTextElideMode(Qt.ElideRight)
        self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.view.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.view.setStyleSheet("QListView { background: #15171c; border: 1px solid #2a2f38; border-radius: 4px; } "
                                "QListView::item { padding: 2px 4px; } QListView::item:selected { background: #2b3446; }")
        self.view.clicked.connect(self._clicked)
        v.addWidget(self.view, 1)
        row = QHBoxLayout()
        self.b_remove = QPushButton(_("Remove"))
        self.b_remove.setToolTip(_("Removes the chosen folders or files (Del); Ctrl+Z undoes"))
        self.b_remove.clicked.connect(self.remove)
        row.addWidget(self.b_remove)
        self.b_unfold = QPushButton(_("Unfold all"))
        self.b_unfold.setObjectName("link")
        self.b_unfold.clicked.connect(self.unfold_all)
        row.addWidget(self.b_unfold)
        row.addStretch(1)
        v.addLayout(row)
        self.refresh()
        bridge.drops.finished.connect(self.refresh)
        self.view.selectionModel().selectionChanged.connect(lambda *_a: self._update_button())

    def _to_common(self):
        if self.go_common is not None:
            self.go_common(self.kind)
        elif self.bridge.go_common is not None:
            self.bridge.go_common(self.kind)

    # ------------------------------------------------------------------ what the list holds
    def held(self):
        """The list the cell holds (its own, or Common's it follows)."""
        d = self.bridge.doc
        if self.owner == COMMON:
            return list(d.graph.common["files"].get(self.kind, []))
        if self.owner not in d.graph.blocks:
            return []
        return list(d.files_of(self.owner, self.kind)[1])

    def entries(self):
        return [] if self.inherited else self.held()

    def excluded(self):
        return set(self.bridge.doc.view.get(EXCLUSIONS, {}).get(f"{self.owner}|{self.kind}", []))

    def entry_text(self, e, opened):
        name = os.path.basename(e.rstrip("/")) or e
        if files.is_folder(self.reader, e):
            fs = self.reader.expand((e,))
            n = "…" if fs is None else str(sum(1 for f in fs if f.path not in self.excluded()))
            return _("{mark} {name}   ·   without sub-folders · {n} files").format(
                mark="▾" if opened else "▸", name=name, n=n)
        return _("  {name}   ·   file").format(name=name)

    def refresh(self):
        src = self.held()
        self.bridge.drops.read(self.owner, self.kind, src)
        n, ok = self.reader.count(src)
        exc = len(self.excluded())
        if not src:
            text = _("no file")
        else:
            parts = [plural(len(src), "{n} entry", "{n} entries"),
                     plural(n - exc, "{n} file", "{n} files") if ok else _("reading…")]
            if exc:
                parts.append(plural(exc, "{n} excluded", "{n} excluded"))
            text = " · ".join(parts)
        self.summary.setText(text)
        self.rows.rebuild()
        self._update_button()

    def _update_button(self):
        self.b_remove.setEnabled(bool(self.view.selectionModel().selectedRows()) and not self.inherited)

    def _clicked(self, index):
        k, e, _t, _p = self.rows.rows[index.row()]
        if k == ENTRY and files.is_folder(self.reader, e):
            self.toggle(e)

    def toggle(self, e):
        if e in self.rows.open:
            self.rows.open.discard(e)
        else:
            self.rows.open.add(e)
        self.rows.rebuild()

    def unfold_all(self):
        es = [e for e in self.entries() if files.is_folder(self.reader, e)]
        if all(e in self.rows.open for e in es):
            self.rows.open.clear()
        else:
            self.rows.open.update(es)
        self.rows.rebuild()

    # ------------------------------------------------------------------ « Remove »
    def remove(self):
        bridge = self.bridge
        chosen = [self.rows.rows[i.row()] for i in self.view.selectionModel().selectedRows()]
        entries = [e for k, e, _t, _p in chosen if k == ENTRY]
        singles = [p for k, e, _t, p in chosen if k == FILE and e not in entries]
        if not entries and not singles:
            return
        view = bridge.doc.view
        if singles:
            rec = view.setdefault(EXCLUSIONS, {})
            key = f"{self.owner}|{self.kind}"
            rec[key] = sorted(set(rec.get(key, [])) | set(singles))
        if entries:
            left = [x for x in self.held() if x not in entries]
            if self.owner == COMMON:
                res = bridge.act("set_common_files", quiet=True, kind=self.kind, paths=left, mode="replace")
            else:
                res = bridge.act("remove_files", quiet=True, session=self.owner, kind=self.kind, paths=entries)
            if not res.ok:
                bridge.toast.emit(bridge.say(res.reason) if res.reason else _("Not possible"))
                return
        else:
            bridge.view_changed()
        n = len(entries) + len(singles)
        bridge.toast.emit(plural(n, "{n} removed", "{n} removed") + (" · Ctrl+Z" if entries else ""))
        if shiboken6.isValid(self):                   # the page may have built a new panel meanwhile
            self.refresh()

    def keyPressEvent(self, e):                                        # noqa: N802
        if e.key() in (Qt.Key_Delete, Qt.Key_Backspace) and self.b_remove.isEnabled():
            self.remove()
            return
        super().keyPressEvent(e)
