"""Files dropped on the graph's Session targets (the panel's drop zone and « Add… », the targets that open over a
Session block). It does what the table does with a drop, through the shared functions of files.py: the drop resolved
at once by the type filter and the engine's rule (a folder entry when the filter keeps what the engine reads there,
else its files one by one), « n ARW added · n JPG left · Take all », « n sub-folders not read · Include » (each
sub-folder holding frames added as an entry of its own), doubles not added (« Already there: nothing added »), one undo
step, nothing stored about the filter. The reader lists the stored entries in the background for the counts shown.

``bridge.rich_toast`` carries (text, links); a link calls ``link(key)``.
"""
from __future__ import annotations

import os

from PySide6.QtCore import QObject, Signal

from ..i18n import _
from . import files
from .table_reader import Reader


class GraphDrops(QObject):
    progress = Signal(object)         # the reader's thread -> this (the interface) thread
    finished = Signal()               # a listing ended: the counts can be drawn again

    def __init__(self, bridge):
        super().__init__()
        self.bridge = bridge
        self.reader = Reader()
        self.reader.listen(self.progress.emit)
        self.progress.connect(self._on_progress)
        self._last = None             # (session, kind, Resolution, filter) of the last drop: its links
        bridge.replaced.connect(self._forget)

    def _forget(self):
        self._last = None

    # ------------------------------------------------------------------ reading the lists the project holds
    def read(self, owner, kind, src):
        """Ask the reader for the entries of a list (cheap when nothing changed): the counts come from it."""
        if not src:
            return None
        for x in src:
            if self.reader.known(x) is None:
                self.reader.note(x, os.path.isdir(x))
        return self.reader.request((owner, kind), list(src))

    def count(self, src) -> tuple[int, bool]:
        return self.reader.count(list(src))

    # ------------------------------------------------------------------ a drop
    def drop(self, session, kind, paths):
        """Files / folders onto the `kind` list of a Session, resolved at once. -> the core's Result, or None when
        nothing was added."""
        bridge, doc = self.bridge, self.bridge.doc
        paths = [files.normalise(p) for p in paths]
        if not paths or session not in doc.graph.blocks:
            return None
        self._last = None
        files.note_paths(self.reader, paths)
        i = files.index()
        res = files.resolve(self.reader, paths, i)
        _src, there = doc.files_of(session, kind)
        ignored = files.without_doubles(res, there)
        self._last = (session, kind, res, i)
        if not res.items:
            self._say(*files.nothing_to_take(res, ignored))
            return None
        out = bridge.act("add_files", quiet=True, session=session, kind=kind, paths=res.items, mode="add")
        if not out.ok:
            self._say(bridge.say(out.reason) if out.reason else _("Not possible"), [])
            return out
        self.read(session, kind, doc.files_of(session, kind)[1])
        taken = res.frames()
        base = files.drop_toast(kind, doc.title(session), len(taken), "add")
        self._say(files.drop_sentence(base, res, "add", ignored, taken), files.drop_links(res, "add"))
        return out

    def _say(self, text, links):
        self.bridge.rich_toast.emit(text, links)

    # ------------------------------------------------------------------ the reader moved on
    def _on_progress(self, job):
        self.bridge.rich_toast_refresh.emit()
        if job.phase in ("done", "stopped"):
            self.finished.emit()

    # ------------------------------------------------------------------ the toast's links
    def _add(self, session, kind, paths, said):
        there = self.bridge.doc.files_of(session, kind)[1]
        paths = [p for p in paths if not files.already(p, there)]
        if not paths:
            self._say(_("Already there: nothing added"), [])
            return None
        res = self.bridge.act("add_files", quiet=True, session=session, kind=kind, paths=paths, mode="add")
        if res.ok:
            self.read(session, kind, self.bridge.doc.files_of(session, kind)[1])
            self._say(said, [(_("Undo"), "undo")])
        return res

    def link(self, key):
        """A link of the rich toast: "undo", "take_all" (the frames the filter left out, as files), "include" (each
        sub-folder holding frames, as an entry of its own)."""
        bridge = self.bridge
        if key == "undo":
            bridge.undo()
            return
        if self._last is None:
            return
        session, kind, res, i = self._last
        if key == "take_all" and res.left:
            for p in res.left:
                self.reader.note(p, False)
            self._add(session, kind, list(res.left), _("All taken"))
        elif key == "include" and res.subs:
            sub = files.include(self.reader, res.subs, i)
            if sub.items:
                self._add(session, kind, sub.items, _("Sub-folders included"))
            else:
                self._say(*files.nothing_to_take(sub, 0))
