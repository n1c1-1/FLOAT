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
        self._filter = None           # a drop held while the user chooses where frames of another filter go
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
        _src, there = doc.files_of(session, kind)
        res = files.resolve(self.reader, paths, i, existing=there)
        ignored = files.without_doubles(res, there)
        if kind == "lights":                            # the same night already in another session of the stack
            files.elsewhere(res, files.stack_lights(doc.graph, session=session))
        self._last = (session, kind, res, i)
        if not res.items:
            self._say(*files.nothing_to_take(res, ignored))
            return None
        if kind == "lights":                            # their header names another filter: the user chooses first
            sf, flt = files.session_filter(doc.graph, session), files.other_filter(doc.graph, session, res.items)
            if flt:
                self._filter = (session, kind, res, ignored)
                self._say(*files.filter_question(flt, sf, doc.title(session)))
                return None
        if kind == "flats":                             # flats of another filter than the lights': asked the same way
            flt = files.flat_filter(doc.graph, session, res.items)
            if flt:
                self._filter = (session, kind, res, ignored)
                self._say(*files.flat_question(flt, files.session_filter(doc.graph, session), doc.title(session)))
                return None
        return self._store(session, kind, res, ignored)

    def _store(self, session, kind, res, ignored):
        bridge, doc = self.bridge, self.bridge.doc
        out = bridge.act("add_files", quiet=True, session=session, kind=kind, paths=res.items, mode="add")
        if not out.ok:
            self._say(bridge.say(out.reason) if out.reason else _("Not possible"), [])
            return out
        self.read(session, kind, doc.files_of(session, kind)[1])
        taken = res.frames()
        base = files.drop_toast(kind, doc.title(session), len(taken), "add")
        note = bridge.take_stale()                      # a computed stack: what it computes again, said too
        self._say(files.drop_sentence(base, res, "add", ignored, taken) + (" · " + note if note else ""),
                  files.drop_links(res, "add"))
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
        if key in ("filter_here", "filter_own", "cancel"):     # frames of another filter: the user's choice
            held, self._filter = self._filter, None
            if held is None or key == "cancel":
                return
            session, kind, res, ignored = held
            if key == "filter_here":
                self._store(session, kind, res, ignored)
                return
            flt = files.header_filter(res.items)
            out = bridge.act_many(files.own_filter_steps(bridge.doc.graph, flt, res.items), quiet=True)
            if out.ok:
                self._say(_("New {f} session added").format(f=flt), [(_("Undo"), "undo")])
            return
        if self._last is None:
            return
        session, kind, res, i = self._last
        if key == "take_all" and res.left:
            for p in res.left:
                self.reader.note(p, False)
            self._add(session, kind, list(res.left), _("All taken"))
        elif key == "add_anyway" and res.elsewhere:
            self._add(session, kind, list(res.elsewhere), _("Added anyway"))
        elif key == "include" and res.subs:
            sub = files.include(self.reader, res.subs, i)
            if sub.items:
                self._add(session, kind, sub.items, _("Sub-folders included"))
            else:
                self._say(*files.nothing_to_take(sub, 0))
