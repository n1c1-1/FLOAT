"""Files dropped on the graph's Session targets (the panel's drop zone and « Add… », the targets that open over a
Session block). It does what the table does with a drop, through the shared functions of files.py: the type filter,
folders listed in the BACKGROUND (the drop answers at once), « n ARW added · n JPG left · Take all », only the direct
files of a folder with « n sub-folders not read · Include », doubles not added (« Already there: nothing added »), one
undo step; a folder with no direct matching file leaves no empty entry (the gesture is taken out again once the
listing says so, and the toast says why).

``bridge.rich_toast`` carries (text function, links function); a link calls ``link(key)``.
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
        self._rest = None
        self._include = None
        self._watching = None
        bridge.replaced.connect(self._forget)

    def _forget(self):
        self._rest = self._include = self._watching = None

    # ------------------------------------------------------------------ reading the lists the project holds
    def read(self, owner, kind, src):
        """Ask the reader for the entries of a list (cheap when nothing changed): the counts come from it."""
        if not src:
            return None
        for x in src:
            if self.reader.known(x) is None:
                self.reader.note(x, os.path.isdir(files.base(x)[0]))
        files.restore_options(self.reader, self.bridge.doc.view, owner, kind, list(src))
        return self.reader.request((owner, kind), list(src))

    def count(self, src) -> tuple[int, bool]:
        return self.reader.count(list(src))

    # ------------------------------------------------------------------ a drop
    def drop(self, session, kind, paths):
        """Files / folders onto the `kind` list of a Session. The model changes and the toast shows at once; the
        folders are listed afterwards, in the background. -> the core's Result, or None when nothing was added."""
        bridge, doc = self.bridge, self.bridge.doc
        paths = [files.normalise(p) for p in paths]
        if not paths or session not in doc.graph.blocks:
            return None
        self._rest = self._include = self._watching = None
        files.note_paths(self.reader, paths)
        items, loose = files.split_loose(self.reader, paths)
        _src, there = doc.files_of(session, kind)
        new = [p for p in items if not files.already(p, there)]
        ignored, items = len(items) - len(new), new
        self._rest = lambda: [("add_files", dict(session=session, kind=kind, paths=self._take(items, loose),
                                                 mode="add"))]
        if not items:
            self._say(*files.nothing_to_take(loose, ignored))
            return None
        res = bridge.act("add_files", quiet=True, session=session, kind=kind, paths=items, mode="add")
        if not res.ok:
            self._say(bridge.say(res.reason) if res.reason else _("Not possible"), [])
            return res
        files.save_options(self.reader, doc.view, session, kind, items)
        self.read(session, kind, self.bridge.doc.files_of(session, kind)[1])
        base = files.drop_toast(self.reader, kind, self.bridge.doc.title(session), items, "add")
        text = files.drop_sentence(self.reader, base, items, loose, "add", ignored)
        self._include = lambda: self._plan_include(session, kind, items)
        self._watching = dict(session=session, kind=kind, items=list(items), loose=list(loose),
                              rev=self.bridge.doc.revision)
        self._say(text, files.drop_links(self.reader, items, loose, "add"))
        self._check_watch()
        return res

    def _say(self, text, links):
        self.bridge.rich_toast.emit(text, links)

    def _take(self, items, loose):
        """The files the type filter left out, made known as single files. -> their paths."""
        left = list(self.reader.left_out(items))
        self.reader.adopt(left)
        for p in loose:
            self.reader.note(p, False)
        return [f.path for f in left] + list(loose)

    def _plan_include(self, session, kind, items):
        """The folders of the drop read at all levels: their entries get the mark. -> (op kwargs, changes)."""
        cur = list(self.bridge.doc.files_of(session, kind)[1])
        aims = [x for x in items if files.is_folder(self.reader, x) and not files.base(x)[1]]
        new = [x + files.MARK if x in aims else x for x in cur]
        chg = [(x, x + files.MARK) for x in cur if x in aims]
        return dict(session=session, kind=kind, paths=new, mode="replace"), chg

    # ------------------------------------------------------------------ the reader moved on
    def _on_progress(self, job):
        self._check_watch()
        self.bridge.rich_toast_refresh.emit()
        if job.phase in ("done", "stopped"):
            self.finished.emit()

    def _check_watch(self):
        """A folder with no direct matching file leaves no empty entry: once every folder of the drop is listed, the
        drop is undone and, when some folders did hold files, made again without the empty ones (one undo step)."""
        w = self._watching
        if not w:
            return
        bridge = self.bridge
        if bridge.doc.revision != w["rev"]:                     # something else was edited meanwhile: leave it
            self._watching = None
            return
        items = w["items"]
        if self.reader.expand(tuple(items)) is None:            # still listing
            return
        self._watching = None
        empties = files.empty_folders(self.reader, items)
        if not empties:
            return
        session, kind = w["session"], w["kind"]
        remaining = [x for x in items if x not in empties]
        bridge.doc.undo()
        bridge.view_changed()
        if remaining:
            res = bridge.act("add_files", quiet=True, session=session, kind=kind, paths=remaining, mode="add")
            if res.ok:
                self.read(session, kind, bridge.doc.files_of(session, kind)[1])
        text = files.drop_sentence(self.reader, _("Folder without a direct file: nothing added"), items, w["loose"],
                                   "add")
        links0 = files.drop_links(self.reader, items, w["loose"], "add")
        self._include = lambda: self._plan_include_empty(session, kind, empties, remaining)
        self._say(text, lambda: [(t, k) for t, k in links0() if k != "undo" or remaining])

    def _plan_include_empty(self, session, kind, empties, remaining):
        marked = [x + files.MARK for x in empties]
        for x in marked:
            self.reader.note(x, True)
            self.reader.opts[x] = self.reader.opts.get(x[:-len(files.MARK)], files.DEFAULT)
        cur = list(self.bridge.doc.files_of(session, kind)[1])
        aims = [x for x in remaining if files.is_folder(self.reader, x) and not files.base(x)[1]]
        new = [x + files.MARK if x in aims else x for x in cur] + marked
        chg = [(x, x + files.MARK) for x in cur if x in aims] + [(x[:-len(files.MARK)], x) for x in marked]
        return dict(session=session, kind=kind, paths=new, mode="replace"), chg

    # ------------------------------------------------------------------ the toast's links
    def link(self, key):
        """A link of the rich toast: "undo", "take_all", "include"."""
        bridge = self.bridge
        if key == "undo":
            bridge.undo()
        elif key == "take_all":
            steps = [s for s in (self._rest() if self._rest else []) if s[1].get("paths")]
            if steps:
                _name, kw = steps[0]
                res = bridge.act("add_files", quiet=True, **kw)
                if res.ok:
                    self.read(kw["session"], kw["kind"], bridge.doc.files_of(kw["session"], kw["kind"])[1])
                    self._say(_("All taken"), [(_("Undo"), "undo")])
        elif key == "include":
            plan = self._include() if self._include else None
            if plan:
                kw, chg = plan
                view = bridge.doc.view
                for old, new in chg:                            # the choice is recorded before the listing starts
                    name = files.option_name(self.reader, view, kw["session"], kw["kind"], old)
                    self.reader.opts[new] = self.reader.opts.get(old, files.DEFAULT)
                    self.reader.note(new, True)
                    view.setdefault(files.OPTIONS, {})[f"{kw['session']}|{kw['kind']}|{new}"] = [name, True]
                res = bridge.act("add_files", quiet=True, **kw)
                if res.ok:
                    self.read(kw["session"], kw["kind"], bridge.doc.files_of(kw["session"], kw["kind"])[1])
                    self._say(_("All levels read"), [(_("Undo"), "undo")])
