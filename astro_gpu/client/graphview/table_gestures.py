"""The table's gestures (a mixin of table.Grid): mouse, keyboard, selection, menus, rename, structure edits. Every
gesture that changes the project goes through Grid.act (one Doc.apply_many = one undo step)."""
from __future__ import annotations

import os

from PySide6.QtCore import QEvent, QMimeData, QPoint, QPointF, Qt, QUrl
from PySide6.QtGui import QCursor, QDrag, QGuiApplication
from PySide6.QtWidgets import QMenu, QToolTip

from ...graphcore.operations import COMMON
from .. import prefs
from ..i18n import _, maybe, plural
from . import files
from .table_logic import BLOCK_TITLE, FOLDED, KIND_TITLE, KINDS, count_text
from .table_paint import HEAD, KIND_TIP, ROW

INTERNAL_MIME = "application/x-float-table"


def common_tip() -> str:
    """What « To Common » does (its name is two words)."""
    return _("Follows Common again: its own files or settings are dropped")


FOCUSABLE = ("name", "box", "cal", "setting", "stack", "ghost_s", "ghost_p", "folded_name", "body")


def undo_link():
    return [(_("Undo"), "undo")]


class Gestures:
    # ------------------------------------------------------------------ selection and focus
    def vy(self, c):
        return c.y - (HEAD + self.common_h) if c.pinned else c.y

    def focusables(self):
        out = []
        for c in self.pinned + self.cells:
            if c.kind not in FOCUSABLE:
                continue
            if c.kind == "box" and (c.id[1].startswith(FOLDED) or (c.id[1] == COMMON and c.id[2] == "lights")):
                continue
            out.append(c)
        return out

    def set_focus(self, c, keep_y=False):
        self.focus = c.id
        if not keep_y:
            self.focus_y = self.vy(c) + min(c.h, ROW) / 2
        self.ensure_visible(c)
        self.viewport().update()

    def select(self, c, mods=Qt.NoModifier):
        if c.kind not in ("name", "box", "stack", "setting", "cal"):
            return
        if c.kind == "box" and c.id[1].startswith(FOLDED):
            return
        if mods & Qt.ShiftModifier and self.focus and self.focus[0] == c.id[0]:
            order = [x.id for x in self.focusables() if x.id[0] == c.id[0] and
                     (c.id[0] != "box" or x.id[2] == c.id[2]) and (c.id[0] != "setting" or x.id[2] == c.id[2])]
            if self.focus in order and c.id in order:
                a, b = sorted((order.index(self.focus), order.index(c.id)))
                self.sel = set(order[a:b + 1])
                self.set_focus(c)
                self.toolbar_changed.emit()
                return
        if mods & Qt.ControlModifier:
            if c.id in self.sel:
                self.sel.discard(c.id)
            else:
                same = all(s[0] == c.id[0] and (c.id[0] not in ("box", "setting") or s[2] == c.id[2]) for s in self.sel)
                if not same:
                    self.sel = set()
                self.sel.add(c.id)
        else:
            self.sel = {c.id}
        self.set_focus(c)
        self.toolbar_changed.emit()
        if len(self.sel) == 1:
            self.open_cell.emit(c.id)

    def toggle_selection(self):
        c = self.cell(self.focus)
        if c is not None:
            self.select(c, Qt.ControlModifier)

    def clear_selection(self):
        if self.sel:
            self.sel = set()
            self.toolbar_changed.emit()
            self.viewport().update()
            self.open_cell.emit(None)
            return True
        return False

    # ------------------------------------------------------------------ mouse
    def mousePressEvent(self, e):                                     # noqa: N802
        if e.button() != Qt.LeftButton:
            return
        self.setFocus()
        self.end_edit()
        pos = e.position().toPoint()
        c = self.cell_at(pos)
        self._press = None
        if c is None:
            return
        z = self.zone_at(c, pos)
        self._press = (pos, c, z)
        if z in ("button", "leave", "reset", "picker", "stop", "chevron"):
            self.click_zone(c, z)
            self._press = None
            return
        if c.kind == "ghost_s":
            self.add_session(c.id[1])
            return
        if c.kind in ("ghost_p", "body"):
            self.add_stack()
            return
        if c.kind == "variant":
            self.sel = {c.id}
            self.toolbar_changed.emit()
            self.viewport().update()
            self.open_cell.emit(c.id)
        elif c.kind in ("name", "box", "stack", "setting", "cal"):
            self.select(c, e.modifiers())
        elif c.kind == "folded_name":
            self.set_focus(c)

    def mouseMoveEvent(self, e):                                      # noqa: N802
        pos = e.position().toPoint()
        if self._press and (e.buttons() & Qt.LeftButton) and self.drag is None:
            p0, c, z = self._press
            if (pos - p0).manhattanLength() > 7:
                self._press = None
                desc = None
                if z == "grip" and not self.multiple_selection():
                    desc = ("session", c.id[1])
                elif c.kind == "box" and self.model.box(c.id[1], c.id[2]).mode == "own" and \
                        not (c.id[1] == COMMON and c.id[2] == "lights"):
                    desc = ("box", c.id[1], c.id[2])
                elif c.kind == "stack" and z != "chevron":
                    desc = ("stack", c.id[1])
                if desc:
                    self._start_qdrag(desc)
                return
        c = self.cell_at(pos)
        nid = c.id if c is not None else None
        if nid != self.hover:
            self.hover = nid
            self.viewport().update()

    def _start_qdrag(self, desc):
        self.start_internal(desc)
        m = QMimeData()
        m.setData(INTERNAL_MIME, b"1")
        d = QDrag(self)
        d.setMimeData(m)
        d.exec(Qt.MoveAction | Qt.CopyAction, Qt.MoveAction)
        self._internal = None

    def mouseReleaseEvent(self, e):                                   # noqa: N802
        self._press = None

    def mouseDoubleClickEvent(self, e):                               # noqa: N802
        c = self.cell_at(e.position().toPoint())
        if c is not None and c.kind in ("name", "stack"):
            self.rename(c)

    def leaveEvent(self, e):                                          # noqa: N802
        self.hover = None
        self.viewport().update()

    def viewportEvent(self, e):                                       # noqa: N802
        if e.type() == QEvent.ToolTip:
            t = self.tooltip_text(e.pos())
            if t:
                QToolTip.showText(e.globalPos(), t, self.viewport())
            else:
                QToolTip.hideText()
            return True
        return super().viewportEvent(e)

    def tooltip_text(self, pos):
        if pos.y() < HEAD:
            for col, (x, w) in self.xs.items():
                if x <= pos.x() < x + w and col in KIND_TIP:
                    return _(KIND_TIP[col])
            return ""
        c = self.cell_at(pos)
        if c is None:
            return ""
        if self.zone_at(c, pos) == "picker":
            return _("Files: add, replace, remove, back to Common")
        mp = self.model
        if c.kind == "box" and not c.id[1].startswith(FOLDED):
            owner, kind = c.id[1], c.id[2]
            b = mp.box(owner, kind)
            t = [b.l1 + (f" · {b.l2}" if b.l2 else "")] if b.l1 else []
            if b.source == "common" and owner != COMMON:
                t.append(_("From Common · every session"))
            elif b.bar:
                t.append(_("Own to this session · Common has {n}").format(n=len(mp.common_list(kind))))
            src = mp.common_list(kind) if owner == COMMON else (mp.own_list(owner, kind) or mp.common_list(kind))
            t += [f"{os.path.basename(x) or x} : " + files.describe(x, files.is_folder(self.reader, x))
                  for x in src[:3]]
            if len(src) > 3:
                t.append(_("… {n} more").format(n=len(src) - 3))
            if owner != COMMON:
                t += [w.text for w in mp.box_warnings(owner, kind)]
            return "\n".join(x for x in t if x)
        if c.kind in ("setting", "cal"):
            _t, own, kind, b = self.cell_text(c)
            if own:
                parts = []
                for k in self.doc.own_keys(b):
                    parts.append(_("{k} {v} (common {c})").format(k=_(k), v=maybe(str(self.doc.effective(b).get(k))),
                                                                  c=maybe(str(self.doc.effective(kind).get(k)))))
                return _("Own: {list}").format(list=", ".join(parts))
            return _("Follows Common") if c.id[1] != COMMON else _("Value for every stack that follows it")
        if c.kind == "stack":
            pl = self.model.stack_by_id(c.id[1])
            if pl is None:
                return ""
            ns, nb, _w = mp.totals(pl)
            return " · ".join([pl["name"], plural(ns, "{n} session", "{n} sessions"), count_text(nb, "lights")])
        return ""

    # ------------------------------------------------------------------ clicks on a cell's buttons
    def click_zone(self, c, z):
        if z == "reset":
            self.back_to_common([c])
        elif z == "picker":
            self.files_menu(c)
        elif z == "stop":
            self.reader.stop((c.id[1], c.id[2]))
            self.reader.stop((COMMON, c.id[2]))
            self.undo()                                   # a cancelled listing takes its drop back: one undo step
            self.status.emit(_("Reading cancelled: the drop is taken back"), 4000)
        elif z == "chevron":
            self.model.folded ^= {c.id[1]}
            self.after_model()
        elif z in ("button", "leave"):
            w = self.model.box(c.id[1], c.id[2]).warn
            if w is None:
                return
            self.leave(w)
            if z == "button" and w.steps:
                self.act(w.steps, w.toast, undo_link())

    def leave(self, w):
        self.model.left.add(w.key)
        self.model.invalidate()
        self.rebuild()
        self.toolbar_changed.emit()

    def focus_cell(self, ident):
        c = self.cell(ident)
        if c is not None:
            self.set_focus(c)

    # ------------------------------------------------------------------ pickers, add
    def _choose_files(self, title):
        from PySide6.QtWidgets import QFileDialog
        filters = files.dialog_filters()
        fs, chosen = QFileDialog.getOpenFileNames(self, title, prefs.last_folder(), ";;".join(filters),
                                                  filters[files.index()])
        if fs:
            files.set_index(files.index_of_filter(chosen))           # remembered: it is also the one drops use
            prefs.remember_folder(os.path.dirname(fs[0]))
            self.filter_changed.emit()
        return fs

    def box_title(self, owner, kind):
        if owner == COMMON:
            return _("{kind} of Common").format(kind=_(KIND_TITLE[kind]))
        return _("{kind} of “{s}”").format(kind=_(KIND_TITLE[kind]), s=self.session_name(owner))

    def picker(self, c, replace=False):
        if c.kind != "box":
            return
        paths = self.choose_files(self.box_title(c.id[1], c.id[2]))
        if paths:
            self.drop_paths(paths, c, replace)

    def files_menu(self, c):
        """The caret of a file box: add, replace, remove, back to Common. Entries that do not apply are greyed, with
        why."""
        mp = self.model
        owner, kind = c.id[1], c.id[2]
        b = mp.box(owner, kind)
        menu = QMenu(self)
        menu.setToolTipsVisible(True)
        title = self.box_title(owner, kind)
        empty = b.mode in ("empty", "greyed")
        follows = owner != COMMON and mp.own_list(owner, kind) is None
        never = _("lights go on a session, not on Common") if owner == COMMON and kind == "lights" else ""
        entries = [
            (_("Add…"), lambda: self.picker(c), never),
            (_("Replace…"), lambda: self.picker(c, replace=True),
             never or (_("nothing to replace: the cell is empty") if empty else "")),
            (_("Remove"), lambda: self.empty_boxes([c]), never or (_("nothing to remove") if empty else "")),
            (_("To Common"), lambda: self.back_to_common([c]),
             _("lights have no Common") if kind == "lights" else
             _("it is the Common row") if owner == COMMON else _("already follows Common") if follows else ""),
        ]
        for txt, cb, why in entries:
            a = menu.addAction(txt + (f"  — {why}" if why else ""))
            a.setEnabled(not why)
            a.setToolTip(why or (common_tip() if txt == _("To Common") else f"{txt.rstrip('…')} : {title}"))
            a.triggered.connect(lambda _c=False, cb=cb: cb())
        z = self.zones(c).get("picker")
        if z is not None:
            y = z.bottom() if c.pinned else z.bottom() - self.scroll_y() + HEAD + self.common_h
            at = self.viewport().mapToGlobal(QPoint(int(z.left()), int(y)))
        else:
            at = QCursor.pos()
        menu.exec(at)

    def drop_paths(self, paths, c, replace=False):
        """Paste / picker: the same as a drop on cell c (left half = add, right half = replace)."""
        from .table import Drag
        g = Drag(paths=[files.normalise(p) for p in paths])
        for p in g.paths:
            d = os.path.isdir(p)
            self.reader.note(p, d)
            g.n_dirs += d
            g.n_files += not d
        target = self.target_for(c, QPointF(c.x + (c.w - 2 if replace else 1), c.y + 1), g)
        if not target.ok:
            self.status.emit(target.why, 4000)
            return
        self.drop_files(target, g)

    def add_session(self, rid):
        r = self.act([("add_night", dict(row=rid))], _("Session added"), undo_link())
        if r.ok:
            self.focus_last_session()

    def add_stack(self):
        r = self.act([("add_row", dict(kind="session"))], _("Stack added"), undo_link())
        if r.ok:
            self.focus_last_session()
            pls = self.model.stacks()
            c = self.cell(("stack", pls[-1]["id"])) if pls else None
            if c:
                self.rename(c)

    # ------------------------------------------------------------------ rename
    def rename(self, c):
        from .table import NameEditor
        if c.kind not in ("name", "stack"):
            self.status.emit(_("Common cannot be renamed") if c.kind == "common_name" else _("Nothing to rename here"),
                             3000)
            return
        self.end_edit()
        self.ensure_visible(c)
        y = c.y - self.scroll_y() + HEAD + self.common_h
        text = self.model.p.blocks[c.id[1]].name if c.kind == "name" else self.model.stack_by_id(c.id[1])["name"]
        ed = NameEditor(self.viewport(), text or "")
        ed.setGeometry(int(c.x + 4), int(y + 4), int(c.w - 8), 24)
        ed.accepted.connect(lambda t, c=c: self._commit_name(c, t))
        ed.cancelled.connect(self.end_edit)
        ed.show()
        ed.setFocus()
        self._editor = ed

    def end_edit(self):
        ed = self._editor
        if ed is not None:
            self._editor = None
            ed.done = True
            ed.hide()
            ed.deleteLater()
            self.setFocus()

    def _commit_name(self, c, t):
        self.end_edit()
        t = t.strip()
        if not t:
            return
        if c.kind == "name":
            if t != self.model.p.blocks[c.id[1]].name:
                self.act([("rename_night", dict(night=c.id[1], name=t))],
                         _("Session renamed “{t}”").format(t=t), undo_link())
        elif t != self.model.stack_by_id(c.id[1])["name"]:
            self.act([("rename_row", dict(row=c.id[1], name=t))], _("Stack renamed “{t}”").format(t=t), undo_link())

    # ------------------------------------------------------------------ remove, reset
    def delete(self):
        mp = self.model
        cs = [self.cell(i) for i in sorted(self.sel, key=str)] if self.focus in self.sel else []
        if not cs and self.focus:
            cs = [self.cell(self.focus)]
        cs = [c for c in cs if c is not None]
        if not cs:
            return
        k = cs[0].kind
        if k == "box":
            self.empty_boxes(cs)
        elif k in ("setting", "cal"):
            self.back_to_common(cs)
        elif k == "name":
            sids = [c.id[1] for c in cs if c.kind == "name"]
            if len(sids) == 1:
                txt = _("“{s}” removed · files kept").format(s=self.session_name(sids[0]))
            else:
                txt = plural(len(sids), "{n} session removed · files kept", "{n} sessions removed · files kept")
            self.sel = set()
            self.act([("remove_night", dict(night=s)) for s in sids], txt, undo_link())
        elif k == "stack":
            steps, names, ref_msg = [], [], ""
            gone = {c.id[1] for c in cs if c.kind == "stack"}
            for rid in gone:
                pl = mp.stack_by_id(rid)
                if pl is None:
                    continue
                names.append(pl["name"])
                steps.append(("delete_row", dict(row=rid)))
                if pl["ref"]:
                    nxt = next((q for q in mp.stacks() if q["id"] not in gone), None)
                    if nxt is not None:
                        ref_msg = _(" · the reference passes to “{p}”").format(p=nxt["name"])
            self.sel = set()
            txt = (_("Stack “{p}” removed · files kept").format(p=names[0]) if len(names) == 1 else
                   plural(len(names), "{n} stack removed · files kept", "{n} stacks removed · files kept"))
            self.act(steps, txt + ref_msg, undo_link())
        else:
            self.status.emit(_("Nothing to remove here"), 3000)

    def empty_boxes(self, cs):
        mp = self.model
        steps, msgs = [], []
        for c in cs:
            owner, kind = c.id[1], c.id[2]
            if owner == COMMON:
                if mp.common_list(kind):
                    left = sum(1 for pl in mp.stacks() for s in pl["sessions"] if mp.own_list(s["sid"], kind) is None)
                    steps.append(("set_common_files", dict(kind=kind, paths=[], mode="replace")))
                    msgs.append(plural(left, "{kind} of Common emptied: {n} session has none left",
                                       "{kind} of Common emptied: {n} sessions have none left",
                                       kind=_(KIND_TITLE[kind])))
            elif owner.startswith(FOLDED):
                continue
            elif mp.own_list(owner, kind) is not None:
                steps.append(("reset_own_files", dict(session=owner, kind=kind)))
                msgs.append(_("{kind} of “{s}” emptied · files kept").format(kind=_(KIND_TITLE[kind]),
                                                                              s=self.session_name(owner)))
        if not steps:
            inherited = any(mp.box(c.id[1], c.id[2]).source == "common" for c in cs if c.id[1] != COMMON)
            self.status.emit(_("This cell follows Common: nothing to remove here") if inherited else
                             _("Cell already empty"), 3000)
            return
        txt = msgs[0] if len(msgs) == 1 else plural(len(msgs), "{n} cell emptied · files kept",
                                                    "{n} cells emptied · files kept")
        self.act(steps, txt, undo_link())

    def back_to_common(self, cs):
        mp = self.model
        steps = []
        cs = [c for c in cs if c is not None]
        for c in cs:
            if c.kind == "box":
                o, k = c.id[1], c.id[2]
                if o != COMMON and mp.own_list(o, k) is not None:
                    steps.append(("reset_own_files", dict(session=o, kind=k)))
            elif c.kind in ("setting", "cal") and c.id[1] != COMMON:
                b = self.block_of(c)
                if b and self.doc.own_keys(b):
                    steps.append(("reset_own", dict(block=b)))
            elif c.kind == "name":
                sid = c.id[1]
                for k in KINDS:
                    if mp.own_list(sid, k) is not None:
                        steps.append(("reset_own_files", dict(session=sid, kind=k)))
                cid = mp.calibration_of(sid)
                if cid and self.doc.own_keys(cid):
                    steps.append(("reset_own", dict(block=cid)))
        if not steps:
            self.status.emit(_("Nothing of its own here: everything already follows Common"), 3000)
            return
        if len(steps) == 1 and cs[0].kind == "box":
            txt = _("“{s}” follows Common again: {kind}").format(s=self.session_name(cs[0].id[1]),
                                                                 kind=_(KIND_TITLE[cs[0].id[2]]))
        elif len(cs) == 1 and cs[0].kind in ("setting", "cal"):
            txt = _("“{p}” follows Common again").format(p=self.owner_name(cs[0]))
        else:
            txt = plural(len(steps), "{n} item back to Common", "{n} items back to Common")
        self.act(steps, txt, undo_link())

    def owner_name(self, c):
        """The name a settings cell is said with: its stack's, or its session's (Calibration)."""
        if c.kind == "setting":
            pl = self.model.stack_by_id(c.id[1])
            return pl["name"] if pl else c.id[1]
        return self.session_name(c.id[1])

    def set_value(self, ident, key, value):
        """Change one setting of a settings cell (what the panel does). -> the Result (None: no such cell)."""
        c = self.cell(ident)
        if c is None:
            return None
        kind = self.setting_kind(c)
        if c.id[1] == COMMON:
            followers = [b for b in self.doc.graph.blocks.values() if b.kind == kind and key not in b.settings]
            done = sum(1 for b in followers if self.doc.state(b.id) == "done")
            block = _(BLOCK_TITLE[kind])
            txt = (plural(done, "Common {block} changed: {n} stack to recompute",
                          "Common {block} changed: {n} stacks to recompute", block=block) if done else
                   _("Common {block} changed").format(block=block))
            return self.act([("set_common_setting", dict(kind=kind, key=key, value=value))], txt, undo_link())
        b = self.block_of(c)
        if b is None:
            return None
        before = key in self.doc.own_keys(b)
        r = self.act([("set_setting", dict(block=b, key=key, value=value))])
        if r.ok:
            after = key in self.doc.own_keys(b)
            name = self.owner_name(c)
            if before and not after:
                self.toast.emit(_("“{p}” follows Common again").format(p=name), undo_link())
            else:
                self.toast.emit(_("Own setting for “{p}”").format(p=name), undo_link())
        return r

    # ------------------------------------------------------------------ structure
    def move_step(self, direction):
        sids = self.chosen_sessions() or ([self.focus[1]] if self.focus and self.focus[0] == "name" else [])
        if len(sids) != 1:
            return
        sid = sids[0]
        pls = self.model.stacks()
        pl = self.model.stack_of(sid)
        names = [s["sid"] for s in pl["sessions"]]
        i = names.index(sid)
        k = [q["id"] for q in pls].index(pl["id"])
        if direction < 0:
            if i > 0:
                args = dict(night=sid, row=pl["id"], before=names[i - 1])
            elif k == 0:
                self.status.emit(_("Already at the top"), 2000)
                return
            else:
                args = dict(night=sid, row=pls[k - 1]["id"])
        else:
            if i + 1 < len(names):
                args = dict(night=sid, row=pl["id"])
                if i + 2 < len(names):
                    args["before"] = names[i + 2]
            elif k + 1 >= len(pls):
                self.status.emit(_("Already at the bottom"), 2000)
                return
            else:
                args = dict(night=sid, row=pls[k + 1]["id"], before=pls[k + 1]["sessions"][0]["sid"])
        self.act([("move_night", args)], _("“{s}” moved").format(s=self.session_name(sid)), undo_link())
        self.focus_cell(("name", sid))

    def move_to(self, sids, rid):
        if not sids:
            return
        if rid is None:
            first = sids[0]
            steps = [("move_night", dict(night=first, row=None))] + [
                ("move_night", lambda outs, s=s: dict(night=s, row=outs[0].created["row"])) for s in sids[1:]]
            where = _("new stack")
        else:
            steps = [("move_night", dict(night=s, row=rid)) for s in sids]
            where = self.model.stack_by_id(rid)["name"]
        txt = (_("“{s}” goes into “{p}”").format(s=self.session_name(sids[0]), p=where) if len(sids) == 1 else
               plural(len(sids), "{n} session goes into “{p}”", "{n} sessions go into “{p}”", p=where))
        self.sel = set()
        self.act(steps, txt, undo_link())

    def duplicate(self):
        c = self.cell(self.focus)
        if c is None:
            return
        mp = self.model
        if c.kind == "stack":
            pl = mp.stack_by_id(c.id[1])
            self.act([("duplicate_row", dict(row=pl["id"]))], _("Copy of “{p}” added").format(p=pl["name"]), undo_link())
        elif c.kind in ("name", "box", "cal") and c.id[1] != COMMON and not str(c.id[1]).startswith(FOLDED):
            sid = c.id[1]
            pl = mp.stack_of(sid)
            names = [s["sid"] for s in pl["sessions"]]
            i = names.index(sid)
            steps = [("add_night", dict(row=pl["id"]))]
            for k in KINDS:
                own = mp.own_list(sid, k)
                if k != "lights" and own is not None:
                    steps.append(("add_files", lambda outs, k=k, own=own: dict(
                        session=outs[0].created["session"], kind=k, paths=list(own), mode="replace")))
            if i + 1 < len(names):
                steps.append(("move_night", lambda outs, nx=names[i + 1]: dict(
                    night=outs[0].created["session"], row=pl["id"], before=nx)))
            self.act(steps, _("“{s}” duplicated without lights").format(s=self.session_name(sid)), undo_link())

    def take_reference(self, rid):
        pl = self.model.stack_by_id(rid)
        self.act([("set_reference", dict(row=rid))], _("“{p}” is the reference").format(p=pl["name"]), undo_link())

    def merge(self, src, dst):
        a, b = self.model.stack_by_id(src), self.model.stack_by_id(dst)
        self.act([("merge_rows", dict(src=src, dst=dst))],
                 _("“{a}” merged into “{b}”: own settings dropped").format(a=a["name"], b=b["name"]), undo_link())

    # ------------------------------------------------------------------ copy / paste
    def copy(self):
        c = self.cell(self.focus)
        if c is None or c.kind != "box":
            return
        owner, kind = c.id[1], c.id[2]
        mp = self.model
        lst = mp.common_list(kind) if owner == COMMON else (mp.own_list(owner, kind) or mp.common_list(kind))
        if not lst:
            self.status.emit(_("Nothing to copy here"), 2000)
            return
        m = QMimeData()
        m.setUrls([QUrl.fromLocalFile(p) for p in lst])
        QGuiApplication.clipboard().setMimeData(m)
        self.status.emit(plural(len(lst), "{n} item copied", "{n} items copied"), 2000)

    def paste(self):
        m = QGuiApplication.clipboard().mimeData()
        if m is None or not m.hasUrls():
            self.status.emit(_("Nothing to paste"), 2000)
            return
        paths = self.paths_of(m)
        cs = [self.cell(i) for i in self.sel] if len(self.sel) > 1 else [self.cell(self.focus)]
        cs = [c for c in cs if c is not None]
        if not cs:
            return
        self.drop_paths(paths, cs[0])

    # ------------------------------------------------------------------ warnings
    def next_warning(self):
        ws = self.model.all_warnings()
        if not ws:
            self.status.emit(_("No warning"), 2500)
            return
        self._warn_idx = (self._warn_idx + 1) % len(ws)
        w = ws[self._warn_idx]
        c = self.cell(("box", w.owner, w.kind))
        if c is not None:
            self.sel = {c.id}
            self.set_focus(c)
            self.toolbar_changed.emit()
        self.status.emit(f"⚠ {w.text}  ({self._warn_idx + 1}/{len(ws)})", 6000)

    # ------------------------------------------------------------------ keyboard
    def neighbour(self, c, d):
        cs = self.focusables()
        fy = self.focus_y
        if d in ("L", "R"):
            cand = [x for x in cs if x is not c and self.vy(x) <= fy < self.vy(x) + x.h and
                    ((x.x >= c.x + c.w - 1) if d == "R" else (x.x + x.w <= c.x + 1))]
            return min(cand, key=lambda x: abs(x.x - c.x)) if cand else None
        top, bottom = self.vy(c), self.vy(c) + c.h
        x0, x1 = c.x, c.x + c.w
        if d == "D":
            cand = [x for x in cs if self.vy(x) >= bottom - 1 and x.x < x1 - 1 and x.x + x.w > x0 + 1 and x is not c]
            if not cand:
                return None
            ymin = min(self.vy(x) for x in cand)
            cand = [x for x in cand if self.vy(x) == ymin]
        else:
            cand = [x for x in cs if self.vy(x) + x.h <= top + 1 and x.x < x1 - 1 and x.x + x.w > x0 + 1 and x is not c]
            if not cand:
                return None
            ymax = max(self.vy(x) + x.h for x in cand)
            cand = [x for x in cand if self.vy(x) + x.h == ymax]
        return min(cand, key=lambda x: abs((x.x + x.w / 2) - (c.x + c.w / 2)))

    def navigate(self, d):
        cs = self.focusables()
        c = self.cell(self.focus)
        if c is None:
            if cs:
                self.set_focus(next((x for x in cs if x.kind == "name"), cs[0]))
            return
        n = self.neighbour(c, d)
        if n is not None:
            self.set_focus(n, keep_y=d in ("L", "R"))
            if d in ("U", "D"):
                if n.h > ROW:
                    self.focus_y = min(max(self.focus_y, self.vy(n) + 4), self.vy(n) + n.h - 4)
                else:
                    self.focus_y = self.vy(n) + n.h / 2

    def focusNextPrevChild(self, forward):                            # noqa: N802
        cs = sorted(self.focusables(), key=lambda x: (round(self.vy(x)), x.x))
        c = self.cell(self.focus)
        if c is None or c not in cs:
            if cs:
                self.set_focus(cs[0] if forward else cs[-1])
                return True
            return False
        i = cs.index(c) + (1 if forward else -1)
        if 0 <= i < len(cs):
            self.set_focus(cs[i])
            return True
        return False

    def activate(self):
        c = self.cell(self.focus)
        if c is None:
            return
        k = c.kind
        if k == "box":
            if self.model.box(c.id[1], c.id[2]).mode == "empty":
                self.picker(c)
            else:
                self.open_cell.emit(c.id)
        elif k in ("name", "stack"):
            self.rename(c)
        elif k == "ghost_s":
            self.add_session(c.id[1])
        elif k in ("ghost_p", "body"):
            self.add_stack()
        elif k in ("setting", "cal"):
            self.open_cell.emit(c.id)

    def keyPressEvent(self, e):                                       # noqa: N802
        k, m = e.key(), e.modifiers()
        ctrl, shift = bool(m & Qt.ControlModifier), bool(m & Qt.ShiftModifier)
        if ctrl and k in (Qt.Key_Z, Qt.Key_Y):
            if self.undo_delegated:                   # the page owns Ctrl+Z / Ctrl+Y
                e.ignore()
                return
            if k == Qt.Key_Y or shift:
                self.redo()
            else:
                self.undo()
        elif ctrl and shift and k in (Qt.Key_Up, Qt.Key_Down):
            self.move_step(-1 if k == Qt.Key_Up else 1)
        elif k in (Qt.Key_Up, Qt.Key_Down, Qt.Key_Left, Qt.Key_Right) and not ctrl:
            self.navigate({Qt.Key_Up: "U", Qt.Key_Down: "D", Qt.Key_Left: "L", Qt.Key_Right: "R"}[k])
        elif k in (Qt.Key_Return, Qt.Key_Enter):
            self.activate()
        elif k == Qt.Key_F2:
            c = self.cell(self.focus)
            if c is not None:
                self.rename(c)
        elif k == Qt.Key_Space:
            self.toggle_selection()
        elif k == Qt.Key_Delete:
            self.delete()
        elif ctrl and k == Qt.Key_C:
            self.copy()
        elif ctrl and k == Qt.Key_V:
            self.paste()
        elif ctrl and k == Qt.Key_D:
            self.duplicate()
        elif k == Qt.Key_F8:
            self.next_warning()
        elif k == Qt.Key_Escape:
            if self.drag is not None:
                self.end_drag()
            else:
                self.clear_selection()
        elif (shift and k == Qt.Key_F10) or k == Qt.Key_Menu:
            c = self.cell(self.focus)
            if c is not None:
                y = c.y if c.pinned else c.y - self.scroll_y() + HEAD + self.common_h
                self.context_menu(c, self.viewport().mapToGlobal(QPoint(int(c.x + 10), int(y + 10))))
        else:
            super().keyPressEvent(e)

    # ------------------------------------------------------------------ context menu
    def contextMenuEvent(self, e):                                    # noqa: N802
        c = self.cell_at(self.viewport().mapFrom(self, e.pos()))
        if c is None:
            return
        if c.id not in self.sel and c.kind in ("name", "box", "stack", "setting", "cal"):
            self.select(c)
        self.context_menu(c, e.globalPos())

    def menu_entries(self, c):
        """[(text, callback, enabled[, tooltip])]; None = a separator. Also what the tests call. A name is two words at
        most; what it does is said in its tooltip."""
        mp = self.model
        out = []
        k = c.kind
        if k == "stack":
            pl = mp.stack_by_id(c.id[1])
            out += [(_("Rename (F2)"), lambda: self.rename(c), True),
                    (_("As reference"), lambda: self.take_reference(pl["id"]), not pl["ref"],
                     _("This stack becomes the reference the other stacks are aligned on")),
                    (_("Duplicate (Ctrl+D)"), self.duplicate, True),
                    (_("Unfold") if pl["folded"] else _("Fold"), lambda: self.click_zone(c, "chevron"), True), None]
            for p2 in mp.stacks():
                if p2["id"] != pl["id"]:
                    out.append((_("Merge into “{p}”").format(p=p2["name"]),
                                lambda s=pl["id"], d=p2["id"]: self.merge(s, d), True))
            out += [None, (_("Delete stack"), self.delete, True)]
        elif k == "name":
            sids = self.chosen_sessions() or [c.id[1]]
            out += [(_("Rename (F2)"), lambda: self.rename(c), len(sids) == 1),
                    (_("Duplicate (Ctrl+D)"), self.duplicate, len(sids) == 1),
                    (_("To Common"), lambda: self.back_to_common([self.cell(("name", s)) for s in sids]), True,
                     common_tip()),
                    None, (_("New stack"), lambda: self.move_to(sids, None), True)]
            for p2 in mp.stacks():
                out.append((_("Move to “{p}”").format(p=p2["name"]), lambda r=p2["id"]: self.move_to(sids, r), True))
            out += [None, (_("Delete"), self.delete, True)]
        elif k == "box" and not c.id[1].startswith(FOLDED):
            owner, kind = c.id[1], c.id[2]
            out += [(_("Show files"), lambda: self.open_cell.emit(c.id), True, _("Its files in the panel on the right")),
                    (_("Add…"), lambda: self.picker(c), owner != COMMON or kind != "lights"),
                    (_("Copy (Ctrl+C)"), self.copy, True)]
            if owner != COMMON:
                out += [(_("No files"), lambda: self.act(
                    [("add_files", dict(session=owner, kind=kind, paths=[], mode="replace"))],
                    _("{kind} of “{s}”: none").format(kind=_(KIND_TITLE[kind]), s=self.session_name(owner)),
                    undo_link()), True, _("This session takes none of these files, not even Common's")),
                    (_("To Common"), lambda: self.back_to_common([c]), mp.own_list(owner, kind) is not None,
                     common_tip())]
            out += [(_("Empty (Del)"), lambda: self.empty_boxes([c]), True)]
        elif k in ("setting", "cal"):
            out += [(_("To Common"), lambda: self.back_to_common([c]), c.id[1] != COMMON and self.is_own(c),
                     common_tip())]
            if k == "setting" and c.id[1] != COMMON and self.block_of(c):
                out += [(_("Variant here"), lambda: self.act([("add_variant", dict(block=self.block_of(c)))],
                                                            _("Variant added"), undo_link()), True)]
        return out

    def context_menu(self, c, gpos):
        menu = QMenu(self)
        menu.setToolTipsVisible(True)
        for ent in self.menu_entries(c):
            if ent is None:
                menu.addSeparator()
                continue
            txt, cb, ok = ent[:3]
            a = menu.addAction(txt)
            a.setEnabled(bool(ok))
            if len(ent) > 3:
                a.setToolTip(ent[3])
            a.triggered.connect(lambda _c=False, cb=cb: cb())
        if menu.actions():
            menu.exec(gpos)
