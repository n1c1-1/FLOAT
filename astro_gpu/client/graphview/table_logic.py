"""The table's reading of the project: stacks, boxes, texts, counts and warnings. Qt-free; it only READS the core
(through the bridge's Doc) and the background listing (table_reader.Reader); it keeps only view sets (warnings left,
stacks folded in the table).

Nothing is read from a file but its name, size and date of change (the listing): no header value, no measured
nature, no date of night. The texts that would need them (exposure, gain, temperature, the darks library's matching,
durations) are not shown.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ...graphcore.model import FILE_KINDS
from ...graphcore.operations import COMMON
from ...graphcore.rules import defaults
from ..i18n import _, maybe, plural

KINDS = FILE_KINDS          # lights darks flats bias flat_darks
#: the column title of a file kind (English; shown through _)
KIND_TITLE = {"lights": "Lights", "darks": "Darks", "flats": "Flats", "bias": "Bias", "flat_darks": "Flat darks"}
#: a count of a file kind: (one, many) for plural()
COUNT = {"lights": ("{n} light", "{n} lights"), "darks": ("{n} dark", "{n} darks"), "flats": ("{n} flat", "{n} flats"),
         "bias": ("{n} bias", "{n} biases"), "flat_darks": ("{n} flat dark", "{n} flat darks")}
#: the short form of a count, for a narrow box (the number is never cut)
SHORT = {"lights": "{n} lt.", "darks": "{n} dk", "flats": "{n} fl.", "bias": "{n} bs.", "flat_darks": "{n} fdk"}
#: the settings columns of a stack: column -> block kind
COLS_BLOCK = {"reg": "registration", "sel": "selection", "stk": "stacking"}
BLOCK_TITLE = {"registration": "Registration", "selection": "Selection", "stacking": "Stacking",
               "calibration": "Calibration"}
FOLDED = "folded:"           # the owner of a folded stack's boxes: "folded:<row id>"


def count_text(n: int, kind: str) -> str:
    return plural(n, *COUNT[kind])


def short_count(n: int, kind: str) -> str:
    return _(SHORT[kind]).format(n=n)


def settings_text(kind: str, eff: dict, short=False) -> str:
    """Text of a settings cell from the core's effective settings: the same words as the block's summary in the graph
    (rules.summary_msg); `short` = the value only (a cell that follows Common in a narrow column)."""
    if kind == "calibration":
        changed = [k for k, v in eff.items() if v != defaults(kind).get(k)]
        if not changed:
            return _("standard")
        if changed == ["Darks scaling"]:
            return _("adjusted") if short else _("darks adjusted")
        if short:
            return _("{n} adj.").format(n=len(changed))
        return plural(len(changed), "{n} setting adjusted", "{n} settings adjusted")
    if kind == "registration":
        v = maybe(str(eff.get("Reference image", "")))
        return v if short else _("reference frame: {v}").format(v=v)
    if kind == "selection":
        v = maybe(str(eff.get("Frame selection", "")))
        return v.split(" · ")[0] if short else v
    thr = str(eff.get("Rejection threshold", ""))
    if eff.get("Method") == "Drizzle":
        m = _("drizzle {scale}").format(scale=maybe(str(eff.get("Drizzle scale", ""))))
    else:
        m = _("classic")
    if short:
        return f"{m} {thr.replace(' ', '')}"
    return _("{m} · rejection at {thr}").format(m=m, thr=thr)


@dataclass
class FileList:
    """What the listing knows of one list of entries (a box's)."""
    sources: tuple
    n: int = 0
    complete: bool = False


@dataclass
class Box:
    """What a file cell shows."""
    owner: str
    kind: str
    mode: str = "empty"            # empty | own | inherited | none | greyed | folded
    l1: str = ""
    l2: str = ""
    n: int = 0
    source: str = "none"
    bar: bool = False
    reading: str = ""              # "Reading…" while listing
    l1s: str = ""                  # a shorter line 1
    l2s: str = ""                  # a shorter line 2
    progress: float | None = None
    warn: object = None            # the first fresh warning, shown as the amber line under the row
    corner: bool = False           # a warning already left
    tip: str = ""
    job: object = None


@dataclass
class Warn:
    key: tuple
    owner: str
    kind: str
    text: str
    button: str = ""               # English label of the action ("" = only « Leave »)
    steps: list = field(default_factory=list)
    toast: str = ""
    stack: str = ""


class TableModel:
    def __init__(self, bridge, reader):
        self.bridge, self.reader = bridge, reader
        self.left: set = set()             # warnings left (« Leave »)
        self.folded: set = set()           # stacks folded to one line (the table's view state: the core folds all)
        self._c: dict = {}

    # ------------------------------------------------------------------ the project
    @property
    def doc(self):
        return self.bridge.doc

    @property
    def p(self):
        return self.bridge.doc.graph

    def invalidate(self):
        self._c.clear()

    def _memo(self, key, f):
        if key not in self._c:
            self._c[key] = f()
        return self._c[key]

    def stacks(self):
        def f():
            out = []
            ref = self.p.reference
            for r in self.p.rows:
                if not r.nights:
                    continue
                out.append({"id": r.id, "name": r.name, "ref": r.id == ref, "folded": r.id in self.folded, "row": r,
                            "badge": self.doc.row_badge(r.id),
                            "sessions": [{"sid": s, "cid": c, "name": self.p.blocks[s].name or self.doc.title(s)}
                                         for s, c in r.nights]})
            return out
        return self._memo(("stacks",), f)

    def variants(self, rid):
        """Variants of stack `rid` (rows without sessions that fork from it, directly or through another variant), in
        rows order: [{"id", "name", "text", "block"}]. Read only: a variant is edited in the graph."""
        rows = {r.id: r for r in self.p.rows}

        def root(r):
            seen = set()
            while r is not None and not r.nights and r.fork and r.id not in seen:
                seen.add(r.id)
                r = rows.get(r.fork[0])
            return r.id if r is not None and r.nights else None

        out = []
        for r in self.p.rows:
            if r.nights or not r.fork or not r.blocks or root(r) != rid:
                continue
            parts = []
            for b in r.blocks:
                k = self.p.blocks[b].kind
                if k in BLOCK_TITLE and k != "calibration":
                    parts.append(f"{_(BLOCK_TITLE[k])} {settings_text(k, self.doc.effective(b))}")
            txt = " · ".join(parts) or self.doc.summary(r.blocks[0])
            out.append({"id": r.id, "name": r.name, "text": txt, "block": r.blocks[0]})
        return out

    def stack_of(self, sid):
        for pl in self.stacks():
            if any(s["sid"] == sid for s in pl["sessions"]):
                return pl
        return None

    def stack_by_id(self, rid):
        return next((pl for pl in self.stacks() if pl["id"] == rid), None)

    def stack_block(self, rid, kind):
        r = next((r for r in self.p.rows if r.id == rid), None)
        if r is None:
            return None
        return next((b for b in r.blocks if self.p.blocks[b].kind == kind), None)     # Selection comes and goes

    def calibration_of(self, sid):
        return next((s["cid"] for pl in self.stacks() for s in pl["sessions"] if s["sid"] == sid), None)

    def own_list(self, sid, kind):
        b = self.p.blocks.get(sid)
        f = b.files if b is not None else None
        return None if f is None or kind not in f else list(f[kind])

    def common_list(self, kind):
        return list(self.p.common["files"].get(kind, []))

    def optional_in_use(self):
        """Optional columns in use (Bias is always shown; only Flat darks waits behind « + columns »)."""
        out = []
        for k in ("flat_darks",):
            if self.common_list(k) or any(self.own_list(s["sid"], k) for pl in self.stacks() for s in pl["sessions"]):
                out.append(k)
        return out

    def session_name(self, sid):
        b = self.p.blocks.get(sid)
        return (b.name if b is not None else None) or sid

    # ------------------------------------------------------------------ lists
    def file_list(self, sources) -> FileList:
        t = tuple(sources)

        def f():
            n, ok = self.reader.count(t)
            fs = self.reader.expand(t)
            if fs is not None:
                n, ok = len(fs), True
            return FileList(t, n, ok)
        return self._memo(("list", t), f)

    def used(self, sid, kind):
        """(source, FileList) of a session's kind: its own list, else Common's (each session takes the whole of
        Common's list: matching a library by exposure needs the headers, which are not read)."""
        own = self.own_list(sid, kind)
        if own is not None:
            return "own", self.file_list(own)
        com = self.common_list(kind)
        if not com or kind == "lights":
            return "none", self.file_list(())
        return "common", self.file_list(com)

    # ------------------------------------------------------------------ a box
    def box(self, owner, kind) -> Box:
        return self._memo(("box", owner, kind), lambda: self._box(owner, kind))

    def _box(self, owner, kind) -> Box:
        b = Box(owner, kind)
        if owner.startswith(FOLDED):
            pl = self.stack_by_id(owner[len(FOLDED):])
            b.mode = "folded"
            b.n = sum(self.file_list(self.own_list(s["sid"], kind) or ()).n for s in (pl["sessions"] if pl else []))
            return b
        if owner == COMMON and kind == "lights":
            b.mode = "greyed"
            return b
        com = self.common_list(kind)
        if owner == COMMON:
            src, L = ("own" if com else "none"), self.file_list(com)
            own_flag = bool(com)
        else:
            src, L = self.used(owner, kind)
            own_flag = src == "own"
        b.source = src
        b.n = L.n
        b.job = self.reader.job((owner, kind)) if own_flag else self.reader.job((COMMON, kind))
        if src == "none":
            b.mode = "empty"
            self._box_warning(b, owner, kind)
            return b
        if src == "own" and not L.sources:
            b.mode = "none"
            b.l1 = _("none")
            b.bar = bool(com) and owner != COMMON
            self._box_warning(b, owner, kind)
            return b
        if src == "own":
            b.mode = "own"
            b.bar = owner != COMMON and bool(com)
            if not L.complete:
                b.l1 = _("Reading… {n}").format(n=L.n)
            else:
                b.l1 = count_text(L.n, kind)
                b.l1s = short_count(L.n, kind)
        else:
            b.mode = "inherited"
            b.l1 = _("↑ common · {n}").format(n=L.n)
            b.l1s = b.l1
        tr = b.job
        if tr is not None and tr.active:
            b.progress = 0.0
            b.reading = _("Reading…")
        self._box_warning(b, owner, kind)
        return b

    def _box_warning(self, b, owner, kind):
        if owner != COMMON and not owner.startswith(FOLDED):
            ws = self.box_warnings(owner, kind)
            fresh = [w for w in ws if w.key not in self.left]
            b.warn = fresh[0] if fresh else None
            b.corner = bool(ws) and not fresh
            if ws:
                b.tip = "\n".join(w.text for w in ws)

    # ------------------------------------------------------------------ warnings
    def box_warnings(self, sid, kind):
        return [w for w in self.session_warnings(sid) if w.kind == kind]

    def session_warnings(self, sid):
        return self._memo(("ws", sid), lambda: self._session_warnings(sid))

    def _session_warnings(self, sid):
        out = []
        # unreadable files (empty or gone): from the listing
        for kind in KINDS:
            own = self.own_list(sid, kind)
            if own:
                fs = self.reader.expand(tuple(own))
                if fs:
                    bad = [f for f in fs if f.size == 0]
                    if bad:
                        out.append(Warn(("unreadable", sid, kind), sid, kind,
                                        plural(len(bad), "{n} file unreadable", "{n} files unreadable"),
                                        "Remove them", [("add_files", dict(session=sid, kind=kind, mode="replace",
                                                                          paths=[f.path for f in fs if f.size]))],
                                        plural(len(bad), "{n} unreadable file removed", "{n} unreadable files removed")))
        # what the model alone decides (the core's warnings: flats missing, a path in two boxes)
        flagged = set()
        for s, kind, text, fix in self.doc.warnings(sid):
            if s != sid:
                continue
            flagged.add(kind)
            out.append(Warn(("core", sid, kind, text), sid, kind, text, "Remove" if fix else "",
                            [fix] if fix else [], _("Removed") if fix else ""))
        # a file listed in two boxes (a folder and a file inside it): from the listing
        for kind, (other, okind) in self._doubles().get(sid, {}).items():
            if kind in flagged:
                continue
            out.append(Warn(("double", sid, kind), sid, kind,
                            _("Also in {kind} of “{t}”").format(kind=_(KIND_TITLE[okind]), t=other)))
        return out

    def _doubles(self):
        def f():
            where, out = {}, {}
            for pl in self.stacks():
                for s in pl["sessions"]:
                    for kind in KINDS:
                        own = self.own_list(s["sid"], kind)
                        if not own:
                            continue
                        for x in (self.reader.expand(tuple(own)) or []):
                            if x.path in where and where[x.path][0] != (s["sid"], kind):
                                (osid, okind), name = where[x.path]
                                if osid != s["sid"] or okind != kind:
                                    out.setdefault(s["sid"], {}).setdefault(kind, (name, okind))
                            else:
                                where.setdefault(x.path, ((s["sid"], kind), s["name"]))
            return out
        return self._memo(("doubles",), f)

    def all_warnings(self):
        out = []
        for pl in self.stacks():
            for s in pl["sessions"]:
                out += self.session_warnings(s["sid"])
        return out

    def open_warnings(self):
        return [w for w in self.all_warnings() if w.key not in self.left]

    # ------------------------------------------------------------------ stack totals, rows
    def totals(self, pl):
        """(sessions, lights listed, warnings) of a stack."""
        def f():
            ns, nb = len(pl["sessions"]), 0
            for s in pl["sessions"]:
                own = self.own_list(s["sid"], "lights")
                if own:
                    nb += self.file_list(own).n
            w = sum(len(self.session_warnings(s["sid"])) for s in pl["sessions"])
            return ns, nb, w
        return self._memo(("tot", pl["id"]), f)

    def has_difference(self, sid, pl):
        """Row kept by « Only differences »: an own value, or a warning."""
        s = self.p.blocks[sid]
        if s.files:
            return True
        cid = next((c for ss, c in pl["row"].nights if ss == sid), None)
        if cid and self.doc.own_keys(cid):
            return True
        if any(self.doc.own_keys(b) for b in pl["row"].blocks if self.p.blocks[b].kind in COLS_BLOCK.values()):
            return True
        return bool(self.session_warnings(sid))

    def readings(self):
        return self.reader.active_jobs()
