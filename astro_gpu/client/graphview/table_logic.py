"""The table's reading of the project: stacks, boxes, texts, counts and warnings. It only READS the core (through
the bridge's Doc), the background listing (table_reader.Reader) and, when the document has a project model, what the
frames' headers say as the current software reads them (table_headers.HeaderReadings: exposure, ISO or gain,
temperature, the nights' dates, the calibration frames' exposures, the darks' matching and the lights of several
exposures, by the Calibration's own rules); it keeps only view sets (warnings left, stacks folded in the table).

Nothing is deduced from a file's or a folder's name. Without a project model (a graph held in memory) the header texts
are not shown.
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
#: a count of a file kind: (one, many) for plural(); "bias" has one English form, the catalogue its French two
#: ("{n} bias#one": "1 offset", "3 offsets")
COUNT = {"lights": ("{n} light", "{n} lights"), "darks": ("{n} dark", "{n} darks"), "flats": ("{n} flat", "{n} flats"),
         "bias": ("{n} bias", "{n} bias"), "flat_darks": ("{n} flat dark", "{n} flat darks")}
#: the short form of a count, for a narrow box (the number is never cut)
SHORT = {"lights": "{n} lt.", "darks": "{n} dk", "flats": "{n} fl.", "bias": "{n} bs.", "flat_darks": "{n} fdk"}
#: the settings columns of a stack: column -> block kind
COLS_BLOCK = {"reg": "registration", "sel": "selection", "stk": "stacking"}
BLOCK_TITLE = {"registration": "Registration", "selection": "Selection", "stacking": "Stacking",
               "calibration": "Calibration"}
FOLDED = "folded:"           # the owner of a folded stack's boxes: "folded:<row id>"


def exposure(t) -> str:
    """An exposure as the current views write it ("300 s", "1/20 s"); "" when unknown."""
    if t is None:
        return ""
    from ..frames import fexp
    return fexp(t)


def light_line(row) -> str:
    """What the lights' headers say: "300 s · Gain 90 · −10 °C" (only what is recorded)."""
    from ..frames import iso_word
    parts = [exposure(row.exposure)]
    if row.iso:
        parts.append(f"{iso_word(row.gain)} {int(row.iso)}")
    if row.temperature is not None:
        parts.append(f"{round(row.temperature):d} °C".replace("-", "\u2212"))
    return " · ".join(x for x in parts if x)


def calib_line(c: dict) -> str:
    """What a session's calibration frames of a kind say (their middle one, as the Calibration reads them):
    "300 s · Gain 90", "master of 20"."""
    from ..frames import iso_word
    parts = [exposure(c.get("exposure"))]
    if c.get("iso"):
        parts.append(f"{iso_word(c.get('gain'))} {int(c['iso'])}")
    if c.get("stacked"):
        parts.append(_("master of {n}").format(n=int(c["stacked"])))
    return " · ".join(x for x in parts if x)


def nights_line(row) -> str:
    """The night(s) of a session's lights from their own dates: "17 Jun 2025", "8 nights · 28/06 to 21/07"."""
    from ..frames import day_text, night_start, nights_text
    nights = list(getattr(row, "nights", None) or [])
    if len(nights) >= 2:
        return nights_text(nights)
    d = night_start(row.first) if row.first else None
    return day_text(d.isoformat()) if d is not None else ""


def seconds_of(row):
    """The time a session's lights add up to (s): their count times their one exposure; None when it is not recorded
    or when they are of several exposures (the reading samples their headers, it does not count each)."""
    from ...stages.calibrate import mixed_exposures
    if row is None or not row.exposure or not row.lights or mixed_exposures(row.light_keys):
        return None
    return float(row.lights) * float(row.exposure)


def duration(sec: float) -> str:
    """A duration as the current views write it ("2 h 30", "45 min")."""
    from ..i18n import hm
    return hm(sec)


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
        self.headers = None                # table_headers.HeaderReadings (None: no header reading)
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
        elif owner != COMMON:
            self._header_line(b, owner, kind)
        self._box_warning(b, owner, kind)
        return b

    # ------------------------------------------------------------------ what the headers say (a project model only)
    def header_row(self, sid):
        """The header reading of a session (services/inspect.SessionRow), or None (being read, or none)."""
        h = self.headers
        return h.row(sid) if h is not None else None

    def headers_pending(self, sid=None) -> bool:
        h = self.headers
        return bool(h is not None and h.pending(sid))

    def _header_line(self, b, sid, kind):
        """Line 2 of a session's box: what its frames' headers say (the lights' exposure, ISO or gain and temperature;
        a calibration kind's exposure and ISO or gain), or « reading… » while they are read."""
        if b.n == 0:
            return
        if self.headers_pending(sid):
            b.l2 = _("reading…")
            return
        row = self.header_row(sid)
        if row is None:
            return
        if kind == "lights":
            b.l2, b.l2s = light_line(row), exposure(row.exposure)
        elif kind in row.calib:
            b.l2, b.l2s = calib_line(row.calib[kind]), exposure(row.calib[kind].get("exposure"))

    def date_text(self, sid) -> str:
        """The night(s) of a session's lights, from their own dates (never a file's date on disk, nor its name), and
        the time its lights add up to: "17 Jun 2025 · 2 h 30"; "" while unknown."""
        row = self.header_row(sid)
        if row is None:
            return _("reading…") if self.headers_pending(sid) and self.own_list(sid, "lights") else ""
        parts = [nights_line(row)]
        sec = seconds_of(row)
        if sec:
            parts.append(duration(sec))
        return " · ".join(x for x in parts if x)

    def seconds(self, pl):
        """The time the lights of a stack add up to (s), or None when one of its sessions' is not known."""
        def f():
            total = 0.0
            for s in pl["sessions"]:
                if not self.own_list(s["sid"], "lights"):
                    continue
                row = self.header_row(s["sid"])
                sec = seconds_of(row) if row is not None else None
                if sec is None:
                    return None
                total += sec
            return total or None
        return self._memo(("sec", pl["id"]), f)

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
        out += self._header_warnings(sid)
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

    def _header_warnings(self, sid):
        """What the headers contradict, as the Sources page says it today: lights of several exposures; darks of
        another exposure or gain than the lights; lights of an exposure no dark matches (the Calibration's own rule)."""
        row = self.header_row(sid)
        if row is None:
            return []
        from ..frames import fexp, iso_word, mixed_exposures_text
        from ..services.inspect import calib_checks
        out = []
        try:
            bad = calib_checks(row, self.headers.camera if self.headers is not None else {})
        except Exception:  # noqa: BLE001 - a reading the checks cannot use: nothing said
            bad = {}
        if bad.get("mixed_exposures"):
            out.append(Warn(("mixed", sid, "lights"), sid, "lights", mixed_exposures_text(bad["mixed_exposures"])))
        dk = row.calib.get("darks")
        by = bad.get("darks_by_exposure")
        if dk and not (by and by[0]["several"]):
            de, le, dg, lg = dk.get("exposure"), row.exposure, dk.get("iso"), row.iso
            same = (de is None or le is None or abs(de - le) < 1e-6) and (dg is None or lg is None or dg == lg)
            if not same:
                word = iso_word(row.gain or dk.get("gain"))
                t = (_("Darks: exposure {exp}, different from the lights; {word} not recorded in the darks.")
                     if dg is None else _("Darks: exposure {exp}, {word} {iso}, different from the lights."))
                out.append(Warn(("darks", sid, "darks"), sid, "darks",
                                t.format(exp=fexp(de), word=word, iso=int(dg or 0))))
        for g in by or []:
            if g["how"] in ("same", "scaled"):
                continue
            e = fexp(g["lights"]) if g["lights"] else _("an exposure not recorded")
            if g["others"]:
                t = _("Lights of {exp}: no dark of {exp}, calibrated without dark (no bias to scale those of "
                      "{others}).").format(exp=e, others=", ".join(map(fexp, g["others"])))
            else:
                t = _("Lights of {exp}: no dark of their exposure and gain, calibrated without dark.").format(exp=e)
            out.append(Warn(("nodark", sid, g["lights"]), sid, "darks", t))
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
