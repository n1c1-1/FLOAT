"""The graph's gestures: files (a night's, the Common ones, a block's own calibration) and sessions by their id."""
import copy
from .model import FILE_KINDS, Row, new_id
from . import rules
from .msg import M
from .rules import LABEL, title_msg
from .ops_base import (COMMON, Refusal, Out, _blk, _names, free_name, session_name, _night, _after_subtree,
                       _auto_align, _kind, _dedup, _file_name)
from .ops_rows import _night_row, op_remove_night, _not_shared, _eff_list, _carry_excluded


# ---------------------------------------------------------------------------------------------------- files, Commun
def _session(q, s):
    b = _blk(q, s)
    if b.kind != "session":
        raise Refusal("files go on a Session")
    return b


def op_add_files(q, session, kind, paths, mode="add"):
    s = _session(q, session)
    _kind(kind)
    paths = list(paths)
    if not paths and mode != "replace":
        raise Refusal("no file")
    if mode == "replace":
        s.files[kind] = _dedup(paths)
    else:
        base = _eff_list(q, session, kind)
        seen = set(base)
        s.files[kind] = base + [x for x in _dedup(paths) if x not in seen]
    pair = (("{n} {one} replaces those of “{t}”", "{n} {kind} replace those of “{t}”") if mode == "replace" else
            ("{n} {one} added to “{t}”", "{n} {kind} added to “{t}”"))
    return Out(rules.files_msg(len(paths), kind, *pair, t=title_msg(q, session)), session)


def op_remove_files(q, session, kind, paths):
    s = _session(q, session)
    _kind(kind)
    rm = set(paths)
    # emptied, the list stays the session's own (none of that kind), never Common's in silence: following Common is
    # a gesture of its own (op_reset_own_files); a list equal to Common's goes at the normalisation
    s.files[kind] = [x for x in _eff_list(q, session, kind) if x not in rm]
    return Out(M("{kind} of “{t}” removed · files kept", kind=rules.FILE_LABEL[kind], t=title_msg(q, session)), session)


def op_use_for_all_nights(q, session, kind):
    _session(q, session)
    _kind(kind)
    r, _n = _night_row(q, session)
    lst = _eff_list(q, session, kind)
    for s, _c in r.nights:
        if s != session:
            q.blocks[s].files[kind] = list(lst)
    return Out(M("{kind} for all the sessions of “{_row}”", kind=rules.FILE_LABEL[kind], _row=r.name), session)


def op_set_common_files(q, kind, paths, mode="add"):
    _kind(kind)
    if kind == "lights":
        raise Refusal("lights go on a session, not on Common")
    cur = q.common["files"].get(kind, [])
    if mode == "replace":
        new = _dedup(paths)
    else:
        new = cur + [x for x in _dedup(paths) if x not in cur]
    q.common["files"][kind] = new
    return Out(M("Common {kind}: {n}", kind=rules.FILE_LABEL[kind], n=len(new)))


def op_set_common_setting(q, kind, key, value):
    if rules.setting_spec(kind, key) is None:
        raise Refusal("unknown setting")
    if (kind, key) in rules.NO_COMMON:
        raise Refusal(rules.NO_COMMON[(kind, key)])
    st = q.common["settings"].setdefault(kind, {})
    if rules.defaults(kind).get(key) == value:
        st.pop(key, None)
    else:
        st[key] = value
    n = len(rules.followers(q, kind)[0])
    return Out(M("Common {kind} changed: {n} to recompute", kind=LABEL.get(kind, kind), n=n))


def op_reset_own(q, block, keys=None):
    b = _blk(q, block)
    b.settings = {} if keys is None else {k: v for k, v in b.settings.items() if k not in keys}
    return Out(M("“{t}” follows Common again", t=title_msg(q, block)), block)


def op_reset_own_files(q, session, kind):
    s = _session(q, session)
    _kind(kind)
    s.files.pop(kind, None)
    return Out(M("“{t}” follows Common again", t=title_msg(q, session)), session)


def _list_of(q, s, kind):
    if s == COMMON:
        return list(q.common["files"].get(kind, []))
    _session(q, s)
    return _eff_list(q, s, kind)


def _set_list(q, s, kind, lst):
    if s == COMMON:
        if kind == "lights":
            raise Refusal("lights go on a session, not on Common")
        q.common["files"][kind] = lst
    else:
        q.blocks[s].files[kind] = lst                 # emptied: its own empty list (op_remove_files)


def op_move_files(q, src_session, src_kind, dst_session, dst_kind, paths=None, copy=False):
    _kind(src_kind)
    _kind(dst_kind)
    src = _list_of(q, src_session, src_kind)
    paths = src if paths is None else _dedup(paths)
    if not paths or any(x not in src for x in paths):
        raise Refusal("these files are not in the box")
    if (dst_session, dst_kind) == (src_session, src_kind):
        return Out(focus=None if dst_session == COMMON else dst_session)
    if not copy:
        rm = set(paths)
        _set_list(q, src_session, src_kind, [x for x in src if x not in rm])
        if (dst_session, dst_kind) == (COMMON, src_kind) and not _list_of(q, src_session, src_kind):
            q.blocks[src_session].files.pop(src_kind, None)     # all its files now Common's: it follows Common
    dst = _list_of(q, dst_session, dst_kind)
    _set_list(q, dst_session, dst_kind, dst + [x for x in paths if x not in dst])
    where = "common" if dst_session == COMMON else M("“{t}”", t=title_msg(q, dst_session))
    pair = (("{n} {one} copied to {where}", "{n} {kind} copied to {where}") if copy else
            ("{n} {one} moved to {where}", "{n} {kind} moved to {where}"))
    return Out(rules.files_msg(len(paths), dst_kind, *pair, where=where),
               None if dst_session == COMMON else dst_session)


def op_split_by(q, session, groups):
    s = _session(q, session)
    r, n = _night_row(q, session)
    if not groups:
        raise Refusal("nothing to split")
    _not_shared(q, session)
    own = {k: list(v) for k, v in s.files.items()}
    allp = [x for g in groups for lst in g.get("files", {}).values() for x in lst]
    if not allp or any(not any(x in v for v in own.values()) for x in allp):
        raise Refusal("files absent from the session")
    rm = set(allp)
    common = {k for g in groups if g.get("into") == "common" for k in g.get("files", {})}
    for k in list(s.files):
        s.files[k] = [x for x in s.files[k] if x not in rm]       # emptied: its own empty list (op_remove_files)
        if not s.files[k] and k in common:
            del s.files[k]                                         # ...unless its files went to Common: followed
    calib = {k: list(v) for k, v in s.files.items() if k != "lights" and v}
    at, made, rows_at = r.nights.index(n) + 1, 0, _after_subtree(q, r.id)
    focus = session
    for g in groups:
        into, files = g.get("into", "here"), {k: _dedup(v) for k, v in g.get("files", {}).items()}
        for k in files:
            _kind(k)
        if into == "here":
            for k, v in files.items():
                base = s.files.get(k, [])
                s.files[k] = base + [x for x in v if x not in base]
        elif into == "common":
            for k, v in files.items():
                _set_list(q, COMMON, k, _list_of(q, COMMON, k) + [x for x in v if x not in _list_of(q, COMMON, k)])
        elif into in ("night", "row"):
            fl = dict(calib)
            fl.update(files)
            name = (g.get("name") or "").strip() or session_name(q)
            if name in _names(q):
                name = free_name(q, name)
            nn = _night(q, name, {k: fl[k] for k in FILE_KINDS if k in fl}, n[1] is not None)   # as the night it
            if nn[1] is not None:                                                               # leaves
                q.blocks[nn[1]].settings = dict(q.blocks[n[1]].settings)
                q.blocks[nn[1]].off = q.blocks[n[1]].off if into == "night" else False
            if into == "night":
                r.nights.insert(at, nn)
                at += 1
            else:
                nr = Row(new_id(q, "r"), name, [], [nn])            # its own stack, being built: its blocks to place
                _auto_align(q, nr)
                q.rows.insert(rows_at, nr)
                rows_at += 1
            _carry_excluded(q, s.sid, own.get("lights", []), files.get("lights", []), s.files.get("lights", []),
                            nn[0])
            made += 1
            focus = nn[0]
        else:
            raise Refusal("unknown destination")
    return Out(M("Split into {n} sessions", n=made + 1) if made else "files distributed", focus)


# ---------------------------------------------------------------------------------------------------- own calibration
def _calib_block(q, block):
    b = _blk(q, block, ("calibration",))
    if b.sid is None:
        raise Refusal("block not found")
    return b


def op_set_own_calibration(q, block, kind, paths, mode="add"):
    """A night's Calibration with frames of its own (a Calibration placed in parallel): that kind stops following
    its Session; the others still do."""
    b = _calib_block(q, block)
    _kind(kind)
    if kind == "lights":
        raise Refusal("lights go on a session, not on Common")
    paths = _dedup(list(paths))
    if not paths and mode != "replace":
        raise Refusal("no file")
    own = copy.deepcopy(b.own) if b.own else {"fork": 1 + max([c.own.get("fork") or 0 for c in q.blocks.values()
                                                                if c.kind == "calibration" and c.own] + [0])}
    cur = own.get(kind, [])
    own[kind] = paths if mode == "replace" else cur + [x for x in paths if x not in cur]
    b.own = own
    return Out(M("{kind}: own files", kind=rules.FILE_LABEL[kind]), block)


def op_follow_session(q, block, kind):
    """A kind of a Calibration with its own frames given back to its Session."""
    b = _calib_block(q, block)
    _kind(kind)
    if not b.own or kind not in b.own:
        raise Refusal("it already follows its session")
    b.own = {k: v for k, v in b.own.items() if k != kind}
    return Out(M("{kind} of the session again", kind=rules.FILE_LABEL[kind]), block)


def op_remove_own_calibration(q, block):
    """The Calibration calibrates its night as its Session says again."""
    b = _calib_block(q, block)
    if not b.own:
        raise Refusal("it already follows its session")
    b.own = None
    return Out("calibrated as its session says", block)


def op_set_night_filter(q, night, filter):  # noqa: A002
    """The filter of a night's lights (stored; nothing is read from the files)."""
    b = _session(q, night)
    _night_row(q, night)
    e = next((n for n in q.nights if n["sid"] == b.sid), None)
    if e is None:
        raise Refusal("session not found")
    e["filter"] = str(filter or "")
    return Out(M("filter: {_f}", _f=e["filter"]), night)


# ---------------------------------------------------------------------------------------------------- step 4c
# What the project model edits beside the drawing: a session by its id (drawn by a stack or not), an Image block's
# file, a tap's name, a workflow's own pauses, options, settings, a row's chain replaced as a whole, a new branch.
def _entry(q, sid):
    """The project's entry of session `sid` (Project.nights)."""
    e = next((n for n in q.nights if not isinstance(sid, bool) and n["sid"] == sid), None)
    if e is None:
        raise Refusal("session not found")
    return e


def _drawn(q, sid):
    """The Session blocks drawing session `sid` (none for a night no stack draws)."""
    return [b for b in q.blocks.values() if b.kind == "session" and b.sid == sid]


def op_set_session_files(q, sid, kind, paths):
    """A session's list of files of a kind, replaced (its Session blocks', or the entry of a night no stack draws)."""
    e = _entry(q, sid)
    _kind(kind)
    if isinstance(paths, str):
        raise Refusal("no file")
    lst = _dedup([str(x) for x in paths])
    drawn = _drawn(q, sid)
    for b in drawn:
        b.files[kind] = list(lst)
    e.setdefault("files", {})[kind] = list(lst)
    return Out(M("{kind} of “{_name}”: {n}", kind=rules.FILE_LABEL[kind], _name=e.get("name", ""), n=len(lst)),
               drawn[0].id if drawn else None)


def op_rename_session(q, sid, name):
    e = _entry(q, sid)
    name = (name or "").strip()
    if not name:
        raise Refusal("a name cannot be empty")
    e["name"] = name
    drawn = _drawn(q, sid)
    for b in drawn:
        b.name = name
    return Out(M("session renamed “{_name}”", _name=name), drawn[0].id if drawn else None)


def op_set_session_filter(q, sid, filter):  # noqa: A002
    """The filter of a session's lights (stored; nothing is read from the files)."""
    e = _entry(q, sid)
    e["filter"] = str(filter or "")
    drawn = _drawn(q, sid)
    return Out(M("filter: {_f}", _f=e["filter"]), drawn[0].id if drawn else None)


def op_set_left_out(q, sid, left_out):
    """A session's calibration frames left out by hand, {kind: [paths]} as a whole (out of its lists, still shown,
    to be taken back)."""
    e = _entry(q, sid)
    if not isinstance(left_out, dict):
        raise Refusal("unknown kind of files")
    for k, v in left_out.items():
        _kind(k)
        if isinstance(v, str):
            raise Refusal("no file")
    e["left_out"] = {k: [str(x) for x in v] for k, v in left_out.items()}
    n = sum(len(v) for v in e["left_out"].values())
    drawn = _drawn(q, sid)
    return Out(M("frames left out of “{_name}”: {n}", _name=e.get("name", ""), n=n), drawn[0].id if drawn else None)


def op_remove_session(q, sid):
    """A session out of the project (its files are never touched): its night out of every stack drawing it (a stack
    of that night alone goes with it), its frames excluded by hand, its own settings in every workflow."""
    e = _entry(q, sid)
    for b in [b.id for b in _drawn(q, sid)]:
        if b in q.blocks:
            op_remove_night(q, b)
    q.nights = [n for n in q.nights if n["sid"] != sid]
    q.excluded = [x for x in q.excluded if x[0] != sid]
    mark = f"@{sid}"
    for f in list(q.rows) + list(q.taps):
        if f.extra and any(k.endswith(mark) for k in f.extra):
            f.extra = {k: v for k, v in f.extra.items() if not k.endswith(mark)}
        if f.sessions is not None and sid in f.sessions:
            f.sessions = [x for x in f.sessions if x != sid]
    for r in q.rows:
        if r.calib and any(k.endswith(mark) for k in r.calib):
            r.calib = {k: v for k, v in r.calib.items() if not k.endswith(mark)}
    return Out(M("“{_name}” out of the project · files kept", _name=e.get("name", "")), style="need")


def op_exclude_frames(q, sid, names, excluded=True):
    """Lights excluded by hand from every stack (or taken back): Project.excluded, by file name."""
    e = _entry(q, sid)
    if isinstance(names, str) or not names:
        raise Refusal("no file")
    names = _dedup([_file_name(x) for x in names])
    have = [list(x) for x in q.excluded]
    if excluded:
        q.excluded = have + [[sid, x] for x in names if [sid, x] not in have]
    else:
        gone = {(sid, x) for x in names}
        q.excluded = [x for x in have if (x[0], x[1]) not in gone]
    return Out(M("{n} lights excluded from “{_name}”" if excluded else "{n} lights taken back in “{_name}”",
                 n=len(names), _name=e.get("name", "")))


def op_exclude_files(q, session, kind, paths, excluded=True):
    """Single files of a Session left out by hand (or taken back), one step: lights excluded from every stack
    (Project.excluded, by file name); calibration frames noted as left out of the session (its night's left_out, which
    the session shows to take them back)."""
    if session == COMMON:
        raise Refusal("Common's files are left out in each session")
    s = _session(q, session)
    _kind(kind)
    if s.sid is None:
        raise Refusal("session not found")
    if isinstance(paths, str) or not paths:
        raise Refusal("no file")
    if kind == "lights":
        return op_exclude_frames(q, s.sid, paths, excluded)
    e = _entry(q, s.sid)
    paths = _dedup([str(x) for x in paths])
    lo = {k: list(v) for k, v in (e.get("left_out") or {}).items()}
    cur = [x for x in lo.get(kind, []) if x not in paths]
    lo[kind] = cur + paths if excluded else cur
    e["left_out"] = lo
    pair = (("{n} {one} left out of “{_name}”", "{n} {kind} left out of “{_name}”") if excluded else
            ("{n} {one} taken back in “{_name}”", "{n} {kind} taken back in “{_name}”"))
    return Out(rules.files_msg(len(paths), kind, *pair, _name=e.get("name", "")), session)


def excluded_files(q, session, kind):
    """The files of a Session left out by hand: lights by file name, calibration frames by path."""
    if session not in q.blocks or q.blocks[session].kind != "session" or kind not in FILE_KINDS:
        return []
    sid = q.blocks[session].sid
    if kind == "lights":
        return [x[1] for x in q.excluded if x[0] == sid]
    e = next((n for n in q.nights if n["sid"] == sid), None)
    return list(((e or {}).get("left_out") or {}).get(kind) or [])


def op_set_image_file(q, block, path=None, stretched=None):
    """The file an Image block reads, and whether it is stretched (its source as stored)."""
    b = _blk(q, block, ("image",))
    if path is None and stretched is None:
        raise Refusal("nothing to change")
    src = copy.deepcopy(b.source) if b.source is not None else {
        "path": "", "stretched": bool(rules.effective(q, block)["Stretched image"])}
    if path is not None:
        src["path"] = str(path)
    if stretched is not None:
        src["stretched"] = bool(stretched)
    b.source = src
    if path is not None:
        return Out(M("image file: {_path}", _path=_file_name(path) if path else ""), block)
    return Out("stretched image" if src["stretched"] else "linear image", block)
