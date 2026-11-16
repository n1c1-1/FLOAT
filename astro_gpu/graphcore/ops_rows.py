"""The graph's gestures: rows and nights (add, remove, move, share, merge, split, duplicate, rename, alignment)."""
import copy
from .model import FILE_KINDS, Index, JOIN_KINDS, Row, new_block, new_id, out_ports
from . import rules
from .msg import M, join, name_of, typed
from .rules import LABEL
from ..core import workflow as wfl
from .ops_base import (Refusal, Out, _flow_chain, _row, _names, free_name, session_name, _night, _copy_calibration,
                       _follow_stack, _to_post, _after_subtree, delete_rows, _auto_align, new_align,
                       _insert_after_frames, detach, _kind, _dedup, _clone_block, _copy_flow, _file_name)
from .ops_after import _not_into_whole_variant


# ---------------------------------------------------------------------------------------------------- rows, nights
def _named(q, name, default):
    name = (name or "").strip()
    if not name:
        return default
    return free_name(q, name)


def _files(files):
    files = files or {}
    for k in files:
        _kind(k)
    return {k: _dedup(files[k]) for k in FILE_KINDS if k in files}


def op_add_row(q, kind, files=None, name=None):
    if kind == "session":
        # the Session alone: a stack being built, each next block placed by the user (the engine runs a stack with
        # no block: nothing computed until one is placed)
        name = _named(q, name, session_name(q))
        r = Row(new_id(q, "r"), name, [], [_night(q, name, _files(files), calibration=False)])
        _auto_align(q, r)
        q.rows.append(r)
        return Out(M("“{_name}” added", _name=name), r.nights[0][0], new=r.nights[0][0])
    if kind == "image":
        if files:
            raise Refusal("an Image takes no files here")
        name = _named(q, name, _image_name(q))
        b = new_block(q, "image")
        r = Row(new_id(q, "r"), name, [b.id])
        q.rows.append(r)
        _to_post(q, b.id)
        return Out(M("“{_name}” added", _name=name), b.id, new=b.id)
    raise Refusal("a row starts with a Session or an Image")


def _image_name(q):
    names, n = _names(q), 1
    while name_of("Image {n}", n=n) in names:
        n += 1
    return name_of("Image {n}", n=n)


def _first(r):
    return r.nights[0][0] if r.nights else r.blocks[0]


def op_delete_row(q, row):
    first = _first(_row(q, row))
    toast = delete_rows(q, [row])
    return Out(toast, old=None if q._astro_moved else first, style="need")


def op_add_night(q, row, files=None, name=None):
    r = _row(q, row)
    if not r.nights:
        raise Refusal("a session joins a stack of lights")
    _not_into_whole_variant(q, r)
    # a stack of one filter's nights: the night joining it takes that filter and is named after it ("Ha 2"), as
    # today's page did
    by_sid = {x["sid"]: x for x in q.nights}
    flts = {(by_sid.get(q.blocks[s].sid) or {}).get("filter", "") for s, _c in r.nights}
    flt = next(iter(flts)) if len(flts) == 1 else ""
    if not (name or "").strip() and flt:
        names = _names(q)
        k = 2
        while f"{flt} {k}" in names:
            k += 1
        name = f"{flt} {k}"
    name = (name or "").strip() or session_name(q)
    n = _night(q, name, _files(files), calibration=False)        # the Session alone, unless the engine needs more:
    if flt:
        sid = _next_sid(q)
        q.blocks[n[0]].sid = sid
        q.nights.append({"sid": sid, "name": name, "filter": flt, "files": {}, "left_out": {}})
    r.nights.append(n)
    why = _follow_stack(q, r, n)                                  # calibrated together, the stack's one Calibration
    toast = M("“{_name}” joins “{_row}”", _name=name, _row=r.name) if flt else         M("{_name} added to “{_row}”: it stacks with its sessions", _name=name, _row=r.name)
    return Out(join([toast, why]) if why else toast, n[0], new=n[0])


def _night_row(q, night):
    for r in q.rows:
        for n in r.nights:
            if n[0] == night:
                return r, n
    raise Refusal("session not found")


def _forget_night(q, sid):
    """A session taken out of the project when no stack draws it any more (its files are never touched): its entry,
    its frames left out, its own settings in the workflows' leftovers."""
    if sid is None or any(b.kind == "session" and b.sid == sid for b in q.blocks.values()):
        return
    q.nights = [n for n in q.nights if n["sid"] != sid]
    q.excluded = [x for x in q.excluded if x[0] != sid]
    for f in list(q.rows) + list(q.taps):
        if f.extra and any(k.endswith(f"@{sid}") for k in f.extra):
            f.extra = {k: v for k, v in f.extra.items() if not k.endswith(f"@{sid}")}


def op_remove_night(q, night):
    r, n = _night_row(q, night)
    sid = q.blocks[night].sid
    if len(r.nights) == 1:
        toast = delete_rows(q, [r.id])
        _forget_night(q, sid)
        return Out(toast, old=None if q._astro_moved else night, style="need")
    r.nights.remove(n)
    for b in n:
        if b is not None:
            del q.blocks[b]
    _forget_night(q, sid)
    return Out("session removed", old=night, style="need")


def _next_sid(q):
    return max([n["sid"] for n in q.nights] + [b.sid for b in q.blocks.values() if b.sid is not None] + [0]) + 1


def op_new_night(q, name=None, filter=""):  # noqa: A002
    """A session of the project that no stack draws yet (a template's, before its frames are known): its name, the
    filter of its lights (stored; nothing is read from the files)."""
    sid = _next_sid(q)
    name = (name or "").strip() or session_name(q)
    q.nights.append({"sid": sid, "name": name, "filter": str(filter or ""), "files": {}, "left_out": {}})
    return Out(M("“{_name}” added", _name=name))


def op_add_stack(q, name, nights, blocks=None, align="auto", all_nights=False, calibration=True):
    """A stack drawing sessions of the project (``nights``: their session ids, drawn in that order), its chain
    ``blocks`` ([{"kind", "method"}], None: Registration, Selection, Stacking: a template's full chain); ``align``
    "auto": on the reference (the first stack keeps its own grid), None: its own grid; ``all_nights``: it stacks every
    night of the project, those added later too; ``calibration``: its nights with their Calibration, or without."""
    entry = {n["sid"]: n for n in q.nights}
    if not nights or any(s not in entry for s in nights) or len(set(nights)) != len(nights):
        raise Refusal("session not found")
    drawn = {b.sid: b for b in q.blocks.values() if b.kind == "session" and b.sid is not None}
    pairs = []
    for sid in nights:
        e = entry[sid]
        src = drawn.get(sid)
        files = {k: list(v) for k, v in src.files.items()} if src is not None else \
            {k: list(v) for k, v in (e.get("files") or {}).items() if v}
        nn = _night(q, src.name if src is not None else (e.get("name") or name_of("Session {n}", n=sid)), files,
                    calibration)
        for x in nn:
            if x is not None:
                q.blocks[x].sid = sid
        pairs.append(nn)
    chain = []
    for n in (blocks if blocks is not None else [{"kind": k} for k in ("registration", "selection", "stacking")]):
        k = n["kind"]
        if k not in LABEL or k in ("session", "calibration", "post", "image") + JOIN_KINDS:
            raise Refusal("unknown block")
        chain.append(new_block(q, k, method=n.get("method")).id)
    r = Row(new_id(q, "r"), free_name(q, (name or "").strip() or session_name(q)), chain, pairs)
    r.all_nights = bool(all_nights)
    if align == "auto":
        _auto_align(q, r)
    elif align is not None:
        raise Refusal("unknown place")
    q.rows.append(r)
    if chain and out_ports(q.blocks[chain[-1]]):
        _to_post(q, chain[-1])
    return Out(M("“{_name}” added", _name=r.name), pairs[0][0], new=pairs[0][0])


def op_share_night(q, night, row):
    """A night drawn in another stack too (a dual-band shot stacked once per filter): one session, its files edited
    once, used by both stacks."""
    r, n = _night_row(q, night)
    t = _row(q, row)
    if not t.nights or t.fork:
        raise Refusal("a session joins a stack of lights")
    _not_into_whole_variant(q, t)
    s = q.blocks[night]
    if t is r or (s.sid is not None and any(q.blocks[x].sid == s.sid for x, _c in t.nights)):
        raise Refusal(M("“{_name}” is already in “{_row}”", _name=s.name, _row=t.name))
    nn = _night(q, s.name, {k: list(v) for k, v in s.files.items()}, n[1] is not None)
    for x in nn:
        if x is not None:
            q.blocks[x].sid = s.sid
    if nn[1] is not None:
        q.blocks[nn[1]].off = q.blocks[n[1]].off
    why = _follow_stack(q, t, nn)
    t.nights.append(nn)
    toast = M("“{_name}” also stacked in “{_row}”", _name=s.name, _row=t.name)
    return Out(join([toast, why]) if why else toast, nn[0], new=nn[0])


def _no_clash(q, t, nights):
    """A night joining stack t: t must not draw that session already."""
    have = {q.blocks[x].sid for x, _c in t.nights}
    for x, _c in nights:
        if q.blocks[x].sid is not None and q.blocks[x].sid in have:
            raise Refusal(M("“{_name}” is already in “{_row}”", _name=q.blocks[x].name, _row=t.name))


def _shared(q, night):
    sid = q.blocks[night].sid
    return sid is not None and sum(b.kind == "session" and b.sid == sid for b in q.blocks.values()) > 1


def _not_shared(q, *nights):
    for s in nights:
        if _shared(q, s):
            raise Refusal(M("“{_name}” is shared by several stacks: its lights stay together", _name=q.blocks[s].name))


def op_move_night(q, night, row=None, before=None):
    """Into stack `row` (at its end, or before its session `before`); row None = its own new row."""
    r, n = _night_row(q, night)
    name = q.blocks[night].name
    if row is None:
        if len(r.nights) == 1:
            raise Refusal("already alone on its row")
        r.nights.remove(n)
        nr = Row(new_id(q, "r"), free_name(q, name), [], [n])       # its own stack, being built: its blocks to place
        _auto_align(q, nr)
        q.rows.append(nr)
        return Out(M("“{_name}” moves to its own row “{_row}”", _name=name, _row=nr.name), night, new=night)
    t = _row(q, row)
    if not t.nights:
        raise Refusal("a session joins a stack of lights")
    if before is not None and before not in [x[0] for x in t.nights]:
        raise Refusal("session not found in this stack")
    if t is not r:
        _not_into_whole_variant(q, t)
        _no_clash(q, t, [n])
    if t is r and (before is None or before == night):
        if before is None and t.nights[-1] is not n:
            t.nights.remove(n)
            t.nights.append(n)
            return Out(M("“{_name}” moved", _name=name), night, new=night)
        return Out(focus=night)                 # dropped where it is: nothing moves, nothing said
    why = _follow_stack(q, t, n) if t is not r and t.nights else ""
    r.nights.remove(n)
    at = [x[0] for x in t.nights].index(before) if before is not None else len(t.nights)
    t.nights.insert(at, n)
    if t is r:
        return Out(M("“{_name}” moved", _name=name), night, new=night)
    toast = M("“{_name}” goes into “{_row}”", _name=name, _row=t.name)
    if why:
        toast = join([toast, why])
    if not r.nights:
        toast = join([toast, delete_rows(q, [r.id])])
    return Out(toast, night, new=night)


def _eff_list(q, s, kind):
    return rules.files_of(q, s, kind)[1]


def _exclude_in(q, night, names):
    """Lights excluded by hand (file names) in the session of night block ``night``, which is given its session id
    now when it has none yet (as op_add_night gives one)."""
    blk = q.blocks[night]
    if blk.sid is None:
        blk.sid = _next_sid(q)
        q.nights.append({"sid": blk.sid, "name": blk.name, "filter": "", "files": {}, "left_out": {}})
    have = [list(x) for x in q.excluded]
    q.excluded = have + [[blk.sid, x] for x in _dedup(names) if [blk.sid, x] not in have]


def _carry_excluded(q, sid, own, moved, kept, night):
    """Lights of session ``sid`` (``own`` before) moved to night block ``night`` (``moved``; ``kept``: the lights it
    keeps): a frame excluded by hand stays excluded where it goes (by file name, as the engine matches them); a name of
    no light (a frame of a folder) stays excluded in both."""
    if sid is None:
        return
    mv, kp, ow = ({_file_name(x) for x in v} for v in (moved, kept, own))
    names = [x[1] for x in q.excluded if x[0] == sid and (x[1] in mv or x[1] not in ow)]
    if not names:
        return
    q.excluded = [x for x in q.excluded if not (x[0] == sid and x[1] in mv and x[1] not in kp)]
    _exclude_in(q, night, names)


def op_merge_nights(q, a, b):
    ra, na = _night_row(q, a)
    rb, nb = _night_row(q, b)
    if ra is not rb or a == b:
        raise Refusal("both sessions must be in the same stack")
    for k in FILE_KINDS:
        if k in q.blocks[b].files:
            base = _eff_list(q, a, k)
            q.blocks[a].files[k] = base + [x for x in q.blocks[b].files[k] if x not in base]
    sid_b = q.blocks[b].sid
    names = [x[1] for x in q.excluded if sid_b is not None and x[0] == sid_b]
    ra.nights.remove(nb)
    for x in nb:
        if x is not None:
            del q.blocks[x]
    if names:                                            # its lights excluded by hand stay excluded in a
        _exclude_in(q, a, names)
    _forget_night(q, sid_b)                              # as a session removed
    return Out(M("“{_name}” holds both sessions", _name=q.blocks[a].name), a, old=b)


def op_split_night(q, night, lights):
    r, n = _night_row(q, night)
    _not_shared(q, night)
    own = q.blocks[night].files.get("lights", [])
    if not lights or any(x not in own for x in lights):
        raise Refusal("these lights are not in the session")
    q.blocks[night].files["lights"] = [x for x in own if x not in lights]
    files = {k: list(v) for k, v in q.blocks[night].files.items() if k != "lights"}
    files["lights"] = list(lights)
    nn = _night(q, session_name(q), files, n[1] is not None)       # as the night it leaves: with its Calibration or not
    if nn[1] is not None:
        q.blocks[nn[1]].settings = dict(q.blocks[n[1]].settings)
        q.blocks[nn[1]].off = q.blocks[n[1]].off
    r.nights.insert(r.nights.index(n) + 1, nn)
    _carry_excluded(q, q.blocks[night].sid, own, lights, q.blocks[night].files["lights"], nn[0])
    return Out(M("{n} lights in “{_name}”", n=len(lights), _name=q.blocks[nn[0]].name), nn[0], new=nn[0])


def op_merge_rows(q, src, dst):
    a, b = _row(q, src), _row(q, dst)
    if not a.nights or not b.nights or a is b:
        raise Refusal("two different stacks of lights are needed")
    _not_into_whole_variant(q, b)
    _no_clash(q, b, a.nights)
    why = ""
    for n in a.nights:
        why = _follow_stack(q, b, n) or why
    b.nights += a.nights
    a.nights = []
    toast = M("“{_a}” merged into “{_b}”: its own settings are dropped", _a=a.name, _b=b.name)
    if why:
        toast = join([toast, why])
    delete_rows(q, [a.id])
    return Out(toast, b.nights[0][0], style="need")


def op_duplicate_row(q, row):
    r = _row(q, row)
    drop_astro = bool(r.nights) and q.reference is not None
    # a copy keeps what each block holds (settings, method, engine settings, switch, review pause: today's Duplicate)
    blocks = [_clone_block(q, b) for b in r.blocks if not (drop_astro and q.blocks[b].kind == "astrometry")]
    nights = []
    for s, c in r.nights:
        o = q.blocks[s]
        nn = _night(q, name_of("{_name} copy", _name=o.name), {k: list(v) for k, v in o.files.items()}, c is not None)
        if c is not None:
            _copy_calibration(q, c, nn[1])
        nights.append(nn)
    nr = Row(new_id(q, "r"), free_name(q, name_of("{_name} copy", _name=r.name)), blocks, nights, list(r.fork) if r.fork else None)
    _copy_flow(nr, r)                                       # its workflow's own settings (Output, pauses...)
    if r.nights:                                            # its own nights' copies, in its order
        nr.sessions = None
    nr.legacy = None                                        # (a new row: made from its blocks, as a variant)
    nr.calib_node, nr.calib_fork, nr.calib = r.calib_node, r.calib_fork, copy.deepcopy(r.calib)
    if r.align:
        nr.align = copy.deepcopy(r.align)
    elif r.nights and not r.fork:
        nr.align = new_align(r.id)
    q.rows.insert(_after_subtree(q, r.id), nr)
    if blocks and q.blocks[blocks[-1]].kind not in JOIN_KINDS and out_ports(q.blocks[blocks[-1]]):
        _to_post(q, blocks[-1])
    first = nights[0][0] if nights else blocks[0]
    return Out(M("“{_new}” drawn under “{_row}”", _new=nr.name, _row=r.name), first, new=first)


def op_rename_row(q, row, name):
    r = _row(q, row)
    name = (name or "").strip()
    if not name:
        raise Refusal("a name cannot be empty")
    if name != r.name and name in {x.name for x in q.rows}:
        raise Refusal(M("“{_name}” already exists", _name=name))
    r.name = name
    return Out(M("renamed “{_name}”", _name=name))       # no focus: what was chosen (a block of the row) stays chosen


def op_rename_night(q, night, name):
    _night_row(q, night)
    name = (name or "").strip()
    if not name:
        raise Refusal("a name cannot be empty")
    q.blocks[night].name = name
    return Out(M("session renamed “{_name}”", _name=name), night)


def op_set_reference(q, row):
    """"Reference for all": every other stack (and each variant with its own Stacking) aligned on this one, from its
    Astrometry when it has one; this one on its own grid. The old reference's Astrometry passes to it when it has
    none (the plate solution the others take)."""
    r = _row(q, row)
    if not r.nights:
        raise Refusal("the reference is a stack of lights")
    old = q.reference
    ix = Index(q)
    mine = set(ix.subtree(row))
    if old != row and old in ix.row and ix.row[old].nights and not any(
            q.blocks[b].kind == "astrometry" for b in r.blocks):
        astro = [b for b in ix.row[old].blocks if q.blocks[b].kind == "astrometry"]
        if astro:
            detach(q, astro[0], keep=True)
            _insert_after_frames(q, _row(q, row), astro[0])
    for x in q.rows:
        own = bool(x.nights) or (x.fork and any(q.blocks[b].kind == "stacking" for b in x.blocks))
        if not own or x.id in mine:
            if x.id in mine and x.align:
                x.align = None
            continue
        a = x.align or new_align(row)
        x.align = {**a, "ref": row, "from": None, "auto": True}
    return Out(M("“{_row}” becomes the registration reference", _row=r.name), row, style="need")


def op_set_align(q, row, ref, block=None, method=None):
    """The target of a row's Align on (INTEGRATION 4.3): `ref` a row (or tap) of another stack or image, None: an
    Align on with its target to choose; `block`: the target's block the alignment leaves from (its Stacking or Image
    by default; from its Astrometry or a block after it, the plate solution comes with the grid)."""
    r = _row(q, row)
    ix = Index(q)
    if not any(q.blocks[b].kind in ("stacking", "image") for b in r.blocks) and not r.nights:
        raise Refusal("an Align on follows a Stacking or an Image of the row")
    if ref is not None:
        flows = {x.id for x in q.rows} | {t.id for t in q.taps}
        if ref not in flows:
            raise Refusal("row not found")
        if block is not None and block not in _flow_chain(q, ix, ref):
            raise Refusal("the block is not in the target's chain")
        why = _join_problem(q, row, ref)
        if why:
            raise Refusal(why)
    a = r.align or new_align(ref)
    if method is not None and method not in wfl.METHODS["align_on"]:
        raise Refusal("unknown method")
    r.align = {**a, "ref": ref, "from": block, "auto": False, "method": method or a.get("method")}
    if ref is None:
        return Out(M("“{_row}”: grid to choose", _row=r.name), row)
    tgt = ix.row[ref].name if ref in ix.row else next(t.name for t in q.taps if t.id == ref)
    return Out(M("“{_row}” on the grid of “{_ref}”", _row=r.name, _ref=tgt), row)


def op_own_grid(q, row):
    """The row keeps its own grid: no Align on."""
    r = _row(q, row)
    if not r.align:
        raise Refusal("already on its own grid")
    r.align = None
    return Out(M("“{_row}” on its own grid", _row=r.name), row)


def op_set_align_off(q, row):
    """Switch the row's Align on off (its image stays on its own grid, the target kept) or on again."""
    r = _row(q, row)
    if not r.align:
        raise Refusal("no Align on in this row")
    r.align = {**r.align, "off": not r.align.get("off")}
    return Out("Align on switched off" if r.align["off"] else "Align on switched on", row, style="off")


def _join_problem(q, row, ref):
    """project2.join_problem on the compiled graph: why `row` cannot be aligned on `ref` (no self, no combination,
    no loop), "" when it can."""
    from ..core.project2 import Project2, join_problem
    from .compile import compile_into
    p = Project2(name="check")
    c = compile_into(q, p)
    a, b = c.wids.get(row), c.wids.get(ref)
    if b is None and ref in Index(q).row:
        from ..core.project2 import JOIN_NO_STACK      # a row that is no workflow (a Combine layers, a layer of Post)
        return JOIN_NO_STACK
    if a is None or b is None:
        return ""
    why = join_problem(p, a, b, reference=True)
    if not why:
        return ""
    from ..core.project2 import JOIN_LOOP
    if why.endswith("it would make a loop"):
        return M(JOIN_LOOP, src=typed(p.workflow(b).name), target=typed(p.workflow(a).name))
    return why


def op_set_calibrate_together(q, on):
    """One calibration for all the nights of a stack (together), or each night its own (INTEGRATION 4.4)."""
    q.calibrate_together = bool(on)
    return Out("calibrated together" if on else "each night calibrated on its own", style="off")
