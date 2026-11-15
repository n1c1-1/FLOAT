"""The graph's gestures: what every gesture keeps after it ran (ids numbered, shared nights, whole variants,
frozen rows thawed, what was made and acted on) and the checks of the result (normalise, legality)."""
from .shapes import Result
from .model import FILE_KINDS, FRAME_KINDS, Index, Row, Tap, check, in_ports, new_id, out_ports
from . import rules
from .msg import M, name_of
from .rules import LABEL, evaluate, inherited, need_greyed, title_msg
from ..core import workflow as wfl
from .ops_base import Refusal, _flow_chain, _night, _copy_calibration


def _ids(q):
    return {"rows": {r.id for r in q.rows}, "blocks": set(q.blocks), "wires": {w.id for w in q.wires}}


def _is_layers_row(q, ix, rid):
    r = ix.row[ix.root(rid)]
    return bool(r.blocks) and q.blocks[r.blocks[0]].kind == "combine_layers"


def _alloc(q):
    """The workflow id of each new row and tap (a Combine layers row is a layer of Post: none), the session id of each
    new night (with its entry in the project's nights), in the order they are made; ids are never reused."""
    new = [f for f in list(q.rows) + list(q.taps) if f.wid is None]
    ix = Index(q) if new else None
    for f in new:
        if isinstance(f, Row) and _is_layers_row(q, ix, f.id):
            continue
        f.wid = q.next_wid
        q.next_wid += 1
        q.order.append(f.wid)
    fresh = sorted((b for b in q.blocks.values() if b.kind == "session" and b.sid is None), key=lambda b: int(b.id[1:]))
    if not fresh:
        return
    nxt = max([n["sid"] for n in q.nights] + [b.sid for b in q.blocks.values() if b.sid is not None] + [0]) + 1
    for b in fresh:
        b.sid = nxt
        q.nights.append({"sid": nxt, "name": b.name or "", "filter": "", "files": {}, "left_out": {}})
        nxt += 1
    for r in q.rows:
        for s, c in r.nights:
            if c is not None:
                q.blocks[c].sid = q.blocks[s].sid


def _sessions_state(q):
    return {b.id: (repr(b.files), b.name) for b in q.blocks.values() if b.kind == "session"}


def _sync_shared(q, before):
    """A night drawn in several stacks is one session: what a gesture changed in one of its Session blocks (its files,
    its name) is the others'; a name changed is the session's."""
    groups = {}
    for b in q.blocks.values():
        if b.kind == "session" and b.sid is not None:
            groups.setdefault(b.sid, []).append(b)
    entry = {n["sid"]: n for n in q.nights}
    for sid, bs in groups.items():
        changed = [b for b in bs if before.get(b.id) != (repr(b.files), b.name)]
        if not changed:
            continue
        src = next((b for b in changed if b.id in before), changed[0])
        for b in bs:
            if b is not src:
                b.files = {k: list(v) for k, v in src.files.items()}
                b.name = src.name
        if sid in entry and src.id in before and before[src.id][1] != src.name:
            entry[sid]["name"] = src.name


#: a session put into a whole variant: it draws its original's sessions (_whole_variants_follow took it out again)
WHOLE_VARIANT = "“{_row}” draws the sessions of “{_orig}”: a session goes into “{_orig}”"


def whole_original(q, r):
    """The stack a whole variant ``r`` draws the sessions of (_whole_variants_follow), or None."""
    if r.trial is None or r.fork or not r.nights:
        return None
    o = next((x for x in q.rows if x.wid == r.trial and x is not r), None)
    return o if o is not None and not o.fork and o.nights else None


def _not_into_whole_variant(q, t):
    o = whole_original(q, t)
    if o is not None:
        raise Refusal(M(WHOLE_VARIANT, _row=t.name, _orig=o.name))


def _whole_variants_follow(q):
    """A variant of a whole stack (op_try_variant from a block before which no variant forks) draws its original's
    sessions: a session added to the original, or taken out of it, is the variant's too, in the original's order (a
    beta tester's variant kept 4 lights after its original's stack grew to 6, and its frames were said twice). Its own
    Calibration of a session it already draws is kept; a new one's is the original's copied."""
    for v in q.rows:
        if v.trial is None or v.fork or not v.nights:
            continue
        o = next((r for r in q.rows if r.wid == v.trial and r is not v), None)
        if o is None or o.fork or not o.nights:
            continue
        sid = {s: q.blocks[s].sid for s, _c in o.nights + v.nights}
        want = [sid[s] for s, _c in o.nights]
        if [sid[s] for s, _c in v.nights] == want or None in want:
            continue
        mine = {sid[s]: [s, c] for s, c in v.nights}
        nights = []
        for s, c in o.nights:
            pair = mine.pop(sid[s], None)
            if pair is None:
                pair = _night(q, q.blocks[s].name, {k: list(x) for k, x in q.blocks[s].files.items()}, c is not None)
                q.blocks[pair[0]].sid = sid[s]
                if c is not None:
                    _copy_calibration(q, c, pair[1], sid[s])
            nights.append(pair)
        for s, c in mine.values():                        # a session its original no longer draws
            gone = {s, c} - {None}
            q.wires = [w for w in q.wires if w.src[0] not in gone and w.dst[0] not in gone]
            for b in gone:
                q.blocks.pop(b, None)
        v.nights = nights


def _flow_sig(q, ix, f):
    if isinstance(f, Row):
        chain = ix.effective(f.id)
        holder = rules.holder_align(q, ix, chain)
    else:
        if f.block not in ix.row_of:
            return None
        chain = _flow_chain(q, ix, f.id)
        holder = rules.holder_align(q, ix, chain)
    return repr(([(q.blocks[b].kind, q.blocks[b].off, q.blocks[b].method) for b in chain],
                 {k: holder.get(k) for k in ("ref", "from", "off", "method")} if holder else None))


def _legacy_chains(q):
    if not any(f.legacy for f in list(q.rows) + list(q.taps)):
        return {}
    ix = Index(q)
    return {f.id: _flow_sig(q, ix, f) for f in list(q.rows) + list(q.taps) if f.legacy}


def _thaw(q, chains):
    """A workflow made before the free graph (kept as its options, frozen) is rebuilt from its blocks by the first
    change of its chain: from then on its blocks are the truth."""
    keep = q.__dict__.pop("_legacy_kept", set())     # a chain replaced with its options given (replace_chain)
    if not chains:
        return
    ix = Index(q)
    for f in list(q.rows) + list(q.taps):
        if f.legacy and f.id in chains and f.id not in keep and _flow_sig(q, ix, f) != chains[f.id]:
            f.legacy = None


def _created(q, before):
    now = {"rows": [r.id for r in q.rows], "blocks": list(q.blocks), "wires": [w.id for w in q.wires]}
    return _summarise(q, {k: [x for x in now[k] if x not in before[k]] for k in now})


def _summarise(q, c):
    """{"rows", "blocks", "wires": new ids} + "row" / "session" / "block": the first new one of each, or None."""
    c = dict(c)
    c["row"] = c["rows"][0] if c["rows"] else None
    c["session"] = next((b for b in c["blocks"] if q.blocks[b].kind == "session"), None)
    c["block"] = next((b for b in c["blocks"] if q.blocks[b].kind not in ("session", "calibration")), None)
    return c


def _acted(graph, view, q, out):
    from .layout import grid
    if out.new and out.new in q.blocks:
        rid = Index(q).row_of.get(out.new)
        g = grid(q, view)
        cell = g.cells.get(out.new) or g.cells.get(f"card:{rid}")
        return (rid, cell[0], cell[1]) if cell else (rid, None, None)
    if out.old:
        g = grid(graph, view)
        cell = g.cells.get(out.old)
        rid = Index(graph).row_of.get(out.old)
        return (rid, cell[0], cell[1]) if cell else (rid, None, None)
    return None


def _view_op(graph, view, name, args):
    p = graph
    if name == "set_mode":
        if args.get("mode") not in ("graph", "table"):
            return Result(False, reason="unknown mode")
        view["mode"] = args["mode"]
        return Result(True)
    if name == "select":
        view["selected"] = args.get("id")
        return Result(True)
    ix = Index(p)
    if name == "unfold_all":
        view["folded"] = False
        return Result(True)
    for r in p.rows:                                        # fold_all
        why = fold_problem(p, ix, r.id) if r.nights else ""
        if why:
            return Result(False, reason=why)
    view["folded"] = True
    return Result(True)


def fold_problem(p, ix, rid):
    for v in ix.children[rid]:
        if p.blocks[ix.row[v].fork[1]].kind in FRAME_KINDS[:-1]:
            return M("a variant starts from the stack “{_row}”: the preprocessing stays unfolded", _row=ix.row[rid].name)
    return ""


# ---------------------------------------------------------------------------------------------------- validation
def finish(q, style=None):
    """Normalise what has one stored form, check structure, refuse what the engine (or the drawing) refuses."""
    normalise(q)
    check(q)
    why = legality(q, style)
    if why:
        raise Refusal(why)


def normalise(q):
    """The stored forms: wires on existing ports, one active wire per port, settings and files equal to what they
    follow dropped, taps on chain blocks (a combination's wire from a mid-row block reads a tap: made, and gone with
    its last wire), one node name per chain, Align on targets that exist."""
    ix = Index(q)
    keep, seen = [], set()
    for w in q.wires:
        if w.src[0] not in q.blocks or w.dst[0] not in q.blocks:
            continue
        if w.src[1] not in out_ports(q.blocks[w.src[0]]) or w.dst[1] not in in_ports(q.blocks[w.dst[0]]):
            continue
        key = (tuple(w.src), tuple(w.dst))
        if key in seen:
            continue
        seen.add(key)
        if q.blocks[w.dst[0]].kind == "post":
            w.active = True
        keep.append(w)
    q.wires = keep
    q.taps = [t for t in q.taps if t.block in ix.row_of and q.blocks[t.block].kind not in ("session", "calibration")]
    ports = {}
    for w in q.wires:
        if q.blocks[w.dst[0]].kind != "post":
            ports.setdefault(tuple(w.dst), []).append(w)
    for ws in ports.values():
        act = [w for w in ws if w.active]
        for w in act[1:]:
            w.active = False
        if not act:
            ws[0].active = True
    _norm_taps(q, ix)
    live = {f.wid for f in list(q.rows) + list(q.taps) if f.wid is not None}
    q.wires = [w for w in q.wires if not (w.wid is not None and w.wid not in live and q.blocks[w.dst[0]].kind == "post")]
    q.order = [x for x in q.order if x in live]
    _norm_names(q, ix)
    _norm_align(q, ix)
    for b in q.blocks.values():
        inh = inherited(q, b.kind)
        b.settings = {k: v for k, v in sorted(b.settings.items()) if inh.get(k, object()) != v}
        if b.files is not None:
            b.files = {k: b.files[k] for k in FILE_KINDS if k in b.files
                       and b.files[k] != q.common["files"].get(k, [])}
    q.common["files"] = {k: q.common["files"][k] for k in FILE_KINDS if q.common["files"].get(k)}
    q.common["settings"] = {k: dict(sorted(v.items())) for k, v in q.common["settings"].items() if v}
    _norm_params(q)


def _norm_params(q):
    """An image block's engine settings are its node's. One whose node name is given by the compile (moved into a
    chain already holding its name, or copied) may hold them under another name, another node's of its chain (a
    second Gradient's settings applied to the first): they are put under the name it is given, which it keeps."""
    late = [b for r in q.rows for b in r.blocks if b in q.blocks and q.blocks[b].kind in wfl.IMAGE_OPS
            and q.blocks[b].engine is None and q.blocks[b].params]
    if not late:
        return
    from ..core.project2 import Project2
    from .compile import compile_into
    names = compile_into(q, Project2(name="names")).names
    for b in late:
        blk, name = q.blocks[b], names.get(b)
        if name is None or list(blk.params) == [name]:
            continue
        mine = {}
        for kv in blk.params.values():
            mine.update(kv)
        blk.engine, blk.params = name, {name: mine}


def _mid_row(ix, b):
    r = ix.row[ix.row_of[b]]
    return ix.pos[b] < len(r.blocks) - 1


def _norm_taps(q, ix):
    """A combination's input from a block where no workflow of a row ends is a tap's (INTEGRATION 4.6); a tap made
    for such a wire goes with its last one, or when its block ends its row again."""
    flows = {f.wid for f in list(q.rows) + list(q.taps) if f.wid is not None}
    want = []
    for w in q.wires:
        if not w.active or q.blocks[w.dst[0]].kind != "combine" or w.src[0] not in ix.row_of:
            continue
        if _mid_row(ix, w.src[0]) and not (w.wid is not None and w.wid in flows) and w.src[0] not in want:
            want.append(w.src[0])
    q.taps = [t for t in q.taps if not t.auto or t.block in want]
    have = {t.block for t in q.taps}
    for b in want:
        if b not in have:
            r = ix.row[ix.row_of[b]]
            q.taps.append(Tap(new_id(q, "t"), b, name_of("{_row} · {lab}", _row=r.name, lab=LABEL[q.blocks[b].kind]),
                              auto=True))


def _norm_names(q, ix):
    """One node name per chain (Project2.validate): a block that comes into a chain holding its stored name (moved,
    or its chain changed) is named again by the compile; the earlier block keeps its name (its results)."""
    for r in q.rows:
        seen = {}
        for b in ix.effective(r.id):
            blk = q.blocks[b]
            if not blk.engine or blk.kind not in wfl.IMAGE_OPS:
                continue
            o = seen.get(blk.engine)
            if o is None or o == b:
                seen[blk.engine] = b
                continue
            early, late = sorted((o, b), key=lambda x: int(x[1:]))
            seen[blk.engine] = early
            q.blocks[late].engine = None


def _norm_align(q, ix):
    flows = {r.id for r in q.rows} | {t.id for t in q.taps}
    for r in q.rows:
        a = r.align
        if not a:
            continue
        if a.get("ref") is not None and a["ref"] not in flows:
            a["ref"] = None
        if a.get("from") is not None and (a["from"] not in q.blocks or a.get("ref") is None
                                          or a["from"] not in _flow_chain(q, ix, a["ref"])):
            a["from"] = None
        if r.fork and not any(q.blocks[b].kind in ("stacking", "image") for b in r.blocks):
            r.align = None               # a variant aligned with its stack: no Align on of its own


def legality(q, style=None):
    """"" when the graph may be kept, else why not (a message): each row's effective chain as the engine runs it,
    then the drawing (rules.chain_verdict), a join's ports, the compiled definition as the engine reads it
    (compile.engine_check: no engine form, Project2.validate, the stage chains)."""
    from .compile import engine_check
    ix = Index(q)
    for r in q.rows:
        why, e = rules.row_verdict(q, ix, r.id)
        if why:
            return _styled(why, e, style)
    for w in q.wires:                     # (before what it makes greyed further: the cause)
        if w.active and q.blocks[w.dst[0]].kind == "combine" and q.blocks[w.src[0]].kind == "synth":
            return M("{lab} is no input of a combination: it reads an image or a Stars/background",
                     lab=LABEL[q.blocks[w.src[0]].kind])
    bad = need_greyed(evaluate(q, ix, wiring=False))
    if bad:
        b, why = next(iter(bad.items()))
        t = title_msg(q, b)
        if style == "need":
            return M("{t} needs it", t=t)
        if style == "off":
            return M("{t} needs it: disable {t} first", t=t)
        return why
    for w in q.wires:                     # the engine's combination reads an Output or a Stars/background
        if w.active and q.blocks[w.dst[0]].kind == "combine" and q.blocks[w.src[0]].kind in ("synth", "combine_layers"):
            return M("{lab} is no input of a combination: it reads an image or a Stars/background",
                     lab=LABEL[q.blocks[w.src[0]].kind])
    if q.calibrate_together:
        for r in q.rows:
            # (a night without its Calibration block is not calibrated: as one switched off, the engine's one node)
            offs = {q.blocks[c].off if c is not None else True for _s, c in r.nights}
            if len(offs) > 1:
                return "calibrated together: one setting for all nights"
    return engine_check(q)


def _styled(why, e, style):
    """A removal or a switch-off the engine refuses because a later block needs the block: said as the engine says
    it for those gestures (wfl.remove_problem, wfl.off_problem)."""
    if e is not None and style in ("need", "off") and e.template in (wfl.WHY_NEEDS, wfl.WHY_HALVES) and \
            getattr(e, "block", ""):
        return M(wfl.WHY_REMOVE_NEEDED if style == "need" else wfl.WHY_OFF_NEEDED,
                 other=wfl.BLOCK_TITLES.get(e.block, e.block))
    return why
