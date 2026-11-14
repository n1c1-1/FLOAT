"""Operations (§4). Every operation runs on a clone of the graph; refused = nothing changes. No history here:
the caller keeps the graphs it wants to go back to."""
import copy
from dataclasses import dataclass

from .shapes import Result
from .model import (Index, Row, Wire, Tap, check, clone, model_json, new_block, new_id, in_ports, out_ports, edges,
                    FILE_KINDS, FRAME_KINDS, JOIN_KINDS, COMBO_PORTS, COMBO_LABEL, LAYER_PORTS, POST_PORT)
from . import rules
from .msg import M, join, typed, name_of, TR
from .rules import LABEL, OFFABLE, evaluate, need_greyed, title_msg, inherited, rank
from ..core import workflow as wfl

COMMON = "common"


class Refusal(Exception):
    """An operation that cannot be done. msg: the reason, a message (see msg.py)."""

    def __init__(self, msg):
        super().__init__(msg)
        self.msg = msg


@dataclass
class Out:
    toast: object = ""        # a message
    focus: object = None
    new: object = None       # block whose cell is where the gesture acted (after)
    old: object = None       # block removed: its cell before
    style: object = None     # None | "need" | "off": how a need refusal is worded
    created: object = None   # {"row": id, "session": id, "block": id, "wire": id}: what the gesture made


# ---------------------------------------------------------------------------------------------------- applying
VIEW_OPS = ("fold_all", "unfold_all", "set_mode", "select")
ALIASES = {"delete_block": "remove_block", "delete_night": "remove_night"}


def new_view():
    """The view state: what the drawing of the graph depends on besides the model."""
    return {"folded": False, "mode": "graph", "zoom": 1.0, "scroll": [0, 0], "selected": None}


def json_of(p):
    """The model's JSON text, kept on the project object (projects are never changed in place)."""
    js = p.__dict__.get("_json")
    if js is None:
        js = p.__dict__["_json"] = model_json(p)
    return js


def apply(graph, name, /, *, view=None, tr=None, **kwargs):
    """Run operation `name` on `graph`. -> (graph2, Result). Refused: graph2 is graph, Result.ok False, reason a
    message. `view` is the view state (new_view()) the fold rules read; `tr` writes the default names that are
    stored (Session 2, copies) in the language of the client."""
    return apply_many(graph, [(name, kwargs)], view, tr)


def apply_many(graph, steps, view=None, tr=None):
    """Several operations as ONE step: steps = [(name, kwargs | f(results so far) -> kwargs), ...]. All or nothing.
    Pure: `graph` is never changed; -> (graph2, Result). View operations (fold_all, unfold_all, set_mode, select,
    alone) change the `view` dict only and give back the same graph."""
    view = new_view() if view is None else view
    if len(steps) == 1 and steps[0][0] in VIEW_OPS:
        return graph, _view_op(graph, view, steps[0][0], steps[0][1])
    token = TR.set(tr)
    try:
        return _apply_many(graph, steps, view)
    finally:
        TR.reset(token)


def _run(q, fn, args):
    """One operation on q, then what every gesture keeps: the new workflows and nights numbered in the order they are
    made, a night shared by several stacks holding one list of files and one name, a frozen row made before the free
    graph rebuilt by the first change of its chain."""
    before = _sessions_state(q)
    chains = _legacy_chains(q)
    o = fn(q, **args)
    _alloc(q)
    _sync_shared(q, before)
    _thaw(q, chains)
    return o


def _apply_many(graph, steps, view):
    q = clone(graph)
    outs = []
    try:
        for name, args in steps:
            fn = OPS.get(ALIASES.get(name, name))
            if fn is None:
                raise Refusal(M("unknown operation: {_name}", _name=name))
            if callable(args):                 # kwargs computed from the earlier steps' Results
                args = args([Result(True, toast=o.toast, focus=o.focus, created=o.created) for o in outs])
            ids = _ids(q)
            o = _run(q, fn, args)
            o.created = _created(q, ids)
            outs.append(o)
        finish(q, outs[-1].style if outs else None)
        fold_guard(graph, q, view)
    except Refusal as e:
        res = Result(False, reason=e.msg)
        if len(steps) == 1 and ALIASES.get(steps[0][0], steps[0][0]) == "remove_block" and isinstance(steps[0][1], dict):
            res.dependants = dependants(graph, steps[0][1].get("block"))      # what remove_with_dependants takes
        return graph, res
    out = outs[-1]
    toast = join([o.toast for o in outs])
    acted = _acted(graph, view, q, out)
    created = {k: [x for o in outs for x in o.created[k] if x in _ids(q)[k]] for k in ("rows", "blocks", "wires")}
    res = Result(True, toast=toast, focus=out.focus, acted=acted, created=_summarise(q, created))
    return (q if json_of(q) != json_of(graph) else graph), res


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


def _flow_chain(q, ix, fid):
    if fid in ix.row:
        return ix.effective(fid)
    t = next((t for t in q.taps if t.id == fid), None)
    if t is None or t.block not in ix.row_of:
        return []
    eff = ix.effective(ix.row_of[t.block])
    return eff[:eff.index(t.block) + 1]


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
            offs = {q.blocks[c].off for _s, c in r.nights}
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


def attempt(graph, steps, view=None):
    """A gesture tried without its refusal (the fuzz's reverse invariant): (state, where, reason). where: "op" (the
    operation itself refused: no state is made, state None), "legal" (refused by legality: state is the graph it
    would have made), "fold" (the folded band), "ok"."""
    q = clone(graph)
    outs = []
    try:
        for name, args in steps:
            fn = OPS.get(ALIASES.get(name, name))
            if fn is None:
                raise Refusal(M("unknown operation: {_name}", _name=name))
            if callable(args):
                args = args([Result(True, toast=o.toast, focus=o.focus, created=o.created) for o in outs])
            o = _run(q, fn, args)
            o.created = {}
            outs.append(o)
    except Refusal as e:
        if getattr(e, "at", None) is not None and len(steps) == 1 and steps[0][0] == "add_block":
            return attempt(graph, [("add_block", {**steps[0][1], "at": e.at})], view)   # the natural place's
        return None, "op", e.msg
    normalise(q)
    check(q)
    if legality(q, outs[-1].style if outs else None):
        return q, "legal", legality(q)        # the reason itself (a removal's or a switch-off's wording aside)
    try:
        fold_guard(graph, q, view or new_view())
    except Refusal as e:
        return q, "fold", e.msg
    return q, "ok", ""


# ---------------------------------------------------------------------------------------------------- helpers
def _blk(q, b, kinds=None):
    if b not in q.blocks:
        raise Refusal("block not found")
    if kinds and q.blocks[b].kind not in kinds:
        raise Refusal(M("{lab} does not fit here", lab=LABEL.get(q.blocks[b].kind, "")))
    return q.blocks[b]


def _row(q, rid):
    for r in q.rows:
        if r.id == rid:
            return r
    raise Refusal("row not found")


def _names(q):
    return {r.name for r in q.rows} | {b.name for b in q.blocks.values() if b.name}


def free_name(q, base, first_plain=True):
    names = {r.name for r in q.rows}
    if first_plain and base not in names:
        return base
    n = 2
    while f"{base} {n}" in names:
        n += 1
    return f"{base} {n}"


def session_name(q):
    names, n = _names(q), 1
    while name_of("Session {n}", n=n) in names:
        n += 1
    return name_of("Session {n}", n=n)


def _night(q, name, files=None):
    s = new_block(q, "session", name=name, files=files)
    c = new_block(q, "calibration")
    return [s.id, c.id]


def _default_chain(q):
    return [new_block(q, k).id for k in ("registration", "selection", "stacking")]


def _to_post(q, b, port=None):
    post = Index(q).post
    port = port or out_ports(q.blocks[b])[0]
    if not any(w.src == [b, port] and w.dst[0] == post for w in q.wires):
        q.wires.append(Wire(new_id(q, "w"), [b, port], [post, POST_PORT]))


def _map_port(blk, port):
    ports = out_ports(blk)
    return port if port in ports else (ports[0] if ports else None)


def _copy_block(q, b):
    o = q.blocks[b]
    return new_block(q, o.kind, settings=dict(o.settings), off=o.off).id


def _after_subtree(q, rid):
    """Index in rows right after the rows of rid's subtree."""
    sub = set(Index(q).subtree(rid))
    return max(i for i, r in enumerate(q.rows) if r.id in sub) + 1


def delete_rows(q, rids):
    """Delete rows with their subtrees, blocks and wires; the rows aligned on a deleted one pass to the first of
    them (its own grid then), with the deleted target's Astrometry when it has none."""
    ix = Index(q)
    q._astro_moved = False
    gone = []
    for r in rids:
        for x in ix.subtree(r):
            if x not in gone:
                gone.append(x)
    left = [r for r in q.rows if r.id not in gone and r.align and r.align.get("ref") in gone]
    if left:
        heir = left[0]
        old = heir.align["ref"]
        heir.align = None
        for r in left[1:]:
            r.align["ref"], r.align["from"] = heir.id, None
            r.align["auto"] = True
        astro = [b for b in ix.row[old].blocks if q.blocks[b].kind == "astrometry"]             if old in ix.row and ix.row[old].nights and heir.nights else []
        if astro and not any(q.blocks[b].kind == "astrometry" for b in heir.blocks):
            ix.row[old].blocks.remove(astro[0])
            _insert_after_frames(q, heir, astro[0])
            q._astro_moved = True
    blocks = {b for r in gone for b in ix.row[r].blocks + [x for n in ix.row[r].nights for x in n]}
    nw = len([w for w in q.wires if w.src[0] in blocks or w.dst[0] in blocks])
    q.wires = [w for w in q.wires if w.src[0] not in blocks and w.dst[0] not in blocks]
    for b in blocks:
        del q.blocks[b]
    q.rows = [r for r in q.rows if r.id not in gone]
    names = join([M("“{_name}”", _name=ix.row[r].name) for r in gone], ", ")
    done = (M("{names} deleted", names=names) if len(gone) == 1
            else M("{n} deleted: {names}", n=len(gone), names=names))
    if nw:
        done = join([done, M(" with {n} wire" if nw == 1 else " with {n} wires", n=nw)], "")
    return done


def _auto_align(q, r):
    """A new stack is aligned on the stack the others are aligned on (the reference), from its Astrometry when it
    has one; none when it is the first stack."""
    ref = q.reference
    if ref is not None and ref != r.id and ref in {x.id for x in q.rows}:
        r.align = new_align(ref)


def new_align(ref, block=None):
    return {"ref": ref, "from": block, "from_name": None, "ref_wid": None, "method": wfl.DEFAULT_METHOD["align_on"],
            "off": False,
            "fork": None, "params": {}, "auto": block is None}


def _night_off(q, r):
    """The switch of the Calibration of a night joining stack r: calibrated together, the stack's (one setting for
    all its nights)."""
    return bool(q.calibrate_together and r.nights and q.blocks[r.nights[0][1]].off)


def _insert_after_frames(q, r, b):
    i = max([j for j, x in enumerate(r.blocks) if q.blocks[x].kind in FRAME_KINDS + ("image",)] or [-1]) + 1
    r.blocks.insert(i, b)


def detach(q, b, keep=False):
    """Take chain block b out of its row: its variants re-fork at its predecessor, its port wires pass to the
    predecessor of its row, an emptied variant row goes (its own variants take its place)."""
    ix = Index(q)
    rid = ix.row_of[b]
    r = ix.row[rid]
    i = ix.pos[b]
    pred = ix.pred(b)
    kids = [v for v in ix.children[rid] if ix.row[v].fork[1] == b]
    if i == 0 and kids and (len(r.blocks) > 1 or pred is None):
        raise Refusal("variants start from this block: delete them first")
    for w in list(q.wires):
        if w.src[0] == b:
            port = _map_port(q.blocks[pred], w.src[1]) if (i > 0 and pred) else None
            if port is None:
                q.wires.remove(w)
            else:
                w.src = [pred, port]
    for v in kids:
        ix.row[v].fork = [ix.row_of[pred], pred] if pred else None
    r.blocks.remove(b)
    if not r.blocks and not r.nights:
        at = q.rows.index(r)
        q.rows = [x for x in q.rows if x.id not in kids]
        for k, v in enumerate(kids):
            q.rows.insert(at + 1 + k, ix.row[v])
        q.rows.remove(r)
    if not keep:
        q.wires = [w for w in q.wires if w.dst[0] != b]
        del q.blocks[b]


def would_cycle(q, src, dst):
    ix = Index(q)
    succ = {}
    for a, c in edges(q, ix):
        succ.setdefault(a, []).append(c)
    stack, seen = [dst], set()
    while stack:
        x = stack.pop()
        if x == src:
            return True
        if x not in seen:
            seen.add(x)
            stack += succ.get(x, [])
    return False


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
        name = _named(q, name, session_name(q))
        r = Row(new_id(q, "r"), name, _default_chain(q), [_night(q, name, _files(files))])
        _auto_align(q, r)
        q.rows.append(r)
        _to_post(q, r.blocks[-1])
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
    name = (name or "").strip() or session_name(q)
    off = _night_off(q, r)
    r.nights.append(_night(q, name, _files(files)))
    q.blocks[r.nights[-1][1]].off = off
    return Out(M("{_name} added to “{_row}”: it stacks with its sessions · Ctrl+Z to undo", _name=name, _row=r.name),
               r.nights[-1][0], new=r.nights[-1][0])


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


def op_add_stack(q, name, nights, blocks=None, align="auto", all_nights=False):
    """A stack drawing sessions of the project (``nights``: their session ids, drawn in that order), its chain
    ``blocks`` ([{"kind", "method"}], None: Registration, Selection, Stacking); ``align`` "auto": on the reference
    (the first stack keeps its own grid), None: its own grid; ``all_nights``: it stacks every night of the project,
    those added later too."""
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
        nn = _night(q, src.name if src is not None else (e.get("name") or name_of("Session {n}", n=sid)), files)
        q.blocks[nn[0]].sid = q.blocks[nn[1]].sid = sid
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
    s = q.blocks[night]
    if t is r or (s.sid is not None and any(q.blocks[x].sid == s.sid for x, _c in t.nights)):
        raise Refusal(M("“{_name}” is already in “{_row}”", _name=s.name, _row=t.name))
    nn = _night(q, s.name, {k: list(v) for k, v in s.files.items()})
    q.blocks[nn[0]].sid = q.blocks[nn[1]].sid = s.sid
    q.blocks[nn[1]].off = _night_off(q, t) if q.calibrate_together else q.blocks[n[1]].off
    t.nights.append(nn)
    return Out(M("“{_name}” also stacked in “{_row}”", _name=s.name, _row=t.name), nn[0], new=nn[0])


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
        nr = Row(new_id(q, "r"), free_name(q, name), _default_chain(q), [n])
        _auto_align(q, nr)
        q.rows.append(nr)
        _to_post(q, nr.blocks[-1])
        return Out(M("“{_name}” moves to its own row “{_row}”", _name=name, _row=nr.name), night, new=night)
    t = _row(q, row)
    if not t.nights:
        raise Refusal("a session joins a stack of lights")
    if before is not None and before not in [x[0] for x in t.nights]:
        raise Refusal("session not found in this stack")
    if t is not r:
        _no_clash(q, t, [n])
    if t is r and (before is None or before == night):
        if before is None and t.nights[-1] is not n:
            t.nights.remove(n)
            t.nights.append(n)
            return Out(M("“{_name}” moved", _name=name), night, new=night)
        return Out(focus=night)                 # dropped where it is: nothing moves, nothing said
    if t is not r and t.nights:
        q.blocks[n[1]].off = _night_off(q, t) if q.calibrate_together else q.blocks[n[1]].off
    r.nights.remove(n)
    at = [x[0] for x in t.nights].index(before) if before is not None else len(t.nights)
    t.nights.insert(at, n)
    if t is r:
        return Out(M("“{_name}” moved", _name=name), night, new=night)
    toast = M("“{_name}” goes into “{_row}”", _name=name, _row=t.name)
    if not r.nights:
        toast = join([toast, delete_rows(q, [r.id])])
    return Out(toast, night, new=night)


def _eff_list(q, s, kind):
    return rules.files_of(q, s, kind)[1]


def op_merge_nights(q, a, b):
    ra, na = _night_row(q, a)
    rb, nb = _night_row(q, b)
    if ra is not rb or a == b:
        raise Refusal("both sessions must be in the same stack")
    for k in FILE_KINDS:
        if k in q.blocks[b].files:
            base = _eff_list(q, a, k)
            q.blocks[a].files[k] = base + [x for x in q.blocks[b].files[k] if x not in base]
    ra.nights.remove(nb)
    for x in nb:
        del q.blocks[x]
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
    nn = _night(q, session_name(q), files)
    q.blocks[nn[1]].settings = dict(q.blocks[n[1]].settings)
    q.blocks[nn[1]].off = q.blocks[n[1]].off
    r.nights.insert(r.nights.index(n) + 1, nn)
    return Out(M("{n} lights in “{_name}”", n=len(lights), _name=q.blocks[nn[0]].name), nn[0], new=nn[0])


def op_merge_rows(q, src, dst):
    a, b = _row(q, src), _row(q, dst)
    if not a.nights or not b.nights or a is b:
        raise Refusal("two different stacks of lights are needed")
    _no_clash(q, b, a.nights)
    if q.calibrate_together:
        for _s, c in a.nights:
            q.blocks[c].off = _night_off(q, b)
    b.nights += a.nights
    a.nights = []
    toast = M("“{_a}” merged into “{_b}”: its own settings are dropped", _a=a.name, _b=b.name)
    delete_rows(q, [a.id])
    return Out(toast, b.nights[0][0], style="need")


def op_duplicate_row(q, row):
    r = _row(q, row)
    drop_astro = bool(r.nights) and q.reference is not None
    blocks = [_copy_block(q, b) for b in r.blocks if not (drop_astro and q.blocks[b].kind == "astrometry")]
    nights = []
    for s, c in r.nights:
        o = q.blocks[s]
        nn = _night(q, name_of("{_name} copy", _name=o.name), {k: list(v) for k, v in o.files.items()})
        q.blocks[nn[1]].settings = dict(q.blocks[c].settings)
        nights.append(nn)
    nr = Row(new_id(q, "r"), free_name(q, name_of("{_name} copy", _name=r.name)), blocks, nights, list(r.fork) if r.fork else None)
    for (_s, c), (_s2, c2) in zip(r.nights, nights):
        q.blocks[c2].off = q.blocks[c].off
    if r.align:
        nr.align = {**r.align, "params": dict(r.align.get("params") or {}), "from": r.align.get("from")}
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
    return Out(M("renamed “{_name}”", _name=name), row)


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


# ---------------------------------------------------------------------------------------------------- blocks
def op_add_block(q, kind, at=None, row=None, _block=None):
    if kind not in LABEL or kind == "post":
        raise Refusal("unknown block")
    if at is None:
        at = natural(q, kind, row)
    t = at.get("type")
    if t == "night":
        if kind != "session":
            raise Refusal("only a Session joins a stack")
        return op_add_night(q, at["row"])
    if t == "row":
        return op_add_row(q, kind)
    if t == "comp":
        return _add_join(q, kind, at.get("after"))
    if t == "port":
        return _add_port_row(q, kind, at["block"], at["port"])
    if kind in ("session", "image") + JOIN_KINDS:
        raise Refusal(M("{lab} is not inserted in a row", lab=LABEL[kind]))
    if kind == "calibration":
        raise Refusal("each session already has its Calibration")
    b = _block or new_block(q, kind).id
    if t == "insert":
        return _insert(q, b, at["row"], at.get("after"))
    if t == "branch":
        return _branch(q, b, at["row"], at.get("y"))
    raise Refusal("unknown place")


def _insert(q, b, rid, after):
    r = _row(q, rid)
    if after is None:
        if not r.fork and not r.nights:
            raise Refusal("nothing is inserted before the head of the row")
        r.blocks.insert(0, b)
    else:
        if after not in r.blocks:
            raise Refusal("the place is gone")
        i = r.blocks.index(after) + 1
        r.blocks.insert(i, b)
        if i == len(r.blocks) - 1:                       # appended at the end: the row's outputs follow
            from .judge import keeps_wire
            for w in q.wires:
                if w.src[0] == after and not keeps_wire(q, q.blocks[b].kind, w):
                    w.src = [b, _map_port(q.blocks[b], w.src[1])] if out_ports(q.blocks[b]) else w.src
    return Out(M("{lab} here, in “{_row}”", lab=LABEL[q.blocks[b].kind], _row=r.name), b, new=b)


def _branch(q, b, rid, y):
    """Variant forked before y (after the row's last block when y is None); b replaces y's copy when same kind."""
    r = _row(q, rid)
    ix = Index(q)
    if not r.blocks:                     # a stack being built (or the block dragged was its only one): no fork block
        raise Refusal("a variant starts after the Registration")
    if y is None:
        x, tail = r.blocks[-1], []
    else:
        if y not in r.blocks:
            raise Refusal("the place is gone")
        if q.blocks[y].kind in ("registration", "image") + JOIN_KINDS or r.blocks.index(y) == 0:
            raise Refusal("a variant starts after the Registration")
        x = ix.pred(y)
        tail = r.blocks[r.blocks.index(y):]
    if tail and q.blocks[tail[0]].kind == q.blocks[b].kind:
        tail = tail[1:]
    return _make_variant(q, rid, x, [b] + [_copy_block(q, t) for t in tail], [None] + tail, r.name)


def _make_variant(q, frow, x, blocks, origin, base):
    """New variant row forking frow after x; copies' port wires copied (alternatives at single ports)."""
    for nb, ob in zip(blocks, origin):
        if ob is None:
            continue
        for w in list(q.wires):
            if w.src[0] == ob:
                act = q.blocks[w.dst[0]].kind == "post"
                q.wires.append(Wire(new_id(q, "w"), [nb, w.src[1]], list(w.dst), act))
    v = Row(new_id(q, "r"), free_name(q, base, False), blocks, [], [frow, x])
    if any(q.blocks[b].kind == "stacking" for b in blocks):
        ix = Index(q)
        a = rules.holder_align(q, ix, ix.effective(frow))
        v.align = copy.deepcopy(a) if a else None
    q.rows.insert(_after_subtree(q, frow), v)
    return Out(M("variant “{_name}” under “{_base}”", _name=v.name, _base=base), blocks[0], new=blocks[0])


def op_add_variant(q, block):
    b = _blk(q, block)
    ix = Index(q)
    if b.kind in ("session", "calibration", "registration", "image", "post") + JOIN_KINDS or ix.pred(block) is None:
        raise Refusal("a variant starts after the Registration")
    rid = ix.row_of[block]
    r = ix.row[rid]
    x = ix.pred(block)
    frow = ix.row_of[x]
    tail = r.blocks[r.blocks.index(block):]
    if frow != rid:                                       # first block of a variant: a sibling of it
        tail = list(r.blocks)
    copies = [_copy_block(q, t) for t in tail]
    return _make_variant(q, frow, x, copies, tail, ix.row[frow].name)


def _add_join(q, kind, after):
    if kind not in JOIN_KINDS:
        raise Refusal("the composition takes a Combination or Combine layers")
    j = new_block(q, kind)
    name = free_name(q, name_of(COMBO_LABEL["sho"] if kind == "combine" else LABEL[kind]))
    q.rows.append(Row(new_id(q, "r"), name, [j.id], legacy=fusion_legacy() if kind == "combine" else None))
    if after is not None:
        a = _blk(q, after, ("stars", "synth"))
        post = Index(q).post
        if kind != "combine_layers" or any(w.src[0] == after and w.dst[0] != post for w in q.wires):
            raise Refusal("this Stars/background already feeds a block")
        for w in q.wires:
            if w.src[0] == after and w.dst[0] == post:      # the row's output passes to the new join
                w.src = [j.id, "image"]
        q.wires.append(Wire(new_id(q, "w"), [a.id, "Starless"], [j.id, "Background"]))
        q.wires.append(Wire(new_id(q, "w"), [a.id, "Stars only"], [j.id, "Stars"]))
    return Out(M("{_name} added", _name=name), j.id, new=j.id)


def _add_port_row(q, kind, jid, port):
    j = _blk(q, jid, ("combine", "combine_layers"))
    if port not in in_ports(j):
        raise Refusal("unknown port")
    if any(w.dst == [jid, port] for w in q.wires):
        raise Refusal("the port is already wired")
    name = free_name(q, name_of(port))
    if kind == "image":
        b = new_block(q, "image")
        q.rows.append(Row(new_id(q, "r"), name, [b.id]))
        q.wires.append(Wire(new_id(q, "w"), [b.id, "image"], [jid, port], channel="L" if j.kind == "combine" else None))
        return Out(M("New row for {port}", port=port), b.id, new=b.id)
    if kind != "session":
        raise Refusal("a port takes a new row: Session or Image")
    ix = Index(q)
    first = next((w for p_ in in_ports(j) for w in q.wires if w.dst == [jid, p_] and w.active), None)
    src_port = "image"
    if first is not None and first.src[0] in ix.row_of and not ix.is_join_row(ix.row_of[first.src[0]]):
        sb = first.src[0]
        chain = ix.effective(ix.row_of[sb])
        chain = chain[:chain.index(sb) + 1]
        if q.blocks[chain[0]].kind == "image":
            chain = chain[1:]
        aligned = q.reference is not None
        keep = [b for b in chain if not (aligned and q.blocks[b].kind == "astrometry")]
        blocks = [_copy_block(q, b) for b in keep]
        if not any(q.blocks[b].kind == "stacking" for b in blocks):
            blocks = _default_chain(q) + blocks
        src_port = first.src[1]
    else:
        blocks = _default_chain(q)
    r = Row(new_id(q, "r"), name, blocks, [_night(q, name)])
    _auto_align(q, r)
    q.rows.append(r)
    q.wires.append(Wire(new_id(q, "w"), [blocks[-1], _map_port(q.blocks[blocks[-1]], src_port)], [jid, port],
                        channel="L" if j.kind == "combine" else None))
    return Out(M("New row for {port}", port=port), r.nights[0][0], new=r.nights[0][0])


def natural(q, kind, row=None):
    """The natural place of a double-click (§2): a place spec."""
    ix = Index(q)
    if kind == "session":
        if row in ix.row and ix.row[row].nights:
            return {"type": "night", "row": row}
        fr = [r.id for r in q.rows if r.nights]
        return {"type": "night", "row": fr[0]} if (row is None and fr) else {"type": "row"}
    if kind == "image":
        return {"type": "row"}
    if kind == "combine":
        return {"type": "comp"}
    if kind == "combine_layers":
        for r in q.rows:
            if r.blocks and q.blocks[r.blocks[-1]].kind in ("stars", "synth") \
                    and not any(w.src[0] == r.blocks[-1] for w in q.wires):
                return {"type": "comp", "after": r.blocks[-1]}
        return {"type": "comp"}
    if kind == "calibration":
        raise Refusal("each session already has its Calibration")
    rows = [r.id for r in q.rows]
    target = row if row in ix.row else (rows[0] if rows else None)
    if target is None:
        raise Refusal("add a Session or an Image first")
    first_reason, first_at = None, None
    for rid in [target] + [r for r in rows if r != target]:
        r = ix.row[rid]
        i0 = 0
        while i0 < len(r.blocks) and rank(q.blocks[r.blocks[i0]].kind) <= rank(kind):
            i0 += 1
        slots = list(range(len(r.blocks) + 1))
        slots.sort(key=lambda i: (abs(i - i0), i))
        for i in slots:
            if i == 0 and not r.fork and not r.nights:
                continue
            at = {"type": "insert", "row": rid, "after": r.blocks[i - 1] if i else None}
            why = try_op(q, "add_block", {"kind": kind, "at": at})
            if not why:
                return at
            if first_reason is None:
                first_reason, first_at = why, at
    e = Refusal(first_reason or M("{lab}: no place", lab=LABEL[kind]))
    e.at = first_at                     # the place its reason is that of (operations.attempt)
    raise e


def fold_guard(before, q, view):
    """Folded band: a gesture that would fork a variant inside a stack is refused (its fork block is hidden in the
    card; unfolding the band by itself would move everything)."""
    if not view or view.get("folded") is not True:
        return
    from .layout import foldable
    ib, iq = Index(before), Index(q)
    for r in iq.row:
        if iq.row[r].nights and not foldable(q, iq, r) and (r not in ib.row or foldable(before, ib, r)):
            raise Refusal("Preprocessing is folded: unfold it first")


def try_op(q, name, args, view=None):
    """"" when operation `name` would be accepted on project q, else its refusal (q untouched)."""
    c = clone(q)
    try:
        out = _run(c, OPS[name], args)
        finish(c, out.style)
        fold_guard(q, c, view)
    except Refusal as e:
        return e.msg
    return ""


def op_move_block(q, block, at):
    b = _blk(q, block)
    ix = Index(q)
    if b.kind in ("session", "calibration", "post"):
        raise Refusal("each session keeps its Calibration")
    if b.kind in ("image",) + JOIN_KINDS:
        raise Refusal("a source stays at the head of its row")
    rid = ix.row_of[block]
    at = dict(at)
    if at.get("type") not in ("insert", "branch"):
        raise Refusal("unknown place for this block")
    r = ix.row[rid]
    if at.get("after") == block:
        at["after"] = r.blocks[ix.pos[block] - 1] if ix.pos[block] > 0 else None
        if at["after"] is None and not r.fork and not r.nights:
            return Out(focus=block)
    if at.get("y") == block:
        nxt = ix.pos[block] + 1
        at["y"] = r.blocks[nxt] if nxt < len(r.blocks) else None
    if own_spot(ix, block, at):
        return Out(focus=block)                 # dropped where it is: nothing moves, nothing said
    detach(q, block, keep=True)
    out = op_add_block(q, b.kind, at, _block=block)
    out.toast = M("{lab} moved", lab=LABEL[b.kind])
    return out


def own_spot(ix, block, at):
    """Whether place spec `at` puts chain block `block` back where it is (after its predecessor in its row)."""
    if at.get("type") != "insert" or at.get("row") != ix.row_of.get(block):
        return False
    i = ix.pos[block]
    r = ix.row[ix.row_of[block]]
    return at.get("after") in ((r.blocks[i - 1],) if i > 0 else (None,)) + (block,)


def op_remove_block(q, block):
    b = _blk(q, block)
    ix = Index(q)
    if b.kind == "post":
        raise Refusal("Post is fixed")
    if b.kind == "calibration":
        raise Refusal("each session has its Calibration: switch it off (D)")
    if b.kind == "session":
        return op_remove_night(q, block)
    rid = ix.row_of[block]
    if ix.pos[block] == 0 and b.kind in ("image",) + JOIN_KINDS:
        toast = delete_rows(q, [rid])
        return Out(toast, old=None if q._astro_moved else block, style="need")
    detach(q, block)
    return Out(M("{lab} deleted", lab=LABEL[b.kind]), old=block, style="need")


def op_switch_off(q, block):
    b = _blk(q, block)
    if b.kind not in OFFABLE:
        raise Refusal(rules.off_reason(b.kind))
    b.off = not b.off
    if b.kind == "calibration" and q.calibrate_together:
        # calibrated together, a stack has one Calibration (the engine's one node): its nights switch as one
        ix = Index(q)
        for _s, c in ix.row[ix.row_of[block]].nights:
            q.blocks[c].off = b.off
    return Out(M("{lab} switched off" if b.off else "{lab} switched on", lab=LABEL[b.kind]), block, style="off")


def op_set_pause(q, block):
    b = _blk(q, block)
    if b.kind == "post":
        raise Refusal("Post is fixed")
    b.pause = not b.pause
    return Out("Check pause" if b.pause else "Check pause removed", block)


def op_set_setting(q, block, key, value):
    b = _blk(q, block)
    if b.kind == "combine" and key == "type":
        return op_set_combination_type(q, block, value)
    if rules.setting_spec(b.kind, key) is None:
        raise Refusal("unknown setting")
    b.settings[key] = value
    return Out(M("{key}: {value}", key=key, value=value), block)


def op_apply_to_rows(q, block, key):
    b = _blk(q, block)
    ix = Index(q)
    val = rules.effective(q, block).get(key)
    if rules.setting_spec(b.kind, key) is None:
        raise Refusal("unknown setting")
    others = [x for x in rules.chain_blocks(q) if q.blocks[x].kind == b.kind
              and ix.row_of[x] != ix.row_of[block]]
    if not others:
        raise Refusal(M("no other row has a {lab}", lab=LABEL[b.kind]))
    for x in others:
        q.blocks[x].settings[key] = val
    return Out(M("{key} applied to {n} other row" if len(others) == 1 else "{key} applied to {n} other rows",
                 key=key, n=len(others)), block)


# ---------------------------------------------------------------------------------------------------- wires
def op_set_combination_type(q, block, type):
    j = _blk(q, block, ("combine",))
    if type not in COMBO_PORTS:
        raise Refusal("unknown type")
    j.settings["type"] = type
    gone = [w for w in q.wires if w.dst[0] == block and w.dst[1] not in COMBO_PORTS[type]]
    q.wires = [w for w in q.wires if w not in gone]
    toast = M("{t}: {ports}", t=title_msg(q, block), ports=join(COMBO_PORTS[type], ", "))
    if gone:
        toast = join([toast, M("removed: {gone}", gone=join(
            [M("wire from “{t}” on {port}", t=title_msg(q, w.src[0]), port=w.dst[1]) for w in gone], ", "))])
    return Out(toast, block)


def op_wire(q, src, src_port, dst, dst_port, channel=None):
    s, d = _blk(q, src), _blk(q, dst)
    ix = Index(q)
    if src_port not in out_ports(s):
        raise Refusal("lights first" if not out_ports(s) else M("no output {port}", port=src_port))
    if dst_port not in in_ports(d) or (d.kind != "post" and ix.pos.get(dst) != 0):
        raise Refusal(M("no input {port}", port=dst_port))
    if any(w.src == [src, src_port] and w.dst == [dst, dst_port] for w in q.wires):
        raise Refusal("this wire already exists")
    if would_cycle(q, src, dst):
        raise Refusal("this wire would close a loop")
    if d.kind == "post":
        ix = Index(q)
        if src in ix.row_of and not _flow_ends(q, ix, src) and not _is_layers_row(q, ix, ix.row_of[src]):
            return op_send_to_post(q, src)

    if d.kind != "post":
        q.wires = [w for w in q.wires if not (w.dst == [dst, dst_port] and w.active)]
    w = Wire(new_id(q, "w"), [src, src_port], [dst, dst_port])
    if d.kind == "combine":                 # a combination's input reads a channel of the image (grey by default)
        w.channel = _channel(channel or "L")
    q.wires.append(w)
    return Out(M("wire {a} → {b}", a=title_msg(q, src), b=title_msg(q, dst)), w.id)


def op_unwire(q, wire):
    w = next((x for x in q.wires if x.id == wire), None)
    if w is None:
        raise Refusal("wire not found")
    q.wires.remove(w)
    t = next((t for t in q.taps if not t.auto and w.wid is not None and t.wid == w.wid
              and q.blocks[w.dst[0]].kind == "post"), None)
    if t is not None and not any(x.wid == t.wid for x in q.wires):
        op_remove_tap(q, t.id)          # a block's image sent to Post: its workflow goes with its wire
    return Out("wire removed")


def op_take_alternative(q, wire):
    w = next((x for x in q.wires if x.id == wire), None)
    if w is None:
        raise Refusal("wire not found")
    if w.active:
        raise Refusal("this wire is already taken")
    for x in q.wires:
        if x.dst == w.dst:
            x.active = x is w
    return Out(M("{t} taken on {port}", t=title_msg(q, w.src[0]), port=w.dst[1]), w.id)


def _flow_ends(q, ix, b):
    """Whether a workflow ends at block b (a row's last block, a tap's)."""
    return any((_flow_chain(q, ix, f.id) or [None])[-1] == b for f in list(q.rows) + list(q.taps))


def _post_tap(q, b, name=None):
    """A mid-row block's image to Post (INTEGRATION 4.6): a workflow of its own, the row's chain up to the block then
    its Output (a tap), its settings those of the row's workflow; the wire to Post carries its id."""
    ix = Index(q)
    r = ix.row[ix.row_of[b]]
    t = Tap(new_id(q, "t"), b, "")
    t.name = (name or "").strip() or name_of("{_row} · {lab}", _row=r.name, lab=LABEL[q.blocks[b].kind])
    names = {x.name for x in q.rows} | {x.name for x in q.taps}
    if t.name in names:
        n = 2
        while f"{t.name} {n}" in names:
            n += 1
        t.name = f"{t.name} {n}"
    _copy_flow(t, r)
    t.legacy = None
    q.taps.append(t)
    _alloc(q)
    from ..core.project2 import Project2
    from .compile import compile_into
    p = Project2(name="tap")
    compile_into(q, p)
    rw, tw = p.workflow(r.wid), p.workflow(t.wid)
    extra = copy.deepcopy(t.extra)
    for st, kv in rw.params.items():                  # as a copy of the row's workflow: every setting it stores
        for k, v in kv.items():
            if (tw.params.get(st) or {}).get(k, object()) != v:
                extra.setdefault(st, {})[k] = copy.deepcopy(v)
    t.extra = extra
    post = ix.post
    q.wires.append(Wire(new_id(q, "w"), [b, out_ports(q.blocks[b])[0]], [post, POST_PORT], True, t.wid))
    return t


def op_send_to_post(q, block, name=None):
    """A block's image to Post: the end of a workflow wired to Post; a mid-row block's, a workflow of its own (a tap)
    up to it."""
    b = _blk(q, block)
    if not out_ports(b):
        raise Refusal("lights first")
    ix = Index(q)
    post = ix.post
    if any(w.src[0] == block and w.dst[0] == post for w in q.wires):
        raise Refusal("already in Post")
    if block in ix.row_of and not _flow_ends(q, ix, block) and not _is_layers_row(q, ix, ix.row_of[block]):
        if b.off:
            raise Refusal(M("{lab} is switched off: its image is its input's", lab=LABEL[b.kind]))
        _post_tap(q, block, name)
    else:
        _to_post(q, block)
    n = sum(1 for w in q.wires if w.dst[0] == post)
    return Out(M("{t} to Post ({n} layer)" if n == 1 else "{t} to Post ({n} layers)", t=title_msg(q, block), n=n), block)


def op_remove_tap(q, tap):
    """A block's image sent to Post taken back: its workflow goes (the row's stays)."""
    t = next((x for x in q.taps if x.id == tap), None)
    if t is None:
        raise Refusal("not found")
    q.taps.remove(t)
    q.wires = [w for w in q.wires if not (w.wid is not None and w.wid == t.wid)]
    q.order = [x for x in q.order if x != t.wid]
    return Out(M("“{_name}” out of Post", _name=t.name))


# ---------------------------------------------------------------------------------------------------- files, Commun
def _session(q, s):
    b = _blk(q, s)
    if b.kind != "session":
        raise Refusal("files go on a Session")
    return b


def _kind(k):
    if k not in FILE_KINDS:
        raise Refusal("unknown kind of files")


def _dedup(paths):
    out, seen = [], set()
    for x in paths:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


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
    n = len(paths)
    kd = rules.FILE_LABEL[kind].lower()
    return Out(M("{n} {kind} replace those of “{t}”" if mode == "replace" else "{n} {kind} added to “{t}”",
                 n=n, kind=kd, t=title_msg(q, session)), session)


def op_remove_files(q, session, kind, paths):
    s = _session(q, session)
    _kind(kind)
    rm = set(paths)
    left = [x for x in _eff_list(q, session, kind) if x not in rm]
    if left:
        s.files[kind] = left
    else:
        s.files.pop(kind, None)
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
    elif lst:
        q.blocks[s].files[kind] = lst
    else:
        q.blocks[s].files.pop(kind, None)


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
    dst = _list_of(q, dst_session, dst_kind)
    _set_list(q, dst_session, dst_kind, dst + [x for x in paths if x not in dst])
    where = "common" if dst_session == COMMON else M("“{t}”", t=title_msg(q, dst_session))
    return Out(M("{n} {kind} copied to {where}" if copy else "{n} {kind} moved to {where}",
                 n=len(paths), kind=rules.FILE_LABEL[dst_kind].lower(), where=where),
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
    for k in list(s.files):
        left = [x for x in s.files[k] if x not in rm]
        if left:
            s.files[k] = left
        else:
            del s.files[k]
    calib = {k: list(v) for k, v in s.files.items() if k != "lights"}
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
            nn = _night(q, name, {k: fl[k] for k in FILE_KINDS if k in fl})
            q.blocks[nn[1]].settings = dict(q.blocks[n[1]].settings)
            q.blocks[nn[1]].off = q.blocks[n[1]].off if into == "night" else False
            if into == "night":
                r.nights.insert(at, nn)
                at += 1
            else:
                nr = Row(new_id(q, "r"), name, _default_chain(q), [nn])
                _auto_align(q, nr)
                q.rows.insert(rows_at, nr)
                rows_at += 1
                _to_post(q, nr.blocks[-1])
            made += 1
            focus = nn[0]
        else:
            raise Refusal("unknown destination")
    return Out(M("Split into {n} sessions", n=made + 1) if made else "files distributed", focus)


# ---------------------------------------------------------------------------------------------------- engine settings
#: the stack stage's settings a Selection or a Crop owns (it has no stage of its own; convert.SELECTION_KEYS)
SELECTION_KEYS = ("min_transparency", "max_fwhm_ratio", "selection", "selection_transparency")


def _engine_name(q, b):
    """The engine node name of chain block b, given now when it has none (the one the compile gives it), so that its
    settings are stored under the stage they belong to."""
    blk = q.blocks[b]
    if blk.engine is None and blk.kind in wfl.IMAGE_OPS + ("registration", "selection", "crop", "stacking",
                                                            "astrometry", "stars", "synth"):
        from ..core.project2 import Project2
        from .compile import compile_into
        normalise(q)
        c = compile_into(q, Project2(name="names"))
        blk.engine = c.names.get(b) or blk.kind
    return blk.engine


def owned_stages(q, b):
    """The engine stages whose settings block b holds (INTEGRATION §2 rule 3): [stage]."""
    blk = q.blocks[b]
    k = blk.kind
    if k == "calibration":
        return [f"inventory@{blk.sid}", f"calibrate@{blk.sid}"] if blk.sid is not None else []
    if k == "registration":
        return ["sessions", "register"]
    if k in ("selection", "crop", "stacking"):
        return ["stack"]
    if k == "astrometry":
        return list(wfl.ASTROMETRY_STAGES)
    if k in wfl.IMAGE_OPS:
        return [_engine_name(q, b)]
    if k == "stars":
        return ["star_removal"]
    if k == "synth":
        return ["synth_stars"]
    if k == "combine":
        return [wfl.BLOCKS_OF_KIND.get(q.blocks[b].settings.get("type", "sho"), "combine")]
    if k == "image":
        return ["image"]
    return []


def check_param(stage, name, value, internal=False):
    """The value as the stage's settings read it (the engine's own check, api._check_param); Refusal naming the
    setting when it is not one, or not valid."""
    from .compile import _key_rules
    rules_ = _key_rules()
    base = wfl.base(stage).split("@")[0]
    if base not in rules_.chains.stages:
        raise Refusal(M("unknown stage {_stage}", _stage=stage))
    model = rules_.params(base)
    if name not in model.model_fields:
        raise Refusal(M("{_stage} has no setting {_name}", _stage=stage, _name=name))
    if not internal and (model.model_fields[name].json_schema_extra or {}).get("level") == "internal":
        raise Refusal(M("{_stage}.{_name} is set by the project", _stage=stage, _name=name))
    try:
        return model(**{name: value}).model_dump(mode="json")[name]
    except ValueError as e:
        raise Refusal(M("{_stage}.{_name}: not a valid value", _stage=stage, _name=name)) from e


def _put(store, stage, name, value):
    d = store.setdefault(stage, {})
    if value is None:
        d.pop(name, None)
    else:
        d[name] = value
    if not d:
        store.pop(stage, None)


def op_set_param(q, block, stage, name, value):
    """A setting of the engine (its real key, stored verbatim; None: back to the suggestion) on the block that owns
    it: a night's Calibration (calibrate, inventory of that night), Registration (register, sessions), Selection,
    Crop, Stacking (their keys of stack), Astrometry, an image block (its node), Stars/background, Synthetic stars, a
    combination, an Image."""
    b = _blk(q, block)
    if b.kind == "calibration" and b.sid is not None and stage in ("inventory", "calibrate"):
        stage = f"{stage}@{b.sid}"
    mine = owned_stages(q, block)
    if stage not in mine:
        raise Refusal(M("{lab} holds no setting of {_stage}", lab=LABEL.get(b.kind, b.kind), _stage=stage))
    if stage == "stack":
        own = "selection" if name in SELECTION_KEYS else "crop" if name == "zone" else "stacking"
        if b.kind != own:
            raise Refusal(M("{lab} holds no setting of {_stage}", lab=LABEL[b.kind], _stage=f"stack.{name}"))
    v = None if value is None else check_param(stage, name, value, internal=name == "zone")
    _put(b.params, stage, name, v)
    _unhide(_flows_through(q, Index(q), block), stage, name)
    return Out(M("{_name}: {_value}", _name=name, _value=str(v)), block)


def _flows_through(q, ix, b):
    """The rows and taps whose workflow goes through block b (its chain, or its stack's nights)."""
    out = []
    for f in list(q.rows) + list(q.taps):
        rid = f.id if isinstance(f, Row) else ix.row_of.get(f.block)
        if rid not in ix.row:
            continue
        root = ix.row[ix.root(rid)]
        if b in _flow_chain(q, ix, f.id) or any(b in n for n in root.nights):
            out.append(f)
    return out


def _unhide(flows, stage, name):
    """A setting stored on its holder shows: the workflows' leftovers holding the same key (applied after the
    holders at the compile) are taken out."""
    for f in flows:
        if f.extra and name in (f.extra.get(stage) or {}):
            d = {k: v for k, v in f.extra[stage].items() if k != name}
            f.extra = {k: v for k, v in f.extra.items() if k != stage} | ({stage: d} if d else {})
        if f.unset and [stage, name] in [list(u) for u in f.unset]:
            f.unset = [u for u in f.unset if list(u) != [stage, name]]


def op_set_row_param(q, row, stage, name, value):
    """A setting a stack holds for all its nights (calibrate, inventory: calibrated together), its Output's
    (render_simple), its Align on's (align_on)."""
    r = _row(q, row)
    ix = Index(q)
    root = ix.row[ix.root(row)]
    if stage in ("inventory", "calibrate"):
        if not root.nights:
            raise Refusal("a stack of lights holds this setting")
        store = root.calib
    elif stage == "render_simple":
        store = {"render_simple": r.output}
    elif stage == "align_on":
        if not r.align:
            raise Refusal("no Align on in this row")
        store = r.align.setdefault("params", {})
    else:
        raise Refusal(M("{_stage}: a block holds this setting", _stage=stage))
    v = None if value is None else check_param(stage, name, value)
    _put(store, stage, name, v)
    if stage == "render_simple":
        r.output = store.get("render_simple", {})
    if stage in ("inventory", "calibrate"):
        _unhide([f for f in list(q.rows) + list(q.taps)
                 if (f.id if isinstance(f, Row) else ix.row_of.get(f.block)) in ix.row
                 and ix.root(f.id if isinstance(f, Row) else ix.row_of[f.block]) == root.id], stage, name)
    else:
        _unhide([r], stage, name)
    return Out(M("{_name}: {_value}", _name=name, _value=str(v)), row)


def op_set_method(q, block, method):
    """The engine method of a block (None: its default), e.g. Gradient's polynomial, Denoise's pretrained network."""
    b = _blk(q, block)
    if b.kind in ("session", "calibration", "post", "image") + JOIN_KINDS:
        raise Refusal(M("{lab} has no method", lab=LABEL.get(b.kind, b.kind)))
    b.method = method
    return Out(M("{lab}: {_method}", lab=LABEL[b.kind], _method=str(method)), block)


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


# ---------------------------------------------------------------------------------------------------- combinations
def fusion_legacy():
    """A combination as the engine stores one made with no block after it: no block list, the default options (the
    first block placed after it gives it its blocks)."""
    return {"options": wfl.check_options({"denoise": None, "stars": None, "gradient": "polynomial",
                                          "deconvolution": None}), "end": None, "frozen": False}


def _channel(ch):
    from ..stages.combine import CHANNELS
    if ch not in CHANNELS:
        raise Refusal("unknown channel")
    return ch


def op_add_combination(q, type="sho", name=None):  # noqa: A002
    """A combination (its type's named inputs), empty, its image a layer of Post."""
    if type not in COMBO_PORTS:
        raise Refusal("unknown type")
    j = new_block(q, "combine", settings={"type": type} if type != "sho" else {})
    base = (name or "").strip() or name_of(COMBO_LABEL[type])
    r = Row(new_id(q, "r"), free_name(q, base), [j.id], legacy=fusion_legacy())
    q.rows.append(r)
    return Out(M("{_name} added", _name=r.name), j.id, new=j.id)


def _input(q, block, port):
    j = _blk(q, block, ("combine",))
    if port not in in_ports(j):
        raise Refusal("unknown port")
    w = next((w for w in q.wires if w.dst == [block, port] and w.active), None)
    if w is None:
        raise Refusal(M("{port}: no input wired", port=port))
    return j, w


def op_set_input(q, block, port, channel=None, weight=None, image=None):
    """A combination's input: the channel of its image it reads (L R G B GB), its weight after the balance (1: as
    balanced, a setting of the combination), the image it reads of its workflow ("output", "starless", "stars")."""
    j, w = _input(q, block, port)
    said = []
    if channel is not None:
        w.channel = _channel(channel)
        said.append(M("channel {ch}", ch=channel))
    if weight is not None:
        try:
            weight = round(float(weight), 3)
        except (TypeError, ValueError):
            raise Refusal("a weight is a number") from None
        if not 0 < weight <= 100:
            raise Refusal("a weight is a number") from None
        st = owned_stages(q, block)[0]
        cur = dict((j.params.get(st) or {}).get("weights") or {})
        role = _role(port)
        cur[role] = weight
        cur = {k: v for k, v in cur.items() if v != 1.0}
        _put(j.params, st, "weights", cur or None)
        said.append(M("weight {w}", w=weight))
    if image is not None:
        _reimage(q, w, image)
        said.append(M("image {im}", im=image))
    if not said:
        raise Refusal("nothing to change")
    return Out(join([M("{port}:", port=port)] + said, " "), block)


def _role(port):
    from .compile import ROLE_OF_PORT
    return ROLE_OF_PORT.get(port, port)


def _reimage(q, w, image):
    """The input reads another image of the same workflow: its Output, or its Stars/background's starless image or
    stars alone."""
    ix = Index(q)
    flows = list(q.rows) + list(q.taps)
    f = next((f for f in flows if w.wid is not None and f.wid == w.wid), None) or         next((f for f in flows if (_flow_chain(q, ix, f.id) or [None])[-1] == w.src[0]), None)
    if f is None:
        raise Refusal("block not found")
    chain = _flow_chain(q, ix, f.id)
    if image in ("starless", "stars"):
        b = next((x for x in chain if q.blocks[x].kind == "stars"), None)
        if b is None:
            raise Refusal("needs Stars/background")
        port = "Starless" if image == "starless" else "Stars only"
    elif image == "output":
        b = next((x for x in reversed(chain) if "image" in out_ports(q.blocks[x])), None)
        if b is None:
            raise Refusal("block not found")
        port = "image"
    else:
        raise Refusal("unknown image")
    w.src = [b, port]
    w.wid = None if chain[-1] == b else f.wid
    w.image = None


def op_add_stars_combination(q, block):
    """“<name> · Stars”: a second combination of the same type reading the stars alone of the Stars/background its
    inputs' starless images come from, each input with its channel; their two images are put together in Post."""
    j = _blk(q, block, ("combine",))
    ix = Index(q)
    ins = [w for w in q.wires if w.dst[0] == block and w.active and w.src[1] == "Starless"]
    if not ins:
        raise Refusal("its inputs read no starless image")
    r = ix.row[ix.row_of[block]]
    t = j.settings.get("type", "sho")
    nj = new_block(q, "combine", settings={"type": t} if t != "sho" else {})
    nr = Row(new_id(q, "r"), free_name(q, name_of("{_name} · Stars", _name=r.name)), [nj.id], legacy=fusion_legacy())
    q.rows.insert(q.rows.index(r), nr)
    for w in ins:
        q.wires.append(Wire(new_id(q, "w"), [w.src[0], "Stars only"], [nj.id, w.dst[1]], True, w.wid, w.channel))
    return Out(M("{_name} added", _name=nr.name), nj.id, new=nj.id)


# ---------------------------------------------------------------------------------------------------- Replace or Try
def _clone_block(q, b):
    """A copy of chain block b with all it holds (settings, engine settings, method, switch, pause); its node name is
    given by the compile in its own chain."""
    o = q.blocks[b]
    return new_block(q, o.kind, settings=dict(o.settings), off=o.off, pause=o.pause, method=o.method,
                     params=copy.deepcopy(o.params), source=copy.deepcopy(o.source)).id


def _copy_flow(dst, src):
    for k in ("all_nights", "sessions", "output", "extra", "unset", "pauses", "options", "end", "legacy"):
        setattr(dst, k, copy.deepcopy(getattr(src, k)))


def _apply_change(q, b, change):
    """A change of the Replace or Try panel on block b: {"param": [stage, name, value]} | {"method": m} | {"off": bool}
    | {"setting": [key, value]}."""
    if "param" in change:
        stage, name, value = change["param"]
        op_set_param(q, b, stage, name, value)
    elif "method" in change:
        op_set_method(q, b, change["method"])
    elif "off" in change:
        if bool(change["off"]) != q.blocks[b].off:
            op_switch_off(q, b)
    elif "setting" in change:
        op_set_setting(q, b, *change["setting"])
    else:
        raise Refusal("unknown change")


def op_try_variant(q, block, change, name=None):
    """Try as variant (INTEGRATION §11): a copy of the workflow through block with the change made on the copy of the
    block, its trial (variant_of) the original's id; both share what comes before the block. A block before which no
    variant forks (a Registration, a night's Calibration, the first block of a stack) makes a whole copy of the stack,
    drawing the same nights."""
    b = _blk(q, block)
    ix = Index(q)
    if b.kind in ("session", "post", "image") + JOIN_KINDS:
        raise Refusal(M("{lab}: no variant from here", lab=LABEL.get(b.kind, b.kind)))
    if b.kind == "calibration":
        rid = ix.row_of[block]
        pos = None
    else:
        rid = ix.row_of[block]
        pos = ix.pos[block]
    r = ix.row[rid]
    if r.wid is None:
        raise Refusal("row not found")
    base = name if (name or "").strip() else name_of("{_row} · variant", _row=r.name)
    pred = ix.pred(block) if pos is not None else None
    if pos is not None and pred is not None and b.kind not in ("registration",):
        frow = ix.row_of[pred]
        tail = r.blocks[pos:] if frow == rid else list(r.blocks)
        copies = [_clone_block(q, t) for t in tail]
        out = _make_variant(q, frow, pred, copies, tail, r.name)
        v = q.rows[[x.id for x in q.rows].index(Index(q).row_of[copies[0]])]
        target = copies[tail.index(block)]
    else:                                                   # the whole stack, drawing the same nights
        root = ix.row[ix.root(rid)]
        if not root.nights or r.fork:
            raise Refusal(M("{lab}: no variant from here", lab=LABEL.get(b.kind, b.kind)))
        copies = [_clone_block(q, t) for t in root.blocks]
        nights = []
        target = None
        for s, c in root.nights:
            nn = _night(q, q.blocks[s].name, {k: list(v_) for k, v_ in q.blocks[s].files.items()})
            q.blocks[nn[0]].sid = q.blocks[nn[1]].sid = q.blocks[s].sid
            q.blocks[nn[1]].off, q.blocks[nn[1]].own = q.blocks[c].off, copy.deepcopy(q.blocks[c].own)
            q.blocks[nn[1]].params = copy.deepcopy(q.blocks[c].params)
            nights.append(nn)
            if c == block:
                target = nn[1]
        v = Row(new_id(q, "r"), base, copies, nights)
        v.calib_node, v.calib_fork, v.calib = root.calib_node, root.calib_fork, copy.deepcopy(root.calib)
        v.align = copy.deepcopy(root.align)
        q.rows.insert(_after_subtree(q, root.id), v)
        for nb, ob in zip(copies, root.blocks):
            for w in list(q.wires):
                if w.src[0] == ob:
                    q.wires.append(Wire(new_id(q, "w"), [nb, w.src[1]], list(w.dst), q.blocks[w.dst[0]].kind == "post"))
        if target is None:
            target = copies[root.blocks.index(block)]
        out = Out(M("variant “{_name}” under “{_base}”", _name=base, _base=r.name), copies[0] if copies else nights[0][0],
                  new=copies[0] if copies else nights[0][0])
    v.name = ""
    v.name = free_name(q, base)
    _copy_flow(v, r)
    v.legacy = None
    v.trial = r.wid
    _apply_change(q, target, change)
    out.toast = M("variant “{_name}”: Keep or Discard", _name=v.name)
    out.focus = target
    return out


def _variant_parts(q, ix, v):
    """For variant row v of a trial: (the original row, its blocks v replaces)."""
    o = next((r for r in q.rows if r.wid == v.trial and r is not v), None)
    if o is None:
        raise Refusal("its original is gone: it is a workflow of its own")
    if v.fork and v.fork[0] == o.id:
        i = o.blocks.index(v.fork[1]) + 1
        return o, o.blocks[i:]
    if (v.fork and o.fork and v.fork == o.fork) or (not v.fork and not o.fork and v.nights and o.nights):
        return o, list(o.blocks)
    raise Refusal("its original is gone: it is a workflow of its own")


def op_keep_variant(q, row):
    """Keep: the variant replaces its original and takes its name; the original's workflow id is retired (Post
    layers, results and marks of the variant's id stay); the variants of the original read the variant's blocks."""
    v = _row(q, row)
    if v.trial is None:
        raise Refusal("not a variant on trial")
    ix = Index(q)
    o, old = _variant_parts(q, ix, v)
    mine = list(v.blocks)
    m = {a: b for a, b in zip(old, mine) if q.blocks[a].kind == q.blocks[b].kind}
    inside = bool(v.fork and v.fork[0] == o.id)          # the variant's blocks continue the original's row
    for c in ix.children[o.id]:
        f = ix.row[c].fork[1]
        if f in old and c != v.id:
            if f not in m:
                raise Refusal(M("a variant of “{_row}” starts from a block the variant does not have", _row=o.name))
            ix.row[c].fork = [o.id if inside else v.id, m[f]]
    for t in q.taps:
        if t.block in old:
            if t.block not in m:
                raise Refusal(M("a block of “{_row}” sent to Post is not in the variant", _row=o.name))
            t.block = m[t.block]
    gone = set(old)
    for w in list(q.wires):
        if w.src[0] in gone:
            twin = next((x for x in q.wires if x.src[0] == m.get(w.src[0]) and x.dst == w.dst), None)
            if twin is not None and w.active:
                twin.active = True
            q.wires.remove(w)
    for r in q.rows:
        if r.align and r.align.get("from") in gone:
            r.align["from"] = m.get(r.align["from"])
    retired, name, trial = o.wid, o.name, o.trial
    if inside:
        o.blocks = o.blocks[:len(o.blocks) - len(old)] + mine
        for c in ix.children[v.id]:
            ix.row[c].fork = [o.id, ix.row[c].fork[1]]
        _copy_flow(o, v)
        o.wid, o.trial = v.wid, trial
        if o.align and o.align.get("ref") == v.id:      # aligned on its own variant: it takes the variant's
            o.align = copy.deepcopy(v.align) if v.align and v.align.get("ref") != o.id else None
        for x in q.rows:
            if x.align and x.align.get("ref") == v.id:
                x.align["ref"] = o.id
        q.rows.remove(v)
        kept = o
    else:                                               # the variant takes the original's place
        at = q.rows.index(o)
        for c in ix.children[o.id]:
            if ix.row[c].fork[0] == o.id and ix.row[c].fork[1] not in gone:
                raise Refusal(M("a variant of “{_row}” starts from a block the variant does not have", _row=o.name))
        for s, c in o.nights:
            del q.blocks[s], q.blocks[c]
        if v.align and v.align.get("ref") == o.id:      # aligned on its original: it takes the original's
            v.align = copy.deepcopy(o.align) if o.align and o.align.get("ref") != v.id else None
        for x in q.rows:
            if x.align and x.align.get("ref") == o.id:
                x.align["ref"] = v.id
        q.rows.remove(v)
        q.rows.insert(at, v)
        q.rows.remove(o)
        v.name, v.trial = name, trial
        kept = v
    for b in gone:
        q.blocks.pop(b, None)
    for f in list(q.rows) + list(q.taps):
        if f.trial == retired:
            f.trial = kept.wid
    q.order = [x for x in q.order if x != retired]
    return Out(M("variant kept: it replaces “{_name}”", _name=name), kept.blocks[0] if kept.blocks else None)


def op_discard_variant(q, row):
    """Discard: the variant goes; its own variants on trial point at its original."""
    v = _row(q, row)
    if v.trial is None:
        raise Refusal("not a variant on trial")
    ix = Index(q)
    if any(ix.row[c].fork[0] == v.id for c in ix.children[v.id]):
        raise Refusal("variants start from this block: delete them first")
    for f in list(q.rows) + list(q.taps):
        if f.trial == v.wid:
            f.trial = v.trial
    name = v.name
    delete_rows(q, [v.id])
    q.order = [x for x in q.order if x != v.wid]
    return Out(M("variant “{_name}” discarded", _name=name))


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


def _file_name(path):
    return str(path).replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]


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
    return Out(M("{n} {kind} left out of “{_name}”" if excluded else "{n} {kind} taken back in “{_name}”",
                 n=len(paths), kind=rules.FILE_LABEL[kind].lower(), _name=e.get("name", "")), session)


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


def _flow_of(q, fid):
    """The row or tap `fid`."""
    f = next((x for x in list(q.rows) + list(q.taps) if x.id == fid), None)
    if f is None:
        raise Refusal("row not found")
    return f


def op_rename_tap(q, tap, name):
    """A block's image sent to Post (or a copy of a workflow) renamed."""
    t = next((x for x in q.taps if x.id == tap), None)
    if t is None:
        raise Refusal("not found")
    name = (name or "").strip()
    if not name:
        raise Refusal("a name cannot be empty")
    if name != t.name and name in {x.name for x in list(q.rows) + list(q.taps) if x is not t}:
        raise Refusal(M("“{_name}” already exists", _name=name))
    t.name = name
    return Out(M("renamed “{_name}”", _name=name), t.block)


def op_set_flow_pauses(q, flow, pauses):
    """A workflow's own pause list (the engine's block names: "calibration@2", "align_on", "output"...), for the pauses
    its blocks do not give (a night's Calibration, an Align on, its Output, a block shared with another workflow);
    None: the pauses of its blocks."""
    f = _flow_of(q, flow)
    if pauses is not None and (isinstance(pauses, str) or not isinstance(pauses, (list, tuple))
                               or any(not isinstance(x, str) or not x for x in pauses)):
        raise Refusal("a pause list names blocks")
    f.pauses = None if pauses is None else list(pauses)
    if pauses is None:
        return Out(M("check pauses of “{_name}”: those of its blocks", _name=f.name))
    return Out(M("check pauses of “{_name}”: {n}", _name=f.name, n=len(f.pauses)))


def op_set_flow_options(q, flow, options=None, end=None):
    """A workflow made from options (as before the free graph: kept as its options until the first change of its
    chain), or one of its blocks again (options None); `end`: its early exit."""
    f = _flow_of(q, flow)
    try:
        end = wfl.check_end(end)
        opts = None if options is None else wfl.check_options(dict(options))
    except (ValueError, TypeError) as e:
        raise Refusal("unknown options") from e
    if opts is None:
        f.legacy, f.end = None, end
        return Out(M("“{_name}” follows its blocks", _name=f.name))
    f.legacy = {"options": opts, "end": end, "frozen": False}
    return Out(M("“{_name}” follows its options", _name=f.name))


def _stage_names(q):
    """The node name of every chain block (stored, else the compile's)."""
    from ..core.project2 import Project2
    from .compile import compile_into
    normalise(q)
    return compile_into(q, Project2(name="names")).names


def _stages_of(q, b, names):
    blk = q.blocks[b]
    if blk.kind in wfl.IMAGE_OPS:
        return [blk.engine or names.get(b) or blk.kind]
    return owned_stages(q, b)


def _holder_of(q, ix, f, stage, name, names):
    """Where workflow f's setting (stage, name) is stored: ("block", id, stage) | ("calib", root row) | ("output", f)
    | ("align", row) | ("extra", f)."""
    base, at, sid = stage.partition("@")
    rid = f.id if isinstance(f, Row) else ix.row_of.get(f.block)
    root = ix.row[ix.root(rid)] if rid in ix.row else None
    if base in ("inventory", "calibrate") and at:
        c = next((c for _s, c in (root.nights if root else []) if str(q.blocks[c].sid) == sid), None)
        return ("block", c, base) if c is not None else ("extra", f)
    if stage in ("inventory", "calibrate"):
        return ("calib", root) if isinstance(f, Row) and root is not None else ("extra", f)
    if stage == "render_simple":
        return ("output", f) if isinstance(f, Row) else ("extra", f)
    if stage == "align_on":
        return ("align", f) if isinstance(f, Row) and f.align else ("extra", f)
    chain = _flow_chain(q, ix, f.id)
    if stage == "stack":
        own = "selection" if name in SELECTION_KEYS else "crop" if name == "zone" else "stacking"
        b = next((b for b in chain if q.blocks[b].kind == own), None)
    else:
        b = next((b for b in chain if stage in _stages_of(q, b, names)), None)
    return ("block", b, stage) if b is not None else ("extra", f)


def _store(q, ix, f, stage, name, v, names):
    """Workflow f's setting stored on its holder; -> (the stage key it is stored under, the workflows it shows in)."""
    h = _holder_of(q, ix, f, stage, name, names)
    if h[0] == "block":
        blk = q.blocks[h[1]]
        st = f"{h[2]}@{blk.sid}" if blk.kind == "calibration" else h[2]
        _put(blk.params, st, name, v)
        return st, _flows_through(q, ix, h[1])
    if h[0] == "calib":
        root = h[1]
        _put(root.calib, stage, name, v)
        return stage, [x for x in list(q.rows) + list(q.taps)
                       if (x.id if isinstance(x, Row) else ix.row_of.get(x.block)) in ix.row
                       and ix.root(x.id if isinstance(x, Row) else ix.row_of[x.block]) == root.id]
    if h[0] == "output":
        store = {"render_simple": f.output}
        _put(store, stage, name, v)
        f.output = store.get("render_simple", {})
        return stage, [f]
    if h[0] == "align":
        _put(f.align.setdefault("params", {}), stage, name, v)
        return stage, [f]
    _put(f.extra, stage, name, v)
    return stage, []


def op_set_flow_param(q, flow, stage, name, value):
    """A workflow's setting (the engine's real key; None: back to the suggestion), stored on what holds it: the block
    of its chain owning the stage, its night's Calibration, its stack, its Output, its Align on; else its leftovers.
    The leftovers that would hide it are taken out."""
    f = _flow_of(q, flow)
    if not isinstance(stage, str) or not isinstance(name, str):
        raise Refusal("unknown setting")
    v = None if value is None else check_param(stage, name, value, internal=name == "zone")
    names = _stage_names(q)
    ix = Index(q)
    st, through = _store(q, ix, f, stage, name, v, names)
    _unhide(through, st, name)
    if not through:
        f.unset = [u for u in f.unset or [] if list(u) != [stage, name]]
    return Out(M("{_name}: {_value}", _name=name, _value=str(v)))


def _store_all(q, f, params):
    if not isinstance(params, dict) or any(not isinstance(kv, dict) for kv in params.values()):
        raise Refusal("unknown setting")
    names = _stage_names(q)
    ix = Index(q)
    for stage, kv in params.items():
        for k, v in kv.items():
            if v is not None:
                try:                         # a value the engine refuses is not stored (one it does not know is)
                    check_param(str(stage), str(k), v, internal=True)
                except Refusal as e:
                    if isinstance(e.msg, tuple) and e.msg[0] == "{_stage}.{_name}: not a valid value":
                        raise
            _store(q, ix, f, str(stage), str(k), copy.deepcopy(v), names)


def op_store_params(q, flow, params):
    """A workflow's settings {stage: {name: value}} stored as given (as the engine stored them: no check) where they
    belong (see set_flow_param): a workflow made, or its chain replaced, keeps its settings."""
    f = _flow_of(q, flow)
    _store_all(q, f, params)
    return Out(M("settings of “{_name}” stored", _name=f.name))


def op_replace_chain(q, row, blocks, calibration=None, align=None, options=None, params=None):
    """A row's chain made of these blocks ([{"kind", "method", "off", "fork", "engine"}], its head Image or
    Combination kept): a template given to a workflow. `calibration`: None keeps the stack's Calibration node, False
    none, {"off": bool} one; `align`: None keeps its Align on (gone without a Stacking), {"ref": row or tap | None,
    "method", "off"} a new one; `options`: the row kept as these options (made before the free graph), None: of its
    blocks; `params`: its settings stored again where the new blocks hold them (its leftovers emptied first)."""
    from .compile import NODE_KINDS
    r = _row(q, row)
    ix = Index(q)
    if r.fork or ix.children[r.id] or any(t.block in r.blocks for t in q.taps):
        raise Refusal("variants or blocks sent to Post start from this row: delete them first")
    if isinstance(blocks, (str, dict)) or any(not isinstance(n, dict) or n.get("kind") not in NODE_KINDS
                                              for n in blocks):
        raise Refusal("unknown block")
    head = [b for b in r.blocks[:1] if q.blocks[b].kind in ("image", "combine", "combine_layers")]
    hk = q.blocks[head[0]].kind if head else None
    if hk == "combine_layers" and blocks:
        raise Refusal(rules.AFTER_LAYERS)
    frame = next((n["kind"] for n in blocks if n["kind"] in FRAME_KINDS), None)
    if frame and not r.nights:
        raise Refusal(M("{lab} belongs to a stack: between the Session and the Stacking", lab=LABEL[frame]))
    if options is not None:
        if hk in JOIN_KINDS:                 # the engine would run the options, the drawing its blocks
            raise Refusal("a combination is made of its blocks")
        try:
            options = wfl.check_options(dict(options))
        except (ValueError, TypeError) as e:
            raise Refusal("unknown options") from e
        end = r.legacy.get("end") if r.legacy else r.end
        src = "frames" if not head else "stretched" if (q.blocks[head[0]].source or {}).get(
            "stretched", rules.effective(q, head[0]).get("Stretched image")) else "image"
        want = [n["block"] for n in wfl.from_options(options, end, src)
                if not n["off"] and n["block"] not in ("image", "output", "align_on", "calibration")]
        kinds = [n["kind"] for n in blocks]
        if kinds not in (want, [k for k in want if k != "color"]):     # a mono stack draws no Color
            raise Refusal("the blocks are not those of the options")     # (the engine runs the options)
    gone = set(r.blocks) - set(head)
    new = [new_block(q, n["kind"], method=n.get("method"), off=bool(n.get("off")), fork=n.get("fork"),
                     engine=(n.get("engine") or None) if n["kind"] in wfl.IMAGE_OPS else None).id for n in blocks]
    for b in gone:
        del q.blocks[b]
    q.wires = [w for w in q.wires if w.src[0] not in gone and w.dst[0] not in gone]
    r.blocks = head + new
    if r.nights and calibration is not None:
        r.calib_node = calibration is not False
        if calibration is not False:
            off = bool(calibration.get("off")) if isinstance(calibration, dict) else False
            if off != any(q.blocks[c].off for _s, c in r.nights):
                for _s, c in r.nights:
                    q.blocks[c].off = off
    if align is not None:
        if not isinstance(align, dict):
            raise Refusal("unknown place")
        ref = align.get("ref")
        if ref is not None and ref not in {x.id for x in q.rows} | {t.id for t in q.taps}:
            raise Refusal("row not found")
        r.align = {**new_align(ref), "method": align.get("method"), "off": bool(align.get("off"))}
    elif r.align and not any(q.blocks[b].kind in ("stacking", "image") for b in r.blocks):
        r.align = None
    if options is None:
        r.legacy = None
    else:
        r.legacy = {**(r.legacy or {"end": r.end, "frozen": False}), "options": options}
        q.__dict__.setdefault("_legacy_kept", set()).add(r.id)
    if params is not None:
        r.extra, r.unset = {}, []
        _store_all(q, r, params)
    return Out(M("“{_row}”: its blocks replaced", _row=r.name), r.blocks[0] if r.blocks else None,
               new=r.blocks[0] if r.blocks else None)


def op_add_branch(q, block, kind, flow=None, name=None, method=None):
    """A new branch: a workflow with the settings of `flow` (the row of `block` by default), its chain up to `block`,
    then a new block of `kind` from its defaults, drawn as a variant forking after `block`. After a night's
    Calibration (kind "calibration"): a stack drawing the same nights, that night's Calibration its own (placed in
    parallel), the others as in the stack it leaves."""
    b = _blk(q, block)
    ix = Index(q)
    if kind not in LABEL:
        raise Refusal("unknown block")
    if block not in ix.row_of or b.kind in ("session", "post") or _is_layers_row(q, ix, ix.row_of[block]):
        raise Refusal(M("{lab}: no variant from here", lab=LABEL.get(b.kind, b.kind)))
    if (b.kind == "calibration") != (kind == "calibration"):
        raise Refusal(M("{lab}: no variant from here", lab=LABEL.get(b.kind, b.kind)))
    if kind in ("session", "post", "image") + JOIN_KINDS:
        raise Refusal(M("{lab} is not inserted in a row", lab=LABEL[kind]))
    rid = ix.row_of[block]
    src = _flow_of(q, flow) if flow is not None else ix.row[rid]
    if flow is not None:
        sroot = ix.root(src.id if isinstance(src, Row) else ix.row_of.get(src.block, rid))
        if block not in _flow_chain(q, ix, src.id) and not (b.kind == "calibration" and sroot == rid):
            raise Refusal("the block is not in this workflow")
    if method is not None and method not in wfl.METHODS.get(kind, ()):
        raise Refusal("unknown method")
    base = (name or "").strip() or name_of("{_row} · {lab}", _row=src.name, lab=LABEL[kind])
    if b.kind == "calibration":
        root = ix.row[rid]
        nights, mine = [], None
        for s, c in root.nights:
            nn = _night(q, q.blocks[s].name, {k: list(v) for k, v in q.blocks[s].files.items()})
            q.blocks[nn[0]].sid = q.blocks[nn[1]].sid = q.blocks[s].sid
            q.blocks[nn[1]].own = copy.deepcopy(q.blocks[c].own)
            nights.append(nn)
            if c == block:
                mine = nn[1]
        v = Row(new_id(q, "r"), free_name(q, base), [], nights)
        v.calib_node, v.calib_fork = True, None
        new = mine
        at = _after_subtree(q, root.id)
    else:
        nb = new_block(q, kind, method=method)
        v = Row(new_id(q, "r"), free_name(q, base), [nb.id], [], [rid, block])
        new = nb.id
        at = _after_subtree(q, rid)
    _copy_flow(v, src)
    v.legacy, v.end, v.trial, v.align = None, None, None, None
    q.rows.insert(at, v)
    _alloc(q)
    if b.kind == "calibration":
        q.blocks[new].own = {"fork": v.wid}
    elif kind in FRAME_KINDS:
        q.blocks[new].fork = v.wid            # a frame block drawn on its own, even while it computes the same
    from ..core.project2 import Project2
    from .compile import compile_into
    p = Project2(name="branch")
    c = compile_into(q, p)
    sw, vw = p.workflow(c.wids[src.id]), p.workflow(v.wid)
    extra = copy.deepcopy(v.extra)
    for st, kv in sw.params.items():                  # as a copy of the workflow it leaves: every setting it stores
        for k, val in kv.items():
            if (vw.params.get(st) or {}).get(k, object()) != val:
                extra.setdefault(st, {})[k] = copy.deepcopy(val)
    fresh = {st for st in extra if st.split("@")[0] in ("inventory", "calibrate")} if b.kind == "calibration" \
        else set(_stages_of(q, new, c.names))
    v.extra = {st: kv for st, kv in extra.items() if st not in fresh}     # its new block: from its defaults
    if vw.breakpoints:
        v.pauses = []                                 # a new workflow pauses nowhere
    return Out(M("branch “{_name}” from “{_base}”", _name=v.name, _base=src.name), new, new=new)


def dependants(q, block):
    """The blocks a removal of chain block `block` takes with it (those after it, in its row and its variants, that
    need it), in the order they go; [] when it goes alone, or when taking later blocks does not make it possible.
    Nothing changes in q."""
    if not isinstance(block, str) or block not in q.blocks:
        return []
    ix = Index(q)
    if block not in ix.row_of or q.blocks[block].kind in ("session", "calibration", "post") or \
            ix.pos.get(block) is None:
        return []
    rid = ix.row_of[block]
    cands = [x for r in ix.subtree(rid) for x in (ix.row[r].blocks[ix.pos[block] + 1:] if r == rid
                                                     else ix.row[r].blocks)]
    fed = True
    while fed:                                  # the joins its images feed, and what follows them
        fed = False
        for w in q.wires:
            d = w.dst[0]
            if w.src[0] in [block] + cands and d not in cands and d != ix.post and d in ix.row_of:
                cands += [x for r in ix.subtree(ix.row_of[d]) for x in ix.row[r].blocks if x not in cands]
                fed = True
    c = clone(q)
    try:
        op_remove_block(c, block)
        normalise(c)
    except Refusal:
        return []
    why, deps = legality(c), []
    while why and cands:
        text = say_text(why)
        live = [d for d in cands if d in c.blocks and d not in deps]
        tried = None
        for d in sorted(live, key=lambda d: say_text(rules.title_msg(c, d)) not in text):    # the one named first
            c2 = clone(c)
            try:
                op_remove_block(c2, d)
                normalise(c2)
            except Refusal:
                continue
            why2 = legality(c2)
            if not why2 or say_text(why2) != text:
                tried = (d, c2, why2)
                break
        if tried is None:
            return []
        deps.append(tried[0])
        c, why = tried[1], tried[2]
    return deps if not why else []


def say_text(m):
    from .msg import say
    return say(m)


def op_remove_with_dependants(q, block):
    """A block deleted with the blocks after it that need it (one step)."""
    b = _blk(q, block)
    deps = dependants(q, block)
    out = op_remove_block(q, block)
    for d in deps:
        if d in q.blocks:
            op_remove_block(q, d)
    if deps:
        out.toast = M("{lab} deleted with {n} block that needs it" if len(deps) == 1
                      else "{lab} deleted with {n} blocks that need it", lab=LABEL[b.kind], n=len(deps))
    return out


OPS = {k[3:]: v for k, v in dict(globals()).items() if k.startswith("op_")}
