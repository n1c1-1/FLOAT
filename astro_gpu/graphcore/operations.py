"""Operations (§4). Every operation runs on a clone of the graph; refused = nothing changes. No history here:
the caller keeps the graphs it wants to go back to."""
from dataclasses import dataclass

from .shapes import Result
from .model import (Index, Row, Wire, check, clone, model_json, new_block, new_id, in_ports, out_ports, edges,
                    FILE_KINDS, FRAME_KINDS, JOIN_KINDS, COMBO_PORTS, COMBO_LABEL, LAYER_PORTS, POST_PORT)
from . import rules
from .msg import M, join, typed, name_of, TR
from .rules import LABEL, OFFABLE, evaluate, need_greyed, title_msg, inherited, rank

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
            o = fn(q, **args)
            o.created = _created(q, ids)
            outs.append(o)
        finish(q, outs[-1].style if outs else None)
        fold_guard(graph, q, view)
    except Refusal as e:
        return graph, Result(False, reason=e.msg)
    out = outs[-1]
    toast = join([o.toast for o in outs])
    acted = _acted(graph, view, q, out)
    created = {k: [x for o in outs for x in o.created[k] if x in _ids(q)[k]] for k in ("rows", "blocks", "wires")}
    res = Result(True, toast=toast, focus=out.focus, acted=acted, created=_summarise(q, created))
    return (q if json_of(q) != json_of(graph) else graph), res


def _ids(q):
    return {"rows": {r.id for r in q.rows}, "blocks": set(q.blocks), "wires": {w.id for w in q.wires}}


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
    """Normalise what has one stored form, check structure, refuse any need left unmet."""
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
    frames = [r.id for r in q.rows if r.nights]
    if q.reference not in frames:
        q.reference = frames[0] if frames else None
    for b in q.blocks.values():
        inh = inherited(q, b.kind)
        b.settings = {k: v for k, v in sorted(b.settings.items()) if inh.get(k, object()) != v}
        if b.files is not None:
            b.files = {k: b.files[k] for k in FILE_KINDS if k in b.files
                       and b.files[k] != q.common["files"].get(k, [])}
    q.common["files"] = {k: q.common["files"][k] for k in FILE_KINDS if q.common["files"].get(k)}
    q.common["settings"] = {k: dict(sorted(v.items())) for k, v in q.common["settings"].items() if v}
    check(q)
    ix = Index(q)
    bad = need_greyed(evaluate(q, ix))
    if not bad:
        for r in q.rows:
            why = rules.chain_problem([q.blocks[b].kind for b in ix.effective(r.id)], rules.root_kind(q, ix, r.id))
            if why:
                raise Refusal(why)
    if bad:
        b, why = next(iter(bad.items()))
        t = title_msg(q, b)
        if style == "need":
            raise Refusal(M("{t} needs it", t=t))
        if style == "off":
            raise Refusal(M("{t} needs it: disable {t} first", t=t))
        raise Refusal(why)


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
    """Delete rows with their subtrees, blocks and wires; the reference passes on with its Astrometry."""
    ix = Index(q)
    q._astro_moved = False
    gone = []
    for r in rids:
        for x in ix.subtree(r):
            if x not in gone:
                gone.append(x)
    if q.reference in gone:
        left = [r.id for r in q.rows if r.nights and r.id not in gone]
        astro = [b for b in ix.row[q.reference].blocks if q.blocks[b].kind == "astrometry"]
        if left and astro:
            ix.row[q.reference].blocks.remove(astro[0])
            _insert_after_frames(q, ix.row[left[0]], astro[0])
            q._astro_moved = True
        q.reference = left[0] if left else None
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
    if i == 0 and len(r.blocks) > 1 and kids:
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
    if not r.blocks:
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
        q.rows.append(r)
        _to_post(q, r.blocks[-1])
        if q.reference is None:
            q.reference = r.id
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
    r.nights.append(_night(q, name, _files(files)))
    return Out(M("{_name} added to “{_row}”: it stacks with its sessions · Ctrl+Z to undo", _name=name, _row=r.name),
               r.nights[-1][0], new=r.nights[-1][0])


def _night_row(q, night):
    for r in q.rows:
        for n in r.nights:
            if n[0] == night:
                return r, n
    raise Refusal("session not found")


def op_remove_night(q, night):
    r, n = _night_row(q, night)
    if len(r.nights) == 1:
        toast = delete_rows(q, [r.id])
        return Out(toast, old=None if q._astro_moved else night, style="need")
    r.nights.remove(n)
    for b in n:
        del q.blocks[b]
    return Out("session removed", old=night, style="need")


def op_move_night(q, night, row=None, before=None):
    """Into stack `row` (at its end, or before its session `before`); row None = its own new row."""
    r, n = _night_row(q, night)
    name = q.blocks[night].name
    if row is None:
        if len(r.nights) == 1:
            raise Refusal("already alone on its row")
        r.nights.remove(n)
        nr = Row(new_id(q, "r"), free_name(q, name), _default_chain(q), [n])
        q.rows.append(nr)
        _to_post(q, nr.blocks[-1])
        return Out(M("“{_name}” moves to its own row “{_row}”", _name=name, _row=nr.name), night, new=night)
    t = _row(q, row)
    if not t.nights:
        raise Refusal("a session joins a stack of lights")
    if before is not None and before not in [x[0] for x in t.nights]:
        raise Refusal("session not found in this stack")
    if t is r and (before is None or before == night):
        if before is None and t.nights[-1] is not n:
            t.nights.remove(n)
            t.nights.append(n)
        return Out(M("“{_name}” moved", _name=name), night, new=night)
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
    own = q.blocks[night].files.get("lights", [])
    if not lights or any(x not in own for x in lights):
        raise Refusal("these lights are not in the session")
    q.blocks[night].files["lights"] = [x for x in own if x not in lights]
    files = {k: list(v) for k, v in q.blocks[night].files.items() if k != "lights"}
    files["lights"] = list(lights)
    nn = _night(q, session_name(q), files)
    q.blocks[nn[1]].settings = dict(q.blocks[n[1]].settings)
    r.nights.insert(r.nights.index(n) + 1, nn)
    return Out(M("{n} lights in “{_name}”", n=len(lights), _name=q.blocks[nn[0]].name), nn[0], new=nn[0])


def op_merge_rows(q, src, dst):
    a, b = _row(q, src), _row(q, dst)
    if not a.nights or not b.nights or a is b:
        raise Refusal("two different stacks of lights are needed")
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
    q.rows.insert(_after_subtree(q, r.id), nr)
    if q.blocks[blocks[-1]].kind not in JOIN_KINDS and out_ports(q.blocks[blocks[-1]]):
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
    r = _row(q, row)
    if not r.nights:
        raise Refusal("the reference is a stack of lights")
    old = q.reference
    if old == row:
        return Out(focus=row)
    ix = Index(q)
    astro = [b for b in ix.row[old].blocks if q.blocks[b].kind == "astrometry"] if old in ix.row else []
    if astro:
        detach(q, astro[0], keep=True)
        _insert_after_frames(q, _row(q, row), astro[0])
    q.reference = row
    return Out(M("“{_row}” becomes the registration reference", _row=r.name), row, style="need")


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
        if not r.fork:
            raise Refusal("nothing is inserted before the head of the row")
        r.blocks.insert(0, b)
    else:
        if after not in r.blocks:
            raise Refusal("the place is gone")
        i = r.blocks.index(after) + 1
        r.blocks.insert(i, b)
        if i == len(r.blocks) - 1:                       # appended at the end: the row's outputs follow
            for w in q.wires:
                if w.src[0] == after:
                    w.src = [b, _map_port(q.blocks[b], w.src[1])] if out_ports(q.blocks[b]) else w.src
    return Out(M("{lab} here, in “{_row}”", lab=LABEL[q.blocks[b].kind], _row=r.name), b, new=b)


def _branch(q, b, rid, y):
    """Variant forked before y (after the row's last block when y is None); b replaces y's copy when same kind."""
    r = _row(q, rid)
    ix = Index(q)
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
    q.rows.append(Row(new_id(q, "r"), name, [j.id]))
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
        q.wires.append(Wire(new_id(q, "w"), [b.id, "image"], [jid, port]))
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
    q.rows.append(r)
    q.wires.append(Wire(new_id(q, "w"), [blocks[-1], _map_port(q.blocks[blocks[-1]], src_port)], [jid, port]))
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
    first_reason = None
    for rid in [target] + [r for r in rows if r != target]:
        r = ix.row[rid]
        i0 = 0
        while i0 < len(r.blocks) and rank(q.blocks[r.blocks[i0]].kind) <= rank(kind):
            i0 += 1
        slots = list(range(len(r.blocks) + 1))
        slots.sort(key=lambda i: (abs(i - i0), i))
        for i in slots:
            if i == 0 and not r.fork:
                continue
            at = {"type": "insert", "row": rid, "after": r.blocks[i - 1] if i else None}
            why = try_op(q, "add_block", {"kind": kind, "at": at})
            if not why:
                return at
            first_reason = first_reason or why
    raise Refusal(first_reason or M("{lab}: no place", lab=LABEL[kind]))


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
        out = OPS[name](c, **args)
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
    if ix.row[rid].nights and b.kind in ("registration", "stacking"):
        raise Refusal(M("{lab} is essential to the stack", lab=LABEL[b.kind]))
    at = dict(at)
    if at.get("type") not in ("insert", "branch"):
        raise Refusal("unknown place for this block")
    r = ix.row[rid]
    if at.get("after") == block:
        at["after"] = r.blocks[ix.pos[block] - 1] if ix.pos[block] > 0 else None
        if at["after"] is None and not r.fork:
            return Out(focus=block)
    if at.get("y") == block:
        nxt = ix.pos[block] + 1
        at["y"] = r.blocks[nxt] if nxt < len(r.blocks) else None
    detach(q, block, keep=True)
    out = op_add_block(q, b.kind, at, _block=block)
    out.toast = M("{lab} moved", lab=LABEL[b.kind])
    return out


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
        return Out(delete_rows(q, [rid]), old=block, style="need")
    if ix.row[rid].nights and b.kind in ("registration", "stacking"):
        raise Refusal(M("{lab} is essential to the stack", lab=LABEL[b.kind]))
    detach(q, block)
    return Out(M("{lab} deleted", lab=LABEL[b.kind]), old=block, style="need")


def op_switch_off(q, block):
    b = _blk(q, block)
    if b.kind not in OFFABLE:
        raise Refusal(rules.off_reason(b.kind))
    b.off = not b.off
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


def op_wire(q, src, src_port, dst, dst_port):
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
    if d.kind != "post":
        q.wires = [w for w in q.wires if not (w.dst == [dst, dst_port] and w.active)]
    w = Wire(new_id(q, "w"), [src, src_port], [dst, dst_port])
    q.wires.append(w)
    return Out(M("wire {a} → {b}", a=title_msg(q, src), b=title_msg(q, dst)), w.id)


def op_unwire(q, wire):
    w = next((x for x in q.wires if x.id == wire), None)
    if w is None:
        raise Refusal("wire not found")
    q.wires.remove(w)
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


def op_send_to_post(q, block):
    b = _blk(q, block)
    if not out_ports(b):
        raise Refusal("lights first")
    post = Index(q).post
    if any(w.src[0] == block and w.dst[0] == post for w in q.wires):
        raise Refusal("already in Post")
    _to_post(q, block)
    n = sum(1 for w in q.wires if w.dst[0] == post)
    return Out(M("{t} to Post ({n} layer)" if n == 1 else "{t} to Post ({n} layers)", t=title_msg(q, block), n=n), block)


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
            if into == "night":
                r.nights.insert(at, nn)
                at += 1
            else:
                nr = Row(new_id(q, "r"), name, _default_chain(q), [nn])
                q.rows.insert(rows_at, nr)
                rows_at += 1
                _to_post(q, nr.blocks[-1])
            made += 1
            focus = nn[0]
        else:
            raise Refusal("unknown destination")
    return Out(M("Split into {n} sessions", n=made + 1) if made else "files distributed", focus)


OPS = {k[3:]: v for k, v in dict(globals()).items() if k.startswith("op_")}
