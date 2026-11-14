"""Fast placement judge: decides a drop from the rules table and the current structure, without applying the
operation. The types of the current project are evaluated once per revision; a drop re-evaluates only the blocks
downstream of it (and stops where nothing changes). tests/fuzz.py checks after every gesture that its verdict is
the one apply() gives, and that a lit place's (line, col) is where the block is drawn."""
from collections import ChainMap

from .model import Index, Block, edges, topo, out_ports, JOIN_KINDS, FRAME_KINDS
from .rules import Ctx, eval_one, evaluate, chain_problem, root_kind, holder_align, is_legacy, COL, LIN

_cache = {}


def memo(name, p, key, fn):
    """One cached value per name, for the latest project object (projects are never changed in place: every
    operation, undo and redo makes a new one)."""
    ent = _cache.get(name)
    if ent is not None and ent[0] is p and ent[1] == key:
        return ent[2]
    val = fn()
    _cache[name] = (p, key, val)
    return val


class Base:
    def __init__(self, p, view):
        from .layout import grid
        self.p = p
        self.view = {"folded": view.get("folded") is True}
        self.ix = Index(p)
        self.info = memo("evaluate_legal", p, None, lambda: evaluate(p, self.ix, wiring=False))
        self.ctx = Ctx(p, self.ix, wiring=False)          # as legality judges (operations.legality): no wiring
        self.order = {b: i for i, b in enumerate(topo(p, self.ix, active_only=True) or list(p.blocks))}
        self.succ = {}
        for a, b in edges(p, self.ix, active_only=True):
            self.succ.setdefault(a, []).append(b)
        self.g = grid(p, self.view, self.ix)
        self.chain_line = {o: i for i, (k, o, _r) in enumerate(self.g.lines) if k == "chain"}


def evaluation(p, ix=None):
    return memo("evaluate", p, None, lambda: evaluate(p, ix or Index(p)))


def base(p, view):
    return memo("base", p, view.get("folded") is True, lambda: Base(p, view))


class _W:
    __slots__ = ("id", "src", "dst", "active")

    def __init__(self, src, dst, active=True):
        self.id, self.src, self.dst, self.active = "", src, dst, active


class _Overlay:
    def __init__(self, B, virtual, head_over, wires_over):
        c = B.ctx
        self.p, self.ix, self.wiring = c.p, c.ix, c.wiring
        self.head_in = ChainMap(head_over, c.head_in)
        self.wires_in = ChainMap(wires_over, c.wires_in)
        self.virtual = virtual

    def blk(self, b):
        return self.virtual.get(b) or self.p.blocks[b]


def _run(B, virtual, chain, head_over, wires_over, starts, pre=None):
    """Evaluate the virtual blocks (in order) then every block downstream of `starts`; first unmet need or "".
    pre: Evs given for virtual sources (a new session)."""
    ov = _Overlay(B, virtual, head_over, wires_over)
    new = dict(pre or {})
    info = ChainMap(new, B.info)
    for b in chain:
        new[b] = eval_one(ov, b, info)
        if new[b].cat == "need":
            return new[b].reason
    todo = set(starts)
    done = set()
    while todo:
        b = min(todo, key=lambda x: B.order.get(x, -1))
        todo.discard(b)
        if b in done:
            continue
        done.add(b)
        ev = eval_one(ov, b, info)
        old = B.info.get(b)
        new[b] = ev
        if ev.cat == "need":
            return ev.reason
        if old is not None and (old.out, old.cat, old.root) == (ev.out, ev.cat, ev.root):
            continue                                       # unchanged: nothing further down changes
        todo.update(B.succ.get(b, ()))
    return ""


def _map_port(blk, port):
    ports = out_ports(blk)
    return port if port in ports else (ports[0] if ports else None)


def _col_after(B, x):
    c = B.g.cells.get(x, (0, 5))[1]
    return c + 1 if c >= LIN else LIN


def judge_insert(B, kind, rid, after):
    p, ix = B.p, B.ix
    r = ix.row[rid]
    if after is None and not r.fork and not r.nights:
        return "nothing is inserted before the head of the row", None
    pos = r.blocks.index(after) + 1 if after is not None else 0
    s = r.blocks[pos] if pos < len(r.blocks) else None
    why = ""
    for x in ix.subtree(rid):                               # the engine's rules and the drawing's first
        eff = ix.effective(x)
        if x != rid and (s is None or s not in eff):
            continue
        at = eff.index(s) if s is not None else len(eff)
        items = [(p.blocks[b].kind, p.blocks[b].off) for b in eff]
        items.insert(at, (kind, False))
        why = chain_problem(items, root_kind(p, ix, x), _align_with(p, ix, eff, at, kind, rid),
                            is_legacy(ix, x))
        if why:
            return why, None
    v = Block("__v", kind)
    if after is not None:
        a = [after]
    elif r.fork:
        a = [r.fork[1]]
    else:
        a = [c if c is not None else s for s, c in r.nights]
    head_over, wires_over, starts = {"__v": a}, {}, []
    if s is not None:
        head_over[s] = ["__v"]
        starts.append(s)
    elif out_ports(v):                                     # appended at the end: the row's outputs follow
        for w in p.wires:
            if w.src[0] == after and not keeps_wire(p, kind, w):
                lst = wires_over.setdefault(w.dst[0], list(B.ctx.wires_in.get(w.dst[0], [])))
                if w.active:
                    lst[lst.index(w)] = _W(["__v", _map_port(v, w.src[1])], w.dst)
                    starts.append(w.dst[0])
    why = _run(B, {"__v": v}, ["__v"], head_over, wires_over, starts)
    if why:
        return why, None
    ca = B.g.cells.get(a[0])
    if ca is not None and ca[0] is None:
        return "", (None, ca[1] + 1)                                    # composition area
    line = B.chain_line[rid]
    if kind in COL:
        return "", (line, 0) if rid in B.g.folded else (line, COL[kind])
    return "", (line, _col_after(B, a[0]) if after is not None or r.fork else LIN)


def keeps_wire(p, kind, w):
    """A Synthetic stars appended after a Stars/background: a combination's wire stays on the Stars (the engine reads
    an Output or a Stars/background, never a Synthetic stars: the Stars becomes a mid-row source, a tap)."""
    return kind == "synth" and p.blocks[w.dst[0]].kind == "combine"


def _align_with(p, ix, eff, at, kind, rid):
    """The Align on the chain eff passes once a block of `kind` is inserted at `at` in row rid's blocks: that of
    the row holding its last Stacking (or Image), the inserted one included."""
    a = None
    for i, b in enumerate(eff + [None]):
        if i == at and kind in ("stacking", "image"):
            a = ix.row[rid].align
        if b is not None and p.blocks[b].kind in ("stacking", "image"):
            a = ix.row[ix.row_of[b]].align
    return a


def judge_branch(B, kind, rid, y):
    p, ix = B.p, B.ix
    r = ix.row[rid]
    if y is None:
        x, tail = r.blocks[-1], []
    else:
        i = r.blocks.index(y)
        if i == 0 or p.blocks[y].kind in ("registration", "image") + JOIN_KINDS:
            return "a variant starts after the Registration", None
        x, tail = r.blocks[i - 1], r.blocks[i:]
    if tail and p.blocks[tail[0]].kind == kind:
        tail = tail[1:]
    virtual = {"__v0": Block("__v0", kind)}
    chain, head_over, prev = ["__v0"], {"__v0": [x]}, "__v0"
    for j, t in enumerate(tail):
        o = p.blocks[t]
        vid = f"__v{j + 1}"
        virtual[vid] = Block(vid, o.kind, dict(o.settings), o.off)
        head_over[vid] = [prev]
        chain.append(vid)
        prev = vid
    eff = ix.effective(rid)
    items = [(p.blocks[b].kind, p.blocks[b].off) for b in eff[:eff.index(x) + 1]] + \
        [(virtual[v].kind, virtual[v].off) for v in chain]
    own = any(virtual[v].kind in ("stacking", "image") for v in chain)
    align = holder_align(p, ix, eff) if own else holder_align(p, ix, eff[:eff.index(x) + 1])
    why = chain_problem(items, root_kind(p, ix, rid), align, is_legacy(ix, rid))
    if not why:
        why = _run(B, virtual, chain, head_over, {}, [])
    if not why and B.view["folded"] and r.nights and p.blocks[x].kind in FRAME_KINDS[:-1]:
        from .layout import foldable
        if foldable(p, ix, rid):
            why = "Preprocessing is folded: unfold it first"
    if why:
        return why, None
    if B.g.cells.get(x, (0, 0))[0] is None and x in B.g.cells:
        return "", (None, B.g.cells[x][1] + 1)
    sub = set(ix.subtree(rid))
    line = max(i for i, (_k, _o, rr) in enumerate(B.g.lines) if rr in sub) + 1
    return "", (line, COL[kind] if kind in COL else _col_after(B, x))


def judge_night(B, rid):
    if rid in B.g.folded:
        return "", (B.chain_line[rid], 0)
    mine = [i for i, (k, _o, rr) in enumerate(B.g.lines) if rr == rid and (k == "night" or i == B.chain_line[rid])]
    return "", (max(mine) + 1, 0)


def judge_port_image(B, j, port):
    v = Block("__i", "image")
    lst = list(B.ctx.wires_in.get(j, [])) + [_W(["__i", "image"], [j, port])]
    why = _run(B, {"__i": v}, ["__i"], {}, {j: lst}, [j])
    return why, (None if why else (len(B.g.lines), 0))


def judge_port_session(B, j, port):
    """Mirror of operations._add_port_row for a Session: the new stack is its Session and the shortest chain of
    operations.port_chains the graph accepts there (a Stacking; Denoise and Stars/background too for a starless image
    or the stars only); the first one's reason when none is."""
    from .rules import Ev, Img
    from .operations import port_chains
    p = B.p
    align = {"ref": p.reference} if p.reference is not None else None
    pre = {"__cal": Ev(out=Img(frames=True, stage=-1, grid="__r"))}
    first = None
    for kinds, src_port, _why in port_chains(p, j, port):
        virtual, chain, head_over, prev = {}, [], {}, "__cal"
        for i, k in enumerate(kinds):
            vid = f"__p{i}"
            virtual[vid] = Block(vid, k)
            head_over[vid] = [prev]
            chain.append(vid)
            prev = vid
        why = chain_problem([(k, False) for k in kinds], "stack", align)
        if not why:
            lst = list(B.ctx.wires_in.get(j, [])) + [_W([prev, _map_port(virtual[prev], src_port)], [j, port])]
            why = _run(B, virtual, chain, head_over, {j: lst}, [j], pre)
        if not why:
            return "", (len(B.g.lines), 0)
        first = first or why
    return first, None


def specs(B, kind):
    """Insert and branch place specs of a palette kind (no geometry): what palette_state needs."""
    p = B.p
    if kind in FRAME_KINDS:          # a stack step goes only after another one (not Empilement): only those spots
        early = FRAME_KINDS[:-1]
        for r in p.rows:
            if r.fork and p.blocks[r.fork[1]].kind in early or r.nights:
                yield {"type": "insert", "row": r.id, "after": None}
            for i, b in enumerate(r.blocks):
                if p.blocks[b].kind in early and b in B.g.cells:
                    yield {"type": "insert", "row": r.id, "after": b}
                if i > 0 and p.blocks[b].kind in FRAME_KINDS and p.blocks[r.blocks[i - 1]].kind in early                         and b in B.g.cells:
                    yield {"type": "branch", "row": r.id, "y": b}
        return
    for r in p.rows:
        if r.fork or r.nights:
            yield {"type": "insert", "row": r.id, "after": None}
        for b in r.blocks:
            if b in B.g.cells or (r.id in B.g.folded and p.blocks[b].kind == "stacking"):
                yield {"type": "insert", "row": r.id, "after": b}
        for i, b in enumerate(r.blocks):
            if i > 0 and p.blocks[b].kind != "registration" and b in B.g.cells:
                yield {"type": "branch", "row": r.id, "y": b}
        if r.blocks:
            yield {"type": "branch", "row": r.id, "y": None}


def judge(B, op, args):
    """(reason, cell) for a place's op; None when this judge does not decide it (the caller simulates)."""
    if op != "add_block":
        return None
    kind, at = args["kind"], args.get("at") or {}
    t = at.get("type")
    if t == "insert":
        return judge_insert(B, kind, at["row"], at.get("after"))
    if t == "branch":
        return judge_branch(B, kind, at["row"], at.get("y"))
    if t == "night":
        return judge_night(B, at["row"])
    if t == "row":
        return "", (len(B.g.lines), 0)
    if t == "comp":
        after = at.get("after")
        k0 = 0
        if after is not None and B.g.cells.get(after, (0, 0))[0] is None:
            k0 = max(B.g.cells[x][1] for x in B.ix.row[B.ix.row_of[after]].blocks) - 100 + 1
        return "", (None, 100 + k0)
    if t == "port" and kind == "image":
        return judge_port_image(B, at["block"], at["port"])
    if t == "port" and kind == "session":
        return judge_port_session(B, at["block"], at["port"])
    return None
