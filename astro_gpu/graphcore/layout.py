"""Layout as a pure function (§3): lines, columns, composition area, Post, wires. Same model + view -> same Drawing.
Guarantees (checked in tests/invariants.py): I1 no overlap, I2 no wire behind a block, I3 rightwards."""
from dataclasses import dataclass

from .shapes import Box, Path, Drawing
from .model import Index, in_ports, out_ports, FRAME_KINDS, JOIN_KINDS
from .rules import COL, LIN, title

BW, BH, GAP, LANE, SROW, TOP, X0 = 120, 54, 16, 80, 68, 74, 24
JW = BW + 46
CARD_W = 2 * BW + GAP          # a folded stack: name + badge + "3 sessions · 78 lights · drizzle auto"


def LAST_FRAME(g):
    """Last column of the frame band (wires out of it start in the gap after it)."""
    return 5 if g.frames else 1 if g.compact else 0
CLEAR = 8
COMP0, POSTCOL = 100, 1000


@dataclass
class Grid:
    lines: list       # [(kind "chain"|"night", owner id, row id)]
    cells: dict       # block -> (line | None, logical col)
    cards: dict       # row -> line
    comp: list        # composition rows in placement order
    frames: bool      # some stack is unfolded: the frame band has its columns
    crop: bool
    nlin: int
    ncomp: int
    folded: set
    compact: bool = False   # stacks exist and all are folded: the band is one card wide
    deps: dict = None       # composition row -> rows it waits for (fork parent, rows feeding its head)


def foldable(p, ix, rid):
    return bool(ix.row[rid].nights) and not any(
        p.blocks[ix.row[v].fork[1]].kind in FRAME_KINDS[:-1] for v in ix.children[rid])


def grid(p, view, ix=None):
    """Lines and logical columns (cheap: no geometry)."""
    ix = ix or Index(p)
    folded = {r.id for r in p.rows if r.nights and foldable(p, ix, r.id)} if view.get("folded") is True else set()
    lines, cells, cards, colcache = [], {}, {}, {}

    def cols(rid):
        if rid in colcache:
            return colcache[rid]
        r = ix.row[rid]
        nxt = LIN
        if r.fork:
            pc = dict(cols(r.fork[0]))[r.fork[1]]
            nxt = pc + 1 if pc >= LIN else LIN
        out = []
        for b in r.blocks:
            k = p.blocks[b].kind
            if k in COL:
                out.append((b, COL[k]))
            else:
                out.append((b, nxt))
                nxt += 1
        colcache[rid] = out
        return out

    def emit(rid):
        r = ix.row[rid]
        L = len(lines)
        lines.append(("chain", rid, rid))
        fold = rid in folded
        if r.nights:
            if fold:
                cards[rid] = L
                cells["card:" + rid] = (L, 0)
            else:
                for i, (s, c) in enumerate(r.nights):
                    if i:
                        lines.append(("night", s, rid))
                    cells[s] = (len(lines) - 1, 0)
                    if c is not None:                  # a night without Calibration: its column stays empty
                        cells[c] = (len(lines) - 1, 1)
        for b, c in cols(rid):
            if not (fold and p.blocks[b].kind in FRAME_KINDS):
                cells[b] = (L, c)
        for v in ix.children[rid]:
            emit(v)

    joinrows = [r.id for r in p.rows if ix.is_join_row(r.id)]
    for r in p.rows:
        if not r.fork and r.id not in joinrows:
            emit(r.id)
    # composition rows: topological order of their inputs, ties in rows order
    jset = set(joinrows)
    deps = {j: set() for j in joinrows}
    for j in joinrows:
        r = ix.row[j]
        if r.fork:
            deps[j].add(r.fork[0])
        elif r.blocks:
            for w in p.wires:
                if w.dst[0] == r.blocks[0] and ix.row_of.get(w.src[0]) in jset:
                    deps[j].add(ix.row_of[w.src[0]])
    order, done = [], set()
    while len(order) < len(joinrows):
        nxt = next((j for j in joinrows if j not in done and deps[j] <= done), None)
        if nxt is None:                       # cannot happen in a checked model; keep the layout total
            nxt = next(j for j in joinrows if j not in done)
        order.append(nxt)
        done.add(nxt)
    kcol = {}
    for j in order:
        r = ix.row[j]
        if r.fork:
            k0 = kcol[r.fork[1]] + 1
        else:
            k0 = 0
            for w in p.wires:
                if r.blocks and w.dst[0] == r.blocks[0] and w.src[0] in kcol:
                    k0 = max(k0, max(kcol[b] for b in ix.row[ix.row_of[w.src[0]]].blocks) + 1)
        for i, b in enumerate(r.blocks):
            kcol[b] = k0 + i
            cells[b] = (None, COMP0 + k0 + i)
    if ix.post:
        cells[ix.post] = (0, POSTCOL)
    src_cols = [c for b, (l, c) in cells.items() if l is not None and c < COMP0]
    nlin = max([c - LIN + 1 for c in src_cols if c >= LIN] + [1])
    # the frame band has its columns when a stack shows its nights, or a block of it (a stack whose nights are all
    # gone, a workflow of the previous format made before its first session: its Registration... still drawn)
    frames = any(r.nights and r.id not in folded for r in p.rows) or \
        any(line is not None and 0 < c < LIN for b, (line, c) in cells.items() if not b.startswith("card:"))
    compact = not frames and any(r.nights for r in p.rows)
    recad = any(p.blocks[b].kind == "crop" and ix.row_of.get(b) not in folded for b in p.blocks)
    ncomp = max([v + 1 for v in kcol.values()] + [1])
    return Grid(lines, cells, cards, order, frames, recad, nlin, ncomp, folded, compact, deps)


def _xs(g):
    open_cols = [0] + (([1, 2, 3] + ([4] if g.crop else []) + [5]) if g.frames else
                       [1] if g.compact else [])
    open_cols += [LIN + i for i in range(g.nlin)]
    xs, x = {}, X0
    for c in open_cols:
        xs[c] = x
        x += BW + GAP
    src_right = x - GAP
    xc = src_right + 3 * GAP
    for k in range(g.ncomp):
        xs[COMP0 + k] = xc + k * (JW + GAP)
    comp_right = xc + g.ncomp * (JW + GAP) - GAP
    xs[POSTCOL] = comp_right + 3 * GAP
    seq = open_cols + [COMP0 + k for k in range(g.ncomp)] + [POSTCOL]
    return xs, seq, src_right, xc, comp_right


def _width(c):
    return JW if COMP0 <= c < POSTCOL else BW


def join_h(b):
    return max(BH, 18 * len(in_ports(b)) + 18)


def layout(p, view, tr=None):
    ix = Index(p)
    g = grid(p, view, ix)
    xs, seq, src_right, xc, comp_right = _xs(g)
    d = Drawing()
    ys, y = [], TOP
    for i, (kind, _o, _r) in enumerate(g.lines):
        if i:
            y += SROW if kind == "night" else LANE
        ys.append(y)
    d.lines = [(o, ys[i]) for i, (_k, o, _r) in enumerate(g.lines)]
    # source boxes
    for b, (line, c) in g.cells.items():
        if line is not None and c < COMP0 and not b.startswith("card:"):
            d.boxes[b] = Box(xs[c], ys[line], BW, BH)
    for rid, line in g.cards.items():
        d.cards[rid] = Box(xs[0], ys[line], CARD_W, BH)        # one narrow card, left-aligned
    src_bottom = max([bx.bottom for bx in d.boxes.values()] + [bx.bottom for bx in d.cards.values()] + [TOP])
    # composition rows, greedy
    placed, mids = [], {}
    rank = {r.id: i for i, r in enumerate(p.rows)}
    left = list(g.comp)                       # topological; among the rows ready, the highest inputs first
    while left:
        done = {pl[0] for pl in placed}
        ready = [j for j in left if g.deps.get(j, set()) <= done] or left[:1]
        cand = []
        for j in ready:
            r = ix.row[j]
            head = p.blocks[r.blocks[0]]
            h = join_h(head) if head.kind in JOIN_KINDS else BH
            if r.fork:
                par = next(pl for pl in placed if pl[0] == r.fork[0])
                want = par[1].bottom + 26
            else:                             # at the barycentre of its inputs
                srcs = [_out_port(p, d, w.src[0], w.src[1]) for w in p.wires if w.dst[0] == r.blocks[0]]
                srcs = [s_ for s_ in srcs if s_]
                want = (sum(s_[1] for s_ in srcs) / len(srcs) - h / 2) if srcs else src_bottom + LANE
            cand.append((want, rank[j], j, h))
        want, _rk, j, h = min(cand)
        left.remove(j)
        r = ix.row[j]
        x0 = xs[g.cells[r.blocks[0]][1]]
        x1 = xs[g.cells[r.blocks[-1]][1]] + _width(g.cells[r.blocks[-1]][1])
        y = max(TOP, round(want))
        moved = True
        while moved:
            moved = False
            for _j, rect in placed:
                # rows keep one free column on their right (the "after the last block" strip) and never abut
                if rect.x < x1 + JW + 2 * GAP and x0 < rect.right + JW + 2 * GAP and y < rect.bottom + 26                         and rect.y < y + h + 26:
                    y = rect.bottom + 26
                    moved = True
        rect = Box(x0, y, x1 - x0, h)
        placed.append((j, rect))
        mid = y + h / 2
        mids[j] = mid
        for b in r.blocks:
            c = g.cells[b][1]
            if p.blocks[b].kind in JOIN_KINDS:
                d.boxes[b] = Box(xs[c], y, JW, h)
            else:
                d.boxes[b] = Box(xs[c], mid - BH / 2, BW, BH)
    if ix.post:
        d.boxes[ix.post] = Box(xs[POSTCOL], TOP, BW, BH)
    d.cells = dict(g.cells)
    # ports
    for b, bx in d.boxes.items():
        blk = p.blocks[b]
        for i, port in enumerate(in_ports(blk)):
            d.ports[(b, port, "in")] = (bx.x, bx.y + BH / 2) if blk.kind == "post" else (bx.x, bx.y + 18 * (i + 1))
        for port in out_ports(blk):
            d.ports[(b, port, "out")] = _out_port(p, d, b, port)
    _wires(p, ix, g, d, xs, seq)
    # captions
    for r in p.rows:
        first = _first_box(p, d, r)
        if first is None:
            continue
        shown = (r.nights and len(r.nights) == 1 and r.id not in d.cards) or r.id in d.cards \
            or (not r.nights and not r.fork and r.blocks and p.blocks[r.blocks[0]].kind in JOIN_KINDS
                and r.name in (title(p, r.blocks[0], ix), title(p, r.blocks[0], ix, tr)))
        if not shown:
            d.captions.append((r.id, r.name, first.x, first.y - 6))
    allb = list(d.boxes.values()) + list(d.cards.values())
    bottom = max([bx.bottom for bx in allb] + [TOP + BH])
    pre_r = (xs[5] if g.frames else xs[1] if g.compact else xs[0]) + BW + 8
    d.bands = [("PREPROCESSING", Box(X0 - 8, TOP - 30, pre_r - X0 + 8, bottom - TOP + 40)),
               ("LINEAR PROCESSING", Box(xs[LIN] - 8, TOP - 30, src_right - xs[LIN] + 16, bottom - TOP + 40)),
               ("COMPOSITION", Box(xc - 8, TOP - 30, comp_right - xc + 16, bottom - TOP + 40))]
    right = max([bx.right for bx in allb] + [comp_right])
    d.extras = {"project": Box(X0, 4, BW, 34),
                "fold_all": Box(pre_r - 8 - 22, TOP - 27, 18, 18),
                "new_workflow": Box(X0, src_bottom + 20 + BH + 8, 160, 20)}
    d.size = (right + 40, max(bottom, d.extras["new_workflow"].bottom) + 40)
    return d


def _first_box(p, d, r):
    if r.id in d.cards:
        return d.cards[r.id]
    ids = ([r.nights[0][0]] if r.nights else []) + r.blocks
    return next((d.boxes[b] for b in ids if b in d.boxes), None)


def _drawn(p, d, b, ix=None):
    """Box of b, or of the card hiding it."""
    if b in d.boxes:
        return d.boxes[b]
    ix = ix or Index(p)
    return d.cards.get(ix.row_of.get(b))


def _out_port(p, d, b, port):
    if b in d.boxes:
        bx = d.boxes[b]
        if p.blocks[b].kind in ("stars", "synth") and port in ("Starless", "Stars only"):
            return (bx.right, bx.y + (18 if port == "Starless" else 36))
        return (bx.right, bx.y + bx.h / 2)
    bx = _drawn(p, d, b)
    return (bx.right, bx.y + BH / 2) if bx else None


def _wires(p, ix, g, d, xs, seq):
    """Chain straight on its line; nights into the fan-in bar; forks and port wires through column gaps."""
    pos = {c: i for i, c in enumerate(seq)}

    def gap_after(c):
        i = pos[c]
        left = xs[c] + _width(c)
        return (i, left, xs[seq[i + 1]])

    def col_of(b):
        if b in g.cells:
            return g.cells[b][1]
        return LAST_FRAME(g)                       # hidden in a folded card: leaves at the band's right edge

    items = {}          # gap index -> [(key, track id)]
    meta = {}           # track id -> (ys entering from the left, y leaving to the right)
    plans = []

    def want(gi, key, tid, ins, out):
        items.setdefault(gi, []).append((key, tid))
        meta[tid] = (ins, out)

    # chains
    for r in p.rows:
        for a, b in zip(r.blocks, r.blocks[1:]):
            if b not in d.boxes:
                continue
            A = _drawn(p, d, a, ix)
            B = d.boxes[b]
            yy = A.y + A.h / 2
            d.wires.append(Path(f"c:{a}>{b}", [(A.right, yy), (B.x, yy)], "chain", a, b))
        if r.nights and r.id not in g.folded and r.blocks:
            head = d.boxes[r.blocks[0]]
            gi = gap_after(1)[0]
            want(gi, (head.y, r.id), "bar:" + r.id, [d.boxes[c if c is not None else s].y + BH / 2
                                                     for s, c in r.nights], head.y + BH / 2)
            for s, c in r.nights:
                S = d.boxes[s]
                if c is not None:
                    C = d.boxes[c]
                    d.wires.append(Path(f"n:{s}>{c}", [(S.right, S.y + BH / 2), (C.x, C.y + BH / 2)], "night", s, c))
                plans.append(("bar", r.id, c if c is not None else s, r.blocks[0], gi))
        if r.fork and r.blocks and r.blocks[0] in d.boxes:
            x = r.fork[1]
            gi = gap_after(col_of(x))[0]
            X = _drawn(p, d, x, ix)
            want(gi, (X.y, "f" + r.id), "f:" + r.id, [X.y + X.h / 2], d.boxes[r.blocks[0]].y + BH / 2)
            plans.append(("fork", r.id, x, r.blocks[0], gi))
    # port wires
    rects = sorted((bx.x, bx.x + bx.w, bx.y, bx.y + bx.h) for bx in list(d.boxes.values()) + list(d.cards.values()))
    for w in p.wires:
        sc, dc = col_of(w.src[0]), col_of(w.dst[0])
        if sc < LAST_FRAME(g):
            sc = LAST_FRAME(g)
        g1 = gap_after(sc)
        g2 = (pos[dc] - 1, xs[seq[pos[dc] - 1]] + _width(seq[pos[dc] - 1]), xs[dc])
        sp = _out_port(p, d, w.src[0], w.src[1])
        dp = d.ports[(w.dst[0], w.dst[1], "in")]
        if g1[0] == g2[0]:
            want(g1[0], (sp[1], w.id), "a:" + w.id, [sp[1]], dp[1])
            plans.append(("port1", w, sp, dp, g1[0]))
            continue
        fy = _free_y(rects, g1[1], g2[2], sp[1], dp[1])
        want(g1[0], (sp[1], w.id), "a:" + w.id, [sp[1]], fy)
        want(g2[0], (sp[1], w.id), "b:" + w.id, [fy], dp[1])
        plans.append(("port2", w, sp, dp, (g1[0], g2[0], fy)))
    # tracks
    track = {}
    for gi, lst in items.items():
        lst = _order_tracks(sorted(lst), meta)
        left = xs[seq[gi]] + _width(seq[gi])
        right = xs[seq[gi + 1]]
        n = len(lst)
        step = min(4.0, (right - left - 4) / (n - 1)) if n > 1 else 0
        c0 = (left + right) / 2 - step * (n - 1) / 2
        for j, (_k, tid) in enumerate(lst):
            track[tid] = c0 + j * step
    for plan in plans:
        kind = plan[0]
        if kind == "bar":
            _k, rid, c, head, gi = plan
            C, H = d.boxes[c], d.boxes[head]
            tx, cy, hy = track["bar:" + rid], C.y + BH / 2, H.y + BH / 2
            d.wires.append(Path(f"n:{c}>{head}", _clean([(C.right, cy), (tx, cy), (tx, hy), (H.x, hy)]),
                                "bar", c, head))
        elif kind == "fork":
            _k, rid, x, v0, gi = plan
            X, V = _drawn(p, d, x, ix), d.boxes[v0]
            tx = track["f:" + rid]
            xy = X.y + X.h / 2
            vy = V.y + BH / 2
            d.wires.append(Path(f"f:{x}>{v0}", _clean([(X.right, xy), (tx, xy), (tx, vy), (V.x, vy)]),
                                "fork", x, v0))
        else:
            w, sp, dp = plan[1], plan[2], plan[3]
            style = "port" if w.active else "port_inactive"
            colour = {"Starless": "starless", "Stars only": "stars_only"}.get(w.src[1], "")
            if kind == "port1":
                t = track["a:" + w.id]
                pts = [sp, (t, sp[1]), (t, dp[1]), dp]
            else:
                _g1, _g2, fy = plan[4]
                t1, t2 = track["a:" + w.id], track["b:" + w.id]
                pts = [sp, (t1, sp[1]), (t1, fy), (t2, fy), (t2, dp[1]), dp]
            d.wires.append(Path(w.id, _clean(pts), style, w.src[0], w.dst[0], colour))


def _cost(i, j):
    """Crossings when track i lies left of track j in one gap (entries come from the left, exits go right)."""
    (ins_i, out_i), (ins_j, out_j) = i, j
    lo_i, hi_i = min(ins_i + [out_i]), max(ins_i + [out_i])
    lo_j, hi_j = min(ins_j + [out_j]), max(ins_j + [out_j])
    return sum(lo_i < y < hi_i for y in ins_j) + (lo_j < out_i < hi_j)


def _order_tracks(lst, meta):
    """Order the tracks of a gap so that its wires cross as little as possible (ties: the sorted order)."""
    n = len(lst)
    if n < 2:
        return lst
    m = [meta[t] for _k, t in lst]
    cost = [[_cost(m[a], m[b]) for b in range(n)] for a in range(n)]
    if n <= 6:
        import itertools
        best, best_c = None, None
        for perm in itertools.permutations(range(n)):
            c = sum(cost[perm[a]][perm[b]] for a in range(n) for b in range(a + 1, n))
            if best_c is None or c < best_c:
                best, best_c = perm, c
                if c == 0:
                    break
        return [lst[k] for k in best]
    order = list(range(n))
    for _pass in range(n):                    # larger gaps: adjacent swaps while they help
        better = False
        for a in range(n - 1):
            x, y = order[a], order[a + 1]
            if cost[y][x] < cost[x][y]:
                order[a], order[a + 1] = y, x
                better = True
        if not better:
            break
    return [lst[k] for k in order]


def _free_y(rects, x0, x1, sy, dy):
    """Nearest y to sy (then dy) whose band ±CLEAR is clear of every box across [x0, x1] (interval union).
    rects: (x, right, y, bottom) sorted by x."""
    iv = []
    for x, right, y, bottom in rects:
        if x >= x1:
            break
        if right > x0:
            iv.append((y - CLEAR, bottom + CLEAR))
    iv.sort()
    merged = []
    for a, b in iv:
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])

    # the merged intervals are disjoint: their ends are free; sy is free unless strictly inside one
    if not any(a < sy < b for a, b in merged):
        return sy
    return min((v for ab in merged for v in ab), key=lambda c: (abs(c - sy), abs(c - dy), c))


def _clean(pts):
    out = []
    for pt in pts:
        pt = (round(pt[0], 2), round(pt[1], 2))
        if not out or out[-1] != pt:
            out.append(pt)
    # drop middle points of straight runs
    res = []
    for pt in out:
        if len(res) >= 2 and (res[-2][0] == res[-1][0] == pt[0] or res[-2][1] == res[-1][1] == pt[1]):
            res[-1] = pt
        else:
            res.append(pt)
    return res
