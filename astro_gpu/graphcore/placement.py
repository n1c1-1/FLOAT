"""Placement grammar (§2): every place for a drag, lit or refused with its reason. Nothing moves during a drag;
each lit place carries the (line, column) where its block will appear (I5)."""
from .shapes import Box, Place
from .msg import M
from .model import Index, clone, in_ports, JOIN_KINDS
from .rules import LABEL, LIN
from .layout import layout, grid, _xs, BW, BH, GAP, X0
from . import operations as ops

STRIP = 18


def _simulate(p, view, name, args):
    q = clone(p)
    try:
        out = ops.OPS[name](q, **args)
        ops.finish(q, out.style)
        ops.fold_guard(p, q, view)
    except ops.Refusal as e:
        return e.msg, None
    cell = None
    if out.new and out.new in q.blocks:
        cell = shown_cell(q, view, out.new)
    return "", cell


def shown_cell(q, view, b):
    """Cell where block b is shown: its own, or its folded stack's card."""
    cells = grid(q, view).cells
    return cells.get(b) or cells.get("card:" + str(Index(q).row_of.get(b)))


def candidates(p, view, kind=None, block=None, d=None):
    """Places without their verdict (lit/reason/line/col filled by judge)."""
    from .judge import memo, base
    B = base(p, view)
    ix, g = B.ix, B.g
    d = d or memo("layout", p, view.get("folded") is True, lambda: layout(p, view))
    xs = _xs(g)[0]
    if block is not None:
        kind = p.blocks[block].kind
    out = []

    def add(pk, rect, tip, name, args):
        if block is not None and _own_place(p, ix, block, name, args):
            return                               # a block dropped where it is: no place (nothing would move)
        out.append(Place(pk, rect, None, None, False, "", tip, (name, args)))

    def op_for(at):
        return ("move_block", {"block": block, "at": at}) if block else ("add_block", {"kind": kind, "at": at})

    chainable = kind not in ("session", "image", "calibration", "post") + JOIN_KINDS
    if chainable:
        for r in p.rows:
            drawn = [b for b in r.blocks if b in d.boxes]
            if r.nights and r.id not in d.cards and r.nights[0][1] in d.boxes:   # a stack's head, after its nights
                cal = d.boxes[r.nights[0][1]]
                x = d.boxes[r.blocks[0]].x - GAP / 2 - 3 if drawn and r.blocks[0] == drawn[0] else cal.right + GAP / 2 - 3
                add("insert", Box(x, cal.y, 6, BH), M("{lab} here, in “{_row}”", lab=LABEL[kind], _row=r.name),
                    *op_for({"type": "insert", "row": r.id, "after": None}))
            if not drawn:
                continue
            if r.fork:
                f = d.boxes[r.blocks[0]]
                add("insert", Box(f.x - GAP / 2 - 3, f.y, 6, BH), M("{lab} here, in “{_row}”", lab=LABEL[kind], _row=r.name),
                    *op_for({"type": "insert", "row": r.id, "after": None}))
            for b in r.blocks:
                bx = d.boxes.get(b)
                if bx is None:
                    if r.id in d.cards and p.blocks[b].kind == "stacking":   # caret at the band's right edge
                        c = d.cards[r.id]
                        bx = Box(xs[LIN] - GAP - BW, c.y, BW, BH)
                    else:
                        continue
                add("insert", Box(bx.right + GAP / 2 - 3, bx.y + (bx.h - BH) / 2, 6, BH),
                    M("{lab} here, in “{_row}”", lab=LABEL[kind], _row=r.name), *op_for({"type": "insert", "row": r.id, "after": b}))
        strips = {}
        for r in p.rows:
            if not r.blocks or r.blocks[0] not in d.boxes and r.id not in d.cards:
                continue
            sub = ix.subtree(r.id)
            bottoms = [d.boxes[b].bottom for s in sub for b in ix.row[s].blocks + [x for n in ix.row[s].nights
                                                                                    for x in n] if b in d.boxes]
            bottoms += [d.cards[s].bottom for s in sub if s in d.cards]
            y = max(bottoms) + 4
            own_y = d.boxes[r.blocks[-1]].y if r.blocks[-1] in d.boxes else d.cards[r.id].y
            ys = [b for i, b in enumerate(r.blocks) if i > 0 and p.blocks[b].kind not in ("registration",)
                  and b in d.boxes]
            last = r.blocks[-1]
            lc = g.cells.get(last, (0, 5))[1]
            for yb in ys + [None]:
                if yb is None:                       # the next column's x (columns are shared by all rows)
                    if lc < LIN or last not in d.boxes:
                        x = xs[LIN]
                    else:
                        x = xs.get(lc + 1, xs[lc] + (BW + 46 if lc >= 100 else BW) + GAP)
                else:
                    x = d.boxes[yb].x
                rect = Box(x, y, BW, STRIP)
                key = (rect.x, rect.y)
                if key in strips and strips[key][0] > own_y:
                    continue
                name, args = op_for({"type": "branch", "row": r.id, "y": yb})
                strips[key] = (own_y, Place("branch", rect, None, None, False, "",
                                            M("new branch: variant of “{_row}”", _row=r.name), (name, args)))
        kept = []
        for _k, v in sorted(strips.items(), key=lambda kv: (kv[0][1], kv[0][0])):
            R = v[1].rect                         # one pixel, one place (only odd composition layouts collide)
            if not any(R.x < K.rect.right and K.rect.x < R.right and R.y < K.rect.bottom and K.rect.y < R.bottom
                       for K in kept):
                kept.append(v[1])
        out += kept
    if kind == "session":
        for r in p.rows:
            if not r.nights:
                continue
            ids = [n[0] for n in r.nights if n[0] in d.boxes]
            bot = max([d.boxes[s].bottom for s in ids] + ([d.cards[r.id].bottom] if r.id in d.cards else []))
            rect = Box(X0, bot + 4, 2 * BW + GAP, STRIP)
            if block:
                add("night", rect, M("Move into “{_row}”", _row=r.name), "move_night", {"night": block, "row": r.id})
            else:
                add("night", rect, M("Add a session to “{_row}”: it stacks with its sessions", _row=r.name),
                    "add_block", {"kind": "session", "at": {"type": "night", "row": r.id}})
    if kind in ("session", "image") and (block is None or kind == "session"):
        allb = list(d.boxes.values()) + list(d.cards.values())
        bottom = max([b.bottom for b in allb if b.x < xs[LIN] + BW] + [d.lines[0][1] if d.lines else 74])
        rect = Box(X0, bottom + 26, xs[LIN] + BW - X0, BH)          # below every strip (they end at +22)
        if block:
            add("row", rect, "Workflow: its own row", "move_night", {"night": block, "row": None})
        else:
            add("row", rect, "Workflow: Session/Image", "add_block", {"kind": kind, "at": {"type": "row"}})
    if kind in JOIN_KINDS and block is None:
        comp = [bx for b, bx in d.boxes.items() if g.cells.get(b, (0, 0))[0] is None]
        xc = xs[100]
        bottom = max([b.bottom for b in comp] + [max([b.bottom for b in d.boxes.values()] + [74])])
        add("comp", Box(xc, bottom + 20, BW + 46, BH), M("{lab}: new join", lab=LABEL[kind]),
            "add_block", {"kind": kind, "at": {"type": "comp"}})
        if kind == "combine_layers":
            for r in p.rows:
                last = r.blocks[-1] if r.blocks else None
                if last in d.boxes and p.blocks[last].kind in ("stars", "synth") \
                        and not any(w.src[0] == last and w.dst[0] != ix.post for w in p.wires):
                    bx = d.boxes[last]
                    add("comp", Box(bx.right + GAP, bx.y, BW, BH), "Combine layers: Starless → Background, Stars only → Stars",
                        "add_block",
                        {"kind": kind, "at": {"type": "comp", "after": last}})
    if kind in ("session", "image") and block is None:
        for r in p.rows:
            if not r.blocks or p.blocks[r.blocks[0]].kind != "combine" or r.fork:
                continue
            j = r.blocks[0]
            for port in in_ports(p.blocks[j]):
                if any(w.dst == [j, port] for w in p.wires):
                    continue
                px, py = d.ports[(j, port, "in")]
                add("port", Box(px - 44, py - STRIP / 2, 40, STRIP), M("New row for {port}", port=port),
                    "add_block", {"kind": kind, "at": {"type": "port", "block": j, "port": port}})
    return out


def _own_place(p, ix, block, name, args):
    """Whether a drag's place puts the dragged block (or Session) back where it is."""
    if name == "move_block":
        return ops.own_spot(ix, block, args["at"])
    if name == "move_night" and args.get("row") is not None and args.get("before") is None:
        r = ix.row.get(args["row"])
        return r is not None and bool(r.nights) and r.nights[-1][0] == block
    return False


def judge(p, view, place):
    from .judge import base, judge as fast
    res = fast(base(p, view), place.op[0], place.op[1])
    why, cell = res if res is not None else _simulate(p, view, place.op[0], place.op[1])
    place.lit, place.reason = not why, why
    if cell:
        place.line, place.col = cell
    return place


def places(p, view, kind=None, block=None):
    return [judge(p, view, c) for c in candidates(p, view, kind, block)]


def palette_state(p, view, kind):
    """(enabled, reason): enabled when one place is lit; else the reason of the natural place."""
    from .judge import base, judge as fast
    if kind in ("session", "image", "combine", "combine_layers"):
        return True, ""
    if kind == "calibration":
        return False, "each session already has its Calibration"
    if not p.rows:
        return False, "add a Session or an Image first"
    from .judge import specs
    from .model import FRAME_KINDS
    B = base(p, view)
    first = (M("{lab} belongs to a stack: between the Session and the Stacking", lab=LABEL[kind])
             if kind in FRAME_KINDS else "")
    for at in specs(B, kind):
        why = fast(B, "add_block", {"kind": kind, "at": at})[0]
        if not why:
            return True, ""
        first = first or why
    return False, first or M("{lab}: no place", lab=LABEL[kind])
