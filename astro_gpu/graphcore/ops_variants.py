"""The graph's gestures: variants (a branch or a copy of a row, Try, Keep, Discard)."""
import copy
from dataclasses import replace

from . import rules
from .model import JOIN_KINDS, Index, Row, new_id
from .msg import M, name_of
from .ops_base import (
    Out,
    Refusal,
    _after_subtree,
    _blk,
    _clone_block,
    _copy_block,
    _copy_calibration,
    _copy_flow,
    _night,
    _row,
    delete_rows,
    free_name,
)
from .ops_blocks import op_set_setting, op_switch_off
from .ops_settings import op_set_method, op_set_param
from .rules import LABEL


def _branch(q, b, rid, y):
    """Variant forked before y (after the row's last block when y is None); b replaces y's copy when same kind."""
    r = _row(q, rid)
    ix = Index(q)
    if q.blocks[b].kind == "synth":
        from .judge import NO_SYNTH_BRANCH
        raise Refusal(NO_SYNTH_BRANCH)
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
                q.wires.append(replace(w, id=new_id(q, "w"), src=[nb, w.src[1]], dst=list(w.dst), active=act))
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
    if b.kind == "synth":
        from .judge import NO_SYNTH_BRANCH
        raise Refusal(NO_SYNTH_BRANCH)
    rid = ix.row_of[block]
    r = ix.row[rid]
    x = ix.pred(block)
    frow = ix.row_of[x]
    tail = r.blocks[r.blocks.index(block):]
    if frow != rid:                                       # first block of a variant: a sibling of it
        tail = list(r.blocks)
    copies = [_copy_block(q, t) for t in tail]
    return _make_variant(q, frow, x, copies, tail, ix.row[frow].name)


# ---------------------------------------------------------------------------------------------------- Replace or Try
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
    if b.kind == "combine" and block in ix.row_of and ix.row[ix.row_of[block]].blocks[:1] == [block]:
        return _combination_variant(q, ix, block, change, name)
    # a Session, Post and an Image start nothing a variant could fork from; a Combine layers is a layer of Post, not
    # a workflow (its row stands for that layer): none of them has a variant
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
            nn = _night(q, q.blocks[s].name, {k: list(v_) for k, v_ in q.blocks[s].files.items()}, c is not None)
            q.blocks[nn[0]].sid = q.blocks[s].sid
            if c is not None:
                _copy_calibration(q, c, nn[1], q.blocks[s].sid)
            nights.append(nn)
            if c is not None and c == block:
                target = nn[1]
        v = Row(new_id(q, "r"), base, copies, nights)
        v.calib_node, v.calib_fork, v.calib = root.calib_node, root.calib_fork, copy.deepcopy(root.calib)
        v.align = copy.deepcopy(root.align)
        q.rows.insert(_after_subtree(q, root.id), v)
        for nb, ob in zip(copies, root.blocks):
            for w in list(q.wires):
                if w.src[0] == ob:
                    q.wires.append(replace(w, id=new_id(q, "w"), src=[nb, w.src[1]], dst=list(w.dst),
                                           active=q.blocks[w.dst[0]].kind == "post"))
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
    out.toast = M("variant “{_name}” under “{_base}”", _name=v.name, _base=r.name)    # a row like another: no header
    out.focus = target
    return out


def _combination_variant(q, ix, block, change, name=None):
    """Try variant from a combination (the editor's draft: client/combineedit.py): a variant forks after a block of
    its row, and a combination is its row's first block, with nothing before it in that row but the wires of its
    inputs, which a fork does not copy (the reason the general way refuses it). So the variant is a second combination
    of the same filters: its row's blocks copied (the recipe, its settings and weights with them), each input wired to
    the same image of the same workflow (port, channel, image read), the change made on the copy, its own workflow and
    its own layer in Post. The original is untouched; the variant is a workflow of its own (no Keep or Discard: there
    is nothing of the original to replace), and one Undo takes it away."""
    from .ops_base import _to_post
    from .ops_wires import fusion_legacy
    r = ix.row[ix.row_of[block]]
    copies = [_clone_block(q, t) for t in r.blocks]
    base = name if (name or "").strip() else name_of("{_row} · variant", _row=r.name)
    v = Row(new_id(q, "r"), "", copies, legacy=copy.deepcopy(r.legacy) or fusion_legacy())
    _copy_flow(v, r)
    v.name = free_name(q, base)
    q.rows.insert(q.rows.index(r) + 1, v)
    for w in [w for w in q.wires if w.dst[0] == block]:              # the same inputs, read the same way
        q.wires.append(replace(w, id=new_id(q, "w"), dst=[copies[0], w.dst[1]]))
    for nb, ob in zip(copies[1:], r.blocks[1:]):                     # the blocks after it: as in the original's row
        for w in [w for w in q.wires if w.src[0] == ob and w.dst[0] in r.blocks]:
            q.wires.append(replace(w, id=new_id(q, "w"), src=[nb, w.src[1]],
                                   dst=[copies[r.blocks.index(w.dst[0])], w.dst[1]]))
    if Index(q).post is not None:                                    # its own layer of Post
        _to_post(q, copies[-1])
    _apply_change(q, copies[0], change)
    out = Out(M("variant “{_name}” of “{_base}”", _name=v.name, _base=r.name), copies[0], new=copies[0])
    out.focus = copies[0]
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
            del q.blocks[s]
            if c is not None:
                del q.blocks[c]
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
