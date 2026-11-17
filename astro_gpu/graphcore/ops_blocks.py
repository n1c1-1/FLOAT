"""The graph's gestures: blocks (insert, joins and their port rows, remove, switch off, pause, settings),
a row's chain replaced, a new branch, what a removal takes with it."""
import copy
from .model import (FRAME_KINDS, Index, JOIN_KINDS, POST_PORT, Row, SYNTH_PORT, Wire, clone, in_ports, new_block,
                    new_id, out_ports)
from . import combos
from . import rules
from . import model as _model
from .msg import M, join, name_of
from .rules import LABEL, OFFABLE, title_msg
from ..core import workflow as wfl
from .ops_base import (Refusal, Out, _flow_chain, _blk, _row, free_name, _night, _copy_calibration, _map_port,
                       _after_subtree, delete_rows, _auto_align, new_align, _night_off, detach, _copy_flow)
from .ops_after import _is_layers_row, _alloc, normalise, legality
from .ops_rows import _night_row, op_remove_night
from .ops_settings import NO_SELECTION, _flow_of, _stages_of, _store_all
from .ops_wires import op_set_combination_type, fusion_legacy


def _insert(q, b, rid, after):
    r = _row(q, rid)
    if after is None:
        if not r.fork and not r.nights:
            raise Refusal("nothing is inserted before the head of the row")
        if q.blocks[b].kind == "synth":                  # (switched off, the engine skips it): never a row's head,
            from .judge import NO_SYNTH_BRANCH           # it reads the stars of the block before it in its row
            raise Refusal(NO_SYNTH_BRANCH)
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
        if q.blocks[b].kind == "synth":                  # its stars feed the Synthetic stars of the Combine layers
            for w in list(q.wires):                      # of its workflow (its two images from the same Stars)
                j = w.dst[0]
                own = any(x.src == [after, "Starless"] and x.dst == [j, "Background"] for x in q.wires)
                if w.src[0] == after and w.src[1] in ("Stars only", "image") and w.dst[1] == "Stars" and w.active \
                        and q.blocks[j].kind == "combine_layers" and (own or _model.SYNTH_ON_STARS_SIDE) and \
                        not any(x.dst == [j, SYNTH_PORT] for x in q.wires):
                    q.wires.append(Wire(new_id(q, "w"), [b, SYNTH_PORT], [j, SYNTH_PORT]))
    return Out(M("{lab} here, in “{_row}”", lab=LABEL[q.blocks[b].kind], _row=r.name), b, new=b)


def _add_join(q, kind, after, type=None):  # noqa: A002
    if kind not in JOIN_KINDS:
        raise Refusal("the composition takes a Combination or Combine layers")
    if kind == "combine":                                   # its recipe (the default one) copied in
        if (type or combos.DEFAULT) not in combos.offered():
            raise Refusal("unknown type")
        settings, source = combos.placed(type or combos.DEFAULT)
        j = new_block(q, kind, settings=settings, source=source)
    else:
        j = new_block(q, kind)
    name = free_name(q, name_of(combos.title(j) if kind == "combine" else LABEL[kind]))
    q.rows.append(Row(new_id(q, "r"), name, [j.id], legacy=fusion_legacy() if kind == "combine" else None))
    post = Index(q).post
    if after is None and kind == "combine_layers" and post is not None:
        # placed on its own, its inputs to wire: its image is a layer of Post from the start, as a combination's is
        # (one family, one way: the block is wired to Post, the layer comes with its inputs)
        q.wires.append(Wire(new_id(q, "w"), [j.id, "image"], [post, POST_PORT]))
    if after is not None:
        a = _blk(q, after, ("stars", "synth"))
        syn = None
        if a.kind == "synth":                               # the stars it reads give the Background and the Stars
            ix = Index(q)
            syn, a = a, q.blocks[ix.pred(after)] if ix.pred(after) is not None else None
            if a is None or a.kind != "stars":
                raise Refusal("this Stars/background already feeds a block")
        post = Index(q).post
        if kind != "combine_layers" or any(w.src[0] in (a.id, after) and w.dst[0] != post for w in q.wires):
            raise Refusal("this Stars/background already feeds a block")
        for w in q.wires:
            if w.src[0] == a.id and w.dst[0] == post:      # the row's output passes to the new join
                w.src = [j.id, "image"]
        if post is not None and not any(w.src[0] == j.id and w.dst[0] == post for w in q.wires):
            q.wires.append(Wire(new_id(q, "w"), [j.id, "image"], [post, POST_PORT]))   # its image: a layer of Post
        q.wires.append(Wire(new_id(q, "w"), [a.id, "Starless"], [j.id, "Background"]))
        q.wires.append(Wire(new_id(q, "w"), [a.id, "Stars only"], [j.id, "Stars"]))
        if syn is not None and not syn.off:
            q.wires.append(Wire(new_id(q, "w"), [syn.id, SYNTH_PORT], [j.id, SYNTH_PORT]))
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
    # the Session and what the port needs, nothing more: an image, which frames give once stacked (the engine's
    # Stacking first); a starless image (or the stars only) when the graph refuses an image with its stars there
    # (what follows the combination reads one): then Denoise and Stars/background too (the engine's needs of a star
    # removal from frames)
    cand = PORT_CHAINS[0]
    for c in port_chains(q, jid, port):
        t = clone(q)
        _port_row(t, jid, port, name, c)
        normalise(t)
        if not legality(t):
            cand = c
            break
    return _port_row(q, jid, port, name, cand)


def _port_row(q, jid, port, name, cand):
    kinds, src_port, why = cand
    blocks = [new_block(q, k).id for k in kinds]
    r = Row(new_id(q, "r"), name, blocks, [_night(q, name, calibration=False)])
    _auto_align(q, r)
    q.rows.append(r)
    q.wires.append(Wire(new_id(q, "w"), [blocks[-1], src_port], [jid, port],
                        channel="L" if q.blocks[jid].kind == "combine" else None))
    return Out(join([M("New row for {port}", port=port), why]), r.nights[0][0], new=r.nights[0][0])


#: what a Session dropped on a combination's port is given: its blocks, the port its wire leaves from, why
PORT_CHAINS = [(("stacking",), "image", "Stacking added: the port reads an image"),
               (("stacking", "denoise", "stars"), "Starless",
                "Stacking, Denoise, Stars/background added: the port reads a starless image"),
               (("stacking", "denoise", "stars"), "Stars only",
                "Stacking, Denoise, Stars/background added: the port reads the stars only")]


def port_chains(q, jid, port):
    """The chains of PORT_CHAINS tried, shortest first, for a new stack feeding `port` of join jid: an image, then
    the image the port reads (Combine layers: Background a starless image, Stars the stars only), or the one the
    combination's first wired input reads (a Stars/background's starless or stars only)."""
    if q.blocks[jid].kind == "combine_layers":
        want = "Starless" if port == "Background" else "Stars only"
    else:
        first = next((w for p_ in in_ports(q.blocks[jid]) for w in q.wires if w.dst == [jid, p_] and w.active), None)
        want = first.src[1] if first is not None else None
    out = [PORT_CHAINS[0]]
    if want in ("Starless", "Stars only"):
        out.append(next(c for c in PORT_CHAINS if c[1] == want))
    return out


def layers_free(q, ix, b):
    """Whether a Combine layers may be put after row-end block b (a Stars/background, or a Synthetic stars after one):
    the Stars/background's images feed no block yet (Post aside), nor the Synthetic stars' own."""
    post = ix.post
    ends = [b]
    if q.blocks[b].kind == "synth":
        a = ix.pred(b)
        if a is None or q.blocks[a].kind != "stars":
            return False
        ends.append(a)
    return not any(w.src[0] in ends and w.dst[0] != post for w in q.wires)


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
        return _remove_calibration(q, block)
    if b.kind == "session":
        return op_remove_night(q, block)
    rid = ix.row_of[block]
    if ix.pos[block] == 0 and b.kind in ("image",) + JOIN_KINDS:
        toast = delete_rows(q, [rid])
        return Out(toast, old=None if q._astro_moved else block, style="need")
    detach(q, block)
    if b.kind == "selection":            # what its absence means (compile: the engine's "All frames")
        return Out(M("{lab} deleted · {why}", lab=LABEL[b.kind], why=NO_SELECTION), old=block, style="need")
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
            if c is not None:
                q.blocks[c].off = b.off
    said = M("{lab} switched off" if b.off else "{lab} switched on", lab=LABEL[b.kind])
    ix = Index(q)
    r = ix.row.get(ix.row_of.get(block))
    if r is not None:                                 # with the workflow's name, as today's page said it
        said = join([said, M("{_row}", _row=r.name)])
    return Out(said, block, style="off")


def _calibrated_as_one(q, r):
    """The nights of stack r whose Calibration a gesture on one of them changes: all of them calibrated together
    (the engine's one Calibration of the stack), else that night alone."""
    return q.calibrate_together and len(r.nights) > 1


def _add_calibration(q, night):
    """A Calibration put back beside a night's Session (calibrated together: beside each night of its stack, the
    engine's one Calibration of the stack)."""
    r, n = _night_row(q, night)
    if n[1] is not None:
        raise Refusal(M("“{_name}” already has its Calibration", _name=q.blocks[night].name))
    off = _night_off(q, r)
    todo = [x for x in r.nights if x[1] is None] if _calibrated_as_one(q, r) else [n]
    for x in todo:
        x[1] = new_block(q, "calibration", sid=q.blocks[x[0]].sid, off=off).id
    r.calib_node = True                   # drawn = compiled (a stack read without the engine's Calibration node)
    if len(todo) > 1:
        toast = M("Calibration added to the {n} sessions of “{_row}”: calibrated together", n=len(todo), _row=r.name)
    else:
        toast = M("{lab} here, in “{_row}”", lab=LABEL["calibration"], _row=r.name)
    return Out(toast, n[1], new=n[1])


def _remove_calibration(q, block):
    """A night's Calibration deleted: its lights are taken as they are (the engine applies no master to them).
    Calibrated together, the Calibration of every night of its stack goes (the engine's one Calibration of the
    stack)."""
    ix = Index(q)
    r = ix.row[ix.row_of[block]]
    todo = [x for x in r.nights if x[1] is not None] if _calibrated_as_one(q, r) else \
        [x for x in r.nights if x[1] == block]
    for x in todo:
        del q.blocks[x[1]]
        x[1] = None
    if len(todo) > 1:
        toast = M("Calibration deleted from the {n} sessions of “{_row}”: calibrated together, their lights taken as "
                  "they are", n=len(todo), _row=r.name)
    else:
        toast = "Calibration deleted: the lights are taken as they are"
    return Out(toast, old=block, style="need")


def _follow_flow_pauses(q, ix, block, on):
    """A workflow holding its own list of pauses (``pauses``: the blocks' pauses do not give them, compile.py) takes
    the pause set or removed on one of its blocks into that list, under the block's engine name (compile's: its
    stored node name, else its kind, "~n" for the n-th of a kind in the chain; a night's Calibration "calibration",
    or "calibration@<session>" when the nights are calibrated each on its own)."""
    from ..core import workflow as wfl_
    b = q.blocks[block]
    for f in list(q.rows) + list(q.taps):
        if f.pauses is None:
            continue
        if hasattr(f, "blocks"):
            chain, root = ix.effective(f.id), ix.row[ix.root(f.id)]
        elif f.block in ix.row_of:
            eff = ix.effective(ix.row_of[f.block])
            chain, root = eff[:eff.index(f.block) + 1], ix.row[ix.root(ix.row_of[f.block])]
        else:
            continue
        if b.kind == "calibration":
            pairs = [(s, c) for s, c in root.nights if c == block]
            if not pairs:
                continue
            sep = not q.calibrate_together and len(root.nights) >= 2
            sid = q.blocks[pairs[0][0]].sid
            name = f"calibration@{sid}" if sep and sid is not None else "calibration"
        elif block in chain:
            same = [x for x in chain[:chain.index(block) + 1] if q.blocks[x].kind == b.kind]
            name = b.engine or (b.kind if len(same) == 1 else f"{b.kind}~{len(same)}")
        else:
            continue
        have = [x for x in f.pauses if x != name]
        f.pauses = sorted(have + [name], key=wfl_.block_order) if on else have


def op_set_pause(q, block):
    b = _blk(q, block)
    if b.kind == "post":
        raise Refusal("Post is fixed")
    if b.kind == "selection" and not b.pause:
        raise Refusal("Selection computes nothing of its own: set the pause on Registration")
    b.pause = not b.pause
    ix = Index(q)
    _follow_flow_pauses(q, ix, block, b.pause)
    lab = title_msg(q, block, ix)
    if not b.pause:
        return Out(M("No more pause after {block}", block=lab), block)
    n = sum(1 for r in q.rows if block in ix.effective(r.id) or any(block in nt for nt in r.nights or []))
    if n > 1:                                          # the workflows it stops, as today's page said it
        return Out(M("Review pause after {block} · {n} workflows", block=lab, n=n), block)
    return Out(M("Review pause after {block}", block=lab), block)


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
    last = r.blocks[-1] if r.blocks and r.blocks[-1] in gone else None
    end = (head + new)[-1] if head + new else None
    for w in q.wires:              # what its image fed (a combination's input) leaves from its new last block
        if w.src[0] == last and end is not None and w.dst[0] not in gone and w.src[1] in out_ports(q.blocks[end]):
            w.src = [end, w.src[1]]
    for b in gone:
        del q.blocks[b]
    q.wires = [w for w in q.wires if w.src[0] not in gone and w.dst[0] not in gone]
    r.blocks = head + new
    if r.nights and calibration is not None:
        r.calib_node = True               # drawn = compiled: a stack without Calibration has none on its nights
        if calibration is False:
            for n in r.nights:
                if n[1] is not None:
                    del q.blocks[n[1]]
                    n[1] = None
        else:
            off = bool(calibration.get("off")) if isinstance(calibration, dict) else False
            for n in r.nights:
                if n[1] is None:
                    n[1] = new_block(q, "calibration", sid=q.blocks[n[0]].sid, off=off).id
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
            nn = _night(q, q.blocks[s].name, {k: list(v) for k, v in q.blocks[s].files.items()}, c is not None)
            q.blocks[nn[0]].sid = q.blocks[s].sid
            if c == block:                            # the branch's own Calibration: a new block, from its defaults
                q.blocks[nn[1]].sid = q.blocks[s].sid
            elif c is not None:                       # the other nights: as in the stack it leaves
                _copy_calibration(q, c, nn[1], q.blocks[s].sid)
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
