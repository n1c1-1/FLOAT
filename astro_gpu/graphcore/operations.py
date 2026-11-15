"""Operations (§4). Every operation runs on a clone of the graph; refused = nothing changes. No history here:
the caller keeps the graphs it wants to go back to."""
import copy
from dataclasses import dataclass, replace

from .shapes import Result
from .model import (Index, Row, Wire, Tap, check, clone, model_json, new_block, new_id, in_ports, out_ports, edges,
                    FILE_KINDS, FRAME_KINDS, JOIN_KINDS, LAYER_PORTS, POST_PORT, SYNTH_PORT)
from . import combos
from . import rules
from . import model as _model
from .msg import M, join, typed, name_of, TR
from .rules import LABEL, OFFABLE, evaluate, need_greyed, title_msg, inherited, rank
from ..core import workflow as wfl

# The gestures live in the modules below by family; operations keeps the applying (apply, the registry OPS, the
# gestures that try another one through it) and gives every name it had, so its importers read them from here.
# ruff: noqa: F401
from .ops_base import (COMMON, Refusal, Out, _flow_chain, _blk, _row, _names, free_name, session_name, _night,
                       _copy_calibration, _stack_calibrated, _follow_stack, _to_post, _map_port, _copy_block,
                       _after_subtree, delete_rows, _auto_align, new_align, _night_off, _insert_after_frames, detach,
                       would_cycle, fold_guard, _kind, _dedup, _clone_block, _copy_flow, _file_name)
from .ops_after import (_ids, _is_layers_row, _alloc, _sessions_state, _sync_shared, WHOLE_VARIANT, whole_original,
                        _not_into_whole_variant, _whole_variants_follow, _flow_sig, _legacy_chains, _thaw, _created,
                        _summarise, _acted, _view_op, fold_problem, finish, normalise, _norm_params, _mid_row,
                        _norm_taps, _norm_names, _norm_align, legality, _styled)
from .ops_rows import (_named, _files, op_add_row, _image_name, _first, op_delete_row, op_add_night, _night_row,
                       _forget_night, op_remove_night, _next_sid, op_new_night, op_add_stack, op_share_night,
                       _no_clash, _shared, _not_shared, op_move_night, _eff_list, _exclude_in, _carry_excluded,
                       op_merge_nights, op_split_night, op_merge_rows, op_duplicate_row, op_rename_row,
                       op_rename_night, op_set_reference, op_set_align, op_own_grid, op_set_align_off, _join_problem,
                       op_set_calibrate_together)
from .ops_files import (_session, op_add_files, op_remove_files, op_use_for_all_nights, op_set_common_files,
                        op_set_common_setting, op_reset_own, op_reset_own_files, _list_of, _set_list, op_move_files,
                        op_split_by, _calib_block, op_set_own_calibration, op_follow_session,
                        op_remove_own_calibration, op_set_night_filter, _entry, _drawn, op_set_session_files,
                        op_rename_session, op_set_session_filter, op_set_left_out, op_remove_session,
                        op_exclude_frames, op_exclude_files, excluded_files, op_set_image_file)
from .ops_settings import (NO_SELECTION, SELECTION_KEYS, _engine_name, owned_stages, check_param, _put, op_set_param,
                           _flows_through, _unhide, op_set_row_param, op_set_method, _flow_of, op_rename_tap,
                           op_set_flow_pauses, op_set_flow_options, _stage_names, _stages_of, _holder_of, _store,
                           op_set_flow_param, _store_all, op_store_params)
from .ops_wires import (op_set_combination_type, op_unwire, op_take_alternative, _flow_ends, _post_tap,
                        op_send_to_post, op_remove_tap, fusion_legacy, _channel, op_add_combination, op_update_recipe,
                        op_set_recipe_value, _recipe_said, _input, op_set_input, _reimage, op_add_stars_combination)


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
    _whole_variants_follow(q)
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


# ---------------------------------------------------------------------------------------------------- blocks
def op_add_block(q, kind, at=None, row=None, _block=None, type=None):  # noqa: A002
    """``type``: a Combination's recipe (its id; None: the default one), placed with its name and its inputs."""
    if kind not in LABEL or kind == "post":
        raise Refusal("unknown block")
    if type is not None and kind != "combine":
        raise Refusal("unknown type")
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
        return _add_join(q, kind, at.get("after"), type)
    if t == "port":
        return _add_port_row(q, kind, at["block"], at["port"])
    if t == "calib":
        if kind != "calibration":
            raise Refusal(M("{lab} is not inserted in a row", lab=LABEL[kind]))
        return _add_calibration(q, at.get("night"))
    if kind in ("session", "image") + JOIN_KINDS:
        raise Refusal(M("{lab} is not inserted in a row", lab=LABEL[kind]))
    if kind == "calibration":
        raise Refusal("a Calibration goes beside a session")
    if _block is None and kind == "stacking":
        # a new stack: the camera white balance as the project's first stack has it, on in a new project (issue #6)
        from .templates import camera_white_balance
        wb = camera_white_balance(q)
    b = _block or new_block(q, kind).id
    if _block is None and kind == "stacking" and wb:
        _put(q.blocks[b].params, "stack", "white_balance", True)
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
                    and not any(w.src[0] == r.blocks[-1] for w in q.wires) and layers_free(q, ix, r.blocks[-1]):
                return {"type": "comp", "after": r.blocks[-1]}
        return {"type": "comp"}
    if kind == "calibration":
        for r in ([ix.row[row]] if row in ix.row else []) + list(q.rows):
            n = next((n for n in r.nights if n[1] is None), None)
            if n is not None:
                return {"type": "calib", "night": n[0]}
        raise Refusal("each session already has its Calibration" if any(r.nights for r in q.rows)
                      else "add a Session first")
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


def op_wire(q, src, src_port, dst, dst_port, channel=None):
    s, d = _blk(q, src), _blk(q, dst)
    ix = Index(q)
    if src_port not in out_ports(s):
        raise Refusal("lights first" if not out_ports(s) else M("no output {port}", port=src_port))
    if d.kind == "synth" and dst_port == SYNTH_IN:
        return _synth_on(q, ix, src, src_port, dst)
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
    if d.kind == "combine":                 # a combination's input reads a channel of the image (its recipe's: grey
        default = next((i["channel"] for i in combos.inputs(d) if i["label"] == dst_port), "L")    # but a dual-band's
        w.channel = _channel(channel or default)                                                   # red for H-alpha)
    q.wires.append(w)
    return Out(M("wire {a} → {b}", a=title_msg(q, src), b=title_msg(q, dst)), w.id)


#: the input a Synthetic stars block is drawn with: the real stars it reads (its chain: the block before it)
SYNTH_IN = "Stars"


def _synth_on(q, ix, src, src_port, synth):
    """A wire pulled from a block's stars to a Synthetic stars' input: the block is put on those stars (moved after the
    Stars/background that gives them, in its workflow); already there, nothing to do."""
    stars_combo = _model.SYNTH_ON_STARS_SIDE and q.blocks[src].kind == "combine" and src_port == "image"
    if not stars_combo and (q.blocks[src].kind != "stars" or src_port != "Stars only"):
        raise Refusal("Synthetic stars read the stars only of a Stars/background")
    if ix.pred(synth) == src:
        raise Refusal("this wire already exists")
    out = op_move_block(q, synth, {"type": "insert", "row": ix.row_of[src], "after": src})
    out.toast = M("{lab} on the stars of “{_row}”", lab=LABEL["synth"], _row=Index(q).row[ix.row_of[src]].name)
    return out


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


OPS = {k[3:]: v for k, v in dict(globals()).items() if k.startswith("op_")}
