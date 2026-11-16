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
from .ops_blocks import (_insert, _add_join, _add_port_row, _port_row, PORT_CHAINS, port_chains, layers_free,
                         own_spot, op_remove_block, op_switch_off, _calibrated_as_one, _add_calibration,
                         _remove_calibration, _follow_flow_pauses, op_set_pause, op_set_setting, op_apply_to_rows,
                         op_replace_chain, op_add_branch, dependants, say_text, op_remove_with_dependants)
from .ops_variants import (_branch, _make_variant, op_add_variant, _apply_change, op_try_variant, _variant_parts,
                           op_keep_variant, op_discard_variant)


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


OPS = {k[3:]: v for k, v in dict(globals()).items() if k.startswith("op_")}
