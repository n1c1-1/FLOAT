"""The graph's gestures: engine settings of a block, a row or a workflow, and a workflow's own pauses,
options and name."""
import copy
from .model import Index, JOIN_KINDS, Row
from . import combos
from .msg import M
from .rules import LABEL
from ..core import workflow as wfl
from .ops_base import Refusal, Out, _flow_chain, _blk, _row
from .ops_after import normalise


# ---------------------------------------------------------------------------------------------------- engine settings
#: what a stack without a Selection block does (compile.NO_SELECTION_STACKS_ALL)
NO_SELECTION = "no Selection: every frame is stacked, same weight"


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
        return [combos.stage(blk)]
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
        c = next((c for _s, c in (root.nights if root else []) if c is not None and str(q.blocks[c].sid) == sid),
                 None)
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
