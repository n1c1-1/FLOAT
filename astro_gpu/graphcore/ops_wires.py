"""The graph's gestures: wires, taps to Post and combinations (recipes, inputs)."""
import copy
from .model import Index, POST_PORT, Row, SYNTH_PORT, Tap, Wire, in_ports, new_block, new_id, out_ports
from . import combos
from .msg import M, join, name_of
from .rules import LABEL, title_msg
from ..core import workflow as wfl
from .ops_base import Refusal, Out, _flow_chain, _blk, free_name, _to_post, _copy_flow
from .ops_after import _is_layers_row, _alloc
from .ops_settings import owned_stages, check_param, _put


# ---------------------------------------------------------------------------------------------------- wires
def op_set_combination_type(q, block, type):
    """A combination of another type: the folder's recipe of that type copied in; the wires into inputs it does not
    have go, and its settings of its former stage with them (the weights of the inputs kept stay)."""
    j = _blk(q, block, ("combine",))
    if type not in combos.offered():
        raise Refusal("unknown type")
    old = combos.stage(j)
    j.settings, j.source = combos.placed(type)
    ports = combos.ports(j)
    new = combos.stage(j)
    if new != old and old in j.params:
        mine = j.params.pop(old)
        roles = {combos.role_of(j, p) for p in ports}
        weights = {r: v for r, v in (mine.get("weights") or {}).items() if r in roles}
        if weights:
            j.params[new] = {"weights": weights}
    gone = [w for w in q.wires if w.dst[0] == block and w.dst[1] not in ports]
    q.wires = [w for w in q.wires if w not in gone]
    toast = M("{t}: {ports}", t=title_msg(q, block), ports=join(ports, ", "))
    if gone:
        toast = join([toast, M("removed: {gone}", gone=join(
            [M("wire from “{t}” on {port}", t=title_msg(q, w.src[0]), port=w.dst[1]) for w in gone], ", "))])
    return Out(toast, block)


def op_unwire(q, wire):
    w = next((x for x in q.wires if x.id == wire), None)
    if w is None:
        raise Refusal("wire not found")
    if w.dst[1] == SYNTH_PORT and q.blocks[w.src[0]].kind == "synth" and not q.blocks[w.src[0]].off:
        # the port follows its workflow's block (Post's synthetic stars row): the block goes, or is switched off
        raise Refusal("this port empties with its Synthetic stars block: delete it or switch it off")
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
    # (Synthetic stars: a workflow of its own too, its image beside the Combine layers it feeds, a layer of Post)
    synth = b.kind == "synth"
    if block in ix.row_of and (synth or not _flow_ends(q, ix, block)) and not _is_layers_row(q, ix, ix.row_of[block]):
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


def op_add_combination(q, type=combos.DEFAULT, name=None):  # noqa: A002
    """A combination of a recipe of the folder (its named inputs), its recipe copied in, empty, its image a layer of
    Post."""
    if type not in combos.offered():
        raise Refusal("unknown type")
    settings, source = combos.placed(type)
    j = new_block(q, "combine", settings=settings, source=source)
    base = (name or "").strip() or name_of(combos.title(j))
    r = Row(new_id(q, "r"), free_name(q, base), [j.id], legacy=fusion_legacy())
    q.rows.append(r)
    return Out(M("{_name} added", _name=r.name), j.id, new=j.id)


def op_update_recipe(q, block):
    """The folder's version of a combination's recipe copied into the block (the user's click, never by itself;
    api.apply_combination_recipe on the block): the wires of inputs it no longer has go, the others follow their role;
    its settings follow the recipe's parameters (those it no longer has go; the block's stage may change)."""
    from ..stages import recipes as rc
    from ..stages.combine import ALL
    j = _blk(q, block, ("combine",))
    new = combos.recipe(combos.kind(j))
    if new is None:
        raise Refusal("the application no longer has this recipe")
    st = combos.status(j)
    if not (st["newer"] or st["changed"]):
        raise Refusal("the recipe is up to date")
    old_stage, old_ports = combos.stage(j), {p: combos.role_of(j, p) for p in combos.ports(j)}
    j.source = {"recipe": new}
    new_stage = combos.stage(j)
    roles = set(rc.roles(new))
    mine = dict(j.params.pop(old_stage, None) or {})
    vals = {**(mine.pop("values", None) or {}), **{k: v for k, v in mine.items() if k != "weights"}}
    vals = {k: v for k, v in vals.items() if k in new["parameters"]}
    kept = {}
    if new_stage == rc.RECIPE_STAGE:
        if vals:
            kept["values"] = vals
    else:
        cls = next(c for c in ALL if c.name == new_stage)
        kept.update({k: v for k, v in vals.items() if k in cls.Params.model_fields})
    weights = {r: x for r, x in (mine.get("weights") or {}).items() if r in roles}
    if weights:
        kept["weights"] = weights
    if kept:
        j.params[new_stage] = kept
    gone = []
    for w in q.wires:
        if w.dst[0] != block:
            continue
        role = old_ports.get(w.dst[1], w.dst[1])
        if role in roles:
            w.dst = [block, combos.port_of(j, role)]
        else:
            gone.append(w)
    q.wires = [w for w in q.wires if w not in gone]
    return Out(M("{t}: recipe version {v}", t=title_msg(q, block), v=new["version"]), block)


def op_set_recipe_value(q, block, name, value=None):
    """A parameter of the combination's recipe (value None: back to its default), where its stage reads it: the
    four types' frozen recipes, their stage's own setting of that name; another recipe, its ``values``, checked against
    the recipe (api.set_recipe_value on the block)."""
    from ..stages import recipes as rc
    from ..stages.combine import recipe_value
    from ..core.stage import StageError
    j = _blk(q, block, ("combine",))
    spec, st = combos.spec(j), combos.stage(j)
    if spec is None or name not in spec["parameters"]:
        raise Refusal("unknown setting")
    if st != rc.RECIPE_STAGE:
        _put(j.params, st, name, None if value is None else check_param(st, name, value))
        return _recipe_said(spec, name, value, block)
    vals = dict((j.params.get(st) or {}).get("values") or {})
    if value is None:
        vals.pop(name, None)
    else:
        try:
            v = recipe_value(type("V", (), {"values": {name: value}})(), spec, name)
        except StageError as e:
            raise Refusal(M("{_why}", _why=str(e))) from None
        if v == spec["parameters"][name]["default"]:
            vals.pop(name, None)
        else:
            vals[name] = v
    _put(j.params, st, "values", vals or None)
    return _recipe_said(spec, name, value, block)


def _recipe_said(spec, name, value, block):
    lab = spec["parameters"][name].get("label") or name
    if value is None:
        return Out(M("{key} back to its default", key=lab), block)
    return Out(M("{key}: {_value}", key=lab, _value=str(value)), block)


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
        role = combos.role_of(j, port)
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
    nj = new_block(q, "combine", settings={k: v for k, v in j.settings.items() if k == "type"},
                   source=copy.deepcopy(j.source))                                       # the same recipe
    nr = Row(new_id(q, "r"), free_name(q, name_of("{_name} · Stars", _name=r.name)), [nj.id], legacy=fusion_legacy())
    q.rows.insert(q.rows.index(r), nr)
    for w in ins:
        q.wires.append(Wire(new_id(q, "w"), [w.src[0], "Stars only"], [nj.id, w.dst[1]], True, w.wid, w.channel))
    return Out(M("{_name} added", _name=nr.name), nj.id, new=nj.id)
