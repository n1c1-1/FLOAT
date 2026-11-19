"""A combination's master layer in Post: what the model tells Post of a combination (``describe``), where its settings
live (the layer's ``tune``), the graph's settings a layer takes a copy of (``layers_graph``, ``put_copy``), and its
filters balanced as the run balances them (``balance_maps``, ``balanced_planes``). No Qt here.

The graph's settings and Post's layers are two sources, bridged by a copy. Inspect sets the GRAPH's settings of a block
of the combination family, kept on its block of the graph (graphcore rules.GRAPH_SETTINGS), never in a result's key:
a combination's hues and intensities (when Post mixes its filters itself, below), a Combine layers' blend, base stretch
and sliders. A layer of Post takes a copy of them when it is made (model.with_graph_copy), and again when the user asks
for it (Post's Graph settings command, one step of Post's Undo); between the two, the layer is its own: Inspect never
changes it, Post never changes the graph. A layer saved before holds its own settings, as it always did.

The Combine layers layer over a combination's Output (``starless`` "output") is the combination's master layer when
Post can make the combination's picture itself: the recipe mixes its balanced filters through hues or a matrix (no
lightness of its own: LRGB and the recipes composed as lightness and colour stay plain layers), its inputs are not
the stars alone (the stars' combination is the layer's Stars side), and no block follows the combination in its
workflow (a Color or Denoise block after it changes its picture, which Post cannot make again).

Its hues and its filters' intensities are then the layer's in Post (``tune`` = {"hues": {role: degrees}, "weights":
{role: intensity}}; none: the block's stored values) and the graph's settings in Inspect (the same shape, on the
block), and they never outdate the block: Post mixes the filters itself. The block's own stored values are left as
they are (its keys do not move). The balance changes the gains the run measures: it stays the block's, a setting that
needs a run. For a plain layer every setting stays the block's.

While Post shows it the master layer holds the description below as ``fusion`` (never saved, given again at each load:
postengine.save_layers drops it):

    {"stage", "roles": [filters, in the recipe's order], "names": {role: shown name}, "hue_params": {role: the
     setting of its hue}, "hues": {role: degrees}, "hue_defaults": {role: its recipe's default hue}, "matrix": [rows
     R, G, B], "weights": {role: intensity}, "balance": "sky" | "signal", "palette": the palette the hues are ("" for
     the user's own)}

The balanced filters are the combination's own, the gains its run measured: the linear maps the run wrote
(drz_meta.json, ``balance``, stages/combine.Combination.steps). A result made before the maps were written, a balance
changed since the run, or inputs computed again since, are balanced by the same ``steps`` on the whole inputs, once
(kept for the session, ``balance_maps``): Post, the editor's preview and the run then show the same gains.
"""
from __future__ import annotations

import json
import re
import threading
from collections import OrderedDict
from pathlib import Path

import numpy as np

#: the stage of a workflow's Output block (its picture is the combination's when nothing follows it)
OUTPUT_STAGE = "render_simple"
#: a master layer's own settings (module docstring)
TUNE = "tune"
#: the own settings of the combinations nested in a layer, by the path of their node (``child``): {path: the same shape
#: as ``tune``}. The layer's own mix (the starless side's, or the combination wired to Post) is ``tune``, as ever.
TUNES = "tunes"


def plain_reason(spec: dict | None) -> str:
    """Why a combination's image cannot be mixed by Post, from its recipe alone ("" when it can): a recipe composed as
    lightness and colour (LRGB...). What the graph puts around a combination (a block after it, its depth) is the layer
    tree's (graphcore/post.mix_tree: a fixed node); the stars' combination is a mix of its own (``describe``)."""
    if spec is None:
        return "lightness"
    if (spec.get("output") or {}).get("lightness") is not None:
        return "lightness"
    return ""


def _hue_names(spec: dict) -> dict[str, str]:
    from .combineedit import hue_params
    return hue_params(spec or {})


def tuned_stored(stage: str, spec: dict, stored: dict, tune: dict | None) -> dict:
    """The block's stored settings with a master layer's own (``tune``) in place of its hues and weights: what the
    editors read of the combination."""
    if not tune:
        return dict(stored)
    from .combineedit import stage_class
    fields = stage_class(stage).Params.model_fields
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in stored.items()}
    for role, name in _hue_names(spec).items():
        if role not in (tune.get("hues") or {}):
            continue
        v = float(tune["hues"][role])
        if name in fields and name != "values":
            out[name] = v
        else:
            out["values"] = {**(out.get("values") or {}), name: v}
    if "weights" in tune:
        out["weights"] = {r: float(w) for r, w in tune["weights"].items() if abs(float(w) - 1.0) > 1e-9}
    return out


def split_change(spec: dict, values: dict) -> tuple[dict, dict]:
    """(the master layer's part, the block's part) of a change of a combination's settings (the editors' kind
    "combine": {a hue's setting: degrees, "weights": {role: intensity}, "balance": ..., ...}): the hues and the
    intensities are the layer's, the rest (the balance, the stars) the block's."""
    by_name = {n: r for r, n in _hue_names(spec).items()}
    layer: dict = {}
    block: dict = {}
    for k, v in values.items():
        if k in by_name:
            layer.setdefault("hues", {})[by_name[k]] = float(v)
        elif k == "weights":
            layer["weights"] = {r: float(w) for r, w in v.items()}
        else:
            block[k] = v
    return layer, block


def new_tune(stage: str, spec: dict, stored: dict, tune: dict | None, part: dict) -> dict:
    """A master layer's ``tune`` after a change of its hues or intensities (``part``, ``split_change``'s): whole
    (every filter's hue and intensity), from what it held, else from the block's stored values."""
    f = describe(stage, spec, tuned_stored(stage, spec, stored, tune), {})
    hues = {**f["hues"], **(part.get("hues") or {})}
    weights = {**f["weights"], **(part.get("weights") or {})}
    return {"hues": {r: float(v) for r, v in hues.items()}, "weights": {r: float(v) for r, v in weights.items()}}


# ----------------------------------------------------------------------------------------------- the layer tree
#: the nodes of the layer tree are named by a path in every saved key: the layer itself is "", its sides "starless" and
#: "stars", a combination's filter ``child(path, role)``: "f:Ha" under the layer's own mix (the key the starless
#: master's filters have always had), "stars/f:Ha" under the stars side's, "f:R/f:Ha" under the mix of the filter R
ROLE = "f:"
SIDES = ("starless", "stars")


def child(path: str, role: str) -> str:
    """The path of the node a combination's input ``role`` is, the combination being at ``path``."""
    return ("" if path in ("", "starless") else path + "/") + ROLE + role


def parent(path: str) -> str:
    """The path of the node above ``path`` ("" for a side or the layer; a side is its own mix's: "starless"
    and "stars" stay what they are)."""
    if path in ("", "starless", "stars"):
        return ""
    head, _sep, _last = path.rpartition("/")
    return head if head else "starless"


def node_paths(tree: dict) -> dict[str, dict]:
    """{path: node} of a layer tree (graphcore/post.mix_tree), every node of it, the layer's own node first. A Combine
    layers is at "" and its sides at "starless" and "stars"; a combination wired straight to Post is at ""."""
    out: dict[str, dict] = {}

    def walk(n, path):
        out[path] = n
        if n["kind"] == "layers":
            for side, c in n["sides"].items():
                walk(c, side)
        elif n["kind"] == "mix":
            for r, c in n["roles"].items():
                walk(c, child(path, r))
    walk(tree, "")
    return out


_PATH = re.compile(r"(?:stars|starless|(?:stars/)?f:[^/\s]+(?:/f:[^/\s]+)*)")


def is_path(path) -> bool:
    """Whether ``path`` can name a node of a layer tree: a side ("starless", "stars") or a filter's path (``child``)."""
    return isinstance(path, str) and _PATH.fullmatch(path) is not None


def root_of(path: str) -> str:
    """The side a node is under: "stars" for the stars side's nodes, "starless" for every other one."""
    return "stars" if path == "stars" or path.startswith("stars/") else "starless"


def depth_of(path: str) -> int:
    """How many steps below its side a node is (a side: 0)."""
    if path in SIDES:
        return 0
    return len((path[6:] if path.startswith("stars/") else path).split("/"))


def mix_kids(node: dict) -> dict:
    """{role: node} of the nested mixes among a mix's inputs."""
    return {r: c for r, c in (node.get("roles") or {}).items() if c.get("kind") == "mix"}


def derived(node: dict, info: dict) -> dict[str, tuple[dict[str, float], float]]:
    """{role: (coefficients, offset)} of the roles of a mix that read a nested mix (their balanced plane is made of
    that mix's picture: ``info["maps"]``); the others are planes of their own."""
    kids = mix_kids(node)
    if not kids:
        return {}
    return {r: info["maps"][r] for r in node["fusion"]["roles"] if any(k in kids for k in info["maps"][r][0])}


def leaf_plan(node: dict, path: str, info_of) -> list[tuple[str, str, str]]:
    """The planes a mix is made of, [(path of the mix, role, kind)], nested mixes included: "bal", the balanced plane of
    a role that reads no nested mix; "read", the channel a role that reads one takes of its other inputs (``info_of(node)``:
    {"maps": {role: ({channel read: coefficient}, offset)}, "channels": {role: channel}}, asked for a mix that has
    nested ones only)."""
    out: list[tuple[str, str, str]] = []
    kids = mix_kids(node)
    der = derived(node, info_of(node)) if kids else {}
    reads: list[str] = []
    for r in node["fusion"]["roles"]:
        if r in der:
            reads += [k for k in der[r][0] if k not in kids and k not in reads]
        else:
            out.append((path, r, "bal"))
    out += [(path, k, "read") for k in reads]
    for r, c in kids.items():
        out += leaf_plan(c, child(path, r), info_of)
    return out


class Mixer:
    """How Post remakes the image of a block of the combination family (a node of the layer tree): ``keys``, the
    fields of a layer that node's own settings fill when the layer is made (the graph's settings copied: ``put_copy``)."""

    def __init__(self, keys: tuple[str, ...]):
        self.keys = keys


def tune_at(lay: dict, path: str) -> dict | None:
    """The own settings of the mix at ``path`` of a layer (None: the block's stored values)."""
    if path in ("", "starless"):
        return lay.get(TUNE)
    return (lay.get(TUNES) or {}).get(path)


def put_tune(lay: dict, path: str, tune: dict | None) -> None:
    """The own settings of the mix at ``path`` set (None: taken off, the block's stored values again)."""
    if path in ("", "starless"):
        if tune:
            lay[TUNE] = {k: dict(v) for k, v in tune.items()}
        else:
            lay.pop(TUNE, None)
        return
    tunes = dict(lay.get(TUNES) or {})
    if tune:
        tunes[path] = {k: dict(v) for k, v in tune.items()}
    else:
        tunes.pop(path, None)
    if tunes:
        lay[TUNES] = tunes
    else:
        lay.pop(TUNES, None)


def put_nested(lay: dict, path: str, tune: dict | None) -> None:
    """The copy of a nested mix's graph settings made: its entry of ``tunes`` is the settings, or None when the graph
    gave it none of its own (the block's stored values). The entry is what says the copy was made: the node is not
    copied again, so the graph's settings changed after it never reach the layer."""
    tunes = dict(lay.get(TUNES) or {})
    tunes[path] = {k: dict(v) for k, v in tune.items()} if tune else None
    lay[TUNES] = tunes


def drop_tunes(lay: dict, paths) -> bool:
    """The own settings of the nested mixes whose node is not in ``paths`` dropped (a level the graph no longer has: it
    starts again from the graph's settings when it comes back; never moved to another level). Whether the layer changed.
    The graph's Undo gives them back with the layer (model: dropped when the graph change is committed)."""
    keep = {k: v for k, v in (lay.get(TUNES) or {}).items() if k in set(paths)}
    if keep == (lay.get(TUNES) or {}):
        return False
    if keep:
        lay[TUNES] = keep
    else:
        lay.pop(TUNES, None)
    return True


def prune(layers: list[dict], lay: dict, paths) -> bool:
    """What a layer of a version names that its tree no longer has (``paths``: every node of it), taken out, when the
    graph change that took the level away is committed (never before: a graph Undo gives the layers back as the step
    found them): an eye or a mask on a missing path is dropped, a filter on one goes to the nearest node above it. The own
    settings (``tunes``) are ``drop_tunes``'. Whether the layers changed."""
    paths = set(paths)
    changed = False

    def gone(k):
        return is_path(k) and k not in SIDES and k not in paths
    for key in ("shown", "side_masks"):
        got = lay.get(key)
        if isinstance(got, dict) and any(gone(k) for k in got):
            lay[key] = {k: v for k, v in got.items() if not gone(k)}
            if not lay[key] and key == "side_masks":
                lay.pop(key)
            changed = True
    for f in layers:
        if f.get("kind") == "filter" and f.get("target") == lay.get("id") and gone(f.get("sub")):
            sub = f["sub"]
            while gone(sub):
                sub = parent(sub)
            f["sub"] = sub
            changed = True
    return changed


#: the settings of a Combine layers layer the graph holds (its block's graph settings) and a layer copies
LAYERS_KEYS = ("blend", "stretch", "sliders")
#: the mixer of each kind of the combination family (graphcore rules.GRAPH_SETTING_KINDS; a test ties the two): a kind
#: that nests has its mixer here, so a new kind cannot exist without its way of being remade
MIXERS: dict[str, Mixer] = {"combine": Mixer((TUNE, TUNES)), "combine_layers": Mixer(LAYERS_KEYS)}


def layers_graph(own: dict | None) -> dict:
    """A Combine layers block's graph settings, whole: its own (``own``) over the defaults a new layer has (Screen,
    the automatic stretch, every slider at its suggestion)."""
    from .postengine import BLENDS, LAYER_STRETCHES
    own = own or {}
    blend = own.get("blend") if own.get("blend") in BLENDS else BLENDS[0]
    stretch = own.get("stretch") if own.get("stretch") in LAYER_STRETCHES else LAYER_STRETCHES[0]
    sliders = {str(k): float(v) for k, v in (own.get("sliders") or {}).items() if abs(float(v) - 2.0) > 1e-9}
    return {"blend": blend, "stretch": stretch, "sliders": sliders}


def put_copy(lay: dict, copied: dict | None) -> bool:
    """A copy of the graph's settings (model.graph_copy) put on a layer of Post, node by node: the layer's own keys
    (``blend``, ``stretch``, ``sliders``), its mix's ``tune``, and ``tunes``, {path: the settings of that nested mix, None:
    none of its own}; a nested mix not named is left as it is. Whether the layer changed."""
    if not copied:
        return False
    import json as _json

    def mark():
        return _json.dumps({k: lay.get(k) for k in (*LAYERS_KEYS, TUNE, TUNES)}, sort_keys=True, default=str)
    before = mark()
    from .postengine import set_layer_stretch
    if "blend" in copied:
        lay["blend"] = copied["blend"]
    if "sliders" in copied:
        lay["sliders"] = dict(copied["sliders"])
    if "stretch" in copied:
        set_layer_stretch(lay, copied["stretch"])
    if TUNE in copied:
        put_tune(lay, "", copied[TUNE])
    for path, tune in (copied.get(TUNES) or {}).items():
        put_nested(lay, path, tune)
    return mark() != before


def channel_row(channel: str) -> np.ndarray:
    """(3,): the linear row of the channel a combination reads of an image (stages/combine.channel_of: its luminance, a
    colour, the mean of green and blue), so that the channel read is this row on the image's red, green and blue."""
    from ..stages.combine import channel_of
    return channel_of(np.eye(3, dtype=np.float32)[:, :, None], channel)[:, 0].astype(np.float64)


#: a leaf plane that is the channel a derived filter reads (not a balanced filter): its path's end
RAW = "|raw"


def _kids(node: dict) -> dict:
    return node.get("children") or {}


def role_forms(node: dict, path: str = "") -> dict[str, tuple[dict[str, float], float]]:
    """{role: ({leaf path: kappa}, offset)} of a combination: each of its balanced filters as a linear form of leaf
    planes, nested combinations included. A filter that reads no combination is a leaf itself (its balanced plane, as
    Post's filters have always been: kappa 1). A filter that reads a combination A (through the channel it names) is
    derived: ``map's coefficient * the channel's row * A's matrix column * A's weight``, per filter of A (premise of
    #181, tests/client/test_post_nest_premises.py: a combination's Output is linear in its balanced filters), and any
    other channel it reads is a plane of its own (the path ``RAW`` ends). ``node``: {"fusion": the combination's
    ``describe``, "maps": {role: ({channel read: coefficient}, offset)} (``balance_maps``), "channels": {role: the
    channel it reads of its input's image}, "children": {role: the same description of the combination it reads, absent
    for a plain input}}."""
    out = {}
    for role in node["fusion"]["roles"]:
        coef, off = node["maps"][role]
        if not any(k in _kids(node) for k in coef):
            out[role] = ({child(path, role): 1.0}, 0.0)
            continue
        k: dict[str, float] = {}
        o = float(off)
        for read, a in coef.items():
            kid = _kids(node).get(read)
            if kid is None:
                leaf = child(path, read) + RAW
                k[leaf] = k.get(leaf, 0.0) + float(a)
                continue
            row = channel_row(node["channels"].get(read, "L"))
            inner = role_forms(kid, child(path, read))
            fa = kid["fusion"]
            mat = np.asarray(fa["matrix"], np.float64)
            for i, ai in enumerate(fa["roles"]):
                g = float(a) * float(row @ mat[:, i]) * float(fa["weights"].get(ai, 1.0))
                for leaf, v in inner[ai][0].items():
                    k[leaf] = k.get(leaf, 0.0) + g * v
                o += g * inner[ai][1]
        out[role] = (k, o)
    return out


def kappas(node: dict, path: str = "") -> tuple[dict[str, np.ndarray], np.ndarray]:
    """A combination's Output (red, green, blue) as a linear map of the planes of the leaves under it: ``({leaf path:
    kappa (3,)}, offset (3,))``, Output = sum of kappa * the leaf's plane + offset (``role_forms`` for the leaves and
    ``node``'s description). Stretching every leaf's plane with one factor and adding them is then the mix stretched."""
    f = node["fusion"]
    mat = np.asarray(f["matrix"], np.float64)
    forms = role_forms(node, path)
    out_k: dict[str, np.ndarray] = {}
    off = np.zeros(3)
    for k, role in enumerate(f["roles"]):
        m = mat[:, k] * float(f["weights"].get(role, 1.0))
        leaves, o = forms[role]
        off += m * o
        for leaf, v in leaves.items():
            out_k[leaf] = out_k.get(leaf, np.zeros(3)) + m * v
    return out_k, off


def describe(stage: str, spec: dict, stored: dict, names: dict, over: dict | None = None,
             inputs=None) -> dict:
    """The master layer's ``fusion`` (module docstring) from the combination's settings ``stored`` (the block's,
    with the layer's own: ``tuned_stored``), ``over`` (settings being moved, not yet kept) on top; ``names``: {role:
    its shown name}. ``inputs``: what the combination reads: the stars alone of every filter make it the stars'
    combination, balanced on the sky and, with the setting Stars at Neutral, mixed as the plain mean (no hue: its
    description says ``stars`` and ``neutral``)."""
    from ..stages.combine import STARS, output_matrix
    from .combineedit import applied_values, palette_of, params_of, stage_class, stars_only
    cls = stage_class(stage)
    values = applied_values(cls, spec, stored)
    for k, v in (over or {}).items():
        values[k] = dict(values.get("weights") or {}, **v) if k == "weights" else v
    roles, mat = output_matrix(params_of(cls, spec, stored, values), spec)
    only = stars_only(inputs)
    neutral = only and bool(spec.get("narrowband")) and values.get("stars", STARS[0]) == STARS[0]
    if neutral:
        mat = [[1.0 / len(roles)] * len(roles) for _ in range(3)]
    hp = _hue_names(spec)
    fields = cls.Params.model_fields
    w = values.get("weights") or {}
    defaults = {r: float(fields[n].default if n in fields and n != "values" else spec["parameters"][n]["default"])
                for r, n in hp.items() if r in roles}
    return {"stage": stage, "roles": list(roles), "names": {r: names.get(r, r) for r in roles},
            "hue_params": {r: hp[r] for r in roles if r in hp},
            "hues": {r: float(values[hp[r]]) for r in roles if r in hp}, "hue_defaults": defaults,
            "matrix": [[float(v) for v in row] for row in mat],
            "weights": {r: float(w.get(r, 1.0)) for r in roles},
            "balance": "sky" if only else values.get("balance", "sky"), "palette": palette_of(spec, values),
            "stars": only, "neutral": neutral}


class _Quiet:
    def log(self, *_a, **_k):
        pass

    def metric(self, *_a, **_k):
        pass

    def progress(self, *_a, **_k):
        pass


def _stage_of(name: str):
    from ..stages.combine import ALL
    return next(c for c in ALL if c.name == name)


def read_inputs(inputs: list[dict], spec: dict) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """The channels a combination reads ({role: (H, W) float32}, stages/combine.channel_of) and the pixels every input
    covers. OSError when an input is not there."""
    from ..stages import recipes
    from ..stages.combine import IMAGES, channel_of
    raw, valid = {}, None
    for inp in inputs:
        src = Path(inp["dir"])
        a = np.load(src / IMAGES[inp.get("image", "output")][1], mmap_mode="r")
        raw[inp["role"]] = np.asarray(channel_of(np.asarray(a), inp.get("channel", recipes.default_channel(
            spec, inp["role"]))), np.float32)
        v = np.load(src / "valid.npy")
        valid = v if valid is None else (valid & v)
    if valid is None:
        raise OSError("the combination read no input")
    return raw, valid


def _stamp(inputs: list[dict]) -> tuple:
    """What a set of inputs is: each one's folder, image and channel, and its file's time."""
    from ..stages.combine import IMAGES
    out = []
    for i in inputs:
        f = Path(i["dir"]) / IMAGES[i.get("image", "output")][1]
        try:
            mt = f.stat().st_mtime
        except OSError:
            mt = None
        out.append((i["role"], str(Path(i["dir"])), i.get("image", "output"), i.get("channel", ""), mt))
    return tuple(sorted(out, key=str))


_MAPS: OrderedDict = OrderedDict()                   # (stage, inputs' stamp, recipe, balance) -> the maps measured
_MAPS_LOCK = threading.Lock()
MAPS_KEPT = 8


def balance_maps(stage: str, spec: dict, inputs: list[dict], balance: str, result_dir=None,
                 raw: tuple | None = None, measure: bool = True) -> dict | None:
    """The balance of a combination's inputs as its run makes it: {"maps": {role: ({channel: coefficient},
    offset)}, "gains": {role: (gain, why)}, "noise": {role: its balanced sky noise}, "from": "run" | "measured"}. The
    run's own (``result_dir``'s drz_meta.json) when it read these inputs with this balance; else the stage's ``steps``
    on the whole inputs (``raw``: (channels, covered) already read), measured once and kept for the session.
    ``measure`` False: None rather than measuring (what is known now, for the interface's thread)."""
    stamp = _stamp(inputs)
    if result_dir is not None:
        d = Path(result_dir)
        try:
            meta = json.loads((d / "drz_meta.json").read_text(encoding="utf-8"))
            man = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            meta, man = {}, {}
        bal = meta.get("balance") or {}
        ran = [i for i in (man.get("params") or {}).get("inputs") or [] if i.get("role") in {x["role"] for x in inputs}]
        if bal.get("how") == balance and bal.get("maps") and _stamp(ran) == stamp:
            return {"maps": {r: ({k: float(c) for k, c in m["coef"].items()}, float(m["offset"]))
                             for r, m in bal["maps"].items()},
                    "gains": {r: (float(g), str(w)) for r, (g, w) in (bal.get("gains") or {}).items()},
                    "noise": {r: float(n) for r, n in (bal.get("noise") or {}).items()}, "from": "run"}
    key = (stage, stamp, json.dumps(spec, sort_keys=True, default=str), balance)
    with _MAPS_LOCK:
        hit = _MAPS.get(key)
        if hit is not None:
            _MAPS.move_to_end(key)
            return hit
    if not measure:
        return None
    chans, valid = raw if raw is not None else read_inputs(inputs, spec)
    chans = {r: np.array(c, np.float32, copy=True) for r, c in chans.items()}
    notes: dict = {}
    maps: dict = {}
    skies = _stage_of(stage)().steps(_Quiet(), spec, inputs, chans, valid, {r: 1.0 for r in chans}, balance, notes,
                                     maps)
    hit = {"maps": maps, "gains": {r: (float(g), str(w)) for r, (g, w) in notes.items()},
           "noise": {r: float(n) for r, (_lv, n) in skies.items()}, "from": "measured"}
    with _MAPS_LOCK:
        _MAPS[key] = hit
        while len(_MAPS) > MAPS_KEPT:
            _MAPS.popitem(last=False)
    return hit


def apply_maps(maps: dict, chans: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """The balanced channels: each its linear map of the channels read (full or reduced: a map is linear, so the
    block means of the balanced channels are the map of the block means)."""
    out = {}
    for r, (coef, off) in maps.items():
        shape = next(iter(chans.values())).shape
        acc = np.full(shape, np.float32(off), np.float32)
        for k, c in coef.items():
            acc += np.float32(c) * chans[k]
        out[r] = acc
    return out


def result_inputs(result_dir) -> tuple[str, dict, list[dict]]:
    """(the stage, the recipe, the inputs) a combination's result was made from (its manifest)."""
    from ..stages import recipes
    d = Path(result_dir)
    man = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    cls = _stage_of(man["stage"])
    params = man.get("params") or {}
    spec = cls().spec(cls.Params.model_validate({"recipe": params.get("recipe") or {}}))
    ports = recipes.roles(spec)
    return man["stage"], spec, [i for i in params.get("inputs") or [] if i.get("role") in ports]


def balanced_planes(result_dir, balance: str) -> tuple[dict[str, np.ndarray], np.ndarray, dict]:
    """({role: its balanced filter (H, W) float32, before its weight}, the pixels every input covers, {role: (gain,
    why)}) of a combination's result: its inputs through ``balance_maps``. OSError when an input is not there."""
    stage, spec, inputs = result_inputs(result_dir)
    raw, valid = read_inputs(inputs, spec)
    got = balance_maps(stage, spec, inputs, balance, result_dir, (raw, valid))
    out = apply_maps(got["maps"], raw)
    for a in out.values():
        a[~valid] = 0
    return out, valid, got["gains"]


def read_planes(result_dir) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """({role: the channel the combination reads of that input (H, W) float32}, the pixels every input covers) of a
    combination's result, before its balance: the planes a role that reads another combination takes of its other
    inputs (``leaf_plan``). OSError when an input is not there."""
    _stage, spec, inputs = result_inputs(result_dir)
    raw, valid = read_inputs(inputs, spec)
    out = {r: np.array(a, np.float32, copy=True) for r, a in raw.items()}
    for a in out.values():
        a[~valid] = 0
    return out, valid


def node_form(result_dir, balance: str) -> dict:
    """{"maps": {role: ({channel read: coefficient}, offset)}, "channels": {role: the channel it reads of its input's
    image}} of a combination's result: its balance as the run made it (``balance_maps``) and what each input is read
    as (``role_forms``'s description of a combination)."""
    from ..stages import recipes
    stage, spec, inputs = result_inputs(result_dir)
    got = balance_maps(stage, spec, inputs, balance, result_dir)
    return {"maps": got["maps"],
            "channels": {i["role"]: i.get("channel", recipes.default_channel(spec, i["role"])) for i in inputs}}
