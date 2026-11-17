"""A combination's master layer in Post: what the model tells Post of a combination (``describe``), where its settings
live (the layer's ``tune``), and its filters balanced as the run balances them (``balance_maps``, ``balanced_planes``).
No Qt here.

The Combine layers layer over a combination's Output (``starless`` "output") is the combination's master layer when
Post can make the combination's picture itself: the recipe mixes its balanced filters through hues or a matrix (no
lightness of its own: LRGB and the recipes composed as lightness and colour stay plain layers), its inputs are not
the stars alone (the stars' combination is the layer's Stars side), and no block follows the combination in its
workflow (a Color or Denoise block after it changes its picture, which Post cannot make again).

Its hues and its filters' intensities are then settings of the LAYER, as a Combine layers layer's blend and sliders
are: kept with the layer in Post's version (``tune`` = {"hues": {role: degrees}, "weights": {role: intensity}},
written only once one was changed: until then the block's stored values, read at each load), shown and changed by
Post, by the block's panel and by Inspect's editor alike (the model routes a change of them to the layer:
graphmodel.GraphEditing.apply_change), and they never outdate the block: Post mixes the filters itself. The block's
own stored values are left as they are (its keys do not move). The balance changes the gains the run measures: it
stays the block's, a setting that needs a run. For a plain layer every setting stays the block's.

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
import threading
from collections import OrderedDict
from pathlib import Path

import numpy as np

#: the stage of a workflow's Output block (its picture is the combination's when nothing follows it)
OUTPUT_STAGE = "render_simple"
#: a master layer's own settings (module docstring)
TUNE = "tune"


def plain_reason(stage: str, spec: dict | None, inputs, chain) -> str:
    """Why a combination's layer stays a plain Combine layers layer ("" when it is a master layer): a recipe composed
    as lightness and colour, the stars alone, a block after the combination."""
    if not stage or spec is None:
        return "stars"
    if (spec.get("output") or {}).get("lightness") is not None:
        return "lightness"
    if inputs and all(i.get("image") == "stars" for i in inputs):
        return "stars"
    if any(s not in (stage, OUTPUT_STAGE) for s in chain or ()):
        return "after"
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


def describe(stage: str, spec: dict, stored: dict, names: dict, over: dict | None = None) -> dict:
    """The master layer's ``fusion`` (module docstring) from the combination's settings ``stored`` (the block's,
    with the layer's own: ``tuned_stored``), ``over`` (settings being moved, not yet kept) on top; ``names``: {role:
    its shown name}."""
    from ..stages.combine import output_matrix
    from .combineedit import applied_values, palette_of, params_of, stage_class
    cls = stage_class(stage)
    values = applied_values(cls, spec, stored)
    for k, v in (over or {}).items():
        values[k] = dict(values.get("weights") or {}, **v) if k == "weights" else v
    roles, mat = output_matrix(params_of(cls, spec, stored, values), spec)
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
            "balance": values.get("balance", "sky"), "palette": palette_of(spec, values)}


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
