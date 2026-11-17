"""A combination's master layer in Post: what the model tells Post of a combination (``describe``), and its filters
balanced from the combination's result (``balanced_planes``, read in the render worker). No Qt here.

The Combine layers layer over a combination's Output (``starless`` "output") is the combination's master layer when
Post can make the combination's picture itself: the recipe mixes its balanced filters through hues or a matrix (no
lightness of its own: LRGB and the recipes composed as lightness and colour stay plain layers), its inputs are not
the stars alone (the stars' combination is the layer's Stars side), and no block follows the combination in its
workflow (a Color or Denoise block after it changes its picture, which Post cannot make again). Its settings (the
hues, the balance, each filter's intensity) are the combination block's own, which Inspect's editor and the run read
too (client/combineedit.py): one source. The master layer holds them as ``fusion`` while Post shows it (never saved,
given again at each load: postengine.save_layers drops it):

    {"stage", "roles": [filters, in the recipe's order], "names": {role: shown name}, "hue_params": {role: the
     setting of its hue}, "hues": {role: degrees}, "hue_defaults": {role: its recipe's default hue}, "matrix": [rows
     R, G, B], "weights": {role: intensity},
     "balance": "sky" | "signal", "palette": the palette the hues are ("" for the user's own)}

The balanced filters are the combination's own: the linear maps its run wrote (drz_meta.json, ``balance``,
stages/combine.Combination.steps) applied to the inputs it read; a result made before the maps were written, or a
balance changed since the run, is balanced again by the same ``steps`` on the same inputs.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

#: the stage of a workflow's Output block (its picture is the combination's when nothing follows it)
OUTPUT_STAGE = "render_simple"


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


def describe(stage: str, spec: dict, stored: dict, names: dict, over: dict | None = None) -> dict:
    """The master layer's ``fusion`` (module docstring) from the block's settings ``stored`` (its stage's), ``over``
    (settings being moved, not yet the block's) on top; ``names``: {role: its shown name}."""
    from ..stages.combine import output_matrix
    from .combineedit import applied_values, hue_params, palette_of, params_of, stage_class
    cls = stage_class(stage)
    values = applied_values(cls, spec, stored)
    for k, v in (over or {}).items():
        values[k] = dict(values.get("weights") or {}, **v) if k == "weights" else v
    roles, mat = output_matrix(params_of(cls, spec, stored, values), spec)
    hp = hue_params(spec)
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


def balanced_planes(result_dir, balance: str) -> tuple[dict[str, np.ndarray], np.ndarray, dict]:
    """({role: its balanced filter (H, W) float32, before its weight}, the pixels every input covers, {role: (gain,
    why)}) of a combination's result: its inputs (manifest.json) through the maps its run wrote, or through the
    stage's ``steps`` again when the balance asked is not the run's or the maps are not there. OSError when an input
    is not there."""
    from ..stages import recipes
    from ..stages.combine import ALL, IMAGES, channel_of
    d = Path(result_dir)
    man = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    meta = json.loads((d / "drz_meta.json").read_text(encoding="utf-8"))
    cls = next(c for c in ALL if c.name == man["stage"])
    params = man.get("params") or {}
    spec = cls().spec(cls.Params.model_validate({"recipe": params.get("recipe") or {}}))
    ports = recipes.roles(spec)
    inputs = [i for i in params.get("inputs") or [] if i.get("role") in ports]
    raw, valid = {}, None
    for inp in inputs:
        src = Path(inp["dir"])
        a = np.load(src / IMAGES[inp.get("image", "output")][1], mmap_mode="r")
        raw[inp["role"]] = channel_of(np.asarray(a), inp.get("channel", recipes.default_channel(spec, inp["role"])))
        v = np.load(src / "valid.npy")
        valid = v if valid is None else (valid & v)
    if valid is None:
        raise OSError("the combination read no input")
    bal = meta.get("balance") or {}
    if bal.get("how") == balance and set(bal.get("maps") or {}) >= set(raw):
        out = {}
        for r, m in bal["maps"].items():
            acc = np.full(valid.shape, np.float32(m["offset"]), np.float32)
            for k, c in m["coef"].items():
                acc += np.float32(c) * raw[k]
            out[r] = acc
        gains = {r: (float(g), str(why)) for r, (g, why) in (bal.get("gains") or {}).items()}
    else:
        out = {r: np.array(c, np.float32, copy=True) for r, c in raw.items()}
        notes: dict = {}
        cls().steps(_Quiet(), spec, inputs, out, valid, {r: 1.0 for r in out}, balance, notes)
        gains = {r: (float(g), str(why)) for r, (g, why) in notes.items()}
    for a in out.values():
        a[~valid] = 0
    return out, valid, gains
