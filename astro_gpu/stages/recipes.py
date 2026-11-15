"""Combination recipes: what a combination of filters does, written in a TOML file instead of engine code.

A recipe is a file of the application's folder ``astro_gpu/resources/combinations`` (``*.toml``; a name starting
with "_" is documentation, not listed: ``_example.toml`` explains every setting). There is no user folder and no
editor: a recipe is written by an expert and shipped with the application. The engine runs every recipe with the
same stage code (stages/combine.py), from a fixed vocabulary of operations it implements and documents:

- ``inputs``: the named channels (``role``: Ha, OIII, R, L...), each a monochrome image (``type = "mono"``), a colour
  image (``"colour"``) or either (``"any"``), and what is read of it by default (``channel``: L the image's grey or
  luminance, R, G, B, or GB the mean of green and blue: stages/combine.channel_of). The project's wire may read
  another channel of a colour image (a dual-band image's red for H-alpha).
- ``parameters``: the settings the user may change, each with its default (a number or one of ``choices``); an
  operation's value written ``"$name"`` is the parameter's value.
- ``steps``, in order, on the channels: ``neutralise_sky`` (first, always: each channel's sky to 0, Rector et al.
  2007), ``linear_match`` (Siril's linear match of channels to a reference input, between two bounds, then the sky
  to 0 again), ``continuum_subtract`` (a narrow-band line image minus its continuum measured by a broadband one, the
  scale measured on the field's stars: Waller 1990). The user's channel weights come after the steps.
- ``output``: the channels mixed into red, green and blue, by ``hues`` (each channel given a hue, balanced so that
  equal channels give a grey: stages/combine.hue_palette) or by an explicit ``matrix`` (rows R, G, B); optionally
  composed as lightness and colour (``lightness``: the channels' noise-weighted or equal mean, or one input's image,
  the L of an LRGB), with the colour smoothed (``smoothing``, ``radius``) and its strength raised under the display's
  gamut (``strength``); ``saturation``: the inputs whose stacks' saturation maps the result hands on.

The four types the application had before recipes (S-H-O, H-O, RGB, LRGB) are four shipped recipes, ``sho``,
``ho``, ``rgb`` and ``lrgb``. Their content at version 1 is frozen here (``LEGACY``): a combination whose recipe is
that content runs on the type's own stage (``combine_sho``...) with the settings it always had, so its key and its
image are those it had before recipes (tests/test_recipes.py). Any other recipe, or one of those four changed, runs on
the stage ``combine_recipe``, whose key holds the recipe's content.

When a combination is placed in a project, the recipe is copied into the workflow's source (``{"fusion": {"type":
the recipe's id, "recipe": the copy, "inputs": [...]}}``): the project computes the same whatever happens to the
folder's file later. A project saved before recipes has no copy: its type's frozen content stands for it. The folder's
version of a recipe may be newer than the project's copy (``status``): nothing changes until the user applies it
(``api.apply_combination_recipe``).

A file that cannot be read is refused with its name, its line and the field (``RecipeError``); the others are listed.
"""
from __future__ import annotations

import copy
import re
import tomllib
from pathlib import Path
from typing import Any

#: the application's folder of recipes
FOLDER = Path(__file__).resolve().parents[1] / "resources" / "combinations"
#: the stage of every recipe that is not one of the four types' frozen content
RECIPE_STAGE = "combine_recipe"
#: what a channel of an input can be read as (stages/combine.channel_of)
CHANNELS = ("L", "R", "G", "B", "GB")
TYPES = ("mono", "colour", "any")
OPS = ("neutralise_sky", "linear_match", "continuum_subtract")
#: the fields that say nothing to the computation (left out of the key and of the frozen content)
DOC_FIELDS = ("name", "version", "description")
PARAM_DOC = ("label", "help", "unit", "step")
INPUT_DOC = ("label",)
_ID = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
_ROLE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,15}$")


class RecipeError(ValueError):
    """A recipe that cannot be used: its file, the line and the field at fault."""

    def __init__(self, source: str, msg: str, line: int | None = None, field: str = ""):
        self.source, self.line, self.field, self.msg = source, line, field, msg
        where = source + (f", line {line}" if line else "") + (f", {field}" if field else "")
        super().__init__(f"{where}: {msg}")


# ---------------------------------------------------------------------------------------------- the frozen four

_NARROW_PARAMS = {"lightness": {"default": "noise", "choices": ["noise", "equal"]},
                  "smoothing": {"default": "auto", "choices": ["auto", "radius"]},
                  "radius": {"default": 0.0, "min": 0.0, "max": 50.0}}
_HUE = {"min": 0.0, "max": 360.0}
_MATCH_HA = [{"op": "neutralise_sky"}, {"op": "linear_match", "reference": "Ha", "channels": [], "low": 1e-7,
                                         "high": 0.875}]


def _narrow(rid: str, roles: list[str], hues: list[float]) -> dict:
    return {"id": rid, "narrowband": True,
            "inputs": [{"role": r, "type": "any", "channel": "L"} for r in roles],
            "parameters": {**{f"hue_{r.lower()}": {"default": h, **_HUE} for r, h in zip(roles, hues)},
                           **copy.deepcopy(_NARROW_PARAMS)},
            "steps": copy.deepcopy(_MATCH_HA),
            "output": {"channels": list(roles), "hues": [f"$hue_{r.lower()}" for r in roles], "matrix": None,
                       "lightness": "channels", "weighting": "$lightness", "smoothing": "$smoothing",
                       "radius": "$radius", "strength": 1.9, "saturation": []}}


_EYE = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
#: the content (``computational``) of the four types' recipes at version 1: what the four blocks did before recipes
LEGACY: dict[str, dict] = {
    "sho": _narrow("sho", ["SII", "Ha", "OIII"], [15.5, 31.4, 202.2]),
    "ho": _narrow("ho", ["Ha", "OIII"], [0.0, 180.0]),
    "rgb": {"id": "rgb", "narrowband": False, "inputs": [{"role": r, "type": "any", "channel": "L"} for r in "RGB"],
            "parameters": {},
            "steps": [{"op": "neutralise_sky"}, {"op": "linear_match", "reference": "R", "channels": [], "low": 1e-7,
                                                  "high": 0.875}],
            "output": {"channels": ["R", "G", "B"], "hues": None, "matrix": _EYE, "lightness": None,
                       "weighting": None, "smoothing": None, "radius": None, "strength": 1.0,
                       "saturation": ["R", "G", "B"]}},
    "lrgb": {"id": "lrgb", "narrowband": False, "inputs": [{"role": r, "type": "any", "channel": "L"} for r in "LRGB"],
             "parameters": {},
             "steps": [{"op": "neutralise_sky"}, {"op": "linear_match", "reference": "L", "channels": [], "low": 1e-7,
                                                   "high": 0.875}],
             "output": {"channels": ["R", "G", "B"], "hues": None, "matrix": _EYE, "lightness": "L",
                        "weighting": None, "smoothing": "radius", "radius": 0.0, "strength": 1.0,
                        "saturation": ["R", "G", "B"]}},
}
#: the four types' names (their blocks' titles)
LEGACY_NAMES = {"sho": "S-H-O", "ho": "H-O", "rgb": "RGB", "lrgb": "LRGB"}


def computational(spec: dict) -> dict:
    """What of a recipe decides the image: everything but its name, version, description and the labels and help
    texts (a copy)."""
    out = {k: copy.deepcopy(v) for k, v in spec.items() if k not in DOC_FIELDS}
    out["inputs"] = [{k: v for k, v in i.items() if k not in INPUT_DOC} for i in spec.get("inputs", [])]
    out["parameters"] = {n: {k: v for k, v in q.items() if k not in PARAM_DOC}
                         for n, q in spec.get("parameters", {}).items()}
    return out


def legacy_spec(kind: str) -> dict:
    """The recipe a project saved before recipes stands for: the type's frozen content, its name, version 1."""
    return {"id": kind, "name": LEGACY_NAMES[kind], "version": 1, "description": "",
            **copy.deepcopy({k: v for k, v in LEGACY[kind].items() if k != "id"})}


def is_legacy(spec: dict) -> bool:
    """True when the recipe is one of the four types' frozen content (its type's own stage runs it)."""
    return spec.get("id") in LEGACY and computational(spec) == LEGACY[spec["id"]]


def recipe_of(fusion: dict) -> dict:
    """The recipe a combination source runs: its copy, else (a project saved before recipes) its type's frozen
    one."""
    r = fusion.get("recipe")
    if r is not None:
        return r
    return legacy_spec(fusion["type"])


def stage_for(fusion: dict) -> str:
    """The stage of a combination source: its type's own (``combine_sho``...) for the frozen content of one of the
    four, else ``combine_recipe``."""
    r = fusion.get("recipe")
    t = fusion.get("type")
    if t in LEGACY and (r is None or is_legacy(r)):
        return f"combine_{t}"
    return RECIPE_STAGE


def roles(spec: dict) -> tuple[str, ...]:
    """The recipe's inputs' roles, in its order (the ports of its block)."""
    return tuple(i["role"] for i in spec["inputs"])


def default_channel(spec: dict, role: str) -> str:
    """What an input is read as when the wire says nothing."""
    return next((i["channel"] for i in spec["inputs"] if i["role"] == role), "L")


# ---------------------------------------------------------------------------------------------- reading a file

def _lines(text: str) -> dict[tuple, int]:
    """{path: line} of the tables and keys of a TOML text (a ``[[steps]]`` table's path holds its index), for the
    messages: the first line of each."""
    where: dict[tuple, int] = {}
    table: tuple = ()
    counts: dict[tuple, int] = {}
    for n, raw in enumerate(text.splitlines(), 1):
        s = raw.strip()
        if s.startswith("#") or not s:
            continue
        if s.startswith("[["):
            name = tuple(x.strip() for x in s[2:s.index("]]")].split("."))
            counts[name] = counts.get(name, -1) + 1
            table = name + (counts[name],)
            where.setdefault(table, n)
        elif s.startswith("["):
            table = tuple(x.strip().strip('"') for x in s[1:s.index("]")].split("."))
            where.setdefault(table, n)
        elif "=" in s:
            key = s.split("=", 1)[0].strip().strip('"')
            where.setdefault(table + (key,), n)
    return where


class _Check:
    """The checks of one recipe's data, raising ``RecipeError`` at the first fault with its line."""

    def __init__(self, source: str, text: str):
        self.source, self.where = source, _lines(text)

    def fail(self, path: tuple, msg: str):
        line = None
        for n in range(len(path), 0, -1):
            if path[:n] in self.where:
                line = self.where[path[:n]]
                break
        field = ".".join(str(x) if not isinstance(x, int) else f"[{x + 1}]" for x in path).replace(".[", "[")
        raise RecipeError(self.source, msg, line, field)

    def known(self, d: dict, path: tuple, allowed: tuple[str, ...]):
        for k in d:
            if k not in allowed:
                self.fail(path + (k,), f"unknown setting (allowed: {', '.join(allowed)})")

    def text(self, d: dict, path: tuple, key: str, required: bool = True, default: str | None = None) -> str | None:
        if key not in d:
            if required:
                self.fail(path, f"{key} is missing")
            return default
        if not isinstance(d[key], str) or not d[key].strip():
            self.fail(path + (key,), "must be a text")
        return d[key]

    def number(self, d: dict, path: tuple, key: str, default: float | None = None, lo: float | None = None,
               hi: float | None = None) -> float | None:
        if key not in d:
            if default is None:
                self.fail(path, f"{key} is missing")
            return default
        v = d[key]
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            self.fail(path + (key,), "must be a number")
        if (lo is not None and v < lo) or (hi is not None and v > hi):
            self.fail(path + (key,), f"must be between {lo:g} and {hi:g}" if lo is not None and hi is not None else
                      f"must be {lo:g} or more" if lo is not None else f"must be {hi:g} or less")
        return float(v)


def parse(text: str, source: str = "recipe") -> dict:
    """The recipe of a TOML text, checked and in its normal form (every optional setting filled); ``RecipeError``
    naming ``source``, the line and the field when it cannot be used."""
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as e:
        m = re.search(r"line (\d+)", str(e))
        raise RecipeError(source, f"not valid TOML ({e})", int(m.group(1)) if m else None) from None
    c = _Check(source, text)
    c.known(data, (), ("id", "name", "version", "description", "narrowband", "inputs", "parameters", "steps",
                       "output"))
    rid = c.text(data, (), "id")
    if not _ID.match(rid):
        c.fail(("id",), "lower-case letters, digits and _, starting with a letter (32 at most)")
    name = c.text(data, (), "name")
    if not isinstance(data.get("version"), int) or isinstance(data.get("version"), bool) or data["version"] < 1:
        c.fail(("version",) if "version" in data else (), "version must be a whole number, 1 or more")
    desc = c.text(data, (), "description", required=False, default="")
    nb = data.get("narrowband", False)
    if not isinstance(nb, bool):
        c.fail(("narrowband",), "must be true or false")

    # the inputs
    ins = data.get("inputs")
    if not isinstance(ins, list) or not ins:
        c.fail(("inputs",) if "inputs" in data else (), "at least one [[inputs]] table is needed")
    inputs = []
    for n, i in enumerate(ins):
        path = ("inputs", n)
        if not isinstance(i, dict):
            c.fail(path, "must be a table")
        c.known(i, path, ("role", "label", "type", "channel"))
        role = c.text(i, path, "role")
        if not _ROLE.match(role):
            c.fail(path + ("role",), "letters, digits and _, starting with a letter (16 at most)")
        if role in {x["role"] for x in inputs}:
            c.fail(path + ("role",), f"the input {role} is named twice")
        typ = c.text(i, path, "type")
        if typ not in TYPES:
            c.fail(path + ("type",), f"must be one of {', '.join(TYPES)}")
        ch = c.text(i, path, "channel", required=False, default="L")
        if ch not in CHANNELS:
            c.fail(path + ("channel",), f"must be one of {', '.join(CHANNELS)}")
        if typ == "mono" and ch != "L":
            c.fail(path + ("channel",), "a monochrome image is read as its grey: L")
        inputs.append({"role": role, "label": c.text(i, path, "label", required=False, default=role), "type": typ,
                       "channel": ch})
    names = [i["role"] for i in inputs]

    # the parameters
    params = {}
    praw = data.get("parameters", {})
    if not isinstance(praw, dict):
        c.fail(("parameters",), "must be tables [parameters.<name>]")
    for pname, q in praw.items():
        path = ("parameters", pname)
        if not _ID.match(pname) or not isinstance(q, dict):
            c.fail(path, "a parameter is a table [parameters.<name>], its name in lower-case letters, digits and _")
        c.known(q, path, ("default", "min", "max", "choices", *PARAM_DOC))
        if "default" not in q:
            c.fail(path, "default is missing")
        if "choices" in q:
            ch = q["choices"]
            if not isinstance(ch, list) or not ch or not all(isinstance(x, str) for x in ch):
                c.fail(path + ("choices",), "must be a list of texts")
            if q["default"] not in ch:
                c.fail(path + ("default",), f"must be one of the choices {ch}")
            p = {"default": q["default"], "choices": list(ch)}
        else:
            lo = c.number(q, path, "min") if "min" in q else None
            hi = c.number(q, path, "max") if "max" in q else None
            if lo is not None and hi is not None and not lo < hi:
                c.fail(path + ("max",), "must be above min")
            p = {"default": c.number(q, path, "default", lo=lo, hi=hi)}
            p |= ({"min": lo} if lo is not None else {}) | ({"max": hi} if hi is not None else {})
        for k in PARAM_DOC:
            if k in q:
                p[k] = q[k] if k != "step" else c.number(q, path, k)
        params[pname] = p

    def value(d: dict, path: tuple, key: str, kind: str, choices: tuple = ()):
        """A literal, or "$name" of a declared parameter of that kind."""
        v = d[key]
        if isinstance(v, str) and v.startswith("$"):
            q = params.get(v[1:])
            if q is None:
                c.fail(path + (key,), f"no parameter {v[1:]!r} is declared")
            if (kind == "number") != ("choices" not in q):
                c.fail(path + (key,), f"the parameter {v[1:]} must be a {'number' if kind == 'number' else 'choice'}")
            if choices and not set(q["choices"]) <= set(choices):
                c.fail(path + (key,), f"the parameter {v[1:]}'s choices must be among {', '.join(choices)}")
            return v
        if kind == "number":
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                c.fail(path + (key,), "must be a number or \"$parameter\"")
            return float(v)
        if v not in choices:
            c.fail(path + (key,), f"must be one of {', '.join(choices)} or \"$parameter\"")
        return v

    def role_of(d: dict, path: tuple, key: str) -> str:
        r = c.text(d, path, key)
        if r not in names:
            c.fail(path + (key,), f"{r} is not one of the inputs ({', '.join(names)})")
        return r

    # the steps
    sraw = data.get("steps")
    if not isinstance(sraw, list) or not sraw:
        c.fail(("steps",) if "steps" in data else (), "at least one [[steps]] table is needed: neutralise_sky first")
    steps = []
    for n, s in enumerate(sraw):
        path = ("steps", n)
        if not isinstance(s, dict):
            c.fail(path, "must be a table")
        op = c.text(s, path, "op")
        if op not in OPS:
            c.fail(path + ("op",), f"unknown operation (the engine's: {', '.join(OPS)})")
        if n == 0 and op != "neutralise_sky":
            c.fail(path + ("op",), "the first step is neutralise_sky: the skies at 0 before any other operation")
        if op == "neutralise_sky":
            c.known(s, path, ("op",))
            steps.append({"op": op})
        elif op == "linear_match":
            c.known(s, path, ("op", "reference", "channels", "low", "high"))
            ref = role_of(s, path, "reference")
            chs = s.get("channels", [])
            if not isinstance(chs, list) or any(x not in names or x == ref for x in chs):
                c.fail(path + ("channels",), "must list inputs other than the reference ([] : all of them)")
            low = c.number(s, path, "low", default=1e-7, lo=0.0, hi=1.0)
            high = c.number(s, path, "high", default=0.875, lo=0.0, hi=1.0)
            if not low < high:
                c.fail(path + ("high",), "must be above low")
            steps.append({"op": op, "reference": ref, "channels": list(chs), "low": low, "high": high})
        else:
            c.known(s, path, ("op", "line", "continuum", "scale"))
            line, cont = role_of(s, path, "line"), role_of(s, path, "continuum")
            if line == cont:
                c.fail(path + ("continuum",), "must be another input than the line")
            st = s.get("scale", "stars")
            if st != "stars" and (isinstance(st, bool) or not isinstance(st, (int, float)) or st <= 0):
                c.fail(path + ("scale",), "must be \"stars\" (measured on the field's stars) or a number above 0")
            steps.append({"op": op, "line": line, "continuum": cont,
                          "scale": st if st == "stars" else float(st)})

    # the output
    o = data.get("output")
    if not isinstance(o, dict):
        c.fail(("output",) if "output" in data else (), "an [output] table is needed")
    path = ("output",)
    c.known(o, path, ("channels", "hues", "matrix", "lightness", "weighting", "smoothing", "radius", "strength",
                      "saturation"))
    chs = o.get("channels")
    if not isinstance(chs, list) or not chs or any(x not in names for x in chs) or len(set(chs)) != len(chs):
        c.fail(path + ("channels",) if "channels" in o else path, "must list inputs, each once")
    if ("hues" in o) == ("matrix" in o):
        c.fail(path, "give either hues or matrix")
    out: dict[str, Any] = {"channels": list(chs), "hues": None, "matrix": None}
    if "hues" in o:
        h = o["hues"]
        if not isinstance(h, list) or len(h) != len(chs) or len(chs) not in (2, 3):
            c.fail(path + ("hues",), "one hue per channel, for 2 or 3 channels")
        out["hues"] = [value({"h": x}, path + ("hues",), "h", "number") for x in h]
    else:
        m = o["matrix"]
        if not isinstance(m, list) or len(m) != 3 or any(not isinstance(r, list) or len(r) != len(chs) for r in m):
            c.fail(path + ("matrix",), f"3 rows (red, green, blue) of {len(chs)} numbers, one per channel")
        out["matrix"] = [[value({"x": x}, path + ("matrix",), "x", "number") for x in r] for r in m]
    light = o.get("lightness")
    if light is not None and light != "channels" and light not in names:
        c.fail(path + ("lightness",), f"must be \"channels\" or one of the inputs ({', '.join(names)})")
    out["lightness"] = light
    out["weighting"] = value(o, path, "weighting", "choice", ("noise", "equal")) if "weighting" in o else None
    out["smoothing"] = value(o, path, "smoothing", "choice", ("auto", "radius")) if "smoothing" in o else None
    out["radius"] = value(o, path, "radius", "number") if "radius" in o else None
    out["strength"] = value(o, path, "strength", "number") if "strength" in o else 1.0
    if isinstance(out["strength"], float) and out["strength"] < 1:
        c.fail(path + ("strength",), "must be 1 (none) or more")
    if light is None:
        for k in ("weighting", "smoothing", "radius"):
            if k in o:
                c.fail(path + (k,), "only with lightness: the plain mix has no lightness to keep")
        if out["strength"] != 1.0:
            c.fail(path + ("strength",), "only with lightness: the plain mix has no lightness to keep")
    else:
        if light == "channels":
            out["weighting"] = out["weighting"] or "noise"
        elif "weighting" in o:
            c.fail(path + ("weighting",), "only with lightness = \"channels\"")
        out["smoothing"] = out["smoothing"] or "auto"
        if out["radius"] is None:
            out["radius"] = 0.0
        if light != "channels" and out["smoothing"] != "radius":
            c.fail(path + ("smoothing",), "a lightness taken from an input needs smoothing = \"radius\" (its "
                                          "noise is not measured)")
    sat = o.get("saturation", [])
    if not isinstance(sat, list) or any(x not in names for x in sat):
        c.fail(path + ("saturation",), "must list inputs")
    out["saturation"] = list(sat)
    used = set(chs) | ({light} if light not in (None, "channels") else set())
    for n, s in enumerate(steps):
        used |= {s.get("reference"), s.get("line"), s.get("continuum")} - {None}
    for i in names:
        if i not in used:
            c.fail(("inputs", names.index(i)), f"the input {i} is used by no step and no output")
    return {"id": rid, "name": name, "version": data["version"], "description": desc, "narrowband": nb,
            "inputs": inputs, "parameters": params, "steps": steps, "output": out}


def load(path: str | Path) -> dict:
    """The recipe of a file (``parse``), its messages naming the file."""
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        raise RecipeError(path.name, f"cannot be read ({e})") from None
    return parse(text, path.name)


def available(folder: str | Path | None = None) -> tuple[dict[str, dict], list[RecipeError]]:
    """({id: recipe}, the files refused) of a folder of recipes (the application's by default), in the files' order
    of names. A file that cannot be used, or that gives an id another file gave first, is refused with its reason;
    the others are listed."""
    folder = Path(folder) if folder is not None else FOLDER
    out: dict[str, dict] = {}
    bad: list[RecipeError] = []
    files = sorted(folder.glob("*.toml")) if folder.is_dir() else []
    for f in files:
        if f.name.startswith("_"):
            continue
        try:
            r = load(f)
        except RecipeError as e:
            bad.append(e)
            continue
        if r["id"] in out:
            bad.append(RecipeError(f.name, f"the id {r['id']!r} is already given by another file", _lines(
                f.read_text(encoding="utf-8")).get(("id",)), "id"))
            continue
        out[r["id"]] = r
    return out, bad


def check(spec: Any, where: str = "recipe") -> dict:
    """A recipe copied into a project, checked as a file would be (its normal form back): ValueError naming
    ``where`` when it is not one."""
    if not isinstance(spec, dict):
        raise RecipeError(where, f"a recipe is a table, not {spec!r}")
    try:
        text = _to_toml(spec)
    except (TypeError, ValueError) as e:
        raise RecipeError(where, f"not a recipe ({e})") from None
    got = parse(text, where)
    if computational(got) != computational(spec):
        raise RecipeError(where, "not in its normal form (made by recipes.parse)")
    return got


def _to_toml(spec: dict) -> str:
    """A recipe in its normal form written back as TOML (the round trip ``check`` reads)."""
    def lit(v):
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, (int, float)):
            return repr(float(v)) if isinstance(v, float) else str(v)
        if isinstance(v, str):
            return '"' + v.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"'
        if isinstance(v, list):
            return "[" + ", ".join(lit(x) for x in v) + "]"
        raise TypeError(f"{v!r} is not a recipe value")

    def table(d, skip_none=True):
        return "".join(f"{k} = {lit(v)}\n" for k, v in d.items() if not (skip_none and v is None))
    s = table({k: spec[k] for k in ("id", "name", "version", "description", "narrowband") if k in spec})
    for i in spec.get("inputs", []):
        s += "[[inputs]]\n" + table(i)
    for n, q in spec.get("parameters", {}).items():
        s += f"[parameters.{n}]\n" + table(q)
    for st in spec.get("steps", []):
        s += "[[steps]]\n" + table(st)
    o = dict(spec.get("output", {}))
    if o.get("lightness") is None:
        o = {k: v for k, v in o.items() if k not in ("weighting", "smoothing", "radius")
             and not (k == "strength" and v == 1.0)}
    elif o.get("lightness") != "channels":
        o.pop("weighting", None)
    s += "[output]\n" + table(o)
    return s


# ---------------------------------------------------------------------------------------------- a project's copy

def status(copy_: dict, folder: dict[str, dict] | None = None) -> dict:
    """The project's copy of a recipe against the folder's file of the same id: {"id", "project": its version,
    "folder": the file's version (None: no such file now), "newer": the file's version is above the copy's,
    "changed": the file computes otherwise than the copy}. Nothing changes until the user applies the file."""
    rid = copy_["id"]
    if folder is None:
        folder = available()[0]
    f = folder.get(rid)
    return {"id": rid, "project": copy_.get("version", 1), "folder": f["version"] if f else None,
            "newer": bool(f) and f["version"] > copy_.get("version", 1),
            "changed": bool(f) and computational(f) != computational(copy_)}


def summary(spec: dict) -> dict:
    """What a list of recipes shows: id, name, version, description, narrow-band or not, the inputs (role, label,
    type, channel) and the parameters."""
    return {"id": spec["id"], "name": spec["name"], "version": spec["version"],
            "description": spec.get("description", ""), "narrowband": spec["narrowband"],
            "inputs": [dict(i) for i in spec["inputs"]], "parameters": copy.deepcopy(spec["parameters"])}


def fusion_source(spec: dict, inputs: list[dict] | None = None) -> dict:
    """A workflow source combining with this recipe, the recipe copied in: what placing the combination stores."""
    return {"fusion": {"type": spec["id"], "recipe": copy.deepcopy(spec), "inputs": list(inputs or [])}}
