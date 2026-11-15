"""The combinations the graph offers and what a Combination block is, read from the engine's recipes (stages/recipes.py:
the files of resources/combinations, through the same reading as api.combination_recipes).

A Combination block stores its type, the recipe's id (``settings["type"]``; absent: "sho"), and, once placed from the
list, the copy of its recipe (``Block.source = {"recipe": the copy}``): its title, its inputs (role, label, type) and
its settings come from that copy, so a project shows and computes the same whatever happens to the file later. A
block of a project saved before recipes has no copy: its type's frozen content stands for it (recipes.legacy_spec),
and it compiles as it always did. The folder's version of a recipe may be newer than the block's copy (``status``):
nothing changes until the user applies it (operations.op_update_recipe).

The list is the folder's, in its order; nothing is guessed from a name."""
from __future__ import annotations

import copy
from functools import lru_cache

from ..stages import recipes as rc

#: the folder of recipes the graph reads (None: the application's); the tests point it to another folder
FOLDER = None
#: the type a Combination card places until another one is chosen in its list
DEFAULT = "sho"


@lru_cache(maxsize=4)
def _read(folder):
    found, _bad = rc.available(folder)
    return found


def _folder() -> dict:
    return _read(str(FOLDER) if FOLDER is not None else None)


def refresh():
    """Read the folder again (a test that writes a recipe)."""
    _read.cache_clear()
    _legacy_in.cache_clear()


# ---------------------------------------------------------------------------------------------- the folder's list
def offered() -> list[str]:
    """The ids of the recipes the folder offers, in its order (api.combination_recipes)."""
    return list(_folder())


def recipe(rid: str) -> dict | None:
    """The folder's recipe of that id, whole (a copy); None when the folder has none."""
    r = _folder().get(rid)
    return copy.deepcopy(r) if r is not None else None


def short(name: str) -> str:
    """A recipe's name as a block's title: its words before a parenthesis ("HOO (colour camera dual-band)" -> "HOO");
    the whole name goes to the tooltip."""
    head = (name or "").split(" (", 1)[0].strip()
    return head or (name or "")


def type_spec(t: str) -> dict | None:
    """The recipe a new block of type ``t`` gets: the folder's; a type of before recipes no longer in the folder: its
    frozen content."""
    r = _folder().get(t)
    if r is not None:
        return r
    return _legacy(t) if t in rc.LEGACY else None


def type_title(t: str) -> str:
    """A type's title (neutral, English): its recipe's short name; the id when the folder has none."""
    s = type_spec(t)
    return short(s["name"]) if s else t


def type_ports(t: str) -> dict[str, str]:
    """{role: port} of a new block of type ``t``, in its recipe's order."""
    s = type_spec(t)
    return {i["role"]: i.get("label") or i["role"] for i in (s or {}).get("inputs", [])}


def type_of_title(title: str, default: str = DEFAULT) -> str:
    """The type whose title is ``title`` (a template names its combination by it); ``default`` when none."""
    return next((t for t in offered() if type_title(t) == title), default)


def titles() -> set[str]:
    """Every title a combination may carry: the folder's recipes' and the four types' of before recipes."""
    return {short(r["name"]) for r in _folder().values()} | set(rc.LEGACY_NAMES.values())


# ---------------------------------------------------------------------------------------------- a block
def kind(b) -> str:
    """A Combination block's type (the recipe's id)."""
    return b.settings.get("type", DEFAULT)


def mine(b) -> dict | None:
    """The block's copy of its recipe; None for a block of a project saved before recipes."""
    src = b.source if isinstance(b.source, dict) else None
    r = src.get("recipe") if src else None
    return r if isinstance(r, dict) else None


def spec(b) -> dict | None:
    """The recipe a Combination block runs: its copy, else its type's frozen content (else the folder's)."""
    r = mine(b)
    if r is not None:
        return r
    t = kind(b)
    if t in rc.LEGACY:
        return _legacy(t)
    return _folder().get(t)


def _legacy(t: str) -> dict:
    """The frozen recipe of one of the four types (a block of a project saved before recipes), with the texts of the
    folder's file of that id (the inputs' labels, the parameters' labels and help: they decide nothing)."""
    return _legacy_in(t, str(FOLDER) if FOLDER is not None else None)


@lru_cache(maxsize=None)
def _legacy_in(t: str, folder) -> dict:
    s = rc.legacy_spec(t)                 # (read only: spec's callers copy what they keep)
    f = _read(folder).get(t)
    if f is not None:
        labels = {i["role"]: i.get("label") for i in f["inputs"]}
        for i in s["inputs"]:
            i["label"] = labels.get(i["role"]) or i["role"]
        for n, q in s["parameters"].items():
            q.update({k: v for k, v in (f["parameters"].get(n) or {}).items() if k in rc.PARAM_DOC})
        s["description"] = s["description"] or f.get("description", "")
    return s


def placed(t: str) -> tuple[dict, dict | None]:
    """(settings, source) of a new block of type ``t``: its type, and the folder's recipe copied in (a type the
    folder no longer has: none, its frozen content stands for it)."""
    r = _folder().get(t)
    return ({"type": t} if t != DEFAULT else {}), ({"recipe": copy.deepcopy(r)} if r is not None else None)


def inputs(b) -> list[dict]:
    """The block's inputs: [{"role", "label", "type", "channel"}], in the recipe's order."""
    s = spec(b)
    return [{"role": i["role"], "label": i.get("label") or i["role"], "type": i.get("type", "any"),
             "channel": i.get("channel", "L")} for i in (s or {}).get("inputs", [])]


def ports(b) -> list[str]:
    """Its input ports: the inputs' labels ("S II", "Hα"...), as the wires store them."""
    return [i["label"] for i in inputs(b)]


def role_of(b, port: str) -> str:
    return next((i["role"] for i in inputs(b) if i["label"] == port), port)


def port_of(b, role: str) -> str:
    return next((i["label"] for i in inputs(b) if i["role"] == role), role)


def title(b) -> str:
    """Its title (neutral, English): its recipe's short name."""
    s = spec(b)
    return short(s["name"]) if s else kind(b)


def stage(b) -> str:
    """The engine stage it runs: its type's own for the four types' frozen content, else combine_recipe."""
    return rc.stage_for({"type": kind(b), "recipe": mine(b)})


def fusion(b, ins: list[dict]) -> dict:
    """Its workflow's source: {"fusion": {"type", "recipe": its copy (none without one), "inputs"}}, what
    api.combination_source gives for the same recipe and inputs."""
    r = mine(b)
    if r is None:
        return {"fusion": {"type": kind(b), "inputs": ins}}
    return rc.fusion_source(r, ins)


def parameters(b) -> dict:
    """The recipe's parameters: {name: {"default", "min", "max" or "choices", "label", "help"}}."""
    return copy.deepcopy((spec(b) or {}).get("parameters") or {})


def status(b) -> dict:
    """The block's recipe against the folder's file of the same id (recipes.status): {"id", "project", "folder",
    "newer", "changed"}; a block without a copy is its type's version 1."""
    return rc.status(spec(b) or {"id": kind(b), "version": 1}, _folder())
