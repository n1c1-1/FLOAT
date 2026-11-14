"""The file says everything: every setting of a block's stage written explicitly, at the stage's current default when
nothing stores it (the owner's rule, 14 Nov). Qt-free (the golden check runs it headless).

What is never written (core/store.py, the rule on defaults): the automatic fields (``AUTO_FIELDS``: a stored value
there means "not automatic"), the settings removed from a stage (``RETIRED``), those without effect on a result
(``NON_KEY``: worker counts, an image's path, a combination's inputs), the injected ones (internal fields: the frames
excluded by hand), and the values a workflow's options or graph impose (``wfl.implied``, ``wfl.node_implied``).
A default is written at the stage's level, never over a session's own value ("calibrate@2" keeps it), and only where
no level of the workflow stores the setting: so the effective value of every setting, hence every key, stays the same
(a non-automatic field stored at its static default is dropped from the key; one whose default changed,
``KEY_BASELINE``, is keyed by its effective value stored or not)."""
from __future__ import annotations

import copy

from ..core import workflow as wfl


def _skipped(rules, stage: str) -> set:
    params = rules.params(stage)
    internal = {n for n, f in params.model_fields.items()
                if ((f.json_schema_extra or {}) if isinstance(f.json_schema_extra, dict) else {}).get("level")
                == "internal"}
    return set(rules.auto_fields(stage)) | set(rules.retired(stage)) | set(rules.non_key(stage)) | internal


def missing_defaults(p, rules=None) -> list[tuple[int, str, str, object]]:
    """[(workflow id, stage, setting, default)] a definition does not store and the file must say, in chain order."""
    from ..core.store import Keys, key_rules
    from ..graphcore.compile import _NoDisk
    rules = rules or key_rules()
    keys = Keys(p, rules, _NoDisk())
    out = []
    for w in p.workflows:
        try:
            chain = keys.chain(w)
        except Exception:  # noqa: BLE001 - a workflow the engine cannot chain: nothing to say of it
            continue
        src = wfl.source_of(w)
        for entry in chain:
            stage = entry.split("@")[0]                       # a session's instance: written at the stage's level
            base = wfl.base(stage)
            try:
                defaults = rules.defaults(base)
            except (KeyError, ValueError):
                continue
            skip = _skipped(rules, base)
            try:
                imposed = wfl.node_implied(w.blocks, entry, src) if w.blocks is not None else \
                    wfl.implied(w.options, base)
            except Exception:  # noqa: BLE001 - nothing imposed that can be read: none written over it either
                imposed = None
            if imposed is None:
                continue
            have = set(w.params.get(stage) or {})
            for name, v in defaults.items():
                if name in skip or name in imposed or name in have:
                    continue
                out.append((w.id, stage, name, copy.deepcopy(v)))
    return out


def with_explicit_defaults(g, tr=None, like=None, rules=None):
    """The graph ``g`` with every missing default written on what holds its setting (where the panel would store it:
    graphcore's own store of a workflow's setting): (graph, count). ``g`` itself is not changed. A default changes no
    effective value, so it is written whatever the graph's state (one a gesture leaves waiting is written too).
    ``like``: the project the graph is compiled into (its reference image decides chains), else an empty one;
    ``rules``: its stages' rules (``api``'s of that project), else the engine's."""
    from ..core.project2 import Project2
    from ..graphcore import model as gm
    from ..graphcore.compile import compile_into
    from ..graphcore.operations import _flow_of, _stage_names, _store
    q = Project2.from_dict(like.to_dict(), like.root) if like is not None else Project2(name="")
    c = compile_into(g, q)
    flow_of = {wid: fid for fid, wid in c.wids.items()}
    seen, items = set(), []
    for wid, stage, name, v in missing_defaults(q, rules):
        f = flow_of.get(wid)
        if f is None or (f, stage, name) in seen:
            continue
        seen.add((f, stage, name))
        items.append((f, stage, name, v))
    if not items:
        return g, 0
    g2 = gm.clone(g)
    names, ix = _stage_names(g2), gm.Index(g2)
    for f, stage, name, v in items:
        _store(g2, ix, _flow_of(g2, f), stage, name, v, names)
    gm.check(g2)
    return g2, len(items)


def strip_defaults(d: dict, like: dict, rules=None) -> dict:
    """The definition ``d`` (to_dict) without the settings it stores at their static default where ``like`` (the same
    project before the defaults were written) stores nothing: what the golden check compares."""
    from ..core.store import key_rules
    rules = rules or key_rules()
    d = copy.deepcopy(d)
    before = {w["id"]: w for w in like.get("workflows") or []}
    for w in d.get("workflows") or []:
        old = (before.get(w["id"]) or {}).get("params") or {}
        for stage, kv in list((w.get("params") or {}).items()):
            try:
                defaults = rules.defaults(wfl.base(stage.split("@")[0]))
            except (KeyError, ValueError):
                continue
            for name in list(kv):
                if name not in (old.get(stage) or {}) and name in defaults and kv[name] == defaults[name]:
                    del kv[name]
            if not kv:
                del w["params"][stage]
    return d


def strip_override_defaults(table: dict, rules=None) -> dict:
    """A Keys table (graphcore.proof.keys_table) whose overrides (stored settings that are not keys but reach the jobs)
    leave out the values stored at their static default: the jobs get the same settings."""
    import json
    from ..core.store import key_rules
    rules = rules or key_rules()
    for row in table.values():
        for st, js in (row.get("overrides") or {}).items():
            base = wfl.base(st.split("@")[0])
            try:
                dft, auto = rules.defaults(base), rules.auto_fields(base)
            except (KeyError, ValueError):
                continue
            d = json.loads(js)
            row["overrides"][st] = json.dumps({k: v for k, v in d.items() if k in auto or k not in dft or v != dft[k]},
                                              sort_keys=True, default=str)
    return table
