"""The proof that a project converted then compiled is the same project for the engine (INTEGRATION §6).

``round_trip(p)``: in this process, the definition ``p`` loaded, ``q`` a copy of it, ``compile_into(convert(p), q)``;
then ``compare(p, q)``: (a) the definitions (sessions, excluded, stack mode, workflows field by field, next id),
(b) every stage key, block key and key override, (c) the review keys; and ``drawing_diff``: the new graph's rows
against the tree today's client draws. Each gives a list of differences, empty when equal; the first one says what,
where and why."""
from __future__ import annotations

import hashlib
import json

from ..core import workflow as wfl
from . import model as gm
from .compile import compile_into
from .convert import Client, convert, node_key, view_key

WF_FIELDS = ("id", "name", "options", "params", "end", "variant_of", "breakpoints", "source", "sessions", "blocks",
             "calibration")


def _first_diff(a, b, where="") -> str:
    if type(a) is not type(b) and not (isinstance(a, (int, float)) and isinstance(b, (int, float))):
        return f"{where}: {a!r} -> {b!r}"
    if isinstance(a, dict):
        for k in list(a) + [k for k in b if k not in a]:
            if k not in b:
                return f"{where}.{k}: present -> absent"
            if k not in a:
                return f"{where}.{k}: absent -> {b[k]!r}"
            d = _first_diff(a[k], b[k], f"{where}.{k}")
            if d:
                return d
        return ""
    if isinstance(a, list):
        if len(a) != len(b):
            return f"{where}: {len(a)} items -> {len(b)} ({a!r:.200} -> {b!r:.200})"
        for i, (x, y) in enumerate(zip(a, b)):
            d = _first_diff(x, y, f"{where}[{i}]")
            if d:
                return d
        return ""
    return "" if a == b else f"{where}: {a!r} -> {b!r}"


def definition_diff(p, q) -> list[str]:
    """(a): the graph's part of two definitions, every field; [] when equal."""
    a, b = p.to_dict(), q.to_dict()
    out = []
    for k in ("sessions", "excluded", "stack_mode", "next_workflow_id"):
        d = _first_diff(a[k], b[k], k)
        if d:
            out.append(d)
    ia, ib = [w["id"] for w in a["workflows"]], [w["id"] for w in b["workflows"]]
    if ia != ib:
        out.append(f"workflows: ids {ia} -> {ib}")
    wb = {w["id"]: w for w in b["workflows"]}
    for w in a["workflows"]:
        w = _normalised(w)
        x = wb.get(w["id"])
        if x is None:
            continue
        for f in WF_FIELDS:
            d = _first_diff(w.get(f), x.get(f), f"workflow {w['id']}.{f}")
            if d:
                out.append(d)
    return out


#: the normalisations of a definition the proof allows, each key-neutral:
#: - a graph's options are those of its blocks (engine review, second): the keys of a workflow with blocks read its
#:   blocks, never its options (core/store.py Keys.chain and Keys._given)
def _normalised(w: dict) -> dict:
    from ..core import workflow as wfl
    if w.get("blocks") is None:
        return w
    try:
        src = "frames" if not w.get("source") else "fusion" if "fusion" in w["source"] else             "stretched" if w["source"].get("stretched") else "image"
        return {**w, "options": wfl.options_of(w["blocks"], src)}
    except (ValueError, KeyError):
        return w


def keys_table(p, rules, scan) -> dict:
    """{wid: {"chain", "stage", "block", "overrides"}} of every workflow (errors as text)."""
    from ..core.store import Keys
    k = Keys(p, rules, scan)
    out = {}
    for w in p.workflows:
        row: dict = {}
        try:
            ch = k.chain(w)
            row["chain"] = list(ch)
            row["stage"] = {s: k.key(w, s) for s in ch}
            row["overrides"] = {s: json.dumps(k.overrides(w, s), sort_keys=True, default=str) for s in ch}
            row["block"] = {b: list(k.block_key(w, b)) for b, _ss in rules.chains.blocks_in(ch)}
        except Exception as e:  # noqa: BLE001
            row["error"] = f"{type(e).__name__}: {e}"
        out[w.id] = row
    return out


def keys_diff(p, q, rules, scan) -> list[str]:
    """(b): every stage key, block key and override of every workflow."""
    a, b = keys_table(p, rules, scan), keys_table(q, rules, scan)
    out = []
    for wid in a:
        d = _first_diff(a[wid], b.get(wid), f"workflow {wid}")
        if d:
            out.append(d)
    return out


def review_keys(p, cl: Client) -> dict:
    """The review key of each block the client draws (client/model.ProjectModel.review_key), without Qt."""
    out = {}
    exc = sorted([list(x) for x in p.excluded])
    for w in p.workflows:
        refs = {view_key(b): key for b, key in cl.blocks[w.id]}
        for k in cl.display_keys(w):
            parts = [_engine(k), list(refs[k]) if k in refs else []]
            st = w.params.get("stack") or {}
            if k == "crop":
                parts += [list(refs.get("alignement", [])), st.get("zone")]
            if k == "selection":
                parts += [list(refs.get("alignement", [])), st.get("min_transparency"), st.get("max_fwhm_ratio"),
                          st.get("selection", "min_weight"), st.get("selection_transparency", "on"), exc]
            raw = json.dumps(parts, sort_keys=True, default=str)
            out[f"{w.id}|{k}"] = f"{_engine(k)}:{hashlib.sha1(raw.encode()).hexdigest()[:16]}"
    return out


def _engine(k):
    from .convert import KEY_ENGINE
    return KEY_ENGINE.get(k, k)


def states_diff(p, q) -> list[str]:
    """(c): the states the API gives (``update=False``: nothing is written)."""
    from .. import api

    def table(x):
        return {f"{wid}|{st}": [s.state, s.reason, s.key, s.previous_key, s.fallback]
                for (wid, st), s in api.states(x, update=False).items()}
    a, b = table(p), table(q)
    return [f"state {k}: {a.get(k)} -> {b.get(k)}" for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)]


# ---------------------------------------------------------------------------------------------- the drawing (d)
#: the differences the graph makes on purpose, each said where it is applied (none is key-relevant: a workflow
#: made before keeps its options, compiled back as they are)
NORMALISATIONS = {
    1: "a mono workflow made before the free graph has no Color (INTEGRATION 5.9: its chain has none; today's page "
       "draws it greyed)",
    2: "a workflow made before the free graph with a stored zone has a Crop (INTEGRATION 5.9; today's page draws none)",
}


def recorded_signature(drawing: dict, legacy: set = frozenset()) -> set:
    """The blocks today's client draws and the workflows each is shared by: chain blocks by the keys from the
    root to them, nights' Session and Calibration blocks by their session. Post's layers (Combine layers) are left
    out: they are Post's, not the graph's rows. ``legacy``: the ids of the workflows made before the free graph
    (NORMALISATION 1)."""
    nodes = drawing["nodes"]
    by = {n["i"]: n for n in nodes}

    def dropped(n):
        return n["key"] == "couleur" and n["grey"] and n["paths"] and set(n["paths"]) <= set(legacy)
    out = set()
    for n in nodes:
        if n["key"] in ("sources", "final", "composition") or dropped(n):
            continue
        paths = tuple(sorted(n["paths"]))
        if n["key"] == "session":
            out.add(("session", n["sid"], paths))
            continue
        if n["key"] == "calibration" and n["sid"] is not None:
            if paths:                       # a night every workflow calibrates apart: its own block is not drawn
                out.add(("night-calibration", n["sid"], paths, bool(n["cut"])))
            continue
        keys, x = [], n
        while x is not None and x["key"] != "sources":
            if not dropped(x):
                keys.append(x["key"])
            x = by.get(x["parent"]) if x["parent"] is not None else None
        out.add(("block", tuple(reversed(keys)), paths, bool(n["cut"]), bool(n["grey"])))
    return out


def graph_signature(g, q, cl: Client) -> set:
    """The same for the new graph: its blocks and the workflows (rows and taps) whose chain holds each."""
    ix = gm.Index(g)
    flows = []
    for r in g.rows:
        flows.append((r, r, ix.effective(r.id)))
    for t in g.taps:
        r = ix.row[ix.row_of[t.block]]
        eff = ix.effective(r.id)
        flows.append((t, r, eff[:eff.index(t.block) + 1]))
    sep = bool(q.separate_sessions())
    seq: dict = {}
    nights: dict = {}
    for rec, r, chain in flows:
        if rec.wid is None:
            continue
        root = ix.row[ix.root(r.id)]
        w = q.workflow(rec.wid)
        mono = cl.is_mono(w)
        items = []
        if root.nights and not sep and root.calib_node:
            # one Calibration of all the nights: the nights' (INTEGRATION 4.1: two stacks of the same nights share
            # it, drawn in each stack), not the row's
            off = all(g.blocks[c].off for _s, c in root.nights)
            ident = ("calib", tuple(g.blocks[s].sid for s, _c in root.nights), root.calib_fork, off,
                     json.dumps([[g.blocks[c].own, g.blocks[c].params] for _s, c in root.nights] + [root.calib],
                                sort_keys=True))
            items.append((ident, "calibration", off, False))
        holder = None
        for b in chain:
            blk = g.blocks[b]
            if blk.kind in ("stacking", "image"):
                holder = ix.row[ix.row_of[b]] if ix.row[ix.row_of[b]].align else None
        for b in chain:
            blk = g.blocks[b]
            if blk.kind == "crop" and rec.legacy:
                continue                    # NORMALISATION 2 (5.9): a Crop given to a workflow made before (its zone)
            if blk.kind == "combine":
                key = "fusion"
            elif blk.kind == "image":
                key = "image"
            else:
                key = node_key({"block": blk.kind, "name": blk.engine or blk.kind})
            grey = blk.kind == "color" and mono
            items.append((b, key, bool(blk.off), grey))
            if blk.kind in ("stacking", "image") and holder is not None:
                items.append((("align", holder.id), "align_on", bool(holder.align.get("off")), False))
        seq[rec.wid] = items
        if sep and root.nights:
            for s, c in root.nights:
                sid = g.blocks[s].sid
                nights.setdefault(("session", sid), set()).add(rec.wid)
                cal = g.blocks[c]
                if root.calib_node:
                    mark = (cal.own or {}).get("fork")
                    nights.setdefault(("night-calibration", sid, mark, cal.off), set()).add(rec.wid)
    through: dict = {}
    for wid, items in seq.items():
        for i, (ident, key, cut, grey) in enumerate(items):
            pre = tuple(k for _i, k, _c, _g in items[:i + 1])
            e = through.setdefault((ident, pre), [set(), cut, grey])
            e[0].add(wid)
    out = set()
    for (ident, pre), (ws, cut, grey) in through.items():
        out.add(("block", pre, tuple(sorted(ws)), cut, grey))
    for k, ws in nights.items():
        if k[0] == "session":
            out.add(("session", k[1], tuple(sorted(ws))))
        else:
            out.add(("night-calibration", k[1], tuple(sorted(ws)), k[3]))
    return out


def drawing_diff(recorded: set, mine: set) -> list[str]:
    a, b = recorded - mine, mine - recorded
    return [f"drawn today, not by the graph: {x}" for x in sorted(a, key=str)] + \
        [f"drawn by the graph, not today: {x}" for x in sorted(b, key=str)]


def legacy_ids(p) -> set:
    return {w.id for w in p.workflows if w.blocks is None and not (isinstance(w.source, dict) and "fusion" in w.source)}


def drawing_check(p, g, q, drawing: dict, cl: Client | None = None) -> tuple[list[str], int]:
    """(d): the differences between the drawing of today's client (``drawing``, as tools/graph_golden records it)
    and the graph's rows; and how many drawn blocks were compared."""
    rec = recorded_signature(drawing, legacy_ids(p))
    return drawing_diff(rec, graph_signature(g, q, cl or Client(q))), len(rec)


# ---------------------------------------------------------------------------------------------- one project
def round_trip(p, block_keys=None, blank=True):
    """(g, q, log): the graph of ``p`` and a copy of ``p`` the graph is compiled into. ``blank``: the copy's graph
    part emptied first (no session, no workflow, the other stack mode, ids from 1), so that nothing in it is equal
    for being left as it was."""
    from ..core.project2 import Project2
    log: list[str] = []
    g = convert(p, block_keys, log)
    q = Project2.from_dict(p.to_dict(), p.root)
    if blank:
        q.sessions, q.workflows, q.excluded = [], [], []
        q.stack_mode = "per_session" if p.stack_mode == "together" else "together"
        q.next_workflow_id = 1
    compile_into(g, q)
    return g, q, log
