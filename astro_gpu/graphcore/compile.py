"""Stored graph -> what the engine runs today: the format-2 definition (INTEGRATION §2).

``compile_into(g, p)`` writes, in the SAME ``Project2`` object, its ``sessions``, ``excluded``, ``stack_mode``,
``workflows`` (the existing ``Workflow`` objects kept, by id) and ``next_workflow_id``; it never touches ``post`` nor
the other fields. One workflow per row and per tap, its id the one stored (never renumbered; a row or tap without
one gets the next free id, given back in ``Compiled.wids``); a block's node name is the one stored (``Block.engine``;
a block without one gets the smallest free name over the chains it is in, given back in ``Compiled.names``).

A workflow's settings: those its holders own (its nights' Calibration, its stack's calibration, its chain's blocks,
its Align on, its Output), then its leftovers (``extra`` set, ``unset`` taken out).

The graph is not changed. A combination's input from a block that ends no workflow is compiled as a workflow of its
own (a tap the graph has not stored yet), said in ``problems``. Rows of Combine layers are Post's layers: no workflow.
Post's wires are not compiled (Post's layers are the engine's: ``Project2.post``).

``engine_problems(g)`` (decision B): what the engine refuses in the compiled definition of a graph."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

from ..core import workflow as wfl
from . import model as gm

#: a combination's roles (stages/combine.py) and the graph's port names
ROLE_OF_PORT = {"S II": "SII", "Hα": "Ha", "O III": "OIII", "L": "L", "R": "R", "G": "G", "B": "B"}
IMAGE_OF_PORT = {"Starless": "starless", "Stars only": "stars"}
#: the graph's chain kinds that are an engine node of the same name
NODE_KINDS = ("registration", "selection", "crop", "stacking", "astrometry", "gradient", "deconv", "denoise", "color",
              "stars", "synth")


@dataclass
class Compiled:
    map: dict = field(default_factory=dict)       # {block id: [(wid, node name)]}
    taps: dict = field(default_factory=dict)      # {tap id: wid}
    wids: dict = field(default_factory=dict)      # {row or tap id: wid}, the new ones included
    names: dict = field(default_factory=dict)     # {block id: node name} given to blocks without one
    problems: list = field(default_factory=list)  # what the compile had to decide (texts)
    split: dict = field(default_factory=dict)     # {block id: [what differs between its workflows]}


class _Flow:
    """A workflow to compile: a row, a tap, or an implicit tap (a combination's mid-row source)."""

    def __init__(self, rec, row, chain, implicit=False):
        self.rec, self.row, self.chain, self.implicit = rec, row, chain, implicit
        self.wid = getattr(rec, "wid", None)


def _effective_align(g, ix, chain):
    """The row holding the Align on a chain passes (the row of its Stacking or Image), None."""
    holder = None
    for b in chain:
        if g.blocks[b].kind in ("stacking", "image"):
            r = ix.row[ix.row_of[b]]
            holder = r if r.align else None
    return holder


def _flows(g, ix, out: Compiled) -> list[_Flow]:
    flows = []
    for r in g.rows:
        root = ix.row[ix.root(r.id)]
        if root.blocks and g.blocks[root.blocks[0]].kind == "combine_layers":
            continue                                   # a Post layer, not a workflow
        flows.append(_Flow(r, r, ix.effective(r.id)))
    for t in g.taps:
        if t.block not in ix.row_of:
            continue
        r = ix.row[ix.row_of[t.block]]
        eff = ix.effective(r.id)
        flows.append(_Flow(t, r, eff[:eff.index(t.block) + 1]))
    nxt = max([g.next_wid] + [f.wid + 1 for f in flows if f.wid is not None])
    for f in flows:
        if f.wid is None:
            f.wid, nxt = nxt, nxt + 1
        out.wids[f.rec.id] = f.wid
        if isinstance(f.rec, gm.Tap):
            out.taps[f.rec.id] = f.wid
    # a combination's input from a block where no workflow ends: an implicit tap
    ends = {}
    for f in flows:
        if f.chain:
            ends.setdefault(f.chain[-1], f)
    for w in g.wires:
        if not w.active or w.dst[0] not in g.blocks or g.blocks[w.dst[0]].kind != "combine":
            continue
        src = w.src[0]
        if src not in ix.row_of or src in ends or (w.wid is not None and any(f.wid == w.wid for f in flows)):
            continue
        r = ix.row[ix.row_of[src]]
        eff = ix.effective(r.id)
        f = _Flow(gm.Tap(f"~{src}", src, f"{r.name} · {g.blocks[src].kind}"), r, eff[:eff.index(src) + 1], True)
        f.wid, nxt = nxt, nxt + 1
        flows.append(f)
        ends[src] = f
        out.problems.append(f"the wire {w.id} leaves a block where no workflow ends: compiled as workflow {f.wid}")
    return flows


def _names(g, flows, out: Compiled):
    """The node names: stored, else the smallest free one over every chain the block is in."""
    used: dict[int, set] = {id(f): {g.blocks[b].engine for b in f.chain if g.blocks[b].engine} for f in flows}
    names = {}
    for f in flows:
        for b in f.chain:
            blk = g.blocks[b]
            if blk.engine or b in names or blk.kind not in NODE_KINDS:
                continue
            mine = [x for x in flows if b in x.chain]
            if blk.kind in wfl.IMAGE_OPS:
                k, name = 1, blk.kind
                while any(name in used[id(x)] for x in mine):
                    k += 1
                    name = f"{blk.kind}~{k}"
            else:
                name = blk.kind
            names[b] = name
            for x in mine:
                used[id(x)].add(name)
    out.names = names
    return names


def _source(g, ix, f, root, ports_wid, out: Compiled):
    head = g.blocks[root.blocks[0]] if root.blocks else None
    if root.nights or head is None or head.kind not in ("image", "combine"):
        return None, "frames"
    if head.kind == "image":
        src = copy.deepcopy(head.source) if head.source is not None else {
            "path": "", "stretched": bool(head.settings.get("Stretched image", False))}
        return src, "stretched" if src.get("stretched") else "image"
    ins = []
    for w in g.wires:
        if not w.active or w.dst[0] != head.id:
            continue
        i = {"pid": ports_wid(w), "role": ROLE_OF_PORT.get(w.dst[1], w.dst[1])}
        if w.channel is not None:
            i["channel"] = w.channel
        image = w.image if w.image is not None else IMAGE_OF_PORT.get(w.src[1])
        if image is not None:
            i["image"] = image
        if w.src[0] in g.blocks and g.blocks[w.src[0]].kind not in ("stars",) and w.src[1] != "image":
            out.problems.append(f"the wire {w.id} leaves a {g.blocks[w.src[0]].kind} block: the engine reads an "
                                "Output or a Stars/background")
        ins.append(i)
    return {"fusion": {"type": gm.combo_type(head), "inputs": ins}}, "fusion"


def _settings(g, f, root, holder) -> dict:
    out: dict = {}

    def put(d):
        for st, kv in (d or {}).items():
            out.setdefault(st, {}).update(copy.deepcopy(kv))
    for _s, c in root.nights:
        put(g.blocks[c].params)
    put(root.calib)
    for b in f.chain:
        put(g.blocks[b].params)
    if holder is not None:
        put(holder.align.get("params"))
    if f.rec.output:
        out.setdefault("render_simple", {}).update(copy.deepcopy(f.rec.output))
    put(f.rec.extra)
    for st, k in f.rec.unset or []:
        (out.get(st) or {}).pop(k, None)
    return {st: kv for st, kv in out.items() if kv}


def compile_into(g, p) -> Compiled:
    """Compile the graph ``g`` into the definition ``p`` (in place); returns what the graph's views need."""
    from ..core.project2 import Session, Workflow
    out = Compiled()
    ix = gm.Index(g)
    flows = _flows(g, ix, out)
    names = _names(g, flows, out)

    def engine(b):
        return g.blocks[b].engine or names.get(b) or g.blocks[b].kind
    # -- nights
    sids = [n["sid"] for n in g.nights]
    fresh = {}
    nxt_sid = max(sids, default=0) + 1
    for r in g.rows:
        for s, _c in r.nights:
            if g.blocks[s].sid is None and s not in fresh:
                fresh[s] = nxt_sid
                nxt_sid += 1

    def sid_of(b):
        return g.blocks[b].sid if g.blocks[b].sid is not None else fresh[b]
    by_sid = {s.id: s for s in p.sessions}
    sessions = []
    for n in g.nights:
        s = by_sid.get(n["sid"]) or Session(n["sid"])
        s.name, s.filter, s.left_out = n.get("name", ""), n.get("filter", ""), copy.deepcopy(n.get("left_out") or {})
        for k in gm.FILE_KINDS:
            setattr(s, k, list((n.get("files") or {}).get(k) or []))
        sessions.append(s)
    for b, sid in fresh.items():
        blk = g.blocks[b]
        s = Session(sid, name=str(blk.name or ""))
        for k in gm.FILE_KINDS:
            setattr(s, k, list(blk.files[k] if k in (blk.files or {}) else g.common["files"].get(k, [])))
        sessions.append(s)
    # -- workflows
    ends = {}
    for f in flows:
        if f.chain:
            ends.setdefault(f.chain[-1], f)
    by_wid = {f.wid: f for f in flows}

    def ports_wid(w):
        if w.wid is not None and w.wid in by_wid:
            return w.wid
        f = ends.get(w.src[0])
        return f.wid if f is not None else -1
    old = {w.id: w for w in p.workflows}
    made = {}
    for f in flows:
        root = ix.row[ix.root(f.row.id)]
        holder = _effective_align(g, ix, f.chain)
        source, kind = _source(g, ix, f, root, ports_wid, out)
        nodes = []
        if kind == "frames" and root.calib_node:
            offs = [g.blocks[c].off for _s, c in root.nights]
            if offs and any(offs) != all(offs):
                out.problems.append(f"“{root.name}”: some nights' Calibration off, others on: compiled as on")
            nodes.append(wfl.node("calibration", off=bool(offs) and all(offs), fork=root.calib_fork))
        for b in f.chain:
            blk = g.blocks[b]
            if blk.kind in ("image", "combine", "combine_layers"):
                if blk.kind == "image" and holder is not None:
                    nodes.append(_align_node(g, holder, by_wid))
                continue
            if blk.kind not in NODE_KINDS:
                continue
            nodes.append(wfl.node(blk.kind, blk.method, blk.off, engine(b), blk.fork))
            if blk.kind == "stacking" and holder is not None:
                nodes.append(_align_node(g, holder, by_wid))
        rec = f.rec
        if rec.legacy:
            blocks, options, end = None, copy.deepcopy(rec.legacy["options"]), rec.legacy.get("end")
        else:
            blocks = wfl.with_output(nodes, kind)
            try:
                options = copy.deepcopy(rec.options) if rec.options is not None else wfl.options_of(blocks, kind)
            except ValueError as e:
                out.problems.append(f"workflow {f.wid}: its options: {e}")
                options = dict(wfl.DEFAULT_OPTIONS)
            end = rec.end
        nights = [sid_of(s) for s, _c in root.nights]
        if rec.sessions is not None:
            sess = list(rec.sessions)
        elif rec.all_nights or kind != "frames" or not nights:
            sess = None
        else:
            sess = nights
        if rec.pauses is not None:
            bps = list(rec.pauses)
        else:
            bps = sorted({engine(b) for b in f.chain if g.blocks[b].pause and g.blocks[b].kind in NODE_KINDS + (
                "image", "combine")}, key=wfl.block_order)
        own = {str(sid_of(s)): copy.deepcopy(g.blocks[c].own) for s, c in root.nights if g.blocks[c].own}
        w = old.get(f.wid) or Workflow(f.wid, rec.name)
        w.name, w.options, w.params, w.end = rec.name, options, _settings(g, f, root, holder), end
        w.variant_of, w.breakpoints, w.source, w.sessions = rec.trial, bps, source, sess
        w.blocks, w.calibration = blocks, own or None
        made[f.wid] = w
        for s, c in root.nights:
            out.map.setdefault(s, []).append((f.wid, "session"))
            out.map.setdefault(c, []).append((f.wid, "calibration"))
        for b in f.chain:
            out.map.setdefault(b, []).append((f.wid, engine(b) if g.blocks[b].kind in NODE_KINDS else
                                              g.blocks[b].kind))
    order = [x for x in g.order if x in made] + [f.wid for f in flows if f.wid not in g.order]
    p.sessions[:] = sessions
    p.excluded[:] = [(int(a), str(b)) for a, b in g.excluded]
    p.stack_mode = "together" if g.calibrate_together else "per_session"
    p.workflows[:] = [made[x] for x in order]
    p.next_workflow_id = max([g.next_wid] + [x + 1 for x in made])
    if p.current is not None and p.current not in made:
        p.current = order[0] if order else None
    _split(g, p, flows, out)
    return out


#: compile(g, base) is compile_into
compile = compile_into  # noqa: A001


def _align_node(g, holder, by_wid):
    a = holder.align
    ref = None
    if a.get("ref") is not None:
        ref = next((f.wid for f in by_wid.values() if f.rec.id == a["ref"]), None)
    elif a.get("ref_wid") is not None:
        ref = a["ref_wid"]
    frm = a.get("from_name")
    if a.get("from") is not None and a["from"] in g.blocks:
        frm = g.blocks[a["from"]].engine or g.blocks[a["from"]].kind
    return wfl.node("align_on", a.get("method"), bool(a.get("off")), None, a.get("fork"), ref, frm)


def _split(g, p, flows, out: Compiled):
    """Blocks drawn once whose workflows differ in what the engine implies (§2): keep_halves, catalogs."""
    ws = {w.id: w for w in p.workflows}
    for b, rows in out.map.items():
        blk = g.blocks[b]
        if len(rows) < 2 or blk.kind not in ("denoise", "astrometry"):
            continue
        seen = set()
        for wid, name in rows:
            w = ws.get(wid)
            if w is None or w.blocks is None:
                continue
            if blk.kind == "denoise":
                seen.add(bool(wfl.node_implied(w.blocks, name).get("keep_halves")))
            else:
                act = [n for n in w.blocks if not n.get("off")]
                try:
                    seen.add(any(n["block"] == "color" for n in act) or p.catalogs_wanted(w))
                except (KeyError, ValueError):
                    pass
        if len(seen) > 1:
            out.split.setdefault(b, []).append("keep_halves" if blk.kind == "denoise" else "catalogs")


# ---------------------------------------------------------------------------------------------- legality (decision B)
class _NoDisk:
    """An input scan that reads no disk: every frame list empty (the structure alone is checked)."""

    def __new__(cls):
        from ..core.store import InputScan

        class Scan(InputScan):
            def session_frames(self, project, session):
                return {k: [] for k in gm.FILE_KINDS}

            def file_id(self, path):
                return [path, None]

            def image_id(self, path):
                return [path, None, None]
        return Scan()


_RULES = []


def engine_problems(g, keys=True) -> list[tuple[str, str]]:
    """What the engine refuses in the compiled definition of a graph: [(class, detail)], [] when it accepts it.
    Classes: "validate" (Project2.validate), "graph" (wfl.graph_problem: its template), "chain" (the stage chain),
    "wiring" (Keys.wiring_problem of a combination or an Align on), "compile" (what the compile had to decide)."""
    from ..core.project2 import Project2
    p = Project2(name="check")
    out: list[tuple[str, str]] = []
    try:
        c = compile_into(g, p)
    except Exception as e:  # noqa: BLE001
        return [("compile-crash", f"{type(e).__name__}: {e}")]
    out += [("compile", x) for x in c.problems]
    try:
        p.validate()
    except ValueError as e:
        out.append(("validate", _template(str(e))))
    for w in p.workflows:
        if w.blocks is not None:
            e = wfl.graph_problem(w.blocks, wfl.source_of(w))
            if e is not None:
                out.append(("graph", f"{e.template} [{getattr(e, 'block', '')}]"))
                continue
        try:
            p.chain(w)
        except (ValueError, KeyError) as e:
            out.append(("chain", _template(str(e))))
    if keys and not any(k in ("validate", "chain", "graph") for k, _ in out):
        try:
            from ..core.store import Keys, key_rules
            if not _RULES:
                _RULES.append(key_rules())
            k = Keys(p, _RULES[0], _NoDisk())
            for w in p.workflows:
                ch = k.chain(w)
                for st in ch:
                    if wfl.base(st) in wfl.BLOCKS_OF_KIND.values() or wfl.base(st) == "align_on":
                        why = k.wiring_problem(w, st)
                        if why:
                            out.append(("wiring", f"{wfl.base(st)}: {_template(why)}"))
        except Exception as e:  # noqa: BLE001
            out.append(("keys", f"{type(e).__name__}: {_template(str(e))}"))
    return out


def _template(text: str) -> str:
    """A message with its varying parts (numbers, names in quotes, lists) taken out: its class."""
    import re
    t = re.sub(r"'[^']*'|\"[^\"]*\"|“[^”]*”", "'…'", text)
    t = re.sub(r"\[[^\]]*\]", "[…]", t)
    return re.sub(r"\d+", "n", t)
