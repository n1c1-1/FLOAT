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
#: a stack without a Selection block stacks every frame with the same weight: its stack stage reads the engine's
#: "All frames" (stack.selection = "all"); False: today's default selection
NO_SELECTION_STACKS_ALL = True


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
            rest = [g.blocks[b].kind for b in ix.effective(r.id)[1:]]
            if rest:                                   # a Post layer, not a workflow: nothing runs after it
                out.problems.append(f"“{r.name}”: {', '.join(rest)} after a Combine layers (a Post layer) is not "
                                    "compiled")
            continue
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
        root = ix.row[ix.root(r.id)]
        if root.blocks and g.blocks[root.blocks[0]].kind == "combine_layers":
            out.problems.append(f"the wire {w.id} leaves a block after a Combine layers (a Post layer): its input "
                                "is not compiled")
            continue
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
        from .rules import effective
        src = copy.deepcopy(head.source) if head.source is not None else {
            "path": "", "stretched": bool(effective(g, head.id)["Stretched image"])}      # Commun flattened
        return src, "stretched" if src.get("stretched") else "image"
    ins = []
    for w in g.wires:
        if not w.active or w.dst[0] != head.id:
            continue
        pid = ports_wid(w)
        if pid < 0:
            continue                                   # said by _flows
        i = {"pid": pid, "role": ROLE_OF_PORT.get(w.dst[1], w.dst[1])}
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
        if c is not None:
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
        blk = g.blocks[b]
        if blk.kind == "combine":
            return wfl.BLOCKS_OF_KIND.get(gm.combo_type(blk), "combine")
        return blk.engine or names.get(b) or blk.kind
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
    drawn = {}                       # a night a stack draws: its files are its Session block's (Commun's when none)
    for r in g.rows:
        for s, _c in r.nights:
            if g.blocks[s].sid is not None:
                drawn.setdefault(g.blocks[s].sid, g.blocks[s])
    sessions = []
    for n in g.nights:
        s = by_sid.get(n["sid"]) or Session(n["sid"])
        s.name, s.filter, s.left_out = n.get("name", ""), n.get("filter", ""), copy.deepcopy(n.get("left_out") or {})
        blk = drawn.get(n["sid"])
        for k in gm.FILE_KINDS:
            if blk is not None:
                setattr(s, k, list(blk.files[k] if k in blk.files else g.common["files"].get(k, [])))
            else:
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
        mixed = []
        cals = [c for _s, c in root.nights if c is not None]
        if kind == "frames" and root.calib_node and (cals or not root.nights):
            # no node when no night has a Calibration: the engine takes the lights as they are (apply_masters off)
            offs = [g.blocks[c].off if c is not None else True for _s, c in root.nights]
            if offs and any(offs) != all(offs):
                if g.calibrate_together or len(root.nights) < 2:
                    out.problems.append(f"“{root.name}”: some nights' Calibration off, others on, calibrated "
                                        "together: compiled as on")
                else:                   # each night its own calibration: the switched-off or absent ones apply no
                    mixed = [sid_of(s) for s, c in root.nights if c is None or g.blocks[c].off]       # master
            nodes.append(wfl.node("calibration", off=bool(offs) and all(offs), fork=root.calib_fork))
        for b in f.chain:
            blk = g.blocks[b]
            if blk.kind in ("image", "combine", "combine_layers"):
                if blk.kind == "image" and holder is not None:
                    nodes.append(_align_node(g, holder, by_wid, names))
                continue
            if blk.kind not in NODE_KINDS:
                continue
            nodes.append(wfl.node(blk.kind, blk.method, blk.off, engine(b), blk.fork))
            if blk.kind == "stacking" and holder is not None:
                nodes.append(_align_node(g, holder, by_wid, names))
        rec = f.rec
        if rec.legacy:
            blocks, options, end = None, copy.deepcopy(rec.legacy["options"]), rec.legacy.get("end")
        else:
            blocks = wfl.with_output(nodes, kind)
            try:                        # a graph's options are those of its blocks (the blocks are the truth)
                options = wfl.options_of(blocks, kind)
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
            # a workflow's own list: a block that left its chain takes its pause with it
            have = {engine(b) for b in f.chain}
            bps = [x for x in rec.pauses if x in have or wfl.base(x) not in NODE_KINDS]
        else:
            bps = {engine(b) for b in f.chain if g.blocks[b].pause and g.blocks[b].kind in NODE_KINDS + (
                "image", "combine")}
            # a night's Calibration with its pause: the workflow pauses after it (each night's own when the nights
            # are calibrated each on its own, project2.separate_sessions)
            sep = not g.calibrate_together and len(root.nights) >= 2 and kind == "frames"
            bps |= {f"calibration@{sid_of(s)}" if sep else "calibration" for s, c in root.nights
                    if c is not None and g.blocks[c].pause}
            bps = sorted(bps, key=wfl.block_order)
        own = {str(sid_of(s)): copy.deepcopy(g.blocks[c].own) for s, c in root.nights
               if c is not None and g.blocks[c].own}
        w = old.get(f.wid) or Workflow(f.wid, rec.name)
        params = _settings(g, f, root, holder)
        kinds = [g.blocks[b].kind for b in f.chain]
        if NO_SELECTION_STACKS_ALL and kind == "frames" and "stacking" in kinds and "selection" not in kinds:
            # a stack without a Selection block: every frame kept, all of the same weight (the engine's "All frames")
            params.setdefault("stack", {})["selection"] = "all"
        for sid in mixed:
            params.setdefault(f"calibrate@{sid}", {})["apply_masters"] = False
        w.name, w.options, w.params, w.end = rec.name, options, params, end
        w.variant_of, w.breakpoints, w.source, w.sessions = rec.trial, bps, source, sess
        w.blocks, w.calibration = blocks, own or None
        made[f.wid] = w
        for s, c in root.nights:
            out.map.setdefault(s, []).append((f.wid, "session"))
            if c is not None:
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
    if p.current not in made:           # as Project2.add_workflow / remove_workflow keep it: the first one
        p.current = order[0] if order else None
    _split(g, p, flows, out)
    return out


#: compile(g, base) is compile_into
compile = compile_into  # noqa: A001


def _align_node(g, holder, by_wid, names):
    a = holder.align
    ref = None
    if a.get("ref") is not None:
        ref = next((f.wid for f in by_wid.values() if f.rec.id == a["ref"]), None)
    elif a.get("ref_wid") is not None:
        ref = a["ref_wid"]
    frm = a.get("from_name")
    if a.get("from") is not None and a["from"] in g.blocks:
        frm = _engine_name(g, a["from"], names)
    elif a.get("auto") and a.get("ref") is not None:
        f = next((f for f in by_wid.values() if f.rec.id == a["ref"]), None)
        astro = next((b for b in (f.chain if f else []) if g.blocks[b].kind == "astrometry" and not g.blocks[b].off),
                     None)
        frm = _engine_name(g, astro, names) if astro is not None else None
    return wfl.node("align_on", a.get("method"), bool(a.get("off")), None, a.get("fork"), ref, frm)


def _engine_name(g, b, names):
    blk = g.blocks[b]
    return blk.engine or names.get(b) or blk.kind


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
    """An input scan that reads no disk: the frames of a session are its definition's entries (one frame per folder
    or file given, by its name: two sessions given the same entries have the same frames, as on disk), every file id
    its path. The keys are those of the definition alone; what the engine says of its wiring (a combination's
    grids, Keys.grid) is exact on it."""

    def __new__(cls):
        from pathlib import Path
        from ..core.store import InputScan

        class Scan(InputScan):
            def session_frames(self, project, session):
                return {k: [(Path(f), 0) for f in getattr(session, k)] for k in gm.FILE_KINDS}

            def useful_areas(self, project, sessions):
                return []

            def file_id(self, path):
                return [path, None]

            def image_id(self, path):
                return [path, None, None]
        return Scan()


_RULES = []


def _key_rules():
    if not _RULES:
        from ..core.store import key_rules
        _RULES.append(key_rules())
    return _RULES[0]


def _engine_message(text: str):
    """The engine's wiring text as a message (template, fields), from the templates it is written with (the client
    translates those); the names it carries are kept as written."""
    import re
    from ..stages.align_on import NO_PLATE_REFERENCE, NO_REFERENCE
    from ..stages.combine import GRID_MANY, GRID_ONE, TO_WIRE
    from .msg import M, typed
    for t in (NO_REFERENCE, NO_PLATE_REFERENCE, GRID_ONE, GRID_MANY, TO_WIRE):
        names = re.findall(r"{(\w+)}", t)
        rx = "^" + re.sub(r"\\{(\w+)\\}", r"(?P<\1>.+?)", re.escape(t)) + "$"
        m = re.match(rx, text)
        if m:
            return M(t, **{k: typed(m.group(k)) for k in names}) if names else t
    return M("{_why}", _why=text)


def _wiring(g, p, c):
    """[(wid, stage, block id | None, message)]: what the engine's Keys.wiring_problem says of each stage that reads
    other workflows (a combination's grids, an Align on's target, a plate solution taken from the target), with the
    block the core greys for it: the combination's head; the first block after the Align on (None: none in the
    row); the block of the stage."""
    from ..core.store import Keys
    k = Keys(p, _key_rules(), _NoDisk())
    ix = gm.Index(g)
    block_of = {(wid, name): b for b, rows in c.map.items() for wid, name in rows}
    flows = {}
    for b, rows in c.map.items():
        for wid, _n in rows:
            flows.setdefault(wid, []).append(b)
    out = []
    for w in p.workflows:
        try:
            ch = k.chain(w)
        except (KeyError, ValueError):
            continue
        aligned = "align_on" in ch
        for st in ch:
            bst = wfl.base(st)
            if not (bst in wfl.BLOCKS_OF_KIND.values() or bst == "align_on" or
                    aligned and ch.index(st) > ch.index("align_on")):
                continue
            why = k.wiring_problem(w, st)
            if not why:
                continue
            blk = None
            mine = [b for b in flows.get(w.id, []) if b in ix.row_of]
            if bst in wfl.BLOCKS_OF_KIND.values():
                blk = next((b for b in mine if g.blocks[b].kind == "combine"), None)
            elif bst == "align_on":
                chain = _chain_of(g, ix, mine)
                at = max((i for i, b in enumerate(chain) if g.blocks[b].kind in ("stacking", "image")), default=None)
                blk = chain[at + 1] if at is not None and at + 1 < len(chain) else None
            else:
                n = wfl.node_of(w.blocks or [], st)
                blk = block_of.get((w.id, n["name"])) if n is not None else None
            out.append((w.id, st, blk, _engine_message(why)))
    return out, k


def _chain_of(g, ix, blocks):
    """The chain (left to right) of the chain blocks a workflow passes."""
    mine = set(blocks)
    rows = [r for r in g.rows if any(b in mine for b in r.blocks)]
    longest = max((ix.effective(r.id) for r in rows), key=len, default=[])
    return [b for b in longest if b in mine]


_MEMO = []


def engine_wiring(g) -> dict:
    """{block id: message}: the blocks the engine leaves out until what they read is wired (Keys.wiring_problem on the
    compiled graph), with its reason. Cached for the latest graph object (graphs are never changed in place)."""
    if _MEMO and _MEMO[0] is g:
        return _MEMO[1]
    from ..core.project2 import Project2
    p = Project2(name="wiring")
    out = {}
    try:
        c = compile_into(g, p)
        probs, _k = _wiring(g, p, c)
        for _wid, _st, b, msg in probs:
            if b is not None:
                out.setdefault(b, msg)
    except Exception:  # noqa: BLE001
        out = {}                     # a graph the engine cannot read: its legality says why (engine_problems)
    _MEMO[:] = [g, out]
    return out


def engine_check(g):
    """The engine's verdict on the compiled definition of a graph, what a gesture may not leave: "" when it accepts
    it, else the reason (a message). The compile having no engine form for something, Project2.validate (the graph
    rules of every workflow, its node names, the workflows read: loops, a combination as an alignment target), the
    stage chain of each workflow."""
    from ..core.project2 import Project2
    from .msg import M
    p = Project2(name="check")
    try:
        c = compile_into(g, p)
    except Exception as e:  # noqa: BLE001
        return M("the engine cannot read it: {_why}", _why=str(e))
    if c.problems:
        return M("the engine has no form for it: {_why}", _why=c.problems[0])
    try:
        p.validate()
    except wfl.BlockRefused as e:
        return M(e.template, **e.fields)
    except ValueError as e:
        return M("the engine refuses it: {_why}", _why=str(e))
    for w in p.workflows:
        try:
            p.chain(w)
        except (ValueError, KeyError) as e:
            return M("the engine refuses it: {_why}", _why=str(e))
    return ""


def engine_problems(g, keys=True, info=None) -> list[tuple[str, str]]:
    """What the engine refuses in the compiled definition of a graph: [(class, detail)], [] when it accepts it. The
    detail is a template (ids, numbers, names taken out), so that one rule broken twice is one class.
    Classes: "graph" (wfl.graph_problem of a workflow: its template and the block), "validate" (Project2.validate, when
    no graph rule is broken), "chain" (the stage chain), "wiring" (a stage Keys.wiring_problem says waits for what it
    reads, and the core does not grey its block with the same reason, or the core greys a block the engine does not
    leave out), "compile" (what the compile had to decide: something the engine has no form for)."""
    from ..core.project2 import Project2
    from .msg import say
    p = Project2(name="check")
    out: list[tuple[str, str]] = []
    try:
        c = compile_into(g, p)
    except Exception as e:  # noqa: BLE001
        return [("compile-crash", f"{type(e).__name__}: {_template(str(e))}")]
    out += [("compile", _template(x)) for x in c.problems]
    graph = False
    for w in p.workflows:
        if w.blocks is not None:
            e = wfl.graph_problem(w.blocks, wfl.source_of(w))
            if e is not None:
                out.append(("graph", f"{e.template} [{getattr(e, 'block', '')}, from {wfl.source_of(w)}]"))
                graph = True
    if not graph:
        try:
            p.validate()
        except ValueError as e:
            out.append(("validate", _template(str(e))))
    for w in p.workflows:
        try:
            p.chain(w)
        except (ValueError, KeyError) as e:
            out.append(("chain", _template(str(e))))
    if keys and not any(k in ("validate", "chain", "graph") for k, _ in out):
        try:
            from .rules import evaluate
            info = info or evaluate(g)
            probs, _k = _wiring(g, p, c)
            said = set()
            for _wid, st, b, msg in probs:
                if b is None:
                    continue
                said.add(b)
                ev = info.get(b)
                ok = ev is not None and (ev.cat == "wiring" and say(ev.reason) == say(msg) or ev.cat == "downstream"
                                         or ev.cat == "port" and wfl.base(st) in wfl.BLOCKS_OF_KIND.values())
                if not ok:
                    out.append(("wiring", f"{wfl.base(st).split('_')[0]}: {_template(say(msg))} [core: "
                                          f"{ev.cat if ev else 'absent'}]"))
            for b, ev in info.items():
                if ev.cat == "wiring" and b not in said and g.blocks[b].kind != "combine_layers":
                    out.append(("wiring", f"core greys {g.blocks[b].kind}: {_template(say(ev.reason))}, the engine "
                                          "does not"))
        except Exception as e:  # noqa: BLE001
            out.append(("keys", f"{type(e).__name__}: {_template(str(e))}"))
    return sorted(set(out))


def _template(text: str) -> str:
    """A message with its varying parts (numbers, names in quotes, lists) taken out: its class."""
    import re
    t = re.sub(r"'[^']*'|\"[^\"]*\"|“[^”]*”", "'…'", text)
    t = re.sub(r"\[[^\]]*\]", "[…]", t)
    t = re.sub(r"^(?:[A-Za-zα ]+, )*[A-Za-zα ]+ to wire$", "… to wire", t)
    t = re.sub(r"[A-Za-zα ]+ is not on the grid of [A-Za-zα ]+", "… is not on the grid of …", t)
    t = re.sub(r"'…': [a-z_]+(, [a-z_]+)* after", "'…': … after", t)
    return re.sub(r"\d+", "n", t)
