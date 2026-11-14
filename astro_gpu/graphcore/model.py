"""Stored model (§1): blocks, rows, port wires, ids. Stored = drawn: nothing here is inferred.

Beside what is drawn, the model holds what the engine's definition needs to be compiled back exactly (compile.py),
filled by the converter of existing projects (convert.py): the workflow id of each row and tap, the engine's node
names and methods, the owned engine settings of each block, the nights as the project stores them, the per-workflow
leftovers. Each of these fields has a default that the operations of the core leave as it is."""
import copy
import json
from dataclasses import dataclass, field

FILE_KINDS = ("lights", "darks", "flats", "bias", "flat_darks")
FRAME_KINDS = ("registration", "selection", "crop", "stacking")
JOIN_KINDS = ("combine", "combine_layers")
SOURCE_KINDS = ("session", "image")
COMBO_PORTS = {"sho": ["S II", "Hα", "O III"], "ho": ["Hα", "O III"], "rgb": ["R", "G", "B"],
               "lrgb": ["L", "R", "G", "B"]}
COMBO_LABEL = {"sho": "S-H-O", "ho": "H-O", "rgb": "RGB", "lrgb": "LRGB"}
LAYER_PORTS = ["Background", "Stars"]
POST_PORT = "layers"
PAIR_PORTS = ["Starless", "Stars only"]


@dataclass
class Block:
    id: str
    kind: str
    settings: dict = field(default_factory=dict)
    off: bool = False
    pause: bool = False
    name: object = None      # Session only: the night's name
    files: object = None     # Session only: {kind: [paths]}
    engine: object = None    # chain block: the engine's node name ("gradient~2"); None: given by the compile
    method: object = None    # chain block: the engine node's method; None: the block's default
    fork: object = None      # frame block placed in parallel: its mark (no key reads it)
    params: dict = field(default_factory=dict)   # the engine settings it owns: {stage entry: {param: value}}
    sid: object = None       # Session / Calibration of a night: the session id (None: a night not compiled yet)
    own: object = None       # Calibration of a night with its own files: {"fork": n, kind: [paths]}
    source: object = None    # Image: {"path", "stretched"} as stored


@dataclass
class Row:
    id: str
    name: str
    blocks: list = field(default_factory=list)   # chain, left to right (Session/Calibration excluded)
    nights: list = field(default_factory=list)   # [[session id, calibration id | None]], first = top (None: a
    #                                              night without Calibration: its lights are taken as they are)
    fork: object = None                          # [row id, block id] for a variant
    # -- the row's workflow (compile.py; see FLOW_FIELDS)
    wid: object = None           # its workflow id (None: a row the compile numbers)
    all_nights: bool = False     # its workflow stacks every night of the project, those added later too
    sessions: object = None      # the workflow's session list when its order is not the row's nights' order
    output: dict = field(default_factory=dict)    # its Output settings (render_simple)
    extra: dict = field(default_factory=dict)     # settings its blocks do not give, verbatim {stage: {param: v}}
    unset: list = field(default_factory=list)     # [stage, param] its blocks give and its workflow has not
    pauses: object = None        # its breakpoints when the blocks' pauses do not give them
    trial: object = None         # the workflow id it is a trial of (Replace or Try pending)
    options: object = None       # its options when they are not those of its blocks
    end: object = None           # its early exit when it has one beside its blocks
    legacy: object = None        # a workflow made before the free graph: {"options", "end", "frozen"}
    # -- the stack (a root frames row)
    calib_node: bool = True      # its chain starts with the engine's Calibration node
    calib_fork: object = None    # that node's mark
    calib: dict = field(default_factory=dict)     # its settings (inventory, calibrate)
    align: object = None         # {"ref": row or tap id | None, "from": block id, "from_name", "ref_wid", "method",
    #                              "off", "fork", "params", "auto"}: the Align on after its Stacking (or Image), its
    #                              target ("ref" None: a target to choose); "auto": from the target's Astrometry
    #                              when it has one (its plate solution comes with the grid), else its Stacking or
    #                              Image; None: the row keeps its own grid


#: the per-workflow fields a row and a tap share
FLOW_FIELDS = ("wid", "all_nights", "sessions", "output", "extra", "unset", "pauses", "trial", "options", "end",
               "legacy")


@dataclass
class Tap:
    """A workflow that is a row's chain up to one of its blocks, then its Output: a block's image sent to Post, a
    workflow equal to another (a duplicate), a mid-row source of a combination (INTEGRATION 4.6)."""
    id: str
    block: str
    name: str
    wid: object = None
    all_nights: bool = False
    sessions: object = None
    output: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)
    unset: list = field(default_factory=list)
    pauses: object = None
    trial: object = None
    options: object = None
    end: object = None
    legacy: object = None
    auto: bool = False       # made for a combination's wire from a mid-row block: goes when no such wire leaves it


@dataclass
class Wire:
    id: str
    src: list      # [block id, out port]
    dst: list      # [block id, in port]
    active: bool = True
    wid: object = None       # the workflow whose image it carries (several end on one block); None: the block's
    channel: object = None   # a combination's input: the channel read ("L", "R", ...); None: absent (L)
    image: object = None     # a combination's input: "output" | "starless" | "stars"; None: absent (output)


@dataclass
class Project:
    name: str = "Project"
    target: str = ""
    rows: list = field(default_factory=list)
    blocks: dict = field(default_factory=dict)
    wires: list = field(default_factory=list)
    common: dict = field(default_factory=lambda: {"files": {}, "settings": {}})
    next_id: int = 1
    # -- the engine's definition (compile.py)
    nights: list = field(default_factory=list)    # [{"sid", "name", "filter", "files", "left_out"}], session order
    excluded: list = field(default_factory=list)  # [[session id, file name]] left out of every stack
    calibrate_together: bool = True               # one calibration for all the nights (else each night its own)
    next_wid: int = 1                             # the next workflow id (ids are never reused)
    order: list = field(default_factory=list)     # the workflow ids in the definition's order
    taps: list = field(default_factory=list)      # [Tap]

    @property
    def reference(self):
        """The stack the others are aligned on (derived, never stored): the target most rows point at (the first of
        them on a tie); with no row aligned, the first stack; None without a stack."""
        ids = [r.id for r in self.rows] + [t.id for t in self.taps]
        refs = [r.align["ref"] for r in self.rows if r.align and r.align.get("ref") in ids]
        if refs:
            return max(dict.fromkeys(refs), key=lambda x: (refs.count(x), -ids.index(x)))
        return next((r.id for r in self.rows if r.nights and not r.fork), None)


def new_id(p, prefix):
    i = p.next_id
    p.next_id += 1
    return f"{prefix}{i}"


def new_block(p, kind, **kw):
    b = Block(new_id(p, "b"), kind, **kw)
    if kind == "session" and b.files is None:
        b.files = {}          # absent kind = follows Commun
    p.blocks[b.id] = b
    return b


def blank_project(name="Project"):
    p = Project(name=name)
    new_block(p, "post")
    return p


# ---------------------------------------------------------------------------------------------------- ports
def combo_type(b):
    return b.settings.get("type", "sho")


def in_ports(b):
    if b.kind == "combine":
        return list(COMBO_PORTS[combo_type(b)])
    if b.kind == "combine_layers":
        return list(LAYER_PORTS)
    if b.kind == "post":
        return [POST_PORT]
    return []


def out_ports(b):
    if b.kind in ("stars", "synth"):
        return list(PAIR_PORTS)
    if b.kind in ("session", "calibration", "registration", "selection", "crop", "post"):
        return []
    return ["image"]


# ---------------------------------------------------------------------------------------------------- reading
class Index:
    """Lookups computed from a project (never stored)."""

    def __init__(self, p):
        self.p = p
        self.row = {r.id: r for r in p.rows}
        self.row_of, self.pos, self.night_of = {}, {}, {}
        for r in p.rows:
            for i, b in enumerate(r.blocks):
                self.row_of[b], self.pos[b] = r.id, i
            for s, c in r.nights:
                self.row_of[s], self.night_of[s] = r.id, s
                if c is not None:
                    self.row_of[c], self.night_of[c] = r.id, s
        self.children = {r.id: [] for r in p.rows}
        for r in p.rows:
            if r.fork and r.fork[0] in self.children:
                self.children[r.fork[0]].append(r.id)
        self.post = next((b.id for b in p.blocks.values() if b.kind == "post"), None)

    def root(self, rid):
        seen = set()
        while self.row[rid].fork and rid not in seen:
            seen.add(rid)
            rid = self.row[rid].fork[0]
        return rid

    def is_join_row(self, rid):
        r = self.row[self.root(rid)]
        return bool(r.blocks) and self.p.blocks[r.blocks[0]].kind in JOIN_KINDS

    def is_frames(self, rid):
        return bool(self.row[rid].nights)

    def subtree(self, rid):
        out = [rid]
        for c in self.children[rid]:
            out += self.subtree(c)
        return out

    def pred(self, b):
        """Predecessor of chain block b in its effective chain (fork block for a variant's first), or None."""
        r = self.row[self.row_of[b]]
        i = self.pos[b]
        if i > 0:
            return r.blocks[i - 1]
        return r.fork[1] if r.fork else None

    def effective(self, rid):
        """Effective chain of a row: its ancestors' chain up to the fork block, then its own blocks."""
        r = self.row[rid]
        if not r.fork:
            return list(r.blocks)
        par = self.effective(r.fork[0])
        return par[:par.index(r.fork[1]) + 1] + list(r.blocks)

    def last(self, rid):
        return self.row[rid].blocks[-1] if self.row[rid].blocks else None


def row_kind(ix, rid):
    r = ix.row[rid]
    if r.fork:
        return "variant"
    if r.nights:
        return "stack"
    k = ix.p.blocks[r.blocks[0]].kind if r.blocks else ""
    if k == "image":
        return "image"
    if k in JOIN_KINDS or not r.all_nights:
        return "join"
    return "stack"                                     # a stack of every night, made before any night was added


# ---------------------------------------------------------------------------------------------------- save / load
_dc = copy.deepcopy


def _flow_dict(f):
    return {k: _dc(getattr(f, k)) for k in FLOW_FIELDS}


def to_dict(p):
    """The dataclasses as plain dicts, fields in declaration order (same result as dataclasses.asdict)."""
    return {"name": p.name, "target": p.target,
            "rows": [{"id": r.id, "name": r.name, "blocks": list(r.blocks), "nights": [list(n) for n in r.nights],
                      "fork": list(r.fork) if r.fork else r.fork, **_flow_dict(r),
                      "calib_node": r.calib_node, "calib_fork": r.calib_fork, "calib": _dc(r.calib),
                      "align": _dc(r.align)} for r in p.rows],
            "blocks": {k: {"id": b.id, "kind": b.kind, "settings": dict(b.settings), "off": b.off, "pause": b.pause,
                           "name": b.name, "files": {f: list(v) for f, v in b.files.items()}
                           if b.files is not None else None, "engine": b.engine, "method": b.method, "fork": b.fork,
                           "params": _dc(b.params), "sid": b.sid, "own": _dc(b.own), "source": _dc(b.source)}
                       for k, b in p.blocks.items()},
            "wires": [{"id": w.id, "src": list(w.src), "dst": list(w.dst), "active": w.active, "wid": w.wid,
                       "channel": w.channel, "image": w.image} for w in p.wires],
            "common": {"files": {k: list(v) for k, v in p.common["files"].items()},
                       "settings": {k: dict(v) for k, v in p.common["settings"].items()}},
            "next_id": p.next_id,
            "nights": _dc(p.nights), "excluded": [list(x) for x in p.excluded],
            "calibrate_together": p.calibrate_together, "next_wid": p.next_wid, "order": list(p.order),
            "taps": [{"id": t.id, "block": t.block, "name": t.name, **_flow_dict(t), "auto": t.auto} for t in p.taps]}


def from_dict(d):
    p = Project(**{k: v for k, v in d.items() if k not in ("rows", "blocks", "wires", "taps", "reference")})
    p.rows = [Row(**r) for r in d["rows"]]
    p.blocks = {k: Block(**b) for k, b in d["blocks"].items()}
    p.wires = [Wire(**w) for w in d["wires"]]
    p.taps = [Tap(**t) for t in d.get("taps") or []]
    return p


def model_json(p):
    return json.dumps(to_dict(p), ensure_ascii=False, sort_keys=False)


def _dcq(x):
    """Deep copy, fast for the empty and plain values most blocks and rows hold."""
    if x is None or isinstance(x, (str, int, float, bool)):
        return x
    return type(x)() if not x else _dc(x)


def clone(p):
    """Fast deep copy of a project."""
    q = Project(p.name, p.target, next_id=p.next_id,
                common={"files": {k: list(v) for k, v in p.common["files"].items()},
                        "settings": {k: dict(v) for k, v in p.common["settings"].items()}},
                nights=_dcq(p.nights), excluded=[list(x) for x in p.excluded],
                calibrate_together=p.calibrate_together, next_wid=p.next_wid, order=list(p.order))
    q.rows = [Row(r.id, r.name, list(r.blocks), [list(n) for n in r.nights], list(r.fork) if r.fork else None,
                  **{k: _dcq(getattr(r, k)) for k in FLOW_FIELDS}, calib_node=r.calib_node, calib_fork=r.calib_fork,
                  calib=_dcq(r.calib), align=_dcq(r.align))
              for r in p.rows]
    q.blocks = {k: Block(b.id, b.kind, dict(b.settings), b.off, b.pause, b.name,
                         {f: list(v) for f, v in b.files.items()} if b.files is not None else None,
                         b.engine, b.method, b.fork, _dcq(b.params), b.sid, _dcq(b.own), _dcq(b.source))
                for k, b in p.blocks.items()}
    q.wires = [Wire(w.id, list(w.src), list(w.dst), w.active, w.wid, w.channel, w.image) for w in p.wires]
    q.taps = [Tap(t.id, t.block, t.name, **{k: _dcq(getattr(t, k)) for k in FLOW_FIELDS}, auto=t.auto) for t in p.taps]
    return q


# ---------------------------------------------------------------------------------------------------- validator
class Invalid(Exception):
    pass


def check(p):
    """Structural validator (§1). Raises Invalid with the first problem found."""
    def need(cond, msg):
        if not cond:
            raise Invalid(msg)
    ix = Index(p)
    need(sum(b.kind == "post" for b in p.blocks.values()) == 1, "exactly one Post")
    seen = {}
    for r in p.rows:
        need(r.name and r.name.strip(), f"row {r.id} unnamed")
        for b in r.blocks + [x for n in r.nights for x in n if x is not None]:
            need(b in p.blocks, f"row {r.id}: unknown block {b}")
            need(b not in seen, f"block {b} in two rows")
            seen[b] = r.id
    for b in p.blocks.values():
        need((b.kind == "post") == (b.id not in seen), f"block {b.id} ({b.kind}) in no row")
        need((b.kind == "session") == (b.files is not None), f"files on {b.id}")
        if b.files is not None:
            need(all(k in FILE_KINDS for k in b.files), f"file kinds of {b.id}")
            need(list(b.files) == [k for k in FILE_KINDS if k in b.files], f"file kinds order of {b.id}")
            for k, v in b.files.items():
                need(v != p.common["files"].get(k, []), f"own list of {b.id}/{k} equal to Commun's")
                need(len(v) == len(set(v)), f"duplicate path in {b.id}/{k}")
    need(set(p.common) == {"files", "settings"}, "common shape")
    need(all(k in FILE_KINDS and k != "lights" and v for k, v in p.common["files"].items()), "common files")
    need(list(p.common["files"]) == [k for k in FILE_KINDS if k in p.common["files"]], "common files order")
    for k, st in p.common["settings"].items():
        need(st and list(st) == sorted(st), f"common settings of {k}")
    for b in p.blocks.values():
        need(list(b.settings) == sorted(b.settings), f"settings order of {b.id}")
    need(len({r.id for r in p.rows}) == len(p.rows), "duplicate row id")
    need(len({r.name for r in p.rows}) == len(p.rows), "duplicate row name")
    for r in p.rows:
        kinds = [p.blocks[b].kind for b in r.blocks]
        for s, c in r.nights:
            need(p.blocks[s].kind == "session" and (c is None or p.blocks[c].kind == "calibration"),
                 f"night pair of {r.id}")
            need(p.blocks[s].name, f"night {s} unnamed")
        for i, k in enumerate(kinds):
            need(k not in ("session", "calibration", "post"), f"{k} in chain of {r.id}")
            need(i == 0 or k not in JOIN_KINDS + ("image",), f"{k} not at head of {r.id}")
        if r.fork:
            need(not r.nights and r.blocks, f"variant {r.id} shape")
            need(r.fork[0] in ix.row and r.fork[1] in ix.row[r.fork[0]].blocks, f"fork of {r.id}")
            need(kinds[0] not in JOIN_KINDS + ("image",), f"variant {r.id} head")
            need(ix.root(r.id) != r.id and not ix.row[ix.root(r.id)].fork, f"fork loop at {r.id}")
        elif r.nights or (r.all_nights and not any(k in JOIN_KINDS + ("image",) for k in kinds)):
            pass                                       # a stack, or a stack being built (no Stacking yet, no block);
            #                                            one of every night made before any night was added
        else:
            need(kinds and kinds[0] in JOIN_KINDS + ("image",), f"row {r.id} has no head")
    by_sid = {}
    for r in p.rows:
        sids = [p.blocks[s].sid for s, _c in r.nights if p.blocks[s].sid is not None]
        need(len(sids) == len(set(sids)), f"row {r.id} draws one session twice")
        for s, _c in r.nights:
            if p.blocks[s].sid is not None:
                by_sid.setdefault(p.blocks[s].sid, []).append(p.blocks[s])
    for sid, bs in by_sid.items():
        need(all(b.files == bs[0].files and b.name == bs[0].name for b in bs), f"session {sid} drawn twice unlike")
    flows = {r.id for r in p.rows} | {t.id for t in p.taps}
    for r in p.rows:
        if r.align is not None:
            need(isinstance(r.align, dict), f"align of {r.id}")
            need(r.align.get("ref") is None or r.align["ref"] in flows, f"align of {r.id}: unknown target")
            need(r.align.get("ref") != r.id, f"align of {r.id}: on itself")
            need(r.align.get("from") is None or r.align["from"] in p.blocks, f"align of {r.id}: unknown block")
    single = {}
    pairs = set()
    for w in p.wires:
        need(w.src[0] in p.blocks and w.dst[0] in p.blocks, f"wire {w.id} dangling")
        sb, db = p.blocks[w.src[0]], p.blocks[w.dst[0]]
        need(w.src[1] in out_ports(sb), f"wire {w.id}: no out port {w.src[1]}")
        need(w.dst[1] in in_ports(db), f"wire {w.id}: no in port {w.dst[1]}")
        need(db.kind == "post" or ix.pos.get(db.id) == 0, f"wire {w.id} into a non-head")
        need((tuple(w.src), tuple(w.dst)) not in pairs, f"duplicate wire {w.id}")
        pairs.add((tuple(w.src), tuple(w.dst)))
        if db.kind != "post":
            single.setdefault(tuple(w.dst), []).append(w.active)
        else:
            need(w.active, f"inactive wire into Post {w.id}")
    for port, acts in single.items():
        need(sum(acts) == 1, f"port {port}: {sum(acts)} active wires")
    need(not has_cycle(p, ix), "cycle")
    for t in p.taps:
        need(t.block in ix.row_of and p.blocks[t.block].kind not in ("session", "calibration"),
             f"tap {t.id} on {t.block}: not a chain block")
        need(t.name and t.name.strip(), f"tap {t.id} unnamed")
    wids = [f.wid for f in list(p.rows) + list(p.taps) if f.wid is not None]
    need(len(wids) == len(set(wids)), "two workflows share an id")
    need(not wids or max(wids) < p.next_wid, "next_wid behind")
    ids = [int(x[1:]) for x in list(p.blocks) + [r.id for r in p.rows] + [w.id for w in p.wires] +
           [t.id for t in p.taps]]
    need(not ids or max(ids) < p.next_id, "next_id behind")
    need(len(ids) == len(set(ids)), "id reused")


def edges(p, ix=None, active_only=False):
    """Data edges (src block, dst block): nights, chain, fork, port wires."""
    ix = ix or Index(p)
    out = []
    for r in p.rows:
        for s, c in r.nights:
            if c is not None:
                out.append((s, c))
            if r.blocks:
                out.append((c if c is not None else s, r.blocks[0]))
        out += list(zip(r.blocks, r.blocks[1:]))
        if r.fork and r.blocks:
            out.append((r.fork[1], r.blocks[0]))
    out += [(w.src[0], w.dst[0]) for w in p.wires if w.active or not active_only]
    return out


def topo(p, ix=None, active_only=False):
    """Blocks in a deterministic topological order (None if a cycle)."""
    ix = ix or Index(p)
    succ = {b: [] for b in p.blocks}
    deg = {b: 0 for b in p.blocks}
    for a, b in edges(p, ix, active_only):
        succ[a].append(b)
        deg[b] += 1
    ready = [b for b in p.blocks if deg[b] == 0]
    order = []
    while ready:
        b = ready.pop(0)
        order.append(b)
        for c in succ[b]:
            deg[c] -= 1
            if deg[c] == 0:
                ready.append(c)
    return order if len(order) == len(p.blocks) else None


def has_cycle(p, ix=None):
    return topo(p, ix) is None


def dumps_model(p):
    return json.dumps(to_dict(p), ensure_ascii=False, indent=1)
