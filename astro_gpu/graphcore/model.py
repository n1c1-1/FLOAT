"""Stored model (§1): blocks, rows, port wires, ids. Stored = drawn: nothing here is inferred."""
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


@dataclass
class Row:
    id: str
    name: str
    blocks: list = field(default_factory=list)   # chain, left to right (Session/Calibration excluded)
    nights: list = field(default_factory=list)   # [[session id, calibration id]], first = top
    fork: object = None                          # [row id, block id] for a variant


@dataclass
class Wire:
    id: str
    src: list      # [block id, out port]
    dst: list      # [block id, in port]
    active: bool = True


@dataclass
class Project:
    name: str = "Project"
    target: str = ""
    rows: list = field(default_factory=list)
    blocks: dict = field(default_factory=dict)
    wires: list = field(default_factory=list)
    reference: object = None
    common: dict = field(default_factory=lambda: {"files": {}, "settings": {}})
    next_id: int = 1


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
                self.row_of[s] = self.row_of[c] = r.id
                self.night_of[s] = self.night_of[c] = s
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
    return "image" if k == "image" else "join"


# ---------------------------------------------------------------------------------------------------- save / load
def to_dict(p):
    """The dataclasses as plain dicts, fields in declaration order (same result as dataclasses.asdict)."""
    return {"name": p.name, "target": p.target,
            "rows": [{"id": r.id, "name": r.name, "blocks": list(r.blocks), "nights": [list(n) for n in r.nights],
                      "fork": list(r.fork) if r.fork else r.fork} for r in p.rows],
            "blocks": {k: {"id": b.id, "kind": b.kind, "settings": dict(b.settings), "off": b.off, "pause": b.pause,
                           "name": b.name, "files": {f: list(v) for f, v in b.files.items()}
                           if b.files is not None else None} for k, b in p.blocks.items()},
            "wires": [{"id": w.id, "src": list(w.src), "dst": list(w.dst), "active": w.active} for w in p.wires],
            "reference": p.reference,
            "common": {"files": {k: list(v) for k, v in p.common["files"].items()},
                       "settings": {k: dict(v) for k, v in p.common["settings"].items()}},
            "next_id": p.next_id}


def from_dict(d):
    p = Project(**{k: v for k, v in d.items() if k not in ("rows", "blocks", "wires")})
    p.rows = [Row(**r) for r in d["rows"]]
    p.blocks = {k: Block(**b) for k, b in d["blocks"].items()}
    p.wires = [Wire(**w) for w in d["wires"]]
    return p


def model_json(p):
    return json.dumps(to_dict(p), ensure_ascii=False, sort_keys=False)


def clone(p):
    """Fast deep copy of a project."""
    q = Project(p.name, p.target, reference=p.reference, next_id=p.next_id,
                common={"files": {k: list(v) for k, v in p.common["files"].items()},
                        "settings": {k: dict(v) for k, v in p.common["settings"].items()}})
    q.rows = [Row(r.id, r.name, list(r.blocks), [list(n) for n in r.nights], list(r.fork) if r.fork else None)
              for r in p.rows]
    q.blocks = {k: Block(b.id, b.kind, dict(b.settings), b.off, b.pause, b.name,
                         {f: list(v) for f, v in b.files.items()} if b.files is not None else None)
                for k, b in p.blocks.items()}
    q.wires = [Wire(w.id, list(w.src), list(w.dst), w.active) for w in p.wires]
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
        for b in r.blocks + [x for n in r.nights for x in n]:
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
            need(p.blocks[s].kind == "session" and p.blocks[c].kind == "calibration", f"night pair of {r.id}")
            need(p.blocks[s].name, f"night {s} unnamed")
        for i, k in enumerate(kinds):
            need(k not in ("session", "calibration", "post"), f"{k} in chain of {r.id}")
            need(i == 0 or k not in JOIN_KINDS + ("image",), f"{k} not at head of {r.id}")
        if r.fork:
            need(not r.nights and r.blocks, f"variant {r.id} shape")
            need(r.fork[0] in ix.row and r.fork[1] in ix.row[r.fork[0]].blocks, f"fork of {r.id}")
            need(kinds[0] not in JOIN_KINDS + ("image",), f"variant {r.id} head")
            need(ix.root(r.id) != r.id and not ix.row[ix.root(r.id)].fork, f"fork loop at {r.id}")
        elif r.nights:
            need(kinds[:1] == ["registration"] and "stacking" in kinds, f"frames row {r.id} lacks a stack")
        else:
            need(kinds and kinds[0] in JOIN_KINDS + ("image",), f"row {r.id} has no head")
    frames = [r.id for r in p.rows if r.nights]
    need((p.reference is None) == (not frames), "reference None iff no frames row")
    need(p.reference is None or p.reference in frames, "reference not a frames row")
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
    ids = [int(x[1:]) for x in list(p.blocks) + [r.id for r in p.rows] + [w.id for w in p.wires]]
    need(not ids or max(ids) < p.next_id, "next_id behind")
    need(len(ids) == len(set(ids)), "id reused")


def edges(p, ix=None, active_only=False):
    """Data edges (src block, dst block): nights, chain, fork, port wires."""
    ix = ix or Index(p)
    out = []
    for r in p.rows:
        for s, c in r.nights:
            out.append((s, c))
            if r.blocks:
                out.append((c, r.blocks[0]))
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
