"""An existing project's definition (format 2) as a stored graph (INTEGRATION §5).

``convert(p) -> Graph`` reads the definition the current code loads and gives the graph that ``compile.compile_into``
turns back into the same definition: every value a key reads is kept as it is (node names, workflow ids and their
order, the settings of stages a chain does not have, the settings of switched-off blocks, stored values equal to the
defaults), and what the graph's blocks do not give is kept per workflow (``Row.extra`` / ``Row.unset``).

The rows are those today's graph page draws: a tree of the workflows on the CLIENT's block key (``client_paths``, a
Qt-free copy of ``client/model.ProjectModel.block_key`` and ``display_keys`` with the cut and greyed branch rule of
``client/blocks.Graph``), where a workflow sharing a prefix with an earlier one is a variant row forking at the last
block they share, and a workflow that adds only its Output to an earlier one is a tap at its last block. The tree is
refined by the nodes' stored attributes (name, method, off, mark, image file), so that a block holds one value of each.

The block keys come from the engine (``api.blocks``: the keys of each block's stages), as the client reads them;
``block_keys`` replaces that reading (tests, or a project whose keys are known)."""
from __future__ import annotations

import copy
import json

from ..core import workflow as wfl
from . import model as gm

#: the views' keys of the engine's blocks (client/blocks.ENGINE_BLOCK)
ENGINE_KEY = {"fusion": "fusion", "align_on": "align_on", "image": "image", "calibration": "calibration",
              "registration": "alignement", "selection": "selection", "crop": "crop", "stacking": "empilement",
              "astrometry": "astrometrie", "color": "couleur", "denoise": "debruitage", "deconv": "deconv",
              "synth": "synth", "stars": "etoiles", "gradient": "fond", "output": "post"}
KEY_ENGINE = {v: k for k, v in ENGINE_KEY.items()}
#: the order display_keys puts the cut and greyed blocks of a workflow made before the free graph in
WANT = ["image", "calibration", "alignement", "selection", "crop", "empilement", "astrometrie", "fond", "couleur",
        "debruitage", "etoiles", "synth", "post"]
#: a combination's roles (stages/combine.py) and the graph's port names
PORT_OF_ROLE = {"SII": "S II", "Ha": "Hα", "OIII": "O III", "L": "L", "R": "R", "G": "G", "B": "B"}
ROLE_OF_PORT = {v: k for k, v in PORT_OF_ROLE.items()}
#: the settings of the stack stage Selection owns (it has no stage of its own)
SELECTION_KEYS = ("min_transparency", "max_fwhm_ratio", "selection", "selection_transparency")


def set_values(stored: dict, stage: str) -> dict:
    """A stage's stored settings as the engine's keys read them: a value stored at the stage's static default counts as
    not stored, but on a field with an automatic rule (stored there, it means "not automatic"). The client's block and
    review keys read them so: the same whether the file writes every default or only the user's changes."""
    from ..core.store import key_rules
    rules = key_rules()
    try:
        defaults, auto = rules.defaults(stage), rules.auto_fields(stage)
    except (KeyError, ValueError):
        return dict(stored)
    return {k: v for k, v in stored.items() if k in auto or k not in defaults or v != defaults[k]}
#: the engine blocks that are a graph block of the same kind
CHAIN_KINDS = ("registration", "selection", "crop", "stacking", "astrometry", "gradient", "deconv", "denoise",
               "color", "stars", "synth")


def view_key(name: str) -> str:
    b, sep, n = name.partition("~")
    return ENGINE_KEY.get(b, b) + (f"~{n}" if sep else "")


def node_key(n: dict) -> str:
    return view_key(n["name"] if n["block"] in wfl.IMAGE_OPS else n["block"])


def _js(x) -> str:
    return json.dumps(x, sort_keys=True, ensure_ascii=False, default=str)


# ---------------------------------------------------------------------------------------------- the client's keys
class Client:
    """What the client's model answers about a definition, without Qt (client/model.py, client/blocks.py)."""

    def __init__(self, p, block_keys=None):
        self.p = p
        if block_keys is None:
            from .. import api

            def block_keys(wid):
                return [(b.block, tuple(b.key)) for b in api.blocks(p, wid)]
        self.blocks = {}
        for w in p.workflows:
            try:
                self.blocks[w.id] = list(block_keys(w.id))
            except (ValueError, KeyError):
                self.blocks[w.id] = []
        self.sep = bool(p.separate_sessions())

    def graph(self, w) -> list[dict]:
        try:
            return [n for n in wfl.graph_of(w) if n["block"] != "output"]
        except (KeyError, ValueError):
            return []

    def is_mono(self, w) -> bool:
        try:
            return w.source is None and self.p.is_mono(w.id)
        except KeyError:
            return False

    def greyed(self, w) -> dict:
        if self.is_mono(w) and any(n["block"] == "color" for n in self.graph(w)):
            return {"color": wfl.MONO_NO_COLOUR}
        return {}

    def block_keys(self, w) -> list[str]:
        return [view_key(b) for b, _k in self.blocks[w.id] if wfl.base(b) in ENGINE_KEY and "@" not in b]

    @staticmethod
    def stretched(w) -> bool:
        return isinstance(w.source, dict) and bool(w.source.get("stretched"))

    def display_keys(self, w) -> list[str]:
        src = w.source
        if isinstance(src, dict) and "fusion" in src:
            try:
                stars = any(n["block"] == "stars" for n in wfl.graph_of(w))
            except (KeyError, ValueError):
                stars = False
            return ["fusion"] + (["etoiles"] if stars else [])
        if w.blocks is not None:
            try:
                g = wfl.graph_of(w)
            except (KeyError, ValueError):
                g = []
            keys = ["image"] if src is not None else []
            sep = self.sep and src is None
            keys += [node_key(n) for n in g if not (sep and n["block"] == "calibration")]
            if "etoiles" in keys and "post" in keys:
                keys.remove("post")
                keys.insert(keys.index("etoiles") + 1, "post")
            if self.stretched(w) and "post" not in keys:
                keys.append("post")
            return keys
        keys = self.block_keys(w)
        if self.stretched(w) and "post" not in keys:
            keys = keys + ["post"]
        if self.sep and src is None:
            keys = [k for k in keys if k != "calibration"]
        have = set(keys)
        extra = {ENGINE_KEY.get(n["block"], n["block"]) for n in self.graph(w) if n["off"]} | \
            {ENGINE_KEY.get(b, b) for b in self.greyed(w)}
        if src is not None:
            extra -= {"calibration", "alignement", "selection", "empilement"}
            if self.stretched(w):
                extra -= {"astrometrie", "fond", "couleur", "debruitage"}
        elif self.sep:
            extra.discard("calibration")
        if w.end:
            end = "empilement" if w.end == "stacking" else None
            extra = {k for k in extra if WANT.index(k) <= WANT.index(end)}
        out = list(keys)
        for k in sorted(extra - have, key=WANT.index):
            before = [i for i, x in enumerate(out) if x in WANT and WANT.index(x) < WANT.index(k)]
            out.insert(before[-1] + 1 if before else 0, k)
        return out

    def is_cut(self, w, k) -> bool:
        return any(n["off"] for n in self.graph(w) if node_key(n) == k)

    def grey(self, w, k) -> str:
        return self.greyed(w).get(KEY_ENGINE.get(k, k), "")

    @staticmethod
    def forks(w) -> tuple:
        return tuple(sorted((int(n), e["fork"]) for n, e in (w.calibration or {}).items()))

    def node_fork(self, w, k):
        return next((n.get("fork") for n in self.graph(w) if node_key(n) == k), None)

    def sessions_of(self, w) -> list[int]:
        return [s.id for s in self.p.sessions_of(w.id)] if w.source is None else []

    def block_key(self, w, k):
        own = (self.node_fork(w, k), self.forks(w))
        st = set_values(w.params.get("stack") or {}, "stack")
        if k == "crop":
            zone = st.get("zone")
            return ("crop", tuple(zone) if zone else None) + own
        if k == "selection":
            base = ("selection", st.get("min_transparency"), st.get("max_fwhm_ratio"),
                    st.get("selection", "min_weight"), st.get("selection_transparency", "on"))
            return base + own if own != (None, ()) else base
        ref = next((key for b, key in self.blocks[w.id] if view_key(b) == k), None)
        src = w.source
        mine = w.id if (isinstance(src, dict) and "fusion" in src) or (
            isinstance(src, dict) and "path" in src and not src.get("path")) else None
        base = (k, ref if ref is not None else (), tuple(sorted(self.sessions_of(w))), mine)
        return base + own if own != (None, ()) else base

    def path(self, w) -> list[dict]:
        """The blocks the client's tree draws for a workflow, in order: {key, bk (its tree key), cut, grey}."""
        out, parent = [], None
        for k in self.display_keys(w):
            cut, grey = self.is_cut(w, k), self.grey(w, k)
            bk = ("cut" if cut else "grey", k, self.block_key(w, k), parent) if (cut or grey) else self.block_key(w, k)
            out.append({"key": k, "bk": bk, "cut": cut, "grey": bool(grey)})
            parent = bk
        return out


# ---------------------------------------------------------------------------------------------- converting
class _TNode:
    __slots__ = ("key", "children", "first", "elem")

    def __init__(self, key, first, elem):
        self.key, self.children, self.first, self.elem = key, {}, first, elem


def _node_dict(n: dict) -> dict:
    return {k: n[k] for k in sorted(n)}


def _legacy_nodes(p, w, cl: Client) -> tuple[list[dict], bool]:
    """INTEGRATION 5.9: a workflow made before the free graph, its blocks: those of its options whose stages its chain
    has (a mono workflow drops Color), a Crop when a zone is stored; frozen when their chain is not its chain."""
    src = wfl.source_of(w)
    nodes = wfl.from_options(w.options, w.end, src)
    try:
        have = set(p.chain(w))
    except (ValueError, KeyError):
        have = set()
    keep = []
    for n in nodes:
        stages = set(wfl.BLOCKS.get(n["block"], ()))
        if n["block"] in ("calibration", "registration", "selection", "output") or stages & have or \
                n["block"] in wfl.IMAGE_OPS and n["name"] in have:
            keep.append(n)
    if (w.params.get("stack") or {}).get("zone") and src == "frames":
        i = next((i for i, n in enumerate(keep) if n["block"] == "stacking"), len(keep))
        keep.insert(i, wfl.node("crop"))
    try:
        mine = wfl.chains().sequence(wfl.sequence_chain(wfl.with_output(keep, src), src,
                                                        reference=bool(p.reference_image),
                                                        sessions=p.separate_sessions(w),
                                                        combine=wfl.combine_stage(w),
                                                        catalogs=p.catalogs_wanted(w)))
        frozen = mine != p.chain(w)
    except (ValueError, KeyError):
        frozen = True
    return [n for n in keep if n["block"] != "output"], frozen


class Converter:
    def __init__(self, p, block_keys=None):
        self.p = p
        self.cl = Client(p, block_keys)
        self.g = gm.Project(name=p.name, target=getattr(p.target, "name", "") or "")
        self.g.blocks = {}
        gm.new_block(self.g, "post")
        self.log: list[str] = []        # what the graph does not draw as today's client does (and why)
        self.flows: dict[int, object] = {}       # wid -> Row or Tap
        self.chain: dict[int, list[str]] = {}    # wid -> its chain's block ids (heads included, nights excluded)
        self.root: dict[int, str] = {}           # wid -> its root row id
        self.align_of: dict[int, str | None] = {}  # wid -> the row holding the Align on it passes, None
        self.nodes: dict[int, dict] = {}         # wid -> {view key: its node}
        self.rows: dict[str, gm.Row] = {}

    # ------------------------------------------------------------------ the tree
    def _elements(self, w) -> list[dict]:
        """The client's path of a workflow, each element with the stored attributes it compiles from."""
        cl = self.cl
        legacy = w.blocks is None and not (isinstance(w.source, dict) and "fusion" in w.source)
        if legacy:
            nodes, frozen = _legacy_nodes(self.p, w, cl)
            self.legacy_frozen[w.id] = frozen
        else:
            nodes = [n for n in cl.graph(w)]
        self.nodes[w.id] = {node_key(n): n for n in nodes}
        if legacy:      # its drawing is from its options' graph (cl.graph), its blocks from the filtered one
            draw = {node_key(n): n for n in cl.graph(w)}
        else:
            draw = self.nodes[w.id]
        out = []
        calib = next((n for n in cl.graph(w) if n["block"] == "calibration"), None)
        for i, e in enumerate(cl.path(w)):
            k = e["key"]
            if k == "post":
                kind, attrs = "post", None
            elif k == "fusion":
                kind, attrs = "combine", None
            elif k == "image":
                kind, attrs = "image", w.source
            elif k == "calibration":
                kind, attrs = "calib", [calib, w.calibration]
            elif k == "align_on":
                kind, attrs = "align", draw.get(k)
            else:
                n = draw.get(k)
                kind = n["block"] if n is not None else KEY_ENGINE.get(k.split("~")[0], k)
                if legacy and e["grey"] and kind == "color":
                    kind = "dropped"            # 5.9: a mono workflow made before has no Color
                elif legacy and k not in self.nodes[w.id]:
                    kind = "dropped"
                attrs = _node_dict(n) if n is not None else None
            if i == 0 and w.source is None:
                attrs = [attrs, calib, w.calibration]
            out.append({**e, "kind": kind, "attrs": attrs})
        return out

    def convert(self) -> gm.Project:
        p, g = self.p, self.g
        self.legacy_frozen: dict[int, bool] = {}
        g.nights = [{"sid": s.id, "name": s.name, "filter": s.filter,
                     "files": {k: list(getattr(s, k)) for k in gm.FILE_KINDS}, "left_out": copy.deepcopy(s.left_out)}
                    for s in p.sessions]
        g.excluded = [[sid, name] for sid, name in p.excluded]
        g.calibrate_together = p.stack_mode == "together"
        g.next_wid = p.next_workflow_id
        g.order = [w.id for w in p.workflows]
        root = _TNode(None, None, None)
        paths = {}
        for w in p.workflows:
            els = self._elements(w)
            cur, path = root, []
            for e in els:
                key = (e["bk"] if not isinstance(e["bk"], list) else tuple(e["bk"]), _js(e["attrs"]))
                nxt = cur.children.get(key)
                if nxt is None:
                    nxt = cur.children[key] = _TNode(key, w.id, e)
                path.append(nxt)
                cur = nxt
            paths[w.id] = path
        self.block_of: dict[int, str] = {}     # id(tree node) -> graph block id
        for w in p.workflows:
            self._place(w, paths[w.id])
        self._settings()
        self._pauses()
        self._wires()
        self._aligns()
        return g

    # ------------------------------------------------------------------ rows and taps
    def _block(self, e, w) -> str:
        n = self.nodes[w.id].get(e["key"]) if e["kind"] not in ("image", "combine") else None
        if e["kind"] == "image":
            b = gm.new_block(self.g, "image", engine="image", source=copy.deepcopy(w.source))
        elif e["kind"] == "combine":
            t = w.source["fusion"]["type"]
            b = gm.new_block(self.g, "combine", settings={"type": t} if t != "sho" else {},
                             engine=wfl.combine_stage(w))
        else:
            b = gm.new_block(self.g, e["kind"], engine=n["name"], method=n.get("method"), off=bool(n.get("off")),
                             fork=n.get("fork"))
        return b.id

    def _crop(self, e, w) -> list[str]:
        """5.9: before the Stacking of a workflow made before the free graph with a stored zone, its Crop (not drawn
        today: its stack reads the zone all the same)."""
        if e["kind"] != "stacking" or w.blocks is not None or "crop" not in self.nodes[w.id]:
            return []
        return [self._block({"kind": "crop", "key": "crop"}, w)]

    def _new_root(self, w, els) -> gm.Row:
        g, p = self.g, self.p
        r = gm.Row(gm.new_id(g, "r"), w.name)
        if w.source is None:
            sids = list(w.sessions) if w.sessions is not None else [s.id for s in p.sessions]
            calib = next((n for n in self.cl.graph(w) if n["block"] == "calibration"), None)
            r.calib_node = calib is not None
            r.calib_fork = calib.get("fork") if calib else None
            for sid in sids:
                s = p.session(sid)
                sb = gm.new_block(g, "session", name=s.name or f"Session {sid}", sid=sid,
                                  files={k: list(getattr(s, k)) for k in gm.FILE_KINDS if getattr(s, k)})
                own = (w.calibration or {}).get(str(sid))
                cb = gm.new_block(g, "calibration", sid=sid, off=bool(calib and calib.get("off")),
                                  own=copy.deepcopy(own) if own else None)
                r.nights.append([sb.id, cb.id])
        g.rows.append(r)
        self.rows[r.id] = r
        return r

    def _place(self, w, path):
        """The workflow's row (a root row, or a variant forking at the last block it shares) or tap."""
        g = self.g
        els = [t.elem for t in path]
        mine = [i for i, t in enumerate(path) if t.first == w.id and els[i]["kind"] != "post"]
        j = mine[0] if mine else len(path)
        shared = path[:j]
        fork_at, why = None, ""
        last = next((i for i in range(len(shared) - 1, -1, -1) if els[i]["kind"] not in ("post", "dropped")), None)
        if last is not None:
            e = els[last]
            if e["kind"] == "calib":
                why = "it shares only the calibration of another workflow"
            elif e["kind"] == "align":
                fork_at = self.block_of[id(path[last - 1])] if last > 0 and id(path[last - 1]) in self.block_of else None
            else:
                fork_at = self.block_of.get(id(path[last]))
            if fork_at is not None:
                holder = self._row_of_block(fork_at)
                nxt = els[j] if j < len(els) else None
                if holder.align and self.g.blocks[fork_at].kind in ("stacking", "image") and e["kind"] != "align":
                    fork_at, why = None, "it has not the Align on of the stack it shares"
                elif nxt is not None and nxt["kind"] == "align":
                    fork_at, why = None, "its own Align on after a Stacking it shares"
        own = [e for e in els[j:] if e["kind"] not in ("post", "dropped")]
        if fork_at is not None and not own:
            t = gm.Tap(gm.new_id(g, "t"), fork_at, w.name)
            g.taps.append(t)
            self._flow(w, t, self._chain_to(fork_at))
            return
        if fork_at is not None and own and own[0]["kind"] not in ("align", "calib"):
            holder = self._row_of_block(fork_at)
            r = gm.Row(gm.new_id(g, "r"), w.name, fork=[holder.id, fork_at])
            g.rows.append(r)
            self.rows[r.id] = r
            for i in range(j, len(path)):
                if els[i]["kind"] in ("post", "dropped", "align", "calib"):
                    continue
                r.blocks += self._crop(els[i], w)
                b = self._block(els[i], w)
                self.block_of[id(path[i])] = b
                r.blocks.append(b)
            self._flow(w, r, self._chain_to(fork_at) + list(r.blocks))
            return
        if last is not None and not why:
            why = "it cannot fork where it leaves the other workflows"
        if shared and why:
            self.log.append(f"workflow {w.id}: drawn as a row of its own: {why}")
        r = self._new_root(w, els)
        fresh = not shared or why
        for i, e in enumerate(els):
            if e["kind"] in ("post", "dropped", "calib"):
                continue
            if e["kind"] == "align":
                r.align = {"wid": (self.nodes[w.id].get("align_on") or {}).get("reference"), "node": e["attrs"]}
                continue
            r.blocks += self._crop(e, w)
            b = self._block(e, w)
            if fresh or i >= j:
                self.block_of.setdefault(id(path[i]), b)
            r.blocks.append(b)
        self._flow(w, r, list(r.blocks))

    def _row_of_block(self, b) -> gm.Row:
        return next(r for r in self.g.rows if b in r.blocks)

    def _chain_to(self, b) -> list[str]:
        r = self._row_of_block(b)
        ix = gm.Index(self.g)
        eff = ix.effective(r.id)
        return eff[:eff.index(b) + 1]

    def _flow(self, w, f, chain):
        self.flows[w.id] = f
        f.wid = w.id
        f.all_nights = w.sessions is None
        self.chain[w.id] = chain
        ix = gm.Index(self.g)
        rid = f.id if isinstance(f, gm.Row) else ix.row_of[f.block]
        self.root[w.id] = ix.root(rid)
        holder = None
        for b in chain:
            if self.g.blocks[b].kind in ("stacking", "image"):
                hr = self.rows[ix.row_of[b]]
                holder = hr.id if hr.align else None
        self.align_of[w.id] = holder
        root = self.rows[self.root[w.id]]
        if w.sessions is not None:
            mine = [self.g.blocks[s].sid for s, _c in root.nights]
            if list(w.sessions) != mine or w.source is not None:
                f.sessions = list(w.sessions)
        if w.blocks is None and not (isinstance(w.source, dict) and "fusion" in w.source):
            f.legacy = {"options": copy.deepcopy(w.options), "end": w.end, "frozen": self.legacy_frozen.get(w.id)}
        else:
            if w.blocks is None:
                f.legacy = {"options": copy.deepcopy(w.options), "end": w.end, "frozen": False}
            else:
                src = wfl.source_of(w)
                try:
                    o = wfl.options_of(w.blocks, src)
                except ValueError:
                    o = None
                if o != w.options:      # the blocks are the truth: their options are compiled (keys read the blocks)
                    self.log.append(f"workflow {w.id}: its stored options differ from its blocks': {_js(w.options)} "
                                    f"-> {_js(o)}")
                if w.end is not None:
                    f.end = w.end
        f.trial = w.variant_of

    # ------------------------------------------------------------------ settings
    def owners(self, wid) -> list:
        """The holders of a workflow's settings, in its chain's order: (kind, object)."""
        g = self.g
        root = self.rows[self.root[wid]]
        out = [("night", g.blocks[c]) for _s, c in root.nights]
        out.append(("calib", root))
        out += [("block", g.blocks[b]) for b in self.chain[wid]]
        if self.align_of[wid]:
            out.append(("align", self.rows[self.align_of[wid]]))
        out.append(("output", self.flows[wid]))
        return out

    def owner(self, wid, stage, param):
        """Which holder a setting of the workflow belongs to (INTEGRATION §2 rule 3), None: its leftovers."""
        g = self.g
        base, _a, sid = stage.partition("@")
        root = self.rows[self.root[wid]]
        if sid:
            if base in ("inventory", "calibrate"):
                return next((("night", g.blocks[c]) for s, c in root.nights if str(g.blocks[c].sid) == sid), None)
            return None
        if stage in ("inventory", "calibrate"):
            return ("calib", root) if root.nights or root.calib_node else None
        if stage == "render_simple":
            return ("output", self.flows[wid])
        if stage == "align_on":
            return ("align", self.rows[self.align_of[wid]]) if self.align_of[wid] else None
        blocks = [g.blocks[b] for b in self.chain[wid]]

        def first(kind):
            return next((("block", b) for b in blocks if b.kind == kind), None)
        if stage in ("sessions", "register"):
            return first("registration")
        if stage == "stack":
            if param in SELECTION_KEYS:
                return first("selection")
            if param == "zone":
                return first("crop")
            return first("stacking")
        if stage == "image":
            return first("image")
        if stage in wfl.BLOCKS_OF_KIND.values():
            return first("combine")
        if stage in wfl.ASTROMETRY_STAGES:
            return first("astrometry")
        if wfl.base(stage) in wfl.IMAGE_OPS:
            return next((("block", b) for b in blocks if b.engine == stage), None)
        if stage == "star_removal":
            return first("stars")
        if stage == "synth_stars":
            return first("synth")
        return None

    @staticmethod
    def store(kind, obj) -> dict:
        """The {stage: {param: value}} a holder keeps its settings in (an Output: {render_simple: its own})."""
        if kind in ("night", "block"):
            return obj.params
        if kind == "calib":
            return obj.calib
        if kind == "align":
            return obj.align.setdefault("params", {})
        return {"render_simple": obj.output}

    def given(self, wid) -> dict:
        out: dict = {}
        for kind, obj in self.owners(wid):
            for st, kv in self.store(kind, obj).items():
                if kv:
                    out.setdefault(st, {}).update(copy.deepcopy(kv))
        return out

    def _settings(self):
        seen = set()
        for w in self.p.workflows:
            for kind, obj in self.owners(w.id):
                if (kind, id(obj)) in seen:
                    continue
                for st, kv in w.params.items():
                    for k, v in kv.items():
                        o = self.owner(w.id, st, k)
                        if o is not None and o[0] == kind and o[1] is obj:
                            self.store(kind, obj).setdefault(st, {})[k] = copy.deepcopy(v)
                seen.add((kind, id(obj)))
        for w in self.p.workflows:
            got, f = self.given(w.id), self.flows[w.id]
            extra = {}
            for st, kv in w.params.items():
                for k, v in kv.items():
                    if k not in got.get(st, {}) or got[st][k] != v:
                        extra.setdefault(st, {})[k] = copy.deepcopy(v)
            f.extra = extra
            f.unset = [[st, k] for st, kv in got.items() for k in kv if k not in (w.params.get(st) or {})]

    # ------------------------------------------------------------------ breakpoints
    def _pauses(self):
        g = self.g
        through: dict[str, list] = {}
        for w in self.p.workflows:
            for b in self.chain[w.id]:
                through.setdefault(b, []).append(w)
        for b, ws in through.items():
            blk = g.blocks[b]
            blk.pause = bool(blk.engine) and all(blk.engine in w.breakpoints for w in ws)
        for w in self.p.workflows:
            got = sorted({g.blocks[b].engine for b in self.chain[w.id] if g.blocks[b].pause}, key=wfl.block_order)
            if got != list(w.breakpoints):
                self.flows[w.id].pauses = list(w.breakpoints)

    # ------------------------------------------------------------------ combinations
    def _wires(self):
        g = self.g
        for w in self.p.workflows:
            if not (isinstance(w.source, dict) and "fusion" in w.source):
                continue
            head = self.chain[w.id][0]
            for i in w.source["fusion"]["inputs"]:
                src = self.chain.get(i["pid"], [])
                image = i.get("image", "output")
                if image in ("starless", "stars"):
                    b = next((x for x in src if g.blocks[x].kind == "stars"), None)
                    port = "Starless" if image == "starless" else "Stars only"
                else:
                    b = next((x for x in reversed(src) if "image" in gm.out_ports(g.blocks[x])), None)
                    port = "image"
                if b is None:
                    self.log.append(f"workflow {w.id}: its input {i['role']} reads workflow {i['pid']}, which has "
                                    "no block to wire from")
                    continue
                g.wires.append(gm.Wire(gm.new_id(g, "w"), [b, port], [head, PORT_OF_ROLE.get(i["role"], i["role"])],
                                       True, i["pid"], i.get("channel"), i.get("image")))

    # ------------------------------------------------------------------ alignment
    def _aligns(self):
        g = self.g
        ids = {f.wid: f.id for f in list(g.rows) + list(g.taps)}
        for r in g.rows:
            if not r.align:
                continue
            ref, node = r.align.pop("wid"), r.align.pop("node") or {}
            frm, frm_name = None, node.get("reference_block")
            if ref is not None and frm_name is not None and ref in self.chain:
                frm = next((b for b in self.chain[ref] if g.blocks[b].engine == frm_name), None)
            r.align.update({"ref": ids.get(ref) if ref is not None else None, "from": frm,
                            "from_name": frm_name if frm is None else None,
                            "ref_wid": ref if ref is not None and ref not in ids else None,
                            "method": node.get("method"), "off": bool(node.get("off")), "fork": node.get("fork")})
            r.align.setdefault("params", {})


def convert(p, block_keys=None, log: list | None = None) -> gm.Project:
    """The stored graph of a project's definition (a Project2 the current code loaded)."""
    c = Converter(p, block_keys)
    g = c.convert()
    if log is not None:
        log += c.log
    return g
