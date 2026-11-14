"""The gallery's templates as operations of the core (INTEGRATION §12 step 3): what the current builders make
(client/graph.GraphPage.make_from_template, make_filters; client/model.ProjectModel.add_sho_rgb,
add_filters_combination, add_dual_x2), as ONE list of steps for ``operations.apply_many`` (one Undo step), and the
Post actions that follow it (Post's layers are Post's: the caller runs them with its own Post functions once the
graph is compiled).

``build(g, t, name, opts, tr=None) -> Plan``: ``t`` the gallery's template (its dict), ``name`` the name the gallery
gives (the palette's for a combination), ``opts`` the views' options (gallery._view_opts). ``Plan.steps`` is
for apply_many; ``Plan.post(created_rows)`` gives the Post actions with the rows' ids:
  ("add_layer", row, part, key)         a layer of a workflow's image, its id kept under ``key``
  ("side", key, side, row, image)       one side of a Combine layers layer
  ("remove_part", row, part)            a workflow's image taken out of Post, in every version
  ("removed", [(row, part)])            images listed as taken out (Post shows nothing else)
Nothing is read from file or folder names: the filters are those the sessions store."""
from __future__ import annotations

from ..core import workflow as wfl
from .model import COMBO_PORTS
from .msg import name_of

#: the views' options as the engine's (client/model.to_engine, Qt-free copy)
DN_ENGINE = {"generique": "pretrained", "affine": "adapted"}
DECONV_ENGINE = {"psf": "rl", "neural": "neural"}
GRADIENT_ENGINE = {"simple": "polynomial", "rbf": "rbf", "auto": "automatic"}
#: the gallery's block keys as the engine's blocks (client/blocks.KEY_ENGINE_BLOCK, the ones a template names)
KEY_ENGINE_BLOCK = {"empilement": "stacking", "alignement": "registration", "astrometrie": "astrometry",
                    "fond": "gradient", "couleur": "color", "debruitage": "denoise", "etoiles": "stars"}
#: a combination type's ports (graph names) by the engine's roles (stages/combine.py)
PORT_OF_ROLE = {"SII": "S II", "Ha": "Hα", "OIII": "O III", "L": "L", "R": "R", "G": "G", "B": "B"}
#: a combination type by its title (stages/combine.COMBINATIONS)
KIND_OF_TITLE = {"S-H-O": "sho", "H-O": "ho", "RGB": "rgb", "LRGB": "lrgb"}


def to_engine(o: dict) -> dict:
    out = {"denoise": DN_ENGINE.get(o.get("dn"), "pretrained") if o.get("debruitage") else None,
           "stars": "neural" if o.get("etoiles") == "retirer" else None,
           "gradient": GRADIENT_ENGINE.get(o.get("fond"), "polynomial"),
           "deconvolution": DECONV_ENGINE.get(o.get("deconv_m"), "rl") if o.get("deconv") else None}
    if o.get("cut"):
        out["cut"] = list(o["cut"])
    return out


def source_of(t: dict) -> str:
    """gallery.template_source."""
    if t.get("stretched"):
        return "stretched"
    if t.get("source") in wfl.SOURCES:
        return t["source"]
    return "image" if t.get("source") == "image" else "frames"


def template_nodes(t: dict, opts: dict, mono: bool = False) -> list[dict]:
    """gallery.template_blocks: the engine nodes a template's workflow has (none cut; a filter's has no Color)."""
    if t.get("blocks") is not None:
        return [dict(n) for n in t["blocks"]]
    end = KEY_ENGINE_BLOCK.get(t.get("end"), t.get("end")) if t.get("end") else None
    g = wfl.from_options(to_engine(dict(opts or {})), end, source_of(t))
    drop = set(t.get("drop") or ())
    if mono or t.get("kind") in ("filters", "fusion"):
        drop.add("color")
    return [n for n in g if n["block"] not in drop and not n["off"]]


def options_nodes(opts: dict, colour: bool) -> list[dict]:
    """A frames workflow's nodes from options (model.mono_blocks; with Color for a colour one)."""
    return [dict(n) for n in wfl.from_options(to_engine(dict(opts)), None, "frames")
            if not n["off"] and (colour or n["block"] != "color")]


def chain_of(nodes: list[dict]) -> list[dict]:
    """The blocks a row holds for these nodes (its Calibration is its nights', its Output is compiled)."""
    return [{"kind": n["block"], "method": n.get("method")} for n in nodes
            if n["block"] not in ("calibration", "output", "align_on")]


class Ref:
    """A row or block made by an earlier step: the k-th row (or block) the plan makes."""

    def __init__(self, what, k):
        self.what, self.k = what, k


def _resolve(x, rows, blocks):
    if isinstance(x, Ref):
        return (rows if x.what == "row" else blocks)[x.k]
    if isinstance(x, dict):
        return {k: _resolve(v, rows, blocks) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(_resolve(v, rows, blocks) for v in x)
    return x


class Plan:
    """Steps for apply_many whose arguments may name what earlier steps made (Ref), and the Post actions."""

    def __init__(self, g):
        self.g = g
        self._steps = []
        self.rows = 0          # rows made so far
        self.blocks = 0        # blocks made so far
        self.actions = []      # Post actions (rows as Ref)
        self.names = {r.name for r in g.rows} | {t.name for t in g.taps}
        self.sessions = {n["name"] for n in g.nights}
        self.sids = [n["sid"] for n in g.nights]

    def op(self, _op, made_rows=0, made_blocks=0, **kw):
        self._steps.append((_op, kw))
        rows = [Ref("row", self.rows + i) for i in range(made_rows)]
        blocks = [Ref("block", self.blocks + i) for i in range(made_blocks)]
        self.rows += made_rows
        self.blocks += made_blocks
        return rows, blocks

    @property
    def steps(self):
        out = []
        for name, kw in self._steps:
            def args(results, kw=kw):
                rows = [x for r in results for x in (r.created or {}).get("rows", [])]
                blocks = [x for r in results for x in (r.created or {}).get("blocks", [])]
                return _resolve(kw, rows, blocks)
            out.append((name, args))
        return out

    def post(self, created_rows):
        """The Post actions with the rows' ids (created_rows: Result.created["rows"] of the whole plan)."""
        return [_resolve(a, list(created_rows), []) for a in self.actions]

    # -- names, as the current builders give them
    def unique_name(self, base, numbered=False):
        """model.unique_name over the workflows' names."""
        names = self.names
        series = any(x.startswith(base + " ") and x[len(base) + 1:].isdigit() for x in names)
        if base not in names and not numbered and not series:
            return base
        n = 1 if numbered or base not in names else 2
        while f"{base} {n}" in names:
            n += 1
        return f"{base} {n}"

    def new_session(self, flt="", name=""):
        """model.add_filter_session: a session of the project, no stack drawing it yet; its id."""
        sid = max(self.sids + [b.sid for b in self.g.blocks.values() if b.sid is not None] + [0]) + 1
        if flt or name:
            base, out, k = flt or name, flt or name, 2
            while out in self.sessions:
                out, k = f"{base} ({k})", k + 1
        else:
            n = sid
            while name_of("Session {n}", n=n) in self.sessions:
                n += 1
            out = name_of("Session {n}", n=n)
        self.sessions.add(out)
        self.sids.append(sid)
        self.op("new_night", name=out, filter=flt)
        return sid

    def stack(self, name, sids, nodes, all_nights=False):
        """A row of these sessions with these nodes' blocks (no Align on): (its Ref, its blocks' Refs by kind)."""
        chain = chain_of(nodes)
        self.names.add(name)
        (row,), blocks = self.op("add_stack", made_rows=1, made_blocks=2 * len(sids) + len(chain), name=name,
                                 nights=list(sids), blocks=chain, align=None, all_nights=all_nights)
        mine = blocks[2 * len(sids):]
        return row, {c["kind"]: b for c, b in zip(chain, mine)}


def _filters(g):
    """{filter: its sessions' ids} (model.filters)."""
    out = {}
    for n in g.nights:
        if n.get("filter"):
            out.setdefault(n["filter"], []).append(n["sid"])
    return out


def _filter_row(plan, g, flt, opts, made):
    """model.filter_workflow + the template's graph: the row stacking a filter's sessions (one made when there is
    none). Its settings: none (the filter workflows of the current builder take no white balance)."""
    sids = _filters(g).get(flt) or made.get(flt)
    name = plan.unique_name(flt)
    return plan.stack(name, sids, options_nodes(opts, colour=False))


def _align_without_astrometry(plan, row, blocks, ref, ref_blocks):
    """model.align_without_astrometry: an Align on after the Stacking, from the reference's Astrometry; no
    Astrometry of its own."""
    plan.op("set_align", row=row, ref=ref, block=ref_blocks.get("astrometry"))
    if "astrometry" in blocks:
        plan.op("remove_block", block=blocks["astrometry"])


def build(g, t: dict, name: str, opts: dict, tr=None) -> Plan:
    """The plan of template t on graph g (see the module's text); ``tr`` writes the default names (Session 1, Image,
    RGB, Stars) in the client's language."""
    from .msg import TR
    token = TR.set(tr)
    try:
        return _build(g, t, name, opts)
    finally:
        TR.reset(token)


def _build(g, t, name, opts):
    plan = Plan(g)
    src = source_of(t)
    if t.get("copy") is not None:
        raise ValueError("a copy of a workflow is the duplicate gesture, not a template")
    if t.get("kind") in ("filters", "fusion"):
        _build_filters(plan, g, t, name, opts)
    elif src in ("image", "stretched"):
        n = int(t.get("images") or 1) if t.get("plain") else 1
        base = name_of("Image")
        default = n > 1 or (t["name"] == "Image" and name == plan.unique_name(base))
        images = [r for r in g.rows if r.blocks and g.blocks[r.blocks[0]].kind == "image" and not r.fork]
        for _i in range(n):
            if default:
                nm = _image_name(plan, g, base, images)
            else:
                nm = plan.unique_name(name)
            plan.names.add(nm)
            (row,), (blk,) = plan.op("add_row", made_rows=1, made_blocks=1, kind="image", name=nm)
            images.append(None)
            if src == "stretched":
                plan.op("set_setting", block=blk, key="Stretched image", value=True)
    else:
        sids = [n_["sid"] for n_ in g.nights]
        if not sids:
            sids = [plan.new_session()]
        # model.add_path: the camera white balance in the Stacking when the project's first workflow has it (or
        # none yet)
        row, blocks = plan.stack(name, sids, template_nodes(t, opts), all_nights=True)
        if _white_balance(g):
            plan.op("set_param", block=blocks["stacking"], stage="stack", name="white_balance", value=True)
    return plan


def _image_name(plan, g, base, images):
    """model.image_name: one Image alone has no number; with others, one still named ``base`` takes the first free
    number, and the new one the next."""
    if not images:
        return plan.unique_name(base)
    if base in plan.names:
        old = next((r for r in g.rows if r.name == base), None)
        n = 1
        while f"{base} {n}" in plan.names:
            n += 1
        if old is not None:
            plan.op("rename_row", row=old.id, name=f"{base} {n}")
        else:
            plan.op("rename_row", row=Ref("row", plan.rows - 1), name=f"{base} {n}")
        plan.names.discard(base)
        plan.names.add(f"{base} {n}")
    n = 1
    while f"{base} {n}" in plan.names:
        n += 1
    return f"{base} {n}"


def camera_white_balance(g) -> bool:
    """model.add_path's rule (issue #6) for a new stack: the camera white balance in its Stacking when the project's
    first stack has it, or when the project has no stack yet (on in a new project), so that the stacks of a project
    share their setting. A Session alone (no Stacking placed yet) is no stack: it decides nothing."""
    stacks = [(r.wid is None, r.wid or 0, i, r, b) for i, r in enumerate(g.rows) for b in r.blocks
              if b in g.blocks and g.blocks[b].kind == "stacking"]
    if not stacks:
        return True
    *_, r, b = min(stacks, key=lambda x: x[:3])          # the first workflow's (compile._settings: block, then leftovers)
    v = (g.blocks[b].params.get("stack") or {}).get("white_balance", False)
    v = ((r.extra or {}).get("stack") or {}).get("white_balance", v)
    return bool(v) and ("stack", "white_balance") not in {tuple(x) for x in r.unset or []}


_white_balance = camera_white_balance


def _build_filters(plan, g, t, name, opts):
    stars = opts.get("etoiles") == "retirer"
    kind = KIND_OF_TITLE.get(t.get("palette"), "sho")
    if t.get("dual_x2"):
        return _dual_x2(plan, g, name, opts)
    have = _filters(g)
    if t.get("rgb_stars"):
        filters = list(t["filters"])
    else:
        filters = (sorted(have) or list(t["filters"])) if t["kind"] == "filters" else list(t["filters"])
    made = {}
    for f in filters:
        if f not in have:
            made[f] = [plan.new_session(f)]
    plan.op("set_calibrate_together", on=False)          # model.filter_workflow: each night on its own
    rows = [(f,) + _filter_row(plan, g, f, opts, made) for f in filters]
    ref, ref_blocks = rows[0][1], rows[0][2]
    if t.get("rgb_stars"):
        sid = plan.new_session("", name_of("RGB"))
        rgb_name = plan.unique_name(name_of("RGB"))
        rgb, rgb_blocks = plan.stack(rgb_name, [sid], options_nodes(opts, colour=True))
        for _f, r, b in rows[1:]:
            _align_without_astrometry(plan, r, b, ref, ref_blocks)
        _align_without_astrometry(plan, rgb, rgb_blocks, ref, ref_blocks)
        fus = _combination(plan, name, kind, [(f, r, b) for f, r, b in rows], "starless")
        _stars_back(plan, fus, rgb, [r for _f, r, _b in rows], [(rgb, "combined")])
        return
    for _f, r, b in rows[1:]:
        _align_without_astrometry(plan, r, b, ref, ref_blocks)
    if t["kind"] == "filters":
        return
    if not stars:
        _combination(plan, name, kind, rows, "output")
        return
    star_fus = _combination(plan, f"{name} · " + name_of("Stars"), kind, rows, "stars")
    fus = _combination(plan, name, kind, rows, "starless")
    lay = ("add_layer", fus, "combined", "lay")
    plan.actions += [lay, ("side", "lay", "starless", fus, "output"), ("side", "lay", "stars", star_fus, "output"),
                     ("remove_part", fus, None), ("remove_part", star_fus, None),
                     ("removed", [(fus, None), (star_fus, None)] + [(r, "combined") for _f, r, _b in rows])]


def _combination(plan, name, kind, rows, image, channels=None):
    """model.add_fusion: a combination of type kind reading the rows' images (channel L unless given)."""
    nm = plan.unique_name(name)
    plan.names.add(nm)
    (fus,), (j,) = plan.op("add_combination", made_rows=1, made_blocks=1, type=kind, name=nm)
    ports = COMBO_PORTS[kind]
    for i, (f, r, b) in enumerate(rows):
        port = PORT_OF_ROLE.get(f, f) if PORT_OF_ROLE.get(f, f) in ports else ports[i]
        if image in ("starless", "stars"):
            src, sport = b["stars"], ("Starless" if image == "starless" else "Stars only")
        else:
            last = [k for k in b if k != "stars"][-1]
            src, sport = b[last], "image"
        plan.op("wire", src=src, src_port=sport, dst=j, dst_port=port,
                channel=(channels or {}).get(f, "L"))
    return fus


def _stars_back(plan, fus, stars_row, filter_rows, extra_removed):
    """The Combine layers layer of the combination's Output with a colour row's stars, and Post shows nothing
    else (model.add_sho_rgb, add_dual_x2)."""
    plan.actions += [("add_layer", fus, "combined", "lay"), ("side", "lay", "starless", fus, "output"),
                     ("side", "lay", "stars", stars_row, "starless"), ("remove_part", fus, None),
                     ("remove_part", stars_row, "combined"),
                     ("removed", [(fus, None)] + extra_removed + [(r, "combined") for r in filter_rows])]


def _dual_x2(plan, g, name, opts):
    """model.add_dual_x2: sessions O III / H-alpha (A), O III / S II (B), Stars (C); rows S (B), H (A), O III (A and B
    stacked together), Stars (C, a colour row); S the reference; S-H-O reading S's red, H's red, O III's green and
    blue, all starless; the Stars row's stars back on it in Post."""
    sa = plan.new_session("", "O III / H-alpha")
    sb = plan.new_session("", "O III / S II")
    sc = plan.new_session("", name_of("Stars"))
    plan.op("set_calibrate_together", on=False)
    wb = _white_balance(g)
    rows = {}
    for key, label, sids, colour in (("S", "S", [sb], False), ("H", "H", [sa], False), ("O", "O III", [sa, sb], False),
                                     ("RGB", name_of("Stars"), [sc], True)):
        nm = plan.unique_name(label)
        rows[key] = plan.stack(nm, sids, options_nodes(opts, colour))
        if wb:
            plan.op("set_param", block=rows[key][1]["stacking"], stage="stack", name="white_balance", value=True)
    for key in ("H", "O", "RGB"):
        _align_without_astrometry(plan, rows[key][0], rows[key][1], rows["S"][0], rows["S"][1])
    fus = _combination(plan, name, "sho", [("SII", rows["S"][0], rows["S"][1]), ("Ha", rows["H"][0], rows["H"][1]),
                                           ("OIII", rows["O"][0], rows["O"][1])], "starless",
                       {"SII": "R", "Ha": "R", "OIII": "GB"})
    _stars_back(plan, fus, rows["RGB"][0], [rows[k][0] for k in ("S", "H", "O")], [(rows["RGB"][0], "combined")])
