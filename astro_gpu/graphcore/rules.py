"""Rules as data (§2): what each block takes and gives, its band and column, its needs (reasons as messages), and
the derived texts: titles, summaries, greyed reasons, wire types. Nothing here is stored.

Every text is a message (see msg.py): an English source string or (template, fields). The functions that give
text to a view (title, summary, describe...) come in two forms: `x_msg` gives the message, `x` the written text
(`tr` = the client's translation function, English when None)."""
import dataclasses
from dataclasses import dataclass

from .model import (Index, topo, in_ports, out_ports, combo_type, FILE_KINDS, FRAME_KINDS, JOIN_KINDS, COMBO_PORTS,
                    COMBO_LABEL)
from .msg import M, say, join, typed

ORDER = ["session", "image", "calibration", "registration", "selection", "crop", "stacking", "astrometry",
         "gradient", "deconv", "denoise", "color", "stars", "synth", "combine", "combine_layers"]
LABEL = {"session": "Session", "image": "Image", "calibration": "Calibration", "registration": "Registration",
         "selection": "Selection", "crop": "Crop", "stacking": "Stacking", "astrometry": "Astrometry",
         "gradient": "Gradient", "deconv": "Deconvolution", "denoise": "Denoise", "color": "Color",
         "stars": "Stars/background", "synth": "Synthetic stars", "combine": "Combination",
         "combine_layers": "Combine layers", "post": "Post"}
GROUP = {"session": "SOURCES", "image": "SOURCES", "calibration": "LIGHTS", "registration": "LIGHTS",
         "selection": "LIGHTS", "crop": "LIGHTS", "stacking": "LIGHTS", "astrometry": "IMAGE",
         "gradient": "IMAGE", "deconv": "IMAGE", "denoise": "IMAGE", "color": "IMAGE",
         "stars": "POST", "synth": "POST", "combine": "COMPOSITION",
         "combine_layers": "COMPOSITION"}
# fixed columns of the frame band (logical); image-band blocks start at LIN
COL = {"session": 0, "image": 0, "calibration": 1, "registration": 2, "selection": 3, "crop": 4, "stacking": 5}
LIN = 6
TAKES = {"session": "nothing", "image": "nothing", "calibration": "lights", "registration": "lights",
         "selection": "lights", "crop": "lights", "stacking": "lights", "stars": "image", "synth": "pair",
         "combine": "ports", "combine_layers": "Background + Stars", "post": "layers"}
GIVES = {"session": "lights", "calibration": "lights", "registration": "lights", "selection": "lights",
         "crop": "lights", "stacking": "image", "stars": "Starless + Stars only",
         "synth": "pair", "post": "nothing"}
OFFABLE = ("calibration", "registration", "selection", "crop", "astrometry", "gradient", "deconv",
           "denoise", "color", "synth")
FILE_LABEL = {"lights": "Lights", "darks": "Darks", "flats": "Flats", "bias": "Bias", "flat_darks": "Flat darks"}


def rank(kind):
    return ORDER.index(kind) if kind in ORDER else len(ORDER)


def off_reason(kind):
    """"" when the block can be switched off, else the greyed menu text (a message)."""
    if kind in OFFABLE:
        return ""
    return M("— {takes} → {gives}, essential", takes=TAKES.get(kind, "image"), gives=GIVES.get(kind, "image"))


# ---------------------------------------------------------------------------------------------------- settings
def S(name, kind, default, options=None, expert=False):
    return (name, kind, default, options or [], expert)


SETTINGS = {
    "image": [("Source", [S("Image file", "text", "image.tif"), S("Stretched image", "bool", False)])],
    "astrometry": [("Solving", [S("Target name", "text", ""), S("Focal length", "text", "auto (header)"),
                                S("Distortion order", "int", 3), S("Blind solve", "bool", True),
                                S("Catalog", "choice", "Gaia DR3", ["Gaia DR3"])])],
    "gradient": [("Background", [S("Method", "choice", "polynomial", ["polynomial", "RBF", "automatic"]),
                                 S("Surface degree", "int", 2), S("Samples per side", "int", 16),
                                 S("Object rejection (σ)", "float", 2.5)])],
    "denoise": [("Network", [S("Method", "choice", "pretrained network",
                               ["pretrained network", "network adapted to the image"]),
                             S("Fast pass", "bool", False)])],
    "deconv": [("Sharpness", [S("Method", "choice", "measured PSF", ["measured PSF", "network (beta)"]),
                              S("Iterations", "int", 30), S("Regularization", "float", 0.002)])],
    "color": [("Calibration", [S("White reference", "choice", "Gaia stars", ["Gaia stars", "galaxies"]),
                               S("Balance", "choice", "auto balance", ["auto balance", "chosen zone"]),
                               S("Minimum stars", "int", 50)])],
    "stars": [("Star removal", [S("Method", "choice", "neural", ["neural"])])],
    "synth": [("Stars", [S("Method", "choice", "image shape", ["image shape", "instrument PSF"]),
                         S("Star width", "text", "2.5 px")])],
    "combine": [("Combination", [S("Luminosity", "choice", "noise", ["noise", "Hα"]),
                                 S("Channel balance", "text", "99.5 %")])],
    "combine_layers": [("Layers", [S("Stars mode", "choice", "Screen", ["Screen", "Linear dodge +", "Lighten"])])],
    "post": [("Output", [S("Stretch", "choice", "Statistical engine", ["Statistical engine", "Arcsinh"]),
                         S("Linked channels", "bool", True)])],
    "calibration": [
        ("Masters", [S("Darks combination", "choice", "Mean with rejection", ["Mean with rejection", "Median"]),
                     S("Darks scaling", "bool", False)]),
        ("Cosmetic correction", [S("Hot pixels", "choice", "Persistence", ["Persistence", "Frame by frame"]),
                                 S("Cold pixels", "bool", False)]),
        ("Flats", [S("Flat defects threshold", "float", 0.3), S("Flat scale", "text", "0 px", expert=True)]),
        ("Debayering", [S("Debayer method", "choice", "RCD", ["RCD", "VNG", "Bilinear"]),
                        S("CFA pattern", "text", "From header", expert=True)])],
    "registration": [
        ("Reference", [S("Reference image", "choice", "sharpest", ["sharpest", "chosen"])]),
        ("Detected stars", [S("Detection threshold", "text", "8 σ"), S("Stars per image", "int", 2500, expert=True)]),
        ("Warp", [S("Warp model", "choice", "Automatic", ["Automatic", "Homography"]),
                  S("Framing", "choice", "Reference image", ["Reference image", "Union"])])],
    "selection": [
        ("Rule", [S("Frame selection", "choice", "all · weight ≥ 5 %",
                    ["all · weight ≥ 5 %", "k × σ filters", "thresholds"]), S("Transparency filter", "bool", True)]),
        ("Other criteria", [S("Minimum roundness", "float", 0.0), S("Keep best", "text", "100 %")])],
    "crop": [("Zone", [S("Drawn zone", "text", "whole field")])],
    "stacking": [
        ("Method", [S("Method", "choice", "Drizzle", ["Drizzle", "Standard stacking"])]),
        ("Drizzle", [S("Drizzle scale", "text", "auto"), S("Drop size", "float", 0.7),
                     S("Drizzle kernel", "text", "Square", expert=True)]),
        ("Outlier rejection", [S("Rejection threshold", "text", "3 σ"), S("Separate thresholds", "bool", False)]),
        ("Half stacks", [S("Block size", "text", "20 frames", expert=True)])],
}
NO_COMMON = {("registration", "Reference image"): "specific to each stack: a frame chosen in its lights"}


def defaults(kind):
    return {it[0]: it[2] for _s, items in SETTINGS.get(kind, []) for it in items}


def setting_spec(kind, key):
    for _s, items in SETTINGS.get(kind, []):
        for it in items:
            if it[0] == key:
                return it
    if kind == "combine" and key == "type":
        return ("type", "choice", "sho", list(COMBO_PORTS), False)
    return None


def inherited(p, kind):
    """What a block of `kind` follows when it has no own value: Common, else default."""
    d = defaults(kind)
    if kind == "combine":
        d["type"] = "sho"
    d.update(p.common["settings"].get(kind, {}))
    return d


def effective(p, b):
    if b in p.blocks:
        blk = p.blocks[b]
        d = inherited(p, blk.kind)
        d.update(blk.settings)
        return d
    return inherited(p, b)


# ---------------------------------------------------------------------------------------------------- types
@dataclass(frozen=True)
class Img:
    frames: bool = False
    stage: int = 9          # last frame step done (0 calibrated, 2 aligned, 3 selected, 4 cropped); 9 = an image
    stars: str = "with"     # with / without / only / pair
    solution: bool = False
    grid: str = ""
    aligned: bool = False   # on another row's grid (the reference's)
    half: bool = False
    linear: bool = True
    denoised: bool = False
    synth: bool = False

    def but(self, **kw):
        return dataclasses.replace(self, **kw)


@dataclass
class Ev:
    out: object = None      # main output Img
    reason: object = ""     # greyed reason (a message, "" = fine)
    cat: str = ""           # "need" | "port" | "downstream"
    root: object = None     # block causing the greying


def outs(p, b, ev, blk=None):
    t = ev.out
    if t is None:
        return {}
    blk = blk or p.blocks[b]
    if blk.kind in ("stars", "synth"):
        return {"Starless": t.but(stars="without"), "Stars only": t.but(stars="only")}
    return {port: t for port in out_ports(blk)}


def _need(p, kind, t, ref_name, structural=False):
    """Reason the block cannot take image t ("" = fine). structural: only the order rules (a switched-off block)."""
    lab = LABEL[kind]
    if t.stars == "pair" and kind != "synth":
        return "only Synthetic stars follows Stars/background"
    if not t.linear and kind not in ("stars", "synth") and not structural:
        return M("no {lab} from a stretched image", lab=lab)
    if kind == "calibration":
        return ""
    if kind in FRAME_KINDS:
        k = COL[kind]
        if not t.frames:
            return M("{lab} comes before {next}", lab=lab, next="the Stacking")
        if t.stage == k:
            return M("{lab} is already in the row", lab=lab)
        if t.stage > k:
            nxt = {3: "Selection", 4: "Crop", 5: "the Stacking"}[t.stage]
            return M("{lab} comes before {next}", lab=lab, next=nxt)
        if kind != "registration" and t.stage < 2:
            return M("{lab} comes after {reg}", lab=lab, reg="Registration")
        return ""
    if t.frames:
        return "Stacking is needed first"
    if structural:
        return ""
    if kind == "astrometry":
        if t.aligned:
            return M("the grid of “{_ref}” gives the solution", _ref=ref_name)
        if t.solution:
            return "only one Astrometry per row"
        if t.stars != "with":
            return "stars are needed"
    if kind in ("deconv", "denoise") and not t.half:
        return "the half stacks are needed (start from the lights)"
    if kind == "color" and not t.solution:
        return "Astrometry is needed first"
    if kind == "stars":
        if not t.denoised:
            return "a denoised image is needed"
        if t.stars != "with":
            return "stars are needed"
    if kind == "synth":
        if t.stars != "pair":
            return "Stars/background is needed first"
        if t.synth:
            return M("{lab} is already in the row", lab=lab)
    return ""


def _make(kind, t):
    if kind in FRAME_KINDS:
        if kind == "stacking":
            return t.but(frames=False, stage=9, half=True)
        return t.but(stage=COL[kind])
    if kind == "astrometry":
        return t.but(solution=True)
    if kind == "denoise":
        return t.but(denoised=True)
    if kind == "stars":
        return t.but(stars="pair", half=False)
    if kind == "synth":
        return t.but(synth=True)
    return t


class Ctx:
    """What evaluating one block needs: its inputs (chain/night/fork) and active port wires, the reference."""

    def __init__(self, p, ix):
        self.p, self.ix = p, ix
        ref = p.reference
        self.ref = ref
        self.ref_name = ix.row[ref].name if ref in ix.row else ""
        self.ref_astro = ref in ix.row and any(p.blocks[b].kind == "astrometry" and not p.blocks[b].off
                                               for b in ix.row[ref].blocks)
        self.head_in, self.wires_in = {}, {}
        for r in p.rows:
            if r.nights and r.blocks:
                self.head_in[r.blocks[0]] = [c for _s, c in r.nights]
            if r.fork and r.blocks:
                self.head_in[r.blocks[0]] = [r.fork[1]]
            for a, b in zip(r.blocks, r.blocks[1:]):
                self.head_in[b] = [a]
            for s, c in r.nights:
                self.head_in[c] = [s]
        for w in p.wires:
            if w.active:
                self.wires_in.setdefault(w.dst[0], []).append(w)
        self.virtual = {}                       # hypothetical blocks (fast placement judge)

    def blk(self, b):
        return self.virtual.get(b) or self.p.blocks[b]


def eval_one(c, b, info):
    """The Ev of block b from its inputs' Evs in info."""
    p, ix = c.p, c.ix
    blk = c.blk(b)
    k = blk.kind
    ev = Ev()
    if k == "session":
        rid = ix.row_of[b]
        ev.out = Img(frames=True, stage=0, grid=c.ref or rid, aligned=rid != c.ref)
        return ev
    if k == "image":
        ev.out = Img(grid=ix.row_of.get(b, b), linear=not effective(p, b)["Stretched image"]
                     if b in p.blocks else not inherited(p, "image")["Stretched image"])
        return ev
    if k == "post":
        return ev
    if k in JOIN_KINDS:
        _join(c, b, blk, ev, info, c.wires_in.get(b, []))
        return ev
    up = [info[s] for s in c.head_in.get(b, []) if s in info]
    bad = next((u for u in up if u.cat), None)
    if bad is not None:
        ev.cat, ev.root = "downstream", bad.root
        return ev
    t = up[0].out if up and up[0].out is not None else None
    if t is None:
        ev.cat, ev.root = "port", b
        return ev
    if k == "stacking" and t.frames and t.aligned and c.ref_astro:
        t = t.but(solution=True)
    why = _need(p, k, t, c.ref_name, structural=blk.off)
    if why:
        ev.reason, ev.cat, ev.root = why, "need", b
        return ev
    ev.out = (t.but(stage=COL[k]) if k in FRAME_KINDS else t) if blk.off else _make(k, t)
    return ev


def evaluate(p, ix=None):
    """{block id: Ev} in topological order (§2: type of a wire = its attributes; needs give greyed reasons)."""
    ix = ix or Index(p)
    order = topo(p, ix, active_only=True) or list(p.blocks)
    c = Ctx(p, ix)
    info = {}
    for b in order:
        info[b] = eval_one(c, b, info)
    for b, ev in info.items():                 # downstream texts
        if ev.cat == "downstream":
            ev.reason = M("downstream of {t}", t=title_msg(p, ev.root, ix))
        elif ev.cat == "port" and not ev.reason:
            ev.reason = "no image in input"
    return info


def _join(c, b, blk, ev, info, wires):
    p = c.p
    ports = in_ports(blk)
    got = {}
    for w in wires:
        si = info.get(w.src[0])
        if si is None:
            continue
        if si.cat:
            ev.cat, ev.root = "downstream", si.root
            continue
        t = outs(p, w.src[0], si, c.blk(w.src[0])).get(w.src[1])
        if t is not None:
            got[w.dst[1]] = t
    lab = COMBO_LABEL[combo_type(blk)] if blk.kind == "combine" else LABEL[blk.kind]
    for port in ports:
        t = got.get(port)
        if t is None:
            continue
        why = ""
        if t.frames:
            why = "Stacking is needed first"
        elif blk.kind == "combine" and not t.linear:
            why = M("no {lab} from a stretched image", lab=lab)
        elif port == "Background" and t.stars != "without":
            why = "the Background port needs a starless image"
        elif port == "Stars" and t.stars != "only":
            why = "the Stars port needs the stars only"
        if why:
            ev.reason, ev.cat, ev.root, ev.out = why, "need", b, None
            return
    vals = [got[q] for q in ports if q in got]
    if len({t.grid for t in vals}) > 1 and not all(t.solution for t in vals):
        ev.reason, ev.cat, ev.root = ("ports on different grids: Astrometry is needed"
                                      if blk.kind == "combine" else "Background and Stars on different grids"), \
            "need", b
        return
    if ev.cat == "downstream":
        return
    if len(vals) < len(ports):
        ev.reason, ev.cat, ev.root = "ports to wire", "port", b
        return
    first = vals[0]
    if blk.kind == "combine":
        ev.out = first.but(solution=all(t.solution for t in vals), half=False, synth=False,
                           denoised=all(t.denoised for t in vals))
    else:
        ev.out = first.but(stars="with", half=False, synth=False)


def need_greyed(info):
    """Blocks greyed by a failed need (the state no operation may create)."""
    return {b: ev.reason for b, ev in info.items() if ev.cat == "need"}


# ---------------------------------------------------------------------------------------------------- texts
def title_msg(p, b, ix=None):
    ix = ix or Index(p)
    blk = p.blocks[b]
    k = blk.kind
    if k == "post":
        return "Post"
    if k == "session":
        r = ix.row[ix.row_of[b]]
        return typed(r.name if len(r.nights) == 1 else blk.name)
    if k == "calibration":
        return "Calibration"
    if k == "combine":
        t = combo_type(blk)
        same = [r.blocks[0] for r in p.rows if r.blocks and p.blocks[r.blocks[0]].kind == "combine"
                and combo_type(p.blocks[r.blocks[0]]) == t]
        n = same.index(b) + 1 if b in same else 1
        return COMBO_LABEL[t] if n == 1 else M("{label} {n}", label=COMBO_LABEL[t], n=n)
    if k == "combine_layers":
        same = [r.blocks[0] for r in p.rows if r.blocks and p.blocks[r.blocks[0]].kind == k]
        n = same.index(b) + 1 if b in same else 1
        return LABEL[k] if n == 1 else M("{label} {n}", label=LABEL[k], n=n)
    chain = ix.effective(ix.row_of[b]) if b in ix.row_of else [b]
    n = [p.blocks[x].kind for x in chain[:chain.index(b) + 1]].count(k)
    return LABEL[k] if n == 1 else M("{label} {n}", label=LABEL[k], n=n)


def title(p, b, ix=None, tr=None):
    return say(title_msg(p, b, ix), tr)


_COUNT = {"lights": ("{n} light", "{n} lights"), "darks": ("{n} dark", "{n} darks"),
          "flats": ("{n} flat", "{n} flats"), "bias": ("{n} bias", "{n} biases"),
          "flat_darks": ("{n} flat dark", "{n} flat darks")}


def _count(n, kind):
    one, many = _COUNT[kind]
    return M(one if n == 1 else many, n=n)


def files_of(p, s, kind):
    f = p.blocks[s].files
    if kind in f:
        return "own", list(f[kind])
    if p.common["files"].get(kind):
        return "common", list(p.common["files"][kind])
    return "none", []


def mother_same(p, ix, rid):
    """True when variant rid is identical to its mother (kinds, settings, off) from its fork on."""
    r = ix.row[rid]
    par = ix.effective(r.fork[0])
    tail = par[par.index(r.fork[1]) + 1:]
    if len(tail) != len(r.blocks):
        return False
    for a, b in zip(tail, r.blocks):
        A, B = p.blocks[a], p.blocks[b]
        if A.kind != B.kind or A.off != B.off or effective(p, a) != effective(p, b):
            return False
    return True


def summary_msg(p, b, info=None, ix=None):
    ix = ix or Index(p)
    info = info or evaluate(p, ix)
    blk = p.blocks[b]
    k = blk.kind
    ev = info.get(b)
    if ev and ev.reason:
        return ev.reason
    if blk.off:
        return "off: image passed on"
    rid = ix.row_of.get(b)
    if rid and ix.row[rid].fork and ix.row[rid].blocks[0] == b and mother_same(p, ix, rid):
        return "like the mother — change a setting"
    e = effective(p, b)
    if k == "session":
        parts = []
        for f in FILE_KINDS:
            src, paths = files_of(p, b, f)
            if f == "lights" and not paths:
                parts.append("lights to add")
            elif paths:
                parts.append(_count(len(paths), f))
        txt = join(parts)
        if p.reference and rid != p.reference:
            txt = join([M("on the grid of {_ref}", _ref=ix.row[p.reference].name), txt])
        return txt
    if k == "calibration":
        changed = [key for key, v in e.items() if v != defaults(k).get(key)]
        if not changed:
            return "standard"
        if changed == ["Darks scaling"]:
            return "darks adjusted"
        return M("{n} setting adjusted" if len(changed) == 1 else "{n} settings adjusted", n=len(changed))
    if k == "registration":
        return M("reference frame: {v}", v=e["Reference image"])
    if k == "selection":
        return e["Frame selection"]
    if k == "crop":
        return e["Drawn zone"]
    if k == "stacking":
        m = M("drizzle {scale}", scale=e["Drizzle scale"]) if e["Method"] == "Drizzle" else "classic"
        return M("{m} · rejection at {thr}", m=m, thr=e["Rejection threshold"])
    if k == "image":
        return join([typed(e["Image file"])] + (["stretched"] if e["Stretched image"] else []))
    if k == "astrometry":
        return M("solves the field · {cat}", cat=e["Catalog"])
    if k in ("gradient", "denoise", "deconv", "stars", "synth"):
        return e["Method"]
    if k == "color":
        return M("{ref} · {bal}", ref=e["White reference"], bal=e["Balance"])
    if k == "combine":
        return join(COMBO_PORTS[combo_type(blk)], ", ")
    if k == "combine_layers":
        return M("stars in {mode}", mode=e["Stars mode"])
    if k == "post":
        n = sum(1 for w in p.wires if w.dst[0] == b)
        return "no layer" if n == 0 else ("1 layer" if n == 1 else M("{n} layers", n=n))
    return ""


def summary(p, b, info=None, ix=None, tr=None):
    return say(summary_msg(p, b, info, ix), tr)


def greyed(p, b, info=None, tr=None):
    info = info or evaluate(p)
    ev = info.get(b)
    return say(ev.reason, tr) if ev else ""


def describe_msg(t):
    if t is None:
        return ""
    if t.frames:
        steps = {0: "calibrated", 2: "aligned", 3: "selected", 4: "cropped"}.get(t.stage, "")
        return M("lights {step}", step=steps) if steps else "lights"
    parts = ["linear image" if t.linear else "stretched image",
             {"with": "with stars", "without": "without stars", "only": "stars only",
              "pair": "without stars + stars only"}[t.stars]]
    if t.solution:
        parts.append("astrometric solution")
    if t.half:
        parts.append("half stacks")
    if t.denoised:
        parts.append("denoised")
    return join(parts)


def describe(t, tr=None):
    return say(describe_msg(t), tr)


def wire_tip_msg(p, w, info=None):
    info = info or evaluate(p)
    ws = {x.id: x for x in p.wires}
    if w in ws:
        src, port = ws[w].src
    else:
        src, port = w[2:].split(">")[0], None
    ev = info.get(src)
    if ev is None or ev.reason or ev.out is None:
        return M("No image: {t} is greyed", t=title_msg(p, src)) if ev and ev.reason else "No image"
    t = outs(p, src, ev).get(port) if port else ev.out
    return describe_msg(t)


def wire_tip(p, w, info=None, tr=None):
    return say(wire_tip_msg(p, w, info), tr)


def stacks(p):
    return [(r.id, [(p.blocks[s].name, s, c) for s, c in r.nights]) for r in p.rows if r.nights]


def chain_blocks(p):
    return [b for r in p.rows for b in [x for n in r.nights for x in n] + r.blocks]


def differs(p, b):
    k = p.blocks[b].kind
    same = [x for x in chain_blocks(p) if p.blocks[x].kind == k]
    if len(same) < 2:
        return False
    vals = [tuple(sorted(effective(p, x).items(), key=lambda kv: kv[0])) for x in same]
    best = max(vals, key=lambda v: (vals.count(v), -vals.index(v)))
    return vals[same.index(b)] != best


def followers(p, kind):
    if kind in FILE_KINDS:
        sess = [s for r in p.rows for s, _c in r.nights]
        return ([s for s in sess if kind not in p.blocks[s].files], [s for s in sess if kind in p.blocks[s].files])
    blocks = [x for x in chain_blocks(p) if p.blocks[x].kind == kind]
    return [x for x in blocks if not p.blocks[x].settings], [x for x in blocks if p.blocks[x].settings]


def warnings(p, scope=None):
    """What the model alone decides: [(session id, kind, message, (op, kwargs) | None)]."""
    ix = Index(p)
    sess = [s for r in p.rows for s, _c in r.nights]
    if scope in ix.row:
        mine = [s for s, _c in ix.row[scope].nights]
    elif scope in p.blocks:
        mine = [scope]
    else:
        mine = sess
    out = []
    any_flats = {s for s in sess if files_of(p, s, "flats")[1]}
    where = {}
    for s in sess:
        for f in FILE_KINDS:
            if f in p.blocks[s].files:
                for path in p.blocks[s].files[f]:
                    where.setdefault(path, []).append((s, f))
    for s in mine:
        if s not in any_flats and any_flats - {s}:
            out.append((s, "flats", "No flats: the other sessions have some", None))
        for f in FILE_KINDS:
            for path in p.blocks[s].files.get(f, []):
                other = [(s2, f2) for s2, f2 in where.get(path, []) if (s2, f2) != (s, f)]
                if other:
                    s2, f2 = other[0]
                    out.append((s, f, M("Also in {kind} of “{t}”", kind=FILE_LABEL[f2], t=title_msg(p, s2, ix)),
                                ("remove_files", {"session": s, "kind": f, "paths": [path]})))
                    break
    return out


def card_summary_msg(p, rid):
    ix = Index(p)
    r = ix.row[rid]
    if not r.nights:
        return ""
    n = len(r.nights)
    lights = sum(len(files_of(p, s, "lights")[1]) for s, _c in r.nights)
    stk = next((b for b in r.blocks if p.blocks[b].kind == "stacking"), None)
    parts = [M("{n} session" if n == 1 else "{n} sessions", n=n), _count(lights, "lights")]
    if stk:
        parts.append(_first_part(summary_msg(p, stk, ix=ix)))
    return join(parts)


def _first_part(m):
    """The part of a summary before its first " · " (the stacking card line keeps the method only)."""
    if isinstance(m, tuple) and m[0].startswith("{m} · "):
        return m[1]["m"]
    return m


def card_summary(p, rid, tr=None):
    return say(card_summary_msg(p, rid), tr)


def project_state_msg(p):
    """("none" | "partial" | "complete", message)."""
    sess = [s for r in p.rows for s, _c in r.nights]
    if not p.rows:
        return "none", "Empty project: add a Session or an Image"
    empty = [title(p, s) for s in sess if not files_of(p, s, "lights")[1]]
    grey = [b for b, ev in evaluate(p).items() if ev.cat]
    if empty:
        return "partial", M("Lights to add: {_names}", _names=", ".join(empty))
    if grey:
        return "partial", M("{n} block greyed" if len(grey) == 1 else "{n} blocks greyed", n=len(grey))
    return "complete", "Ready to run"


def project_state(p, tr=None):
    state, m = project_state_msg(p)
    return state, say(m, tr)


# ---------------------------------------------------------------------------------------------------- structure
AFTER_JOIN = ("gradient", "color", "stars", "synth")


def chain_problem(kinds, root):
    """Order rules of one effective chain, whatever the greying (CHANGES.md, "What follows what").
    kinds: the chain's block kinds left to right; root: "stack" | "image" | "join". -> a message or ""."""
    last_frame, prev, n_ef, n_syn = -1, None, 0, 0
    for i, k in enumerate(kinds):
        lab = LABEL[k]
        if k in FRAME_KINDS:
            if root != "stack" or last_frame == 5 or (prev is not None and prev not in FRAME_KINDS):
                return M("{lab} belongs to a stack: between the Session and the Stacking", lab=lab)
            if (i == 0) != (k == "registration") or COL[k] <= last_frame:
                return M("{lab}: the order is Registration, Selection, Crop, Stacking", lab=lab)
            last_frame = COL[k]
        elif i > 0:
            if k in ("session", "calibration", "image", "post") + JOIN_KINDS:
                return M("{lab} is only placed at the head of a row", lab=lab)
            if root == "stack" and last_frame != 5:
                return M("{lab} comes after the Stacking", lab=lab)
            if root == "join" and k not in AFTER_JOIN:
                return "after a join: Gradient, Color or Stars/background"
        if prev == "stars" and k != "synth":
            return "only Synthetic stars follows Stars/background"
        if k == "synth" and prev != "stars":
            return "Stars/background is needed first"
        n_ef += k == "stars"
        n_syn += k == "synth"
        if n_ef > 1 or n_syn > 1:
            return M("only one {lab} per row", lab=lab)
        prev = k
    if root == "stack" and last_frame != 5:
        return "Stacking is essential to the stack"
    return ""


def root_kind(p, ix, rid):
    r = ix.row[ix.root(rid)]
    if r.nights:
        return "stack"
    return "image" if p.blocks[r.blocks[0]].kind == "image" else "join"


def structure(p, ix=None):
    """Problems of the stricter structural validator (empty = fine): every row's effective chain obeys
    chain_problem, and nothing follows Post (Post is in no row). -> [message]."""
    ix = ix or Index(p)
    out = []
    for r in p.rows:
        why = chain_problem([p.blocks[b].kind for b in ix.effective(r.id)], root_kind(p, ix, r.id))
        if why:
            out.append(M("“{_row}”: {why}", _row=r.name, why=why))
    return out
