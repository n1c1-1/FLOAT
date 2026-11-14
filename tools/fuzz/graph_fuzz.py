"""Fuzz (§6): random gestures from blank and from each template. After each: check(), I1–I6, refusals inert,
undo then redo = same JSON, every row drawn. A failure prints its seed and the shrunk gesture list.
Above 30 rows, 40 % of gestures prune (delete a row or a block) so graphs stay near the size people build.
Each run prints a digest of its final state: the same seed gives the same digest (temp file per process).
Also: apply never changes the graph it is given. Not part of pytest (the smoke is tests/graphcore/test_fuzz_smoke.py).
Usage: python tools/fuzz/graph_fuzz.py [first_seed] [last_seed] [gestures] [start,start…]

The engine's legality (decision B, INTEGRATION §3), on by default (``--no-engine``: off), both directions:
- after every accepted gesture the graph's compiled definition is one the engine accepts, and the core greys each
  block the engine leaves out with the engine's reason (graphcore.compile.engine_problems: the graph rules of every
  workflow, Project2.validate, the stage chains, Keys.wiring_problem against the core's greyed blocks);
- after every refused gesture the engine refuses the state the gesture would have made (operations.attempt: the same
  operation without its refusal, compiled), unless the refusal is one of the core's own rules listed below
  (CORE_ONLY: drawing rules, each with why the engine cannot say it), or the operation itself refused before making a
  state (GESTURE: a gesture that names nothing that exists, or one the model cannot hold; listed too).
``--survey``: violations are counted and printed at the end instead of stopping the run (``--json <file>`` writes
the first gesture sequence of each class).
Usage: python tools/fuzz/graph_fuzz.py [first_seed] [last_seed] [gestures] [start,start…] [--survey] [--json <file>]"""
import copy
import hashlib
import os
import random
import sys
import time
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests", "graphcore"))
from astro_gpu import graphcore  # noqa: E402
from astro_gpu.graphcore import layout, COMMON  # noqa: E402
from astro_gpu.graphcore.model import check, Index, out_ports, in_ports, FILE_KINDS, COMBO_PORTS, model_json  # noqa: E402
from astro_gpu.graphcore.rules import ORDER, SETTINGS, structure  # noqa: E402
from astro_gpu.graphcore.placement import candidates, judge  # noqa: E402
from astro_gpu.graphcore.msg import say  # noqa: E402
from fixtures import Doc, templates  # noqa: E402
import invariants as inv  # noqa: E402
from astro_gpu.graphcore.compile import engine_problems  # noqa: E402
from astro_gpu.graphcore import operations as ops, rules as core_rules  # noqa: E402

#: the core's own refusals of a state the engine would run (reverse invariant), each with why the engine cannot say it
CORE_ONLY = {
    core_rules.FRAME_BAND: "the frame band is drawn in fixed columns before the image band: a frame block (switched "
                           "off: absent for the engine) after an image block, or outside a stack, has no column",
    core_rules.FRAME_ORDER: "the same: two frame blocks of one kind, or out of order, would share a column",
    core_rules.IMAGE_BAND: "the same: an image block (switched off: absent for the engine) drawn among the frame "
                           "columns",
    "the Background port needs a starless image": "Combine layers is a layer of Post, no workflow: its ports are "
                                                  "the core's (the engine never sees them)",
    "the Stars port needs the stars only": "the same",
    "Preprocessing is folded: unfold it first": "a view rule: the folded band hides the block a variant would fork "
                                                "from",
    "no {lab} from a stretched image": "a combination works on linear images (stages/combine reads an Output's or a "
                                       "Stars/background's linear image): a stretched image's workflow has no Output, "
                                       "its starless is stretched; Project2.validate does not look at what an input "
                                       "gives, the run would find nothing to read",
}
#: refusals an operation gives with the engine's own verdict, before any state (project2.join_problem on the compiled
#: graph: the reverse invariant holds by construction)
ENGINE_SAID = {"a workflow cannot read itself", "no stack", "{src} already reads {target}: it would make a loop"}
#: the operations' own refusals: no state is made (an argument naming nothing that exists, or a shape the model has no
#: form for); a refusal of this kind that is not listed fails the run
GESTURE = {
    # names nothing that exists, or the wrong kind of thing
    "block not found", "row not found", "wire not found", "session not found", "unknown operation: {_name}",
    "unknown block", "unknown place", "unknown setting", "unknown type", "unknown port", "unknown kind of files",
    "unknown destination", "unknown mode", "the place is gone", "session not found in this stack",
    "{lab} does not fit here", "files go on a Session", "these lights are not in the session",
    "these files are not in the box", "files absent from the session", "no file", "nothing to split",
    "no output {port}", "no input {port}", "lights first", "a name cannot be empty", "“{_name}” already exists",
    "an Image takes no files here", "lights go on a session, not on Common",
    "the composition takes a Combination or Combine layers", "a port takes a new row: Session or Image",
    "this wire already exists", "already in Post", "this wire is already taken", "the port is already wired",
    "already alone on its row", "both sessions must be in the same stack", "two different stacks of lights are needed",
    "no other row has a {lab}", "this Stars/background already feeds a block", "already on its own grid",
    "no Align on in this row",
    # step 3b: the engine's settings by their real keys (the engine's own check), variants on trial, a combination's
    # inputs, shared nights, taps, a Calibration's own files
    "unknown stage {_stage}", "{_stage} has no setting {_name}", "{_stage}.{_name} is set by the project",
    "{_stage}.{_name}: not a valid value", "{lab} holds no setting of {_stage}", "a stack of lights holds this setting",
    "{_stage}: a block holds this setting", "{lab} has no method", "unknown change", "{lab}: no variant from here",
    "not a variant on trial", "its original is gone: it is a workflow of its own",
    "a variant of “{_row}” starts from a block the variant does not have",
    "a block of “{_row}” sent to Post is not in the variant", "unknown channel", "{port}: no input wired",
    "a weight is a number", "nothing to change", "needs Stars/background", "unknown image",
    "its inputs read no starless image", "“{_name}” is already in “{_row}”",
    "“{_name}” is shared by several stacks: its lights stay together", "not found", "it already follows its session",
    "{lab} is switched off: its image is its input's",
    "unknown method",
    # the model's shapes: a row starts with its head, a variant forks after a block of its row, a night's
    # Calibration is its night's, Post is one, a source stays at its row's head
    "nothing is inserted before the head of the row", "a variant starts after the Registration",
    "each session already has its Calibration", "each session has its Calibration: switch it off (D)",
    "each session keeps its Calibration", "only a Session joins a stack", "a session joins a stack of lights",
    "{lab} is not inserted in a row", "a row starts with a Session or an Image", "a source stays at the head of its row",
    "Post is fixed", "unknown place for this block", "variants start from this block: delete them first",
    "the reference is a stack of lights", "an Align on follows a Stacking or an Image of the row",
    "the block is not in the target's chain", "add a Session or an Image first", "{lab}: no place",
    # Commun is the graph's own (the compile flattens it): a setting specific to each stack has no common value
    "specific to each stack: a frame chosen in its lights",
    # wires: a loop is not a graph; the engine's joins read an image or a Stars/background
    "this wire would close a loop",
    # a switch-off of a block that is not switched (the engine's off_problem says the same for its own nodes)
    "— {takes} → {gives}, essential",
}

PATHS = [f"D:/night{n}/IMG_{i:04d}.fit" for n in range(3) for i in range(8)]
NAMES = ["M31", "Ha", "OIII", "", "  ", "Session 1", "SII", "RGB", "M31 2"]


class Fail(Exception):
    pass


def pick(rng, seq):
    seq = list(seq)
    return rng.choice(seq) if seq else None


def gen(rng, doc):
    """One gesture as a replayable record: ("op", name, kwargs, place|None) or ("undo",) etc."""
    p = doc.project
    ix = Index(p)
    blocks = list(p.blocks)
    rows = [r.id for r in p.rows]
    sess = [s for r in p.rows for s, _c in r.nights]
    wires = [w.id for w in p.wires]
    joins = [b for b in blocks if p.blocks[b].kind == "combine"]
    roll = rng.random()
    if len(rows) > 30 and rng.random() < 0.4:          # keep graphs of a size people build: prune
        if rng.random() < 0.6:
            return ("op", "delete_row", {"row": pick(rng, rows)}, None)
        return ("op", "remove_block", {"block": pick(rng, blocks)}, None)
    if roll < 0.25:
        kind = pick(rng, ORDER)
        cands = candidates(p, doc.view, kind)
        if not cands:
            return ("op", "add_block", {"kind": kind, "row": pick(rng, rows + [None])}, None)
        pl = judge(p, doc.view, rng.choice(cands))
        return ("op", pl.op[0], pl.op[1], pl)
    if roll < 0.32:
        return ("op", "add_block", {"kind": pick(rng, ORDER), "row": pick(rng, rows + [None])}, None)
    if roll < 0.40:
        b = pick(rng, [x for x in blocks if x in ix.row_of])
        if b is None:
            return ("undo",)
        if rng.random() < 0.15:
            return ("op", "remove_block", {"block": b}, None)
        cands = candidates(p, doc.view, block=b)
        if not cands:
            return ("op", "remove_block", {"block": b}, None)
        pl = judge(p, doc.view, rng.choice(cands))
        return ("op", pl.op[0], pl.op[1], pl)
    g = rng.choice([
        lambda: ("op", "remove_block", {"block": pick(rng, blocks)}),
        lambda: ("op", "delete_row", {"row": pick(rng, rows)}),
        lambda: ("op", "remove_night", {"night": pick(rng, sess)}),
        lambda: ("op", "add_variant", {"block": pick(rng, blocks)}),
        lambda: ("op", "take_alternative", {"wire": pick(rng, wires)}),
        lambda: ("op", "set_reference", {"row": pick(rng, rows)}),
        lambda: _align(rng, p, rows),
        lambda: ("op", "own_grid", {"row": pick(rng, rows)}),
        lambda: ("op", "set_align_off", {"row": pick(rng, rows)}),
        lambda: ("op", "set_calibrate_together", {"on": rng.random() < 0.5}),
        lambda: ("op", "set_combination_type", {"block": pick(rng, joins or blocks), "type": pick(rng, COMBO_PORTS)}),
        lambda: _wire(rng, p),
        lambda: ("op", "unwire", {"wire": pick(rng, wires)}),
        lambda: ("op", "switch_off", {"block": pick(rng, blocks)}),
        lambda: ("op", "set_pause", {"block": pick(rng, blocks)}),
        lambda: ("op", "rename_row", {"row": pick(rng, rows), "name": pick(rng, NAMES)}),
        lambda: ("op", "rename_night", {"night": pick(rng, sess), "name": pick(rng, NAMES)}),
        lambda: ("op", "duplicate_row", {"row": pick(rng, rows)}),
        lambda: ("op", "add_night", {"row": pick(rng, rows)}),
        lambda: ("op", "move_night", {"night": pick(rng, sess), "row": pick(rng, rows + [None])}),
        lambda: ("op", "move_night", {"night": pick(rng, sess), "row": pick(rng, rows), "before": pick(rng, sess)}),
        lambda: ("op", "add_night", {"row": pick(rng, rows), "files": {"lights": rng.sample(PATHS, 2)},
                                     "name": pick(rng, NAMES)}),
        lambda: ("op", "add_row", {"kind": pick(rng, ["session", "image"]), "name": pick(rng, NAMES),
                                   "files": {pick(rng, FILE_KINDS): rng.sample(PATHS, 2)} if rng.random() < .5
                                   else None}),
        lambda: ("op", "merge_nights", {"a": pick(rng, sess), "b": pick(rng, sess)}),
        lambda: _split(rng, p, sess),
        lambda: ("op", "add_files", {"session": pick(rng, sess), "kind": pick(rng, FILE_KINDS),
                                     "paths": rng.sample(PATHS, rng.randint(0, 5)),
                                     "mode": pick(rng, ["add", "replace"])}),
        lambda: ("op", "remove_files", {"session": pick(rng, sess), "kind": pick(rng, FILE_KINDS),
                                        "paths": rng.sample(PATHS, rng.randint(1, 8))}),
        lambda: ("op", "use_for_all_nights", {"session": pick(rng, sess), "kind": pick(rng, FILE_KINDS)}),
        lambda: _setting(rng, p, blocks),
        lambda: ("op", "apply_to_rows", {"block": pick(rng, blocks), "key": _key(rng, p, pick(rng, blocks))}),
        lambda: ("op", "send_to_post", {"block": pick(rng, blocks)}),
        lambda: ("op", "fold_all", {}),
        lambda: ("op", "unfold_all", {}),
        lambda: ("op", "set_common_files", {"kind": pick(rng, FILE_KINDS), "paths": rng.sample(PATHS, 3),
                                            "mode": pick(rng, ["add", "replace"])}),
        lambda: _common_setting(rng),
        lambda: ("op", "reset_own", {"block": pick(rng, blocks), "keys": None if rng.random() < .5 else
                                     [_key(rng, p, pick(rng, blocks))]}),
        lambda: ("op", "reset_own_files", {"session": pick(rng, sess), "kind": pick(rng, FILE_KINDS)}),
        lambda: ("op", "move_files", {"src_session": pick(rng, sess + [COMMON]), "src_kind": pick(rng, FILE_KINDS),
                                      "dst_session": pick(rng, sess + [COMMON]), "dst_kind": pick(rng, FILE_KINDS),
                                      "paths": None if rng.random() < .5 else rng.sample(PATHS, 2),
                                      "copy": rng.random() < .3}),
        lambda: _split_by(rng, p, sess),
        lambda: ("op", "merge_rows", {"src": pick(rng, rows), "dst": pick(rng, rows)}),
        lambda: _try_variant(rng, p, blocks),
        lambda: ("op", "keep_variant", {"row": pick(rng, [r.id for r in p.rows if r.trial is not None] or rows)}),
        lambda: ("op", "discard_variant", {"row": pick(rng, [r.id for r in p.rows if r.trial is not None] or rows)}),
        lambda: _set_input(rng, p, joins),
        lambda: ("op", "add_stars_combination", {"block": pick(rng, [j for j in joins if any(
            w.dst[0] == j and w.src[1] == "Starless" for w in p.wires)] or joins or blocks)}),
        lambda: ("op", "add_combination", {"type": pick(rng, list(COMBO_PORTS) + ["x"]), "name": pick(rng, NAMES)}),
        lambda: _share(rng, p, sess),
        lambda: _template(rng, p),
        lambda: ("op", "set_night_filter", {"night": pick(rng, sess), "filter": pick(rng, ["Ha", "OIII", ""])}),
        lambda: ("op", "send_to_post", {"block": pick(rng, blocks), "name": pick(rng, NAMES)}),
        lambda: ("op", "remove_tap", {"tap": pick(rng, [t.id for t in p.taps] or ["t0"])}),
        lambda: _set_param(rng, p, blocks),
        lambda: ("op", "set_row_param", {"row": pick(rng, rows), "stage": pick(rng, ["render_simple", "calibrate",
                                                                                   "align_on", "stack"]),
                                         "name": pick(rng, ["linked", "apply_masters", "method"]),
                                         "value": pick(rng, [True, False, "stars"])}),
        lambda: ("op", "set_method", {"block": pick(rng, blocks), "method": pick(rng, METHOD_CHOICES)}),
        lambda: ("op", "set_own_calibration", {"block": pick(rng, [c for r in p.rows for _s, c in r.nights] or blocks),
                                               "kind": pick(rng, FILE_KINDS), "paths": rng.sample(PATHS, 2),
                                               "mode": pick(rng, ["add", "replace"])}),
        lambda: ("op", "follow_session", {"block": pick(rng, [c for r in p.rows for _s, c in r.nights] or blocks),
                                          "kind": pick(rng, FILE_KINDS)}),
        lambda: ("op", "remove_own_calibration", {"block": pick(rng, [c for r in p.rows for _s, c in r.nights]
                                                                      or blocks)}),
        lambda: ("many", [_setting(rng, p, blocks)[1:3], ("add_night", {"row": pick(rng, rows)})]),
        lambda: ("undo",), lambda: ("undo",), lambda: ("redo",),
        lambda: ("save",),
    ])()
    return g if g[0] != "op" or len(g) == 4 else g + (None,)


def _align(rng, p, rows):
    """Align a row on another row or tap, from one of its blocks (or its Stacking/Image), or with its target to
    choose."""
    ref = pick(rng, rows + [t.id for t in p.taps] + [None])
    blocks = []
    if ref is not None:
        ix = Index(p)
        blocks = ix.effective(ref) if ref in ix.row else []
    return ("op", "set_align", {"row": pick(rng, rows), "ref": ref,
                                "block": pick(rng, blocks) if blocks and rng.random() < 0.5 else None,
                                "method": pick(rng, [None, None, "stars", "plate", "x"])}, None)


def _wire(rng, p):
    srcs = [b for b in p.blocks if out_ports(p.blocks[b])]
    dsts = [b for b in p.blocks if in_ports(p.blocks[b])]
    s, d = pick(rng, srcs or list(p.blocks)), pick(rng, dsts or list(p.blocks))
    return ("op", "wire", {"src": s, "src_port": pick(rng, out_ports(p.blocks[s]) or ["image"]),
                           "dst": d, "dst_port": pick(rng, in_ports(p.blocks[d]) or ["layers"])})


def _key(rng, p, b):
    items = [it for _s, its in SETTINGS.get(p.blocks[b].kind, []) for it in its] if b else []
    return pick(rng, [it[0] for it in items]) or "Method"


def _value(rng, it):
    name, typ, default, options, _e = it
    if typ == "choice":
        return pick(rng, options)
    if typ == "bool":
        return rng.random() < 0.5
    if typ == "int":
        return rng.randint(1, 5)
    if typ == "float":
        return rng.choice([0.5, 0.7, 1.0])
    return rng.choice([default, "x", "y"])


def _setting(rng, p, blocks):
    b = pick(rng, blocks)
    items = [it for _s, its in SETTINGS.get(p.blocks[b].kind, []) for it in its]
    if not items:
        return ("op", "set_setting", {"block": b, "key": "nothing", "value": 1}, None)
    it = rng.choice(items)
    return ("op", "set_setting", {"block": b, "key": it[0], "value": _value(rng, it)}, None)


def _common_setting(rng):
    kind = pick(rng, [k for k in SETTINGS])
    it = rng.choice([it for _s, its in SETTINGS[kind] for it in its])
    return ("op", "set_common_setting", {"kind": kind, "key": it[0], "value": _value(rng, it)}, None)


#: the engine settings the fuzz sets (stage, name, values: valid ones and refused ones)
PARAMS = [("register", "min_pairs", [3, 10, 20, 2, 500]), ("stack", "method", ["drizzle", "classic", "nope"]),
          ("stack", "min_transparency", [0.6, 0.2]), ("stack", "kappa", [2.5, 3.0]), ("denoise", "fast", [True, False]),
          ("calibrate", "apply_masters", [False, True]), ("gradient", "nothing", [1]), ("stack", "zone", [None]),
          ("combine_sho", "lightness", ["noise", "equal"]), ("render_simple", "linked", [False])]
#: by block kind, the settings it holds (most gestures set one of its own)
OWN = {"registration": [("register", "min_pairs", [3, 10, 20, 2])], "selection": [("stack", "min_transparency", [0.6])],
       "stacking": [("stack", "method", ["drizzle", "classic"]), ("stack", "kappa", [2.5])],
       "crop": [("stack", "zone", [None])], "denoise": [(None, "fast", [True, False])],
       "gradient": [(None, "nothing", [1])], "calibration": [("calibrate", "apply_masters", [False, True])],
       "combine": [(None, "lightness", ["noise", "equal"])], "astrometry": [("astrometry", "nothing", [1])]}
METHOD_CHOICES = ["polynomial", "rbf", "automatic", "pretrained", "adapted", "rl", "neural", "measured", None, "x"]


def _param(rng, p, b):
    blk = p.blocks[b] if b else None
    if blk is not None and blk.kind in OWN and rng.random() < 0.8:
        st, name, vals = rng.choice(OWN[blk.kind])
        if st is None:
            st = ops.owned_stages(p, b)[0] if blk.kind == "combine" else (blk.engine or blk.kind)
        return st, name, rng.choice(vals)
    st, name, vals = rng.choice(PARAMS)
    return st, name, rng.choice(vals)


def _change(rng, p, b):
    k = rng.random()
    if k < 0.45:
        return {"param": list(_param(rng, p, b))}
    if k < 0.7:
        return {"method": pick(rng, METHOD_CHOICES)}
    if k < 0.95:
        return {"off": rng.random() < 0.5}
    return {"nothing": 1}


def _try_variant(rng, p, blocks):
    ix = Index(p)
    b = pick(rng, [x for x in blocks if x in ix.row_of and p.blocks[x].kind not in ("session", "image", "combine",
                                                                                    "combine_layers")] or blocks)
    return ("op", "try_variant", {"block": b, "change": _change(rng, p, b), "name": pick(rng, NAMES + [None])}, None)


def _set_param(rng, p, blocks):
    b = pick(rng, blocks)
    st, name, v = _param(rng, p, b)
    return ("op", "set_param", {"block": b, "stage": st, "name": name, "value": v}, None)


#: the gallery's templates (client/gallery.GALLERY, its data only: the fuzz runs without Qt)
_FULL = dict(debruitage=True, etoiles="retirer", fond="simple")
_CLASSIC = dict(debruitage=False, etoiles="garder", fond="simple")
GALLERY = [dict(key="colour_min", name="Colour minimal", opts=_CLASSIC),
           dict(key="colour_full", name="Colour full", opts=_FULL),
           dict(key="image", name="Image", opts=_CLASSIC, source="image", plain=True, blocks=[]),
           dict(key="filters_min", name="Filters minimal", opts=_CLASSIC, kind="filters", palette="H-O",
                filters=["Ha", "OIII"]),
           dict(key="filters_full", name="Filters full", opts=_FULL, kind="fusion", palette="S-H-O",
                filters=["SII", "Ha", "OIII"]),
           dict(key="sho_rgb", name="SHO + RGB stars", opts=_FULL, kind="fusion", palette="S-H-O",
                filters=["SII", "Ha", "OIII"], rgb_stars=True),
           dict(key="images", name="Several images", opts=_CLASSIC, source="image", plain=True, blocks=[], images=3),
           dict(key="dual_x2", name="Dual-band x2", opts=_FULL, kind="fusion", palette="S-H-O",
                filters=["O III / H-alpha", "O III / S II"], dual_x2=True)]


def _template(rng, p):
    """A gallery template made on the graph as it is: its steps as one gesture."""
    from astro_gpu.graphcore import templates as tpl
    t = rng.choice(GALLERY)
    plan = tpl.build(p, t, t.get("palette") if t.get("kind") == "fusion" else t["name"], dict(t["opts"]))
    return ("many", plan.steps)


def _share(rng, p, sess):
    stacks = [r.id for r in p.rows if r.nights and not r.fork]
    return ("op", "share_night", {"night": pick(rng, sess), "row": pick(rng, stacks or [r.id for r in p.rows])}, None)


def _set_input(rng, p, joins):
    j = pick(rng, joins or list(p.blocks))
    ports = [w.dst[1] for w in p.wires if j and w.dst[0] == j and w.active] if rng.random() < 0.85 else []
    ports = ports or (in_ports(p.blocks[j]) if j else [])
    kw = {"block": j, "port": pick(rng, ports or ["R"])}
    r = rng.random()
    if r < 0.4:
        kw["channel"] = pick(rng, ["L", "R", "G", "B", "GB", "Q"])
    elif r < 0.7:
        kw["weight"] = pick(rng, [0.5, 1.0, 2.0, -1, "x"])
    else:
        kw["image"] = pick(rng, ["output", "starless", "stars", "other"])
    return ("op", "set_input", kw, None)


def _split(rng, p, sess):
    s = pick(rng, sess)
    own = p.blocks[s].files.get("lights", []) if s else []
    return ("op", "split_night", {"night": s, "lights": rng.sample(own, rng.randint(0, len(own))) if own
                                  else ["absent.fit"]}, None)


def _split_by(rng, p, sess):
    s = pick(rng, sess)
    own = [x for v in (p.blocks[s].files.values() if s else []) for x in v]
    groups = []
    for _i in range(rng.randint(1, 3)):
        part = rng.sample(own, min(len(own), rng.randint(0, 3))) if own else ["absent.fit"]
        groups.append({"into": pick(rng, ["here", "common", "night", "row"]),
                       "files": {pick(rng, FILE_KINDS): part}, "name": pick(rng, NAMES + [None])})
    return ("op", "split_by", {"session": s, "groups": groups}, None)


# ---------------------------------------------------------------------------------------------------- running
def template(m):
    return m if isinstance(m, str) else m[0] if m else ""


REVERSE = {}    # --survey: {(where, template): [count, first example]}
ENGINE_ON = [True]
SURVEY = [False]


def reverse_check(prev, view, steps, res):
    """A refused gesture: the engine refuses the state it would have made, or the refusal is listed (CORE_ONLY,
    GESTURE). -> a problem text or ""."""
    if len(steps) == 1 and steps[0][0] in ops.VIEW_OPS:
        return ""
    state, where, why = ops.attempt(prev, steps, view)
    t = template(why)
    if where == "op":
        bad = "" if t in GESTURE or t in ENGINE_SAID else f"operation refusal not listed: {t!r}"
    elif where == "fold":
        bad = "" if t in CORE_ONLY else f"fold refusal not listed: {t!r}"
    elif where == "ok":
        bad = f"apply refused ({res.reason_text()}) but the gesture is accepted when tried again"
    else:
        eng = engine_problems(state, keys=False)
        bad = "" if eng or t in CORE_ONLY else f"core refuses, the engine accepts: {say(why)!r}"
    if bad and SURVEY[0]:
        REVERSE.setdefault((where, t if not bad.startswith("core") else say(why)[:90]), [0, steps])[0] += 1
        return ""
    return bad


def run_one(doc, g, before_d):
    """Execute gesture g on doc (may replace doc on save/load); verify; return (doc, drawing)."""
    p0 = doc._json
    prev = doc.project
    nu, nr = len(doc._undo), len(doc._redo)
    if g[0] == "undo":
        doc.undo()
        res, op, place = None, None, None
    elif g[0] == "redo":
        doc.redo()
        res, op, place = None, None, None
    elif g[0] == "save":
        text = doc.dumps()
        doc2 = Doc.loads(text)
        if doc2.dumps() != text:
            raise Fail("save/load not byte-identical")
        doc2._undo, doc2._redo = doc._undo, doc._redo
        doc = doc2
        res, op, place = None, None, None
    else:
        if g[0] == "many":
            steps = [tuple(x) for x in g[1]]
            res = doc.apply_many(steps)
            op, place = None, None
        else:
            _k, op, args, place = g
            steps = [(op, args)]
            if place is not None:                       # judged on the current state (replays, shrinking)
                place = judge(doc.project, doc.view, copy.copy(place))
            view0 = dict(doc.view)
            res = doc.apply(op, **args)
            if place is not None and place.lit != res.ok:
                raise Fail(f"place lit={place.lit} ({place.reason_text()}) but apply ok={res.ok} ({res.reason_text()})")
        if model_json(prev) != p0:
            raise Fail("apply changed the graph it was given")
        if not res.ok:
            if doc._json != p0 or len(doc._undo) != nu or len(doc._redo) != nr:
                raise Fail(f"refusal changed something: {res.reason_text()}")
            if ENGINE_ON[0]:
                bad = reverse_check(prev, view0 if g[0] != "many" else doc.view, steps, res)
                if bad:
                    raise Fail(f"reverse: {bad}")
        elif doc._json != p0:
            if len(doc._undo) != nu + 1 or doc._redo:
                raise Fail("accepted change did not push exactly one step")
            after = doc._json
            doc.undo()
            if doc._json != p0:
                raise Fail("undo did not restore the previous model")
            doc.redo()
            if doc._json != after:
                raise Fail("redo did not restore the model")
    check(doc.project)
    d = layout(doc.project, doc.view)
    probs = inv.static(doc, d) + [f"structure: {x}" for x in structure(doc.project)]
    text = doc.dumps()
    loaded = Doc.loads(text)
    if loaded.dumps() != text:
        probs.append("save/load not byte-identical")
    probs += inv.i6(doc, d, loaded)
    if res is not None and res.ok and op:
        probs += inv.i4(before_d, d, op, res.acted)
        probs += inv.i5(d, res.focus, place, doc.project)
    if ENGINE_ON[0] and res is not None and res.ok:
        eng = [] if SURVEY[0] else engine_problems(doc.project)      # --survey: collected by fuzz()
        if eng:
            probs.append("engine: " + "; ".join(f"[{c}] {d}" for c, d in eng[:3]))
    if probs:
        raise Fail("; ".join(probs[:5]))
    return doc, d


def start_doc(key):
    return Doc.from_template(key)


def replay(key, gestures):
    doc = start_doc(key)
    d = layout(doc.project, doc.view)
    for g in gestures:
        doc, d = run_one(doc, g, d)
    return doc


def fails(key, gestures, sig):
    try:
        replay(key, gestures)
    except Exception as e:
        return _sig(e) == sig
    return False


def _sig(e):
    msg = e.args[0] if isinstance(e, Fail) and e.args else type(e).__name__
    return str(msg).split(" ")[0] + (str(msg).split(" ")[1] if " " in str(msg) else "")


def shrink(key, gestures, sig, budget=90.0):
    t0 = time.time()
    cur = list(gestures)
    n = 2
    while len(cur) > 1 and time.time() - t0 < budget:
        chunk = max(1, len(cur) // n)
        done = False
        for i in range(0, len(cur), chunk):
            cand = cur[:i] + cur[i + chunk:]
            if cand and fails(key, cand, sig):
                cur, n, done = cand, max(n - 1, 2), True
                break
        if not done:
            if chunk == 1:
                break
            n = min(len(cur), n * 2)
    return cur


def show(g):
    if g[0] == "op":
        return f"{g[1]}({g[2]})" + (f" place={g[3].kind}@{(g[3].line, g[3].col)}" if g[3] else "")
    if g[0] == "many":
        return f"apply_many({g[1]})"
    return g[0]


def probe(rng, doc, d):
    """The view's and the table's queries must answer on any reachable state (and change nothing)."""
    js = doc._json
    b = pick(rng, doc.project.blocks)
    doc.title(b), doc.summary(b), doc.greyed(b), doc.differs(b), doc.effective(b)
    r = pick(rng, [x.id for x in doc.project.rows])
    if r:
        doc.title(r), doc.summary(r), doc.warnings(r)
    doc.stacks(), doc.warnings(), doc.followers(pick(rng, FILE_KINDS)), doc.followers(pick(rng, ORDER))
    k = pick(rng, ORDER)
    pls = graphcore.places(doc.project, doc.view, kind=k)
    for i, a in enumerate(pls):                         # one pixel, one place: a drop cannot hit two
        for c in pls[i + 1:]:
            A, C = a.rect, c.rect
            if A.x < C.right and C.x < A.right and A.y < C.bottom and C.y < A.bottom:
                raise Fail(f"places overlap for {k}: {a.kind} {a.op[1]} / {c.kind} {c.op[1]}")
    if any(pl.lit for pl in pls) and not doc.palette_state(k)[0]:
        raise Fail(f"palette card {k} greyed while a place is lit")
    w = pick(rng, [x.id for x in d.wires])
    if w:
        doc.wire_tip(w)
    if doc._json != js:
        raise Fail("a query changed the model")


STATS = []      # wire crossings of each state of the current run
ENGINE = {}     # --engine: {(class, detail): {"start", "seed", "at", "gestures"}}, the first sequence showing each


def engine_classes(p):
    return {(c, d) for c, d in engine_problems(p)}


def light_replay(key, gestures):
    """The gestures replayed without the drawing's invariants (shrinking only needs the final graph)."""
    doc = start_doc(key)
    for g in gestures:
        if g[0] == "undo":
            doc.undo()
        elif g[0] == "redo":
            doc.redo()
        elif g[0] == "many":
            doc.apply_many(g[1])
        elif g[0] == "op":
            doc.apply(g[1], **g[2])
    return doc


def shows(key, gestures, cls):
    """Whether replaying the gestures ends on a state whose compiled definition shows the class."""
    try:
        doc = light_replay(key, gestures)
    except Exception:  # noqa: BLE001
        return False
    return cls in engine_classes(doc.project)


def shrink_engine(key, gestures, cls, budget=45.0):
    t0 = time.time()
    cur = list(gestures)
    n = 2
    while len(cur) > 1 and time.time() - t0 < budget:
        chunk = max(1, len(cur) // n)
        done = False
        for i in range(0, len(cur), chunk):
            cand = cur[:i] + cur[i + chunk:]
            if cand and shows(key, cand, cls):
                cur, n, done = cand, max(n - 1, 2), True
                break
        if not done:
            if chunk == 1:
                break
            n = min(len(cur), n * 2)
    return cur


def fuzz(seed, key, n, engine=False):
    if engine and not SURVEY[0]:
        engine = False                              # the invariant stops the run (run_one): nothing to collect
    rng = random.Random(f"{seed}:{key}")
    doc = start_doc(key)
    d = layout(doc.project, doc.view)
    done = []
    STATS[:] = []
    for i in range(n):
        g = gen(rng, doc)
        done.append(g)
        try:
            doc, d = run_one(doc, g, d)
            STATS.append(inv.crossings(d))
            if i % 10 == 0:
                probe(rng, doc, d)
            if engine:
                for cls in engine_classes(doc.project) - set(ENGINE):
                    ENGINE[cls] = {"start": key, "seed": seed, "at": i, "gestures": list(done)}
        except Exception as e:
            msg = e.args[0] if isinstance(e, Fail) else traceback.format_exc(limit=-6)
            print(f"FAIL seed={seed} start={key} gesture {i}: {msg}")
            small = shrink(key, done, _sig(e)) if not os.environ.get("NOSHRINK") else done[-3:]
            print(f"  shrunk to {len(small)} gestures:" + (" (last 25 shown)" if len(small) > 25 else ""))
            for x in small[-25:]:
                print("   ", show(x))
            return False
    return hashlib.md5(doc.dumps().encode("utf-8")).hexdigest()[:10]   # final state: a run is repeatable


def main():
    try:                                            # a long run never takes the machine from the others
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 10)
    except Exception:  # noqa: BLE001
        pass
    argv = list(sys.argv[1:])
    ENGINE_ON[0] = "--no-engine" not in argv
    SURVEY[0] = "--survey" in argv
    engine = ENGINE_ON[0]
    out = argv[argv.index("--json") + 1] if "--json" in argv else None
    argv = [x for i, x in enumerate(argv) if x not in ("--engine", "--no-engine", "--survey", "--json")
            and (i == 0 or argv[i - 1] != "--json")]
    a = int(argv[0]) if len(argv) > 0 else 1
    b = int(argv[1]) if len(argv) > 1 else 5
    n = int(argv[2]) if len(argv) > 2 else 2000
    keys = [k for k, _t, _d in templates()]
    keys = ["blank"] + [k for k in keys if k != "blank"]
    if len(argv) > 3:
        keys = argv[3].split(",")
    total, bad = 0, 0
    t0 = time.time()
    for seed in range(a, b + 1):
        for key in keys:
            ok = fuzz(seed, key, n, engine)
            total += 1
            bad += not ok
            cr = f"crossings mean {sum(STATS) / max(1, len(STATS)):.1f} max {max(STATS or [0])}"
            print(f"seed {seed} {key}: {'ok ' + ok if ok else 'FAILED'} {cr} ({time.time() - t0:.0f} s)", flush=True)
    print(f"{total - bad}/{total} runs clean, {n} gestures each" + (" (engine invariants on)" if engine else ""))
    for (where, t), (cnt, steps) in sorted(REVERSE.items(), key=lambda kv: -kv[1][0]):
        print(f"REVERSE [{where}] x{cnt}: {t}   e.g. {steps}"[:400])
    if engine and SURVEY[0]:
        found = []
        if out:
            import json
            with open(out, "w", encoding="utf-8") as f:      # first unshrunk: a long shrink may be cut short
                json.dump([{"class": c, "detail": d, "start": e["start"], "seed": e["seed"],
                            "gestures": [show(x) for x in e["gestures"][:e["at"] + 1]]}
                           for (c, d), e in sorted(ENGINE.items())], f, indent=1, ensure_ascii=False)
        for (cls, detail), e in sorted(ENGINE.items()):
            small = shrink_engine(e["start"], e["gestures"][:e["at"] + 1], (cls, detail))
            found.append({"class": cls, "detail": detail, "start": e["start"], "seed": e["seed"],
                          "gestures": [show(x) for x in small]})
            print(f"ENGINE REFUSES [{cls}] {detail}  (from {e['start']}, {len(small)} gestures)")
            for x in small:
                print("   ", show(x))
        print(f"{len(ENGINE)} classes the engine refuses in states the core accepts")
        if out:
            import json
            with open(out, "w", encoding="utf-8") as f:
                json.dump(found, f, indent=1, ensure_ascii=False)
    return bad == 0


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
