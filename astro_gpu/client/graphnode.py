"""The blocks of the graph as the run, Inspect, the Report and the panel know them, apart from the drawing: the block
table (``STEPS``, ``BY_KEY``, ``TOOLS``), the views' block keys and their titles, the state words and their order, the
block-level answers over a project model (stages, state, duration, summary), ``Node`` (one block over the workflows
going through it) and ``BlockMap`` (the blocks of a stored graph, each as a ``Node``, from the compiled definition).

It imports neither the model nor the graph page, so that both can import it (``BlockMap`` takes the model as an
argument). The constants and the functions above ``Node`` are the same as those of ``blocks.py``, which the graph
page of today still reads: tests/client/graphview/test_graphnode.py checks that they stay equal.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..stages.combine import BY_KIND as COMBINE_KINDS
from ..stages.combine import STAGES as COMBINE
from .i18n import _, plural

# the columns of the graph, left to right; optional columns only appear when a workflow uses them
COLUMNS = ["sources", "calibration", "alignement", "selection", "crop", "empilement", "align_on", "astrometrie", "fond",
           "couleur", "deconv", "debruitage", "etoiles", "synth", "post", "fusion", "composition", "final"]
OPTIONAL = {"crop", "debruitage", "deconv", "etoiles", "synth", "post", "fusion", "composition"}
PHASES = [("Pre-processing", "calibration", "empilement"), ("Linear processing", "astrometrie", "etoiles"),
          ("Composition", "fusion", "composition")]
TOOL_COLUMN = {"debruitage": "debruitage", "etoiles": "etoiles", "deconv": "deconv",
               "auto": "fond", "rbf": "fond", "simple": "fond"}
TITLES = {"align_on": "Align on", "sources": "Frames", "final": "Post", "image": "Image", "session": "Session",
          "fusion": "Combination",
          "calibration": "Calibration",
          "alignement": "Registration",
          "selection": "Selection", "crop": "Crop", "empilement": "Stacking", "astrometrie": "Astrometry", "fond": "Gradient",
          "couleur": "Color", "debruitage": "Denoise", "deconv": "Deconvolution",
          "etoiles": "Stars/background", "synth": "Synthetic stars",
          "post": "Output", "demi_piles": "Half-stacks", "composition": "Combine layers"}
#: a combination's named inputs as the user reads them (stages/combine.py ports)
PORT_NAMES = {"SII": "S II", "Ha": "Hα", "OIII": "O III", "L": "L", "R": "R", "G": "G", "B": "B"}


def combine_kind(q) -> str:
    """The type of the combination a workflow starts from ("sho"...), "" for another workflow."""
    f = (getattr(q, "source", None) or {}).get("fusion") if isinstance(getattr(q, "source", None), dict) else None
    return str(f.get("type") or "") if isinstance(f, dict) else ""


def combine_title(kind: str) -> str:
    """A type of combination's block name (S-H-O, H-O, RGB, LRGB), translated; the group's for none."""
    return _(COMBINE_KINDS[kind].title) if kind in COMBINE_KINDS else _("Combination")


def port_name(role: str) -> str:
    """A named input's name (S II, Hα, O III, R...), translated (G is V in French)."""
    return _(PORT_NAMES.get(role, role))


# the optional blocks, in the toolbar's order
TOOLBOX = ["debruitage", "deconv", "etoiles"]
WARN_MARK = "⚠ "                       # a summary that opens with it: the block draws a warning icon (issue #58)
#: the two images of the Stars block (the parts of Post's layers) and their words on the wires
STAR_OUTPUTS = (("starless", "Starless image"), ("stars", "Stars alone"))
# the engine's block names (core/workflow.BLOCKS) and the views' keys
ENGINE_BLOCK = {"fusion": "fusion", "align_on": "align_on", "image": "image", "calibration": "calibration",
                "registration": "alignement",
                "selection": "selection", "crop": "crop",
                "stacking": "empilement", "astrometry": "astrometrie", "color": "couleur", "denoise": "debruitage",
                "deconv": "deconv", "synth": "synth", "stars": "etoiles", "gradient": "fond", "output": "post"}
KEY_ENGINE_BLOCK = {v: k for k, v in ENGINE_BLOCK.items()}
# the wires that are not the chain: what a block takes from a block further back, shown when it is selected
SIDE_INPUTS = {
    "debruitage": [("empilement", "half-stacks")],
    "couleur": [("astrometrie", "Gaia spectra")],
}
TRUNK = ["inventory", "calibrate", "register", "stack"]
#: the image blocks: placed in any order and more than once (issue #47), drawn in columns by their rank
IMAGE_KEYS = {"align_on", "astrometrie", "fond", "couleur", "debruitage", "deconv"}
#: the frame blocks, in their columns
FRAME_KEYS = ["calibration", "alignement", "selection", "crop", "empilement"]

# every state a block can be in: up to date, pending, outdated, and what a run adds
STATE_WORD = {"done": "Up to date", "pending": "Not computed yet", "stale": "Outdated", "running": "Running",
              "failed": "Failed", "interrupted": "Interrupted", "skipped": "Skipped"}
#: a block done with its declared fallback (Graph.fallback): drawn in orange, its reasons after this
FALLBACK_WORD = "Done with fallback"
# worst first: a block is as bad as its worst stage in any of its workflows
STATE_ORDER = ["running", "failed", "interrupted", "stale", "skipped", "pending", "done"]

# the review points ("Review pause"): a dot like a debugger's, red while the run will stop there, yellow with
# the pause bars while the run is stopped there for the user, green once the user has checked these results
BREAK_COL, PAUSE_COL, REVIEWED_COL = "#e5484d", "#f5c542", "#5fbf7f"
# the most urgent first: what a block shows when its workflows are not all in the same case
REVIEW_ORDER = ["paused", "pending", "reviewed"]

# the block table: key, phase, title, kind (fixed | tool | auto | slot), engine stages, what it does
STEPS = [
    dict(key="session", phase=0, title="Session", kind="fixed", stages=[],
         what="One night: its light frames and the dark, flat and bias frames chosen for it, and the filter."),
    dict(key="image", phase=0, title="Image", kind="fixed", stages=["image"],
         what="An image file (a linear stack) this workflow starts from: the blocks after it read it as a stack."),
    dict(key="align_on", phase=3, title="Align on", kind="slot", stages=["align_on"],
         what="Brings this workflow's image onto the grid of another workflow's image, through their stars or their plate "
              "solutions: it takes the size and framing of its reference, what overflows is cut. Placed right after "
              "Stacking, while the stars are there."),
    dict(key="calibration", phase=1, title="Calibration", kind="fixed", stages=["inventory", "calibrate"],
         what="Dark and flat masters, then each light calibrated: thermal signal, vignetting and defective pixels "
              "removed."),
    dict(key="alignement", phase=2, title="Registration", kind="fixed", stages=["sessions", "register"],
         what="Measures the stars of each frame (size, transparency) and aligns all frames on the sharpest one."),
    dict(key="selection", phase=2, title="Selection", kind="fixed", stages=[], run=["register"],
         what="Which frames go into the stack: by star size and transparency thresholds, or one by one."),
    dict(key="crop", phase=2, title="Crop", kind="fixed", stages=[], run=["register"],
         what="The part of the field that is stacked and processed: a small target no longer costs the whole frame. "
              "Calibration and Registration keep the whole field."),
    dict(key="empilement", phase=3, title="Stacking", kind="fixed", stages=["stack"],
         what="Stacks the frames straight from the sensor's color pixels (drizzle without debayering) and rejects "
              "trails and outlier pixels."),
    dict(key="demi_piles", phase=3, title="Half-stacks", kind="auto", stages=[], run=["stack"],
         what="Two independent stacks of about 360 frames each: their difference measures the noise, which lets "
              "Denoise work without inventing anything. Computed during stacking, at no extra time."),
    dict(key="astrometrie", phase=4, title="Astrometry", kind="fixed",
         stages=["astrometry", "reference", "catalogs"],
         what="Finds where the image points on the sky by matching its stars to Gaia, then fetches the catalogue "
              "stars."),
    dict(key="couleur", phase=4, title="Color", kind="fixed", stages=["color"],
         what="Measures the sensor's response on the Gaia spectra of the field stars, then the matrix to the true "
              "sky colors."),
    dict(key="synth", phase=4, title="Synthetic stars", kind="fixed", stages=["synth_stars"],
         what="Draws the detected stars again, with the shape of this image or of the instrument."),
    dict(key="deconv", phase=4, title="Deconvolution", kind="tool", tool="deconv", stages=["deconv"],
         what="Sharpens the image with the point spread function measured on the stars."),
    dict(key="debruitage", phase=4, title="Denoise", kind="tool", tool="debruitage", stages=["denoise"],
         what="A network learns on the two half-stacks to predict one from the other: it removes the noise without "
              "inventing anything."),
    dict(key="etoiles", phase=4, title="Stars/background", kind="tool", tool="etoiles", stages=["star_removal"],
         what="A network trained on our stacks takes the stars off the image: two images, the starless image and the "
              "stars alone, sent to Post as they are or combined by a Combine layers block."),
    dict(key="fond", phase=4, title="Gradient", kind="slot", stages=["gradient"],
         what="Models the sky gradient (light pollution, the Moon, vignetting) on the stack and removes it from what "
              "every later block reads. Polynomial: a smooth surface fitted on the sky between the objects. "
              "Automatic: a more flexible surface that follows complex gradients."),
    dict(key="post", phase=5, title="Output", kind="fixed", stages=["render_simple"],
         what="Prepares the render: what this workflow gives in Post, its image and its sliders."),
    # the combinations of filters (stages/combine.py): one block per type, its title and its settings its stage's
    dict(key="fusion", phase=6, title="Combination", kind="slot", stages=list(COMBINE),
         what="Combines the images of several workflows (one per filter, or a dual-band image's channels) into one "
              "colour image, aligned through their plate solutions: S-H-O, H-O, RGB or LRGB, by what was "
              "photographed."),
    dict(key="composition", phase=6, title="Combine layers", kind="slot", stages=[],
         what="Combines the Stars/background block's two images: the starless image and the stars, each stretched, the stars "
              "put back on top (Screen, Linear dodge or Lighten). Its settings are those of its layer in Post."),
]
BY_KEY = {b["key"]: b for b in STEPS}
# what the optional blocks say they do (the toolbar cards' tooltips)
TOOLS = {
    "debruitage": dict(title="Denoise", phase=4,
                       what="Removes the noise without inventing anything: a network learns on the two half-stacks "
                            "to predict one from the other."),
    "etoiles": dict(title="Stars/background", phase=4,
                    what="Takes the stars off the image with a trained network: the starless image and the stars "
                         "alone, two images for Post (combined by a Combine layers block, or as they are)."),
    "synth": dict(title="Synthetic stars", phase=4,
                  what="Draws the detected stars again, with the shape of this image or of the instrument."),
    "deconv": dict(title="Deconvolution", phase=4,
                   what="Sharpens the image with the point spread function measured on the stars."),
}


def kbase(key: str) -> str:
    """A view's block key without its session or its number: "calibration@2" → "calibration", "debruitage~2" →
    "debruitage"."""
    return key.partition("@")[0].partition("~")[0]


def knum(key: str) -> str:
    """The number of a graph's later node of a block ("2" for "debruitage~2", issue #47), "" otherwise."""
    return key.partition("@")[0].partition("~")[2]


def block_title(p, key: str) -> str:
    """The name of a block, translated. The output of a workflow carries the workflow's name; a block placed twice
    in a graph, its number ("Denoise 2")."""
    if key == "post" and p is not None:
        return p.name
    n = knum(key)
    base = kbase(key)                                 # a session's block: its block's name
    t = _(TITLES.get(base) or BY_KEY[base]["title"])
    if p is not None and getattr(p, "model", None) is not None:   # numbered left to right in its workflow, the
        try:                                                       # same name everywhere (GUI exploration 4.1)
            same = [k for k in chain(p) if kbase(k) == base]
        except (KeyError, ValueError, AttributeError):
            same = []
        if key in same:
            return f"{t} {same.index(key) + 1}" if len(same) > 1 else t
    return f"{t} {n}" if n else t


def calibration_counts(darks, flats, bias=None, gone=()) -> str:
    """"3 darks · 1 flat (· 0 bias)": each count with its noun in the right form, as the lights' (GUI exploration 2).
    A kind in ``gone`` (``gone_kinds``) without a frame: its folder is not found, not empty (GUI exploration 4)."""
    def one(kind, n, counted, missing):
        return _(missing) if kind in gone and not int(n or 0) else counted
    parts = [one("darks", darks, plural(int(darks or 0), "{n} dark", "{n} darks"), "darks not found"),
             one("flats", flats, plural(int(flats or 0), "{n} flat", "{n} flats"), "flats not found")]
    if bias is not None:
        parts.append(one("bias", bias, plural(int(bias or 0), "{n} bias", "{n} bias"), "bias not found"))
    return " · ".join(parts)


def gone_kinds(s, sids=None) -> set[str]:
    """The kinds of frames (lights, darks...) of these sessions (all when None) one of whose folders is not there:
    moved or renamed outside, its disk unplugged (GUI exploration 4). Each folder is looked at once, never each file."""
    if not hasattr(s, "session") or getattr(s, "p", None) is None:
        return set()
    out: set[str] = set()
    for se in s.p.sessions:
        if sids is not None and se.id not in sids:
            continue
        for kind in ("lights", "darks", "flats", "bias", "flat_darks"):
            folders = {(p if not p.suffix else p.parent) for p in (Path(s.p.abspath(e)) for e in getattr(se, kind))}
            if any(not f.exists() for f in folders):
                out.add(kind)
    return out


def own_calibration_summary(s, p, sid: int) -> str:
    """Under a Calibration placed in parallel: its counts, and whether they are its own or still its Session's."""
    n = {k: len(s.calibration_files(p, sid, k)) for k in ("darks", "flats")}
    follows = all(s.calibration_follows(p, sid, k) for k in ("darks", "flats", "bias", "flat_darks"))
    return (_("as Session") if follows else _("own files")) + " · " + calibration_counts(n["darks"], n["flats"])


def session_summary(s, sid: int) -> str:
    """Under a Session block's name: its lights, its calibration frames, its filter; a kind whose folder is not
    found says so (GUI exploration 4)."""
    ses = next((x for x in s.sessions if x["id"] == sid), {})
    gone = gone_kinds(s, {sid})
    if not int(ses.get("lights") or 0):                # issue #48: a Session placed empty says what it needs
        se = s.session(sid) if hasattr(s, "session") else None
        flt = getattr(se, "filter", "")
        return (_("lights folder not found") if "lights" in gone else _("lights to add")) + (f" · {flt}" if flt else "")
    text = plural(int(ses.get("lights") or 0), "{n} light", "{n} lights")
    text += " · " + calibration_counts(ses.get("darks", 0), ses.get("flats", 0), gone=gone)
    flt = getattr(s.session(sid), "filter", "") if hasattr(s, "session") else ""
    warn = WARN_MARK if hasattr(s, "file_warnings") and s.file_warnings(sid) else ""   # issue #58: see its panel
    return warn + text + (f" · {flt}" if flt else "")


def node_title(s, n) -> str:
    """The name a block shows: a session's name for its Session block, the workflow's for its Output, the frames'
    block named after its one session (or "All nights")."""
    if n.key == "session":
        return s.session(n.sid).name or _("Session {n}").format(n=n.sid)
    if n.key == "post" and not n.handle:              # issue #47: the block's name; the workflow's under it
        return _("Output")
    if n.key == "composition" and n.num:              # several Combine layers blocks: all numbered, top to bottom, as
        return f"{block_title(None, 'composition')} {n.num}"     # Gradient 1, Gradient 2 (control round)
    if n.num and n.key in FRAME_KEYS:                 # frame blocks placed in parallel, the same way
        return f"{block_title(None, n.key)} {n.num}"
    if n.key == "image" and n.paths and n.paths[0].name:   # its workflow's name, which carries the number (#155: one
        return n.paths[0].name                              # source; numbered by place, a deletion shifted them)
    if n.key == "fusion" and n.paths:                 # a combination: its type's name (S-H-O...); several of a type:
        kind = combine_kind(n.paths[0])               # numbered in the project's order (exploration 2, control round);
        bk = getattr(s, "block_key", None)            # a branch shares its source's block: counted once (round 4)
        ident = (lambda q: bk(q, n.key)) if bk is not None else (lambda q: q.pid)
        same = []
        for q in getattr(s, "paths", []):
            if combine_kind(q) == kind and ident(q) not in same:
                same.append(ident(q))
        mine = ident(n.paths[0])
        title = combine_title(kind)
        return f"{title} {same.index(mine) + 1}" if len(same) > 1 and mine in same else title
    if n.paths and "~" in n.key or (n.paths and n.key in ("fond", "couleur", "debruitage")):
        # issue #47: a block placed twice, numbered left to right; shared by a branch that has it twice: numbered
        # as in that branch ("Gradient 1" before its "Gradient 2", control round)
        base = kbase(n.key)
        p = max(n.paths, key=lambda q: sum(kbase(k) == base for k in _safe_chain(q)))
        return block_title(p, n.key)
    if n.key == "sources" and hasattr(s, "session") and s.sessions:
        if len(s.sessions) == 1:
            return s.sessions[0]["name"] or _("Session")
        if getattr(s, "stack_mode", "together") == "together":
            return _("All nights")
    return block_title(n.paths[0] if n.paths else None, n.key)


def review_title(key: str) -> str:
    """The name of a block in the texts about a pause (the output of a workflow is just "Output" there)."""
    return _("Output") if key == "post" else block_title(None, key)


def _safe_chain(p) -> list[str]:
    try:
        return chain(p) if getattr(p, "model", None) is not None else []
    except (KeyError, ValueError, AttributeError):
        return []


def chain(p) -> list[str]:
    """The block keys of a workflow, in order (the frames excluded, its Output last); a workflow pulled to the post
    from a block stops there, then has its output."""
    m = p.model
    return list(m.display_keys(p) if hasattr(m, "display_keys") else m.block_keys(p))


# ------------------------------------------------------------------ block-level answers over a model
def stages_of(s, p, key: str) -> list[str]:
    """The engine stages of this block in the workflow's chain; a session's block ("calibration@2"), that session's
    instances (issue #43)."""
    allst = s.all_stages(p)
    k, _at, sid = key.partition("@")
    k, _t, num = k.partition("~")                     # a graph's later node (issue #47): its own stage
    return [x for x in (f"{st}@{sid}" if sid else f"{st}~{num}" if num else st for st in BY_KEY[k]["stages"])
            if x in allst]


def run_stages(s, p, key: str) -> list[str]:
    """What Run to here runs for this block (Selection: Registration)."""
    return BY_KEY[kbase(key)].get("run") or stages_of(s, p, key)


def block_state(s, p, key: str) -> str:
    """The state of a block in one workflow: the worst of its stages (Selection follows Registration)."""
    st = run_stages(s, p, key)
    if not st:
        return "done"
    states = {s.state(p, x) for x in st}
    return next((w for w in STATE_ORDER if w in states), "pending")


def block_fallback(s, p, key: str) -> list[str]:
    """Why the stages of a block, done, took their declared fallback in one workflow (core/stage.Context.fallback: the
    engine's words, each once); [] when they did their usual work."""
    info = getattr(s, "state_info", None)
    out: list[str] = []
    for x in run_stages(s, p, key) if info is not None else ():
        st = info(p, x)
        why = getattr(st, "fallback", "") if st is not None and st.state == "done" else ""
        if why and why not in out:
            out.append(why)
    return out


def block_seconds(s, p, key: str) -> float:
    return sum(s.seconds(x, p) for x in stages_of(s, p, key))


def fmt_int(n) -> str:
    return f"{int(n):,}".replace(",", " ")


def drizzle_auto(s, p) -> bool:
    """Whether the Stacking's drizzle scale is the engine's automatic one and not known yet (the frames are not aligned):
    the block says "auto", never the default 1.5 the engine would then replace (GUI exploration 2: "×1,5" before the
    run, "×1" after, without any action)."""
    if "scale" in (p.params.get("stack") or {}) or not hasattr(s, "param_info"):
        return False
    info = s.param_info(p, "stack", "scale")
    return info is None or (info.origin != "user" and not info.suggestion_known)


def method_summary(s, p, key: str) -> str:
    from .model import method_of, methods_of
    m = method_of(p, key)                       # "fond~2": its own node's method (round 3)
    return _(next(x["short"] for x in methods_of(kbase(key)) if x["key"] == m))


@dataclass
class Node:
    key: str                                   # step key, or "sources" / "final"
    paths: list = field(default_factory=list)  # the workflows going through this block
    parent: Node | None = None
    children: list = field(default_factory=list)
    col: int = 0
    y: float = 0.0
    w: float = 0.0                             # its width in the graph (the column's)
    lid: int | None = None                     # a Stars and background block: its layer in Post
    handle: bool = False                       # the Stars block of a workflow whose Output block it holds (#17)
    sid: int | None = None                     # a Session block, or a session's Calibration block: its session
    cut: bool = False                          # cut by the user in its workflows (issue #43)
    grey: str = ""                             # why it cannot run (a cut before it), translated; "" when it can
    broken: str = ""                           # why it cannot run at all (a fusion's channel gone...): circled in red
    missing: str = ""                          # why it cannot run on this computer: the data of the sky it needs is
    #                                            not on the disk (core/sky_data.py); left out of the runs, the rest runs
    slot: int = -1                             # an image block: its rank among its workflow's image blocks (#47); a
    #                                            combination: its rank among the combinations it reads (#165)
    h: float = 0.0                             # its height when it is not the standard one (a join with many inputs)
    after: int = -1                            # a Stars/background after a combination: that combination's rank (#165)
    waits: str = ""                            # a join with an input to wire (or inputs not on one grid): why, translated;
    #                                            the block is greyed, left out of the runs with what follows, never red
    num: int = 0                               # a Combine layers block among several, or a frame block among those
    #                                            placed in parallel: its number (control round)
    _bk: object = None

    @property
    def ref(self) -> str:
        """The key the views select the block by: its key, with its session ("calibration@2")."""
        return f"{self.key}@{self.sid}" if self.sid is not None else self.key

    @property
    def shared(self) -> bool:
        return len(self.paths) > 1


# ------------------------------------------------------------------ the blocks of a stored graph
#: the views' key of a graph block kind that is not an engine node of its own name (the others: ``ENGINE_BLOCK``)
VIEW_OF_KIND = {"session": "session", "calibration": "calibration", "image": "image", "combine": "fusion",
                "combine_layers": "composition", "post": "final"}


def view_key(kind: str, name: str = "") -> str:
    """The views' key of a graph block: from the engine's node name it compiles to ("gradient~2" -> "fond~2"), else from
    its kind."""
    if kind in VIEW_OF_KIND:
        return VIEW_OF_KIND[kind]
    base, sep, n = (name or kind).partition("~")
    return ENGINE_BLOCK.get(base, base) + (f"~{n}" if sep else "")


class BlockMap:
    """The blocks of a stored graph as the run, Inspect, the Report and the panel know them: one ``Node`` per block id,
    over the workflows it is compiled into (``compiled.map``: {block id: [(workflow id, node name)]}, given by
    ``graphcore.compile.compile_into``). ``model`` gives the workflows (``paths``, each with its ``pid``) and, when it
    has them, the run answers (``all_stages``, ``state``); ``graph`` is the stored graph (block kinds, nights).

    ``node(block id)``, ``node_of(path, view key)``, ``nodes`` (the graph's block order), ``address(block id)`` ->
    (workflow id, view key) the selection mirrors to, ``state(node)`` (the worst over its workflows), ``summary``."""

    def __init__(self, compiled, model, graph=None):
        self.compiled, self.model, self.graph = compiled, model, graph
        paths = {getattr(p, "pid", None): p for p in getattr(model, "paths", None) or []}
        blocks = getattr(graph, "blocks", None) or {}
        separate = graph is not None and not getattr(graph, "calibrate_together", True)
        self._nodes: dict[str, Node] = {}
        self._ids: dict[int, str] = {}
        self._by_path: dict[tuple, Node] = {}
        for bid, pairs in (getattr(compiled, "map", None) or {}).items():
            blk = blocks.get(bid)
            kind = blk.kind if blk is not None else ""
            name = pairs[0][1] if pairs else ""
            key = view_key(kind or name, name)
            sid = None
            if blk is not None and (kind == "session" or (kind == "calibration" and separate)):
                sid = getattr(blk, "sid", None)
            n = Node(key, [paths[w] for w, _n in pairs if w in paths], sid=sid, cut=bool(getattr(blk, "off", False)))
            self._nodes[bid] = n
            self._ids[id(n)] = bid
            for w, _n in pairs:
                self._by_path.setdefault((w, n.ref), n)

    @property
    def nodes(self) -> list:
        """Every block's node, in the graph's block order."""
        return list(self._nodes.values())

    def node(self, block: str):
        return self._nodes.get(block)

    def block_of(self, node) -> str | None:
        """The block id of one of these nodes."""
        return self._ids.get(id(node))

    def node_of(self, p, key: str):
        """The node of a view key ("fond~2", "calibration@3") in one workflow, or None."""
        return self._by_path.get((getattr(p, "pid", p), key))

    def address(self, block: str):
        """(workflow id, view key) of a block: what the Inspect strip and the run window select it by; None for a block
        no workflow goes through (a Post, a row being built)."""
        pairs = (getattr(self.compiled, "map", None) or {}).get(block)
        n = self._nodes.get(block)
        if not pairs or n is None:
            return None
        return pairs[0][0], n.ref

    def state(self, node) -> str:
        """The worst state of a block over its workflows ("done" when the model has no run answers)."""
        if not node.paths or not hasattr(self.model, "all_stages") or not hasattr(self.model, "state"):
            return "done" if not node.paths else "pending"
        states = {block_state(self.model, p, node.ref) for p in node.paths}
        return next((w for w in STATE_ORDER if w in states), "pending")

    def summary(self, node, tr=None) -> str:
        """The one line under a block's name: the stored graph's own summary of the block."""
        bid = self.block_of(node)
        if bid is None or self.graph is None:
            return ""
        from ..graphcore import api
        return api.summary(self.graph, bid, tr or _)
