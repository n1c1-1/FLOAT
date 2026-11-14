"""Window fuzz of the new graph page: random REAL gestures (Qt events through gestures.py) on a GraphPage editing a
graph held in memory, and after EACH gesture checks on what is painted.

    python tools/fuzz/graphview_fuzz.py --seed 1 --starts 3 --gestures 1000 --out DIR
    python tools/fuzz/graphview_fuzz.py --replay DIR/<start>_seed<n>_minimal.json      (replays a gesture list)
    python tools/fuzz/graphview_fuzz.py --model ...      the page on the client's project model (GraphEditing), each
                                                         start compiled into a read-only project under DIR/projects

Gestures: a palette card dragged onto a place (lit or refused), a palette card double-clicked, a placed block dragged
onto a place, a wire pulled from an output port to an input (accepted or refused), a context-menu item, the delete
cross (then sometimes the toast's « Delete too »), keys D / Del / Ctrl+D / Ctrl+Z / Ctrl+Y / Ctrl+J, the display
control (Blocks | Folded | Table, and back), a click on a wire (then sometimes Del), real files and folders dropped on a
Session's targets (then sometimes a link of the toast), made in a folder under --out.

Checks after every gesture:
  a  no two block items (blocks, stack cards) overlap
  b  no painted wire passes behind a block other than its own two ends (their cards when folded)
  c  dropped = shown: what the gesture made (bridge.last_shown) is drawn, and fully on screen
  d  undo exact: after an accepted gesture Ctrl+Z gives back exactly the graph before (json_of), Ctrl+Y the one after
     (and a, b, f hold on both)
  e  a refused gesture shows a message (the toast, visible, not empty); an accepted place / port changes the graph
  f  what is painted is the core's Drawing (an item per box and card at its box, a wire item per path with its ends),
     the bridge's cached Drawing is the current one, no item of a drag is left; no Python exception, no Qt warning
A failure writes the replayable gesture list and a minimised one (delta debugging by replay), and, when the core's
own Drawing has the defect, the minimal list of document operations that reproduces it.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import re
import sys
import time
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)                                    # the worktree before any installed copy of the package
sys.path.insert(1, os.path.join(ROOT, "tests", "graphcore"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if __name__ == "__main__" and "--out" in sys.argv:          # the client's settings file goes with the run's output
    _o = sys.argv[sys.argv.index("--out") + 1]
    os.makedirs(_o, exist_ok=True)
    os.environ.setdefault("ASTRO_GPU_SETTINGS", os.path.join(_o, "settings.json"))

from fixtures import Doc as FixtureDoc  # noqa: E402
from PySide6.QtCore import QRectF, QtMsgType, qInstallMessageHandler  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from astro_gpu.client.graphview import menu as gv_menu  # noqa: E402
from astro_gpu.client.graphview.document import MemoryDocument  # noqa: E402
from astro_gpu.client.graphview.gestures import Gestures, pump  # noqa: E402
from astro_gpu.client.graphview.items import BlockItem, FileTarget, GhostItem, RubberWire, StackItem  # noqa: E402
from astro_gpu.client.graphview.page import GraphPage  # noqa: E402
from astro_gpu.graphcore.layout import layout as raw_layout  # noqa: E402  (uncached)
from astro_gpu.graphcore.model import FILE_KINDS, JOIN_KINDS  # noqa: E402
from astro_gpu.graphcore.operations import json_of  # noqa: E402

#: the start graphs, in the order --starts takes them (tests/graphcore/fixtures.py)
STARTS = ["blank", "sho_rgb", "colour_complete", "filters_minimal", "several_images", "filters_complete",
          "colour_minimal"]
SIZE = (1500, 900)
TOL = 0.6
#: menu items whose action is not a graph change (or opens an editor): no expectation
MENU_FREE = {"inspect", "rename", "mark_reviewed", "submenu"}
#: the real files dropped on Session targets, made once under <out>/files (names relative to it)
FILE_TREE = {"lights_a": ["IMG_0001.ARW", "IMG_0002.ARW", "IMG_0003.ARW", "IMG_0004.ARW", "PREVIEW_1.JPG",
                          "PREVIEW_2.JPG"],
             "lights_b": ["L_001.fits", "L_002.fits", "L_003.fits"],
             "nested/a": ["S_1.ARW", "S_2.ARW"], "nested/b": ["S_3.ARW"],
             "empty": [],
             "loose": ["IMG_9001.ARW", "IMG_9002.CR2", "notes.txt"]}
DROPS = ["lights_a", "lights_b", "nested", "empty", "loose/IMG_9001.ARW", "loose/IMG_9002.CR2", "loose/notes.txt"]
#: known defects of the CORE, skipped by their signature until the core answers (the request's words in
#: _work/s4a/CORE_REQUESTS.md); never a weakened check
KNOWN_CORE_CASES = [
    ("e", "a lit move_block place gives back the same graph (the own place of the block?), its toast says it acted: ''",
     "places(block=b) offers the block's own spot as a lit place (move_block at after=None / after=b itself); "
     "apply accepts it, gives back the same graph and still toasts “{lab} moved”"),
    ("e", "a lit move_night place gives back the same graph (the own place of the block?), its toast says it acted: ''",
     "the same for a Session dragged: places(block=session) offers its own spot as a lit move_night place"),
]

WEIGHTS = [("palette", 22), ("double", 6), ("move", 14), ("wire", 9), ("menu", 12), ("key", 12), ("cross", 5),
           ("wclick", 4), ("display", 4), ("ctrlj", 2), ("files", 7)]


class Failure(Exception):
    def __init__(self, check, msg, core=False):
        super().__init__(f"({check}) {msg}")
        self.check, self.msg, self.core = check, msg, core

    @property
    def sig(self):
        return signature(self.check, self.msg)


def signature(check, msg):
    """A failure's class: its check and its message without ids, numbers and quoted names."""
    m = re.sub(r"“[^”]*”|'[^']*'|\"[^\"]*\"", "_", msg)
    m = re.sub(r"\b[bwrcfn]:?\w*\d+\b|\b[bwr]\d+\b", "#", m)
    m = re.sub(r"-?\d+(\.\d+)?", "0", m)
    return f"{check}:{m[:90]}"


#: signature -> the request's words
KNOWN_CORE = {signature(c, m): r for c, m, r in KNOWN_CORE_CASES}


# ============================================================================================ geometry
def inter(a, b):
    return a[0] < b[0] + b[2] - TOL and b[0] < a[0] + a[2] - TOL and a[1] < b[1] + b[3] - TOL and b[1] < a[1] + a[3] - TOL


def segment_inside(p, q, r):
    """True when the segment p-q passes through the inside of rect r=(x,y,w,h) shrunk by 2 px (Liang-Barsky)."""
    x0, y0, x1, y1 = r[0] + 2, r[1] + 2, r[0] + r[2] - 2, r[1] + r[3] - 2
    if x1 <= x0 or y1 <= y0:
        return False
    dx, dy = q[0] - p[0], q[1] - p[1]
    t0, t1 = 0.0, 1.0
    for pp, qq in ((-dx, p[0] - x0), (dx, x1 - p[0]), (-dy, p[1] - y0), (dy, y1 - p[1])):
        if pp == 0:
            if qq < 0:
                return False
        else:
            t = qq / pp
            if pp < 0:
                if t > t1:
                    return False
                t0 = max(t0, t)
            else:
                if t < t0:
                    return False
                t1 = min(t1, t)
    return t0 < t1 - 1e-9


def ends_of(graph, ident):
    """The ids a wire end stands for: itself and its row (the card it is folded in)."""
    out = {ident}
    for r in graph.rows:
        if ident == r.id or ident in r.blocks or any(ident in n for n in r.nights):
            out.add(r.id)
    return out


def overlaps(rects):
    ids = sorted(rects, key=lambda i: rects[i][0])
    for n, a in enumerate(ids):
        A = rects[a]
        for b in ids[n + 1:]:
            B = rects[b]
            if B[0] >= A[0] + A[2]:
                break
            if inter(A, B):
                return a, b
    return None


def crossing(polys, rects, ends):
    """(segment, block) of a polyline passing behind a block that is not one of its ends, or None."""
    for pts in polys:
        for p, q in zip(pts, pts[1:]):
            lx, hx, ly, hy = min(p[0], q[0]), max(p[0], q[0]), min(p[1], q[1]), max(p[1], q[1])
            for i, r in rects.items():
                if i in ends or hx < r[0] or lx > r[0] + r[2] or hy < r[1] or ly > r[1] + r[3]:
                    continue
                if segment_inside(p, q, r):
                    return (p, q), i
    return None


def core_problems(graph, view):
    """The checks a and b on the core's own Drawing (uncached): '' or the defect."""
    d = raw_layout(graph, view)
    rects = {i: (b.x, b.y, b.w, b.h) for i, b in list(d.boxes.items()) + list(d.cards.items())}
    ov = overlaps(rects)
    if ov:
        return f"core Drawing: boxes {ov[0]} and {ov[1]} overlap"
    for w in d.wires:
        hit = crossing([w.points], rects, ends_of(graph, w.src) | ends_of(graph, w.dst))
        if hit:
            return f"core Drawing: wire {w.id} ({w.style}) passes behind {hit[1]}"
    return ""


# ============================================================================================ the document
class RecordingDocument(MemoryDocument):
    """The graph held in memory, with the log of its accepted operations (the core-only minimal reproduction)."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.log: list = []

    def apply(self, steps):
        steps = list(steps)
        res = super().apply(steps)
        if res.ok:
            self.log.append(("apply", [(n, k if callable(k) else copy.deepcopy(k)) for n, k in steps]))
        return res

    def undo(self):
        res = super().undo()
        if res.ok:
            self.log.append(("undo", None))
        return res

    def redo(self):
        res = super().redo()
        if res.ok:
            self.log.append(("redo", None))
        return res


def start_document(start):
    h = FixtureDoc.from_template(start)
    return RecordingDocument(graph=h.project, view=h.view)


#: --model: the folder the starts' projects are written in (None: the graph held in memory)
MODEL_ROOT = None


class CountingStack(list):
    """The model's undo list, its steps counted: + a step pushed, - a step popped or taken back from its end; the
    oldest dropped at the 200 cap are not counted off."""

    def __init__(self, items=()):
        super().__init__(items)
        self.net = len(self)

    def append(self, x):
        super().append(x)
        self.net += 1

    def pop(self, *a):
        x = super().pop(*a)
        self.net -= 1
        return x

    def __delitem__(self, key):
        n = len(self)
        if isinstance(key, slice) and key.start is not None and key.start >= 0 and key.stop is None:
            self.net -= len(range(*key.indices(n)))   # the end taken back (default_post_layer)
        super().__delitem__(key)


class ModelDocument:
    """The page's document on the client's project model (``M(GraphEditing, ProjectModel)``, read-only): what the fuzz
    reads of a document (``_undo``, ``log``) over the model's own ``GraphDocument``."""

    def __init__(self, model, worker):
        self.model, self.worker = model, worker
        self.inner = model.document
        self.log: list = []                     # no core-only reproduction on a model: its steps are the model's
        model._undo = CountingStack(model._undo)

    def __getattr__(self, name):
        return getattr(self.inner, name)

    @property
    def _undo(self):
        """Its steps counted (the model keeps its last 200 only: a length would stop growing)."""
        return range(self.model._undo.net)

    def close(self):
        self.model.close(save=False)
        self.worker.stop()


def start_model(start):
    """A start's graph compiled into a new read-only project, opened by the project model with GraphEditing."""
    import tempfile

    from astro_gpu.client.graphmodel import GraphEditing
    from astro_gpu.client.model import ProjectModel
    from astro_gpu.client.services.worker import ApiWorker
    from astro_gpu.core import project2
    from astro_gpu.core.project import Target
    from astro_gpu.graphcore.compile import compile_into

    class M(GraphEditing, ProjectModel):
        pass
    os.makedirs(MODEL_ROOT, exist_ok=True)
    root = tempfile.mkdtemp(prefix=f"{start}_", dir=MODEL_ROOT)
    h = FixtureDoc.from_template(start)
    p = project2.create(os.path.join(root, "P"), start, Target(name="M 31"))
    compile_into(h.project, p)
    project2.save(p)
    w = ApiWorker()
    m = M(project2.load(os.path.join(root, "P")), w, readonly=True)
    if not w.wait(120) or not m.ready:
        raise RuntimeError(f"the project model could not open {start}: {m.error}")
    return ModelDocument(m, w)


def core_replay(start, log):
    doc = start_document(start)
    for name, steps in log:
        try:
            doc.apply(steps) if name == "apply" else getattr(doc, name)()
        except Exception:                                                      # noqa: BLE001
            return None
    return doc


def core_defect(start, log):
    doc = core_replay(start, log)
    if doc is None:
        return "core exception"
    try:
        return core_problems(doc.graph, doc.view)
    except Exception as e:                                                     # noqa: BLE001
        return f"layout exception {type(e).__name__}: {e}"


def minimise(items, fails, budget):
    """Delta debugging: the smallest sub-list of items for which fails(sub) is still True (within budget seconds)."""
    t0 = time.time()
    cur, n = list(items), 2
    while len(cur) >= 2 and time.time() - t0 < budget:
        size = max(1, len(cur) // n)
        done = False
        for i in range(0, len(cur), size):
            cand = cur[:i] + cur[i + size:]
            if cand and fails(cand):
                cur, n, done = cand, max(n - 1, 2), True
                break
            if time.time() - t0 > budget:
                break
        if not done:
            if size == 1:
                break
            n = min(len(cur), n * 2)
    return cur


# ============================================================================================ files
def make_files(root):
    """The real files and folders a drop takes (empty files: the listing reads names only). Made once, kept."""
    for folder, names in FILE_TREE.items():
        d = os.path.join(root, folder)
        os.makedirs(d, exist_ok=True)
        for nm in names:
            p = os.path.join(d, nm)
            if not os.path.exists(p):
                open(p, "a").close()
    return root


# ============================================================================================ one run
class Run:
    """A page from one start, its gestures and the checks."""

    def __init__(self, start, seed, files_root):
        self.start, self.seed = start, seed
        self.rng = random.Random(f"{seed}:{start}")
        self.files_root = files_root
        self.errors: list[str] = []
        self.doc = start_model(start) if MODEL_ROOT else start_document(start)
        self.page = GraphPage(self.doc.model, None) if MODEL_ROOT else GraphPage(self.doc)
        self.page.resize(*SIZE)
        self.page.show()
        self.page.activateWindow()
        pump()
        self.page.palette.finish()
        self.G = Gestures(self.page)
        self.bridge, self.canvas = self.page.bridge, self.page.canvas
        self.dropped, self.dropped_same = None, False
        own = self.bridge.drop

        def drop(place):                                # what the canvas dropped on (the hot ghost's place)
            self.dropped = place
            before = self.bridge.graph
            res = own(place)
            self.dropped_same = bool(res.ok and self.bridge.graph is before)
            return res
        self.bridge.drop = drop
        self.done: list[dict] = []
        self.stats = {"accepted": 0, "refused": 0, "skipped": 0, "undo_checks": 0}
        self.by_kind: dict[str, int] = {}

    def close(self):
        self.page.close()
        self.page.deleteLater()
        pump()
        if isinstance(self.doc, ModelDocument):
            self.doc.close()

    # ------------------------------------------------------------------ reading
    @property
    def graph(self):
        return self.bridge.graph

    def js(self):
        if MODEL_ROOT:          # the model's undo gives the graph back from its sorted JSON: compared as content
            return json.dumps(json.loads(json_of(self.graph)), sort_keys=True)
        return json_of(self.graph)

    def boxes(self):
        d = self.bridge.drawing()
        return list(d.boxes), list(d.cards)

    def blocks_drawn(self, kinds=None, cards=False):
        b, c = self.boxes()
        out = [x for x in b if kinds is None or self.graph.blocks[x].kind in kinds]
        return out + (c if cards else [])

    # ------------------------------------------------------------------ choosing a gesture (no event, no change)
    def choose(self):
        rng = self.rng
        big = len(self.graph.blocks) > 45
        for _i in range(12):
            if big and rng.random() < 0.35:
                g = self._prune()
            else:
                kind = rng.choices([k for k, _w in WEIGHTS], [w for _k, w in WEIGHTS])[0]
                g = getattr(self, "_c_" + kind)()
            if g is not None:
                return g
        return None

    def _prune(self):
        rows = [r for r in self.graph.rows]
        if not rows:
            return None
        r = self.rng.choice(rows)
        drawn = set(self.blocks_drawn(cards=True))
        b = next((x for x in [r.id] + list(r.blocks) + [s for n in r.nights for s in n] if x in drawn), None)
        if b is None:
            return None
        text = next((t for t, op, _a, ok, _w in self._menu_leaves(b) if op == "delete_row"), None)
        return dict(g="menu", block=b, text=text) if text else None

    def _c_palette(self):
        kind = self.rng.choice(list(self.page.palette.cards))
        places = self.G.places(kind=kind)
        if not places:
            return None
        lit = [i for i, p in enumerate(places) if p.lit]
        grey = [i for i, p in enumerate(places) if not p.lit]
        pool = lit if lit and (not grey or self.rng.random() < 0.65) else grey
        return dict(g="palette", kind=kind, place=self.rng.choice(pool))

    def _c_double(self):
        return dict(g="double", kind=self.rng.choice(list(self.page.palette.cards)))

    def _c_move(self):
        bs = self.blocks_drawn()
        if not bs:
            return None
        b = self.rng.choice(bs)
        places = self.G.places(block=b)
        if not places:
            return None
        lit = [i for i, p in enumerate(places) if p.lit]
        grey = [i for i, p in enumerate(places) if not p.lit]
        pool = lit if lit and (not grey or self.rng.random() < 0.65) else grey
        return dict(g="move", block=b, place=self.rng.choice(pool))

    def _c_wire(self):
        d = self.bridge.drawing()
        outs = [(b, p) for (b, p, s) in d.ports if s == "out" and b in d.boxes]
        if not outs:
            return None
        src, sport = self.rng.choice(outs)
        cands = [(b, p) for (b, p, s) in d.ports if s == "in" and b in d.boxes and b != src
                 and self.graph.blocks[b].kind in JOIN_KINDS]
        cands += [(b, "layers") for b in d.boxes if self.graph.blocks[b].kind == "post"]
        others = [(b, p) for (b, p, s) in d.ports if s == "in" and b in d.boxes and b != src
                  and self.graph.blocks[b].kind not in JOIN_KINDS]
        if not cands and not others:
            return None
        ok = [c for c in cands if not self.bridge.check_wire(src, sport, *c)]
        bad = [c for c in cands if c not in ok]
        r = self.rng.random()
        pool = ok if ok and r < 0.5 else bad if bad and r < 0.9 else others or cands
        dst, dport = self.rng.choice(pool)
        return dict(g="wire", src=src, sport=sport, dst=dst, dport=dport)

    def _menu_leaves(self, b):
        out = []

        def walk(items):
            for e in items:
                if e is None:
                    continue
                if e[1] == "submenu":
                    walk(e[2])
                else:
                    out.append(e)
        walk(gv_menu.menu(self.bridge.doc, b))
        return out

    def _c_menu(self):
        bs = self.blocks_drawn(cards=True)
        if not bs:
            return None
        b = self.rng.choice(bs)
        leaves = self._menu_leaves(b)
        texts = [e[0] for e in leaves]
        # the gesture picks the first item whose text CONTAINS the text: only texts no other item contains
        usable = [e for e in leaves if e[1] not in ("rename", "inspect") and sum(e[0] in t for t in texts) == 1]
        if not usable:
            return None
        return dict(g="menu", block=b, text=self.rng.choice(usable)[0])

    def _c_key(self):
        key = self.rng.choice(["D", "Del", "Ctrl+D", "Ctrl+Z", "Ctrl+Y", "D", "Del"])
        if key in ("Ctrl+Z", "Ctrl+Y"):
            return dict(g="key", key=key, block=None)
        bs = self.blocks_drawn(cards=True)
        if not bs:
            return None
        return dict(g="key", key=key, block=self.rng.choice(bs))

    def _c_cross(self):
        bs = [b for b in self.blocks_drawn() if self.graph.blocks[b].kind not in ("post", "calibration")]
        if not bs:
            return None
        return dict(g="cross", block=self.rng.choice(bs), link=self.rng.random() < 0.6)

    def _c_wclick(self):
        ws = sorted(self.canvas.wires)
        if not ws:
            return None
        return dict(g="wclick", wire=self.rng.choice(ws), delete=self.rng.random() < 0.6)

    def _c_display(self):
        key = self.rng.choice(["blocks", "folded", "table"])
        return dict(g="display", key=key, back=self.rng.choice(["blocks", "folded"]))

    def _c_ctrlj(self):
        return dict(g="ctrlj")

    def _c_files(self):
        ss = self.blocks_drawn(kinds=("session",))
        if not ss:
            return None
        paths = self.rng.sample(DROPS, self.rng.randint(1, 3))
        link = self.rng.choice([None, None, "take_all", "include", "undo"])
        return dict(g="files", session=self.rng.choice(ss), kind=self.rng.choice(FILE_KINDS), paths=paths, link=link)

    # ------------------------------------------------------------------ performing a gesture (real events)
    def perform(self, g):
        """-> (expectation, history) or None when the gesture no longer applies (replay of a shortened list).
        expectation: "change" | "message" | "change_or_message" | "none" (no change) | "free"."""
        return getattr(self, "_p_" + g["g"])(g)

    def _drop_expect(self, aimed):
        pl = self.dropped
        if pl is None:
            raise Failure("e", f"a drop aimed at the {aimed.kind} place ({self.bridge.say(aimed.tip)!r}) hit no place")
        if pl.lit and self.dropped_same:
            raise Failure("e", f"a lit {pl.op[0]} place gives back the same graph (the own place of the block?), its "
                               f"toast says it acted: {self.canvas.toast.plain!r}", core=True)
        return ("change" if pl.lit else "message"), False

    def _p_palette(self, g):
        places = self.G.places(kind=g["kind"])
        if g["place"] >= len(places):
            return None
        self.dropped = None
        self.G.drag_palette(g["kind"], g["place"])
        return self._drop_expect(places[g["place"]])

    def _p_double(self, g):
        if g["kind"] not in self.page.palette.cards:
            return None
        self.G.double_click_palette(g["kind"])
        return "change_or_message", False

    def _p_move(self, g):
        b = g["block"]
        if b not in self.bridge.drawing().boxes:
            return None
        places = self.G.places(block=b)
        if g["place"] >= len(places):
            return None
        self.dropped = None
        self.G.move_block(b, g["place"])
        return self._drop_expect(places[g["place"]])

    def _p_wire(self, g):
        d = self.bridge.drawing()
        src, dst = g["src"], g["dst"]
        if (src, g["sport"], "out") not in d.ports or dst not in d.boxes:
            return None
        kind = self.graph.blocks[dst].kind
        if kind != "post" and (dst, g["dport"], "in") not in d.ports:
            return None
        why = self.bridge.check_wire(src, g["sport"], dst, g["dport"])
        self.G.pull_wire(src, g["sport"], dst, g["dport"])
        if kind in JOIN_KINDS or kind == "post":
            return ("message" if why else "change"), False
        return "none", False

    def _p_menu(self, g):
        b = g["block"]
        if b not in self.canvas.blocks:
            return None
        entry = next((e for e in self._menu_leaves(b) if g["text"] and g["text"] in e[0]), None)
        if entry is None:
            return None
        self.G.menu_pick(b, g["text"])
        if entry[1] in MENU_FREE or not entry[3]:
            return ("none" if not entry[3] else "free"), False
        return "change_or_message", False

    def _p_key(self, g):
        key, b = g["key"], g["block"]
        if b is not None:
            if b not in self.canvas.blocks:
                return None
            self.G.click(b)
            if self.bridge.selected != b:
                raise Failure("e", f"a click on {b} did not select it (selected {self.bridge.selected})")
        self.G.key(key, target=self.canvas)
        if key in ("Ctrl+Z", "Ctrl+Y"):
            return "change_or_message", True
        card = b in self.bridge.drawing().cards
        if key == "Del":
            return "change_or_message", False
        if key == "D":
            return ("none" if card else "change_or_message"), False
        in_row = card or self.bridge.row_of(b) is not None
        return ("change_or_message" if in_row else "none"), False

    def _p_cross(self, g):
        b = g["block"]
        if b not in self.canvas.blocks or self.canvas.blocks[b].d["kind"] == "card" or \
                not self.canvas.blocks[b].d["removable"]:
            return None
        self.G.click_cross(b)
        if g.get("link") and any(k == "delete_too" for _t, k in self.canvas.toast.links()):
            self.canvas.toast.link.emit("delete_too")
            pump()
        return "change_or_message", False

    def _p_wclick(self, g):
        w = g["wire"]
        if w not in self.canvas.wires:
            return None
        self.G.click_wire(w)
        if g.get("delete") and self.canvas.selected_wire == w:
            self.G.key("Del", target=self.canvas)
            return "change_or_message", False
        return "free", False

    def _p_display(self, g):
        want, refused = self._shown_after(g["key"])
        self.G.display(g["key"])
        if self.page.position() != want:
            raise Failure("f", f"the display control on {g['key']} shows {self.page.position()}")
        if g["key"] == "table":
            if self.page.centre.currentWidget() is not self.page.table_host:
                raise Failure("f", "Table is chosen but the table is not shown")
            want, refused = self._shown_after(g["back"])
            self.G.display(g["back"])
            if self.page.position() != want:
                raise Failure("f", f"the display control on {g['back']} shows {self.page.position()}")
        if self.page.centre.currentWidget() is not self.canvas:
            raise Failure("f", f"{self.page.position()} is chosen but the graph is not shown")
        return ("message" if refused else "none"), False

    def _shown_after(self, key):
        """The position a display choice gives: Folded is refused by the core in some graphs (a variant forking from a
        frame block): the blocks stay unfolded and the toast says why. -> (position, refused)."""
        if key == "folded" and self.page.position() != "folded" and \
                not self.bridge.doc.try_many([("fold_all", {})]).ok:
            return "blocks", True
        return key, False

    def _p_ctrlj(self, g):
        order = ["blocks", "folded", "table"]
        p0 = self.page.position()
        want, refused = self._shown_after(order[(order.index(p0) + 1) % 3])
        if refused:                                   # Ctrl+J passes a refused Folded (its reason said) to Table
            want = "table"
        self.G.key("Ctrl+J", target=self.canvas)
        if refused:
            if self.page.position() != want:
                raise Failure("f", f"Ctrl+J to a refused Folded from {p0} shows {self.page.position()}")
            return "message", False
        if self.page.position() != want:
            fw, aw = QApplication.focusWidget(), QApplication.activeWindow()
            raise Failure("f", f"Ctrl+J from {p0} shows {self.page.position()}, not {want} (focus "
                               f"{type(fw).__name__ if fw else None}, active window {type(aw).__name__ if aw else None})")
        if want == "table":
            self.G.key("Ctrl+J", target=self.page.table_host)
            if self.page.position() != "blocks":
                raise Failure("f", f"Ctrl+J from the table shows {self.page.position()}, not blocks")
        return "none", False

    def _p_files(self, g):
        s = g["session"]
        if s not in self.bridge.drawing().boxes or self.graph.blocks[s].kind != "session":
            return None
        paths = [os.path.join(self.files_root, p).replace("\\", "/") for p in g["paths"]]
        res = self.G.drop_files(s, g["kind"], paths)
        if not res.get("targets"):
            raise Failure("e", f"files dragged over the Session {s} showed no target {g['kind']}")
        self.settle_files()
        link = g.get("link")
        if link and self.canvas.toast.isVisible() and any(k == link for _t, k in self.canvas.toast.links()):
            self.canvas.toast.link.emit(link)
            self.link_used = True
            pump()
            self.settle_files()
            if link == "undo":
                return "free", True
        return "change_or_message", False

    def settle_files(self):
        drops = self.bridge._drops
        if drops is None:
            return
        for _i in range(3):
            drops.reader.wait(10)
            QTest.qWait(15)
            pump()

    # ------------------------------------------------------------------ the checks
    def check_painted(self, when=""):
        """f, a, b on the painted scene."""
        cv, bridge = self.canvas, self.bridge
        if self.errors:
            raise Failure("f", "Python exception or Qt warning: " + self.errors[0])
        if self.page.position() == "table":
            return
        if cv.gesture is not None or cv.ghosts or cv.targets or cv.rubber is not None or cv.floater is not None:
            raise Failure("f", f"a drag's items are left{when}: gesture {cv.gesture and cv.gesture.get('kind')}, "
                               f"{len(cv.ghosts)} ghosts, {len(cv.targets)} targets")
        d = bridge.drawing()
        fresh = bridge.doc.layout()
        if fresh != d:
            raise Failure("f", f"the bridge's cached Drawing is not the current graph's{when}")
        want = set(d.boxes) | set(d.cards)
        if set(cv.blocks) != want:
            raise Failure("f", f"items without a box / boxes without an item{when}: {sorted(set(cv.blocks) ^ want)}")
        scene_blocks = [it for it in cv.scene().items() if isinstance(it, BlockItem)]
        if len(scene_blocks) != len(cv.blocks):
            raise Failure("f", f"{len(scene_blocks)} block items in the scene for {len(cv.blocks)} boxes{when}")
        if any(isinstance(it, (GhostItem, FileTarget, RubberWire)) for it in cv.scene().items()):
            raise Failure("f", f"a ghost, file target or rubber wire is left in the scene{when}")
        rects = {}
        for i, it in cv.blocks.items():
            box = d.boxes.get(i) or d.cards[i]
            if isinstance(it, StackItem) != (i in d.cards):
                raise Failure("f", f"{i} is painted as a {'card' if isinstance(it, StackItem) else 'block'}{when}")
            r = (it.pos().x(), it.pos().y(), it.w, it.h)
            if max(abs(r[0] - box.x), abs(r[1] - box.y), abs(r[2] - box.w), abs(r[3] - box.h)) > 0.05:
                raise Failure("f", f"item {i} painted at {r}, the Drawing says {(box.x, box.y, box.w, box.h)}{when}")
            rects[i] = r
        ids = [p.id for p in d.wires]
        if len(set(ids)) != len(ids):
            twice = sorted({x for x in ids if ids.count(x) > 1})
            raise Failure("f", f"two paths of the Drawing share an id{when}: {twice}", core=True)
        if set(cv.wires) != set(ids):
            raise Failure("f", f"wire items differ from the Drawing's paths{when}: {sorted(set(cv.wires) ^ set(ids))}")
        for p in d.wires:
            it = cv.wires[p.id]
            a, b = it.end_points
            if max(abs(a.x() - p.points[0][0]), abs(a.y() - p.points[0][1]), abs(b.x() - p.points[-1][0]),
                   abs(b.y() - p.points[-1][1])) > 0.05 or it.style_name != p.style:
                raise Failure("f", f"wire {p.id} painted from {(a.x(), a.y())} to {(b.x(), b.y())} ({it.style_name}), "
                                   f"the Drawing says {p.points[0]} to {p.points[-1]} ({p.style}){when}")
        ov = overlaps(rects)
        if ov:
            raise Failure("a", f"items overlap{when}: {ov[0]} {rects[ov[0]]} / {ov[1]} {rects[ov[1]]}")
        g = self.graph
        for wid, it in cv.wires.items():
            src, dst = it.ends
            ends = ends_of(g, src) | ends_of(g, dst)
            polys = [[(q.x(), q.y()) for q in poly] for poly in it.path().toSubpathPolygons()]
            hit = crossing(polys, rects, ends)
            if hit:
                raise Failure("b", f"wire {wid} ({it.style_name} {src}>{dst}) passes behind {hit[1]} "
                                   f"{tuple(round(v) for v in rects[hit[1]])}{when}")

    def check_shown(self):
        """c: what the gesture made is drawn and fully on screen."""
        cv, bridge = self.canvas, self.bridge
        if self.page.position() == "table":
            return
        view = cv.mapToScene(cv.viewport().rect()).boundingRect().adjusted(-0.5, -0.5, 0.5, 0.5)
        for x in bridge.last_shown:
            is_row = bridge.row(x) is not None
            if x in cv.blocks:
                it = cv.blocks[x]
            else:
                r = bridge.row_of(x) if not is_row else None
                if is_row and x not in cv.blocks:
                    continue                                   # an unfolded row: drawn as its blocks
                it = cv.blocks.get(r.id) if r is not None else None
                if it is None:
                    raise Failure("c", f"{x} made by the gesture is not drawn")
            box = QRectF(it.pos().x(), it.pos().y(), it.w, it.h)
            if not view.contains(box):
                raise Failure("c", f"{x} made by the gesture is not fully on screen: item {box.getRect()}, "
                                   f"view {view.getRect()}, zoom {cv.zoom:.2f}")

    def check_message(self, expect, changed, toasts0):
        toast = self.canvas.toast if self.page.position() != "table" else getattr(self.page, "table_toast", None)
        shown = len(self.G.toasts) > toasts0 and toast is not None and toast.isVisible() and toast.plain.strip()
        if expect == "change" and not changed:
            raise Failure("e", f"an accepted place / port left the graph unchanged (toast {self.canvas.toast.plain!r})",
                          core=True)
        if expect in ("message", "none") and changed:
            raise Failure("e", f"a gesture that should change nothing ({expect}) changed the graph")
        if expect == "message" and not shown:
            raise Failure("e", "a refused gesture left no visible message")
        if expect == "change_or_message" and not changed and not shown:
            raise Failure("e", "a gesture neither changed the graph nor showed a message")

    def check_undo(self, before, after, steps):
        """d: a gesture is ONE undo step (two when a toast's link was clicked after it); Ctrl+Z as many times gives
        back the graph before, Ctrl+Y the one after (and the scene stays right)."""
        self.stats["undo_checks"] += 1
        if steps not in ((1, 2) if self.link_used else (1,)):
            raise Failure("d", f"the gesture made {steps} undo steps" + (" with a toast's link" if self.link_used else ""))
        for _i in range(steps):
            self.G.key("Ctrl+Z", target=self.canvas)
        if self.js() != before:
            raise Failure("d", "Ctrl+Z did not give back the graph before the gesture")
        self.check_painted(" after Ctrl+Z")
        for _i in range(steps):
            self.G.key("Ctrl+Y", target=self.canvas)
        if self.js() != after:
            raise Failure("d", "Ctrl+Y did not give back the graph after the gesture")
        self.check_painted(" after Ctrl+Y")

    # ------------------------------------------------------------------ one gesture
    def step_known(self, g, known):
        """step(g), a known core defect (KNOWN_CORE) counted in `known` and its state taken back instead of failing."""
        try:
            return self.step(g)
        except Failure as e:
            if e.sig not in KNOWN_CORE:
                raise
            known[e.sig] = known.get(e.sig, 0) + 1
            self.errors.clear()
            if self.js() != self.last_before and self.bridge.doc.can_undo:
                self.bridge.undo()
                pump()
            return True

    def step(self, g):
        """Perform one gesture and check. -> True when performed, False when it does not apply (replay)."""
        if self.page.position() == "table":
            self.G.display("blocks")
        if QApplication.activeWindow() is not self.page:   # offscreen: a closed menu leaves no active window
            self.page.activateWindow()
            pump(1)
        before = self.last_before = self.js()
        toasts0 = len(self.G.toasts)
        undo0 = len(self.doc._undo)
        self.bridge.last_shown = ()
        self.link_used = False
        out = self.perform(g)
        if out is None:
            self.stats["skipped"] += 1
            return False
        expect, history = out
        pump()
        after = self.js()
        changed = after != before
        self.stats["accepted" if changed else "refused"] += 1
        self.by_kind[g["g"]] = self.by_kind.get(g["g"], 0) + 1
        self.check_painted()
        self.check_shown()
        self.check_message(expect, changed, toasts0)
        if changed and not history:
            self.check_undo(before, after, len(self.doc._undo) - undo0)
        return True


# ============================================================================================ runner
class Catcher:
    """Python exceptions raised in Qt slots and Qt warnings go to run.errors while a run is on."""

    def __init__(self):
        self.run = None

    def __enter__(self):
        self._hook, sys.excepthook = sys.excepthook, self._exc
        self._prev = qInstallMessageHandler(self._qt)
        return self

    def __exit__(self, *a):
        sys.excepthook = self._hook
        qInstallMessageHandler(self._prev)

    def _exc(self, typ, val, tb):
        if self.run is not None:
            self.run.errors.append("".join(traceback.format_exception(typ, val, tb))[-1500:])

    def _qt(self, mode, ctx, msg):
        if msg.startswith("This plugin does not support"):          # the offscreen platform's own limit
            return
        if mode in (QtMsgType.QtWarningMsg, QtMsgType.QtCriticalMsg, QtMsgType.QtFatalMsg) and self.run is not None:
            self.run.errors.append(f"Qt {mode.name}: {msg}")


def app():
    a = QApplication.instance()
    if a is None:
        from astro_gpu.client.widgets import QSS
        a = QApplication([])
        a.setStyleSheet(QSS)
    return a


def classify(f: Failure, run):
    """A failure is the core's when the core's own Drawing has it (a, b), or when it says so."""
    if f.core:
        return True
    if f.check in ("a", "b"):
        return bool(core_problems(run.graph, run.bridge.view))
    return "graphcore" in f.msg and "Traceback" in f.msg


def replay(start, gestures, files_root, catcher=None):
    """Replay a gesture list on a new page. -> (Failure or None, run)."""
    app()
    own = catcher is None
    catcher = catcher or Catcher().__enter__()
    run = Run(start, 0, files_root)
    catcher.run = run
    fail = None
    try:
        for g in gestures:
            run.step_known(g, {})
    except Failure as e:
        fail = e
    except Exception:                                                          # noqa: BLE001
        fail = Failure("x", "exception in the fuzz: " + traceback.format_exc(limit=-4)[-600:])
    finally:
        catcher.run = None
        run.close()
        if own:
            catcher.__exit__(None, None, None)
    return fail, run


def fuzz(start, seed, n, files_root, out=None, shrink_budget=240.0, verbose=False):
    """One start: n random gestures. -> summary dict (clean True/False, the failure, counts)."""
    app()
    t0 = time.time()
    summary = dict(start=start, seed=seed, gestures=n, clean=True, failure=None)
    with Catcher() as catcher:
        run = Run(start, seed, files_root)
        catcher.run = run
        done = 0
        fail = None
        known: dict[str, int] = {}
        try:
            while done < n:
                g = run.choose()
                if g is None:
                    g = dict(g="double", kind="session")
                run.done.append(g)
                run.step_known(g, known)
                done += 1
                if verbose and done % 100 == 0:
                    print(f"   {start} seed {seed}: {done} gestures, {time.time() - t0:.0f} s, "
                          f"{len(run.graph.blocks)} blocks", flush=True)
        except Failure as e:
            fail = e
        except Exception:                                                      # noqa: BLE001
            fail = Failure("x", "exception in the fuzz: " + traceback.format_exc(limit=-4)[-600:])
        catcher.run = None
        summary.update(run=done, stats=run.stats, by_kind=run.by_kind, blocks_at_end=len(run.graph.blocks),
                       seconds=round(time.time() - t0, 1), known_core_skipped=known)
        if fail is not None:
            core = classify(fail, run)
            summary.update(clean=False, failure=dict(check=fail.check, message=fail.msg, signature=fail.sig, core=core,
                                                     at=done + 1, last=run.done[-8:]))
            log = list(run.doc.log)
            gestures = list(run.done)
            run.close()
            if out is not None:
                _write_failure(out, start, seed, fail, core, gestures, log, files_root, catcher, shrink_budget, summary)
        else:
            run.close()
    if out is not None:
        with open(os.path.join(out, f"{start}_seed{seed}.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=1, ensure_ascii=False, default=repr)
    return summary


def _write_failure(out, start, seed, fail, core, gestures, log, files_root, catcher, budget, summary):
    base = os.path.join(out, f"{start}_seed{seed}")
    with open(base + "_gestures.json", "w", encoding="utf-8") as f:
        json.dump(dict(start=start, signature=fail.sig, gestures=gestures), f, indent=1, ensure_ascii=False)

    def fails(sub):
        e, _r = replay(start, sub, files_root, catcher)
        return e is not None and e.sig == fail.sig
    first, _r = replay(start, gestures, files_root, catcher)
    replayable = first is not None and first.sig == fail.sig
    small = minimise(gestures, fails, budget) if replayable else gestures
    with open(base + "_minimal.json", "w", encoding="utf-8") as f:
        json.dump(dict(start=start, signature=fail.sig, replayable=replayable, gestures=small), f, indent=1,
                  ensure_ascii=False)
    summary["failure"].update(replayable=replayable, minimal=len(small))
    if core:
        defect = core_defect(start, log)
        if defect:
            mini = minimise(log, lambda sub: bool(core_defect(start, sub)), budget)
            summary["failure"]["core_ops"] = [[n, s] for n, s in mini]
            summary["failure"]["core_defect"] = core_defect(start, mini)
            with open(base + "_core_ops.json", "w", encoding="utf-8") as f:
                json.dump(dict(start=start, defect=summary["failure"]["core_defect"], ops=mini), f, indent=1,
                          ensure_ascii=False, default=repr)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--starts", type=int, default=3, help=f"how many starts, in the order {STARTS}")
    ap.add_argument("--start", default=None, help="one start by its key instead")
    ap.add_argument("--gestures", type=int, default=1000)
    ap.add_argument("--out", default=os.path.join(ROOT, "_work", "graphview_fuzz"))
    ap.add_argument("--shrink-budget", type=float, default=240.0, help="seconds for each minimisation")
    ap.add_argument("--replay", default=None, help="a gesture list (JSON) written by a failure")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--model", action="store_true", help="the page on the client's project model, not in memory")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    if a.model:
        global MODEL_ROOT
        MODEL_ROOT = os.path.join(a.out, "projects")
    from astro_gpu.client import i18n, prefs
    prefs.load()
    i18n.set_language("en")
    files_root = make_files(os.path.join(a.out, "files"))
    if a.replay:
        d = json.load(open(a.replay, encoding="utf-8"))
        fail, run = replay(d["start"], d["gestures"], files_root)
        print(f"replay {d['start']}, {len(d['gestures'])} gestures: " + (f"FAIL {fail}" if fail else "clean"))
        return 1 if fail else 0
    starts = [a.start] if a.start else STARTS[:a.starts]
    clean, total, t0 = 0, 0, time.time()
    for st in starts:
        s = fuzz(st, a.seed, a.gestures, files_root, a.out, a.shrink_budget, a.verbose)
        total += 1
        clean += s["clean"]
        f = s["failure"]
        line = f"{st:18s} seed {a.seed}  {s['run']}/{a.gestures} gestures  {s['seconds']:.0f} s  "
        line += ("clean" if f is None else f"FAIL ({'core' if f['core'] else 'view'}) at gesture {f['at']}: "
                 f"{f['message'][:200]}")
        if s["known_core_skipped"]:
            line += f"  (known core defects skipped: {s['known_core_skipped']})"
        print(line, flush=True)
    print(f"{clean}/{total} starts clean, {time.time() - t0:.0f} s, output in {a.out}")
    return 0 if clean == total else 1


if __name__ == "__main__":
    sys.exit(main())
