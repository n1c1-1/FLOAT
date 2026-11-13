"""Fuzz (§6): random gestures from blank and from each template. After each: check(), I1–I6, refusals inert,
undo then redo = same JSON, every row drawn. A failure prints its seed and the shrunk gesture list.
Above 30 rows, 40 % of gestures prune (delete a row or a block) so graphs stay near the size people build.
Each run prints a digest of its final state: the same seed gives the same digest (temp file per process).
Also: apply never changes the graph it is given. Not part of pytest (the smoke is tests/graphcore/test_fuzz_smoke.py).
Usage: python tools/fuzz/graph_fuzz.py [first_seed] [last_seed] [gestures] [start,start…]

``--engine`` (decision B, INTEGRATION §3): after every gesture the graph's compiled definition must be one the engine
accepts (graphcore.compile.engine_problems: Project2.validate, the graph rules, the stage chain, the joins' wiring).
Until the core's rules are the engine's (step 3) this does not stop a run: every class of violation is collected with
the first gesture sequence that shows it, shrunk, and printed at the end (``--json <file>`` writes them).
Usage: python tools/fuzz/graph_fuzz.py --engine [first_seed] [last_seed] [gestures] [start,start…] [--json <file>]"""
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
        lambda: ("many", [_setting(rng, p, blocks)[1:3], ("add_night", {"row": pick(rng, rows)})]),
        lambda: ("undo",), lambda: ("undo",), lambda: ("redo",),
        lambda: ("save",),
    ])()
    return g if g[0] != "op" or len(g) == 4 else g + (None,)


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
            res = doc.apply_many(g[1])
            op, place = None, None
        else:
            _k, op, args, place = g
            if place is not None:                       # judged on the current state (replays, shrinking)
                place = judge(doc.project, doc.view, copy.copy(place))
            res = doc.apply(op, **args)
            if place is not None and place.lit != res.ok:
                raise Fail(f"place lit={place.lit} ({place.reason_text()}) but apply ok={res.ok} ({res.reason_text()})")
        if model_json(prev) != p0:
            raise Fail("apply changed the graph it was given")
        if not res.ok:
            if doc._json != p0 or len(doc._undo) != nu or len(doc._redo) != nr:
                raise Fail(f"refusal changed something: {res.reason_text()}")
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
    argv = list(sys.argv[1:])
    engine = "--engine" in argv
    out = argv[argv.index("--json") + 1] if "--json" in argv else None
    argv = [x for i, x in enumerate(argv) if x != "--engine" and x != "--json" and (i == 0 or argv[i - 1] != "--json")]
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
    print(f"{total - bad}/{total} runs clean, {n} gestures each")
    if engine:
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
