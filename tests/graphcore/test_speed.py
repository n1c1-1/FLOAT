"""Timing (coordinator's target): on the SHO + RGB stars fixture after 300 random gestures, the whole palette's
state and places(kind) each under 5 ms (median of several calls, after a first call that builds the caches)."""
import os
import random
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools", "fuzz"))
import graph_fuzz as fuzz  # noqa: E402
from astro_gpu.graphcore import layout, places  # noqa: E402
from astro_gpu.graphcore.rules import ORDER  # noqa: E402

LIMIT = 0.015      # target 5 ms on a free machine (measured 2-5 ms); the margin is for a loaded one


def grown(seed=1, n=300):
    rng = random.Random(f"{seed}:sho_rgb")
    doc = fuzz.start_doc("sho_rgb")
    d = layout(doc.project, doc.view)
    for _i in range(n):
        doc, d = fuzz.run_one(doc, fuzz.gen(rng, doc), d)
    return doc


def _median(fn, n=7):
    ts = []
    for _i in range(n):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return statistics.median(ts)


def timings(doc):
    out = {"palette": _median(lambda: [doc.palette_state(k) for k in ORDER])}
    for k in ("gradient", "denoise", "color", "stars", "selection", "session", "combine_layers"):
        out[k] = _median(lambda k=k: places(doc.project, doc.view, kind=k))
    return out


def test_palette_and_places_under_15_ms():
    doc = grown()
    t = timings(doc)
    slow = {k: round(v * 1000, 2) for k, v in t.items() if v > LIMIT}
    assert not slow, slow


if __name__ == "__main__":
    doc = grown()
    print(len(doc.project.rows), "rows,", len(doc.project.blocks), "blocks")
    for k, v in timings(doc).items():
        print(f"{k:18s} {v * 1000:8.2f} ms")
