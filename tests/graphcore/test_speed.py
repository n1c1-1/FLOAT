"""Timing (coordinator's target): on the SHO + RGB stars fixture after 300 random gestures, the whole palette's
state and places(kind) each in a time proportional to the graph's size: 0.25 ms of processor time per block (77
blocks measured 2-5 ms on a free machine; a slowness of 60 ms per card is caught many times over).

What is measured is the processor time of the calling thread (time.thread_time), not the wall time: under a parallel
test run the wall time of the same work is doubled or tripled by the other processes, the processor time is not. The
thread clock ticks every 15.6 ms on Windows, so a call is timed over a batch of calls (the best of three batches),
after a first call that builds the caches. The budget follows the number of blocks, so that a fixture grown larger by
a change of the fuzz gestures (77, then 101 blocks) does not turn into a failure while the speed per block is the same."""
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools", "fuzz"))
import graph_fuzz as fuzz  # noqa: E402
from astro_gpu.graphcore import layout, places  # noqa: E402
from astro_gpu.graphcore.rules import ORDER  # noqa: E402

PER_BLOCK = 0.00025     # seconds of processor time per block of the graph, for the whole palette or one places()
BATCH = 10              # calls per batch: 10 calls of a few ms are far above the clock's 15.6 ms tick
BATCHES = 3


def grown(seed=1, n=300):
    """The fixture. The engine's checks of each gesture are left out (the fuzz runs them; they do not change the graph
    made, and took half of the time)."""
    rng = random.Random(f"{seed}:sho_rgb")
    doc = fuzz.start_doc("sho_rgb")
    d = layout(doc.project, doc.view)
    was, fuzz.ENGINE_ON[0] = fuzz.ENGINE_ON[0], False
    try:
        for _i in range(n):
            doc, d = fuzz.run_one(doc, fuzz.gen(rng, doc), d)
    finally:
        fuzz.ENGINE_ON[0] = was
    return doc


def _cpu_per_call(fn):
    fn()                                            # the first call builds the caches
    best = float("inf")
    for _b in range(BATCHES):
        t = time.thread_time()
        for _i in range(BATCH):
            fn()
        best = min(best, (time.thread_time() - t) / BATCH)
    return best


def timings(doc):
    out = {"palette": _cpu_per_call(lambda: [doc.palette_state(k) for k in ORDER])}
    for k in ("gradient", "denoise", "color", "stars", "selection", "session", "combine_layers"):
        out[k] = _cpu_per_call(lambda k=k: places(doc.project, doc.view, kind=k))
    return out


def test_palette_and_places_take_a_time_per_block():
    doc = grown()
    blocks = len(doc.project.blocks)
    assert blocks >= 50, blocks                     # the fixture is a real graph, not one the gestures emptied
    limit = PER_BLOCK * blocks
    t = timings(doc)
    slow = {k: round(v * 1000, 2) for k, v in t.items() if v > limit}
    assert not slow, (f"over {limit * 1000:.1f} ms for {blocks} blocks", slow)


if __name__ == "__main__":
    doc = grown()
    print(len(doc.project.rows), "rows,", len(doc.project.blocks), "blocks,",
          f"limit {PER_BLOCK * len(doc.project.blocks) * 1000:.1f} ms")
    for k, v in timings(doc).items():
        print(f"{k:18s} {v * 1000:8.2f} ms")
