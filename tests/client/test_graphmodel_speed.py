"""Timing of the graph page's gestures over the project model (the owner's rule: FLOAT stays fast on a computer without
a GPU). Each gesture is measured in processor time of the whole process (time.process_time: the GUI thread and the
worker, never the wall clock, which a parallel test run doubles), from the gesture to the end of the work it starts
(the drawing, the panel, the palette's cards), the median of three, and must stay under a budget per block of the
graph. Level 3 (tests/levels.toml): it is a measure, not a check of behaviour.

Measured on this machine, CPU only, on the SHO + RGB stars template (37 blocks), in ms of processor per gesture, before
and after the work of November 2026 on the page's speed: place 484 -> 203, delete 188 -> 172, undo 469 -> 250, choose
a block 312 -> 62, Folded 234 -> 172; that is 13 ms per block before at the worst, 7 after. The budget, 10 ms per
block, fails a gesture that computes the whole page twice again."""
from __future__ import annotations

import statistics
import time

import pytest

from tests.client.graphview.legacy import World
from tests.client.pagehelp import qapp  # noqa: F401

#: seconds of processor time per block of the graph, for one gesture and all the work it starts
PER_BLOCK = 0.010
REPEATS = 3


@pytest.fixture()
def world(qapp, tmp_path):  # noqa: F811
    w = World(tmp_path / "w", workflows=(), filters=("SII", "Ha", "OIII"), nights=0)
    yield w
    w.close()


def _cpu(W, gesture, back=None) -> float:
    """The median processor time of a gesture and of the work it starts (its undo, ``back``, not counted)."""
    times = []
    for _i in range(REPEATS):
        W.settle(2)
        t = time.process_time()
        gesture()
        W.settle(2)
        times.append(time.process_time() - t)
        if back is not None:
            back()
            W.settle(2)
    return statistics.median(times)


def test_the_gestures_of_the_graph_page_take_a_time_per_block(world):
    W = world
    W.template("sho_rgb")
    b = W.b
    blocks = len(W.graph.blocks)
    assert blocks >= 30, blocks                         # a real graph: the template's rows, combinations, Post
    stack = W.kinds("stacking")[0]
    W.choose(stack)
    b.act("add_block", kind="deconv", row=b.row_of(stack).id)    # something to undo
    W.settle()
    limit = PER_BLOCK * blocks
    got = {
        "place": _cpu(W, lambda: W.page.palette.double_click("deconv"), b.undo),
        "delete": _cpu(W, lambda: b.act("remove_block", block=W.kinds("deconv")[0]), b.undo),
        "undo": _cpu(W, b.undo, b.redo),
        "choose": _cpu(W, lambda: b.choose(W.kinds("denoise")[0]), lambda: b.choose(stack)),
        "folded": _cpu(W, lambda: W.page.set_display("folded"), lambda: W.page.set_display("blocks")),
    }
    print("processor ms per gesture,", blocks, "blocks:", {k: round(v * 1000) for k, v in got.items()})
    slow = {k: round(v * 1000) for k, v in got.items() if v > limit}
    assert not slow, (f"over {limit * 1000:.0f} ms of processor for {blocks} blocks", slow,
                      {k: round(v * 1000) for k, v in got.items()})
