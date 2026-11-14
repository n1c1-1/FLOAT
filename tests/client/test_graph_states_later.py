"""Draw first, states after: a gesture of the graph page on the project model is drawn at once from the stored graph,
while the worker reads the keys, states, chains and Run all's plan of the edit (``model.ASYNC_STATES``); the blocks the
edit can change keep their last state, drawn faint ("being read"), a new block is "pending", the others keep what
they showed; when the answer comes, the states, lines and badges are painted in place. Nothing reads the states of
an earlier edit: a reader that cannot wait reads them at once (``forced_syncs``), which a gesture never needs.

The timing test (level 3, tests/levels.toml) measures processor time of the whole process (the GUI thread and the
worker) on the SHO + RGB stars template: to the gesture's drawing, and to its states painted."""
from __future__ import annotations

import statistics
import time

import pytest
from PySide6.QtWidgets import QApplication

from astro_gpu.client import model as client_model
from tests.client.graphview.legacy import World
from tests.client.pagehelp import qapp  # noqa: F401

#: what a block's item shows that the run and the project model say
SHOWN = ("state", "summary", "grey", "review", "fallback", "broken", "missing", "warned", "title", "badge", "own", "dots")
#: seconds of processor time per block of the graph: to the gesture's drawing, and to its states painted
FIRST_PER_BLOCK = 0.005
STATES_PER_BLOCK = 0.010


@pytest.fixture()
def world(qapp, tmp_path, monkeypatch):  # noqa: F811
    monkeypatch.setattr(client_model, "ASYNC_STATES", True)
    w = World(tmp_path / "w", workflows=(), filters=("SII", "Ha", "OIII"), nights=0)
    yield w
    w.close()


def _states_come(W):
    end = time.time() + 60
    while W.m.states_pending() and time.time() < end:
        QApplication.processEvents()
        time.sleep(0.002)
    assert not W.m.states_pending(), "the states of the edit never came"
    for _i in range(4):
        QApplication.processEvents()


def _painted_as_computed(W):
    cv = W.page.canvas
    d = W.b.drawing()
    for b, it in cv.blocks.items():
        now = cv.card_data(b, d.cards[b]) if b in d.cards else cv.block_data(b, d.boxes[b])
        for k in SHOWN:
            assert it.d.get(k) == now.get(k), (b, k, it.d.get(k), now.get(k))
        assert not it.d.get("reading"), b


def test_a_gesture_is_drawn_first_and_its_states_painted_when_the_worker_has_read_them(world):
    W = world
    W.template("sho_rgb")
    b, cv, m = W.b, W.page.canvas, W.m
    stack = W.kinds("stacking")[0]
    W.choose(stack)
    _states_come(W)
    forced = m.forced_syncs
    before = {x: dict(it.d) for x, it in cv.blocks.items()}

    b.act("add_block", kind="deconv", row=b.row_of(stack).id)            # placed
    new = W.kinds("deconv")[0]
    assert m.states_pending()                                           # drawn, the worker reads the states
    assert cv.blocks[new].d["reading"] and cv.blocks[new].d["state"] == "pending"
    far = next(x for x in W.kinds("session") if x in cv.blocks and x not in b.row_of(stack).blocks)
    assert not cv.blocks[far].d["reading"] and cv.blocks[far].d["state"] == before[far]["state"]
    _states_come(W)
    assert m.forced_syncs == forced                                     # nobody read the states before the answer
    _painted_as_computed(W)

    b.act("remove_block", block=new)                                    # deleted
    assert m.states_pending()
    _states_come(W)
    _painted_as_computed(W)
    b.undo()                                                            # undone: the block back
    assert m.states_pending() and new in cv.blocks
    _states_come(W)
    _painted_as_computed(W)
    assert m.forced_syncs == forced
    assert m.state(m.paths[0], m.all_stages(m.paths[0])[0]) in ("pending", "done", "stale")   # a decision reads now


def test_a_reader_that_cannot_wait_reads_the_states_of_the_edit_at_once(world):
    W = world
    W.template("sho_rgb")
    b, m = W.b, W.m
    stack = W.kinds("stacking")[0]
    W.choose(stack)
    _states_come(W)
    b.act("add_block", kind="deconv", row=b.row_of(stack).id)
    assert m.states_pending()
    forced = m.forced_syncs
    p = next(q for q in m.paths if any(s.startswith("deconv") for s in m.all_stages(q)))   # chains: no wait needed
    assert m.forced_syncs == forced and m.states_pending()
    st = next(s for s in m.all_stages(p) if s.startswith("deconv"))
    assert m.state(p, st) == "pending"                                  # the new block's stage: read at once
    assert m.forced_syncs == forced + 1 and not m.states_pending()
    _states_come(W)                                                     # the worker's answer is dropped, the page told
    _painted_as_computed(W)


def _computed(m):
    """The project as if run once: a stage is done while its key is the one it had (each time the model takes states)."""
    from astro_gpu import api
    keys0 = {}
    install = m._install

    def done(got):
        install(got)
        for w in m.p.workflows:
            for st in m.__dict__["_ch"].get(w.id, []):
                k = api.stage_key(m.p, w.id, st)
                if keys0.setdefault((w.id, st), k) == k:
                    m.__dict__["_st"][(w.id, st)] = api.StageState("done", "", k)
    m._install = done
    m._sync()


def test_the_toast_says_the_gesture_at_once_and_what_it_computes_again_once_the_states_are_read(world):
    W = world
    W.template("sho_rgb")
    b, m = W.b, W.m
    _computed(m)
    _states_come(W)
    toasts = []
    b.rich_toast.connect(lambda t, ls: toasts.append((t, ls)))
    refreshed = []
    b.rich_toast_refresh.connect(lambda: refreshed.append(True))
    forced = m.forced_syncs
    grad = W.kinds("gradient")[0]
    b.act("switch_off", block=grad)
    assert m.states_pending() and toasts
    text, links = toasts[-1]
    first = text() if callable(text) else text
    assert "Computes again" not in first and [k for _w, k in links] == ["undo"]     # the action at once, with Undo
    _states_come(W)
    now = text()
    assert refreshed and now.startswith(first + " · ") and "Computes again: " in now      # completed in place
    assert m.forced_syncs == forced

    toasts.clear()                                                      # Undo before the answer
    b.act("switch_off", block=W.kinds("gradient")[1])
    text, _links = toasts[-1]
    b.undo()
    assert m.states_pending()
    _states_come(W)
    assert "Computes again" not in text()                              # nothing said for an edit taken back
    _painted_as_computed(W)
    assert m.done_count() == sum(1 for st in m._states.values() if st.state == "done")


def _cpu(W, gesture, back):
    """(processor time to the gesture's drawing, to its states painted), medians of three."""
    first, full = [], []
    for _i in range(3):
        _states_come(W)
        W.settle(2)
        t = time.process_time()
        gesture()
        QApplication.processEvents()
        first.append(time.process_time() - t)
        _states_come(W)
        full.append(time.process_time() - t)
        back()
        _states_come(W)
    return statistics.median(first), statistics.median(full)


@pytest.mark.level3
def test_the_gestures_are_drawn_first_within_a_time_per_block(world):
    W = world
    W.template("sho_rgb")
    b = W.b
    blocks = len(W.graph.blocks)
    stack = W.kinds("stacking")[0]
    W.choose(stack)
    b.act("add_block", kind="deconv", row=b.row_of(stack).id)            # something to undo
    _states_come(W)
    got = {
        "place": _cpu(W, lambda: W.page.palette.double_click("deconv"), b.undo),
        "delete": _cpu(W, lambda: b.act("remove_block", block=W.kinds("deconv")[0]), b.undo),
        "undo": _cpu(W, b.undo, b.redo),
    }
    print("processor ms per gesture (to the drawing, to the states painted),", blocks, "blocks:",
          {k: (round(a * 1000), round(c * 1000)) for k, (a, c) in got.items()})
    slow = {k: (round(a * 1000), round(c * 1000)) for k, (a, c) in got.items()
            if a > FIRST_PER_BLOCK * blocks or c > STATES_PER_BLOCK * blocks}
    assert not slow, (f"over {FIRST_PER_BLOCK * blocks * 1000:.0f} / {STATES_PER_BLOCK * blocks * 1000:.0f} ms", slow)
