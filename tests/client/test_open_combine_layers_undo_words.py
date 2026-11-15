"""Opening a workflow's Combine layers in Post makes its layer real without a step of Undo (opening is never one), and
leaves Undo as it was: each Undo names its own step, Redo keeps its words, and the project is not marked modified by
the opening. The step was taken back but its words stayed: every later Undo said the next step's words, Redo had
none, and the title said the project had changes."""
from __future__ import annotations

import pytest

from tests.client.graphview.legacy import World
from tests.client.pagehelp import qapp  # noqa: F401

ROW = ["calibration", "registration", "selection", "stacking", "astrometry", "gradient", "denoise", "stars"]


@pytest.fixture()
def world(qapp, tmp_path):  # noqa: F811
    W = World(tmp_path / "w", workflows=(), chains={"Session 1": ROW})
    for q in W.m.paths:
        q.rendered = True
    yield W
    W.close()


def test_opening_a_combine_layers_keeps_undos_words_and_the_saved_state(world):
    m = world.m
    pid = m.paths[0].pid
    m.add_post_layer(pid, None)
    m.name_step("Step A")
    m.add_post_layer(pid, "starless")
    m.name_step("Step B")
    m.undo()
    m.redo()                                             # a Redo's words exist on the way too
    assert m.flush(force=True) and not m.dirty
    n = len(m._undo)
    m.default_post_layer(pid)
    assert len(m._undo) == n and not m.dirty
    assert m.next_words("undo") == "Step B"
    assert m.undo() and m.restored == "Step B"
    assert m.next_words("undo") == "Step A" and m.next_words("redo") == "Step B"
    assert m.redo() and m.restored == "Step B"
