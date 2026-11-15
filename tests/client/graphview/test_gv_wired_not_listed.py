"""What a wire of the graph chooses is chosen with the wires, never in the panel (the owner's rule): the reference of
an Align on, the inputs of a combination and the two sides of a Combine layers are shown read-only in the panel, the
workflow wired or "not wired", its tooltip saying where it is chosen; no list of workflows is left there."""
from __future__ import annotations

import pytest
from PySide6.QtWidgets import QComboBox, QLabel

from tests.client.graphview.legacy import World


@pytest.fixture()
def world(qapp, tmp_path):
    made = []

    def make(**kw):
        w = World(tmp_path / f"w{len(made)}", **kw)
        made.append(w)
        return w
    yield make
    for w in made:
        w.close()


def wired(W):
    return [x for x in W.page.panel.findChildren(QLabel) if x.objectName() == "wired" and x.isVisibleTo(W.page.panel)]


def no_workflow_list(W):
    pids = {q.pid for q in W.m.paths}
    return not any(c.findData(pid) >= 0 for c in W.page.panel.findChildren(QComboBox) for pid in pids
                   if c.objectName() != "recipe")


def row_named(W, name):
    return next(r for r in W.graph.rows if r.name == name)


def test_a_combination_and_an_align_on_show_what_is_wired_read_only(world):
    W = world(workflows=(), filters=("Ha", "OIII"))
    W.page.show_templates()
    card = W.page.templates.card("filters_full")
    card.set_palette("H-O")
    card.mousePressEvent(None)
    W.settle()
    W.choose(W.kinds("combine")[0])
    shown = [x.text() for x in wired(W)]
    assert {x.split(" · ")[0] for x in shown} == {"Ha", "OIII"}, shown         # the workflow, and which of its images
    assert all("wire on the graph" in x.toolTip() for x in wired(W))
    assert no_workflow_list(W)
    # an input unwired (the Table view or the wire's gesture): said "not wired", nothing to pick in the panel
    fu = next(q for q in W.m.paths if "fusion" in (q.source or {}))
    assert W.m.clear_fusion_input(fu, "OIII")
    W.settle()
    W.choose(W.kinds("combine")[0])
    assert "not wired" in [x.text() for x in wired(W)] and no_workflow_list(W)
    # Align on: its reference wired by the block's menu, shown read-only
    ha = row_named(W, "Ha")
    W.g.menu_pick(row_named(W, "OIII").blocks[-1], "Ha")
    W.settle()
    assert row_named(W, "OIII").align["ref"] == ha.id
    from types import SimpleNamespace
    pn = W.page.panel                                      # its row of the panel (Inspect's block: no box here)
    pn._refresh(None)
    pn.reference_row(SimpleNamespace(paths=[W.view("OIII")]))
    W.settle(1)
    assert [x.text() for x in wired(W)] == ["Ha"] and no_workflow_list(W)


def test_combine_layers_shows_its_two_sides_read_only(world):
    W = world(workflows=("Full",))
    W.choose(W.one("combine_layers"))
    assert [x.text() for x in wired(W)] == ["Full · starless image", "Full · stars alone"]
    assert no_workflow_list(W)
