"""Every block is set the same way, the Gradient's (the owner's rule): a change of a combination's recipe parameter, of
a Combine layers' values and of the Synthetic stars' method is a draft of its block when its results are up to date
(Apply, Try variant, Reset), never made at the click. On a scratch project (nothing computed): its results are said
up to date by the model's cost of a change."""
from __future__ import annotations

import pytest
from PySide6.QtWidgets import QDoubleSpinBox

from astro_gpu.client.blockdraft import DraftBar
from astro_gpu.client.graphmodel import ChangeCost
from astro_gpu.client.panel import MethodCard
from astro_gpu.client.post3 import ContSlider
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


def up_to_date(W, monkeypatch):
    """Its results said computed and up to date: a change of setting is a draft."""
    monkeypatch.setattr(W.m, "change_cost", lambda ch, paths=None: ChangeCost(["Block"], [(1, "x")], 60.0))


def bar(W):
    bars = [b for b in W.page.panel.findChildren(DraftBar) if b.isVisibleTo(W.page.panel)]
    return bars[0] if bars else None


def add_synth(W):
    W.choose(W.one("stars"))
    W.g.double_click_palette("synth")
    W.settle()
    return W.one("synth")


def test_a_recipe_parameter_is_a_draft_then_applied(world, monkeypatch):
    W = world(workflows=(), filters=("Ha", "OIII"))
    W.page.show_templates()
    card = W.page.templates.card("filters_full")
    card.set_palette("H-O")
    card.mousePressEvent(None)
    W.settle()
    up_to_date(W, monkeypatch)
    b = next(x for x in W.kinds("combine") if W.graph.blocks[x].params.get("combine_ho") is not None
             or "ho" in str(W.graph.blocks[x].source))
    monkeypatch.setattr(W.m, "recipe_parameters", lambda p: {"smoothing": {
        "label": "Colour smoothing", "min": 0, "max": 180, "step": 1, "default": 0,
        "value": (W.graph.blocks[b].params.get("combine_ho") or {}).get("smoothing", 0)}})
    W.choose(b)
    sp = next(x for x in W.page.panel.findChildren(QDoubleSpinBox) if x.maximum() == 180)
    sp.setValue(12)
    sp.editingFinished.emit()
    W.settle(2)
    d = W.m.pending_change
    assert d and [e["ch"]["kind"] for e in d.vals.values()] == ["recipe"]
    assert bar(W) is not None
    W.page.panel.draft_reset()
    W.settle(2)
    # its intensity, as every setting of the combination: a draft, then applied in one step
    it = next(x for x in W.page.panel.findChildren(QDoubleSpinBox) if x.objectName() == "intensity")
    it.setValue(0.5)
    it.editingFinished.emit()
    W.settle(2)
    d = W.m.pending_change
    assert d and [e["ch"]["kind"] for e in d.vals.values()] == ["combine"]
    assert not (W.graph.blocks[b].params.get("combine_ho") or {}).get("weights")      # nothing made at the click
    assert bar(W).b_variant.isEnabled(), bar(W).b_variant.toolTip()     # a variant: a second combination
    assert W.page.panel.draft_apply()
    W.settle(2)
    assert W.graph.blocks[b].params["combine_ho"]["weights"] and not W.m.pending_change


def test_synthetic_stars_method_and_combine_layers_values_are_drafts(world, monkeypatch):
    W = world(workflows=("Full",))
    syn = add_synth(W)
    up_to_date(W, monkeypatch)
    W.choose(syn)
    was = W.graph.blocks[syn].method
    other = next(c for c in W.page.panel.findChildren(MethodCard) if c.isVisibleTo(W.page.panel) and not c.on and c.ok)
    other.clicked.emit()
    W.settle(2)
    d = W.m.pending_change
    assert d and [e["ch"]["kind"] for e in d.vals.values()] == ["method"]
    assert W.graph.blocks[syn].method == was and bar(W) is not None
    assert W.page.panel.draft_apply()
    W.settle(2)
    assert W.graph.blocks[syn].method != was and not W.m.pending_change
    # Combine layers: a slider let go is a draft of the block, its layer of Post unchanged until Apply
    W.choose(W.one("combine_layers"))
    sl = next(x for x in W.page.panel.findChildren(ContSlider) if x.isVisibleTo(W.page.panel))
    sl.done.emit(3.0)
    W.settle(2)
    d = W.m.pending_change
    assert d and [e["ch"]["kind"] for e in d.vals.values()] == ["layer"]
    lid = next(iter(d.vals.values()))["ch"]["lid"]
    assert not (W.m.post_layer(lid) or {}).get("sliders")
    assert bar(W) is not None and not bar(W).b_variant.isEnabled()          # a layer of Post: no variant
    assert W.page.panel.draft_apply()
    W.settle(2)
    assert list(W.m.post_layer(lid)["sliders"].values()) == [3.0]

