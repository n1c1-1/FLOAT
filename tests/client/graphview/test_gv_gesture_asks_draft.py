"""A gesture of the graph or of the Table on a block holding an unapplied draft asks first the question leaving the
block asks (Apply, Try variant, Discard, Stay), Stay taking the gesture back; a gesture that leaves that block as it
is asks nothing and the draft stays (each change of the model used to drop it silently)."""
from __future__ import annotations

from astro_gpu.client.blockdraft import DRAFT_ANSWER_ENV
from astro_gpu.client.panel import MethodCard
from tests.client.graphview.test_gv_drafts_everywhere import add_synth, up_to_date, world  # noqa: F401  (fixture)


def test_a_gesture_on_the_drafted_block_asks_first_and_stay_takes_it_back(world, monkeypatch):
    W = world(workflows=("Full",))
    syn = add_synth(W)
    up_to_date(W, monkeypatch)
    W.choose(syn)
    next(c for c in W.page.panel.findChildren(MethodCard) if c.isVisibleTo(W.page.panel) and not c.on
         and c.ok).clicked.emit()
    W.settle(2)
    d = W.m.pending_change
    assert d
    monkeypatch.setenv(DRAFT_ANSWER_ENV, "stay")
    res = W.b.act("remove_block", block=syn)                                # its block deleted: asked, Stay
    W.settle(2)
    assert not res.ok and syn in W.graph.blocks and W.m.pending_change is d
    # a gesture that leaves the block as it is asks nothing (Stay would have kept it from happening): the draft stays
    row = next(r for r in W.graph.rows if r.wid is not None)
    assert W.b.act("rename_row", row=row.id, name="Other name").ok
    W.settle(2)
    assert W.m.pending_change is d
    monkeypatch.setenv(DRAFT_ANSWER_ENV, "discard")
    res = W.b.act("remove_block", block=syn)                                # Discard: the draft goes, the gesture is made
    W.settle(2)
    assert res.ok and syn not in W.graph.blocks and not W.m.pending_change
