"""Synthetic stars on the stars side read from another workflow, on the page over a project model (the owner's decision
of 14 November): the S-H-O template (after its "S-H-O · Stars" combination) and Dual-band x2 (after its star row's
Stars/background). Behind graphcore.model.SYNTH_ON_STARS_SIDE: these tests switch it on and run once the engine runs
the block there and Post puts its row in the Combine layers layer of the stars-side workflow (skipped before, saying
why). Added: one undo step, the third input wired, the stars-side workflow wanted by Post; removed: the input empty."""
from __future__ import annotations

import pytest

from astro_gpu.graphcore import model as gm
from astro_gpu.graphcore.model import SYNTH_PORT
from tests.client.graphview.legacy import World
from tests.client.pagehelp import qapp  # noqa: F401


@pytest.fixture()
def on(monkeypatch):
    monkeypatch.setattr(gm, "SYNTH_ON_STARS_SIDE", True)


def post_reads_stars_side():
    """Whether Post puts the synthetic stars row by the stars-side workflow (main's change)."""
    import inspect
    from astro_gpu.client import postengine as pe
    return "stars_pid" in inspect.getsource(pe.with_synth_rows)


@pytest.mark.parametrize("key,filters", [("filters_full", ("SII", "Ha", "OIII")), ("dual_x2", ())])
def test_synthetic_stars_on_the_stars_side_of_a_template(qapp, tmp_path, on, key, filters):  # noqa: F811
    W = World(tmp_path / "w", workflows=(), filters=filters, nights=0 if filters else 1)
    try:
        W.template(key)
        g = W.graph
        cl = W.one("combine_layers")
        src = next(w.src for w in g.wires if w.dst == [cl, "Stars"] and w.active)
        W.choose(src[0])
        steps = W.steps()
        W.g.double_click_palette("synth")
        W.settle()
        if not W.kinds("synth"):
            pytest.skip(f"the engine does not run Synthetic stars there yet: {W.said[-1:]}")
        if not post_reads_stars_side():
            pytest.skip("Post does not put the synthetic stars row by the stars-side workflow yet")
        g = W.graph
        syn = W.one("synth")
        ix = gm.Index(g)
        assert ix.pred(syn) == src[0] and W.steps() == steps + 1
        assert any(w.src == [syn, SYNTH_PORT] and w.dst == [cl, SYNTH_PORT] for w in g.wires)
        stars_pid = next(r.wid for r in g.rows if r.id == ix.row_of[syn])
        assert stars_pid in W.m.synth_wanted()
        W.m.undo(graph=True)
        W.settle()
        assert not W.kinds("synth")
        assert not any(w.dst == [W.one("combine_layers"), SYNTH_PORT] for w in W.graph.wires)
    finally:
        W.close()
