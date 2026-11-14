"""Synthetic stars on the stars side, read back and waited for (14 November): (1) a "S-H-O · Stars" workflow whose blocks
are ["output", "synth"] (the block placed outside the page, or by the engine's own call) converts to the graph with the
block after the combination, drawn under its row, wired to the Combine layers' third input, and selectable, removable
and undone on the page; (2) a Combine layers whose stars side has a Synthetic stars not computed yet is not "done":
it waits for it as for an input to compute (a colour workflow, the S-H-O template, a star row)."""
from __future__ import annotations

import pytest

from astro_gpu import api
from astro_gpu.graphcore import model as gm
from astro_gpu.graphcore.compile import compile_into
from astro_gpu.graphcore.convert import convert
from astro_gpu.graphcore.layout import layout
from astro_gpu.graphcore.model import SYNTH_PORT
from tests.client.graphview.legacy import World
from tests.client.pagehelp import qapp  # noqa: F401

CASES = [("Full", (), None), ("filters_full", ("SII", "Ha", "OIII"), "template"), ("dual_x2", (), "template")]


def world(tmp_path, key, filters, how):
    if how is None:
        W = World(tmp_path / "w", workflows=(key,))
    else:
        W = World(tmp_path / "w", workflows=(), filters=filters, nights=0 if filters else 1)
        W.template(key)
    return W


def stars_side(W):
    cl = W.one("combine_layers")
    return cl, next(w.src for w in W.graph.wires if w.dst == [cl, "Stars"] and w.active)


def place(W):
    cl, src = stars_side(W)
    wid = next(f.wid for f in W.graph.rows if src[0] in f.blocks)
    W.m.insert_block(W.m.view(wid), "synth")          # the engine's own call (Post's), outside the page
    W.settle()
    return cl, src, W.one("synth"), wid


def computed(m, but=()):
    """Every stage taken as computed and up to date, but the stages named."""
    for w in m.p.workflows:
        for st in m._chains.get(w.id, []):
            if st in but:
                m._states.pop((w.id, st), None)
                continue
            try:
                m._states[(w.id, st)] = api.StageState("done", "", api.stage_key(m.p, w.id, st))
            except (ValueError, KeyError):
                pass


@pytest.mark.parametrize("key,filters,how", CASES)
def test_a_combine_layers_waits_for_the_synthetic_stars_of_its_stars_side(qapp, tmp_path, key, filters, how):  # noqa: F811
    W = world(tmp_path, key, filters, how)
    try:
        cl, _src, syn, _wid = place(W)
        bm = W.m.block_map
        computed(W.m, but=("synth_stars",))
        assert bm.state(bm.node(syn)) == "pending"
        assert bm.state(bm.node(cl)) != "done"                    # never "Up to date" before its Synthetic stars
        computed(W.m)
        assert bm.state(bm.node(cl)) == "done"
    finally:
        W.close()


def test_a_stars_combination_with_synthetic_stars_converts_draws_and_is_removed_on_the_page(qapp, tmp_path):  # noqa: F811
    W = world(tmp_path, "filters_full", ("SII", "Ha", "OIII"), "template")
    try:
        cl, src, syn, wid = place(W)
        assert [n["block"] for n in W.m.p.workflow(wid).blocks] == ["output", "synth"]
        # converted from that definition: the block after the combination, the same definition back
        g = convert(W.m.p)
        row = next(r for r in g.rows if r.wid == wid)
        assert [g.blocks[b].kind for b in row.blocks] == ["combine", "synth"]
        q = type(W.m.p).from_dict(W.m.p.to_dict(), W.m.p.root)
        compile_into(g, q)
        assert [n["block"] for n in q.workflow(wid).blocks] == ["output", "synth"]
        d = layout(g, {"folded": False})
        s, head = row.blocks[1], row.blocks[0]
        assert d.boxes[s].y >= d.boxes[head].bottom and d.boxes[s].x >= d.boxes[head].right
        # on the page: drawn under the combination, into the third input; chosen, deleted, undone
        assert any(w.src == [syn, SYNTH_PORT] and w.dst == [cl, SYNTH_PORT] for w in W.graph.wires)
        W.choose(syn)
        assert W.b.selected == syn and W.page.canvas.blocks[syn].d["selected"]
        steps = W.steps()
        W.g.click_cross(syn)
        W.settle()
        assert not W.kinds("synth") and W.steps() == steps + 1
        assert not any(w.dst[1] == SYNTH_PORT for w in W.graph.wires)
        W.g.key("Ctrl+Z", W.page.canvas)
        W.settle()
        assert W.kinds("synth") and any(w.dst == [W.one("combine_layers"), SYNTH_PORT] for w in W.graph.wires)
        W.choose(W.one("synth"))
        W.g.key("Del", W.page.canvas)
        W.settle()
        assert not W.kinds("synth")
        assert gm.Index(W.graph).row[gm.Index(W.graph).row_of[src[0]]].blocks == [src[0]]
    finally:
        W.close()
