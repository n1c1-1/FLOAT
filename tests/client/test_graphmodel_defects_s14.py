"""What a CPU-only laptop found on the new graph page over the project model (legacy.World: a scratch project, nothing
computed): blocks placed by hand go at the end of their chain, Align on offers only legal targets in the client's
words, a second Combine layers is numbered as the old page numbered it."""
from __future__ import annotations

import pytest

from tests.client.graphview.legacy import World
from tests.client.pagehelp import qapp  # noqa: F401


@pytest.fixture()
def world(qapp, tmp_path):  # noqa: F811
    made = []

    def make(**kw):
        w = World(tmp_path / f"w{len(made)}", **kw)
        made.append(w)
        return w
    yield make
    for w in made:
        w.close()


def _chain(W, row):
    return [W.graph.blocks[b].kind for b in row.blocks]


@pytest.mark.parametrize("folded", [False, True])
def test_blocks_placed_from_the_palette_go_at_the_end_of_the_chain_in_order(world, folded):
    """A stack built by hand from the palette (Session, Registration, Stacking), then Astrometry, Gradient, Denoise
    double-clicked in that order: each goes after the last block of the chain, in the engine's order, never right after
    Stacking (unfolded and folded)."""
    W = world(workflows=(), nights=1)
    W.page.palette.double_click("session")
    W.settle()
    if folded:
        assert W.b.act("fold_all").ok
        W.settle()
    for k in ("registration", "stacking", "astrometry", "gradient", "denoise"):
        W.page.palette.double_click(k)
        W.settle()
    row = next(r for r in W.graph.rows if r.nights)
    assert _chain(W, row) == ["registration", "stacking", "astrometry", "gradient", "denoise"]
    q = W.m.paths[0]
    assert [n["block"] for n in W.m.graph(q)][-3:] == ["astrometry", "gradient", "denoise"]


def _align_menu(W, rid):
    from astro_gpu.client.graphview.menu import menu
    items = menu(W.page.bridge.doc, rid)
    from astro_gpu.client.i18n import _
    return next(i for i in items if i and i[1] == "submenu" and i[0] == _("Align on"))[2]


def test_align_on_offers_only_workflows_the_engine_accepts_and_says_why_in_the_clients_words(world):
    """Align on (the row's menu) never offers a Combine layers (a layer of Post, no workflow) nor a combination (never a
    reference: the engine's join_problem); a workflow that would make a loop is greyed with the old words, translated;
    each target offered is accepted by the model (no "row not found")."""
    from astro_gpu.client import i18n
    W = world(workflows=(), filters=("SII", "Ha", "OIII"), nights=0)
    W.template("filters_full")
    g = W.graph
    sii = next(r for r in g.rows if r.name == "SII")
    targets = [i for i in _align_menu(W, sii.id) if i and i[1] == "set_align"]
    names = {i[0].lstrip("✓ ") for i in targets}
    assert names == {"Ha", "OIII"}, names                 # no S-H-O, S-H-O · Stars, Combine layers
    assert all(not i[3] and i[4] == f"{i[0]} already reads SII: it would make a loop" for i in targets)
    i18n.set_language("fr")
    try:
        why = [i[4] for i in _align_menu(W, sii.id) if i and i[1] == "set_align"]
        assert why and all("cela ferait une boucle" in w for w in why), why
    finally:
        i18n.set_language("en")
    ha = next(r for r in g.rows if r.name == "Ha")
    for it in _align_menu(W, ha.id):
        if it and it[1] == "set_align" and it[3]:
            res = W.b.act("set_align", **it[2])
            assert res.ok, W.b.say(res.reason)
            W.settle()


def test_a_combine_layers_row_is_refused_as_a_target_with_the_engines_words(world):
    """A Combine layers row is no workflow: never offered, and the core's judge refuses it as the engine words a target
    without a stack."""
    W = world(workflows=("Full",))
    g = W.graph
    cl = next(r for r in g.rows if r.blocks and g.blocks[r.blocks[0]].kind == "combine_layers")
    row = next(r for r in g.rows if r.nights)
    assert not any(i and i[0].lstrip("✓ ") == cl.name for i in _align_menu(W, row.id) if i and i[1] == "set_align")
    res = W.b.doc.try_many([("set_align", {"row": row.id, "ref": cl.id})])
    assert not res.ok and W.b.say(res.reason) == "no stack"


def _layers(W):
    g = W.graph
    return [(r.blocks[0], W.page.canvas.blocks[r.blocks[0]].d["title"]) for r in g.rows
            if r.blocks and g.blocks[r.blocks[0]].kind == "combine_layers"]


def test_a_second_combine_layers_is_numbered_and_the_first_keeps_its_name(world):
    """Stars/background draws its Combine layers; a second one placed from the palette is the new one, numbered
    ("Combine layers 2", Post's "Full · Combine layers 2"), the wired first one keeps its block and its name; one Undo
    takes the second away."""
    W = world(workflows=("Full",))
    W.page.refresh()
    W.settle()
    (first, title), = _layers(W)
    assert title == "Combine layers"
    W.page.palette.double_click("combine_layers")
    W.settle()
    W.page.refresh()
    W.settle()
    got = dict(_layers(W))
    assert got.get(first) == "Combine layers" and len(got) == 2, got
    new = next(b for b in got if b != first)
    assert got[new] == "Combine layers 2"
    assert sorted(x.get("name") for x in W.m.post_version()["layers"]) == ["Full · Combine layers",
                                                                           "Full · Combine layers 2"]
    W.b.undo()
    W.settle()
    assert [b for b, _t in _layers(W)] == [first]
