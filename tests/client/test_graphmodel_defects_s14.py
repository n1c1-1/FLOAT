"""What a CPU-only laptop found on the new graph page over the project model (legacy.World: a scratch project, nothing
computed): blocks placed by hand go at the end of their chain, Align on offers only legal targets in the client's
words, a second Combine layers is numbered as the old page numbered it."""
from __future__ import annotations

from pathlib import Path

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


COPY64 = Path("C:/Users/Nicolas/sandbox/float_graphe_etude/copies/64_smart_seestar_s30_m16_dualband")


def _cl_on_its_line(g, view):
    """(Combine layers box, its Stars/background box) of each Combine layers of the drawing: on its line, the line
    through the middle of the Stars/background crosses the Combine layers."""
    from astro_gpu.graphcore import api
    d = api.layout(g, view)
    out = []
    for b, blk in g.blocks.items():
        if blk.kind != "combine_layers":
            continue
        src = next(w.src[0] for w in g.wires if w.dst == [b, "Background"])
        out.append((d.boxes[b], d.boxes[src]))
    return out


def test_a_combine_layers_after_a_combinations_stars_is_drawn_on_its_line(world):
    """A combination's Stars/background (a colour stack's H-O): its Combine layers continues the row on its line in the
    Composition band, as after a stack's Stars/background, never pushed below it."""
    W = world(workflows=("Full",), chains={"Colour full": ["calibration", "registration", "selection", "stacking",
                                                               "astrometry", "gradient", "denoise", "color"]})
    m = W.m
    q = W.view("Colour full")
    m.add_fusion("HOO", "ho", [{"pid": q.pid, "role": "Ha", "channel": "R"},
                               {"pid": q.pid, "role": "OIII", "channel": "GB"}])
    W.settle()
    W.choose(W.one("combine"))
    W.page.palette.double_click("stars")
    W.settle()
    pairs = _cl_on_its_line(W.graph, dict(m.graph_view))
    assert len(pairs) == 2
    for cl, stars in pairs:
        assert cl.y <= stars.y + stars.h / 2 <= cl.bottom and cl.x > stars.right, (cl, stars)


@pytest.mark.skipif(not COPY64.is_dir(), reason="no study copy on this machine")
def test_a_converted_project_draws_its_combine_layers_as_a_fresh_one(qapp, tmp_path):  # noqa: F811
    """The copy of a project of the previous format (a colour stack, its H-O combination with Stars/background): its
    Combine layers is drawn on its row's line in the Composition band, as a project made from scratch draws it."""
    import json
    import shutil

    from astro_gpu.client.model import ProjectModel
    from astro_gpu.client.services.worker import ApiWorker
    from astro_gpu.core import project2
    root = tmp_path / "copy"
    root.mkdir()
    shutil.copy(COPY64 / "project.astroproj", root / "project.astroproj")
    d = json.loads((root / "project.astroproj").read_text(encoding="utf-8"))
    d["scratch_dir"] = (tmp_path / "scratch").as_posix()
    (root / "project.astroproj").write_text(json.dumps(d, indent=2), encoding="utf-8")
    w = ApiWorker()
    m = ProjectModel(project2.load(root), w, readonly=True)
    try:
        assert w.wait(120) and m.ready, m.error
        pairs = _cl_on_its_line(m.stored, dict(m.graph_view))
        assert len(pairs) == 1
        cl, stars = pairs[0]
        assert cl.y <= stars.y + stars.h / 2 <= cl.bottom and cl.x > stars.right, (cl, stars)
    finally:
        m.close(save=False)
        w.stop()


def _layer_rows(W):
    """{block: (its row's name in the graph, its title on the canvas, its panel's title, its layer's name in Post)} of
    each Combine layers."""
    from astro_gpu.client.graphnode import node_title
    g = W.graph
    out = {}
    for r in g.rows:
        if r.blocks and g.blocks[r.blocks[0]].kind == "combine_layers":
            b = r.blocks[0]
            n = W.bm.node(b)
            lay = W.m.post_layer(n.lid) if n is not None and n.lid is not None else None
            out[b] = (r.name, W.page.canvas.blocks[b].d["title"], node_title(W.m, n), (lay or {}).get("name"))
    return out


def _reopened_names(W):
    """The Combine layers rows' names and Post's Combine layers layers' names of the project saved and opened again."""
    from astro_gpu.client.model import ProjectModel
    from astro_gpu.core import project2
    assert W.m.flush()
    m2 = ProjectModel(project2.load(W.tmp / "P"), W.w)
    try:
        assert W.w.wait(120) and m2.ready, m2.error
        g = m2.stored
        return ([r.name for r in g.rows if r.blocks and g.blocks[r.blocks[0]].kind == "combine_layers"],
                sorted(x["name"] for x in m2.post_version()["layers"] if x.get("part") == "combined"))
    finally:
        m2.close(save=False)


def test_a_second_combine_layers_takes_the_second_name_in_the_graph_too(world):
    """The hand pass on "SHO + RGB stars": a second Combine layers double-clicked in the palette renamed the template's
    row "Combine layers 2" and took "Combine layers" for its own row, while its block was titled "Combine layers 2"
    (the gesture is made on the graph without the rows drawn for Post's layers: the new row took the free name, the
    one drawn back the next). The one already there keeps its name and the new one is the second, as its block's
    title says, in the notice too; saved and opened again, the same."""
    W = world(workflows=(), filters=("SII", "Ha", "OIII", "R", "G", "B"), nights=0)
    W.template("sho_rgb")
    W.page.refresh()
    W.settle()
    (first, was), = _layer_rows(W).items()
    assert was[:2] == ("Combine layers", "Combine layers")
    W.said.clear()
    W.page.palette.double_click("combine_layers")
    W.settle()
    W.page.refresh()
    W.settle()
    got = _layer_rows(W)
    new = next(b for b in got if b != first)
    assert got[first] == was, got
    assert got[new][:2] == ("Combine layers 2", "Combine layers 2"), got
    assert W.said and "“Combine layers 2”" in W.said[-1], W.said
    assert _reopened_names(W)[0] == ["Combine layers", "Combine layers 2"]
    W.b.undo()
    W.settle()
    assert _layer_rows(W) == {first: was}


def test_two_combine_layers_of_a_workflow_have_one_numbering_in_the_graph_the_panel_and_post(world):
    """Two Combine layers over one workflow's Stars/background: the second is "Combine layers 2" as its row's name in
    the graph, its block's title, its panel's title and its layer in Post; the first keeps its row's name and its
    layer's; saved and opened again, the same."""
    W = world(workflows=("Full",))
    W.page.refresh()
    W.settle()
    (first, was), = _layer_rows(W).items()
    assert was[:2] == ("Combine layers", "Combine layers")      # (its layer is Post's default one: none stored yet)
    W.page.palette.double_click("combine_layers")
    W.settle()
    W.page.refresh()
    W.settle()
    got = _layer_rows(W)
    new = next(b for b in got if b != first)
    assert (got[first][0], got[first][1], got[first][3]) == ("Combine layers", "Combine layers", "Full · Combine layers")
    assert got[new] == ("Combine layers 2", "Combine layers 2", "Combine layers 2", "Full · Combine layers 2"), got
    assert _reopened_names(W) == (["Combine layers", "Combine layers 2"],
                                  ["Full · Combine layers", "Full · Combine layers 2"])
