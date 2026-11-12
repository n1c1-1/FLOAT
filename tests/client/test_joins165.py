"""Issue #165 in the client: free joins. Inside a workflow the blocks connect by their order; at a join (a combination's
named inputs, Combine layers' two sides, Align on's reference) a wire is dropped onto the input. An empty input is a
hollow dot, the block greyed with the reason; a loop is refused; one Ctrl+Z per wire; no wire goes right to left; the
"SHO + RGB stars" template makes the graph of the issue. Offscreen; nothing is computed."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if Path("C:/Windows/Fonts").is_dir():
    os.environ.setdefault("QT_QPA_FONTDIR", "C:/Windows/Fonts")
pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication  # noqa: E402

from astro_gpu.client import i18n  # noqa: E402
from astro_gpu.client.blocks import Graph  # noqa: E402
from astro_gpu.client.gallery import GALLERY, _view_opts  # noqa: E402
from astro_gpu.client.graph import GraphPage  # noqa: E402
from astro_gpu.client.model import ProjectModel  # noqa: E402
from astro_gpu.client.services.worker import ApiWorker  # noqa: E402
from astro_gpu.core import project2  # noqa: E402
from astro_gpu.core.project import Target  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture()
def world(qapp, tmp_path, monkeypatch):
    monkeypatch.setenv("ASTRO_GPU_SETTINGS", str(tmp_path / "settings.json"))
    i18n.set_language("en")
    p = project2.create(tmp_path / "P", "P", Target(name="M 31"))
    project2.save(p)
    w = ApiWorker()
    m = ProjectModel(p, w)
    assert w.wait(120)
    m._sync()
    page = GraphPage(m)
    page.resize(1800, 900)
    page.said = []
    page.toasted.connect(page.said.append)
    yield {"m": m, "w": w, "page": page}
    page.close()
    m.close(save=False)
    w.stop()


def settle(world):
    world["w"].wait(60)
    QApplication.processEvents()
    world["page"].refresh()
    QApplication.processEvents()


def template(world, key: str):
    t = next(c for c in GALLERY if c["key"] == key)
    world["page"].make_from_template(t, world["m"].unique_name(t["palette"]), _view_opts(t))
    settle(world)
    return t


def by_name(m, name):
    return next(q for q in m.paths if q.name == name)


def no_wire_goes_left(page):
    """Every wire a -> b: right(a) <= left(b); the one exception, the vertical reference wire of an Align on block
    aligned on another of its column."""
    page.graph.rebuild()
    wires = page.graph.wire_log
    assert wires
    for a, b, how in wires:
        if how == "vertical":
            assert a.key == "align_on" and b.key == "align_on" and a.col == b.col
            continue
        assert a.col + a.w <= b.col + 1e-6, (a.key, b.key, a.col + a.w, b.col)
    return wires


def test_the_sho_rgb_stars_template_makes_the_graph_of_the_issue(world):
    m = world["m"]
    template(world, "sho_rgb")
    names = [q.name for q in m.paths]
    assert names == ["SII", "Ha", "OIII", "RGB", "S-H-O"]
    a, b, c, rgb, sho = (by_name(m, n) for n in names)
    for q in (a, b, c, rgb):                          # each: ... Stacking, [Align on], Astrometry, Gradient, Denoise, Stars
        blocks = [n["block"] for n in m.graph(q)]
        assert blocks[blocks.index("stacking") + (2 if q is not a else 1)] == "astrometry"
        assert blocks[-1] == "stars" and blocks.index("gradient") < blocks.index("denoise") < blocks.index("stars")
        assert ("align_on" in blocks) == (q is not a)
        assert (m.align_reference(q) == a.pid) == (q is not a)                       # B, C and RGB on A
    assert "color" in [n["block"] for n in m.graph(rgb)] and "color" not in [n["block"] for n in m.graph(b)]
    f = m.p.workflow(sho.pid).source["fusion"]
    assert f["type"] == "sho"
    assert [(i["role"], i["pid"], i["image"]) for i in f["inputs"]] == [("SII", a.pid, "starless"),
                                                                       ("Ha", b.pid, "starless"),
                                                                       ("OIII", c.pid, "starless")]
    assert [n["block"] for n in m.graph(sho)] == []                                   # no Stars/background after it
    layers = [x for x in m.post_layers() if x.get("kind") == "workflow"]
    assert len(layers) == 1
    assert (layers[0]["pid"], layers[0]["part"], layers[0]["starless"], layers[0]["stars_pid"]) == (
        sho.pid, "combined", "output", rgb.pid)
    assert {tuple(x) for x in m.p.post["removed"]} >= {(sho.pid, None), (rgb.pid, "combined")}
    g = Graph(m)
    assert [name for name, _pid in g.post_layers] == [layers[0]["name"]]            # Post shows exactly this picture
    comp = g.compositions[0]
    (starless, _s), (stars, _t) = g.composition_inputs[id(comp)]
    assert starless is g.node_of(sho, "fusion") and stars is g.node_of(rgb, "etoiles")
    fn = g.node_of(sho, "fusion")
    assert [(a_ is g.node_of(q, "etoiles"), role, image) for (a_, role, image), q in
            zip(g.fusion_inputs[id(fn)], (a, b, c))] == [(True, "SII", "starless"), (True, "Ha", "starless"),
                                                        (True, "OIII", "starless")]
    stack_a = g.node_of(a, "empilement")
    assert [g.align_refs[id(g.node_of(q, "align_on"))] is stack_a for q in (b, c, rgb)] == [True] * 3
    assert not fn.broken and not comp.waits and not [n for n in g.nodes if n.key == "align_on" and n.waits]
    assert [n["block"] for n in m.graph(rgb)].count("stars") == 1                     # the stars are taken off once only


def test_the_sho_rgb_template_shows_no_ghost_combine_layers_and_names_its_colour_workflow_rgb(world):
    m = world["m"]
    template(world, "sho_rgb")
    g = Graph(m)
    assert len(g.compositions) == 1 and not g.awaits_post                  # the issue's Combine layers, nothing else
    assert [n for n in g.nodes if n.key == "composition" and n.paths[0].name in ("SII", "Ha", "OIII")] == []
    assert m.post_defaults() == []
    assert [s.name for s in m.p.sessions][-1] == "RGB"


def test_a_combination_waiting_for_frames_is_greyed_never_red(world):
    m = world["m"]
    template(world, "sho_rgb")
    fn = Graph(m).node_of(by_name(m, "S-H-O"), "fusion")
    assert fn.waits and not fn.broken and "no lights" in fn.waits


def test_the_names_of_the_inputs_never_sit_under_the_title_nor_the_subtitle(world):
    from PySide6.QtCore import QRectF
    from PySide6.QtGui import QFont, QFontMetricsF
    page = world["page"]
    template(world, "sho_rgb")
    page.graph.rebuild()
    items = [it for it in page.graph.scene().items() if hasattr(it, "inputs") and getattr(it, "inputs", None)]
    assert {it.node.key for it in items} >= {"fusion", "composition", "align_on"}
    for it in items:
        names = [QRectF(9, d["y"] - 9, it.name_w, 18) for d in it.inputs]
        assert all(a.bottom() <= b.top() + 1e-6 for a, b in zip(names, names[1:]))        # one line each
        f = QFont("Segoe UI", 10)
        f.setBold(True)
        x0 = 11 + it.name_w
        full = QFontMetricsF(f).horizontalAdvance(it.title)
        title = QRectF(x0, 6, full, 20)
        sub = QRectF(x0, 28, 10, 18)
        assert it.w - 7 - x0 >= full, (it.node.key, it.title, it.w)                      # the title whole, right of the names
        for r in names:
            assert not r.intersects(title) and not r.intersects(sub)
            assert r.right() <= x0


def test_the_wires_into_a_join_each_bend_at_their_own_place(world):
    page = world["page"]
    template(world, "sho_rgb")
    page.graph.rebuild()
    from astro_gpu.client.graph import stagger
    assert len({stagger(i, 3, True) for i in range(3)}) == 3 and stagger(0, 1, True) == 0


def test_no_wire_goes_right_to_left(world):
    m, page = world["m"], world["page"]
    template(world, "sho_rgb")
    wires = no_wire_goes_left(page)
    assert any(a.key == "etoiles" and b.key == "fusion" for a, b, _h in wires)                 # the joins are in the log
    assert any(b.key == "composition" for _a, b, _h in wires) and any(b.key == "align_on" for _a, b, _h in wires)
    # a Stars/background after a combination (the graph is free) is drawn after it, a second combination after the first
    sho = by_name(m, "S-H-O")
    page.place_block("stars", sho.pid)
    settle(world)
    page.place_fusion("ho")
    settle(world)
    ho = m.paths[-1]
    m.set_fusion_input(ho, "Ha", sho.pid)                                       # a combination reading a combination
    m.set_fusion_input(m.view(ho.pid), "OIII", by_name(m, "Ha").pid)
    settle(world)
    no_wire_goes_left(page)
    g = page.graph.graph
    assert page.graph.col_of(g.node_of(sho, "fusion")) == "f0" and page.graph.col_of(g.node_of(m.view(ho.pid), "fusion")) == "f1"
    assert page.graph.col_of(g.node_of(sho, "etoiles")) == "s0"
    cols = page.graph.columns(g)
    assert cols.index("f0") < cols.index("s0") < cols.index("f1") < cols.index("composition")


def test_the_filters_templates_align_the_others_on_the_first(world):
    m = world["m"]
    template(world, "filters_full")
    first, second, third, fusion = m.paths[-4:]
    assert m.align_reference(first) is None and m.align_reference(second) == first.pid == m.align_reference(third)
    assert [q.pid for q in m.paths if q.source and q.source.get("fusion")] == [fusion.pid]
    assert [n["block"] for n in m.graph(fusion)] == []                                  # no Stars/background after it
    g = Graph(m)
    assert "to wire" not in g.node_of(fusion, "fusion").waits             # nothing to wire (it waits for lights)
    assert [name for name, _pid in g.post_layers] == [fusion.name]                      # inputs feed it; it goes to Post


def test_a_combination_placed_from_the_palette_is_empty_and_each_input_is_a_hollow_dot(world):
    m, page = world["m"], world["page"]
    template(world, "filters_min")
    n = len(m.paths)
    page.place_fusion("sho")
    settle(world)
    fu = m.paths[-1]
    assert len(m.paths) == n + 1 and fu.source["fusion"]["inputs"] == []              # never chosen by the filters
    g = Graph(m)
    node = g.node_of(fu, "fusion")
    assert node.waits == "S II, Hα, O III to wire"
    page.graph.rebuild()
    from astro_gpu.client.graph import BlockItem
    it = next(x for x in page.graph.scene().items() if isinstance(x, BlockItem) and x.node.key == "fusion")
    assert [(d["name"], d["wired"]) for d in it.inputs] == [("S II", False), ("Hα", False), ("O III", False)]
    assert "to wire" in it.toolTip() and it.h >= 3 * 18 + 10                              # names spread, block grown
    assert len({d["y"] for d in it.inputs}) == 3 and all(0 < d["y"] < it.h for d in it.inputs)


def test_a_wire_dropped_on_an_input_wires_it_one_step_each_and_undo_takes_it_back(world):
    m, page = world["m"], world["page"]
    template(world, "filters_min")                                                       # Ha and OIII (H-O)
    ha, oiii = m.paths[-2:]
    page.place_fusion("ho")
    settle(world)
    fu = m.paths[-1]
    g = Graph(m)
    gv = page.graph
    src = g.node_of(ha, "empilement")
    dst = g.node_of(fu, "fusion")
    go, why = gv.plan_wire(g.leaves[ha.pid], None, dst, ("port", "Ha"))
    assert go is not None and not why
    n0 = len(m._undo)
    page.wire_to_input(g.leaves[ha.pid], None, dst, ("port", "Ha"))
    settle(world)
    assert len(m._undo) == n0 + 1 and m.view(fu.pid).source["fusion"]["inputs"][0]["pid"] == ha.pid
    g = Graph(m)
    page.wire_to_input(g.leaves[oiii.pid], None, g.node_of(m.view(fu.pid), "fusion"), ("port", "OIII"))
    settle(world)
    assert len(m._undo) == n0 + 2
    g = Graph(m)
    assert not g.node_of(m.view(fu.pid), "fusion").waits.startswith("Hα") or "no lights" in g.node_of(m.view(fu.pid), "fusion").waits                # one input left to wire at most
    assert m.undo()
    settle(world)
    assert [i["role"] for i in m.view(fu.pid).source["fusion"]["inputs"]] == ["Ha"]
    assert m.undo()
    settle(world)
    assert m.view(fu.pid).source["fusion"]["inputs"] == []
    assert src is not None


def test_the_stars_alone_cannot_feed_a_combination_and_a_loop_is_refused(world):
    m, page = world["m"], world["page"]
    template(world, "sho_rgb")
    sho, rgb, a = by_name(m, "S-H-O"), by_name(m, "RGB"), by_name(m, "SII")
    g = Graph(m)
    gv = page.graph
    go, why = gv.plan_wire(g.node_of(rgb, "etoiles"), "stars", g.node_of(sho, "fusion"), ("port", "SII"))
    assert go is None and "stars alone" in why
    # a workflow that the combination reads cannot read the combination: Align on's reference is a loop, and as an input
    assert "loop" in m.join_problem(a.pid, sho.pid) and m.join_problem(sho.pid, sho.pid)
    page.place_fusion("ho")
    settle(world)
    ho = m.paths[-1]
    m.set_fusion_input(ho, "Ha", sho.pid)
    settle(world)
    n0 = len(m._undo)
    go, why = gv.plan_wire(Graph(m).leaves[ho.pid], None, Graph(m).node_of(m.view(sho.pid), "fusion"), ("port", "SII"))
    assert go is None and "loop" in why                                                # S-H-O reads H-O which reads S-H-O
    page.wire_to_input(Graph(m).leaves[ho.pid], None, Graph(m).node_of(m.view(sho.pid), "fusion"), ("port", "SII"))
    assert "loop" in page.said[-1]
    assert m.view(sho.pid).source["fusion"]["inputs"][0]["pid"] == a.pid
    assert len(m._undo) == n0                                                           # the refused wire made no step


def test_the_two_sides_of_combine_layers_are_wired_and_unwired(world):
    m, page = world["m"], world["page"]
    template(world, "sho_rgb")
    rgb, a = by_name(m, "RGB"), by_name(m, "SII")
    g = Graph(m)
    comp = g.compositions[0]
    lid = comp.lid
    assert lid is not None and not comp.waits
    gv = page.graph
    # the stars side takes the stars alone of a Stars/background block, not an Output, not the starless image
    go, why = gv.plan_wire(g.node_of(a, "etoiles"), "starless", comp, ("side", "stars"))
    assert go is None and "stars alone" in why.lower()
    n0 = len(m._undo)
    page.wire_to_input(g.node_of(a, "etoiles"), "stars", comp, ("side", "stars"))          # another workflow's stars
    settle(world)
    assert m.post_layer(lid)["stars_pid"] == a.pid and len(m._undo) == n0 + 1
    assert m.undo()
    settle(world)
    assert m.post_layer(lid)["stars_pid"] == rgb.pid
    assert m.set_layer_side(lid, "stars", None)                                          # unwired: the side is empty
    settle(world)
    g = Graph(m)
    assert g.compositions[0].waits == "Stars to wire"                                  # greyed with its reason
    assert [d["wired"] for d in page.graph._inputs_of(g, g.compositions[0])] == [True, False]
    no_wire_goes_left(page)
    assert m.set_layer_side(lid, "stars", rgb.pid)


def test_the_names_are_french_with_aligner_sur(world):
    from astro_gpu.client.blocks import TITLES, block_title
    assert TITLES["align_on"] == "Align on"
    i18n.set_language("fr")
    try:
        assert block_title(None, "align_on") == "Aligner sur"
        assert i18n._("reference to wire") == "référence à brancher"
        assert i18n._("SHO + RGB stars") != "SHO + RGB stars"
    finally:
        i18n.set_language("en")
