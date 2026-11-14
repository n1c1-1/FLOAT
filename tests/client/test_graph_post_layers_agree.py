"""The graph and Post agree on the Combine layers of a project opened as it was saved (the owner's project of 14
November: saved with Synthetic stars in the chain, its Combine layers made by a gesture after the Stars/background,
the block had no wire to Post while Post held its layer). A Combine layers that Post shows as a layer has its wire to
Post; Post's layers and the graph's blocks are as many; no Combine layers stands for a layer Post does not hold.
Scratch projects, nothing computed."""
from __future__ import annotations

import copy
import json

import pytest
from PySide6.QtWidgets import QApplication

from astro_gpu.graphcore.model import SYNTH_PORT
from astro_gpu.graphcore.post import layer_rows, stars_side
from tests.client.graphview.legacy import World
from tests.client.pagehelp import qapp  # noqa: F401


def agree(m, g):
    """The invariant: every Combine layers block stands for a Combine layers layer of the shown version (or the one
    Post shows by default) of its workflow, with its wire to Post; and Post's layers are as many as the blocks."""
    post = next(b for b, x in g.blocks.items() if x.kind == "post")
    rows = layer_rows(g)
    heads = {r.blocks[0] for r in g.rows if r.blocks and g.blocks[r.blocks[0]].kind == "combine_layers"}
    drawn = {r.blocks[0] for rids in rows.values() for r in g.rows if r.id in rids}
    assert heads == drawn
    for j in heads:
        assert any(w.src == [j, "image"] and w.dst[0] == post for w in g.wires), j
    want = {}
    for x in m.post_version().get("layers") or []:
        if x.get("kind") == "workflow" and x.get("part") == "combined":
            want[x["pid"]] = want.get(x["pid"], 0) + 1
    for q in m.post_defaults():
        if m.post_default_part(q) == "combined":
            want[q.pid] = want.get(q.pid, 0) + 1
    for pid, n in want.items():
        assert len(rows.get(stars_side(g, pid), [])) == n, (pid, n, rows)
    assert sum(want.values()) == len(heads)


def settle(w, n=12):
    for _i in range(n):
        w.wait(50)
        QApplication.processEvents()


@pytest.fixture()
def saved(qapp, tmp_path):  # noqa: F811
    """A Full workflow with Synthetic stars and its Combine layers layer in Post, saved; its file's dict."""
    from astro_gpu.core import project2
    W = World(tmp_path / "w", workflows=("Full",))
    W.m.insert_block(W.m.paths[0], "synth")
    W.settle()
    W.m.add_post_layer(W.m.paths[0].pid, "combined", keep_default=False)
    W.settle()
    agree(W.m, W.graph)
    assert W.m.flush()
    path = W.m.p.root / project2.FILE
    W.close()
    yield path, json.loads(path.read_text(encoding="utf-8"))


def previous_drawing(d, wire_to_post=False):
    """The graph as the previous drawing saved it: the synthetic stars on the Stars input (the real stars an inactive
    alternative there), the starless image read from the Synthetic stars; its Combine layers made by a gesture after
    the Stars/background: no wire to Post."""
    g = d["graph"]
    kind = {b: x["kind"] for b, x in g["blocks"].items()}
    post = next(b for b, k in kind.items() if k == "post")
    out = []
    syn = next(b for b, k in kind.items() if k == "synth")
    for w in g["wires"]:
        if w["dst"][1] == SYNTH_PORT:
            w["src"], w["dst"] = [syn, "Stars only"], [w["dst"][0], "Stars"]
        elif w["dst"][1] == "Stars" and kind[w["dst"][0]] == "combine_layers":
            w["active"] = False
        elif w["dst"][1] == "Background":
            w["src"] = [syn, "Starless"]
        elif kind.get(w["src"][0]) == "combine_layers" and w["dst"][0] == post and not wire_to_post:
            continue
        out.append(w)
    g["wires"] = out
    return d


def reopen(qapp, tmp_path, d):  # noqa: F811
    from astro_gpu.client.graphview.page import GraphPage
    from astro_gpu.client.model import ProjectModel
    from astro_gpu.client.services.worker import ApiWorker
    from astro_gpu.core import project2
    root = tmp_path / "again"
    root.mkdir()
    (root / "project.astroproj").write_text(json.dumps(d), encoding="utf-8")
    w = ApiWorker()
    m = ProjectModel(project2.load(root), w)
    assert w.wait(120) and m.ready, m.error
    page = GraphPage(m)
    page.resize(1500, 800)
    page.show()
    settle(w)
    return w, m, page


def close(w, m, page):
    page.close()
    page.deleteLater()
    m.close(save=False)
    w.stop()


def layer(pid, lid, name):
    return {"id": lid, "kind": "workflow", "pid": pid, "part": "combined", "starless": "starless", "stars_pid": pid,
            "name": name, "visible": True, "opacity": 100.0, "mode": "Normal", "base": None, "sliders": {},
            "mask": None, "blend": "Screen", "shown": {"stars": True, "starless": True}}


@pytest.mark.parametrize("case", ["one layer", "the defect's two layers", "removed from the shown version only"])
def test_a_project_saved_in_the_previous_drawing_opens_with_graph_and_post_agreeing(qapp, tmp_path, saved, case):  # noqa: F811
    path, d = saved
    d = previous_drawing(copy.deepcopy(d))
    pid = d["graph"]["rows"][0]["wid"]
    versions = d["post"]["versions"]
    mine = next(x for x in versions[0]["layers"] if x.get("part") == "combined")
    if case == "the defect's two layers":                 # the earlier defect's second layer, above the first
        versions[0]["layers"].insert(0, layer(pid, 9001, mine["name"].replace(" · ", " 2 · ")))
    if case == "removed from the shown version only":     # a second version holds it, the shown one took it out
        other = copy.deepcopy(versions[0])
        other.update(id=2, name="Version 2")
        versions[0]["layers"] = [x for x in versions[0]["layers"] if x.get("part") != "combined"]
        versions[0]["removed"] = [[pid, "combined"]]
        versions.append(other)
    w, m, page = reopen(qapp, tmp_path, d)
    try:
        g = page.bridge.graph
        agree(m, g)
        cl = [b for b, x in g.blocks.items() if x.kind == "combine_layers"]
        assert len(cl) == {"one layer": 1, "the defect's two layers": 2, "removed from the shown version only": 0}[case]
        syn = next(b for b, x in g.blocks.items() if x.kind == "synth")
        for j in cl:                                       # drawn as now: its three inputs
            ins = {x.dst[1]: x.src for x in g.wires if x.dst[0] == j and x.active}
            assert ins[SYNTH_PORT] == [syn, SYNTH_PORT] and ins["Stars"][1] == "Stars only"
        if case == "removed from the shown version only":
            assert any(x.get("part") == "combined" for x in m.p.post["versions"][1]["layers"])
    finally:
        close(w, m, page)


def test_a_combine_layers_placed_after_a_stars_background_has_its_wire_to_post():
    """The cause: a Combine layers put after a Stars/background whose image did not go to Post had no wire to Post."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "graphcore"))
    from fixtures import Doc
    from test_points import ok, row
    doc = Doc.from_template("colour_complete")
    p = doc.project
    post = next(b for b, x in p.blocks.items() if x.kind == "post")
    j = next(b for b, x in p.blocks.items() if x.kind == "combine_layers")
    ok(doc, "delete_row", row=row(doc, "Combine layers").id)
    stars = row(doc, "Session 1").blocks[-1]
    assert not any(w.src[0] == stars and w.dst[0] == post for w in doc.project.wires)
    ok(doc, "add_block", kind="combine_layers", at={"type": "comp", "after": stars})
    j = next(b for b, x in doc.project.blocks.items() if x.kind == "combine_layers")
    assert any(w.src == [j, "image"] and w.dst[0] == post for w in doc.project.wires)


def test_the_projection_keeps_the_wires_to_post_it_finds():
    """Post's wires drawn again over a graph that already has them change nothing (ids, order, next id): an Undo gives
    the stored graph back exactly (the window fuzz of 14 November: S-H-O + RGB, a block of OIII moved, then a
    Registration deleted; Ctrl+Z gave the S-H-O's wire to Post back under a new id)."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "graphcore"))
    from fixtures import Doc
    from astro_gpu.graphcore.model import model_json
    from astro_gpu.graphcore.post import wired_images, with_post_wires
    for key in ("colour_minimal", "several_images", "sho_rgb", "colour_complete", "filters_complete"):
        g = Doc.from_template(key).project
        assert model_json(with_post_wires(g, wired_images(g))) == model_json(g), key


def test_deleting_a_registration_of_s_h_o_and_rgb_is_undone_exactly(qapp, tmp_path):  # noqa: F811
    from astro_gpu.graphcore.model import model_json
    W = World(tmp_path / "w", workflows=(), filters=("SII", "Ha", "OIII"), nights=0)
    try:
        W.template("sho_rgb")
        oiii = next(r for r in W.graph.rows if r.name == "OIII")
        reg = next(b for b in oiii.blocks if W.graph.blocks[b].kind == "registration")
        before = model_json(W.graph)
        assert W.m.apply([("remove_block", {"block": reg})]).ok
        W.settle()
        W.m.undo(graph=True)
        W.settle()
        assert model_json(W.graph) == before
    finally:
        W.close()
