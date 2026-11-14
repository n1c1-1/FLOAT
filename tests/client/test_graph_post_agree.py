"""The graph and Post agree after every gesture that takes away a source of Post's layers (the engine owner's rule): a
block or a wire removed in the graph takes what it gave Post out of every version, in the same undo step, and Ctrl+Z
brings both back; the Post block's count, every version's layers and the Post page's list say the same images. On the
owner's trial: a hand-built chain ending Stars/background -> Combine layers -> Post, the Stars block deleted with its
Combine layers while Post was open. Offscreen, scratch project, nothing computed (the workflows marked rendered)."""
from __future__ import annotations

import pytest

from astro_gpu.client.post3 import PostPage3
from astro_gpu.client.services.render import RenderWorker
from astro_gpu.graphcore.post import layer_rows, place_of, stars_side, wired_images
from tests.client.graphview.legacy import World
from tests.client.pagehelp import qapp  # noqa: F401

ROW = ["calibration", "registration", "selection", "stacking", "astrometry", "gradient", "denoise", "stars"]


@pytest.fixture()
def world_page(qapp, tmp_path):  # noqa: F811
    """Colour full made from the gallery on a blank project, Post's page open."""
    W = World(tmp_path / "w", workflows=())
    W.template("colour_full")
    for q in W.m.paths:
        q.rendered = True
    page3 = PostPage3(W.m, worker=RenderWorker(None, resolver=lambda sid: None, device="cpu"))
    page3.view.sync_paths(initial=True)
    W.m._undo.clear()
    yield W, page3
    page3.shutdown()
    page3.deleteLater()
    W.close()


@pytest.fixture()
def bench(qapp, tmp_path):
    W = World(tmp_path / "w", workflows=(), chains={"Session 1": ROW})
    m = W.m
    for q in m.paths:
        q.rendered = True
    q = m.paths[0]
    m.add_post_layer(q.pid, None)                     # Post opened and worked in: its layers written, two versions
    m.add_post_layer(q.pid, "starless")
    m.default_post_layer(q.pid)
    m.new_post_version()
    page3 = PostPage3(m, worker=RenderWorker(None, resolver=lambda sid: None, device="cpu"))
    page3.view.sync_paths(initial=True)
    m._undo.clear()
    yield W, page3
    page3.shutdown()
    page3.deleteLater()
    W.close()


def images(layers) -> set:
    return {(x.get("pid"), x.get("part")) for x in layers or [] if x.get("kind") == "workflow"}


def check(W, page3):
    """Every version holds only images the graph still gives; the Post page shows the active version; the Post block
    counts what Post shows."""
    m = W.m
    W.settle(2)
    for q in m.paths:
        q.rendered = True                             # the run computed every workflow
    m.pathsChanged.emit()
    W.settle(1)
    g = m._g

    def gives(pid, part):
        return stars_side(g, pid) is not None if part == "combined" else place_of(g, pid, part) is not None
    for v in m.p.post["versions"]:
        gone = {(pid, part) for pid, part in images(v.get("layers")) if not gives(pid, part)}
        assert not gone, (v["id"], gone)
    layers = page3.view.layers
    shown = images(layers)
    assert shown == images(m.post_version().get("layers")) | {(q.pid, m.post_default_part(q))
                                                              for q in m.post_defaults()}
    assert {pid for pid, _part in shown} == {pid for _name, pid in m.block_map.post_layers}
    assert len(m.block_map.post_layers) == len([x for x in layers if x.get("kind") == "workflow"])
    assert wired_images(g) == {x for x in shown if x[1] != "combined"}
    # a Combine layers block for each Combine layers layer Post shows, wired from its workflow's Stars block
    rows = layer_rows(g)
    for pid in {pid for pid, _part in shown}:
        n = sum(1 for x in layers if x.get("kind") == "workflow" and x.get("pid") == pid and x.get("part") == "combined")
        assert len(rows.get(stars_side(g, pid), [])) == n, (pid, n, rows)
    assert sum(len(r) for r in rows.values()) == len([b for b in g.blocks.values() if b.kind == "combine_layers"])
    return shown


def test_deleting_stars_with_its_combine_layers_takes_their_layers_out_of_every_version_and_undo_brings_them(bench):
    W, page3 = bench
    m = W.m
    pid = m.paths[0].pid
    before = check(W, page3)
    assert {(pid, "starless"), (pid, "combined"), (pid, None)} <= before
    stars = W.one("stars")
    cl = W.kinds("combine_layers")
    res = m.apply([("remove_block", {"block": b}) for b in cl] + [("remove_block", {"block": stars})])
    assert res.ok, res.reason
    after = check(W, page3)
    assert after == {(pid, None)}
    for v in m.p.post["versions"]:
        assert images(v["layers"]) == {(pid, None)}
    m.undo(graph=True)
    assert check(W, page3) == before
    for v in m.p.post["versions"]:
        assert {(pid, "starless"), (pid, "combined")} <= images(v["layers"])
    m.redo(graph=True)
    assert check(W, page3) == after


def test_switching_a_block_off_or_deleting_the_row_keeps_graph_and_post_agreeing(bench):
    """(Stars/background itself cannot be switched off: it turns an image into two.)"""
    W, page3 = bench
    m = W.m
    before = check(W, page3)
    assert m.apply([("switch_off", {"block": W.one("gradient")})]).ok
    assert check(W, page3) == before
    m.undo(graph=True)
    assert check(W, page3) == before
    row = W.graph.rows[0].id
    assert m.apply([("delete_row", {"row": row})]).ok
    assert check(W, page3) == set()
    assert all(not images(v["layers"]) for v in m.p.post["versions"])
    m.undo(graph=True)
    assert check(W, page3) == before


def test_a_variant_discarded_or_kept_and_a_wire_to_post_removed_keep_them_agreeing(bench):
    W, page3 = bench
    m = W.m
    before = check(W, page3)

    def trial():
        res = m.apply([("try_variant", {"block": W.kinds("gradient")[0], "change": {"off": True}})])
        assert res.ok, res.reason
        check(W, page3)
        return next(r for r in W.graph.rows if r.trial is not None)
    var = trial()
    assert m.apply([("discard_variant", {"row": var.id})]).ok
    assert check(W, page3) == before
    var = trial()
    assert m.apply([("keep_variant", {"row": var.id})]).ok
    check(W, page3)
    m.undo(graph=True)
    m.undo(graph=True)
    assert check(W, page3) == before
    post = next(b for b, x in W.graph.blocks.items() if x.kind == "post")
    for wire in [w for w in W.graph.wires if w.dst[0] == post]:
        assert m.apply([("unwire", {"wire": wire.id})]).ok
        check(W, page3)
        m.undo(graph=True)
        assert check(W, page3) == before


# ---------------------------------------------------------------------------------------------- Synthetic stars (#93)
def _into_stars(W):
    cl = W.one("combine_layers")
    return sorted(W.graph.blocks[w.src[0]].kind for w in W.graph.wires if w.dst == [cl, "Stars"])


def test_synthetic_stars_extend_the_one_combine_layers_in_place_and_one_undo_takes_them_back(world_page):
    """The old model and page (issue #93): Synthetic stars goes into the SAME workflow after Stars/background (never a
    second workflow nor a second Combine layers); Post's one Combine layers layer keeps its id and eye and gets the
    synthetic stars row in its Stars group; the graph's Combine layers takes the real stars and the synthetic ones on its
    one Stars side (two wires); one Undo takes block and row back; switched off, the synthetic stars' wire goes."""
    import astro_gpu.client.postengine as pe
    W, page3 = world_page
    m = W.m
    pid = m.paths[0].pid
    page3.view._save()                                             # Post's layer written: its id
    lid = next(x["id"] for x in m.post_version()["layers"] if x.get("part") == "combined")
    eye = m.post_layer(lid)["visible"]
    assert _into_stars(W) == ["stars"]
    kinds = [p.op[1]["at"]["type"] for p in __import__("astro_gpu.graphcore", fromlist=["places"]).places(
        W.graph, m.graph_view, kind="synth") if p.lit]
    assert "branch" not in kinds                                   # no branch of its own: never "<name> 2"
    W.b.act("add_block", kind="synth")
    W.settle()
    assert [q.pid for q in m.paths] == [pid] and len(W.kinds("combine_layers")) == 1
    row = next(r for r in W.graph.rows if W.one("synth") in r.blocks)
    assert W.one("stars") in row.blocks and row.blocks.index(W.one("synth")) == row.blocks.index(W.one("stars")) + 1
    assert _into_stars(W) == ["stars", "synth"]
    assert pid in m.synth_wanted()
    check(W, page3)
    syn = [x for x in page3.view.layers if pe.is_synth(x)]
    assert len(syn) == 1 and syn[0]["target"] == lid
    assert m.post_layer(lid) is not None and m.post_layer(lid)["visible"] == eye
    assert m.apply([("switch_off", {"block": W.one("synth")})]).ok
    W.settle()
    assert _into_stars(W) == ["stars"]
    m.undo(graph=True)
    W.settle()
    assert _into_stars(W) == ["stars", "synth"]
    m.undo(graph=True)                                             # one step: the block and its row
    W.settle()
    assert not W.kinds("synth") and _into_stars(W) == ["stars"]
    assert not any(pe.is_synth(x) for x in page3.view.layers) and m.post_layer(lid) is not None
    check(W, page3)


def test_a_project_holding_two_combine_layers_opens_as_it_is(world_page):
    """A project that already holds two Combine layers layers of one workflow (the defect's second one included) opens
    with two Combine layers blocks, never merged."""
    W, page3 = world_page
    m = W.m
    pid = m.paths[0].pid
    page3.view._save()
    m.add_post_layer(pid, "combined")
    W.settle()
    assert len(W.kinds("combine_layers")) == 2
    m.flush(force=True) if not m.readonly else None
    from astro_gpu.client.model import ProjectModel
    from astro_gpu.core import project2
    w2 = type(W.w)()
    m2 = ProjectModel(project2.load(W.m.root), w2, readonly=True)
    assert w2.wait(120) and m2.ready
    assert len([b for b in m2.stored.blocks.values() if b.kind == "combine_layers"]) == 2
    assert len([x for x in m2.post_version()["layers"] if x.get("part") == "combined"]) == 2
    m2.close(save=False)
    w2.stop()
    check(W, page3)
