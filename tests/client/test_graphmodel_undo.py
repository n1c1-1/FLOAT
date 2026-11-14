"""GraphEditing's Undo (INTEGRATION §9) and its document interface (the new graph page drives the model).

The definition is the same object whatever is done (the engine's per-project context is kept); the graph's and Post's
steps keep their pages; a fold, a selection or a frame excluded by hand is never a step; a recovery copy reopens to
the same project."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "graphcore"))
import test_h1_old_way as h1  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from astro_gpu.client.graphmodel import GraphEditing  # noqa: E402
from astro_gpu.client.model import ProjectModel  # noqa: E402
from astro_gpu.client.services.worker import ApiWorker  # noqa: E402
from astro_gpu.core import project2  # noqa: E402
from astro_gpu.graphcore.model import Index  # noqa: E402


class M(GraphEditing, ProjectModel):
    pass


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture()
def model(qapp, tmp_path):
    """M on a "Colour minimal" project of one session (built by the gallery's builder, saved, reopened)."""
    root = tmp_path / "P"
    w = h1.old_way(root, next(t for t in h1.GALLERY if t["key"] == "colour_min"), h1.FAKE[:1])
    project2.save(w.p)
    w.close()
    worker = ApiWorker()
    m = M(project2.load(root), worker, readonly=True)
    assert worker.wait(120) and m.ready
    yield m
    m.close(save=False)
    worker.stop()


def row(m, name="Colour minimal"):
    return next(r for r in m.stored.rows if r.name == name)


def block(m, kind, name="Colour minimal"):
    return next(b for b in Index(m.stored).effective(row(m, name).id) if m.stored.blocks[b].kind == kind)


def definition(m) -> str:
    d = m.p.to_dict()
    for k in ("created", "current_block"):            # the selection shown is view state (kept for the next opening)
        d.pop(k, None)
    return json.dumps(d, sort_keys=True, default=str)


def test_the_definition_is_the_same_object_over_edits_and_undo(model):
    p = model.p
    first = definition(model)
    res = model.apply([("add_block", {"kind": "deconv", "row": row(model).id})])
    assert res.ok and model.p is p and definition(model) != first
    assert model.can_undo(graph=True)
    assert model.undo(graph=True) and model.p is p and definition(model) == first
    assert model.redo(graph=True) and model.p is p and definition(model) != first
    assert model._undo[-1][3] is None and json.loads(model._undo[-1][4]) == model.p.post


def test_a_refused_gesture_changes_nothing_and_is_no_step(model):
    before, steps = definition(model), len(model._undo)
    res = model.apply([("remove_block", {"block": "no such block"})])
    assert not res.ok and res.reason and definition(model) == before and len(model._undo) == steps


def test_graph_post_graph_then_the_graphs_undo_twice_and_posts_undo(model):
    wid = view_pid = next(q.pid for q in model.paths if q.name == "Colour minimal")
    model.apply([("add_block", {"kind": "deconv", "row": row(model).id})])               # graph
    with model.from_post():
        model.new_post_version()                                                          # Post's
    model.apply([("switch_off", {"block": block(model, "deconv")})])                     # graph
    assert len(model._undo) == 3 and model._undo[1].from_post
    assert model.undo(graph=True)                                                        # the switch back on
    assert not model.stored.blocks[block(model, "deconv")].off
    assert not model.can_undo(graph=True)                                                # Post's step is Post's
    assert model.undo()                                                                  # Post's version goes
    assert len(model.p.post["versions"]) == 1
    assert model.undo(graph=True)                                                        # the Deconvolution goes
    assert not any(model.stored.blocks[b].kind == "deconv" for b in model.stored.blocks)
    assert view_pid == wid


def test_a_fold_or_a_selection_is_no_step_and_no_edit(model):
    steps, rev = len(model._undo), model.revision
    res = model.apply([("fold_all", {})])
    assert res.ok and model.graph_view["folded"] is True
    model.document.apply([("select", {"id": block(model, "stacking")})])
    assert model.graph_view["selected"] == block(model, "stacking")
    assert len(model._undo) == steps and model.revision == rev and not model.dirty


def test_a_frame_excluded_by_hand_is_no_step_but_goes_into_the_graph(model):
    steps = len(model._undo)
    model.set_excluded([(1, "L0.fit")], True)
    assert len(model._undo) == steps
    assert [1, "L0.fit"] in [list(x) for x in model.stored.excluded]
    model.apply([("add_block", {"kind": "deconv", "row": row(model).id})])
    assert (1, "L0.fit") in model.frames_excluded                                        # a compile kept it
    model.undo(graph=True)
    assert (1, "L0.fit") in model.frames_excluded                                        # and Undo did not take it
    assert model.graph_rebuilds == 0


def test_undo_selects_on_block_ids(model):
    model.apply([("add_block", {"kind": "deconv", "row": row(model).id})])
    d = block(model, "deconv")
    pred = Index(model.stored).pred(d)
    model.graph_view["selected"] = d
    model.undo(graph=True)
    assert model.graph_view["selected"] == pred                                          # the one before it


def test_a_recovery_copy_whose_last_step_is_posts_reopens_to_the_same_project(model, tmp_path):
    model.apply([("add_block", {"kind": "deconv", "row": row(model).id})])
    with model.from_post():
        model.new_post_version()
    rec = model.p.to_dict()                                                              # what the recovery writes
    w = ApiWorker()
    back = M(project2.Project2.from_dict(json.loads(json.dumps(rec)), model.p.root), w, readonly=True)
    try:
        assert w.wait(120)
        assert definition(back) == definition(model)
        assert [r.name for r in back.stored.rows] == [r.name for r in model.stored.rows]
        assert sorted(b.kind for b in back.stored.blocks.values()) == sorted(b.kind for b in model.stored.blocks.values())
    finally:
        back.close(save=False)
        w.stop()


def test_the_graph_page_drives_the_model(model):
    """The new page on the model's document: a gesture through its bridge edits the definition, Undo takes it back."""
    from astro_gpu.client.graphview.page import GraphPage
    page = GraphPage(model.document)
    try:
        before = definition(model)
        res = page.bridge.act("add_block", kind="deconv", row=row(model).id)
        assert res.ok and definition(model) != before
        assert page.bridge.doc.state(block(model, "deconv")) in ("pending", "done", "stale")
        assert isinstance(page.bridge.doc.cost(), float)
        assert page.bridge.doc.can_undo
        res = page.bridge.undo()
        assert res.ok and definition(model) == before
        assert page.bridge.doc.can_redo
    finally:
        page.deleteLater()


def test_a_wire_to_post_removed_in_the_graph_leaves_every_version(model):
    """Engine review 3: a layer deleted in one version only leaves the graph's wire; the wire removed in the graph
    takes the image out of every version."""
    wid = next(q.pid for q in model.paths if q.name == "Colour minimal")
    model.add_post_layer(wid, None, None)
    model.new_post_version()

    class PostPage:
        is_post_page = True
    model.set_post_layers([x for x in model.post_layers() if x.get("pid") != wid], PostPage(), scope="active")
    post = next(b for b in model.stored.blocks if model.stored.blocks[b].kind == "post")
    w = next(x for x in model.stored.wires if x.dst[0] == post)                          # the wire stays
    res = model.apply([("unwire", {"wire": w.id})])
    assert res.ok
    assert all(not any(x.get("pid") == wid and x.get("part") is None for x in v.get("layers") or [])
               for v in model.p.post["versions"])
    assert (wid, None) in model.post_removed()
