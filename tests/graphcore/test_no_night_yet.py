"""A project whose workflow was made from options before any session (the CLI's add-workflow, an empty project of the
API): its graph is a stack of every night with none drawn yet, which the core keeps and edits; the definition compiles
back unchanged, and a night added later is drawn in it."""
from __future__ import annotations

from astro_gpu import api
from astro_gpu.core.project2 import Project2
from astro_gpu.graphcore.compile import compile_into
from astro_gpu.graphcore.convert import convert
from astro_gpu.graphcore.model import check
from astro_gpu.graphcore.operations import apply_many


def empty_with_a_workflow(tmp_path):
    p = api.create_project(tmp_path / "P", "P")
    api.add_workflow(p, "Full", "Full")
    return p


def test_a_workflow_made_before_any_night_converts_to_a_graph_the_core_keeps(tmp_path):
    p = empty_with_a_workflow(tmp_path)
    g = convert(p)
    check(g)                                                   # no "row has no head" any more
    q = Project2.from_dict(p.to_dict(), None)
    compile_into(g, q)
    assert q.to_dict()["workflows"] == p.to_dict()["workflows"]


def test_it_is_edited_and_takes_the_first_night(tmp_path):
    p = empty_with_a_workflow(tmp_path)
    g = convert(p)
    r = g.rows[0]
    g2, res = apply_many(g, [("rename_row", {"row": r.id, "name": "Renamed"})])
    assert res.ok, res.reason
    assert g2.rows[0].name == "Renamed"
    g3, res = apply_many(g2, [("new_night", {"name": "Night 1", "filter": ""})])
    assert res.ok, res.reason
    q = Project2.from_dict(p.to_dict(), None)
    compile_into(g3, q)
    assert [w.name for w in q.workflows] == ["Renamed"] and q.workflows[0].sessions is None


def test_a_workflow_without_calibration_is_drawn_without_one(tmp_path):
    """A night's Calibration may be absent (deleted): such a workflow converts to nights with no Calibration block,
    which compile back to no calibration node (drawn = stored = compiled)."""
    from astro_gpu.core import workflow as wfl
    p = api.create_project(tmp_path / "P", "P")
    raw = tmp_path / "raw"
    raw.mkdir()
    api.add_session(p, lights=[raw])
    w = api.add_workflow(p, "Plain", "Full")
    p.set_blocks(w.id, [wfl.node("registration"), wfl.node("stacking")])
    p.validate()
    assert not any(n["block"] == "calibration" for n in p.workflow(w.id).blocks)
    g = convert(p)
    check(g)
    assert [c for _s, c in g.rows[0].nights] == [None]
    assert not any(b.kind == "calibration" for b in g.blocks.values())
    q = Project2.from_dict(p.to_dict(), None)
    compile_into(g, q)
    assert q.to_dict()["workflows"] == p.to_dict()["workflows"]
