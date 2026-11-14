"""A stack built on the graph page has the camera white balance in its Stacking as a new workflow of today's model had
it (issue #6, ``model.add_path``): on in a new project, so that a one-shot colour stack and every image before Color
(Stacking, Astrometry, Gradient...) no longer look green; a later stack follows the project's first one, so that a
project computed without it keeps its stacks shared. A Session alone (no Stacking placed yet) decides nothing."""
from __future__ import annotations

from astro_gpu.core.project2 import Project2
from astro_gpu.graphcore import model as gm
from astro_gpu.graphcore.compile import compile_into
from astro_gpu.graphcore.operations import apply
from astro_gpu.graphcore.templates import camera_white_balance


def _ok(g, op, /, **kw):
    g2, res = apply(g, op, **kw)
    assert res.ok, res.reason
    return g2


def _stack_by_hand(g):
    """A Session row, then Registration, Selection and Stacking placed one by one from the palette: -> (graph, row)."""
    g = _ok(g, "add_row", kind="session")
    rid = g.rows[-1].id
    for k in ("registration", "selection", "stacking"):
        g = _ok(g, "add_block", kind=k, row=rid)
    return g, rid


def _wb(g) -> dict:
    """{row id: the camera white balance its compiled workflow runs with}."""
    p = Project2(name="wb")
    c = compile_into(g, p)
    return {rid: bool((p.workflow(wid).params.get("stack") or {}).get("white_balance", False))
            for rid, wid in c.wids.items() if any(g.blocks[b].kind == "stacking" for b in
                                                  next(r for r in g.rows if r.id == rid).blocks)}


def _stacking(g, rid):
    return next(b for b in next(r for r in g.rows if r.id == rid).blocks if g.blocks[b].kind == "stacking")


def test_a_stack_built_block_by_block_has_the_camera_white_balance_in_a_new_project():
    g, rid = _stack_by_hand(gm.blank_project("P"))
    assert _wb(g) == {rid: True}
    g, rid2 = _stack_by_hand(g)                                   # a second stack: as the first one
    assert _wb(g)[rid2] is True


def test_a_later_stack_follows_the_projects_first_stack():
    g, rid = _stack_by_hand(gm.blank_project("P"))
    g = _ok(g, "set_param", block=_stacking(g, rid), stage="stack", name="white_balance", value=False)
    g, rid2 = _stack_by_hand(g)                                   # computed before without it: the stacks shared
    assert _wb(g) == {rid: False, rid2: False}


def test_a_session_alone_decides_nothing():
    g = _ok(gm.blank_project("P"), "add_row", kind="session")    # frames dropped first, no Stacking yet
    assert camera_white_balance(g) is True                        # a template placed now takes it too
