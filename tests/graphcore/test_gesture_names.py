"""A gesture creates what it names, nothing more, unless the engine needs more for the graph to be valid at all (then
the toast says which and why). Calibration, Registration, Selection and Stacking are removed like any block and put
back from the palette; each absence compiles to what the engine runs (lights taken as they are, frames taken as
aligned, no selection rule of a block); the engine's refusals are said at the moment they matter."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from fixtures import Doc  # noqa: E402
from test_points import ok, row, kinds, of_kind, refused_inert, where, drop, session_doc  # noqa: E402
from test_engine_rules import compiled, engine_ok  # noqa: E402
from astro_gpu.core import workflow as wfl  # noqa: E402
from astro_gpu.graphcore import templates as tpl  # noqa: E402


def bare(doc, r):
    """[(Session kind, has a Calibration)] of a stack's nights."""
    return [(doc.project.blocks[s].kind, c is not None) for s, c in r.nights]


def test_a_session_placed_on_a_blank_page_comes_alone():
    for how in ("double-click", "row place", "files", "menu"):
        doc = Doc()
        if how == "double-click":
            ok(doc, "add_block", kind="session")
        elif how == "row place":
            drop(doc, where(doc, "row", kind="session")[0])
        elif how == "files":
            ok(doc, "add_row", kind="session", files={"lights": ["L1.fit", "L2.fit"]})
        else:
            ok(doc, "add_row", kind="session")
        r = doc.project.rows[0]
        assert bare(doc, r) == [("session", False)] and r.blocks == [] and doc.project.wires == [], how
        engine_ok(doc)


def test_the_owners_star_removal_without_calibration_nor_registration():
    doc = Doc()
    ok(doc, "add_row", kind="session", files={"lights": ["L1.fit", "L2.fit"]})
    rid = doc.project.rows[0].id
    # what the engine refuses, said where it matters: a star removal needs stacked frames, then a Denoise
    assert refused_inert(doc, "add_block", kind="stars", row=rid) == "Stars/background needs Stacking first"
    ok(doc, "add_block", kind="stacking", row=rid)
    assert refused_inert(doc, "add_block", kind="stars", row=rid) == "needs Denoise first"
    ok(doc, "add_block", kind="denoise", row=rid)
    ok(doc, "add_block", kind="stars", row=rid)
    assert kinds(doc, row(doc, "Session 1")) == ["stacking", "denoise", "stars"]
    engine_ok(doc)
    p, _c = compiled(doc)
    w = p.workflows[0]
    assert [n["block"] for n in w.blocks] == ["stacking", "denoise", "output", "stars"]     # Output is automatic
    assert p.chain(w) == ["inventory", "calibrate", "register", "stack", "denoise", "render_simple", "star_removal"]
    assert wfl.node_implied(w.blocks, "calibrate") == {"apply_masters": False}      # the lights taken as they are
    assert wfl.node_implied(w.blocks, "register") == {"align": False}               # the frames taken as aligned
    r = row(doc, "Session 1")
    assert refused_inert(doc, "remove_block", block=of_kind(doc, r, "denoise")) == "needed by Stars/background"
    assert refused_inert(doc, "remove_block", block=of_kind(doc, r, "stacking")) == "Denoise needs Stacking first"


def test_calibration_registration_selection_stacking_removed_and_put_back():
    doc = session_doc()
    r = doc.project.rows[0]
    ok(doc, "add_block", kind="gradient", row=r.id)
    for k in ("registration", "selection"):
        ok(doc, "remove_block", block=of_kind(doc, row(doc, "Session 1"), k))
        engine_ok(doc)
    cal = row(doc, "Session 1").nights[0][1]
    r = ok(doc, "remove_block", block=cal)
    assert "taken as they are" in r.toast_text() and bare(doc, row(doc, "Session 1")) == [("session", False)]
    assert cal not in doc.project.blocks
    engine_ok(doc)
    p, _c = compiled(doc)
    assert [n["block"] for n in p.workflows[0].blocks] == ["stacking", "gradient", "output"]
    assert doc.palette_state("calibration")[0]
    s = row(doc, "Session 1").nights[0][0]
    pl = where(doc, "calib", kind="calibration", night=s)
    assert pl and pl[0].lit
    drop(doc, pl[0])
    assert bare(doc, row(doc, "Session 1")) == [("session", True)]
    assert doc.palette_state("calibration") == (False, "each session already has its Calibration")
    for k in ("registration", "selection"):                                # back from the palette, in their order
        ok(doc, "add_block", kind=k, row=row(doc, "Session 1").id)
    assert kinds(doc, row(doc, "Session 1")) == ["registration", "selection", "stacking", "gradient"]
    p, _c = compiled(doc)
    assert [n["block"] for n in p.workflows[0].blocks][:4] == ["calibration", "registration", "selection", "stacking"]
    engine_ok(doc)
    ok(doc, "remove_block", block=of_kind(doc, row(doc, "Session 1"), "gradient"))
    ok(doc, "remove_block", block=of_kind(doc, row(doc, "Session 1"), "stacking"))     # frames only: the engine
    engine_ok(doc)                                                                        # runs it (nothing stacked)


def test_calibration_together_is_one_per_stack_each_night_its_own_apart():
    doc = session_doc()
    rid = doc.project.rows[0].id
    r = ok(doc, "add_night", row=rid)                           # calibrated together: the stack's one Calibration
    assert bare(doc, row(doc, "Session 1")) == [("session", True)] * 2 and "calibrated together" in r.toast_text()
    r = ok(doc, "remove_block", block=row(doc, "Session 1").nights[1][1])
    assert bare(doc, row(doc, "Session 1")) == [("session", False)] * 2 and "2 sessions" in r.toast_text()
    engine_ok(doc)
    ok(doc, "add_night", row=rid)                               # no Calibration in the stack: the Session alone
    assert bare(doc, row(doc, "Session 1")) == [("session", False)] * 3
    ok(doc, "add_block", kind="calibration", row=rid)           # put back: on every night, as one
    assert bare(doc, row(doc, "Session 1")) == [("session", True)] * 3
    ok(doc, "set_calibrate_together", on=False)                 # each night its own: one goes alone
    ok(doc, "remove_block", block=row(doc, "Session 1").nights[1][1])
    assert [c for _k, c in bare(doc, row(doc, "Session 1"))] == [True, False, True]
    engine_ok(doc)
    p, _c = compiled(doc)
    sid = doc.project.blocks[row(doc, "Session 1").nights[1][0]].sid
    w = p.workflows[0]
    assert w.blocks[0]["block"] == "calibration" and w.params[f"calibrate@{sid}"] == {"apply_masters": False}
    assert refused_inert(doc, "set_calibrate_together", on=True) == "calibrated together: one setting for all nights"
    ok(doc, "add_night", row=rid)                               # each night its own: the Session alone
    assert bare(doc, row(doc, "Session 1"))[-1] == ("session", False)


def test_other_creating_gestures_make_what_they_name():
    doc = session_doc()
    rid = doc.project.rows[0].id
    ok(doc, "add_night", row=rid, files={"lights": ["a.fit", "b.fit"]})
    s2 = row(doc, "Session 1").nights[1][0]
    ok(doc, "move_night", night=s2, row=None)                  # its own row: a stack being built, nothing placed
    new = doc.project.rows[-1]
    assert new.blocks == [] and len(new.nights) == 1 and not any(w.src[0] in new.blocks for w in doc.project.wires)
    engine_ok(doc)
    ok(doc, "duplicate_row", row=rid)                           # a copy is what it copies
    copy = doc.project.rows[1]
    assert kinds(doc, copy) == kinds(doc, row(doc, "Session 1")) and bare(doc, copy) == [("session", True)]
    s = new.nights[0][0]
    ok(doc, "split_by", session=s, groups=[{"into": "row", "files": {"lights": ["b.fit"]}, "name": "Part"}])
    assert row(doc, "Part").blocks == []
    engine_ok(doc)


def test_a_session_on_a_port_gets_what_the_port_needs():
    doc = Doc.from_template("filters_minimal")                  # H-O to Post: an image is enough
    j = next(b for b in doc.project.blocks.values() if b.kind == "combine").id
    ok(doc, "set_combination_type", block=j, type="sho")
    r = drop(doc, where(doc, "port", kind="session", match={"block": j, "port": "S II"})[0])
    new = doc.project.rows[-1]
    assert kinds(doc, new) == ["stacking"] and bare(doc, new) == [("session", False)]
    assert "the port reads an image" in r.toast_text()
    engine_ok(doc)


def test_templates_keep_their_full_chains():
    doc = Doc()
    ok(doc, "new_night", name="N1")
    t = dict(key="colour_min", name="Colour minimal", opts=dict(debruitage=False, etoiles="garder", fond="simple"))
    plan = tpl.build(doc.project, t, t["name"], dict(t["opts"]))
    assert doc.apply_many(plan.steps).ok
    r = doc.project.rows[-1]
    assert bare(doc, r) == [("session", True)] and kinds(doc, r)[:3] == ["registration", "selection", "stacking"]


def test_order_is_the_only_drawing_rule():
    doc = session_doc()
    r = doc.project.rows[0]
    g = ok(doc, "add_block", kind="gradient", row=r.id)
    ok(doc, "switch_off", block=of_kind(doc, row(doc, "Session 1"), "gradient"))
    ok(doc, "remove_block", block=of_kind(doc, row(doc, "Session 1"), "stacking"))     # a switched-off Gradient: the
    engine_ok(doc)                                                                     # engine runs it (absent)
    assert g.ok
    stk = where(doc, "insert", kind="stacking", match={"row": r.id, "after": of_kind(doc, row(doc, "Session 1"),
                                                                                      "gradient")})
    assert stk and not stk[0].lit and "Stacking" in stk[0].reason_text()       # after an image block: the order
