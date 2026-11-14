"""What the project model edits beside the drawing, now operations of the core (step 4c): a session by its id (drawn
or not: its files, name, filter, frames left out, its lights excluded by hand, the session taken out), an Image block's
file, a tap renamed, a workflow's own pauses, a workflow made from options, a workflow's setting stored on what holds
it (the leftovers that hid it taken out), a row's chain replaced as a whole, a new branch; the variants of a stack; a
block or a Session dropped on its own spot is no place. Each change is one undo step; each refusal changes nothing."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from fixtures import Doc  # noqa: E402
from test_points import ok, row, of_kind, refused_inert  # noqa: E402
from test_engine_rules import compiled, engine_ok  # noqa: E402
from astro_gpu import graphcore  # noqa: E402
from astro_gpu.graphcore import places  # noqa: E402
from astro_gpu.graphcore.model import Index  # noqa: E402


def wf(p, name):
    return next(w for w in p.workflows if w.name == name)


def undo_redo(doc):
    after = doc._json
    doc.undo()
    before = doc._json
    doc.redo()
    assert doc._json == after
    return before


# ---------------------------------------------------------------------------------------------------- sessions by id
def test_a_night_no_stack_draws_takes_its_files_name_filter_and_frames_left_out():
    doc = Doc.from_template("colour_minimal")
    ok(doc, "new_night", name="Later")
    sid = doc.project.nights[-1]["sid"]
    ok(doc, "set_session_files", sid=sid, kind="lights", paths=["D:/n/a.fit", "D:/n/b.fit", "D:/n/a.fit"])
    ok(doc, "rename_session", sid=sid, name="Night 2")
    ok(doc, "set_session_filter", sid=sid, filter="Ha")
    ok(doc, "set_left_out", sid=sid, left_out={"darks": ["D:/n/d1.fit"]})
    p, _ = compiled(doc)
    s = p.session(sid)
    assert s.lights == ["D:/n/a.fit", "D:/n/b.fit"] and s.name == "Night 2" and s.filter == "Ha"
    assert s.left_out == {"darks": ["D:/n/d1.fit"]}
    undo_redo(doc)
    assert refused_inert(doc, "rename_session", sid=99, name="x") == "session not found"
    assert refused_inert(doc, "rename_session", sid=sid, name="  ") == "a name cannot be empty"
    assert refused_inert(doc, "set_session_files", sid=sid, kind="nope", paths=[]) == "unknown kind of files"


def test_a_drawn_session_edited_by_its_id_is_the_same_in_every_stack_drawing_it():
    doc = Doc.from_template("filters_minimal")
    ha = row(doc, "Ha")
    s = ha.nights[0][0]
    sid = doc.project.blocks[s].sid
    ok(doc, "share_night", night=s, row=row(doc, "OIII").id)
    ok(doc, "set_session_files", sid=sid, kind="lights", paths=["D:/x/1.fit"])
    ok(doc, "rename_session", sid=sid, name="Dual")
    blocks = [b for b in doc.project.blocks.values() if b.kind == "session" and b.sid == sid]
    assert len(blocks) == 2 and all(b.files["lights"] == ["D:/x/1.fit"] and b.name == "Dual" for b in blocks)
    p, _ = compiled(doc)
    assert p.session(sid).lights == ["D:/x/1.fit"] and p.session(sid).name == "Dual"
    engine_ok(doc)


def test_a_session_taken_out_of_the_project_and_lights_excluded_by_hand():
    doc = Doc.from_template("colour_minimal")
    r = doc.project.rows[0]
    ok(doc, "add_night", row=r.id, files={"lights": ["D:/b/1.fit"]})
    sid = doc.project.blocks[doc.project.rows[0].nights[1][0]].sid
    ok(doc, "exclude_frames", sid=sid, names=["D:/b/1.fit", "D:/b/2.fit"])
    p, _ = compiled(doc)
    assert sorted(p.excluded) == [(sid, "1.fit"), (sid, "2.fit")]
    ok(doc, "exclude_frames", sid=sid, names=["2.fit"], excluded=False)
    assert [list(x) for x in doc.project.excluded] == [[sid, "1.fit"]]
    ok(doc, "remove_session", sid=sid)
    p, _ = compiled(doc)
    assert sid not in [s.id for s in p.sessions] and not p.excluded and len(doc.project.rows[0].nights) == 1
    engine_ok(doc)
    undo_redo(doc)
    assert refused_inert(doc, "exclude_frames", sid=sid, names=["a.fit"]) == "session not found"


# ---------------------------------------------------------------------------------------------------- images, taps
def test_an_image_block_takes_its_file():
    doc = Doc.from_template("several_images")
    img = doc.project.rows[0].blocks[0]
    ok(doc, "set_image_file", block=img, path="D:/img/m31.tif")
    assert doc.project.blocks[img].source == {"path": "D:/img/m31.tif", "stretched": False}
    ok(doc, "set_image_file", block=img, stretched=True)
    p, _ = compiled(doc)
    assert p.workflows[0].source == {"path": "D:/img/m31.tif", "stretched": True}
    undo_redo(doc)
    assert refused_inert(doc, "set_image_file", block=img) == "nothing to change"
    reason = refused_inert(doc, "set_image_file", block=doc.project.rows[0].id, path="x")
    assert reason == "block not found"


def test_a_tap_renamed_and_its_name_kept_unique():
    doc = Doc.from_template("colour_minimal")
    r = doc.project.rows[0]
    ok(doc, "send_to_post", block=of_kind(doc, r, "astrometry"), name="Plate")
    t = doc.project.taps[0]
    ok(doc, "rename_tap", tap=t.id, name="Solved")
    p, _ = compiled(doc)
    assert "Solved" in [w.name for w in p.workflows]
    assert refused_inert(doc, "rename_tap", tap=t.id, name=r.name) == "“Session 1” already exists"
    assert refused_inert(doc, "rename_tap", tap="t999", name="x") == "not found"


# ---------------------------------------------------------------------------------------------------- workflows
def test_a_workflow_own_pause_list_and_back_to_its_blocks():
    doc = Doc.from_template("colour_minimal")
    r = doc.project.rows[0]
    ok(doc, "set_flow_pauses", flow=r.id, pauses=["calibration", "output"])
    p, _ = compiled(doc)
    assert p.workflows[0].breakpoints == ["calibration", "output"]
    ok(doc, "set_pause", block=of_kind(doc, r, "gradient"))
    ok(doc, "set_flow_pauses", flow=r.id, pauses=None)
    p, _ = compiled(doc)
    assert p.workflows[0].breakpoints == ["gradient"]
    assert refused_inert(doc, "set_flow_pauses", flow=r.id, pauses="gradient") == "a pause list names blocks"


def test_a_workflow_made_from_options_until_its_chain_changes():
    doc = Doc.from_template("colour_minimal")
    r = doc.project.rows[0]
    opts = {"denoise": None, "stars": None, "gradient": "polynomial", "deconvolution": None}
    ok(doc, "set_flow_options", flow=r.id, options=opts, end="stacking")
    p, _ = compiled(doc)
    assert p.workflows[0].blocks is None and p.workflows[0].end == "stacking"
    assert p.workflows[0].options["gradient"] == "polynomial"
    ok(doc, "set_flow_options", flow=r.id)
    p, _ = compiled(doc)
    assert p.workflows[0].blocks is not None and p.workflows[0].end is None
    assert refused_inert(doc, "set_flow_options", flow=r.id, options={"nope": 1}) == "unknown options"


def test_a_workflow_setting_goes_to_its_holder_and_a_leftover_no_longer_hides_it():
    doc = Doc.from_template("colour_minimal")
    r = doc.project.rows[0]
    reg = of_kind(doc, r, "registration")
    ok(doc, "store_params", flow=r.id, params={"register": {"min_pairs": 7}, "nowhere": {"x": 1}})
    assert doc.project.blocks[reg].params == {"register": {"min_pairs": 7}}
    assert doc.project.rows[0].extra == {"nowhere": {"x": 1}}
    q = doc.project.rows[0]
    q.extra = {"register": {"min_pairs": 3}}            # a leftover of a converted project
    doc.project.__dict__.pop("_json", None)
    ok(doc, "set_flow_param", flow=r.id, stage="register", name="min_pairs", value=20)
    p, _ = compiled(doc)
    assert p.workflows[0].params["register"]["min_pairs"] == 20 and not doc.project.rows[0].extra
    ok(doc, "set_flow_param", flow=r.id, stage="render_simple", name="linked", value=False)
    assert doc.project.rows[0].output == {"linked": False}
    ok(doc, "set_param", block=reg, stage="register", name="min_pairs", value=10)
    assert compiled(doc)[0].workflows[0].params["register"]["min_pairs"] == 10
    reason = refused_inert(doc, "set_flow_param", flow=r.id, stage="register", name="nope", value=1)
    assert reason == "register has no setting nope"


def test_a_row_chain_replaced_keeps_its_settings_where_the_new_blocks_hold_them():
    doc = Doc.from_template("colour_minimal")
    r = doc.project.rows[0]
    chain = [{"kind": k} for k in ("registration", "selection", "stacking", "astrometry", "gradient", "denoise",
                                    "color")]
    ok(doc, "replace_chain", row=r.id, blocks=chain, calibration={"off": False},
       params={"register": {"min_pairs": 9}, "denoise": {"fast": True}})
    r = doc.project.rows[0]
    assert [doc.project.blocks[b].kind for b in r.blocks] == [c["kind"] for c in chain]
    p, _ = compiled(doc)
    assert p.workflows[0].params["register"] == {"min_pairs": 9} and p.workflows[0].params["denoise"] == {"fast": True}
    engine_ok(doc)
    opts = {"denoise": None, "stars": None, "gradient": "polynomial", "deconvolution": None}
    ok(doc, "replace_chain", row=r.id, blocks=chain[:5], options=opts)
    assert doc.project.rows[0].legacy["options"]["gradient"] == "polynomial"     # kept: its options are given
    undo_redo(doc)
    assert refused_inert(doc, "replace_chain", row=r.id, blocks=[{"kind": "post"}]) == "unknown block"


def test_a_new_branch_from_a_block_and_beside_a_night_calibration():
    doc = Doc.from_template("colour_minimal")
    r = doc.project.rows[0]
    astro = of_kind(doc, r, "astrometry")
    ok(doc, "set_param", block=of_kind(doc, r, "registration"), stage="register", name="min_pairs", value=20)
    ok(doc, "add_branch", block=astro, kind="gradient", method="rbf", name="Session 1 · RBF")
    v = row(doc, "Session 1 · RBF")
    assert v.fork == [r.id, astro] and v.trial is None
    p, _ = compiled(doc)
    w = wf(p, "Session 1 · RBF")
    assert [n["block"] for n in w.blocks][-2:] == ["gradient", "output"]
    assert next(n for n in w.blocks if n["block"] == "gradient")["method"] == "rbf"
    assert w.params["register"] == {"min_pairs": 20} and w.variant_of is None
    engine_ok(doc)
    ok(doc, "add_branch", block=of_kind(doc, r, "registration"), kind="stacking", name="Raw")
    st = doc.project.blocks[row(doc, "Raw").blocks[0]]
    assert st.fork == wf(compiled(doc)[0], "Raw").id
    cal = r.nights[0][1]
    ok(doc, "add_branch", block=cal, kind="calibration", name="Own calibration")
    oc = row(doc, "Own calibration")
    assert oc.nights and not oc.blocks
    assert doc.project.blocks[oc.nights[0][1]].own == {"fork": wf(compiled(doc)[0], "Own calibration").id}
    assert refused_inert(doc, "add_branch", block=astro, kind="session") == "Session is not inserted in a row"
    assert refused_inert(doc, "add_branch", block=cal, kind="gradient") == "Calibration: no variant from here"


# ---------------------------------------------------------------------------------------------------- queries, drags
def test_the_variants_of_a_stack():
    doc = Doc.from_template("colour_minimal")
    r = doc.project.rows[0]
    ok(doc, "add_variant", block=of_kind(doc, r, "gradient"))
    v = doc.project.rows[-1]
    ok(doc, "add_variant", block=v.blocks[-1])
    assert graphcore.variants(doc.project, r.id) == [x.id for x in doc.project.rows[1:]]
    assert graphcore.variants(doc.project, "r999") == []


def test_a_block_or_a_session_dropped_on_its_own_spot_is_no_place():
    doc = Doc.from_template("sho_rgb")
    ix = Index(doc.project)
    for b in [x for x in doc.project.blocks if x in ix.row_of and ix.pos.get(x) is not None][:12]:
        for pl in places(doc.project, doc.view, block=b):
            if pl.op[0] == "move_block" and pl.lit:
                q, res = graphcore.apply(doc.project, "move_block", view=doc.view, **pl.op[1])
                assert q is not doc.project, (b, pl.op)
    doc = Doc.from_template("colour_complete")
    s = doc.project.rows[0].nights[-1][0]
    assert not [pl for pl in places(doc.project, doc.view, block=s)
                if pl.op[0] == "move_night" and pl.op[1].get("row") == doc.project.rows[0].id and pl.lit]
    q, res = graphcore.apply(doc.project, "move_night", night=s, row=doc.project.rows[0].id)
    assert res.ok and q is doc.project and not res.toast


def test_no_flats_said_only_of_a_session_with_lights():
    doc = Doc.from_template("colour_minimal")
    r = doc.project.rows[0]
    s1 = r.nights[0][0]
    ok(doc, "add_files", session=s1, kind="flats", paths=["D:/f/1.fit"])
    ok(doc, "add_night", row=r.id)
    s2 = doc.project.rows[0].nights[1][0]
    assert not [w for w in graphcore.warnings(doc.project, s2) if w[1] == "flats"]
    ok(doc, "add_files", session=s2, kind="lights", paths=["D:/l/1.fit"])
    assert [w for w in graphcore.warnings(doc.project, s2) if w[1] == "flats"]


def test_a_block_deleted_with_the_blocks_that_need_it():
    doc = Doc.from_template("colour_complete")
    r = doc.project.rows[0]
    den = of_kind(doc, r, "denoise")
    assert refused_inert(doc, "remove_block", block=den)                  # Stars/background needs it
    deps = graphcore.dependants(doc.project, den)
    assert deps and all(doc.project.blocks[d].kind != "denoise" for d in deps)
    res = ok(doc, "remove_with_dependants", block=den)
    assert all(d not in doc.project.blocks for d in deps + [den]) and res.toast
    engine_ok(doc)
    undo_redo(doc)
    grad = of_kind(doc, doc.project.rows[0], "gradient")
    assert graphcore.dependants(doc.project, grad) == []


def test_the_fold_problem_said_before_the_click():
    doc = Doc.from_template("colour_minimal")
    assert graphcore.fold_problem(doc.project) == ""
    r = doc.project.rows[0]
    ok(doc, "add_variant", block=of_kind(doc, r, "stacking"))
    assert graphcore.fold_problem(doc.project).startswith("a variant starts from the stack")


def test_a_block_moved_onto_a_branch_of_a_row_it_leaves_empty_is_refused():
    doc = Doc.from_template("colour_minimal")
    sid = doc.project.blocks[doc.project.rows[0].nights[0][0]].sid
    ok(doc, "add_stack", name="Built", nights=[sid], blocks=[{"kind": "registration"}], align=None)
    r = row(doc, "Built")
    reg = r.blocks[0]
    reason = refused_inert(doc, "move_block", block=reg, at={"type": "branch", "row": r.id, "y": None})
    assert reason == "a variant starts after the Registration"
    for pl in places(doc.project, doc.view, block=reg):              # every place answers, none raises
        q, res = graphcore.apply(doc.project, pl.op[0], view=doc.view, **pl.op[1])
        assert res.ok == pl.lit, (pl.op, res.reason_text())


def test_a_refused_deletion_says_the_blocks_that_need_it():
    doc = Doc.from_template("colour_complete")
    den = of_kind(doc, doc.project.rows[0], "denoise")
    res = doc.apply("remove_block", block=den)
    assert not res.ok and res.dependants == graphcore.dependants(doc.project, den) and res.dependants
    assert doc.apply("remove_block", block="b999").dependants == []


def test_single_files_left_out_by_hand_in_one_step():
    doc = Doc.from_template("colour_minimal")
    s = doc.project.rows[0].nights[0][0]
    ok(doc, "exclude_files", session=s, kind="lights", paths=["D:/l/a.fit"])
    ok(doc, "exclude_files", session=s, kind="darks", paths=["D:/d/1.fit", "D:/d/2.fit"])
    assert graphcore.excluded_files(doc.project, s, "lights") == ["a.fit"]
    assert graphcore.excluded_files(doc.project, s, "darks") == ["D:/d/1.fit", "D:/d/2.fit"]
    p, _ = compiled(doc)
    sid = doc.project.blocks[s].sid
    assert p.excluded == [(sid, "a.fit")] and p.session(sid).left_out == {"darks": ["D:/d/1.fit", "D:/d/2.fit"]}
    ok(doc, "exclude_files", session=s, kind="darks", paths=["D:/d/1.fit"], excluded=False)
    assert graphcore.excluded_files(doc.project, s, "darks") == ["D:/d/2.fit"]
    undo_redo(doc)
    reason = refused_inert(doc, "exclude_files", session="common", kind="darks", paths=["x"])
    assert reason == "Common's files are left out in each session"
