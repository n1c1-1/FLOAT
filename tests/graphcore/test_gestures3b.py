"""What the current software does on the graph page, now gestures of the core (step 3b): Replace or Try as variant
then Keep or Discard, a combination's inputs (channel, weight, image) and its "· Stars" twin, a night shared by two
stacks, a block's image to Post as a workflow of its own (a tap), a Calibration with its own files, the engine's own
settings (Min pairs...), a row made before the free graph rebuilt by its first change. Each compiled definition is one
the engine accepts; ids are never renumbered; Undo gives back the graph before."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from fixtures import Doc  # noqa: E402
from test_points import ok, row, of_kind, refused_inert  # noqa: E402
from test_engine_rules import compiled, engine_ok  # noqa: E402
from astro_gpu.graphcore.model import Index  # noqa: E402


def wf(p, name):
    return next(w for w in p.workflows if w.name == name)


# ---------------------------------------------------------------------------------------------------- 1. variants
def test_try_a_variant_then_keep_it_retires_the_original_id():
    doc = Doc.from_template("colour_minimal")
    r = doc.project.rows[0]
    grad = of_kind(doc, r, "gradient")
    p0, _ = compiled(doc)
    wid0 = p0.workflows[0].id
    ok(doc, "try_variant", block=grad, change={"method": "rbf"}, name="Session 1 · RBF")
    v = row(doc, "Session 1 · RBF")
    assert v.trial == wid0 and v.fork == [r.id, of_kind(doc, r, "astrometry")]
    p, _ = compiled(doc)
    var = wf(p, "Session 1 · RBF")
    assert var.variant_of == wid0 and var.id != wid0
    assert [n["block"] for n in var.blocks] == [n["block"] for n in p.workflow(wid0).blocks]
    assert next(n for n in var.blocks if n["block"] == "gradient")["method"] == "rbf"
    engine_ok(doc)
    vid = var.id
    ok(doc, "keep_variant", row=v.id)
    p, _ = compiled(doc)
    assert [w.id for w in p.workflows] == [vid] and p.workflows[0].name == "Session 1"
    assert p.workflows[0].variant_of is None
    assert next(n for n in p.workflows[0].blocks if n["block"] == "gradient")["method"] == "rbf"
    assert wid0 not in doc.project.order and doc.project.next_wid == vid + 1
    engine_ok(doc)
    doc.undo()
    doc.undo()
    p, _ = compiled(doc)
    assert [w.id for w in p.workflows] == [wid0]


def test_discard_a_variant_and_a_whole_stack_trial_shares_its_nights():
    doc = Doc.from_template("colour_minimal")
    r = doc.project.rows[0]
    ok(doc, "add_night", row=r.id)
    reg = of_kind(doc, r, "registration")
    ok(doc, "try_variant", block=reg, change={"param": ["register", "min_pairs", 20]})
    v, r = doc.project.rows[-1], doc.project.rows[0]
    assert not v.fork and len(v.nights) == 2
    assert [doc.project.blocks[s].sid for s, _c in v.nights] == [doc.project.blocks[s].sid for s, _c in r.nights]
    p, _ = compiled(doc)
    assert len(p.sessions) == 2                                   # the nights are stored once
    assert p.workflows[1].params["register"] == {"min_pairs": 20} and p.workflows[1].variant_of == p.workflows[0].id
    assert p.workflows[1].sessions == p.workflows[0].sessions
    engine_ok(doc)
    ok(doc, "discard_variant", row=v.id)
    p, _ = compiled(doc)
    assert len(p.workflows) == 1 and len(p.sessions) == 2
    ok(doc, "try_variant", block=reg, change={"param": ["register", "min_pairs", 20]})
    keep = doc.project.rows[-1]
    ok(doc, "keep_variant", row=keep.id)
    p, _ = compiled(doc)
    assert len(p.workflows) == 1 and p.workflows[0].name == "Session 1" and p.workflows[0].params["register"]
    assert len(doc.project.rows) == 1 and len(p.sessions) == 2
    engine_ok(doc)


def test_keep_moves_the_variants_and_wires_of_the_original():
    doc = Doc.from_template("filters_complete")
    sii = row(doc, "SII")
    den = of_kind(doc, sii, "denoise")
    ok(doc, "try_variant", block=den, change={"method": "adapted"}, name="SII fine")
    v = row(doc, "SII fine")
    stars_v = v.blocks[-1]
    ok(doc, "keep_variant", row=v.id)
    s = row(doc, "SII")
    assert s.blocks[-1] == stars_v                                 # the variant's blocks continue the row
    j = row(doc, "S-H-O").blocks[0]
    assert any(w.src[0] == stars_v and w.dst[0] == j and w.active for w in doc.project.wires)
    engine_ok(doc)


# ---------------------------------------------------------------------------------------------------- 2. combinations
def test_combination_input_channel_weight_image_and_its_stars_twin():
    doc = Doc.from_template("filters_complete")
    j = row(doc, "S-H-O").blocks[0]
    ok(doc, "set_input", block=j, port="O III", channel="GB", weight=1.5)
    p, _ = compiled(doc)
    fus = wf(p, "S-H-O")
    oiii = next(i for i in fus.source["fusion"]["inputs"] if i["role"] == "OIII")
    assert oiii["channel"] == "GB" and oiii["image"] == "starless"
    assert fus.params["combine_sho"]["weights"] == {"OIII": 1.5}
    assert "unknown channel" in refused_inert(doc, "set_input", block=j, port="O III", channel="Q")
    ok(doc, "set_input", block=j, port="Hα", image="output")
    p, _ = compiled(doc)
    ha = next(i for i in wf(p, "S-H-O").source["fusion"]["inputs"] if i["role"] == "Ha")
    assert "image" not in ha and ha["pid"] == wf(p, "Ha").id
    engine_ok(doc)
    ok(doc, "set_input", block=j, port="Hα", image="starless")
    ok(doc, "add_stars_combination", block=j)
    st = row(doc, "S-H-O · Stars")
    p, _ = compiled(doc)
    ins = wf(p, "S-H-O · Stars").source["fusion"]["inputs"]
    assert {i["role"]: (i["image"], i["channel"]) for i in ins} == {"SII": ("stars", "L"), "Ha": ("stars", "L"),
                                                                     "OIII": ("stars", "GB")}
    assert wf(p, "S-H-O · Stars").blocks is None                  # as the engine stores a combination with no block
    assert st.legacy is not None
    ok(doc, "add_block", kind="stars", at={"type": "insert", "row": st.id, "after": st.blocks[0]})
    p, _ = compiled(doc)
    assert [n["block"] for n in wf(p, "S-H-O · Stars").blocks] == ["output", "stars"]   # its first block: rebuilt
    engine_ok(doc)


def test_a_new_combination_is_stored_as_the_engine_stores_one():
    doc = Doc.from_template("filters_minimal")
    ok(doc, "add_combination", type="ho", name="H-O nebula")
    p, _ = compiled(doc)
    w = wf(p, "H-O nebula")
    assert w.source == {"fusion": {"type": "ho", "inputs": []}} and w.blocks is None
    assert w.options == {"denoise": None, "stars": None, "gradient": "polynomial", "deconvolution": None}


# ---------------------------------------------------------------------------------------------------- 3. shared nights
def test_a_night_shared_by_two_stacks_is_one_session_edited_once():
    doc = Doc.from_template("filters_minimal")
    ha, o3 = row(doc, "Ha"), row(doc, "OIII")
    n = ha.nights[0][0]
    ok(doc, "share_night", night=n, row=o3.id)
    o3 = row(doc, "OIII")
    twin = o3.nights[-1][0]
    assert doc.project.blocks[twin].sid == doc.project.blocks[n].sid and twin != n
    ok(doc, "add_files", session=twin, kind="lights", paths=["D:/a/1.fit", "D:/a/2.fit"])
    assert doc.project.blocks[n].files["lights"] == ["D:/a/1.fit", "D:/a/2.fit"]
    ok(doc, "rename_night", night=n, name="Dual 1")
    assert doc.project.blocks[twin].name == "Dual 1"
    p, _ = compiled(doc)
    sid = doc.project.blocks[n].sid
    assert [s.id for s in p.sessions].count(sid) == 1 and p.session(sid).name == "Dual 1"
    assert p.session(sid).lights == ["D:/a/1.fit", "D:/a/2.fit"]
    assert sid in wf(p, "Ha").sessions and sid in wf(p, "OIII").sessions
    engine_ok(doc)
    assert "already in" in refused_inert(doc, "share_night", night=n, row=o3.id)
    assert "shared by several stacks" in refused_inert(doc, "split_night", night=n, lights=["D:/a/1.fit"])
    ok(doc, "remove_night", night=twin)                         # out of one stack: the session stays for the other
    p, _ = compiled(doc)
    assert sid in [s.id for s in p.sessions]
    ok(doc, "remove_night", night=n)                            # the last stack drawing it: the session goes
    p, _ = compiled(doc)
    assert sid not in [s.id for s in p.sessions]


def test_a_night_set_its_filter_and_ids_follow_the_order_made():
    doc = Doc()
    ok(doc, "add_row", kind="session", name="SII")
    ok(doc, "add_row", kind="session", name="Ha")
    s = doc.project.rows[1].nights[0][0]
    ok(doc, "set_night_filter", night=s, filter="Ha")
    p, _ = compiled(doc)
    assert [(x.id, x.filter) for x in p.sessions] == [(1, ""), (2, "Ha")]
    assert [w.id for w in p.workflows] == [1, 2] and doc.project.order == [1, 2]


# ---------------------------------------------------------------------------------------------------- 5. taps, settings
def test_a_mid_row_block_to_post_is_a_workflow_of_its_own_and_goes_with_its_wire():
    doc = Doc.from_template("colour_complete")
    r = doc.project.rows[0]
    ok(doc, "set_param", block=of_kind(doc, r, "denoise"), stage="denoise", name="fast", value=True)
    grad = of_kind(doc, r, "gradient")
    ok(doc, "send_to_post", block=grad, name="Session 1 · Gradient")
    t = doc.project.taps[0]
    assert t.block == grad and not t.auto and t.wid is not None
    p, c = compiled(doc)
    w = p.workflow(t.wid)
    assert w.name == "Session 1 · Gradient" and [n["block"] for n in w.blocks][-2:] == ["gradient", "output"]
    assert w.params.get("denoise") == {"fast": True}             # a copy of the row's workflow keeps its settings
    engine_ok(doc)
    wire = next(x for x in doc.project.wires if x.wid == t.wid)
    ok(doc, "unwire", wire=wire.id)
    assert doc.project.taps == [] and t.wid not in doc.project.order


def test_engine_settings_by_their_real_keys_and_own_calibration_files():
    doc = Doc.from_template("colour_minimal")
    r = doc.project.rows[0]
    reg, st = of_kind(doc, r, "registration"), of_kind(doc, r, "stacking")
    ok(doc, "set_param", block=reg, stage="register", name="min_pairs", value=10)
    assert "not a valid value" in refused_inert(doc, "set_param", block=reg, stage="register", name="min_pairs", value=2)
    assert "holds no setting" in refused_inert(doc, "set_param", block=st, stage="register", name="min_pairs", value=12)
    assert "holds no setting" in refused_inert(doc, "set_param", block=st, stage="stack", name="min_transparency",
                                               value=0.6)
    ok(doc, "set_param", block=st, stage="stack", name="method", value="drizzle")
    ok(doc, "set_row_param", row=r.id, stage="render_simple", name="linked", value=False)
    c = r.nights[0][1]
    ok(doc, "set_param", block=c, stage="calibrate", name="apply_masters", value=False)
    ok(doc, "set_own_calibration", block=c, kind="darks", paths=["D:/d/1.fit"])
    p, _ = compiled(doc)
    w = p.workflows[0]
    sid = doc.project.blocks[c].sid
    assert w.params["register"] == {"min_pairs": 10}               # the default value too: stored as given
    assert w.params["stack"]["method"] == "drizzle" and w.params["render_simple"] == {"linked": False}
    assert w.params[f"calibrate@{sid}"] == {"apply_masters": False}
    assert w.calibration == {str(sid): {"fork": 1, "darks": ["D:/d/1.fit"]}}
    engine_ok(doc)
    ok(doc, "follow_session", block=c, kind="darks")
    ok(doc, "remove_own_calibration", block=c)
    p, _ = compiled(doc)
    assert p.workflows[0].calibration is None


def test_a_frozen_row_made_before_the_free_graph_is_rebuilt_by_its_first_change():
    doc = Doc.from_template("colour_minimal")
    g = doc.project
    r = g.rows[0]
    opts = {"denoise": None, "stars": None, "gradient": "polynomial", "deconvolution": None}
    r.legacy = {"options": dict(opts), "end": None, "frozen": True}
    doc.project = g
    p, _ = compiled(doc)
    assert p.workflows[0].blocks is None and p.workflows[0].options == opts
    ok(doc, "set_param", block=of_kind(doc, r, "registration"), stage="register", name="min_pairs", value=12)
    assert doc.project.rows[0].legacy is not None                # a setting is no change of its chain
    ok(doc, "switch_off", block=of_kind(doc, doc.project.rows[0], "gradient"))
    assert doc.project.rows[0].legacy is None
    p, _ = compiled(doc)
    assert p.workflows[0].blocks is not None
    assert Index(doc.project).row[r.id].legacy is None


# ---------------------------------------------------------------------------------------------------- 4. Post
def test_the_wires_to_post_are_the_shared_set_of_images():
    from astro_gpu.graphcore.post import post_actions, shared_images, wired_images, with_post_wires
    doc = Doc.from_template("colour_complete")
    r = doc.project.rows[0]
    ok(doc, "send_to_post", block=of_kind(doc, r, "gradient"), name="Session 1 · Gradient")
    g = doc.project
    wid, tap = g.rows[0].wid, g.taps[0].wid
    post = {"versions": [{"id": 1, "layers": [{"kind": "workflow", "pid": wid, "part": "starless"}]},
                         {"id": 2, "layers": [{"kind": "workflow", "pid": tap, "part": None}],
                          "removed": [[wid, "starless"]]}], "active": 2}
    shared = shared_images(post, defaults={(wid, "stars")})
    assert shared == {(wid, "starless"), (wid, "stars"), (tap, None)}   # a version's own removal keeps the image
    g2 = with_post_wires(g, shared)
    assert wired_images(g2) == shared
    stars = of_kind(doc, r, "stars")
    assert {(w.src[0], w.src[1]) for w in g2.wires if g2.blocks[w.dst[0]].kind == "post"} >= {
        (stars, "Starless"), (stars, "Stars only"), (of_kind(doc, r, "gradient"), "image")}
    g3 = with_post_wires(g2, shared - {(wid, "stars")})
    assert post_actions(g2, g3) == [("remove_post_part", wid, "stars")]
    assert post_actions(g3, g2) == [("add_post_layer", wid, "stars")]


# ---------------------------------------------------------------------------------------------------- 6. templates
GALLERY = [dict(key="filters_full", name="Filters full", opts=dict(debruitage=True, etoiles="retirer", fond="simple"),
                kind="fusion", palette="S-H-O", filters=["SII", "Ha", "OIII"]),
           dict(key="dual_x2", name="Dual-band x2", opts=dict(debruitage=True, etoiles="retirer", fond="simple"),
                kind="fusion", palette="S-H-O", filters=["O III / H-alpha", "O III / S II"], dual_x2=True)]


def test_templates_are_one_step_of_core_operations_the_engine_accepts():
    from astro_gpu.graphcore import templates
    for t in GALLERY:
        doc = Doc()
        plan = templates.build(doc.project, t, "S-H-O", dict(t["opts"]))
        res = doc.apply_many(plan.steps)
        assert res.ok, res.reason_text()
        assert len(doc._undo) == 1
        engine_ok(doc)
        p, c = compiled(doc)
        assert [w.name for w in p.workflows][-2:] in (["S-H-O · Stars", "S-H-O"], ["Stars", "S-H-O"])
        acts = plan.post(res.created["rows"])
        assert acts[0][0] == "add_layer" and acts[-1][0] == "removed"
    o3 = row(doc, "O III")                                        # the dual-band's O III row: both nights, shared
    assert [doc.project.blocks[s].sid for s, _c in o3.nights] == [1, 2]
    assert sorted(len(r.nights) for r in doc.project.rows if r.nights) == [1, 1, 1, 2]


def test_keep_a_trial_aligned_on_its_original_takes_the_original_alignment():
    doc = Doc.from_template("colour_minimal")
    r = doc.project.rows[0]
    ok(doc, "try_variant", block=of_kind(doc, r, "registration"), change={"param": ["register", "min_pairs", 20]})
    v = doc.project.rows[-1]
    ok(doc, "set_align", row=v.id, ref=r.id, block=of_kind(doc, doc.project.rows[0], "astrometry"))
    ok(doc, "keep_variant", row=v.id)
    assert len(doc.project.rows) == 1 and doc.project.rows[0].align is None
    engine_ok(doc)
