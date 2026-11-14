"""The core's legality is the engine's (INTEGRATION §3, decision B): each rule of the table, the model pieces it needs
(per-row alignment target, rows being built, Registration off or absent, calibration together, taps for a
combination's mid-row inputs), and the compiled definition the engine accepts after each gesture."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from fixtures import Doc  # noqa: E402
from test_points import ok, row, kinds, of_kind, refused_inert, where  # noqa: E402
from astro_gpu.core.project2 import Project2  # noqa: E402
from astro_gpu.graphcore import layout  # noqa: E402
from astro_gpu.graphcore.compile import compile_into, engine_problems  # noqa: E402
from astro_gpu.graphcore.operations import legality  # noqa: E402
import invariants as inv  # noqa: E402


def compiled(doc):
    p = Project2(name="t")
    c = compile_into(doc.project, p)
    return p, c


def engine_ok(doc):
    assert engine_problems(doc.project) == []
    assert legality(doc.project) == ""
    assert not inv.static(doc, layout(doc.project, doc.view))


def test_after_a_combination_only_stars_and_nothing_after_combine_layers():
    doc = Doc.from_template("filters_minimal")
    j = row(doc, "S-H-O").blocks[0]
    at = {"type": "insert", "row": row(doc, "S-H-O").id, "after": j}
    assert refused_inert(doc, "add_block", kind="gradient", at=at) == "no Gradient from combination"
    assert refused_inert(doc, "add_block", kind="color", at=at) == "no Color from combination"
    ok(doc, "add_block", kind="stars", at=at)
    s = row(doc, "S-H-O").blocks[1]
    assert refused_inert(doc, "add_block", kind="synth", at={"type": "insert", "row": row(doc, "S-H-O").id,
                                                              "after": s}) == "no Synthetic stars from combination"
    engine_ok(doc)
    doc = Doc.from_template("colour_complete")
    cl = row(doc, "Combine layers")
    assert refused_inert(doc, "add_block", kind="gradient", at={"type": "insert", "row": cl.id, "after": cl.blocks[0]}) \
        == "nothing follows Combine layers: its image is a layer of Post"


def test_a_mid_row_input_is_a_tap_and_synthetic_stars_is_no_input():
    doc = Doc.from_template("filters_minimal")
    ha, j = row(doc, "Ha"), row(doc, "S-H-O").blocks[0]
    astro = of_kind(doc, ha, "astrometry")
    ok(doc, "wire", src=astro, src_port="image", dst=j, dst_port="Hα")      # Astrometry has a Gradient after it
    assert [t.block for t in doc.project.taps] == [astro] and doc.project.taps[0].auto
    p, c = compiled(doc)
    tap = c.taps[doc.project.taps[0].id]
    assert [n["block"] for n in p.workflow(tap).blocks][-2:] == ["astrometry", "output"]
    fus = next(w for w in p.workflows if w.source and "fusion" in w.source)
    assert {i["role"]: i["pid"] for i in fus.source["fusion"]["inputs"]}["Ha"] == tap
    engine_ok(doc)
    w = next(x for x in doc.project.wires if x.src[0] == astro)
    ok(doc, "unwire", wire=w.id)
    assert doc.project.taps == []                                           # gone with its last wire
    doc = Doc.from_template("filters_complete")
    sii = row(doc, "SII")
    stars = sii.blocks[-1]
    ok(doc, "add_block", kind="synth", at={"type": "insert", "row": sii.id, "after": stars})
    assert all(w.src[0] == stars for w in doc.project.wires if doc.project.blocks[w.dst[0]].kind == "combine"
               and w.src[0] in row(doc, "SII").blocks)                      # the combination keeps reading the Stars
    synth = row(doc, "SII").blocks[-1]
    j = row(doc, "S-H-O").blocks[0]
    assert "no input of a combination" in refused_inert(doc, "wire", src=synth, src_port="Starless", dst=j, dst_port="S II")
    engine_ok(doc)


def test_rows_being_built_and_registration_off_or_absent():
    doc = Doc.from_template("colour_minimal")
    r = doc.project.rows[0]
    reg = of_kind(doc, r, "registration")
    ok(doc, "switch_off", block=reg)                                       # a pass-through for the engine
    ok(doc, "remove_block", block=reg)
    assert kinds(doc, row(doc, "Session 1"))[0] == "selection"
    engine_ok(doc)
    stk = of_kind(doc, row(doc, "Session 1"), "stacking")
    assert refused_inert(doc, "remove_block", block=stk) == "Astrometry needs Stacking first"
    doc = Doc()
    ok(doc, "add_block", kind="session")
    r = doc.project.rows[0]
    for b in list(reversed(r.blocks)):
        ok(doc, "remove_block", block=b)
    r = doc.project.rows[0]
    assert r.blocks == [] and r.nights                                      # the stack stays, being built
    engine_ok(doc)
    p, _c = compiled(doc)
    assert [n["block"] for n in p.workflows[0].blocks] == ["calibration"]
    pl = where(doc, "insert", kind="stacking", match={"row": r.id, "after": None})
    assert pl and pl[0].lit
    ok(doc, "add_block", kind="stacking", at={"type": "insert", "row": r.id, "after": None})
    ok(doc, "add_block", kind="registration", at={"type": "insert", "row": r.id, "after": None})
    assert kinds(doc, doc.project.rows[0]) == ["registration", "stacking"]
    engine_ok(doc)


def test_needs_are_the_engines():
    doc = Doc.from_template("several_images")
    r = doc.project.rows[0]
    ok(doc, "add_block", kind="stars", at={"type": "insert", "row": r.id, "after": r.blocks[0]})   # no Denoise
    engine_ok(doc)
    doc = Doc.from_template("colour_minimal")
    r = doc.project.rows[0]
    ok(doc, "add_variant", block=of_kind(doc, r, "gradient"))
    v = next(x for x in doc.project.rows if x.fork)
    assert refused_inert(doc, "add_block", kind="astrometry", at={"type": "insert", "row": v.id,
                                                                 "after": v.blocks[0]}) == "one Astrometry per workflow"
    doc = Doc.from_template("several_images")                              # one node name per workflow
    r = doc.project.rows[0]
    ok(doc, "add_block", kind="astrometry", at={"type": "insert", "row": r.id, "after": r.blocks[0]})
    a = doc.project.rows[0].blocks[1]
    ok(doc, "switch_off", block=a)
    assert refused_inert(doc, "add_block", kind="astrometry", at={"type": "insert", "row": r.id, "after": a}) \
        == "Astrometry already here, off: switch on (D)"


def test_alignment_target_per_row():
    doc = Doc.from_template("sho_rgb")
    sii, ha, o3, rgb = (row(doc, n) for n in ("SII", "Ha", "OIII", "RGB"))
    assert [x.align and x.align["ref"] for x in (sii, ha, o3, rgb)] == [None, sii.id, sii.id, sii.id]
    p, c = compiled(doc)
    al = [n for n in p.workflow(c.wids[rgb.id]).blocks if n["block"] == "align_on"][0]
    assert al["reference"] == c.wids[sii.id] and al["reference_block"] == "astrometry"     # the plate comes along
    ok(doc, "add_block", kind="astrometry", at={"type": "insert", "row": ha.id, "after": of_kind(doc, ha, "stacking")})
    ok(doc, "set_align", row=o3.id, ref=ha.id)                              # A -> B -> C
    assert doc.project.reference == sii.id and doc.project.rows[2].align["ref"] == ha.id
    assert refused_inert(doc, "set_align", row=ha.id, ref=o3.id) == "OIII already reads Ha: it would make a loop"
    assert refused_inert(doc, "set_align", row=ha.id, ref=row(doc, "S-H-O").id) == "no stack"
    engine_ok(doc)
    ok(doc, "set_align", row=rgb.id, ref=None)                              # an Align on with its target to choose
    assert doc.row_badge(rgb.id) == "grid to choose"
    grad, colour = (of_kind(doc, row(doc, "RGB"), k) for k in ("gradient", "color"))
    assert doc.greyed(grad) == "reference to wire"                          # the engine leaves it out, and what
    assert doc.greyed(colour) == "downstream of Gradient"                   # follows
    engine_ok(doc)
    assert refused_inert(doc, "own_grid", row=rgb.id) == "needs Astrometry first"   # its Color, then
    ok(doc, "own_grid", row=ha.id)
    assert row(doc, "Ha").align is None and doc.row_badge(ha.id) == "reference"     # OIII is on its grid
    ok(doc, "set_reference", row=row(doc, "RGB").id)                        # reference for all
    assert all(x.align and x.align["ref"] == rgb.id for x in doc.project.rows if x.nights and x.id != rgb.id)
    engine_ok(doc)


def test_calibration_together_and_each_night():
    doc = Doc.from_template("colour_minimal")
    r = doc.project.rows[0]
    ok(doc, "add_night", row=r.id)
    c2 = doc.project.rows[0].nights[1][1]
    assert doc.project.calibrate_together
    assert refused_inert(doc, "switch_off", block=c2) == "calibrated together: one setting for all nights"
    ok(doc, "set_calibrate_together", on=False)
    ok(doc, "switch_off", block=c2)
    p, _c = compiled(doc)
    sid = doc.project.blocks[doc.project.rows[0].nights[1][0]].sid
    w = p.workflows[0]
    assert p.stack_mode == "per_session" and not w.blocks[0]["off"]
    assert any(v == {"apply_masters": False} for k, v in w.params.items() if k.startswith("calibrate@")) or sid is None
    engine_ok(doc)
    assert refused_inert(doc, "set_calibrate_together", on=True) == "calibrated together: one setting for all nights"
    ok(doc, "switch_off", block=doc.project.rows[0].nights[0][1])
    ok(doc, "set_calibrate_together", on=True)
    ok(doc, "add_night", row=doc.project.rows[0].id)                       # a night joining takes the stack's switch
    assert all(doc.project.blocks[c].off for _s, c in doc.project.rows[0].nights)
    engine_ok(doc)
