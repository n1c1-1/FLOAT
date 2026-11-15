"""What a gesture copies or carries keeps all it had: a variant's wires keep their channel, image and workflow; a
split or merged night keeps its frames left out by hand; a duplicated stack copies its whole workflow."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from fixtures import Doc  # noqa: E402
from test_points import ok, row, of_kind  # noqa: E402
from test_engine_rules import compiled, engine_ok  # noqa: E402
from astro_gpu.graphcore.model import Index  # noqa: E402


def _oiii_wire(doc, j):
    return next(w for w in doc.project.wires if w.dst == [j, "O III"] and w.active)


def test_a_variant_s_wires_keep_the_channel_image_and_workflow_they_read():
    _variant_wires("add_variant")


def test_a_tried_variant_s_wires_keep_the_channel_image_and_workflow_they_read_once_kept():
    _variant_wires("try_variant")


def _variant_wires(op):
    doc = Doc.from_template("filters_complete")
    j = row(doc, "S-H-O").blocks[0]
    ok(doc, "set_input", block=j, port="O III", channel="GB")
    w0 = _oiii_wire(doc, j)
    w0.wid = 7                            # a workflow named on the wire (several end on one block)
    kw = {"change": {"method": "adapted"}} if op == "try_variant" else {}
    src_row = Index(doc.project).row_of[w0.src[0]]
    ok(doc, op, block=of_kind(doc, Index(doc.project).row[src_row], "denoise"), **kw)
    copies = [w for w in doc.project.wires if w.dst == [j, "O III"] and w.id != w0.id]
    assert copies, op
    for w in copies:
        assert (w.channel, w.image, w.wid) == (w0.channel, w0.image, w0.wid), op
        assert w.src[0] != w0.src[0] and not w.active
    if op == "try_variant":
        ok(doc, "keep_variant", row=next(r for r in doc.project.rows if r.trial is not None).id)
        kept = _oiii_wire(doc, j)
        assert (kept.channel, kept.image, kept.wid) == ("GB", w0.image, 7)


# ---------------------------------------------------------------------------------------------------- 4.4
def _night_with_exclusions():
    doc = Doc.from_template("colour_minimal")
    s = doc.project.rows[0].nights[0][0]
    ok(doc, "add_files", session=s, kind="lights", paths=["D:/l/1.fit", "D:/l/2.fit", "D:/l/3.fit"], mode="replace")
    sid = doc.project.blocks[s].sid
    ok(doc, "exclude_frames", sid=sid, names=["1.fit", "3.fit"])
    return doc, s, sid


def _excluded_of(doc, lights):
    p, _ = compiled(doc)
    s = next(x for x in p.sessions if x.lights == lights)
    return sorted(n for sid, n in p.excluded if sid == s.id)


def test_a_light_excluded_stays_excluded_in_the_session_a_split_moves_it_to():
    doc, s, sid = _night_with_exclusions()
    ok(doc, "split_night", night=s, lights=["D:/l/1.fit"])
    assert _excluded_of(doc, ["D:/l/1.fit"]) == ["1.fit"]
    assert _excluded_of(doc, ["D:/l/2.fit", "D:/l/3.fit"]) == ["3.fit"]
    engine_ok(doc)
    doc, s, sid = _night_with_exclusions()
    ok(doc, "split_by", session=s, groups=[{"into": "night", "files": {"lights": ["D:/l/3.fit"]}},
                                          {"into": "row", "files": {"lights": ["D:/l/1.fit"]}, "name": "Part"}])
    assert _excluded_of(doc, ["D:/l/3.fit"]) == ["3.fit"] and _excluded_of(doc, ["D:/l/1.fit"]) == ["1.fit"]
    assert _excluded_of(doc, ["D:/l/2.fit"]) == []


def test_a_merged_session_s_exclusions_go_to_the_session_that_absorbs_it_and_it_leaves_the_project():
    doc, s, sid = _night_with_exclusions()
    r = doc.project.rows[0]
    ok(doc, "add_night", row=r.id, files={"lights": ["D:/m/4.fit"]})
    s2 = doc.project.rows[0].nights[1][0]
    sid2 = doc.project.blocks[s2].sid
    if sid2 is None:                                       # a night compiled later: named by the compile
        p, _ = compiled(doc)
        sid2 = next(x.id for x in p.sessions if x.lights == ["D:/m/4.fit"])
        doc.project.blocks[s2].sid = sid2
        doc.project.__dict__.pop("_json", None)
        doc.project.nights.append({"sid": sid2, "name": doc.project.blocks[s2].name, "filter": "", "files": {},
                                   "left_out": {}})
    ok(doc, "exclude_frames", sid=sid2, names=["4.fit"])
    ok(doc, "merge_nights", a=s, b=s2)
    p, _ = compiled(doc)
    assert sid2 not in [x.id for x in p.sessions]
    assert sorted(p.excluded) == [(sid, "1.fit"), (sid, "3.fit"), (sid, "4.fit")]
    engine_ok(doc)


# ---------------------------------------------------------------------------------------------------- 4.5
def test_a_duplicated_stack_copies_its_whole_workflow_and_owns_its_copies():
    doc = Doc.from_template("colour_minimal")
    r = doc.project.rows[0]
    r.output = {"linked": False}
    r.extra = {"stack": {"min_transparency": 0.6}}
    r.calib = {"calibrate": {"hot_sigma": 6.0}}
    r.align = {**r.align, "params": {"k": [1]}} if r.align else None
    cal = doc.project.blocks[r.nights[0][1]]
    cal.params = {"calibrate": {"hot_sigma": 5.0}}
    doc.project.__dict__.pop("_json", None)
    ok(doc, "duplicate_row", row=r.id)
    o, c = doc.project.rows[0], doc.project.rows[1]
    for k in ("output", "extra", "unset", "options", "end", "calib", "calib_node", "calib_fork", "all_nights",
              "pauses"):
        assert getattr(c, k) == getattr(o, k), k
    ccal = doc.project.blocks[c.nights[0][1]]
    assert ccal.params == cal.params and ccal.params is not cal.params
    c.output["linked"] = True
    c.calib["calibrate"]["hot_sigma"] = 3.0
    assert o.output == {"linked": False} and o.calib["calibrate"]["hot_sigma"] == 6.0
    if o.align:
        c.align["params"]["k"].append(2)
        assert o.align["params"]["k"] == [1]
    p, _ = compiled(doc)
    assert p.workflows[1].params.get("render_simple") == {"linked": True}
    assert p.workflows[1].params["stack"]["min_transparency"] == 0.6


# ---------------------------------------------------------------------------------------------------- 13.2
def _own_darks():
    doc = Doc.from_template("colour_minimal")
    s = doc.project.rows[0].nights[0][0]
    ok(doc, "set_common_files", kind="darks", paths=["D:/c/1.fit", "D:/c/2.fit"], mode="replace")
    ok(doc, "add_files", session=s, kind="darks", paths=["D:/s/1.fit"], mode="replace")
    return doc, s


def _darks(doc, s):
    p, _ = compiled(doc)
    return p.session(doc.project.blocks[s].sid).darks


def test_a_session_s_own_list_emptied_stays_its_own_empty_list_until_it_follows_common_again():
    doc, s = _own_darks()
    assert _darks(doc, s) == ["D:/s/1.fit"]
    ok(doc, "remove_files", session=s, kind="darks", paths=["D:/s/1.fit"])
    assert _darks(doc, s) == []                                   # no dark, never Common's in silence
    ok(doc, "reset_own_files", session=s, kind="darks")           # the Table's "follows Common again"
    assert _darks(doc, s) == ["D:/c/1.fit", "D:/c/2.fit"]
    doc, s = _own_darks()
    ok(doc, "move_files", src_session=s, src_kind="darks", dst_session=s, dst_kind="flats")
    assert _darks(doc, s) == []
    doc, s = _own_darks()
    ok(doc, "split_by", session=s, groups=[{"into": "night", "files": {"darks": ["D:/s/1.fit"]}}])
    assert _darks(doc, s) == []
    doc, s = _own_darks()
    ok(doc, "move_files", src_session=s, src_kind="darks", dst_session="common", dst_kind="darks")
    assert _darks(doc, s) == ["D:/c/1.fit", "D:/c/2.fit", "D:/s/1.fit"]     # its files Common's: Common followed
