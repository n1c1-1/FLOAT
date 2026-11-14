"""The owner's points 3–8 (BRIEF.md) and the acceptance scenarios of CONCEPTION §6 the core alone can decide."""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import invariants as inv  # noqa: E402
from fixtures import Doc  # noqa: E402
from astro_gpu.graphcore import layout, places, COMMON  # noqa: E402
from astro_gpu.graphcore.model import Index, check  # noqa: E402


# ---------------------------------------------------------------------------------------------------- helpers
def ok(doc, op, **kw):
    r = doc.apply(op, **kw)
    assert r.ok, (op, kw, r.reason_text())
    check(doc.project)
    return r


def row(doc, name):
    return next(r for r in doc.project.rows if r.name == name)


def kinds(doc, r):
    return [doc.project.blocks[b].kind for b in r.blocks]


def of_kind(doc, r, k):
    return next(b for b in r.blocks if doc.project.blocks[b].kind == k)


def where(doc, pk, kind=None, block=None, match=None, **more):
    match = dict(match or {}, **more)
    out = []
    for p in places(doc.project, doc.view, kind=kind, block=block):
        at = p.op[1].get("at", p.op[1])
        if p.kind == pk and all(at.get(k) == v or p.op[1].get(k) == v for k, v in match.items()):
            out.append(p)
    return out


def drop(doc, place):
    before = layout(doc.project, doc.view)
    assert place.lit, place.reason_text()
    r = doc.drop(place)
    assert r.ok, r.reason_text()
    after = layout(doc.project, doc.view)
    assert not inv.static(doc, after), inv.static(doc, after)
    assert not inv.i5(after, r.focus, place, doc.project)
    assert not inv.i4(before, after, place.op[0], r.acted)
    return r


def boxes_of(d, doc, rid):
    r = Index(doc.project).row[rid]
    return {b: d.boxes.get(b) for b in r.blocks + [x for n in r.nights for x in n if x is not None]}


def session_doc():
    """A Session (it comes alone), then its Calibration, Registration, Selection, Stacking placed one by one, the
    Stacking sent to Post."""
    doc = Doc()
    ok(doc, "add_block", kind="session")
    rid = doc.project.rows[0].id
    for k in ("calibration", "registration", "selection", "stacking"):
        ok(doc, "add_block", kind=k, row=rid)
    ok(doc, "send_to_post", block=doc.project.rows[0].blocks[-1])
    return doc


def refused_inert(doc, op, **kw):
    js, n = doc._json, len(doc._undo)
    r = doc.apply(op, **kw)
    assert not r.ok
    assert doc._json == js and len(doc._undo) == n
    return r.reason_text()


# ---------------------------------------------------------------------------------------------------- owner's points
def test_point3_identical_variant_is_allowed():
    doc = session_doc()
    r = doc.project.rows[0]
    parent = list(r.blocks)
    sel = of_kind(doc, r, "selection")
    drop(doc, where(doc, "branch", kind="selection", y=sel)[0])
    v = doc.project.rows[1]
    assert v.fork == [r.id, of_kind(doc, r, "registration")] and kinds(doc, v) == ["selection", "stacking"]
    assert doc.summary(v.blocks[0]) == "like the mother — change a setting"
    assert doc.project.rows[0].blocks == parent


def test_point4_second_session_joins_or_starts_its_own_row():
    doc = session_doc()
    r = doc.project.rows[0]
    assert where(doc, "night", kind="session", row=r.id)[0].lit
    assert where(doc, "row", kind="session")[0].lit
    drop(doc, where(doc, "night", kind="session", row=r.id)[0])
    s2 = r.nights[1][0] if len(r.nights) > 1 else doc.project.rows[0].nights[1][0]
    drop(doc, where(doc, "row", block=s2)[0])
    assert [x.name for x in doc.project.rows] == ["Session 1", "Session 2"]


def test_point5_every_listed_row_is_drawn():
    doc = session_doc()
    ok(doc, "duplicate_row", row=doc.project.rows[0].id)
    d = layout(doc.project, doc.view)
    assert len(doc.project.rows) == 2 and not inv.rows_drawn(doc.project, d)
    emp = of_kind(doc, doc.project.rows[0], "stacking")
    ok(doc, "set_setting", block=emp, key="Drop size", value=1.0)
    assert {c[1] for c in d.captions} | {r.name for r in doc.project.rows if len(r.nights) == 1} \
        >= {r.name for r in doc.project.rows}


def test_point6_branch_place_never_goes_in_series():
    doc = Doc.from_template("colour_complete")
    r = doc.project.rows[0]
    before = list(r.blocks)
    col = of_kind(doc, r, "color")
    drop(doc, where(doc, "branch", kind="color", y=col)[0])
    assert doc.project.rows[0].blocks == before
    assert kinds(doc, r).count("color") == 1
    v = next(x for x in doc.project.rows if x.fork)
    assert kinds(doc, v)[0] == "color" and v.fork[1] == of_kind(doc, r, "gradient")


def test_point7_send_to_post_moves_nothing():
    doc = Doc.from_template("sho_rgb")
    before = layout(doc.project, doc.view)
    n = len(doc.project.rows)
    r = ok(doc, "send_to_post", block=of_kind(doc, row(doc, "Ha"), "gradient"))
    after = layout(doc.project, doc.view)
    assert len(doc.project.rows) == n and not inv.static(doc, after)
    assert not inv.i4(before, after, "send_to_post", r.acted)


def test_point8_drop_on_lower_row_moves_only_that_row():
    doc = Doc.from_template("sho_rgb")
    oiii = row(doc, "OIII")
    ok(doc, "add_variant", block=of_kind(doc, oiii, "denoise"))
    before = layout(doc.project, doc.view)
    grad = of_kind(doc, oiii, "gradient")
    p = where(doc, "insert", kind="denoise", row=oiii.id, after=grad)
    assert p and p[0].lit
    drop(doc, p[0])
    after = layout(doc.project, doc.view)
    v = next(x for x in doc.project.rows if x.fork)
    moved = {b for b in before.boxes if b in after.boxes and before.boxes[b] != after.boxes[b]}
    allowed = set(oiii.blocks) | set(v.blocks)
    assert moved and moved <= allowed, moved - allowed
    assert all(after.boxes[b].x > before.boxes[b].x and after.boxes[b].y == before.boxes[b].y for b in moved)


# ---------------------------------------------------------------------------------------------------- acceptance
def test_a01_double_click_session():
    doc = Doc()
    ok(doc, "add_block", kind="session")                     # the Session alone: nothing else is placed for the user
    r = doc.project.rows[0]
    s, c = r.nights[0]
    assert doc.title(s) == "Session 1" and c is None and kinds(doc, r) == [] and doc.project.wires == []
    assert [b.kind for b in doc.project.blocks.values()] == ["post", "session"]
    doc = session_doc()
    r = doc.project.rows[0]
    assert kinds(doc, r) == ["registration", "selection", "stacking"]
    ix = Index(doc.project)
    assert [(w.src[0], w.dst[0]) for w in doc.project.wires] == [(r.blocks[-1], ix.post)]


def test_a02_drop_files_counts_and_speed():
    doc = session_doc()
    s = doc.project.rows[0].nights[0][0]
    ok(doc, "add_files", session=s, kind="lights", paths=[f"L{i}.fit" for i in range(40)])
    assert doc.summary(s) == "40 lights"
    ok(doc, "add_files", session=s, kind="darks", paths=[f"D{i}.fit" for i in range(12)])
    assert doc.summary(s) == "40 lights · 12 darks"
    assert doc.summary(doc.project.rows[0].nights[0][1]) == "standard"      # settings, see CHANGEMENTS.md
    t = time.perf_counter()
    ok(doc, "add_files", session=s, kind="flats", paths=[f"F{i}.fit" for i in range(1000)])
    assert time.perf_counter() - t < 0.05


def _three():
    doc = session_doc()
    r = doc.project.rows[0]
    drop(doc, where(doc, "night", kind="session", row=r.id)[0])
    return doc


def test_a03_second_night_fans_in():
    doc = _three()
    r = doc.project.rows[0]
    assert len(r.nights) == 2
    bars = [w for w in layout(doc.project, doc.view).wires if w.style == "bar"]
    assert len(bars) == 2 and {w.dst for w in bars} == {r.blocks[0]}


def test_a04_a05_night_to_own_row_and_back():
    doc = _three()
    js3, d3 = doc._json, layout(doc.project, doc.view)
    r1 = doc.project.rows[0]
    s2 = r1.nights[1][0]
    keep = boxes_of(d3, doc, r1.id)
    drop(doc, where(doc, "row", block=s2)[0])
    d4 = layout(doc.project, doc.view)
    assert doc.project.rows[1].name == "Session 2"
    assert all(d4.boxes.get(b) == bx for b, bx in keep.items() if b != s2 and b != r1.nights[1][1]
               if b in d4.boxes)
    drop(doc, where(doc, "night", block=s2, row=r1.id)[0])
    assert len(doc.project.rows) == 1 and len(doc.project.rows[0].nights) == 2
    doc.undo()
    doc.undo()
    assert doc._json == js3 and layout(doc.project, doc.view) == d3


def test_a06_a07_variant_mark_and_parent_still():
    doc = session_doc()
    r = doc.project.rows[0]
    drop(doc, where(doc, "branch", kind="selection", y=of_kind(doc, r, "selection"))[0])
    v = doc.project.rows[1]
    assert v.name == "Session 1 2" and doc.summary(v.blocks[0]).startswith("like the mother")
    d0 = layout(doc.project, doc.view)
    rr = ok(doc, "set_setting", block=of_kind(doc, v, "stacking"), key="Drop size", value=1.0)
    assert not doc.summary(v.blocks[0]).startswith("like the mother")
    d1 = layout(doc.project, doc.view)
    assert boxes_of(d0, doc, r.id) == boxes_of(d1, doc, r.id)
    assert not inv.i4(d0, d1, "set_setting", rr.acted)


def test_a08_a09_variant_in_colour_template():
    doc = Doc.from_template("colour_complete")
    r = doc.project.rows[0]
    drop(doc, where(doc, "branch", kind="color", y=of_kind(doc, r, "color"))[0])
    v = next(x for x in doc.project.rows if x.fork)
    assert kinds(doc, r).count("color") == 1
    before = layout(doc.project, doc.view)
    p = where(doc, "insert", kind="denoise", row=v.id, after=None)[0]
    assert p.lit, p.reason_text()
    drop(doc, p)
    after = layout(doc.project, doc.view)
    moved = {b for b in before.boxes if b in after.boxes and before.boxes[b] != after.boxes[b]
             and before.cells[b][1] < 100}                  # Post and composition stay right of the longest row
    assert moved <= set(v.blocks)


def test_a10_vers_post_on_empilement():
    doc = session_doc()
    ok(doc, "add_block", kind="gradient")
    emp = of_kind(doc, doc.project.rows[0], "stacking")
    ok(doc, "send_to_post", block=emp)
    post = Index(doc.project).post
    assert doc.summary(post) == "2 layers" and len(doc.project.rows) == 1
    assert not inv.static(doc, layout(doc.project, doc.view))


def test_a11_sho_template_grid_and_fold():
    doc = Doc.from_template("sho_rgb")
    assert doc.project.reference == row(doc, "SII").id
    for n in ("Ha", "OIII", "RGB"):
        assert doc.summary(row(doc, n).nights[0][0]).startswith("on the grid of SII")
    assert doc.apply("fold_all").ok
    d = layout(doc.project, doc.view)
    assert len(d.cards) == 4 and not inv.static(doc, d) and d.size[0] <= 1920


def test_a12_reference_moves_with_astrometrie():
    doc = Doc.from_template("sho_rgb")
    astro = of_kind(doc, row(doc, "SII"), "astrometry")
    js = doc._json
    ok(doc, "set_reference", row=row(doc, "RGB").id)
    assert doc.project.reference == row(doc, "RGB").id and astro in row(doc, "RGB").blocks
    doc.undo()
    assert doc._json == js


def test_a13_a14_variant_alternative_at_port():
    doc = Doc.from_template("sho_rgb")
    ha = row(doc, "Ha")
    ok(doc, "add_variant", block=of_kind(doc, ha, "gradient"))
    v = next(x for x in doc.project.rows if x.fork)
    ix = Index(doc.project)
    assert ix.children[ha.id] == [v.id]
    d = layout(doc.project, doc.view)
    jw = [w for w in doc.project.wires if w.dst[1] == "Hα"]
    alt = next(w for w in jw if w.src[0] in v.blocks)
    assert not alt.active and next(p for p in d.wires if p.id == alt.id).style == "port_inactive"
    j = alt.dst[0]
    post = Index(doc.project).post
    layers = [w.id for w in doc.project.wires if w.dst[0] == post]
    ok(doc, "take_alternative", wire=alt.id)
    assert [w.active for w in doc.project.wires if w.dst[1] == "Hα" and w.dst[0] == j].count(True) == 1
    assert next(w for w in doc.project.wires if w.id == alt.id).active
    assert [w.id for w in doc.project.wires if w.dst[0] == post] == layers


def test_a15_a16_change_type_then_fill_port():
    doc = Doc.from_template("sho_rgb")
    j = row(doc, "S-H-O").blocks[0]
    cc = row(doc, "Combine layers").blocks[0]
    ccw = [(w.src, w.dst) for w in doc.project.wires if w.dst[0] == cc]
    r = ok(doc, "set_combination_type", block=j, type="ho")
    assert "S II" in r.toast_text() and {w.dst[1] for w in doc.project.wires if w.dst[0] == j} == {"Hα", "O III"}
    assert [(w.src, w.dst) for w in doc.project.wires if w.dst[0] == cc] == ccw
    ok(doc, "set_combination_type", block=j, type="sho")
    assert doc.greyed(j) == "S II to wire"
    p = where(doc, "port", kind="session", match={"block": j, "port": "S II"})[0]
    r = drop(doc, p)
    new = doc.project.rows[-1]
    assert new.name == "S II" and new.nights[0][1] is None
    # what the graph needs there, nothing more: the S-H-O is Combine layers' Background, a starless image
    assert kinds(doc, new) == ["stacking", "denoise", "stars"] and "starless" in r.toast_text()
    assert any(w.dst == [j, "S II"] and w.src == [new.blocks[-1], "Starless"] for w in doc.project.wires)


def test_a17_delete_reference_row():
    doc = Doc.from_template("sho_rgb")
    astro = of_kind(doc, row(doc, "SII"), "astrometry")
    r = ok(doc, "delete_row", row=row(doc, "SII").id)
    assert doc.project.reference == row(doc, "Ha").id and astro in row(doc, "Ha").blocks
    assert "SII" in r.toast_text()


def test_a18_duplicate_drawn_below():
    doc = Doc.from_template("sho_rgb")
    ha = row(doc, "Ha")
    ok(doc, "duplicate_row", row=ha.id)
    d = layout(doc.project, doc.view)
    lines = [o for o, _y in d.lines]
    cp = row(doc, "Ha copy")
    assert lines.index(cp.id) == lines.index(ha.id) + 1
    g = of_kind(doc, ha, "gradient")
    ok(doc, "set_setting", block=g, key="Surface degree", value=3)
    assert doc.apply("apply_to_rows", block=g, key="Surface degree").ok
    assert doc.effective(of_kind(doc, row(doc, "Ha copy"), "gradient"))["Surface degree"] == 3


def test_a19_couleur_on_image_row_refused():
    doc = Doc.from_template("several_images")
    ps = [p for p in places(doc.project, doc.view, kind="color") if p.kind == "insert"]
    assert ps and all(not p.lit and p.reason_text() == "needs Astrometry first" for p in ps)
    refused_inert(doc, ps[0].op[0], **ps[0].op[1])


def test_a20_a21_switch_off_and_needed():
    doc = Doc.from_template("colour_minimal")
    r = doc.project.rows[0]
    g = of_kind(doc, r, "gradient")
    ok(doc, "switch_off", block=g)
    assert doc.summary(g) == "off: image passed on"
    a = of_kind(doc, r, "astrometry")
    assert refused_inert(doc, "switch_off", block=a) == "needed by Color: switch Color off first"
    assert refused_inert(doc, "remove_block", block=a) == "needed by Color"


def test_a24_debruitage_needs_half_stacks():
    doc = Doc.from_template("several_images")
    assert len([w for w in doc.project.wires]) == 3
    ps = [p for p in places(doc.project, doc.view, kind="denoise") if p.kind == "insert"]
    assert all(p.reason_text() == "needs half-stacks (from frames)" for p in ps)


def test_a25_a26_combinaison_unwired_then_loop():
    doc = Doc.from_template("filters_complete")
    p = where(doc, "comp", kind="combine")[0]
    drop(doc, p)
    j = doc.project.rows[-1].blocks[0]
    assert doc.greyed(j) == "S II, Hα, O III to wire"
    ha = row(doc, "Ha")
    ok(doc, "wire", src=ha.blocks[-1], src_port="Starless", dst=j, dst_port="Hα")
    sho = row(doc, "S-H-O").blocks[0]
    cc = row(doc, "Combine layers").blocks[0]
    assert refused_inert(doc, "wire", src=cc, src_port="image", dst=sho, dst_port="S II") == \
        "this wire would close a loop"


def test_a27_fold_band_not_undone_rows_keep_order():
    doc = Doc.from_template("sho_rgb")
    for r in doc.project.rows:
        if r.nights:
            ok(doc, "add_night", row=r.id)
    n = len(doc._undo)
    d0 = layout(doc.project, doc.view)
    res = doc.apply("fold_all")
    d1 = layout(doc.project, doc.view)
    assert res.ok and not inv.static(doc, d1) and not inv.i4(d0, d1, "fold_all", res.acted)
    stacks = [r.id for r in doc.project.rows if r.nights]
    assert set(d1.cards) == set(stacks) and all(c.w == 2 * 120 + 16 for c in d1.cards.values())
    assert [o for o, _y in d1.lines] == [r.id for r in doc.project.rows if not r.blocks
                                         or doc.project.blocks[r.blocks[0]].kind not in ("combine",
                                                                                          "combine_layers")]
    band = d1.bands[0][1]
    assert band.right < d0.bands[0][1].right                      # the band shrinks to the card width
    assert all(b.right <= band.right for b in d1.cards.values())
    for r in stacks:                                              # name and badge readable folded
        assert doc.title(r) and doc.summary(r).startswith("2 sessions")
    assert doc.row_badge(row(doc, "SII").id) == "reference" and doc.row_badge(row(doc, "Ha").id).startswith("on the grid")
    doc.undo()
    assert doc.view["folded"] is True                             # folding is not undone
    doc.redo()
    res = doc.apply("unfold_all")
    assert res.ok and layout(doc.project, doc.view) == d0 and len(doc._undo) == n


def test_a28_merge_then_split():
    doc = _three()
    r = doc.project.rows[0]
    (s1, _c1), (s2, _c2) = r.nights
    ok(doc, "add_files", session=s1, kind="lights", paths=[f"A{i}" for i in range(30)])
    ok(doc, "add_files", session=s2, kind="lights", paths=[f"B{i}" for i in range(20)])
    j0 = doc._json
    ok(doc, "merge_nights", a=s1, b=s2)
    assert doc.summary(s1) == "50 lights" and len(doc.project.rows[0].nights) == 1
    j1 = doc._json
    ok(doc, "split_night", night=s1, lights=[f"B{i}" for i in range(20)])
    s3 = doc.project.rows[0].nights[1][0]
    assert doc.summary(s1) == "30 lights" and doc.summary(s3) == "20 lights"
    doc.undo()
    assert doc._json == j1
    doc.undo()
    assert doc._json == j0


def test_a29_save_reopen_identical():
    for key in ("sho_rgb", "colour_complete"):
        doc = Doc.from_template(key)
        ok(doc, "add_variant", block=doc.project.rows[0].blocks[3])
        assert doc.apply("fold_all").ok
        text = doc.dumps()
        doc2 = Doc.loads(text)
        assert doc2.dumps() == text
        assert layout(doc2.project, doc2.view) == layout(doc.project, doc.view)


# ---------------------------------------------------------------------------------------------------- table contract
def test_common_files_follow_and_own():
    doc = _three()
    (s1, _), (s2, _) = doc.project.rows[0].nights
    ok(doc, "set_common_files", kind="darks", paths=["D1", "D2"])
    assert doc.files_of(s1, "darks") == ("common", ["D1", "D2"])
    ok(doc, "add_files", session=s2, kind="darks", paths=["X"], mode="replace")
    assert doc.followers("darks") == ([s1], [s2])
    ok(doc, "add_files", session=s2, kind="darks", paths=["D1", "D2"], mode="replace")
    assert doc.followers("darks") == ([s1, s2], [])           # equal to Common: follows again
    ok(doc, "add_files", session=s1, kind="flats", paths=["F1"])
    ok(doc, "move_files", src_session=s1, src_kind="flats", dst_session=COMMON, dst_kind="flats")
    assert doc.project.common["files"]["flats"] == ["F1"] and doc.files_of(s1, "flats")[0] == "common"
    assert not doc.apply("set_common_files", kind="lights", paths=["L"]).ok


def test_common_settings_and_apply_many():
    doc = Doc.from_template("filters_complete")
    ok(doc, "set_common_setting", kind="stacking", key="Rejection threshold", value="5 σ")
    e = of_kind(doc, row(doc, "Ha"), "stacking")
    assert doc.effective(e)["Rejection threshold"] == "5 σ" and doc.own_keys(e) == []
    ok(doc, "set_setting", block=e, key="Rejection threshold", value="8 σ")
    assert doc.own_keys(e) == ["Rejection threshold"] and doc.differs(e)
    n = len(doc._undo)
    r = doc.apply_many([("add_night", {"row": row(doc, "Ha").id}), ("reset_own", {"block": e})])
    assert r.ok and len(doc._undo) == n + 1 and doc.own_keys(e) == []
    assert not doc.apply_many([("add_night", {"row": row(doc, "Ha").id}), ("delete_row", {"row": "r999"})]).ok
    assert len(doc._undo) == n + 1


def test_split_by_and_merge_rows():
    doc = session_doc()
    s = doc.project.rows[0].nights[0][0]
    ok(doc, "add_files", session=s, kind="lights", paths=[f"L{i}" for i in range(10)] + ["D1", "D2", "F1"])
    ok(doc, "split_by", session=s, groups=[{"into": "common", "files": {"darks": ["D1", "D2"]}},
                                          {"into": "here", "files": {"flats": ["F1"]}},
                                          {"into": "night", "files": {"lights": ["L5", "L6"]}},
                                          {"into": "row", "files": {"lights": ["L7"]}, "name": "Ha"}])
    assert doc.project.common["files"]["darks"] == ["D1", "D2"]
    assert len(doc.project.rows[0].nights) == 2 and doc.project.rows[1].name == "Ha"
    assert doc.summary(s) == "7 lights · 2 darks · 1 flat"
    ok(doc, "merge_rows", src=doc.project.rows[1].id, dst=doc.project.rows[0].id)
    assert len(doc.project.rows) == 1 and len(doc.project.rows[0].nights) == 3


if __name__ == "__main__":
    names = [n for n in dict(globals()) if n.startswith("test_")]
    bad = 0
    for n in names:
        try:
            globals()[n]()
            print("ok  ", n)
        except Exception as e:
            bad += 1
            import traceback
            print("FAIL", n, repr(e))
            traceback.print_exc(limit=4)
    print(f"{len(names) - bad}/{len(names)} passed")
