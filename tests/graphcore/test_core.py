"""Unit tests of the graph core, in the order of the work: model, rules, operations, layout, fixtures, requests."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import invariants as inv  # noqa: E402
from fixtures import Doc, templates  # noqa: E402
from astro_gpu import graphcore  # noqa: E402
from astro_gpu.graphcore import apply, apply_many, layout, new_view, say, M  # noqa: E402
from astro_gpu.graphcore.model import check, Invalid, clone, Wire, Index, to_dict, from_dict, model_json  # noqa: E402
from astro_gpu.graphcore.rules import evaluate  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CORE = os.path.join(HERE, "astro_gpu", "graphcore")


def test_no_qt_in_core():
    for f in os.listdir(CORE):
        if f.endswith(".py"):
            src = open(os.path.join(CORE, f), encoding="utf-8").read()
            assert "PySide" not in src and "astro_gpu.client" not in src, f


def test_no_accented_letter_in_core():
    """No French (or any accented) literal in the core: English source strings only (the French is in the locale)."""
    for f in os.listdir(CORE):
        if f.endswith(".py"):
            src = open(os.path.join(CORE, f), encoding="utf-8").read()
            bad = [c for c in src if c in "éèêëàâôùûçîïÉÈÀÊ«»œ"]
            assert not bad, (f, bad[:5])


def test_api_names():
    for n in ("apply", "apply_many", "layout", "places", "new_view", "title", "summary", "greyed", "wire_tip",
              "palette_state", "stacks", "differs", "effective", "own_keys", "followers", "files_of", "warnings",
              "COMMON", "project_state", "row_badge", "wire_check"):
        assert hasattr(graphcore, n), n
    for m in ("apply", "apply_many", "undo", "redo", "drop", "from_template"):
        assert hasattr(Doc, m), m


# ---------------------------------------------------------------------------------------------------- 1 model
def test_every_fixture_checks_and_round_trips_through_dicts():
    for key, _t, _d in templates():
        doc = Doc.from_template(key)
        check(doc.project)
        d = to_dict(doc.project)
        text = json.dumps(d, ensure_ascii=False, indent=1)
        back = from_dict(json.loads(text))
        check(back)
        assert model_json(back) == model_json(doc.project)
        assert json.dumps(to_dict(back), ensure_ascii=False, indent=1) == text
        h = Doc.loads(doc.dumps())
        assert h.json == doc.json


def test_kinds_are_the_engine_block_names():
    doc = Doc.from_template("sho_rgb")
    kinds = {b.kind for b in doc.project.blocks.values()}
    assert kinds == {"session", "calibration", "registration", "selection", "stacking", "gradient", "denoise",
                     "astrometry", "color", "stars", "combine", "combine_layers", "post"}, kinds
    types = {b.settings.get("type", "sho") for b in doc.project.blocks.values() if b.kind == "combine"}
    assert types == {"sho"}


def test_check_catches_broken_models():
    doc = Doc.from_template("sho_rgb")
    p = doc.project
    breakers = [
        lambda q: q.rows[0].blocks.append(q.rows[1].blocks[0]),            # block in two rows
        lambda q: q.rows[0].blocks.append("b999"),                          # dangling id
        lambda q: setattr(q, "reference", None),
        lambda q: q.wires.append(Wire("w999", [q.rows[0].blocks[-1], "Starless"], [q.rows[3].blocks[0], "Hα"])),
        lambda q: setattr(q.rows[1], "name", q.rows[0].name),
        lambda q: q.rows[0].blocks.remove(q.rows[0].blocks[0]),             # frames row without Registration
        lambda q: setattr(q, "next_id", 3),
    ]
    for brk in breakers:
        q = clone(p)
        brk(q)
        try:
            check(q)
        except Invalid:
            continue
        raise AssertionError("not caught")


# ---------------------------------------------------------------------------------------------------- 2 rules
def test_titles_numbers_and_reasons():
    doc = Doc.from_template("colour_minimal")
    r = doc.project.rows[0]
    ok = doc.apply("add_block", kind="gradient", at={"type": "insert", "row": r.id, "after": r.blocks[-1]})
    assert ok.ok and doc.title(ok.focus) == "Gradient 2"
    grads = [b for b in r.blocks if doc.project.blocks[b].kind == "gradient"]
    assert doc.title(grads[0]) == "Gradient"
    img = Doc.from_template("several_images")
    row = img.project.rows[0]
    res = img.apply("add_block", kind="denoise", at={"type": "insert", "row": row.id, "after": row.blocks[0]})
    assert res.reason_text() == "the half stacks are needed (start from the lights)"
    res = img.apply("add_block", kind="selection", at={"type": "insert", "row": row.id, "after": row.blocks[0]})
    assert res.reason_text() == "Selection comes before the Stacking"
    f = Doc.from_template("filters_minimal")
    o3 = f.project.rows[1]
    res = f.apply("add_block", kind="astrometry", at={"type": "insert", "row": o3.id, "after": o3.blocks[2]})
    assert res.reason_text() == "the grid of “Ha” gives the solution"


def test_greyed_ports_and_downstream():
    doc = Doc.from_template("filters_minimal")
    j = doc.project.rows[2].blocks[0]
    w = next(x for x in doc.project.wires if x.dst[0] == j)
    assert doc.apply("add_block", kind="gradient", at={"type": "insert", "row": doc.project.rows[2].id,
                                                       "after": j}).ok
    assert doc.apply("unwire", wire=w.id).ok
    assert doc.greyed(j) == "ports to wire"
    g = doc.project.rows[2].blocks[1]
    assert doc.greyed(g) == "downstream of H-O" and doc.summary(g) == "downstream of H-O"
    assert "No image" in doc.wire_tip(f"c:{j}>{g}")


def test_summaries():
    doc = Doc.from_template("sho_rgb")
    blocks = doc.project.blocks
    texts = {blocks[b].kind: doc.summary(b) for b in blocks}
    assert texts["selection"] == "all · weight ≥ 5 %" and texts["stacking"] == "drizzle auto · rejection at 3 σ"
    assert texts["combine"] == "S II, Hα, O III" and texts["combine_layers"] == "stars in Screen"
    assert texts["post"] == "1 layer" and texts["astrometry"] == "solves the field · Gaia DR3"


def test_messages_are_written_by_the_translation_given():
    def tr(s):
        return {"Gradient": "Dégradé", "downstream of {t}": "en aval de {t}"}.get(s, s)
    doc = Doc.from_template("filters_minimal")
    doc.tr = tr
    j = doc.project.rows[2].blocks[0]
    w = next(x for x in doc.project.wires if x.dst[0] == j)
    doc.apply("add_block", kind="gradient", at={"type": "insert", "row": doc.project.rows[2].id, "after": j})
    doc.apply("unwire", wire=w.id)
    g = doc.project.rows[2].blocks[1]
    assert doc.greyed(g) == "en aval de H-O"
    assert say(M("{a} and {_b}", a="Gradient", _b="Gradient"), tr) == "Dégradé and Gradient"


def test_default_names_follow_the_language():
    table = {"Session {n}": "Nuit {n}", "{_name} copy": "{_name} copie"}

    def tr(s):
        return table.get(s, s)
    g = graphcore.blank_project()
    g, r = apply(g, "add_row", kind="session", tr=tr)
    assert g.rows[0].name == "Nuit 1", g.rows[0].name
    g, r = apply(g, "duplicate_row", row=g.rows[0].id, tr=tr)
    assert g.rows[1].name == "Nuit 1 copie"


# ---------------------------------------------------------------------------------------------------- 3 operations
def test_apply_is_pure_and_refusal_is_inert():
    g = Doc.from_template("colour_minimal").project
    js = model_json(g)
    for op, kw in [("remove_block", {"block": Index(g).post}), ("delete_row", {"row": "r999"}),
                   ("rename_row", {"row": g.rows[0].id, "name": " "}), ("frobnicate", {})]:
        g2, r = apply(g, op, **kw)
        assert not r.ok and r.reason and g2 is g and model_json(g) == js
    g2, r = apply(g, "add_night", row=g.rows[0].id)
    assert r.ok and g2 is not g and model_json(g) == js and len(g2.rows[0].nights) == 2
    g3, r = apply(g, "rename_row", row=g.rows[0].id, name=g.rows[0].name)
    assert r.ok and g3 is g                                    # no change: the same graph comes back


def test_undo_redo_of_the_history_helper_is_exact():
    doc = Doc.from_template("colour_minimal")
    js, n = doc._json, len(doc._undo)
    doc.apply("add_night", row=doc.project.rows[0].id)
    after = doc._json
    assert doc.undo().toast_text().startswith("Undone: ") and doc._json == js
    assert doc.redo().toast_text().startswith("Redone: ") and doc._json == after
    r = doc.apply("rename_row", row=doc.project.rows[0].id, name=doc.project.rows[0].name)
    assert r.ok and len(doc._undo) == n + 1                   # no change, no step


def test_no_variant_forked_inside_a_folded_stack():
    doc = Doc.from_template("colour_minimal")
    r = doc.project.rows[0]
    assert doc.apply("fold_all").ok
    stk = r.blocks[2]
    js = doc._json
    res = doc.apply("add_variant", block=stk)
    assert not res.ok and res.reason_text() == "Preprocessing is folded: unfold it first" and doc._json == js
    assert doc.apply("unfold_all").ok and doc.apply("add_variant", block=stk).ok
    res = doc.apply("fold_all")
    assert not res.ok and "the preprocessing stays unfolded" in res.reason_text()


def test_view_ops_not_undone():
    doc = Doc.from_template("sho_rgb")
    n = len(doc._undo)
    assert doc.apply("fold_all").ok
    assert doc.apply("set_mode", mode="table").ok
    assert len(doc._undo) == n and doc.view["mode"] == "table"
    g, r = apply(doc.project, "select", id="b1", view=doc.view)
    assert g is doc.project and doc.view["selected"] == "b1"


# ---------------------------------------------------------------------------------------------------- 4 layout
def test_layout_invariants_on_fixtures_and_folds():
    for key, _t, _d in templates():
        doc = Doc.from_template(key)
        d = layout(doc.project, doc.view)
        assert not inv.static(doc, d), (key, inv.static(doc, d))
        doc.apply("fold_all")
        d = layout(doc.project, doc.view)
        assert not inv.static(doc, d), (key, inv.static(doc, d))
        assert layout(doc.project, doc.view) == d


def test_places_lit_match_apply():
    doc = Doc.from_template("sho_rgb")
    for kind in ("gradient", "denoise", "color", "session", "combine_layers", "selection"):
        for p in graphcore.places(doc.project, doc.view, kind=kind):
            g2, r = apply(clone(doc.project), p.op[0], view=dict(doc.view), **p.op[1])
            assert r.ok == p.lit, (kind, p.kind, p.reason_text(), r.reason_text())
            if r.ok:
                d = layout(g2, doc.view)
                assert not inv.i5(d, r.focus, p, g2), (kind, p.kind, p.op)


# ---------------------------------------------------------------------------------------------------- 5 fixtures
def test_fixtures_have_no_greyed_block():
    for key, _t, _d in templates():
        doc = Doc.from_template(key)
        bad = {b: e.reason for b, e in evaluate(doc.project).items() if e.cat}
        assert not bad, (key, bad)
        assert not doc._undo


# ---------------------------------------------------------------------------------------------------- requests
def test_view_and_table_requests():
    doc = Doc.from_template("filters_minimal")
    ha, o3, j = doc.project.rows[0], doc.project.rows[1], doc.project.rows[2].blocks[0]
    assert doc.wire_check(o3.blocks[-1], "image", j, "Hα") == ""
    assert doc.wire_check(j, "image", j, "Hα") == "this wire would close a loop"
    assert doc.project_state()[0] == "partial"
    assert doc.title(ha.id) == "Ha" and doc.summary(ha.id) == "1 session · 0 lights · drizzle auto"
    d = layout(doc.project, doc.view)
    assert set(d.extras) == {"project", "new_workflow", "fold_all"}
    r = doc.apply("add_row", kind="session")
    assert r.created["row"] == doc.project.rows[-1].id and r.created["session"] == r.focus
    doc.undo()
    n = len(doc._undo)
    r = doc.apply_many([("add_night", {"row": ha.id}),
                        ("add_files", lambda outs: {"session": outs[-1].created["session"], "kind": "lights", "paths": ["a"]})])
    assert r.ok and len(doc._undo) == n + 1
    assert doc.summary(doc.project.rows[0].nights[-1][0]) == "1 light"
    s1, s2 = [x[0] for x in doc.project.rows[0].nights]
    assert doc.apply("move_night", night=s2, row=ha.id, before=s1).ok
    assert [x[0] for x in doc.project.rows[0].nights] == [s2, s1]
    assert doc.apply("add_row", kind="session", name="Ha", files={"lights": ["x", "y"]}).ok
    new = doc.project.rows[-1]
    assert new.name == "Ha 2" and doc.summary(new.nights[0][0]).endswith("2 lights")


def test_apply_many_is_all_or_nothing():
    g = Doc.from_template("colour_minimal").project
    js = model_json(g)
    g2, r = apply_many(g, [("add_night", {"row": g.rows[0].id}), ("frobnicate", {})])
    assert not r.ok and g2 is g and model_json(g) == js
    g2, r = apply_many(g, [("add_night", {"row": g.rows[0].id}), ("add_night", {"row": g.rows[0].id})], new_view())
    assert r.ok and len(g2.rows[0].nights) == 3 and " · " in r.toast_text()


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
            traceback.print_exc(limit=-3)
    print(f"{len(names) - bad}/{len(names)} passed")
