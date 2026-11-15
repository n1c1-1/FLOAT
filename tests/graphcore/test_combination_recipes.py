"""The Combination block reads the engine's recipes (graphcore/combos.py, stages/recipes.py): every recipe of the
application's folder is placed, titled, given its inputs as the recipe says and compiled to the source the engine's
own placing gives (api.combination_source); the project file is the graph, its copy of the recipe in the block, so a
save and a reopening change nothing; a project saved before recipes keeps its compiled definition; a newer recipe is
applied only by the user's gesture (update_recipe); a recipe's parameter is set on the block."""
import json
import os
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from fixtures import Doc  # noqa: E402
from test_points import ok, row, refused_inert  # noqa: E402
from test_engine_rules import compiled, engine_ok  # noqa: E402
from astro_gpu import api  # noqa: E402
from astro_gpu.graphcore import api as g  # noqa: E402
from astro_gpu.graphcore import combos  # noqa: E402
from astro_gpu.graphcore import model as gm  # noqa: E402
from astro_gpu.graphcore.convert import convert  # noqa: E402
from astro_gpu.graphcore.operations import apply  # noqa: E402
from astro_gpu.stages import recipes  # noqa: E402

LISTED = [r["id"] for r in api.combination_recipes()["recipes"]]


def _combination(doc, name):
    return doc.project.blocks[row(doc, name).blocks[0]]


def _only_combination(doc):
    """The fixture's one combination (its row keeps the name it was placed with)."""
    return next(b for b in doc.project.blocks.values() if b.kind == "combine")


def _row_of(doc, b):
    return next(r for r in doc.project.rows if r.blocks and r.blocks[0] == b.id)


def _wf(p, name):
    return next(w for w in p.workflows if w.name == name)


def test_the_list_is_the_folders_and_holds_the_new_recipes():
    assert combos.offered() == LISTED
    assert {"sho", "ho", "rgb", "lrgb", "hoo", "hargb"} <= set(LISTED)


@pytest.mark.parametrize("rid", LISTED)
def test_each_recipe_is_placed_titled_and_compiled_as_the_engine_places_it(rid):
    spec = api.combination_recipe(rid)
    p, r = apply(g.blank_project(), "add_combination", type=rid)
    assert r.ok, r.reason
    b = next(x for x in p.blocks.values() if x.kind == "combine")
    assert b.source == {"recipe": spec}                                    # its recipe copied in
    assert gm.in_ports(b) == [i["label"] for i in spec["inputs"]]           # its inputs: the recipe's
    assert combos.inputs(b) == [{k: i[k] for k in ("role", "label", "type", "channel")} for i in spec["inputs"]]
    title = g.title(p, b.id)
    assert title == spec["name"].split(" (")[0] and len(title.split()) <= 2     # a name of two words at most
    assert p.rows[-1].name == title
    q = __import__("astro_gpu.core.project2", fromlist=["Project2"]).Project2(name="t")
    from astro_gpu.graphcore.compile import compile_into
    compile_into(p, q)
    assert q.workflows[0].source == api.combination_source(rid, [])
    assert combos.stage(b) == recipes.stage_for(q.workflows[0].source["fusion"])


@pytest.mark.parametrize("rid", ["hoo", "hargb"])
def test_hoo_and_hargb_are_wired_saved_reopened_and_compiled(rid):
    doc = Doc.from_template("colour_minimal")
    ok(doc, "add_combination", type=rid, name="Mine")
    j = _combination(doc, "Mine")
    src = row(doc, "Session 1").blocks[-1]
    for port in gm.in_ports(j):
        ok(doc, "wire", src=src, src_port="image", dst=j.id, dst_port=port)
    p, _c = compiled(doc)
    wid = _wf(p, "Session 1").id
    ins = [{"pid": wid, "role": i["role"], "channel": i["channel"]} for i in api.combination_recipe(rid)["inputs"]]
    assert _wf(p, "Mine").source == api.combination_source(rid, ins)
    assert _wf(p, "Mine").source["fusion"]["type"] == rid
    engine_ok(doc)
    # the project file is the graph: written and read back, the same graph, the same definition
    again = gm.from_dict(json.loads(json.dumps(gm.to_dict(doc.project))))
    assert gm.model_json(again) == gm.model_json(doc.project)
    from astro_gpu.core.project2 import Project2
    from astro_gpu.graphcore.compile import compile_into
    q = Project2(name="t")
    compile_into(again, q)
    assert q.to_dict()["workflows"] == p.to_dict()["workflows"]


def test_a_hoo_input_reads_the_channel_its_recipe_gives_unless_the_wire_says_another():
    doc = Doc.from_template("colour_minimal")
    ok(doc, "add_combination", type="hoo", name="Dual")
    j = _combination(doc, "Dual")
    src = row(doc, "Session 1").blocks[-1]
    ok(doc, "wire", src=src, src_port="image", dst=j.id, dst_port="Hα")
    ok(doc, "wire", src=src, src_port="image", dst=j.id, dst_port="O III", channel="L")
    p, _c = compiled(doc)
    f = _wf(p, "Dual").source["fusion"]
    got = {i["role"]: i.get("channel", recipes.default_channel(f["recipe"], i["role"])) for i in f["inputs"]}
    assert got == {"Ha": "R", "OIII": "L"}


def test_a_project_saved_before_recipes_keeps_its_definition():
    """A combination with no copy (the four types before recipes): converted, drawn with its type's inputs, compiled
    back without a copy (its key is the one it always had)."""
    doc = Doc.from_template("filters_minimal")
    p, _c = compiled(doc)
    d = p.to_dict()
    for w in d["workflows"]:
        f = (w.get("source") or {}).get("fusion")
        if f:
            f.pop("recipe", None)                                      # as saved before recipes
    from astro_gpu.core.project2 import Project2
    from astro_gpu.graphcore.compile import compile_into
    old = Project2.from_dict(d)
    gr = convert(old)
    j = next(b for b in gr.blocks.values() if b.kind == "combine")
    assert j.source is None and gm.in_ports(j) == ["Hα", "O III"]
    q = Project2(name="t")
    compile_into(gr, q)
    f = next(w for w in q.workflows if w.source and "fusion" in w.source).source["fusion"]
    assert "recipe" not in f and {i["role"] for i in f["inputs"]} == {"Ha", "OIII"}


def test_a_converted_combination_keeps_its_copy():
    doc = Doc.from_template("colour_minimal")
    ok(doc, "add_combination", type="hoo", name="Dual")
    p, _c = compiled(doc)
    gr = convert(p)
    j = next(b for b in gr.blocks.values() if b.kind == "combine")
    assert j.source == {"recipe": api.combination_recipe("hoo")} and combos.stage(j) == recipes.RECIPE_STAGE


def test_another_type_takes_its_recipe_and_its_inputs():
    doc = Doc.from_template("filters_minimal")
    j = _only_combination(doc).id
    ok(doc, "set_combination_type", block=j, type="hoo")
    b = doc.project.blocks[j]
    assert b.source == {"recipe": api.combination_recipe("hoo")} and gm.in_ports(b) == ["Hα", "O III"]
    assert combos.stage(b) == recipes.RECIPE_STAGE
    ok(doc, "set_combination_type", block=j, type="hargb")
    assert gm.in_ports(doc.project.blocks[j]) == ["R", "G", "B", "Hα"]
    assert {w.dst[1] for w in doc.project.wires if w.dst[0] == j} <= {"Hα"}    # O III had no place: its wire went
    assert refused_inert(doc, "set_combination_type", block=j, type="nothing") == "unknown type"


def test_the_stars_twin_runs_the_same_recipe():
    doc = Doc.from_template("filters_complete")
    j = row(doc, "S-H-O").blocks[0]
    ok(doc, "set_combination_type", block=j, type="hoo")              # S II has no place: its wire goes
    assert {w.dst[1] for w in doc.project.wires if w.dst[0] == j} == {"Hα", "O III"}
    ok(doc, "add_stars_combination", block=j)
    twin = _combination(doc, "S-H-O · Stars")
    assert twin.source == doc.project.blocks[j].source and twin.settings == doc.project.blocks[j].settings
    engine_ok(doc)


# ---------------------------------------------------------------------------------------------- a newer recipe
@pytest.fixture()
def folder(tmp_path, monkeypatch):
    """The application's recipes copied to a folder of the test's own, which the graph reads."""
    d = tmp_path / "combinations"
    shutil.copytree(recipes.FOLDER, d)
    monkeypatch.setattr(combos, "FOLDER", d)
    combos.refresh()
    yield d
    monkeypatch.setattr(combos, "FOLDER", None)
    combos.refresh()


def _newer(folder: Path, rid: str, old: str, new: str):
    f = folder / f"{rid}.toml"
    t = f.read_text(encoding="utf-8")
    assert t.count("version = 1") == 1 and t.count(old) == 1
    f.write_text(t.replace("version = 1", "version = 2").replace(old, new), encoding="utf-8")
    combos.refresh()


def test_a_newer_recipe_is_said_and_applied_only_by_the_users_gesture(folder):
    doc = Doc.from_template("colour_minimal")
    ok(doc, "add_combination", type="hoo", name="Dual")
    j = _combination(doc, "Dual").id
    src = row(doc, "Session 1").blocks[-1]
    for port in ("Hα", "O III"):
        ok(doc, "wire", src=src, src_port="image", dst=j, dst_port=port)
    assert refused_inert(doc, "update_recipe", block=j) == "the recipe is up to date"
    _newer(folder, "hoo", 'label = "O III"', 'label = "OIII line"')
    st = combos.status(doc.project.blocks[j])
    assert st["newer"] and st["project"] == 1 and st["folder"] == 2
    p, _c = compiled(doc)
    assert _wf(p, "Dual").source["fusion"]["recipe"]["version"] == 1            # nothing changes by itself
    ok(doc, "update_recipe", block=j)
    b = doc.project.blocks[j]
    assert b.source["recipe"]["version"] == 2 and not combos.status(b)["newer"]
    assert sorted(w.dst[1] for w in doc.project.wires if w.dst[0] == j) == ["Hα", "OIII line"]   # kept by role
    p, _c = compiled(doc)
    assert {i["role"] for i in _wf(p, "Dual").source["fusion"]["inputs"]} == {"Ha", "OIII"}
    engine_ok(doc)
    assert doc.undo().ok and doc.project.blocks[j].source["recipe"]["version"] == 1


def test_a_frozen_type_changed_runs_on_the_recipe_stage_and_its_parameters_are_the_blocks(folder):
    doc = Doc.from_template("filters_minimal")
    jb = _only_combination(doc)
    j, name = jb.id, _row_of(doc, jb).name
    ok(doc, "set_param", block=j, stage="combine_ho", name="hue_oiii", value=170.0)
    _newer(folder, "ho", "max = 50", "max = 40")
    ok(doc, "update_recipe", block=j)
    b = doc.project.blocks[j]
    assert combos.stage(b) == recipes.RECIPE_STAGE
    assert b.params == {recipes.RECIPE_STAGE: {"values": {"hue_oiii": 170.0}}}   # the setting kept, by its name
    ok(doc, "set_recipe_value", block=j, name="radius", value=12.0)
    assert doc.project.blocks[j].params[recipes.RECIPE_STAGE]["values"] == {"hue_oiii": 170.0, "radius": 12.0}
    assert refused_inert(doc, "set_recipe_value", block=j, name="radius", value=45.0)      # above its max
    ok(doc, "set_recipe_value", block=j, name="hue_oiii", value=None)
    assert doc.project.blocks[j].params[recipes.RECIPE_STAGE]["values"] == {"radius": 12.0}
    p, _c = compiled(doc)
    w = _wf(p, name)
    assert w.params[recipes.RECIPE_STAGE]["values"] == {"radius": 12.0}
    engine_ok(doc)
