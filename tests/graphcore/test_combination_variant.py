"""Try variant from a combination (the combination editor's draft bar): a second combination of the same filters, the
same inputs wired the same way, the change on the copy, its own workflow and its own layer in Post; the original
untouched; one Undo step. The other joins keep their refusal: a Combine layers is a layer of Post, not a workflow."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from fixtures import Doc  # noqa: E402
from test_engine_rules import compiled, engine_ok  # noqa: E402
from test_points import ok, refused_inert  # noqa: E402

from astro_gpu.graphcore.model import Index  # noqa: E402


def _inputs(doc, j):
    return sorted((w.src[0], w.src[1], w.dst[1], w.channel, w.image, w.active) for w in doc.project.wires
                  if w.dst[0] == j)


def test_a_combination_s_variant_is_a_second_combination_of_the_same_filters():
    doc = Doc.from_template("filters_minimal")
    j = next(r.blocks[0] for r in doc.project.rows if r.blocks and doc.project.blocks[r.blocks[0]].kind == "combine")
    stage = next(iter(doc.project.blocks[j].params), "combine_ho")
    before, n = doc._json, len(doc._undo)
    params0 = dict(doc.project.blocks[j].params)
    rows0 = len(doc.project.rows)
    r = ok(doc, "try_variant", block=j, change={"param": [stage, "hue_oiii", 200.0]})
    assert len(doc.project.rows) == rows0 + 1 and len(doc._undo) == n + 1
    v = r.focus
    assert v != j and doc.project.blocks[v].kind == "combine"
    assert _inputs(doc, v) == _inputs(doc, j)                            # the same filters, read the same way
    assert doc.project.blocks[v].params[stage]["hue_oiii"] == 200.0
    assert doc.project.blocks[j].params == params0                       # the original untouched
    ix = Index(doc.project)
    assert any(w.src[0] == v and w.dst[0] == ix.post for w in doc.project.wires)     # its own layer of Post
    vr = ix.row[ix.row_of[v]]
    assert vr.trial is None and "variant" in vr.name
    engine_ok(doc)
    p, _c = compiled(doc)
    fus = [w for w in p.workflows if (w.source or {}).get("fusion")]
    assert len(fus) == 2
    assert [(i["pid"], i["role"]) for i in fus[0].source["fusion"]["inputs"]] == \
        [(i["pid"], i["role"]) for i in fus[1].source["fusion"]["inputs"]]
    doc.undo()
    assert doc._json == before


def test_a_combine_layers_still_has_no_variant():
    doc = Doc.from_template("colour_complete")
    cl = next(b for b, x in doc.project.blocks.items() if x.kind == "combine_layers")
    assert "no variant from here" in refused_inert(doc, "try_variant", block=cl, change={"off": False})


def test_a_recipe_round_trip_keeps_the_settings_both_recipes_have_and_gives_each_its_own_back():
    """H-O with Balance On sky, O III hue 240 and O III weight 0.5, then S-H-O, then H-O again: the balance and the
    weight go with each change, the H-O's hues come back with it (the reviewer's round trip lost them)."""
    doc = Doc.from_template("filters_minimal")
    j = next(r.blocks[0] for r in doc.project.rows if r.blocks and doc.project.blocks[r.blocks[0]].kind == "combine")
    ok(doc, "set_param", block=j, stage="combine_ho", name="balance", value="sky")
    ok(doc, "set_param", block=j, stage="combine_ho", name="hue_oiii", value=240.0)
    ok(doc, "set_input", block=j, port="O III", weight=0.5)
    ok(doc, "set_combination_type", block=j, type="sho")
    sho = doc.project.blocks[j].params["combine_sho"]
    assert sho["balance"] == "sky" and sho["weights"] == {"OIII": 0.5} and "hue_oiii" not in sho
    ok(doc, "set_combination_type", block=j, type="ho")
    ho = doc.project.blocks[j].params["combine_ho"]
    assert ho["balance"] == "sky" and ho["hue_oiii"] == 240.0 and ho["weights"] == {"OIII": 0.5}
    assert "combine_sho" not in doc.project.blocks[j].params          # nothing of the other recipe is computed
    engine_ok(doc)
