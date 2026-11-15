"""An image block whose node name is given again by the compile (moved into a chain that already holds its name: the
second Gradient becomes « gradient~2 ») takes its settings with it: they stayed under its old name, so they were
applied to the first Gradient of the chain and the second one ran with its defaults."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from fixtures import Doc  # noqa: E402
from test_points import ok  # noqa: E402
from test_engine_rules import compiled, engine_ok  # noqa: E402
from astro_gpu.graphcore.operations import owned_stages  # noqa: E402


def _gradient(doc, name):
    r = next(r for r in doc.project.rows if r.name == name)
    return r, next(b for b in r.blocks if doc.project.blocks[b].kind == "gradient")


def test_a_gradient_moved_beside_another_keeps_its_own_settings():
    doc = Doc.from_template("filters_minimal")
    ha, gh = _gradient(doc, "Ha")
    _o3, go = _gradient(doc, "OIII")
    ok(doc, "set_param", block=gh, stage=owned_stages(doc.project, gh)[0], name="degree", value=1)
    ok(doc, "set_param", block=go, stage=owned_stages(doc.project, go)[0], name="degree", value=3)
    ok(doc, "move_block", block=go, at={"type": "insert", "row": ha.id, "after": gh})
    p, _c = compiled(doc)
    w = next(w for w in p.workflows if w.name == "Ha")
    assert w.params["gradient"] == {"degree": 1}                  # the first Gradient's own (it was the second's)
    assert w.params["gradient~2"] == {"degree": 3}                # the moved one's, under its new name
    assert owned_stages(doc.project, go) == ["gradient~2"]
    assert doc.project.blocks[go].params == {"gradient~2": {"degree": 3}}       # what its panel reads
    engine_ok(doc)
