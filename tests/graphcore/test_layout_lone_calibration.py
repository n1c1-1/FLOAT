"""A Calibration placed beside a Session that has no other block yet is drawn wired to its session (the owner's
trial of 15 November: the block was linked in the model, only its wire was not drawn, so it looked loose)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from fixtures import Doc  # noqa: E402
from test_points import ok  # noqa: E402
from astro_gpu.graphcore.layout import layout  # noqa: E402


def night_wires(doc):
    d = layout(doc.project, doc.view)
    return [w.id for w in d.wires if w.id.startswith("n:")]


def test_a_lone_calibration_is_wired_to_its_session():
    doc = Doc()
    ok(doc, "add_block", kind="session")
    rid = doc.project.rows[0].id
    assert night_wires(doc) == []
    ok(doc, "add_block", kind="calibration", row=rid)
    s, c = doc.project.rows[0].nights[0]
    assert night_wires(doc) == [f"n:{s}>{c}"]
    ok(doc, "add_block", kind="registration", row=rid)          # with a stack block: the same wire, once
    assert night_wires(doc).count(f"n:{s}>{c}") == 1
