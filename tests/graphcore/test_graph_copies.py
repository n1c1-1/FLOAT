"""What a gesture copies or carries keeps all it had: a variant's wires keep their channel, image and workflow; a
split or merged night keeps its frames left out by hand; a duplicated stack copies its whole workflow."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from fixtures import Doc  # noqa: E402
from test_points import ok, row, of_kind  # noqa: E402
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
