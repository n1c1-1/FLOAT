"""A block's line and tooltip never print a list, a dict, an array or a long string raw: a Gradient variant with
manual samples says "307 samples", and no block kind of a full template prints a long value injected in a setting."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "graphcore"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from fixtures import Doc  # noqa: E402
from test_points import ok, of_kind  # noqa: E402
from astro_gpu.graphcore import api  # noqa: E402
from astro_gpu.client import graphnode  # noqa: E402
from astro_gpu.client.i18n import set_language  # noqa: E402

BOUND = 120


def line_of(doc, b):
    """The line the graph's tooltip carries for a block (document.summary's own composition)."""
    line = api.summary(doc.project, b)
    diff = graphnode.variant_diff(doc.project, b)
    return f"{line} · {diff}" if diff and diff not in line else line


def _pts(n=307):
    return [[267.0 + i, 89.0] for i in range(n)]


def test_a_gradient_variant_with_manual_samples_says_their_count():
    for lang, words in (("en", "307 samples"), ("fr", "307")):
        set_language(lang)
        doc = Doc.from_template("colour_minimal")
        r = doc.project.rows[0]
        ok(doc, "add_variant", block=of_kind(doc, r, "gradient"))
        first = doc.project.rows[-1].blocks[0]
        doc.project.blocks[first].params = {"gradient": {"surface_degree": 5, "sample_points": _pts()}}
        line = line_of(doc, first)
        assert words in line and "[" not in line and "267.0" not in line and len(line) <= BOUND, line
    set_language("en")


def test_no_block_kind_of_a_full_template_prints_a_long_value():
    set_language("en")
    for tpl in ("colour_minimal", "sho_rgb"):
        doc = Doc.from_template(tpl)
        for b in list(doc.project.blocks):
            blk = doc.project.blocks[b]
            for stage in ("gradient", "stack", "crop", "denoise", "color"):
                for val in (_pts(), {"k": _pts(50)}, "x" * 300):
                    blk.params = {stage: {"some_setting": val, "exclude_brush": _pts(40)}}
                    line = line_of(doc, b)
                    assert "[" not in line and "x" * 50 not in line and "267.0" not in line, (tpl, blk.kind, line)
                    assert len(line) <= BOUND * 2, (tpl, blk.kind, len(line))
            blk.params = {}
