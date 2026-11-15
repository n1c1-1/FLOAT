"""A greyed palette card names the requirement the block misses in THAT chain, the engine's (the night test pass:
"Colour needs Stacking first" beside a Stacking while Astrometry was missing; Stars/background said the same). A place
too early in the chain (before the Stacking) only says the place is wrong: another place may take the block."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from fixtures import Doc  # noqa: E402
import astro_gpu.client.names  # noqa: E402,F401  (the names as the window shows them: the engine's "Color" is "Colour")
from test_points import ok  # noqa: E402


def _stacked():
    doc = Doc()
    ok(doc, "add_row", kind="session", files={"lights": ["L1.fit", "L2.fit"]})
    rid = doc.project.rows[0].id
    for k in ("calibration", "registration", "stacking", "gradient"):
        if k != "calibration":
            ok(doc, "add_block", kind=k, row=rid)
    return doc, rid


def test_color_and_stars_name_what_they_miss_beside_a_stacking():
    doc, rid = _stacked()
    assert doc.palette_state("color") == (False, "needs Astrometry first")
    assert doc.palette_state("stars") == (False, "needs Denoise first")
    ok(doc, "add_block", kind="astrometry", row=rid)
    assert doc.palette_state("color")[0]                                 # its requirement met: placed
    ok(doc, "add_block", kind="denoise", row=rid)
    assert doc.palette_state("stars")[0] or "Stacking" not in doc.palette_state("stars")[1]


def test_without_any_stacking_the_stacking_is_what_is_missing():
    doc = Doc()
    ok(doc, "add_row", kind="session", files={"lights": ["L1.fit", "L2.fit"]})
    assert doc.palette_state("color") == (False, "Colour needs Stacking first")
