"""The Stacking block's line says what its settings and its last result say: the method the result used when it is
known (Classic when the automatic choice fell back to it), and the Classic combination when it is not the mean, with
no rejection named for a combination that has none. After a Classic stack by median it read « drizzle auto ·
rejection at 3 σ »."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from fixtures import Doc  # noqa: E402
from test_points import ok, of_kind  # noqa: E402
from astro_gpu.graphcore import rules  # noqa: E402


def _stacking():
    doc = Doc.from_template("colour_minimal")
    r = doc.project.rows[0]
    return doc, of_kind(doc, r, "stacking")


def test_a_classic_stack_by_median_says_so_without_rejection():
    doc, b = _stacking()
    ok(doc, "set_param", block=b, stage="stack", name="method", value="classic")
    ok(doc, "set_param", block=b, stage="stack", name="classic_combination", value="median")
    assert rules.summary(doc.project, b) == "classic · median"


def test_the_method_the_result_used_is_said():
    doc, b = _stacking()
    ok(doc, "set_param", block=b, stage="stack", name="classic_combination", value="median")
    line = rules.say(rules.stacking_msg(doc.project, b))
    assert line.startswith("drizzle")                              # before a run: the automatic method
    assert rules.say(rules.stacking_msg(doc.project, b, used="classic")) == "classic · median"


def test_the_mean_keeps_its_rejection():
    doc, b = _stacking()
    ok(doc, "set_param", block=b, stage="stack", name="method", value="classic")
    assert rules.say(rules.stacking_msg(doc.project, b)) == "classic · rejection at 3 σ"
