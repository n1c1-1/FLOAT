"""The Stacking block's line on the graph and its Table cell say the method its last result used (Classic when the
automatic choice fell back to it) and its Classic combination without a rejection it does not have: after a Classic
stack by median they read « drizzle auto · rejection at 3 σ »."""
from types import SimpleNamespace

from tests.client.graphview.test_gv_table import Bench
from tests.client.graphview.test_gv_table_follow_mark import _lines


class Model:
    """What the project model says of the Stacking: its last result used the Classic method."""

    def __init__(self, b):
        self.block_map = SimpleNamespace(node=lambda x: SimpleNamespace(paths=["wf"]) if x == b else None)

    def param_info(self, p, stage, name):
        assert (p, stage, name) == ("wf", "stack", "method")
        return SimpleNamespace(drift=True, result_value="classic", origin="auto", suggestion_known=False,
                               value="drizzle")


def test_the_stacking_line_says_the_classic_median_its_result_used(qapp):
    b = Bench(1920)
    stk = next(x for x, blk in b.doc.graph.blocks.items() if blk.kind == "stacking")
    assert b.bridge.act("set_param", quiet=True, block=stk, stage="stack", name="classic_combination",
                        value="median").ok
    assert b.doc.summary(stk).startswith("drizzle")                 # nothing known of a result: the automatic method
    b.bridge.document.model = Model(stk)
    b.g.after_model()
    assert b.doc.summary(stk) == "classic · median"
    lines = _lines(b, ("setting", b.stacks()[0]["id"], "stk"))
    assert lines[0] == "classic · median" and "rejection" not in " ".join(lines)
    b.close()
