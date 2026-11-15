"""The Table's beta-tester findings: frames dropped again on a box's « replace » half go through the rule of doubles
(4 lights replaced by 2 of them, before); the Stacking column says the method the settings panel stored (« drizzle
auto » for a stack set to Classic, before)."""
from __future__ import annotations

from tests.client.graphview.test_gv_table import Bench, files_in, make


def test_frames_already_there_dropped_on_the_replace_half_are_doubles(qapp, tmp_path):
    night = make(tmp_path / "night", 4)
    b = Bench(1920)
    sid = b.sessions()[0]
    b.drop([night], ("box", sid, "lights"))
    b.read()
    assert len(b.own(sid, "lights")) == 4
    n0 = b.steps()
    two = files_in(night)[:2]
    b.drop(two, ("box", sid, "lights"), fx=0.8)                      # the replace half
    b.read()
    assert len(b.own(sid, "lights")) == 4 and b.steps() == n0        # nothing replaced, nothing added
    assert "Already there" in b.t.toast_text, b.t.toast_text
    more = make(tmp_path / "more", 2, prefix="NEW")
    b.drop(two + files_in(more), ("box", sid, "lights"), fx=0.8)     # some new: added, the doubles left
    b.read()
    assert len(b.own(sid, "lights")) == 6 and b.steps() == n0 + 1
    b.close()


def test_the_stacking_column_says_the_method_stored_on_the_block(qapp):
    from astro_gpu.client.graphview.table_logic import settings_text
    b = Bench(1920)
    g = b.doc.graph
    st = next(x for x, blk in g.blocks.items() if blk.kind == "stacking")
    eff = b.doc.effective(st)
    assert settings_text("stacking", eff).startswith("drizzle")
    blk = type("B", (), {"params": {"stack": {"method": "classic", "kappa": 2.5}}})()
    assert settings_text("stacking", eff, block=blk) == "classic · rejection at 2.5 σ"
    assert settings_text("stacking", eff, short=True, block=blk) == "classic 2.5σ"
    b.close()
