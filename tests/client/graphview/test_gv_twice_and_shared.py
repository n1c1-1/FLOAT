"""The same night added to two sessions of one stack, and calibration frames shared between sessions. The engine reads
the union of a stack's sessions once per file (Project2.inputs keeps unique paths, ``list_frames`` a set): the screen
counted the lights session by session (432 shown for 372 read). Now: lights of another session of the stack are said
at the add and not added (« Add anyway »), the stack counts distinct lights and says how many are listed twice, the
later session flags them; the same darks in two sessions are a sharing said in Inspect's words, never a warning."""
from __future__ import annotations

from astro_gpu.core.project import list_frames, stage_graph
from astro_gpu.core.workflow import chains
from tests.client.graphview.test_gv_files import Toasts, settle
from tests.client.graphview.test_gv_table import Bench, fwd, make

stage_graph()             # the engine's stages loaded at collection, not in the first test's time
chains()


def test_gv_twice_lights_of_another_session_are_said_and_counted_once(qapp, tmp_path):
    from astro_gpu.graphcore import rules
    night = make(tmp_path / "night", 6, "L")
    b = Bench(1366, sessions=2)
    s1, s2 = b.sessions()
    b.drop([night], ("box", s1, "lights"))
    b.read()
    n0 = b.steps()
    b.drop([night], ("box", s2, "lights"))                         # the same night again, on Session 2
    b.read()
    assert b.steps() == n0 and not b.own(s2, "lights")
    name1 = b.g.model.session_name(s1)
    assert b.t.toast_text == f"6 already in “{name1}”: not added" and "add_anyway" in b.t.toast_w.text()
    b.t.link("add_anyway")                                         # the user's own choice
    b.read()
    assert len(b.own(s2, "lights")) == 6 and b.steps() == n0 + 1
    pl = b.stacks()[0]
    b.g.model.invalidate()
    assert b.g.model.totals(pl)[1] == 6 and b.g.model.twice(pl) == 6
    assert "6 lights · 6 listed twice" in b.t.summary()
    union = b.own(s1, "lights") + b.own(s2, "lights")
    assert len(list_frames(union)) == 6                            # what the engine reads of the stack
    ws = [w for w in b.g.model.session_warnings(s2) if w.kind == "lights"]
    assert len(ws) == 1 and ws[0].text == f"6 also in Lights of “{name1}”" and ws[0].steps
    assert not [w for w in b.g.model.session_warnings(s1) if w.kind == "lights"]    # said once, on the later one
    card = rules.card_summary_msg(b.doc.graph, pl["id"])
    assert "6 lights · 6 listed twice" in b.bridge.say(card)
    b.close()


def test_gv_twice_on_a_session_block_and_as_a_new_session(qapp, tmp_path):
    night = make(tmp_path / "night", 4, "L")
    b = Bench(1366, sessions=2)
    s1, s2 = b.sessions()
    rid = b.stacks()[0]["id"]
    b.drop([night], ("box", s1, "lights"))
    b.read()
    toasts = Toasts(b.bridge)
    assert b.bridge.drops.drop(s2, "lights", [fwd(night)]) is None  # a Session block's target, its panel's rows
    assert "4 already in" in toasts.text and "add_anyway" in toasts.keys
    n = len(b.sessions())
    b.drop([night], ("ghost_s", rid))                               # a new session of the stack
    assert len(b.sessions()) == n and "4 already in" in b.t.toast_text
    b.t.link("add_anyway")
    assert len(b.sessions()) == n + 1
    b.bridge.drops.link("add_anyway")
    settle(b.reader)
    assert len(b.own(s2, "lights")) == 4
    b.close()


def test_gv_shared_darks_are_a_sharing_not_warnings(qapp, tmp_path):
    from astro_gpu.client.graphview.files_panel import FilesPanel
    darks = make(tmp_path / "darks", 5, "D")
    b = Bench(1366, sessions=3)
    ss = b.sessions()
    for s in ss:
        b.drop([make(tmp_path / f"L{s}", 3, "L")], ("box", s, "lights"))
        b.drop([darks], ("box", s, "darks"))
    b.read()
    b.g.model.invalidate()
    assert not [w for s in ss for w in b.g.model.session_warnings(s) if w.kind == "darks"]
    assert not [w for s in ss for s2, k, _t, _f in b.doc.warnings(s) if k == "darks"]
    others = [b.g.model.session_name(s) for s in ss[1:]]
    tip = b.g.tooltip_text(b.pos(("box", ss[0], "darks"), 0.3, 0.5))
    assert b.g.model.shared_with(ss[0], "darks") == others
    p = FilesPanel(b.bridge, ss[0], "darks")
    assert not p.shared.isHidden() and p.shared.text() == f"shared with {', '.join(others)}: one master"
    assert f"shared with {', '.join(others)}: one master" in tip
    b.close()
