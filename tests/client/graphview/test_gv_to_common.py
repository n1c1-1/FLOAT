"""« To Common » on darks shared by two sessions: they became nothing (the session followed an empty Common, its darks
removed), and the notice's Undo link and Ctrl+Z did nothing in the window. Now the frames become Common's and the
sessions holding them follow it (nothing lost); a session that followed the empty Common keeps none; one undo step,
which the notice's link, the table's Ctrl+Z and Edit > Undo all take back."""
from __future__ import annotations

from astro_gpu.core.project import stage_graph
from astro_gpu.core.workflow import chains
from tests.client.graphview.test_gv_table import Bench, make

stage_graph()             # the engine's stages loaded at collection, not in the first test's time
chains()


def test_gv_to_common_promotes_shared_darks_and_undoes(qapp, tmp_path):
    from PySide6.QtCore import Qt
    from astro_gpu.graphcore.operations import COMMON
    darks = make(tmp_path / "darks", 4, "D")
    b = Bench(1366, sessions=3)
    s1, s2, s3 = b.sessions()
    for s in (s1, s2, s3):
        b.drop([make(tmp_path / f"L{s}", 2, "L")], ("box", s, "lights"))
    for s in (s1, s2):
        b.drop([darks], ("box", s, "darks"))
    b.read()
    shared = b.own(s1, "darks")
    n0 = b.steps()
    b.g.back_to_common([b.g.cell(("box", s1, "darks"))])               # the cell menu's « To Common »
    assert b.steps() == n0 + 1
    assert b.g.model.common_list("darks") == shared                    # the frames are Common's now
    assert b.own(s1, "darks") is None and b.own(s2, "darks") is None   # both follow it: nothing lost
    assert b.own(s3, "darks") == []                                    # had none, keeps none
    assert b.box(s1, "darks").n == 4 and b.box(s3, "darks").n == 0
    assert "are Common's now · 2 sessions follow them" in b.t.toast_text and "undo" in b.t.toast_w.text()
    b.t.link("undo")                                                   # the notice's Undo
    assert b.own(s1, "darks") == shared and b.own(s2, "darks") == shared and not b.g.model.common_list("darks")
    b.g.back_to_common([b.g.cell(("box", s1, "darks"))])
    b.key(Qt.Key_Z, Qt.ControlModifier)                                 # Ctrl+Z on the table
    assert b.own(s1, "darks") == shared and b.box(COMMON, "darks").n == 0
    b.close()


def test_gv_to_common_undo_goes_through_the_window_history(qapp, tmp_path):
    """In the window the table's undo is the window's (bridge.history): what it returns is not the core's Result;
    the table is drawn again all the same, and nothing raises."""
    darks = make(tmp_path / "darks", 3, "D")
    b = Bench(1366, sessions=2)
    s1, s2 = b.sessions()
    for s in (s1, s2):
        b.drop([darks], ("box", s, "darks"))
    b.read()
    calls = []
    b.bridge.history = (lambda: calls.append("undo") or b.bridge.undo_graph() and None,
                        lambda: calls.append("redo") or b.bridge.redo_graph() and None)
    b.g.back_to_common([b.g.cell(("box", s1, "darks"))])
    b.t.link("undo")
    assert calls == ["undo"] and b.own(s1, "darks") is not None and b.box(s1, "darks").mode == "own"
    b.close()
