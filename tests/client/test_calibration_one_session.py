"""A Calibration is a session's own block ("calibration@1") only when that session is calibrated on its own (two or
more sessions in the per-session mode): a project of one session saved in that mode drew its Calibration as
"calibration@1" while Inspect asked for "calibration", and the Calibration page of the computed project stayed blank
(the testers' pass 2, 3 of 3)."""
from __future__ import annotations

from astro_gpu.client.graphnode import BlockMap
from astro_gpu.graphcore import model as gm
from tests.client.test_one_undo import qapp, win  # noqa: F401  (the fixtures are used by name)


def test_the_calibration_of_one_session_is_the_block_inspect_shows(win, monkeypatch):
    s = win.s
    p = s.path
    g = gm.clone(s._g)
    g.calibrate_together = False                                    # the per-session mode, one session
    bm = BlockMap(s.compiled, s, g)
    node = bm.node_of(p, "calibration")
    assert node is not None and node.ref == "calibration" and node.key == "calibration"
    sid = s.p.sessions[0].id
    monkeypatch.setattr(s, "separate_sessions", lambda: [sid])      # calibrated on its own: the session's block
    bm = BlockMap(s.compiled, s, g)
    assert bm.node_of(p, f"calibration@{sid}") is not None and bm.node_of(p, f"calibration@{sid}").sid == sid
