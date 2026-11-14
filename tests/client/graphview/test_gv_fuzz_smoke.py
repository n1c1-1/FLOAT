"""Window fuzz smoke: one seed, twenty random real gestures on the graph page from one start, every check of
tools/fuzz/graphview_fuzz.py after each, inside the fast suite's budget (under 10 s of the process's own time). The
long run (3 starts x 1,000 gestures, on a graph in memory and on the project model) is the tool's, before a merge."""
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools", "fuzz"))
import graphview_fuzz  # noqa: E402

GESTURES = 20


def test_window_fuzz_one_seed_twenty_gestures(qapp, tmp_path):
    t0 = time.process_time()          # the process's own time: a machine busy with other work does not count
    files = graphview_fuzz.make_files(str(tmp_path / "files"))
    s = graphview_fuzz.fuzz("sho_rgb", 1, GESTURES, files)
    assert s["clean"], s["failure"]
    assert s["run"] == GESTURES
    assert s["stats"]["accepted"] >= 4 and s["stats"]["refused"] >= 2, s["stats"]
    assert time.process_time() - t0 < 10, time.process_time() - t0
