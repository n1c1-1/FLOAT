"""Fuzz smoke: one seed, 300 random gestures from blank and from one fixture (the full fuzz is tools/fuzz/graph_fuzz.py,
run before each merge, 5 seeds x 2,000 gestures from every start)."""
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools", "fuzz"))
import graph_fuzz  # noqa: E402


def test_fuzz_smoke_one_seed_300_gestures():
    t0 = time.process_time()          # the process's own time: a machine busy with other work does not count
    for key in ("blank", "sho_rgb"):
        digest = graph_fuzz.fuzz(1, key, 300)
        assert digest, f"fuzz failed from {key}"
    assert time.process_time() - t0 < 30, time.process_time() - t0
