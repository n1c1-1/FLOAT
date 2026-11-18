"""tools/t.py: the timed tests (marker ``alone``, Post's speed gate) run after the parallel part, alone in one process,
the library threads as a user has them (in the full run, beside eleven workers, the gate took 2.5 s for its 1.6 s)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _t():
    spec = importlib.util.spec_from_file_location("tools_t", ROOT / "tools" / "t.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_a_parallel_run_runs_the_timed_tests_after_it_alone(monkeypatch):
    t = _t()
    calls = []
    monkeypatch.setattr(t.subprocess, "call", lambda cmd, cwd, env, creationflags: (calls.append((cmd, env)), 0)[1])
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    assert t.main(["3", "-j", "4"]) == 0
    (par, penv), (alone, aenv) = calls
    assert "-n" in par and par[par.index("pytest") + 2] == "(not slow and not network) and not alone"
    assert "-n" not in alone and alone[alone.index("pytest") + 2] == "(not slow and not network) and alone"
    assert penv["OMP_NUM_THREADS"] == "1" and "OMP_NUM_THREADS" not in aenv     # the user's threads for the timed ones


def test_one_process_runs_everything_at_once_and_the_codes_combine(monkeypatch):
    t = _t()
    calls = []
    monkeypatch.setattr(t.subprocess, "call", lambda cmd, cwd, env, creationflags: (calls.append(cmd), 0)[1])
    assert t.main(["3", "-j", "0"]) == 0 and len(calls) == 1 and "alone" not in calls[0][calls[0].index("pytest") + 2]
    assert t.combined(0, 5) == 0 and t.combined(5, 0) == 0 and t.combined(5, 5) == 5
    assert t.combined(1, 0) == 1 and t.combined(0, 1) == 1 and t.combined(5, 1) == 1
