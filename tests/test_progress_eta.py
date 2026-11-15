"""A stage's ETA follows its recent pace, not the mean pace since its start: a stage whose first part weighs more of
its bar than of its time (a stack's debayering: 60 % of the bar, a tenth of the time) said "1 min left" for 10."""
from __future__ import annotations

from types import SimpleNamespace

from astro_gpu.core import stage as st


def _ctx(monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(st.time, "time", lambda: clock[0])
    sent = []
    c = st.Context.__new__(st.Context)
    c._t0, c._last_progress, c._pace = clock[0], 0.0, []
    c.stage = SimpleNamespace(name="stack")
    c.journal = SimpleNamespace(emit=lambda kind, **k: sent.append(k))
    c.check_stop = lambda: None
    return c, clock, sent


def test_the_eta_follows_the_pace_of_the_slow_part_that_follows_a_fast_one(monkeypatch):
    c, clock, sent = _ctx(monkeypatch)
    for s in range(1, 61):                       # a fast part: 60 % of the bar in 60 s
        clock[0] += 1
        c.progress(0.6 * s / 60)
    assert round(sent[-1]["eta"]) == 40          # (at the pace so far, the rest would take 40 s)
    for s in range(1, 41):                       # the slow part: 1 % of the bar per 10 s, 400 s for the 40 %
        clock[0] += 10
        c.progress(0.6 + 0.01 * s)
    mid = [x for x in sent if abs(x["frac"] - 0.63) < 1e-9][0]
    assert 300 < mid["eta"] < 420                # a mean pace since the start would have said 61 s


def test_a_steady_stage_keeps_the_mean_pace(monkeypatch):
    c, clock, sent = _ctx(monkeypatch)
    for s in range(1, 51):
        clock[0] += 2
        c.progress(s / 100)
    assert abs(sent[-1]["eta"] - 100) < 1.0
