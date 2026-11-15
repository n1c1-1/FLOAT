"""Inspect on a combination block not computed yet (S-H-O of the filters template): the page shows "Not computed yet"
and its Run to here button with the combination's own plan. It raised "unknown block 'fusion'" where the window
showed the crash box (a beta tester's window stopped answering on that click)."""
from __future__ import annotations

from PySide6.QtWidgets import QLabel, QPushButton

from astro_gpu.client.services.run import plan_to
from tests.client.test_combination_recipes_page import _place
from tests.client.test_swap_smoke import env, made_from, qapp, settle, win  # noqa: F401  (fixtures by name)


def test_inspect_on_a_combination_not_computed_shows_its_plan(win, env):
    made_from(win, env, "filters_min")
    _place(win, "sho")
    s = win.s
    q = next(q for q in s.paths if isinstance(q.source, dict) and "fusion" in q.source)
    plan = plan_to(s, q, "fusion")                       # the combination's own stage (no frames here: nothing to run)
    assert plan.jobs == [] or any(j.stage.startswith("combine_") for j in plan.jobs)
    s.select(index=s.paths.index(q), step="fusion")
    win.set_mode(1)
    settle(win, 6)
    texts = [x.text() for x in win.tune.findChildren(QLabel) if x.isVisible()]
    assert "Not computed yet" in texts, texts
    run = [b for b in win.tune.findChildren(QPushButton) if b.isVisible() and b.text().startswith("Run to here")]
    assert run, "no Run to here button"
    assert not env["errors"], env["errors"]
