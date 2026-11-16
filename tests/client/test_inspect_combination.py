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


def test_inspect_on_a_computed_combination_shows_the_combined_image(win, env):
    """A combination computed (H-O, S-H-O of a mono filters workflow) said "No image for this block": the block had
    no picture of its own. It shows the combined image its stage saves, as the Stacking block shows the stack."""
    import numpy as np

    from astro_gpu.client.services.inspect import service_of
    from astro_gpu.stages.combine import STAGES
    from tests.client.test_sources_import import fits_file
    from tests.test_api import fake_done
    made_from(win, env, "filters_min")
    _place(win, "ho")
    s = win.s
    for x in s.p.sessions:                                # each filter's lights, and the two filters wired
        s.add_session_entries(x.id, "lights", [str(fits_file(env["tmp"] / "raw" / x.name / f"L_{k}.fits", x.filter))
                                               for k in range(3)], [])
    settle(win)
    q = next(q for q in s.paths if isinstance(q.source, dict) and "fusion" in q.source)
    for role, pid in (("Ha", 1), ("OIII", 2)):
        assert s.set_fusion_input(q, role, pid)
        settle(win, 3)
        q = s.view(q.pid)
    stage = next(x for x in s.all_stages(q) if x in STAGES)
    for v in list(s.paths):                               # computed, without running it: every stage's done result
        for st in s.all_stages(v):
            if st != stage:
                fake_done(s.p, v.pid, st)
    rgb = np.random.default_rng(1).normal(100, 5, (3, 48, 64)).astype(np.float32)
    fake_done(s.p, q.pid, stage, files={"drz_full.npy": rgb})
    s._sync()
    settle(win, 6)
    s.select(index=s.paths.index(s.view(q.pid)), step="fusion")
    win.set_mode(1)
    settle(win, 6)
    service_of(s).wait(30)
    settle(win, 6)
    texts = [x.text() for x in win.tune.findChildren(QLabel) if x.isVisible()]
    assert "No image for this block." not in texts, texts
    img = win.tune.view.image() if win.tune.view is not None else None
    assert img is not None and img.name == "Combined image", texts
    assert not env["errors"], env["errors"]
