"""Inspect's settings panel is the same panel as the Graph's (panel.Panel): a setting of a computed block changed there
is the block's draft, shown at once, its bar at the panel's foot, nothing outdated. The beta tester's crash replayed: a
Calibration setting changed in Inspect, the Graph asked for (the question: Stay) and Inspect again, then Try variant: a valid
variant row (a Calibration held by the stack, not by a block: the variant copies the whole stack), never an exception.
The same on Registration and Stacking. On a copy of the computed test project (tests/client/test_inspect.py's window)."""
from __future__ import annotations

import pytest
from PySide6.QtWidgets import QMessageBox

from astro_gpu.client import prefs
from astro_gpu.client.blockdraft import DRAFT_ANSWER_ENV, DraftBar
from astro_gpu.client.panel import CompactRow, Panel
from tests.client.test_inspect import inspect, open_window, qapp, settle  # noqa: F401  (qapp: the fixture)
from conftest import needs_test_project, scratch_project  # noqa: E402  (tests/conftest.py)

pytestmark = needs_test_project


@pytest.fixture(scope="module")
def dwin(qapp, tmp_path_factory):
    import os
    folder = scratch_project(tmp_path_factory.mktemp("draft") / "heart")
    os.environ["ASTRO_GPU_SETTINGS"] = str(folder.parent / "settings.json")
    prefs.load()
    w = open_window(folder)
    yield w
    w.close()
    os.environ.pop("ASTRO_GPU_SETTINGS", None)


@pytest.fixture(autouse=True)
def _no_box(monkeypatch):
    for kind in ("information", "warning", "critical"):
        monkeypatch.setattr(QMessageBox, kind, lambda *a, **k: pytest.fail(f"a message box: {a[1:3]}"))


def _states(s):
    return {k: (v.state, v.key) for k, v in s._states.items()}


def _bar(panel):
    return next(iter(panel.findChildren(DraftBar)), None)


def _change_a_setting(win, stage):
    """A setting of ``stage`` changed in Inspect's panel that makes a draft (the first one that does)."""
    pn = win.tune.panel
    pn.expert = True
    pn.refresh(pn.node)
    settle(win)
    for row in [r for r in pn.findChildren(CompactRow) if r.r.get("stage") == stage and not r.r.get("locked")]:
        r = row.r
        v = r["value"]
        nv = (not v) if isinstance(v, bool) else next((c for c, _t in r["choices"] if c != v), None) \
            if r.get("choices") else (round(v + (r.get("step") or 1), 6) if isinstance(v, (int, float)) else None)
        if nv is None or (r.get("max") is not None and nv > r["max"]):
            continue
        pn.set_setting(r, nv)
        settle(win, 2)
        if win.s.pending_change:
            return r["name"], nv
    pytest.fail(f"no setting of {stage} makes a draft")


@pytest.mark.parametrize("step, stage", [("calibration", "calibrate"), ("alignement", "register"),
                                         ("empilement", "stack")])
def test_a_setting_changed_in_inspect_left_to_the_graph_and_back_then_try_variant(dwin, monkeypatch, step, stage):
    win, s = dwin, dwin.s
    monkeypatch.setenv(DRAFT_ANSWER_ENV, "stay")       # the question asked on the way out is answered Stay
    inspect(win, step, 0)
    assert isinstance(win.tune.panel, Panel)           # the Graph's panel component
    n, states = len(s.paths), _states(s)
    name, value = _change_a_setting(win, stage)
    assert _states(s) == states and s.pending_change.step == step        # a draft: nothing outdated
    bar = _bar(win.tune.panel)
    assert bar is not None and "→" in bar.what.text() and bar.cost.text().startswith("Computes again")
    win.set_mode(0)                                    # the Graph: the question is asked, Stay (the page keeps the draft)
    settle(win)
    assert win.mode == 1
    win.set_mode(1)
    settle(win)
    assert s.pending_change and win.mode == 1 and s.sel_step == step
    bar = _bar(win.tune.panel)
    assert bar is not None
    if not bar.b_variant.isEnabled():                  # impossible by design: greyed with its reason, never raised
        assert bar.b_variant.toolTip().startswith("No variant: ")
        win.tune.panel.draft_reset()
        return
    bar.b_variant.click()
    settle(win)
    assert len(s.paths) == n + 1 and not s.pending_change
    q = s.paths[-1]
    assert q.variant_of == s.paths[0].pid and s.param_value(q, stage, name) == value
    assert s.param_value(s.paths[0], stage, name) != value
    assert {k: v for k, v in _states(s).items() if k[0] == s.paths[0].pid} == \
        {k: v for k, v in states.items() if k[0] == s.paths[0].pid}               # the original is still computed
    s.undo()
    settle(win)
    assert len(s.paths) == n
    win.tune.panel.expert = False
