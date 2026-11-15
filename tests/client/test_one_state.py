"""One state per block, said the same everywhere (the night test pass: a red Stacking whose tooltip said "Up to date",
Inspect "Outdated", the Gradient view "The stack is not computed", Post "Not computed yet"). The model's state through
``BlockMap`` is the source; the graph's block and its tooltip, the panel's state line, Inspect's strip, the Post block's
rows and Run to here's plan all say it, on a computed project: computed, outdated by a setting, outdated by an upstream
change, never computed, interrupted, failed. An Outdated block says why in a few words."""
from __future__ import annotations

import os
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402
from conftest import full_only, needs_test_project, scratch_project  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from astro_gpu import api  # noqa: E402
from astro_gpu.client import app as appmod  # noqa: E402
from astro_gpu.client import prefs  # noqa: E402
from astro_gpu.client.graphnode import STATE_WORD  # noqa: E402

pytestmark = needs_test_project


@pytest.fixture(scope="module")
def qapp():
    from astro_gpu.client.widgets import QSS
    a = QApplication.instance() or QApplication([])
    a.setStyleSheet(QSS + appmod.EXTRA_QSS)
    return a


def settle(w, n=8):
    for _i in range(n):
        w.worker.wait(30)
        QApplication.processEvents()
        time.sleep(0.02)
    QApplication.processEvents()


@pytest.fixture()
def win(qapp, tmp_path, monkeypatch):
    monkeypatch.setenv("ASTRO_GPU_SETTINGS", str(tmp_path / "settings.json"))
    prefs.load()
    prefs.set("home_sections", [])
    appmod.apply_language("en")
    folder = scratch_project(tmp_path / "heart")
    p = api.open_project(folder)
    full_only(p)
    api.save(p)
    w = appmod.Main()
    w.resize(1600, 950)
    w.show()
    assert w.open_project(folder)
    end = time.monotonic() + 180
    while time.monotonic() < end and not w.s.ready:
        w.worker.wait(1)
        QApplication.processEvents()
    assert w.s.ready, w.s.error
    settle(w)
    yield w
    w.close_project(discard=True)
    w.close()
    w.deleteLater()
    QApplication.processEvents()


def _block(w, kind):
    return next(k for k, x in w.graph.bridge.graph.blocks.items() if x.kind == kind)


def said(w, kind) -> dict:
    """What each view says of the block of ``kind`` in the first workflow: {view: state key}, and the graph's tooltip."""
    s, page = w.s, w.graph
    g = s.block_map
    b = _block(w, kind)
    node = g.node(b)
    page.canvas.update_states()
    item = page.canvas.blocks[b]
    words = {v: k for k, v in STATE_WORD.items()}
    tip = item.toolTip()
    tip_state = next((k for line in tip.splitlines() for word, k in words.items()
                      if line == word or line.startswith(word + " · ")), None)
    pn = w.graph.panel                                  # the graph page's settings panel
    panel = pn.state_chip(node)
    out = {"model": g.state(node), "item": item.d["state"], "tooltip": tip_state, "panel": words.get(panel.text())}
    q = node.paths[0]
    last = g._last_nodes().get(q.pid) is node
    if last:
        out["post block"] = pn.output_state(q)
    from astro_gpu.client.services.run import plan_to
    plan = plan_to(s, q, node.ref)
    out["run plan"] = "done" if not getattr(plan, "jobs", plan) else out["model"]
    return out, tip, panel.toolTip()


def _strip_word(w, key):
    w.s.select(index=0, step=key)
    w.set_mode(1)
    settle(w, 4)
    st = w.tune.strip
    words = {v: k for k, v in STATE_WORD.items()}
    tips = [x.toolTip() for x in st.findChildren(type(st.current)) if x is st.current]
    w.set_mode(0)
    settle(w, 2)
    return words.get(tips[0].split(" · ")[0]) if tips else None


def _agree(got, want):
    vals = {k: v for k, v in got.items()}
    assert set(vals.values()) == {want}, vals


def test_one_state_everywhere_and_why_an_outdated_block_is(win):
    s = win.s
    full = s.paths[0]
    if s.state(full, "stack") != "done" or s.state(full, "gradient") != "done":
        pytest.skip("the test project's Full is not up to date")
    # computed
    got, tip, _t = said(win, "stacking")
    _agree(got, "done")
    assert _strip_word(win, "empilement") == "done"
    # outdated by a setting: the block says which one, the one after says where the change was made
    s.set_param(full, "stack", "kappa", 4.0)
    settle(win)
    got, tip, chip_tip = said(win, "stacking")
    _agree({k: v for k, v in got.items() if k != "run plan"}, "stale")
    assert got["run plan"] == "stale"
    assert "Outdated · " in tip and "→ 4" in tip, tip
    assert "→ 4" in chip_tip, chip_tip
    assert _strip_word(win, "empilement") == "stale"
    got, tip, _t = said(win, "gradient")
    _agree(got, "stale")
    assert "Outdated · Stacking changed" in tip, tip
    s.set_param(full, "stack", "kappa", None)
    settle(win)
    _agree(said(win, "stacking")[0], "done")


@pytest.mark.parametrize("state", ["pending", "interrupted", "failed"])
def test_never_computed_interrupted_and_failed_are_said_alike(win, state):
    """The run's states as the engine gives them (put in the model as a run would leave them)."""
    s = win.s
    full = s.paths[0]
    from astro_gpu.client.graphnode import run_stages
    g = s.block_map
    node = g.node(_block(win, "gradient"))
    for x in run_stages(s, full, node.ref):
        s._states[(full.pid, x)] = api.StageState(state, "")
    got, tip, _t = said(win, "gradient")
    want = {k: v for k, v in got.items() if k != "run plan"}
    _agree(want, state)
    assert STATE_WORD[state] in tip.splitlines()


def test_the_gradient_preview_reads_the_earlier_stack_of_an_outdated_one(win):
    """The Gradient view of an Outdated stack shows its earlier result: its preview reads that one too, never "The
    stack is not computed"."""
    from astro_gpu.client.gradientedit import _reduced_stack
    s = win.s
    full = s.paths[0]
    if s.state(full, "stack") != "done":
        pytest.skip("the test project's Full is not up to date")
    s.set_param(full, "stack", "kappa", 4.0)
    settle(win)
    assert s.state(full, "stack") == "stale"
    sm = _reduced_stack(s.p, full.pid)
    assert sm is not None and Path(str(api.stage_outputs(s.p, full.pid, "stack", which="previous").dir)).is_dir()
