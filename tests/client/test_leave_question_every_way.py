"""The leave question is asked on every way out of a block holding a draft, for every block editor (the pages Inspect
opens: Calibration, Registration, Selection, Stacking, Astrometry, Gradient, Denoise, Color). The owner changed
Calibration's hot pixel threshold and went back to the Graph: no question, it came only when he entered another block.
The rule is the same for every block, so a block added to BLOCKS must pass it: the Graph, Post, Report, another block
of the flow bar, closing the project. Stay keeps the block, its page and its draft. On a copy of the computed test
project (tests/client/test_inspect.py's window)."""
from __future__ import annotations

import pytest
from PySide6.QtWidgets import QMessageBox

from astro_gpu.client import blockdraft, prefs
from astro_gpu.client.panel import CompactRow
from tests.client.test_inspect import inspect, open_window, qapp, settle  # noqa: F401  (qapp: the fixture)
from conftest import needs_test_project, scratch_project  # noqa: E402  (tests/conftest.py)

pytestmark = needs_test_project

#: the blocks of the test project whose page holds a draft (a Panel page, the Gradient editor)
BLOCKS = ("calibration", "alignement", "selection", "empilement", "astrometrie", "fond", "debruitage", "couleur")
WAYS = ("graph", "post", "report", "other_block", "close")


@pytest.fixture(scope="module")
def lwin(qapp, tmp_path_factory):
    import os
    folder = scratch_project(tmp_path_factory.mktemp("leave") / "heart")
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


def make_draft(win, step):
    """A setting of the block shown changed and not applied: its draft."""
    s = win.s
    if step == "fond":
        ed = win.tune.grad_editor
        ed.set_draft({"method": "rbf"})
        assert ed.has_draft()
        return
    pn = win.tune.panel
    pn.expert = True
    pn.refresh(pn.node)
    settle(win)
    for row in [r for r in pn.findChildren(CompactRow) if not r.r.get("locked")]:
        r = row.r
        v = r["value"]
        nv = (not v) if isinstance(v, bool) else next((c for c, _t in r["choices"] if c != v), None) \
            if r.get("choices") else (round(v + (r.get("step") or 1), 6) if isinstance(v, (int, float)) else None)
        if nv is None or (r.get("max") is not None and nv > r["max"]):
            continue
        pn.set_setting(r, nv)
        settle(win, 2)
        if s.pending_change:
            return
    pytest.fail(f"no setting of {step} makes a draft")


def drop_draft(win):
    win.s.pending_change = None
    win.tune.grad_editor.drafts.clear()
    settle(win, 2)


@pytest.mark.parametrize("way", WAYS)
@pytest.mark.parametrize("step", BLOCKS)
def test_every_way_out_of_a_block_with_a_draft_asks_first(lwin, monkeypatch, step, way):
    win, s = lwin, lwin.s
    asked = []
    monkeypatch.setattr(blockdraft, "ask_leave", lambda parent, block, lines=(), cost="", variant=True:
                        asked.append((block, list(lines))) or "stay")
    inspect(win, step, 0)
    make_draft(win, step)
    other = next(k for k in ("calibration", "alignement", "selection") if k != step)
    try:
        if way == "graph":
            win.set_mode(0)
        elif way == "post":
            win.set_mode(2)
        elif way == "report":
            win.set_mode(3)
        elif way == "other_block":
            win.tune.select(step=other)                    # the flow bar's click
        else:
            assert win.close_project(go_home=True) is False
        settle(win, 2)
        assert asked, f"{step} left by {way}: no question"
        assert asked[0][1], "the question names the change"
        assert win.mode == 1 and s.sel_step == step          # Stay: the block's page, as it was
        assert s.pending_change or win.tune.grad_editor.has_draft()
    finally:
        drop_draft(win)


@pytest.mark.parametrize("way", ("graph", "post", "other_block"))
@pytest.mark.parametrize("step", ("fusion", "composition"))
def test_real_every_way_out_of_a_combination_block_with_a_draft_asks_first(qapp, tmp_path, monkeypatch, step, way):
    """The combination family's editors (a combination, Combine layers) on the owner's M27 (a scratch copy of its
    definition, its results read in place): the same rule."""
    from test_combination_family import M27, _whole, pump

    from astro_gpu.client.graphnode import combine_kind
    w = _whole(tmp_path, M27)
    try:
        s = w.s
        asked = []
        monkeypatch.setattr(blockdraft, "ask_leave", lambda parent, block, lines=(), cost="", variant=True:
                            asked.append((block, list(lines))) or "stay")
        q = next(q for q in s.paths if combine_kind(q) and
                 not all(i.get("image") == "stars" for i in q.source["fusion"]["inputs"]))
        s.select(index=s.index_of(q.pid), step=step)
        w.set_mode(1)
        settle(w)
        pump(3)
        ed = w.tune.comb_editor if step == "fusion" else w.tune.layers_editor
        assert ed.page_shown()
        if step == "fusion":
            ed.set_intensity("SII", 1.5)
        else:
            ed.set_slider("saturation", 3.0)
        assert ed.has_draft()
        if way == "graph":
            w.set_mode(0)
        elif way == "post":
            w.set_mode(2)
        else:
            w.tune.select(step="calibration")
        pump(0.3)
        assert asked and asked[0][1], f"{step} left by {way}: no question"
        assert w.mode == 1 and s.sel_step == step and ed.has_draft()
    finally:
        w.s.pending_change = None
        w.close()


# ---------------------------------------------------------------- the draft bar is at one place in every block's page
BAR_TOP_MAX = 260        # px under the top of Inspect's page: the bar is the first thing of the settings column


def bar_of(win, step):
    """The draft bar of the block page shown (the one visible, None when there is none)."""
    from astro_gpu.client.blockdraft import DraftBar
    bars = [b for b in win.tune.findChildren(DraftBar) if b.isVisible()]
    assert len(bars) <= 1, f"{step}: {len(bars)} draft bars"
    return bars[0] if bars else None


def assert_bar_at_the_top(win, step):
    """The rule of every block's page: no bar without a draft (nothing greyed), with a draft one bar at the top of the
    settings column, outside any scroll area, with its three buttons."""
    from PySide6.QtCore import QPoint
    from PySide6.QtWidgets import QScrollArea
    bar = bar_of(win, step)
    assert bar is not None, f"{step}: a draft with no bar"
    assert not any(isinstance(a, QScrollArea) for a in _parents(bar, win.tune)), f"{step}: the bar scrolls away"
    top = bar.mapTo(win.tune, QPoint(0, 0)).y() - win.tune.pages.mapTo(win.tune, QPoint(0, 0)).y()
    assert top <= BAR_TOP_MAX, f"{step}: the bar is at {top} px, not at the top of the column"
    assert bar.b_apply.isEnabled() and bar.b_reset.isEnabled()


def _parents(w, stop):
    w = w.parentWidget()
    while w is not None and w is not stop:
        yield w
        w = w.parentWidget()


@pytest.mark.parametrize("step", BLOCKS)
def test_the_draft_bar_is_at_the_top_of_every_block_page_with_a_draft_only(lwin, step):
    win = lwin
    inspect(win, step, 0)
    settle(win, 2)
    assert bar_of(win, step) is None, f"{step}: a bar without a draft"
    make_draft(win, step)
    settle(win, 2)
    try:
        assert_bar_at_the_top(win, step)
    finally:
        if step == "fond":
            win.tune.grad_editor.reset()                   # Reset, as the bar's button
        else:
            win.tune.panel.draft_reset()
    settle(win, 2)
    assert bar_of(win, step) is None, f"{step}: the bar stays after the draft went"


@pytest.mark.parametrize("step", ("fusion", "composition"))
def test_real_the_draft_bar_is_at_the_top_of_the_combination_editors_with_a_draft_only(qapp, tmp_path, step):
    from test_combination_family import M27, _whole, pump

    from astro_gpu.client.graphnode import combine_kind
    w = _whole(tmp_path, M27)
    try:
        s = w.s
        q = next(q for q in s.paths if combine_kind(q) and
                 not all(i.get("image") == "stars" for i in q.source["fusion"]["inputs"]))
        s.select(index=s.index_of(q.pid), step=step)
        w.set_mode(1)
        settle(w)
        pump(3)
        ed = w.tune.comb_editor if step == "fusion" else w.tune.layers_editor
        assert ed.page_shown() and bar_of(w, step) is None
        if step == "fusion":
            ed.set_intensity("SII", 1.5)
        else:
            ed.set_slider("saturation", 3.0)
        pump(0.3)
        assert_bar_at_the_top(w, step)
        ed.reset()
        pump(0.3)
        assert bar_of(w, step) is None
    finally:
        w.s.pending_change = None
        w.close()
