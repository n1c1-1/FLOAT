"""One way to set a block (blockdraft.py), the Gradient view's: on a block with computed results a change of setting is
a DRAFT of the block, shown at once (the radio checked, the value typed), nothing outdated, nothing recomputed; one bar
pinned at the panel's foot says what changed (old → new), what Apply computes again and how long, with Apply, Try
variant and Reset. Leaving the block asks Apply / Try variant / Discard / Stay; a block never computed has no draft;
the preference "Confirm changes" off applies at once. A gesture of the graph never waits: it applies, its toast says
what it made stale ("Computes again") or added ("To compute"), with Undo. On the computed test project of
test_gv_legacy_model (its fixtures and helpers)."""
from __future__ import annotations

import os
from pathlib import Path

import pytest
from PySide6.QtWidgets import QLabel

from astro_gpu.client import prefs
from astro_gpu.client.blockdraft import DRAFT_ANSWER_ENV, DraftBar
from tests.client.graphview.test_gv_legacy_model import (  # noqa: F401  (the fixtures are used by name)
    _edit_kappa,
    block_of,
    button,
    computed_but_stars,
    env,
    live_rows,
    method_cards,
    model,
    needs_test_project,
    node,
    page,
    select_node,
    settle,
    texts_of,
    world,
)

pytestmark = needs_test_project
SHOTS = os.environ.get("FLOAT_CONFIRM_SHOTS")      # a folder: the captures of the bar are written there


def _states(m):
    return {k: v.state for k, v in m._states.items()}


def _edit(page, key, value):
    row = next(r for r in live_rows(page) if r.r["key"] == key)
    page.panel.set_setting(row.r, value)
    settle(page.s.worker, page, 3)
    return row.r


def _bar(page) -> DraftBar | None:
    bars = [b for b in page.panel.findChildren(DraftBar) if b.isVisibleTo(page.panel)]
    assert len(bars) <= 1
    return bars[0] if bars else None


def _change(page):
    return page.panel.draft_change(page.s.pending_change)


def _denoise_other(page):
    cards = method_cards(page)
    on = [c for c in cards if c.on]
    other = next(c for c in cards if not c.on and c.ok)
    return on[0], other


def test_a_click_on_denoise_method_checks_it_at_once_and_outdates_nothing(world, model, page):
    select_node(page, node(page, "debruitage", 1))
    settle(world["w"], page)
    undo, states, run = len(model._undo), _states(model), model.total_seconds()
    was, other = _denoise_other(page)
    name, was_name = other.findChildren(QLabel)[0].text(), was.findChildren(QLabel)[0].text()
    other.clicked.emit()
    settle(world["w"], page)
    now = [c for c in method_cards(page) if c.on]
    assert len(now) == 1 and now[0].findChildren(QLabel)[0].text() == name     # checked at once
    assert len(model._undo) == undo and _states(model) == states and model.total_seconds() == run
    assert model.view(1).opts["dn"] != _change(page)["value"]                   # the workflow is unchanged
    bar = _bar(page)
    assert bar is not None and bar.b_apply.text() == "Apply" and bar.b_variant.text() == "Try variant" \
        and bar.b_reset.text() == "Reset"
    assert "Method:" in bar.what.text() and "→" in bar.what.text()
    assert bar.cost.text().startswith("Computes again: Denoise") and "min" in bar.cost.text()
    assert not any(t in ("Replace", "Cancel", "Keep", "Discard", "Do not ask") for t in texts_of(page.panel))
    # pinned: the bar is the panel's last widget, outside its scroll area
    assert bar.parentWidget().parentWidget() is page.panel and not page.panel._sc.isAncestorOf(bar)
    if SHOTS:
        Path(SHOTS).mkdir(parents=True, exist_ok=True)
        page.panel.grab().save(str(Path(SHOTS) / "draft_panel.png"))
    bar.b_reset.click()                                     # Reset: back to the applied method, nothing changed
    settle(world["w"], page)
    assert not model.pending_change and _bar(page) is None and _states(model) == states
    assert [c for c in method_cards(page) if c.on][0].findChildren(QLabel)[0].text() == was_name


def test_several_edits_are_listed_on_the_bar_and_applied_as_one_step(world, model, page):
    select_node(page, node(page, "empilement"))
    settle(world["w"], page)
    undo = len(model._undo)
    page.panel.expert = True
    _edit_kappa(page, 4.0)
    pix = next(r for r in live_rows(page) if r.r["key"] == "stack.pixfrac")
    new_pix = 0.7 if pix.r["value"] != 0.7 else 0.6
    _edit(page, "stack.pixfrac", new_pix)
    _edit(page, "stack.kappa", 3.5)                             # the same setting again: the later value
    ch = _change(page)
    assert ch["kind"] == "many" and len(ch["changes"]) == 2 and [c["value"] for c in ch["changes"]] == [3.5, new_pix]
    said = _bar(page).what.text()
    assert "Rejection threshold: 3 → 3.5" in said and "Drop size: 1 → " in said.split("<br>")[1], said
    assert next(r for r in live_rows(page) if r.r["key"] == "stack.kappa").r["value"] == 3.5     # shown at once
    told = []
    page.toasted.connect(told.append)
    _bar(page).b_apply.click()
    settle(world["w"], page)
    assert len(model._undo) == undo + 1                         # one step for both
    assert all(q.params["stack"]["kappa"] == 3.5 and q.params["stack"]["pixfrac"] == new_pix for q in model.paths)
    assert model.state(model.paths[0], "stack") == "stale" and _bar(page) is None
    page.bridge.undo()                                          # Ctrl+Z: the values back, the blocks up to date at once
    settle(world["w"], page)
    assert computed_but_stars(model, model.paths[0]) and model.state(model.paths[0], "stack") == "done"
    assert any("results up to date again" in t for t in told), told
    page.panel.expert = False


def test_a_setting_back_to_its_value_leaves_the_draft(world, model, page):
    select_node(page, node(page, "empilement"))
    settle(world["w"], page)
    r = _edit_kappa(page, 4.0)
    assert model.pending_change and _bar(page) is not None
    _edit(page, "stack.kappa", r["applied"])
    assert not model.pending_change and _bar(page) is None


def test_the_draft_steps_are_undone_one_by_one(world, model, page):
    select_node(page, node(page, "empilement"))
    settle(world["w"], page)
    page.panel.expert = True
    _edit_kappa(page, 4.0)
    _edit(page, "stack.kappa", 4.5)
    d = model.pending_change
    assert d.top("undo") is not None and len(d.undo) == 2
    d.move("undo")
    model.draftChanged.emit()
    settle(world["w"], page)
    assert next(r for r in live_rows(page) if r.r["key"] == "stack.kappa").r["value"] == 4.0
    d.move("undo")
    model.draftChanged.emit()
    settle(world["w"], page)
    assert not model.pending_change and _bar(page) is None          # no draft left: the applied value
    d.move("redo")
    model.draftChanged.emit()
    settle(world["w"], page)
    assert _bar(page) is not None
    _bar(page).b_reset.click()
    page.panel.expert = False


def test_try_variant_keeps_the_original_computed(world, model, page):
    select_node(page, node(page, "debruitage", 1))
    settle(world["w"], page)
    n = len(model.paths)
    _was, other = _denoise_other(page)
    other.clicked.emit()
    settle(world["w"], page)
    _bar(page).b_variant.click()
    settle(world["w"], page)
    assert len(model.paths) == n + 1 and not model.pending_change
    q = model.paths[-1]
    assert q.variant_of == 1 and computed_but_stars(model, model.view(1))
    assert not any("Variant of" in t for t in texts_of(page.panel))          # a variant is a row like another
    model.undo()
    settle(world["w"], page)


@pytest.mark.parametrize("answer", ["stay", "discard", "apply", "variant"])
def test_leaving_a_block_with_a_draft_asks_and_each_answer_does_what_it_says(world, model, page, monkeypatch, answer):
    monkeypatch.setenv(DRAFT_ANSWER_ENV, answer)
    stack = block_of(page, node(page, "empilement"))
    select_node(page, node(page, "empilement"))
    settle(world["w"], page)
    n, undo, states = len(model.paths), len(model._undo), _states(model)
    _edit_kappa(page, 4.0)
    settle(world["w"], page)
    other = block_of(page, node(page, "fond", 1))
    page.bridge.choose(other)
    settle(world["w"], page)
    if answer == "stay":                                    # the block stays chosen, its draft as it was
        assert page.bridge.selected == stack and model.pending_change and _bar(page) is not None
        page.panel.draft_reset()
        return
    assert page.bridge.selected == other and not model.pending_change    # never a draft left behind
    if answer == "discard":
        assert len(model._undo) == undo and _states(model) == states and len(model.paths) == n
    elif answer == "apply":
        assert len(model._undo) == undo + 1 and all(q.params["stack"]["kappa"] == 4.0 for q in model.paths)
    else:
        assert len(model.paths) == n + 1 and model.paths[-1].params["stack"]["kappa"] == 4.0
        assert computed_but_stars(model, model.view(1))
    page.panel.expert = False


def test_no_draft_on_a_block_never_computed(world, model, page):
    classic = model.view(2)                                     # computed up to its Color, its Output never
    n = node(page, "post", 2)
    assert model.state(classic, "render_simple") != "done" and n.paths == [classic]
    select_node(page, n)
    settle(world["w"], page)
    undo = len(model._undo)
    page.panel.propose("render_simple", "linked", True, None, "Linked: no → yes", "linked")
    settle(world["w"], page)
    assert not model.pending_change and _bar(page) is None and len(model._undo) == undo + 1      # applied at once
    model.undo()
    settle(world["w"], page)


def test_the_preference_off_applies_a_setting_at_once(world, model, page):
    prefs.set("confirm_changes", False)
    try:
        select_node(page, node(page, "empilement"))
        settle(world["w"], page)
        undo = len(model._undo)
        _edit_kappa(page, 4.0)
        settle(world["w"], page)
        assert not model.pending_change and _bar(page) is None and len(model._undo) == undo + 1
        assert all(q.params["stack"]["kappa"] == 4.0 for q in model.paths)
        model.undo()
        settle(world["w"], page)
    finally:
        prefs.set("confirm_changes", True)


def test_deleting_a_variants_original_leaves_the_variant_whole_and_computed(world, model, page):
    """A variant is removed by deleting it, and preferred by deleting its original: the original's row deleted, the
    variant keeps the blocks it shared and its results, nothing is outdated."""
    select_node(page, node(page, "empilement", 1))
    _edit_kappa(page, 3.5)
    settle(world["w"], page)
    _bar(page).b_variant.click()
    settle(world["w"], page)
    q = model.paths[-1]
    assert q.variant_of == 1
    done_before = {st for st in model.all_stages(q) if model.state(q, st) == "done"}
    keys = {st: model.stage_key(q, st) for st in model.all_stages(q)}
    full_row = next(r for r in page.bridge.graph.rows if r.wid == 1)          # the original's own row
    v_row = next(r for r in page.bridge.graph.rows if r.wid == q.pid)
    assert v_row.trial == 1
    res = page.bridge.act("delete_row", row=full_row.id)
    settle(world["w"], page, 20)
    assert res.ok, res.reason
    v = model.view(q.pid)
    assert v is not None and model.view(1) is None, (q.pid, [(x.pid, x.name) for x in model.paths], res.toast)
    assert {st: model.stage_key(v, st) for st in model.all_stages(v)} == keys        # nothing outdated
    assert {st for st in model.all_stages(v) if model.state(v, st) == "done"} == done_before
    assert {"calibrate", "register"} <= done_before
    model.undo()
    settle(world["w"], page)
    model.undo()
    settle(world["w"], page)


def _toast(page):
    t = page.canvas.toast
    return t.plain if t.isVisible() else ""


def test_a_graph_gesture_applies_at_once_says_what_it_made_stale_and_undo_gives_it_back(world, model, page):
    grad = block_of(page, node(page, "fond", 1))          # Gradient: switched off, the blocks after it change
    page.bridge.choose(grad)
    settle(world["w"], page)
    undo = len(model._undo)
    res = page.bridge.act("switch_off", block=grad)
    settle(world["w"], page)
    assert res.ok and page.bridge.graph.blocks[grad].off and len(model._undo) == undo + 1   # applied: no question
    assert model.state(model.view(1), "color") == "stale"
    said = _toast(page)
    assert "switched off" in said and "Computes again: Denoise, Color" in said and "Output" not in said, said
    assert [k for _t, k in page.canvas.toast.links()] == ["undo"]
    assert "Replace" not in said and "Do not ask" not in said
    told = []
    page.toasted.connect(told.append)
    page.canvas.toast.link.emit("undo")                     # its Undo: the block back, its results up to date at once
    settle(world["w"], page)
    assert not page.bridge.graph.blocks[grad].off and computed_but_stars(model, model.view(1))
    assert any("results up to date again" in t for t in told), told


def _place(g, kind, row, after):
    pls = g.places(kind=kind)
    return next(i for i, p in enumerate(pls) if p.lit and p.op[0] == "add_block"
                and (p.op[1].get("at") or {}).get("row") == row and (p.op[1].get("at") or {}).get("after") == after)


def test_a_block_appended_after_computed_blocks_says_what_is_to_compute(world, model, page):
    """Classic stack is computed up to its Color, its automatic Output never computed: a Deconvolution dragged to its
    end only adds work. The drop places the block (a real drag), and the toast says what is now to compute."""
    from astro_gpu.client.graphview.gestures import Gestures
    g = Gestures(page)
    colour = block_of(page, node(page, "couleur", 2))
    row = page.bridge.row_of(colour).id
    i = _place(g, "deconv", row, colour)
    n = len(page.bridge.graph.blocks)
    g.drag_palette("deconv", i)
    settle(world["w"], page)
    assert len(page.bridge.graph.blocks) == n + 1
    said = _toast(page)
    assert said.startswith("Deconvolution added to “Classic stack”"), said
    assert "To compute: Deconvolution" in said and "Computes again" not in said and "Output" not in said, said
    model.undo()
    settle(world["w"], page)


def test_a_block_inserted_before_computed_blocks_names_them(world, model, page):
    from astro_gpu.client.graphview.gestures import Gestures
    g = Gestures(page)
    grad = block_of(page, node(page, "fond", 1))
    row = page.bridge.row_of(grad).id
    i = _place(g, "deconv", row, grad)
    g.drag_palette("deconv", i)
    settle(world["w"], page)
    said = _toast(page)
    assert said.startswith("Deconvolution added to “Full”") and "here" not in said, said    # never the bubble's words
    assert "Computes again: Denoise" in said and "Output" not in said, said
    assert [k for _t, k in page.canvas.toast.links()] == ["undo"]
    model.undo()
    settle(world["w"], page)
    assert computed_but_stars(model, model.view(1))


def test_a_drop_on_a_lit_place_always_places_the_block(world, model, page):
    """Every lit place of a Deconvolution and a Gradient, with a real drag, on the computed project: the block is
    there (never a question, never nothing)."""
    from astro_gpu.client.graphview.gestures import Gestures
    g = Gestures(page)
    for kind in ("deconv", "gradient"):
        for i, pl in enumerate(g.places(kind=kind)):
            if not pl.lit:
                continue
            n = sum(1 for b in page.bridge.graph.blocks.values() if b.kind == kind)
            page.canvas.toast.plain = ""      # what the toast said, kept once it is hidden: a loaded machine settles
            g.drag_palette(kind, i)              # longer than a toast stays shown
            settle(world["w"], page, 4)
            m = sum(1 for b in page.bridge.graph.blocks.values() if b.kind == kind)
            assert m > n, (kind, i, page.bridge.say(pl.tip), _toast(page))       # (a branch copies the blocks after)
            assert page.canvas.toast.plain
            page.bridge.undo()
            settle(world["w"], page, 4)


def test_no_draft_when_nothing_up_to_date_is_lost(world, model, page):
    full = model.view(1)
    ch = dict(kind="param", stage="render_simple", name="linked", value=True, suggested=None)
    # Full's Output is computed, but it is the compile's own node: a change only it would compute again asks nothing
    assert model.state(full, "render_simple") == "done"
    assert not model.change_cost(dict(ch), [full]).lost
    # a pause changes no key: nothing
    assert not model.change_cost([("set_pause", {"block": block_of(page, node(page, "fond", 1))})]).lost


def test_an_undo_or_a_redo_is_costed_as_a_gesture(world, model, page):
    """The window asks before an Undo or a Redo that would compute again results computed and up to date (the testers'
    pass 2: one Ctrl+Z after a run went back before it, every block Outdated, no question): ``history_cost`` reads it as
    a gesture's cost. An Undo that gives results back loses nothing."""
    grad = block_of(page, node(page, "fond", 1))
    assert not model.history_cost("undo").lost                 # the steps before the run: nothing computed since
    res = page.bridge.act("switch_off", block=grad)
    settle(world["w"], page)
    assert res.ok and not model.history_cost("undo").lost     # its Undo gives the results back
    page.bridge.undo()
    settle(world["w"], page)
    assert computed_but_stars(model, model.view(1))
    cost = model.history_cost("redo")
    assert cost.lost and "Denoise" in cost.blocks and "Color" in cost.blocks and "Output" not in cost.text()
    page.bridge.redo()
    settle(world["w"], page)
    page.bridge.undo()
    settle(world["w"], page)
    assert computed_but_stars(model, model.view(1))
