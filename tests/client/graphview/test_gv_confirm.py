"""A change that would compute again results computed and up to date waits for Replace / Try variant / Cancel (the
owner's rule, every block): on the panel at its foot, on the graph in the page's toast; nothing in the project changes
while it waits; several edits of a block are asked once; another block chosen meanwhile asks first; the preference
"Confirm changes" off applies at once; after Replace, Ctrl+Z gives the results back at once. On the computed test
project of test_gv_legacy_model (its fixtures and helpers)."""
from __future__ import annotations

import os
from pathlib import Path

from PySide6.QtWidgets import QLabel

from astro_gpu.client import prefs
from astro_gpu.client.graphmodel import ASKED
from astro_gpu.client.graphview import bridge as bridge_mod
from tests.client.graphview.test_gv_legacy_model import (  # noqa: F401  (the fixtures are used by name)
    _edit_kappa,
    block_of,
    button,
    computed_but_stars,
    env,
    live_rows,
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
SHOTS = os.environ.get("FLOAT_CONFIRM_SHOTS")      # a folder: the two captures of the question are written there


def _states(m):
    return {k: v.state for k, v in m._states.items()}


def _edit(page, key, value):
    row = next(r for r in live_rows(page) if r.r["key"] == key)
    page.panel.set_setting(row.r, value)
    settle(page.s.worker, page, 3)
    return row.r


def test_the_reason_word_is_the_same_on_both_sides():
    assert bridge_mod.ASKED == ASKED


def test_a_panel_change_waits_and_nothing_changes_until_its_answer(world, model, page):
    select_node(page, node(page, "empilement"))
    settle(world["w"], page)
    undo, states, run = len(model._undo), _states(model), model.total_seconds()
    _edit_kappa(page, 4.0)
    settle(world["w"], page)
    ch = model.pending_change
    assert ch and ch["cost"].lost
    assert len(model._undo) == undo and _states(model) == states and model.total_seconds() == run
    assert all((q.params.get("stack") or {}).get("kappa") != 4.0 for q in model.paths)
    said = page.panel.findChild(QLabel, "choice_cost").text()
    assert said.startswith("Computes again: Stacking, Astrometry, Gradient") and "min" in said
    kappa = next(r for r in live_rows(page) if r.r["key"] == "stack.kappa")
    assert kappa.r["value"] == 4.0                              # what was typed stays shown while it waits
    assert page.panel.findChild(QLabel, "choice_never") is not None
    if SHOTS:
        Path(SHOTS).mkdir(parents=True, exist_ok=True)
        page.panel.grab().save(str(Path(SHOTS) / "confirm_panel.png"))
    assert page.answer_change("cancel")                         # Esc
    settle(world["w"], page)
    assert model.pending_change is None and len(model._undo) == undo and _states(model) == states


def test_several_edits_of_a_block_are_asked_once_and_undone_at_once(world, model, page):
    select_node(page, node(page, "empilement"))
    settle(world["w"], page)
    undo = len(model._undo)
    page.panel.expert = True
    _edit_kappa(page, 4.0)
    pix = next(r for r in live_rows(page) if r.r["key"] == "stack.pixfrac")
    new_pix = 0.7 if pix.r["value"] != 0.7 else 0.6
    _edit(page, "stack.pixfrac", new_pix)
    _edit(page, "stack.kappa", 3.5)                             # the same setting again: the later value
    ch = model.pending_change
    assert ch["kind"] == "many" and len(ch["changes"]) == 2 and len(ch["whats"]) == 2
    assert [c["value"] for c in ch["changes"]] == [3.5, new_pix]
    assert len(page.panel.findChildren(QLabel, "choice_cost")) == 1 and len(model._undo) == undo
    told = []
    page.toasted.connect(told.append)
    button(page.panel, "Replace").click()
    settle(world["w"], page)
    assert len(model._undo) == undo + 1                         # one step for both
    assert all(q.params["stack"]["kappa"] == 3.5 and q.params["stack"]["pixfrac"] == new_pix for q in model.paths)
    assert model.state(model.paths[0], "stack") == "stale"
    page.bridge.undo()                                          # Ctrl+Z: the values back, the blocks up to date at once
    settle(world["w"], page)
    assert computed_but_stars(model, model.paths[0]) and model.state(model.paths[0], "stack") == "done"
    assert any("results up to date again" in t for t in told), told
    page.panel.expert = False


def test_another_block_chosen_while_a_change_waits_asks_first(world, model, page):
    stack = block_of(page, node(page, "empilement"))
    select_node(page, node(page, "empilement"))
    settle(world["w"], page)
    _edit_kappa(page, 4.0)
    settle(world["w"], page)
    other = block_of(page, node(page, "fond", 1))
    page.bridge.choose(other)
    settle(world["w"], page)
    assert page.bridge.selected == stack and model.pending_change is not None    # the choice stays, nothing lost
    assert page.panel.findChild(QLabel, "choice_first") is not None
    button(page.panel, "Cancel").click()
    settle(world["w"], page)
    assert model.pending_change is None and page.bridge.selected == other       # then the block chosen is shown


def test_a_graph_gesture_waits_in_the_toast_and_replace_is_undone_up_to_date(world, model, page):
    den = block_of(page, node(page, "fond", 1))          # Gradient: switched off, the blocks after it change
    page.bridge.choose(den)
    settle(world["w"], page)
    undo, states, rev = len(model._undo), _states(model), model.revision
    res = page.bridge.act("switch_off", block=den)
    settle(world["w"], page)
    assert not res.ok and res.reason == ASKED and model.pending_gesture is not None
    assert len(model._undo) == undo and _states(model) == states and model.revision == rev
    assert not page.bridge.graph.blocks[den].off
    toast = page.canvas.toast
    assert toast.isVisible() and "Gradient" in toast.plain and "Computes again: Denoise, Color" in toast.plain
    assert [k for _t, k in toast.links()] == ["confirm_replace", "confirm_variant", "confirm_cancel", "confirm_never"]
    if SHOTS:
        Path(SHOTS).mkdir(parents=True, exist_ok=True)
        page.root.grab().save(str(Path(SHOTS) / "confirm_graph.png"))
    toast.link.emit("confirm_cancel")
    settle(world["w"], page)
    assert model.pending_gesture is None and len(model._undo) == undo and not page.bridge.graph.blocks[den].off
    page.bridge.act("switch_off", block=den)                    # again: Enter answers Try variant
    settle(world["w"], page)
    n = len(model.paths)
    assert page.answer_change("default")
    settle(world["w"], page)
    assert len(model.paths) == n + 1 and not page.bridge.graph.blocks[den].off
    assert computed_but_stars(model, model.view(1))             # the computed chain kept
    model.undo()
    settle(world["w"], page)
    page.bridge.act("switch_off", block=den)
    settle(world["w"], page)
    told = []
    page.toasted.connect(told.append)
    toast.link.emit("confirm_replace")
    settle(world["w"], page)
    assert page.bridge.graph.blocks[den].off and model.state(model.view(1), "color") == "stale"
    assert any("Ctrl+Z gives them back" in t for t in told), told
    page.bridge.undo()
    settle(world["w"], page)
    assert not page.bridge.graph.blocks[den].off and computed_but_stars(model, model.view(1))
    assert any("results up to date again" in t for t in told), told


def test_the_preference_off_applies_at_once(world, model, page):
    prefs.set("confirm_changes", False)
    try:
        select_node(page, node(page, "empilement"))
        settle(world["w"], page)
        undo = len(model._undo)
        _edit_kappa(page, 4.0)
        settle(world["w"], page)
        assert model.pending_change is None and len(model._undo) == undo + 1
        assert all(q.params["stack"]["kappa"] == 4.0 for q in model.paths)
        model.undo()
        settle(world["w"], page)
        den = block_of(page, node(page, "fond", 1))          # Gradient: switched off, the blocks after it change
        res = page.bridge.act("switch_off", block=den)
        assert res.ok and model.pending_gesture is None and page.bridge.graph.blocks[den].off
    finally:
        prefs.set("confirm_changes", True)


def test_do_not_ask_turns_the_preference_off_and_says_where(world, model, page):
    den = block_of(page, node(page, "fond", 1))          # Gradient: switched off, the blocks after it change
    page.bridge.act("switch_off", block=den)
    settle(world["w"], page)
    try:
        page.canvas.toast.link.emit("confirm_never")
        assert prefs.get("confirm_changes") is False and model.pending_gesture is not None   # the question waits
        assert "Settings › General" in page.canvas.toast.plain
        page.answer_change("cancel")
        assert model.pending_gesture is None and not page.bridge.graph.blocks[den].off
    finally:
        prefs.set("confirm_changes", True)


def test_no_question_when_nothing_up_to_date_is_lost(world, model, page):
    classic = model.view(2)                                     # computed but its Output
    n = node(page, "post", 2)
    assert model.state(classic, "render_simple") != "done" and n.paths == [classic]
    ch = dict(kind="param", stage="render_simple", name="linked", value=True, suggested=None)
    assert not model.change_cost(ch, n.paths).lost
    select_node(page, n)
    settle(world["w"], page)
    undo = len(model._undo)
    page.panel.propose("render_simple", "linked", True, None, "Linked: no → yes", "linked")
    settle(world["w"], page)
    assert model.pending_change is None and len(model._undo) == undo + 1
    model.undo()
    settle(world["w"], page)
    # a pause changes no key: no question
    assert not model.change_cost([("set_pause", {"block": block_of(page, node(page, "fond", 1))})]).lost
