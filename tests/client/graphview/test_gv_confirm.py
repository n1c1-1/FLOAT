"""A change of setting in the panel that would compute again results computed and up to date waits at the panel's foot
for Replace / Try variant / Cancel; nothing in the project changes while it waits; several edits of a block are asked
once; another block chosen meanwhile asks first; the preference "Confirm changes" off applies at once; after Replace,
Ctrl+Z gives the results back at once. A gesture of the graph never waits: it applies (the gesture is the validation)
and its toast says what it made stale ("Computes again") or added ("To compute"), with Undo; the compile's own nodes
(the automatic Output) never count. On the computed test project of test_gv_legacy_model (its fixtures and
helpers)."""
from __future__ import annotations

import os
from pathlib import Path

from PySide6.QtWidgets import QLabel

from astro_gpu.client import prefs
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
    if SHOTS:
        Path(SHOTS).mkdir(parents=True, exist_ok=True)
        page.root.grab().save(str(Path(SHOTS) / "confirm_graph.png"))
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
            g.drag_palette(kind, i)
            settle(world["w"], page, 4)
            m = sum(1 for b in page.bridge.graph.blocks.values() if b.kind == kind)
            assert m > n, (kind, i, page.bridge.say(pl.tip), _toast(page))       # (a branch copies the blocks after)
            assert _toast(page)
            page.bridge.undo()
            settle(world["w"], page, 4)


def test_the_preference_off_applies_a_setting_at_once(world, model, page):
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
    finally:
        prefs.set("confirm_changes", True)


def test_do_not_ask_turns_the_preference_off_and_says_where(world, model, page):
    select_node(page, node(page, "empilement"))
    settle(world["w"], page)
    _edit_kappa(page, 4.0)
    settle(world["w"], page)
    told = []
    page.toasted.connect(told.append)
    try:
        page.panel.findChild(QLabel, "choice_never").linkActivated.emit("never")
        assert prefs.get("confirm_changes") is False and model.pending_change is not None    # the question waits
        assert any("Settings › General" in t for t in told), told
        button(page.panel, "Cancel").click()
        settle(world["w"], page)
        assert model.pending_change is None
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
    # Full's Output is computed, but it is the compile's own node: a change only it would compute again asks nothing
    full = model.view(1)
    assert model.state(full, "render_simple") == "done"
    assert not model.change_cost(dict(ch), [full]).lost
    # a pause changes no key: nothing
    assert not model.change_cost([("set_pause", {"block": block_of(page, node(page, "fond", 1))})]).lost
