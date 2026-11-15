"""The Combination card shows the recipe it places and its list on the card (a beta tester double-clicked it twice and
got S-H-O both times, never seeing RGB); the block's panel changes its recipe afterwards, its default name following;
an input's list greys a workflow of a Session alone with the reason, never « no workflow » once picked."""
from __future__ import annotations

from PySide6.QtWidgets import QComboBox

from tests.client.test_swap_smoke import env, made_from, qapp, ready, settle, win  # noqa: F401  (fixtures by name)


def test_the_card_shows_its_recipe_and_the_panel_changes_it(win, env):
    made_from(win, env, "filters_min")
    page = win.graph
    card = page.palette.cards["combine"]
    assert card.title.text() == "S-H-O" and card.choose is not None and card.choose.isVisibleTo(card)
    m = page.palette.combination_types(card, run=False)
    assert {"S-H-O", "RGB", "LRGB"} <= {a.text() for a in m.actions()}
    before = {b for b, x in win.s.stored.blocks.items() if x.kind == "combine"}
    page.palette.double_click("combine")
    settle(win, 4)
    j = next(b for b, x in win.s.stored.blocks.items() if x.kind == "combine" and b not in before)
    page.bridge.choose(j)
    settle(win, 3)
    cb = next(x for x in page.panel.findChildren(QComboBox) if x.objectName() == "recipe")
    assert cb.currentText() == "S-H-O" and cb.findText("RGB") >= 0
    row = next(r for r in win.s.stored.rows if r.blocks and r.blocks[0] == j)
    name0 = row.name
    cb.activated.emit(cb.findText("RGB"))
    settle(win, 3)
    blk = win.s.stored.blocks[j]
    assert blk.settings.get("type") == "rgb"
    row = next(r for r in win.s.stored.rows if r.blocks and r.blocks[0] == j)
    assert row.name.startswith("RGB") and row.name != name0
    win.s.undo()
    settle(win, 3)
    assert win.s.stored.blocks[j].settings.get("type", "sho") == "sho"
    assert not env["errors"], env["errors"]


def test_an_input_list_greys_a_session_alone(win, env):
    made_from(win, env, "filters_min")
    s, page = win.s, win.graph
    page.bridge.act("add_row", kind="session")                 # a workflow of a Session alone: no image yet
    settle(win, 3)
    page.palette.double_click("combine")
    settle(win, 4)
    p = next(q for q in s.paths if (q.source or {}).get("fusion"))
    choices = dict((q.pid, why) for q, why in s.input_choices(p, "SII"))
    lone = [q for q in s.paths if not s._chain(q.pid) or all(
        s.stored.blocks[b].kind in ("session", "calibration") for b in s._chain(q.pid))]
    assert lone and all(choices.get(q.pid) == "no image yet: place its Stacking first" for q in lone), choices
    assert not env["errors"], env["errors"]
