"""The Combination card places any recipe of the application on the window's graph page (HOO here): its list is the
recipes', the block is titled and has the inputs the recipe gives, its panel shows them; saved, closed and opened
again the project file gives the same graph and the recipe's copy; a newer recipe is said in the block's tooltip and
by an "Update recipe" action of its panel, applied only on the click, one Undo step, no message box."""
from __future__ import annotations

import shutil

from PySide6.QtWidgets import QLabel, QPushButton

from astro_gpu import api
from astro_gpu.graphcore import combos
from astro_gpu.graphcore.model import model_json
from astro_gpu.stages import recipes
from tests.client.test_swap_smoke import env, made_from, qapp, ready, settle, win  # noqa: F401  (fixtures by name)


def _place(w, rid):
    page = w.graph
    pal = page.palette
    m = pal.combination_types(pal.cards["combine"], run=False)
    title = combos.type_title(rid)
    next(a for a in m.actions() if a.text() == title).trigger()
    assert pal.cards["combine"].title.text() == title
    before = {b for b, x in w.s.stored.blocks.items() if x.kind == "combine"}
    pal.double_click("combine")
    settle(w, 4)
    new = [b for b, x in w.s.stored.blocks.items() if x.kind == "combine" and b not in before]
    assert len(new) == 1
    return new[0]


def _panel_texts(w):
    return [x.text() for x in w.graph.panel.findChildren(QLabel) if x.isVisible()]


def test_hoo_is_placed_shown_saved_and_opened_again(win, env):
    root = made_from(win, env, "filters_min")
    b = _place(win, "hoo")
    blk = win.s.stored.blocks[b]
    assert blk.source == {"recipe": api.combination_recipe("hoo")}
    it = win.graph.canvas.blocks[b]
    assert it.d["title"] == "HOO" and [i[0] for i in it.d["inputs"]] == ["Hα", "O III"]
    assert "HOO (colour camera dual-band)" in it.d["tip"]
    win.graph.bridge.choose(b)
    settle(win, 3)
    texts = _panel_texts(win)
    from PySide6.QtWidgets import QComboBox
    recipe = next(x for x in win.graph.panel.findChildren(QComboBox) if x.objectName() == "recipe")
    assert recipe.currentText() == "HOO" and "<b>Hα</b>" in texts and "<b>O III</b>" in texts
    assert not any(x.text() == "Update recipe" for x in win.graph.panel.findChildren(QPushButton))
    q = next(q for q in win.s.paths if (q.source or {}).get("fusion", {}).get("type") == "hoo")
    assert win.s.p.workflow(q.pid).source == api.combination_source("hoo", [])
    win.save()
    settle(win)
    stored = model_json(win.s.stored)
    assert win.close_project(go_home=False)
    assert win.open_project(root)
    ready(win)
    assert model_json(win.s.stored) == stored
    q = next(q for q in win.s.paths if (q.source or {}).get("fusion", {}).get("type") == "hoo")
    assert win.s.p.workflow(q.pid).source["fusion"]["recipe"] == api.combination_recipe("hoo")
    assert not env["errors"], env["errors"]


def test_a_newer_recipe_is_said_and_applied_on_the_users_click(win, env, tmp_path, monkeypatch):
    made_from(win, env, "filters_min")
    b = _place(win, "hoo")
    d = tmp_path / "combinations"
    shutil.copytree(recipes.FOLDER, d)
    f = d / "hoo.toml"
    f.write_text(f.read_text(encoding="utf-8").replace("version = 1", "version = 2"), encoding="utf-8")
    monkeypatch.setattr(combos, "FOLDER", d)
    combos.refresh()
    try:
        win.graph.bridge.choose(b)
        win.graph.refresh()
        settle(win, 3)
        assert "A newer version of its recipe exists" in win.graph.canvas.blocks[b].d["tip"]
        assert win.s.stored.blocks[b].source["recipe"]["version"] == 1            # nothing applied by itself
        up = next(x for x in win.graph.panel.findChildren(QPushButton) if x.text() == "Update recipe")
        assert "version 2" in up.toolTip() and "version 1" in up.toolTip()
        steps = len(win.s._undo)
        up.click()
        settle(win, 3)
        assert win.s.stored.blocks[b].source["recipe"]["version"] == 2 and len(win.s._undo) == steps + 1
        win.s.undo(graph=True)
        settle(win, 3)
        assert win.s.stored.blocks[b].source["recipe"]["version"] == 1
    finally:
        monkeypatch.setattr(combos, "FOLDER", None)
        combos.refresh()
    assert not env["errors"], env["errors"]


def test_the_parameters_of_a_recipe_run_on_its_own_stage_are_set_in_the_panel(win, env, tmp_path, monkeypatch):
    """An H-O whose recipe the application changed (it no longer is the frozen H-O: it runs on the recipe stage):
    updated, its panel shows the recipe's parameters, by their names, and a change is stored in the block."""
    from PySide6.QtWidgets import QDoubleSpinBox
    made_from(win, env, "filters_min")
    b = _place(win, "ho")
    d = tmp_path / "combinations"
    shutil.copytree(recipes.FOLDER, d)
    f = d / "ho.toml"
    text = f.read_text(encoding="utf-8")
    assert text.count("version = 2") == 1 and text.count('max = 360\nlabel = "O III hue"') == 1
    f.write_text(text.replace("version = 2", "version = 3").replace('max = 360\nlabel = "O III hue"', 'max = 300\nlabel = "O III hue"'), encoding="utf-8")
    monkeypatch.setattr(combos, "FOLDER", d)
    combos.refresh()
    try:
        win.graph.bridge.choose(b)
        win.graph.refresh()
        settle(win, 3)
        next(x for x in win.graph.panel.findChildren(QPushButton) if x.text() == "Update recipe").click()
        settle(win, 3)
        assert combos.stage(win.s.stored.blocks[b]) == recipes.RECIPE_STAGE
        assert "Recipe" in _panel_texts(win) and "O III hue" in _panel_texts(win)
        spins = [x for x in win.graph.panel.findChildren(QDoubleSpinBox) if x.maximum() == 300.0]
        assert len(spins) == 1
        spins[0].setValue(190.0)
        spins[0].editingFinished.emit()
        settle(win, 3)
        assert win.s.stored.blocks[b].params[recipes.RECIPE_STAGE]["values"]["hue_oiii"] == 190.0
    finally:
        monkeypatch.setattr(combos, "FOLDER", None)
        combos.refresh()
    assert not env["errors"], env["errors"]
