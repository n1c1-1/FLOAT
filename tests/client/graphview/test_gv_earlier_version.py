"""A block whose result was computed with an earlier version of its method: it stays "Up to date", with a small mark
and one line in its tooltip, "Computed with an earlier version (5 -> 6)"; no message box, no banner. Its menu and its
panel have a visible command, "Recompute", which opens the run's plan (what it computes again and
how long) before anything runs. A block made from a result recomputed since is Outdated and says why. On a graph held
in memory; the panel and the plan on the computed test project (test_real_*)."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from PySide6.QtWidgets import QMessageBox

from astro_gpu import api
from conftest import full_only, needs_test_project, scratch_project  # (tests/conftest.py)

EN = "Computed with an earlier version (5 -> 6)"
FR = "Calculé avec une version antérieure (5 -> 6)"
CMD_EN, CMD_FR = "Recompute", "Recalculer"


def _page(doc):
    from astro_gpu.client.graphview.gestures import pump
    from astro_gpu.client.graphview.page import GraphPage
    p = GraphPage(doc)
    p.resize(1400, 800)
    p.show()
    pump()
    return p


def _doc():
    from astro_gpu.client.graphview.document import MemoryDocument
    from tests.graphcore import fixtures

    class Badged(MemoryDocument):
        given = {"gradient": {"earlier": [("5", "6")]}}

        def badges(self, b):
            return self.given.get(self.graph.blocks[b].kind, {})
    h = fixtures.Doc.from_template("colour_minimal")
    g = MemoryDocument(h.project, h.view)
    return Badged(g.graph, g.view, states={b: "done" for b in g.graph.blocks})


@pytest.fixture
def boxes(monkeypatch):
    shown = []
    for kind in ("information", "warning", "critical", "question"):
        monkeypatch.setattr(QMessageBox, kind, lambda *a, _k=kind, **k: shown.append(_k))
    return shown


def test_the_block_says_it_discreetly_and_stays_up_to_date(qapp, boxes):
    from astro_gpu.client.graphview.gestures import pump
    p = _page(_doc())
    by = {it.d["kind"]: it for it in p.canvas.blocks.values()}
    gr, st = by["gradient"], by["stacking"]
    assert gr.d["earlier"] == [("5", "6")] and gr.d["state"] == "done"
    tip = gr.toolTip().split("\n")
    assert EN in tip and "Earlier version" in tip and "Up to date" not in tip   # one truth: not up to date
    assert not st.d["earlier"] and "earlier version" not in st.toolTip()
    gr.update()                                                 # painted with its mark (no exception)
    p.canvas.viewport().repaint()
    pump()
    assert gr.earlier_rect().isValid() and not boxes
    # amber, as a block that needs an update (the reference node editors): the band, with its reason and its action
    from astro_gpu.client.graphview.style import STATE_COL
    assert gr.band_colour() == STATE_COL["earlier"] != STATE_COL["done"] and st.band_colour() == STATE_COL["done"]
    assert "Recompute to bring it up to date" in gr.toolTip()
    banner = getattr(p, "banner", None)
    assert banner is None or not banner.isVisible()
    p.close()


def test_its_menu_has_recompute_current_version_which_asks_the_window_and_changes_nothing(qapp, boxes):
    from astro_gpu.client.graphview.menu import menu
    from astro_gpu.graphcore.operations import json_of
    p = _page(_doc())
    ids = {it.d["kind"]: b for b, it in p.canvas.blocks.items()}
    items = {e[0]: e for e in menu(p.canvas.doc, ids["gradient"]) if e}
    assert CMD_EN in items and items[CMD_EN][1] == "recompute" and items[CMD_EN][3]
    assert CMD_EN not in {e[0] for e in menu(p.canvas.doc, ids["stacking"]) if e}      # only where it applies
    asked, before = [], json_of(p.bridge.graph)
    p.view_action.connect(lambda op, ident: asked.append((op, ident)))
    p.canvas.run_menu(ids["gradient"], "recompute", {"block": ids["gradient"]})
    assert asked == [("recompute", ids["gradient"])] and json_of(p.bridge.graph) == before and not boxes
    p.close()


def test_the_words_are_french(qapp):
    from astro_gpu.client import app as appmod
    from astro_gpu.client.graphview.menu import menu
    appmod.apply_language("fr")
    try:
        p = _page(_doc())
        ids = {it.d["kind"]: b for b, it in p.canvas.blocks.items()}
        assert FR in p.canvas.blocks[ids["gradient"]].toolTip().split("\n")
        assert CMD_FR in {e[0] for e in menu(p.canvas.doc, ids["gradient"]) if e}
        from astro_gpu.client.i18n import _
        assert _("the block before it was recomputed") == "le bloc qui le précède a été recalculé"
        assert _("Recompute “{block}” (current version)").format(block="Gradient") == \
            "Recalculer « Gradient » (version actuelle)"
        # a block that failed on a result today's code can no longer read: what to do, the program's error left out
        from astro_gpu.client.i18n import is_raw_error
        from astro_gpu.client.services.run import engine_text
        from astro_gpu.core.runner2 import UNREADABLE
        msg = UNREADABLE.format(title="Stars/background", old="5", new="6", error="KeyError: 'model'")
        assert not is_raw_error(msg) and engine_text(msg) == (
            "le résultat du bloc Étoiles/fond a été calculé avec une version antérieure (5 -> 6), que cette version "
            "ne sait peut-être plus lire : recalculez-le avec « Recalculer » sur ce bloc")
        appmod.apply_language("en")
        assert engine_text(msg) == (
            "the result of the Stars/background block was computed with an earlier version (5 -> 6), which this "
            "version may no longer read: recompute it with “Recompute” on that block")
        p.close()
    finally:
        appmod.apply_language("en")


def test_a_block_s_earlier_versions_are_those_of_its_done_stages():
    from astro_gpu.client.graphnode import block_earlier, earlier_words
    info = {"gradient": api.StageState("done", "", "k1", earlier=("5", "6")),
            "surveys": api.StageState("done", "", "k2"),
            "color": api.StageState("stale", "x", "k3", "k3", earlier=("1", "2"))}
    m = SimpleNamespace(state_info=lambda p, s: info.get(s), state=lambda p, s: info[s].state,
                        all_stages=lambda p: list(info))
    p = SimpleNamespace(pid=1, params={}, opts={})
    assert block_earlier(m, p, "fond") == [("5", "6")]
    assert block_earlier(m, p, "couleur") == []                 # Outdated: it is recomputed anyway
    assert earlier_words([("5", "6")]) == EN
    assert earlier_words([("5", "6"), ("5", "6"), ("1", "3")]) == \
        "Computed with an earlier version (5 -> 6, 1 -> 3)"


def test_the_earlier_version_line_says_version_numbers_not_the_network_file_s_hash():
    """Stars/background's version is "<n>-<hash of the network file>": the user reads the numbers only, the hash stays
    in the manifest; a new network file alone (same number) is said in words."""
    from astro_gpu.client.graphnode import earlier_words
    assert earlier_words([("4-9b974a78cb", "5-9b974a78cb")]) == "Computed with an earlier version (4 -> 5)"
    assert earlier_words([("3+classic2", "3+classic3")]) == "Computed with an earlier version (3+classic2 -> 3+classic3)"
    assert earlier_words([("5-2ea85a0755", "5-9b974a78cb")]) == \
        "Computed with an earlier version (5, new network file)"


def test_a_block_made_from_a_result_recomputed_since_says_why_it_is_outdated():
    from astro_gpu.client.model import ProjectModel
    st = api.StageState("stale", api.REASON_UPSTREAM_REDONE, "k", "k")
    assert ProjectModel._outdated_why(SimpleNamespace(), SimpleNamespace(pid=1), "color", st, 0) == \
        ("upstream", "the block before it was recomputed")


# ---------------------------------------------------------------------------------------------- the computed project

@needs_test_project
def test_real_panel_line_and_command_plan_the_recompute_and_post_still_renders(qapp, tmp_path, monkeypatch):
    """On the computed test project, with the Gradient's method at a new version: every block stays computed, the
    Gradient's panel says the earlier version and has the command, whose plan recomputes the Gradient and what follows
    it under the same keys; Post's layer renders from the results as they are."""
    from astro_gpu.client import app as appmod
    from astro_gpu.client import prefs
    from astro_gpu.client.graphview.page import GraphPage
    from astro_gpu.client.model import ProjectModel
    from astro_gpu.client.services.worker import ApiWorker
    from astro_gpu.stages.gradient import Gradient
    from tests.client.graphview.test_gv_legacy_model import button, node, select_node, settle, texts_of
    monkeypatch.setenv("ASTRO_GPU_SETTINGS", str(tmp_path / "settings.json"))
    prefs.load()
    appmod.apply_language("en")
    p = api.open_project(scratch_project(tmp_path / "heart"))
    full_only(p)
    before = {k: (v.state, v.key) for k, v in api.states(p, update=False).items()}
    assert before[(1, "gradient")][0] == "done" and before[(1, "render_simple")][0] == "done"
    monkeypatch.setattr(Gradient, "version", Gradient.version + "-next")
    api.refresh(p)
    st = api.states(p, update=False)
    assert {k: (v.state, v.key) for k, v in st.items()} == before          # nothing Outdated, no key moved
    assert st[(1, "gradient")].earlier == (Gradient.key_label, Gradient.key_label + "-next")
    img = api.render_layer(p, 1, scale=0.1)                                # Post's layer, from the results as they are
    assert img.rgb.ndim == 3 and img.rgb.shape[0] == 3 and float(img.rgb.max()) > 0
    w = ApiWorker()
    m = ProjectModel(p, w)
    assert w.wait(180) and m.ready, m.error
    try:
        pg = GraphPage(m)
        pg.resize(1700, 900)
        pg.show()
        settle(w, pg)
        n = node(pg, "fond", 1)
        select_node(pg, n)
        settle(w, pg, 3)
        words = f"Computed with an earlier version ({Gradient.key_label} -> {Gradient.key_label}-next)"
        assert words in texts_of(pg.panel)
        assert "Earlier version" in texts_of(pg.panel) and "Up to date" not in texts_of(pg.panel)   # one truth
        asked = []
        pg.view_action.connect(lambda op, ident: asked.append(op))
        button(pg.panel, CMD_EN).click()
        assert asked == ["recompute"]
        plan = m.plan_recompute(n.paths, n.ref)
        stages = [j.stage for j in plan.jobs]
        assert stages[0] == "gradient" and "render_simple" in stages and all(j.redo for j in plan.jobs)
        assert "stack" not in stages and plan.seconds > 0
        assert {j.key for j in plan.jobs} <= {k for _s, k in before.values()}       # the same keys: results replaced
        pg.close()
    finally:
        m.flush()
        w.wait(30)


def test_a_block_of_an_earlier_version_is_counted_by_one_rule_for_the_graph_the_panel_and_the_status():
    from astro_gpu.client.graphnode import earlier_blocks, node_earlier
    info = {"gradient": api.StageState("done", "", "k1", earlier=("5", "6")),
            "color": api.StageState("done", "", "k3")}
    p = SimpleNamespace(pid=1, params={}, opts={})
    node = lambda key, **k: SimpleNamespace(key=key, ref=key, paths=[p], sid=None, cut=False, grey="", **k)  # noqa: E731
    nodes = [node("fond"), node("couleur"), node("session")]
    bm = SimpleNamespace(nodes=nodes, state=lambda n: "done", session_stages=lambda q, sid: None)
    m = SimpleNamespace(state_info=lambda q, s: info.get(s), state=lambda q, s: info[s].state, block_map=bm,
                        all_stages=lambda q: list(info))
    assert node_earlier(m, nodes[0]) == [("5", "6")] and node_earlier(m, nodes[1]) == []
    assert earlier_blocks(m) == 1
    nodes[0].cut = True                                         # cut: nothing is said of a block that is not run
    assert earlier_blocks(m) == 0


def test_run_is_never_up_to_date_over_a_block_of_an_earlier_version_and_says_where_the_command_is():
    """The Run button greyed with "All up to date" while a block's result is of an earlier version said the opposite
    of the graph's amber block: it says how many, and where Recompute is."""
    from astro_gpu.client import app as appmod
    info = {"gradient": api.StageState("done", "", "k1", earlier=("5", "6"))}
    p = SimpleNamespace(pid=1, params={}, opts={})
    node = SimpleNamespace(key="fond", ref="fond", paths=[p], sid=None, cut=False, grey="", waits="")
    bm = SimpleNamespace(nodes=[node], state=lambda n: "done", session_stages=lambda q, sid: None)
    m = SimpleNamespace(ready=True, paths=[p], has_data=lambda: True, p=SimpleNamespace(workflows=[SimpleNamespace(blocks=None)]),
                        state_info=lambda q, st: info.get(st), state=lambda q, st: "done", block_map=bm,
                        all_stages=lambda q: list(info))
    fake = SimpleNamespace(s=m, waiting_text=lambda: "")
    none = SimpleNamespace(jobs=0)
    assert appmod.Main.run_refusal(fake, none) == "1 block in an earlier version: Recompute, on the block"
    info.clear()
    assert appmod.Main.run_refusal(fake, none) == "All up to date"


def test_the_blocks_of_an_earlier_version_of_some_workflows_are_found_in_the_graph_s_order():
    from astro_gpu.client.graphnode import earlier_nodes_of
    info = {"gradient": api.StageState("done", "", "k1", earlier=("5", "6")),
            "color": api.StageState("done", "", "k3")}
    p1, p2 = SimpleNamespace(pid=1, params={}, opts={}), SimpleNamespace(pid=2, params={}, opts={})
    node = lambda key, q: SimpleNamespace(key=key, ref=key, paths=[q], sid=None, cut=False, grey="")  # noqa: E731
    info["astrometry"] = api.StageState("done", "", "k4", earlier=("1", "2"))
    nodes = [node("fond", p1), node("couleur", p1), node("fond", p2), node("astrometrie", p1)]
    bm = SimpleNamespace(nodes=nodes, state=lambda n: "done", session_stages=lambda q, sid: None)
    m = SimpleNamespace(state_info=lambda q, st: info.get(st), state=lambda q, st: "done", block_map=bm,
                        all_stages=lambda q: list(info))
    # workflow 1 has two blocks of an earlier version: the last of its chain, the one its layer reads
    assert earlier_nodes_of(m, {1}) == [nodes[3]] and earlier_nodes_of(m, {2}) == [nodes[2]]
    assert earlier_nodes_of(m, {1, 2}) == [nodes[2], nodes[3]] and earlier_nodes_of(m, set()) == []


def test_recompute_of_several_blocks_is_one_plan_in_one_window(qapp):
    """Post's Recompute (several blocks, one plan: a result two of them share is computed once)."""
    from astro_gpu.client.services.run import RunController
    seen = {}

    class Model:
        scope = None

        def view(self, pid):
            return SimpleNamespace(pid=pid)

        def plan_recompute(self, views, key):
            return SimpleNamespace(jobs=[SimpleNamespace(key=f"{key}-{v.pid}") for v in views] +
                                   [SimpleNamespace(key="shared")], targets=[], skipped={}, rules=None)

        def _one_plan(self, plans):
            seen["plans"] = plans
            return plans

    opened = {}
    fake = SimpleNamespace(model=Model(), running=False, before_run=lambda: True, scope=None,
                           block_name=lambda p, key: key,
                           _plan_and_open=lambda planner, title, **k: opened.update(plan=planner(), title=title()))
    p1, p2 = SimpleNamespace(pid=1), SimpleNamespace(pid=2)
    RunController.recompute_blocks(fake, [([p1], "fond"), ([p2], "etoiles")])
    assert len(seen["plans"]) == 2 and fake.scope == {1, 2}
    assert opened["title"] == "Recompute “fond, etoiles” (current version)"
