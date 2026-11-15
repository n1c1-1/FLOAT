"""No two visible texts drawn over each other, at the screens' usual sizes (1250 x 685, 1366 x 768, 1920 x 1080), in
French and in English, on: the empty graph, the template gallery, a graph with one template, a graph with a
combination of every recipe, Inspect's list of frames;
and the tabs of the project bar keep their place when the project's name gets longer (a long name is cut, whole in its
tooltip).

The boxes are the texts' own (tests/client/langscan.py): a widget's text clipped to what shows of it, an item of the
graph as it paints it. The window fuzz (tools/fuzz/graphview_fuzz.py) checks the same on the graph's own gestures."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if Path("C:/Windows/Fonts").is_dir():
    os.environ.setdefault("QT_QPA_FONTDIR", "C:/Windows/Fonts")
pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication  # noqa: E402

from tests.client import langscan  # noqa: E402
from tests.client.test_swap_smoke import (  # noqa: E402,F401 - fixtures of the application's smoke test
    env,
    made_from,
    qapp,
    ready,
    settle,
    template,
    win,
)

SIZES = [(1250, 685), (1366, 768), (1920, 1080)]
LANGS = ["en", "fr"]


def _crossed(w) -> list:
    QApplication.processEvents()
    return [f"{a!r} over {b!r}" for a, b in langscan.overlaps(langscan.texts(w))]


def _window(win, size, lang):
    win.change_language(lang)
    win.resize(*size)
    settle(win, 3)


@pytest.mark.parametrize("lang", LANGS)
@pytest.mark.parametrize("size", SIZES)
def test_the_empty_graph_and_the_gallery_draw_no_text_over_another(win, env, size, lang):
    _window(win, (1100, 700), lang)                      # built small, the window made larger while the graph is hidden
    win.new_project()
    ready(win)
    win.set_mode(1)
    settle(win, 3)
    win.resize(*size)
    settle(win, 3)
    win.set_mode(0)
    settle(win, 3)
    assert not _crossed(win), _crossed(win)
    win.graph.show_templates()
    settle(win, 3)
    assert not _crossed(win), _crossed(win)


@pytest.mark.parametrize("lang", LANGS)
@pytest.mark.parametrize("size", SIZES)
def test_a_graph_with_one_template_draws_no_text_over_another(win, env, size, lang):
    _window(win, size, lang)
    made_from(win, env, "colour_full")
    settle(win, 3)
    assert not _crossed(win), _crossed(win)
    win.set_mode(1)                                       # Inspect, on the workflow made
    settle(win, 3)
    assert not _crossed(win), _crossed(win)


@pytest.mark.parametrize("lang", LANGS)
@pytest.mark.parametrize("size", SIZES)
def test_the_combinations_of_every_recipe_draw_no_text_over_another(win, env, size, lang):
    """Each recipe of the application placed from the Combination card's list (S-H-O... HOO, HaRGB) and chosen: the
    graph's drawing, the palette, and the block's panel, each on its own (on a small screen the automatic panel opens
    over the graph: the drawing under it is hidden, not crossed); then undone, the next one placed."""
    from astro_gpu.graphcore import combos
    _window(win, size, lang)
    made_from(win, env, "filters_min")
    page = win.graph
    pal = page.palette
    for rid in combos.offered():
        pal.set_combination_type(rid)
        before = {b for b, x in win.s.stored.blocks.items() if x.kind == "combine"}
        pal.double_click("combine")
        settle(win, 2)
        new = next(b for b, x in win.s.stored.blocks.items() if x.kind == "combine" and b not in before)
        page.bridge.choose(new)
        settle(win, 3)
        for part in (page.canvas, pal, page.panel):
            assert not _crossed(part), (rid, _crossed(part))
        win.s.undo(graph=True)
        settle(win, 2)


@pytest.mark.parametrize("lang", LANGS)
def test_the_tabs_keep_their_place_when_the_name_gets_longer(win, env, lang):
    _window(win, (1250, 685), lang)
    win.new_project()
    ready(win)
    where = {}
    for name in ("Hi", "Heart nebula in Cassiopeia", "A very long name for a project that goes on and on and on " * 2):
        win.project.definition.name = name
        win.project.definition.target.name = "IC 1805"
        win.title.setText(win.project_name())
        QApplication.processEvents()
        where[name[:6]] = [b.geometry().x() for b in win.mode_btns]
        assert win.title.width() == win.title.maximumWidth()
        if len(name) > 40:
            assert name in win.title.toolTip()               # the whole name, where the cut one is
    assert len({tuple(v) for v in where.values()}) == 1, where


# ---------------------------------------------------------------------------------------------- Inspect's frame list
def _test_project():
    from conftest import HAVE_TEST_PROJECT, TEST_PROJECT
    return TEST_PROJECT if HAVE_TEST_PROJECT else None


@pytest.mark.skipif(_test_project() is None, reason="the test project is not on this machine")
@pytest.mark.parametrize("lang", LANGS)
@pytest.mark.parametrize("size", SIZES)
def test_real_inspect_frame_list_draws_no_text_over_another(win, env, size, lang):
    """The first session's frames: the viewer, the facts of the frame shown (its time, its state) and the strip."""
    _window(win, size, lang)
    win.readonly = True                                    # a project of the test machine: nothing is written in it
    assert win.open_project(_test_project())
    ready(win, 300)
    win.s.select(step="session@1")
    win.set_mode(1)
    settle(win, 6)
    from astro_gpu.client.imageview import default_loader
    win.browser.set_mode(1)                                # the viewer: the frame's facts under the image
    for _i in range(40):                                   # its first frames are read in threads
        default_loader().wait(10)
        settle(win, 2)
        if win.browser.viewer.band.state.isVisible() and win.browser.viewer.band.name_l.full_text():
            break
    assert win.browser.viewer.band.name_l.full_text(), "the viewer shows no frame"
    assert not _crossed(win), _crossed(win)


@pytest.mark.parametrize("lang", LANGS)
@pytest.mark.parametrize("size", SIZES)
def test_a_blocks_draft_bar_at_the_top_of_the_panel_draws_no_text_over_another(win, env, size, lang):
    """A block holding a draft (blockdraft.py): its bar pinned under the block's header, above Apply to and the settings,
    its state word "Draft"; in the Graph page's panel and in Inspect's, no text over another."""
    from astro_gpu.client.blockdraft import Draft, DraftBar
    from astro_gpu.client.panel import Panel
    _window(win, size, lang)
    made_from(win, env, "colour_full")
    s, page = win.s, win.graph
    stack = next(b for b, x in s.stored.blocks.items() if x.kind == "stacking")
    page.bridge.choose(stack)
    settle(win, 3)
    node = page.panel.node
    d = Draft(node.key, [q.pid for q in node.paths])
    d.title = "Stacking"
    for name, v in (("kappa", 4.5), ("pixfrac", 0.7)):
        d.put({("param", "stack", name): {"ch": {"kind": "param", "stage": "stack", "name": name, "value": v,
                                                 "suggested": None},
                                          "what": f"{name}: 1 → {v}", "short": name}})
    s.pending_change = d
    try:
        for mode in (0, 1):
            win.set_mode(mode)
            settle(win, 3)
            pn = page.panel if mode == 0 else next(p for p in win.tune.findChildren(Panel))
            pn.refresh(pn.node)
            settle(win, 2)
            bar = pn.findChild(DraftBar)
            from PySide6.QtWidgets import QLabel
            assert bar is not None and bar.isVisibleTo(pn) and not pn._sc.isAncestorOf(bar)      # pinned, at the top
            assert pn.hv.indexOf(bar) >= 0                 # in the block's header, above Apply to and the settings
            labels = [pn.hv.itemAt(i).widget() for i in range(pn.hv.count())]
            scope = [i for i, w in enumerate(labels) if isinstance(w, QLabel) and w.text().rstrip(" :") in
                     ("Apply to", "Appliquer à")]
            assert all(pn.hv.indexOf(bar) < i for i in scope)
            pn.layout().activate()
            from PySide6.QtWidgets import QLabel
            word = pn.findChild(QLabel, "draft_state")
            assert word is not None and word.text() == ("Brouillon" if lang == "fr" else "Draft")
            assert not _crossed(pn), (mode, _crossed(pn))
    finally:
        s.pending_change = None
