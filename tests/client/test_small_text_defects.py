"""Small text and view defects the testers quoted: File > Recent projects lists a project just saved (the menu was
built once and never read the settings again), and "Show all" never takes the graph below the floor of the automatic
view (it went to 40 % at 1366 x 768, titles nobody could read)."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if Path("C:/Windows/Fonts").is_dir():
    os.environ.setdefault("QT_QPA_FONTDIR", "C:/Windows/Fonts")
pytest.importorskip("PySide6")

from tests.client.test_swap_smoke import (  # noqa: E402,F401 - fixtures of the application's smoke test
    env,
    made_from,
    qapp,
    ready,
    settle,
    win,
)


def test_recent_projects_lists_a_project_just_saved(win, env, monkeypatch):
    from astro_gpu.client import app as appmod
    win.new_project()
    ready(win)
    root = env["tmp"] / "Saved one"
    monkeypatch.setattr(appmod, "ask_project_name", lambda parent, suggested: (root, "Saved one"))
    file_menu = win.menuBar().actions()[0].menu()
    file_menu.aboutToShow.emit()
    assert not win._recent_menu.isEnabled()                                    # nothing saved yet
    assert win.name_project()                                                  # the first Save names it
    file_menu.aboutToShow.emit()                                               # the menu opens
    assert win._recent_menu.isEnabled()
    assert [a.text() for a in win._recent_menu.actions()][0].startswith("Saved one")


def test_show_all_never_goes_below_the_readable_size(win, env):
    """The whole graph of a big template in a 1366 x 768 window: Show all fits what it can at the readable size of
    the automatic view (its floor, 50 %), and scrolls from the left end for the rest, never 40 %."""
    from astro_gpu.client.graphview.canvas import MIN_AUTO
    win.resize(1366, 768)
    made_from(win, env, "filters_full")
    settle(win, 3)
    canvas = win.graph.canvas
    canvas.show_all()
    settle(win, 2)
    assert canvas.zoom >= MIN_AUTO - 1e-9, canvas.zoom
    assert canvas.b_zoom.text() == f"{round(canvas.zoom * 100)} %"
    win.resize(1920, 1080)                                         # room: the whole graph, as large as it fits
    settle(win, 2)
    canvas.show_all()
    assert MIN_AUTO <= canvas.zoom <= 1.2


@pytest.mark.parametrize("size", [(1600, 900), (1366, 768)])
def test_the_automatic_view_shows_the_whole_graph_beside_the_open_panel(win, env, size):
    """With the settings panel open the automatic view stopped at 75 %: the Composition area and the Post block of
    "Color full" were off the canvas. It goes down until the graph fits what the sides leave, to 50 % at the least."""
    from astro_gpu.client.graphview.canvas import MIN_AUTO
    win.resize(*size)
    made_from(win, env, "colour_full")
    win.graph.set_layout("docked", "docked")
    settle(win, 3)
    canvas, graph = win.graph.canvas, win.graph.bridge.graph
    assert canvas.auto and win.graph.panel.isVisible()
    post = canvas.blocks[next(b for b, x in graph.blocks.items() if x.kind == "post")]
    right = canvas.mapFromScene(post.mapToScene(post.w, post.h))
    left = canvas.mapFromScene(post.mapToScene(0, 0))
    vp = canvas.viewport().rect()
    whole = vp.contains(left) and vp.contains(right)
    assert whole or abs(canvas.zoom - MIN_AUTO) < 1e-6, (canvas.zoom, right.x(), vp.width())
    assert MIN_AUTO == 0.5 and canvas.b_zoom.text() == f"{round(canvas.zoom * 100)} %"
