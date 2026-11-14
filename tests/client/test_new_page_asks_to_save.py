"""The window with the new graph page: an edit made on the page waits for Save (the title's star), and every way that
replaces the project (New project, Open project, after a workflow made from the blank page's template) asks the
window's one question first; Cancel keeps the project and its edits."""
from __future__ import annotations

import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from astro_gpu.client import app as appmod  # noqa: E402
from astro_gpu.client import prefs  # noqa: E402


@pytest.fixture()
def win(qapp, tmp_path, monkeypatch):
    monkeypatch.setenv("ASTRO_GPU_SETTINGS", str(tmp_path / "settings.json"))     # the drafts go beside it
    prefs.load()
    appmod.apply_language("en")
    asked = []
    monkeypatch.setattr(appmod.Main, "ask_save", lambda self: (asked.append(self.project.root), "cancel")[1])
    w = appmod.Main()
    w.resize(1400, 900)
    w.show()
    w.asked = asked
    yield w
    if w.project is not None:
        w.close_project(discard=True)
    w.close()
    w.deleteLater()
    QApplication.processEvents()


@pytest.fixture(scope="module")
def qapp():
    from astro_gpu.client.widgets import QSS
    a = QApplication.instance() or QApplication([])
    a.setStyleSheet(QSS + appmod.EXTRA_QSS)
    return a


def _ready(w):
    end = time.monotonic() + 120
    while time.monotonic() < end and not (w.s is not None and w.s.ready):
        w.worker.wait(1)
        QApplication.processEvents()
        time.sleep(0.02)
    assert w.s.ready, w.s.error
    for _i in range(5):
        QApplication.processEvents()


def test_an_edit_on_the_graph_page_waits_for_save_and_new_or_open_asks_first(win, tmp_path):
    from astro_gpu.client.gallery import GALLERY
    win.new_project()
    _ready(win)
    page, first = win.graph, win.project.root
    assert not win.s.dirty and not win.windowTitle().endswith("*")
    page.show_templates()                                              # the blank page's « From template »
    page.templates.chosen.emit(next(t for t in GALLERY if t.get("tab") == "frames"))
    QApplication.processEvents()
    assert page.bridge.graph.rows and win.s.dirty and win.windowTitle().endswith(" *")
    win.new_project()                                                  # New project: asked, Cancel keeps it
    assert win.asked == [first] and win.project.root == first and win.s.dirty
    from astro_gpu.core import project2
    from astro_gpu.core.project import Target
    other = tmp_path / "other"
    project2.save(project2.create(other, "Other", Target(name="M 31")))
    win.asked.clear()
    assert win.open_project(other) is False                            # Open project: asked too
    assert win.asked == [first] and win.project.root == first
    res = page.bridge.act("add_row", kind="session")                   # an edit of the page itself
    assert res.ok and win.s.dirty
