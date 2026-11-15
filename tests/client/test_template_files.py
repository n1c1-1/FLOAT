"""Templates are blank project files (issue #167): the shipped files are the reference (a changed default is caught by
the tool script that writes them), the gallery's tabs are the folders of the shipped root and of the user's, a template
chosen on a blank project makes a new project that is its copy, and Save as template writes the open project's graph,
without frames or paths, into the user's folder."""
from __future__ import annotations

import importlib.util
import json
import os
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from astro_gpu.client import app as appmod  # noqa: E402
from astro_gpu.client import prefs, templatefiles  # noqa: E402
from astro_gpu.core import project2  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def qapp():
    from astro_gpu.client.widgets import QSS
    a = QApplication.instance() or QApplication([])
    a.setStyleSheet(QSS + appmod.EXTRA_QSS)
    return a


@pytest.fixture()
def settings(tmp_path, monkeypatch):
    monkeypatch.setenv("ASTRO_GPU_SETTINGS", str(tmp_path / "settings.json"))
    prefs.load()
    return tmp_path


def _tool():
    spec = importlib.util.spec_from_file_location("make_templates", ROOT / "tools" / "make_templates.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_shipped_files_are_what_the_tool_writes(qapp, settings):
    """Every shipped template file loads as a project, and equals what the tool script regenerates from the
    construction: a default that changed without the files written again fails here (python tools/make_templates.py)."""
    tool = _tool()
    made = tool.construction()
    files = [f for tab in templatefiles.SHIPPED_TABS for f, mine in templatefiles.files(tab) if not mine]
    assert sorted(f.name for f in files) == sorted(tool.file_of(t).name for t in made)
    for t in made:
        f = tool.file_of(t)
        assert f.parent.name in templatefiles.SHIPPED_TABS and f.parent.name == t["tab"]
        d = json.loads(f.read_text(encoding="utf-8"))
        project2.Project2.from_dict({**d, "name": "Untitled"}, f.parent)
        assert d == json.loads(json.dumps(tool.build(t))), f"{f.name}: write the files again (tools/make_templates.py)"


def test_tabs_are_folders_of_both_roots(settings):
    mine = templatefiles.user_dir()
    assert mine == settings / "templates"
    assert templatefiles.tabs() == ["Color", "Mono"]
    d = {"graph": {}, "format": 3}
    for tab in ("Mono", "Dual band"):                      # one in a shipped tab, one a tab of its own
        (mine / tab).mkdir(parents=True)
        (mine / tab / ("Mine" + templatefiles.SUFFIX)).write_text(json.dumps(d), encoding="utf-8")
    (mine / "Empty").mkdir()
    assert templatefiles.tabs() == ["Color", "Mono", "Dual band", "Empty"]        # a folder added is a tab, empty
    mono = templatefiles.cards("Mono")
    assert [c["name"] for c in mono] == ["Filters full", "Filters minimal", "Mono LRGB", "Mono RGB", "Mine"]
    assert [c["user"] for c in mono] == [False] * 4 + [True]
    assert [c["name"] for c in templatefiles.cards("Dual band")] == ["Mine"]
    assert not any(c["user"] for c in templatefiles.cards("Color"))


def test_gallery_shows_the_tabs(qapp, settings):
    from astro_gpu.client.gallery import TemplateGallery, TemplatePage
    f = templatefiles.save_user({"graph": {"blocks": {}, "nights": [], "common": {}}}, "Dual band", "Mine")
    page = TemplatePage()
    page.build()
    assert page.tab_names == ["Color", "Mono", "Dual band"]
    assert {c.t["tab"] for c in page.cards} == {"Color"}
    page.show_tab("Dual band")
    assert [c.t["name"] for c in page.cards] == ["Mine"] and page.cards[0].t["file"] == f.as_posix()
    d = TemplateGallery(type("S", (), {"paths": []})())
    assert d.tab_keys == ["Color", "Mono", "Dual band"]


def _settle(w, n=8):
    for _i in range(n):
        w.worker.wait(30)
        QApplication.processEvents()
        time.sleep(0.01)


@pytest.fixture()
def win(qapp, settings):
    appmod.apply_language("en")
    w = appmod.Main()
    w.resize(1500, 900)
    w.show()
    w.new_project()
    end = time.monotonic() + 120
    while time.monotonic() < end and not w.s.ready:
        w.worker.wait(1)
        QApplication.processEvents()
    yield w
    w.close_project(discard=True)
    w.close()
    w.deleteLater()
    QApplication.processEvents()


def _rows(w):
    return [r.name for r in w.graph.bridge.graph.rows]


def test_a_template_makes_a_new_project_that_is_its_copy(win):
    first = win.project.root
    card = next(c for c in templatefiles.shipped_cards() if c["key"] == "colour_full")
    win.graph.show_templates()
    win.graph.templates.chosen.emit(card)
    _settle(win, 30)
    assert win.project is not None and win.project.root != first
    assert "Colour full" in _rows(win)
    g = win.graph.bridge.graph
    assert all(not n["files"] for n in g.nights) and all(not b.files for b in g.blocks.values())      # no frames


def test_save_as_template_round_trip(win, monkeypatch):
    from astro_gpu.client import gallery
    card = next(c for c in templatefiles.shipped_cards() if c["key"] == "colour_min")
    win.graph.show_templates()
    win.graph.templates.chosen.emit(card)
    _settle(win, 30)
    monkeypatch.setattr(gallery, "template_dialog", lambda parent, name: ("My colour", "Dual band"))
    win.save_template()
    f = templatefiles.user_dir() / "Dual band" / ("My colour" + templatefiles.SUFFIX)
    assert f.is_file()
    d = json.loads(f.read_text(encoding="utf-8"))
    assert d["created"] == "" and d["scratch_dir"] == "" and all(not n["files"] for n in d["graph"]["nights"])
    assert templatefiles.tabs()[-1] == "Dual band"
    mine = templatefiles.cards("Dual band")[0]
    assert mine["name"] == "My colour" and mine["user"] and mine["file"] == f.as_posix()
    win.new_project()
    _settle(win, 10)
    win.graph.show_templates()
    win.graph.templates.show_tab("Dual band")
    win.graph.templates.chosen.emit(win.graph.templates.cards[0].t)
    _settle(win, 30)
    assert "Colour minimal" in _rows(win)
