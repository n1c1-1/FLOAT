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


def test_a_users_template_adds_its_workflow_to_a_project_that_holds_sessions(win, monkeypatch):
    """What a user's template adds is read from its file's graph, as a shipped template's is: the same blocks, and the
    workflow it adds to a project that already holds a session."""
    from astro_gpu.client import gallery
    shipped = next(c for c in templatefiles.shipped_cards() if c["key"] == "colour_min")
    win.graph.show_templates()
    win.graph.templates.chosen.emit(shipped)
    _settle(win, 30)
    monkeypatch.setattr(gallery, "template_dialog", lambda parent, name: ("My colour", "Mine"))
    win.save_template()
    mine = templatefiles.cards("Mine")[0]
    kinds = lambda c: [n["block"] for n in gallery.template_blocks(c) if n["block"] not in ("calibration", "output")]      # noqa: E731
    assert mine["source"] == "frames" and kinds(mine) == kinds(shipped) and kinds(mine)
    before = len(_rows(win))
    res = win.s.apply_template(mine, "Added", gallery._view_opts(mine))
    assert res.ok and len(_rows(win)) == before + 1 and "Added" in _rows(win)


def test_the_templates_kept_in_the_settings_are_written_once_as_files(qapp, settings):
    """At start each old template of the settings becomes a blank project file of the tab My templates, once; the settings
    entry stays; the file's card lists the blocks the old template had."""
    old = [{"key": "u1", "tab": "Mine", "name": "My old one", "text": "Kept.", "source": "frames",
            "blocks": [{"block": b, "name": b, "method": None, "off": False}
                       for b in ("calibration", "registration", "selection", "stacking", "astrometry")]}]
    prefs.set("templates", old)
    first = templatefiles.import_old()
    assert [f.name for f in first] == ["My old one" + templatefiles.SUFFIX] and first[0].parent.name == templatefiles.OLD_TAB
    assert prefs.get("templates") == old and prefs.get("templates_moved")
    c = templatefiles.cards(templatefiles.OLD_TAB)[0]
    assert c["name"] == "My old one" and c["text"] == "Kept." and c["user"]
    assert [n["block"] for n in c["blocks"]] == ["calibration", "registration", "selection", "stacking", "astrometry"]
    assert templatefiles.import_old() == [] and len(templatefiles.cards(templatefiles.OLD_TAB)) == 1


def test_the_version_is_one_number_everywhere():
    """The window, the About box and the log read astro_gpu.__version__; the project file and the installer carry the
    same number."""
    import re
    from astro_gpu import __version__
    assert __version__ == "0.2.0"
    assert re.search(r'^version = "([^"]+)"', (ROOT / "pyproject.toml").read_text(encoding="utf-8"), re.M).group(1) == __version__
    assert f'#define AppVersion "{__version__}"' in (ROOT / "installer" / "float.iss").read_text(encoding="utf-8")


def _from_shipped(win, key):
    card = next(c for c in templatefiles.shipped_cards() if c["key"] == key)
    win.graph.show_templates()
    win.graph.templates.chosen.emit(card)
    _settle(win, 30)


def test_a_template_saved_in_a_new_tab_shows_in_both_galleries(win, monkeypatch):
    """The tester's case: saved into a new tab, the template's card is in the blank page's gallery (it was an empty
    tab there) as in New workflow, the same cards."""
    from astro_gpu.client import gallery
    _from_shipped(win, "sho_rgb")
    monkeypatch.setattr(gallery, "template_dialog", lambda parent, name: ("Night SHO", "Tests nuit"))
    win.save_template()
    _from_shipped(win, "colour_min")
    monkeypatch.setattr(gallery, "template_dialog", lambda parent, name: ("Simple", "Tests nuit"))
    win.save_template()
    win.new_project()
    _settle(win, 10)
    win.graph.show_templates()
    page = win.graph.templates
    page.show_tab("Tests nuit")
    QApplication.processEvents()
    assert [c.t["name"] for c in page.cards] == ["Night SHO", "Simple"]
    # a card without a text was 0 px tall: an empty wrapped label gives no height for its width
    assert all(c.isVisible() and c.height() >= c.pic.height() + 20 for c in page.cards), \
        [(c.isVisible(), c.height()) for c in page.cards]
    d = gallery.TemplateGallery(win.s)
    d._show_tab("Tests nuit")
    assert [c.t["name"] for c in d.cards] == ["Night SHO", "Simple"]
    d.close()


def test_a_users_multi_row_template_adds_its_whole_graph(win, monkeypatch):
    """The tester's case: a project made from SHO + RGB stars, saved as a template, came back as one chain (its first
    row). On a blank project it makes the copy of the whole file; on a project that holds workflows it adds every row,
    the S-H-O block and Combine layers, wired, as the shipped template does; its card draws the whole graph."""
    from astro_gpu.client import gallery
    _from_shipped(win, "sho_rgb")
    shipped_rows = _rows(win)
    assert len(shipped_rows) == 6
    monkeypatch.setattr(gallery, "template_dialog", lambda parent, name: ("Night SHO", "Tests nuit"))
    win.save_template()
    mine = templatefiles.cards("Tests nuit")[0]
    assert mine.get("kind") == "fusion" and mine.get("rgb_stars")              # drawn and added as the shipped one
    win.new_project()
    _settle(win, 10)
    win.graph.show_templates()
    win.graph.templates.show_tab("Tests nuit")
    win.graph.templates.chosen.emit(win.graph.templates.cards[0].t)          # a blank project: the file's copy
    _settle(win, 30)
    assert _rows(win) == shipped_rows
    before = len(_rows(win))                                                   # a project with workflows: all added
    res = win.s.apply_template(mine, "Again", gallery._view_opts(mine))
    assert res.ok, res.reason
    g = win.graph.bridge.graph
    assert len(_rows(win)) == before + 6
    kinds = [g.blocks[b].kind for r in g.rows for b in r.blocks]
    assert kinds.count("combine") == 2 and kinds.count("combine_layers") == 2


def test_every_shipped_multi_row_shape_is_known_again(win, monkeypatch):
    """Each shipped template of several rows, made into a project and saved as the user's: its card is the shipped
    recipe (it adds and draws the whole graph); a template of one row keeps the recipe of its row."""
    from astro_gpu.client import gallery
    for c in templatefiles.shipped_cards():
        if c["key"] == "sho_rgb" or c.get("source") == "image":
            continue
        win.new_project()                                    # each on a blank project: the file's copy
        _settle(win, 10)
        _from_shipped(win, c["key"])
        monkeypatch.setattr(gallery, "template_dialog", lambda parent, name, k=c["key"]: (k, "Shapes"))
        win.save_template()
        mine = next(x for x in templatefiles.cards("Shapes") if x["name"] == c["key"])
        if c.get("kind"):
            want = {k: v for k, v in c.items() if k not in templatefiles._OWN}
            assert {k: mine.get(k) for k in want} == want, c["key"]
        else:
            assert not mine.get("kind") and mine["blocks"], c["key"]


def _in_dialog(fn):
    """Run ``fn`` on the Save as template dialog once it is open, then close it; -> what fn returned."""
    from PySide6.QtCore import QTimer
    from astro_gpu.client import gallery
    got = []

    def look():
        d = QApplication.activeModalWidget()
        if d is None:
            QTimer.singleShot(20, look)
            return
        try:
            got.append(fn(d))
        finally:
            d.reject()
    QTimer.singleShot(0, look)
    assert gallery.template_dialog(None, "Night: SHO") is None
    return got[0]


def test_the_tab_field_says_a_new_name_makes_a_new_tab(qapp, settings):
    from PySide6.QtWidgets import QComboBox
    got =_in_dialog(lambda d: (d.findChild(QComboBox).lineEdit().placeholderText(), d.findChild(QComboBox).toolTip()))
    assert got == ("An existing tab, or a new name", "An existing tab, or a new name")


def test_save_says_it_replaces_a_template_of_the_same_file(qapp, settings):
    """13.33: a name that writes the same file as a template of the tab ("Night: SHO" and "Night SHO") replaced it
    without a word; Save now says it on hover. Another name: nothing said."""
    from PySide6.QtWidgets import QComboBox, QDialogButtonBox, QLineEdit
    templatefiles.save_user({"graph": {"blocks": {}, "nights": [], "common": {}}}, "Color", "Night SHO")

    def read(d):
        save = d.findChild(QDialogButtonBox).buttons()[0]
        out = [save.toolTip()]
        d.findChild(QLineEdit).setText("Another one")
        out.append(save.toolTip())
        d.findChild(QLineEdit).setText("night sho")                 # the same file on this system
        d.findChild(QComboBox).setCurrentText("Mono")
        out.append(save.toolTip())
        return out
    first, other, mono = _in_dialog(read)
    assert "Night SHO" in first and "replace" in first.lower()
    assert other == "" and mono == ""


def test_a_template_that_cannot_be_written_is_said_in_the_banner(win, monkeypatch):
    from astro_gpu.client import gallery
    monkeypatch.setattr(gallery, "template_dialog", lambda parent, name: ("Mine", "Tests nuit"))

    def refused(*a, **k):
        raise OSError("the disk is full")
    monkeypatch.setattr(templatefiles, "save_user", refused)
    said = []
    monkeypatch.setattr(win.banner, "say", lambda text, *a, **k: said.append(text))
    win.save_template()
    assert said and "the disk is full" in said[0]
