"""Every way out of a project asks the window's one question (Save / Don't save / Cancel) when anything is unsaved,
whatever the kind of edit: the graph, the table, a block's setting, a combination's, Post's composition (its last edit
held by Post until the event loop comes back included) and the project's form. One function decides (``Main.unsaved``);
the paths: Close project (Ctrl+W), Quit, the window's close box, New project, Open project…, Recent projects, Home.
Cancel keeps the project and its edits; with nothing unsaved, nothing is asked."""
from __future__ import annotations

import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from astro_gpu.client import app as appmod  # noqa: E402
from astro_gpu.client import home as homemod  # noqa: E402
from astro_gpu.client import prefs  # noqa: E402

ORIGINAL_ASK = appmod.Main.ask_save


@pytest.fixture(scope="module")
def qapp():
    from astro_gpu.client.widgets import QSS
    a = QApplication.instance() or QApplication([])
    a.setStyleSheet(QSS + appmod.EXTRA_QSS)
    return a


@pytest.fixture()
def win(qapp, tmp_path, monkeypatch):
    monkeypatch.setenv("ASTRO_GPU_SETTINGS", str(tmp_path / "settings.json"))
    prefs.load()
    prefs.set("home_sections", [])
    appmod.apply_language("en")
    answer = {"now": "cancel"}
    asked = []
    monkeypatch.setattr(appmod.Main, "ask_save", lambda self: (asked.append(self.project.root), answer["now"])[1])
    from astro_gpu.core import project2
    from astro_gpu.core.project import Target
    other = tmp_path / "other"
    project2.save(project2.create(other, "Other", Target(name="M 31")))
    prefs.add_recent(other.as_posix(), "Other", "M 31")
    monkeypatch.setattr(homemod, "choose_project", lambda parent: other.as_posix())
    w = appmod.Main()
    w.resize(1400, 900)
    w.show()
    w.asked, w.answer, w.other = asked, answer, other
    yield w
    answer["now"] = "discard"
    if w.project is not None:
        w.close_project(discard=True)
    w.close()
    w.deleteLater()
    QApplication.processEvents()


def _ready(w):
    end = time.monotonic() + 120
    while time.monotonic() < end and not (w.s is not None and w.s.ready):
        w.worker.wait(1)
        QApplication.processEvents()
        time.sleep(0.02)
    assert w.s.ready, w.s.error
    for _i in range(5):
        QApplication.processEvents()


def _saved_from(win, key):
    """A new project, its workflow made from the gallery's template ``key``, saved: nothing unsaved."""
    from astro_gpu.client.gallery import GALLERY
    win.new_project()
    _ready(win)
    win.graph.show_templates()
    win.graph.templates.chosen.emit(dict(next(t for t in GALLERY if t["key"] == key), file=""))
    QApplication.processEvents()
    assert win.s.flush() and not win.s.dirty
    return win.graph


def _block(page, kind):
    return next(k for k, x in page.bridge.graph.blocks.items() if x.kind == kind)


# ------------------------------------------------------------------------------------------------ the kinds of edit
def _graph(win, page):
    assert page.bridge.act("add_row", kind="session").ok


def _table(win, page):
    page.set_display("table")
    QApplication.processEvents()
    r = page.bridge.graph.rows[0]
    assert page.table_host.table.grid.act([("rename_row", dict(row=r.id, name="Renamed"))]).ok
    page.set_display("blocks")


def _setting(win, page):
    res = page.bridge.act("set_node_method", block=_block(page, "gradient"), method="rbf")
    if not res.ok:
        res = page.bridge.act("set_method", block=_block(page, "gradient"), method="rbf")
    assert res.ok, page.bridge.say(res.reason)


def _combination(win, page):
    blk = next(k for k, x in page.bridge.graph.blocks.items() if x.kind.startswith("combine") and x.kind != "combine_layers")
    res = page.bridge.act("set_param", block=blk, stage="combine_sho", name="hue_ha", value=120.0)
    assert res.ok, page.bridge.say(res.reason)


def _post(win, page):
    s = win.s
    lid = s.add_post_layer(s.paths[0].pid, None)
    s.flush()
    assert not s.dirty
    s.set_post_layer(lid, visible=False)


def _post_held(win, page):
    """Post keeps its last edit (a slider) until the event loop comes back: the question takes it first."""
    win.s.flush()

    class HeldEdit:
        def flush(self):
            win.s.edited()

    class Holder:
        page = type("P", (), {"view": HeldEdit()})()
    win._post_holder = Holder()


def _form(win, page):
    win.s.set_target_name("M 42")


KINDS = {"graph": _graph, "table": _table, "setting": _setting, "combination": _combination, "post": _post,
         "post_held": _post_held, "form": _form}


# ------------------------------------------------------------------------------------------------ the ways out
def _action(win, text):
    for m in win.menuBar().actions():
        for a in m.menu().actions():
            if a.text() == text:
                return a
            if a.menu() is not None:
                for b in a.menu().actions():
                    if b.text().startswith(text):
                        return b
    raise AssertionError(f"no menu item {text!r}")


PATHS = {
    "close (Ctrl+W)": lambda w: _action(w, "Close project").trigger(),
    "quit": lambda w: _action(w, "Quit").trigger(),
    "close box": lambda w: w.close(),
    "new": lambda w: _action(w, "New project").trigger(),
    "open": lambda w: _action(w, "Open project…").trigger(),
    "recent": lambda w: _action(w, "Other").trigger(),
    "home": lambda w: _action(w, "Home").trigger(),
}


def test_the_close_and_home_items_keep_their_shortcuts(win):
    assert _action(win, "Close project").shortcut().toString() == "Ctrl+W"
    assert _action(win, "Home").shortcut().toString() == "Ctrl+H"


@pytest.mark.parametrize("kind", list(KINDS))
def test_every_way_out_asks_for_every_kind_of_edit_and_cancel_keeps_it(win, kind):
    page = _saved_from(win, "filters_full" if kind == "combination" else "colour_full")
    root = win.project.root
    KINDS[kind](win, page)
    assert win.unsaved(), f"{kind}: the edit is not seen as unsaved"
    for name, go in PATHS.items():
        win.asked.clear()
        go(win)
        QApplication.processEvents()
        assert win.asked == [root], f"{kind} · {name}: not asked"
        assert win.project is not None and win.project.root == root and win.s.dirty, f"{kind} · {name}: not kept"
        assert win.isVisible() and win.screens.currentWidget() is win.proj_screen, f"{kind} · {name}: left"


def test_nothing_unsaved_nothing_asked_and_home_closes_the_project(win):
    _saved_from(win, "colour_full")
    win.asked.clear()
    _action(win, "Home").trigger()
    QApplication.processEvents()
    assert not win.asked and win.project is None and win.screens.currentWidget() is win.home


def test_save_and_dont_save_do_what_they_say(win, tmp_path):
    from astro_gpu.core import project2
    root = tmp_path / "ws" / "Named"
    assert win.create_project(root.as_posix(), "Named", "IC 1805")
    _ready(win)
    win.s.set_target_name("M 42")
    win.answer["now"] = "discard"
    _action(win, "Other").trigger()                                       # Recent projects
    QApplication.processEvents()
    assert win.project is not None and win.project.root == win.other          # the other one is open
    assert project2.load(root).target.name == "IC 1805"                      # the edit dropped, as answered
    assert win.open_project(root)
    _ready(win)
    win.s.set_target_name("M 42")
    win.answer["now"] = "save"
    _action(win, "Home").trigger()
    QApplication.processEvents()
    assert win.project is None and project2.load(root).target.name == "M 42"  # saved, then closed


def test_the_question_is_asked_without_a_screen_unless_the_run_answers_it(win, monkeypatch):
    """A usability session draws the window without a screen and answers the box as a user would: the question is
    shown; only an unattended run gives its answer beforehand."""
    from PySide6.QtWidgets import QMessageBox
    _saved_from(win, "colour_full")
    win.s.set_target_name("M 42")
    shown = []
    monkeypatch.setattr(QMessageBox, "exec", lambda self: shown.append(self.text()) or 0)    # closed: no button
    monkeypatch.delenv(appmod.SAVE_ANSWER_ENV, raising=False)
    assert QApplication.platformName() == "offscreen"
    assert ORIGINAL_ASK(win) == "cancel" and shown and "Save the changes" in shown[0]
    monkeypatch.setenv(appmod.SAVE_ANSWER_ENV, "discard")
    shown.clear()
    assert ORIGINAL_ASK(win) == "discard" and not shown


def test_the_projects_picture_is_post_as_shown_its_crop_applied(win, tmp_path):
    """The report's and Home's picture of the project (``cache/thumb.jpg``) is what Post shows: with a crop set, the
    picture has the crop's shape, not the whole canvas'."""
    import numpy as np
    from PIL import Image

    from astro_gpu.client.home import THUMB
    root = tmp_path / "ws" / "Pic"
    assert win.create_project(root.as_posix(), "Pic", "M 31")
    _ready(win)
    half = np.full((200, 300, 3), 90, np.uint8)                       # a 600 x 400 canvas, at half resolution

    class Src:
        img1 = half

        @staticmethod
        def size():
            return (600, 400)

    class View:
        src, crop = Src(), [100, 0, 300, 400]                         # 200 x 400: taller than wide

    class Holder:
        page = type("P", (), {"view": View()})()
    win._post_holder = Holder()
    win._leave_thumbnail()
    win.worker.wait(30)
    w, h = Image.open(root / THUMB).size
    assert h > w and abs(w / h - 0.5) < 0.02, (w, h)
    View.crop = None
    win._leave_thumbnail()
    win.worker.wait(30)
    w, h = Image.open(root / THUMB).size
    assert abs(w / h - 1.5) < 0.02, (w, h)
    win._post_holder = None
