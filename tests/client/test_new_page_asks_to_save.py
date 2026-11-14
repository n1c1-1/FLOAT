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


# ======================================================================================================= quitting
def _colour_full_saved(win):
    """A new project made from the gallery's « Colour full » on the blank page, then saved: nothing left to ask."""
    from astro_gpu.client.gallery import GALLERY
    win.new_project()
    _ready(win)
    page = win.graph
    page.show_templates()
    page.templates.chosen.emit(next(t for t in GALLERY if t["key"] == "colour_full"))
    QApplication.processEvents()
    assert page.bridge.graph.rows and win.s.dirty                       # made from a template: an edit
    assert win.s.flush() and not win.s.dirty
    return page


def _quit_asks(win, how) -> bool:
    """Close the window (its close box, or File > Quit); the question answered Cancel. -> whether it was asked."""
    win.asked.clear()
    if how == "quit":
        quit_ = next(a for m in win.menuBar().actions() for a in m.menu().actions() if a.text() == "Quit")
        quit_.trigger()
    else:
        win.close()
    QApplication.processEvents()
    if win.asked:
        assert win.project is not None and win.isVisible()             # Cancel: the window and its project stay
    return bool(win.asked)


def _place(page):
    pl = next(p for p in page.bridge.places(kind="denoise") if p.lit)
    return page.bridge.drop(pl)


def _wire(page):
    b = page.bridge
    d = b.drawing()
    post = next(k for k, x in b.graph.blocks.items() if x.kind == "post")
    for (src, port, sense) in d.ports:
        if sense == "out" and b.check_wire(src, port, post, "layers") == "":
            return b.act("wire", src=src, src_port=port, dst=post, dst_port="layers")
    w = next(p.id for p in d.wires if p.dst and p.dst[0] == post)      # none to add: one taken out
    return b.act("unwire", wire=w)


def _delete(page):
    b = page.bridge
    for kind in ("denoise", "stars", "gradient"):
        blk = next((k for k, x in b.graph.blocks.items() if x.kind == kind), None)
        if blk is not None:
            res = b.act("remove_block", block=blk)
            if res.ok:
                return res
    raise AssertionError("no block could be deleted")


def _table(page):
    page.set_display("table")
    QApplication.processEvents()
    r = page.bridge.graph.rows[0]
    res = page.table_host.table.grid.act([("rename_row", dict(row=r.id, name="Renamed"))])
    page.set_display("blocks")
    return res


def _files(page, tmp_path):
    b = page.bridge
    s = next(k for k, x in b.graph.blocks.items() if x.kind == "session")
    d = tmp_path / "lights"
    d.mkdir()
    for i in range(3):
        (d / f"L_{i:03d}.fits").write_bytes(bytes(64))
    res = b.drops.drop(s, "lights", [str(d / f"L_{i:03d}.fits") for i in range(3)])
    b.drops.reader.wait(30)
    return res


def _panel_setting(page, win):
    from astro_gpu.client.panel import CompactRow
    b = page.bridge
    blk = next(k for k, x in b.graph.blocks.items() if x.kind == "gradient")
    page.set_layout("docked", "docked")
    b.choose(blk)
    end = time.monotonic() + 60                                          # its settings come from the worker
    while time.monotonic() < end and not page.panel.findChildren(CompactRow):
        win.worker.wait(1)
        QApplication.processEvents()
        time.sleep(0.02)
    rows = page.panel.findChildren(CompactRow)
    row = next(r for r in rows if r.r.get("type") in ("int", "float")
               and not r.r.get("locked") and r.isEnabled())
    v = row.r["value"] if row.r.get("value") is not None else row.r["suggested"]
    row.changed.emit(row.r, v + 1 if row.r["type"] == "int" else v * 1.5 + 0.1)
    QApplication.processEvents()
    return None


GESTURES = ("none", "place", "wire", "delete", "table", "files", "panel")


@pytest.mark.parametrize("gesture", GESTURES)
def test_quitting_after_a_template_and_a_gesture_of_the_page_asks(win, tmp_path, gesture):
    page = _colour_full_saved(win)
    if gesture == "none":                                                # just made and saved: nothing to ask
        assert not _quit_asks(win, "close")
        return
    res = {"place": _place, "wire": _wire, "delete": _delete, "table": _table}.get(gesture, lambda p: None)(page)
    if gesture == "files":
        res = _files(page, tmp_path)
    elif gesture == "panel":
        res = _panel_setting(page, win)
    assert res is None or res.ok, page.bridge.say(res.reason)
    assert win.s.dirty, f"{gesture} did not mark the project as edited"
    assert _quit_asks(win, "close") and _quit_asks(win, "quit")


# ======================================================================================================= Run saves
def _saved_project(win, tmp_path):
    """A project with its own folder (named), its workflow made from « Colour full », its target IC 1805, saved."""
    from astro_gpu.client.gallery import GALLERY
    root = tmp_path / "ws" / "Saved"
    assert win.create_project(root.as_posix(), "Saved", "IC 1805"), win.banner.text.text()
    _ready(win)
    win.graph.show_templates()
    win.graph.templates.chosen.emit(next(t for t in GALLERY if t["key"] == "colour_full"))
    QApplication.processEvents()
    assert win.s.flush() and not win.s.dirty
    return root


def test_run_on_a_saved_project_saves_its_edits_first_and_says_so(win, tmp_path):
    from astro_gpu.core import project2
    root = _saved_project(win, tmp_path)
    win.s.set_target_name("M 42")                                       # an edit not saved
    assert win.s.dirty and project2.load(root).target.name == "IC 1805"
    assert win.run.before_run() is True                                 # what Run and Run to here do first
    assert not win.asked and not win.s.dirty
    assert project2.load(root).target.name == "M 42"                    # on the disk before the plan is made
    assert win.status.text() == "Project saved"
    win.s.set_target_name("M 31")
    win.s.flush = lambda force=False: False                             # another window saved it meanwhile
    assert win.run.before_run() is False and win.s.dirty                # the run waits, the reason said
    assert "nothing is replaced" in win.banner.text.text()
    assert project2.load(root).target.name == "M 42"
    del win.s.flush


def test_run_on_a_project_never_saved_names_it_first_as_before(win, monkeypatch):
    win.new_project()
    _ready(win)
    win.s.set_target_name("M 42")
    named = []
    monkeypatch.setattr(appmod.Main, "name_project", lambda self: named.append(1) or False)     # the name asked: Cancel
    assert win.run.before_run() is False and named == [1] and win.s.dirty
    assert win.status.text() != "Project saved"


def test_quitting_after_view_changes_only_does_not_ask(win):
    page = _colour_full_saved(win)
    b, cv = page.bridge, page.canvas
    page.set_display("folded")
    page.set_display("table")
    page.set_display("blocks")
    cv.zoom_to(2.0)
    b.choose(next(iter(b.graph.blocks)))
    b.choose(None)
    page.palette.toggle()
    QApplication.processEvents()
    assert not win.s.dirty
    assert not _quit_asks(win, "close")
