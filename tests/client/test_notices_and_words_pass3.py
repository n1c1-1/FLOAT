"""The beta testers' third pass, notices and words: the « run goes on » strip goes with the run; files dropped where no
page takes them are said; Open project takes a project's own folder, says a project already open, and asks about
unsaved edits before its file dialog; the status line never says « all up to date » with nothing computed; quotes
of the interface's language around names; « Add lights » opens the file dialog."""
from __future__ import annotations

from PySide6.QtCore import QMimeData, QPoint, QPointF, Qt, QUrl
from PySide6.QtGui import QDragEnterEvent, QDragMoveEvent, QDropEvent
from PySide6.QtWidgets import QApplication

from astro_gpu.client import i18n
from astro_gpu.core.project2 import Project2
from tests.client.test_one_undo import _set_rbf, _settle, qapp, win  # noqa: F401  (fixtures by name)


def test_the_run_goes_on_strip_goes_with_the_run(win):
    rc = win.run
    rc._snapshot = Project2.from_dict(win.s.p.to_dict(), win.s.p.root)
    rc.phase, rc._told_edit = "running", False
    _set_rbf(win)
    _settle(win, 2)
    assert win.banner.isVisible()
    rc.phase = "idle"                                                  # stopped, or ended
    assert not win.banner.isVisible()
    win.banner.say("Something else")                                   # another notice is never taken away
    rc.phase = "running"
    rc.phase = "idle"
    assert win.banner.isVisible()
    win.banner.hide()


def _drop_on_window(win, paths):
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(p)) for p in paths])
    at = QPoint(5, 5)
    QApplication.sendEvent(win, QDragEnterEvent(at, Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier))
    QApplication.sendEvent(win, QDragMoveEvent(at, Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier))
    QApplication.sendEvent(win, QDropEvent(QPointF(at), Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier))
    QApplication.processEvents()


def test_files_dropped_where_nothing_takes_them_are_said(win, tmp_path):
    f = tmp_path / "a.fits"
    f.write_bytes(bytes(64))
    win.set_mode(1)
    _settle(win, 2)
    _drop_on_window(win, [f])
    assert win.status.text().startswith("Frames go on a Session block")
    win.set_mode(0)
    win.graph.show_templates()
    _drop_on_window(win, [f])
    assert win.status.text().startswith("Choose a template first")
    win.graph.hide_templates()


def test_open_project_on_the_open_project_says_so(win):
    win.open_project(win.project.root)
    assert win.status.text().endswith("is already open")


def test_choose_project_takes_the_projects_own_folder(tmp_path, monkeypatch):
    from astro_gpu.client import home
    d = tmp_path / "Pelican"
    d.mkdir()
    monkeypatch.setattr(home.QFileDialog, "getOpenFileName", lambda *a, **k: (d.as_posix(), ""))
    assert home.choose_project(None) == d.as_posix()


def test_open_project_asks_about_unsaved_edits_before_its_dialog(win, monkeypatch):
    from astro_gpu.client import home
    _set_rbf(win)
    order = []
    monkeypatch.setattr(win, "ask_save", lambda: order.append("asked") or "cancel")
    monkeypatch.setattr(home, "choose_project", lambda parent: order.append("dialog") or None)
    win.open_other()
    assert order == ["asked"]                                          # cancelled: no dialog
    monkeypatch.setattr(win, "ask_save", lambda: order.append("asked") or "discard")
    win.open_other()
    assert order == ["asked", "asked", "dialog"]


def test_the_status_line_never_says_all_up_to_date_with_nothing_computed(win):
    win.update_run_button()
    assert "all up to date" not in win.status.text()


def test_names_are_quoted_in_the_interfaces_language():
    i18n.set_language("fr")
    try:
        assert i18n.quoted("Session 1") == "« Session 1 »"
    finally:
        i18n.set_language("en")
    assert i18n.quoted("Session 1") == "“Session 1”"


def test_add_lights_opens_the_file_dialog(win, tmp_path, monkeypatch):
    from astro_gpu.client import panel as panelmod
    seen = []
    monkeypatch.setattr(panelmod.QFileDialog if hasattr(panelmod, "QFileDialog") else
                        __import__("PySide6.QtWidgets", fromlist=["QFileDialog"]).QFileDialog,
                        "getOpenFileNames", lambda *a, **k: seen.append(True) or ([], ""))
    pn = win.graph.panel
    pn.pick_lights(next(iter(x.id for x in win.s.p.sessions)))
    assert seen == [True]
