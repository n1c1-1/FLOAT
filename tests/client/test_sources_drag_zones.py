"""Session files page (#168): while files are dragged over it the centre shows four large zones with their counts; a
drop in a zone is files of that kind; they go away after the drop or when the drag leaves; an empty kind keeps them and
a click opens the file dialog for that kind."""
from __future__ import annotations

from PySide6.QtCore import QMimeData, QPoint, QPointF, Qt, QUrl
from PySide6.QtGui import QDragEnterEvent, QDragLeaveEvent, QDragMoveEvent, QDropEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from tests.client.graphview.test_gv_kind_is_the_place import labelled
from tests.client.test_sources_import import close, open_page, qapp  # noqa: F401


def mime_of(paths):
    m = QMimeData()
    m.setUrls([QUrl.fromLocalFile(str(p)) for p in paths])
    return m


def shown(tmp_path, monkeypatch):
    from astro_gpu.client.sources import SourcesView
    w, m, page = open_page(tmp_path, monkeypatch, sessions=1, workflows=("Full",))
    view = SourcesView(m)
    view.resize(1400, 800)
    view.show()
    view.show_session(1, "lights")
    QApplication.processEvents()
    return w, m, page, view


def test_drag_shows_four_zones_drop_on_darks_adds_darks_only_and_leave_hides(qapp, tmp_path, monkeypatch):
    from astro_gpu.client.sources import DropZone
    w, m, page, view = shown(tmp_path, monkeypatch)
    try:
        assert view.stack.currentIndex() == view.mode and view.held("lights")          # the list, as today
        files = labelled(tmp_path / "x", 2, "Ha", "Light Frame", "light")              # they say lights: not read
        mime, at = mime_of(files), QPoint(20, 20)
        QApplication.sendEvent(view.grid.viewport(), QDragEnterEvent(at, Qt.CopyAction, mime, Qt.LeftButton,
                                                                     Qt.NoModifier))
        QApplication.processEvents()
        zones = view.stack.currentWidget().findChildren(DropZone)
        assert view.stack.currentIndex() == 3 and [z.key for z in zones] == ["lights", "darks", "flats", "bias"]
        assert zones[0].count.text() == f"{view.held('lights')} files" and zones[1].count.text() == "No file"
        lights = list(m.p.session(1).lights)
        target = view.zones["darks"]
        QApplication.sendEvent(target, QDragEnterEvent(at, Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier))
        QApplication.sendEvent(target, QDragMoveEvent(at, Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier))
        QApplication.sendEvent(target, QDropEvent(QPointF(at), Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier))
        QTest.qWait(250)
        s = m.p.session(1)
        assert len(s.darks) == 2 and list(s.lights) == lights and not s.flats and not s.bias
        assert view.stack.currentIndex() == view.mode                                   # the zones went away
        # a drag that leaves the page
        QApplication.sendEvent(view.grid.viewport(), QDragEnterEvent(at, Qt.CopyAction, mime, Qt.LeftButton,
                                                                     Qt.NoModifier))
        assert view.stack.currentIndex() == 3
        QApplication.sendEvent(view.grid.viewport(), QDragLeaveEvent())
        QTest.qWait(250)
        assert view.stack.currentIndex() == view.mode
    finally:
        view.close()
        close(w, m, page)


def test_empty_session_keeps_the_zones_and_a_click_opens_the_dialog_for_its_kind(qapp, tmp_path, monkeypatch):
    from astro_gpu.client import sources
    w, m, page, view = shown(tmp_path, monkeypatch)
    try:
        view.show_session(1, "flats")
        QApplication.processEvents()
        assert view.stack.currentIndex() == 3 and set(view.zones) == {"lights", "darks", "flats", "bias"}
        files = labelled(tmp_path / "b", 2, "Ha", "Light Frame", "b")
        asked = []
        monkeypatch.setattr(sources.QFileDialog, "getOpenFileNames",
                            staticmethod(lambda *a, **k: (asked.append(1) or [str(f) for f in files], "")))
        QTest.mouseClick(view.zones["bias"], Qt.LeftButton)
        assert asked and len(m.p.session(1).bias) == 2 and not m.p.session(1).flats
    finally:
        view.close()
        close(w, m, page)
