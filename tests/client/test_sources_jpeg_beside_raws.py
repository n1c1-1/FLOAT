"""Sources / Inspect « Add files… », « Add folder… » and a drop on a kind's row: a camera's JPEGs beside its raw or FITS
frames are left (the engine reads a stored file as it is: each JPEG would be one more frame, counted and possibly
stacked instead of the raws), said with « Take all »; a session saved before with such JPEGs says it, with a button
that takes them out. The stored entries and the engine's listing agree with what is said."""
from __future__ import annotations

from pathlib import Path

from astro_gpu.core.project import list_frames
from tests.client.test_sources_import import close, fits_file, open_page, qapp, settle, texts  # noqa: F401


def night(folder: Path, n=8):
    for k in range(n):
        fits_file(folder / f"DSC{k:05d}.fits")
        (folder / f"DSC{k:05d}.JPG").write_bytes(bytes(64))
    return sorted(str(x) for x in folder.iterdir())


def test_sources_jpeg_beside_raws_left_taken_and_removed(qapp, tmp_path, monkeypatch):
    from astro_gpu.client.sources import SourcesView
    picked = night(tmp_path / "night")
    w, m, page = open_page(tmp_path, monkeypatch, sessions=1, workflows=("Full",))
    view = SourcesView(m)
    said: list = []
    view.toast.connect(said.append)
    try:
        sid = m.add_filter_session("")
        view.show_session(sid, "lights")
        assert view.add_files("lights", picked) == 8                       # the file dialog: 16 files selected
        stored = m.p.session(sid).lights
        assert len(stored) == 8 and all(x.endswith(".fits") for x in stored) and len(list_frames(stored)) == 8
        assert "8 JPG left: beside raw frames" in said[-1]
        settle(w, page)
        view.refresh()
        assert "Take all" in texts(view) and any("8 JPG left" in t for t in texts(view))
        view.take_all("lights")                                            # the user's own choice
        assert len(list_frames(m.p.session(sid).lights)) == 16
        settle(w, page)
        view.refresh()
        assert len(view.beside_raws("lights")) == 8 and "Remove JPG" in texts(view)
        assert not any("not frames the engine reads here" in t for t in view.anomalies("lights"))
        view.remove_beside_raws("lights")
        assert len(m.p.session(sid).lights) == 8 and not view.beside_raws("lights")
        sid2 = m.add_filter_session("")                                    # « Add folder… »
        view.show_session(sid2, "lights")
        assert view.add_files("lights", [str(tmp_path / "night")]) == 8 and "8 JPG left" in said[-1]
        assert len(list_frames(m.p.session(sid2).lights)) == 8
    finally:
        view.close()
        close(w, m, page)


def test_sources_lights_of_another_session_of_the_stack_are_said_not_added(qapp, tmp_path, monkeypatch):
    from astro_gpu.client.sources import SourcesView
    w, m, page = open_page(tmp_path, monkeypatch, sessions=2, workflows=("Full",))
    view = SourcesView(m)
    said: list = []
    view.toast.connect(said.append)
    try:
        first = m.session_files(1, "lights")
        view.show_session(2, "lights")
        before = list(m.p.session(2).lights)
        assert view.add_files("lights", first) == 0 and m.p.session(2).lights == before
        assert said[-1] == "3 already in “night 1”: not added"
        settle(w, page)
        view.refresh()
        assert "Add anyway" in texts(view)
        view.add_anyway("lights")
        assert len(m.p.session(2).lights) == len(before) + 3
    finally:
        view.close()
        close(w, m, page)


def _drop(widget, paths, pt=None):
    """Files dropped as the Explorer sends them: enter, a move (a fresh event, not accepted beforehand), the drop."""
    from PySide6.QtCore import QMimeData, QPoint, QPointF, Qt, QUrl
    from PySide6.QtGui import QDragEnterEvent, QDragMoveEvent, QDropEvent
    from PySide6.QtWidgets import QApplication
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(p) for p in paths])
    pt = QPoint(10, 10) if pt is None else pt
    enter = QDragEnterEvent(pt, Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier)
    QApplication.sendEvent(widget, enter)
    move = QDragMoveEvent(pt, Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier)
    move.ignore()
    QApplication.sendEvent(widget, move)
    if not (enter.isAccepted() and move.isAccepted()):
        return False
    QApplication.sendEvent(widget, QDropEvent(QPointF(pt), Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier))
    return True


def test_sources_take_dropped_files_on_the_page_and_on_a_kinds_row(qapp, tmp_path, monkeypatch):
    """The testers' pass 2: in Inspect's Files page every drop was refused, the zone "Drop your files here" included:
    the page took the files as they entered, not as they moved over it."""
    from astro_gpu.client.sources import GroupRow, SourcesView
    darks = [x for x in night(tmp_path / "darks") if x.endswith(".fits")][:2]
    w, m, page = open_page(tmp_path, monkeypatch, sessions=1, workflows=("Full",))
    view = SourcesView(m)
    try:
        sid = m.p.sessions[0].id
        view.show_session(sid, "darks")
        mid = view.stack.mapTo(view, view.stack.rect().center())  # on the kind shown (elsewhere: asked)
        assert _drop(view, [darks[0]], mid) and len(m.p.session(sid).darks) == 1
        view.refresh()
        row = next(r for r in view.findChildren(GroupRow) if r.key == "flats")
        assert _drop(row, [darks[1]])
        assert len(m.p.session(sid).flats) == 1
    finally:
        view.close()
        close(w, m, page)
