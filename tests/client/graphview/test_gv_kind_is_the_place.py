"""The software never decides what a file is. The kind of a frame is the place where the user drops or adds it (a
Session block is the place of lights; a kind's target or row, of its kind); the filter of a session is a name the user
or the template gives. FITS labels (FILTER, IMAGETYP…) are never read to file, name, ask or warn: the frames below say
the opposite of the place in their headers and their names, on purpose."""
from __future__ import annotations

import numpy as np

from tests.client.graphview.test_gv_page import B, _drop_raw, page, pump, settle  # noqa: F401 - the fixtures


def labelled(folder, n, flt, kind, name):
    """``n`` FITS frames whose header says filter ``flt`` and image type ``kind``, named ``name``_k.fits."""
    from astropy.io import fits
    folder.mkdir(parents=True, exist_ok=True)
    out = []
    for k in range(n):
        h = fits.PrimaryHDU(np.zeros((4, 4), np.float32))
        h.header["FILTER"] = flt
        h.header["IMAGETYP"] = kind
        h.header["FRAMETYP"] = kind
        h.header["OBJECT"] = "Flat field"
        h.writeto(folder / f"{name}_{k}.fits")
        out.append(str(folder / f"{name}_{k}.fits").replace("\\", "/"))
    return out


def night_filter(graph, block):
    sid = graph.blocks[block].sid
    return str(next((n for n in graph.nights if n.get("sid") == sid), {}).get("filter") or "")


def test_frames_are_filed_by_where_they_are_dropped_never_by_their_headers(B, tmp_path):  # noqa: N803
    from PySide6.QtCore import QMimeData, QUrl, Qt
    from PySide6.QtGui import QDragEnterEvent, QDragMoveEvent
    from PySide6.QtWidgets import QApplication
    B.install().setup(("add_row", dict(kind="session")))
    s = B.one("session")
    B.setup(("set_night_filter", dict(night=s, filter="Ha")))           # the user's name for the session
    darks = labelled(tmp_path / "d", 2, "OIII", "Master Dark", "masterDark")
    B.g._see(B.g.block_point(s))
    mime = QMimeData()                                                  # dragged over the block: it says lights
    mime.setUrls([QUrl.fromLocalFile(p) for p in darks])
    vp, at = B.g.vp, B.g.block_point(s)
    QApplication.sendEvent(vp, QDragEnterEvent(at, Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier))
    QApplication.sendEvent(vp, QDragMoveEvent(at, Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier))
    pump()
    assert B.canvas.tip.text == "Drop lights"
    B.canvas.dragLeaveEvent(None)
    before = B.steps()
    _drop_raw(B, at, darks)                                             # on the Session block itself: lights
    settle(B)
    got = B.graph.blocks[s].files or {}
    assert got.get("lights") == darks and not got.get("darks"), got
    assert not hasattr(B.canvas, "kind_menu") and B.steps() == before + 1          # nothing asked, one undo step
    assert "2 lights added" in B.canvas.toast.plain and [k for _t, k in B.canvas.toast.links()] == ["undo"]
    assert night_filter(B.graph, s) == "Ha"                             # the header's OIII never read
    lights = labelled(tmp_path / "f", 1, "SII", "Light Frame", "light")
    B.b.drops.drop(s, "flats", lights)                                  # on the Flats target: a flat
    assert (B.graph.blocks[s].files or {}).get("flats") == lights and B.steps() == before + 2
    assert night_filter(B.graph, s) == "Ha" and B.links[-1] == [("Undo", "undo")]


def test_a_new_stack_made_by_a_drop_takes_lights_and_no_filter(qapp, tmp_path):
    from tests.client.graphview.test_gv_table import Bench
    folder = tmp_path / "x"
    labelled(folder, 3, "Ha", "Dark Frame", "dark")
    b = Bench(1366, blank=True)
    b.drop([folder], ("ghost_p",))                                      # « + Stack »
    s = b.sessions()[0]
    assert len(b.doc.files_of(s, "lights")[1]) == 3 and not b.doc.files_of(s, "darks")[1]
    assert night_filter(b.doc.graph, s) == "" and "New stack added" in b.t.toast_text
    b.close()


def test_inspect_takes_files_on_a_kind_row_only(qapp, tmp_path, monkeypatch):
    from PySide6.QtCore import QMimeData, QPoint, QPointF, Qt, QUrl
    from PySide6.QtGui import QDragEnterEvent, QDragMoveEvent, QDropEvent
    from PySide6.QtWidgets import QApplication
    from astro_gpu.client.sources import GroupRow, SourcesView
    from tests.client.test_sources_import import close, open_page
    w, m, page = open_page(tmp_path, monkeypatch, sessions=1, workflows=("Full",))
    view = SourcesView(m)
    view.resize(1400, 800)
    view.show()
    said: list = []
    view.toast.connect(said.append)
    try:
        view.show_session(1, "lights")
        QApplication.processEvents()
        flt = m.p.session(1).filter
        darks = labelled(tmp_path / "darks", 2, "OIII", "Dark Frame", "dark")

        def drop(target, paths, at):
            mime = QMimeData()
            mime.setUrls([QUrl.fromLocalFile(p) for p in paths])
            QApplication.sendEvent(target, QDragEnterEvent(at, Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier))
            QApplication.sendEvent(target, QDragMoveEvent(at, Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier))
            QApplication.sendEvent(target, QDropEvent(QPointF(at), Qt.CopyAction, mime, Qt.LeftButton,
                                                      Qt.NoModifier))
            QApplication.processEvents()
        lights = list(m.p.session(1).lights)
        for at in (view._left_w.mapTo(view, QPoint(150, 790)), view.stack.mapTo(view, QPoint(40, 40))):
            drop(view, darks, at)                                       # a heading, the frames: nothing added
            assert m.p.session(1).lights == lights and not m.p.session(1).darks and not hasattr(view, "kind_menu")
            assert said[-1] == "Drop on a row"
        row = next(r for r in view.findChildren(GroupRow) if r.key == "bias")
        drop(row, darks, QPoint(10, 10))                                # on the Bias row: biases, whatever they say
        assert len(m.p.session(1).bias) == 2 and not m.p.session(1).darks and m.p.session(1).lights == lights
        assert m.p.session(1).filter == flt
    finally:
        view.close()
        close(w, m, page)


def test_inspect_centre_takes_no_drop_and_says_so(qapp, tmp_path, monkeypatch):
    from PySide6.QtCore import QMimeData, QPoint, QPointF, Qt, QUrl
    from PySide6.QtGui import QDragEnterEvent, QDragMoveEvent, QDropEvent
    from PySide6.QtWidgets import QApplication, QLabel
    from astro_gpu.client import i18n
    from astro_gpu.client.sources import GroupRow, SourcesView
    from tests.client.test_sources_import import close, open_page
    w, m, page = open_page(tmp_path, monkeypatch, sessions=1, workflows=("Full",))
    view = SourcesView(m)
    view.resize(1400, 800)
    view.show()
    said: list = []
    view.toast.connect(said.append)
    try:
        view.show_session(1, "lights")
        QApplication.processEvents()
        darks = labelled(tmp_path / "darks", 2, "OIII", "Dark Frame", "dark")
        mime = QMimeData()
        mime.setUrls([QUrl.fromLocalFile(p) for p in darks])
        lights = list(m.p.session(1).lights)
        for target in (view.grid.viewport(), view.table.viewport(), view.viewer, view.stack):
            at = QPoint(20, 20)
            ev = QDragEnterEvent(at, Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier)
            QApplication.sendEvent(target, ev)
            assert ev.isAccepted() and ev.dropAction() == Qt.IgnoreAction     # the cursor says not allowed
            QApplication.sendEvent(target, QDragMoveEvent(at, Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier))
            assert all("dashed" in r.styleSheet() for r in view.findChildren(GroupRow))   # the rows are lit
            said.clear()
            QApplication.sendEvent(target, QDropEvent(QPointF(at), Qt.CopyAction, mime, Qt.LeftButton,
                                                      Qt.NoModifier))
            assert said == ["Drop on a row"]
            assert m.p.session(1).lights == lights and not m.p.session(1).darks and not m.p.session(1).bias
            assert not any("dashed" in r.styleSheet() for r in view.findChildren(GroupRow))
        view.show_session(1, "darks")                                   # an empty kind: one plain message
        QApplication.processEvents()
        assert view.stack.currentIndex() == 3
        texts = [x.text() for x in view.stack.currentWidget().findChildren(QLabel)]
        assert "Drop your files on their zone, or use Add files." in texts
        i18n.set_language("fr")
        try:
            assert i18n._("Drop your files on their zone, or use Add files.") == (
                "Déposez vos fichiers sur leur zone, ou utilisez Ajouter fichiers.")
            assert i18n._("Drop on a row") == "Déposez sur une ligne"
        finally:
            i18n.set_language("en")
    finally:
        view.close()
        close(w, m, page)


def test_an_empty_kind_shows_one_drop_zone_per_kind_and_each_takes_its_own(qapp, tmp_path, monkeypatch):
    from PySide6.QtCore import QMimeData, QPoint, QPointF, Qt, QUrl
    from PySide6.QtGui import QDragEnterEvent, QDragMoveEvent, QDropEvent
    from PySide6.QtWidgets import QApplication
    from astro_gpu.client.sources import GROUPS, DropZone, SourcesView
    from tests.client.test_sources_import import close, open_page
    w, m, page = open_page(tmp_path, monkeypatch, sessions=1, workflows=("Full",))
    view = SourcesView(m)
    view.resize(1400, 800)
    view.show()
    try:
        view.show_session(1, "darks")
        QApplication.processEvents()
        assert view.stack.currentIndex() == 3
        zones = {z.key: z for z in view.stack.currentWidget().findChildren(DropZone)}
        assert list(zones) == [k for k, _t, _s in GROUPS]               # one per kind, flat darks too
        assert not m.p.session(1).flats
        files = labelled(tmp_path / "f", 2, "OIII", "Dark Frame", "dark")      # what the files say is not read
        mime = QMimeData()
        mime.setUrls([QUrl.fromLocalFile(p) for p in files])
        at = QPoint(10, 10)
        target = zones["flats"]
        QApplication.sendEvent(target, QDragEnterEvent(at, Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier))
        QApplication.sendEvent(target, QDragMoveEvent(at, Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier))
        QApplication.sendEvent(target, QDropEvent(QPointF(at), Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier))
        QApplication.processEvents()
        assert len(m.p.session(1).flats) == 2 and not m.p.session(1).darks     # the zone is the kind
    finally:
        view.close()
        close(w, m, page)
