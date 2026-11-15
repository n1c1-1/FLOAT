"""Calibration files dropped together where no kind is meant (a beta tester: a master dark and a master flat dropped on
a Session block asked one kind for both, « Darks » filed the flat as a dark). Each file whose FITS header says its kind
(IMAGETYP) is filed by it, the others are asked; the told ones are one undo step. In Inspect, a drop on a heading (not
on a kind's row nor on the frames of the kind shown) asks too: a folder of darks dropped on « Calibration » became
lights. Only headers are read: the names say the opposite on purpose."""
from __future__ import annotations

import numpy as np

from astro_gpu.client.graphview import files
from tests.client.graphview.test_gv_page import B, _drop_raw, page, pump, settle  # noqa: F401 - the fixtures


def fits_file(folder, name, kind, flt=""):
    from astropy.io import fits
    folder.mkdir(parents=True, exist_ok=True)
    h = fits.PrimaryHDU(np.zeros((4, 4), np.float32))
    h.header["IMAGETYP"] = kind
    if flt:
        h.header["FILTER"] = flt
    h.writeto(folder / name)
    return str(folder / name).replace("\\", "/")


def test_header_kinds_are_read_from_the_header_only(tmp_path):
    a = fits_file(tmp_path, "flat_0.fits", "Master Dark")
    b = fits_file(tmp_path, "dark_0.fits", "Flat Field")
    c = fits_file(tmp_path, "x.fits", "Bias Frame")
    d = fits_file(tmp_path, "y.fits", "DARKFLAT")
    e = fits_file(tmp_path, "z.fits", "LIGHT")
    f = fits_file(tmp_path, "w.fits", "")
    raw = tmp_path / "IMG_1.CR2"
    raw.write_bytes(bytes(64))
    told, rest = files.by_header_kind([a, b, c, d, e, f, str(raw)])
    assert told == {"darks": [a], "flats": [b], "bias": [c], "flat_darks": [d], "lights": [e]}
    assert rest == [f, str(raw).replace("\\", "/")]
    sub = tmp_path / "darks"
    for k in range(3):
        fits_file(sub, f"f{k}.fits", "Dark Frame")
    assert files.by_header_kind([str(sub)])[0] == {"darks": [str(sub).replace("\\", "/")]}
    fits_file(sub, "odd.fits", "Flat Field")                     # a folder of mixed kinds: asked
    assert files.by_header_kind([str(sub)])[1] == [str(sub).replace("\\", "/")]


def test_a_dark_and_a_flat_dropped_on_a_session_block_are_filed_by_their_headers(B, tmp_path):  # noqa: N803
    B.install().setup(("add_row", dict(kind="session")))
    s = B.one("session")
    dark = fits_file(tmp_path / "m", "masterFlat_SII.fit", "Master Dark")
    flat = fits_file(tmp_path / "m", "masterDark_300s.fit", "Master Flat")
    unknown = fits_file(tmp_path / "u", "a.fits", "")
    before = B.steps()
    B.g._see(B.g.block_point(s))
    B.canvas.kind_menu = None
    _drop_raw(B, B.g.block_point(s), [dark, flat])
    settle(B)
    g = B.graph.blocks[s].files
    assert g.get("darks") == [dark] and g.get("flats") == [flat] and not g.get("lights")
    assert B.steps() == before + 1 and B.canvas.kind_menu is None          # one undo step, nothing asked
    _drop_raw(B, B.g.block_point(s), [unknown])                            # a header that says nothing: asked
    menu = B.canvas.kind_menu
    assert menu is not None and [a.data() for a in menu.actions()] == ["lights", "darks", "flats", "bias"]
    assert B.steps() == before + 1


def test_inspect_drop_on_a_heading_asks_and_files_by_headers(qapp, tmp_path, monkeypatch):
    from PySide6.QtCore import QMimeData, QPoint, QPointF, Qt, QUrl
    from PySide6.QtGui import QDragEnterEvent, QDragMoveEvent, QDropEvent
    from PySide6.QtWidgets import QApplication
    from astro_gpu.client.sources import SourcesView
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
        darks = tmp_path / "darks"
        for k in range(2):
            fits_file(darks, f"L_{k}.fits", "Dark Frame")
        unknown = fits_file(tmp_path / "u", "a.fits", "")

        def drop(paths, at):
            mime = QMimeData()
            mime.setUrls([QUrl.fromLocalFile(p) for p in paths])
            QApplication.sendEvent(view, QDragEnterEvent(at, Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier))
            QApplication.sendEvent(view, QDragMoveEvent(at, Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier))
            QApplication.sendEvent(view, QDropEvent(QPointF(at), Qt.CopyAction, mime, Qt.LeftButton,
                                                    Qt.NoModifier))
            QApplication.processEvents()
        heading = view._left_w.mapTo(view, QPoint(150, 790))           # the left column, under its rows
        lights = list(m.p.session(1).lights)
        drop([str(darks)], heading)
        assert m.p.session(1).lights == lights and len(m.p.session(1).darks) == 2, said
        view.kind_menu = None
        drop([unknown], heading)
        assert view.kind_menu is not None and m.p.session(1).lights == lights      # asked, nothing added before
        next(a for a in view.kind_menu.actions() if a.data() == "bias").trigger()
        assert [str(x).replace("\\", "/") for x in m.p.session(1).bias] == [unknown], (m.p.session(1).bias, said)
        assert view.group == "bias"
        mid = view.stack.mapTo(view, QPoint(40, 40))                   # on the frames of the kind shown: that kind,
        other = fits_file(tmp_path / "o", "b.fits", "Dark Frame")      # whatever the header says
        drop([other], mid)
        assert len(m.p.session(1).bias) == 2 and len(m.p.session(1).darks) == 2
    finally:
        view.close()
        close(w, m, page)
