"""#186: a session's frames have two tabs. Load shows the five kinds' zones with their counts before and after the
lights alike (a click on a zone is the file dialog for its kind, a drop on it files of its kind); Inspect shows the
frames. Inspect opens on Load, from the graph too. A drop off every zone and row is refused, said in the window's strip
(not a word in the status line, gone before it was read). The graph's notice after a drop names the session."""
from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if Path("C:/Windows/Fonts").is_dir():
    os.environ.setdefault("QT_QPA_FONTDIR", "C:/Windows/Fonts")
pytest.importorskip("PySide6")
from PySide6.QtCore import QMimeData, QPoint, QPointF, Qt, QUrl  # noqa: E402
from PySide6.QtGui import QDragEnterEvent, QDragMoveEvent, QDropEvent  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication, QPushButton  # noqa: E402

from astro_gpu.client import app as appmod  # noqa: E402
from astro_gpu.client import i18n, prefs  # noqa: E402
from tests.client.graphview.test_gv_kind_is_the_place import labelled  # noqa: E402
from tests.client.pagehelp import choose, watch_toasts  # noqa: E402
from tests.client.test_sources_import import close, open_page, settle  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    from astro_gpu.client.widgets import QSS
    a = QApplication.instance() or QApplication([])
    a.setStyleSheet(QSS + appmod.EXTRA_QSS)
    return a


def _settle(w, n=8):
    for _i in range(n):
        w.worker.wait(30)
        QApplication.processEvents()
        time.sleep(0.01)


@pytest.fixture()
def win(qapp, tmp_path, monkeypatch):
    monkeypatch.setenv("ASTRO_GPU_SETTINGS", str(tmp_path / "settings.json"))
    prefs.load()
    appmod.apply_language("en")
    from astro_gpu.client.gallery import GALLERY
    w = appmod.Main()
    w.resize(1500, 900)
    w.show()
    w.new_project()
    end = time.monotonic() + 120
    while time.monotonic() < end and not w.s.ready:
        w.worker.wait(1)
        QApplication.processEvents()
    w.graph.show_templates()
    w.graph.templates.chosen.emit(next(t for t in GALLERY if t["key"] == "colour_full"))
    _settle(w)
    yield w
    w.close_project(discard=True)
    w.close()
    w.deleteLater()
    QApplication.processEvents()


def drop_on(target, paths, at=QPoint(10, 10)):
    """Files from the file explorer dragged onto ``target`` and dropped there."""
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(p)) for p in paths])
    QApplication.sendEvent(target, QDragEnterEvent(at, Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier))
    QApplication.sendEvent(target, QDragMoveEvent(at, Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier))
    QApplication.sendEvent(target, QDropEvent(QPointF(at), Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier))
    QApplication.processEvents()


def session_block(s, sid):
    return next(b for b, x in s.stored.blocks.items() if x.kind == "session" and getattr(x, "sid", None) == sid)


def titles(view):
    return {k: z.title.text() for k, z in view.zones.items()}


def test_load_shows_the_five_kinds_after_the_lights_and_inspect_opens_on_it(win, tmp_path, monkeypatch):
    from astro_gpu.client import sources
    s, page = win.s, win.graph
    sid = s.p.sessions[0].id
    block = session_block(s, sid)
    # the lights, dropped on the session in the graph; then « Open frames » of its panel
    page.bridge.drops.drop(block, "lights", labelled(tmp_path / "l", 3, "", "Dark Frame", "light"))
    _settle(win)
    lights = list(s.p.session(sid).lights)
    assert lights
    choose(page, block)
    _settle(win)
    opens = [b for b in page.panel.findChildren(QPushButton) if b.text() == "Open frames" and b.isVisible()]
    assert len(opens) == 1
    opens[0].click()
    _settle(win)
    view, frames = win.browser, win.frames
    assert win.mode == 1 and frames.pages.currentIndex() == 1 and view.tab == "load"
    assert frames.tabs["load"].isChecked() and view.stack.currentIndex() == 3 and view.sid == sid
    assert titles(view) == {"lights": "Lights · 3", "darks": "Darks · 0", "flats": "Flats · 0", "bias": "Bias · 0",
                            "flat_darks": "Dark flats · 0"}
    assert not view._right_w.isVisible() and not view._bar_w.isVisible()          # the zones take the room
    # a click on Darks: the file dialog, its files darks (their headers say lights: never read)
    darks = labelled(tmp_path / "d", 2, "", "Light Frame", "d")
    monkeypatch.setattr(sources.QFileDialog, "getOpenFileNames", staticmethod(lambda *a, **k: (darks, "")))
    QTest.mouseClick(view.zones["darks"], Qt.LeftButton)
    _settle(win)
    ses = s.p.session(sid)
    assert len(ses.darks) == 2 and list(ses.lights) == lights and not ses.flats
    assert view.tab == "load" and view.stack.currentIndex() == 3 and titles(view)["darks"] == "Darks · 2"
    # a drop on Flats: flats only
    drop_on(view.zones["flats"], labelled(tmp_path / "f", 2, "", "Dark Frame", "f"))
    _settle(win)
    ses = s.p.session(sid)
    assert len(ses.flats) == 2 and len(ses.darks) == 2 and not ses.bias and titles(view)["flats"] == "Flats · 2"
    # Inspect: the frames; a drop off every zone and row is refused, said in the strip until files are added
    frames.tabs["inspect"].click()
    _settle(win)
    assert view.tab == "inspect" and view.stack.currentIndex() == view.mode and view._bar_w.isVisible()
    bias = labelled(tmp_path / "b", 1, "", "Bias Frame", "b")
    drop_on(view.viewer, bias, QPoint(40, 40))                                  # on the picture
    _settle(win)
    ses = s.p.session(sid)
    assert not ses.bias and len(ses.flats) == 2 and len(ses.darks) == 2
    assert win.banner.isVisible() and win.banner.text.text() == sources.DROP_REFUSED
    assert sources.DROP_REFUSED not in win.status.text()                       # not the word that went at once
    drop_on(view.zones["bias"], bias)                                           # (the zones show while dragging)
    _settle(win)
    assert len(s.p.session(sid).bias) == 1 and not win.banner.isVisible()
    # back to the graph and to Inspect again: Load
    win.set_mode(0)
    _settle(win, 3)
    win.set_mode(1)
    _settle(win)
    assert view.tab == "load" and frames.tabs["load"].isChecked() and view.stack.currentIndex() == 3


def test_the_graph_notice_after_a_drop_names_the_session(qapp, tmp_path, monkeypatch):
    """A one-night workflow's Session block is titled by its workflow (« Full »): the notice named it, where Inspect's
    names the session."""
    w, m, page = open_page(tmp_path, monkeypatch, sessions=1, workflows=("Full",))
    try:
        toasts = watch_toasts(page)
        block = session_block(m, 1)
        assert page.bridge.doc.title(block) == "Full"
        page.bridge.drops.drop(block, "darks", labelled(tmp_path / "d", 2, "", "Light Frame", "d"))
        settle(w, page)
        assert toasts and toasts[-1].startswith("2 darks added to “night 1”") and "Full" not in toasts[-1]
    finally:
        close(w, m, page)


def test_the_tabs_and_the_refusal_have_their_french():
    from astro_gpu.client.frames import TABS
    from astro_gpu.client.sources import DROP_REFUSED
    i18n.set_language("fr")
    try:
        assert [i18n.in_context("frames", w) for _k, w, _r, _t in TABS] == ["Charger", "Inspecter", "Sessions"]
        assert all(i18n._(t) != t for _k, _w, _r, t in TABS)
        assert i18n._(DROP_REFUSED).startswith("Non ajoutés : déposez les fichiers sur la zone ou la ligne")
    finally:
        i18n.set_language("en")
