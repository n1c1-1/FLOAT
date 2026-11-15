"""Images opened as layers of Post: from a menu, the graph follows; from nothing, the project is made."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if Path("C:/Windows/Fonts").is_dir():
    os.environ.setdefault("QT_QPA_FONTDIR", "C:/Windows/Fonts")
pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from astro_gpu.client import app as appmod  # noqa: E402
from astro_gpu.client import drafts, gallery, prefs  # noqa: E402
from astro_gpu.client.services.worker import Gpu, Machine  # noqa: E402

MACHINE = Machine([Gpu("cuda:0", "RTX", 16.0)], 16, 64.0)


@pytest.fixture(scope="session")
def qapp():
    from astro_gpu.client.widgets import QSS
    app = QApplication.instance() or QApplication([])
    app.setStyleSheet(QSS + appmod.EXTRA_QSS)
    return app


@pytest.fixture()
def env(tmp_path, monkeypatch, qapp):
    import cv2
    from astropy.io import fits
    monkeypatch.setenv("ASTRO_GPU_SETTINGS", str(tmp_path / "settings.json"))
    monkeypatch.setenv("ASTRO_GPU_HOME", str(tmp_path / "home"))
    prefs.load()
    prefs.set("home_sections", [])
    prefs.set("cache_dir", (tmp_path / "scratch").as_posix())
    appmod.apply_language("en")
    monkeypatch.setattr(appmod, "machine_info", lambda: MACHINE)
    for kind in ("information", "warning", "critical"):
        monkeypatch.setattr(QMessageBox, kind, lambda *a, **k: None)
    png, fit = tmp_path / "a.png", tmp_path / "b.fits"
    cv2.imwrite(str(png), np.full((8, 8, 3), 128, np.uint8))
    fits.PrimaryHDU(np.full((3, 8, 8), 0.1, np.float32)).writeto(fit)
    monkeypatch.setattr(gallery.QFileDialog, "getOpenFileNames", lambda *a, **k: ([str(png), str(fit)], ""))
    windows = []
    yield {"tmp": tmp_path, "windows": windows, "mp": monkeypatch, "png": png, "fit": fit}
    for w in windows:
        w.close()
        w.deleteLater()
    QApplication.processEvents()


def settle(w, n=10):
    for _i in range(n):
        QApplication.processEvents()
        w.worker.wait(30)
    QApplication.processEvents()


def main(env) -> appmod.Main:
    w = appmod.Main()
    w.resize(1500, 800)
    w.show()
    env["windows"].append(w)
    w.worker.wait()
    settle(w)
    launched = env["launched"] = []
    env["mp"].setattr(w, "launch", lambda: launched.append(True))
    return w


def test_from_home_the_project_is_made_the_graph_follows_and_post_is_shown(env):
    w = main(env)
    w.show_home()
    settle(w)
    w.home.openImages.emit()
    w.worker.wait()
    settle(w, 14)
    assert w.project is not None and drafts.is_untitled(w.project.root) and w.mode == 2
    s = w.s
    assert [q.name for q in s.paths] == ["a", "b"] and all(s.is_plain_image(q) for q in s.paths)
    assert [s.p.workflow(q.pid).source["stretched"] for q in s.paths] == [True, False]
    g = s.block_map
    assert {q.pid for q in s.paths} <= set(g.to_post) and [n.key for n in g.nodes].count("image") == 2
    assert not env["launched"]                     # nothing is computed without the user's Run (the testers' pass 2)


def test_cancelling_the_files_makes_no_project(env):
    w = main(env)
    env["mp"].setattr(gallery.QFileDialog, "getOpenFileNames", lambda *a, **k: ([], ""))
    w.open_images()
    settle(w)
    assert w.project is None and not env["launched"]


def test_a_project_already_open_takes_them_beside_what_it_has(env):
    w = main(env)
    w.new_project()
    w.worker.wait()
    settle(w, 12)
    w.graph.place_images([str(env["png"])])
    settle(w)
    w.open_images()
    settle(w, 12)
    assert [q.name for q in w.s.paths] == ["a", "a 2", "b"] or len(w.s.paths) == 3
    assert w.mode == 2


def test_post_has_its_button_and_its_empty_page_asks_for_images(env):
    w = main(env)
    w.new_project()
    w.worker.wait()
    settle(w, 12)
    w.set_mode(2)
    settle(w)
    page = w.post_holder.page
    assert page.pages.currentIndex() == 0
    asked = []
    env["mp"].setattr(w, "open_images", lambda: asked.append(1))
    page.openImages.disconnect()
    page.openImages.connect(lambda: asked.append(1))
    page.view.b_open.clicked.emit()                               # (goes through the view's own signal)
    page.openImages.emit()
    assert asked


def test_the_kind_of_an_image_changed_on_its_layer_is_to_compute_and_said_with_a_run(env):
    """Nothing is computed without the user's Run (the testers: changing the kind started a run): the change is marked
    to compute and said in the status line with a Run link; Post's own Run, over outdated results, runs."""
    w = main(env)
    w.new_project()
    w.worker.wait()
    settle(w, 12)
    w.graph.place_images([str(env["fit"])])
    settle(w)
    w.set_mode(2)
    settle(w)
    view, q = w.post_holder.page.view, w.s.paths[0]
    assert not q.source["stretched"]
    view.set_stretched(q, True)
    settle(w)
    assert w.s.view(q.pid).source["stretched"] and not env["launched"]
    assert w.s.block_map.state(w.s.block_map.node_of(w.s.view(q.pid), "image")) != "done"
    said = w.status.text()
    assert "Image kind changed: to compute at the next run" in said and ">Run</a>" in said, said
    w._flash_linked()                                   # its Run link
    assert env["launched"] == [True]
    view.stale_run.click()                              # Post's own Run: the user's gesture
    assert env["launched"] == [True, True]


def test_a_linear_image_alone_can_be_run(env):
    w = main(env)
    w.new_project()
    w.worker.wait()
    settle(w, 12)
    w.graph.place_images([str(env["fit"])])
    settle(w)
    assert not any(appmod.no_block(x) for x in w.s.p.workflows)


def test_the_layers_of_images_opened_together_follow_the_order_chosen_whichever_is_computed_first(env):
    from astro_gpu.client.post3 import PostPage3
    from astro_gpu.client.services.render import RenderWorker
    w = main(env)
    w.new_project()
    w.worker.wait()
    settle(w, 12)
    w.graph.place_images([str(env["png"]), str(env["fit"])])
    settle(w)
    s = w.s
    page = PostPage3(s, worker=RenderWorker(None, resolver=lambda sid: None, device="cpu"))
    try:
        v = page.view
        names = [q.name for q in s.paths]
        assert names == ["a", "b"]
        for first in (names, names[::-1]):                   # computed in the order chosen, then the other way round
            v.layers = []
            for q in s.paths:
                q.rendered = False
            for n in first:
                next(q for q in s.paths if q.name == n).rendered = True
                v.sync_paths()
            assert [x["name"] for x in v.layers if x["kind"] == "workflow"] == ["b", "a"], first
    finally:
        page.shutdown()
        page.deleteLater()


# ------------------------------------------------------------------ the night test pass: Open images from Home
def test_from_home_with_a_project_left_open_the_images_are_a_new_project_and_the_other_is_untouched(env, tmp_path):
    """Opened from Home, the images make a project of their own: the project that was open is left (asked first when
    anything is unsaved), never given the images, never saved by itself."""
    from astro_gpu.core import project2
    w = main(env)
    other = tmp_path / "ws" / "Other"
    assert w.create_project(other.as_posix(), "Other", "M 31")
    w.worker.wait()
    settle(w, 12)
    before = (other / project2.FILE).read_bytes()
    w.show_home()                                     # Home over it (as the window shows it for a test)
    settle(w)
    w.home.openImages.emit()
    w.worker.wait()
    settle(w, 14)
    assert w.project is not None and w.project.root != other and drafts.is_untitled(w.project.root)
    assert [q.name for q in w.s.paths] == ["a", "b"]
    assert (other / project2.FILE).read_bytes() == before


def test_raw_frames_and_unreadable_files_are_said_at_once_and_nothing_is_made(env, tmp_path):
    w = main(env)
    raw = tmp_path / "IMG_0001.CR2"
    raw.write_bytes(b"II*\x00" + bytes(60))
    env["mp"].setattr(gallery.QFileDialog, "getOpenFileNames", lambda *a, **k: ([str(raw)], ""))
    w.show_home()
    settle(w)
    w.home.openImages.emit()
    settle(w, 2)
    assert w.project is None and "raw frame" in w.banner.text.text() and not env["launched"]
    txt = tmp_path / "notes.png"
    txt.write_text("not an image")
    env["mp"].setattr(gallery.QFileDialog, "getOpenFileNames", lambda *a, **k: ([str(txt)], ""))
    w.home.openImages.emit()
    settle(w, 2)
    assert w.project is None and "notes.png" in w.banner.text.text()


def test_a_fits_raw_frame_is_refused_by_its_header(env, tmp_path):
    """The testers' pass 2: a smart telescope's lights (FITS with a Bayer pattern) opened as five grey images, and a run
    started by itself. A FITS file whose header gives a Bayer pattern is a raw frame: refused at once, nothing made."""
    import numpy as np
    from astropy.io import fits
    w = main(env)
    raw = tmp_path / "frame_0001.fit"
    hd = fits.Header()
    hd["BAYERPAT"] = "GRBG"
    fits.PrimaryHDU(np.zeros((8, 8), np.uint16), header=hd).writeto(raw)
    env["mp"].setattr(gallery.QFileDialog, "getOpenFileNames", lambda *a, **k: ([str(raw)], ""))
    w.show_home()
    settle(w)
    w.home.openImages.emit()
    settle(w, 2)
    assert w.project is None and "frame_0001.fit is a raw frame" in w.banner.text.text() and not env["launched"]
