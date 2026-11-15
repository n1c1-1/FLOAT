"""The viewer when the image asked for cannot be read: Export and Send to Post never write or send the picture shown
before it under the new name (greyed, the reason on hover); an export window is freed once closed."""
from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
pytest.importorskip("PySide6")
import shiboken6  # noqa: E402
from PySide6.QtCore import QEvent  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from astro_gpu.client import imageview as iv  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture()
def view(qapp):
    ld = iv.Loader()
    v = iv.ImageView(loader=ld, post=True)
    v.resize(900, 600)
    v.show()
    QApplication.processEvents()
    yield v
    v.close()
    v.deleteLater()
    QApplication.processEvents()
    ld.stop()


def _sky(seed=1):
    rng = np.random.default_rng(seed)
    return (200 + rng.normal(0, 6, (3, 120, 160))).astype(np.float32)


def _show(v, image):
    v.set_image(image)
    assert v.wait(30)
    QApplication.processEvents()


def _unreadable():
    raise OSError("disk gone")


def test_an_unreadable_image_is_never_exported_or_sent_as_the_previous_one(view, tmp_path):
    _show(view, iv.ViewImage(_sky(), "First", key="first"))
    assert view.export_b.isEnabled() and view.post_b.isEnabled()
    sent = []
    view.sendToPost.connect(sent.append)
    _show(view, iv.ViewImage(_unreadable, "Second", key="second"))
    assert not view.export_b.isEnabled() and not view.post_b.isEnabled()
    said = view.canvas.message                                   # the reason, as the image says it
    assert said and view.export_b.toolTip() == said and view.post_b.toolTip() == said
    view._send_post()
    assert sent == []
    assert view.export_dialog() is None
    with pytest.raises(RuntimeError):
        view.export(tmp_path / "x.fits")
    assert not (tmp_path / "x.fits").exists()
    _show(view, iv.ViewImage(_sky(2), "Third", key="third"))  # a readable one again: both back
    assert view.export_b.isEnabled() and view.post_b.isEnabled()
    view._send_post()
    assert sent == ["third"]


def test_an_export_window_is_freed_once_closed(view, tmp_path):
    _show(view, iv.ViewImage(_sky(), "First", key="first"))
    dlg = view.export_dialog()
    dlg.reject()
    QApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    QApplication.processEvents()
    assert not shiboken6.isValid(dlg)
    dlg = view.export_dialog()                                    # written, then closed: freed too
    dlg.dest.setText((tmp_path / "x.fits").as_posix())
    dlg.go.click()
    assert dlg.wait(30) and dlg.result_path is not None
    dlg.go.click()
    QApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    QApplication.processEvents()
    assert not shiboken6.isValid(dlg) and (tmp_path / "x.fits").is_file()


@pytest.mark.parametrize("kw", [{}, {"units": "count"}, {"units": ""}, {"units": "norm"}, {"stretched": True}])
def test_a_pixel_that_is_not_a_number_is_read_out_without_an_error(view, kw):
    img = _sky()
    img[:, 10, 20] = np.nan                                       # outside the covered area of a stack
    _show(view, iv.ViewImage(img, "Holes", key=("nan", str(kw)), **kw))
    text = view.pixel_text(20.5, 10.5)
    assert text.startswith("x    20  y    10")
    if kw.get("units") == "count" or kw.get("stretched"):         # whole numbers: a dash where there is none
        assert "R –   G –   B –" in text
