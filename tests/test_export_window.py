"""The export windows go back to ready after an export, and their folder button picks a folder only."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QApplication, QFileDialog

from astro_gpu.client import exportdlg, imageview
from astro_gpu.client.imageview import ImageExportDialog


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


class FakeView(QObject):
    toasted = Signal(str)

    def __init__(self, folder):
        super().__init__()
        self.export_dir = str(folder)
        self.default_dir = None
        self.prep = SimpleNamespace(width=8, height=6, wcs=None, vi=SimpleNamespace(name="Master flat"))
        self.canvas = SimpleNamespace(whole_visible=lambda: True)

    def prepared(self):
        return self.prep

    def image(self):
        return self.prep.vi

    def stretch(self):
        return None

    def _size_text(self, prep):
        return "8 x 6"

    def visible_area(self):
        return (0, 0, 8, 6)


def fake_writer(monkeypatch):
    def write(path, *a, **k):
        from pathlib import Path
        Path(path).write_text("x")
        return Path(path)
    monkeypatch.setattr(imageview, "export_image", write)


def test_a_change_after_an_export_makes_the_next_one_possible(app, tmp_path, monkeypatch):
    fake_writer(monkeypatch)
    d = ImageExportDialog(FakeView(tmp_path))
    d._go()
    assert d.wait() and d.result_path.suffix == ".fits"
    assert d.go.text() == "Close" and not d.cancel_b.isEnabled()
    d.fmt.setCurrentIndex(d.fmt.findData("tiff16"))
    assert d.go.text() == "Export" and d.cancel_b.isEnabled() and "Written" not in d.status.text()
    d._go()
    assert d.wait() and d.result_path.suffix == ".tif" and d.result_path.exists()
    assert len(list(tmp_path.iterdir())) == 2


def test_post_export_window_goes_back_to_ready(app):
    """ExportDialog._ready on a bare object: the button, Cancel and the status after a first export."""
    from PySide6.QtWidgets import QProgressBar, QPushButton

    from astro_gpu.client.i18n import _
    o = SimpleNamespace(running=False, go=QPushButton(_("Close")), cancel=QPushButton(), bar=QProgressBar(),
                        status=QPushButton("12 files written"))
    o.cancel.setEnabled(False)
    exportdlg.ExportDialog._ready(o)
    assert o.go.text() == "Export" and o.cancel.isEnabled() and o.status.text() == ""
def test_the_folder_button_picks_a_folder_and_keeps_the_name(app, tmp_path, monkeypatch):
    other = tmp_path / "other"
    other.mkdir()
    d = ImageExportDialog(FakeView(tmp_path))
    d.dest.setText((tmp_path / "my_name.fits").as_posix())
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *a, **k: str(other))
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: pytest.fail("the save dialog was opened"))
    d._browse()
    assert d.dest.text() == (other / "my_name.fits").as_posix()
    assert d.view.export_dir == other.as_posix()


