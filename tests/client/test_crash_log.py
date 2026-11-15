"""An unexpected error is added to the crash log in the application's own folder (beside its settings): its date, the
program's version, the page shown and the whole traceback, so that it can be read after the box is closed."""
from __future__ import annotations

import re
import sys

from astro_gpu import __version__
from astro_gpu.client import app as appmod
from astro_gpu.client import prefs


def test_a_crash_is_written_with_its_date_version_page_and_traceback(tmp_path, monkeypatch):
    monkeypatch.setenv("ASTRO_GPU_SETTINGS", str(tmp_path / "settings.json"))
    prefs.load()
    try:
        raise RuntimeError("the cause")
    except RuntimeError:
        import traceback
        text = "".join(traceback.format_exception(*sys.exc_info()))
    f = appmod.log_crash(text, appmod.page_shown(None))
    assert f == tmp_path / appmod.CRASH_LOG
    appmod.log_crash("second", "Inspect · Calibration")
    out = f.read_text(encoding="utf-8")
    heads = re.findall(r"^===== (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) · (.+)$", out, re.M)
    assert len(heads) == 2 and heads[0][1].endswith(f"{__version__} · Home") and heads[1][1].endswith("Calibration")
    assert "RuntimeError: the cause" in out and "Traceback" in out          # appended, the first one kept


def test_an_error_while_the_crash_box_is_open_is_written_without_another_box(tmp_path, monkeypatch):
    """The box is modal: an error repeated by a timer while it is open was shown in another box over it, and again."""
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication, QMessageBox
    QApplication.instance() or QApplication([])
    monkeypatch.setenv("ASTRO_GPU_SETTINGS", str(tmp_path / "settings.json"))
    prefs.load()
    boxes = []

    def exec_(box):
        boxes.append(box)
        if len(boxes) < 5:                                   # an error raised while the box is open
            try:
                raise ValueError(f"again {len(boxes)}")
            except ValueError:
                appmod.show_error(*sys.exc_info())
        return 0
    monkeypatch.setattr(QMessageBox, "exec", exec_)
    try:
        raise RuntimeError("first")
    except RuntimeError:
        appmod.show_error(*sys.exc_info())
    assert len(boxes) == 1
    out = (tmp_path / appmod.CRASH_LOG).read_text(encoding="utf-8")
    assert "RuntimeError: first" in out and "ValueError: again 1" in out
    try:
        raise RuntimeError("later")
    except RuntimeError:
        appmod.show_error(*sys.exc_info())
    assert len(boxes) == 2                                   # the box closed: the next error has its own


def test_a_log_whose_place_cannot_be_found_costs_only_the_log(monkeypatch):
    def lost():
        raise OSError("no folder")
    monkeypatch.setattr(appmod, "crash_log_file", lost)
    assert appmod.log_crash("text", "Home") is None
