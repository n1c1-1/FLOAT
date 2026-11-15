"""A chart without curves takes the mouse wheel without an error, and an export that cannot be written is said (it
raised out of the button)."""
from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
from PySide6.QtCore import QPoint, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QWheelEvent  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from astro_gpu.client import chartview as cv  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _wheel(w, pos=QPointF(100, 100)):
    ev = QWheelEvent(pos, w.mapToGlobal(pos), QPoint(0, 0), QPoint(0, 120), Qt.NoButton, Qt.NoModifier,
                     Qt.NoScrollPhase, False)
    w.wheelEvent(ev)


def test_the_wheel_on_an_empty_chart_does_nothing(qapp):
    v = cv.ChartView("Empty")
    v.resize(500, 300)
    v.show()
    QApplication.processEvents()
    assert not v.panels
    _wheel(v)                                                    # raised IndexError
    v.close()


def _chart():
    v = cv.ChartView("Test chart", export_name="t")
    p = v.add_panel("Level (u)", "u", digits=1)
    v.set_x("Time", "s", kind="num")
    v.add_line("a", np.arange(10.0), np.arange(10.0) ** 2, panel=p, color=cv.BLUE)
    v.fit()
    v.resize(500, 300)
    return v


def test_an_export_that_cannot_be_written_is_said(qapp, tmp_path, monkeypatch):
    said = []
    monkeypatch.setattr(cv.QToolTip, "showText", lambda *a, **k: said.append(a[1]))
    v = _chart()
    bad = tmp_path / "no such folder" / "x"
    assert v.export_csv(str(bad) + ".csv") is None
    assert v.export_png(str(bad) + ".png") is None
    assert len(said) == 2 and all(s.startswith("This file cannot be written") for s in said)
    ok = tmp_path / "x.png"
    assert v.export_png(str(ok)) == str(ok) and ok.stat().st_size > 0
    assert v.export_csv(str(tmp_path / "x.csv")) == str(tmp_path / "x.csv")
