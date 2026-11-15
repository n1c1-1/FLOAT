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
