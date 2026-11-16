"""A slider of Post lands on the values of its step, and stays where it was let go.

Dehaze's Scale gave 249 or 257 px for a wanted 256, and the value moved by one notch at the end of a drag. The slider
counted in thousandths of its length whatever its step: an arrow key moved it by less than a step (ten presses for 1 %
of Opacity, one press in two for 1 px of Scale), and a click on the rail placed the handle with another measure of the
rail than the drag that follows, so the first pixel of that drag changed the value again."""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402
from PySide6.QtCore import QPoint, Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication, QStyle, QStyleOptionSlider  # noqa: E402

from astro_gpu.client import app as appmod  # noqa: E402
from astro_gpu.client.post3 import ContSlider  # noqa: E402
from astro_gpu.client.postengine import FILTERS  # noqa: E402
from astro_gpu.client.widgets import QSS  # noqa: E402

WIDTH = 386                                              # a slider of the settings panel at its usual width


@pytest.fixture(scope="module")
def qapp():
    a = QApplication.instance() or QApplication([])
    a.setStyleSheet(QSS + appmod.EXTRA_QSS)
    return a


def _param(filt, key):
    return next(q for q in FILTERS[filt]["params"] if q["key"] == key)


def _slider(q, value=None):
    """The slider Post makes for a filter's setting (``layer_props``), shown."""
    sl = ContSlider(q["label"], q["lo"], q["hi"], q["default"] if value is None else value, q["default"],
                    unit=q["unit"], step=q["step"])
    sl.resize(WIDTH, sl.sizeHint().height())
    sl.setVisible(True)
    QApplication.processEvents()
    sl.log = {"moved": [], "done": []}
    sl.moved.connect(sl.log["moved"].append)
    sl.done.connect(sl.log["done"].append)
    return sl


def _handle(s):
    opt = QStyleOptionSlider()
    s.initStyleOption(opt)
    return s.style().subControlRect(QStyle.CC_Slider, opt, QStyle.SC_SliderHandle, s)


def _on_step(sl, v) -> bool:
    n = (v - sl.lo) / sl.step
    return abs(n - round(n)) < 1e-6


SCALE = _param("dehaze", "scale")                        # 32 to 512 px by 1: more values than the rail has pixels
SETTINGS = [SCALE, _param("dehaze", "amount"), _param("sharpen", "radius"), _param("levels", "black"),
            dict(label="Opacity", lo=0, hi=100, default=100, unit="%", step=1)]


@pytest.mark.parametrize("q", SETTINGS, ids=lambda q: f"{q['label']} {q['lo']}-{q['hi']}")
def test_a_drag_shows_values_of_the_step_and_the_release_leaves_the_last_one(qapp, q):
    sl = _slider(q, q["lo"])
    s, y = sl.s, sl.s.height() // 2
    at = _handle(s).center()
    QTest.mousePress(s, Qt.LeftButton, Qt.NoModifier, at)
    seen = set()
    for x in range(at.x(), s.width() - 4):
        QTest.mouseMove(s, QPoint(x, y))
        QApplication.processEvents()
        assert _on_step(sl, sl.value()), (x, sl.value())
        seen.add(sl.value())
    QTest.mouseMove(s, QPoint(s.width() * 2 // 3, y))
    QApplication.processEvents()
    last = sl.value()
    assert sl.log["moved"][-1] == last
    QTest.mouseRelease(s, Qt.LeftButton, Qt.NoModifier, QPoint(s.width() * 2 // 3, y))
    QApplication.processEvents()
    assert sl.log["done"] == [last] and sl.value() == last           # let go: the value of the last move
    assert float(sl.val.text().split()[0].replace("%", "")) == last  # shown as it is, to its step's last decimal
    steps = round((q["hi"] - q["lo"]) / q["step"])
    assert q["hi"] in seen and len(seen) >= min(steps, 100)          # the whole range, to its end


@pytest.mark.parametrize("q", SETTINGS, ids=lambda q: f"{q['label']} {q['lo']}-{q['hi']}")
def test_a_click_on_the_rail_then_a_move_of_no_pixel_keeps_the_value(qapp, q):
    """The handle comes under the pointer, and the drag that follows starts from there: back at the pixel of the click,
    the value is the one of the click."""
    for x in range(14, WIDTH - 14, 7):
        sl = _slider(q, q["lo"] if x > WIDTH // 2 else q["hi"])
        s, y = sl.s, sl.s.height() // 2
        QTest.mousePress(s, Qt.LeftButton, Qt.NoModifier, QPoint(x, y))
        QApplication.processEvents()
        clicked = sl.value()
        assert _handle(s).contains(QPoint(x, y)), x                  # the handle is where it was clicked
        # centred on the click, to the nearest step (a slider of few steps: a step is several pixels of its rail)
        assert abs(_handle(s).center().x() - x) <= 2 + WIDTH / (2 * s.maximum()), (x, _handle(s))
        QTest.mouseMove(s, QPoint(x + 15, y))
        QApplication.processEvents()
        assert sl.value() != clicked or x + 15 > WIDTH - 12          # it follows the pointer (not at the rail's end)
        QTest.mouseMove(s, QPoint(x, y))
        QApplication.processEvents()
        assert sl.value() == clicked, (x, clicked, sl.value())
        QTest.mouseRelease(s, Qt.LeftButton, Qt.NoModifier, QPoint(x, y))
        QApplication.processEvents()
        assert sl.log["done"] == [clicked] and sl.value() == clicked and _on_step(sl, clicked)
        sl.setVisible(False)
        sl.deleteLater()


@pytest.mark.parametrize("q", SETTINGS, ids=lambda q: f"{q['label']} {q['lo']}-{q['hi']}")
def test_an_arrow_key_moves_by_one_step(qapp, q):
    start = q["lo"] + 20 * q["step"]
    sl = _slider(q, start)
    assert sl.value() == pytest.approx(start)
    for i in range(1, 8):
        QTest.keyClick(sl.s, Qt.Key_Right)
        assert sl.value() == pytest.approx(start + i * q["step"]), i
        assert sl.log["done"][-1] == sl.value()                      # each key: a value kept, a step of Undo
    for i in range(6, -4, -1):
        QTest.keyClick(sl.s, Qt.Key_Left)
        assert sl.value() == pytest.approx(start + i * q["step"]), i


def test_scale_is_set_to_256_exactly(qapp):
    """A drag brings Scale near 256 px, the arrow keys onto it; every value of its range is kept as it is given."""
    sl = _slider(SCALE)
    s, y = sl.s, sl.s.height() // 2
    QTest.mousePress(s, Qt.LeftButton, Qt.NoModifier, _handle(s).center())
    near = None
    for x in range(_handle(s).center().x(), s.width()):
        QTest.mouseMove(s, QPoint(x, y))
        QApplication.processEvents()
        if sl.value() >= 250:
            near = sl.value()
            QTest.mouseRelease(s, Qt.LeftButton, Qt.NoModifier, QPoint(x, y))
            break
    assert near is not None and 250 <= near <= 256
    for _i in range(int(256 - sl.value())):
        QTest.keyClick(s, Qt.Key_Right)
    assert sl.value() == 256 and sl.val.text() == "256 px" and sl.log["done"][-1] == 256
    for v in range(SCALE["lo"], SCALE["hi"] + 1):                    # a value given (a panel built again) is shown as it is
        sl.set_value(v)
        assert sl.value() == v


def test_a_slider_without_a_step_glides_as_before(qapp):
    sl = ContSlider("Sky", 0.0, 4.0, 2.0, 2.0)
    assert sl.value() == 2.0 and sl.s.maximum() == 1000
    sl.set_value(3.4)
    assert sl.value() == pytest.approx(3.4)
