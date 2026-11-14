"""Inspect > Gradient on a laptop window: the tools that act on the image sit on a row above it, the method cards take one
line each and the stretch column is folded; a large window keeps the full cards and the column."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
from conftest import needs_test_project  # noqa: E402
from PySide6.QtWidgets import QApplication, QPushButton  # noqa: E402
from test_inspect import F_PROJECT, open_window, settle  # noqa: E402

from astro_gpu.client import app as appmod  # noqa: E402
from astro_gpu.client import prefs  # noqa: E402
from astro_gpu.client.panel import MethodCard  # noqa: E402
from astro_gpu.client.widgets import QSS  # noqa: E402

pytestmark = needs_test_project


@pytest.fixture(scope="module")
def qapp():
    a = QApplication.instance() or QApplication([])
    a.setStyleSheet(QSS + appmod.EXTRA_QSS)
    return a


def _gradient(qapp, tmp_path, size):
    os.environ["ASTRO_GPU_SETTINGS"] = str(tmp_path / "settings.json")
    prefs.load()
    appmod.apply_language("en")
    w = open_window(Path(F_PROJECT), size=size)
    settle(w)
    w.s.select(index=0, step="fond")
    w.set_mode(1)
    settle(w)
    w.tune.dock.set_state("docked")
    settle(w)
    return w


def _button(view, text):
    return next(b for b in view.tool_row.findChildren(QPushButton) if b.text() == text)


def test_a_laptop_window_has_the_sample_tools_on_the_image_and_one_line_cards(qapp, tmp_path):
    w = _gradient(qapp, tmp_path, (1250, 685))
    try:
        t = w.tune
        v = t.view
        assert v is not None and v.tool_row.isVisibleTo(v)
        win = w.rect()
        for text in ("Samples", "Brush", "Ellipse"):
            b = _button(v, text)
            assert b.isVisibleTo(w)
            r = b.mapTo(w, b.rect().topLeft())
            assert 0 <= r.x() and r.x() + b.width() <= win.width() and 0 <= r.y() and r.y() + b.height() <= win.height()
        assert not v.panel_box.isVisibleTo(v)                     # the stretch column is folded (H opens it)
        assert v.hist_b.isVisibleTo(v) and not v.hist_b.isChecked()
        cards = t.panel.findChildren(MethodCard)
        assert len(cards) >= 3
        assert all(c.sizeHint().height() <= 36 for c in cards), [c.sizeHint().height() for c in cards]
        # clicking Samples is the tool on, as the panel's button was
        _button(v, "Samples").click()
        assert t.grad_editor.mode == "points"
        assert not any(b.text() == "Samples" for b in t.panel.findChildren(QPushButton))
        settle(w)
    finally:
        w.close()
        appmod.apply_language("en")


def test_a_large_window_keeps_the_full_cards_and_the_stretch_column(qapp, tmp_path):
    w = _gradient(qapp, tmp_path, (1920, 1080))
    try:
        t = w.tune
        assert t.view.panel_box.isVisibleTo(t.view)
        assert all(c.sizeHint().height() > 50 for c in t.panel.findChildren(MethodCard))
        assert t.view.tool_row.isVisibleTo(t.view)
    finally:
        w.close()
        appmod.apply_language("en")


def test_the_new_texts_are_in_french(qapp):
    appmod.apply_language("fr")
    try:
        from astro_gpu.client.i18n import _
        assert _("The brush adds an excluded area (X)") != "The brush adds an excluded area (X)"
        assert _("The brush takes the excluded area away (X)") != "The brush takes the excluded area away (X)"
    finally:
        appmod.apply_language("en")
