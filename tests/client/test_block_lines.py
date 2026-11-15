"""The line under a block's name says the block's CURRENT stored settings, after every change (the night test pass: a
Stacking set to Classic still read "drizzle auto"); a variant's first block says what differs from its original (one
made with RBF read "like the mother — change a setting")."""
from __future__ import annotations

import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from astro_gpu.client import app as appmod  # noqa: E402
from astro_gpu.client import prefs  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    from astro_gpu.client.widgets import QSS
    a = QApplication.instance() or QApplication([])
    a.setStyleSheet(QSS + appmod.EXTRA_QSS)
    return a


@pytest.fixture()
def page(qapp, tmp_path, monkeypatch):
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
    yield w.graph
    w.close_project(discard=True)
    w.close()
    w.deleteLater()
    QApplication.processEvents()


def _settle(w):
    for _i in range(10):
        w.worker.wait(30)
        QApplication.processEvents()
        time.sleep(0.01)


def _line(page, b):
    page.canvas.update_states()
    return page.canvas.blocks[b].d["summary"]


def _of(page, kind, row=None):
    g = page.bridge.graph
    if row is not None:
        return next(b for b in row.blocks if g.blocks[b].kind == kind)
    return next(k for k, x in g.blocks.items() if x.kind == kind)


def test_the_stacking_line_follows_its_stored_settings(page):
    s = page.s
    b = _of(page, "stacking")
    assert _line(page, b).startswith("drizzle")
    s.set_param(s.paths[0], "stack", "method", "classic")
    s.set_param(s.paths[0], "stack", "kappa", 4.0)
    _settle(page.window())
    line = _line(page, b)
    assert line.startswith("classic") and "drizzle" not in line, line
    assert "4 σ" in line or "no Selection" in line, line


@pytest.mark.parametrize("change, said", [({"method": "rbf"}, "RBF"), ({"param": ["gradient", "degree", 3]}, "3")])
def test_a_variant_says_what_differs_from_its_original(page, change, said):
    b = _of(page, "gradient")
    res = page.bridge.act("try_variant", block=b, change=change, name="Variant")
    assert res.ok, page.bridge.say(res.reason)
    _settle(page.window())
    g = page.bridge.graph
    row = next(r for r in g.rows if r.fork)
    first = row.blocks[0]
    line = _line(page, first)
    assert "like the" not in line and said in line, line
