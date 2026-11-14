"""Shared fixtures of the new graph page's tests: Qt offscreen, the client's style, English texts, a bridge on a graph
held in memory (nothing is written anywhere)."""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402


@pytest.fixture(scope="session")
def qapp():
    from astro_gpu.client.widgets import QSS
    app = QApplication.instance() or QApplication([])
    app.setStyleSheet(QSS)
    return app


@pytest.fixture(autouse=True)
def _english_and_private_settings(tmp_path, monkeypatch):
    """English texts (the source language) and a settings file of the test's own."""
    from astro_gpu.client import i18n, prefs
    monkeypatch.setenv("ASTRO_GPU_SETTINGS", str(tmp_path / "settings.json"))
    prefs.load()
    i18n.set_language("en")
    yield
    i18n.set_language("en")


@pytest.fixture()
def bridge(qapp):
    from astro_gpu.client.graphview.bridge import Bridge
    from astro_gpu.client.graphview.document import MemoryDocument
    return Bridge(MemoryDocument())
