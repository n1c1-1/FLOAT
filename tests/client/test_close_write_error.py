"""A project that cannot be written when it is closed (a full disk, a locked file) stays open: the failure is said in
the banner, its run is still followed, and the window does not close."""
from __future__ import annotations

from PySide6.QtWidgets import QApplication

from astro_gpu import api
from astro_gpu.client import i18n
from astro_gpu.client import app as appmod
from tests.client.test_leaving_asks_to_save import _ready, qapp, win  # noqa: F401  (fixtures used by name)


def test_a_write_error_at_closing_keeps_the_project_open_and_says_it(win, tmp_path, monkeypatch):
    root = tmp_path / "ws" / "Named"
    assert win.create_project(root.as_posix(), "Named", "IC 1805")
    _ready(win)
    win.s.set_target_name("M 42")
    win.answer["now"] = "save"

    def full(_p):
        raise OSError(28, "No space left on device")
    saving = api.save
    monkeypatch.setattr(api, "save", full)
    appmod.apply_language("fr")
    try:
        assert win.close_project() is False
        assert win.project is not None and win.s is not None and win.run is not None and win.unsaved()
        assert win.banner.isVisible() and "plein" in win.banner.text.text()
        assert not i18n.missing(), sorted(i18n.missing())
        assert win.close() is False and win.isVisible()                 # quitting: the window stays
    finally:
        appmod.apply_language("en")
    monkeypatch.setattr(api, "save", saving)
    assert win.close_project() is True and win.project is None          # written now, then closed
    QApplication.processEvents()
