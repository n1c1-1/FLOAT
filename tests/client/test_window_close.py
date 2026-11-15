"""The window's close box and Alt+F4 with changes not saved (the testers' pass 2 said "nothing happens"): a real close
event asks Save / Don't save / Cancel; Cancel keeps the window and its changes, Don't save closes it. (The tester's
screenshots draw a title bar over the dialogs: its close box and Alt+F4 now send that close event too.)"""
from __future__ import annotations

from astro_gpu.client import app as appmod
from tests.client.test_one_undo import _set_rbf, qapp, win  # noqa: F401  (the fixtures are used by name)


def test_closing_the_window_with_changes_not_saved_asks_first(win, monkeypatch):
    _set_rbf(win)
    assert win.unsaved()
    monkeypatch.setenv(appmod.SAVE_ANSWER_ENV, "cancel")
    assert win.close() is False and win.isVisible() and win.project is not None and win.unsaved()
    monkeypatch.setenv(appmod.SAVE_ANSWER_ENV, "discard")
    assert win.close() is True and not win.isVisible() and win.project is None


def test_home_never_shows_a_composition_not_saved(win, monkeypatch):
    """The testers' pass 2: Home's picture of a project showed a version the user chose not to save."""
    from types import SimpleNamespace as NS

    import numpy as np
    img = np.zeros((4, 4, 3), np.float32)
    view = NS(src=NS(img1=img, size=lambda: (8, 8)), crop=None)
    monkeypatch.setattr(win, "_post_holder", NS(page=NS(view=view)))
    written = []
    monkeypatch.setattr(win.worker, "submit", lambda ch, fn, *a: written.append(ch) if ch.startswith("thumb:") else None)
    _set_rbf(win)                                                     # a change not saved
    win._leave_thumbnail()                                            # Home shown, or closed without saving
    assert written == []
    win._leave_thumbnail(saving=True)                                 # closed with Save: the picture of what is saved
    assert len(written) == 1
