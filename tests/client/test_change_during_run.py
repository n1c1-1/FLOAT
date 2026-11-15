"""A change of the project while a run goes (a session added, a setting): the run goes on with the frames and settings
it started with, and the window says so once, with Stop (the testers' pass 2: a run went on 23 minutes on inputs made
stale meanwhile, without a word)."""
from __future__ import annotations

from PySide6.QtWidgets import QPushButton

from astro_gpu.core.project2 import Project2
from tests.client.test_one_undo import _set_rbf, _settle, qapp, win  # noqa: F401  (the fixtures are used by name)


def test_a_change_during_a_run_is_said_once_with_stop(win, monkeypatch):
    rc = win.run
    stopped = []
    monkeypatch.setattr(rc, "stop", lambda: stopped.append(True))
    rc._snapshot = Project2.from_dict(win.s.p.to_dict(), win.s.p.root)      # the definition the run started with
    rc.phase, rc._told_edit = "running", False
    try:
        rc._edited_while_running()                                           # nothing changed: nothing said
        assert not win.banner.isVisible()
        _set_rbf(win)                                                        # a setting changed meanwhile
        _settle(win, 2)
        assert win.banner.isVisible() and "goes on with the frames and settings it started with" in \
            win.banner.text.text()
        stop = next(b for b in win.banner.findChildren(QPushButton) if b.text() == "Stop")
        stop.click()
        assert stopped == [True] and not win.banner.isVisible()
        rc._edited_while_running()                                           # said once a run
        assert not win.banner.isVisible()
    finally:
        rc.phase = "idle"
