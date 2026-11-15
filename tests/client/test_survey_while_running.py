"""Use survey moves results to the trash, and its Undo / Redo move them back and forth: while a run is in progress
(it reads those results) the button and that Undo / Redo are greyed, the reason on hover, and do nothing."""
from __future__ import annotations

from PySide6.QtWidgets import QApplication, QPushButton

from astro_gpu import api
from astro_gpu.client import i18n
from astro_gpu.client import app as appmod
from tests.client.test_one_undo import qapp, win  # noqa: F401  (the fixtures are used by name)


def _running(w, on: bool):
    """The window's run said in progress (or ended), as the run controller says it."""
    w.run.__dict__["_phase"] = "running" if on else "idle"
    w.run._emit_state(on)
    QApplication.processEvents()


def _moves(monkeypatch):
    moved = []
    monkeypatch.setattr(api, "use_survey", lambda p, wf: moved.append("use") or
                        [{"stage": "gradient", "key": "k", "trash": "t"}])
    monkeypatch.setattr(api, "put_back_survey", lambda p, m: moved.append("back") or (["gradient"], []))
    monkeypatch.setattr(api, "trash_survey_again", lambda p, m: moved.append("again") or (["gradient"], []))
    monkeypatch.setattr(api, "survey_stale", lambda p, wf: ["gradient"])
    return moved


def test_use_survey_and_its_undo_are_greyed_while_a_run_is_in_progress(win, monkeypatch):
    moved = _moves(monkeypatch)
    from astro_gpu.client.panel import Panel
    pn = Panel(win.s)
    pn.setParent(win)
    p = win.s.path
    pn._use_survey_row(p)
    use = next(b for b in pn.findChildren(QPushButton) if b.text() == "Use survey")
    assert use.isEnabled()
    _running(win, True)
    assert not use.isEnabled() and "run is in progress" in use.toolTip()
    use.click()
    assert moved == []
    _running(win, False)
    assert use.isEnabled()
    use.click()
    assert moved == ["use"]
    _running(win, True)
    win._update_actions()
    assert not win.a_undo.isEnabled() and "run is in progress" in win.a_undo.toolTip()
    win.undo()                                                     # a notice's link, Ctrl+Z routed by a page
    assert moved == ["use"]
    _running(win, False)
    win._update_actions()
    assert win.a_undo.isEnabled()
    win.undo()
    assert moved == ["use", "back"]
    _running(win, True)
    win._update_actions()
    assert not win.a_redo.isEnabled() and "run is in progress" in win.a_redo.toolTip()
    win.redo()
    assert moved == ["use", "back"]
    _running(win, False)
    win.redo()
    assert moved == ["use", "back", "again"]


def test_the_reasons_have_their_french(win, monkeypatch):
    _moves(monkeypatch)
    from astro_gpu.client.panel import Panel
    win.s.use_survey(win.s.path.pid)
    appmod.apply_language("fr")
    try:
        pn = Panel(win.s)
        pn.setParent(win)
        _running(win, True)
        pn._use_survey_row(win.s.path)
        win._update_actions()
        win.undo()
        assert not i18n.missing(), sorted(i18n.missing())
    finally:
        _running(win, False)
        appmod.apply_language("en")
