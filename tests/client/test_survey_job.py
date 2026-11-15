"""Get survey's fetch is the workflow's of one project: a fetch under way for a workflow of another project (the same
workflow number) is not shown as this one's, its end is not taken for this one's, and this one's Get survey, greyed
while the program fetches the other, says why."""
from __future__ import annotations

import threading
import time
from types import SimpleNamespace as NS

from PySide6.QtWidgets import QApplication, QLabel, QPushButton

from astro_gpu import api
from astro_gpu.client.services import survey
from tests.client.test_one_undo import qapp, win  # noqa: F401  (the fixtures are used by name)


def test_a_fetch_of_another_project_is_not_this_one_s(win, tmp_path, monkeypatch):
    gate = threading.Event()
    monkeypatch.setattr(api, "fetch_survey", lambda p, wf, progress=None: gate.wait(20) and "Survey")
    monkeypatch.setattr(api, "survey_state", lambda p, wf: {"missing": True, "available": False})
    changed = []
    monkeypatch.setattr(win.s, "results_changed", lambda *a: changed.append(1))
    other = NS(root=(tmp_path / "other project").as_posix())
    j = survey.job()
    pid = win.s.path.pid
    assert j.start(other, pid)                                   # the same workflow number, in another project
    try:
        from astro_gpu.client.panel import Panel
        pn = Panel(win.s)
        pn.setParent(win)
        pn.survey_row(win.s.path)
        texts = [x.text() for x in pn.findChildren(QLabel)]
        assert "H-alpha survey: not on this computer" in texts, texts
        get = next(b for b in pn.findChildren(QPushButton) if b.text() == "Get survey")
        assert not get.isEnabled() and "being fetched" in get.toolTip()
        from astro_gpu.client import app as appmod
        from astro_gpu.client import i18n
        appmod.apply_language("fr")
        try:
            Panel(win.s, parent=win).survey_row(win.s.path)
            assert not i18n.missing(), sorted(i18n.missing())
        finally:
            appmod.apply_language("en")
    finally:
        gate.set()
    end = time.monotonic() + 20
    while j.running() and time.monotonic() < end:
        time.sleep(0.01)
    for _i in range(5):
        QApplication.processEvents()
    assert changed == []                                         # the other project's end: nothing to compute here
    assert get.isEnabled()
