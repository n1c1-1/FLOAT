"""Get survey: the H-alpha survey of a workflow's field fetched off the GUI thread (``api.fetch_survey`` does the work).

An action of the user, outside any run (a run never downloads: core/online.py). One job for the whole program, as the
extracts' download (services/extracts.py): a plain thread, its progress and its end told by signals, so that the
Gradient block's panel, built again at every change, shows where it is.

    j = job()
    j.progress.connect(slot, Qt.QueuedConnection)       # (fraction, text)
    j.finished.connect(slot, Qt.QueuedConnection)       # (project, workflow id, "" when done | the error)
    j.start(project, pid)

A fetch is the workflow's of one project (``where(project)``): the same workflow number in another project is another.
"""
from __future__ import annotations

import threading
from pathlib import Path

from PySide6.QtCore import QObject, Signal

from ... import api


def where(project) -> str:
    """The project a fetch is for, by its folder (a definition without one: by itself)."""
    root = getattr(project, "root", None)
    return Path(root).as_posix().lower() if root else f"id:{id(project)}"


class SurveyJob(QObject):
    progress = Signal(float, str)
    finished = Signal(str, int, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.pid: int | None = None                  # the workflow whose survey is being fetched; None when idle
        self.root: str | None = None                 # its project (where())
        self.last = (0.0, "")
        self.title = ""                              # the survey fetched last

    def running(self, pid: int | None = None, project=None) -> bool:
        """A fetch under way: any (no argument), or the one of this workflow of this project."""
        if self.pid is None:
            return False
        return pid is None or (pid == self.pid and project is not None and where(project) == self.root)

    def start(self, project, pid: int) -> bool:
        """Fetch the survey of the workflow's field unless a fetch runs (GUI thread)."""
        if self.running():
            return False
        self.pid, self.root, self.last = pid, where(project), (0.0, "")
        threading.Thread(target=self._run, args=(project, pid), daemon=True, name=f"survey-{pid}").start()
        return True

    def _progress(self, f: float, text: str):
        self.last = (float(f), text)
        self.progress.emit(float(f), text)

    def _run(self, project, pid: int):
        why = ""
        try:
            self.title = api.fetch_survey(project, pid, self._progress)
        except Exception as e:  # noqa: BLE001 - said in the panel, in plain words
            from ...core import online
            why = e.why if isinstance(e, online.ServiceDown) else str(e)[:200]
        root, self.pid = self.root, None
        self.finished.emit(root, pid, why)


_JOB: list[SurveyJob] = []


def job() -> SurveyJob:
    if not _JOB:
        _JOB.append(SurveyJob())
    return _JOB[0]
