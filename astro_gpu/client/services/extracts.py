"""The download of the star catalogue and of the bright stars, off the GUI thread (core/sky_catalogue.py and
core/bright_stars.py do the work).

One job for the whole program (``job()``): a plain thread runs the download and tells the GUI by signals, so that it
goes on when Settings is closed and a Settings opened later shows where it is. The ApiWorker is not used: it is one
thread shared by every short request of the client, and a download takes minutes to hours.

    j = job()
    j.progress.connect(slot, Qt.QueuedConnection)       # (done, total, text)
    j.finished.connect(slot, Qt.QueuedConnection)       # (kind, "" when done | "cancelled" | the error)
    j.start(sc.KIND)
"""
from __future__ import annotations

import threading

from PySide6.QtCore import QCoreApplication, QObject, Signal

from ...core import bright_stars as bs
from ...core import gaia_extracts as ge
from ...core import sky_catalogue as sc
from ..i18n import _, number


class ExtractsJob(QObject):
    progress = Signal(object, object, str)        # done, total: byte counts, past 2^31 for the big extracts (a C int would raise)
    finished = Signal(str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.kind: str | None = None                 # what is being downloaded (sc.KIND or bs.KIND); None when idle
        self.last = (0, 1, "")
        self._cancel = threading.Event()
        self._thread: threading.Thread | None = None
        app = QCoreApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self.cancel)

    def running(self) -> bool:
        return self.kind is not None

    def start(self, kind: str) -> bool:
        """Start the download of ``kind`` unless one is running (GUI thread)."""
        if self.running():
            return False
        ge.translate, ge.number = _, number          # the language of the moment
        self.kind = kind
        self.last = (0, 1, _("Asking for the star lists") if kind == bs.KIND else _("Checking the files"))
        self._cancel = threading.Event()
        self._thread = threading.Thread(target=self._run, args=(kind, self._cancel), daemon=True,
                                        name=f"gaia-extracts-{kind}")
        self._thread.start()
        return True

    def cancel(self):
        self._cancel.set()

    def _progress(self, done: int, total: int, text: str):
        self.last = (done, total, text)
        self.progress.emit(done, total, text)

    def _run(self, kind: str, cancel: threading.Event):
        outcome = ""
        try:
            if kind == bs.KIND:                      # the bright stars: lists asked of the services, indexes built
                bs.download(self._progress, cancel)
            else:
                sc.download(self._progress, cancel)
        except ge.Cancelled:
            outcome = "cancelled"
        except ge.DownloadError as e:
            outcome = str(e)
        except Exception as e:  # noqa: BLE001 - whatever happens goes to the screen as text
            outcome = f"{type(e).__name__}: {e}"
        self.kind = None
        self.finished.emit(kind, outcome)


_JOB: ExtractsJob | None = None


def job() -> ExtractsJob:
    """The program's download job (created in the GUI thread, on first use)."""
    global _JOB
    if _JOB is None:
        _JOB = ExtractsJob(QCoreApplication.instance())
    return _JOB
