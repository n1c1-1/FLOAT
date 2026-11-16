"""Quitting while the worker still runs a long request (Home's scan of a slow disk) ends the program cleanly. Its thread
was a QThread: on the way out the Qt bindings destroy every Qt object without a parent that Python holds, the running
QThread with them, and Qt aborts the program then ("QThread: Destroyed while thread is still running"); keeping a
reference to it changed nothing. The worker's thread is now a Python thread, which no Qt object's end can stop."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

SCRIPT = r"""
import gc, sys, threading, time
from PySide6.QtCore import QCoreApplication, QObject
from astro_gpu.client.services.worker import ApiWorker
app = QCoreApplication([])
owner = QObject()
w = ApiWorker(owner)
gate = threading.Event()
w.submit("home", gate.wait, float(sys.argv[1]))     # a request still running at the end
time.sleep(0.2)
w.stop(100)                                         # the wait at quitting ends while the request runs
owner.deleteLater()
del w, owner
gc.collect()
app.processEvents()
print("closed", flush=True)
"""


@pytest.mark.parametrize("seconds", [3.0, 0.3])     # still running when the program ends, or ending during its end
def test_a_worker_busy_at_quitting_ends_the_program_cleanly(seconds):
    env = {**os.environ, "QT_QPA_PLATFORM": "offscreen", "CUDA_VISIBLE_DEVICES": "-1", "PYTHONPATH": str(ROOT),
           "QT_FORCE_STDERR_LOGGING": "1"}
    env.pop("PYTEST_CURRENT_TEST", None)
    r = subprocess.run([sys.executable, "-c", SCRIPT, str(seconds)], cwd=ROOT, env=env, capture_output=True, text=True,
                       timeout=60)
    assert r.returncode == 0 and "closed" in r.stdout, (r.returncode, r.stdout, r.stderr[-800:])
    assert "Destroyed while thread" not in r.stderr, r.stderr[-800:]


def test_a_worker_stopped_while_busy_is_kept_until_its_request_ends_and_says_what_it_does():
    """As the render thread and the earlier-version dialog do: a worker whose request still runs when the wait of
    ``stop`` ends is kept in a list until its thread ends, and ``busy()`` says what is true: the requests dropped by
    the stop no longer count, the one running counts until it ends."""
    import threading
    import time

    from PySide6.QtWidgets import QApplication

    from astro_gpu.client.services import worker as wk
    app = QApplication.instance() or QApplication([])
    w = wk.ApiWorker()
    gate, started = threading.Event(), threading.Event()
    w.submit("slow", lambda: (started.set(), gate.wait(20)))
    assert started.wait(5)
    w.submit("next", lambda: 1)                           # waiting behind it: the stop drops it
    try:
        w.stop(50)                                        # the wait ends while the request runs
        assert w in wk._RETIRED and w.busy()              # kept, and still at work
        w.submit("late", lambda: 2)                       # asked after the stop: never run, never counted
    finally:
        gate.set()
    end = time.monotonic() + 5
    while w.busy() and time.monotonic() < end:
        app.processEvents()
        time.sleep(0.005)
    assert not w.busy() and w._thread.wait(5000)
    wk.ApiWorker().stop()                                 # any later stop lets go of the ones that ended
    assert w not in wk._RETIRED
    quiet = wk.ApiWorker()                                # nothing running: not kept, not busy
    quiet.submit("a", lambda: 3)
    assert quiet.wait(5)
    quiet.stop()
    assert quiet not in wk._RETIRED and not quiet.busy()
