"""Quitting while Post's render thread still opens an engine ends the program cleanly. Its thread was a QThread, kept in
a list when the wait at stopping ended first: on the way out the Qt bindings destroy every Qt object Python holds, the
running QThread with them, and Qt aborts the program ("QThread: Destroyed while thread is still running"), as the API
worker did (its own test). The render thread is now a Python thread, which no Qt object's end can stop."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

SCRIPT = r"""
import gc, sys, time
from PySide6.QtCore import QCoreApplication
from astro_gpu.client.services import render
app = QCoreApplication([])


class Slow(render.RenderWorker):
    def _make(self):                                  # an engine that takes long to open
        time.sleep(float(sys.argv[1]))
        self.renderer = None


w = Slow(None, resolver=lambda pid: None, device="cpu")
time.sleep(0.2)
w.stop(100)                                           # the wait at quitting ends while the engine opens
del w
gc.collect()
app.processEvents()
print("closed", flush=True)
"""


@pytest.mark.parametrize("seconds", [3.0, 0.3])     # still opening when the program ends, or ending during its end
def test_post_render_busy_at_quitting_ends_the_program_cleanly(seconds):
    env = {**os.environ, "QT_QPA_PLATFORM": "offscreen", "CUDA_VISIBLE_DEVICES": "-1", "PYTHONPATH": str(ROOT),
           "QT_FORCE_STDERR_LOGGING": "1"}
    env.pop("PYTEST_CURRENT_TEST", None)
    r = subprocess.run([sys.executable, "-c", SCRIPT, str(seconds)], cwd=ROOT, env=env, capture_output=True, text=True,
                       timeout=60)
    assert r.returncode == 0 and "closed" in r.stdout, (r.returncode, r.stdout, r.stderr[-800:])
    assert "Destroyed while thread" not in r.stderr, r.stderr[-800:]
