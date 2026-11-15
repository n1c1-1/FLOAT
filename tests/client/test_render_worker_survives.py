"""The render thread of Post and Inspect survives a failure while it hands memory back on leaving Post: the next
request is still computed."""
from __future__ import annotations

import os
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication  # noqa: E402

from astro_gpu.client.services.render import RenderWorker  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def test_a_failed_release_leaves_the_thread_running(qapp):
    w = RenderWorker(None, resolver=lambda sid: None, device="cpu")
    try:
        calls = []

        def broken():
            calls.append(1)
            if len(calls) == 1:
                raise OSError("the images could not be handed over")

        w._take_images = broken
        w.release()
        end = time.monotonic() + 20
        while not calls and time.monotonic() < end:
            qapp.processEvents()
            time.sleep(0.01)
        assert calls, "the release did not run"
        time.sleep(0.2)
        assert w.isRunning(), "the render thread stopped"
        w.refresh()
        with w._cond:                                            # a later request is still taken
            w._release = True
            w._cond.notify()
        end = time.monotonic() + 20
        while len(calls) < 2 and time.monotonic() < end:
            time.sleep(0.01)
        assert len(calls) == 2
    finally:
        w.stop()
