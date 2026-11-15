"""Quitting while the card test is starting leaves no child process running: the stop came before the child was
launched (reading the card's name and driver takes a moment), found no process to kill, and the child then ran on
after the program had quit. Now a stop asked at any moment kills the child, or keeps it from starting."""
from __future__ import annotations

import threading

from PySide6.QtWidgets import QApplication

from astro_gpu.client.services import selftest as jobmod


class Child:
    """A test child that runs until it is killed (at most 3 s here)."""

    def __init__(self):
        self.killed = threading.Event()

    def poll(self):
        return 0 if self.killed.is_set() else None

    def kill(self):
        self.killed.set()


def test_a_stop_while_the_test_starts_kills_its_child(monkeypatch):
    QApplication.instance() or QApplication([])
    reading, go = threading.Event(), threading.Event()

    def slow_identity(dev):
        reading.set()
        go.wait(5)                                      # the card's name being read: no child yet
        return "RTX 4060 Ti", "610.88"
    monkeypatch.setattr(jobmod.st, "card_identity", slow_identity)
    monkeypatch.setattr(jobmod.st, "card_busy", lambda dev: False)
    children = []

    def runner(device, on_proc=None):
        c = Child()
        children.append(c)
        on_proc(c)
        c.killed.wait(3)                                # the child runs until killed
        return []
    t = jobmod.CardTest()
    t.runner = runner
    assert t.start("auto", "cuda:0", force=True)
    assert reading.wait(5)
    t.stop()                                            # quitting now: the child is not launched yet
    go.set()
    t._thread.join(10)
    assert not t._thread.is_alive()
    assert all(c.killed.is_set() for c in children)     # never left running (or never started)
