"""A run whose events can no longer be read (its drive unplugged, its folder refused) is not followed for ever: after
the patience the follower has for a run gone silent, it says so once, in plain words, and the run is no longer shown
as running. One failed read alone is asked again."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QApplication

from astro_gpu.client import i18n
from astro_gpu.client.services.run import INTERRUPTED, NOT_STARTED, RunFollower, RunTrack, render_line


class Worker(QObject):
    done = Signal(str, int, object)
    failed = Signal(str, int, str)

    def __init__(self):
        super().__init__()
        self.asked = 0

    def submit(self, channel, fn, *args, **kwargs):
        self.asked += 1
        return self.asked

    def cancel(self, channel):
        pass


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def follower(started: bool = True):
    t = RunTrack([{"key": "ka", "stage": "a", "block": "stacking", "wf": [1]}], "r1")
    if started:
        t.apply([{"type": "run_start", "t": 1.0, "run": "r1", "pid": 1},
                 {"type": "stage_start", "t": 2.0, "run": "r1", "stage": "a", "key": "ka"}])
    w = Worker()
    f = RunFollower(None, w, t, SimpleNamespace(offset=0, run_id="r1", pid=None, log_file=""))
    got = SimpleNamespace(batches=[], ended=0)
    f.batch.connect(got.batches.append)
    f.ended.connect(lambda: setattr(got, "ended", got.ended + 1))
    return f, w, t, got


@pytest.mark.parametrize("started", [True, False])
def test_reads_that_keep_failing_end_the_following_and_say_so_once(qapp, started):
    f, w, t, got = follower(started)
    f.tick()
    w.failed.emit(f.channel, 1, "OSError: [Errno 5] Input/output error")
    assert not t.finished and got.ended == 0                    # one failed read: asked again
    f.tick()
    assert w.asked == 2
    w.failed.emit(f.channel, 2, "OSError: [Errno 5] Input/output error")
    assert t.finished and t.status == "interrupted" and t.lost and t.unread and got.ended == 1
    assert not t.never_started and t.error.startswith("OSError")
    assert [j.state for j in t.jobs] == [INTERRUPTED if started else NOT_STARTED]
    said = [x for b in got.batches for x in b.lines if x[1] == "run_unread"]
    assert len(said) == 1 and got.batches[-1].ended
    i18n.set_language("en")
    assert render_line(said[0]).endswith("Run no longer followed: its progress cannot be read")
    i18n.set_language("fr")
    try:
        assert render_line(said[0]).endswith("Exécution plus suivie : sa progression ne peut pas être lue")
    finally:
        i18n.set_language("en")
    f.tick()                                                    # over: nothing more is asked, nothing said again
    w.failed.emit(f.channel, 3, "OSError")
    assert w.asked == 2 and got.ended == 1


def test_a_read_that_works_again_forgets_the_failed_one(qapp):
    f, w, t, got = follower()
    for n in (1, 2, 3):                                         # failed, read, failed: never two in a row
        f.tick()
        if n == 2:
            w.done.emit(f.channel, n, ([], 0, True, True, ""))
        else:
            w.failed.emit(f.channel, n, "OSError")
    assert not t.finished and got.ended == 0
