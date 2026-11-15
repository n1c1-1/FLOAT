""""+ N min to run" after a change: never while the states of an edit are being read (the time is not known yet), and
taken back when a later change (an undo) brings the time down: the testers' pass 2 read "+ 4 min to run" over blocks all
up to date, Run greyed."""
from __future__ import annotations

from types import SimpleNamespace as NS

from astro_gpu.client.graphview.host import SidesHost


class _Model:
    ready = True

    def __init__(self):
        self.sec, self.pending, self.paths = 0.0, False, [1]

    def total_seconds(self):
        return self.sec

    def cost(self):
        return NS(seconds=self.sec, measured=True)

    def has_data(self):
        return True

    def states_pending(self):
        return self.pending

    def run_problems(self):
        return []


class _Toast:
    def __init__(self):
        self.plain, self.shown = "", False

    def isVisible(self):  # noqa: N802 - Qt's name
        return self.shown

    def hide(self):
        self.shown = False


class _Host:
    def __init__(self):
        self.s, self._last, self.said_texts = _Model(), None, []
        self.canvas = NS(toast=_Toast())

    def toast(self, text):
        self.said_texts.append(text)
        self.canvas.toast.plain, self.canvas.toast.shown = text, True


def test_the_time_to_run_is_never_said_while_the_states_are_read_and_goes_with_an_undo():
    h = _Host()
    SidesHost.cost_toast(h)                                  # the first reading
    h.s.sec, h.s.pending = 240.0, True                       # an edit, its states not read yet
    SidesHost.cost_toast(h)
    assert h.said_texts == []
    h.s.pending = False                                      # read: the time it adds is said
    SidesHost.cost_toast(h)
    assert h.said_texts == ["+ 4 min to run"] and h.canvas.toast.shown
    h.s.sec = 0.0                                            # an undo: all up to date again
    SidesHost.cost_toast(h)
    assert not h.canvas.toast.shown and h.said_texts == ["+ 4 min to run"]

