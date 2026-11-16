"""One chronological order for the window's Undo: every history of the window (the workflows' steps, Post's own steps,
the Gradient view's draft) stamps its steps with one clock, so that Ctrl+Z takes back the latest change wherever it was
made, and Ctrl+Shift+Z the latest one undone, while nothing was changed since.

The workflows' steps and the draft's carry their stamp (``stamp``); a history whose code is not ours to stamp (Post's
page) is read by its lists: the steps past the ones already seen are new (``seen``), read each time its lists change.
"""
from __future__ import annotations

_clock = [0]


def tick() -> int:
    """The next stamp: later than every stamp given before."""
    _clock[0] += 1
    return _clock[0]


def now() -> int:
    """The latest stamp given: a step stamped later was pushed after this call."""
    return _clock[0]


def stamp(step) -> None:
    """A step pushed to a list of Undo or Redo now: its stamp (an object that cannot hold one is left as it is)."""
    try:
        step.seq = tick()
    except AttributeError:
        pass


def seq_of(step) -> int:
    return int(getattr(step, "seq", 0) or 0)


class Lists:
    """The stamps of lists that cannot hold one per step (Post's page): per list, one stamp per step, read again each
    time the list may have changed (``seen``). A list grown has new steps (stamped now), a list shrunk lost its last."""

    def __init__(self):
        self._of: dict[int, tuple[list, list[int]]] = {}

    def seen(self, steps: list) -> list[int]:
        ent = self._of.get(id(steps))
        if ent is None or ent[0] is not steps:
            ent = (steps, [])
            self._of[id(steps)] = ent
        seqs = ent[1]
        del seqs[len(steps):]
        while len(seqs) < len(steps):
            seqs.append(tick())
        return seqs

    def top(self, steps: list) -> int:
        """The stamp of a list's last step, 0 when it has none."""
        seqs = self.seen(steps)
        return seqs[-1] if seqs else 0
