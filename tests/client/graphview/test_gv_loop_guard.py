"""The GUI thread never waits for the worker: on a computed project whose worker answers late, the settings panel's
controls (the method cards first: a usability session froze the window for minutes on a click on Denoise's
"Pretrained + adapted to the image") never keep the event loop away for more than ``LIMIT`` seconds.

``LoopWatch`` is the proof: a timer of the GUI thread beats; a watchdog thread that sees no beat for ``LIMIT`` seconds
takes the stack of every thread at that moment (what the GUI thread was waiting for), and ``faulthandler`` writes them
all and ends the run if the GUI thread never comes back (a deadlock that holds the interpreter's lock)."""
from __future__ import annotations

import faulthandler
import random
import sys
import threading
import time
import traceback

import pytest
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QAbstractSlider,
    QAbstractSpinBox,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QMessageBox,
)

from astro_gpu.client import model as client_model
from astro_gpu.client import prefs
from astro_gpu.client.panel import MethodCard, StackMethodCard
from tests.client.graphview.legacy import World
from tests.client.pagehelp import qapp  # noqa: F401  (the fixture, by name)

#: seconds the event loop may be kept away, at most
LIMIT = 2.0
#: seconds after which a GUI thread that never came back is a deadlock: every stack written, the run ended
HANG = 120.0


class LoopWatch:
    """Watches the GUI thread: ``worst`` (s), the longest time it was kept from its event loop, and ``stacks``, the
    threads' stacks taken while it was kept longer than ``limit``."""

    def __init__(self, limit: float = LIMIT, hang: float = HANG):
        self.limit, self.hang = limit, hang
        self.last = time.monotonic()
        self.worst = 0.0
        self.stacks: list[str] = []
        self._gui = threading.get_ident()
        self._end = threading.Event()
        self.timer = QTimer()
        self.timer.timeout.connect(self.beat)
        self.timer.start(20)
        self._thread = threading.Thread(target=self._watch, daemon=True, name="loop-watch")
        self._thread.start()
        faulthandler.dump_traceback_later(hang, exit=True)

    def beat(self):
        now = time.monotonic()
        self.worst = max(self.worst, now - self.last)
        self.last = now

    def _watch(self):
        taken = 0.0
        while not self._end.wait(0.1):
            gap = time.monotonic() - self.last
            if gap > self.limit and taken != self.last:
                taken = self.last
                frames = sys._current_frames()
                text = [f"--- the GUI thread kept from its event loop {gap:.1f} s; every thread:"]
                for ident, frame in frames.items():
                    name = next((t.name for t in threading.enumerate() if t.ident == ident), "?")
                    mark = " (GUI)" if ident == self._gui else ""
                    text.append(f"Thread {name}{mark}:\n" + "".join(traceback.format_stack(frame)))
                self.stacks.append("\n".join(text))

    def pump(self, seconds: float):
        """The event loop turns for ``seconds`` (the GUI thread's own work in it counts)."""
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            QApplication.processEvents()
            self.beat()
            time.sleep(0.005)

    def close(self):
        faulthandler.cancel_dump_traceback_later()
        self._end.set()
        self.timer.stop()
        self._thread.join(2)

    def check(self):
        assert self.worst <= self.limit and not self.stacks, \
            f"the event loop was kept {self.worst:.1f} s\n" + "\n".join(self.stacks[:2])


class SlowWorker:
    """The worker's jobs made slow (``delay`` s each, the states of the edits ``states_delay``): it is always behind."""

    def __init__(self, m, delay=0.15, states_delay=0.8):
        submit = m.worker.submit

        def slow(channel, fn, *a, **k):
            d = states_delay if channel.startswith("model.states:") else delay

            def late(*aa, **kk):
                time.sleep(d)
                return fn(*aa, **kk)
            return submit(channel, late, *a, **k)
        m.worker.submit = slow


def _computed(m):
    """The project as if run once: a stage is done while its key is the one it had (each time the model takes states;
    the recipe of test_graph_states_later)."""
    from astro_gpu import api
    keys0 = {}
    install = m._install

    def done(got):
        install(got)
        for w in m.p.workflows:
            for st in m.__dict__["_ch"].get(w.id, []):
                try:
                    k = api.stage_key(m.p, w.id, st)
                except (ValueError, KeyError):
                    continue
                if keys0.setdefault((w.id, st), k) == k:
                    m.__dict__["_st"][(w.id, st)] = api.StageState("done", "", k)
        install((m.__dict__["_st"], m.__dict__["_ch"], m.__dict__["_bl"]))
    m._install = done
    m._sync()


@pytest.fixture(autouse=True)
def _own_settings(tmp_path, monkeypatch):
    """A settings file of the test's own, English texts, the question before a change that loses results on."""
    from astro_gpu.client import i18n
    monkeypatch.setenv("ASTRO_GPU_SETTINGS", str(tmp_path / "settings.json"))
    prefs.load()
    i18n.set_language("en")
    prefs.set("confirm_changes", True)
    yield
    i18n.set_language("en")


@pytest.fixture()
def no_modal(monkeypatch):
    """A control that would open a dialog gets it answered at once (a modal dialog is not what is measured here)."""
    monkeypatch.setattr(QDialog, "exec", lambda self, *a, **k: 0)
    monkeypatch.setattr(QMessageBox, "exec", lambda self, *a, **k: QMessageBox.Cancel)
    for name in ("getOpenFileName", "getSaveFileName"):
        monkeypatch.setattr(QFileDialog, name, staticmethod(lambda *a, **k: ("", "")))
    monkeypatch.setattr(QFileDialog, "getOpenFileNames", staticmethod(lambda *a, **k: ([], "")))
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", staticmethod(lambda *a, **k: ""))


def _world(tmp_path, kind):
    if kind == "colour":
        return World(tmp_path / "w")
    w = World(tmp_path / "w", workflows=(), filters=("SII", "Ha", "OIII"), nights=0)
    w.template("sho_rgb")
    return w


def _answer(W, watch, rnd):
    """A draft at the panel's foot: answered (Apply, Reset or Try variant)."""
    if W.m.pending_change:
        W.page.answer_draft(rnd.choice(("apply", "reset", "reset", "variant")))
        watch.beat()
        watch.pump(0.05)


def _controls(panel):
    kinds = (MethodCard, StackMethodCard, QCheckBox, QComboBox, QAbstractSpinBox, QAbstractSlider)
    return [c for k in kinds for c in panel.findChildren(k) if c.isVisibleTo(panel) and c.isEnabled()]


def _touch(c, rnd):
    """One gesture on a control, as a hand would make it."""
    if isinstance(c, (MethodCard, StackMethodCard)):
        c.mousePressEvent(None)
    elif isinstance(c, QCheckBox):
        c.click()
    elif isinstance(c, QComboBox):
        if c.count() > 1:
            c.setCurrentIndex((c.currentIndex() + 1) % c.count())
            c.activated.emit(c.currentIndex())
    elif isinstance(c, QAbstractSpinBox):
        c.stepUp() if rnd.random() < 0.5 else c.stepDown()
        c.editingFinished.emit()
    elif isinstance(c, QAbstractSlider):
        c.setValue(c.value() + max(1, (c.maximum() - c.minimum()) // 7))
        c.sliderReleased.emit()


def _session(W, watch, rnd, per_block=8):
    """Each block chosen in turn; its panel's controls touched (each found again: the panel is made again after a
    change), the questions answered, Undo and Redo at random; the worker always behind."""
    blocks = list(W.graph.blocks)
    touched = 0
    for b in blocks:
        if b not in W.graph.blocks:
            continue
        W.b.choose(b)
        watch.beat()
        watch.pump(0.15)
        for i in range(per_block):
            cs = _controls(W.page.panel)
            if not cs:
                break
            c = cs[i % len(cs)] if i < len(cs) else rnd.choice(cs)
            try:
                _touch(c, rnd)
            except RuntimeError:                      # the control went with a panel made again meanwhile
                pass
            touched += 1
            watch.beat()
            watch.pump(rnd.choice((0.0, 0.03, 0.2)))
            _answer(W, watch, rnd)
            if rnd.random() < 0.2:
                W.b.undo()
                watch.beat()
                watch.pump(0.03)
            if rnd.random() < 0.1:
                W.b.redo()
                watch.beat()
        _answer(W, watch, rnd)
    return touched


def test_clicking_the_method_cards_of_a_computed_denoise_while_the_worker_is_behind_never_blocks(
        qapp, tmp_path, monkeypatch, no_modal):
    """The usability session's freeze, replayed: Denoise of a computed project chosen, its method cards clicked again
    and again, with Undo between them, while the worker's answers come late. The question of the change (results
    computed and up to date would be computed again) is asked, and the window never waits."""
    monkeypatch.setattr(client_model, "ASYNC_STATES", True)
    W = _world(tmp_path, "colour")
    watch = None
    try:
        m = W.m
        dn = W.kinds("denoise")[0]
        for method in ("adapted", "pretrained"):        # the method on the block itself, as a project saved by the graph
            m._do([("set_method", {"block": dn, "method": method})])
        W.settle()
        assert any(n.get("method") == "pretrained" for w in m.p.workflows for n in (w.blocks or [])
                   if isinstance(n, dict))
        _computed(m)
        W.settle()
        W.choose(dn)
        assert m.state(m.path, "denoise") == "done"
        SlowWorker(m)
        watch = LoopWatch()
        asked = 0
        for i in range(8):
            end = time.monotonic() + 10                  # a change made: the panel is made again with its states
            while not (cards := [c for c in W.page.panel.findChildren(MethodCard)
                                 if c.isVisibleTo(W.page.panel) and c.ok]) and time.monotonic() < end:
                watch.pump(0.05)
            assert cards, "Denoise shows its method cards"
            for c in cards:
                try:
                    c.mousePressEvent(None)
                except RuntimeError:                     # the click made a draft: the panel was built again under it
                    break
                watch.beat()
                watch.pump(0.02 * (i % 3))
            if m.pending_change:
                asked += 1
                assert W.page.panel.draft_change(m.pending_change)
                W.page.answer_draft("apply" if i % 2 else "reset")
            watch.beat()
            if i % 3 == 2:
                W.b.undo()
                watch.beat()
            watch.pump(0.05)
        watch.pump(1.5)
        watch.check()
        assert asked, "a method change of a computed block asks first"
    finally:
        if watch is not None:
            watch.close()
        W.close()


@pytest.mark.parametrize("kind", ["colour", "filters"])
def test_every_panel_control_of_every_block_leaves_the_event_loop_free_with_a_slow_worker(
        qapp, tmp_path, monkeypatch, no_modal, kind):
    """A scripted session on a computed project (colour, and filters): each block chosen, the controls of its panel
    touched, the questions answered, Undo and Redo between them, the worker always behind: the event loop is never
    kept away more than ``LIMIT`` seconds."""
    monkeypatch.setattr(client_model, "ASYNC_STATES", True)
    W = _world(tmp_path, kind)
    watch = None
    try:
        _computed(W.m)
        W.settle()
        SlowWorker(W.m)
        watch = LoopWatch()
        touched = _session(W, watch, random.Random(7))
        watch.pump(1.5)
        watch.check()
        assert touched > 20, touched
    finally:
        if watch is not None:
            watch.close()
        W.close()
