"""Inspect's final images are rendered on Post's render worker once Post lends it: a final image asked of the worker
used before, still under way when the other one is lent, is asked again of the new one, so that the next images do not
wait for an answer that will never come (the service stayed busy, the image never came)."""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from types import SimpleNamespace as NS  # noqa: E402

from PySide6.QtCore import QObject, Signal  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from astro_gpu.client.services.inspect import InspectService, VisualSpec  # noqa: E402


class Renderer(QObject):
    """A render worker that answers only when told."""
    rendered = Signal(object)
    failed = Signal(int, str, str)

    def __init__(self):
        super().__init__()
        self.asked = []

    def request(self, req):
        self.asked.append(req)
        return 100 + len(self.asked)                             # its number, as the render worker gives it

    def stop(self):
        pass


class Model:
    p = None
    gpu = False

    def view(self, wid):
        return NS(pid=wid, name="Full")

    def state_info(self, v, stage):
        return NS(state="done", key="k1", previous_key="")


def test_a_final_image_under_way_when_post_lends_its_worker_is_asked_again(monkeypatch):
    QApplication.instance() or QApplication([])
    import astro_gpu.client.postengine as pe
    monkeypatch.setattr(pe, "new_workflow_layer", lambda wid, name, on: {"wid": wid})
    sv = InspectService(Model())
    try:
        old, new = Renderer(), Renderer()
        sv.use_renderer(old)
        spec = VisualSpec(key="final", caption="Final", kind="render", stage="render_simple")
        got = []
        sv.imageReady.connect(lambda slot, token, im: got.append(token))
        t1 = sv.request_image(0, 1, spec)
        assert len(old.asked) == 1
        sv.use_renderer(new)                                     # Post shown: its worker lent while one renders
        assert len(new.asked) == 1                               # asked again where it can be answered
        new.rendered.emit(NS(kind="inspect", seq=101, rgb=__import__("numpy").zeros((4, 4, 3), "uint8"), canvas=None))
        assert got == [t1] and not sv.busy()
        sv.request_image(0, 1, spec)
        assert len(new.asked) == 2                               # the next one is not left waiting
    finally:
        sv.stop()


def test_a_late_answer_is_never_placed_as_the_one_asked(monkeypatch):
    """The answer of an earlier render (another page's, one let go) arrives while another is asked: it is dropped, and
    the slot gets its own image."""
    QApplication.instance() or QApplication([])
    import numpy as np

    import astro_gpu.client.postengine as pe
    monkeypatch.setattr(pe, "new_workflow_layer", lambda wid, name, on: {"wid": wid})
    sv = InspectService(Model())
    try:
        r = Renderer()
        sv.use_renderer(r)
        spec = VisualSpec(key="final", caption="Final", kind="render", stage="render_simple")
        got, failed = [], []
        sv.imageReady.connect(lambda slot, token, im: got.append((slot, token)))
        sv.imageFailed.connect(lambda slot, token, why: failed.append((slot, token)))
        t = sv.request_image(1, 1, spec)
        r.rendered.emit(NS(kind="inspect", seq=7, rgb=np.zeros((4, 4, 3), "uint8"), canvas=None))   # an old one
        r.failed.emit(7, "inspect", "an old failure")
        assert got == [] and failed == [] and sv.busy()
        r.rendered.emit(NS(kind="inspect", seq=101, rgb=np.zeros((4, 4, 3), "uint8"), canvas=None))
        assert got == [(1, t)] and not sv.busy()
    finally:
        sv.stop()


class FailingModel(Model):
    """A project whose files cannot be read."""
    p = None


def test_a_reading_that_fails_is_said(monkeypatch):
    """Frames, a group of frames, the measurements of a result, the frames kept and a frame's statistics: a reading that
    fails is answered (the page said "Reading the frames…" for ever)."""
    QApplication.instance() or QApplication([])
    import astro_gpu.client.services.inspect as si

    def broken(*_a, **_k):
        raise OSError("the disk is gone")

    for name in ("project_frames", "lights_of", "browse_group", "frame_stats"):
        monkeypatch.setattr(si, name, broken)
    monkeypatch.setattr(si.api, "stage_outputs", broken)
    monkeypatch.setattr(si.api, "selection_preview", broken)
    sv = InspectService(FailingModel())
    try:
        said = []
        sv.readFailed.connect(lambda what, req, why: said.append((what, req, why)))
        sv.request_frames(1)
        tg = sv.request_group(1, "darks")
        sv.request_outputs(1, "stack", "current", "k1")
        tp = sv.request_preview(1, 0.5, 2.0)
        sv.request_stats("a.fits")
        assert sv.wait(20)
        got = {what: (req, why) for what, req, why in said}
        assert got == {"frames": (None, "OSError: the disk is gone"), "group": (tg, "OSError: the disk is gone"),
                       "outputs": ((1, "stack", "current"), "OSError: the disk is gone"),
                       "preview": (tp, "OSError: the disk is gone"), "stats": ("a.fits", "OSError: the disk is gone")}
    finally:
        sv.stop()


def test_the_pages_say_a_reading_that_failed():
    """Where the pages waited for a reading: the frames of a kind (Sources' count and image), the health check, a
    block's measurements."""
    QApplication.instance() or QApplication([])
    from PySide6.QtWidgets import QLabel

    from astro_gpu.client import sources
    from astro_gpu.client.steps import Steps
    cls = sources.SourcesView
    canvas = NS(message="", update=lambda: None)
    me = NS(_reading=True, items=[], sid=1, group="darks", held=lambda k: 3, count=QLabel(), _group_token=4,
            viewer=NS(canvas=canvas), excluded=lambda i: False, _read_why={}, data=None)
    me.update_count = lambda: cls.update_count(me)
    cls._read_failed(me, "group", 3, "an older request")                  # not the one asked: nothing said
    assert me._reading and me._read_why == {}
    cls._read_failed(me, "group", 4, "the disk is gone")
    assert canvas.message == "The frames cannot be read: the disk is gone" == me.count.text()
    me.pages = NS(currentIndex=lambda: 0)
    cls._read_failed(me, "frames", None, "the disk is gone")
    assert cls.checks(me) == [(False, "The frames cannot be read: the disk is gone")]

    class Service(QObject):
        outputsReady = Signal(int, str, str)
        readFailed = Signal(str, object, str)

        def outputs_of(self, *a):
            return None

        def request_outputs(self, *a):
            pass

    class Model:
        sel_step = "alignement"

        def state_info(self, p, stage):
            return NS(state="done", key="k1", previous_key="")

        def all_stages(self, p):
            return ["register"]

    sv = Service()
    st = Steps(Model(), sv)
    p = NS(pid=1)
    assert st.facts(p, "alignement") == []                                # being read
    sv.readFailed.emit("outputs", (1, "register", "current"), "the disk is gone")
    assert st.facts(p, "alignement") == [("Measurements", "cannot be read: the disk is gone")]
