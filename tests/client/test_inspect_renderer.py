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
        new.rendered.emit(NS(kind="inspect", rgb=__import__("numpy").zeros((4, 4, 3), "uint8"), canvas=None))
        assert got == [t1] and not sv.busy()
        sv.request_image(0, 1, spec)
        assert len(new.asked) == 2                               # the next one is not left waiting
    finally:
        sv.stop()
