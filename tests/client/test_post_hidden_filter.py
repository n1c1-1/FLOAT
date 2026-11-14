"""A change to a row that is not shown (a hidden filter's slider, a filter under a hidden layer) renders nothing and shows
no draft; its settings are kept, undone, and used when the eye is turned on."""
from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QApplication

from test_post_versions import _done, _page_of, by_name, model, qapp, two  # noqa: F401
from astro_gpu.client.postengine import snapshot, visible_signature


class _Slider(QObject):
    moved = Signal(float)
    done = Signal(float)


def _settle(v):
    """What the timers do: the last final render asked."""
    if v._final_t.isActive():
        v._final_t.stop()
        v._send_final()
    v._draft_t.stop()


def _setup(model):
    model.add_post_layer(by_name(model, "Full").pid, "stars")
    page = _page_of(model)
    v = page.view
    assert v.layers, "the project has a layer"
    top = next(x for x in v.layers if x["kind"] != "filter")
    v.sel = (top["id"], "layer")
    fl = v.add_filter("curves")
    fl["sliders"] = {}
    _settle(v)
    return page, v, top, fl


def _drag(v, fl, value):
    sl = _Slider()
    v._wire(sl, lambda x: fl["sliders"].__setitem__("shadows", x))
    sent = []
    v.worker.request = lambda r: sent.append(r.kind) or 1
    v._send_draft = lambda: sent.append("draft")
    sl.moved.emit(value)
    if v._draft_t.isActive():
        v._draft_t.stop()
        v._send_draft()
    sl.done.emit(value)
    _settle(v)
    return sent


def test_a_hidden_filters_slider_requests_no_render_and_keeps_the_value(model):
    page, v, top, fl = _setup(model)
    try:
        v.set_visible(fl["id"], False)
        _settle(v)
        shown = v.src.img1
        sent = _drag(v, fl, 3.0)
        assert "draft" not in sent and not {"half", "refine", "draft"} & set(sent)
        assert v.src.img1 is shown
        assert fl["sliders"]["shadows"] == 3.0
        assert v.undo, "the change is a step of Undo"
        v.do_undo()
        assert "shadows" not in v.layer(fl["id"])["sliders"]
        v.redo.clear()
    finally:
        _done(page)


def test_turning_the_eye_on_uses_the_saved_value_and_renders(model):
    page, v, top, fl = _setup(model)
    try:
        v.set_visible(fl["id"], False)
        _settle(v)
        _drag(v, fl, 3.0)
        before = visible_signature(v.layers)
        v.set_visible(fl["id"], True)
        assert visible_signature(v.layers) != before
        assert v._final_t.isActive()
        assert v.layer(fl["id"])["sliders"]["shadows"] == 3.0
    finally:
        _done(page)


def test_a_visible_filters_slider_still_renders(model):
    page, v, top, fl = _setup(model)
    try:
        sent = _drag(v, fl, 3.0)
        assert "draft" in sent and "half" in sent or "refine" in sent
    finally:
        _done(page)


def test_a_filter_under_a_hidden_layer_is_not_shown():
    top = {"id": 1, "kind": "workflow", "visible": False, "name": "a"}
    f = {"id": 2, "kind": "filter", "target": 1, "sub": None, "visible": True, "name": "f", "sliders": {"x": 1}}
    a = visible_signature([f, top])
    f2 = dict(f, sliders={"x": 2})
    assert visible_signature([f2, dict(top)]) == a
    top2 = dict(top, visible=True)
    assert visible_signature([f2, top2]) != visible_signature([f, top2])
    assert visible_signature([f2, dict(top)], want=[2]) != a      # its selected mask is on the screen
