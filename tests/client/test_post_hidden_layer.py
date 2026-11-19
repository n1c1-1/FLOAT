"""Post with two Stars and background layers (issue #171): the top one hides the lower one, so a slider of the top
one's last filter makes nothing of the lower one."""
from __future__ import annotations

import os
import time

import numpy as np

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
pytest.importorskip("PySide6")

from test_post import WF, DEV, Folder, ToyModel, stars_prep  # noqa: E402
from test_post_speed import STACK, Work, drag, filters, slider_of  # noqa: E402

from astro_gpu.client import postengine as pe  # noqa: E402
from astro_gpu.client.post3 import PostPage3  # noqa: E402
from astro_gpu.client.services.render import RenderWorker  # noqa: E402

pytestmark = [pytest.mark.level3, pytest.mark.alone]
H, W = int(os.environ.get("HID_H", 2000)), int(os.environ.get("HID_W", 3000))


@pytest.fixture(scope="module")
def two(tmp_path_factory, qapp):
    t = tmp_path_factory.mktemp("hidden")
    folders = {}
    for pid, seed in ((1, 3), (2, 5)):
        out, st = stars_prep(t / f"wf{pid}", H, W, seed=seed)
        folders[pid], folders[("stars", pid)] = Folder(out), Folder(st)
    model = ToyModel([WF(1, "Full", "simple", True, stars=True), WF(2, "Copy", "simple", True, stars=True)])
    worker = RenderWorker(None, resolver=lambda pid: folders.get(pid), device=DEV)
    page = PostPage3(model, worker=worker)
    page.resize(1500, 900)
    page.show()
    v = page.view

    class S:
        pass
    s = S()
    s.qapp, s.worker, s.view, s.page = qapp, worker, v, page

    def settle(timeout=300):
        end = time.monotonic() + timeout
        idle = 0
        while time.monotonic() < end:
            worker.settle(5)
            qapp.processEvents()
            idle = idle + 1 if not worker.busy() and not v._final_t.isActive() and not v._draft_t.isActive() \
                and not v._region_t.isActive() else 0
            if idle == 2:
                return True
        return False
    s.settle = settle
    assert settle()
    for lay in v.layers:
        if lay["kind"] == "workflow":
            lay["visible"] = True
    for top in [x for x in v.layers if pe.is_pixel(x)][::-1]:          # the lower layer first, as the user's copy
        for key, params in (STACK[:1] + STACK[3:6] + STACK[-4:] if top is not None else ()):
            v.select(top["id"], "layer")
            fl = v.add_filter(key)
            fl["params"].update(params)
    v.request(final_only=True, force=True)
    assert settle()
    yield s
    page.shutdown()
    page.deleteLater()


def test_a_drag_makes_the_top_layer_only_and_shows_no_mosaic(two, monkeypatch):
    """Issue #171: under the top Stars and background layer (Normal, 100 %, no mask) nothing of the lower one is made
    by a drag or its release, whether the lower is shown, hidden by its eye or deleted: its images and its filters are
    not made, and the drag's pictures are shown smoothly (no blocks of the draft's pixels)."""
    s, v = two, two.view
    top, low = [x for x in v.layers if pe.is_pixel(x)][:2]
    made = []
    real_rgb, real_apply = pe.Composer.layer_rgb, pe.Composer._apply

    def layer_rgb(self, lay, *a, **k):
        made.append(("image", lay.get("pid")))
        return real_rgb(self, lay, *a, **k)

    def apply(self, fl, *a, **k):
        made.append(("filter", fl.get("target")))
        return real_apply(self, fl, *a, **k)
    monkeypatch.setattr(pe.Composer, "layer_rgb", layer_rgb)
    monkeypatch.setattr(pe.Composer, "_apply", apply)
    times = {}
    for label in ("shown", "hidden", "deleted"):
        if label == "hidden":
            low["visible"] = False
            v.request(final_only=True, force=True)
        elif label == "deleted":
            v.layers.remove(low)
            v.request(final_only=True, force=True)
        assert s.settle()
        last = next(x for x in v.layers if x["kind"] == "filter" and x.get("target") == top["id"])
        sl = slider_of(v, last, "Strength")
        assert s.settle()
        made.clear()
        t, rel = drag(s, sl, steps=6, release=True)
        assert s.settle()
        assert ("image", low["pid"]) not in made, (label, made)
        assert {x[1] for x in made if x[0] == "filter"} <= {top["id"]}, (label, made)
        times[label] = (min(t), rel)
    print("drag draft / release (s):", times)


def test_the_draft_of_a_drag_is_enlarged_smoothly():
    from types import SimpleNamespace

    from astro_gpu.client.post3 import PostSource
    src = PostSource(SimpleNamespace(show_mask=False, mask1=None))
    src.canvas, src.level1, src.level2 = (64, 64), 1, 3
    src.img1 = np.zeros((32, 32, 3), np.uint8)
    src.img2 = np.random.default_rng(0).integers(0, 255, (8, 8, 3)).astype(np.uint8)
    src.seq1, src.seq2 = 1, 2
    a = src._half()
    assert a.shape == (32, 32, 3)
    assert (a[:, 1:] == a[:, :-1]).mean() < 0.1, "the draft is shown as blocks"
    assert src._half() is a                                                   # enlarged once, not at each paint
    src.img2 = np.full((8, 8, 3), 77, np.uint8)
    assert (src._half() == 77).all()


def test_a_drag_zoomed_in_is_drawn_at_the_screen_s_size_not_in_blocks(two):
    """The drag's picture of a view zoomed in is made for the part shown, at a level whose pixels are a few screen
    pixels (four at most), not the half picture enlarged (blocks as large as the zoom: 20 to 30 px)."""
    s, v = two, two.view
    top = next(x for x in v.layers if pe.is_pixel(x))
    last = next(x for x in v.layers if x["kind"] == "filter" and x.get("target") == top["id"])
    sl = slider_of(v, last, "Strength")
    v.viewer.zoom_to(3.0)
    assert s.settle()
    got = []
    v.rendered.connect(got.append)
    drag(s, sl, steps=5, release=True)
    v.rendered.disconnect(got.append)
    drafts = [r for r in got if r.kind == "draft"]
    assert drafts and all(r.box is not None for r in drafts), "a zoomed drag made the whole canvas"
    for r in drafts:
        block = 3.0 * 2.0 ** r.level
        assert block <= 4.5, f"a draft pixel is {block:.1f} screen pixels"
        assert (r.box[2] - r.box[0]) * (r.box[3] - r.box[1]) < 1.5e6
    v.viewer.zoom_to(0.2)
    assert s.settle()


def test_zoomed_in_the_status_says_refining_until_the_full_resolution_is_there(two):
    """Zoom 100 % shows the enlarged half picture until the part shown comes at full resolution (15 s on a processor):
    the status line says Refining... for all that time, and no longer after."""
    s, v = two, two.view
    said = []
    v.refining.connect(said.append)
    v.viewer.zoom_to(1.0)
    v._view_changed()
    assert v.is_refining(), "zoomed in, the part shown is awaited and nothing says so"
    assert s.settle()
    assert not v.is_refining() and said and said[0] is True and said[-1] is False, said
    v.refining.disconnect(said.append)
    v.viewer.zoom_to(0.2)
    assert s.settle()
