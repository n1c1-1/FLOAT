"""Post's reaction time, the gate (level 3: run it alone, ``pytest tests/client/test_post_speed.py``): synthetic
projects of the size the owner works on (a 24 Mpx Combine layers layer with twelve filters, heavy ones among them; a
combination's master layer whose filters' masks read the picture), CPU only.

Two kinds of checks:
- the work a gesture does, counted (each filter applied, with the pixels it was applied to): a light drag makes the
  filter moved and those above it only, at the drag's size; nothing is made at full resolution before the screen's
  picture of a release;
- the time it takes, the best of five: a light gesture's picture in 0.15 s, a heavy filter's own slider in 0.4 s, the
  screen's picture of a release in 1 s. Against a calibration kernel: on a machine more than three times slower than
  the one the thresholds were set on, the timed checks are skipped, never passed."""
from __future__ import annotations

import copy
import os
import time

import cv2
import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
pytest.importorskip("PySide6")

from test_post import Setup, stars_prep  # noqa: E402

from astro_gpu.client import postengine as pe  # noqa: E402
from astro_gpu.client.postengine import Composer  # noqa: E402

pytestmark = pytest.mark.level3

H, W = 4000, 6000                                   # 24 Mpx
#: the calibration kernel's time on the machine the thresholds were set on (CPU, s): a bigger time, a slower machine
CALIBRATION_S = 0.10
#: bottom first: the owner's kind of stack, heavy filters among light ones; the top one is light
STACK = (("dehaze", {"amount": 0.25, "scale": 256.0}),
         ("ml_denoise", {"l1": 2.5, "l2": 1.5, "l3": 0.8, "amount": 1.0}),
         ("denoise", {"luminance": 0.1, "color": 0.6}),
         ("scnr", {"strength": 0.4}),
         ("balance", {"temperature": 0.6, "tint": -0.3}),
         ("mixer", {"sat_red": 0.2, "hue_blue": 0.1}),
         ("local_contrast", {"strength": 0.4, "scale": 60.0}),
         ("ml_sharpen", {"l3": 0.3, "l4": 0.4, "l5": 0.3}),
         ("highlights", {"strength": 0.3}),
         ("clahe", {"amount": 0.3}),
         ("levels", {"gamma": 1.1}),
         ("saturation", {"strength": 30.0, "protect": 1.0}))


def calibration() -> float:
    """The best of three of a fixed piece of work (blurs and arithmetic on a 6 Mpx image)."""
    a = np.random.default_rng(0).random((2000, 3000, 3), dtype=np.float32)
    best = 1e9
    for _ in range(3):
        t = time.perf_counter()
        b = cv2.GaussianBlur(a, (0, 0), 6)
        c = np.clip(a * 1.3 - b * 0.3, 0, 1)
        cv2.GaussianBlur(c, (0, 0), 2)
        best = min(best, time.perf_counter() - t)
    return best


@pytest.fixture(scope="module")
def slow_factor():
    return calibration() / CALIBRATION_S


def timed(slow_factor, seconds: float, best: float, what: str):
    if slow_factor > 3:
        pytest.skip(f"this machine is {slow_factor:.1f} times slower than the thresholds' one: {what} not timed")
    assert best <= seconds, f"{what}: {best:.3f} s, over {seconds} s"


# ---------------------------------------------------------------------------------------------- the 24 Mpx stack
@pytest.fixture(scope="module")
def big(tmp_path_factory, qapp):
    out, st = stars_prep(tmp_path_factory.mktemp("speed") / "big", H, W)
    s = Setup({"full": out, "full_stars": st}, qapp, which=(1,), stars=True)
    assert s.settle(600)
    v = s.view
    lay = next(x for x in v.layers if pe.is_pixel(x))
    for key, params in STACK:                           # each new filter goes on top of the layer's own filters
        v.select(lay["id"], "layer")
        fl = v.add_filter(key)
        fl["params"].update(params)
    v.request(final_only=True, force=True)
    assert s.settle(900)
    yield s
    s.close()


class Work:
    """Every filter applied, by the composer: (its id, its pixels, the composition's level, a draft)."""

    def __init__(self, monkeypatch):
        self.log = []
        real = Composer._apply

        def apply(comp, fl, under, level, abox, canvas, ctx, *a, **k):
            self.log.append((fl["id"], under.shape[0] * under.shape[1], level, bool(ctx.draft), time.perf_counter()))
            return real(comp, fl, under, level, abox, canvas, ctx, *a, **k)
        monkeypatch.setattr(Composer, "_apply", apply)


def filters(view):
    """The layer's own filters, bottom first."""
    return list(reversed([x for x in view.layers if x["kind"] == "filter"]))


def slider_of(view, fl, name):
    """The slider ``name`` of the filter ``fl``'s panel, the filter chosen (its panel built and shown)."""
    from PySide6.QtCore import QEvent
    from PySide6.QtWidgets import QApplication

    from astro_gpu.client.post3 import ContSlider
    view.select(fl["id"], "layer")
    end = time.monotonic() + 10
    while time.monotonic() < end:
        QApplication.processEvents()
        QApplication.sendPostedEvents(None, QEvent.DeferredDelete)      # (the former panel's sliders gone)
        got = [s for s in view.findChildren(ContSlider) if s.name.text() == name and s.isVisible()]
        if got:
            return got[-1]
        time.sleep(0.02)
    raise AssertionError(f"no slider {name!r} shown for {fl.get('filter')}: "
                         f"{[s.name.text() for s in view.findChildren(ContSlider) if s.isVisible()]}")


def pump_until(s, cond, timeout=120):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        s.qapp.processEvents()
        if cond():
            return True
        time.sleep(0.002)
    return False


def drag(s, sl, steps=8, every=0.06, work=None, release=False):
    """A drag of a slider as a hand makes it: pressed (its first draft, kept apart, is waited for), then moved every
    ``every`` s ``steps`` times. Returns (the time from each draft's request to its picture shown, what was shown).
    ``work``: its log is emptied once the handle is taken (what the moves make is counted); ``release``: let go right
    after the last move, and the time to the screen's picture said in place of what was shown."""
    v = s.view
    shown, asked = [], {}

    def note(res):
        shown.append((time.perf_counter(), res.kind, res.seq))
    v.rendered.connect(note)
    real = s.worker.request

    def request(r):
        seq = real(r)
        if r.kind == "draft":
            asked[seq] = time.perf_counter()
        return seq
    s.worker.request = request
    try:
        sl.s.setSliderDown(True)
        assert pump_until(s, lambda: not s.worker.busy(), 300)          # (the press's draft, not shown)
        if work is not None:
            work.log.clear()
        p0 = sl.s.value()
        way = -3 if p0 > sl.s.maximum() / 2 else 3                    # (away from the nearer end: every move moves)
        for k in range(steps):
            sl.s.setValue(p0 + way * (k + 1))
            end = time.monotonic() + every
            while time.monotonic() < end:
                s.qapp.processEvents()
                time.sleep(0.002)
        released = None
        # the handle still held: the drafts asked are shown (a hand waits for the picture before letting go)
        assert pump_until(s, lambda: not s.worker.busy() and not v._draft_t.isActive(), 300)
        if release:                                     # let go after the last draft, as a hand does
            t = time.perf_counter()
            sl.s.setSliderDown(False)

            def current():                              # the screen's picture is the settings' (the full one may come)
                return not v._final_t.isActive() and not v._draft_t.isActive() and v._done_seq >= v._main_seq
            s.qapp.processEvents()
            assert pump_until(s, current), (v._main_seq, v._done_seq, v._final_kind)
            released = time.perf_counter() - t
            s.screen_at = time.perf_counter()
        assert pump_until(s, lambda: not s.worker.busy() and not v._draft_t.isActive(), 300)
    finally:
        s.worker.request = real
        v.rendered.disconnect(note)
    times = [t - asked[q] for t, kind, q in shown if kind == "draft" and q in asked]
    assert times, f"no draft was shown: {sl.name.text()} {[(k, q) for _t, k, q in shown]} asked {sorted(asked)}"
    return times, released


def test_a_light_drag_makes_the_filter_moved_only_at_the_drag_s_size(big, monkeypatch):
    s, v = big, big.view
    top = filters(v)[-1]
    sl = slider_of(v, top, "Strength")
    assert s.settle(600)
    work = Work(monkeypatch)
    drag(s, sl, steps=6, work=work)
    drafts = [w for w in work.log if w[3]]
    assert drafts, "no draft was made"
    assert {w[0] for w in drafts} == {top["id"]}, f"a light drag made other filters: {sorted({w[0] for w in drafts})}"
    biggest = max(w[1] for w in drafts)
    assert biggest <= (W >> v.draft_level) * (H >> v.draft_level) * 1.05


def test_a_release_makes_nothing_at_full_resolution_before_the_screen_s_picture(big, monkeypatch):
    s, v = big, big.view
    sl = slider_of(v, filters(v)[-1], "Strength")
    assert s.settle(600)
    work = Work(monkeypatch)
    drag(s, sl, steps=2, release=True)
    early = [w for w in work.log if w[2] == 0 and w[4] < s.screen_at]
    assert not early, "a filter was made at full resolution before the screen's picture"
    s.settle(900)


def test_the_times_of_a_light_gesture_a_heavy_slider_and_a_release(big, slow_factor):
    s, v = big, big.view
    fl = filters(v)
    light, heavy = fl[-1], next(x for x in fl if x["filter"] == "ml_denoise")
    best_light, best_heavy, best_release = 1e9, 1e9, 1e9
    for _ in range(5):
        sl = slider_of(v, light, "Strength")
        assert s.settle(600)
        times, released = drag(s, sl, release=True)
        best_light = min(best_light, min(times))
        best_release = min(best_release, released)
        assert s.settle(900)
        sl = slider_of(v, heavy, "Amount")
        assert s.settle(600)
        times, _shown = drag(s, sl)
        best_heavy = min(best_heavy, min(times))
        sl.s.setSliderDown(False)
        assert s.settle(900)
    print()
    print(f"gate: light {best_light:.3f} s, heavy {best_heavy:.3f} s, release {best_release:.3f} s, "
          f"machine x{slow_factor:.2f}")
    timed(slow_factor, 0.15, best_light, "a light gesture's picture")
    timed(slow_factor, 0.40, best_heavy, "a heavy filter's own slider")
    timed(slow_factor, 1.0, best_release, "the screen's picture after a release")


# ---------------------------------------------------------------------------------------------- a combination
def test_a_combination_s_light_gesture_makes_its_top_filter_only(monkeypatch, slow_factor):
    """A combination's master layer, its filters' masks reading the picture, a stack on top: a light gesture (a draft
    of the top filter) makes the top filter only, and in time."""
    from test_post_fusion import Filters, _lum_part, master
    hh, ww = 2000, 3000
    rng = np.random.default_rng(1)
    planes = {r: (cv2.GaussianBlur(rng.random((hh, ww)).astype(np.float32), (0, 0), 30) +
                  rng.normal(0, .004, (hh, ww))).astype(np.float32) for r in ("SII", "Ha", "OIII")}
    prov = Filters(planes)
    prov.canvas = lambda pids: pe.Canvas(ww, hh)
    g = master(side_masks={"f:SII": _lum_part(**{"from": "picture"}), "f:OIII": _lum_part(**{"from": "picture"})})
    tops = []
    for key, params in (("local_contrast", {"strength": 0.4}), ("levels", {"gamma": 1.1}),
                        ("saturation", {"strength": 30.0})):
        f = pe.new_filter_layer(key, g["id"])
        f["params"].update(params)
        tops.append(f)
    layers = pe.normalize(list(reversed(tops)) + [g])
    cp = Composer(prov)
    cp.compose(copy.deepcopy(layers), 3, draft=True, track=True)
    work = Work(monkeypatch)
    best = 1e9
    for k in range(5):
        changed = copy.deepcopy(layers)
        next(x for x in changed if x.get("filter") == "saturation")["params"]["strength"] = 31.0 + k
        t = time.perf_counter()
        cp.compose(changed, 3, draft=True, track=True)
        best = min(best, time.perf_counter() - t)
    assert {w[0] for w in work.log} == {tops[-1]["id"]}, "a light gesture made other filters"
    timed(slow_factor, 0.15, best, "a combination's light gesture")


def test_the_times_of_a_layer_setting_under_the_stack_at_the_window_s_size(big, slow_factor):
    """The owner's case: a setting of the layer itself (Sky) under its twelve filters, the Post view as large as his
    window (2560 x 1440): every filter is made again, at the drag's size while it moves and at the screen's after the
    release; the drafts in the light gesture's time, the screen's picture in a second."""
    s, v = big, big.view
    s.page.resize(2560, 1440)
    assert s.settle(900)
    lay = next(x for x in v.layers if pe.is_pixel(x))
    best_draft, best_release = 1e9, 1e9
    for _ in range(3):
        sl = slider_of(v, lay, "Sky")
        assert s.settle(900)
        times, released = drag(s, sl, steps=6, release=True)
        best_draft = min(best_draft, min(times))
        best_release = min(best_release, released)
        assert s.settle(900)
    print()
    print(f"gate, a layer's setting: draft {best_draft:.3f} s, release {best_release:.3f} s, machine x{slow_factor:.2f}")
    timed(slow_factor, 0.15, best_draft, "a layer setting's draft under the stack")
    timed(slow_factor, 1.0, best_release, "the screen's picture of a layer setting under the stack")
