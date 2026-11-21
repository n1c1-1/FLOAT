"""Post's reaction time, the gate (level 3: run it alone, ``pytest tests/client/test_post_speed.py``): synthetic
projects of the size the owner works on (a 24 Mpx Combine layers layer with twelve filters, heavy ones among them; a
combination's master layer whose filters' masks read the picture), CPU only.

Two kinds of checks:
- the work a gesture does, counted (each filter applied, with the pixels it was applied to): a light drag makes the
  filter moved and those above it only, at the drag's size; nothing is made at full resolution before the screen's
  picture of a release;
- the time it takes, the best of five: a light gesture's picture in 0.15 s, a heavy filter's own slider in 0.4 s, the
  screen's picture of a release in 1 s (2 s for a setting of the layer under its twelve filters, which are all made
  again). Against a calibration kernel: on a machine more than three times slower than
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

#: alone: times against thresholds, never beside other tests (tools/t.py runs it after the parallel part)
pytestmark = [pytest.mark.level3, pytest.mark.alone]

if os.environ.get("ASTRO_GPU_POST_THREADS"):        # a smaller machine's budget (a laptop's 4 cores): every library's
    import torch                                    # threads held to it, Post's bands too (client/postengine)
    torch.set_num_threads(int(os.environ["ASTRO_GPU_POST_THREADS"]))
    cv2.setNumThreads(int(os.environ["ASTRO_GPU_POST_THREADS"]))

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
    # 2 s, not 1: a setting of the layer itself makes all twelve filters again, heavy ones among them, at the
    # screen's size (the owner: 1.6 s is acceptable behind twelve filters); its drafts keep the light gesture's 0.15 s
    timed(slow_factor, 2.0, best_release, "the screen's picture of a layer setting under the stack")


# ---------------------------------------------------------------------------------------------- the deepest tree
class _Deep(pe.Provider):
    """A layer tree at its maximum depth (#181): Combine layers whose starless side is an S-H-O (B) with an H-O (A) as its Hα,
    its stars the stars' combination C. 6 Mpx planes; every plane asked for is counted (``asked``); the levels are block means,
    kept (a file is read once: what a gesture must not do again)."""

    def __init__(self, hh, ww):
        rng = np.random.default_rng(2)

        def blob(s):
            return (cv2.GaussianBlur(rng.random((hh, ww)).astype(np.float32), (0, 0), s) * 0.3 +
                    rng.normal(0, .004, (hh, ww)).astype(np.float32))
        self.hh, self.ww = hh, ww
        self.planes = {6: {"Ha": blob(30), "OIII": blob(40)},                              # the H-O (A)
                       5: {"SII": blob(35), "Ha": blob(30), "OIII": blob(25)},             # the S-H-O (B)
                       7: {"R": blob(20), "G": blob(25), "B": blob(30)}}                   # the stars' combination
        self.reads = {r: blob(15) for r in ("SII", "OIII")}
        self.asked = 0
        self._lv: dict = {}

    def version(self, pid):
        return 0

    def available(self, sid):
        return isinstance(sid, tuple) and sid[0] == pe.FUSION or sid in (("stars", 7), 7)

    def canvas(self, pids):
        return pe.Canvas(self.ww, self.hh)

    def _at(self, key, a, level):
        k = (key, level)
        if k not in self._lv:
            f = 2 ** level
            h, w = self.hh // f * f, self.ww // f * f
            self._lv[k] = a[:h, :w].reshape(h // f, f, w // f, f).mean(axis=(1, 3)).astype(np.float32) if f > 1 else a
        return self._lv[k]

    def fusion_planes(self, lay, level, box):
        self.asked += 1
        x0, y0, x1, y1 = box
        out = {r: self._at((lay["pid"], r), a, level)[y0:y1, x0:x1] for r, a in self.planes[lay["pid"]].items()}
        return out, np.ones(next(iter(out.values())).shape, bool)

    def fusion_form(self, lay):
        return {"maps": {"SII": ({"SII": 1.1}, 0.0), "Ha": ({"Ha": 1.2, "OIII": 0.05}, 0.01), "OIII": ({"OIII": 0.9}, 0.0)},
                "channels": {"SII": "L", "Ha": "L", "OIII": "L"}}

    def fusion_reads(self, lay, level, box):
        self.asked += 1
        x0, y0, x1, y1 = box
        return {r: self._at(("read", r), a, level)[y0:y1, x0:x1] for r, a in self.reads.items()}

    def layer(self, lay, level, box, draft=False, stop=None):
        assert lay.get("stars_over") is not None, lay
        x0, y0, x1, y1 = box
        x = sum(self._at((7, r), a, level) for r, a in self.planes[7].items()) / 3
        e = np.repeat((1.0 - np.exp(-5.0 * np.clip(x, 0, None)))[..., None], 3, -1)
        return e[y0:y1, x0:x1].astype(np.float32)


def _deep_layers(prov):
    """The layers of the deepest tree: the Combine layers, its S-H-O whose Hα is an H-O, the stars' combination; the
    deepest leaves' masks read the picture; a stack of twelve filters on top, and one on a deepest leaf."""
    from test_post_fusion import _lum_part
    from test_post_nest import fusion_c, group, tree
    stars = {"kind": "mix", "block": 4, "pid": 7, "title": "S-H-O · Stars", "fusion": dict(
        fusion_c(), roles=["R", "G", "B"], names={"R": "R", "G": "G", "B": "B"}, matrix=[[1 / 3] * 3] * 3,
        weights={"R": 1.0, "G": 1.0, "B": 1.0}, stars=True, neutral=True),
        "roles": {r: {"kind": "image"} for r in "RGB"}}
    t = tree(None, None, stars)
    sho = t["sides"]["starless"]
    ho = sho["roles"]["Ha"]
    sho["pid"], ho["pid"] = 5, 6
    sho["fusion"] = dict(sho["fusion"], roles=["SII", "Ha", "OIII"], names={"SII": "S II", "Ha": "Hα", "OIII": "O III"},
                         matrix=[[1.0, 0.0, 0.2], [0.0, 1.0, 0.3], [0.0, 0.0, 0.9]],
                         weights={"SII": 1.0, "Ha": 1.0, "OIII": 1.0})
    ho["fusion"] = dict(ho["fusion"], roles=["Ha", "OIII"], matrix=[[1.0, 0.0], [0.0, 0.5], [0.0, 0.5]],
                        weights={"Ha": 1.0, "OIII": 1.0})
    sho["roles"] = {"SII": {"kind": "image"}, "Ha": ho, "OIII": {"kind": "image"}}
    ho["roles"] = {"Ha": {"kind": "image"}, "OIII": {"kind": "image"}}
    g = group(t, stars_pid=7, stars_image="output",
              side_masks={"f:Ha/f:Ha": _lum_part(**{"from": "picture"}), "f:Ha/f:OIII": _lum_part(**{"from": "picture"})})
    deep = pe.new_filter_layer("levels", g["id"], "f:Ha/f:Ha")
    deep["params"].update(gamma=0.9)
    tops = []
    for key, params in STACK:
        f = pe.new_filter_layer(key, g["id"])
        f["params"].update(params)
        tops.append(f)
    return pe.normalize(list(reversed(tops)) + [deep, g]), g, deep, tops


def test_the_deepest_tree_s_gestures_make_what_they_change_only_and_in_time(monkeypatch, slow_factor):
    """Combine layers over an S-H-O whose Hα is an H-O (three levels), the stars' combination as its stars, the deepest
    leaves' masks reading the picture, twelve filters on top: a light drag on the top filter and on a deepest leaf's filter
    make that filter only, at the drag's size, and no plane is read again; the times of a deepest leaf's light gesture, an
    inner hue drag's draft and the screen's picture of a release."""
    prov = _Deep(2000, 3000)
    layers, g, deep, tops = _deep_layers(prov)
    cp = Composer(prov)
    cp.compose(copy.deepcopy(layers), 3, draft=True, track=True)
    cp.compose(copy.deepcopy(layers), 1, track=True)               # (the screen's picture, kept)
    cp.compose(copy.deepcopy(layers), 3, draft=True, track=True)
    work = Work(monkeypatch)
    read = prov.asked

    def edited(change):
        out = copy.deepcopy(layers)
        change(out)
        return out

    def set_top(strength):
        return lambda ls: next(x for x in ls if x.get("filter") == "saturation")["params"].__setitem__("strength", strength)
    best_top = best_deep = best_hue = best_release = 1e9
    for k in range(5):
        changed = edited(set_top(31.0 + k))
        t0 = time.perf_counter()
        cp.compose(changed, 3, draft=True, track=True)
        best_top = min(best_top, time.perf_counter() - t0)
    assert {w[0] for w in work.log} == {tops[-1]["id"]}, "a light drag on the top filter made other filters"
    biggest = max(w[1] for w in work.log)
    assert biggest <= (3000 >> 3) * (2000 >> 3) * 1.05
    work.log.clear()
    for k in range(5):
        def gamma(ls, k=k):
            next(x for x in ls if x["id"] == deep["id"])["params"]["gamma"] = 0.8 + 0.01 * k
        changed = edited(gamma)
        t0 = time.perf_counter()
        cp.compose(changed, 3, draft=True, track=True)
        best_deep = min(best_deep, time.perf_counter() - t0)
    assert deep["id"] in {w[0] for w in work.log}
    assert prov.asked == read, "a drag of a deepest leaf's filter read a plane again"
    for k in range(5):                                             # an inner hue: the H-O's colours
        def hue(ls, k=k):
            n = next(x for x in ls if pe.is_group(x))
            n["nest"] = copy.deepcopy(n["nest"])
            n["nest"]["sides"]["starless"]["roles"]["Ha"]["fusion"]["matrix"] = [
                [1.0, 0.0], [0.0, 0.3 + 0.02 * k], [0.0, 0.7 - 0.02 * k]]
        changed = edited(hue)
        t0 = time.perf_counter()
        cp.compose(changed, 3, draft=True, track=True)
        best_hue = min(best_hue, time.perf_counter() - t0)
    assert prov.asked == read, "a hue drag of the inner combination read a plane again"
    for k in range(5):                                             # the screen's picture after a release of the top filter
        changed = edited(set_top(40.0 + k))
        t0 = time.perf_counter()
        cp.compose(changed, 1, track=True)
        best_release = min(best_release, time.perf_counter() - t0)
    print()
    print(f"gate, deepest tree: top filter {best_top:.3f} s, deepest leaf's filter {best_deep:.3f} s, inner hue "
          f"{best_hue:.3f} s, release {best_release:.3f} s, machine x{slow_factor:.2f}")
    timed(slow_factor, 0.15, best_top, "the light gesture's picture at the deepest tree")
    timed(slow_factor, 0.15, best_deep, "a deepest leaf's light gesture")
    timed(slow_factor, 0.15, best_hue, "an inner hue drag's draft")
    timed(slow_factor, 1.0, best_release, "the screen's picture after a release, at the deepest tree")
