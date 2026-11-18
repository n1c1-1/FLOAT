"""The screen's picture is rendered at the size the view shows it (a 27 Mpx image fitted in a 2560 x 1440 window:
2.9 Mpx, where the half was 6.8), at a level between two powers of two: the filters' radii, given in full-resolution
pixels, are brought to that level, and the picture stays close to the exact full-size picture reduced to its size
(the full size and the export are made at level 0, as before)."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from astro_gpu.client import postengine as pe
from astro_gpu.client.postengine import Canvas, Composer, Provider, new_filter_layer, new_workflow_layer
from astro_gpu.client.postfilters.detail import at_level
from astro_gpu.client.services.render import screen_level, to_u8


class Sky(Provider):
    """A night sky at full resolution: a sky background with its noise, a nebula, a few hundred stars; a level is the
    full resolution reduced by area (what a stage's result rendered smaller gives)."""

    def __init__(self, W=1200, H=800, seed=1):
        self.W, self.H = W, H
        rng = np.random.default_rng(seed)
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        neb = 0.25 * np.exp(-(((xx - W * .45) / (W * .25)) ** 2 + ((yy - H * .5) / (H * .3)) ** 2))
        neb += 0.1 * (np.sin(xx / 37) * np.cos(yy / 53) + 1) * np.exp(-(((xx - W * .6) / (W * .2)) ** 2))
        a = np.stack([0.06 + neb, 0.07 + neb * 0.45, 0.08 + neb * 0.6], -1)
        a += rng.normal(0, 0.012, a.shape)
        for _ in range(W * H // 4000):
            x, y, s, b = rng.uniform(0, W), rng.uniform(0, H), rng.uniform(0.7, 2.2), rng.uniform(0.1, 0.9)
            r, xi, yi = int(4 * s) + 2, int(x), int(y)
            ys, xs = slice(max(0, yi - r), min(H, yi + r + 1)), slice(max(0, xi - r), min(W, xi + r + 1))
            g = np.exp(-(((xx[ys, xs] - x) ** 2 + (yy[ys, xs] - y) ** 2) / (2 * s * s)))
            a[ys, xs] += b * g[..., None] * np.array([1.0, 0.95, 0.9])
        self.img = np.clip(a, 0, 1).astype(np.float32)

    def available(self, pid):
        return True

    def version(self, pid):
        return 0

    def canvas(self, pids):
        return Canvas(self.W, self.H)

    def has(self, name):
        return False

    def data_mask(self, name, level, box):
        return None

    def layer(self, lay, level, box, draft=False, stop=None):
        a = self.img
        if level > 0:
            a = cv2.resize(a, Canvas(self.W, self.H).size(level), interpolation=cv2.INTER_AREA)
        x0, y0, x1, y1 = box
        return np.ascontiguousarray(a[y0:y1, x0:x1])


#: the filters whose result depends on the scale: radii, meshes, tiles, wavelet layers, whole-image measures
HEAVY = [("dehaze", {"amount": 0.5}), ("ml_denoise", {"l1": 3, "l2": 2, "l3": 1.5, "l4": 1, "l5": 1, "amount": 0.8}),
         ("ml_sharpen", {"l1": 0.5, "l2": 0.8, "l3": 0.6, "l4": 0.4, "l5": 0.3}), ("clahe", {"amount": 0.5}),
         ("highlights", {"strength": 0.8, "knee": 0.3}), ("local_contrast", {"strength": 0.5}),
         ("sharpen", {"strength": 0.8}), ("denoise", {"luminance": 0.5, "color": 0.5}),
         ("faint_smooth", {"amount": 0.5}), ("saturation", {"strength": 40, "protect": 1.0}),
         ("mixer", {"sat_red": 30, "hue_aqua": 10})]


def stack(filters):
    a = new_workflow_layer(1, "A", True)
    fl = []
    for key, params in filters:
        f = new_filter_layer(key, a["id"])
        f["params"].update(params)
        fl.append(f)
    return pe.normalize(list(reversed(fl)) + [a])


def gap(a, b) -> tuple[float, float]:
    """(mean, 99.9th percentile) of the largest channel difference, in 8-bit levels."""
    d = np.abs(to_u8(a).astype(np.int16) - to_u8(b).astype(np.int16)).max(-1)
    return float(d.mean()), float(np.percentile(d, 99.9))


def test_the_screen_level_is_the_size_shown():
    assert screen_level(0.6) == 1 and screen_level(0.5) == 1                # zoomed in: the half, then regions
    lv = screen_level(1300 / 4272)                                          # NGC 7000 fitted in 2560 x 1440
    assert lv == 1.625
    W, H = 6384, 4272
    w, h = Canvas(W, H).size(lv)
    assert (w, h) == (2070, 1386) and w >= W * 1300 / 4272 and w <= 1.1 * W * 1300 / 4272
    assert Canvas(W, H).size(1.0) == Canvas(W, H).size(1) == (3192, 2136)  # a power of two: as always


def test_the_wavelet_layers_are_read_between_two_at_a_level_between_two():
    v = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert at_level(v, 2) == at_level(v, 2.0) == [3.0, 4.0, 5.0]           # a power of two: the same as always
    assert at_level(v, 1.5) == pytest.approx([2.5, 3.5, 4.5, 2.5])         # the layer past the last: none


@pytest.mark.parametrize("lv", [1.625, 1.25])
def test_the_screen_picture_stays_close_to_the_exact_picture_reduced(lv):
    """Every filter whose result depends on the scale, together: the picture at the size shown against the exact
    full-size picture reduced to that size, and against what the half gave (its picture reduced to the same size)."""
    pr = Sky()
    layers = stack(HEAVY)
    cv = Canvas(pr.W, pr.H)
    exact = cv2.resize(np.clip(Composer(pr).compose(pe.snapshot(layers), 0).rgb, 0, 1), cv.size(lv),
                       interpolation=cv2.INTER_AREA)
    shown = Composer(pr).compose(pe.snapshot(layers), lv).rgb
    assert shown.shape[:2] == cv.size(lv)[::-1]
    half = cv2.resize(np.clip(Composer(pr).compose(pe.snapshot(layers), 1).rgb, 0, 1), cv.size(lv),
                      interpolation=cv2.INTER_AREA)
    mean, p999 = gap(shown, exact)
    hmean, hp999 = gap(half, exact)
    assert mean < 1.6 and p999 < 12, (mean, p999, hmean, hp999)
    assert mean < hmean + 0.5, (mean, hmean)


@pytest.mark.parametrize("key,params", HEAVY)
def test_each_filter_at_a_level_between_two_is_as_close_as_at_the_half(key, params):
    """One filter at a time: at the size shown, no farther from the exact picture than the half's picture is."""
    pr = Sky(600, 400)
    layers = stack([(key, params)])
    cv = Canvas(pr.W, pr.H)
    lv = 1.625
    exact = cv2.resize(np.clip(Composer(pr).compose(pe.snapshot(layers), 0).rgb, 0, 1), cv.size(lv),
                       interpolation=cv2.INTER_AREA)
    shown = Composer(pr).compose(pe.snapshot(layers), lv).rgb
    half = cv2.resize(np.clip(Composer(pr).compose(pe.snapshot(layers), 1).rgb, 0, 1), cv.size(lv),
                      interpolation=cv2.INTER_AREA)
    mean, p999 = gap(shown, exact)
    hmean, hp999 = gap(half, exact)
    assert mean <= hmean + 0.5 and p999 <= hp999 + 3, (mean, p999, hmean, hp999)


def test_a_new_screen_size_lets_the_former_one_s_images_go():
    """The first picture is made before the view knows its size (at the half), the next ones at the size shown: the
    half's stack must not stay, it took the room of the full resolution's steps (every exact view of the owner's
    NGC 7000 made its twelve filters again)."""
    pr = Sky(600, 400)
    cp = Composer(pr, cache_bytes=400_000_000)
    layers = pe.snapshot(stack(HEAVY[4:8]))
    cp.screen_at(1)
    cp.compose(layers, 1, track=True)
    held = [lru for lru in (cp.images, cp.steps, cp.backdrops) if any(t == 1 for t, _g in lru._used.values())]
    assert held
    cp.screen_at(1.5)
    cp.compose(layers, 1.5, track=True)
    for lru in (cp.images, cp.steps, cp.backdrops, cp.parts):
        assert not any(t == 1 for t, _g in lru._used.values())
    assert any(t == 1.5 for t, _g in cp.steps._used.values())
    cp.screen_at(1.5)                                        # the same size again: nothing goes
    assert any(t == 1.5 for t, _g in cp.steps._used.values())
