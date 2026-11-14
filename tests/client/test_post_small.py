"""An image of another shape is fitted inside Post's canvas, its aspect kept."""
from __future__ import annotations

import numpy as np

from astro_gpu.client.postengine import Composer, new_image_layer
from astro_gpu.client.services.render import LayerSource


def _source(**imgs):
    src = LayerSource(lambda pid: None, service=None)
    src.image_sources.update(imgs)
    return src


def test_an_image_of_another_shape_is_fitted_inside_the_canvas_its_aspect_kept():
    rng = np.random.default_rng(1)
    ref = (rng.random((3, 160, 256)) * 0.2).astype(np.float32)          # the canvas: 256 x 160
    pic = (rng.random((3, 70, 100)) * 0.9 + 0.1).astype(np.float32)    # 100 x 70: another shape (1.4286 / 1.6)
    src = _source(a=ref, b=pic)
    cp = Composer(src)
    layers = [new_image_layer("b", "b"), new_image_layer("a", "a")]
    for lay in layers:
        lay["sliders"] = {}
    cv = cp.canvas_of(layers)
    assert (cv.W, cv.H) == (256, 160)                      # the picture does not shrink it
    cov = src.cover(layers[0], 0, (0, 0, cv.W, cv.H))
    assert cov is not None
    ys, xs = np.nonzero(cov)
    w, h = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
    assert abs(w / h - 100 / 70) < 0.02                    # the picture's own aspect
    assert h >= cv.H - 1 and w < cv.W                       # fitted by its height, centred
    assert abs((xs.min() + xs.max()) / 2 - (cv.W - 1) / 2) <= 1
    assert src.cover(layers[1], 0, (0, 0, cv.W, cv.H)) is None      # the first layer covers all
    out = cp.compose(layers, 0, cache=False).rgb
    below = cp.compose(layers[1:], 0, cache=False).rgb
    assert np.array_equal(out[:, : xs.min()], below[:, : xs.min()])    # around it, the layer below shows untouched
    assert np.abs(out[:, xs.min() + 2: xs.max() - 2] - below[:, xs.min() + 2: xs.max() - 2]).mean() > 0.05
    half = cp.compose(layers, 1, cache=False).rgb           # the same at the half resolution
    assert half.shape[:2] == cv.size(1)[::-1]
    assert np.array_equal(half[:, :4], cp.compose(layers[1:], 1, cache=False).rgb[:, :4])


def test_an_image_of_the_same_shape_still_fills_the_canvas():
    ref = np.full((3, 160, 256), 0.2, np.float32)
    big = np.full((3, 320, 512), 0.8, np.float32)           # a drizzle's grid: the same shape
    src = _source(a=ref, b=big)
    cp = Composer(src)
    layers = [new_image_layer("b", "b"), new_image_layer("a", "a")]
    cv = cp.canvas_of(layers)
    assert src.cover(layers[0], 0, (0, 0, cv.W, cv.H)) is None
