"""A combination's master layer keeps what a release does not change (client/postengine.py): its stretched filters by
what the stretch reads (Sky, Faint tones, the stretch), its starless image before the end of the stretch by all it
depends on, the planes of its masks by the images they read. The pixels are the same as computed from nothing."""
from __future__ import annotations

import copy

import numpy as np

from astro_gpu.client import postengine as pe
from astro_gpu.client.postengine import Composer
from astro_gpu.render.simple import SimpleRenderEngine

from test_post_fusion import FiltersAndStars, _lum_part, master


def _brush_mask():
    m = pe.new_mask()
    part = pe.new_part("brush")
    part["strokes"] = [(30.0, 22.0, 8.0, 100.0, True)]
    m["parts"] = [part]
    return m


def scene(picture=True):
    """A master layer with its stars side, a filter of its own on O III, a mask on S II (a Luminance range reading
    the picture, or its own filter) and on H-alpha (a brush), and a Saturation filter over it with a Luminance mask."""
    sii = _lum_part(**{"from": "picture"}) if picture else _lum_part()
    g = master(stars_pid=4, side_masks={"f:SII": sii, "f:Ha": _brush_mask()})
    lv = pe.new_filter_layer("levels", g["id"], pe.role_sub("OIII"))
    lv["params"].update(black=0.05, gamma=0.8)
    sat = pe.new_filter_layer("saturation", g["id"])
    sat["params"]["strength"] = 30.0
    sat["mask"] = _lum_part()
    return pe.normalize([sat, lv, g])


def _edits(layers):
    """The releases: Post's sliders of the layer, a filter's setting, a mask's, then back to the first state."""
    def at(f):
        new = copy.deepcopy(layers)
        f(new)
        return new
    g = next(x for x in layers if x["kind"] == "workflow")["id"]

    def lay(x):
        return next(y for y in x if y["id"] == g)

    def flt(x, key):
        return next(y for y in x if y.get("filter") == key)
    return [layers,
            at(lambda x: lay(x)["sliders"].update(contrast=3.0)),
            at(lambda x: lay(x)["sliders"].update(contrast=3.0, saturation=3.4)),
            at(lambda x: lay(x)["sliders"].update(sky=2.6)),
            at(lambda x: flt(x, "levels")["params"].update(gamma=1.2)),
            at(lambda x: lay(x)["side_masks"]["f:SII"].update(feather=6.0)),
            at(lambda x: flt(x, "saturation")["params"].update(strength=-40.0)),
            at(lambda x: flt(x, "saturation")["mask"].update(feather=4.0)),
            layers]


def test_a_release_s_picture_is_bit_identical_to_one_computed_from_nothing():
    """One composer through a series of releases (the results it keeps used), against a new composer that keeps
    nothing, at each release: the same pixels, bit for bit, the same masks and mask thumbnails; at the screen's level
    (with the thumbnails, as Post's) and at full resolution (the release's exact view), with or without a mask that
    reads the picture."""
    for picture in (True, False):
        warm = Composer(FiltersAndStars())
        for layers in _edits(scene(picture)):
            g = next(x for x in layers if x["kind"] == "workflow")["id"]
            for level, thumbs in ((1, True), (0, False)):
                got = warm.compose(copy.deepcopy(layers), level, thumbs=thumbs, track=True)
                cold = Composer(FiltersAndStars()).compose(copy.deepcopy(layers), level, thumbs=thumbs, cache=False)
                assert np.array_equal(got.rgb, cold.rgb), (picture, level)
                assert got.mask_thumbs.keys() == cold.mask_thumbs.keys()
                assert all(np.array_equal(got.mask_thumbs[k], cold.mask_thumbs[k]) for k in cold.mask_thumbs)
            # a mask asked for (Post's selected one): made, the same
            want = [(g, "f:SII")]
            got = warm.compose(copy.deepcopy(layers), 1, want=want, thumbs=True)
            cold = Composer(FiltersAndStars()).compose(copy.deepcopy(layers), 1, want=want, thumbs=True, cache=False)
            assert np.array_equal(got.rgb, cold.rgb) and np.array_equal(got.masks[want[0]], cold.masks[want[0]])


def test_contrast_rebuilds_neither_the_starless_image_nor_the_stretched_filters(monkeypatch):
    """Contrast and Saturation come after the stretch: a change of them redoes the end of the stretch only, at the
    screen's level and at full resolution, when no mask of a filter reads the picture; Sky moves the stretch: both are
    made again."""
    calls = {"side": 0, "factor": 0}
    side, factor = Composer._fusion_side_blend, SimpleRenderEngine.colour_factor

    def counted_side(self, *a, **k):
        calls["side"] += 1
        return side(self, *a, **k)

    def counted_factor(*a, **k):
        calls["factor"] += 1
        return factor(*a, **k)
    monkeypatch.setattr(Composer, "_fusion_side_blend", counted_side)
    monkeypatch.setattr(SimpleRenderEngine, "colour_factor", staticmethod(counted_factor))
    layers = scene(picture=False)
    eds = _edits(layers)
    comp = Composer(FiltersAndStars())
    for level in (1, 0):
        before = comp.compose(copy.deepcopy(layers), level).rgb
        calls.update(side=0, factor=0)
        got = comp.compose(copy.deepcopy(eds[1]), level).rgb              # Contrast
        assert calls == {"side": 0, "factor": 0}, level
        assert np.abs(got - before).max() > 1e-3
        assert np.array_equal(got, Composer(FiltersAndStars()).compose(copy.deepcopy(eds[1]), level, cache=False).rgb)
        calls.update(side=0, factor=0)
        comp.compose(copy.deepcopy(eds[3]), level)                         # Sky
        assert calls["side"] == 1 and calls["factor"] >= 1, level
