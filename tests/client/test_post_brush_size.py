"""The mask brush can be as large as the image: its Size stopped at 1200 px, and covering the centre of a 4200 px wide
image (a disc of about 2800 px) took several strokes. The upper end of Size is the image's long side (1200 px at least,
as before, on a smaller image); the slider stays even by ratio, so the small sizes keep their room; [ and ] go over the
whole range."""
from __future__ import annotations

import pytest

from test_post import Setup, _settings, post, preps, qapp, wide, widgets_of  # noqa: F401

from astro_gpu.client.post3 import BRUSH_MAX, BRUSH_MIN, ContSlider, brush_pos, brush_px


def test_the_law_of_the_slider_is_even_by_ratio_up_to_the_long_side():
    top = 4202.0
    assert brush_px(0.0, top) == BRUSH_MIN and brush_px(1.0, top) == 4202
    for px in (4, 10, 60, 300, 1200, 2800, 4202):
        assert brush_px(brush_pos(px, top), top) == px
    assert brush_pos(60.0, top) > 0.35                               # the small sizes are not squeezed at the start
    assert brush_pos(2800.0, top) < 0.95                             # nor a disc over the field's centre at the very end
    # a thousandth of the rail (the slider's finest move) is under 1 % of the size, wherever it is
    assert all(abs(brush_px(t + 0.001, top) - brush_px(t, top)) <= max(1, 0.01 * brush_px(t, top)) for t in (0.1, 0.5, 0.9))


def _brush(s, top_layer=True):
    v = s.view
    lay = v.layers[-1]
    v.select(lay["id"], "layer")
    v.add_mask(lay)
    part = v.add_part(v.mask_owner(), "brush")
    assert s.settle(180)
    return v, part


def _size(v):
    return next(x for x in widgets_of(v, ContSlider) if x.name.text() == "Size")


def test_on_a_wide_image_the_brush_goes_to_its_long_side_with_the_slider_and_the_keys(wide):  # noqa: F811
    v, part = _brush(wide)
    assert v.src.canvas[0] > BRUSH_MAX
    long_side = max(v.src.canvas)
    assert v.brush_max() == long_side
    sz = _size(v)
    assert sz.val.text() == "60 px"
    sz.s.setValue(sz.s.maximum())                                    # the slider at its end
    assert part["size"] == long_side and v.viewer.brush_r == long_side / 2
    assert _size(v).val.text() == f"{long_side} px"
    part["size"] = 60.0
    for _i in range(40):                                             # ] again and again: up to the long side, no further
        v.brush_size(1.25)
    assert part["size"] == long_side and _size(v).val.text() == f"{long_side} px"
    for _i in range(60):                                             # [ : down to the smallest
        v.brush_size(1 / 1.25)
    assert part["size"] == BRUSH_MIN and _size(v).val.text() == "4 px"
    sz = _size(v)
    sz.set_value(brush_pos(2000.0, v.brush_max()))                   # a size over the old limit is shown (to the
    assert abs(int(sz.val.text().split()[0]) - 2000) <= 10           # slider's thousandth of its rail)


def test_on_a_small_image_the_brush_still_goes_to_1200_px(post):  # noqa: F811
    v, part = _brush(post)
    assert max(v.src.canvas) < BRUSH_MAX and v.brush_max() == BRUSH_MAX
    sz = _size(v)
    sz.s.setValue(sz.s.maximum())
    assert part["size"] == 1200
