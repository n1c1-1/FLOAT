"""The size the Crop panel announces is the size of the exported picture: it said 4202 x 2777 px for a JPEG of
4202 x 2772 px. A crop is kept in the pixels of Post's image; when that image gets smaller afterwards (it is the area
every layer covers: a layer added, a workflow computed again), the box kept goes past its edge. The export wrote what
is inside the image, the panel and the history beside the export said the whole box."""
from __future__ import annotations

import pytest

from test_post import Setup, _settings, exporting, preps, qapp  # noqa: F401

from astro_gpu.client.postengine import crop_within, snapshot
from astro_gpu.client.services.render import ExportJob, run_export


def _exported(s):
    """The picture the export writes for the page as it is: ((width, height), its history)."""
    got = []
    v = s.view
    run_export(s.worker.renderer, ExportJob(layers=snapshot(v.layers), folder=str(s.out), name="c", crop=v.crop,
                                            history=True),
               lambda f, m: None, writer=lambda rgb, out, opts, meta: got.append((rgb.shape, meta)) or [])
    (shape, meta), = got
    return (shape[2], shape[1]), meta["history"]


def _said(v):
    """The size the Crop panel announces, its tool taken."""
    if not v.b_crop.isChecked():
        v.b_crop.setChecked(True)
    w, h = v.crop_size.text().replace("px", "").split("×")
    return int(w), int(h)


def _draw(v, box):
    """A box drawn with the Crop tool (what a drag of its handles ends with)."""
    v.b_crop.setChecked(True)
    v.viewer.crop_box = tuple(box)
    v._crop_drawn()


def _agree(s):
    v = s.view
    size, hist = _exported(s)
    assert _said(v) == size, (v.crop, _said(v), size)
    assert (hist["canvas"]["width"], hist["canvas"]["height"]) == size
    if hist["crop"] is not None:
        x0, y0, x1, y1 = hist["crop"]
        assert (x1 - x0, y1 - y0) == size
    return size


def test_the_size_announced_is_the_exported_pictures_for_a_free_and_a_locked_crop(exporting):  # noqa: F811
    s = exporting
    v = s.view
    W, H = v.src.size()
    assert _agree(s) == (W, H)                                       # no crop: the whole image
    _draw(v, (10.4, 20.6, 310.3, 187.5))                             # free, between pixels
    assert _agree(s) == (300, 167)
    _draw(v, (0, 0, W, H - 37))                                      # to three edges
    assert _agree(s) == (W, H - 37)
    for i in range(1, v.crop_ratio_cb.count()):                      # each fixed proportion, as its list sets it
        v.crop_ratio_cb.setCurrentIndex(0)
        _draw(v, (33.0, 21.0, 401.0, 333.0))
        v.crop_ratio_cb.setCurrentIndex(i)
        size = _agree(s)
        q = v.crop_ratio_cb.itemData(i)
        assert abs(size[0] / size[1] - q) < 0.02, (v.crop_ratio_cb.itemText(i), size)


@pytest.mark.parametrize("over", [(0, 5), (7, 0), (3, 12)])
def test_a_crop_kept_from_a_larger_image_says_the_size_it_will_be_exported_at(exporting, over):  # noqa: F811
    """The box of a version saved when Post's image was larger: its part inside the image is the picture."""
    s = exporting
    v = s.view
    W, H = v.src.size()
    v._show_crop([40, 30, W + over[0], H + over[1]])
    assert s.settle()
    assert _agree(s) == (W - 40, H - 30)
    assert v.viewer.crop_box == (40, 30, W, H)                       # shown as it is exported
    assert v.crop == [40, 30, W + over[0], H + over[1]]              # the box kept is not rewritten by looking at it
    _hist = _exported(s)[1]
    assert _hist["crop"] == [40, 30, W, H]


def test_a_box_with_nothing_inside_the_image_is_no_crop():
    assert crop_within([10, 20, 110, 80], 500, 300) == (10, 20, 110, 80)
    assert crop_within([10, 20, 510, 305], 500, 300) == (10, 20, 500, 300)
    assert crop_within([-4, -2, 110, 80], 500, 300) == (0, 0, 110, 80)
    assert crop_within([600, 20, 700, 80], 500, 300) is None
    assert crop_within(None, 500, 300) is None
