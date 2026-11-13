"""A Combine layers layer that puts one row's stars on another's image is a combination of two images: its frame is the
rectangle where both have data (as a combination's: render_simple.whole_box), where a single stack is not cropped."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PySide6")
from astro_gpu.client.postengine import Composer, new_workflow_layer  # noqa: E402
from astro_gpu.client.services.render import Folder, LayerSource  # noqa: E402

W, H = 256, 192


def folder(root: Path, name: str, valid: np.ndarray, stars: bool) -> Folder:
    d = root / name
    d.mkdir()
    np.save(d / "valid.npy", valid)
    prep = {"crop": [0, W, 0, H]}                                 # a single stack: no crop at all
    (d / ("prep_stars.json" if stars else "prep_simple.json")).write_text(json.dumps(prep), encoding="utf-8")
    return Folder(d)


def source(tmp_path, stars_valid):
    full = np.ones((H, W), bool)
    folders = {1: folder(tmp_path, "nebula", full, False), ("stars", 2): folder(tmp_path, "stars", stars_valid, True)}
    return LayerSource(folders.get, None)


def layer():
    lay = new_workflow_layer(1, "Combine", True, part="combined")
    return dict(lay, starless="output", stars_pid=2)


def test_the_combine_layers_frame_is_where_both_images_have_data(tmp_path):
    turned = np.ones((H, W), bool)
    yy, xx = np.mgrid[0:H, 0:W]
    turned[xx < 24 + (yy * 0.18)] = False                          # a turned frame: a diagonal edge on the left
    cv = Composer(source(tmp_path, turned)).canvas_of([layer()])
    ox, oy = cv.origin
    assert ox + cv.W <= W and oy + cv.H <= H and cv.W < W
    sub = turned[oy:oy + cv.H, ox:ox + cv.W]
    assert sub.all(), "every pixel of the frame has stars data"
    assert ox >= 24 + int(0.18 * (H - 1)) - 1                      # the diagonal's far end is inside the frame's left edge


def test_without_a_turned_frame_nothing_is_cropped(tmp_path):
    cv = Composer(source(tmp_path, np.ones((H, W), bool))).canvas_of([layer()])
    assert (cv.W, cv.H) == (W, H) and cv.origin == (0, 0)


def test_a_single_stack_alone_is_not_cropped(tmp_path):
    turned = np.ones((H, W), bool)
    turned[:, :60] = False
    src = source(tmp_path, turned)
    alone = new_workflow_layer(1, "Nebula", True, part=None)
    assert Composer(src).canvas_of([alone]).W == W
