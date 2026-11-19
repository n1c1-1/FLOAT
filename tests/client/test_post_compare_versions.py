"""Post's Compare (#172): the picture split by a divider, another version left of it, the version shown right, driven as
the block editors' Split is. Real gestures on the page (keys, mouse), offscreen."""
from __future__ import annotations

import copy
import json

import pytest
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QImage, QMouseEvent, QPainter
from PySide6.QtWidgets import QApplication
from PySide6.QtTest import QTest

from test_post import _settings, named, post, preps, qapp, shown  # noqa: F401

from astro_gpu.client import i18n, imageview_canvas, post3, postengine, viewer as viewer_mod
from astro_gpu.client.post3 import SUGGESTED

pytestmark = pytest.mark.usefixtures("_settings")


def _two_versions(post):
    """The project's Post with two versions, the first shown."""
    v, p = post.view, post.model.p
    first = p.post["versions"][0]
    first["layers"] = postengine.save_layers(v.layers)
    second = {"id": 2, "name": "Version 2", "layers": postengine.with_new_ids(first["layers"]), "crop": None}
    p.post = {**p.post, "versions": [first, second], "active": 1}
    v._post_changed(None)

    def pick(vid):                                              # the model's set_post_version, as the page asks it
        p.post = {**p.post, "active": vid}
        v._post_changed(None)
    post.model.set_post_version = pick
    return first, second


def _key(v, key):
    v.viewer.setFocus()
    QTest.keyClick(v.viewer, key)


def _hover(vw, x):
    """A mouse move to (x, 60), delivered as Qt delivers it."""
    pos = QPointF(x, 60)
    QApplication.sendEvent(vw, QMouseEvent(QEvent.MouseMove, pos, vw.mapToGlobal(pos), Qt.NoButton, Qt.NoButton,
                                           Qt.NoModifier))


def _spy(v):
    sent = []
    real = v.worker.request
    v.worker.request = lambda req: (sent.append(req), real(req))[1]
    return sent, real


def test_c_compares_with_the_other_version_and_v_flips(post):
    v = post.view
    # one version: Suggested
    _key(v, Qt.Key_C)
    assert v.b_cmp.isChecked() and v.cb_cmp.currentData() == SUGGESTED
    assert v.viewer.compare_names == ("Suggested", "Version 1")
    _key(v, Qt.Key_C)
    assert not v.b_cmp.isChecked() and v.viewer.compare is None
    # two versions: the other one, not Suggested; Suggested is the list's last item
    _two_versions(post)
    assert [v.cb_cmp.itemText(i) for i in range(v.cb_cmp.count())] == ["Version 2", "Suggested"]
    _key(v, Qt.Key_C)
    assert v.cb_cmp.currentData() == 2 and v.viewer.compare is v.before
    assert v.viewer.compare_names == ("Version 2", "Version 1")
    assert post.settle() and v.before.img1 is not None
    assert v.hint.full_text().startswith("Drag the divider") and "Version 2" in v.hint.full_text()
    _key(v, Qt.Key_V)
    assert v.viewer.curtain == 1.0
    _key(v, Qt.Key_V)
    assert v.viewer.curtain == 0.0


def test_the_divider_is_dragged_with_the_mouse_and_kept_when_the_item_changes(post):
    v = post.view
    _two_versions(post)
    v.b_cmp.setChecked(True)
    assert post.settle()
    vw = v.viewer
    w = vw.width()
    xs = int(w * vw.curtain)
    _hover(vw, xs + 9)                                                       # within 12 px of the line
    assert vw.cursor().shape() == Qt.SplitHCursor
    _hover(vw, xs + 30)
    assert vw.cursor().shape() != Qt.SplitHCursor
    QTest.mousePress(vw, Qt.LeftButton, Qt.NoModifier, QPoint(xs + 9, 60))
    for x in (xs + 40, xs + 120, int(w * 0.3)):
        QTest.mouseMove(vw, QPoint(x, 60))
    QTest.mouseRelease(vw, Qt.LeftButton, Qt.NoModifier, QPoint(int(w * 0.3), 60))
    assert abs(vw.curtain - 0.3) < 0.01
    QTest.mousePress(vw, Qt.LeftButton, Qt.NoModifier, QPoint(w - 2, 60))      # far from it: the picture pans, no divider
    QTest.mouseRelease(vw, Qt.LeftButton, Qt.NoModifier, QPoint(w - 2, 60))
    assert abs(vw.curtain - 0.3) < 0.01
    v.cb_cmp.setCurrentIndex(v.cb_cmp.findData(SUGGESTED))                   # another item: the divider stays
    assert abs(vw.curtain - 0.3) < 0.01 and vw.compare_names[0] == "Suggested"
    QTest.mousePress(vw, Qt.LeftButton, Qt.NoModifier, QPoint(int(w * 0.3), 60))
    QTest.mouseMove(vw, QPoint(-50, 60))                                     # clamped
    QTest.mouseRelease(vw, Qt.LeftButton, Qt.NoModifier, QPoint(-50, 60))
    assert vw.curtain == 0.02


def test_zoomed_in_both_sides_share_the_view_and_the_compared_one_has_its_full_resolution(post):
    v = post.view
    _two_versions(post)
    v.b_cmp.setChecked(True)
    assert post.settle()
    v.viewer.zoom_to(1.0, 200, 150)
    v.viewer.pan_by(30, 20)
    assert post.settle()
    box = v.visible_box()
    assert box is not None and v.before.full is not None and v.src.full is not None
    (bx0, by0, bx1, by1), arr = v.before.full
    assert (bx0, by0) <= (box[0], box[1]) and (bx1, by1) >= (box[2], box[3])
    assert arr.shape[:2] == (by1 - by0, bx1 - bx0)
    assert v.before.size() == v.src.size()                                   # one grid, one transform for both sides
    a = v.before.region(0, box[0], box[1], box[2], box[3])
    assert a.shape[:2] == (box[3] - box[1], box[2] - box[0])
    assert (a == arr[box[1] - by0:box[3] - by0, box[0] - bx0:box[2] - bx0]).all()   # the exact pixels, not the half enlarged


def test_a_slider_re_renders_the_shown_side_only(post):
    v = post.view
    _two_versions(post)
    v.b_cmp.setChecked(True)
    assert post.settle()
    before_img = v.before.img1
    sent, real = _spy(v)
    named(v, "Full")["sliders"]["sky"] = 3.5
    v.request(final_only=True)
    assert post.settle()
    kinds = [r.kind for r in sent]
    assert "half" in kinds and "before" not in kinds and "before_region" not in kinds
    assert v.before.img1 is before_img
    v.worker.request = real


def test_the_compared_version_shown_swaps_and_a_deleted_one_falls_back(post):
    v = post.view
    first, second = _two_versions(post)
    v.b_cmp.setChecked(True)
    assert post.settle() and v.cb_cmp.currentData() == 2
    v.cb_version.setCurrentIndex(v.cb_version.findData(2))
    v._version_picked(v.cb_version.currentIndex())                           # the compared one picked as the shown one
    assert v._version == 2 and v.cb_cmp.currentData() == 1                   # the one just left is now compared
    assert v.viewer.compare_names == ("Version 1", "Version 2")
    assert post.settle()
    # a third version, compared, then deleted
    third = {"id": 3, "name": "Version 3", "layers": copy.deepcopy(first["layers"]), "crop": None}
    p = post.model.p
    p.post = {**p.post, "versions": [first, second, third]}
    v._post_changed(None)
    v.cb_cmp.setCurrentIndex(v.cb_cmp.findData(3))
    assert v.viewer.compare_names[0] == "Version 3"
    p.post = {**p.post, "versions": [first, second]}
    v._post_changed(None)                                                    # deleted: back to a default, no error
    assert v.cb_cmp.currentData() in (1, SUGGESTED) and v.b_cmp.isChecked()
    assert post.settle() and v.before.img1 is not None
    second["name"] = "Bold"                                                  # renamed: its tag follows (the shown one's)
    v._fill_versions()
    assert v.viewer.compare_names[1] == "Bold"


def test_crop_and_spot_healing_turn_compare_off_and_grey_it(post):
    v = post.view
    v.b_cmp.setChecked(True)
    _key(v, Qt.Key_R)                                                        # R with compare on: compare off
    assert v.b_crop.isChecked() and not v.b_cmp.isChecked() and v.viewer.compare is None
    assert not v.b_cmp.isEnabled() and v.b_cmp.toolTip().startswith("Compare (C)") and "cropping" in v.b_cmp.toolTip()
    _key(v, Qt.Key_C)                                                        # greyed: the key does nothing
    assert not v.b_cmp.isChecked()
    v.b_crop.setChecked(False)
    assert v.b_cmp.isEnabled()
    v.b_cmp.setChecked(True)
    v.b_heal.setChecked(True)
    assert not v.b_cmp.isChecked() and not v.b_cmp.isEnabled() and "healing" in v.b_cmp.toolTip()
    v.b_heal.setChecked(False)
    assert v.b_cmp.isEnabled() and v.b_cmp.toolTip().startswith("Compare (C)") and "Compare with another version" in v.b_cmp.toolTip()


def test_compare_is_view_state_only(post):
    v = post.view
    _two_versions(post)
    undo, redo = list(v.undo), list(v.redo)
    saved = json.dumps(postengine.save_layers(v.layers), sort_keys=True)
    post_state, touched = copy.deepcopy(post.model.p.post), post.model.touched
    v.b_cmp.setChecked(True)
    assert post.settle()
    vw = v.viewer
    QTest.mousePress(vw, Qt.LeftButton, Qt.NoModifier, QPoint(int(vw.width() * 0.5), 50))
    QTest.mouseMove(vw, QPoint(int(vw.width() * 0.7), 50))
    QTest.mouseRelease(vw, Qt.LeftButton, Qt.NoModifier, QPoint(int(vw.width() * 0.7), 50))
    v.cb_cmp.setCurrentIndex(v.cb_cmp.findData(SUGGESTED))
    v.b_cmp.setChecked(False)
    assert v.undo == undo and v.redo == redo and post.model.touched == touched
    assert json.dumps(postengine.save_layers(v.layers), sort_keys=True) == saved
    assert post.model.p.post == post_state


def test_both_drawings_use_the_shared_divider_and_the_french_is_there(post, monkeypatch):
    calls = []
    real = imageview_canvas.draw_split

    def spy(*a, **k):
        calls.append(tuple(a[3:5]))
        return real(*a, **k)
    monkeypatch.setattr(viewer_mod, "draw_split", spy)
    monkeypatch.setattr(imageview_canvas, "draw_split", spy)
    v = post.view
    _two_versions(post)
    v.b_cmp.setChecked(True)
    assert post.settle()
    v.viewer.grab()
    assert calls and calls[-1] == ("Version 2", "Version 1")
    n = len(calls)
    c = imageview_canvas.ImageCanvas()
    c.resize(300, 200)
    c.split_names = ("A", "B")
    img = QImage(300, 200, QImage.Format_RGB32)
    p = QPainter(img)
    c._split_line(p)
    p.end()
    assert len(calls) == n + 1 and calls[-1] == ("A", "B")
    i18n.set_language("fr")
    try:
        assert post3._("Compare") == "Comparer"
        for t in (post3.COMPARE_TIP, "Compared version (left of the divider)",
                  "Not while cropping: the crop box is drawn on the whole picture",
                  "Not while healing spots: the strokes act on the version shown",
                  "Drag the divider · V: one side whole · Left: “{name}”"):
            assert post3._(t) != t, t
    finally:
        i18n.set_language("en")


def test_the_shown_picture_is_not_the_compared_one_and_both_are_on_one_canvas(post):
    v = post.view
    _two_versions(post)
    sent, real = _spy(v)
    v.b_cmp.setChecked(True)
    assert post.settle()
    req = [r for r in sent if r.kind == "before"][-1]
    assert req.plain and req.canvas is v._canvas                              # the shown version's grid
    assert shown(v) is not None and v.before.img1.shape == shown(v).shape
    v.worker.request = real


def _best_shift(a, b, r=3):
    """The (dy, dx) within +-r that makes grey ``a`` closest to ``b`` (0, 0 when they are on one grid)."""
    import numpy as np
    a = a.astype(np.float32).mean(-1)
    b = b.astype(np.float32).mean(-1)
    h, w = a.shape
    best = None
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            d = np.abs(a[r + dy:h - r + dy, r + dx:w - r + dx] - b[r:h - r, r:w - r]).mean()
            if best is None or d < best[0]:
                best = (d, (dy, dx))
    return best[1], best[0]


@pytest.mark.parametrize("crop", [False, True])
def test_two_versions_differing_by_a_slider_are_on_one_grid_across_the_divider(post, crop):
    """Both sides show the same canvas positions: the compared picture, rendered for the shown version's canvas, lines
    up with the shown one pixel for pixel (a star at the divider continues across it), at the screen's level, zoomed in
    and panned, and with a crop on one version."""
    v = post.view
    first, second = _two_versions(post)
    second["layers"] = copy.deepcopy(first["layers"])
    for lay in second["layers"]:                                              # differing by one slider value only
        if lay.get("kind") == "workflow" and "sky" in lay.get("sliders", {}):
            lay["sliders"]["sky"] = lay["sliders"]["sky"] + 0.4
            break
    if crop:
        second["crop"] = [40, 30, 400, 300]
    v._post_changed(None)
    v.b_cmp.setChecked(True)
    assert post.settle()
    assert v.before.img1.shape == v.src.img1.shape
    shift, diff = _best_shift(v.before.img1, v.src.img1)
    assert shift == (0, 0) and diff < 12, (shift, diff)
    v.viewer.zoom_to(1.5, 250, 180)
    v.viewer.pan_by(-40, 25)
    assert post.settle()
    box = v.visible_box()
    x0, y0, x1, y1 = box
    a = v.before.region(0, x0, y0, x1, y1)
    b = v.src.region(0, x0, y0, x1, y1)
    shift, diff = _best_shift(a, b)
    assert shift == (0, 0) and diff < 12, (shift, diff)
    # the picture drawn: the columns on each side of the divider are the same canvas columns
    assert v.viewer.compare.size() == v.viewer.source.size()
