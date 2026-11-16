"""A mask put on another layer: "Copy mask" in the menu of a row that has one, "Paste mask" on any row, and the mask's
thumbnail dragged onto another row. Two sky corrections under the same luminance mask meant building the mask twice.

The copy is its own afterwards (a change of one leaves the other), the paste is one step of Undo named "Mask pasted on
<name>", it replaces the mask the row had, and every kind of mask goes: a layer's, a filter's, the synthetic stars',
and the masks of a Combine layers layer's two images."""
from __future__ import annotations

import copy

import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMessageBox

from test_post import Setup, _settings, preps, qapp  # noqa: F401

from astro_gpu.client import i18n
from astro_gpu.client import postengine as pe


@pytest.fixture()
def post(preps, qapp):  # noqa: F811
    s = Setup(preps, qapp, which=(1, 2), stars=True)
    assert s.settle()
    yield s
    s.close()


def action(menu, text):
    return next((a for a in menu.actions() if a.text() == text), None)


def built(mask):
    """A mask as its settings say it, without its parts' numbers."""
    return dict(mask, parts=[{k: v for k, v in p.items() if k != "id"} for p in mask["parts"]])


def sky_mask():
    """A mask as one builds it for the sky: a luminance range, a brush stroke taken off, feathered, not at full
    density."""
    m = pe.new_mask()
    lum = pe.new_part("luminance")
    lum.update(lo=0.0, hi=35.0)
    brush = pe.new_part("brush")
    brush.update(op="sub", strokes=[(120.0, 90.0, 40.0, 60.0, True)])
    m.update(parts=[lum, brush], feather=12.0, density=80.0)
    return m


def with_two_filters(post):
    v = post.view
    v.select(v.layers[-1]["id"], "layer")
    a = v.add_filter("saturation")
    b = v.add_filter("levels")
    a["mask"] = sky_mask()
    v.rebuild()
    post.settle()
    return v, v.layer(a["id"]), v.layer(b["id"])


def row_of(v, lay, sub=None):
    return next(r for r in v.rows if r.lay["id"] == lay["id"] and (r.lay.get("sub") if r.lay["kind"] == "sub" else None) == sub)


def test_copy_mask_then_paste_mask_puts_a_copy_of_its_own_on_another_filter_in_one_named_step(post, monkeypatch):
    boxes = []
    monkeypatch.setattr(QMessageBox, "exec", lambda self: boxes.append(self) or QMessageBox.Ok)
    v, a, b = with_two_filters(post)
    said = []
    v.toasted.connect(said.append)
    # nothing copied yet: Paste mask is there, greyed, and says why; Copy mask only on the row that has a mask
    paste = action(v.layer_menu_of(b), "Paste mask")
    assert paste is not None and not paste.isEnabled() and "No mask copied" in paste.toolTip()
    assert action(v.layer_menu_of(b), "Copy mask") is None
    action(v.layer_menu_of(a), "Copy mask").trigger()
    assert "copied" in said[-1]
    steps = len(v.undo)
    paste = action(v.layer_menu_of(b), "Paste mask")
    assert paste.isEnabled()
    paste.trigger()
    post.settle()
    b = v.layer(b["id"])
    assert built(b["mask"]) == built(a["mask"]) and b["mask"] is not a["mask"]
    assert not {p["id"] for p in b["mask"]["parts"]} & {p["id"] for p in a["mask"]["parts"]}
    assert len(v.undo) == steps + 1                                    # one step
    assert v.next_words("undo") == "Mask pasted on “Levels”"
    assert said[-1] == "Mask pasted on “Levels” · Ctrl+Z"
    assert row_of(v, b).m is not None                                  # its thumbnail is on its row
    # each is its own: a change of one leaves the other, the mask copied too (pasted again, it is as it was copied)
    was = copy.deepcopy(built(a["mask"]))
    b["mask"]["feather"] = 30.0
    b["mask"]["parts"][0]["hi"] = 60.0
    b["mask"]["parts"][1]["strokes"].append((10.0, 10.0, 5.0, 50.0, True))
    assert built(a["mask"]) == was
    a["mask"]["density"] = 40.0
    assert b["mask"]["density"] == 80.0
    c = v.add_filter("curves")
    v.paste_mask(c)
    assert built(v.layer(c["id"])["mask"]) == was
    v.do_undo()                                                        # the paste on Curves
    v.do_undo()                                                        # Curves
    # Undo takes the paste back, Redo makes it again, each named
    said.clear()
    v.do_undo()
    assert v.layer(b["id"])["mask"] is None and said[-1] == "Undone: Mask pasted on “Levels”"
    assert v.next_words("redo") == "Mask pasted on “Levels”"
    v.do_redo()
    assert v.layer(b["id"])["mask"] is not None and said[-1] == "Redone: Mask pasted on “Levels”"
    assert v.next_words("undo") == "Mask pasted on “Levels”"
    assert not boxes                                                   # never a message box


def test_paste_mask_replaces_the_mask_the_row_had_and_undo_brings_it_back(post):
    v, a, b = with_two_filters(post)
    old = pe.new_mask()
    old["parts"] = [pe.new_part("gradient")]
    b["mask"] = old
    v.rebuild()
    v.copy_mask(a)
    steps = len(v.undo)
    assert "Replaces" in action(v.layer_menu_of(b), "Paste mask").toolTip()
    action(v.layer_menu_of(b), "Paste mask").trigger()
    assert built(v.layer(b["id"])["mask"]) == built(a["mask"]) and len(v.undo) == steps + 1
    assert v.next_words("undo") == "Mask pasted on “Levels”"
    v.do_undo()
    assert built(v.layer(b["id"])["mask"]) == built(old)


def test_the_mask_shown_in_the_settings_follows_the_paste(post):
    v, a, b = with_two_filters(post)
    v.add_mask(b)                                                      # its (empty) mask is the one being edited
    assert v.sel == (b["id"], "mask")
    v.copy_mask(a)
    v.paste_mask(b)
    post.settle()
    assert v.sel == (b["id"], "mask") and len(v.layer(b["id"])["mask"]["parts"]) == 2
    from astro_gpu.client.post3 import ContSlider
    from test_post import widgets_of
    feather = next(s for s in widgets_of(v, ContSlider) if s.name.text() == "Feather")
    assert feather.value() == 12


def test_the_masks_of_a_layer_of_its_two_images_and_of_the_synthetic_rows_are_the_same_kind(post):
    v, a, _b = with_two_filters(post)
    g = next(x for x in v.layers if pe.is_group(x))
    v.copy_mask(a)
    # onto the layer itself, then onto its Stars image (its mask is its side's)
    action(v.layer_menu_of(g), "Paste mask").trigger()
    assert built(g["mask"]) == built(a["mask"])
    stars = pe.sub_row(g, "stars")
    assert action(v.layer_menu_of(stars), "Copy mask") is None         # no mask on that image yet
    action(v.layer_menu_of(stars), "Paste mask").trigger()
    assert built(g["side_masks"]["stars"]) == built(a["mask"]) and "starless" not in g["side_masks"]
    assert v.next_words("undo") == "Mask pasted on “Stars”"
    assert row_of(v, g, "stars").m is not None
    # from that image's mask to another layer
    g["side_masks"]["stars"]["feather"] = 3.0
    other = next(x for x in v.layers if x["kind"] == "workflow" and not pe.is_group(x))
    action(v.layer_menu_of(pe.sub_row(g, "stars")), "Copy mask").trigger()
    action(v.layer_menu_of(other), "Paste mask").trigger()
    assert other["mask"]["feather"] == 3.0 and built(other["mask"]) == built(g["side_masks"]["stars"])
    # the real stars have no mask of their own: greyed, with why
    real = action(v.layer_menu_of(pe.sub_row(g, "real")), "Paste mask")
    assert real is not None and not real.isEnabled() and "no mask of its own" in real.toolTip()
    # the whole composition still renders, and is saved and read back with its masks
    assert post.settle()
    again = pe.load_layers(pe.save_layers(v.layers), {1, 2})
    assert built(next(x for x in again if pe.is_group(x))["side_masks"]["stars"]) == built(g["side_masks"]["stars"])


def _drag(v, src, x_from, dst, qapp):  # noqa: F811
    """A real drag from ``x_from`` of the row ``src`` to the middle of the row ``dst``."""
    start = QPoint(x_from, src.height() // 2)
    end = src.mapFromGlobal(dst.mapToGlobal(dst.rect().center()))
    QTest.mousePress(src, Qt.LeftButton, Qt.NoModifier, start)
    for k in range(1, 6):
        QTest.mouseMove(src, start + (end - start) * k / 5)
        qapp.processEvents()
    return end


def test_a_masks_thumbnail_dragged_onto_another_row_copies_the_mask_there(post, qapp):  # noqa: F811
    v, a, b = with_two_filters(post)
    src, dst = row_of(v, a), row_of(v, b)
    order = [x["id"] for x in v.layers]
    steps = len(v.undo)
    end = _drag(v, src, src.m.geometry().center().x(), dst, qapp)
    assert dst.property("drop") and dst.sub.full_text() == "Copy mask here"    # the row says what the drop does
    QTest.mouseRelease(src, Qt.LeftButton, Qt.NoModifier, end)
    post.settle()
    b = v.layer(b["id"])
    assert built(b["mask"]) == built(a["mask"]) and b["mask"] is not a["mask"]
    assert a["mask"] is not None and [x["id"] for x in v.layers] == order      # copied: nothing moved
    assert len(v.undo) == steps + 1 and v.next_words("undo") == "Mask pasted on “Levels”"
    assert v._mask_clip is None                                        # the drag is not a Copy mask
    assert "Drag it onto another layer" in row_of(v, a).m.toolTip()
    # let go on its own row, or outside the rows: nothing
    src = row_of(v, a)
    steps = len(v.undo)
    QTest.mousePress(src, Qt.LeftButton, Qt.NoModifier, src.m.geometry().center())
    QTest.mouseMove(src, src.m.geometry().center() + QPoint(40, 0))
    QTest.mouseRelease(src, Qt.LeftButton, Qt.NoModifier, src.m.geometry().center() + QPoint(40, 0))
    qapp.processEvents()
    assert len(v.undo) == steps


def test_a_row_dragged_by_anything_else_than_its_mask_still_moves(post, qapp):  # noqa: F811
    v, a, b = with_two_filters(post)
    src, dst = row_of(v, a), row_of(v, b)
    end = _drag(v, src, src.n.geometry().center().x(), dst, qapp)
    QTest.mouseRelease(src, Qt.LeftButton, Qt.NoModifier, end)
    post.settle()
    assert v.layer(b["id"])["mask"] is None                            # no mask copied


def test_the_two_commands_in_french(post):
    v, a, b = with_two_filters(post)
    i18n.set_language("fr")
    try:
        assert action(v.layer_menu_of(a), "Copier masque") is not None
        paste = action(v.layer_menu_of(b), "Coller masque")
        assert paste is not None and paste.toolTip().startswith("Aucun masque copié")
        v.copy_mask(a)
        v.paste_mask(b)
        assert v.next_words("undo") == "Masque collé sur « Niveaux »"
    finally:
        i18n.set_language("en")
