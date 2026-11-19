"""Post's tool column (#173): the tools that act on the picture (Crop, Spot healing, Compare, Show mask, Sky grid,
Object names) are a column of icons at the right edge of the image, never in a "..." menu; the bar above keeps the view,
the versions, the options of the tool at work, Undo, Redo and Export, and the centre never gets narrower than they need."""
from __future__ import annotations

import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from test_post import _settings, full_layer, post, preps  # noqa: F401
from test_small_screens import _bar_fits, _open, _post_view, _settle, env, qapp  # noqa: F401

from astro_gpu.client import app as appmod
from astro_gpu.client import post3

SIZES = ((1366, 689), (1093, 640), (1024, 640), (1920, 1001), (1536, 785), (1280, 641))


def _tools(view):
    return (view.b_crop, view.b_heal, view.b_cmp, view.b_mask, view.b_grid, view.b_names)


def _bar_widgets(view):
    bar = view.top_bar
    return [bar.itemAt(i).widget() for i in range(bar.count()) if bar.itemAt(i).widget() is not None]


def _rect_in(w, widget):
    return widget.rect().translated(widget.mapTo(w, QPoint(0, 0)))


def _everything_fits(w, view, why):
    """The six column buttons and the bar's controls are shown, inside the window, none over another."""
    _settle(w, 4)
    assert not hasattr(view, "b_more") and not hasattr(view, "more_menu"), why
    must = [view.b_fit, view.zoom_lab, view.cb_version, view.b_version_new, view.b_version_del, view.b_undo,
            view.b_redo, view.b_export] + list(_tools(view))
    for b in must:
        assert b.isVisible(), (why, b.toolTip() or b.text())
    shown = [x for x in _bar_widgets(view) + list(_tools(view)) if x.isVisible()]
    win = w.rect()
    for x in shown:
        assert win.contains(_rect_in(w, x)), (why, x.toolTip() or x.text(), _rect_in(w, x), win)
    rects = [(x, _rect_in(w, x)) for x in shown]
    for i, (a, ra) in enumerate(rects):
        for b, rb in rects[i + 1:]:
            assert not ra.intersects(rb), (why, a.toolTip() or a.text(), b.toolTip() or b.text())
    _bar_fits(view)


@pytest.mark.parametrize("lang", ["en", "fr"])
@pytest.mark.parametrize("size", SIZES)
def test_the_tools_and_the_bar_all_fit_at_every_laptop_size(env, size, lang):  # noqa: F811
    appmod.apply_language(lang)
    w = _open(env, size)
    view = _post_view(w)
    _everything_fits(w, view, ("plain", size, lang))
    least, centre = view.centre.minimumWidth(), view.centre.width()
    view.b_cmp.setChecked(True)                                         # Compare's list in the bar
    _everything_fits(w, view, ("compare", size, lang))
    assert view.cb_cmp.isVisible()
    view.b_cmp.setChecked(False)
    view.b_crop.setChecked(True)                                        # the crop's Apply and Cancel in the bar
    _everything_fits(w, view, ("cropping", size, lang))
    assert view.b_crop_ok.isVisible() and view.b_crop_cancel.isVisible()
    assert view.centre.minimumWidth() == least                          # the minimum is the same whichever moment it is
    view.b_crop.setChecked(False)
    _settle(w)
    assert view.centre.width() == centre                                # and the splitter never moved for a tool
    view.b_crop.setChecked(False)


@pytest.mark.parametrize("size", [(1366, 689), (1024, 640), (1920, 1001)])
def test_the_properties_panel_dragged_widest_leaves_the_bar_whole(env, size):  # noqa: F811
    w = _open(env, size)
    view = _post_view(w)
    view.ldock._choose("docked")
    view.dock._choose("docked")
    _settle(w)
    for compare in (False, True):
        view.b_cmp.setChecked(compare)
        view.split.setSizes([0, 0, post3.PANEL_MAX])
        _settle(w)
        assert view.centre.width() >= view.centre.minimumWidth() >= post3.CENTRE_MIN
        if size[0] < 1500:
            assert view.centre.width() == view.centre.minimumWidth(), view.split.sizes()
        _everything_fits(w, view, ("widest", size, compare))
    assert view.centre.minimumWidth() >= view._bar_need(len(view._bar_steps()), (view.cb_cmp,)) + post3.STRIP_W


def test_each_tool_button_names_itself_with_its_key_and_the_keys_and_clicks_toggle_it(post):
    v = post.view
    v.add_mask(full_layer(v))
    v.select(full_layer(v)["id"], "layer")
    assert post.settle()
    named = ((v.b_crop, "Crop", "R"), (v.b_heal, "Spot healing", ""), (v.b_cmp, "Compare", "C"),
             (v.b_mask, "Show mask", "M"), (v.b_grid, "Sky grid", ""), (v.b_names, "Object names", ""))
    for b, name, key in named:
        head = b.toolTip().split("\n")[0]
        assert head == name + (f" ({key})" if key else ""), b.toolTip()
        assert b.text() == "" and b.isCheckable() and not b.icon().isNull()
    for b, key in ((v.b_cmp, Qt.Key_C), (v.b_mask, Qt.Key_M), (v.b_crop, Qt.Key_R)):
        v.viewer.setFocus()
        was = b.isChecked()
        QTest.keyClick(v.viewer, key)
        assert b.isChecked() != was, b.toolTip()
        QTest.keyClick(v.viewer, key)
        assert b.isChecked() == was, b.toolTip()
    for b in (v.b_cmp, v.b_mask, v.b_crop, v.b_heal):                    # a real click on the icon
        if not b.isEnabled():
            continue
        QTest.mouseClick(b, Qt.LeftButton, Qt.NoModifier, b.rect().center())
        QApplication.processEvents()
        assert b.isChecked(), b.toolTip()
        QTest.mouseClick(b, Qt.LeftButton, Qt.NoModifier, b.rect().center())
        QApplication.processEvents()
        assert not b.isChecked(), b.toolTip()
    assert v.b_crop.isChecked() is False and v.viewer.crop_edit is False
    v.b_crop.click()
    assert v.viewer.crop_edit and v.b_crop_ok.isVisibleTo(v)               # the tool is on, its options in the bar
    v.b_crop.click()


def test_the_column_is_one_fixed_strip_beside_the_picture_and_french_has_its_names(env):  # noqa: F811
    from astro_gpu.client.layout import STRIP_W
    appmod.apply_language("fr")
    w = _open(env, (1366, 689))
    view = _post_view(w)
    _settle(w)
    col = view.tools
    assert col.width() == STRIP_W and col.isVisible()
    assert col.geometry().left() >= view.viewer.geometry().right()          # at the picture's right edge
    heads = [b.toolTip().split("\n")[0] for b in _tools(view)]
    assert heads == ["Recadrage (R)", "Tampon correcteur", "Comparer (C)", "Voir masque (M)", "Grille du ciel",
                     "Noms des objets"]
    appmod.apply_language("en")
