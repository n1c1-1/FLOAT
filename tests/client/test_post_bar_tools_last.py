"""Post's tools never go into a "…" menu (#173): Spot healing and Crop stay in their column at every width, in both
languages (in French the bar is about 113 px wider, and Spot healing went into "…" where the English bar showed it);
New version and Export give up their text for their icon (their name on hover) when the bar has no room."""
from __future__ import annotations

import pytest

from test_small_screens import _bar_fits, _open, _post_view, _settle, env, qapp  # noqa: F401

from astro_gpu.client import app as appmod
from astro_gpu.client import post3

WIDTHS = (1250, 1366, 1600)


@pytest.mark.parametrize("width", WIDTHS)
@pytest.mark.parametrize("lang", ["en", "fr"])
def test_every_tool_stays_in_its_column_in_both_languages(env, width, lang):  # noqa: F811
    appmod.apply_language(lang)
    w = _open(env, (width, 800))
    view = _post_view(w)
    _settle(w)
    _bar_fits(view)                                                   # nothing cut, nothing over another
    assert not hasattr(view, "b_more")
    for b in (view.b_heal, view.b_crop, view.b_names, view.b_grid, view.b_mask, view.b_cmp):
        assert b.isVisible() and b.toolTip(), (width, lang, b.toolTip())
    if view.b_version_new.text() == "":                               # no room: their icon, their name on hover
        assert view.b_export.text() == "" and post3._("New version") in view.b_version_new.toolTip()
    if view.b_export.text() == "":
        assert view.b_export.toolTip()
