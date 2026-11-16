"""Post's bar keeps the tools one works with: New version and Export give up their text (their icon alone, their name
on hover) before any tool goes into "…", in both languages. In French, about 113 px wider, Spot healing went into "…"
at a window width where the English bar still showed it."""
from __future__ import annotations

import pytest

from test_small_screens import _bar_fits, _open, _post_view, _settle, env, qapp  # noqa: F401

from astro_gpu.client import app as appmod
from astro_gpu.client import post3

#: the window's width -> whether Spot healing was in the bar in English before this change (measured)
HEAL_SHOWN = {1250: True, 1366: False, 1600: True}


@pytest.mark.parametrize("width", sorted(HEAL_SHOWN))
@pytest.mark.parametrize("lang", ["en", "fr"])
def test_spot_healing_stays_in_the_bar_in_french_where_it_does_in_english(env, width, lang):  # noqa: F811
    appmod.apply_language(lang)
    w = _open(env, (width, 800))
    view = _post_view(w)
    _settle(w)
    _bar_fits(view)                                                   # nothing cut, nothing over another
    if HEAL_SHOWN[width]:
        assert view.b_heal.isVisible() and view.b_crop.isVisible(), (width, lang)
    hidden = [b for b in view._overflow() if b.isHidden()]
    if hidden:                                                        # a tool in "…": the two had given their text first
        assert view.b_export.text() == "" and view.b_version_new.text() == ""
        assert view.b_export.toolTip() and post3._("New version") in view.b_version_new.toolTip()
    if view.b_heal.isHidden():                                        # the tools one works with go after the view's aids
        assert all(b.isHidden() for b in (view.b_names, view.b_grid, view.b_mask, view.b_cmp))
