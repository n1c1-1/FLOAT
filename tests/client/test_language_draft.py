"""A change of language builds the project's screen again: a block's draft is answered first (Apply, Try variant,
Discard, Stay), as Close, Save, Run and Home ask it; Stay keeps the screen, its draft and the language."""
from __future__ import annotations

from astro_gpu.client import app as appmod
from astro_gpu.client import blockdraft, i18n, prefs
from tests.client.test_one_undo import qapp, win  # noqa: F401  (the fixtures are used by name)


def test_a_change_of_language_asks_for_the_draft_first(win, monkeypatch):
    pn = win.graph.panel
    asked, reset = [], []
    answer = {"now": "stay"}
    monkeypatch.setattr(pn, "draft_pending", lambda: ("Gradient", ["Method: RBF"], None))
    monkeypatch.setattr(pn, "draft_reset", lambda: reset.append(1))
    monkeypatch.setattr(blockdraft, "ask_leave", lambda parent, block, lines, cost, variant=True:
                        asked.append(block) or answer["now"])
    screen = win.proj_screen
    prefs.set("language", "fr")                                     # as the settings' list writes it
    try:
        win.change_language("fr")
        assert asked == ["Gradient"] and reset == []
        assert win.proj_screen is screen and i18n.language() == "en" and prefs.get("language") == "en"
        answer["now"] = "discard"
        prefs.set("language", "fr")
        win.change_language("fr")
        assert reset == [1] and win.proj_screen is not screen and i18n.language() == "fr"
    finally:
        monkeypatch.setattr(pn, "draft_pending", lambda: None)
        appmod.apply_language("en")
