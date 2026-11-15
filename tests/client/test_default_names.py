"""The names the application gives get their language when they are shown: a workflow made from a template, a numbered
block, a copy, a variant, a session. They are stored in English, shown in the interface's language while they are the
default, and a name the user typed is shown as typed. A project saved by an earlier version with such a name in French is
shown in either language too."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if Path("C:/Windows/Fonts").is_dir():
    os.environ.setdefault("QT_QPA_FONTDIR", "C:/Windows/Fonts")
pytest.importorskip("PySide6")

from astro_gpu.client import i18n, names  # noqa: E402
from astro_gpu.graphcore.msg import M, name_of, say  # noqa: E402
from tests.client.test_swap_smoke import (  # noqa: E402,F401 - fixtures of the application's smoke test
    env,
    made_from,
    qapp,
    ready,
    settle,
    win,
)


@pytest.fixture(autouse=True)
def english():
    i18n.set_language("en")
    yield
    i18n.set_language("en")


def test_a_default_name_is_shown_in_the_language_of_the_interface():
    for stored, fr in (("Colour full", "Couleur complet"), ("Filters full", "Filtres complet"),
                       ("Combine layers 2", "Combiner calques 2"), ("Ha · Denoise", "Ha · Débruitage"),
                       ("Ha copy", "Ha copie"), ("Heart · variant", "Heart · variante"), ("Stars", "Étoiles"),
                       ("Image 3", "Image 3"), ("S-H-O", "S-H-O")):
        i18n.set_language("fr")
        assert names.shown(stored) == fr
        i18n.set_language("en")
        assert names.shown(stored) == names.SHOWN.get(stored, stored)       # "Colour full" shown "Color full"
        assert names.neutral(stored) == stored


def test_a_name_made_in_french_by_an_earlier_version_follows_the_interface_too():
    for old, en in (("Couleur complet", "Colour full"), ("Combiner calques 2", "Combine layers 2"),
                    ("Débruitage 2 copie", "Denoise 2 copy"), ("Ha · Débruitage", "Ha · Denoise")):
        assert names.shown(old) == names.SHOWN.get(en, en) and names.neutral(old) == en
        i18n.set_language("fr")
        assert names.shown(old) == old
        i18n.set_language("en")


def test_a_name_the_user_typed_is_shown_as_typed():
    for typed in ("Mon test", "Rosette nuit 2", "Cœur 12", "M 31 · Hα", "Image finale du 12"):
        for lang in ("en", "fr"):
            i18n.set_language(lang)
            assert names.shown(typed) == typed and names.neutral(typed) == typed


def test_the_names_the_core_stores_are_english_whatever_the_language():
    i18n.set_language("fr")
    assert name_of("Session {n}", n=2) == "Session 2"
    assert name_of("{_row} · {lab}", _row="Ha", lab="Denoise") == "Ha · Denoise"
    assert name_of("{_name} copy", _name="Full") == "Full copy"
    # a message names a row as the interface shows it
    m = M("{_name} added", _name="Combine layers 2")
    assert "Combiner calques 2" in say(m, i18n._)
    i18n.set_language("en")
    assert "Combine layers 2" in say(m, i18n._)


def test_a_workflow_made_from_a_template_in_french_is_stored_in_english_and_shown_in_both(win, env):
    win.change_language("fr")
    made_from(win, env, "colour_full")
    settle(win, 3)
    s = win.s
    assert [w.name for w in s.p.workflows] == ["Colour full"]                 # stored in English
    assert [q.name for q in s.paths] == ["Couleur complet"]                    # shown in French
    assert s.unique_name("Couleur complet") == "Colour full 2"                 # a second one, as the first's series
    row = win.graph.bridge.doc.graph.rows[0]
    assert row.name == "Colour full"
    win.change_language("en")
    settle(win, 3)
    assert [q.name for q in s.paths] == ["Color full"]                         # and in English, at once (shown
    assert [w.name for w in s.p.workflows] == ["Colour full"]                 # as the interface spells it)


def test_the_colour_block_is_written_one_way_and_an_earlier_name_follows():
    """The colour block is written one way, "Color" (the engine owner's spelling, the engine's and the graph core's own
    word); the names projects stored for a while with "Colour" ("Ha · Colour") are shown and stored so too."""
    from astro_gpu.client.graphnode import TITLES
    assert TITLES["couleur"] == "Color"
    i18n.set_language("en")
    assert names.shown("Ha · Colour") == "Ha · Color" and names.neutral("Ha · Colour") == "Ha · Color"
    assert names.shown("Ha · Color") == "Ha · Color"
    i18n.set_language("fr")
    try:
        assert names.shown("Ha · Color") == "Ha · Couleur" and names.shown("Colour") == "Couleur"
    finally:
        i18n.set_language("en")


def test_french_agrees_with_the_kind_of_frames():
    """The testers' pass 2: « 107 brutes ajoutés »."""
    from astro_gpu.client.i18n import _
    from astro_gpu.graphcore.msg import M, say
    i18n.set_language("fr")
    try:
        said = say(M("{n} {kind} added to “{t}”", n=7, kind="lights", t="Session 1"), _)
        assert said == "7 brutes ajoutées à « Session 1 »", said
        assert say(M("{n} {kind} added to “{t}”", n=7, kind="darks", t="Session 1"), _).startswith("7 darks ajoutés")
        assert _("{kind} copied to {dest}").format(kind=_("Lights"), dest="x") == "Brutes copiées vers x"
    finally:
        i18n.set_language("en")
