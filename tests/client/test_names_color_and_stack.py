"""One name for one thing: in French a stack is "Empilement" everywhere (the Table view's "+ Empilement", its column
group, its messages), as its button and its block; the two colour templates are shown "Color full" and "Color
minimal", as the interface spells "Color", while what projects and templates store stays as it was."""
from __future__ import annotations

import json
import re

from astro_gpu.client import i18n, names
from astro_gpu.client.gallery import GALLERY


def test_french_names_a_stack_empilement_everywhere():
    lang = i18n.language()
    try:
        i18n.set_language("fr")
        assert i18n._("+ Stack") == "+ Empilement" and i18n._("STACK") == "EMPILEMENT"
        assert i18n._("Stacks") == "Empilements" and i18n._("New stack") == "Nouvel empilement"
    finally:
        i18n.set_language(lang)
    for f in sorted((i18n.LOCALE / "fr").glob("*.json")):
        for en, fr in json.loads(f.read_text(encoding="utf-8")).items():
            assert not re.search(r"\b[Pp]iles?\b|\bPILE\b", fr), (f.name, en, fr)


def test_the_colour_templates_are_shown_color_and_stored_as_before():
    lang = i18n.language()
    try:
        i18n.set_language("en")
        assert names.shown("Colour full") == "Color full" and names.shown("Colour minimal 2") == "Color minimal 2"
        assert names.shown("Colour full · Denoise") == "Color full · Denoise"
        i18n.set_language("fr")
        assert names.shown("Colour full") == "Couleur complet" and names.shown("Colour minimal") == "Couleur minimal"
    finally:
        i18n.set_language(lang)
    # stored as before: a project saved with these names reads them, and a new one stores the same
    assert names.neutral("Colour full") == "Colour full" and names.neutral("Couleur complet") == "Colour full"
    keys = {t["key"]: t["name"] for t in GALLERY}
    assert keys["colour_full"] == "Colour full" and keys["colour_min"] == "Colour minimal"
