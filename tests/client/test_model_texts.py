"""Texts of the project model: the target's coordinates with a decimal comma in French, and the results Use survey's
Undo left in the trash counted with a real plural ("1 result", "2 results"), never "result(s)"."""
from __future__ import annotations

import re

from astro_gpu.client import app as appmod
from astro_gpu.client import i18n
from tests.client.test_one_undo import qapp, win  # noqa: F401  (the fixtures are used by name)


def test_coordinates_and_counts_read_well_in_both_languages(win):
    s = win.s
    s.p.target.name = "M 42"
    try:
        en = s.target_state().note
        assert re.search(r"RA \d+\.\d{3}°, Dec [+-]\d+\.\d{3}°", en), en
        s.survey_skipped = ["gradient"]
        assert s._survey_words("Undone").endswith("1 result computed again since: kept")
        s.survey_skipped = ["gradient", "color"]
        assert s._survey_words("Undone").endswith("2 results computed again since: kept")
        appmod.apply_language("fr")
        fr = s.target_state().note
        assert re.search(r"AD \d+,\d{3}°, Déc [+-]\d+,\d{3}°", fr), fr
        s.survey_skipped = ["gradient"]
        assert s._survey_words("Annulé").endswith("1 résultat recalculé depuis : gardé")
        s.survey_skipped = ["gradient", "color"]
        assert s._survey_words("Annulé").endswith("2 résultats recalculés depuis : gardés")
        assert not i18n.missing(), sorted(i18n.missing())
    finally:
        s.survey_skipped = []
        appmod.apply_language("en")
