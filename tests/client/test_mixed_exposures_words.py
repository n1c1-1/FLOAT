"""A session mixing exposure times says them with the language's decimal mark: "0,5 s" in French (it said "0.5 s")."""
from __future__ import annotations

from astro_gpu.client import i18n
from astro_gpu.client.frames import mixed_exposures_text


def test_mixed_exposure_times_have_a_decimal_comma_in_french():
    try:
        i18n.set_language("fr")
        fr = mixed_exposures_text([0.5, 30.0, 120.0])
        assert "0,5 s" in fr and "0.5" not in fr and "30 s" in fr and "120 s" in fr, fr
    finally:
        i18n.set_language("en")
    assert "0.5 s" in mixed_exposures_text([0.5, 30.0])
