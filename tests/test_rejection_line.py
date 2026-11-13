"""The stack says which rejection the Bayer drizzle uses and why, in English and in French."""
from astro_gpu.client import i18n
from astro_gpu.client.services.run import engine_text
from astro_gpu.stages.stack import rejection_line, rejection_suggestion, values_per_point


def test_the_line_gives_the_automatic_choice_and_its_figures(monkeypatch):
    few = rejection_line("by_count", False, 3, 1.0, 1.0, "RGGB", 3.0)
    v, _s = values_per_point(3, 1.0, 1.0, "RGGB")
    assert rejection_suggestion(3, 1.0, 1.0, "RGGB", 3.0).value == "by_count"
    assert few.startswith("drizzle rejection: by frame count (automatic: ") and f"{v:.1f} values per point" in few
    assert " under the " in few
    many = rejection_line("own", False, 200, 1.0, 1.0, "RGGB", 3.0)
    assert many.startswith("drizzle rejection: sigma clipping on the raw pixels (automatic: ") and "at least the" in many
    assert rejection_line("integration", True, 9, 1.0, 1.0, "RGGB", 3.0) == \
        "drizzle rejection: integration of the debayered frames (set by hand)"


def test_the_line_is_translated():
    try:
        i18n.set_language("fr")
        for line in (rejection_line("by_count", False, 3, 1.0, 1.0, "RGGB", 3.0),
                     rejection_line("own", False, 200, 1.0, 1.0, "RGGB", 3.0),
                     rejection_line("own", True, 9, 1.0, 1.0, "RGGB", 3.0)):
            fr = engine_text(line)
            assert fr.startswith("rejet du drizzle : ") and "drizzle rejection" not in fr and " frame count" not in fr, fr
    finally:
        i18n.set_language("en")


def test_the_stacking_method_line_says_why_and_is_translated():
    from astro_gpu.core.params import Auto
    from astro_gpu.core import coverage as cv
    from astro_gpu.stages.stack import method_line
    auto = Auto("classic", cv.M_FEW, {"n": 9})
    assert method_line("classic", "auto", auto) == "stacking method: Classic (automatic: 9 frames (drizzle needs 15): Classic)"
    assert method_line("drizzle", "user", auto) == "stacking method: Bayer drizzle (set by hand)"
    drz = Auto("drizzle", cv.M_DRIZZLE_FINE, {"fwhm": 1.8, "cov": 7.5, "scale": 2.0})
    try:
        i18n.set_language("fr")
        fr = engine_text(method_line("classic", "auto", auto))
        assert fr == "méthode d'empilement : Classique (automatique : 9 images (le drizzle en demande 15) : Classique)" \
            or fr.startswith("méthode d'empilement : Classique (automatique : "), fr
        assert "frames" not in fr and "automatic" not in fr, fr
        fr = engine_text(method_line("drizzle", "auto", drz))
        assert "stacking" not in fr and "," in fr, fr
        assert engine_text(method_line("drizzle", "user", auto)) == "méthode d'empilement : Drizzle Bayer (réglée à la main)"
    finally:
        i18n.set_language("en")
