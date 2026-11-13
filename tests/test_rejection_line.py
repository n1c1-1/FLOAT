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
