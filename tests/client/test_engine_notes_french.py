"""The engine's notes the Inspect page shows are in the interface's language (the tutorials' film: "No darks: …" and
"No flats: …" in English in the French application)."""
from astro_gpu import texts
from astro_gpu.api import CALIBRATION_NOTES


def test_the_calibration_notes_have_their_french():
    for kind, text in CALIBRATION_NOTES.items():
        assert texts.has_translation(text, "fr"), kind
        assert texts.translate(text, "fr") != text


def test_the_no_darks_note_says_the_hot_pixels_are_still_found():
    """The tutorials' film: the note said the hot pixels were "left to the bad pixel map" while the log counted the hot
    pixels found: they are found, by the map."""
    assert "found by the bad pixel map" in CALIBRATION_NOTES["darks"]
