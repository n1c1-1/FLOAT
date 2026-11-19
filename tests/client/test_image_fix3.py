"""Issue #174, the tester's second round: the workflow of an Image follows its file's name, the warning says the header line
once (EN and FR), and a reset Undo stretch is said in the log."""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402

from astro_gpu.client import i18n  # noqa: E402
from tests.client.test_image_source_fixes import model, qapp  # noqa: E402,F401
from tests.client.test_image_unstretch import autostretched  # noqa: E402
from tests.test_unstretch import write_fits  # noqa: E402


def two_files(tmp_path):
    lin, a = autostretched(tmp_path / "st_mono_mtf.fits")
    _l, b = autostretched(tmp_path / "st_mono_autostretch.fits",
                          "Autostretch (shadows: -2.80, target bg: 0.25, unlinked)")
    return a, b


def test_changing_the_image_renames_the_workflow_that_bore_its_file_name(model, tmp_path):
    a, b = two_files(tmp_path)
    q = model.add_image_workflow(model.unique_name(a.stem), str(a))
    model.set_graph(q, [])
    assert model.p.workflow(q.pid).name == "st_mono_mtf"
    model.set_image_file(q, str(b))
    assert model.p.workflow(q.pid).name == "st_mono_autostretch"
    assert [w.name for w in model.p.workflows].count("st_mono_autostretch") == 1
    model.rename(q.pid, "My nebula")                          # a name typed by the user stays
    model.set_image_file(q, str(a))
    assert model.p.workflow(q.pid).name == "My nebula"


def test_the_header_line_is_said_once_in_english_and_french(model, tmp_path):
    _a, b = two_files(tmp_path)
    colour = write_fits(tmp_path / "rgb_ghs.fits", np.full((3, 64, 80), 0.4, np.float32),
                        ["GHS (pivot: 0.010, amount: 19.09, local: 1.0 [0.00, 1.00])"])
    q = model.add_image_workflow("Image", str(b))
    model.set_graph(q, [])
    for lang in ("en", "fr"):
        i18n.set_language(lang)
        for f, line in ((b, "Autostretch (shadows"), (colour, "GHS (pivot")):
            model.set_image_file(q, str(f))
            text = model.image_state(q)["warning"]
            assert text.count(line) == 1, text
            assert "Its header lists" not in text if lang == "fr" else "Son en-tête" not in text
    i18n.set_language("en")


def test_a_reset_undo_stretch_is_said_in_the_log(model, tmp_path):
    a, b = two_files(tmp_path)
    q = model.add_image_workflow("Image", str(a))
    model.set_graph(q, [])
    model.set_image_file(q, unstretch="estimated")
    n = len(model.history)
    model.set_image_file(q, str(b))
    assert "unstretch" not in model.p.workflow(q.pid).source
    assert any("Undo stretch reset" in h[1] for h in model.history[n:])
    n = len(model.history)
    model.set_image_file(q, str(a))                           # nothing was set: nothing said
    assert not any("Undo stretch reset" in h[1] for h in model.history[n:])
