"""What the tests of today's graph page also checked that is not the page: the block table's words, the panel's and the
window's widgets, Post's composition engine and the stage's progress. Their modules did not change with the page; the
tests went with the files that imported the old one (PORT_TABLE.md says which)."""
from __future__ import annotations

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication, QHBoxLayout, QLineEdit, QPushButton, QToolButton, QWidget


# ---------------------------------------------------------------------------------------------- the block table's words
def test_calibration_counts_take_the_plural(qapp):
    """GUI exploration 2: "3 darks · 1 flat", as the lights ("6 lights"); French singular for 0 and 1."""
    from astro_gpu.client import i18n
    from astro_gpu.client.graphnode import calibration_counts
    assert calibration_counts(3, 1) == "3 darks · 1 flat"
    assert calibration_counts(5, 0, 0) == "5 darks · 0 flats · 0 bias"
    i18n.set_language("fr")
    try:
        assert calibration_counts(3, 1, 1) == "3 darks · 1 flat · 1 offset"
        assert calibration_counts(0, 2, 4) == "0 dark · 2 flats · 4 offsets"
    finally:
        i18n.set_language("en")


def test_the_names_are_french_with_aligner_sur(qapp):
    from astro_gpu.client import i18n
    from astro_gpu.client.graphnode import TITLES, block_title
    assert TITLES["align_on"] == "Align on"
    i18n.set_language("fr")
    try:
        assert block_title(None, "align_on") == "Aligner sur"
        assert i18n._("reference to wire") == "référence à brancher"
        assert i18n._("SHO + RGB stars") != "SHO + RGB stars"
    finally:
        i18n.set_language("en")


# ---------------------------------------------------------------------------------------------- widgets
def test_a_shown_widget_without_parent_never_opens_as_a_window(qapp):
    from astro_gpu.client.panel import retain
    b = retain(QPushButton("x"), True)
    assert not b.isWindow() or not b.isVisible()                          # it closed the Stacking block's menu
    assert retain(QPushButton("y"), False).isHidden()


def test_a_field_going_away_writes_nothing(qapp):
    """A panel's field hidden as its block goes (Delete, Ctrl+Z) does not say "editing finished" any more: it would
    write into a block, session or workflow that is gone."""
    from astro_gpu.client.widgets import clear
    host = QWidget()
    lay = QHBoxLayout(host)
    box = QWidget()
    inner = QHBoxLayout(box)
    ed, ed2 = QLineEdit("x"), QLineEdit("y")
    inner.addWidget(ed2)
    lay.addWidget(ed)
    lay.addWidget(box)
    host.show()
    ed.setFocus()
    calls = []
    ed.editingFinished.connect(lambda: calls.append(1))
    ed2.editingFinished.connect(lambda: calls.append(2))
    clear(lay)
    QApplication.processEvents()
    assert calls == []
    host.close()


def test_a_banner_stays_until_it_is_closed(qapp):
    from astro_gpu.client.app import Banner
    b = Banner()
    assert b.isHidden()
    b.say("The project cannot be saved")
    assert not b.isHidden() and b.text.text() == "The project cannot be saved"
    QApplication.processEvents()
    assert not b.isHidden()
    b.findChild(QToolButton).click()
    assert b.isHidden()


# ---------------------------------------------------------------------------------------------- the stage's progress
def test_a_stage_phase_moves_the_bar_from_its_first_file():
    """Issue #50: progress within a phase fills that phase's share of the bar."""
    from types import SimpleNamespace

    from astro_gpu.core.stage import Context
    sent = []
    ctx = Context.__new__(Context)
    ctx.stage = SimpleNamespace(name="calibrate")
    ctx.journal = SimpleNamespace(emit=lambda kind, **kw: sent.append(kw["frac"]))
    ctx._t0, ctx._last_progress = 0.0, 0.0
    ctx.check_stop = lambda: None
    ctx.phase(0.0, 0.4)
    ctx.progress(0.5, "reading darks", force=True)
    ctx.phase(0.4, 1.0)
    ctx.progress(0.5, "lights", force=True)
    assert sent == [0.2, 0.7]


# ---------------------------------------------------------------------------------------------- Post's composition
def test_the_default_layer_of_a_stars_workflow_puts_the_stars_back_in_screen():
    from astro_gpu.client.model import POST_PART_NAMES
    from astro_gpu.client.postengine import new_workflow_layer
    lay = new_workflow_layer(1, "Full · " + POST_PART_NAMES["combined"], True, part="combined")
    assert lay["part"] == "combined" and lay["blend"] == "Screen" and lay["mode"] == "Normal"
    out = new_workflow_layer(2, "Classic stack", True)
    assert out["part"] is None and "blend" not in out and out["mode"] == "Normal"
    assert POST_PART_NAMES == {"starless": "Starless image", "stars": "Stars alone", "combined": "Combine layers"}


def test_a_part_layer_asks_the_stars_block_for_its_images():
    from astro_gpu.client import postengine
    from astro_gpu.client.postengine import new_workflow_layer, source_id
    assert source_id(new_workflow_layer(4, "Full")) == 4
    for part in ("starless", "stars", "combined"):
        assert source_id(new_workflow_layer(4, "Full", part=part)) == ("stars", 4)
    assert source_id(postengine.new_image_layer("stack/abc", "Stack")) == ("image", "stack/abc")
    assert new_workflow_layer(4, "Full", part="stars")["mode"] == "Screen"   # black around the stars
    assert new_workflow_layer(4, "Full", part="starless")["mode"] == "Normal"
    with pytest.raises(ValueError):
        new_workflow_layer(4, "Full", part="halos")


def test_load_keeps_a_valid_blend_and_drops_an_unknown_part():
    from astro_gpu.client.postengine import BLENDS, load_layers, new_workflow_layer, save_layers
    good = new_workflow_layer(1, "Full · Stars and background", True, part="combined")
    good["blend"] = "Linear dodge (add)"
    odd = new_workflow_layer(1, "Full · Stars and background", True, part="combined")
    odd["blend"] = "Overlay"
    gone = new_workflow_layer(1, "Full · halos", True)
    gone["part"] = "halos"                                                  # a part this version does not know
    other = new_workflow_layer(2, "Classic", True)                          # its workflow is gone
    kept = load_layers(save_layers([good, odd, gone, other]), {1})
    assert [(x["id"], x.get("blend")) for x in kept] == [(good["id"], "Linear dodge (add)"), (odd["id"], BLENDS[0])]


def test_linear_dodge_adds_the_layers():
    from astro_gpu.client.postengine import BLENDS, MODES, Composer
    assert "Linear dodge (add)" in MODES
    rng = np.random.default_rng(1)
    a = rng.random((8, 8, 3)).astype(np.float32)
    b = rng.random((8, 8, 3)).astype(np.float32) * 0.5
    got = Composer.blend("Linear dodge (add)", a, b)
    assert np.allclose(got, np.clip(a + b, 0, 1), atol=1e-6) and got.max() == 1.0
    assert np.allclose(Composer.blend("Screen", a, b), 1 - (1 - a) * (1 - b), atol=1e-6)
    assert BLENDS == ("Screen", "Linear dodge (add)", "Lighten") and set(BLENDS) <= set(MODES)
