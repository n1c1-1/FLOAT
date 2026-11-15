"""A greyed control says why on hover (the testers' pass 2: 4 of 5 tried said nothing): the window shows its tooltips
active or not (the run's window in front, a dialog just closed), the naming dialog's Save says why it is greyed, and a
palette card is not restyled again while its state stays (restyled under the mouse, it never showed its reason)."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt

from tests.client.test_one_undo import qapp, win  # noqa: F401  (the fixtures are used by name)


def test_the_window_shows_its_tooltips_when_it_is_not_the_active_one(win):
    assert win.testAttribute(Qt.WA_AlwaysShowToolTips)


def test_the_naming_dialogs_save_says_why_it_is_greyed(qapp, tmp_path):
    from astro_gpu.client.home import NewProjectDialog
    d = NewProjectDialog(tmp_path, "", None)
    d.name.setText("")
    assert not d.ok.isEnabled() and d.ok.toolTip() == "Type the project's name first"
    (tmp_path / "M33").mkdir()
    d.name.setText("M33")
    assert not d.ok.isEnabled() and d.ok.toolTip().startswith("A folder of this name is already there")
    d.name.setText("M33 lab")
    assert d.ok.isEnabled() and d.ok.toolTip()
    d.deleteLater()


def test_a_palette_card_keeps_its_style_while_its_state_stays(win, monkeypatch):
    pal = win.graph.palette
    pal.finish()
    card = next(c for c in pal.cards.values() if not c.enabled and c.reason)
    styled = []
    monkeypatch.setattr(card, "setStyleSheet", lambda *a: styled.append(a))
    card.set_state(card.enabled, card._reason, card._hint)
    assert styled == [] and card.toolTip()


def test_a_project_is_proposed_in_documents_float(qapp, tmp_path, monkeypatch):
    from astro_gpu.client import home
    monkeypatch.setattr(home.prefs, "recent", lambda: [])
    monkeypatch.setattr(home.prefs, "get", lambda k, d=None: None)
    loc = Path(home.projects_location())
    assert loc.name == "FLOAT" and loc.parent.name == "Documents"
