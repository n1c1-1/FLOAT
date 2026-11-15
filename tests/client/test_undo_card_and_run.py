"""The beta testers' Undo findings: Ctrl+Z during a run took the lights out without a word (it now asks, saying that the
run goes on with what it started with); the card's two answers were both « Annuler » in French (Défaire / Garder);
an Undo is always said with what it took back; the Gradient draft's steps keep Edit › Redo up to date."""
from __future__ import annotations

from PySide6.QtWidgets import QLabel

from astro_gpu.client import i18n
from tests.client.test_one_undo import _method, _set_rbf, _settle, qapp, win  # noqa: F401  (fixtures by name)


def test_an_undo_during_a_run_asks_and_says_the_run_goes_on(win):
    _set_rbf(win)
    rc = win.run
    rc.phase = "running"
    try:
        win.undo()
        card = win._history_card
        assert card.isVisible() and _method(win) == "rbf"                     # nothing taken back before the answer
        texts = " ".join(x.text() for x in card.findChildren(QLabel))
        assert "goes on with the frames and settings it started with" in texts
        assert card.ok.text() == "Undo" and card.no.text() == "Keep"
        card.accept()
        _settle(win, 3)
        assert _method(win) != "rbf" and win.status.text().startswith("Undone: ")
    finally:
        rc.phase = "idle"


def test_the_cards_answers_never_share_a_word_in_french(win):
    from astro_gpu.client.graphmodel import ChangeCost
    from astro_gpu.client.historycard import HistoryCard
    i18n.set_language("fr")
    try:
        card = HistoryCard(win, "undo", "x", ChangeCost(["Gradient"], [(1, "gradient")], 30), lambda: None)
        assert (card.ok.text(), card.no.text()) == ("Défaire", "Garder")
        card.cancel()
        card = HistoryCard(win, "redo", "x", None, lambda: None, running=True)
        assert card.ok.text() != card.no.text() and card.findChild(QLabel, "history_running") is not None
        assert card.findChild(QLabel, "history_cost") is None                 # nothing computed again: not said
        card.cancel()
    finally:
        i18n.set_language("en")


def test_an_undo_is_always_said_with_what_it_took_back(win, monkeypatch):
    _set_rbf(win)
    monkeypatch.setattr(win.s, "restored", "", raising=False)
    win._say_restored("Undone", "Undone: {what}", words="")
    assert win.status.text() == "Undone: a change of the workflows (Graph)"
    win._say_restored("Undone", "Undone: {what}", words="Method: RBF")
    assert win.status.text() == "Undone: Method: RBF"


def test_a_session_added_is_named_in_its_undo(win):
    sid = win.s.add_filter_session("Ha")
    _settle(win, 3)
    assert "Ha" in win.s.next_words("undo") and "added" in win.s.next_words("undo")
    assert sid


def test_the_drafts_steps_keep_edit_redo_up_to_date(win):
    win._inspect()
    ed = win._grad_editor()
    assert ed is not None and ed.receivers("2historyChanged()") >= 1       # the window's Edit › Undo / Redo follow
