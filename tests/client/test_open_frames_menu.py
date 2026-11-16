"""View > Open frames: the session panel's button in the menus (it was nowhere else). Greyed with its reason when no
session is chosen; chosen, it opens the session's files in Inspect, as the button."""
from __future__ import annotations

from PySide6.QtWidgets import QApplication, QMessageBox

from tests.client.test_one_undo import _gradient, _settle, qapp, win  # noqa: F401 - the fixtures


def _action(w):
    view = next(a.menu() for a in w.menuBar().actions() if a.text() == "View")
    view.aboutToShow.emit()                              # the menu opens
    return next(a for a in view.actions() if a.text() == "Open frames")


def test_open_frames_is_in_the_view_menu(win, monkeypatch):  # noqa: F811
    for kind in ("information", "warning", "critical"):
        monkeypatch.setattr(QMessageBox, kind, lambda *a, **k: (_ for _ in ()).throw(AssertionError("a message box")))
    b = win.graph.bridge
    b.choose(_gradient(win))
    _settle(win, 3)
    a = _action(win)
    assert not a.isEnabled() and a.toolTip() == "Choose a session first"
    a.trigger()                                          # greyed: nothing happens
    QApplication.processEvents()
    assert win.mode == 0
    b.choose(next(k for k, x in b.graph.blocks.items() if x.kind == "session"))
    _settle(win, 3)
    a = _action(win)
    assert a.isEnabled() and a.toolTip().startswith("Opens its files in Inspect")
    a.trigger()
    _settle(win, 3)
    assert win.mode == 1 and win.s.sel_step.startswith(("sources", "session"))


def test_open_frames_has_the_buttons_french_words():
    import json
    from pathlib import Path
    fr = Path(__file__).resolve().parents[2] / "astro_gpu" / "client" / "locale" / "fr"
    words = {}
    for f in fr.glob("*.json"):
        words.update(json.loads(f.read_text(encoding="utf-8")))
    assert words["Open frames"] == "Ouvrir images" and words["Choose a session first"] == "Choisir d'abord une session"
