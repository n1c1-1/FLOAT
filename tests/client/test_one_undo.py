"""One Undo for the window: Ctrl+Z (wherever the focus is), Edit > Undo and a notice's Undo link do the same thing (the
night test pass: Ctrl+Z did nothing in the Graph while the menu worked, a notice's Undo did nothing). One chronological
order for the window (the second testers' pass: Ctrl+Z on the Graph after a gesture in Post undid an older gesture of
the Graph): the latest change is taken back wherever it was made; made on another page, it is said with a link that
undoes it there. What was undone is said in the interface's words, never the engine's raw names."""
from __future__ import annotations

import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from astro_gpu.client import app as appmod  # noqa: E402
from astro_gpu.client import prefs  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    from astro_gpu.client.widgets import QSS
    a = QApplication.instance() or QApplication([])
    a.setStyleSheet(QSS + appmod.EXTRA_QSS)
    return a


def _settle(w, n=8):
    for _i in range(n):
        w.worker.wait(30)
        QApplication.processEvents()
        time.sleep(0.01)


@pytest.fixture()
def win(qapp, tmp_path, monkeypatch):
    monkeypatch.setenv("ASTRO_GPU_SETTINGS", str(tmp_path / "settings.json"))
    prefs.load()
    appmod.apply_language("en")
    from astro_gpu.client.gallery import GALLERY
    w = appmod.Main()
    w.resize(1500, 900)
    w.show()
    w.activateWindow()
    w.new_project()
    end = time.monotonic() + 120
    while time.monotonic() < end and not w.s.ready:
        w.worker.wait(1)
        QApplication.processEvents()
    w.graph.show_templates()
    w.graph.templates.chosen.emit(next(t for t in GALLERY if t["key"] == "colour_full"))
    _settle(w)
    yield w
    w.close_project(discard=True)
    w.close()
    w.deleteLater()
    QApplication.processEvents()


def _gradient(w):
    g = w.graph.bridge.graph
    return next(k for k, x in g.blocks.items() if x.kind == "gradient")


def _method(w):
    return w.graph.bridge.graph.blocks[_gradient(w)].method


def _set_rbf(w):
    res = w.graph.bridge.act("set_node_method", block=_gradient(w), method="rbf")
    if not res.ok:
        res = w.graph.bridge.act("set_method", block=_gradient(w), method="rbf")
    assert res.ok
    _settle(w, 3)
    assert _method(w) == "rbf"


@pytest.mark.parametrize("how", ["ctrl+z on the graph", "edit menu", "notice link"])
def test_ctrl_z_the_menu_and_a_notice_link_undo_the_same_step(win, how):
    _set_rbf(win)
    if how == "ctrl+z on the graph":
        win.graph.canvas.setFocus()
        QApplication.processEvents()
        QTest.keyClick(QApplication.focusWidget() or win.graph.canvas, Qt.Key_Z, Qt.ControlModifier)
    elif how == "edit menu":
        next(a for m in win.menuBar().actions() for a in m.menu().actions() if a.text() == "Undo").trigger()
    else:
        win.graph.bridge.drops.link("undo")
    _settle(win, 3)
    assert _method(win) != "rbf", how
    said = win.status.text()
    assert said.startswith("Undone: ") and "RBF" in said and "rbf" not in said and "method:" not in said, said


def test_the_page_shortcut_never_competes_with_the_menu(win):
    from PySide6.QtGui import QKeySequence, QShortcut
    live = [sc for sc in win.graph.findChildren(QShortcut) if sc.isEnabled() and sc.key() == QKeySequence("Ctrl+Z")]
    assert live == []
    assert win.a_undo.shortcut() == QKeySequence("Ctrl+Z")


def test_post_with_nothing_of_its_own_offers_the_graphs_step(win):
    _set_rbf(win)
    win.set_mode(2)
    _settle(win, 3)
    assert win._post_shown()
    win.undo()
    said = win.status.text()
    assert "The last change was made in the Graph" in said and "Undo there" in said and _method(win) == "rbf"
    win._flash_linked("do")                                          # the link
    _settle(win, 3)
    assert win.mode == 0 and _method(win) != "rbf"


def test_the_graph_with_posts_step_last_offers_it(win):
    s = win.s
    win.set_mode(0)
    lid = s.add_post_layer(s.paths[0].pid, None)
    with s.from_post():
        s.set_post_layer(lid, visible=False)
    assert s.can_undo() and not s.can_undo(graph=True)
    win.undo()
    said = win.status.text()
    assert "The last change was made in Post" in said and "Undo there" in said
    win._flash_linked("do")
    _settle(win, 3)
    assert win.mode == 2


def test_a_gesture_in_post_is_undone_before_an_older_gesture_of_the_graph(win):
    """The testers' pass 2: "Sky -1" in Post, then Ctrl+Z on the Graph undid an older gesture of the Graph (the lights
    added): the project went "Untitled", every block Outdated. The latest change is Post's: said, never the Graph's."""
    _set_rbf(win)
    win.set_mode(2)
    _settle(win, 3)
    view = win.post.view
    n = len(view.undo)
    view.push()                                                      # a step of Post's own, after the Graph's
    _settle(win, 2)
    win.set_mode(0)
    _settle(win, 2)
    win.undo()
    said = win.status.text()
    assert "The last change was made in Post" in said and "Undo there" in said and _method(win) == "rbf"
    win._flash_linked("do")                                          # its link: Post shown, its step undone there
    _settle(win, 3)
    assert win.mode == 2 and len(view.undo) == n and _method(win) == "rbf"
    win.undo()                                                       # the Graph's step is now the latest one
    assert "The last change was made in the Graph" in win.status.text() and _method(win) == "rbf"


def test_redo_never_comes_back_over_a_later_change(win):
    _set_rbf(win)
    win.undo()
    _settle(win, 3)
    assert _method(win) != "rbf"
    win.set_mode(2)
    _settle(win, 3)
    win.post.view.push()                                             # a change made since, in Post
    _settle(win, 2)
    win.set_mode(0)
    _settle(win, 2)
    win.redo()
    assert win.status.text() == "Nothing to redo" and _method(win) != "rbf"


class _Draft:
    """A Gradient draft as the window reads it (gradientedit.GradientEditor.history_top / undo)."""

    def __init__(self, pid):
        from astro_gpu.client import journal
        self.view = self.p = None
        self.pid, self.taken = pid, []
        self.top = (journal.tick(), pid)

    def history_top(self, how):
        return self.top if how == "undo" else None

    def undo(self, pid=None):
        self.taken.append(pid)
        self.top = None
        return True

    def draft_pending(self):                       # its draft is a step only: nothing asked when its page is left
        return None


def test_the_gradient_drafts_step_is_undone_before_an_older_step_of_the_workflows(win, monkeypatch):
    """The testers' pass 2: in the Gradient view, Ctrl+Z undid an earlier "variant discarded" instead of the sample edit
    just made in the draft."""
    _set_rbf(win)
    draft = _Draft(win.s.paths[0].pid)                               # a draft step made after the Graph's
    monkeypatch.setattr(win, "_grad_editor", lambda: draft)
    win.undo()
    assert "The last change was made in the Gradient draft" in win.status.text() and _method(win) == "rbf"
    win._flash_linked("do")
    _settle(win, 3)
    assert draft.taken == [win.s.paths[0].pid] and win.mode == 1 and _method(win) == "rbf"


def test_an_undo_never_gives_a_saved_project_back_its_untitled(win, tmp_path, monkeypatch):
    """The testers' pass 2: an Undo of a step made before the first Save renamed the saved project "Untitled"."""
    _set_rbf(win)                                                    # a step made before the project is named
    monkeypatch.setattr(appmod, "ask_project_name", lambda *_a, **_k: (tmp_path / "lab", "M33 lab"))
    assert win.name_project()
    assert win.s.p.name == "M33 lab"
    win.undo()
    _settle(win, 3)
    assert _method(win) != "rbf" and win.s.p.name == "M33 lab" and "M33 lab" in win.project_name()
    win.redo()
    _settle(win, 3)
    assert _method(win) == "rbf" and win.s.p.name == "M33 lab"


def test_an_undo_that_would_compute_again_asks_first(win, monkeypatch):
    """An Undo or a Redo that makes results computed and up to date stale asks as a change of setting does: Undo /
    Cancel, nothing changed until then; "Confirm changes" off: at once."""
    from PySide6.QtWidgets import QLabel

    from astro_gpu.client.graphmodel import ChangeCost
    _set_rbf(win)
    monkeypatch.setattr(win.s, "history_cost", lambda how="undo": ChangeCost(["Gradient", "Denoise"],
                                                                           [(1, "gradient"), (1, "denoise")], 90))
    win.undo()
    card = win._history_card
    assert card.isVisible() and _method(win) == "rbf"
    assert card.findChild(QLabel, "history_cost").text().startswith("Computes again: Gradient, Denoise")
    assert "Undo: " in " ".join(x.text() for x in card.findChildren(QLabel))
    win.undo()                                                       # a question waits: it is answered first
    assert _method(win) == "rbf"
    card.cancel()
    _settle(win, 2)
    assert _method(win) == "rbf"
    win.undo()
    win._history_card.accept()
    _settle(win, 3)
    assert _method(win) != "rbf"
    prefs.set("confirm_changes", False)
    try:
        win.redo()                                                   # the preference off: at once
        _settle(win, 3)
        assert _method(win) == "rbf"
    finally:
        prefs.set("confirm_changes", True)


def test_a_link_stays_long_enough_and_while_the_mouse_is_over_it(win):
    from astro_gpu.client.graphview.canvas import LINK_MS
    assert appmod.LINK_MS >= 10000 and LINK_MS >= 10000
    win.flash("Something", ("Undo", lambda: None))
    assert win._flash_timer.interval() >= 10000
    win.refresh()                                                    # the status line read again: the link stays
    assert "<a href" in win.status.text()
    toast = win.graph.canvas.toast
    toast.say_links("Lights added", [("Undo", "undo")])
    assert toast.timer.interval() >= 10000


def test_the_overview_map_takes_the_blocks_states_after_an_undo(win):
    """The testers' pass 2: after an undo the overview map stayed orange over blocks all up to date."""
    from PySide6.QtGui import QColor

    from astro_gpu.client.graphview.style import STATE_COL
    _set_rbf(win)
    win.undo()
    _settle(win, 6)
    cv = win.graph.canvas
    cv.minimap.rects = [(r, QColor("#ff8800")) for r, _c in cv.minimap.rects]   # the colours an earlier state left
    cv.update_states()                                               # the states told again, no edit waited for
    want = [QColor(STATE_COL.get(it.d["state"], "#5d6470")).name() for it in cv.blocks.values()]
    assert [c.name() for _r, c in cv.minimap.rects] == want


def test_a_blocks_draft_step_is_undone_before_an_older_step_and_redone(win):
    """A block's draft (blockdraft.Draft, in the panel) holds its edits as steps of the window's one Undo order: Ctrl+Z
    takes the latest draft edit back before an older step of the workflows, Ctrl+Shift+Z makes it again."""
    from astro_gpu.client.blockdraft import Draft
    _set_rbf(win)                                                    # a step of the workflows, older
    d = Draft("fond", [win.s.paths[0].pid])
    d.title = "Gradient"
    d.put({("param", "gradient", "degree"): {"ch": {"kind": "param", "stage": "gradient", "name": "degree",
                                                    "value": 3, "suggested": None},
                                             "what": "Degree: 2 → 3", "short": "degree 3"}})
    win.s.pending_change = d
    win.undo()
    _settle(win, 3)
    assert not d and _method(win) == "rbf" and "Gradient draft" in win.status.text()     # the draft's step only
    win.redo()
    _settle(win, 3)
    assert d and _method(win) == "rbf"
    win.s.pending_change = None
