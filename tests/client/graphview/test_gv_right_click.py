"""A right click on a block opens its context menu and nothing else. The menu opens once the button is up: opened at
the press (the default of the systems other than Windows, and of the off-screen window a beta tester drove), the
release of the same click fell on the menu and chose an entry, and a variant was made at once with no menu seen."""
from __future__ import annotations

from PySide6.QtCore import QEvent, QObject, Qt, QTimer
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMenu

from astro_gpu.client.app import context_menus_on_release
from tests.client.graphview.test_gv_page import B, page, pump  # noqa: F401  (fixtures by name)


class _Shown(QObject):
    """When a menu shows, whether the right button was still down."""

    def __init__(self):
        super().__init__()
        self.menus = []

    def eventFilter(self, obj, ev):                                   # noqa: N802
        if ev.type() == QEvent.Show and isinstance(obj, QMenu):
            self.menus.append(bool(QApplication.mouseButtons() & Qt.RightButton))
        return False


def test_a_right_click_on_a_block_opens_its_menu_at_the_release_and_changes_nothing(B, qapp):  # noqa: N803
    hints = qapp.styleHints()
    was = hints.contextMenuTrigger()
    context_menus_on_release(qapp)
    spy = _Shown()
    qapp.installEventFilter(spy)
    try:
        B.tpl("colour_minimal")
        gr = B.one("gradient")
        B.g._see(B.g.block_point(gr))
        pt = B.g.block_point(gr)
        vp = B.canvas.viewport()
        win = vp.window().windowHandle()                   # through the window, as the system gives a click
        pt = vp.mapTo(vp.window(), pt)
        before = B.js()
        seen = []

        def close_menu():
            m = QApplication.activePopupWidget()
            if isinstance(m, QMenu):
                seen.append([a.text() for a in m.actions() if not a.isSeparator()])
                m.close()

        QTimer.singleShot(80, lambda: QTest.mouseRelease(win, Qt.RightButton, Qt.NoModifier, pt))
        QTimer.singleShot(250, close_menu)
        QTest.mousePress(win, Qt.RightButton, Qt.NoModifier, pt)
        pump(0.5)
        assert seen and "Variant here" in seen[0], seen
        assert spy.menus == [False], spy.menus             # once, the button up
        B.exact(before)                                     # nothing made, nothing changed
    finally:
        qapp.removeEventFilter(spy)
        hints.setContextMenuTrigger(was)
