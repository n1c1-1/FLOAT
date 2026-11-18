"""The graph page hidden behind another page of the window (Post) does not follow the model at each change: each of
Post's releases drew the hidden graph's blocks again (0.4 s of the window's own time on the owner's NGC 7000, at the
moment the next picture was asked). It follows once, when it is shown again."""
from __future__ import annotations

import time

from PySide6.QtWidgets import QApplication, QLabel, QStackedWidget


def pump(sec=0.05):
    end = time.monotonic() + sec
    while time.monotonic() < end:
        QApplication.processEvents()


def test_a_hidden_graph_page_follows_the_model_when_it_is_shown_again(qapp):
    from astro_gpu.client.graphview.document import MemoryDocument
    from astro_gpu.client.graphview.page import GraphPage
    win = QStackedWidget()
    page = GraphPage(MemoryDocument())
    other = QLabel("Post")
    win.addWidget(page)
    win.addWidget(other)
    win.resize(900, 600)
    win.show()
    pump()
    seen = []
    real = page.refresh
    page.refresh = lambda states=True: (seen.append(states), real(states))[1]
    page.request_refresh()
    pump()
    assert len(seen) == 1                                   # shown: it follows at once, as before
    win.setCurrentWidget(other)
    pump()
    seen.clear()
    for _ in range(5):                                      # Post's changes, the graph behind it
        page.request_refresh()
    pump()
    assert seen == []
    win.setCurrentWidget(page)
    pump()
    assert len(seen) == 1                                   # once, when shown again
    win.hide()
    win.deleteLater()
    pump()
