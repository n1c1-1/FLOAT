"""The palette's entries keep their place while blocks are added (the tutorials' film: a click aimed at an entry
landed on its neighbour). The line "for “name”" under the palette's title appeared with the first block chosen and
pushed every entry down by a line, by two when the name was long enough to wrap."""
from __future__ import annotations

import pytest
from PySide6.QtCore import QPoint
from PySide6.QtWidgets import QApplication

from tests.client.graphview.legacy import World


@pytest.fixture()
def world(qapp, tmp_path):
    made = []

    def make(**kw):
        w = World(tmp_path / f"w{len(made)}", **kw)
        made.append(w)
        return w
    yield make
    for w in made:
        w.close()


def places(W) -> dict:
    """Where each entry of the palette is on the page."""
    QApplication.processEvents()
    return {k: (c.mapTo(W.page, QPoint(0, 0)).x(), c.mapTo(W.page, QPoint(0, 0)).y())
            for k, c in W.page.palette.cards.items() if c.isVisible()}


@pytest.mark.parametrize("size", [(1500, 800), (1600, 900)])
def test_the_entries_stay_where_they_are_when_blocks_are_added(world, size):
    W = world(workflows=(), size=size)
    pal = W.page.palette
    assert pal.state == "docked" and W.b.selected is None
    start = places(W)
    assert len(start) >= 10
    for kind in ("session", "registration", "stacking"):
        pal.double_click(kind)
        W.settle()
        assert places(W) == start, kind
    assert pal.for_row.isVisible() and pal.for_row.text().startswith("for “")
    # a long name: said on its one line, cut, whole in its tooltip
    long = "A very long name of a workflow typed by the user"
    assert W.b.act("rename_row", row=W.graph.rows[0].id, name=long).ok
    W.settle()
    assert places(W) == start
    assert long in pal.for_row.toolTip() and pal.for_row.text() != pal.for_row.toolTip()
    W.b.choose(None)
    W.settle()
    assert places(W) == start
