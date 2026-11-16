"""Dropping calibration frames on a session's rows in Inspect, as a tutorial was filmed: the rows keep their place when
the line "shared with…" appears (the Flats row moved down by a line, and a drop at its former place answered "Drop on
a row"), and a drop goes to the session the page shows at the moment of the drop."""
from __future__ import annotations

from PySide6.QtCore import QPoint
from PySide6.QtWidgets import QApplication, QComboBox, QLabel

from tests.client.test_sources_import import close, fits_file, open_page, qapp, settle  # noqa: F401


def rows_y(view, events: bool = True) -> dict:
    """Where each kind's row is on the page, as a drop finds them (``events``: once the pending events are done)."""
    from astro_gpu.client.sources import GroupRow
    if events:
        QApplication.processEvents()
    rows = view.findChildren(GroupRow)
    assert all(r.isVisibleTo(view) for r in rows), [r.key for r in rows if not r.isVisibleTo(view)]
    return {r.key: r.mapTo(view, QPoint(0, 0)).y() for r in rows}


def shown(tmp_path, monkeypatch):
    from astro_gpu.client.sources import SourcesView
    w, m, page = open_page(tmp_path, monkeypatch, sessions=2, workflows=("Full",))
    view = SourcesView(m)
    view.resize(1300, 800)
    view.show()
    return w, m, page, view


def test_the_rows_keep_their_place_when_a_kind_becomes_shared(qapp, tmp_path, monkeypatch):
    w, m, page, view = shown(tmp_path, monkeypatch)
    try:
        darks = [str(fits_file(tmp_path / "d" / f"D_{k}.fits")) for k in range(4)]
        view.show_session(1, "lights")
        assert view.add_files("darks", darks) == 4
        settle(w, page)
        view.show_session(2, "darks")
        view.refresh()
        before = rows_y(view)
        assert len(before) == 5 and not m.shared_with(2, "darks")
        assert view.add_files("darks", darks) == 4          # the same darks: one master for the two sessions
        settle(w, page)
        view.refresh()
        assert m.shared_with(2, "darks") == [1]
        assert rows_y(view) == before, (before, rows_y(view))
        said = [x.text() for x in view.findChildren(QLabel) if "shared with" in x.text()]
        assert said == ["Darks · shared with night 1: one master"], said
        # built again (a reading of the worker that arrives, an addition): the rows are in place at once, where a
        # drop sent before the next round of events finds them (it fell on the page: "Drop on a row")
        view.build_left()
        assert rows_y(view, events=False) == before
    finally:
        view.close()
        close(w, m, page)
