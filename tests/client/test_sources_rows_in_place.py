"""The kinds' rows of a session in Inspect, and their + buttons, stay in place when files are added: the list of the
kind shown was drawn between two rows and pushed the rows below down at each addition (a tester's next click landed
on another row four times). It is drawn under all the rows now."""
from __future__ import annotations

from tests.client.test_sources_import import close, fits_file, open_page, qapp, settle  # noqa: F401


def layout_order(view):
    """The kinds' rows and the file list, in the order of the column's layout."""
    from astro_gpu.client.sources import GroupRow
    out = []
    for i in range(view.lv.count()):
        w = view.lv.itemAt(i).widget()
        if isinstance(w, GroupRow):
            out.append(w.key)
        elif w is not None and w is view.file_list:
            out.append("files")
    return out


def test_sources_rows_stay_in_place_when_files_are_added(qapp, tmp_path, monkeypatch):
    from astro_gpu.client.sources import SourcesView
    w, m, page = open_page(tmp_path, monkeypatch, sessions=1, workflows=("Full",))
    view = SourcesView(m)
    try:
        view.show_session(1, "lights")
        view.refresh()
        rows = ["lights", "darks", "flats", "bias", "flat_darks"]
        assert layout_order(view) == rows + ["files"]                   # the lights' list under every row
        assert view.add_files("darks", [str(fits_file(tmp_path / "d" / f"D_{k}.fits")) for k in range(6)]) == 6
        settle(w, page)
        view.refresh()
        assert view.group == "darks" and layout_order(view) == rows + ["files"] and view.file_list.count() == 6
    finally:
        view.close()
        close(w, m, page)


def test_sources_a_mono_sensor_has_no_pattern_wording():
    from astro_gpu.client.sources import sensor_text
    assert sensor_text({"width": 4656, "height": 3520, "cfa": "GGGG"}) == "4656 × 3520 px, monochrome"
    assert sensor_text({"width": 4656, "height": 3520, "cfa": ""}) == "4656 × 3520 px, monochrome"
    assert sensor_text({"width": 6000, "height": 4000, "cfa": "RGGB"}) == "6000 × 4000 px, RGGB pattern"
