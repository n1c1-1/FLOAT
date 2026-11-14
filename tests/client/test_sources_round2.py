"""GUI exploration of 7 November, round 2 (sources, import, templates, reopening): what it found stays fixed. Offscreen
on scratch projects; nothing is computed."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if Path("C:/Windows/Fonts").is_dir():
    os.environ.setdefault("QT_QPA_FONTDIR", "C:/Windows/Fonts")
pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication  # noqa: E402
from test_sources_import import close, open_page, settle  # noqa: E402

from astro_gpu import api  # noqa: E402
from astro_gpu.client.model import ProjectModel  # noqa: E402
from astro_gpu.client.services.worker import ApiWorker  # noqa: E402
from astro_gpu.core import project2  # noqa: E402
from tests.client.pagehelp import choose, watch_toasts  # noqa: E402


def session_block(m, sid):
    """The graph's Session block of a night."""
    return next(b for b, x in m.stored.blocks.items() if x.kind == "session" and getattr(x, "sid", None) == sid)


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def fits_frame(path: Path, flt: str = "", when: str = "2026-11-06T22:00:00") -> Path:
    from astropy.io import fits
    path.parent.mkdir(parents=True, exist_ok=True)
    h = fits.PrimaryHDU(np.zeros((4, 4), np.uint16))
    h.header["DATE-OBS"] = when
    h.header["EXPTIME"] = 60.0
    if flt:
        h.header["FILTER"] = flt
    h.writeto(path, overwrite=True)
    return path


def frames(folder: Path, flt: str, n: int = 3) -> Path:
    for k in range(n):
        fits_frame(folder / f"L_{k:03d}.fits", flt)
    return folder


def test_two_filters_in_one_workflow_are_said_by_the_health_check(tmp_path):
    p = project2.create(tmp_path / "P", "P")
    p.sessions += [project2.Session(1, "a", filter="Ha", lights=[str(frames(tmp_path / "a", "Ha"))]),
                   project2.Session(2, "b", filter="OIII", lights=[str(frames(tmp_path / "b", "OIII"))])]
    p.add_workflow("W", {})
    got = [c for c in api.health_checks(p) if c.key == "filters_mixed"]
    assert got and got[0].level == "warn" and got[0].args["filters"] == ["Ha", "OIII"]


def test_group_sessions_split_filters_and_names_are_unique(tmp_path):
    """One folder holding Ha and OIII of a night: two sessions; a name taken gets a number."""
    folder = tmp_path / "mix"
    fits_frame(folder / "a.fits", "Ha")
    fits_frame(folder / "b.fits", "OIII")
    scan = api.scan_files({"lights": [folder]})
    gs = api.group_sessions(scan.frames["lights"])
    assert [(g.name, g.filter, g.count) for g in gs] == [("2026-11-06 · Ha", "Ha", 1), ("2026-11-06 · OIII", "OIII", 1)]
    p = project2.create(tmp_path / "P", "P")
    made = api.add_frames(p, scan)
    assert [(s.name, s.filter, len(s.lights)) for s in made] == [("2026-11-06 · Ha", "Ha", 1),
                                                                 ("2026-11-06 · OIII", "OIII", 1)]
    assert api.add_session(p, name="2026-11-06 · Ha").name == "2026-11-06 · Ha (2)"


def test_the_selected_block_is_kept_without_an_edit(qapp, tmp_path, monkeypatch):
    """Save (or close) without an edit keeps the block shown (in the settings: the project is not written); an Image
    selected is taken back on reopening."""
    w, m, page = open_page(tmp_path, monkeypatch, sessions=1, workflows=("Full",))
    try:
        page.palette.double_click("image")
        settle(w, page)
        m.flush()
        q = m.paths[-1]
        page.select(index=m.paths.index(q), step="image")
        assert not m.dirty
        before = (m.root / project2.FILE).read_bytes()
        m.flush()                                                        # Ctrl+S with nothing changed
        assert (m.root / project2.FILE).read_bytes() == before
        assert m._remembered_selection() == {"current": q.pid, "block": "image"}
        page.select(index=0, step="alignement")
    finally:
        close(w, m, page)                                                # closed without saving: kept too
    w2 = ApiWorker()
    m2 = ProjectModel(project2.load(tmp_path / "P"), w2)
    try:
        assert m2.path.pid == m.paths[0].pid and m2.sel_step == "alignement"
        m2.select(index=m2.paths.index(m2.view(q.pid)), step="image")
    finally:
        m2.close(save=True)
        w2.stop()
    w3 = ApiWorker()
    m3 = ProjectModel(project2.load(tmp_path / "P"), w3)
    try:
        assert m3.path.pid == q.pid and m3.sel_step == "image"
    finally:
        m3.close(save=False)
        w3.stop()
    w2 = ApiWorker()
    m2 = ProjectModel(project2.load(tmp_path / "P"), w2)
    try:
        assert m2.path.pid == q.pid and m2.sel_step == "image"
    finally:
        m2.close(save=False)
        w2.stop()


@pytest.mark.xfail(strict=True, reason="on the new page, the frames chosen in a project of one Image workflow choose "
                   "nothing (the old page showed the Image block)")
def test_an_image_only_project_shows_its_image_not_the_frames(qapp, tmp_path, monkeypatch):
    w, m, page = open_page(tmp_path, monkeypatch)
    try:
        page.palette.double_click("image")
        settle(w, page)
        page.select(step="sources")                                      # nothing chosen: no frames to import
        node = page.current_node()
        assert node is not None and node.key == "image"
        r = page.palette.current_row()                                   # the palette places in the image's workflow
        assert r is not None and m.compiled.wids.get(r.id) == m.paths[0].pid
    finally:
        close(w, m, page)


def test_files_dropped_on_a_session_go_where_the_user_says(qapp, tmp_path, monkeypatch):
    """Decision 9: files from the Explorer onto a Session block: the target they are dropped on (Lights, Darks...)
    says what they are; nothing is guessed from their names (a file called dark, dropped as lights, is a light)."""
    w, m, page = open_page(tmp_path, monkeypatch, sessions=2, workflows=("Full",))
    try:
        toasts = watch_toasts(page)
        dark = fits_frame(tmp_path / "cal" / "darks" / "d_001.fits")
        page.bridge.drops.drop(session_block(m, 2), "lights", [str(dark)])   # on the Lights target of night 2
        settle(w, page)
        s = m.p.session(2)
        assert any("d_001" in x for x in s.lights) and not s.darks
        assert toasts and "night 2" in toasts[-1]
    finally:
        close(w, m, page)


def test_plus_session_joins_the_flow_it_is_drawn_with(qapp, tmp_path, monkeypatch):
    """Two flows: "+ Session" (the palette's Session double-clicked) joins the flow of the session chosen, never the
    other one."""
    w, m, page = open_page(tmp_path, monkeypatch, sessions=2, workflows=("Full",))
    try:
        q1 = m.paths[0]
        m.p.workflow(q1.pid).sessions = [1]
        m.add_graph_workflow("Flow", [], sessions=[2])
        settle(w, page)
        choose(page, session_block(m, 2))
        settle(w, page)
        pid = page.graph.plus_session_pid
        assert pid is not None and m.p.workflow(pid).sessions == [2]
        page.palette.double_click("session")
        settle(w, page)
        new = max(s.id for s in m.p.sessions)
        assert new in m.p.workflow(pid).sessions and new not in m.p.workflow(q1.pid).sessions
    finally:
        close(w, m, page)


@pytest.mark.xfail(strict=True, reason="the new page's Session and Calibration blocks say '1 dark' and 'standard' under "
                   "their name for a darks folder not found (the panel and Locate say it)")
def test_a_folder_not_found_says_so_and_is_chosen_again_from_its_row(qapp, tmp_path, monkeypatch):
    """GUI exploration 4: a darks (or lights) folder renamed outside: the Session and Calibration blocks and the
    panel say it is not found, never "0 dark" or "none"; in the Session's files, Locate chooses where it is now (one
    undo step)."""
    from PySide6.QtWidgets import QPushButton
    from test_sources_import import texts

    from astro_gpu.client.sources import SourcesView
    w, m, page = open_page(tmp_path, monkeypatch, sessions=1, workflows=("Full",))
    view = SourcesView(m)
    try:
        darks = tmp_path / "cal" / "darks"
        for k in range(2):
            fits_frame(darks / f"D_{k:03d}.fits")
        m.p.session(1).darks = [(tmp_path / "cal" / "darks_old").as_posix()]      # renamed outside
        m.reload_sources()
        settle(w, page)
        g = m.block_map
        assert "darks not found" in g.summary(g.node(session_block(m, 1)))     # the one Session block
        cal = next(b for b, x in m.stored.blocks.items() if x.kind == "calibration")
        assert "darks not found" in g.summary(g.node(cal))
        choose(page, session_block(m, 1))
        page.refresh()
        assert any(t.startswith("Darks") and "folder not found" in t for t in texts(page.panel))
        view.refresh()
        said: list = []
        view.toast.connect(said.append)
        monkeypatch.setattr(SourcesView, "ask_folder", lambda self, title, start: darks.as_posix())
        next(b for b in view.findChildren(QPushButton) if b.text() == "Locate…").click()
        settle(w, page)
        assert [Path(x).name for x in m.p.session(1).darks] == ["D_000.fits", "D_001.fits"]   # its files, once for all
        assert said and "found again" in said[-1]
        g = m.block_map
        assert "2 darks" in g.summary(g.node(session_block(m, 1)))
        m.undo()
        assert m.p.session(1).darks == [(tmp_path / "cal" / "darks_old").as_posix()]
        m.p.session(1).lights = [(tmp_path / "raw" / "gone").as_posix()]         # the lights' folder too
        m.reload_sources()
        settle(w, page)
        page.refresh()
        assert any(t.startswith("Lights") and "folder not found" in t for t in texts(page.panel))
    finally:
        view.close()
        close(w, m, page)
