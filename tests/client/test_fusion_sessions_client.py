"""GUI exploration 2, the graph's structure: sessions joining a flow, a Combine channels's channels and what it adds, names,
Image blocks without a file, a session's Calibration, panel fields going away; offscreen on a scratch project (nothing is
computed)."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if Path("C:/Windows/Fonts").is_dir():
    os.environ.setdefault("QT_QPA_FONTDIR", "C:/Windows/Fonts")
pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication, QHBoxLayout, QLineEdit, QPushButton, QWidget  # noqa: E402

from astro_gpu.client import i18n, prefs  # noqa: E402
from astro_gpu.client.blocks import Graph, node_title  # noqa: E402
from astro_gpu.client.graph import GraphPage  # noqa: E402
from astro_gpu.client.model import ProjectModel  # noqa: E402
from astro_gpu.client.services.worker import ApiWorker  # noqa: E402
from astro_gpu.client.widgets import clear  # noqa: E402
from astro_gpu.core import project2  # noqa: E402
from astro_gpu.core import workflow as wfl  # noqa: E402
from astro_gpu.core.project import Target  # noqa: E402

CHAIN = ["calibration", "registration", "stacking", "astrometry"]


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def frames(folder: Path, names) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    for i, n in enumerate(names):
        (folder / n).write_bytes(b"\0" * (100 + i))
    return folder


def open_page(tmp_path, monkeypatch, filters=(), workflows=("Full",)):
    """One colour session and a template's workflow; or (``filters``) one session per filter, each its own flow."""
    monkeypatch.setenv("ASTRO_GPU_SETTINGS", str(tmp_path / "settings.json"))
    prefs.load()
    i18n.set_language("en")
    p = project2.create(tmp_path / "P", "P", Target(name="M 31"))
    if filters:
        for i, f in enumerate(filters, 1):
            p.sessions.append(project2.Session(i, f, lights=[str(frames(tmp_path / "raw" / f, ["a1.fits"]))],
                                               filter=f))
        p.stack_mode = "per_session"
    else:
        p.sessions.append(project2.Session(1, "night 1", lights=[str(frames(tmp_path / "raw" / "n1", ["a1.ARW"]))]))
        for name in workflows:
            p.add_workflow(name, wfl.TEMPLATES[name])
    project2.save(p)
    w = ApiWorker()
    m = ProjectModel(p, w)
    assert w.wait(120)
    m._sync()
    if filters:
        for i, f in enumerate(filters, 1):
            m.add_graph_workflow(f, [wfl.node(b) for b in CHAIN], sessions=[i])
        m._undo.clear()
    page = GraphPage(m)
    page.resize(1500, 800)
    page.toasts = []
    page.toast = page.toasts.append
    return w, m, page


def settle(w, page):
    w.wait(60)
    QApplication.processEvents()
    page.refresh()


def done(w, m, page):
    page.close()
    m.close(save=False)
    w.stop()


def blocks_of(m, name):
    return [n["block"] for n in m.graph(next(q for q in m.paths if q.name == name))]


def test_a_second_session_joins_the_chosen_flow_never_in_place_of_its_nights(qapp, tmp_path, monkeypatch):
    """Decision 1: double-click on the Session card (or a drop on "+ Session") adds the night to the chosen flow: the
    first night stays read, it says so; Ctrl+Z takes it back."""
    w, m, page = open_page(tmp_path, monkeypatch)
    try:
        full = m.paths[0]
        page.select(index=0, step="sources")
        page.palette_action("session")
        settle(w, page)
        assert [s.id for s in m.p.sessions_of(full.pid)] == [1, 2] and len(m.paths) == 1
        page.palette_action("session")                             # a third night: still all three
        settle(w, page)
        assert [s.id for s in m.p.sessions_of(full.pid)] == [1, 2, 3]
        m.undo()
        m.undo()
        assert [s.id for s in m.p.sessions] == [1]
    finally:
        done(w, m, page)




def fits_of(folder: Path, flt: str) -> Path:
    import numpy as np
    from astropy.io import fits
    folder.mkdir(parents=True, exist_ok=True)
    fits.PrimaryHDU(np.zeros((8, 8), np.uint16), header=fits.Header({"FILTER": flt})).writeto(folder / "l1.fits")
    return folder / "l1.fits"


@pytest.mark.parametrize("filters", [("Ha",), ("Ha", "OIII")])
def test_a_session_added_to_a_filter_flow_follows_the_filter_of_its_frames(qapp, tmp_path, monkeypatch, filters):
    """The night test pass: a second session added to the H-alpha flow, given OIII frames, kept the H-alpha filter, and
    the two filters would have been stacked together. Its frames' filter wins: it goes to the OIII flow (made when
    there is none, with the H-alpha flow's blocks)."""
    w, m, page = open_page(tmp_path, monkeypatch, filters=filters)
    try:
        ha = next(q for q in m.paths if q.name == "Ha")
        sid = m.new_session(ha.pid)
        assert m.p.session(sid).filter == "Ha"                                  # joined Ha: took its filter
        m.add_session_entries(sid, "lights", [str(fits_of(tmp_path / "oiii2", "OIII"))])
        settle(w, page)
        assert m.p.session(sid).filter == "OIII"
        assert m.sessions_of(m.view(ha.pid)) == [1]
        oiii = [q for q in m.paths if q.source is None and sid in m.sessions_of(q)]
        assert len(oiii) == 1 and blocks_of(m, oiii[0].name) == blocks_of(m, "Ha")
        assert sorted(m.filters()) == ["Ha", "OIII"]
    finally:
        done(w, m, page)




def test_one_image_sent_to_post_from_a_block_is_not_numbered(qapp, tmp_path, monkeypatch):
    """Round 4: a block of an Image's chain sent to Post (a branch with the same source) made the only Image block
    "Image 1"; it shows its workflow's name, not the branch's. A second Image file: each block its workflow's name, which
    carries the number (#155), a name given as it is."""
    w, m, page = open_page(tmp_path, monkeypatch, workflows=())
    try:
        tif = frames(tmp_path / "img", ["a.tif", "b.tif"])
        src = {"path": (tif / "a.tif").as_posix(), "stretched": False}
        p = m.add_graph_workflow("Image", [wfl.node("astrometry"), wfl.node("gradient"),
                                           wfl.node("gradient", name="gradient~2")], source=src)
        settle(w, page)
        m.send_to_post(m.view(p.pid), "fond", "Image · Gradient")
        settle(w, page)
        g = Graph(m)
        images = {id(g.node_of(q, "image")): g.node_of(q, "image") for q in m.paths}
        assert len(images) == 1 and [node_title(m, n) for n in images.values()] == ["Image"]
        m.add_graph_workflow("Other", [wfl.node("astrometry")],
                             source={"path": (tif / "b.tif").as_posix(), "stretched": False})
        settle(w, page)
        g = Graph(m)
        images = {id(g.node_of(q, "image")): g.node_of(q, "image") for q in m.paths}
        assert sorted(node_title(m, n) for n in images.values()) == ["Image", "Other"]
    finally:
        done(w, m, page)


def test_a_stars_block_after_a_combination_is_drawn_after_it(qapp, tmp_path, monkeypatch):
    """One block, one function: a combination has no stars of its own; a Stars/background placed after it takes them
    off the combined image, drawn after its block (its Combine layers in Post)."""
    w, m, page = open_page(tmp_path, monkeypatch, filters=("Ha", "OIII"))
    try:
        chans = [{"pid": q.pid, "role": q.name, "channel": "L"} for q in m.paths]
        f = m.add_fusion("H-O", "ho", chans)
        settle(w, page)
        fv = m.view(f.pid)
        assert m.place_indices(fv, "stars") and not m.palette_reason(fv, "stars")
        page.place_block("stars", f.pid)
        settle(w, page)
        fv = m.view(f.pid)
        assert m.display_keys(fv) == ["fusion", "etoiles"] and m.all_stages(fv)[-1] == "star_removal"
        g = Graph(m)
        assert g.node_of(fv, "etoiles").parent is g.node_of(fv, "fusion")
    finally:
        done(w, m, page)




def test_two_images_without_a_file_are_two_blocks(qapp, tmp_path, monkeypatch):
    """Decision 4: each Image placed is its own block, removed alone."""
    w, m, page = open_page(tmp_path, monkeypatch)
    try:
        page.palette_action("image")
        page.palette_action("image")
        settle(w, page)
        imgs = [q for q in m.paths if q.source is not None]
        g = Graph(m)
        nodes = [g.node_of(q, "image") for q in imgs]
        assert len(imgs) == 2 and nodes[0] is not nodes[1]
        page.remove_tool(nodes[0], None)
        settle(w, page)
        assert [q.pid for q in m.paths if q.source is not None] == [imgs[1].pid]
    finally:
        done(w, m, page)


def test_names_never_empty_nor_another_s(qapp, tmp_path, monkeypatch):
    """Decision 8: a session or a workflow cannot take an empty name or another's: refused with a short message."""
    w, m, page = open_page(tmp_path, monkeypatch, filters=("Ha", "OIII"))
    try:
        ha, oiii = (next(q for q in m.paths if q.name == n) for n in ("Ha", "OIII"))
        assert m.name_problem("OIII", sid=1) and m.name_problem("  ", sid=1) and not m.name_problem("Ha 2", sid=1)
        assert not m.rename_session(1, "OIII") and m.p.session(1).name == "Ha"
        page.rename(ha.pid, "OIII")
        page.rename(ha.pid, "")
        assert m.view(ha.pid).name == "Ha" and len(page.toasts) == 2 and "refused" in page.toasts[0]
        assert not m.rename(ha.pid, "OIII") and m.rename(ha.pid, "H-alpha") and m.view(ha.pid).name == "H-alpha"
        page.select(index=m.paths.index(m.view(oiii.pid)), step="session@2")
        page.refresh()
        ed = next(e for e in page.panel.findChildren(QLineEdit) if e.text() == "OIII")
        ed.setText("Ha")
        ed.editingFinished.emit()
        assert m.p.session(2).name == "OIII" and ed.text() == "OIII" and "refused" in page.toasts[-1]
    finally:
        done(w, m, page)


def test_removing_a_flow_or_its_session_says_what_goes(qapp, tmp_path, monkeypatch):
    """Two flows: the Session removed takes its flow, and says so; Delete workflow takes the Session only it read (no
    Session left unwired, its frames said used); Ctrl+Z brings both back."""
    w, m, page = open_page(tmp_path, monkeypatch, filters=("Ha", "OIII"))
    try:
        page.session_action("remove", 1)
        settle(w, page)
        assert [q.name for q in m.paths] == ["OIII"] and "workflow" in page.toasts[-1] and "Ha" in page.toasts[-1]
        m.undo()
        settle(w, page)
        oiii = next(q for q in m.paths if q.name == "OIII")
        page.workflow_action("delete", oiii.pid)
        settle(w, page)
        assert [s.id for s in m.p.sessions] == [1] and "OIII" in page.toasts[-1]
        m.undo()
        assert [s.id for s in m.p.sessions] == [1, 2] and len(m.paths) == 2
    finally:
        done(w, m, page)


def test_a_session_calibration_shows_its_scope_and_has_its_buttons(qapp, tmp_path, monkeypatch):
    """One Calibration for every session: switched off, every session's block looks so and the message says "all
    sessions"; its panel has Delete and Switch off; a pause that cannot be kept is never announced. The Session panel
    has its Delete (greyed with the reason when it cannot go)."""
    w, m, page = open_page(tmp_path, monkeypatch)
    try:
        page.select(index=0, step="sources")
        page.palette_action("session")
        settle(w, page)
        g = Graph(m)
        c1 = g.node_of(m.paths[0], "calibration@1")
        page.select_node(c1)
        page.refresh()
        texts = [b.text() for b in page.panel.findChildren(QPushButton)]
        assert "Delete" in texts and "Switch off" in texts
        page.toggle_cut(c1, None)
        settle(w, page)
        g = Graph(m)
        assert all(cn.cut for _s, cn in g.sessions) and "all sessions" in page.toasts[-1]
        page.toggle_pause(g.sessions[0][1])                      # announced only when it is kept, else said why
        kept = bool(m.p.workflow(m.paths[0].pid).breakpoints)
        assert ("Review pause" in page.toasts[-1]) == kept and (kept or "Registration" in page.toasts[-1])
        page.select_node(g.sessions[1][0])
        page.refresh()
        x = next(b for b in page.panel.findChildren(QPushButton) if b.text() == "Delete")
        assert x.isEnabled()
    finally:
        done(w, m, page)


def test_a_field_going_away_writes_nothing(qapp):
    """A panel's field hidden as its block goes (Delete, Ctrl+Z) does not say "editing finished" any more: it would
    write into a block, session or workflow that is gone."""
    host = QWidget()
    lay = QHBoxLayout(host)
    box = QWidget()
    inner = QHBoxLayout(box)
    ed, ed2 = QLineEdit("x"), QLineEdit("y")
    inner.addWidget(ed2)
    lay.addWidget(ed)
    lay.addWidget(box)
    host.show()
    ed.setFocus()
    calls = []
    ed.editingFinished.connect(lambda: calls.append(1))
    ed2.editingFinished.connect(lambda: calls.append(2))
    clear(lay)
    QApplication.processEvents()
    assert calls == []
    host.close()


def test_a_filter_s_second_night_is_named_and_drawn_with_its_filter(qapp, tmp_path, monkeypatch):
    """GUI exploration 4: a second night joining Ha is "Ha 2" (the filter's other night shows it), drawn with the Ha
    sessions, level with the Ha chain, not under OIII; each session level with its chain (also in big projects); the
    sessions' bars of two filters never run into each other."""
    w, m, page = open_page(tmp_path, monkeypatch, filters=("Ha", "OIII"))
    try:
        ha, oiii = (next(q for q in m.paths if q.name == n) for n in ("Ha", "OIII"))
        page.refresh()
        lanes, g = page.graph._lanes, page.graph.graph
        assert {sn.sid: sn.y for sn, _c in g.sessions} == {1: lanes[ha.pid], 2: lanes[oiii.pid]}
        for q in (ha, oiii):
            page.select(index=m.paths.index(m.view(q.pid)), step=f"session@{m.sessions_of(m.view(q.pid))[0]}")
            page.palette_action("session")
            settle(w, page)
        assert [(s.name, s.filter) for s in m.p.sessions] == [("Ha", "Ha"), ("OIII", "OIII"), ("Ha 2", "Ha"),
                                                               ("OIII 2", "OIII")]
        assert page.toasts[0].startswith("“Ha 2” joins “Ha”")
        page.refresh()
        lanes, g = page.graph._lanes, page.graph.graph
        y = {sn.sid: sn.y for sn, _c in g.sessions}
        assert y[1] < y[3] < y[2] < y[4]                                          # Ha, Ha 2, then OIII, OIII 2
        assert (y[1] + y[3]) / 2 == lanes[ha.pid] and (y[2] + y[4]) / 2 == lanes[oiii.pid]
        assert max(y[3], lanes[ha.pid]) < min(y[2], lanes[oiii.pid])             # two bars, apart
    finally:
        done(w, m, page)


def test_a_new_session_s_name_is_never_another_s(qapp, tmp_path, monkeypatch):
    """GUI exploration 4: Session 1 renamed "Session 2" by the user: the next session is not "Session 2" too."""
    w, m, page = open_page(tmp_path, monkeypatch)
    try:
        assert m.rename_session(1, "Session 2")
        sid = m.add_filter_session("")
        assert sid == 2 and m.p.session(sid).name == "Session 3"
        sid = m.add_session()
        names = [s.name for s in m.p.sessions]
        assert len(set(names)) == len(names) == 3
    finally:
        done(w, m, page)


def test_the_deconvolution_and_synthetic_stars_cards_get_their_durations(qapp, tmp_path, monkeypatch):
    """The night test pass: under each method of these two blocks, "…" for good: the worker never estimated them."""
    w, m, page = open_page(tmp_path, monkeypatch)
    try:
        q = m.paths[0]
        m._prefetch_estimates(False)
        for block in ("deconv", "synth"):
            for meth in ("psf", "neural") if block == "deconv" else ("measured", "instrument"):
                got = m.method_times(q, block, meth)
                assert got is not None and set(got) <= {"gpu", "cpu"} and got, (block, meth, got)
    finally:
        done(w, m, page)
