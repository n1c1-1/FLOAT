"""« Add session » on a workflow whose first session has a name of its own (a tutorial's: its night's date) made
"Session 1", the project's second session. A new session is "Session n", n its number in the project, in English and
in French; its lights dropped on it change no name, and a name the user typed stays."""
from __future__ import annotations

import numpy as np

from astro_gpu.graphcore.model import Index
from tests.client.graphview.test_gv_files import settle as read
from tests.client.pagehelp import done, open_page, qapp, settle  # noqa: F401


def lights(folder, day, n=2):
    from astropy.io import fits
    folder.mkdir(parents=True)
    out = []
    for i in range(n):
        h = fits.PrimaryHDU(np.zeros((8, 8), np.uint16))
        h.header["DATE-OBS"] = f"{day}T22:0{i}:00"
        h.writeto(folder / f"L_{i}.fits")
        out.append((folder / f"L_{i}.fits").as_posix())
    return out


def add_session(page):
    """The stack's menu, « Add session »: its item run as a click on it does. -> the new Session block."""
    from astro_gpu.client.graphview import menu
    from astro_gpu.client.i18n import _
    g = page.bridge.graph
    rid = next(r.id for r in g.rows if r.nights)
    it = next(x for x in menu.row_items(page.bridge.doc, Index(g), rid) if x[0] == _("Add session"))
    assert it[3], it
    res = page.bridge.act(it[1], **it[2])
    assert res.ok
    return next(r for r in page.bridge.graph.rows if r.id == rid).nights[-1][0]


def names(m, page):
    """The sessions' names: the project's, and as the page's Session blocks are titled."""
    g = page.bridge.graph
    blocks = [page.bridge.doc.title(s) for r in g.rows for s, _c in r.nights]
    return [s.name for s in m.p.sessions], blocks


def check(qapp, tmp_path, monkeypatch, language):
    from astro_gpu.client import i18n
    w, m, page = open_page(tmp_path, monkeypatch)             # one session, "night 1", in a template's workflow
    try:
        i18n.set_language(language)
        s2 = add_session(page)
        settle(w, page)
        assert names(m, page) == (["night 1", "Session 2"], ["night 1", "Session 2"])
        page.bridge.drops.drop(s2, "lights", lights(tmp_path / "raw" / "n2", "2024-12-30"))
        read(page.bridge.drops.reader)
        settle(w, page)
        assert len(m.p.session(2).lights) == 2
        assert names(m, page) == (["night 1", "Session 2"], ["night 1", "Session 2"])       # its lights: no new name
        s3 = add_session(page)
        assert page.bridge.act("rename_night", night=s3, name="Moonless").ok                # typed by the user
        page.bridge.drops.drop(s3, "lights", lights(tmp_path / "raw" / "n3", "2024-12-31"))
        read(page.bridge.drops.reader)
        settle(w, page)
        assert names(m, page) == (["night 1", "Session 2", "Moonless"], ["night 1", "Session 2", "Moonless"])
        add_session(page)
        settle(w, page)
        assert names(m, page)[0][-1] == "Session 4"
    finally:
        i18n.set_language("en")
        done(w, m, page)


def test_gv_an_added_session_has_its_number_in_the_project(qapp, tmp_path, monkeypatch):
    check(qapp, tmp_path, monkeypatch, "en")


def test_gv_an_added_session_has_its_number_in_the_project_in_french(qapp, tmp_path, monkeypatch):
    check(qapp, tmp_path, monkeypatch, "fr")
