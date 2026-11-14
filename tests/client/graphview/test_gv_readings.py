"""The table's readings are what the current software reads from the frames' headers (services/inspect.project_frames,
as the Sources page shows them), read off the GUI thread by the project model's service: the lights' exposure, gain
and temperature, the darks' exposure and their matching with the lights, the night's date, the time the lights add up
to, « reading… » while they are read, and lights of several exposures. A graph held in memory shows none of them.

Small FITS files on a temporary project; the folders' and files' names say nothing the table could use. The same
project model also checks the page's glue with the window (today's page's methods and signals, on the model)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from astro_gpu.client.graphmodel import GraphEditing
from astro_gpu.client.model import ProjectModel


M = ProjectModel                      # the graph is its truth (GraphEditing)


def _fits(path: Path, **hdr) -> Path:
    from astropy.io import fits
    path.parent.mkdir(parents=True, exist_ok=True)
    h = fits.PrimaryHDU(np.full((8, 12), 900, np.uint16))
    for k, v in hdr.items():
        h.header[k] = v
    h.writeto(path, overwrite=True)
    return path


def _project(root: Path) -> Path:
    from astro_gpu.core import project2
    from astro_gpu.core.project import Target
    p = project2.create(root / "P", "P", Target(name="M 42"))
    a, b, d = root / "x1", root / "x2", root / "x3"
    for i in range(3):
        _fits(a / f"f{i}.fits", EXPTIME=300.0, GAIN=90, **{"CCD-TEMP": -10.0, "DATE-OBS": "2026-11-06T22:00:00"})
    for i in range(2):
        _fits(d / f"f{i}.fits", EXPTIME=60.0, GAIN=90, **{"DATE-OBS": "2026-11-06T08:00:00"})
    for i, t in enumerate((300.0, 300.0, 120.0)):
        _fits(b / f"f{i}.fits", EXPTIME=t, GAIN=90, **{"DATE-OBS": "2026-11-07T23:00:00"})
    p.sessions.append(project2.Session(1, "One", lights=[str(a)], darks=[str(d)]))
    p.sessions.append(project2.Session(2, "Two", lights=[str(b)]))
    project2.save(p)
    return root / "P"


def _colour_min():
    from astro_gpu.client.gallery import GALLERY
    return next(t for t in GALLERY if t["key"] == "colour_min")


def _wait(page, model=None, timeout=60.0):
    from astro_gpu.client.graphview.gestures import pump
    if model is not None:
        from astro_gpu.client.services.inspect import service_of
        assert service_of(model).wait(timeout)
    grid = page.table_host.table.grid
    grid.reader.wait(timeout)
    for _i in range(20):
        pump()
        if model is None or not grid.model.headers_pending():
            break
        service_of(model).wait(timeout)
    return grid


@pytest.fixture(scope="module")
def model(qapp, tmp_path_factory):
    """One project model for the module (opening one is the costly part): read-only, nothing written."""
    from astro_gpu.client.services.worker import ApiWorker
    from astro_gpu.core import project2
    w = ApiWorker()
    m = M(project2.load(_project(tmp_path_factory.mktemp("readings"))), w, readonly=True)
    assert w.wait(120) and m.ready, m.error
    yield m
    m.close(save=False)
    w.stop()


def _page(model):
    """The page on the model, with one stack of its two sessions (the gallery's Colour minimal, once)."""
    from astro_gpu.client.graphview.gestures import pump
    from astro_gpu.client.graphview.page import GraphPage
    page = GraphPage(model, None)
    page.resize(1500, 800)
    page.show()
    pump()
    if not page.bridge.graph.rows:
        page.template_chosen(_colour_min())
        pump()
    return page


def _sessions(grid):
    pl = grid.model.stacks()[0]
    return {s["name"]: s["sid"] for s in pl["sessions"]}, pl


def test_the_table_shows_what_the_headers_say(model):
    page = _page(model)
    page.set_display("table")
    grid = _wait(page, model)
    mp = grid.model
    by, pl = _sessions(grid)
    one, two = by["One"], by["Two"]
    assert mp.box(one, "lights").l2 == "300 s · Gain 90 · −10 °C"
    assert mp.box(one, "darks").l2 == "60 s · Gain 90"
    assert mp.date_text(one) == "6 Nov 2026 · 15 min"            # the night it began, 3 x 300 s
    assert mp.date_text(two) == "7 Nov 2026"                       # several exposures: no sum said
    assert mp.seconds(pl) is None
    texts = [w.text for w in mp.session_warnings(one)]
    assert "Darks: exposure 60 s, Gain 90, different from the lights." in texts, texts
    mixed = [w.text for w in mp.session_warnings(two) if w.kind == "lights"]
    assert mixed and mixed[0].startswith("This session mixes lights of 300 s and 120 s"), mixed
    assert not mp.headers_pending()
    page.close()


def test_reading_is_said_while_the_headers_are_read(model, monkeypatch):
    from astro_gpu.client.graphview.gestures import pump
    from astro_gpu.client.services.inspect import service_of
    asked = []
    monkeypatch.setattr(service_of(model), "request_job", lambda *a: asked.append(a) or -1)   # its answer never comes
    page = _page(model)
    page.set_display("table")                           # (the model keeps the display: the table may be shown at once)
    pump()
    grid = page.table_host.table.grid
    grid.reader.wait(30)
    pump()
    mp = grid.model
    mp.invalidate()
    by, _pl = _sessions(grid)
    assert asked and asked[0][0] == "graphview.headers"
    assert mp.headers_pending(by["One"])
    assert mp.box(by["One"], "lights").l2 == "reading…"
    assert mp.date_text(by["One"]) == "reading…"
    page.table_host.table._update_bars()
    assert page.table_host.table.counter.text() == "Reading the frames…"
    page.close()


def test_a_graph_in_memory_shows_no_header_reading(qapp, tmp_path):
    from astro_gpu.client.graphview.document import open_project
    from astro_gpu.client.graphview.page import GraphPage
    page = GraphPage(open_project(_project(tmp_path)))
    page.resize(1500, 800)
    page.show()
    page.template_chosen(_colour_min())
    page.set_display("table")
    grid = _wait(page)
    mp = grid.model
    by, pl = _sessions(grid)
    assert mp.box(by["One"], "lights").l1 == "3 lights"           # the listing's count
    assert mp.box(by["One"], "lights").l2 == "" and mp.box(by["One"], "darks").l2 == ""
    assert mp.date_text(by["One"]) == "" and mp.seconds(pl) is None
    assert not [w for w in mp.session_warnings(by["Two"]) if w.key[0] in ("mixed", "darks", "nodark")]
    assert not grid.headers.active and not mp.headers_pending()
    page.close()


def test_the_page_on_the_project_model_answers_as_todays_page(model):
    """GraphPage(model, steps): the page's selection is the model's (Inspect, the panel, the run read it), the window's
    select() chooses the block on the page, Enter opens it, a variant made here is told with its original."""
    from astro_gpu.client.graphview.gestures import Gestures, pump
    page = _page(model)
    page.set_display("blocks")
    pump()
    b, bm = page.bridge, model.block_map
    stacking = next(x for x, blk in b.graph.blocks.items() if blk.kind == "stacking")
    Gestures(page).click(stacking)
    wid, key = bm.address(stacking)
    assert model.path is not None and model.path.pid == wid and model.sel_step == key
    node = page.current_node()
    assert node is bm.node(stacking) and page.plus_session_pid == wid
    b.choose(None)
    page.select(index=[q.pid for q in model.paths].index(wid), step=key)
    assert b.selected == stacking
    opened, variants = [], []
    page.openBlock.connect(lambda: opened.append(1))
    page.variantMade.connect(lambda a, o: variants.append((a, o)))
    Gestures(page).key("Return", target=page.canvas)
    assert opened
    gradient = next(x for x, blk in b.graph.blocks.items() if blk.kind == "gradient")
    assert b.act("add_variant", block=gradient).ok
    pump()
    assert variants and variants[0][1] == wid and variants[0][0] != wid
    b.undo()
    b.choose("project")
    assert page.project_open() and model.sel_step == ""
    page.close()
