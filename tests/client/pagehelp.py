"""What the client tests take from the graph page (graphview): a page opened on a scratch project, a block's item found
by its key, kind or title, a click or a double-click on it, its selection, the project model's block map.

The page draws one ``BlockItem`` per block of the stored graph (``page.canvas.blocks``: {block id: item}); each item
carries its block's data (``item.d``: kind, title, summary, state, tip...) and its run node (``item.node``, a
``graphnode.Node``: ``key``, ``paths``, ``sid``). The model's ``block_map`` gives the node of a workflow's view key
(``node_of(q, "empilement")``) and the block id of a node (``block_of``)."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if Path("C:/Windows/Fonts").is_dir():
    os.environ.setdefault("QT_QPA_FONTDIR", "C:/Windows/Fonts")
pytest.importorskip("PySide6")
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from astro_gpu.client import i18n, prefs  # noqa: E402
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


def watch_toasts(page) -> list:
    """The toasts the page shows, in order (``page.toasts``)."""
    page.toasts = []
    page.toasted.connect(page.toasts.append)
    return page.toasts


def open_page(tmp_path, monkeypatch, filters=(), workflows=("Full",)):
    """One colour session and a template's workflow; or (``filters``) one session per filter, each its own flow; the
    graph page on its project model. -> (worker, model, page); ``page.toasts`` lists the toasts shown."""
    from astro_gpu.client.graphview.page import GraphPage
    from astro_gpu.client.model import ProjectModel
    from astro_gpu.client.services.worker import ApiWorker
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
    watch_toasts(page)
    return w, m, page


def settle(w, page=None):
    w.wait(60)
    QApplication.processEvents()
    if page is not None:
        page.refresh()
        QApplication.processEvents()


def done(w, m, page):
    page.close()
    m.close(save=False)
    w.stop()


# ---------------------------------------------------------------------------------------------- blocks of the page
def drawn(page):
    """The page drawn as it is now (the canvas laid out again if the graph changed): {block id: item}."""
    if page.bridge.stale():
        page.bridge.view_changed()
    QApplication.processEvents()
    return page.canvas.blocks


def items(page, key=None, kind=None, title=None) -> list:
    """The block items of the page with this view key (``item.node.key``, "empilement", "calibration"...), graph kind
    (``item.d["kind"]``, "stacking", "session"...) or title, in the graph's order."""
    out = []
    for it in drawn(page).values():
        n = getattr(it, "node", None)
        if key is not None and (n is None or n.key != key):
            continue
        if kind is not None and it.d.get("kind") != kind:
            continue
        if title is not None and it.d.get("title") != title:
            continue
        out.append(it)
    return out


def item(page, key=None, kind=None, title=None, q=None):
    """The one block item of the page with this key / kind / title (of workflow ``q`` when given)."""
    its = items(page, key, kind, title)
    if q is not None:
        pid = getattr(q, "pid", q)
        its = [it for it in its if it.node is not None and any(p.pid == pid for p in it.node.paths)]
    assert its, f"no block {key or kind or title} on the page: {[(it.d['kind'], it.d['title']) for it in drawn(page).values()]}"
    return its[0]


def block_id(m, q, key) -> str | None:
    """The graph block id of a workflow's view key ("empilement", "calibration@1"), or None."""
    bm = m.block_map
    n = bm.node_of(q, key)
    return bm.block_of(n) if n is not None else None


def node(m, q, key):
    """The run node (graphnode ``Node``) of a workflow's view key, with its marks read now (broken, missing)."""
    bm = m.block_map
    bid = block_id(m, q, key)
    return bm.node(bid) if bid is not None else bm.node_of(q, key)


def marked(m) -> list:
    """Every block's node of the model's graph, with the marks that follow the disk read now (broken, missing)."""
    bm = m.block_map
    return [bm.node(b) for b in m.stored.blocks if bm.node(b) is not None]


def choose(page, ident):
    """Select a block (or row, or the Project block) as a click on it does."""
    page.bridge.choose(ident)
    QApplication.processEvents()


def click(page, it, double=False):
    """A real click (or double-click) of the mouse on a block item of the canvas."""
    c = page.canvas
    if not page.isVisible():
        page.show()
    QApplication.processEvents()
    c._show(it.id)
    pos = c.centre_of(it.id)
    if double:
        QTest.mouseDClick(c.viewport(), Qt.LeftButton, Qt.NoModifier, pos)
    else:
        QTest.mouseClick(c.viewport(), Qt.LeftButton, Qt.NoModifier, pos)
    QApplication.processEvents()
