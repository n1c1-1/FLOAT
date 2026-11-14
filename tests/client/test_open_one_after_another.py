"""Projects opened one after another in one process, as the owner opens his in the running application: a project of
the previous format (one with a session, one made before its first session), one of the graph's format and a blank one,
in every order, through the project model and its graph page, then through the main window: each opens, its graph is
drawn, nothing is logged as an error and no message box is shown. (Seen: a stack of the previous format with no night
yet, opened once the engine was warm, failed in the layout: a frame column of a band said closed.)"""
from __future__ import annotations

import itertools
import logging

import pytest
from PySide6.QtWidgets import QApplication

from astro_gpu import api
from astro_gpu.client.graphview.page import GraphPage
from astro_gpu.client.model import ProjectModel
from astro_gpu.client.services.worker import ApiWorker
from astro_gpu.core import project2
from astro_gpu.core.project import Target
from astro_gpu.graphcore.layout import layout
from tests.client.graphview.legacy import frames
from tests.client.test_swap_smoke import env, qapp, ready, settle, win  # noqa: F401  (the window's fixtures)


def _projects(folder):
    """{kind: project folder}: previous format with a session, previous format before its first session, the graph's
    format, blank."""
    out = {}
    p = project2.create(folder / "old", "Old", Target(name="M 31"))
    p.sessions.append(project2.Session(1, "night 1", lights=[str(frames(folder / "raw", ["a1.ARW", "a2.ARW"]))]))
    api.add_workflow(p, "Full", "Full", None)
    api.add_workflow(p, "Classic stack", "Classic stack", None)
    project2.save(p)
    out["old"] = folder / "old"
    p = project2.create(folder / "old_no_night", "Old, no night", Target(name="M 42"))
    api.add_workflow(p, "Full", "Full", None)
    project2.save(p)
    out["old_no_night"] = folder / "old_no_night"
    project2.save(project2.create(folder / "blank", "Blank", Target(name="M 45")))
    out["blank"] = folder / "blank"
    # the graph's format: the first one opened by the model and saved after an edit
    import shutil
    shutil.copytree(folder / "old", folder / "new")
    w = ApiWorker()
    m = ProjectModel(project2.load(folder / "new"), w)
    assert w.wait(120) and m.ready
    m.rename(m.paths[0].pid, "Wide")
    m.flush()
    m.close(save=False)
    w.stop()
    out["new"] = folder / "new"
    assert project2.load(folder / "new").graph is not None
    return out


class _Errors(logging.Handler):
    def __init__(self):
        super().__init__(logging.ERROR)
        self.got = []

    def emit(self, record):
        self.got.append(self.format(record))


@pytest.fixture(scope="module")
def projects(tmp_path_factory, qapp):  # noqa: F811
    return _projects(tmp_path_factory.mktemp("one_after_another"))


ORDERS = list(itertools.permutations(("old", "new", "blank", "old_no_night"), 3))


def test_every_stack_of_the_previous_format_is_laid_out_even_before_its_first_night():
    from astro_gpu.graphcore.convert import convert
    from astro_gpu.graphcore.operations import new_view
    import tempfile
    from pathlib import Path
    p = project2.create(Path(tempfile.mkdtemp(prefix="nonight_")) / "P", "P", Target(name="M 31"))
    api.add_workflow(p, "Full", "Full", None)
    g = convert(p)
    assert g.rows and not g.rows[0].nights
    for folded in (False, True):
        v = new_view()
        v["folded"] = folded
        d = layout(g, v)
        assert all(b in d.boxes for b in g.rows[0].blocks)


@pytest.mark.parametrize("order", ORDERS, ids=["-".join(o) for o in ORDERS])
def test_projects_opened_one_after_another_on_the_model_and_its_page(qapp, projects, order):  # noqa: F811
    errors = _Errors()
    logging.getLogger("astro_gpu").addHandler(errors)
    try:
        w = ApiWorker()
        for kind in order:
            m = ProjectModel(project2.load(projects[kind]), w, readonly=True)
            page = GraphPage(m)
            page.resize(1400, 800)
            page.show()
            assert w.wait(120) and m.ready, (kind, m.error)
            for _i in range(4):
                QApplication.processEvents()
            page.refresh()
            QApplication.processEvents()
            d = page.bridge.drawing()
            assert all(b in d.boxes for r in m.stored.rows for b in r.blocks), kind
            page.close()
            page.deleteLater()
            m.close(save=False)
            QApplication.processEvents()
        w.stop()
    finally:
        logging.getLogger("astro_gpu").removeHandler(errors)
    assert not errors.got, errors.got


@pytest.mark.parametrize("order", [("new", "old_no_night", "old", "blank"), ("blank", "old", "new", "old_no_night"),
                                   ("old_no_night", "blank", "new", "old")])
def test_projects_opened_one_after_another_in_the_window(win, env, projects, order):  # noqa: F811
    win.readonly = True
    try:
        for kind in order:
            assert win.open_project(projects[kind]), kind
            ready(win)
            assert isinstance(win.graph, GraphPage), kind
            win.graph.refresh()
            settle(win, 3)
            d = win.graph.bridge.drawing()
            assert all(b in d.boxes for r in win.s.stored.rows for b in r.blocks), kind
        assert win.close_project(go_home=False, discard=True)
    finally:
        win.readonly = False
    assert not env["errors"], env["errors"]
