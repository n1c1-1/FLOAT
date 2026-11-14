"""Opening a project never waits for the disk nor for the engine's stages in the GUI thread (the old model's rule): a
project of the previous format opened before the engine's rules are loaded shows its nights until the worker has warmed
the API, then the graph converted from its definition, the same as a conversion made at once; an edit before that
finishes the conversion first. Read-only on the test project."""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication  # noqa: E402

from astro_gpu import api  # noqa: E402
from astro_gpu.client import graphmodel  # noqa: E402
from astro_gpu.client.model import ProjectModel  # noqa: E402
from astro_gpu.client.services.worker import ApiWorker  # noqa: E402
from astro_gpu.graphcore import model as gm  # noqa: E402
from conftest import TEST_PROJECT, full_only, needs_test_project  # noqa: E402  (tests/conftest.py)

pytestmark = needs_test_project
ROOT = Path(__file__).resolve().parents[2]
FOUR = (("Classic stack", "Classic stack", {"render_simple": {"linked": False}}),
        ("Classic + denoise", "Classic + denoise", None),
        ("Full twin", "Full", None))


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _open_four():
    p = full_only(api.open_project(TEST_PROJECT))
    for name, template, params in FOUR:
        api.add_workflow(p, name, template, params)
    return p


def _shape(g):
    return [(r.name, [g.blocks[b].kind for b in r.blocks], r.fork is not None) for r in g.rows], \
        sorted(t.name for t in g.taps)


def test_real_a_project_opened_cold_shows_its_nights_then_the_same_graph_as_a_conversion_at_once(qapp, monkeypatch):
    w = ApiWorker()
    now = ProjectModel(_open_four(), w, readonly=True)           # the engine's rules loaded: converted at once
    assert w.wait(180) and now.ready
    want, want_def = _shape(now._g), [w_.name for w_ in now.p.workflows]
    now.close(save=False)
    monkeypatch.setattr(graphmodel.GraphEditing, "_rules_loaded", staticmethod(lambda: False))
    p = _open_four()
    before = p.to_dict()
    m = ProjectModel(p, w, readonly=True)
    g = m.document.graph                                         # what the page draws meanwhile: the nights
    assert m._provisional and not g.rows and [n["sid"] for n in g.nights] == [s.id for s in p.sessions]
    assert p.to_dict()["workflows"] == before["workflows"]       # the definition untouched meanwhile
    assert [v.name for v in m.paths] == ["Full", "Classic stack", "Classic + denoise", "Full twin"]
    assert m.block_map is not None and m._provisional            # the block map drawn as it is
    assert w.wait(180) and m.ready
    assert not m._provisional and _shape(m._g) == want and [x.name for x in m.p.workflows] == want_def
    assert all(not os.path.isdir(f) for s in m.p.sessions for k in gm.FILE_KINDS for f in getattr(s, k))
    m.close(save=False)
    # an edit before the worker answered: the conversion is finished first, then the edit made on it
    m = ProjectModel(_open_four(), w, readonly=True)
    assert m._provisional
    m.rename(m.paths[1].pid, "Wide")
    assert not m._provisional and _shape(m._g)[0][1][0] == "Wide" and m.p.workflows[1].name == "Wide"
    assert w.wait(180) and m.ready and _shape(m._g)[0][1][0] == "Wide"
    m.close(save=False)
    w.stop()


def test_real_the_model_opens_in_under_a_second_in_a_fresh_process():
    """The old model's ``< 1 s``: a fresh process (the stages never imported), the test project and three workflows."""
    code = textwrap.dedent(f"""
        import os, sys, tempfile, time
        os.environ["ASTRO_GPU_HOME"] = tempfile.mkdtemp(prefix="opening_home_")
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
        sys.path[:0] = [{str(ROOT)!r}, {str(ROOT / "tests")!r}]
        from PySide6.QtWidgets import QApplication
        app = QApplication([])
        from astro_gpu import api
        from astro_gpu.client.model import ProjectModel
        from astro_gpu.client.services.worker import ApiWorker
        from conftest import TEST_PROJECT, full_only
        p = full_only(api.open_project(TEST_PROJECT))
        for name, template, params in {FOUR!r}:
            api.add_workflow(p, name, template, params)
        w = ApiWorker()
        t0 = time.perf_counter()
        m = ProjectModel(p, w, readonly=True)
        dt = time.perf_counter() - t0
        ok = w.wait(180) and m.ready and len(m._g.rows) > 0
        print("TIME", dt, ok)
        m.close(save=False)
        w.stop()
    """)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300, cwd=str(ROOT))
    line = next((x for x in r.stdout.splitlines() if x.startswith("TIME")), None)
    assert line is not None, r.stderr[-2000:]
    _t, dt, ok = line.split()
    assert ok == "True" and float(dt) < 1.0, line
