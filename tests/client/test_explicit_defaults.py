"""The file says everything (the owner's rule, 14 Nov): every setting of a block is written, at the stage's default when
nobody changed it, and nothing else moves: a reviewed Selection stays reviewed, the compile adds no setting value of its
own, and the panel marks as changed only a value that is not the stage's default."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "graphcore"))
import test_golden  # noqa: E402
import test_h1_old_way as h1  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from astro_gpu.client import graphmodel  # noqa: E402
from astro_gpu.client.model import ProjectModel  # noqa: E402
from astro_gpu.client.services.worker import ApiWorker  # noqa: E402
from astro_gpu.core import project2  # noqa: E402
from astro_gpu.core.project2 import Project2  # noqa: E402
from astro_gpu.graphcore.compile import compile_into  # noqa: E402
from astro_gpu.graphcore.convert import convert  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def opened(d: dict, root: Path, explicit: bool, monkeypatch):
    monkeypatch.setattr(graphmodel, "EXPLICIT_DEFAULTS", explicit)
    w = ApiWorker()
    m = ProjectModel(Project2.from_dict(json.loads(json.dumps(d)), root), w, readonly=True)
    assert w.wait(120)
    return m, w


def test_a_reviewed_selection_stays_reviewed_once_every_default_is_written(qapp, tmp_path, monkeypatch):
    """The review mark of Selection is tied to the thresholds it applies: written at their defaults, the same mark."""
    rec = h1.template_recording(next(t for t in h1.GALLERY if t["key"] == "colour_full"), h1.FAKE)
    d = rec.states[0]["definition"]
    old, w1 = opened(d, tmp_path / "a", False, monkeypatch)
    new, w2 = opened(d, tmp_path / "b", True, monkeypatch)
    try:
        assert old.p.to_dict()["workflows"] != new.p.to_dict()["workflows"]       # the defaults are written
        stack = new.p.workflows[0].params["stack"]
        assert stack["max_fwhm_ratio"] == 2.0 and "min_transparency" not in stack   # an automatic field stays unset
        for a, b in zip(old.paths, new.paths):
            for key in ("selection", "empilement", "alignement", "crop"):
                assert old.review_key(a, key) == new.review_key(b, key), key
    finally:
        for m, w in ((old, w1), (new, w2)):
            m.close(save=False)
            w.stop()


@pytest.mark.parametrize("shape", list(test_golden.SHAPES))
def test_the_compile_adds_no_setting_of_its_own(shape):
    """Every setting of the compiled definition is one the graph stores: a block's, a stack's (its calibration, its
    Output), an Align on's, a workflow's leftovers. Structural values the compile writes itself: none."""
    from astro_gpu.client.graphdefaults import with_explicit_defaults
    p = test_golden.SHAPES[shape]()
    g, _n = with_explicit_defaults(convert(p), like=p)
    stored = set()

    def take(params):
        for st, kv in (params or {}).items():
            stored.update((st, k) for k in kv)
    for b in g.blocks.values():
        take(b.params)
    for f in list(g.rows) + list(g.taps):
        take(getattr(f, "calib", None))
        take(getattr(f, "extra", None))
        if getattr(f, "output", None):
            stored.update(("render_simple", k) for k in f.output)
        if getattr(f, "align", None):
            take(f.align.get("params"))
    q = Project2.from_dict(p.to_dict(), None)
    compile_into(g, q)
    made = {(st, k) for w in q.workflows for st, kv in w.params.items() for k in kv}
    structural = {(st, k) for st, k in made if k == "apply_masters" and st.startswith("calibrate@")}   # a night's
    assert made - stored - structural == set()                                         # Calibration switched off


def test_the_panel_marks_only_a_value_off_its_default(qapp, monkeypatch):
    from types import SimpleNamespace

    from astro_gpu.client.settings import user_changed
    monkeypatch.setattr(graphmodel, "EXPLICIT_DEFAULTS", True)
    at = SimpleNamespace(origin="user", value=2.0, default=2.0)
    assert not user_changed("stack", "max_fwhm_ratio", at)                 # written at its default: not a change
    assert user_changed("stack", "max_fwhm_ratio", SimpleNamespace(origin="user", value=1.5, default=2.0))
    assert user_changed("stack", "min_transparency", SimpleNamespace(origin="user", value=0.6, default=0.6))  # auto
    assert not user_changed("stack", "max_fwhm_ratio", SimpleNamespace(origin="default", value=2.0, default=2.0))


def test_a_new_project_saves_its_graph_and_keeps_the_old_file_once(qapp, tmp_path, monkeypatch):
    """A project of the previous format: its first save writes the graph (format 3) and keeps the old file beside it as
    project.astroproj.v2.bak, once."""
    monkeypatch.setattr(graphmodel, "EXPLICIT_DEFAULTS", True)
    root = tmp_path / "P"
    w = h1.World(root, h1.FAKE[:1])
    w.m.apply_template(next(t for t in h1.GALLERY if t["key"] == "colour_min"), "Colour minimal", {})
    p0 = project2.Project2.from_dict(w.m.p.to_dict(), root)
    w.close()
    p0.graph = None
    project2.save(p0)                                                  # a file of the previous format
    old = (root / project2.FILE).read_bytes()
    wk = ApiWorker()
    m = ProjectModel(project2.load(root), wk)
    try:
        assert wk.wait(120)
        m.set_param(m.paths[0], "stack", "kappa", 2.5)
        m.flush()
        assert json.loads((root / project2.FILE).read_text(encoding="utf-8"))["format"] == project2.GRAPH_FORMAT
        bak = root / (project2.FILE + ".v2.bak")
        assert bak.read_bytes() == old
        m.set_param(m.paths[0], "stack", "kappa", 2.0)
        m.flush()
        assert bak.read_bytes() == old                                 # once
        assert project2.load(root).workflows[0].params["stack"]["kappa"] == 2.0
    finally:
        m.close(save=False)
        wk.stop()
