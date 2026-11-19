"""Opening a project computes and reads only what its first screen quotes (issue #170): the automatic values a stage
suggested at an earlier opening come back from the program's state folder (not worked out again, the drizzle one reading
every frame's alignment), and the first opening quotes the stages' durations only, the method cards', the toolbar's and
the New workflow dialog's following it. Read in place on the test project."""
from __future__ import annotations

import json
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication  # noqa: E402

from astro_gpu import api  # noqa: E402
from astro_gpu.client.model import ProjectModel  # noqa: E402
from astro_gpu.client.services.worker import ApiWorker  # noqa: E402
from conftest import TEST_PROJECT, full_only, needs_test_project, scratch_project  # noqa: E402

pytestmark = needs_test_project


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _open(tmp_path):
    return full_only(api.open_project(scratch_project(tmp_path / "p", TEST_PROJECT)))


def test_the_automatic_values_of_a_stage_are_kept_with_the_project(tmp_path, monkeypatch):
    p = _open(tmp_path)
    stage = api._keys(p).rules.chains.stages["stack"]
    calls = []
    real = stage.auto
    monkeypatch.setattr(stage, "auto", lambda ctx: calls.append(1) or real(ctx), raising=False)
    first = api.describe(p, 1, "stack")
    assert calls, "the first reading works the automatic values out"
    assert api._autos_path(p).is_file()
    assert not (p.cache / "autos.json").exists(), "nothing is written in the project"
    n = len(calls)
    again = api.describe(_open(tmp_path), 1, "stack")                  # another opening: a new project object
    assert len(calls) == n, "the values kept are read, not worked out again"
    assert [(r.name, r.value, r.suggested, r.reason, r.reason_args) for r in again] == \
        [(r.name, r.value, r.suggested, r.reason, r.reason_args) for r in first]


def test_a_value_kept_for_another_key_is_not_taken(tmp_path, monkeypatch):
    p = _open(tmp_path)
    api.describe(p, 1, "stack")
    f = api._autos_path(p)
    kept = json.loads(f.read_text(encoding="utf-8"))
    f.write_text(json.dumps({"stack|not-its-key": next(iter(kept.values()))}), encoding="utf-8")
    stage = api._keys(p).rules.chains.stages["stack"]
    calls = []
    real = stage.auto
    monkeypatch.setattr(stage, "auto", lambda ctx: calls.append(1) or real(ctx), raising=False)
    api.describe(_open(tmp_path), 1, "stack")
    assert calls, "a value kept under another key is worked out again"


def test_the_first_opening_quotes_the_stage_durations_only(qapp, tmp_path, monkeypatch):
    seen = []
    for name in ("method_seconds", "tool_cost", "opts_seconds"):
        real = getattr(ProjectModel, name)
        monkeypatch.setattr(ProjectModel, name, lambda self, *a, _r=real, _n=name, **k: seen.append(_n) or _r(self, *a, **k))
    w = ApiWorker()
    m = ProjectModel(_open(tmp_path), w, readonly=True)
    got = []
    m.readyChanged.connect(lambda: got.append(list(seen)))
    import time
    end = time.time() + 180
    while time.time() < end and not got:
        qapp.processEvents()
        time.sleep(0.01)
    assert got, "the model never said it was ready"
    assert got[0] == [], f"the first screen's quotes read more than the stages' durations: {sorted(set(got[0]))}"
    assert w.wait(180)                                              # the cards' and dialogs' durations follow
    assert "method_seconds" in seen and "opts_seconds" in seen
    m.close(save=False)
