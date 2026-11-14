"""The keys a run is planned with are the keys its worker computes.

The client plans on its definition (``api._keys``: a copy of the compiled definition); the run freezes another copy in
its plan file (``RunService`` takes a snapshot, ``prepare_run`` writes it); the worker and a job's child process rebuild
the project from that file and check each job's key before it runs ("the inputs of ... changed since the run was
planned"). Checked here, for every gallery template and for a chain built block by block on a blank project (Session,
Calibration, Registration, Selection, Stacking, run; then Astrometry, Gradient, Denoise, Stars/background and Combine
layers): every job's key, and the whole key table, read by the worker from the plan file equal the planned ones, before
any result exists and after each job's result exists. And the worker, started in the project's folder, imports the code
that planned the run (a worker importing another copy of FLOAT keyed Stars/background with that copy's network)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "graphcore"))
import test_h1_old_way as h1  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

import astro_gpu  # noqa: E402
from astro_gpu import api  # noqa: E402
from astro_gpu.core import runner2  # noqa: E402
from astro_gpu.core.project2 import Project2  # noqa: E402
from astro_gpu.core.store import Store, StoreProject  # noqa: E402

TEMPLATES = [t for t in h1.GALLERY if t.get("copy") is None]


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def table(keys) -> dict:
    """{workflow: {stage: key}} of every workflow (an error as its text)."""
    out = {}
    for w in keys.project.workflows:
        try:
            out[w.id] = {s: keys.key(w, s) for s in keys.chain(w)}
        except (KeyError, ValueError) as e:
            out[w.id] = f"{type(e).__name__}: {e}"
    return out


def frozen(m):
    """The worker of a run of the model's Run all, as a run makes it: the plan, the snapshot of the definition the
    run window takes, the plan file ``prepare_run`` writes, the worker built from that file."""
    plan, _pause = m.plan_run()
    snapshot = Project2.from_dict(m.p.to_dict(), m.p.root)
    info = runner2.prepare_run(snapshot, plan, "cpu")
    d = json.loads(Path(info.plan_file).read_text(encoding="utf-8"))
    return plan, runner2._Worker(Path(info.root), d, None, False)


def check_frozen(m, what: str) -> None:
    """Every job's key and the whole key table, from the plan file, equal the planned ones; then again after each job's
    result exists (marked done in the store, in the plan's order)."""
    plan, worker = frozen(m)
    assert table(worker.keys) == table(api._keys(m.p)), what
    for j in plan.jobs:
        sp = StoreProject(worker.keys, worker.project.workflow(j.run_wf), j.stage)
        assert sp.key == j.key, (what, j.stage, j.run_wf)
    store = Store(m.p.root)
    for j in plan.jobs:
        folder = store.dir(j.stage, j.key)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "manifest.json").write_text(json.dumps({"stage": j.stage, "status": "done", "key": j.key}),
                                              encoding="utf-8")
        api.refresh(m.p)
        again = runner2._Worker(worker.root, worker.d, None, False)
        assert table(again.keys) == table(api._keys(m.p)), (what, "after", j.stage)
        for x in m.plan_run()[0].jobs:                     # what is left to run, planned again, keyed alike
            assert StoreProject(again.keys, again.project.workflow(x.run_wf), x.stage).key == x.key, \
                (what, "after", j.stage, x.stage)


@pytest.mark.parametrize("t", TEMPLATES, ids=[t["key"] for t in TEMPLATES])
def test_a_template_runs_with_the_keys_it_was_planned_with(qapp, tmp_path, t):
    w = h1.old_way(tmp_path / "P", t, h1.FAKE)
    try:
        check_frozen(w.m, t["key"])
    finally:
        w.close()


def test_a_chain_built_by_hand_then_extended_runs_with_its_planned_keys(qapp, tmp_path):
    w = h1.World(tmp_path / "P", [("Night 1", "")])
    m = w.m
    try:
        lights = list(m.p.sessions[0].lights)                # a stack of its own, reading the night's frames
        assert lights and m.apply([("add_row", {"kind": "session", "files": {"lights": lights}})]).ok
        assert w.w.wait(120) and m.ready                   # the sessions read again once files are added
        row = m.stored.rows[0].id
        for kind in ("calibration", "registration", "selection", "stacking"):
            res = m.apply([("add_block", {"kind": kind, "row": row})])
            assert res.ok, (kind, res.reason)
        check_frozen(m, "first chain")
        for kind in ("astrometry", "gradient", "denoise", "stars"):
            res = m.apply([("add_block", {"kind": kind, "row": row})])
            assert res.ok, (kind, res.reason)
        res = m.apply([("add_block", {"kind": "combine_layers"})])
        assert res.ok, res.reason
        stages = {j.stage for j in m.plan_run()[0].jobs}
        assert {"denoise", "star_removal"} <= stages, stages
        check_frozen(m, "extended")
    finally:
        w.close()


def test_the_worker_imports_the_code_that_planned_the_run(tmp_path, monkeypatch):
    started = []

    class Fake:
        pid = 1

    def popen(cmd, **kw):
        started.append((cmd, kw))
        return Fake()
    with monkeypatch.context() as mp:
        mp.setattr(runner2.subprocess, "Popen", popen)
        runner2._spawn_worker(tmp_path, tmp_path / "x.plan.json", "0", tmp_path / "x.log")
    (cmd, kw), = started
    assert cmd[1:3] == ["-m", "astro_gpu.core.runner2"]
    probe = subprocess.run([cmd[0], "-c", "import astro_gpu; print(astro_gpu.__file__)"], cwd=kw["cwd"],
                           env=kw["env"], capture_output=True, text=True, timeout=120)
    assert probe.returncode == 0 and Path(probe.stdout.strip()).resolve() == Path(astro_gpu.__file__).resolve(), \
        probe.stdout + probe.stderr
