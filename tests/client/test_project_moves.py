"""A computed project moved, copied, saved as, saved and opened again, made into a template: the classes of defects the
owner met with his projects. Each test makes a tiny project in its own folder (three lights and three darks of 24 x 32
pixels in FITS) and computes it for real, on the processor, in this process (its Calibration block: the engine's own
stage, as tests/test_parallel_calibration.py runs it); then the application's window opens it.

"Up to date" is said twice: by the engine (a plan of everything holds no job) and by the window (every block that
computes something is "done" on the graph)."""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from astro_gpu.core import project2, runner2
from astro_gpu.core import workflow as wfl
from astro_gpu.core.project2 import Session
from tests.client.test_swap_smoke import drawn, env, qapp, ready, settle, win  # noqa: F401  (the window's fixtures)

ACCENTED = "Nébuleuse d'essai"


def fits_frames(folder: Path, prefix: str, level: float, seed: int, n: int = 3) -> Path:
    from astropy.io import fits
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        rng = np.random.default_rng(seed + i)
        a = np.clip(rng.normal(level, 4, (24, 32)), 0, 65535).astype(np.uint16)
        h = fits.Header()
        for k, v in {"BAYERPAT": "RGGB", "EXPTIME": 15.0, "GAIN": 100.0, "DATE-OBS": "2026-08-16T22:00:00",
                     "INSTRUME": "TESTCAM"}.items():
            h[k] = v
        fits.PrimaryHDU(a, h).writeto(folder / f"{prefix}{i}.fits")
    return folder


def computed(root: Path, inside: bool = False, scratch: str | None = None) -> Path:
    """A project computed in ``root``: its frames in ``root``/raw named relative to it (``inside``), else beside it,
    by their absolute paths. Its calibrated frames go to ``scratch`` (as the window sets it: <cache folder>/<project
    folder>), by default the engine's own place, ``root``/scratch."""
    raw = root / "raw" if inside else root.parent / f"{root.name}_raw"
    fits_frames(raw / "L", "L", 400, 0)
    fits_frames(raw / "D", "D", 50, 10)
    p = project2.create(root, root.name)
    p.stack_mode = "per_session"
    p.devices = {"main": "cpu", "aux": ""}
    if scratch is not None:
        p.scratch_dir = scratch
    lights, darks = ("raw/L", "raw/D") if inside else (str(raw / "L"), str(raw / "D"))
    p.sessions.append(Session(1, "night 1", lights=[lights], darks=[darks]))
    p.add_workflow("Main", wfl.DEFAULT_OPTIONS)
    p.set_blocks(1, [wfl.node("calibration")])
    p.workflows[0].params["calibrate"] = {"workers": 1}
    project2.save(p)
    plan = runner2.plan_all(p)
    assert plan.jobs
    assert runner2.run_plan(p.root, runner2.prepare_run(p, plan, "cpu").plan_file, echo=False) == "done"
    assert to_compute(root) == []
    return root


def to_compute(root: Path) -> list[str]:
    """The stages a run of every workflow of the project in ``root`` would compute, read from the disk."""
    return sorted(j.stage for j in runner2.plan_all(project2.load(root)).jobs)


def block_states(w) -> dict:
    bm = w.s.block_map
    return {n.key: bm.state(n) for n in bm.nodes if n.paths and n.key not in ("session",) and not n.cut}


def all_done(w, timeout: float = 60.0) -> dict:
    """The window's states once they settle; every block that computes something "done"."""
    from tests.client.test_swap_smoke import pump
    pump(lambda: block_states(w) and all(v == "done" for v in block_states(w).values()), timeout)
    return block_states(w)


def opened(win, root: Path):  # noqa: F811
    assert win.open_project(root)
    ready(win)
    return win.s


# ---------------------------------------------------------------------------------------------- a. the folder copied
def test_a_computed_project_copied_elsewhere_is_up_to_date_there(win, env):  # noqa: F811
    root = computed(env["tmp"] / "here" / "P")
    other = env["tmp"] / "there" / "P"
    shutil.copytree(root, other)
    os.rename(root, root.with_name("P_gone"))               # nothing can be read where it was computed
    assert to_compute(other) == []
    opened(win, other)
    states = all_done(win)
    assert states and set(states.values()) == {"done"}, states
    assert not env["errors"], env["errors"]


# ---------------------------------------------------------------------------------------------- b. its frames inside it
@pytest.mark.parametrize("name", ["P", ACCENTED])
def test_a_project_holding_its_frames_moved_elsewhere_is_up_to_date_there(win, env, name):  # noqa: F811
    root = computed(env["tmp"] / "here" / name, inside=True)
    d = json.loads((root / project2.FILE).read_text(encoding="utf-8"))
    assert d["sessions"][0]["lights"] == ["raw/L"]          # named relative to the project, as written
    other = env["tmp"] / "là bas" / name
    shutil.copytree(root, other)
    os.rename(root, root.with_name(name + " parti"))
    assert to_compute(other) == []
    opened(win, other)
    states = all_done(win)
    assert states and set(states.values()) == {"done"}, states
    assert not env["errors"], env["errors"]


# ---------------------------------------------------------------------------------------------- c. Save as
def _save_as(win, env, monkeypatch, root: Path, new: Path):  # noqa: F811
    from PySide6.QtWidgets import QFileDialog
    opened(win, root)
    assert set(all_done(win).values()) == {"done"}
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", staticmethod(lambda *a, **k: new.as_posix()))
    win.save_as()
    ready(win)
    assert Path(win.project.root).resolve() == new.resolve()
    assert (new / "store").is_dir() and (new / project2.FILE).is_file()
    assert to_compute(new) == []
    states = all_done(win)
    assert states and set(states.values()) == {"done"}, states
    assert not env["errors"], env["errors"]


def test_save_as_moves_the_results_and_the_copy_is_up_to_date(win, env, monkeypatch):  # noqa: F811
    """As the window makes a project: its calibrated frames in the cache folder, outside the project's folder."""
    from astro_gpu.client import prefs
    root = computed(env["tmp"] / "P", scratch=(Path(prefs.get("cache_dir")) / "P").as_posix())
    _save_as(win, env, monkeypatch, root, env["tmp"] / "Q")
    assert sorted(x.name for x in root.iterdir()) == [project2.FILE]       # the results moved, never copied


@pytest.mark.xfail(strict=True, reason="Save as moves store, logs, state.json and the thumbnail, not the calibrated "
                   "frames of a project whose scratch folder is its own (no scratch_dir: <project>/scratch): they stay "
                   "behind in the folder left, while the copy is given a new scratch folder in the cache")
def test_save_as_leaves_nothing_but_the_project_file_when_the_scratch_is_inside(win, env, monkeypatch):  # noqa: F811
    root = computed(env["tmp"] / "P")
    assert (root / "scratch").is_dir()
    _save_as(win, env, monkeypatch, root, env["tmp"] / "Q")
    assert sorted(x.name for x in root.iterdir()) == [project2.FILE]


# ---------------------------------------------------------------------------------------------- d. saved and opened again
def test_saved_by_the_window_closed_and_opened_again_is_the_same_and_up_to_date(win, env):  # noqa: F811
    root = computed(env["tmp"] / "P")
    s = opened(win, root)
    assert set(all_done(win).values()) == {"done"}
    win.save()
    settle(win)
    graph, post = drawn(s.stored), json.dumps(s.p.to_dict()["post"], sort_keys=True)
    assert win.close_project(go_home=False)
    s = opened(win, root)
    assert drawn(s.stored) == graph
    assert json.dumps(s.p.to_dict()["post"], sort_keys=True) == post
    states = all_done(win)
    assert states and set(states.values()) == {"done"}, states
    assert to_compute(root) == []
    assert not env["errors"], env["errors"]


# ---------------------------------------------------------------------------------------------- e. and f. a template
def _blocks_and_wires(g) -> list:
    rows, _taps, wires = drawn(g)
    return [[(wid, name, blocks) for wid, name, blocks, _nights in rows], wires]


def test_a_template_saved_from_a_project_makes_a_project_of_the_same_graph_without_frames_nor_results(
        win, env, monkeypatch):  # noqa: F811
    from astro_gpu.client import gallery, templatefiles
    root = computed(env["tmp"] / "P")
    s = opened(win, root)
    graph = _blocks_and_wires(s.stored)
    monkeypatch.setattr(gallery, "template_dialog", lambda *a, **k: ("Mine", "My templates"))
    win.save_template()
    settle(win)
    files = [f for f, mine in templatefiles.files("My templates") if mine]
    assert [f.stem for f in files] == ["Mine"]
    assert win.close_project(go_home=False)
    assert win.new_project_from_template(files[0])
    ready(win)
    s = win.s
    assert _blocks_and_wires(s.stored) == graph
    assert all(not getattr(x, k) for x in s.p.sessions for k in project2.KINDS)          # no frames
    made = Path(win.project.root)
    assert made.resolve() != root.resolve()
    assert not (made / "store").is_dir() or not any((made / "store").iterdir())         # no results
    assert "done" not in block_states(win).values()
    # f. its Table view says no file unreadable
    page = win.graph
    page.set_display("table")
    settle(win)
    t = page.table_host.table.grid
    assert t.reader.wait(30)
    assert [x for pl in t.model.stacks() for x in pl["sessions"]]       # a session is shown, its files read
    settle(win)
    assert not [w.text for w in t.model.all_warnings() if "unreadable" in w.text], t.model.all_warnings()
    assert not env["errors"], env["errors"]


def test_the_table_of_a_computed_project_says_no_file_unreadable(win, env):  # noqa: F811
    """f. on a project that holds its frames, opened from another folder than the one the application runs in."""
    root = computed(env["tmp"] / ACCENTED, inside=True)
    opened(win, root)
    page = win.graph
    page.set_display("table")
    settle(win)
    t = page.table_host.table.grid
    assert t.reader.wait(30)
    assert [x for pl in t.model.stacks() for x in pl["sessions"]]       # a session is shown, its files read
    settle(win)
    assert not [w.text for w in t.model.all_warnings() if "unreadable" in w.text], t.model.all_warnings()
    assert not env["errors"], env["errors"]
