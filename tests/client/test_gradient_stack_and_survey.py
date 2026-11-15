"""The Gradient editor and what it keeps of a stack: the H-alpha survey is only downloaded on the user's click, the
reduced copy, its size, the block's own samples and the survey follow a new stack result, and an Outdated stack still
gives its size (the samples are never placed in a 1 px square)."""
from __future__ import annotations

import numpy as np

from astro_gpu.client import gradientedit as ge
from tests.client.test_gradient_draft import View, ed, qapp, small_stack, wait  # noqa: F401  (fixtures by name)


def test_changing_workflow_with_the_survey_shown_downloads_nothing(ed, monkeypatch):
    asked = []

    def survey_small(project, pid, sm, fetch=False):
        asked.append((pid, fetch))
        return {"level": np.zeros(sm["arr"].shape[1:], np.float32), "title": "H-alpha"}

    monkeypatch.setattr(ge, "_survey_small", survey_small)
    ed.survey = {"pid": 1, "level": np.zeros((3, 3), np.float32), "title": "H-alpha"}
    ed.survey_shown = True
    q = View(2, "Variant")
    ed.s.paths.append(q)
    ed._load(q)
    wait(ed)
    assert asked, "the survey of the workflow now shown was not read"
    assert all(not fetch for _pid, fetch in asked), asked


class State:
    def __init__(self, key, state="done", previous_key=None):
        self.key, self.state, self.previous_key = key, state, previous_key


def test_a_new_stack_result_is_read_again(ed, monkeypatch):
    """The stack run again on a crop: the reduced copy, its size, the block's own samples and the survey are the new
    result's, so a sample clicked or generated is placed on the new stack."""
    states = {"stack": State("k1")}
    monkeypatch.setattr(ed.s, "state_info", lambda p, stage: states.get(stage), raising=False)
    p = ed.s.path
    ed._load(p)
    ed._run("load")
    wait(ed)
    assert ed._stack_size() == (900, 600)
    ed.defaults[p.pid] = [[800.0, 500.0]]
    ed.survey = {"pid": p.pid, "level": np.zeros((300, 450), np.float32), "title": "H-alpha"}
    crop = small_stack()
    crop = {**crop, "arr": crop["arr"][:, :150, :200].copy(), "valid": crop["valid"][:150, :200].copy(),
            "W": 400, "H": 300}
    monkeypatch.setattr(ge, "_reduced_stack", lambda project, pid: {**crop, "pid": pid})
    states["stack"] = State("k2")                                 # the stack computed again, cropped
    ed._load(p)
    assert p.pid not in ed.defaults and ed.survey is None
    ed._run("load")
    wait(ed)
    assert ed._stack_size() == (400, 300)
    pts = np.asarray(ed.grid_points_of())
    assert pts[:, 0].max() < 400 and pts[:, 1].max() < 300


def test_a_copy_read_from_the_replaced_stack_is_not_kept(ed, monkeypatch):
    """The stack's result changes while its copy is being read: the copy that arrives is the old one's, read again."""
    states = {"stack": State("k1")}
    monkeypatch.setattr(ed.s, "state_info", lambda p, stage: states.get(stage), raising=False)
    p = ed.s.path
    ed._load(p)
    ed.small = None
    sm = small_stack()
    reads = []
    monkeypatch.setattr(ge, "_reduced_stack", lambda project, pid: (reads.append(pid),
                                                                    {**sm, "pid": pid, "W": 900 + 100 * (len(reads) > 1)})[1])
    ed._run("load")
    states["stack"] = State("k2")                                 # before the answer is taken
    ed._load(p)
    wait(ed)
    assert ed._stack_size() == (1000, 600)


class _Out:
    def __init__(self, d):
        self.dir = str(d)


class _Canvas:
    """What the tool reads of the image: its displayed size and zoom."""
    scale = 1.0
    display = type("D", (), {"size": lambda self: (900, 600)})()


def test_an_outdated_stack_gives_its_previous_size(ed, monkeypatch, tmp_path):
    from astro_gpu import api
    np.save(tmp_path / "drz_full.npy", np.zeros((3, 40, 70), np.float32))
    monkeypatch.setattr(api, "stage_outputs", lambda project, pid, stage, which="current":
                        _Out(tmp_path) if stage == "stack" and which == "previous" else None)
    ed.small, ed._dims = None, None
    assert ed._stack_size() == (70, 40)
    pts = np.asarray(ed.grid_points_of())
    assert pts[:, 0].max() > 35 and pts[:, 1].max() > 20


def test_no_sample_is_placed_while_the_stack_size_is_unknown(ed, monkeypatch):
    from astro_gpu import api
    monkeypatch.setattr(api, "stage_outputs", lambda *a, **k: None)
    ed.small, ed._dims = None, None
    ed.set_mode("points")
    ed.small = None                                               # its reading did not come back
    monkeypatch.setattr(ed, "_run", lambda *a, **k: None)
    before = [list(x) for x in ed.work["sample_points"]]
    ed._press(_Canvas(), 450.0, 300.0, ge.Qt.LeftButton)          # a quick click
    assert ed.work["sample_points"] == before
    ed.act("generate")
    assert [list(x) for x in ed.work["sample_points"]] == before
