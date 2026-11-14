"""Inspect > Gradient tuned as a draft (client/gradientedit.py): the draft never touches the project, its preview is the
stage's own result on the reduced copy, Apply / Try variant / Reset do what they say, leaving with a draft asks in the
notice strip, every tool is in the window at 1250 x 685, and the CPU time of a preview.

The first tests drive the editor on a made-up stack with a stand-in model (no project on disk); the window tests read the
test project through a scratch copy (nothing is computed, nothing is written into it)."""
from __future__ import annotations

import os
import time

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
pytest.importorskip("PySide6")
from PySide6.QtCore import Signal  # noqa: E402
from PySide6.QtWidgets import QApplication, QWidget  # noqa: E402

from astro_gpu.client import gradientedit as ge  # noqa: E402
from astro_gpu.stages.gradient import GradientParams, read_grid, sky_model_of  # noqa: E402

H, W = 300, 450


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def small_stack(seed=3):
    """A reduced copy as the editor makes it: a corner glow and a tilt on a colour sky, stars, noise."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W] / np.array([H, W], np.float32)[:, None, None]
    sky = 100 + 20 * xx + 10 * yy ** 2 + 15 * np.exp(-np.hypot(xx - 1, yy) / 0.3)
    img = np.stack([sky * k for k in (1.0, 1.2, 0.8)]) + rng.normal(0, 0.5, (3, H, W))
    for _i in range(80):
        y, x = rng.integers(3, H - 3), rng.integers(3, W - 3)
        img[:, y - 1:y + 2, x - 1:x + 2] += rng.uniform(20, 200)
    return {"pid": 1, "arr": img.astype(np.float32), "valid": np.ones((H, W), bool), "f": 0.5, "W": 2 * W, "H": 2 * H}


class View:
    def __init__(self, pid, name="Full"):
        self.pid, self.name, self.params = pid, name, {"gradient": {}}


class Model:
    """What the editor asks of the project model; records the changes made."""

    def __init__(self):
        self.p = object()
        self.path = View(1)
        self.paths = [self.path]
        self.sel_step = "fond"
        self.changes, self.variants, self.logs = [], [], []

    def param_value(self, p, stage, name):
        return p.params.get(stage, {}).get(name, GradientParams.model_fields[name].default)

    def state(self, p, stage):
        return "done"

    def stage_key(self, p, stage):
        return ("key", tuple(sorted((k, repr(v)) for k, v in p.params.get("gradient", {}).items())))

    def scope_paths(self):
        return [self.path]

    def view(self, pid):
        return next((q for q in self.paths if q.pid == pid), None)

    def apply_change(self, paths, ch):
        self.changes.append(ch)
        for q in paths:
            if ch["kind"] == "params":
                for st, nm, v in ch["items"]:
                    q.params.setdefault(st, {})[nm] = v
            elif ch["kind"] == "opt":
                q.params.setdefault("gradient", {})["method"] = {"simple": "polynomial", "rbf": "rbf",
                                                                 "auto": "automatic"}[ch["value"]]
        return []

    def add_variant(self, p, ch, title):
        q = View(len(self.paths) + 1, title)
        q.params = {"gradient": dict(p.params["gradient"])}
        self.paths.append(q)
        self.variants.append((p.pid, ch, title))
        self.apply_change([q], ch)
        return q

    def cost_of_change(self, paths, ch):
        return type("Cost", (), {"seconds": 90.0, "measured": False})()

    def log(self, template, **k):
        self.logs.append(template.format(**k))


class Banner:
    def __init__(self):
        self.asked = []

    def ask(self, text, *actions):
        self.asked.append((text, actions))

    def hide(self):
        pass


class Tune(QWidget):
    toasted = Signal(str)

    def __init__(self, model):
        super().__init__()
        self.s = model
        self.show_hist = True
        self._dirty = False
        self._center_sig = "kept"
        self.banner = Banner()
        self.refreshed = 0
        self.made = []

    def request_refresh(self):
        self.refreshed += 1

    def _has_previous(self, p, k):
        return False

    def _variant_made(self, pid, orig):
        self.made.append((pid, orig))


@pytest.fixture()
def ed(qapp, monkeypatch):
    sm = small_stack()
    monkeypatch.setattr(ge, "_reduced_stack", lambda project, pid: {**sm, "pid": pid})
    monkeypatch.setattr(ge, "_applied_small", lambda project, pid, s: None)
    monkeypatch.setattr(ge.GradientEditor, "_computed_samples", lambda self: None)
    m = Model()
    t = Tune(m)
    t.window = lambda: t
    e = t.grad = ge.GradientEditor(t)
    e._load(m.path)
    return e


def wait(e, timeout=30.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        QApplication.processEvents()
        if not e.busy and not e._timer.isActive() and e._pending is None:
            QApplication.processEvents()
            if not e.busy:
                return
        time.sleep(0.01)
    raise AssertionError("the preview did not come")


PTS = [[x, y] for x in (60.0, 300.0, 520.0, 760.0, 860.0) for y in (60.0, 300.0, 540.0)]


def test_a_draft_never_changes_the_project(ed):
    m, p = ed.s, ed.s.path
    key = m.stage_key(p, "gradient")
    ed.set_draft({"method": "rbf"})
    ed.set_draft({"rbf_smooth": 0.2})
    ed.work["sample_points"] = [list(x) for x in PTS]
    ed._commit()
    ed._set_avoid(False)
    ed.work["exclude_ellipses"].append([400.0, 300.0, 60.0, 40.0, 0.0])
    ed._commit()
    wait(ed)
    assert m.changes == [] and m.variants == [] and p.params == {"gradient": {}}
    assert m.stage_key(p, "gradient") == key
    assert sorted(ed._draft()["vals"]) == ["avoid_emission", "exclude_ellipses", "method", "rbf_smooth",
                                           "sample_points"]
    assert ed.result is not None and ed.tag_text().startswith("Draft preview")


@pytest.mark.parametrize("draft", [{"method": "polynomial", "degree": 3},
                                   {"method": "rbf", "rbf_smooth": 0.3, "sample_points": PTS},
                                   {"method": "polynomial", "degree": 1, "sample_points": PTS,
                                    "exclude_ellipses": [[450.0, 300.0, 100.0, 80.0, 20.0]]}])
def test_the_preview_is_the_stage_result_on_the_reduced_copy(ed, draft):
    ed.set_draft(draft)
    wait(ed)
    sm = small_stack()
    prm = GradientParams(**draft)
    mo = sky_model_of(sm["arr"], sm["valid"], prm, f=sm["f"])
    g = read_grid(mo["grid"], mo["info"], H, W)
    lvl = np.array([np.median(g[c]) for c in range(3)], np.float32)[:, None, None]
    want = sm["arr"] - g + lvl
    assert ed.result.shape == want.shape
    assert float(np.abs(ed.result - want).max()) < 1e-3


def test_apply_makes_the_draft_the_block_settings(ed):
    m, p = ed.s, ed.s.path
    ed.set_draft({"method": "rbf", "rbf_smooth": 0.25})
    ed.work["sample_points"] = [list(x) for x in PTS]
    ed._commit()
    wait(ed)
    ed._update_cost()
    assert "about" in ed.cost_text() and ed.s.changes == []          # asked first, nothing changed by asking
    ed.apply()
    assert [c["kind"] for c in m.changes] == ["opt", "params"]
    assert p.params["gradient"]["method"] == "rbf" and p.params["gradient"]["rbf_smooth"] == 0.25
    assert len(p.params["gradient"]["sample_points"]) == len(PTS)
    assert not ed.has_draft() and ed._applied_value("method") == "rbf"


def test_apply_is_one_step_through_the_general_confirmation(ed):
    m = ed.s
    got = []
    m.apply_as = lambda ch, how="replace", paths=None, title="": got.append((ch, how, [q.pid for q in paths]))
    ed.set_draft({"method": "automatic", "scale": 7.0})
    ed.apply()
    (ch, how, pids), = got
    assert how == "replace" and pids == [1] and ch["kind"] == "many"
    assert [c["kind"] for c in ch["changes"]] == ["opt", "params"]


def test_try_variant_keeps_the_workflow_and_shows_the_variant(ed):
    m, p = ed.s, ed.s.path
    ed.set_draft({"degree": 4})
    ed.try_variant()
    assert p.params == {"gradient": {}}
    (pid, ch, title), = m.variants
    assert pid == 1 and ch["kind"] == "params" and ("gradient", "degree", 4) in ch["items"] and "Full" in title
    q = m.paths[-1]
    assert q.params["gradient"]["degree"] == 4 and ed.tv.made == [(q.pid, 1)] and not ed.has_draft(1)


def test_reset_and_the_draft_undo(ed):
    ed.set_draft({"degree": 3})
    ed.set_draft({"degree": 4})
    ed.undo()
    assert ed.value("degree") == 3
    ed.redo()
    assert ed.value("degree") == 4
    ed.set_draft({"degree": 2})                        # back to the applied value: no draft
    assert not ed.has_draft()
    ed.set_draft({"method": "rbf"})
    ed.reset()
    assert not ed.has_draft() and ed.value("method") == "polynomial" and ed.s.changes == []


def test_leaving_with_a_draft_asks_in_the_notice_strip(ed):
    m, p = ed.s, ed.s.path
    ed.set_draft({"degree": 3})
    ed._check_left()                                   # the view hidden: the block is not the one shown
    (text, actions), = ed.tv.banner.asked
    assert "draft is not applied" in text and [a[0] for a in actions] == ["Apply", "Try variant", "Reset", "Later"]
    ed._check_left()
    assert len(ed.tv.banner.asked) == 1                # said once
    dict(actions)["Later"]()
    assert ed.has_draft() and m.changes == []          # Later: kept
    dict(actions)["Apply"]()
    assert p.params["gradient"]["degree"] == 3 and not ed.has_draft()


def test_a_removed_sample_has_a_visible_control(ed):
    ed.set_draft({"sample_points": PTS})
    ed.mode, ed._selpt = "points", 2
    ed.remove_selected()
    assert len(ed.value("sample_points")) == len(PTS) - 1 and PTS[2] not in ed.value("sample_points")


@pytest.mark.slow
@pytest.mark.parametrize("method", ["polynomial", "rbf"])
def test_cpu_time_of_a_preview(method):
    """The preview on a reduced copy of one megapixel, in CPU time of the process (one thread)."""
    rng = np.random.default_rng(1)
    h, w = 820, 1230
    yy, xx = np.mgrid[0:h, 0:w] / np.array([h, w], np.float32)[:, None, None]
    sky = 100 + 20 * xx + 10 * yy ** 2
    arr = (np.stack([sky] * 3) + rng.normal(0, 0.5, (3, h, w))).astype(np.float32)
    sm = {"arr": arr, "valid": np.ones((h, w), bool), "f": 0.25, "W": 4 * w, "H": 4 * h}
    pts = [[x, y] for x in np.linspace(200, 4 * w - 200, 12) for y in np.linspace(200, 4 * h - 200, 8)]
    prm = GradientParams(method=method, sample_points=pts if method == "rbf" else [])
    ge._preview(sm, prm)
    best = min(_cpu(ge._preview, sm, prm) for _i in range(3))
    print(f"{method}: {best * 1000:.0f} ms CPU")
    assert best < 0.6, best                            # the budget is 0.3 s; twice that on a slow shared machine


def _cpu(fn, *a):
    t0 = time.process_time()
    fn(*a)
    return time.process_time() - t0


# ---------------------------------------------------------------------------------------------- the real window
def _window(tmp_path, size):
    from conftest import needs_test_project, scratch_project  # noqa: F401
    from test_inspect import inspect, open_window, settle

    from astro_gpu.client import app as appmod
    from astro_gpu.client import prefs
    from astro_gpu.client.widgets import QSS
    QApplication.instance().setStyleSheet(QSS + appmod.EXTRA_QSS)
    os.environ["ASTRO_GPU_SETTINGS"] = str(tmp_path / "settings.json")
    prefs.load()
    appmod.apply_language("en")
    folder = scratch_project(tmp_path / "proj")
    w = open_window(folder, size=size, readonly=False)
    inspect(w, "fond", index=0)
    w.tune.dock.set_state("docked")                    # the panel open: the narrowest room for the image
    settle(w)
    return w, settle


def _test_project():
    from conftest import HAVE_TEST_PROJECT
    if not HAVE_TEST_PROJECT:
        pytest.skip("the test project is not here")


@pytest.mark.parametrize("size", [(1250, 685), (1366, 728)])
def test_every_tool_is_in_the_window_on_a_laptop(qapp, tmp_path, size):
    _test_project()
    w, settle = _window(tmp_path, size)
    try:
        t, ed = w.tune, w.tune.grad_editor
        if ed.view is None:
            pytest.skip("Gradient of the test project is not computed")
        assert not t.view.panel_box.isVisibleTo(t.view)          # the stretch column folded: the image keeps the room
        for k in ("method", "main", "avoid", "reset", "variant", "apply", "points", "brush", "ellipse", "grid",
                  "density", "survey", "marks", "remove", "clear", "undo", "redo"):
            b = ed._ui[k]
            assert b.isVisibleTo(w), k
            r = b.rect().translated(b.mapTo(w, b.rect().topLeft()))
            assert r.left() >= 0 and r.top() >= 0 and r.right() <= w.width() and r.bottom() <= w.height(), (k, r)
            assert b.width() >= b.sizeHint().width() - 1, k                       # never squeezed: its name whole
        ed.set_mode("brush")
        for k in ("paint", "erase", "size"):
            b = ed._ui[k]
            r = b.rect().translated(b.mapTo(w, b.rect().topLeft()))
            assert b.isVisibleTo(w) and r.right() <= w.width(), k
        ed.set_mode(None)
    finally:
        w.close()


def test_the_draft_leaves_the_graph_up_to_date_until_apply(qapp, tmp_path):
    _test_project()
    w, settle = _window(tmp_path, (1250, 685))
    try:
        s, ed = w.s, w.tune.grad_editor
        p = s.path
        if ed.view is None:
            pytest.skip("Gradient of the test project is not computed")
        before = {st: (s.state(p, st), s.stage_key(p, st)) for st in s.all_stages(p)}
        ed.set_draft({"degree": 3, "avoid_emission": False})
        settle(w)
        for _i in range(200):
            QApplication.processEvents()
            if ed.result is not None and not ed.busy:
                break
            time.sleep(0.05)
        assert {st: (s.state(p, st), s.stage_key(p, st)) for st in s.all_stages(p)} == before
        assert ed.result is not None and w.tune.view.image().name == "Draft preview"
        ed.apply()
        settle(w)
        assert s.state(p, "gradient") == "stale" and s.param_value(p, "gradient", "degree") == 3
        assert not ed.has_draft()
    finally:
        w.close()
