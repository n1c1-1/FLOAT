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
from PySide6.QtCore import Qt, Signal  # noqa: E402
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
    assert ge.preview_cell(prm, (H, W)) is None                # placed samples, or the polynomial: the run's own cell
    mo = sky_model_of(sm["arr"], sm["valid"], prm, f=sm["f"])
    g = read_grid(mo["grid"], mo["info"], H, W)
    lvl = np.array([np.median(g[c]) for c in range(3)], np.float32)[:, None, None]
    want = sm["arr"] - g + lvl
    assert ed.result.shape == want.shape
    assert float(np.abs(ed.result - want).max()) < 1e-3


def test_the_rbf_preview_on_its_own_samples_is_near_the_run(ed):
    """The RBF on the samples the block chooses is previewed on a coarser cell (gradientedit.preview_cell): the sky it
    takes off stays within 0.3 % of the sky's level of the run's (measured 0.2 % on the test project, 0.48 ADU of 240)."""
    ed.set_draft({"method": "rbf"})
    wait(ed)
    sm = small_stack()
    prm = GradientParams(method="rbf")
    assert ge.preview_cell(prm, (H, W)) == max(8, int(H / ge.PREVIEW_CELL_SHARE))
    mo = sky_model_of(sm["arr"], sm["valid"], prm, f=sm["f"])
    g = read_grid(mo["grid"], mo["info"], H, W)
    want = sm["arr"] - g + np.array([np.median(g[c]) for c in range(3)], np.float32)[:, None, None]
    sky = float(np.median(sm["arr"]))
    d = float(np.abs(ed.result - want).max())
    assert d < 0.003 * sky, (d, sky)


def test_steps_in_quick_succession_make_one_preview_of_the_last(ed):
    """Five steps of a setting within 300 ms (a spin box's arrows): one preview computed and painted, of the last
    value; never a burst."""
    row = {"name": "degree", "suggested": 2}
    for v in (3, 4, 5, 4, 3):
        ed.draft_setting(row, v)
        end = time.monotonic() + 0.05
        while time.monotonic() < end:
            QApplication.processEvents()
            time.sleep(0.005)
    wait(ed)
    assert (ed.computed, ed.painted) == (1, 1)
    sm = small_stack()
    want = ge._preview(sm, GradientParams(degree=3))["result"]
    assert float(np.abs(ed.result - want).max()) < 1e-3


def test_a_change_during_a_slow_preview_paints_only_the_latest(ed):
    """A setting changed while a slow preview runs (the RBF on its own 32 per side): that preview is let go or its
    result dropped, one preview of the latest draft is made, and only it is painted."""
    ed.set_draft({"method": "rbf", "samples": 32})
    end = time.monotonic() + 10
    while not ed.busy and time.monotonic() < end:     # the slow one under way
        QApplication.processEvents()
        time.sleep(0.005)
    assert ed.busy
    ed.set_draft({"rbf_smooth": 0.2})
    wait(ed, 60)
    assert ed.computed == 2 and ed.painted == 1
    sm = small_stack()
    want = ge._preview(sm, GradientParams(method="rbf", samples=32, rbf_smooth=0.2))["result"]
    assert float(np.abs(ed.result - want).max()) < 1e-3


def test_the_block_s_own_samples_are_one_editable_grid(ed):
    """The samples the block places by itself are drawn as ordinary samples, not in the draft; the first change makes
    them the draft's own (what is seen is what is used); Undo gives the block's own placement back."""
    ed._run_points(["generate", "objects", "nebula_disk"], [], default=True)
    wait(ed)
    grid = ed.defaults[1]
    assert grid and ed.work["sample_points"] == grid and not ed.has_draft()
    assert ed._params().sample_points == []                       # previewed as the stage would choose
    ed.work["sample_points"].append([500.0, 300.0])
    ed._commit()
    assert ed.value("sample_points")[:-1] == grid and len(ed.value("sample_points")) == len(grid) + 1
    ed.undo()
    assert not ed.has_draft() and ed.work["sample_points"] == grid


def test_the_actions_on_the_samples(ed):
    """Generate grid places a regular grid of its pitch; Avoid objects keeps the samples of sky (the stage's rule);
    Clear all leaves none and nothing chosen behind the user: Apply refused, the preview removes nothing; each one step
    of Undo, each said in the image's tag."""
    ed._run("load")                                            # the stack's size, as the view has it
    wait(ed)
    ed.set_draft({"samples": 8})
    ed.act("generate")
    n = len(ed.value("sample_points"))
    assert n == 8 * 12 and ed.tag_text().endswith("samples on a grid")     # 900 x 600: 8 by 12
    ed.act("objects")
    wait(ed)
    assert 0 < len(ed.value("sample_points")) <= n and "removed" in ed.tag_text()
    ed.act("clearall")
    wait(ed)
    assert ed._explicit_empty() and ed.too_few() and ed.value("sample_points") == []
    ed.apply()
    ed.try_variant()
    assert ed.s.changes == [] and ed.s.variants == []              # never "the block chooses" instead
    ed._said = None
    assert ed.tag_text().startswith("Too few samples")
    ed.undo()
    assert len(ed.value("sample_points")) > 0 and not ed.too_few()


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
    n = ed._notice
    assert n is not None and n.parentWidget() is ed.tv.window()   # over the window, in no layout: nothing moves
    assert [b.text() for b in n.buttons] == ["Apply", "Try variant", "Reset", "Later"]
    ed._check_left()
    assert ed._notice is n                             # said once
    n.buttons[3].click()
    assert ed.has_draft() and m.changes == [] and n.isHidden()    # Later: kept
    ed.apply()
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


PAGE_CONTROLS = ("before", "method", "points", "brush", "ellipse", "eraser", "generate", "pitch", "objects", "nebula",
                 "clearall", "size", "marks", "reset", "variant", "apply")


def _controls(w, ed) -> list[str]:
    """The interactive controls of the page outside its settings panel (and that panel's folded strip)."""
    from PySide6.QtWidgets import QAbstractButton, QAbstractSlider, QAbstractSpinBox, QComboBox
    seen, out = set(), []
    for k in (QAbstractButton, QAbstractSlider, QAbstractSpinBox, QComboBox):
        for x in ed.page.findChildren(k):
            if not x.isVisible() or ed.drawer.isAncestorOf(x) or ed.dock.strip.isAncestorOf(x):
                continue
            seg = x.parentWidget() if x.parentWidget().objectName() == "segment" else x
            if id(seg) not in seen:
                seen.add(id(seg))
                out.append(seg.objectName() + ":" + (x.text() if isinstance(x, QAbstractButton) else type(x).__name__))
    return out


@pytest.mark.parametrize("size", [(1250, 685), (1366, 728)])
def test_every_tool_is_in_the_window_on_a_laptop(qapp, tmp_path, size):
    _test_project()
    w, settle = _window(tmp_path, size)
    try:
        ed = w.tune.grad_editor
        if ed.view is None:
            pytest.skip("Gradient of the test project is not computed")
        # the settings panel automatic below 1500 px (its strip at the image's right), Inspect's own panel away
        assert ed.dock.state == "auto" and not ed.drawer.isVisible() and not w.tune.panel.isVisible()
        for k in PAGE_CONTROLS:
            b = ed._ui[k]
            assert b.isVisible(), k
            r = b.rect().translated(b.mapTo(w, b.rect().topLeft()))
            assert r.left() >= 0 and r.top() >= 0 and r.right() <= w.width() and r.bottom() <= w.height(), (k, r)
            assert b.width() >= b.sizeHint().width() - 1, k                       # never squeezed: its name whole
        # at most 16 outside the settings panel: Before / After, Method, the four tools and their size, the four
        # actions on the samples and the grid's pitch, Marks, Reset, Try variant, Apply (one more fails)
        assert len(_controls(w, ed)) <= 16, _controls(w, ed)
    finally:
        w.close()


def _geometry(w, ed):
    from PySide6.QtCore import QPoint
    c = ed.gv.canvas
    st = ed.gv.view_state()
    return (c.rect().translated(c.mapTo(w, QPoint(0, 0))).getRect(), round(st.get("zoom", 0), 2),
            round(st.get("fx", 0), 4), round(st.get("fy", 0), 4))


@pytest.mark.parametrize("size", [(1250, 685), (1920, 1080)])
def test_the_image_never_moves_through_a_session(qapp, tmp_path, size):
    """Every tool, every method, a sample placed, moved and erased, a painted area, the grid, Before / After, Marks,
    the settings panel opened over the image and closed, leaving with a draft and coming back, Reset, Apply: the
    image's rectangle and zoom stay (the panel's state is the user's choice of layout: see the next test)."""

    _test_project()
    w, settle = _window(tmp_path, size)
    try:
        s, t, ed = w.s, w.tune, w.tune.grad_editor
        if ed.view is None:
            pytest.skip("Gradient of the test project is not computed")

        def idle():
            settle(w)
            for _i in range(300):
                QApplication.processEvents()
                if not ed.busy and not ed._timer.isActive() and ed._held is None:
                    break
                time.sleep(0.02)
            settle(w)

        def go(step):
            s.select(step=step)
            t.select(step=step)
            t.request_refresh()
            idle()
        idle()
        g0 = _geometry(w, ed)
        W, H = ed._stack_size()
        moved = []

        def check(what):
            if _geometry(w, ed) != g0:
                moved.append((what, _geometry(w, ed)))
        for m in ("points", "brush", "ellipse", "eraser"):
            ed.set_mode(m)
            idle()
            check(m)
        for m in ("rbf", "automatic", "polynomial"):
            cb = ed._ui["method"]
            cb.setCurrentIndex(cb.findData(m))
            cb.activated.emit(cb.currentIndex())
            idle()
            check(m)
        for a in ("generate", "objects", "clearall", "generate"):
            ed._ui[a].click()
            idle()
            check(a)
        ed.work["sample_points"].append([W * 0.5, H * 0.5])
        ed._commit()
        idle()
        ed.work["sample_points"][-1] = [W * 0.55, H * 0.5]
        ed._commit()
        idle()
        check("sample placed and moved")
        n = len(ed.work["sample_points"])
        ed.set_mode("eraser")
        x, y = ed.work["sample_points"][-1]
        f = ed._f(ed.gv.canvas)
        ed.press(ed.gv.canvas, x * f, y * f, Qt.LeftButton)
        ed.release(ed.gv.canvas)
        idle()
        assert len(ed.work["sample_points"]) == n - 1                 # the Eraser takes the sample under it
        check("sample erased")
        ed.set_mode("brush")
        ed.press(ed.gv.canvas, W * 0.3 * f, H * 0.3 * f, Qt.LeftButton)
        ed.move(ed.gv.canvas, W * 0.35 * f, H * 0.3 * f)
        ed.release(ed.gv.canvas)
        idle()
        check("painted")
        for b in ("before", "before", "marks", "marks"):
            ed._ui[b].click()
            idle()
            check(b)
        if ed.dock.state != "docked":                  # the panel over the image: the image does not move
            ed.dock.open_overlay()
            idle()
            check("settings over the image")
            ed.dock.close_overlay()
            idle()
            check("settings closed")
        go("empilement")
        assert ed._notice is not None and ed._notice.isVisible()      # leaving with a draft: asked, over the window
        go("fond")
        check("came back")
        ed._ui["reset"].click()
        idle()
        check("reset")
        ed.set_draft({"degree": 3})
        idle()
        ed._ui["apply"].click()
        idle()
        check("apply")
        assert s.state(s.path, "gradient") == "stale"
        assert not moved, (g0, moved)
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


def test_the_settings_panel_has_the_window_s_three_states(qapp, tmp_path):
    """Kept open, automatic, kept folded: the window's side-panel states. Automatic and folded leave a strip at the
    image's right and open the panel over the image (the same image's rectangle in both); kept open, the panel takes its
    room beside the image. The user's choice is remembered; without one, the window's width chooses."""
    _test_project()
    w, settle = _window(tmp_path, (1920, 1080))
    try:
        ed = w.tune.grad_editor
        if ed.view is None:
            pytest.skip("Gradient of the test project is not computed")
        assert ed.dock.state == "docked" and ed.drawer.isVisible()       # a large window: kept open
        rect = {}
        for st in ("auto", "icons", "docked"):
            ed.dock._choose(st)
            settle(w)
            rect[st] = _geometry(w, ed)[0]
            assert ed.dock.strip.isVisible() == (st != "docked")
            if st != "docked":
                ed.dock.open_overlay()
                settle(w)
                assert ed.drawer.isVisible() and _geometry(w, ed)[0] == rect[st]
                ed.dock.close_overlay()
                settle(w)
        assert rect["auto"] == rect["icons"] and rect["docked"][2] < rect["auto"][2]
        from astro_gpu.client import prefs
        assert prefs.get("gradient_settings_side") == "docked"            # the last choice, remembered
        assert ed._side_state() == "docked"
    finally:
        w.close()


def test_marks_hide_and_come_back_with_a_tool(qapp, tmp_path):
    """Marks: hidden to judge the result, shown again as soon as a marks tool is used; M tapped toggles them, M held
    hides them only while it is down; never part of the draft."""
    from PySide6.QtCore import QEvent
    from PySide6.QtGui import QKeyEvent
    _test_project()
    w, settle = _window(tmp_path, (1250, 685))
    try:
        ed = w.tune.grad_editor
        if ed.view is None:
            pytest.skip("Gradient of the test project is not computed")
        mk = ed._ui["marks"]
        assert mk.isVisible() and mk.isChecked() and ed.shown
        mk.click()
        assert not ed.shown and not mk.isChecked() and not ed.has_draft()
        ed._ui["brush"].click()
        assert ed.shown and mk.isChecked()                               # a tool: the marks back
        c = ed.gv.canvas

        def key(kind):
            QApplication.sendEvent(c, QKeyEvent(kind, Qt.Key_M, Qt.NoModifier, "m"))
        key(QEvent.KeyPress)
        assert not ed.shown
        ed._m_down = (ed._m_down[0] - 1.0, ed._m_down[1])                # held a second
        key(QEvent.KeyRelease)
        assert ed.shown                                                  # held: back on release
        key(QEvent.KeyPress)
        key(QEvent.KeyRelease)
        assert not ed.shown                                              # tapped: hidden
        key(QEvent.KeyPress)
        key(QEvent.KeyRelease)
        assert ed.shown
    finally:
        w.close()


def test_each_method_offers_only_the_marks_it_reads(qapp, tmp_path):
    """The stage reads the samples with the polynomial and the RBF, the excluded areas with all three methods, and no
    sample with the automatic one: its Sample and Grid are greyed with their reason, its samples not drawn nor erased
    (the draft keeps them, they come back with a method that reads them); Marks stays (its areas are read)."""
    _test_project()
    w, settle = _window(tmp_path, (1250, 685))
    try:
        ed = w.tune.grad_editor
        if ed.view is None:
            pytest.skip("Gradient of the test project is not computed")
        assert ge.MARK_KINDS == {"polynomial": ("samples", "areas"), "rbf": ("samples", "areas"),
                                 "automatic": ("areas",)}
        W, H = ed._stack_size()
        ed.set_draft({"sample_points": [[W * 0.5, H * 0.5]]})
        for m in ("polynomial", "rbf", "automatic"):
            ed.set_draft({"method": m})
            settle(w)
            samples = m != "automatic"
            u = ed._ui
            assert u["points"].isEnabled() == samples and u["generate"].isEnabled() == samples, m
            assert u["objects"].isEnabled() == samples and u["nebula"].isEnabled() == samples, m
            assert u["brush"].isEnabled() and u["ellipse"].isEnabled() and u["eraser"].isEnabled(), m
            assert u["marks"].isEnabled(), m
            if not samples:
                assert "no samples" in u["points"].toolTip()
                ed.set_mode("eraser")
                f = ed._f(ed.gv.canvas)
                ed.press(ed.gv.canvas, W * 0.5 * f, H * 0.5 * f, Qt.LeftButton)
                ed.release(ed.gv.canvas)
                assert ed.value("sample_points") == [[W * 0.5, H * 0.5]]     # unseen: untouched
                ed.set_mode(None)
        ed.set_draft({"method": "rbf"})
        assert ed.value("sample_points") == [[W * 0.5, H * 0.5]] and ed._ui["points"].isEnabled()
        # the survey, simplified: no "Away from emission", a status line and Show survey in the settings panel
        from PySide6.QtWidgets import QCheckBox
        assert not [c for c in ed.page.findChildren(QCheckBox) if c.text() == "Away from emission"]
        assert ed._ui["survey"].text() == "Show survey"
        assert ed._ui["survey"].isEnabled() == ed.survey_here()
    finally:
        w.close()


def test_try_variant_keeps_the_original_and_takes_the_preview(qapp, tmp_path):
    """Try variant from a draft: the original's Gradient keeps its key, its state and its picture, no draft left on it;
    the variant shows the draft's preview (not computed) with the original's result as Before; an Undo of the variant
    gives the draft back to the original."""
    _test_project()
    w, settle = _window(tmp_path, (1250, 685))
    try:
        s, t, ed = w.s, w.tune, w.tune.grad_editor
        if ed.view is None:
            pytest.skip("Gradient of the test project is not computed")
        p = s.path

        def idle():
            settle(w)
            for _i in range(300):
                QApplication.processEvents()
                if not ed.busy and not ed._timer.isActive() and ed._held is None:
                    break
                time.sleep(0.02)
            settle(w)
        idle()
        before = (s.stage_key(p, "gradient"), s.state(p, "gradient"), ed.gv.image().identity())
        ed.set_draft({"degree": 3})
        idle()
        assert ed.result is not None
        ed._ui["variant"].click()
        idle()
        v = s.path
        assert v.pid != p.pid and s.state(v, "gradient") == "pending"
        assert ed.page.isVisible() and ed.gv.image().name == "Preview" and "not computed" in ed.tag_text()
        ed._set_before(True)
        idle()
        assert ed.gv.image().name.startswith("Result of")             # the original's result, to compare
        ed._set_before(False)
        s.select(index=s.index_of(p.pid), step="fond")
        t.request_refresh()
        idle()
        q = s.path
        assert q.pid == p.pid and not ed.has_draft(p.pid)
        assert (s.stage_key(q, "gradient"), s.state(q, "gradient"), ed.gv.image().identity()) == before
        s.undo()
        idle()
        s.select(index=s.index_of(p.pid), step="fond")
        t.request_refresh()
        idle()
        assert s.view(v.pid) is None and ed.has_draft(p.pid) and ed.value("degree") == 3
    finally:
        w.close()


_SCALED = r'''
import os, sys, time
from pathlib import Path
sys.path[:0] = [sys.argv[1], sys.argv[1] + "/tests", sys.argv[1] + "/tests/client"]
os.environ["ASTRO_GPU_SETTINGS"] = sys.argv[3]
from conftest import scratch_project
from test_inspect import inspect, open_window, settle
from PySide6.QtWidgets import QAbstractButton, QApplication, QLabel, QWidget
from astro_gpu.client import app as appmod, prefs
from astro_gpu.client.widgets import QSS
a = QApplication.instance() or QApplication([])
a.setStyleSheet(QSS + appmod.EXTRA_QSS)
prefs.load()
appmod.apply_language("fr")                                  # the longest names
w = open_window(scratch_project(Path(sys.argv[2])), size=(1250, 685), readonly=False)
inspect(w, "fond", index=0)
w.tune.dock.set_state("docked")
settle(w)
ed = w.tune.grad_editor
if ed.page is None or not ed.page.isVisible():
    print("SKIP no page")
    sys.exit(0)
bad = []
for x in ed.page.findChildren(QWidget):
    if not x.isVisible() or x is ed.drawer or ed.drawer.isAncestorOf(x) or ed.gv.isAncestorOf(x):
        continue
    par = x.parentWidget()
    if par is not None and not par.rect().contains(x.geometry()):
        bad.append(("outside its parent", getattr(x, "text", lambda: type(x).__name__)()))
    r = x.rect().translated(x.mapTo(w, x.rect().topLeft()))
    if not w.rect().contains(r):
        bad.append(("outside the window", getattr(x, "text", lambda: type(x).__name__)()))
    if isinstance(x, (QAbstractButton, QLabel)) and not (isinstance(x, QLabel) and x.wordWrap()) and \
            x.width() < x.sizeHint().width() - 1:
        bad.append(("cut", x.text()))
print("BAD", bad if bad else "none")
w.close()
'''


@pytest.mark.parametrize("scale", ["1", "1.25", "1.5"])
def test_nothing_on_the_page_is_cut_at_any_display_scaling(tmp_path, scale):
    """The application's own style, the French names (the longest), a 1250 x 685 window at 100, 125 and 150 % display
    scaling: every control of the page whole, inside its parent and inside the window (a fixed width once cut the
    tools in half on a real launch)."""
    import subprocess
    import sys
    from pathlib import Path
    _test_project()
    root = str(Path(__file__).resolve().parents[2])
    script = tmp_path / "scaled.py"
    script.write_text(_SCALED, encoding="utf-8")
    env = dict(os.environ, QT_SCALE_FACTOR=scale, QT_QPA_PLATFORM="offscreen", PYTHONPATH=root,
               CUDA_VISIBLE_DEVICES="")
    out = subprocess.run([sys.executable, str(script), root, str(tmp_path / "proj"), str(tmp_path / "s.json")],
                         capture_output=True, text=True, env=env, timeout=300)
    lines = [ln for ln in out.stdout.splitlines() if ln.startswith(("BAD", "SKIP"))]
    assert lines, out.stdout[-2000:] + out.stderr[-2000:]
    if lines[0].startswith("SKIP"):
        pytest.skip("Gradient of the test project is not computed")
    assert lines[0] == "BAD none", lines[0]


def test_a_long_preview_is_said_at_the_top_of_the_image(ed):
    """A preview running longer than BUSY_SHOWN_MS is said on the image ("Computing preview…", its bar along the top);
    a fast one is not (no flicker); once current, the tag gives the preview's time."""
    ed.set_draft({"method": "rbf", "samples": 32})                   # slow: over a second on the fixture
    end = time.monotonic() + 10
    while not ed._busy_shown and time.monotonic() < end:
        QApplication.processEvents()
        time.sleep(0.01)
    assert ed.busy and ed._busy_shown and ed.tag_text().startswith("Computing preview")
    assert ed._anim.isActive()
    wait(ed, 60)
    assert not ed._busy_shown and not ed._anim.isActive() and ed.tag_text().startswith("Draft preview")
    ed.set_draft({"method": "polynomial", "samples": 16})            # fast: never said
    seen = []
    end = time.monotonic() + 10
    while (ed.busy or ed._timer.isActive()) and time.monotonic() < end or not seen:
        QApplication.processEvents()
        seen.append(ed._busy_shown)
        time.sleep(0.005)
        if len(seen) > 2000:
            break
    wait(ed)
    assert not any(seen) or ge.BUSY_SHOWN_MS < ed.result_ms


def test_an_excluded_area_leaves_its_samples_red_and_fades(ed):
    """A sample under a painted area or in an ellipse is not used (drawn red, said on hover); the area itself shows
    filled only a moment after it is painted, then fades: never a red block over the image."""
    ed.set_draft({"sample_points": [[100.0, 100.0], [500.0, 300.0], [800.0, 500.0]],
                  "exclude_brush": [[500.0, 300.0, 40.0, 1.0]], "exclude_ellipses": [[800.0, 500.0, 30.0, 20.0, 0.0]]})
    assert list(ed._excluded()) == [False, True, True]
    ed.set_draft({"exclude_brush": [[500.0, 300.0, 40.0, 1.0], [500.0, 300.0, 50.0, 0.0]]})     # erased again
    assert list(ed._excluded()) == [False, False, True]
    ed._fade_t0 = time.monotonic()
    assert ed._fade_alpha() > 0.5
    ed._fade_t0 = time.monotonic() - 2 * ge.FADE_MS / 1000
    assert ed._fade_alpha() == 0.0


def test_the_gradient_view_s_aids_never_reach_another_view(qapp, tmp_path):
    """The survey map and the marks shown in the Gradient view stay there: Denoise's compare with "Gradient removed"
    (Split) is pixel for pixel the same with the survey shown or hidden in the Gradient view."""
    from PySide6.QtGui import QImage
    _test_project()
    w, settle = _window(tmp_path, (1250, 685))
    try:
        s, t, ed = w.s, w.tune, w.tune.grad_editor
        if ed.view is None:
            pytest.skip("Gradient of the test project is not computed")

        def go(step):
            s.select(step=step)
            t.select(step=step)
            t.request_refresh()
            settle(w)
            settle(w)

        def denoise_split():
            go("debruitage")
            v = t.view
            i = next(k for k, x in enumerate(v._targets) if x.name == "Gradient removed")
            v.set_compare_mode("split")
            v.set_compare(i)
            assert v.wait(60)
            settle(w)
            img = v.canvas.grab().toImage().convertToFormat(QImage.Format.Format_RGB888)
            raw = np.frombuffer(img.constBits(), np.uint8, img.sizeInBytes()).reshape(img.height(), img.bytesPerLine())
            return raw[:, : img.width() * 3].copy()
        a0 = denoise_split()
        go("fond")
        if not ed.survey_here():
            pytest.skip("no H-alpha survey of the test project's field on this computer")
        ed._set_survey_shown(True)
        ed._set_shown(True)
        settle(w)
        a1 = denoise_split()
        assert a0.shape == a1.shape and (a0 == a1).all()
    finally:
        w.close()
