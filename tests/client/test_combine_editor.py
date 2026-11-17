"""Inspect > a combination tuned as a draft (client/combineedit.py), as the Gradient block is: the draft bar, the draft
never touches the project until Apply, Apply is one step and the block outdated, Try variant, the leave question, the
preview computed with the stage's own functions, the eyes, the palettes, the French names, the block's title following
its recipe.

The first tests drive the editor on made-up filters with a stand-in model (nothing on disk); the window tests read the
owner's M27 (S-H-O) and IC 443 (H-O) projects through a scratch copy of their definition (nothing computed, nothing
written into them)."""
from __future__ import annotations

import copy
import os
import time

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
pytest.importorskip("PySide6")
from PySide6.QtCore import Signal  # noqa: E402
from PySide6.QtWidgets import QApplication, QVBoxLayout, QWidget  # noqa: E402

from astro_gpu.client import combineedit as ce  # noqa: E402
from astro_gpu.stages import combine as C  # noqa: E402
from astro_gpu.stages import recipes  # noqa: E402

H, W = 120, 180
SHO = ("SII", "Ha", "OIII")


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def filters(seed=3, H=H, W=W):
    """Three filters on skies of their own level and noise, a nebula of a different strength in each, a few stars."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W]
    neb = np.exp(-((xx - W / 2) ** 2 + (yy - H / 2) ** 2) / (2 * 25.0 ** 2)).astype(np.float32)
    off, amp, noise = {"SII": 0.3, "Ha": 2.5, "OIII": 0.8}, {"SII": 0.02, "Ha": 0.3, "OIII": 0.12}, \
        {"SII": 0.002, "Ha": 0.0015, "OIII": 0.001}
    ch = {r: (off[r] / 1000 + amp[r] * neb + rng.normal(0, noise[r], (H, W))).astype(np.float32) for r in SHO}
    for _i in range(25):
        y, x = rng.integers(3, H - 3), rng.integers(3, W - 3)
        for r in SHO:
            ch[r][y - 1:y + 2, x - 1:x + 2] += 0.2
    return ch


def small_of(ch) -> dict:
    return {"chans": {r: v.copy() for r, v in ch.items()}, "valid": np.ones((H, W), bool), "k": 2, "H": 2 * H,
            "W": 2 * W, "inputs": [{"role": r, "channel": "L", "full_scale": 1.0} for r in SHO]}


class WF:
    def __init__(self, pid, source, params):
        self.id, self.source, self.params = pid, source, params


class View:
    def __init__(self, pid, name, wf):
        self.pid, self.name, self._wf = pid, name, wf

    @property
    def source(self):
        return self._wf.source


class Proj:
    def __init__(self):
        self.workflows = []

    def workflow(self, pid):
        return next(w for w in self.workflows if w.id == pid)


class Model:
    """What the editor asks of the project model; records the changes made."""

    def __init__(self, kind="sho", image="output", params=None):
        self.p = Proj()
        ins = [{"pid": 10 + n, "role": r, "channel": "L", "image": image} for n, r in enumerate(
            SHO if kind == "sho" else ("Ha", "OIII"))]
        wf = WF(1, {"fusion": {"type": kind, "inputs": ins}}, {f"combine_{kind}": dict(params or {})})
        self.p.workflows.append(wf)
        self.stage = f"combine_{kind}"
        self.path = View(1, "S-H-O" if kind == "sho" else "H-O", wf)
        self.paths = [self.path]
        self.sel_step = "fusion"
        self.changes, self.variants, self.logs = [], [], []

    def state(self, p, stage):
        return "done"

    def view(self, pid):
        return next((q for q in self.paths if q.pid == pid), None)

    def scope_paths(self):
        return [self.path]

    def stage_key(self, q, st):
        return None

    def all_stages(self, q):
        return [self.stage]

    def apply_change(self, paths, ch):
        self.changes.append(ch)
        for q in paths:
            st = q._wf.params.setdefault(self.stage, {})
            for k, v in ch["values"].items():
                if k == "weights":
                    st["weights"] = {**(st.get("weights") or {}), **v}
                else:
                    st[k] = v
        return []

    def add_variant(self, p, ch, title):
        wf = WF(len(self.paths) + 1, copy.deepcopy(p._wf.source), copy.deepcopy(p._wf.params))
        self.p.workflows.append(wf)
        q = View(wf.id, title, wf)
        self.paths.append(q)
        self.variants.append((p.pid, ch, title))
        self.apply_change([q], ch)
        return q

    def log(self, template, **k):
        self.logs.append(template.format(**k))


class Tune(QWidget):
    toasted = Signal(str)

    def __init__(self, model):
        super().__init__()
        self.s = model
        self.center = QVBoxLayout(self)
        self.panel = QWidget(self)
        self.head_box = None
        self.dock = None
        self.view, self.views = None, []
        self._center_sig = None
        self.refreshed = 0
        self.made = []

    def _clear_center(self):
        pass

    def request_refresh(self):
        self.refreshed += 1

    def _variant_made(self, pid, orig):
        self.made.append((pid, orig))


class Node:
    cut = False


def make(monkeypatch, qapp, **kw):
    ch = filters()
    monkeypatch.setattr(ce, "_reduced_for", lambda project, pid, stage, spec: small_of(ch))
    monkeypatch.setattr(ce, "_applied_for", lambda project, pid, stage, k, settings=None: None)
    m = Model(**kw)
    t = Tune(m)
    t.resize(1400, 900)
    e = t.comb = ce.CombineEditor(t)
    return e, t, ch


@pytest.fixture()
def ed(qapp, monkeypatch):
    e, t, _ch = make(monkeypatch, qapp)
    assert e.center(t, Node(), t.s.path)
    wait(e)
    return e


def wait(e, timeout=30.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        QApplication.processEvents()
        if not e.busy and e._want is None:
            QApplication.processEvents()
            if not e.busy:
                return
        time.sleep(0.005)
    raise AssertionError("the preview did not come")


def expected(e, ch, off=(), solo=None):
    """The preview as the stage's own functions make it on the same reduced filters."""
    sm = small_of(ch)
    spec = e._spec()
    cls = ce.stage_class(e._stage())
    p = ce.params_of(cls, spec, e._stored(), e.values())
    bal = ce.balance(sm, e._stage(), spec, e._balance_used())
    return ce.output_render(ce.mixed(sm, bal, e._stage(), spec, p, off, solo))


# ---------------------------------------------------------------------------------------------- the stage's own result
def test_the_preview_mix_is_the_stage_s_own_on_the_same_inputs(tmp_path):
    """The reduced inputs, balanced and mixed by the editor's pieces, are the combination's run on those inputs,
    pixel for pixel: the stage's steps, its weights, its hues (here a hue, a weight and each balance)."""
    from tests.test_blocks43 import _output
    from tests.test_combine import _run
    ch = filters()
    ins = []
    for r in SHO:
        _output(tmp_path / r, np.repeat(ch[r][None], 3, 0))
        ins.append({"role": r, "channel": "L", "dir": str(tmp_path / r), "pid": 1, "full_scale": 1.0})
    spec = recipes.legacy_spec("sho")
    sm = ce.reduced_inputs(ins, spec, pixels=10 ** 9)                 # the copy at full size: the run's own input
    assert sm["k"] == 1
    for how in ("signal", "sky"):
        prm = C.CombineSHO.Params(inputs=ins, balance=how, hue_ha=45.0, weights={"OIII": 1.6})
        out = _run(tmp_path, C.CombineSHO, prm, f"run_{how}")
        bal = ce.balance(sm, "combine_sho", spec, how)
        got = ce.mixed(sm, bal, "combine_sho", spec, prm)
        assert np.allclose(got, np.load(out / "drz_full.npy"), atol=1e-6), how
        assert set(bal["notes"]) == {"SII", "OIII"} and all(g > 0 for g, _w in bal["notes"].values())


def test_the_preview_is_the_stage_result_through_the_output_s_default_render(ed):
    """What the editor shows is the stage's mix on its reduced copy through the Output's default render."""
    ch = filters()
    assert ed.result is not None and ed.result.shape == (3, H, W)
    assert float(np.abs(ed.result - expected(ed, ch)).max()) < 1e-5
    ed.set_hue("OIII", 207.0)
    wait(ed)
    assert float(np.abs(ed.result - expected(ed, ch)).max()) < 1e-5
    shown = ed.result
    ed.set_hue("OIII", 230.0)                          # hues with no grey between them: mixed all the same
    wait(ed)
    assert ed.result is not shown and not ed.error
    assert float(np.abs(ed.result - expected(ed, ch)).max()) < 1e-5


# ---------------------------------------------------------------------------------------------- the draft
def test_the_block_shows_the_draft_bar_its_palette_cards_and_balance(ed):
    from astro_gpu.client.blockdraft import DraftBar
    u = ed._ui
    assert isinstance(u["bar"], DraftBar) and u["bar"].isVisibleTo(ed.page)
    assert u["bar"].b_apply.text() == "Apply" and not u["apply"].isEnabled()
    assert [u["palette"].itemText(i) for i in range(u["palette"].count())] == ["Hubble (S-H-O)", "Custom"]
    assert u["palette"].currentData() == "hubble"
    for r in SHO:
        assert f"hue_{r}" in u and f"int_{r}" in u and f"eye_{r}" in u and f"thumb_{r}" in u
        assert u[f"thumb_{r}"].pixmap() is not None and not u[f"thumb_{r}"].pixmap().isNull()
    # no run's value yet: the reduced copy's estimate, said so; what the reason means on hover
    assert u["gain_SII"].text().startswith("≈ ×") and "from the preview" in u["gain_SII"].text()
    assert u["gain_Ha"].text() == "reference" and "estimate" in u["gain_SII"].toolTip()
    assert "On sky:" in u["gain_SII"].toolTip()                                  # the default balance
    assert "on sky" in u["gain_OIII"].text()
    assert "balance_signal" in u and "balance_sky" in u and "stars_neutral" not in u
    assert u["bar"].what.text().find("Applied settings") >= 0


def test_a_hue_change_makes_a_draft_and_leaves_the_project_until_apply(ed):
    m, p = ed.s, ed.s.path
    before = copy.deepcopy(p._wf.params)
    ed._ui["hue_Ha"].setValue(60)                       # a key or the wheel: one step
    wait(ed)
    assert ed.has_draft() and ed.value("hue_ha") == 60.0
    assert m.changes == [] and p._wf.params == before
    assert "Draft · 1 change" in ed._ui["bar"].what.text() and ed._ui["apply"].isEnabled()
    assert ed._ui["palette"].currentData() == ""      # no palette's hues any more: Custom
    ed.set_hue("Ha", C.HUBBLE_HUES[1])                  # back to the applied value: no draft
    assert not ed.has_draft()


def test_a_drag_follows_the_slider_and_is_one_step(ed):
    sl = ed._ui["int_OIII"]
    sl.setSliderDown(True)
    ed.begin_drag()
    for v in (round(ce.intensity_u(x) * ce.INTENSITY_POSITIONS) for x in (1.1, 1.2, 1.3, 1.4)):   # (its rail's places)
        sl.setValue(v)
        QApplication.processEvents()
    sl.setSliderDown(False)
    ed.end_drag()
    wait(ed)
    assert ed.weight("OIII") == pytest.approx(1.4) and len(ed._draft().undo) == 1
    assert float(np.abs(ed.result - expected(ed, filters())).max()) < 1e-5     # the last value, shown
    ed.undo()
    assert not ed.has_draft()


def test_a_press_on_the_rail_then_a_drag_is_one_step(ed):
    """The hand pass: one drag of a hue slider made two Undo steps (205, then 267): the press on the rail moved the
    handle as a step of its own, the drag that followed another. The gesture, press to release, is one step, as in
    Post."""
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QStyleFactory
    if "windows11" not in QStyleFactory.keys():
        pytest.skip("the Windows 11 style is not on this machine")
    app = QApplication.instance()
    was = app.style().name()
    app.setStyle("windows11")                     # Windows' own: a press on the rail takes the handle there
    try:
        sl = ed._ui["hue_OIII"]
        ed.page.resize(900, 700)
        ed.page.show()
        QApplication.processEvents()
        y = sl.height() // 2
        a, b = QPoint(int(sl.width() * 0.2), y), QPoint(int(sl.width() * 0.4), y)  # O III at 202°: the rail, away
        v0 = sl.value()
        QTest.mousePress(sl, Qt.LeftButton, Qt.NoModifier, a)
        v1 = sl.value()
        QTest.mouseMove(sl, b)
        QTest.mouseRelease(sl, Qt.LeftButton, Qt.NoModifier, b)
        wait(ed)
        assert v0 != v1 != sl.value() and ed.has_draft() and len(ed._draft().undo) == 1
    finally:
        app.setStyle(was)
    ed.undo()
    assert not ed.has_draft()


def test_apply_is_one_change_of_the_block(ed):
    m = ed.s
    got = []
    m.apply_as = lambda ch, how="replace", paths=None, title="": got.append((ch, how, [q.pid for q in paths]))
    ed.set_hue("SII", 0.0)
    ed.set_intensity("OIII", 1.5)
    ed.set_draft({"balance": "signal"})
    ed.apply()
    (ch, how, pids), = got
    assert how == "replace" and pids == [1] and ch["kind"] == "combine"
    assert ch["values"] == {"hue_sii": 0.0, "weights": {"OIII": 1.5}, "balance": "signal"}
    assert not ed.has_draft() and ed.tv.refreshed


def test_try_variant_puts_the_draft_on_a_variant(ed):
    m, p = ed.s, ed.s.path
    ed.set_hue("OIII", 180.0)
    ed.try_variant()
    (pid, ch, title), = m.variants
    assert pid == 1 and ch["kind"] == "combine" and ch["values"] == {"hue_oiii": 180.0} and "S-H-O" in title
    assert p._wf.params[m.stage] == {}                     # the workflow keeps its settings
    q = m.paths[-1]
    assert q._wf.params[m.stage]["hue_oiii"] == 180.0 and ed.tv.made == [(q.pid, 1)] and not ed.has_draft(1)


@pytest.mark.parametrize("answer", ["stay", "discard", "apply", "variant"])
def test_leaving_with_a_draft_asks_and_each_answer_does_what_it_says(ed, monkeypatch, answer):
    from astro_gpu.client.blockdraft import DRAFT_ANSWER_ENV, settle
    m, p = ed.s, ed.s.path
    ed.set_hue("Ha", 50.0)
    block, lines, _cost = ed.draft_pending()
    assert block.startswith("S-H-O") and lines and lines[0] == "Hα hue of “S-H-O”: 31° → 50°"     # named as every step
    monkeypatch.setenv(DRAFT_ANSWER_ENV, answer)
    assert settle(ed, None) == (answer != "stay")
    if answer == "stay":
        assert ed.has_draft() and m.changes == []
    elif answer == "discard":
        assert not ed.has_draft() and m.changes == []
    elif answer == "apply":
        assert p._wf.params[m.stage]["hue_ha"] == 50.0 and not ed.has_draft()
    else:
        assert m.variants and not ed.has_draft(p.pid) and ed.draft_pending() is None


def test_the_drafts_steps_are_in_the_window_s_one_order_of_undo(ed):
    from astro_gpu.client import journal
    assert ed.history_top("undo") is None
    before = journal.tick()
    ed.set_hue("Ha", 40.0)
    seq, pid = ed.history_top("undo")
    assert seq > before and pid == 1
    assert ed.undo(pid) and not ed.has_draft() and ed.history_top("redo") is not None
    assert ed.redo() and ed.value("hue_ha") == 40.0


def test_an_eye_off_takes_the_filter_out_of_the_mix_only_in_the_view(ed):
    ch = filters()
    ed._ui["eye_OIII"].click()
    wait(ed)
    assert "OIII" in ed.off and not ed.has_draft()
    assert float(np.abs(ed.result - expected(ed, ch, off={"OIII"})).max()) < 1e-5
    assert float(np.abs(ed.result - expected(ed, ch)).max()) > 0.01        # it did change the picture
    ed.set_solo("Ha")                                     # its thumbnail clicked: that filter alone
    wait(ed)
    assert float(np.abs(ed.result - expected(ed, ch, off={"OIII"}, solo="Ha")).max()) < 1e-5
    ed.set_solo(None)
    ed._ui["eye_OIII"].click()
    wait(ed)
    assert not ed.off and float(np.abs(ed.result - expected(ed, ch)).max()) < 1e-5


def test_a_palette_sets_the_hues_in_one_step(qapp, monkeypatch):
    e, t, _ch = make(monkeypatch, qapp, kind="ho")
    assert e.center(t, Node(), t.s.path)
    wait(e)
    pal = e._ui["palette"]
    assert [pal.itemData(i) for i in range(pal.count())] == ["hoo", ""] and pal.currentData() == "hoo"
    e.set_hue("OIII", 200.0)
    assert pal.currentData() == ""
    i = pal.findData("hoo")
    pal.setCurrentIndex(i)
    pal.activated.emit(i)
    assert e.value("hue_ha") == 0.0 and e.value("hue_oiii") == 180.0 and not e.has_draft()
    e2, t2, _c = make(monkeypatch, qapp, params={"hue_sii": 0.0, "hue_ha": 120.0, "hue_oiii": 240.0})
    assert e2.center(t2, Node(), t2.s.path)
    assert e2._ui["palette"].currentData() == ""
    assert e2.set_palette("hubble") and len(e2._draft().undo) == 1
    assert [e2.value(n) for n in ("hue_sii", "hue_ha", "hue_oiii")] == [round(h, 2) for h in C.HUBBLE_HUES]


def test_the_stars_combination_shows_only_its_stars(qapp, monkeypatch):
    e, t, _ch = make(monkeypatch, qapp, image="stars")
    assert e.center(t, Node(), t.s.path)
    wait(e)
    u = e._ui
    assert "stars_neutral" in u and "stars_palette" in u
    assert not any(k.startswith(("hue_", "int_", "balance_")) for k in u) and "palette" not in u
    assert u["stars_neutral"].property("on") is True
    u["stars_palette"].clicked.emit()
    assert e.value("stars") == "palette" and e.has_draft()
    wait(e)
    assert float(np.abs(e.result - expected(e, filters())).max()) < 1e-5


def test_french_names(qapp, monkeypatch):
    from astro_gpu.client import i18n
    i18n.set_language("fr")
    try:
        e, t, _ch = make(monkeypatch, qapp)
        assert e.center(t, Node(), t.s.path)
        wait(e)
        from PySide6.QtWidgets import QAbstractButton, QLabel
        texts = {x.text() for x in e.page.findChildren(QLabel) + e.page.findChildren(QAbstractButton) if x.text()}
        for fr in ("Appliquer", "Palette", "Filtres", "Teinte", "Intensité", "Équilibre", "Sur le signal",
                   "Sur le ciel", "Brouillon", "Appliqué", "Rideau", "référence"):
            assert any(fr in x for x in texts), (fr, sorted(texts))
        for en in ("Hue", "Intensity", "Balance", "On signal", "On sky", "Filters", "reference", "Applied settings"):
            assert not any(x == en or x.startswith(en + " ") for x in texts), en
        e.set_hue("Ha", 50.0)
        assert "Brouillon · 1 changement" in e._ui["bar"].what.text()
        assert "Teinte Hα" in e.draft_change()["what"] or "teinte" in e.draft_change()["what"]
    finally:
        i18n.set_language("en")


# ---------------------------------------------------------------------------------------------- the graph's side
def test_a_recipe_s_title_in_a_workflow_name_follows_its_recipe():
    from astro_gpu.graphcore import combos
    assert combos.retitled("S-H-O · Stars", "H-O") == "H-O · Stars"
    assert combos.retitled("S-H-O 2 · Stars", "H-O") == "H-O 2 · Stars"
    assert combos.retitled("S-H-O", "S-H-O") == "S-H-O" and combos.retitled("M27 colours", "H-O") == "M27 colours"
    assert combos.title_stem("H-O · Stars") == "H-O" and combos.title_stem("Mine") == ""


# ---------------------------------------------------------------------------------------------- the owner's projects
REAL = {"m27": "F:/astro_essais/tutoriels_ref/MONO/m27", "ic443": "F:/astro_essais/tutoriels_ref/MONO/ic443"}


def _real_window(tmp_path, name, unstore: tuple[str, str] | None = None):
    """A scratch copy of the owner's project (its store read in place, no link), opened read only. ``unstore``:
    (a stored setting, its value) taken out of the copy's definition first, as if the project had stored nothing."""
    from pathlib import Path

    from conftest import scratch_project
    from test_inspect import Harness

    from astro_gpu.client import app as appmod
    from astro_gpu.client import prefs
    src = Path(REAL[name])
    if not (src / "project.astroproj").is_file():
        pytest.skip(f"the project {name} is not here")
    os.environ["ASTRO_GPU_SETTINGS"] = str(tmp_path / "settings.json")
    prefs.load()
    appmod.apply_language("en")

    class Whole(Harness):
        def make_model(self):
            appmod.Main.make_model(self)
    w = Whole(readonly=True)
    w.resize(1500, 900)
    w.show()
    folder = scratch_project(tmp_path / name, src)
    if unstore:
        import json

        def drop(x):
            if isinstance(x, dict):
                if x.get(unstore[0]) == unstore[1]:
                    x.pop(unstore[0])
                for v in x.values():
                    drop(v)
            elif isinstance(x, list):
                for v in x:
                    drop(v)
        f = folder / "project.astroproj"
        d = json.loads(f.read_text(encoding="utf-8"))
        drop(d)
        f.write_text(json.dumps(d, indent=1), encoding="utf-8")
    w.open_project(folder)
    for _i in range(400):
        if w.s.ready:
            break
        w.worker.wait(5)
        time.sleep(0.05)
    assert w.s.ready, w.s.error
    return w


def _open_combination(w):
    from test_inspect import settle

    from astro_gpu.client.graphnode import combine_kind
    s = w.s
    q = next(q for q in s.paths if combine_kind(q) and
             not all(i.get("image") == "stars" for i in q.source["fusion"]["inputs"]))
    s.select(index=s.index_of(q.pid), step="fusion")
    w.set_mode(1)
    ed = w.tune.comb_editor
    settle(w)
    wait(ed, 120)
    settle(w)
    return q, ed


def test_on_m27_the_editor_applies_in_one_step_and_tries_a_variant(qapp, tmp_path):
    from test_inspect import settle
    w = _real_window(tmp_path, "m27")
    try:
        s = w.s
        q, ed = _open_combination(w)
        assert ed.page_shown() and ed._ui["bar"].isVisible() and not w.tune.panel.isVisible()
        assert ed.result is not None and ed.small["k"] >= 2           # the reduced copy of the inputs
        st = ed._stage()
        key = s.stage_key(q, st)
        state = s.state(q, st)
        ed.set_intensity("SII", 1.5)
        assert s.stage_key(q, st) == key and s.state(q, st) == state  # a draft: the project as it was
        n = len(s._undo)
        ed.apply()
        settle(w)
        assert len(s._undo) == n + 1 and not ed.has_draft()
        # Post shows this S-H-O by its master layer: an intensity is the layer's (client/postfusion.py), the block's
        # key and state as they were, nothing to run
        assert s.stage_key(q, st) == key and s.state(q, st) == state
        assert s.master_layer(q.pid)["tune"]["weights"]["SII"] == 1.5
        assert (s.p.workflow(q.pid).params.get(st) or {}).get("weights") is None
        s.undo()
        settle(w)
        assert s.stage_key(q, st) == key and "tune" not in s.master_layer(q.pid)
        # Try variant of a setting the engine computes (the balance): a second S-H-O of the same filters, its own
        # workflow, the original untouched (a master layer's hue or intensity goes to a new version of Post instead)
        q, ed = _open_combination(w)
        ins = [(i["pid"], i["role"], i.get("image")) for i in q.source["fusion"]["inputs"]]
        before = {x.pid for x in s.paths}
        other = "sky" if ed.value("balance") == "signal" else "signal"
        ed.set_draft({"balance": other})
        ed.try_variant()
        settle(w)
        assert len(s.paths) == len(before) + 1 and len(s._undo) == n + 1
        v = next(x for x in s.paths if x.pid not in before)
        assert [(i["pid"], i["role"], i.get("image")) for i in v.source["fusion"]["inputs"]] == ins
        assert (s.p.workflow(v.pid).params.get(st) or {}).get("balance") == other
        assert s.stage_key(q, st) == key                               # the original untouched
        s.undo()
        settle(w)
        assert {x.pid for x in s.paths} == before and s.view(v.pid) is None
    finally:
        w.close()


def test_on_ic443_the_h_o_block_is_titled_h_o(qapp, tmp_path):
    """IC 443's combinations were placed as S-H-O and made H-O: their names still say S-H-O. The block's title, the
    graph's and the editor's, follow the recipe; a recipe change renames nothing stored."""
    from astro_gpu.graphcore import rules
    w = _real_window(tmp_path, "ic443")
    try:
        s = w.s
        q, ed = _open_combination(w)
        # the editor's title is the block's name as the path bar and the graph show it (one name everywhere): H-O,
        # numbered among the project's H-O blocks
        from astro_gpu.client.graphnode import node_title
        assert ed._ui["title"].text() == node_title(s, ed.node) and ed._ui["title"].text().startswith("H-O")
        g = s._g
        titles = {rules.title(g, r.blocks[0]) for r in g.rows if r.blocks and g.blocks[r.blocks[0]].kind == "combine"}
        assert titles == {"H-O", "H-O · Stars"}, titles
        # what the project stores stays as it is (an existing project opens, converts and keys the same): its names
        # are shown with the recipe's title, never written so
        assert {x.name for x in s.paths if x.source and x.source.get("fusion")} == {"S-H-O", "S-H-O · Stars"}
        stars = next(x for x in s.paths if x.name.endswith("· Stars"))
        n = len(s._undo)
        said = s.set_combination_type(stars, "rgb")          # its inputs are S II-less: Hα and O III go
        assert len(s._undo) == n + 1 and "removed" in said and "Hα" in said and "O III" in said
        assert "renamed" not in said                          # no rename the user did not make
        assert s.view(stars.pid).name == "S-H-O · Stars"      # stored as it was, shown with the recipe's title
        jb = next(r.blocks[0] for r in s._g.rows if r.blocks and r.name == "S-H-O · Stars")
        assert rules.title(s._g, jb) == "RGB · Stars"
        s.undo()
        assert s.view(stars.pid).name == "S-H-O · Stars"
    finally:
        w.close()


def test_the_graph_panel_s_settings_of_a_combination_are_one_draft(qapp, tmp_path):
    """The reviewer's test: the weight applied at once and left the header saying Up to date, the recipe applied at
    once while a hue waited for Apply. Every setting of the panel now goes to the block's draft (the project as it
    was until Apply, one step), and the waiting block is said on the status line."""
    from test_inspect import settle

    from astro_gpu.client.panel import Panel
    # the project stores the Balance On signal, the default it was made with: that choice is not computed in its store,
    # whose combination was computed at that default; taken out, the block has its result, as the test needs
    w = _real_window(tmp_path, "ic443", unstore=("balance", "signal"))
    try:
        s = w.s
        q, ed = _open_combination(w)
        st = ed._stage()
        key = s.stage_key(q, st)
        pn = Panel(s, w.steps)
        node = s.block_map.node_of(q, "fusion")
        pn.refresh(node)
        assert pn.findChild(QWidget, "adjust") is not None    # Adjust: the editor in Inspect
        # the panel sums the settings up (Adjust opens them in Inspect): no intensity box of its own (the family's one
        # layout); a change of an intensity made by the panel's draft is still one draft with the recipe's
        from PySide6.QtWidgets import QDoubleSpinBox
        assert not [x for x in pn.findChildren(QDoubleSpinBox) if x.objectName() == "intensity"]
        pn.propose_weight(q, "OIII", 0.5, 1.0)
        assert s.pending_change and s.stage_key(q, st) == key
        pn.propose_recipe_type(q, "sho", "ho")
        d = s.pending_change
        assert d and len(d) == 2 and s.stage_key(q, st) == key  # a draft: nothing changed yet
        pn.refresh(node)                                      # the draft's recipe: its inputs shown, S II too
        from PySide6.QtWidgets import QLabel
        assert "S II" in {x.text().replace("<b>", "").replace("</b>", "") for x in pn.findChildren(QLabel)}
        what = " ".join(pn.draft_change(d).get("whats") or [])
        assert "O III intensity of “H-O”: 1.00 → 0.50" in what and "Recipe: H-O → S-H-O" in what
        n = len(s._undo)
        assert pn.draft_apply()
        settle(w)
        assert len(s._undo) == n + 1
        f = s.p.workflow(q.pid)
        assert f.source["fusion"]["type"] == "sho"
        # an intensity of a combination Post shows by its master layer is the layer's (client/postfusion.py)
        lay = s.master_layer(q.pid)
        assert (lay["tune"]["weights"]["OIII"] if lay is not None else
                (f.params.get("combine_sho") or {}).get("weights", {}).get("OIII")) == 0.5
        w._update_actions() if hasattr(w, "_update_actions") else None
        assert w.waiting_text() == "1 block waits for an input"      # S II to wire: said, never "all up to date"
        assert w.run_refusal(type("C", (), {"jobs": 0})()) == "1 block waits for an input"
        s.undo()
        settle(w)
        assert s.stage_key(q, st) == key
    finally:
        w.close()


def test_file_open_given_the_project_file_opens_its_folder(qapp, tmp_path):
    from pathlib import Path

    from conftest import scratch_project
    from test_inspect import Harness
    src = Path(REAL["ic443"])
    if not (src / "project.astroproj").is_file():
        pytest.skip("the project ic443 is not here")
    folder = scratch_project(tmp_path / "p", src)
    w = Harness(readonly=True)
    try:
        w.show()
        w.banner.say("No project in this folder: x")
        assert w.open_project(folder / "project.astroproj") is not False
        assert w.project is not None and w.project.root == folder and not w.banner.isVisible()
    finally:
        w.close()


# ---------------------------------------------------------------------------------------------- round 2
def test_the_draft_bar_keeps_its_height_whatever_the_draft_holds(ed):
    """A narrow draft bar (this editor's, Gradient's) never grows: it pushed the cards down between a press and a click
    (a click on On sky landed on On signal)."""
    from astro_gpu.client.blockdraft import DraftBar
    bar = ed._ui["bar"]
    QApplication.processEvents()
    h0, y0 = bar.sizeHint().height(), ed._ui["balance_sky"].mapTo(ed.page, ed._ui["balance_sky"].rect().topLeft())
    ed.set_hue("SII", 10.0)
    ed.set_hue("OIII", 205.0)
    ed.set_intensity("Ha", 1.7)
    ed.set_draft({"balance": "signal"})
    QApplication.processEvents()
    assert bar.sizeHint().height() == h0
    assert ed._ui["balance_sky"].mapTo(ed.page, ed._ui["balance_sky"].rect().topLeft()) == y0
    assert "Draft · 4 changes" in bar.what.text() and "S II hue" in bar.what.toolTip()
    g = DraftBar(narrow=True)
    g.resize(300, 120)
    g.show_draft(["Draft · 1 change"])
    h1 = g.sizeHint().height()
    g.show_draft(["Draft · 3 changes", "Gradient draft: RBF, 40 samples, exclusions", "another line", "and one more"],
                 "Computes again · about 2 min")
    assert g.sizeHint().height() == h1                      # Gradient's bar: the same


def test_on_a_block_without_result_every_change_is_made_at_once(qapp, monkeypatch):
    """The rule of every control: a block with a result (computed or Outdated) takes a draft; without one (never
    computed, waiting for an input), the change is made at once, one step."""
    e, t, _ch = make(monkeypatch, qapp)
    t.s.state = lambda p, stage: "pending"
    assert e.center(t, Node(), t.s.path)
    wait(e)
    e.set_hue("OIII", 205.0)
    assert not e.has_draft() and [c["kind"] for c in t.s.changes] == ["combine"]
    assert t.s.path._wf.params[t.s.stage]["hue_oiii"] == 205.0
    sl = e._ui["int_SII"]                                   # a drag: once, when it is let go
    sl.setSliderDown(True)
    e.begin_drag()
    for v in (22, 26):
        sl.setValue(v)
    sl.setSliderDown(False)
    e.end_drag()
    assert len(t.s.changes) == 2 and not e.has_draft()


def test_french_numbers_and_the_whole_hue_rail(qapp, monkeypatch):
    from astro_gpu.client import i18n
    i18n.set_language("fr")
    try:
        e, t, _ch = make(monkeypatch, qapp)
        assert e.center(t, Node(), t.s.path)
        wait(e)
        u = e._ui
        assert u["intv_SII"].text() == "1,00" and "," in u["gain_SII"].text()
        e.set_intensity("SII", 2.2)
        assert "1,00 → 2,20" in e.draft_change()["what"]
    finally:
        i18n.set_language("en")
    assert "sub-page:horizontal, QSlider::add-page:horizontal { background: transparent; }" in \
        u["hue_SII"].styleSheet()


def test_the_preview_renders_as_post_shows_the_workflow_s_layer():
    """Post's layer of the combination sets the render (its sliders); none: the default."""
    from astro_gpu.render.simple import SimpleSettings
    assert ce.layer_settings([], 3) is None
    first = {"kind": "workflow", "pid": 3, "part": None, "sliders": {}, "base": None}
    darker = dict(first, sliders={"sky": 0})
    assert ce.layer_settings([first], 3)["sky"] == pytest.approx(0.25) == SimpleSettings().sky
    assert ce.layer_settings([darker], 3)["sky"] == pytest.approx(0.15)
    rgb = np.abs(np.random.default_rng(2).normal(0.1, 0.02, (3, 60, 80))).astype(np.float32)
    a, b = ce.output_render(rgb, ce.layer_settings([darker], 3)), ce.output_render(rgb)
    assert float(np.median(a)) < float(np.median(b))
