"""Star data not on this computer: no block is marked or left out; the run window says the installation is incomplete
and, at Run, downloads what is missing and starts by itself (a failure says why and the run does not start). Nothing is
downloaded during a run (tests/test_no_download_in_run.py)."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QLabel, QPushButton  # noqa: E402

from astro_gpu.client import i18n  # noqa: E402
from astro_gpu.core import sirilcat  # noqa: E402
from tests.client.pagehelp import done, item, marked, node, open_page, qapp, settle  # noqa: E402, F401


@pytest.fixture
def no_catalogue(tmp_path, monkeypatch):
    """No offline extract anywhere, no cache of earlier downloads."""
    monkeypatch.setenv("ASTRO_GPU_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "userhome"))
    for v in ("LOCALAPPDATA", "APPDATA", "XDG_CONFIG_HOME", "XDG_DATA_HOME"):
        monkeypatch.setenv(v, str(tmp_path / "nowhere"))
    monkeypatch.setenv("ASTRO_GPU_GAIA_EXTRACTS", "")
    monkeypatch.setenv("ASTRO_GPU_SIRIL_CATALOGUES", "")
    sirilcat.reset()
    yield
    i18n.set_language("en")
    sirilcat.reset()


def _button(root, text):
    return next(b for b in root.findChildren(QPushButton) if b.text() == text and b.isVisibleTo(root))


def _texts(root) -> list[str]:
    return [w.text() for w in root.findChildren(QLabel) if w.isVisibleTo(root) and w.text()]


def test_blocks_without_star_data_are_not_marked_and_the_plan_runs_them(qapp, tmp_path, monkeypatch, no_catalogue):  # noqa: F811
    """No star data on this computer: nothing is greyed or marked, no block is left out of the plan; the run installs them
    (tests below)."""
    w, m, page = open_page(tmp_path, monkeypatch, workflows=("Full",))
    try:
        settle(w, page)
        assert all(not hasattr(n, "missing") for n in marked(m))
        assert not hasattr(m, "sky_missing")
        tip = item(page, key="astrometrie").toolTip()
        assert "Cannot run" not in tip and "Install it" not in tip
        plan = m.plan_all()
        assert plan.left_out == {}
        assert {"astrometry", "catalogs"} <= {j.stage for j in plan.jobs}
        page.select(step="astrometrie")
        settle(w, page)
        assert not [t for t in _texts(page.panel) if "Cannot run" in t]
        assert not [b for b in page.panel.findChildren(QPushButton) if b.text() == "Install catalogue"]
    finally:
        done(w, m, page)


def test_the_run_window_says_the_star_data_are_incomplete_and_their_size(qapp, tmp_path, monkeypatch, no_catalogue):  # noqa: F811
    from astro_gpu.client.runs import INFO_NOTE, RunDialog
    from astro_gpu.core import sky_catalogue as sc
    w, m, page = open_page(tmp_path, monkeypatch, workflows=("Full",))
    try:
        settle(w, page)
        assert sc.missing_bytes() == sc.total_bytes() > 2e9             # nothing on this computer: the whole manifest
        d = RunDialog(m, m.plan_all(), "Run")
        note = [t for t in d.warnings() if "Star data incomplete" in t]
        assert len(note) == 1 and note[0].startswith(INFO_NOTE) and "GB to download" in note[0], note
        assert not [r for r in d.refused() if "star" in r[1].lower()]   # said, not blocked
        i18n.set_language("fr")
        assert [t for t in d.warnings() if "Données d'étoiles incomplètes" in t]
        i18n.set_language("en")
        monkeypatch.setattr(sc, "missing", lambda: [])                  # everything installed: nothing said
        assert not [t for t in d.warnings() if "Star data" in t]
        d.close()
    finally:
        done(w, m, page)


def _controller(m, monkeypatch):
    """A RunController that records the starts instead of launching a worker."""
    from astro_gpu.client.services.run import RunController
    c = RunController(m)
    c._snapshot = object()
    monkeypatch.setattr(RunController, "refusal", lambda self, pl: [])      # the scratch project has one light frame
    started = []
    monkeypatch.setattr(RunController, "_launch", lambda self, pl: started.append(pl) or True)
    return c, started


def _wait(cond, seconds=10):
    import time
    t0 = time.monotonic()
    while not cond() and time.monotonic() - t0 < seconds:
        QApplication.processEvents()
        time.sleep(0.01)
    QApplication.processEvents()


def test_run_downloads_the_missing_star_data_then_starts_by_itself(qapp, tmp_path, monkeypatch, no_catalogue):  # noqa: F811
    from astro_gpu.core import sky_catalogue as sc
    w, m, page = open_page(tmp_path, monkeypatch, workflows=("Full",))
    try:
        settle(w, page)
        plan = m.plan_all()
        c, started = _controller(m, monkeypatch)
        have = []
        monkeypatch.setattr(sc, "missing", lambda: [] if have else sc.pins()[:1])

        def fake(progress=None, cancel=None, attempts=3):
            progress(0, 100, "File 1 of 1: downloading 0 B of 100 B")
            progress(100, 100, "Done")
            have.append(1)
        monkeypatch.setattr(sc, "download", fake)
        assert c.begin(plan) and c.installing and not started          # downloading: the run has not started
        _wait(lambda: started)
        assert started == [plan] and not c.installing and c.install_error == ""
        assert c.install_state[:2] == (100, 100)
    finally:
        done(w, m, page)


def test_run_that_cannot_download_says_why_does_not_start_and_the_next_one_resumes(qapp, tmp_path, monkeypatch,  # noqa: F811
                                                                                   no_catalogue):
    from astro_gpu.client.runs import RunDialog
    from astro_gpu.core import sky_catalogue as sc
    from astro_gpu.core.gaia_extracts import DownloadError
    w, m, page = open_page(tmp_path, monkeypatch, workflows=("Full",))
    try:
        settle(w, page)
        plan = m.plan_all()
        c, started = _controller(m, monkeypatch)
        d = RunDialog(m, plan, "Run", controller=c)
        have, tries = [], []
        monkeypatch.setattr(sc, "missing", lambda: [] if have else sc.pins()[:1])

        def fake(progress=None, cancel=None, attempts=3):
            tries.append(1)
            if len(tries) == 1:
                raise DownloadError("The connection failed (no network). What was downloaded is kept: start again to go on.")
            have.append(1)
        monkeypatch.setattr(sc, "download", fake)
        assert c.begin(plan)
        _wait(lambda: not c.installing)
        assert not started and "no network" in c.install_error and "next Run goes on" in c.install_error
        d.refresh()
        assert d.summary.text() == "Star data not downloaded" and "no network" in d.detail.text()
        assert c.begin(plan)
        _wait(lambda: started)
        assert started == [plan] and c.install_error == "" and len(tries) == 2
        d.close()
    finally:
        done(w, m, page)


def test_a_plan_that_reads_no_star_data_starts_at_once(qapp, tmp_path, monkeypatch, no_catalogue):  # noqa: F811
    from astro_gpu.core import runner2
    from astro_gpu.core import sky_catalogue as sc
    w, m, page = open_page(tmp_path, monkeypatch, workflows=("Full",))
    try:
        settle(w, page)
        plan = m.plan_all()
        plan = runner2.Plan([j for j in plan.jobs if j.stage.split("@")[0] not in sc.STEPS])
        assert not sc.needed_by(plan)
        c, started = _controller(m, monkeypatch)
        assert c.begin(plan) and started == [plan] and not c.installing
    finally:
        done(w, m, page)


def test_get_survey_fetches_it_outside_the_run_and_the_next_run_uses_it(qapp, tmp_path, monkeypatch,  # noqa: F811
                                                                         no_catalogue):
    """The survey of the field not on the disk: the run gave 'none'. The Gradient block's panel says it and Get survey
    fetches it (an action of the user, its progress shown), then the results made without it are computed again."""
    import numpy as np

    from astro_gpu.client.graphview.page import GraphPage
    from astro_gpu.client.model import ProjectModel
    from astro_gpu.client.services import survey as survey_job
    from astro_gpu.client.services.worker import ApiWorker
    from astro_gpu.core import project2, runner2
    from astro_gpu.core.project import Target
    from astro_gpu.core.workflow import TEMPLATES, node
    from astro_gpu.stages import surveys
    from tests import skyfield
    monkeypatch.setenv("ASTRO_GPU_SETTINGS", str(tmp_path / "settings.json"))
    sky = skyfield.field(tmp_path / "sky", spectra=False)
    monkeypatch.setenv("ASTRO_GPU_GAIA_EXTRACTS", str(sky["extracts"]))
    sirilcat.reset()
    p = project2.create(tmp_path / "P", "P", Target(**sky["target"]))
    wf = p.add_workflow("Image", TEMPLATES["Classic stack"])
    wf.source = {"path": str(sky["image"]), "stretched": False}
    p.set_blocks(wf.id, [node("astrometry"), node("gradient")])
    project2.save(p)
    plan = runner2.plan_all(p)
    assert runner2.run_plan(p.root, runner2.prepare_run(p, plan, "cpu").plan_file, echo=False) == "done"
    fetched = []

    def fake(wd, W, H, scale, which="auto", min_coverage=0.95, log=print, progress=None, fetch=False):
        if not fetch:                                             # a look at the disk: only what was fetched is there
            if not fetched:
                raise surveys.SurveyMissing(surveys.MISSING)
            return np.zeros((H, W), np.float32), "nsns"
        for k in range(3):
            progress(k / 3, f"tile {k + 1} of 3")
        fetched.append((W, H))
        return np.zeros((H, W), np.float32), "nsns"
    monkeypatch.setattr(surveys, "survey_on_grid", fake)
    worker = ApiWorker()
    m = ProjectModel(project2.load(p.root), worker)
    assert worker.wait(120)
    m._sync()
    page = GraphPage(m)
    try:
        page.resize(1500, 800)
        page.select(step="fond")
        settle(worker, page)
        assert "H-alpha survey: not on this computer" in _texts(page.panel)
        get = _button(page.panel, "Get survey")
        assert "Nothing is downloaded during a run" in get.toolTip()
        from astro_gpu import api as _api
        assert _api.survey_state(m.p, wf.id)["available"] is False
        j = survey_job.job()
        ended = []
        j.finished.connect(lambda root, pid, why: ended.append((pid, why)))     # (its project, its workflow, why)
        get.click()
        import time
        t0 = time.monotonic()
        while not ended and time.monotonic() - t0 < 20:
            QApplication.processEvents()
        QApplication.processEvents()
        assert fetched == [(skyfield.W, skyfield.H)] and ended == [(wf.id, "")]
        store = p.root / "store"
        assert not (store / "_trash").exists() or not list((store / "_trash").glob("*-survey-*"))   # nothing moved
        assert not {"surveys", "gradient"} & {x.stage for x in runner2.plan_all(project2.load(p.root)).jobs}
        m.results_changed()
        settle(worker, page)
        assert "H-alpha survey: on this computer, this result was made without it" in _texts(page.panel)
        use = _button(page.panel, "Use survey")
        assert "Undo" in use.toolTip()
        use.click()
        QApplication.processEvents()
        # it names what it computes again and is one Undo step: it never asks Replace / Try variant / Cancel too
        assert m.pending_change is None
        assert page.panel.findChild(QLabel, "choice_cost") is None
        again = runner2.plan_all(project2.load(p.root))
        assert {"surveys", "gradient", "render_simple"} <= {x.stage for x in again.jobs}       # computed again
        assert {"image", "astrometry"}.isdisjoint({x.stage for x in again.jobs})                # kept
        assert any((store / "_trash").glob("surveys-*-survey-*"))                               # never deleted
        assert m.can_undo()
        assert m.undo()                                                                         # one step: all back
        assert not {"surveys", "gradient", "render_simple"} & {x.stage for x in
                                                                  runner2.plan_all(project2.load(p.root)).jobs}
        assert m.redo()
        assert {"surveys", "gradient"} <= {x.stage for x in runner2.plan_all(project2.load(p.root)).jobs}
    finally:
        page.close()
        m.close(save=False)
        worker.stop()


def test_the_run_window_says_when_asterion_is_not_installed(qapp, tmp_path, monkeypatch, no_catalogue):  # noqa: F811
    """The plan has an Astrometry: the window says, as information and not as a warning, that FLOAT's own search finds
    the field (the run's log and Inspect's checks already said it); said no more once the solver is there."""
    from astro_gpu.client.runs import INFO_NOTE, RunDialog
    from astro_gpu.core import asterion
    from tests import skyfield
    monkeypatch.setenv("ASTRO_GPU_GAIA_EXTRACTS", str(skyfield.field(tmp_path / "sky", spectra=True)["extracts"]))
    sirilcat.reset()
    w, m, page = open_page(tmp_path, monkeypatch, workflows=("Full",))
    try:
        settle(w, page)
        monkeypatch.setattr(asterion, "status", lambda *a, **k: asterion.Status(False, why="not installed", missing=True))
        d = RunDialog(m, m.plan_all(), "Run")
        note = [t for t in d.warnings() if "Asterion" in t]
        assert len(note) == 1 and note[0].startswith(INFO_NOTE) and "is not installed" in note[0]
        d.refresh()
        assert any("Asterion is not installed" in x.text() for x in d.findChildren(QLabel))
        d.close()
        monkeypatch.setattr(asterion, "status", lambda *a, **k: asterion.Status(True))
        d = RunDialog(m, m.plan_all(), "Run")
        assert not [t for t in d.warnings() if "Asterion" in t]
        d.close()
    finally:
        done(w, m, page)


def test_a_block_left_out_for_its_wiring_is_named_and_translated(qapp, tmp_path, monkeypatch):  # noqa: F811
    """A combination left out because its inputs are not on one grid: its block by its displayed name, not its engine
    stage, the reason translated with the names of its ports, and no "Install it" sentence (that one is for the sky's
    data only)."""
    w, m, page = open_page(tmp_path, monkeypatch, workflows=("Full",))
    try:
        settle(w, page)
        q = m.paths[0]
        plan = m.plan_all()
        plan.left_out = {q.pid: {"combine_sho": "Hα, O III are not on the grid of S II: place Align on after their "
                                                "Stacking"}}
        from astro_gpu.client.runs import RunDialog
        d = RunDialog(m, plan, "Run")
        en = [t for t in d.warnings() if "not run" in t]
        assert en == [f'"{q.name}": Combination and the blocks after it not run: Hα, O III are not on the grid of S II: '
                      "place Align on after their Stacking"]
        i18n.set_language("fr")
        fr = [t for t in d.warnings() if "non exécutés" in t]
        assert len(fr) == 1 and "combine_sho" not in fr[0] and "Installer" not in fr[0] and "installer" not in fr[0]
        assert "ne sont pas sur la grille de" in fr[0] and "are not on the grid" not in fr[0], fr
        d.close()
    finally:
        i18n.set_language("en")
        done(w, m, page)


def test_use_survey_puts_back_what_is_not_recomputed_and_says_the_rest(tmp_path, monkeypatch):
    """Undo of Use survey: a key computed since is left as it is, the others come back from the trash."""
    from astro_gpu import api
    from astro_gpu.core import project2, store as st
    from astro_gpu.core.project import Target
    p = project2.create(tmp_path / "P", "P", Target())
    s = st.Store(p.root)
    for stage in ("surveys", "gradient"):
        (s.dir(stage, "0123456789abcdef")).mkdir(parents=True)
    moved = [{"stage": "surveys", "key": "0123456789abcdef", "trash": "surveys-0123456789abcdef-survey-x"},
             {"stage": "gradient", "key": "0123456789abcdef", "trash": "gradient-0123456789abcdef-survey-x"}]
    done, skipped = api.trash_survey_again(p, moved)
    assert done == ["surveys", "gradient"] and not s.dir("gradient", "0123456789abcdef").exists()
    s.dir("gradient", "0123456789abcdef").mkdir(parents=True)                      # recomputed meanwhile
    done, skipped = api.put_back_survey(p, moved)
    assert done == ["surveys"] and skipped == ["gradient"]
    assert (s.trash / "gradient-0123456789abcdef-survey-x").exists()


def test_use_survey_refuses_during_a_run_and_a_failed_move_puts_back_what_moved(tmp_path, monkeypatch):
    # review 2.1: a run reads these results; and a move Windows refuses half way lost the list Undo needs
    import os

    import pytest

    from astro_gpu import api
    from astro_gpu.core import project2, runner2, store as st
    from astro_gpu.core.project import Target
    p = project2.create(tmp_path / "P", "P", Target())
    s = st.Store(p.root)
    moved = [{"stage": x, "key": "0123456789abcdef", "trash": f"{x}-0123456789abcdef-survey-x"}
             for x in ("surveys", "gradient", "color")]
    for m in moved:
        s.dir(m["stage"], m["key"]).mkdir(parents=True)
    monkeypatch.setattr(runner2, "running", lambda root: {"pid": 1})
    for f in (api.use_survey, api.put_back_survey, api.trash_survey_again):
        with pytest.raises(api.RunInProgress):
            f(p, 1 if f is api.use_survey else moved)
    assert all(s.dir(m["stage"], m["key"]).exists() for m in moved)
    monkeypatch.setattr(runner2, "running", lambda root: None)
    real = os.replace

    def refuse_the_third(a, b):
        if "color" in str(a):
            raise PermissionError("held open")
        real(a, b)
    monkeypatch.setattr(api.os, "replace", refuse_the_third)
    with pytest.raises(OSError):
        api.trash_survey_again(p, moved)
    assert all(s.dir(m["stage"], m["key"]).exists() for m in moved)            # all back where they were
    assert not any((s.trash / m["trash"]).exists() for m in moved)
