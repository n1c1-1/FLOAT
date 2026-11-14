"""A session of several nights (8 nights of NGC 6995, 29 June to 22 July) and a project typed as another object
(NGC 602, Dec -73.6, over lights pointing at Dec +31.7): the lights row says the nights as they are, the Session block's
tooltip too, and the pointing against the typed target is a warning in the matching lines, in the checks before a run
and in the Project block (orange, with the reason)."""
from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
from astropy.io import fits  # noqa: E402

from astro_gpu import api  # noqa: E402
from astro_gpu.client import i18n  # noqa: E402
from astro_gpu.client.frames import check_text, nights_of, nights_text  # noqa: E402
from astro_gpu.client.model import ProjectModel  # noqa: E402
from astro_gpu.client.services.inspect import project_frames  # noqa: E402
from astro_gpu.client.sources import SourcesView  # noqa: E402
from astro_gpu.core import project2  # noqa: E402
from astro_gpu.core.project import Target  # noqa: E402
from astro_gpu.io.frames import light_facts  # noqa: E402
from tests.client.pagehelp import item, qapp  # noqa: E402,F401  (a fixture)

NIGHTS = ["2023-06-29", "2023-07-05", "2023-07-08", "2023-07-11", "2023-07-15", "2023-07-18", "2023-07-21", "2023-07-22"]


def _project(root: Path, target: Target, nights=NIGHTS, **extra) -> project2.Project2:
    p = project2.create(root / "P", "P", target)
    for i, day in enumerate(nights):
        for j in range(2):
            h = fits.Header()
            for k, v in {"EXPTIME": 300.0, "GAIN": 90, "RA": 314.14, "DEC": 31.7, "OBJECT": "NGC 6995",
                         "DATE-OBS": f"{day}T23:{10 + j}:00", **extra}.items():
                h[k] = v
            f = root / "L" / f"l{i}_{j}.fits"
            f.parent.mkdir(parents=True, exist_ok=True)
            fits.PrimaryHDU(np.full((24, 32), 900, np.uint16), h).writeto(f, overwrite=True)
    p.sessions.append(project2.Session(1, "SII", lights=[str(root / "L")], filter="SII"))
    project2.save(p)
    return p


def test_several_nights_are_said_as_they_are(tmp_path):
    p = _project(tmp_path, Target(name="NGC 6995"))
    row = project_frames(p, None).sessions[0]
    assert len(row.nights) == 8 and row.nights[0] == "2023-06-29" and row.nights[-1] == "2023-07-22"
    sub = SourcesView._sub(SimpleNamespace(), "lights", 16, row)
    assert sub == "16 lights · 300 s · Gain 90 · 8 nights · 29/06 to 22/07", sub
    # the night's own day, by the rule of night_start: a frame of 00:16 belongs to the night before
    assert nights_of(["2023-07-01T00:16:00", "2023-06-30T22:00:00"]) == ["2023-06-30"]
    assert nights_text(["2023-06-30"]) == "" and nights_text(["2023-06-30", "2023-07-02"]) == "2 nights · 30/06 to 02/07"
    try:
        i18n.set_language("fr")
        assert SourcesView._sub(SimpleNamespace(), "lights", 16, row).endswith("8 nuits · du 29/06 au 22/07")
        assert not i18n.missing()
    finally:
        i18n.set_language("en")


def test_one_night_is_said_as_today(tmp_path):
    p = _project(tmp_path, Target(name="NGC 6995"), nights=["2023-06-29"])
    row = project_frames(p, None).sessions[0]
    assert SourcesView._sub(SimpleNamespace(), "lights", 2, row) == "2 lights · 300 s · Gain 90 · night of 29/06"


def test_the_session_block_s_tooltip_says_the_nights(qapp, tmp_path):  # noqa: F811
    from astro_gpu.client.graphview.page import GraphPage
    from astro_gpu.client.graphview.style import kind_what
    from astro_gpu.client.services.worker import ApiWorker
    assert kind_what("session").startswith("One night")
    p = _project(tmp_path, Target(name="NGC 6995"))
    w = p.add_workflow("Workflow", {})
    p.set_blocks(w.id, [{"block": "calibration"}, {"block": "registration"}, {"block": "stacking"}])
    project2.save(p)
    worker = ApiWorker()
    m = ProjectModel(p, worker, readonly=True, gpu=False, device="cpu")
    assert worker.wait(120) and m.ready, m.error
    page = GraphPage(m)
    try:
        assert m.nights_text(1) == "8 nights · 29/06 to 22/07"
        page.refresh()
        tip = item(page, kind="session").toolTip()
        assert "8 nights · 29/06 to 22/07: their light frames" in tip, tip
        assert "One night" not in tip
    finally:
        page.close()
        m.close(save=False)
        worker.stop()


def test_the_model_gives_the_nights_of_a_session(tmp_path):
    _project(tmp_path, Target(name="NGC 6995"))
    files = sorted((tmp_path / "L").glob("*.fits"))
    m = SimpleNamespace(_facts=SimpleNamespace(lights={1: light_facts(files)}))
    assert ProjectModel.nights_text(m, 1) == "8 nights · 29/06 to 22/07" and ProjectModel.nights_text(m, 2) == ""


def test_lights_far_from_the_target_are_information_in_the_matching_lines_and_the_checks(tmp_path):
    p = _project(tmp_path, Target(name="NGC 602", ra=314.14, dec=-73.6))
    d = project_frames(p, None)
    row = d.sessions[0]
    assert row.target_far and round(row.target_far["sep"]) == 105 and row.target_far["object"] == "NGC 6995"
    view = SimpleNamespace(data=d, s=SimpleNamespace(frames_excluded=[]), _row=lambda: row, black=lambda: 0.0)
    want = ("Target: the frames point 105° away from NGC 602 (their header says NGC 6995): check the project's "
            "target.")
    assert (None, want) in SourcesView.checks(view)
    checks = [c for c in d.checks if c.key == "target_far"]
    assert len(checks) == 1 and checks[0].level == "info" and checks[0].args["sessions"] == ["SII"], d.checks
    assert check_text(checks[0]) == want
    assert "Target:" in checks[0].text


def test_no_object_name_in_the_header_and_lights_near_the_target(tmp_path):
    p = _project(tmp_path, Target(name="NGC 602", ra=314.14, dec=-73.6), OBJECT="")
    row = project_frames(p, None).sessions[0]
    assert row.target_far["object"] == ""
    from astro_gpu.client.frames import target_far_text
    assert target_far_text(row.target_far) == "Target: the frames point 105° away from NGC 602: check the project's target."
    near = _project(tmp_path / "n", Target(name="NGC 6995", ra=314.2, dec=31.6))
    d = project_frames(near, None)
    assert d.sessions[0].target_far is None and not any(c.key == "target_far" for c in d.checks)


def test_the_project_block_is_not_orange_for_a_far_target(tmp_path):
    p = _project(tmp_path, Target(name="NGC 602", ra=314.14, dec=-73.6))
    files = sorted((tmp_path / "L").glob("*.fits"))
    full = {k: "x" for k in project2.ACQUISITION}
    p.acquisition.update(full)
    m = SimpleNamespace(p=p, _facts=SimpleNamespace(lights={1: light_facts(files)}),
                        target_state=lambda: SimpleNamespace(name="NGC 602"))
    st = ProjectModel.project_state(m)
    assert st.level == "complete" and not st.missing and st.line.startswith("Complete")
    right = _project(tmp_path / "r", Target(name="NGC 6995", ra=314.2, dec=31.6))
    right.acquisition.update(full)
    m2 = SimpleNamespace(p=right, _facts=SimpleNamespace(lights={1: light_facts(files)}),
                         target_state=lambda: SimpleNamespace(name="NGC 6995"))
    st2 = ProjectModel.project_state(m2)
    assert st2.level == "complete" and st2.line.startswith("Complete")
    assert api.target_far(right, {1: light_facts(files)}) == []
