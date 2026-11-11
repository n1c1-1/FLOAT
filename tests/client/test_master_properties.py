"""The properties of a session's master flat: the exposure as the matching line says it (0.05 s is 1/20 s, never 1 s),
the session's own flat (not the project's first), and the day in the file's header (not the day it was copied)."""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
from astropy.io import fits  # noqa: E402

from astro_gpu.client.imageview import exposure_text  # noqa: E402
from astro_gpu.client.services.inspect import browse_group, project_frames  # noqa: E402
from astro_gpu.client.sources import calib_rows  # noqa: E402
from astro_gpu.core import project2  # noqa: E402
from astro_gpu.core.project import Target  # noqa: E402


def _fits(path: Path, level: float, **hdr) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    h = fits.Header()
    for k, v in hdr.items():
        h[k] = v
    fits.PrimaryHDU(np.full((24, 32), level, np.float32), h).writeto(path, overwrite=True)
    return path


def _project(root: Path, flat_hdr: dict) -> project2.Project2:
    p = project2.create(root / "P", "P", Target(name="NGC 6995"))
    ha = _fits(root / "cal" / "flat_ha.fit", 0.5, EXPTIME=1.0, STACKCNT=30)
    oiii = _fits(root / "cal" / "flat_oiii.fit", 0.5, EXPTIME=0.05, STACKCNT=30, **flat_hdr)
    for i, (flt, flat) in enumerate((("Ha", ha), ("OIII", oiii))):
        for j in range(2):
            _fits(root / f"L{i}" / f"l{j}.fits", 900, EXPTIME=300.0, **{"DATE-OBS": f"2023-06-29T23:0{j}:00"})
        p.sessions.append(project2.Session(i + 1, flt, lights=[str(root / f"L{i}")], flats=[str(flat)], filter=flt))
    project2.save(p)
    return p


def test_the_exposure_of_a_fraction_of_a_second_is_not_rounded():
    assert exposure_text(0.05) == "1/20 s" and exposure_text(1.0) == "1 s" and exposure_text(0.7) == "0.7 s"
    assert exposure_text(0.001) == "1/1000 s" and exposure_text(300) == "300 s"


def test_a_session_shows_its_own_flat_and_the_header_s_day(tmp_path):
    p = _project(tmp_path, {"DATE-OBS": "2022-10-18T03:36:00"})
    d = project_frames(p, None)
    oiii = next(r for r in d.sessions if r.filter == "OIII")
    items = browse_group(p, None, "flats", {}, session=oiii.id)
    assert items[0].time[:10] == "2022-10-18"                        # the copy's date on disk is today's
    rows = dict(calib_rows(len(items), oiii.calib["flats"], items[0].time))
    assert rows["Exposure"] == "1/20 s" and rows["Taken on"] == "2022-10-18", rows
    ha = next(r for r in d.sessions if r.filter == "Ha")
    assert dict(calib_rows(1, ha.calib["flats"], "x"))["Exposure"] == "1 s"


def test_without_a_date_in_the_header_the_file_s_date_is_shown(tmp_path):
    p = _project(tmp_path, {})
    d = project_frames(p, None)
    oiii = next(r for r in d.sessions if r.filter == "OIII")
    items = browse_group(p, None, "flats", {}, session=oiii.id)
    assert items[0].time[:4] == str(time.localtime().tm_year)
    assert dict(calib_rows(1, oiii.calib["flats"], items[0].time))["Taken on"] == items[0].time[:10]
