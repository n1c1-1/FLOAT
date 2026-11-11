"""A monochrome astro camera (FITS, GAIN 90, 300 s) and a master dark of 300 s without a GAIN keyword: the matching line
said "exposure 300 s and ISO 0, different from the lights" (the missing gain read as 0). A value one side does not
record is no mismatch; frames that carry a GAIN are said "Gain", the cameras that record an ISO say "ISO"."""
from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
from astropy.io import fits  # noqa: E402

from astro_gpu.client import i18n  # noqa: E402
from astro_gpu.client.services.inspect import project_frames  # noqa: E402
from astro_gpu.client.sources import SourcesView  # noqa: E402
from astro_gpu.core import project2  # noqa: E402
from astro_gpu.core.project import Target  # noqa: E402


def _fits(path: Path, level: float, **hdr) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    h = fits.Header()
    for k, v in hdr.items():
        h[k] = v
    fits.PrimaryHDU(np.full((24, 32), level, np.float32), h).writeto(path, overwrite=True)
    return path


def _lines(root: Path, light: dict, dark: dict) -> tuple[list, object]:
    p = project2.create(root / "P", "P", Target(name="NGC 6995"))
    for j in range(2):
        _fits(root / "L" / f"l{j}.fits", 900, **light)
    d = _fits(root / "cal" / "dark.fit", 0.01, STACKCNT=20, **dark)
    p.sessions.append(project2.Session(1, "Ha", lights=[str(root / "L")], darks=[str(d)], filter="Ha"))
    project2.save(p)
    data = project_frames(p, None)
    row = data.sessions[0]
    view = SimpleNamespace(data=data, s=SimpleNamespace(frames_excluded=[]), _row=lambda: row, black=lambda: 0.0)
    return SourcesView.checks(view), row


def test_a_gain_missing_in_the_master_dark_is_no_mismatch(tmp_path):
    lines, row = _lines(tmp_path, dict(EXPTIME=300.0, GAIN=90), dict(EXPTIME=300.0))
    assert (True, "Darks: exposure 300 s, same as the lights; Gain not recorded in the darks.") in lines, lines
    assert not any("different" in t for _ok, t in lines)


def test_a_gain_that_differs_or_an_exposure_that_differs_is_said(tmp_path):
    lines, _row = _lines(tmp_path, dict(EXPTIME=300.0, GAIN=90), dict(EXPTIME=300.0, GAIN=120))
    assert (False, "Darks: exposure 300 s, Gain 120, different from the lights.") in lines, lines
    lines, _row = _lines(tmp_path / "b", dict(EXPTIME=300.0, GAIN=90), dict(EXPTIME=180.0, GAIN=90))
    assert (False, "Darks: exposure 180 s, Gain 90, different from the lights.") in lines, lines
    lines, _row = _lines(tmp_path / "c", dict(EXPTIME=300.0, GAIN=90), dict(EXPTIME=300.0, GAIN=90))
    assert (True, "Darks: exposure 300 s, Gain 90, same as the lights.") in lines, lines


def test_the_row_and_the_matching_lines_say_gain_for_fits_and_iso_for_an_iso_speed(tmp_path):
    _lines_, row = _lines(tmp_path, dict(EXPTIME=300.0, GAIN=90), dict(EXPTIME=300.0))
    assert row.gain
    sub = SourcesView._sub(SimpleNamespace(), "lights", 2, row)
    assert sub.startswith("2 lights · 300 s · Gain 90 · night of "), sub
    _l, row2 = _lines(tmp_path / "b", dict(EXPTIME=300.0, ISOSPEED=1600), dict(EXPTIME=300.0, ISOSPEED=1600))
    assert not row2.gain
    assert SourcesView._sub(SimpleNamespace(), "lights", 2, row2).startswith("2 lights · 300 s · ISO 1600 · ")
    try:
        i18n.set_language("fr")
        assert "Gain 90 · nuit du" in SourcesView._sub(SimpleNamespace(), "lights", 2, row)
        lines, _r = _lines(tmp_path / "c", dict(EXPTIME=300.0, GAIN=90), dict(EXPTIME=300.0))
        assert (True, "Darks : pose 300 s, comme les brutes ; Gain non noté dans les darks.") in lines, lines
    finally:
        i18n.set_language("en")
