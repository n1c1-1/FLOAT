"""Astrometry when the typed target and the pointing in the frames' headers disagree (a project typed as NGC 602 over
lights pointing at Dec +31.7): the typed target is tried first; when it does not solve the field, the header's pointing
is the centre, and the log says which one solved. When they agree, nothing changes."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from test_astrometry_asterion import DEC, RA, SCALE, H, W, _field, _written

from astro_gpu.core import asterion
from astro_gpu.core.stage import StageError
from astro_gpu.stages import astrometry as am
from astro_gpu.stages.astrometry import AstrometryParams


def _run(tmp_path, monkeypatch, target, header, **par):
    """The stage on the rendered field, whose catalogue answers only for a centre within 10 deg of the field: (logs,
    metrics)."""
    logs, metrics, asked = [], {}, []
    gs, gm, o = _field()
    empty = gs[:0]

    def gaia(ra, dec, *a, **k):
        asked.append((round(ra, 2), round(dec, 2)))
        near = am.angular_separation(ra, dec, RA, DEC) < 10
        return (gs, gm) if near else (empty, gm[:0])
    monkeypatch.setattr(am, "detect", lambda g: o)
    monkeypatch.setattr(am, "gaia_bright", gaia)
    monkeypatch.setattr(am, "remember_instrument", lambda *a: None)
    monkeypatch.setattr(am, "mirror_hint", lambda *a: (True, False))
    monkeypatch.setattr(am, "focal_hints", lambda *a: [])
    monkeypatch.setattr(asterion, "status", lambda *a, **k: asterion.Status(False, why="", missing=True))
    point = {"pointing_ra": header[0], "pointing_dec": header[1]} if header else {}
    ctx = SimpleNamespace(load_json=lambda stage, name: {"scale": 1.0},
                          load=lambda *a, **k: np.zeros((3, H, W), np.float32),
                          log=lambda m, level="info": logs.append((level, m)), progress=lambda *a, **k: None,
                          metric=metrics.__setitem__, out=lambda n: tmp_path / n,
                          metric_of=lambda stage, key, *a: point.get(key), scratch=tmp_path, check_stop=lambda: None)
    try:
        am.Astrometry().run(ctx, AstrometryParams(ra=target[0], dec=target[1], target="NGC 602", scale=SCALE,
                                                  blind=False, on_failure="stop", **par))
    finally:
        _run.logs, _run.metrics, _run.asked = logs, metrics, asked
    return logs, metrics


def test_a_wrong_target_far_from_the_header_pointing_is_followed_by_the_header_s_pointing(tmp_path, monkeypatch):
    logs, metrics = _run(tmp_path, monkeypatch, (13.0, -73.6), (RA + 0.2, DEC - 0.1))
    w = _written(tmp_path / "wcs.fits")
    c = w.pixel_to_world(W / 2, H / 2)
    assert am.angular_separation(c.ra.deg, c.dec.deg, RA, DEC) < 0.01                  # solved at the true centre
    assert metrics["centre_from"] == "header"
    text = " | ".join(m for _l, m in logs)
    assert "no solution around the typed target NGC 602" in text and "solved around the pointing in the frames' headers" in text
    assert (13.0, -73.6) in _run.asked and any(abs(a - RA) < 1 for a, _d in _run.asked)    # the target first, then the header


def test_a_right_target_far_from_nothing_is_not_changed(tmp_path, monkeypatch):
    logs, metrics = _run(tmp_path, monkeypatch, (RA + 0.3, DEC - 0.2), (RA + 0.2, DEC - 0.1))
    assert "centre_from" not in metrics and not any("typed target" in m for _l, m in logs)
    assert all(am.angular_separation(a, d, RA, DEC) < 1 for a, d in _run.asked)


def test_a_far_header_that_the_target_confirms_is_said(tmp_path, monkeypatch):
    """The target right and the header wrong (a mount that was not synced): the field solves around the target."""
    logs, metrics = _run(tmp_path, monkeypatch, (RA + 0.3, DEC - 0.2), (200.0, -30.0))
    assert metrics["centre_from"] == "target"
    assert any("where the typed target NGC 602 puts it" in m for _l, m in logs)


def test_no_header_pointing_changes_nothing(tmp_path, monkeypatch):
    with pytest.raises(StageError):
        _run(tmp_path, monkeypatch, (13.0, -73.6), None)
    assert not any("pointing in the frames" in m for _l, m in _run.logs)
