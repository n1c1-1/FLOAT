"""What the sessions' page reads for its Calibration matching lines: the exposure and gain of each calibration frame
come from its header, read once and without decoding the frame (each dark, flat and bias of a raw set was decoded
whole for them, and the flats and bias read a second time by the health checks: "Reading the frames…" lasted minutes
on a processor alone)."""
from __future__ import annotations

import collections
from pathlib import Path

import numpy as np

from astro_gpu.client.services.inspect import project_frames
from astro_gpu.core import project2
from astro_gpu.core.project import Target
from astro_gpu.io import frames as fr
from tests.client.test_calib_matching import H, W, _fits


def test_each_calibration_frame_header_is_read_once_and_never_decoded_for_it(tmp_path, monkeypatch):
    p = project2.create(tmp_path / "P", "P", Target(name="NGC 7000"))
    for j in range(3):
        _fits(tmp_path / "L" / f"l{j}.fits", np.full((H, W), 900, np.uint16), EXPTIME=300.0,
              **{"DATE-OBS": f"2026-06-10T23:0{j}:00"})
    kinds = {}
    for kind, level, exp in (("darks", 510, 300.0), ("flats", 9000, 0.5), ("bias", 500, 0.001)):
        for j in range(5):
            _fits(tmp_path / kind / f"{kind[0]}{j}.fits", np.full((H, W), level, np.uint16), EXPTIME=exp)
        kinds[kind] = [str(tmp_path / kind)]
    p.sessions.append(project2.Session(1, "night 1", lights=[str(tmp_path / "L")], **kinds))
    project2.save(p)
    fr._LIGHT_FACTS.clear()
    reads, real = collections.Counter(), fr.read_info
    alone = collections.Counter()                           # read with no frame of its camera given: a raw is decoded

    def read_info(path, like=None):
        kind = Path(path).parent.name
        reads[Path(path).name] += 1
        alone[kind] += like is None
        return real(path, like=like)
    monkeypatch.setattr(fr, "read_info", read_info)
    d = project_frames(p, None)
    row = d.sessions[0]
    assert len(row.dark_keys) == len(row.flat_keys) == len(row.bias_keys) == 5
    assert row.dark_keys[0]["exposure"] == 300.0 and row.flat_keys[0]["exposure"] == 0.5
    middle = {"d2.fits", "f2.fits", "b2.fits"}              # the one whose levels are measured: read for them too
    again = {n: c for n, c in reads.items() if n[0] in "dfb" and c > (2 if n in middle else 1)}
    assert not again, again
    # the first of a kind and the one whose levels are measured: the others only their header
    assert all(alone[k] <= 2 for k in ("darks", "flats", "bias")), alone


def test_the_health_checks_read_no_light_header_the_page_has_read(tmp_path, monkeypatch):
    """The health checks asked where the lights point with a sample of 8 of their headers, kept apart from the sample
    the page (and their own check of the exposure times) reads: 8 headers of each session were read a second time.
    They now ask the one sample: after the page's reading, the health checks read no light's header."""
    from astro_gpu import api
    p = project2.create(tmp_path / "P", "P", Target(name="NGC 7000"))
    for j in range(12):
        _fits(tmp_path / "L" / f"l{j:02d}.fits", np.full((H, W), 900, np.uint16), EXPTIME=300.0,
              **{"DATE-OBS": f"2026-06-10T23:{j:02d}:00"})
    p.sessions.append(project2.Session(1, "night 1", lights=[str(tmp_path / "L")]))
    project2.save(p)
    fr._LIGHT_FACTS.clear()
    reads, real = collections.Counter(), fr.read_info

    def read_info(path, like=None):
        reads[Path(path).name] += 1
        return real(path, like=like)
    monkeypatch.setattr(fr, "read_info", read_info)
    d = project_frames(p, None)
    assert len(d.sessions[0].light_keys) == 12
    # each light once; the first and the last once more for the session's dates, the first again for the frame size
    # of the disk check (23 before: 8 more for the sample apart)
    assert sum(reads.values()) == 12 + 3, reads
    reads.clear()
    api.health_checks(p)                                    # asked again: nothing is read
    assert not reads, reads
    fr._LIGHT_FACTS.clear()
    api.health_checks(p)                                    # their sample not kept (no page before): each light once
    assert set(reads.values()) == {1} and len(reads) == 12, reads
