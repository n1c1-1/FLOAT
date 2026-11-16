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
