"""The same shot twice: a light file in two sessions or folders, as the same path or as a copy (name, size and the
capture date of its header): stacked once, first session wins, said by the health check, in both stack modes."""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
from astropy.io import fits

from astro_gpu import api
from astro_gpu.core import project2 as p2
from astro_gpu.core import workflow as wfl
from astro_gpu.core import shots
from astro_gpu.core.project2 import Session
from astro_gpu.core.store import InputScan, Keys


def _fits(path: Path, when: str | None, level: int = 100) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    h = fits.PrimaryHDU(np.full((8, 8), level, np.uint16))
    if when:
        h.header["DATE-OBS"] = when
    h.writeto(path, overwrite=True)
    return path


def _two_nights(tmp_path, mode: str):
    a = tmp_path / "evening" / "lights"
    for i in range(3):
        _fits(a / f"L{i}.fits", f"2026-08-16T22:0{i}:00")
    b = tmp_path / "copy" / "lights"                 # the same shots copied elsewhere, plus a new one
    b.mkdir(parents=True)
    for i in range(3):
        shutil.copy(a / f"L{i}.fits", b / f"L{i}.fits")
    _fits(b / "L9.fits", "2026-08-17T22:00:00")
    p = p2.create(tmp_path / "proj", "proj")
    p.sessions = [Session(1, "Night 1", lights=[str(a)]), Session(2, "Night 2", lights=[str(b)])]
    p.stack_mode = mode
    p.add_workflow("W", wfl.TEMPLATES["Classic stack"])
    return p, a, b


def test_repeated_needs_the_name_the_size_and_the_header_date(tmp_path):
    a = _fits(tmp_path / "x" / "L0.fits", "2026-08-16T22:00:00")
    same = tmp_path / "y" / "L0.fits"
    same.parent.mkdir()
    shutil.copy(a, same)
    other_date = _fits(tmp_path / "z" / "L0.fits", "2026-08-17T22:00:00")       # same name and size, another shot
    no_date = _fits(tmp_path / "w" / "L0.fits", None)
    no_date2 = tmp_path / "v" / "L0.fits"
    no_date2.parent.mkdir()
    shutil.copy(no_date, no_date2)

    def sz(f):
        return (f, f.stat().st_size)
    got = shots.repeated([(1, [sz(a)]), (2, [sz(same), sz(other_date), sz(no_date)]), (3, [sz(no_date2), sz(a)])])
    assert set(got) == {(2, same), (3, a)}           # a file without a date of its own is never taken for a copy
    assert got[(2, same)] == (1, a)


def test_together_stacks_a_copied_shot_once(tmp_path):
    p, a, b = _two_nights(tmp_path, "together")
    lights = p.inputs(1).lights
    assert len(lights) == 4 and not any("copy" in x and not x.endswith("L9.fits") for x in lights)
    assert (2, b / "L0.fits") in p.repeated_lights(1) and p.repeated_lights(1)[(2, b / "L0.fits")] == (1, a / "L0.fits")
    keys = Keys(p, scan=InputScan())
    assert sum(f[0] == "lights" for f in keys.scan.frames(p)) == 4


def test_per_session_stacks_a_copied_shot_once(tmp_path):
    p, a, b = _two_nights(tmp_path, "per_session")
    assert p.session_inputs(1, 1).lights == [p.abspath(a)]
    assert [Path(x).name for x in p.session_inputs(2, 1).lights] == ["L9.fits"]
    keys = Keys(p, scan=InputScan())
    assert [f[1] for f in keys.scan.frames(p, 2, None, [p.session(1)]) if f[0] == "lights"] == ["L9.fits"]


def test_the_same_path_in_two_sessions_is_stacked_once_too(tmp_path):
    p, a, b = _two_nights(tmp_path, "together")
    p.sessions[1].lights = [str(a), str(b)]
    assert len(p.inputs(1).lights) == 4


def test_health_check_names_both_sessions_and_the_count_for_copies(tmp_path):
    for mode in ("together", "per_session"):
        p, a, b = _two_nights(tmp_path / mode, mode)
        c = next(c for c in api.health_checks(p) if c.key == "same_frames")
        assert c.level == "warn" and c.args == {"first": "Night 1", "other": "Night 2", "n": 3}


def test_a_project_without_repeats_keeps_its_frames_and_inputs(tmp_path):
    p, a, b = _two_nights(tmp_path, "together")
    for f in (b / "L0.fits", b / "L1.fits", b / "L2.fits"):
        f.unlink()
    assert p.inputs(1).lights == [p.abspath(a), p.abspath(b)]      # folders as they were
    assert not p.repeated_lights(1) and not any(c.key == "same_frames" for c in api.health_checks(p))
