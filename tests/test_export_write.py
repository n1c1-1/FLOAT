"""An export replaces its file in one step (issue #85)."""
from __future__ import annotations

import os

import pytest

from astro_gpu.render import layers


def write(text):
    return lambda t: t.write_text(text)


def test_an_existing_file_is_replaced_and_no_temporary_stays(tmp_path):
    p = tmp_path / "a.tif"
    p.write_text("old")
    layers._safely(p, write("new"))
    assert p.read_text() == "new" and [f.name for f in tmp_path.iterdir()] == ["a.tif"]


def test_the_temporary_keeps_the_suffix_that_says_the_format(tmp_path):
    seen = []
    layers._safely(tmp_path / "a.png", lambda t: (seen.append(t.suffix), t.write_text("x")))
    assert seen == [".png"]


def test_a_refusal_that_clears_is_retried(tmp_path, monkeypatch):
    p = tmp_path / "a.tif"
    p.write_text("old")
    real, calls = os.replace, []

    def flaky(a, b):
        calls.append(1)
        if len(calls) < 3:
            raise OSError(22, "Invalid argument")
        real(a, b)
    monkeypatch.setattr(layers.os, "replace", flaky)
    monkeypatch.setattr(layers, "REPLACE_WAIT", 0.0)
    layers._safely(p, write("new"))
    assert p.read_text() == "new" and len(calls) == 3


def test_a_failure_leaves_the_old_file_and_names_the_file(tmp_path, monkeypatch):
    p = tmp_path / "a.tif"
    p.write_text("old")

    def refused(a, b):
        raise OSError(22, "Invalid argument")
    monkeypatch.setattr(layers.os, "replace", refused)
    monkeypatch.setattr(layers, "REPLACE_WAIT", 0.0)
    with pytest.raises(layers.WriteFailed) as e:
        layers._safely(p, write("new"))
    assert e.value.name == "a.tif" and p.read_text() == "old" and [f.name for f in tmp_path.iterdir()] == ["a.tif"]


def test_a_stacked_picture_says_its_exposure_as_the_sum_of_the_frames_it_used_and_each_filters_share(tmp_path, monkeypatch):
    """The header's EXPTIME was the inventory's rounded hours of one row: 0.08 h = 288 s for a row of 20 x 15 s...
    here the exact sums of the frames the stacks used, the combination's over its inputs (Siril's LIVETIME / STACKCNT:
    the sum of the stacked frames' exposures and their count), each filter's share in HISTORY."""
    import json

    import numpy as np
    from astropy.io import fits

    from astro_gpu import api
    from astro_gpu.core import project2
    from astro_gpu.core.project import Target
    from astro_gpu.render.layers import export_rgb
    p = project2.create(tmp_path / "P", "P", Target(name="Rosette Nebula"))
    rows = {}
    for name, n, exp in (("S", 20, 15.0), ("H", 20, 15.0), ("O", 40, 15.0)):
        w = p.add_workflow(name, {})
        folder = tmp_path / f"stack_{name}"
        folder.mkdir()
        frames = []
        for i in range(n):
            f = tmp_path / f"{name}_{i}.fits"
            h = fits.Header()
            h["EXPTIME"], h["DATE-OBS"] = exp, f"2026-11-0{1 + (name == 'O')}T21:{i:02d}:00"
            fits.PrimaryHDU(np.zeros((4, 4), np.float32), header=h).writeto(f)
            frames.append({"index": i, "path": str(f), "weight": 1.0 if i != 3 else 0.0})   # one rejected: not counted
        (folder / "diagnostics.json").write_text(json.dumps({"frames": frames}))
        rows[w.id] = folder
    fus = p.add_workflow("S-H-O", {})
    fus.source = {"fusion": {"type": "sho", "inputs": [{"pid": i, "role": r} for i, r in zip(rows, "SHO")]}}
    monkeypatch.setattr(api, "_dir", lambda pr, wf, stage, which="current": rows.get(getattr(wf, "id", wf))
                        if stage == "stack" else None)
    got = api.shot_facts(p, fus)
    assert got["frames"] == 77 and got["seconds"] == 77 * 15.0 and [x["frames"] for x in got["parts"]] == [19, 19, 39]
    assert got["first"] == "2026-11-01T21:00:00" and got["last"].startswith("2026-11-02T21:39:")
    facts = {"target": "Rosette Nebula", "date_obs": got["first"], "date_end": got["last"], "exposure_s": got["seconds"],
             "stackcnt": got["frames"], "parts": got["parts"]}
    files = export_rgb(np.random.default_rng(0).uniform(0, 1, (3, 6, 8)).astype(np.float32), tmp_path / "out",
                       {"name": "x", "formats": ["fits32f"]}, facts=facts)
    h = fits.getheader(next(f for f in files if f.suffix == ".fits"))
    assert h["EXPTIME"] == h["LIVETIME"] == 1155.0 and h["STACKCNT"] == 77 and h["OBJECT"] == "Rosette Nebula"
    assert [str(x) for x in h["HISTORY"] if str(x).startswith("exposure ")] == [
        "exposure S: 19 frames, 285 s", "exposure H: 19 frames, 285 s", "exposure O: 39 frames, 585 s"]
