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


def test_a_picture_counts_each_distinct_frame_once_across_its_rows_and_its_stars_row(tmp_path, monkeypatch):
    """Dual-band x2: S and H stack 20 lights each, O III the same 40 (both filters in one exposure), and a Stars row of
    20 other frames gives the stars of the Combine layers picture: 60 distinct frames, 900 s, each light once; DATE-OBS /
    DATE-END over the same set; HISTORY keeps each row's count and says which share lights (Siril's LIVETIME: the sum of
    the stacked exposures, STACKCNT: their count)."""
    import json

    import numpy as np
    from astropy.io import fits

    from astro_gpu import api
    from astro_gpu.core import project2
    from astro_gpu.core.project import Target
    from astro_gpu.render.layers import export_rgb
    p = project2.create(tmp_path / "P", "P", Target(name="Rosette Nebula"))

    def frames(prefix, n, hour):
        out = []
        for i in range(n):
            f = tmp_path / f"{prefix}_{i}.fits"
            h = fits.Header()
            h["EXPTIME"], h["DATE-OBS"] = 15.0, f"2026-11-01T{hour}:{i:02d}:00"
            fits.PrimaryHDU(np.zeros((4, 4), np.float32), header=h).writeto(f)
            out.append(str(f))
        return out
    lights, stars = frames("L", 40, 21), frames("T", 20, "05")
    rows = {}
    for name, files in (("S", lights[:20]), ("H", lights[20:]), ("O", lights), ("Stars", stars)):
        w = p.add_workflow(name, {})
        folder = tmp_path / f"stack_{name}"
        folder.mkdir()
        recs = [{"index": i, "path": f, "weight": 1.0 if not (name == "S" and i == 3) else 0.0}
                for i, f in enumerate(files)]
        (folder / "diagnostics.json").write_text(json.dumps({"frames": recs}))
        rows[w.id] = folder
    ids = list(rows)
    fus = p.add_workflow("S-H-O", {})
    fus.source = {"fusion": {"type": "sho", "inputs": [{"pid": i, "role": r} for i, r in zip(ids[:3], "SHO")]}}
    monkeypatch.setattr(api, "_dir", lambda pr, wf, stage, which="current": rows.get(getattr(wf, "id", wf))
                        if stage == "stack" else None)
    got = api.shot_facts(p, [fus, ids[3]])
    assert got["frames"] == 60 and got["seconds"] == 900.0
    assert [(x["name"], x["frames"], x["shared"]) for x in got["parts"]] == [
        ("S", 19, 0), ("H", 20, 0), ("O", 40, 39), ("Stars", 20, 0)]
    assert got["first"] == "2026-11-01T05:00:00" and got["last"].startswith("2026-11-01T21:39:")
    assert api.shot_facts(p, fus)["frames"] == 40 and api.shot_facts(p, [fus, fus])["seconds"] == 600.0
    facts = {"target": "Rosette Nebula", "date_obs": got["first"], "date_end": got["last"], "exposure_s": got["seconds"],
             "stackcnt": got["frames"], "parts": got["parts"]}
    files = export_rgb(np.random.default_rng(0).uniform(0, 1, (3, 6, 8)).astype(np.float32), tmp_path / "out",
                       {"name": "x", "formats": ["fits32f"]}, facts=facts)
    h = fits.getheader(next(f for f in files if f.suffix == ".fits"))
    assert h["EXPTIME"] == h["LIVETIME"] == 900.0 and h["STACKCNT"] == 60 and h["DATE-OBS"].startswith("2026-11-01T05:00")
    lines = [str(x) for x in h["HISTORY"] if str(x).startswith("exposure ")]
    assert lines[2] == "exposure O: 40 frames, 600 s, 39 also in an earlier row" and lines[0] == "exposure S: 19 frames, 285 s"
