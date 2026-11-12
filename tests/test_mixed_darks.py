"""A session of lights of two exposures (M101 in R, G and B: lights of 300 and 60 s, a master dark of each made by another
program, no bias, a black level of 0): each light is calibrated with the dark of its own exposure. Before, the two
masters were averaged into one dark of 180 s, scaled by 1.667 with the camera's offset: every calibrated sky came out
under 0 (-260 ADU for the 300 s frames, -500 for the 60 s ones). A dark of another exposure is scaled only above a bias
(or a known black level); without, the lights go without dark, said, rather than with a wrong one."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from astropy.io import fits

from astro_gpu.core.project import Inputs, Project
from astro_gpu.core.runner import Pipeline
from astro_gpu.stages import calibrate as cal
from astro_gpu.stages.calibrate import Calibrate, dark_plan, exposure_groups
from astro_gpu.stages.inventory import Inventory

H, W = 40, 48
OFFSET = 300.0                      # the camera's offset (its black level is 0 in the headers)
THERMAL = 50.0 / 300                # ADU of dark signal a second
SKY = 800.0 / 300                   # ADU of sky a second


def _write(path: Path, data, **hdr) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    h = fits.Header()
    for k, v in hdr.items():
        h[k] = v
    fits.PrimaryHDU(data=data, header=h).writeto(path, overwrite=True)
    return path


def _dark(t: float) -> np.ndarray:
    d = np.full((H, W), OFFSET + THERMAL * t, np.float32)
    d[7, 9] += 20 * t                                       # a hot pixel, growing with the exposure
    return d


def _session(tmp_path: Path, exposures=(300.0, 60.0), darks=(300.0, 60.0), bias: bool = False, n: int = 3):
    """Monochrome lights of each of ``exposures`` (``n`` each), a master dark of each of ``darks`` on the 0-1 scale (as
    another program writes them), a master bias of the offset with ``bias``. (manifest, cache.json, log lines, the
    calibrated frames in ADU, the lights' exposures in the cache's order)."""
    L, D, B = (tmp_path / k for k in "LDB")
    k = 0
    for t in exposures:
        for i in range(n):
            a = _dark(t) + SKY * t + np.random.default_rng(k).normal(0, 2, (H, W))
            _write(L / f"l{t:05.0f}_{i}.fits", np.round(a).astype(np.uint16), EXPTIME=t, GAIN=139,
                   **{"DATE-OBS": f"2026-04-1{k // 10}T22:{k % 10:02d}:00"})
            k += 1
    for t in darks:
        _write(D / f"master_dark_{t:g}.fit", _dark(t) / np.float32(65535), EXPTIME=t, STACKCNT=20, GAIN=139)
    if bias:
        _write(B / "bias.fit", np.full((H, W), OFFSET / 65535, np.float32), STACKCNT=50)
    pr = Project.create(tmp_path / "p", "p", Inputs(lights=[str(L)], darks=[str(D)] if darks else [],
                                                    bias=[str(B)] if bias else []))
    pr.config.devices = {"main": "cpu"}
    pr.config.params = {"calibrate": {"workers": 1, "hot_from_lights": False}}
    pr.save()
    Pipeline(pr, [Inventory(), Calibrate()]).run()
    d = Path(pr.stage_dir("calibrate"))
    m = json.loads((d / "manifest.json").read_text())
    assert m["status"] == "done", m.get("error")
    c = json.loads((d / "cache.json").read_text())
    fr = np.asarray(np.memmap(c["path"], dtype=np.float16, mode="r", shape=(c["n"], H, W)), np.float32) / c["scale"]
    logs = [json.loads(x) for x in (pr.logs / "events.jsonl").read_text().splitlines()]
    said = [(e.get("level"), e["msg"]) for e in logs if e.get("type") == "log" and e.get("stage") == "calibrate"]
    inv = json.loads((Path(pr.stage_dir("inventory")) / "frames.json").read_text())["lights"]
    return m, c, said, fr, [f["exposure"] for f in inv], d


def _sky(fr: np.ndarray) -> float:
    return float(np.nanmedian(fr))


def test_the_groups_and_the_plan_follow_the_usual_practice():
    g = exposure_groups([(300.0, 139.0), (60.0, 139.0), (299.99, 139.0), (None, 139.0), (60.0, 0.0)])
    assert [(x["exposure"], x["gain"], x["of"]) for x in g] == [(299.995, 139.0, [0, 2, 3]), (60.0, 139.0, [1]),
                                                                  (60.0, 0.0, [4])]
    lights = exposure_groups([(300.0, 139.0), (60.0, 139.0), (120.0, 200.0)])
    darks = exposure_groups([(300.0, 139.0)])
    assert dark_plan(lights, darks, base=False) == [("same", 0), ("none", None), ("none", None)]
    assert dark_plan(lights, darks, base=True) == [("same", 0), ("scaled", 0), ("none", None)]   # never across gains
    assert dark_plan(lights, exposure_groups([(None, None)]), base=False)[1] == ("same", 0)        # not recorded


def test_two_exposures_with_a_master_dark_each_get_their_own(tmp_path):
    m, c, said, fr, exp, d = _session(tmp_path)
    for i, t in enumerate(exp):
        assert abs(_sky(fr[i]) - SKY * t) < 3, (t, _sky(fr[i]))            # 800 and 160 ADU, never under 0
    assert np.nanmin(fr) > 0
    msgs = [t for _lv, t in said]
    assert "the lights of 300 s are calibrated with the darks of 300 s" in msgs, msgs
    assert "the lights of 60 s are calibrated with the darks of 60 s" in msgs
    assert not any("180" in t or "scaled" in t for t in msgs) and "dark_scale" not in m["metrics"]
    z = np.load(d / cal.GROUP_DARKS)
    assert z["darks"].shape == (2, H, W) and sorted(set(z["rank"].tolist())) == [0, 1]
    assert np.array_equal(np.load(d / "master_dark.npy"), z["darks"][0])
    assert c["dark_level"] == float(np.median(_dark(60.0)))               # the lowest: the cache's widest range


def test_lights_without_a_dark_of_their_exposure_and_no_bias_go_without_dark_said(tmp_path):
    _m, _c, said, fr, exp, _d = _session(tmp_path, darks=(300.0,))
    assert ("warning", "no dark of 60 s: the lights of 60 s are calibrated without dark (no bias frames nor black "
                       "level: the darks of 300 s would scale the offset with their thermal signal)") in said, said
    for i, t in enumerate(exp):
        want = SKY * t if t == 300 else SKY * t + OFFSET + THERMAL * t   # the 60 s ones keep their offset, honestly
        assert abs(_sky(fr[i]) - want) < 3, (t, _sky(fr[i]))


def test_with_a_bias_a_missing_exposure_gets_another_one_s_thermal_part_scaled(tmp_path):
    m, _c, said, fr, exp, _d = _session(tmp_path, darks=(300.0,), bias=True)
    assert "no dark of 60 s: the lights of 60 s are calibrated with the darks of 300 s, their thermal signal scaled " \
           "by 0.200" in [t for _lv, t in said], said
    for i, t in enumerate(exp):
        assert abs(_sky(fr[i]) - SKY * t) < 3, (t, _sky(fr[i]))
    assert "dark_scale" not in m["metrics"] or m["metrics"]["dark_scale"] == 0.2


def test_a_session_of_one_exposure_uses_its_one_dark_for_every_frame_as_before(tmp_path):
    """The same files in the same order, the same arithmetic: each frame is (raw - master dark) / flat x scale, the
    master dark the dark of the lights' exposure as it is; no file of several darks (the cache-only rebuild reads it)."""
    _m, c, said, _fr, exp, d = _session(tmp_path, exposures=(300.0,), darks=(300.0,))
    assert set(exp) == {300.0}
    dark, flat, bad = (np.load(d / n) for n in cal.MASTER_FILES)
    assert np.allclose(dark, _dark(300.0), atol=0.01)
    assert not (d / cal.GROUP_DARKS).exists()
    lights = sorted((tmp_path / "L").iterdir())
    for i, f in enumerate(lights):
        raw = fits.getdata(f).astype(np.float32)
        x = np.minimum((raw - dark) / flat * c["scale"], cal.VALID_MAX - 1)
        x[bad] = np.nan
        got = np.asarray(np.memmap(c["path"], dtype=np.float16, mode="r", shape=(c["n"], H, W))[i])
        assert np.array_equal(got, x.astype(np.float16), equal_nan=True)
    assert not any("calibrated with" in t or "without dark" in t for _lv, t in said)     # nothing new said


def test_lights_of_one_exposure_among_masters_of_two_take_their_own_not_the_mean(tmp_path):
    _m, _c, said, fr, _exp, d = _session(tmp_path, exposures=(300.0,))
    assert abs(_sky(fr) - SKY * 300) < 3 and np.allclose(np.load(d / "master_dark.npy"), _dark(300.0), atol=0.01)
    assert "the lights of 300 s are calibrated with the darks of 300 s" in [t for _lv, t in said]
