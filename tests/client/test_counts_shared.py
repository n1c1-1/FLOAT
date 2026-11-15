"""The figures of a project of two nights sharing their calibration frames (the testers' pass 2): the report said
"Darks: 20" for 10 darks on the disk, the Home card "2 nights · 24 min" for 48 min of light."""
from __future__ import annotations

import json
from types import SimpleNamespace as NS

from astro_gpu.client.home import _integration
from astro_gpu.client.model import Facts, ProjectModel


def test_the_report_counts_shared_calibration_frames_once():
    darks = [f"C:/d/D_{i}.fits" for i in range(10)]
    facts = Facts()
    facts.inventory = {"n_lights": 24, "n_darks": 20, "n_flats": 60}       # the engine's sum over the sessions
    facts.files = {1: {"lights": ["a"], "darks": darks, "flats": [f"C:/f/F_{i}.fits" for i in range(30)]},
                   2: {"lights": ["b"], "darks": list(darks), "flats": [f"c:/f/F_{i}.fits" for i in range(30)]}}
    m = NS(_facts=facts, sessions=[], name="M33 lab", target="M 33")
    info = ProjectModel.info(m)
    assert (info["n_darks"], info["n_flats"], info["n_bias"], info["n_lights"]) == (10, 30, 0, 24)


def test_the_home_card_sums_the_nights_light(tmp_path):
    def inv(k, first, h):
        d = tmp_path / "store" / "inventory" / k
        d.mkdir(parents=True)
        (d / "manifest.json").write_text(json.dumps({"status": "done", "metrics": {"integration_h": h,
                                                                                  "first_time": first}}))
    inv("aaa", "2021-11-03T20:00", 0.4)
    inv("bbb", "2021-11-04T20:00", 0.4)
    inv("ccc", "2021-11-04T20:00", 0.3)                  # the same night read with a frame limit: counted once
    (tmp_path / "state.json").write_text(json.dumps({"last_keys": {
        "1": {"inventory": "aaa", "inventory@1": "aaa", "inventory@2": "bbb"}, "2": {"inventory@2": "ccc"}}}))
    assert _integration(tmp_path) == 0.8
