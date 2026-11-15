"""The Gradient editor and what it keeps of a stack: the H-alpha survey is only downloaded on the user's click, the
reduced copy, its size, the block's own samples and the survey follow a new stack result, and an Outdated stack still
gives its size (the samples are never placed in a 1 px square)."""
from __future__ import annotations

import numpy as np

from astro_gpu.client import gradientedit as ge
from tests.client.test_gradient_draft import View, ed, qapp, small_stack, wait  # noqa: F401  (fixtures by name)


def test_changing_workflow_with_the_survey_shown_downloads_nothing(ed, monkeypatch):
    asked = []

    def survey_small(project, pid, sm, fetch=False):
        asked.append((pid, fetch))
        return {"level": np.zeros(sm["arr"].shape[1:], np.float32), "title": "H-alpha"}

    monkeypatch.setattr(ge, "_survey_small", survey_small)
    ed.survey = {"pid": 1, "level": np.zeros((3, 3), np.float32), "title": "H-alpha"}
    ed.survey_shown = True
    q = View(2, "Variant")
    ed.s.paths.append(q)
    ed._load(q)
    wait(ed)
    assert asked, "the survey of the workflow now shown was not read"
    assert all(not fetch for _pid, fetch in asked), asked
