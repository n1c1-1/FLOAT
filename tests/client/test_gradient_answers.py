"""The Gradient editor's answers carry their workflow and request: a preview computed for one workflow is never shown
on another one shown meanwhile, even with the same settings; the workflow shown gets its own."""
from __future__ import annotations

import time

import numpy as np
from PySide6.QtWidgets import QApplication

from astro_gpu.client import gradientedit as ge
from astro_gpu.stages.gradient import GradientParams
from tests.client.test_gradient_draft import View, ed, qapp, small_stack, wait  # noqa: F401  (fixtures by name)

DRAFT = {"method": "rbf", "samples": 32}                         # slow enough to change workflow while it runs


def test_a_preview_of_another_workflow_is_never_shown(ed, monkeypatch):
    sm = small_stack()
    monkeypatch.setattr(ge, "_reduced_stack", lambda project, pid: {**sm, "arr": sm["arr"] * pid, "pid": pid})
    shown = []
    present = ed._present
    monkeypatch.setattr(ed, "_present", lambda *a, **k: (shown.append((ed.p.pid, ed.result.copy())),
                                                         present(*a, **k))[1])
    q = View(2, "Variant")
    ed.s.paths.append(q)
    ed.set_draft(DRAFT)
    end = time.monotonic() + 10
    while not ed.busy and time.monotonic() < end:                # workflow 1's preview under way
        QApplication.processEvents()
        time.sleep(0.005)
    assert ed.busy
    ed._load(q)                                                  # workflow 2 shown, the same settings drafted
    ed.set_draft(DRAFT)
    wait(ed, 90)
    want = ge._preview({**sm, "arr": sm["arr"] * 2, "pid": 2}, GradientParams(**DRAFT))["result"]
    mine = [r for pid, r in shown if pid == 2]
    assert mine, "workflow 2's preview was not shown"
    for r in mine:                                               # every picture shown on workflow 2 is its own
        assert float(np.abs(r - want).max()) < 1e-3 * float(np.abs(want).max())
