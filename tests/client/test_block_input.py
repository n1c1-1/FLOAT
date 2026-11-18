"""Inspect: the picture a block is compared with is what the block really reads (the graph's order), not the stack with
its gradient removed whatever the graph is. The tab of a block that reads Color's output was compared with the
uncalibrated image, shown with the calibrated image's stretch (a yellow before)."""
from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from astro_gpu import api  # noqa: E402
from astro_gpu.client.services import inspect_sources as src  # noqa: E402
from astro_gpu.client.steps import Steps  # noqa: E402

P = SimpleNamespace(pid=1, name="WF")
DEPS = {"gradient": ("stack", "color"), "color": ("stack", "gradient"), "denoise": ("stack", "gradient", "color"),
        "deconv": ("stack", "gradient", "color")}


class FakeModel:
    def __init__(self, chain):
        self.chain = chain
        self.p = SimpleNamespace()

    def all_stages(self, p):
        return list(self.chain)

    def state_info(self, p, stage):
        return SimpleNamespace(state="done", key=f"k-{stage}", previous_key="") if stage in self.chain else None


def steps_of(chain, monkeypatch) -> Steps:
    monkeypatch.setattr(api, "chain", lambda p, wid: list(chain))
    monkeypatch.setattr(api, "deps", lambda p, wid, stage: tuple(d for d in DEPS[stage] if d in chain
                                                                  and chain.index(d) < chain.index(stage)))
    s = Steps.__new__(Steps)
    s.s = FakeModel(chain)
    return s


def test_denoise_after_color_is_compared_with_the_colour_calibrated_input(monkeypatch):
    steps = steps_of(["stack", "gradient", "color", "denoise"], monkeypatch)
    den = steps.visuals(P, "debruitage")[0]
    assert den.compare[0] == "input:denoise" and not den.camera           # reads Color's output: not camera colours
    before = steps.picture(P, "input:denoise")
    assert before.kind == "input" and not before.camera
    assert before.caption == "What the block reads (Stack, gradient removed, colour-calibrated)"
    assert steps.stages_of(P, before) == ["denoise", "stack", "gradient", "color"]      # its key is theirs


def test_denoise_before_color_reads_the_camera_colours(monkeypatch):
    steps = steps_of(["stack", "gradient", "denoise", "color"], monkeypatch)
    den = steps.visuals(P, "debruitage")[0]
    assert den.camera
    before = steps.picture(P, "input:denoise")
    assert before.camera and before.caption == "What the block reads (Stack, gradient removed)"


def test_a_block_reading_the_stack_alone_is_compared_with_the_stack(monkeypatch):
    steps = steps_of(["stack", "denoise"], monkeypatch)
    assert steps.picture(P, "input:denoise").key == "stack"


def test_the_gradient_block_after_color_removes_its_sky_from_the_calibrated_image(monkeypatch):
    steps = steps_of(["stack", "color", "gradient"], monkeypatch)
    spec = steps.visuals(P, "fond")[0]
    assert spec.compare[0] == "input:gradient" and not spec.camera
    assert steps.stages_of(P, spec) == ["gradient", "stack", "color"]


def test_the_sources_apply_the_blocks_in_the_graphs_order(monkeypatch):
    stack = np.stack([np.full((4, 4), v, np.float32) for v in (10, 20, 30)])
    M = np.diag([2.0, 1.0, 0.5]).astype(np.float32)
    sky = np.full((3, 4, 4), 1.0, np.float32)
    monkeypatch.setattr(api, "chain", lambda p, wid: ["stack", "gradient", "color", "denoise"])
    monkeypatch.setattr(api, "deps", lambda p, wid, e: DEPS[e])
    monkeypatch.setattr(src, "_stk", lambda *a: "stack")
    monkeypatch.setattr(src, "_array", lambda p, wid, st, name, which: {"drz_full": stack, "M": M}[name].copy())
    monkeypatch.setattr(src, "_sky_model", lambda p, wid, which, H, W, entry="gradient": (sky, None))
    got = src.input_source(None, 1, "denoise")()
    want = np.stack([(stack[c] - 1.0) * M[c, c] for c in range(3)])
    assert np.allclose(got, want)
    # Gradient placed after Color takes its sky off the calibrated image
    monkeypatch.setattr(api, "chain", lambda p, wid: ["stack", "color", "gradient"])
    monkeypatch.setattr(api, "deps", lambda p, wid, e: ("stack", "color"))
    got = src.gradient_source(None, 1, "gradient")()
    assert np.allclose(got, np.stack([stack[c] * M[c, c] - 1.0 for c in range(3)]))
