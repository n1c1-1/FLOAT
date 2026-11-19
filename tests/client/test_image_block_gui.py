"""An Image block's workflow in the window (the tester's pass on M27): the Gradient editor reads the Image block's result
as its stack, the notice of a new Image block names no placeholder, Open images fills a block with no file as its panel's
Choose image… does, and Kind says that a file whose stretch is undone is read as linear."""
from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
from PySide6.QtWidgets import QPushButton  # noqa: E402

from astro_gpu.client import gradientedit as ge  # noqa: E402
from astro_gpu.client.model import ProjectModel  # noqa: E402
from astro_gpu.client.steps import Steps  # noqa: E402
from tests.client.test_sources_import import close, open_page, qapp, settle, texts  # noqa: E402,F401


class Chain:
    """What the project model says of a workflow's chain."""

    def __init__(self, stages):
        self.stages, self.states = stages, {}

    def all_stages(self, p=None):
        return list(self.stages)

    input_stage = ProjectModel.input_stage

    def state(self, p, stage):
        return self.states.get(stage, "pending")

    def state_info(self, p, stage):
        s = self.states.get(stage)
        return SimpleNamespace(state=s, key="k", previous_key=None) if s else None


def test_the_stack_of_an_image_workflow_is_its_image_block():
    img, stk = Chain(["image", "gradient"]), Chain(["register", "stack", "gradient"])
    assert img.input_stage(None) == "image" and stk.input_stage(None) == "stack"
    assert Steps.read_stage(SimpleNamespace(s=img), None, "stack") == "image"
    assert Steps.read_stage(SimpleNamespace(s=stk), None, "stack") == "stack"
    assert Steps.read_stage(SimpleNamespace(s=img), None, "gradient") == "gradient"


def test_the_gradient_editor_has_its_input_when_the_image_block_is_done():
    """It looked for the state of a "stack" stage that an Image workflow does not have: no input, the page black."""
    m = Chain(["image", "gradient"])
    ed = SimpleNamespace(s=m, p=SimpleNamespace(pid=1), drafting=lambda p: False, carried={}, tv=None)
    ed._input_stage = lambda p: ge.GradientEditor._input_stage(ed, p)
    m.states["image"] = "pending"
    assert ge.GradientEditor.no_input(ed, ed.p) is True
    m.states["image"] = "done"
    assert ge.GradientEditor.no_input(ed, ed.p) is False
    assert ge.GradientEditor._stack_key(ed) == "k"


def _png(tmp_path):
    import cv2
    png = tmp_path / "done.png"
    cv2.imwrite(str(png), np.full((8, 8, 3), 128, np.uint8))
    return png


def test_a_new_image_block_names_no_placeholder(qapp, tmp_path, monkeypatch):
    w, m, page = open_page(tmp_path, monkeypatch)
    try:
        said = []
        page.bridge.toast.connect(said.append)
        page.palette.double_click("image")
        settle(w, page)
        assert said and "Image 1" not in said[-1] and said[-1].startswith("Image added")
    finally:
        close(w, m, page)


def test_open_images_fills_a_block_with_no_file(qapp, tmp_path, monkeypatch):
    png = _png(tmp_path)
    w, m, page = open_page(tmp_path, monkeypatch)
    try:
        page.palette.double_click("image")
        settle(w, page)
        n = len(m.paths)
        assert page.place_images([str(png)]) is not None
        settle(w, page)
        assert len(m.paths) == n and m.p.workflow(m.paths[-1].pid).source["path"] == png.as_posix()
        assert page.placed_new is False and m.next_words("undo") == "Image chosen: done.png"
        page.place_images([str(png)])                              # none waits now: a workflow of its own
        settle(w, page)
        assert len(m.paths) == n + 1 and page.placed_new is True
    finally:
        close(w, m, page)


def test_kind_says_when_the_stretch_is_undone(qapp, tmp_path, monkeypatch):
    png = _png(tmp_path)
    w, m, page = open_page(tmp_path, monkeypatch)
    try:
        page.palette.double_click("image")
        settle(w, page)
        q = m.paths[-1]
        m.set_image_file(q, str(png), stretched=False)
        page.select(index=m.paths.index(q), step="image")
        monkeypatch.setattr(m, "image_state", lambda p: {"warning": "", "estimate_off": "", "estimated": False,
                                                          "exact": True, "lines": "x"})
        page.refresh()
        assert "Linear (undone)" in texts(page.panel) and "Linear" not in texts(page.panel)
        monkeypatch.undo()
        page.refresh()
        assert "Linear" in texts(page.panel)
    finally:
        close(w, m, page)
