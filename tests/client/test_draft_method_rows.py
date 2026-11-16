"""A method chosen on a computed block is the block's draft: the Graph's settings panel lists the settings of the
method the draft holds, at once, not those of the applied method until Apply (the tutorials' pass: Gradient set to RBF
kept showing "Surface degree", a setting of Polynomial). On a copy of the computed test project."""
from __future__ import annotations

import os

import pytest
from PySide6.QtWidgets import QLabel

from astro_gpu.client import prefs
from astro_gpu.client.graphnode import Node
from astro_gpu.client.panel import CompactRow, MethodCard, Panel
from tests.client.test_inspect import open_window, qapp, settle  # noqa: F401  (qapp: the fixture)
from conftest import needs_test_project, scratch_project  # noqa: E402  (tests/conftest.py)

pytestmark = needs_test_project


@pytest.fixture()
def win(qapp, tmp_path):  # noqa: F811
    folder = scratch_project(tmp_path / "heart")
    os.environ["ASTRO_GPU_SETTINGS"] = str(tmp_path / "settings.json")
    prefs.load()
    w = open_window(folder)
    yield w
    w.s.pending_change = None
    w.close()
    os.environ.pop("ASTRO_GPU_SETTINGS", None)


def _rows(pn):
    return {r.r["name"] for r in pn.findChildren(CompactRow) if r.r.get("stage") == "gradient"}


def test_real_rows_are_those_of_the_method_the_draft_holds(win):
    s = win.s
    node = Node("fond", [s.path])
    pn = Panel(s, win.steps)                           # the Graph's panel (Inspect's Gradient has its own editor)
    pn.changed.connect(lambda: pn.refresh(pn.node))
    pn.refresh(node)
    settle(win, 2)
    assert s.path.opts["fond"] == "simple" and "degree" in _rows(pn)
    rbf = next(c for c in pn.findChildren(MethodCard)
               if c.ok and not c.on and any(lb.text() == "RBF" for lb in c.findChildren(QLabel)))
    rbf.clicked.emit()
    settle(win, 2)
    assert s.pending_change is not None and s.path.opts["fond"] == "simple"        # a draft: nothing applied
    assert _rows(pn) == {"rbf_smooth"}                 # RBF's setting, not Polynomial's degree
    pn.draft_reset()
    pn.refresh(node)
    settle(win, 2)
    assert "degree" in _rows(pn) and "rbf_smooth" not in _rows(pn)
