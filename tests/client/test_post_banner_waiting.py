"""The empty-canvas banner tells layers all hidden by their eye from visible layers still waiting for a result."""
from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")
from astro_gpu.client import post3  # noqa: E402
from astro_gpu.client.postengine import new_workflow_layer  # noqa: E402


def _banner(layers):
    return post3.PostView._empty_banner(SimpleNamespace(layers=layers))


def test_banner_hidden_versus_waiting():
    a, b = new_workflow_layer(1, "A", False), new_workflow_layer(2, "B", False)
    assert _banner([a, b]) == "Every image layer is hidden"
    b["visible"] = True
    assert _banner([a, b]) == "Waiting for the layers' results"
    assert _banner([]) == "No layers yet"
