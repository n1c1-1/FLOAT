"""The Project block asked for "filter" over sessions that each had their filter (the filters template sets one per
session): a filter given by every session counts as given."""
from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
from astro_gpu.client.model import ProjectModel  # noqa: E402


def _state(filters, acquisition=None):
    p = SimpleNamespace(sessions=[SimpleNamespace(filter=f) for f in filters], acquisition=acquisition or {})
    return ProjectModel.project_state(SimpleNamespace(p=p, target_state=lambda: SimpleNamespace(name="NGC 6995"),
                                              target_far_texts=lambda: []))


def test_a_filter_given_by_every_session_is_not_missing():
    st = _state(["Ha", "OIII", "SII"])
    assert "filter" not in st.missing and st.missing == ["instrument", "mount", "location", "date"], st.missing


def test_a_session_without_filter_leaves_it_missing_unless_the_project_gives_one():
    assert "filter" in _state(["Ha", ""]).missing
    assert "filter" in _state([]).missing
    assert "filter" not in _state(["Ha", ""], {"filter": "Ha"}).missing
