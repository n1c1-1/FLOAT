"""test_post_versions.py run against the model with the graph as its truth: every test of that file, its fixture
building ``M(GraphEditing, ProjectModel)`` instead of ``ProjectModel`` (the file itself is not changed)."""
from __future__ import annotations

import pytest

from astro_gpu import api
from astro_gpu.client.graphmodel import GraphEditing
from astro_gpu.client.model import ProjectModel
from astro_gpu.client.services.worker import ApiWorker
from astro_gpu.core import project2
from astro_gpu.core.project import Target
from test_post_versions import *  # noqa: F401,F403  (its tests, collected here with the fixture below)


class M(GraphEditing, ProjectModel):
    pass


@pytest.fixture(scope="module")
def two(qapp, tmp_path_factory):  # noqa: F811 - the same project as test_post_versions.two, on M
    root = tmp_path_factory.mktemp("versions") / "P"
    p = project2.create(root, "P", Target(name="M 1"))
    raw = root.parent / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    (raw / "a1.ARW").write_bytes(b"\0" * 100)
    p.sessions.append(project2.Session(1, "night 1", lights=[str(raw)]))
    api.add_workflow(p, "Full", "Full")
    api.add_workflow(p, "Classic stack", "Classic stack")
    w = ApiWorker()
    m = M(p, w, readonly=True)
    assert w.wait(120) and m.ready, m.error
    assert isinstance(m, GraphEditing)
    yield m
    m.close(save=False)
    w.stop()
