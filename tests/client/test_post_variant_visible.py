"""A variant's layer arrives visible in Post."""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication  # noqa: E402

from astro_gpu import api  # noqa: E402
from astro_gpu.client.model import ProjectModel  # noqa: E402
from astro_gpu.client.post3 import PostPage3  # noqa: E402
from astro_gpu.client.services.render import RenderWorker  # noqa: E402
from astro_gpu.client.services.worker import ApiWorker  # noqa: E402
from astro_gpu.core import project2  # noqa: E402
from astro_gpu.core.project import Target  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def test_variant_layer_arrives_visible(qapp, tmp_path):
    root = tmp_path / "P"
    p = project2.create(root, "P", Target(name="M 1"))
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "a1.ARW").write_bytes(b"\0" * 100)
    p.sessions.append(project2.Session(1, "night 1", lights=[str(raw)]))
    api.add_workflow(p, "Classic stack", "Classic stack")
    api.add_workflow(p, "Other", "Classic + denoise")
    project2.save(p)
    w = ApiWorker()
    m = ProjectModel(project2.load(root), w)
    assert w.wait(120) and m.ready, m.error
    for v in m.paths:
        v.rendered = True
    page = PostPage3(m, worker=RenderWorker(None, resolver=lambda sid: None, device="cpu"))
    try:
        v = page.view
        assert len(m.paths) >= 2
        orig, var = m.paths[0], m.paths[1]
        keep = next(x for x in v.layers if x["kind"] == "workflow" and x["pid"] == orig.pid)
        v.layers = [x for x in v.layers if x["kind"] != "workflow" or x["pid"] == orig.pid]
        keep["visible"] = False
        var.variant_of = orig.pid
        v.sync_paths()
        lay = next(x for x in v.layers if x["kind"] == "workflow" and x["pid"] == var.pid)
        assert lay["visible"] is True                      # the variant: made to be looked at
        assert keep["visible"] is False                    # the original's stays as it was
    finally:
        page.shutdown()
        page.deleteLater()
        m.close(save=False)
        w.stop()
