"""Post's edits wait for Save like the graph's: nothing of them is in the project's file before it (no autosave)."""
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


def _open(root):
    w = ApiWorker()
    m = ProjectModel(project2.load(root), w)
    assert w.wait(120) and m.ready, m.error
    _rendered(m)
    return m, w


def _rendered(model):
    """The workflows as if computed (the page then lists their layers)."""
    orig = model._sync

    def fake(*a, **k):
        orig(*a, **k)
        for v in model.paths:
            v.rendered = True
    model._sync = fake
    model._sync()


def _post_on_disk(root):
    p = project2.load(root)
    return [x.get("mode") for x in project2.active_version(p.post).get("layers") or []]


def test_post_edits_reach_the_file_only_with_save(qapp, tmp_path):
    root = tmp_path / "P"
    p = project2.create(root, "P", Target(name="M 1"))
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "a1.ARW").write_bytes(b"\0" * 100)
    p.sessions.append(project2.Session(1, "night 1", lights=[str(raw)]))
    api.add_workflow(p, "Classic stack", "Classic stack")
    project2.save(p)
    m, w = _open(root)
    page = PostPage3(m, worker=RenderWorker(None, resolver=lambda sid: None, device="cpu"))
    try:
        v = page.view
        lay = v.layers[0]
        v._set(lay, "mode", "Screen")
        v.flush()
        qapp.processEvents()
        assert m.dirty                                    # the edit is the unsaved state of the project
        assert "Screen" not in _post_on_disk(root)        # nothing written: reopening shows the saved composition
        assert m.flush()                                  # Save
        assert "Screen" in _post_on_disk(root)
    finally:
        page.shutdown()
        page.deleteLater()
        m.close(save=False)
        w.stop()


def test_post_edits_dropped_on_close_without_saving(qapp, tmp_path):
    root = tmp_path / "P"
    p = project2.create(root, "P", Target(name="M 1"))
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "a1.ARW").write_bytes(b"\0" * 100)
    p.sessions.append(project2.Session(1, "night 1", lights=[str(raw)]))
    api.add_workflow(p, "Classic stack", "Classic stack")
    project2.save(p)
    m, w = _open(root)
    page = PostPage3(m, worker=RenderWorker(None, resolver=lambda sid: None, device="cpu"))
    v = page.view
    v._set(v.layers[0], "mode", "Screen")
    v._set_crop((1, 1, 5, 5))
    page.shutdown()                                       # the window closing flushes Post's last edit into the model
    page.deleteLater()
    m.close(save=False)
    w.stop()
    assert "Screen" not in _post_on_disk(root)
    assert project2.active_version(project2.load(root).post).get("crop") is None
