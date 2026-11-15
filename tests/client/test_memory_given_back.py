"""Memory given back: the master darks and flats Inspect read go when their project closes; the engine of an image
sent to Post goes once its layer is deleted (its source stays, for an Undo)."""
from __future__ import annotations

import os

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from types import SimpleNamespace as NS  # noqa: E402

from PySide6.QtWidgets import QApplication  # noqa: E402

from astro_gpu.client.postengine import new_image_layer  # noqa: E402
from astro_gpu.client.services import inspect as si  # noqa: E402
from astro_gpu.client.services.render import LayerSource, Renderer, Request  # noqa: E402


def test_the_masters_of_a_project_go_when_it_closes(tmp_path):
    QApplication.instance() or QApplication([])
    mine, other = tmp_path / "mine", tmp_path / "other"
    for root in (mine, other):
        d = root / "store" / "calibrate" / "k1"
        d.mkdir(parents=True)
        np.save(d / "master_dark.npy", np.zeros((4, 4), np.float32))
        np.save(d / "master_flat.npy", np.ones((4, 4), np.float32))
        np.save(d / "badmask.npy", np.zeros((4, 4), bool))
        si._calib_masters(str(d))
    try:
        assert len(si._masters) == 2
        sv = si.InspectService(NS(p=NS(root=str(mine))))
        sv.stop()                                                # the project closed
        assert list(si._masters) == [str(other / "store" / "calibrate" / "k1")]   # another window's stays
    finally:
        si._masters.clear()


def test_the_engine_of_a_deleted_image_layer_goes():
    src = LayerSource(lambda pid: None, service=None)
    src.image_sources.update(a=np.full((3, 40, 60), 0.2, np.float32), b=np.full((3, 40, 60), 0.6, np.float32))
    r = Renderer(src)
    a, b = new_image_layer("a", "a"), new_image_layer("b", "b")
    r.run(Request(kind="half", layers=[b, a], level=0))
    assert set(src._img) == {"a", "b"}
    r.run(Request(kind="half", layers=[a], level=0))            # b deleted
    assert set(src._img) == {"a"} and "b" in src.image_sources
    r.run(Request(kind="half", layers=[b, a], level=0))         # Undo: read again
    assert set(src._img) == {"a", "b"}
