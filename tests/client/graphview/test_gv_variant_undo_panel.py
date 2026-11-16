"""A variant tried then undone (Ctrl+Z): its block is gone, and the page's panel shows the block the variant came from,
as a block removed leaves the one before it chosen (it stayed open and empty)."""
from __future__ import annotations

import pytest

from tests.client.graphview.test_gv_readings import M, _project
from tests.client.graphview.test_gv_sides import _model_page, _wait


@pytest.fixture()
def model(qapp, tmp_path_factory):
    from astro_gpu.client.services.worker import ApiWorker
    from astro_gpu.core import project2
    w = ApiWorker()
    m = M(project2.load(_project(tmp_path_factory.mktemp("variant_undo"))), w, readonly=True)
    assert w.wait(120) and m.ready, m.error
    yield m
    m.close(save=False)
    w.stop()


def test_the_panel_leaves_a_tried_variant_undone(model):
    page = _model_page(model)
    try:
        p = model.paths[0]
        q = model.add_variant(p, {"kind": "param", "stage": "calibrate", "name": "hot_sigma", "value": 0.7},
                              f"{p.name} · variant")
        _wait(50)
        made = next(r for r in page.bridge.graph.rows if r.wid == q.pid)
        orig = next(r for r in page.bridge.graph.rows if r.wid == p.pid)
        assert not made.fork and made.blocks                 # a whole stack's copy: no block before it remains
        b, came_from = made.blocks[0], orig.blocks[0]
        page.bridge.choose(b)
        _wait(50)
        assert page.panel.node is not None
        model.undo()
        _wait(100)
        assert b not in page.bridge.graph.blocks
        assert page.bridge.selected == came_from and page.panel.node is not None
    finally:
        page.close()

