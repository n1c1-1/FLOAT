"""A variant's copy of a block is the block with its method and its engine settings: the panel of a Gradient variant of
an RBF Gradient shows the RBF settings as the original's does (it showed the polynomial ones: the copy had lost its
method), with the original's values. On a project model (legacy.World: a scratch project, nothing computed)."""
from __future__ import annotations

import pytest

from astro_gpu.client.settings import catalog_key, step_settings
from tests.client.graphview.legacy import World


@pytest.fixture()
def W(qapp, tmp_path):
    w = World(tmp_path / "w", workflows=(), chains={"Main": ["calibration", "registration", "selection", "stacking",
                                                             "gradient"]})
    yield w
    w.close()


def _rows(W, p):
    for _i in range(3):                              # the worker describes the stages' settings
        W.m.params(p, "gradient")
        W.settle(10)
    n = next(n for n in W.bm.nodes if n.key == "fond" and p in n.paths)
    return catalog_key(p, n.key), [(r["name"], r["value"]) for _t, rs in step_settings(W.m, p, n.key) for r in rs]


def test_a_gradient_variant_shows_the_rbf_settings_of_its_original(W):
    m = W.m
    g = W.one("gradient")
    main = m.paths[0]
    assert m.set_node_method([main], "fond", "rbf")
    W.settle()
    m.set_param(main, "gradient", "rbf_smooth", 0.4)
    W.settle()
    assert W.b.act("add_variant", block=g).ok
    W.settle()
    copy = next(b for r in W.graph.rows if r.fork for b in r.blocks)
    assert W.graph.blocks[copy].method == "rbf" and W.graph.blocks[copy].params == W.graph.blocks[g].params
    var = next(q for q in m.paths if q.pid != main.pid)
    want = _rows(W, main)
    assert want[0] == "fond:rbf" and ("rbf_smooth", 0.4) in want[1]
    assert _rows(W, var) == want
