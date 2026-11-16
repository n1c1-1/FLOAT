"""A layer duplicated in Post (Ctrl+J), then every step it made undone: the project's graph is the one it was. The Post
sweep (test_sweep_post.py, level 3) found a Combine layers block left in the graph after that Undo, on the test project;
here on the small project of test_post_versions.py, each layer duplicated in turn."""
from __future__ import annotations

import pytest

from tests.client.sweep import definition, first_difference
from tests.client.test_post_versions import _done, _page_of, by_name, model, qapp, two  # noqa: F401  (fixtures by name)


DEFECT = ("the layer of a workflow's part (Full, stars alone) duplicated in Post adds a Combine layers block to the "
          "graph that no Undo takes away: the definition keeps it once every step of the duplicate is undone")


@pytest.mark.parametrize("which", [0, pytest.param(-1, marks=pytest.mark.xfail(strict=True, reason=DEFECT))])
def test_a_layer_duplicated_in_post_then_undone_leaves_the_graph_as_it_was(model, which):  # noqa: F811
    model.add_post_layer(by_name(model, "Full").pid, "stars")
    model.add_post_layer(by_name(model, "Classic stack").pid, None)
    page = _page_of(model)
    v = page.view
    try:
        v._save()
        n, before = len(model._undo), definition(model)
        v.select(v.layers[which]["id"], "layer")
        v.duplicate()
        v._save()
        k = 0
        while len(model._undo) > n and k < 20:
            model.undo()
            k += 1
        after = definition(model)
        assert after == before, first_difference(before, after)
    finally:
        _done(page)
