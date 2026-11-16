"""A layer duplicated in Post (Ctrl+J), then every step it made undone: the project's graph is the one it was. The Post
sweep (test_sweep_post.py, level 3) found a Combine layers block left in the graph after that Undo, on the test project;
here on the small project of test_post_versions.py, each layer duplicated in turn. A duplicate is a step of Post's own
Undo (the window undoes it on Post, a step of the workflows Post made included): undone so, the graph follows back. The
check undid the workflows' list alone, which never held that step."""
from __future__ import annotations

import pytest

from tests.client.sweep import definition, first_difference
from tests.client.test_post_versions import _done, _page_of, by_name, model, qapp, two  # noqa: F401  (fixtures by name)


@pytest.mark.parametrize("which", [0, -1])
def test_a_layer_duplicated_in_post_then_undone_leaves_the_graph_as_it_was(model, which):  # noqa: F811
    model.add_post_layer(by_name(model, "Full").pid, "stars")
    model.add_post_layer(by_name(model, "Classic stack").pid, None)
    page = _page_of(model)
    v = page.view
    try:
        v._save()
        n, m, before = len(model._undo), len(v.undo), definition(model)
        v.select(v.layers[which]["id"], "layer")
        v.duplicate()
        v._save()
        k = 0
        while len(v.undo) > m and k < 20:                  # Post's own steps, as the window undoes them on Post
            v.do_undo()
            k += 1
        while len(model._undo) > n and k < 40:
            model.undo()
            k += 1
        after = definition(model)
        assert after == before, first_difference(before, after)
    finally:
        _done(page)
