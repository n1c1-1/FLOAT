"""Post's Undo and what the Graph made: a step made in the Graph is not Post's to undo, and the undone step of one version
leaves nothing in the others."""
from test_post_versions import _done, _layers_of, _page_of, by_name, model, qapp, two  # noqa: F401


def _names(layers):
    return [x["name"] for x in layers]


def _synth_rows(layers):
    return [x for x in layers if x.get("kind") == "filter" and x.get("filter") == "synth"]


def test_a_combine_layers_made_in_the_graph_is_not_undone_by_post_and_the_synthetic_stars_leave_every_version(model):
    full = by_name(model, "Full")
    page = _page_of(model)
    v = page.view
    try:
        first = model.p.post["active"]
        model.add_post_layer(full.pid, "combined")                 # Stars' Combine layers, made in the Graph
        made = _names(v.layers)
        assert made.count("Full · Combine layers 2") == 1 or any("Combine layers" in n for n in made)
        v._version_new()
        second = model.p.post["active"]
        model.set_post_version(first)
        group = next(x for x in v.layers if x.get("part") == "combined" and not x["name"].endswith("2"))
        v.add_synth(group)                                         # Stars row > Synthetic stars, in the first version
        assert _synth_rows(v.layers)
        model.set_post_version(second)
        v.add_filter("curves")                                     # in the second
        v._save()
        model.set_post_version(first)
        v.do_undo()                                                # the synthetic stars go from the first version
        assert not _synth_rows(v.layers) and not _synth_rows(_layers_of(model, first))
        assert not _synth_rows(_layers_of(model, second))          # and are nowhere else
        assert any(x["name"] == "Curves" for x in _layers_of(model, second))
        n = len(v.layers)
        said = []
        v.toasted.connect(said.append)
        v.do_undo()                                                # the Graph's Combine layers stays: not a step of Post
        assert len(v.layers) == n and _names(v.layers) == made
        assert said and said[-1].startswith(("Nothing more to undo in this version", "Nothing to undo here"))
    finally:
        _done(page)
