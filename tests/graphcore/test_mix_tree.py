"""#181: the layer tree Post nests (graphcore/post.mix_tree), derived from the graph: what a Combine layers or a
combination wired to Post shows as a mix of what it reads, which blocks fix a combination's image (said, never silent),
and how deep the nesting goes (rules.MAX_NEST). Pure on block ids, no Qt."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from fixtures import Doc, _join, _last, _ok, _stack  # noqa: E402
from test_gestures3b import GALLERY  # noqa: E402
from astro_gpu.graphcore import post, rules, templates  # noqa: E402
from astro_gpu.graphcore.model import Index  # noqa: E402


def kind_of(g, b):
    return g.blocks[b].kind


def layers(doc):
    g = doc.project
    return next(b for b, x in g.blocks.items() if x.kind == "combine_layers")


def into(g, block, port):
    return next(w.src for w in g.wires if w.dst == [block, port] and w.active)


def shape(g, n):
    """A node as the kinds it is made of: easy to read in a test."""
    k = n["kind"]
    if k == "layers":
        return ("layers", {s: shape(g, c) for s, c in n["sides"].items()})
    if k == "mix":
        return ("mix", {r: shape(g, c) for r, c in n["roles"].items()})
    if k == "fixed":
        return ("fixed", n["why"], g.blocks[n["by"]].kind if n["by"] else None)
    return "image"


def test_the_nestable_kinds_are_the_combination_family_and_the_depth_is_three():
    assert [k for k in rules.ORDER if rules.nestable(k)] == ["combine", "combine_layers"]
    assert tuple(k for k in rules.ORDER if rules.nestable(k)) == rules.GRAPH_SETTING_KINDS
    assert rules.MAX_NEST == 3


def test_filters_full_both_sides_are_a_mix_of_their_filters():
    doc = Doc.from_template("filters_complete")
    g = doc.project
    t = post.mix_tree(g, layers(doc), "image")
    assert t["kind"] == "layers" and t["block"] == layers(doc)
    assert shape(g, t) == ("layers", {"starless": ("mix", {"SII": "image", "Ha": "image", "OIII": "image"}),
                                      "stars": ("mix", {"SII": "image", "Ha": "image", "OIII": "image"})})
    sl, st = t["sides"]["starless"], t["sides"]["stars"]
    assert sl["block"] == into(g, layers(doc), "Background")[0] and st["block"] == into(g, layers(doc), "Stars")[0]
    assert sl["roles"]["Ha"]["port"] == "Starless" and st["roles"]["Ha"]["port"] == "Stars only"


def test_sho_with_the_stars_of_an_rgb_stack_is_a_mix_over_a_plain_stars_image():
    doc = Doc.from_template("sho_rgb")
    g = doc.project
    t = post.mix_tree(g, layers(doc), "image")
    assert shape(g, t) == ("layers", {"starless": ("mix", {"SII": "image", "Ha": "image", "OIII": "image"}),
                                      "stars": "image"})


def test_a_combination_wired_straight_to_post_is_a_mix_root():
    doc = Doc.from_template("filters_minimal")
    g = doc.project
    ix = Index(g)
    w = next(w for w in g.wires if w.dst[0] == ix.post)
    assert shape(g, post.mix_tree(g, w.src[0], w.src[1])) == ("mix", {"Ha": "image", "OIII": "image"})


@pytest.mark.parametrize("t", GALLERY, ids=lambda t: t["key"])
def test_the_gallery_s_combinations_are_mixes_of_the_filters_they_read(t):
    doc = Doc()
    plan = templates.build(doc.project, t, "S-H-O", dict(t["opts"]))
    assert doc.apply_many(plan.steps).ok
    g = doc.project
    joins = [b for b, x in g.blocks.items() if x.kind == "combine"]
    assert joins                                        # (the starless S-H-O; its "· Stars" twin when there is one)
    for j in joins:
        assert shape(g, post.mix_tree(g, j, "image")) == ("mix", {"SII": "image", "Ha": "image", "OIII": "image"})


def _background_through(doc, kind):
    """The starless S-H-O of a layers' Background read through a block of ``kind`` placed after it in its row."""
    g = doc.project
    cc = layers(doc)
    j = into(g, cc, "Background")[0]
    _ok(doc, "add_block", kind=kind, at={"type": "insert", "row": Index(g).row_of[j], "after": j})
    k = _last(doc, Index(g).row_of[j])
    return cc, j, k


def test_a_stars_background_after_the_combination_fixes_it_and_names_that_block():
    doc = Doc.from_template("filters_complete")
    g = doc.project
    cc, j, k = _background_through(doc, "stars")
    _ok(doc, "unwire", wire=next(w.id for w in doc.project.wires if w.dst == [cc, "Background"]))
    _ok(doc, "wire", src=k, src_port="Starless", dst=cc, dst_port="Background")
    g = doc.project
    t = post.mix_tree(g, cc, "image")
    n = t["sides"]["starless"]
    assert n == {"kind": "fixed", "block": j, "by": k, "why": "after"} and kind_of(g, k) == "stars"
    assert t["sides"]["stars"]["kind"] == "mix"


def test_a_color_between_the_combination_and_post_fixes_it_and_names_the_color():
    doc = Doc.from_template("filters_minimal")
    g = doc.project
    ix = Index(g)
    j = next(b for b, x in g.blocks.items() if x.kind == "combine")
    _ok(doc, "add_block", kind="color", at={"type": "insert", "row": ix.row_of[j], "after": j})
    g = doc.project
    k = _last(doc, ix.row_of[j])
    t = post.mix_tree(g, k, "image")
    assert t == {"kind": "fixed", "block": j, "by": k, "why": "after"} and kind_of(g, k) == "color"


def test_a_block_after_the_combination_switched_off_passes_the_image_as_it_is():
    doc = Doc.from_template("filters_minimal")
    g = doc.project
    ix = Index(g)
    j = next(b for b, x in g.blocks.items() if x.kind == "combine")
    _ok(doc, "add_block", kind="color", at={"type": "insert", "row": ix.row_of[j], "after": j})
    g = doc.project
    k = _last(doc, ix.row_of[j])
    g.blocks[k].off = True
    assert shape(g, post.mix_tree(g, k, "image")) == ("mix", {"Ha": "image", "OIII": "image"})


def _chain(doc, n):
    """``n`` combinations one reading the next one's image as its Hα (the first reads two stacks), the Combine layers of
    ``doc`` reading the last as its Background."""
    a = _stack(doc, "A", ["gradient", "denoise", "stars"])
    b = _stack(doc, "B", ["gradient", "denoise", "stars"])
    cur = None
    for i in range(n):
        j = _join(doc, "combine", "ho")
        if cur is None:
            _ok(doc, "wire", src=_last(doc, a), src_port="Starless", dst=j, dst_port="Hα")
        else:
            _ok(doc, "wire", src=cur, src_port="image", dst=j, dst_port="Hα")
        _ok(doc, "wire", src=_last(doc, b), src_port="Starless", dst=j, dst_port="O III")
        cur = j
    cc = _join(doc, "combine_layers")
    _ok(doc, "wire", src=cur, src_port="image", dst=cc, dst_port="Background")
    return cc, cur, doc.project


def test_a_combination_reading_a_combination_is_a_mix_in_a_mix():
    doc = Doc()
    cc, top, g = _chain(doc, 2)
    t = post.mix_tree(g, cc, "image")
    assert shape(g, t) == ("layers", {"starless": ("mix", {"Ha": ("mix", {"Ha": "image", "OIII": "image"}),
                                                           "OIII": "image"})})


def test_the_fourth_level_is_fixed_and_names_the_combination_that_reads_it():
    doc = Doc()
    cc, top, g = _chain(doc, 3)               # layers (1) > C (2) > B (3) > A (4)
    t = post.mix_tree(g, cc, "image")
    a = t["sides"]["starless"]["roles"]["Ha"]["roles"]["Ha"]
    b = t["sides"]["starless"]["roles"]["Ha"]
    assert a == {"kind": "fixed", "block": into(g, b["block"], "Hα")[0], "by": b["block"], "why": "depth"}
    assert shape(g, t)[1]["starless"][1]["Ha"][1]["Ha"][:2] == ("fixed", "depth")


def test_three_combinations_without_a_combine_layers_are_the_most_and_the_fourth_is_fixed():
    doc = Doc()
    cc, top, g = _chain(doc, 4)
    t = post.mix_tree(g, top, "image")        # C4 (1) > C3 (2) > C2 (3) > C1 (4)
    assert t["kind"] == "mix"
    third = t["roles"]["Ha"]["roles"]["Ha"]
    assert third["kind"] == "mix" and third["roles"]["Ha"]["kind"] == "fixed"
    assert third["roles"]["Ha"]["why"] == "depth" and third["roles"]["Ha"]["by"] == third["block"]


def test_a_recipe_composed_as_lightness_and_colour_is_fixed_by_its_own_reason():
    doc = Doc()
    r = [_stack(doc, n, []) for n in ("L", "R", "G", "B")]
    j = _join(doc, "combine", "lrgb")
    for rid, port in zip(r, ("L", "R", "G", "B")):
        _ok(doc, "wire", src=_last(doc, rid), src_port="image", dst=j, dst_port=port)
    t = post.mix_tree(doc.project, j, "image")
    assert t == {"kind": "fixed", "block": j, "by": None, "why": "lightness"}


def test_a_filter_row_is_an_image_even_when_it_is_a_stars_row_of_a_combination_free_workflow():
    doc = Doc.from_template("colour_complete")
    g = doc.project
    t = post.mix_tree(g, layers(doc), "image")
    assert shape(g, t) == ("layers", {"starless": "image", "stars": "image"})
