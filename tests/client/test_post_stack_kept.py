"""A layer with a stack of its own filters (the owner's NGC 7000: twelve on a 27 Mpx image): a release of one filter
makes that filter again, not the stack under it, at the screen's levels and at full resolution where there is room;
full-resolution images never take the room of the screen's; the pixels are those computed from nothing."""
from __future__ import annotations

import numpy as np
from test_post import ToyProvider  # noqa: F401 - the toy layers
from test_post_cache import Toy

from astro_gpu.client import postengine as pe
from astro_gpu.client.postengine import ByteLRU, Composer, new_filter_layer, new_mask, new_part, new_workflow_layer


def stack():
    """A workflow layer with eight filters of its own, one with a luminance mask; returned top first."""
    a = new_workflow_layer(1, "A", True)
    fl = []
    for key, params in (("dehaze", {"amount": 0.3}), ("levels", {"gamma": 1.3}), ("saturation", {"strength": 30.0}),
                        ("balance", {"temperature": 0.6}), ("local_contrast", {"strength": 0.4}),
                        ("sharpen", {"strength": 0.5}), ("highlights", {"strength": 0.3}),
                        ("saturation", {"strength": 20.0})):
        f = new_filter_layer(key, a["id"])
        f["params"].update(params)
        fl.append(f)
    m = new_mask()
    p = new_part("luminance")
    p.update(lo=10.0, hi=60.0)
    m["parts"] = [p]
    fl[3]["mask"] = m
    return pe.normalize(list(reversed(fl)) + [a])          # top first: the last filter on top


def release(cp, layers):
    """What Post composes after a release: the quarter, the half (with the masks' thumbnails), the exact view."""
    return [cp.compose(pe.snapshot(layers), lv, thumbs=lv == 1, track=True) for lv in (2, 1, 0)]


def with_param(layers, i, key, value):
    out = pe.snapshot(layers)
    filters = [x for x in out if x["kind"] == "filter"]
    filters[i]["params"][key] = value
    return out


def test_a_release_of_the_top_filter_makes_that_filter_only_at_every_level(monkeypatch):
    applied = []
    real = Composer._apply

    def counted(self, fl, under, level, *a, **k):
        applied.append((fl["id"], level))
        return real(self, fl, under, level, *a, **k)
    monkeypatch.setattr(Composer, "_apply", counted)
    layers = stack()
    top = next(x for x in layers if x["kind"] == "filter")["id"]
    cp = Composer(ToyProvider(), cache_bytes=200_000_000)
    release(cp, layers)
    for k, v in enumerate((25.0, 31.0, 44.0)):
        applied.clear()
        release(cp, with_param(layers, 0, "strength", v))
        if k:                                       # (the first release of a filter finds what it reads at once too)
            assert applied == [(top, 2), (top, 1), (top, 0)], applied


def test_a_release_in_the_middle_of_the_stack_makes_it_from_there():
    layers = stack()
    cp = Composer(ToyProvider(), cache_bytes=200_000_000)
    release(cp, layers)
    filters = [x for x in layers if x["kind"] == "filter"]
    applied = []
    real = cp._apply
    cp._apply = lambda fl, under, level, *a, **k: (applied.append((fl["id"], level)), real(fl, under, level, *a, **k))[1]
    for v in (1.1, 1.2):
        applied.clear()
        release(cp, with_param(layers, 6, "gamma", v))
    above = [f["id"] for f in reversed(filters[:7])]          # levels and the filters over it, bottom first
    for lv in (2, 1, 0):
        assert [i for i, level in applied if level == lv] == above


def test_every_release_is_bit_identical_to_a_composition_from_nothing():
    """Releases of the top filter, of one in the middle, of a mask, an eye off and on, then back to the start: at each
    level, the picture and the thumbnails of a composer that keeps everything it may, and of one that keeps nothing."""
    layers = stack()
    states = [layers, with_param(layers, 0, "strength", 25.0), with_param(layers, 0, "strength", 31.0),
              with_param(layers, 6, "gamma", 1.1), with_param(layers, 6, "gamma", 1.2)]
    m = pe.snapshot(states[-1])
    next(x for x in m if x["kind"] == "filter" and x.get("mask"))["mask"]["feather"] = 6.0
    states.append(m)
    off = pe.snapshot(m)
    [x for x in off if x["kind"] == "filter"][2]["visible"] = False
    states += [off, m, layers]
    for cap in (200_000_000, 3_000_000):                    # room for all, room for little
        pr = ToyProvider()
        cp = Composer(pr, cache_bytes=cap)
        for s in states:
            got = release(cp, s)
            for lv, g in zip((2, 1, 0), got):
                ref = Composer(pr).compose(pe.snapshot(s), lv, thumbs=lv == 1, cache=False)
                assert np.array_equal(g.rgb, ref.rgb), (cap, lv)
                assert g.mask_thumbs.keys() == ref.mask_thumbs.keys()
                assert all(np.array_equal(g.mask_thumbs[k], ref.mask_thumbs[k]) for k in ref.mask_thumbs)
            for c in (cp.images, cp.steps, cp.backdrops, cp.parts):
                assert c.bytes <= c.cap                     # memory bounded


def test_full_resolution_never_takes_the_screen_s_room_and_goes_first_for_it():
    small = np.zeros((10, 10, 3), np.float32)
    big = np.zeros((20, 20, 3), np.float32)
    lru = ByteLRU(small.nbytes * 4)
    lru.begin(1)
    for k in range(3):
        assert lru.put(("screen", k), small)
    lru.begin(0)
    assert not lru.put("full", big)                          # no room without a screen image: not kept
    assert all(("screen", k) in lru.items for k in range(3))
    lru.begin(1)
    lru.items.pop(("screen", 2))                            # (room made by hand for a full image)
    lru.bytes -= small.nbytes
    lru.begin(0)
    assert lru.put("full2", np.zeros((10, 20, 3), np.float32))
    lru.begin(2)
    assert lru.put(("screen", 9), small) and lru.put(("screen", 10), small)
    assert "full2" not in lru.items                          # the screen's image took the full one's room first
    assert all(("screen", k) in lru.items for k in range(2))
    assert lru.bytes <= lru.cap


def test_a_combine_layers_layer_s_stack_follows_its_sides_filters():
    """A Combine layers layer with a filter on its stars side and a stack of its own: a release of the side's filter,
    then of the top filter, gives the pixels of a composition from nothing (the stack's steps depend on the side's)."""
    g = new_workflow_layer(1, "G", True, part="combined")
    side = new_filter_layer("saturation", g["id"], "stars")
    side["params"]["strength"] = 30.0
    tops = []
    for key, params in (("levels", {"gamma": 1.3}), ("local_contrast", {"strength": 0.4}), ("saturation", {"strength": 20.0})):
        f = new_filter_layer(key, g["id"])
        f["params"].update(params)
        tops.append(f)
    layers = pe.normalize(list(reversed(tops)) + [side, g])
    sid = side["id"]

    def at(f):
        out = pe.snapshot(layers)
        f({x["id"]: x for x in out})
        return out
    states = [layers, at(lambda d: d[sid]["params"].update(strength=60.0)),
              at(lambda d: d[sid]["params"].update(strength=60.0) or d[tops[-1]["id"]]["params"].update(strength=5.0)),
              at(lambda d: d[sid]["params"].update(strength=-20.0) or d[tops[-1]["id"]]["params"].update(strength=5.0)),
              layers]
    pr = Toy()
    cp = Composer(pr, cache_bytes=200_000_000)
    for s in states:
        for lv, got in zip((2, 1, 0), release(cp, s)):
            ref = Composer(pr).compose(pe.snapshot(s), lv, thumbs=lv == 1, cache=False)
            assert np.array_equal(got.rgb, ref.rgb), lv


def side_stack():
    """A Combine layers layer whose starless image has a stack of its own filters (the owner's NGC 7000), top first."""
    g = new_workflow_layer(1, "G", True, part="combined")
    side = []
    for key, params in (("levels", {"gamma": 1.3}), ("local_contrast", {"strength": 0.4}), ("balance", {"tint": 0.4}),
                        ("saturation", {"strength": 20.0})):
        f = new_filter_layer(key, g["id"], "starless")
        f["params"].update(params)
        side.append(f)
    m = new_mask()
    p = new_part("luminance")
    p.update(lo=10.0, hi=60.0)
    m["parts"] = [p]
    side[2]["mask"] = m
    return pe.normalize(list(reversed(side)) + [g])


def test_a_release_of_the_top_filter_of_a_side_makes_that_filter_only(monkeypatch):
    applied = []
    real = Composer._apply

    def counted(self, fl, under, level, *a, **k):
        applied.append((fl["id"], level))
        return real(self, fl, under, level, *a, **k)
    monkeypatch.setattr(Composer, "_apply", counted)
    layers = side_stack()
    top = next(x for x in layers if x["kind"] == "filter")["id"]
    pr = Toy()
    cp = Composer(pr, cache_bytes=200_000_000)
    release(cp, layers)
    for k, v in enumerate((25.0, 31.0, 44.0)):
        applied.clear()
        state = with_param(layers, 0, "strength", v)
        got = release(cp, state)
        if k:
            assert applied == [(top, 2), (top, 1), (top, 0)], applied
        for lv, g in zip((2, 1, 0), got):
            ref = Composer(pr).compose(pe.snapshot(state), lv, thumbs=lv == 1, cache=False)
            assert np.array_equal(g.rgb, ref.rgb)
            assert all(np.array_equal(g.mask_thumbs[k2], ref.mask_thumbs[k2]) for k2 in ref.mask_thumbs)


def test_a_drag_s_drafts_make_the_filter_moved_only_and_show_what_a_draft_from_nothing_shows():
    """While a slider moves, each draft starts from the steps kept under the filter moved (drafts kept apart from the
    exact images: a filter may take a quick look in a draft); a draft is the one a composer with only the layer's image
    gives."""
    layers = stack()
    pr = ToyProvider()
    cp = Composer(pr, cache_bytes=200_000_000)
    release(cp, layers)
    top = next(x for x in layers if x["kind"] == "filter")["id"]
    applied = []
    real = cp._apply
    cp._apply = lambda fl, under, level, *a, **k: (applied.append((fl["id"], level)), real(fl, under, level, *a, **k))[1]
    for k, v in enumerate((22.0, 26.0, 30.0, 34.0)):
        applied.clear()
        state = with_param(layers, 0, "strength", v)
        got = cp.compose(pe.snapshot(state), 3, draft=True, track=True)
        if k:
            assert applied == [(top, 3)], applied
        ref_cp = Composer(pr)
        ref_cp.compose(pe.snapshot(state), 3)                         # (the layer's exact image, as the warm one has)
        ref = ref_cp.compose(pe.snapshot(state), 3, draft=True, cache=False)
        assert np.array_equal(got.rgb, ref.rgb)
