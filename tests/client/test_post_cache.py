"""Post's composition reuses what a change leaves as it was, bit for bit: the composite under the row being edited (its
backdrop), the rows hidden under an opaque one, the images kept by content (scan-resistant), the mask thumbnails."""
from __future__ import annotations

import numpy as np
import pytest

from test_post import ToyProvider  # noqa: F401 - the toy layers

from astro_gpu.client import postengine as pe
from astro_gpu.client.postengine import ByteLRU, Composer, new_filter_layer, new_mask, new_part, new_workflow_layer


class Toy(ToyProvider):
    """ToyProvider with a Stars block for workflow 1 and 2 (a group layer reads ("stars", pid)) and, optionally, a
    layer fitted inside the canvas (``fitted``: its pid covers only the left half)."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.fitted = set()

    def available(self, pid):
        return (pid[1] if isinstance(pid, tuple) else pid) in self.img

    def cover(self, lay, level, box):
        if lay.get("pid") not in self.fitted:
            return None
        x0, y0, x1, y1 = box
        m = np.zeros((y1 - y0, x1 - x0), np.float32)
        m[:, : max(0, (self.W >> level) // 2 - x0)] = 1
        return m


def heavy():
    """Bottom first: A with filters, B Screen 60 % with a luminance mask, a group, a filter on everything below,
    C Multiply, D Normal 100 % on top (opaque). Returned top first, with the rows by name."""
    a = new_workflow_layer(1, "A", True)
    fa = new_filter_layer("levels", a["id"])
    fa["params"].update(gamma=1.4)
    fs = new_filter_layer("saturation", a["id"])
    fs["params"].update(strength=40.0)
    b = new_workflow_layer(2, "B", True)
    b.update(mode="Screen", opacity=60.0, mask=new_mask())
    lp = new_part("luminance")
    lp.update(lo=30.0, hi=90.0)
    b["mask"]["parts"] = [lp]
    fb = new_filter_layer("sharpen", b["id"])
    fb["params"].update(strength=0.8)
    g = new_workflow_layer(1, "G", True, part="combined")
    g.update(mode="Lighten", opacity=50.0)
    fg = new_filter_layer("saturation", g["id"], "stars")
    fg["params"].update(strength=30.0)
    glob = new_filter_layer("local_contrast", None)
    glob["params"].update(strength=0.4)
    glob["mask"] = new_mask()
    gp = new_part("gradient")
    glob["mask"]["parts"] = [gp]
    c = new_workflow_layer(2, "C", True)
    c.update(mode="Multiply", opacity=70.0)
    fc = new_filter_layer("levels", c["id"])
    fc["params"].update(black=0.05)
    d = new_workflow_layer(1, "D", True)
    fd = new_filter_layer("saturation", d["id"])
    fd["params"].update(strength=10.0)
    layers = pe.normalize([fd, d, fc, c, glob, fg, g, fb, b, fs, fa, a])
    return layers, {"A": a, "B": b, "G": g, "glob": glob, "C": c, "D": d, "fa": fa, "fs": fs, "fb": fb, "fc": fc, "fd": fd}


def fresh(pr, layers, level=0, **kw):
    """The composition by a new composer that keeps nothing: the reference."""
    return Composer(pr).compose(pe.snapshot(layers), level, cache=False, **kw)


def same(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b)


@pytest.mark.parametrize("level", [0, 1])
def test_a_change_starts_from_the_backdrop_of_the_row_edited_bit_for_bit(level):
    pr = Toy()
    layers, L = heavy()
    L["D"]["visible"] = False                                       # no opaque top: every row shows
    cp = Composer(pr)
    run = lambda: cp.compose(pe.snapshot(layers), level, track=True)    # noqa: E731
    assert same(run().rgb, fresh(pr, layers, level).rgb) and cp.skipped == 0
    n = len(cp._edits)
    for k, v in enumerate((20.0, 30.0, 35.0)):                      # C's filter, three times
        L["fc"]["params"]["black"] = v / 1000
        out = run()
        assert same(out.rgb, fresh(pr, layers, level).rgb)
        if k:                                                       # from the second change: C alone is computed
            assert cp.skipped == n - 2 and out.drawn == 0           # (D hidden is the last row)
    # a change below invalidates the backdrop: everything from there up again, the same image as from scratch
    L["fa"]["params"]["gamma"] = 1.2
    out = run()
    assert cp.skipped == 0 and same(out.rgb, fresh(pr, layers, level).rgb)
    # its mask, its mode, its opacity: each a change of that row (the composite under it is reused)
    for change in (lambda: L["B"]["mask"]["parts"][0].update(lo=40.0), lambda: L["B"].update(mode="Lighten"),
                   lambda: L["B"].update(opacity=45.0), lambda: L["B"]["sliders"].update(sky=2.6),
                   lambda: L["glob"].update(opacity=50.0), lambda: L["G"].update(blend="Lighten")):
        change()
        out = run()
        assert same(out.rgb, fresh(pr, layers, level).rgb)
        out = run()                                                 # nothing changed: the last composite, the same
        assert same(out.rgb, fresh(pr, layers, level).rgb) and cp.skipped == n
    assert cp.plan_errors == 0


def test_a_hidden_row_and_a_row_shown_again_keep_the_image_exact():
    pr = Toy()
    layers, L = heavy()
    cp = Composer(pr)
    for flip in ("B", "glob", "C", "B", "glob", "C"):
        L[flip]["visible"] = not L[flip]["visible"]
        L["D"]["visible"] = flip != "C"
        for lvl in (2, 1, 0):
            assert same(cp.compose(pe.snapshot(layers), lvl, track=True).rgb, fresh(pr, layers, lvl).rgb)
    assert cp.plan_errors == 0


def test_rows_under_an_opaque_layer_are_not_computed_and_the_image_is_its_own():
    pr = Toy()
    layers, L = heavy()
    cp = Composer(pr)
    out = cp.compose(pe.snapshot(layers), 0)
    alone = Composer(pr).compose(pe.snapshot([L["fd"], dict(L["D"])]), 0, cache=False)
    assert cp.skipped == 5                                          # A, B, G, the filter, C
    assert out.drawn == 1 and same(out.rgb, alone.rgb)
    # skipped only when truly opaque: 99 %, a mask, another mode, an image that does not cover the canvas
    for change, undo in ((lambda: L["D"].update(opacity=99.0), lambda: L["D"].update(opacity=100.0)),
                         (lambda: L["D"].update(mask=new_mask()), lambda: L["D"].update(mask=None)),
                         (lambda: L["D"].update(mode="Screen"), lambda: L["D"].update(mode="Normal")),
                         (lambda: pr.fitted.add(1), lambda: pr.fitted.clear())):
        change()
        out = cp.compose(pe.snapshot(layers), 0)
        assert cp.skipped == 0
        undo()
    # a mask asked for under it (the mask being edited) or a histogram: computed, and the image is the same
    out = cp.compose(pe.snapshot(layers), 0, want=(L["B"]["id"],), hists=(L["fa"]["id"],))
    assert L["B"]["id"] in out.masks and L["fa"]["id"] in out.hists and same(out.rgb, alone.rgb)
    # the export path (nothing kept) skips them too: the same pixels
    assert same(fresh(pr, layers).rgb, alone.rgb)


def test_mask_thumbnails_of_rows_not_computed_are_those_they_had():
    pr = Toy()
    layers, L = heavy()
    L["D"]["opacity"] = 99.0
    cp = Composer(pr)
    first = cp.compose(pe.snapshot(layers), 1, thumbs=True, track=True)
    assert {L["B"]["id"], L["glob"]["id"]} <= set(first.mask_thumbs)
    L["D"]["opacity"] = 100.0                                      # now opaque: the rows under it are not computed
    out = cp.compose(pe.snapshot(layers), 1, thumbs=True, track=True)
    assert cp.skipped > 0
    ref = Composer(pr).compose(pe.snapshot(layers), 1, thumbs=True)    # nothing kept: every mask computed
    assert out.mask_thumbs.keys() == ref.mask_thumbs.keys()
    assert all(np.array_equal(out.mask_thumbs[k], ref.mask_thumbs[k]) for k in ref.mask_thumbs)
    # a mask changed under the opaque layer: its thumbnail is not kept, so its row is computed again
    L["B"]["mask"]["parts"][0]["lo"] = 50.0
    out = cp.compose(pe.snapshot(layers), 1, thumbs=True, track=True)
    ref = Composer(pr).compose(pe.snapshot(layers), 1, thumbs=True)
    assert np.array_equal(out.mask_thumbs[L["B"]["id"]], ref.mask_thumbs[L["B"]["id"]])
    assert same(out.rgb, ref.rgb)


def test_the_kept_images_resist_a_scan_and_stay_in_their_budget():
    """A composition reads its images in the same order: when they do not all fit, the ones kept stay (a plain LRU
    drops each just before it is read again, and finds none)."""
    a = [np.full((10, 10, 3), k, np.float32) for k in range(5)]
    lru = ByteLRU(a[0].nbytes * 3)
    for _ in range(3):
        lru.begin()
        hits = 0
        for k, x in enumerate(a):
            if lru.get(k) is not None:
                hits += 1
            else:
                lru.put(k, x)
        assert lru.bytes <= lru.cap
    assert hits == 3
    lru.begin()                                                     # a new composition reads others: the old go first
    for k in range(5, 8):
        assert lru.put(k, a[0])
    assert set(lru.items) == {5, 6, 7}


def test_the_caches_share_the_budget():
    c = Composer(None, cache_bytes=1_000_000_000)
    assert c.images.cap + c.steps.cap + c.backdrops.cap + c.parts.cap <= 1_000_000_000
