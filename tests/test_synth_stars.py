"""The Synthetic stars block: the instrument's shape, the detection, the two methods, and its row in Post."""
import numpy as np

from astro_gpu.client import postengine as pe
from astro_gpu.core import workflow as wfl
from astro_gpu.stages import synth_stars_lib as lib


def small_instrument(**kw):
    return lib.Instrument(**{**dict(diameter_mm=150, focal_mm=750, pixel_um=3.72, seeing_px=2.5), **kw})


def test_a_spider_makes_spikes_and_a_refractor_makes_none():
    R = 90
    newton = lib.theoretical_shape(small_instrument(obstruction=0.33, spider_arms=4, spider_mm=1.5), R)
    lens = lib.theoretical_shape(small_instrument(), R)
    assert np.allclose(newton.sum((1, 2)), 1, atol=1e-4) and newton.shape == (3, 2 * R + 1, 2 * R + 1)
    g = newton[1]
    assert g[R, R + 50] > 5 * g[R + 35, R + 35]                        # along the arms, against between them
    assert lens[1][R, R + 50] < 1.5 * lens[1][R + 35, R + 35]
    turned = lib.theoretical_shape(small_instrument(obstruction=0.33, spider_arms=4, spider_mm=1.5, angle_deg=45), R)
    assert turned[1][R + 35, R + 35] > 5 * turned[1][R, R + 50]


def stars_alone(ring=False, n=49, size=640):
    """A stars-alone image (3, H, W) with stars on a grid; ``ring``: defocused, a hollow disc."""
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[-16:17, -16:17]
    rr = np.hypot(yy, xx)
    stamp = (np.exp(-((rr - 6) ** 2) / 2) if ring else np.exp(-(rr ** 2) / (2 * 1.3 ** 2))).astype(np.float32)
    stamp /= stamp.sum()
    s = rng.normal(0, 0.05, (3, size, size)).astype(np.float32)
    pts = [(y, x) for y in np.linspace(50, size - 51, int(n ** 0.5)).astype(int)
           for x in np.linspace(50, size - 51, int(n ** 0.5)).astype(int)]
    fl = rng.uniform(800, 9000, len(pts))
    fl[0] = 30000.0                                                    # the brightest: the others are not saturated
    for (y, x), f in zip(pts, fl):
        s[:, y - 16:y + 17, x - 16:x + 17] += f * stamp * np.array([1.2, 1.0, 0.8], np.float32)[:, None, None]
    return s, np.ones((size, size), bool), np.array(pts), fl


def test_a_defocused_ring_is_one_star_and_is_drawn_narrow_with_its_light():
    s, valid, pts, fl = stars_alone(ring=True)
    L = s.mean(0)
    ys, xs, _p, _T, size = lib.detect(L, valid)
    assert size > 3 and len(ys) == len(pts)
    drawn, info = lib.from_the_instrument(s, valid, small_instrument(), None)
    assert info["stars_drawn"] == len(pts)
    y, x = pts[24]
    a = drawn[1, y - 16:y + 17, x - 16:x + 17]
    assert a.argmax() in range(16 * 33 + 13, 16 * 33 + 20)              # the light at the centre: not a ring any more
    assert abs(float(a.sum()) / fl[24] - 1) < 0.15


def test_the_image_shape_keeps_each_star_in_its_place_and_colour():
    s, valid, pts, fl = stars_alone()
    drawn, info = lib.from_the_image(s, valid)
    assert info["stars_drawn"] + info["stars_kept"] == len(pts)
    y, x = pts[6]
    for c, k in ((0, 1.2), (2, 0.8)):
        assert abs(float(drawn[c, y - 12:y + 13, x - 12:x + 13].sum()) / float(drawn[1, y - 12:y + 13, x - 12:x + 13].sum()) - k) < 0.05


def _centroid(a):
    n = a.shape[0] // 2
    yy, xx = np.mgrid[-n:n + 1, -n:n + 1]
    q = np.clip(a, 0, None)
    return float((q * yy).sum() / q.sum()), float((q * xx).sum() / q.sum())


def test_a_star_off_its_pixel_is_measured_centred_and_drawn_where_it_was():
    # review 13.4: moved(p, dy, dx) shifts by -(dy, dx); the measure shifted a star 0.4 px off by +0.4 (0.8 off), the
    # drawing put it 0.4 px the other side
    off = (0.4, -0.3)
    yy, xx = np.mgrid[-32:33, -32:33]
    L = np.zeros((640, 640), np.float32)
    pts = [(y, x) for y in range(200, 441, 60) for x in range(200, 441, 60)]
    for y, x in pts:
        L[y - 32:y + 33, x - 32:x + 33] += np.exp(-((yy - off[0]) ** 2 + (xx - off[1]) ** 2) / (2 * 1.4 ** 2))
    L[40, 40] = 5.0                                         # a brighter star far off: these ones are not saturated
    ys, xs = np.array(pts).T
    shape = lib.measured_shape(L, ys, xs, L[ys, xs])
    cy, cx = _centroid(shape)
    assert abs(cy) < 0.05 and abs(cx) < 0.05
    centred = np.exp(-(yy ** 2 + xx ** 2) / (2 * 1.4 ** 2)).astype(np.float32)
    centred /= centred.sum()
    drawn = lib.draw(200, 200, np.array([100]), np.array([100]), [off], np.full((3, 1), 1000.0, np.float32),
                     centred, 1e-3)
    cy, cx = _centroid(drawn[1, 100 - 20:100 + 21, 100 - 20:100 + 21])
    assert abs(cy - off[0]) < 0.05 and abs(cx - off[1]) < 0.05


def test_the_saturated_stars_take_their_light_from_the_catalogue():
    fluxes = np.array([[120.0, 100.0, 80.0]] * 12 + [[50.0, 50.0, 50.0]]).T
    ys = np.arange(13) * 40.0
    xs = np.arange(13) * 40.0
    gaia = {"x": xs + 1, "y": ys, "g": np.array([15.0] * 12 + [10.0]), "bp": np.full(13, 11.0), "rp": np.full(13, 10.0)}
    sat = np.array([False] * 12 + [True])
    out, n = lib.saturated_fluxes(ys, xs, sat, fluxes, gaia)
    assert n == 1 and abs(out[1, 12] / 10000 - 1) < 0.01 and abs(out[0, 12] / out[1, 12] - 1.2) < 0.01
    no_colour = {**gaia, "bp": np.full(13, np.nan)}                     # the offline extract
    out, n = lib.saturated_fluxes(ys, xs, sat, fluxes, no_colour)
    assert n == 1 and abs(out[1, 12] / 10000 - 1) < 0.01 and abs(out[0, 12] / out[1, 12] - 1.2) < 0.01


def test_the_block_comes_after_the_stars_and_says_its_method():
    full = {"denoise": "pretrained", "stars": "neural"}
    blocks = wfl.from_options(full)
    assert wfl.palette_problem(blocks, "frames", "synth") is None
    assert wfl.palette_problem(wfl.from_options({"denoise": "pretrained"}), "frames", "synth") is not None
    with_it = blocks + [wfl.node("synth", "instrument")]
    assert "synth_stars" in wfl.sequence_chain(with_it, "frames")
    assert wfl.node_implied(with_it, "synth_stars") == {"method": "instrument"}
    assert "synth_stars" not in wfl.sequence_chain(blocks, "frames")


def test_the_layer_of_post_opens_open_with_both_methods():
    """Issue #93: one start for the image shape and the instrument, the mask open (its brush subtracts); the Stars group
    of Post: tests/client/test_post_stars_group.py."""
    group = pe.new_workflow_layer(1, "Full", part="combined")
    layers = pe.with_synth_rows([group], {1: "measured"})
    row = next(x for x in layers if x.get("filter") == "synth")
    assert row["target"] == group["id"] and row["sub"] == "stars" and row["mask"]["parts"][0]["strokes"] == []
    assert row["mask"]["parts"][0]["op"] == "sub"
    assert pe.with_synth_rows(layers, {1: "measured"}) == layers          # once
    open_ = next(x for x in pe.with_synth_rows([pe.new_workflow_layer(2, "F", part="combined")], {2: "instrument"})
                 if x.get("filter") == "synth")
    assert [p["op"] for p in open_["mask"]["parts"]] == ["sub"] and open_["mask"]["parts"][0]["strokes"] == []
    assert [x for x in pe.with_synth_rows(layers, {}) if x.get("filter") == "synth"] == []
    assert pe.source_id(dict(group, part="synth")) == ("synth", 1) and pe.part_sliders({"star_size": 1, "noise": 2}, "synth") == {"star_size": 1}


def field_of_stars(n=240, size=640, seed=3, sigma=1.4):
    """A stars-alone luminance with ``n`` round stars of every brightness, on a grid kept clear of the centre."""
    rng = np.random.default_rng(seed)
    L = rng.normal(0, 0.5, (size, size)).astype(np.float32)
    yy, xx = np.mgrid[-12:13, -12:13]
    stamp = np.exp(-(yy ** 2 + xx ** 2) / (2 * sigma ** 2)).astype(np.float32)
    pts = []
    for y in range(40, size - 40, 40):
        for x in range(40, size - 40, 40):
            if abs(y - size // 2) < 80 and abs(x - size // 2) < 80:
                continue
            pts.append((y, x))
    pts = pts[:n]
    for (y, x), a in zip(pts, rng.uniform(150, 600, len(pts))):
        L[y - 12:y + 13, x - 12:x + 13] += a * stamp
    return L, np.array(pts), stamp


def test_an_extended_blob_is_not_taken_for_a_star():
    L, pts, _ = field_of_stars()
    h = L.shape[0] // 2
    yy, xx = np.mgrid[:L.shape[0], :L.shape[1]]
    L += 300 * np.exp(-((yy - h) ** 2 + (xx - h) ** 2) / (2 * 7.0 ** 2)).astype(np.float32)   # the nebula's texture
    ys, xs, _p, _T, size = lib.detect(L, np.ones(L.shape, bool))
    keep, why = lib.vet(L, None, ys, xs, size)
    on_blob = np.hypot(ys - h, xs - h) < 15
    assert on_blob.any() and not keep[on_blob].any()                   # nothing drawn on the blob
    assert keep[~on_blob].mean() > 0.9                                 # the field's stars stay
    assert why["stars_not_stellar"] >= on_blob.sum()


def test_a_bump_on_the_structure_of_the_starless_image_is_texture_and_a_bright_star_is_not():
    L, pts, stamp = field_of_stars()
    rng = np.random.default_rng(5)
    import cv2
    tex = cv2.GaussianBlur(rng.normal(0, 1, L.shape).astype(np.float32), (0, 0), 2.0) * 800   # a textured nebula, in the middle only
    gy, gx = np.mgrid[:L.shape[0], :L.shape[1]]
    tex *= np.exp(-((gy - 321) ** 2 + (gx - 345) ** 2) / (2 * 50.0 ** 2)).astype(np.float32)
    y, x = 321, 330
    L[y - 12:y + 13, x - 12:x + 13] += 300 * stamp                    # a star-like bump, small against the texture
    y2, x2 = 321, 360
    L[y2 - 12:y2 + 13, x2 - 12:x2 + 13] += 20000 * stamp              # a real, bright star
    ys, xs, _p, _T, size = lib.detect(L, np.ones(L.shape, bool))
    keep, _why = lib.vet(L, tex, ys, xs, size)
    at = lambda yy_, xx_: np.nonzero((abs(ys - yy_) < 3) & (abs(xs - xx_) < 3))[0]
    assert len(at(y, x)) == 1 and not keep[at(y, x)[0]]
    assert len(at(y2, x2)) == 1 and keep[at(y2, x2)[0]]
    plain, _ = lib.vet(L, None, ys, xs, size)                         # without the starless image the bump passes
    assert plain[at(y, x)[0]]


def test_two_touching_stars_are_two_stars_or_none_never_one_elongated():
    L, pts, stamp = field_of_stars()
    h = L.shape[0] // 2
    for dx in (-2, 2):                                                # 4 px apart: one maximum, an elongated blob
        L[h - 12:h + 13, h + dx - 12:h + dx + 13] += 150 * stamp
    ys, xs, _p, _T, size = lib.detect(L, np.ones(L.shape, bool))
    keep, _why = lib.vet(L, None, ys, xs, size)
    near = np.hypot(ys - h, xs - h) < 8
    assert near.any()
    assert keep[near].sum() in (0, 2)                                 # two drawn stars or none, not one elongated one


def test_the_block_is_at_a_new_version_after_vetting():
    from astro_gpu.stages.synth_stars import SynthStars
    assert int(SynthStars.version) >= 3
