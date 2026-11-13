"""The hot pixels of the lights (stages/calibrate.py): the persistence over the session, Siril's cosmetic correction per
frame, and the replacement of every excluded pixel by the mean of its same-colour neighbours."""
import numpy as np
import pytest

from astro_gpu.stages import calibrate as cal

H, W = 192, 256


def _star(yy, xx, y, x, amp=3000.0, sig=1.5):
    return amp * np.exp(-0.5 * (((xx - x) / sig) ** 2 + ((yy - y) / sig) ** 2))


def _sequence(rng, n=24, step=3.0):
    """A Bayer sequence (RGGB) of a sky with stars drifting by ``step`` px a frame, and fixed defects: isolated hot
    pixels (some saturated, those lighting their four nearest neighbours by crosstalk), and pairs hot together."""
    yy, xx = np.mgrid[0:H, 0:W]
    stars0 = np.stack([rng.uniform(30, H - 30, 25), rng.uniform(20, W - 100, 25)], 1)
    hot = np.zeros((H, W), np.float32)
    isolated = [(40, 41), (77, 120), (101, 200), (150, 67), (20, 230), (131, 15)]
    for i, (y, x) in enumerate(isolated):
        hot[y, x] = 65535 if i % 2 == 0 else 30000                      # half of them saturated
        if i % 2 == 0:
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                hot[y + dy, x + dx] += 2500                              # crosstalk of a saturated pixel
    pairs = [((60, 80), (60, 82)), ((120, 160), (122, 160))]            # two pixels of a colour, side by side
    for a, b in pairs:
        hot[a] = hot[b] = 25000
    frames = []
    for k in range(n):
        sky = 100 + sum(_star(yy, xx, y, x + k * step) for y, x in stars0)
        frames.append(np.minimum(sky + hot + rng.normal(0, 5, (H, W)), 65535).astype(np.float32))
    defects = isolated + [p for pair in pairs for p in pair]
    return frames, defects, stars0, step


def _read_from(monkeypatch, frames):
    table = {f"f{i}": f for i, f in enumerate(frames)}
    monkeypatch.setattr(cal, "read_cfa", lambda p: table[p].copy())
    return list(table)


def test_the_persistence_flags_the_defects_saturated_or_in_pairs_and_no_star(monkeypatch):
    from scipy.ndimage import median_filter
    rng = np.random.default_rng(1)
    frames, defects, stars0, step = _sequence(rng)
    paths = _read_from(monkeypatch, frames)
    dark = np.full((H, W), 100, np.float32)
    flat = np.ones((H, W), np.float32)
    bad = np.zeros((H, W), bool)
    msg = cal._hot_from_lights(paths, dark, flat, bad, median_filter, cfa="RGGB")
    assert "persistence" in msg and str(int(bad.sum())) in msg
    assert all(bad[y, x] for y, x in defects)                            # a saturated one too, a pair too
    n = len(frames)
    for y, x in stars0:                                                  # no core of a star, at any of its places
        for k in range(0, n, 4):
            assert not bad[int(round(y)), int(round(x + k * step))]


def test_siril_correction_flags_the_isolated_hot_pixels_and_replaces_them_by_the_mean_of_the_neighbours():
    rng = np.random.default_rng(2)
    yy, xx = np.mgrid[0:H, 0:W]
    x = (100 + _star(yy, xx, 90, 120, 20000, 2.0) + rng.normal(0, 5, (H, W))).astype(np.float32)
    isolated = [(40, 41), (77, 120), (101, 201), (150, 67)]
    for y, c in isolated:
        x[y, c] += 30000
    x[20, 230] = 65535.0                                                 # saturated
    a, b = (60, 80), (60, 82)                                            # a pair of the same colour, side by side
    x[a] += 25000
    x[b] += 25000
    x[1, 1] += 30000                                                     # at the corner: fewer neighbours
    before = x.copy()
    ys, xs, vals = cal.siril_hot_pixels(x, 2)
    flagged = set(zip(ys.tolist(), xs.tolist()))
    for p in [*isolated, (20, 230), (1, 1)]:
        assert p in flagged
    assert a not in flagged and b not in flagged                         # the pair lights each other's mean
    assert not any(abs(y - 90) <= 4 and abs(c - 120) <= 4 for y, c in flagged)   # the star's core untouched
    assert len(flagged) <= 6 + 3                                         # and little else
    assert np.array_equal(x, before)                                     # corrected by the caller, on a copy
    i = list(zip(ys.tolist(), xs.tolist())).index((40, 41))
    nb = [before[40 + dy, 41 + dx] for dy in (-2, 0, 2) for dx in (-2, 0, 2) if dy or dx]
    assert vals[i] == pytest.approx(np.mean(nb), rel=1e-5)
    i = list(zip(ys.tolist(), xs.tolist())).index((1, 1))
    nb = [before[y, c] for y, c in ((1, 3), (3, 1), (3, 3))]             # the neighbours inside the frame only
    assert vals[i] == pytest.approx(np.mean(nb), rel=1e-5)


def _worker(monkeypatch, frame, bad, hot, dark=100.0):
    """The calibration of one light by ``_calib_one``, its masters set as the pool's initialiser does."""
    mm = np.zeros((1, H, W), np.float32)
    cal._G.clear()
    cal._G.update(darks=[np.full((H, W), dark, np.float32)], flat=np.ones((H, W), np.float32), bad=bad, mm=mm,
                  sat=65000.0, scale=1.0, fill=cal._neighbours(bad, "RGGB"), cfa="RGGB", hot=hot)
    monkeypatch.setattr(cal, "read_cfa", lambda p: frame.copy())
    idx, stats, n_hot = cal._calib_one((0, "x", 0))
    cal._G.clear()
    return mm[0], n_hot


def test_a_light_is_calibrated_with_its_hot_pixels_corrected_per_frame_and_a_star_core_left_saturated(monkeypatch):
    rng = np.random.default_rng(3)
    yy, xx = np.mgrid[0:H, 0:W]
    frame = (100 + _star(yy, xx, 90, 120, 90000, 2.0) + rng.normal(0, 5, (H, W))).astype(np.float32)
    frame = np.minimum(frame, 65535)
    frame[40, 41] = 65535                                                # a saturated hot pixel
    frame[77, 120] += 30000                                              # a hot one
    bad = np.zeros((H, W), bool)
    bad[130, 30] = True
    frame[130, 30] = 9000                                                # an excluded pixel
    for hot in ("per_frame", "persistence"):
        out, n_hot = _worker(monkeypatch, frame, bad, hot)
        assert out[130, 30] == pytest.approx(np.mean([frame[130 + dy, 30 + dx] - 100 for dy in (-2, 0, 2)
                                                      for dx in (-2, 0, 2) if dy or dx]), rel=1e-4)
        assert out[90, 120] == cal.SAT_MARK                              # a star's core is saturated, as it was
        if hot == "per_frame":
            assert n_hot >= 2 and out[40, 41] < 1000 and out[77, 120] < 1000     # corrected, not marked saturated
        else:
            assert n_hot == 0 and out[40, 41] == cal.SAT_MARK                    # the persistence marks it in bad


def test_the_hot_method_asked_for_and_the_one_that_runs():
    class Q:
        hot_from_lights = True
        hot_method = "persistence"

    q = Q()
    assert cal.hot_method(q, "RGGB", 40) == ("persistence", "")
    assert cal.hot_method(q, "RGGB", 10) == ("off", "")                  # too few frames for a persistence
    q.hot_method = "per_frame"
    assert cal.hot_method(q, "RGGB", 3) == ("per_frame", "")             # any session
    xt = "GRGGBG" * 6
    assert cal.hot_method(q, xt, 40)[0] == "persistence" and cal.hot_method(q, xt, 40)[1]   # said
    assert cal.hot_method(q, xt, 10)[0] == "off"
    q.hot_from_lights = False
    assert cal.hot_method(q, "RGGB", 40) == ("off", "")


def test_the_hot_pixel_method_is_a_choice_with_french_words():
    from astro_gpu import texts
    assert cal.CalibrateParams().hot_method == "persistence"
    assert list(cal.HOT_METHODS) == ["persistence", "per_frame"]
    assert texts.choice_label("persistence", "fr") == "Persistance"
    assert texts.choice_label("per_frame", "fr") == "Pose par pose"
    assert texts.param_label("calibrate", "hot_method", "fr") == "Pixels chauds"


def test_the_persistence_threshold_is_in_sigmas_of_one_frame_not_of_the_quartile(monkeypatch):
    """A fixed pattern of 2.5 sigmas of one frame's noise (the dark current's spread) is no defect: averaged over the
    frames the quartile's own noise is a third of a frame's, and 5 of its sigmas flagged every such pixel. A pixel 12
    sigmas of a frame above its neighbours is."""
    from scipy.ndimage import median_filter
    rng = np.random.default_rng(5)
    sigma = 10.0
    pattern = np.where(rng.random((H, W)) < 0.05, 2.5 * sigma, 0.0).astype(np.float32)
    hot = np.zeros((H, W), np.float32)
    hot[60, 80] = hot[100, 150] = 12 * sigma
    frames = [(1000 + pattern + hot + rng.normal(0, sigma, (H, W))).astype(np.float32) for _ in range(30)]
    paths = _read_from(monkeypatch, frames)
    bad = np.zeros((H, W), bool)
    cal._hot_from_lights(paths, np.zeros((H, W), np.float32), np.ones((H, W), np.float32), bad, median_filter)
    assert bad[60, 80] and bad[100, 150]
    assert bad.sum() < 20                                              # not the 5 % of the pattern


def test_the_persistence_leaves_a_star_that_stayed_on_its_pixels_and_still_flags_the_defects(monkeypatch):
    """Frames that hardly moved: the lower quartile holds the star, so its core is no defect; the isolated defect is."""
    from scipy.ndimage import median_filter
    rng = np.random.default_rng(7)
    yy, xx = np.mgrid[0:H, 0:W]
    sky = 100 + _star(yy, xx, 96, 128, 6000, 2.0) + _star(yy, xx, 40, 40, 3000, 1.6)
    sky[150, 200] += 2000                                              # a defect, away from the stars
    frames = [(sky + rng.normal(0, 5, (H, W))).astype(np.float32) for _ in range(24)]
    paths = _read_from(monkeypatch, frames)
    bad = np.zeros((H, W), bool)
    msg = cal._hot_from_lights(paths, np.full((H, W), 100, np.float32), np.ones((H, W), np.float32), bad, median_filter)
    assert bad[150, 200]
    assert not bad[94:99, 126:131].any() and not bad[38:43, 38:43].any()
    assert "left to the stack's rejection" in msg
