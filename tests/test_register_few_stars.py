"""Frames of a few dozen trailed stars (an untracked telescope: short exposures, a steady drift): every frame of the
field is registered, on the count of its stars brought onto the reference's and by a linear map; a frame of another
field stays out; the log says what was tried, with the value and the limit."""
from __future__ import annotations

import json

import numpy as np
import pytest
from scipy.spatial import cKDTree
from test_register_sparse import _aperture, _run, _sky

from astro_gpu.stages import register as rg
from astro_gpu.stages.register import apply_poly, fit_frame, fit_sparse

W, H = 3054, 2042
#: the scatter (px, per axis) of the centre of a trailed star: pairs of such stars lie 1.1 px apart (root mean square)
TRAIL = 0.55


def _drifting(n_frames=15, n_stars=30, drift=57.0, noise=TRAIL, seed=3):
    """The catalogues of ``n_frames`` frames of one sky that drifts by ``drift`` px a frame, no rotation: about
    ``n_stars`` stars in each, their centres scattered by ``noise``."""
    rng = np.random.default_rng(seed)
    span = drift * n_frames
    n = int(round(n_stars * (W + span) / W))
    sky = np.c_[rng.uniform([-span / 2, 0], [W + span / 2, H], (n, 2)),
                np.sort(rng.pareto(1.5, n) + 1)[::-1] * 1000, np.full(n, 5.0)]
    cats = []
    for i in range(n_frames):
        xy = sky[:, :2] + [drift * (i - n_frames // 2), 0.0] + rng.normal(0, noise, (n, 2))
        inside = (xy[:, 0] > 0) & (xy[:, 0] < W) & (xy[:, 1] > 0) & (xy[:, 1] < H)
        cats.append(np.c_[xy, sky[:, 2:]][inside])
    return cats


def test_thirty_trailed_stars_with_a_drift_are_registered(tmp_path, monkeypatch):
    cats = _drifting()
    assert all(24 <= len(c) <= 36 for c in cats)
    z, logs, metrics = _run(tmp_path, [_aperture(c) for c in cats], monkeypatch)
    ok = np.isfinite(z["rms"])
    assert ok.all() and metrics["aligned"] == 15
    assert z["rms"][ok].max() < rg.STAR_TOL
    few = z["nmatch"] < rg.CUBIC_MATCHES
    assert few.sum() >= 5 and not z["coef"][few][:, :, 3:].any()  # under 30 star pairs: a linear map, no cubic
    ref = int(z["ref"])
    for i in (0, 14):                                             # the map is the drift: 57 px a frame
        x, _ = apply_poly(z["coef"][i], np.array([W / 2]), np.array([H / 2]), W, H)
        assert x[0] - W / 2 == pytest.approx(-57.0 * (i - ref), abs=1.0)
    assert not any("left out" in m or "could not be aligned" in m for _, m in logs)


def test_a_frame_of_another_field_among_them_is_rejected(tmp_path, monkeypatch):
    cats = _drifting()
    cats[-1] = _sky(31, seed=77)                                   # as many stars, of another field
    z, logs, metrics = _run(tmp_path, [_aperture(c) for c in cats], monkeypatch)
    ok = np.isfinite(z["rms"])
    assert ok[:14].all() and not ok[14] and metrics["aligned"] == 14


def _twelve():
    """(frame, reference): a reference of 14 stars, a frame of 12 of them, each 1.2 px from its place."""
    rng = np.random.default_rng(11)
    ref = _sky(14, seed=4)
    ang = rng.uniform(0, 2 * np.pi, 12)
    cat = ref[:12].copy()
    cat[:, :2] += [25.0, -40.0] + 1.2 * np.c_[np.cos(ang), np.sin(ang)]
    return cat, ref


def test_twelve_stars_within_the_tolerance_are_accepted_by_the_few_stars_path():
    """The count of stars brought within the tolerance decides (the cited method's rule): no limit on the residual."""
    cat, ref = _twelve()
    r = fit_sparse(cat, ref, cKDTree(ref[:, :2]), W, H)
    assert r is not None
    coef, nmatch, rms, _ = r
    assert nmatch == 12 and 1.0 < rms < rg.STAR_TOL
    x, y = apply_poly(coef, cat[:, 0], cat[:, 1], W, H)
    assert np.hypot(x - ref[:12, 0], y - ref[:12, 1]).max() < rg.STAR_TOL


def test_stars_beyond_the_tolerance_do_not_count():
    """Nine stars within the tolerance and three well beyond it: under the ten pairs asked."""
    cat, ref = _twelve()
    cat[9:, :2] += 30.0
    assert fit_sparse(cat, ref, cKDTree(ref[:, :2]), W, H) is None


def test_the_usual_path_takes_a_frame_of_under_thirty_stars_by_a_linear_map():
    cats = _drifting(n_frames=3, noise=0.2)
    ref, cat = cats[1], cats[2]
    assert len(cat) < 40
    r = fit_frame(cat[:26], ref, cKDTree(ref[:, :2]), W, H)
    assert r is not None
    coef, nmatch, rms, _ = r
    assert 20 <= nmatch <= 26 and rms < 0.6 and not coef[:, 3:].any()


def test_the_usual_path_keeps_its_cubic_from_thirty_pairs():
    sky = _sky(60)
    cat = sky.copy()
    cat[:, :2] += [12.0, 7.0]
    r = fit_frame(cat, sky, cKDTree(sky[:, :2]), W, H)
    assert r is not None and r[1] >= rg.CUBIC_MATCHES and r[0][:, 3:].any()


@pytest.mark.parametrize("n", [20, 31, 300, 1200])
def test_the_usual_path_rejects_another_field(n):
    """Whatever its number of stars: the pairs chance makes are under the share of the stars a true match pairs."""
    ref = _sky(n, seed=1)
    assert fit_frame(_sky(n, seed=2), ref, cKDTree(ref[:, :2]), W, H) is None


def test_the_log_says_what_was_tried_with_the_value_and_the_limit(tmp_path, monkeypatch):
    cats = _drifting()
    cats[-1] = _sky(31, seed=77)
    _z, logs, _ = _run(tmp_path, [_aperture(c) for c in cats], monkeypatch)
    warn = [m for level, m in logs if level == "warning"]
    assert warn[0] == ("1 frame(s) could not be aligned and will be ignored (too few star pairs with the reference: "
                       "each one below)")
    line = next(m for m in warn if m.startswith("f14.fits left out: "))
    assert "too few to match" not in line and "residual too large" not in line
    assert line.startswith("f14.fits left out: 31 stars detected, ")
    assert " star pairs with the reference within 4 px after the shift of the stars' vote (10 needed)" in line
    assert line.endswith(" stars brought within 2 px (10 needed)") and "the triangles of stars: " in line


def test_the_frames_aligned_by_a_linear_map_are_said(tmp_path, monkeypatch):
    z, logs, _ = _run(tmp_path, [_aperture(c) for c in _drifting()], monkeypatch)
    n = int((z["nmatch"] < rg.CUBIC_MATCHES).sum())
    assert n >= 5
    assert ("info", f"sparse field: {n} frame(s) aligned from under 30 star pairs (a linear map, no cubic terms)") in logs


def test_the_new_lines_reach_the_french_window():
    from astro_gpu import texts
    from astro_gpu.client import i18n
    from astro_gpu.client.services.run import engine_text
    fr = json.loads((texts.LOCALE / "fr" / "run.json").read_text(encoding="utf-8"))
    lines = ["2 frame(s) could not be aligned and will be ignored (too few star pairs with the reference: each one below)",
             "a.fits left out: 31 stars detected, 4 star pairs with the reference within 4 px after the shift of the "
             "stars' vote (10 needed)",
             "a.fits left out: 31 stars detected, 4 star pairs with the reference within 4 px after the shift of the "
             "stars' vote (10 needed); the triangles of stars: 3 stars brought within 2 px (10 needed)",
             "a.fits left out: 31 stars detected, 27 star pairs with the reference within 4 px after the shift of the "
             "stars' vote, 8 kept by the fit (10 needed)",
             "sparse field: 15 frame(s) aligned from under 30 star pairs (a linear map, no cubic terms)"]
    i18n.set_language("fr")
    try:
        said = [engine_text(m) for m in lines]
    finally:
        i18n.set_language("en")
    assert all(s != m for s, m in zip(said, lines))
    assert said[0] == ("2 image(s) non alignée(s), écartée(s) (trop peu de paires d'étoiles avec la référence : le "
                       "détail de chacune suit)")
    assert "4 paires d'étoiles" in said[1] and "(10 exigées)" in said[1] and "a.fits écartée" in said[1]
    assert "3 étoiles amenées à moins de 2 px" in said[2] and "8 gardées par l'ajustement" in said[3]
    assert "15 image(s)" in said[4] and "moins de 30 paires" in said[4]
    assert not any(k for k in fr if "too few to match" in k or "residual too large" in k)


# ---------------------------------------------------------------------------------------- the stars' centres


def test_the_centre_of_a_round_star_is_where_it_was(tmp_path):
    """Stars about 4 px wide, the size the fixed window was made for: their centres to a twentieth of a pixel."""
    rng = np.random.default_rng(6)
    S = 512
    yy, xx = np.mgrid[0:S, 0:S]
    img = 1000 + rng.normal(0, 3, (S, S))
    mid = np.c_[rng.uniform(40, S - 40, 40), rng.uniform(40, S - 40, 40)]
    keep = [k for k in range(40) if all(np.hypot(*(mid[k] - mid[j])) > 30 for j in range(k))]
    mid = mid[keep]
    for x, y in mid:
        img += 6000 * np.exp(-0.5 * (((xx - x) / 1.7) ** 2 + ((yy - y) / 1.7) ** 2))
    f = tmp_path / "lights.f16"
    mm = np.memmap(f, dtype=np.float16, mode="w+", shape=(1, S, S))
    mm[0] = img
    mm.flush()
    del mm
    rg._init(str(f), 1, S, S, 8.0, 2500, "MONO")
    try:
        cat = rg._catalog(0)[1]
    finally:
        rg._G.clear()
    d, _ = cKDTree(cat[:, :2]).query(mid)
    assert len(cat) == len(mid) and np.median(d) < 0.05 and d.max() < 0.2
