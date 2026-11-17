"""The star halos the network leaves (stages/star_removal.take_off_halos): the extended profile of the stars is built
from the image's own bright stars (Slater, Harding & Mihos 2009; Infante-Sainz et al. 2020), scaled to each bright
star on its core in the stars image, and the part of the wing the starless image still holds is moved to the stars
image."""
from __future__ import annotations

import numpy as np
import pytest

from astro_gpu.core.params import params_hash
from astro_gpu.stages import star_removal as sr

SIGMA = 0.05          # the sky noise per pixel
FWHM = 5.0
N = 1200


def _star(rr, core, wing):
    """A Moffat core (FWHM 5 px, beta 3) and an r^-3 wing."""
    a = FWHM / (2 * np.sqrt(2 ** (1 / 3) - 1))
    return core * (1 + (rr / a) ** 2) ** -3 + wing * (1 + (rr / FWHM) ** 2) ** -1.5


def _network_keeps(rr):
    """What the fake network leaves of a star at each radius: nothing of the core (4 FWHM), half of the wing at 40 px,
    more further out (the measured behaviour on the bench)."""
    return np.where(rr <= 4 * FWHM, 0.0, 1 - 1 / (1 + ((rr - 4 * FWHM) / 20.0) ** 2))


def _scene(stars, nebula=None, seed=0):
    """(lin, starless after the fake network, the true sky): each star (y, x, core, wing); the fake network takes the
    core off, leaves the wing as _network_keeps says, and takes a uniform pedestal off (the faint stars of the field)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:N, :N].astype(np.float32)
    sky = np.zeros((N, N), np.float32)
    if nebula is not None:
        y0, x0, amp, s = nebula
        sky += amp * np.exp(-((yy - y0) ** 2 + (xx - x0) ** 2) / (2 * s ** 2))
    sky += SIGMA * rng.standard_normal((N, N)).astype(np.float32)
    lin, taken = sky.copy(), np.zeros((N, N), np.float32)
    for y, x, core, wing in stars:
        rr = np.hypot(yy - y, xx - x)
        s = _star(rr, core, wing).astype(np.float32)
        lin += s
        taken += s * (1 - _network_keeps(rr)).astype(np.float32)
    starless = lin - taken - 0.2
    return np.repeat(lin[None], 3, 0), np.repeat(starless[None], 3, 0), sky - 0.2


GRID = [(y, x, 2000.0, 136.0) for y in (300, 600, 900) for x in (300, 600, 900)]


def _ring_means(img, y, x, radii):
    yy, xx = np.mgrid[:N, :N]
    rr = np.hypot(yy - y, xx - x)
    return np.array([img[(rr >= a) & (rr < a + 4)].mean() for a in radii])


@pytest.fixture(scope="module")
def grid():
    lin, starless, truth = _scene(GRID)
    valid = np.ones((N, N), bool)
    out, info = sr.take_off_halos(lin, starless, valid)
    return lin, starless, truth, out, info


def test_the_wing_left_by_the_network_is_taken_off_to_within_the_noise(grid):
    lin, starless, truth, out, info = grid
    assert info["psf_stars"] >= 5 and len(info["stars"]) == 9
    radii = np.arange(12, 140, 8)
    before = _ring_means(starless[0] - truth, 600, 600, radii) / SIGMA
    after = _ring_means(out[0] - truth, 600, 600, radii) / SIGMA
    assert before.max() > 2.0, before                          # the defect: a disc of a few sigma
    assert np.abs(after).max() < 0.5, after                    # gone, within the noise


def test_starless_and_stars_still_add_up_to_the_image(grid):
    lin, starless, truth, out, info = grid
    stars = lin - out
    assert np.array_equal(out + stars, lin) or np.allclose(out + stars, lin, atol=1e-5)
    assert np.allclose(out[0], out[1]) and np.allclose(out[1], out[2])       # a grey image stays grey
    assert (starless - out).min() >= -1e-6                                     # only light taken off, never added


def test_a_star_on_a_bright_nebula_leaves_the_nebula_within_the_noise():
    stars = GRID[:4] + [(800, 800, 2000.0, 136.0)] + GRID[5:]       # one star on the nebula's brightest part
    lin, starless, truth = _scene(stars, nebula=(800, 800, 60 * SIGMA, 90.0), seed=1)
    out, info = sr.take_off_halos(lin, starless, np.ones((N, N), bool))
    radii = np.arange(12, 140, 8)
    after = _ring_means(out[0] - truth, 800, 800, radii) / SIGMA
    assert np.abs(after).max() < 0.6, after                    # no hole dug in the nebula, no halo left


def test_a_faint_star_is_left_alone():
    faint = (450, 450, 40 * SIGMA, 136.0 * 40 * SIGMA / 2000.0)      # its wing far below the noise
    lin, starless, truth = _scene(GRID + [faint], seed=2)
    out, info = sr.take_off_halos(lin, starless, np.ones((N, N), bool))
    assert all(np.hypot(y - 450, x - 450) > 3 for y, x in info["stars"])
    assert len(info["stars"]) == 9


def test_keep_is_the_previous_result_exactly():
    lin, starless, truth = _scene(GRID[:2], seed=3)
    valid = np.ones((N, N), bool)
    assert sr.halos_step(lin, starless, valid, "keep")[0] is starless


def test_the_setting_and_its_key():
    """Star halos: Remove by default, absent from the key (the results computed before keep their keys; the version
    says they were computed with an earlier one); Keep is a key of its own."""
    p = sr.StarRemovalParams()
    assert p.halos == "remove" and sr.StarRemovalParams.model_fields["halos"].json_schema_extra["choices"] == [
        "remove", "keep"]
    assert params_hash(p) == "44136fa355b3678a"                # the key of a result made before the setting
    assert params_hash(sr.StarRemovalParams(halos="keep")) != params_hash(p)
    assert sr.StarRemoval().version.startswith("5-")
    from astro_gpu import texts
    assert texts.choice_label("remove", "fr") == "Retirer" and texts.choice_label("keep", "fr") == "Garder"
    assert texts.param_label("star_removal", "halos", "fr") == "Halos d'étoiles"
