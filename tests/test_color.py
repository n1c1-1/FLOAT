"""Colour calibration: the stars' white balance (one factor per colour), the Gaia spectra on one grid."""
from __future__ import annotations

import numpy as np


def test_spectra_of_two_grids_are_brought_to_one(tmp_path, monkeypatch):
    """The Catalogues step reads spectra of two grids (the 2 nm sampled ones of the extract, the 10 nm ones an earlier
    version cached): one grid, 400-800 nm every 10 nm, for all."""
    from astro_gpu.core import skycache, sirilcat
    from astro_gpu.stages import catalogs
    monkeypatch.setenv("ASTRO_GPU_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(skycache.Path, "home", classmethod(lambda cls: tmp_path / "userhome"))
    monkeypatch.setenv("ASTRO_GPU_GAIA_EXTRACTS", "")
    monkeypatch.setenv("ASTRO_GPU_SIRIL_CATALOGUES", "")
    sirilcat.reset()
    wide = (sirilcat.XP_WAVELENGTHS.astype(float), np.linspace(1, 2, len(sirilcat.XP_WAVELENGTHS)))
    narrow = (catalogs.XP_GRID.copy(), np.linspace(1.1, 1.5, 41))
    skycache.save_npy(skycache.cache_dir() / "xp" / "11.npy", np.stack(wide))
    skycache.save_npy(skycache.cache_dir() / "xp" / "12.npy", np.stack(narrow))

    class Ctx:
        def log(self, *a):
            pass

        def progress(self, *a):
            pass
    rows = catalogs._xp_spectra(Ctx(), np.array([11, 13, 12]), 0.85, 1.0)
    assert rows[1] is None and all(len(r[0]) == 41 for r in (rows[0], rows[2]))
    assert np.allclose(rows[0][1], np.interp(catalogs.XP_GRID, *wide))
    assert np.array_equal(rows[2][1], narrow[1])


def test_the_stars_white_balance_brings_the_white_reference_to_white():
    """SPCC's white balance: a line through the measured against the catalogue colour ratios, evaluated at the white
    reference. Stars whose colours a demosaiced stack pulled towards grey and tinted orange (red x 1.15, blue x 0.9
    at white, as heart's Classic stack) get factors that bring the white reference back to white; exact colours get
    none; a fifth of wild stars does not move it."""
    from astro_gpu.stages.color import star_white_balance
    rng = np.random.default_rng(3)
    n = 2000
    e = np.c_[rng.lognormal(0, 0.25, n), np.ones(n), rng.lognormal(-0.05, 0.3, n)] * rng.uniform(1, 50, (n, 1))
    wb = star_white_balance(e, e)
    assert np.allclose(wb["k"], 1, atol=1e-9) and wb["stars"] == n
    ratio = e / e[:, 1:2]
    m = np.c_[1.15 * ratio[:, 0] ** 0.8, np.ones(n), 0.9 * ratio[:, 2] ** 0.75] * rng.lognormal(0, 0.08, (n, 3))
    m[: n // 5] *= rng.lognormal(0, 1.0, (n // 5, 3))                                      # wild stars
    wb = star_white_balance(m, e)
    assert wb["k"][1] == 1.0
    assert abs(wb["k"][0] * 1.15 - 1) < 0.03 and abs(wb["k"][2] * 0.9 - 1) < 0.03, wb
    assert star_white_balance(m[:10], e[:10])["k"].tolist() == [1.0, 1.0, 1.0]           # too few stars: none


def test_a_mono_image_has_no_colours_to_calibrate_before_any_star_count(tmp_path):
    import pytest
    from astro_gpu.core.stage import StageError
    from astro_gpu.core.workflow import MONO_NO_COLOUR
    from astro_gpu.stages.color import Color, ColorParams

    class Ctx:
        def has(self, stage):
            return True

        def metric_of(self, stage, key, default=None):
            return True if (stage, key) == ("inventory", "mono") else default

    for st, p in ((Color(), ColorParams()),):   # the catalogues stay: an aligned colour workflow reads a mono reference's
        with pytest.raises(StageError, match="mono image has no colours"):
            st.run(Ctx(), p)
    assert "mono image" in MONO_NO_COLOUR
