"""Narrow-band sessions of lights of 300 s with a master dark of 300 s (level 848 ADU) and a normalised master flat of 1 s
(floats on the 0-1 scale, M27 in SII, Ha, OIII): the master flat is used as it is, never dark-subtracted, and its
vignetting is a number. A flat that cannot be one (not a number, no light left once the dark level is taken off) stops the
calibration naming it, instead of a vignetting of NaN found by the registration ("too few stars")."""
from __future__ import annotations

import numpy as np
import pytest

from astro_gpu.stages.calibrate import master_flat
from tests.test_mixed_darks import H, W, _write

LEVEL = 848.0                      # a black level here, never a dark's


class _Ctx:
    _files_read = _files_total = 0

    def log(self, *a, **k): ...
    def progress(self, *a, **k): ...


def _flat(level: float) -> np.ndarray:
    yy, xx = np.mgrid[0:H, 0:W]
    return (level * (1 - 0.1 * (((yy - H / 2) / H) ** 2 + ((xx - W / 2) / W) ** 2))).astype(np.float32)


@pytest.mark.parametrize("level", [0.12, 0.5])
@pytest.mark.parametrize("master", [True, False])
def test_a_normalised_flat_keeps_its_vignetting(tmp_path, level, master):
    p = _write(tmp_path / "flat.fit", _flat(level), EXPTIME=1.0)
    v = master_flat([str(p)], [], None, LEVEL, H, W, 0, _Ctx(), master=master, cfa="MONO")
    assert np.isfinite(v).all() and 0.8 < v.min() < 1.0     # 0.12 x 65535 = 7864 ADU, 848 off: the vignetting stays


def test_a_flat_with_nan_stops_naming_it(tmp_path):
    f = _flat(0.12)
    f[3, 4] = np.nan
    p = _write(tmp_path / "masterFlat_Ha.fit", f, EXPTIME=1.0)
    for master in (True, False):
        with pytest.raises(ValueError, match="masterFlat_Ha.fit"):
            master_flat([str(p)], [], None, LEVEL, H, W, 0, _Ctx(), master=master, cfa="MONO")


def test_a_flat_under_the_dark_level_stops_naming_it(tmp_path):
    f = _flat(0.12)
    f[0, 0] = 2.0                                 # a hot pixel above the 0-1 scale: the file reads as ADU, 0.12 ADU
    p = _write(tmp_path / "masterFlat_SII.fit", f, EXPTIME=1.0)
    with pytest.raises(ValueError, match="masterFlat_SII.fit.*given as a master"):
        master_flat([str(p)], [], None, LEVEL, H, W, 0, _Ctx(), master=False, cfa="MONO")
    v = master_flat([str(p)], [], None, LEVEL, H, W, 0, _Ctx(), master=True, cfa="MONO")
    assert np.isfinite(v).all()
