"""The solution file Astrometry writes is a header with no image: reading it raises no astropy note."""
import warnings

from astropy.io import fits
from astropy.wcs import WCS

from astro_gpu.core.imgutil import read_wcs


def test_a_header_only_solution_is_read_without_a_warning(tmp_path):
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval, w.wcs.crpix, w.wcs.cdelt = [10.0, 20.0], [50.0, 40.0], [-1e-3, 1e-3]
    f = tmp_path / "wcs.fits"
    fits.PrimaryHDU(header=w.to_header(relax=True)).writeto(f)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        got = read_wcs(f)
    assert abs(got.wcs.crval[0] - 10.0) < 1e-9
