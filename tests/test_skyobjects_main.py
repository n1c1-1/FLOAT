"""The object that names a field: the most prominent one covering its centre, whatever the frames that made the stack
(a field centred on the Pelican was named Sh2-109, an 18 degree region lying over it, because it is the biggest)."""
from __future__ import annotations

import pytest
from astropy.wcs import WCS

from astro_gpu.core import skyobjects as so


def _wcs(ra, dec, w, h, scale_arcsec=3.0):
    x = WCS(naxis=2)
    x.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    x.wcs.crval = [ra, dec]
    x.wcs.crpix = [(w + 1) / 2, (h + 1) / 2]
    x.wcs.cdelt = [-scale_arcsec / 3600, scale_arcsec / 3600]
    return x


def obj(name, ra, dec, major, minor=None, angle=0.0):
    return so.SkyObject(name, "", "HII region", ra, dec, major, minor or major, angle)


FIELD = [obj("Sh2-109", 308.41, 40.34, 1080), obj("NGC 7000", 314.82, 44.53, 120, 30),
         obj("IC 5070", 312.75, 44.40, 60, 50), obj("IC 5068", 312.62, 42.48, 40, 30)]


@pytest.mark.parametrize("w,h", [(3000, 2000), (3050, 2010), (2800, 1900)])      # grids of 7, 9 and 12 frames: cropped alike
def test_the_pelican_field_is_named_by_the_object_at_its_centre_not_the_biggest(w, h):
    wcs = _wcs(312.8, 44.3, w, h)
    objs = so.objects_in_field(wcs, w, h, FIELD)
    assert objs[0].name == "Sh2-109"                                    # biggest first, as the annotations list it
    assert so.main_object(wcs, w, h, objects=objs).name == "IC 5070"


def test_messier_outranks_ngc_ic_and_sharpless_when_they_all_cover_the_centre():
    o = [obj("Sh2-155", 10.0, 41.0, 300), obj("IC 1", 10.0, 41.0, 100), obj("NGC 1", 10.0, 41.0, 100), obj("M 31", 10.0, 41.0, 190)]
    wcs = _wcs(10.0, 41.0, 2000, 1500)
    assert so.main_object(wcs, 2000, 1500, o).name == "M 31"
    assert so.main_object(wcs, 2000, 1500, o[:3]).name == "NGC 1"       # NGC and IC share a tier: the designation order decides
    assert so.main_object(wcs, 2000, 1500, o[:1]).name == "Sh2-155"


def test_no_object_covering_the_centre_falls_back_on_the_nearest_in_its_size():
    wcs = _wcs(10.0, 41.0, 2000, 1500)
    near, far = obj("NGC 5", 10.4, 41.0, 20), obj("NGC 6", 10.6, 41.0, 20)
    assert so.main_object(wcs, 2000, 1500, [far, near]).name == "NGC 5"
    assert so.main_object(wcs, 2000, 1500, []) is None
