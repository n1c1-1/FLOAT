"""The balanced channels of a combination as linear maps of its inputs (stages/combine.Combination.steps ``maps``):
every step of the recipes is linear in the channels, so Post's combination layer can mix the filters again from the
inputs (client/postfusion.py). The run writes the maps in drz_meta.json; no setting or key changes."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from astro_gpu.stages import combine as C
from astro_gpu.stages import recipes
from tests.test_combine import _sky_field


class _Quiet:
    def log(self, *_a, **_k):
        pass

    def metric(self, *_a, **_k):
        pass


def _apply(maps, raw):
    return {r: sum(np.float64(c) * raw[k].astype(np.float64) for k, c in coef.items()) + off
            for r, (coef, off) in maps.items()}


@pytest.mark.parametrize("kind,how", [("sho", "sky"), ("sho", "signal"), ("ho", "sky"), ("rgb", "sky"),
                                      ("lrgb", "sky")])
def test_the_maps_give_the_balanced_channels_of_every_type(kind, how):
    cls = C.BY_KIND[kind]
    ports = cls.ports
    off = {r: 100.0 + 13 * i for i, r in enumerate(ports)}
    noise = {r: 2.0 + i for i, r in enumerate(ports)}
    raw, _neb = _sky_field(off, noise)
    valid = np.ones(next(iter(raw.values())).shape, bool)
    chans = {r: v.copy() for r, v in raw.items()}
    maps: dict = {}
    cls().steps(_Quiet(), cls().spec(None), [{"role": r, "full_scale": 65535.0} for r in ports], chans, valid, {r: 1.0 for r in ports}, how,
                {}, maps)
    got = _apply(maps, raw)
    for r in ports:
        scale = float(np.abs(chans[r]).max())
        assert np.abs(got[r] - chans[r]).max() <= 1e-5 * scale, r


def test_a_continuum_subtraction_is_a_map_of_two_channels():
    spec = recipes.load(Path(C.__file__).parents[1] / "resources" / "combinations" / "hargb.toml")
    ports = recipes.roles(spec)
    raw, _neb = _sky_field({r: 100.0 + i for i, r in enumerate(ports) if r != "Ha"} | {"Ha": 90.0},
                           {r: 1.5 for r in ports})
    yy, xx = np.mgrid[0:raw["Ha"].shape[0], 0:raw["Ha"].shape[1]]
    for k, (y, x) in enumerate([(20, 30), (40, 150), (120, 40), (130, 160), (80, 20), (25, 100), (140, 100)]):
        star = 4000.0 * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * 1.6 ** 2))
        for r in ports:
            raw[r] = (raw[r] + star * (0.6 if r == "Ha" else 1.0)).astype(np.float32)
    valid = np.ones(raw["Ha"].shape, bool)
    chans = {r: v.copy() for r, v in raw.items()}
    maps: dict = {}
    C.CombineRecipe().steps(_Quiet(), spec, [{"role": r, "full_scale": 65535.0} for r in ports], chans, valid,
                            {r: 1.0 for r in ports}, "sky", {}, maps)
    assert len(maps["Ha"][0]) == 2                    # the H-alpha minus the scaled red: two channels
    got = _apply(maps, raw)
    for r in ports:
        assert np.abs(got[r] - chans[r]).max() <= 1e-4 * float(np.abs(chans[r]).max()), r


def test_the_output_matrix_is_the_one_the_run_mixes_with():
    p = C.SHOParams(hue_sii=10.0, hue_ha=40.0, hue_oiii=200.0)
    roles, mat = C.output_matrix(p, C.CombineSHO().spec(None))
    assert roles == ("SII", "Ha", "OIII")
    assert mat == C.hue_palette((10.0, 40.0, 200.0))
    roles, mat = C.output_matrix(None, C.CombineRGB().spec(None))
    assert roles == ("R", "G", "B") and np.allclose(mat, np.eye(3))
