"""Compact star spectra (tools/catalogue_compact.py): each spectrum stored as its coordinates on a basis that holds the
span of the colour matching functions exactly; core/sirilcat.py reads them in place of the sampled ones, and the colour
the program computes from them (stages/catalogs._xyz) is the sampled spectra's within the int16 rounding."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from astro_gpu.core import gaia_extracts as ge
from astro_gpu.core import sirilcat
from astro_gpu.stages import catalogs

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_catalogue_reduce import _isolated, _sky, _use  # noqa: E402, F401 - the fixture: no extract anywhere
from test_skycache import _write_siril  # noqa: E402

from tools import catalogue_compact as cc  # noqa: E402

XPDIR = "siril_cat1_healpix8_xpsamp"


@pytest.fixture(autouse=True)
def _blocks(monkeypatch):
    monkeypatch.setattr(cc, "BLOCK", 97)                # several blocks, cut in the middle of pixels


def _planck(T):
    lam = sirilcat.XP_WAVELENGTHS[None] * 1e-9
    return 1.0 / lam ** 5 / np.expm1(0.014388 / (lam * np.asarray(T)[:, None]))


def _full(tmp_path):
    """A spectra extract of one chunk: stars as smooth spectra (black bodies 3000-12000 K, 1 % noise)."""
    import astropy.units as u
    from astropy_healpix import HEALPix
    st = _sky()
    h1 = HEALPix(nside=2, order="nested")
    chunk = int(h1.lonlat_to_healpix(40.0 * u.deg, 61.0 * u.deg))
    inside = np.asarray(h1.lonlat_to_healpix(st["ra"] * u.deg, st["dec"] * u.deg)) == chunk
    rng = np.random.default_rng(5)
    xp = {k: v[inside] for k, v in st.items()}
    xp["g"] = np.clip(xp["g"], 5, 15)
    n = len(xp["g"])
    xp["e"] = rng.integers(14, 17, n).astype(np.int8)
    f = _planck(rng.uniform(3000, 12000, n)) * (1 + 0.01 * rng.standard_normal((n, 343)))
    xp["flux"] = f / f.max(1, keepdims=True) * rng.uniform(0.5, 2.0, (n, 1)) * 10.0 ** (-xp["e"][:, None].astype(float))
    src = tmp_path / "full"
    _write_siril(src / XPDIR / f"siril_cat1_healpix8_xpsamp_{chunk}.dat", 2, 8, xp, chunk=(1, chunk))
    return src, xp, chunk


def test_the_compact_file_holds_the_stars_of_the_range_in_their_order_with_the_basis_in_its_header(tmp_path):
    src, _xp, chunk = _full(tmp_path)
    out = tmp_path / "compact"
    res = cc.compact_all(src, out, k=12, xp_gmax=14.0, per_chunk=10_000, log=lambda *_: None)
    name = f"float_xp_compact_{chunk}.dat"
    assert set(res) == {name}
    full = sirilcat.read_header(src / XPDIR / f"siril_cat1_healpix8_xpsamp_{chunk}.dat")
    cf = sirilcat.read_header(out / sirilcat.COMPACT_DIR / name)
    assert cf is not None and cf.kind == sirilcat.TYPE_XPCOMPACT and cf.k == 12 and cf.cut == (sirilcat.NO_BRIGHT_CUT, 14011)
    assert (cf.level, cf.chunk_level, cf.chunk, cf.first, cf.last) == \
        (full.level, full.chunk_level, full.chunk, full.first, full.last)
    B = cf.basis()
    assert np.allclose(B.T @ B, np.eye(12), atol=1e-6)
    # the colour matching functions lie in the basis' span: a star's XYZ is kept whatever its spectrum
    W = cc.xyz_weights()
    assert np.allclose(B @ (B.T @ W), W, atol=1e-5 * np.abs(W).max())
    everything = np.arange(full.n_index) + full.first
    r_full, p_full = full.read(everything)
    r_c, p_c = cf.read(everything)
    keep = r_full["g"] <= 14011
    assert 0 < keep.sum() < len(r_full) and res[name]["kept"] == keep.sum()
    assert (p_c == p_full[keep]).all()
    for k in ("ra", "dec", "g"):
        assert (r_c[k] == r_full[keep][k]).all()
    f, fc = full.spectra(r_full[keep]), cf.spectra(r_c)
    lam = sirilcat.XP_WAVELENGTHS
    x, xc = catalogs._xyz(lam, f), catalogs._xyz(lam, fc)
    assert np.abs(xc / x - 1).max() < 2e-4
    # the shape: smooth spectra come back close, the noise left out
    assert np.median(np.linalg.norm(fc - f, axis=1) / np.linalg.norm(f, axis=1)) < 0.02
    # truncated, or of a later format version: not an extract
    bad = tmp_path / "bad" / name
    bad.parent.mkdir()
    data = (out / sirilcat.COMPACT_DIR / name).read_bytes()
    bad.write_bytes(data[:-5])
    assert sirilcat.read_header(bad) is None
    bad.write_bytes(data[:sirilcat.COMPACT_AT] + bytes([sirilcat.COMPACT_VERSION + 1]) + data[sirilcat.COMPACT_AT + 1:])
    assert sirilcat.read_header(bad) is None


def test_the_engine_reads_the_compact_spectra_as_the_sampled_ones(tmp_path, monkeypatch):
    src, xp, chunk = _full(tmp_path)
    out = tmp_path / "compact"
    cc.compact_all(src, out, k=12, xp_gmax=14.0, xp_gmin=7.5, per_chunk=10_000, log=lambda *_: None)
    sel = (xp["g"] > 7.5) & (xp["g"] < 14.0)
    pos, g = (xp["ra"][sel], xp["dec"][sel]), np.round(xp["g"][sel] * 1000) / 1000
    _use(monkeypatch, src)
    rows_full = sirilcat.xp_spectra(*pos, g)
    _use(monkeypatch, out)
    assert len(sirilcat.locate()["xp"]) == 1 and sirilcat.limits()["xp"] == (7.489, 14.011)
    assert ge.status(ge.XP).files == 1 and ge.status(ge.XP).g_max == 14.011
    assert sirilcat.installed_folders() == [out]
    rows = sirilcat.xp_spectra(*pos, g)
    assert all(r is not None for r in rows) and all(r is not None for r in rows_full)
    lam = rows[0][0]
    assert np.array_equal(lam, sirilcat.XP_WAVELENGTHS)
    x = catalogs._xyz(lam, np.stack([r[1] for r in rows_full]))
    xc = catalogs._xyz(lam, np.stack([r[1] for r in rows]))
    assert np.abs(xc / x - 1).max() < 2e-4
    # both installed in one folder: the sampled spectra are read
    both = tmp_path / "both"
    for d in (XPDIR, sirilcat.COMPACT_DIR):
        (both / d).mkdir(parents=True)
        for p in ((src if d == XPDIR else out) / d).iterdir():
            (both / d / p.name).write_bytes(p.read_bytes())
    _use(monkeypatch, both)
    assert next(iter(sirilcat.locate()["xp"].values())).kind == sirilcat.TYPE_XPSAMP
