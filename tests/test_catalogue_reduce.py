"""Cut extracts (tools/catalogue_reduce.py): the same format with only a G range of the stars, read unchanged by
core/sirilcat.py, which knows the cut from the header, and answers the same as the full extract within it."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from astro_gpu.core import gaia_extracts as ge
from astro_gpu.core import sirilcat

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_skycache import _write_siril  # noqa: E402

from tools import catalogue_reduce as cr  # noqa: E402

ASTRO = "siril_cat_healpix8_astro.dat"
XPDIR = "siril_cat1_healpix8_xpsamp"


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    """A private ASTRO_GPU_HOME and user home, no other program's data folder, no extract anywhere."""
    home = tmp_path / "userhome"
    home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.setenv("ASTRO_GPU_HOME", str(tmp_path / "aghome"))
    (tmp_path / "nowhere").mkdir()
    for v in ("LOCALAPPDATA", "APPDATA", "XDG_CONFIG_HOME", "XDG_DATA_HOME"):
        monkeypatch.setenv(v, str(tmp_path / "nowhere"))
    monkeypatch.setenv("ASTRO_GPU_GAIA_EXTRACTS", "")
    monkeypatch.setenv("ASTRO_GPU_SIRIL_CATALOGUES", "")
    monkeypatch.setenv("ASTRO_GPU_OFFLINE", "1")
    monkeypatch.setattr(cr, "BLOCK", 97)                # several blocks, cut in the middle of pixels
    sirilcat.reset()
    yield
    sirilcat.reset()


def _sky(seed=3, n=600, per_pixel=60):
    """Stars around (40, 61) over several level-8 pixels, as the extract has them: the brightest ``per_pixel`` of each
    pixel; a crowded corner fills its pixel (complete only to its faintest star), the others are not full."""
    import astropy.units as u
    from astropy_healpix import HEALPix
    rng = np.random.default_rng(seed)
    ra = np.r_[40.0 + rng.uniform(-0.6, 0.6, n), 40.3 + rng.uniform(-0.02, 0.02, 300)]
    dec = np.r_[61.0 + rng.uniform(-0.3, 0.3, n), 61.1 + rng.uniform(-0.01, 0.01, 300)]
    g = rng.uniform(5, 19, len(ra))
    pix = np.asarray(HEALPix(nside=256, order="nested").lonlat_to_healpix(ra * u.deg, dec * u.deg))
    o = np.lexsort((g, pix))
    rank = np.arange(len(o)) - np.searchsorted(pix[o], pix[o])
    keep = np.sort(o[rank < per_pixel])
    return {"ra": ra[keep], "dec": dec[keep], "g": g[keep]}


def _full(tmp_path, chunk_of=None):
    import astropy.units as u
    from astropy_healpix import HEALPix
    st = _sky()
    src = tmp_path / "full"
    _write_siril(src / ASTRO, 1, 8, st)
    h1 = HEALPix(nside=2, order="nested")
    chunk = int(h1.lonlat_to_healpix(40.0 * u.deg, 61.0 * u.deg))
    inside = np.asarray(h1.lonlat_to_healpix(st["ra"] * u.deg, st["dec"] * u.deg)) == chunk
    rng = np.random.default_rng(4)
    xp = {k: v[inside] for k, v in st.items()}
    xp["g"] = np.clip(xp["g"], 5, 15)                    # the spectra extract stops at G 15
    xp["e"] = rng.integers(14, 17, len(xp["g"])).astype(np.int8)
    xp["flux"] = rng.uniform(0.5, 2.0, (len(xp["g"]), 343)) * 10.0 ** (-xp["e"][:, None].astype(float))
    _write_siril(src / XPDIR / f"siril_cat1_healpix8_xpsamp_{chunk}.dat", 2, 8, xp, chunk=(1, chunk))
    return src, st, xp, chunk


def _use(monkeypatch, folder):
    monkeypatch.setenv("ASTRO_GPU_GAIA_EXTRACTS", str(folder))
    sirilcat.reset()


def test_the_cut_files_keep_the_records_of_the_range_in_their_order_and_their_header(tmp_path):
    src, _st, _xp, chunk = _full(tmp_path)
    out = tmp_path / "cut"
    res = cr.reduce_all(src, out, astro_gmax=16.0, xp_gmax=14.0, log=lambda *_: None)
    assert set(res) == {ASTRO, f"siril_cat1_healpix8_xpsamp_{chunk}.dat"}
    for name, sub, hi in ((ASTRO, "", 15999), (f"siril_cat1_healpix8_xpsamp_{chunk}.dat", XPDIR, 14011)):
        full, cut = sirilcat.read_header(src / sub / name), sirilcat.read_header(out / sub / name)
        assert cut is not None and cut.cut == (sirilcat.NO_BRIGHT_CUT, hi) and full.cut is None
        assert (cut.kind, cut.level, cut.chunk_level, cut.chunk, cut.first, cut.last) == \
            (full.kind, full.level, full.chunk_level, full.chunk, full.first, full.last)
        assert cut.cap() == full.cap() and cut.full_cap == full.cap()
        # the header is the full file's but for the cut's field, in bytes the format leaves empty
        a, b = (open(p / sub / name, "rb").read(128) for p in (src, out))
        assert a[:sirilcat.CUT_AT] == b[:sirilcat.CUT_AT] and a[sirilcat.CUT_AT:] == bytes(128 - sirilcat.CUT_AT)
        everything = np.arange(full.n_index) + full.first
        r_full, p_full = full.read(everything)
        r_cut, p_cut = cut.read(everything)
        keep = r_full["g"] <= hi
        assert 0 < keep.sum() < len(r_full)
        assert r_cut.tobytes() == r_full[keep].tobytes() and (p_cut == p_full[keep]).all()
        assert res[name]["kept"] == keep.sum() and res[name]["bytes"] == (out / sub / name).stat().st_size


def test_cut_extracts_answer_as_the_full_ones_within_the_cut(tmp_path, monkeypatch):
    src, st, xp, chunk = _full(tmp_path)
    out = tmp_path / "cut"
    cr.reduce_all(src, out, astro_gmax=16.0, xp_gmax=14.0, log=lambda *_: None)
    queries = [(40.0, 61.0, 0.5, 16.0), (40.3, 61.1, 0.08, 14.5), (40.2, 61.0, 0.3, 12.0), (39.6, 60.8, 0.2, 16.0)]
    sel = (xp["g"] > 7.5) & (xp["g"] < 14.0)
    pos, g = (xp["ra"][sel], xp["dec"][sel]), np.round(xp["g"][sel] * 1000) / 1000
    answers = {}
    for which in (src, out):
        _use(monkeypatch, which)
        answers[which] = ([sirilcat.astro_cone(*q) for q in queries], sirilcat.xp_spectra(*pos, g))
    for full, cut in zip(answers[src][0], answers[out][0]):
        for k in ("RA_ICRS", "DE_ICRS", "Gmag", "pmRA", "pmDE"):
            assert full[k].tobytes() == cut[k].tobytes()
        # the completeness as the stages use it: to the depth they ask (G 16 at most)
        assert min(full["complete_to"], 16.0) == min(cut["complete_to"], 16.0)
    assert any(np.isfinite(a["complete_to"]) and a["complete_to"] < 16 for a in answers[src][0])  # a full pixel
    rows_full, rows_cut = answers[src][1], answers[out][1]
    assert all(r is not None for r in rows_full)
    assert all(a[1].tobytes() == b[1].tobytes() for a, b in zip(rows_full, rows_cut))


def test_the_cut_is_known_and_said_where_the_stars_are_counted(tmp_path, monkeypatch):
    src, _st, _xp, _chunk = _full(tmp_path)
    _use(monkeypatch, src)
    assert sirilcat.limits() == {"astro": None, "xp": None}
    out = tmp_path / "cut"
    cr.reduce_all(src, out, astro_gmax=16.0, xp_gmax=14.0, xp_gmin=7.5, log=lambda *_: None)
    _use(monkeypatch, out)
    assert sirilcat.limits() == {"astro": (-np.inf, 15.999), "xp": (7.489, 14.011)}
    assert ge.status(ge.ASTRO).state == "installed" and ge.status(ge.ASTRO).g_max == 15.999
    assert ge.status(ge.XP).g_max == 14.011
    # a cut file cut again keeps the narrower range; a truncated one is not an extract
    again = cr.reduce_file(out / ASTRO, tmp_path / "again" / ASTRO, None, 17000, log=lambda *_: None)
    assert sirilcat.read_header(tmp_path / "again" / ASTRO).cut[1] == 15999 and again["kept"] == again["records"]
    bad = tmp_path / "bad" / ASTRO
    bad.parent.mkdir()
    bad.write_bytes((out / ASTRO).read_bytes()[:-16])
    assert sirilcat.read_header(bad) is None
    with pytest.raises(ValueError):
        cr.reduce_file(bad, tmp_path / "x" / ASTRO, None, 16000)


def test_the_colour_star_limits_end_at_the_cut_and_say_why_on_hover(tmp_path, monkeypatch):
    from astro_gpu import api, texts

    def infos():
        return [api.ParamInfo(n, n, "", "", "float", 14.0, 14.0, 14.0, True, "default", le=le, step=0.5)
                for n, le in (("xp_gmin", 15.0), ("xp_gmax", 17.5), ("xp_max", 20000))]
    src, _st, _xp, _chunk = _full(tmp_path)
    _use(monkeypatch, src)
    full = infos()
    api._bounded_by_spectra(full)                       # the full extract: the bounds as declared
    assert [i.le for i in full] == [15.0, 17.5, 20000] and all(i.tip == "" for i in full)
    out = tmp_path / "cut"
    cr.reduce_all(src, out, astro_gmax=16.0, xp_gmax=14.0, log=lambda *_: None)
    _use(monkeypatch, out)
    cut = infos()
    api._bounded_by_spectra(cut)
    assert [i.le for i in cut] == [14.0, 14.0, 20000]
    assert [i.tip for i in cut] == [texts.SPECTRA_CUT_TIP] * 2 + [""]
