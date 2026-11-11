"""The full Gaia extracts still read offline (core/gaia_extracts.py ``status``, core/sirilcat.py): the places the engine
looks in, an install of the full extracts that keeps working, and the engine without any extract. The download of these
extracts is gone (Settings fetches FLOAT's own catalogue, tests/test_sky_catalogue.py).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from astro_gpu.core import gaia_extracts as ge
from astro_gpu.core import online, sirilcat

ASTRO_NAME = "siril_cat_healpix8_astro.dat"


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    """A private ASTRO_GPU_HOME and user home, no other program's data folder, no extract anywhere, no network setting."""
    home = tmp_path / "userhome"
    home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.setenv("ASTRO_GPU_HOME", str(tmp_path / "aghome"))
    (tmp_path / "nowhere").mkdir()
    for v in ("LOCALAPPDATA", "APPDATA", "XDG_CONFIG_HOME", "XDG_DATA_HOME"):
        monkeypatch.setenv(v, str(tmp_path / "nowhere"))
    monkeypatch.setenv("ASTRO_GPU_GAIA_EXTRACTS", "")
    monkeypatch.setenv("ASTRO_GPU_SIRIL_CATALOGUES", "")
    monkeypatch.delenv("ASTRO_GPU_OFFLINE", raising=False)
    online.reset()
    sirilcat.reset()
    yield
    online.reset()
    sirilcat.reset()


# ------------------------------------------------------------------ fake extract files

def _header(level: int, kind: int, chunk: int | None) -> bytearray:
    h = bytearray(128)
    h[:48] = b"Siril test extract".ljust(48, b"\0")
    h[48:53] = bytes([3, level, kind, 1 if chunk is not None else 0, level if chunk is not None else 0])
    if chunk is not None:
        for off in (53, 57, 61):
            h[off:off + 4] = int(chunk).to_bytes(4, "little")
    return h


def _astro_file(n: int = 3000, seed: int = 1) -> bytes:
    """A valid astrometric extract of HEALPix level 1 (48 pixels) with n random stars, all in pixel 5 (random bytes: it
    does not compress, so that the compressed file spans several chunks)."""
    rng = np.random.default_rng(seed)
    rec = np.zeros(n, sirilcat.ASTRO_DT)
    for name, dt in sirilcat.ASTRO_DT.fields.items():
        rec[name] = rng.integers(-1000, 1000, n).astype(dt[0])
    idx = np.where(np.arange(48) >= 5, n, 0).astype("<u4")
    return bytes(_header(1, sirilcat.TYPE_ASTRO, None)) + idx.tobytes() + rec.tobytes()


def _xp_file(chunk: int, n: int = 2, seed: int = 2) -> bytes:
    """A valid spectra extract holding one level-1 pixel (the chunk) and n stars."""
    rng = np.random.default_rng(seed + chunk)
    rec = np.zeros(n, sirilcat.XP_DT)
    rec["ra"] = rng.integers(0, 1000, n)
    rec["dec"] = rng.integers(0, 1000, n)
    rec["g"] = 12000 + np.arange(n)
    rec["flux"] = rng.uniform(0.5, 2.0, (n, 343)).astype(np.float16)
    return bytes(_header(1, sirilcat.TYPE_XPSAMP, chunk)) + np.array([n], "<u4").tobytes() + rec.tobytes()


# ------------------------------------------------------------------ where the engine looks

def test_our_folder_is_searched_and_the_environment_comes_first(tmp_path, monkeypatch):
    own = sirilcat.own_dir()
    assert own == tmp_path / "aghome" / "cache" / "gaia_extracts"
    (own / "siril_cat1_healpix8_xpsamp").mkdir(parents=True)
    (own / ASTRO_NAME).write_bytes(_astro_file(n=10))
    (own / "siril_cat1_healpix8_xpsamp" / "siril_cat1_healpix8_xpsamp_7.dat").write_bytes(_xp_file(7))
    (own / "siril_cat1_healpix8_xpsamp" / "siril_cat1_healpix8_xpsamp_8.dat.part").write_bytes(_xp_file(8)[:300])
    found = sirilcat.locate()
    assert found["astro"].path == own / ASTRO_NAME and list(found["xp"]) == [7]      # a .part is not an extract
    # a folder of the environment wins over ours (the new name, and the older one still accepted)
    other = tmp_path / "other"
    other.mkdir()
    (other / ASTRO_NAME).write_bytes(_astro_file(n=20, seed=5))
    for var in ("ASTRO_GPU_GAIA_EXTRACTS", "ASTRO_GPU_SIRIL_CATALOGUES"):
        monkeypatch.setenv("ASTRO_GPU_GAIA_EXTRACTS", "")
        monkeypatch.setenv("ASTRO_GPU_SIRIL_CATALOGUES", "")
        monkeypatch.setenv(var, str(other))
        sirilcat.reset()
        assert sirilcat.locate()["astro"].path == other / ASTRO_NAME


def test_another_programs_copy_is_an_extra_place_only(tmp_path, monkeypatch):
    share = tmp_path / "nowhere" / "siril"                  # where that program keeps its data (LOCALAPPDATA/siril)
    share.mkdir(parents=True)
    (share / ASTRO_NAME).write_bytes(_astro_file(n=10))
    found = sirilcat.locate()
    assert found["astro"].path == share / ASTRO_NAME
    st = ge.status(ge.ASTRO)
    assert st.state == "installed" and not st.own and st.outside == share
    # ours, once downloaded or copied there, is preferred
    (sirilcat.own_dir()).mkdir(parents=True)
    (sirilcat.own_dir() / ASTRO_NAME).write_bytes(_astro_file(n=10))
    sirilcat.reset()
    assert sirilcat.locate()["astro"].path == sirilcat.own_dir() / ASTRO_NAME and ge.status(ge.ASTRO).own


def test_no_extract_anywhere_the_engine_answers_none(tmp_path):
    assert sirilcat.locate() == {"astro": None, "xp": {}}
    assert sirilcat.astro_cone(315.3, 68.17, 1.0, 14.0) is None
    assert sirilcat.xp_spectra(np.array([315.3]), np.array([68.17]), np.array([10.0])) == [None]
    assert sirilcat.describe() == "none"
    assert not sirilcat.own_dir().exists()                       # looking never creates the folder
    s = ge.status(ge.XP)
    assert (s.state, s.files, s.total, s.part_bytes, s.outside) == ("missing", 0, 48, 0, None)


def test_an_install_of_the_full_extracts_is_still_read_and_counts_as_installed(tmp_path):
    """The extracts a former Settings downloaded (48 spectra chunks of the full form, in our folder) are read by the
    engine and ``sky_catalogue.status`` says installed: nothing to download again."""
    from astro_gpu.core import sky_catalogue
    own = sirilcat.own_dir()
    (own / "siril_cat1_healpix8_xpsamp").mkdir(parents=True)
    (own / ASTRO_NAME).write_bytes(_astro_file(n=10))
    st = sky_catalogue.status()
    assert st.state == "partial" and st.files == 1
    for n in range(48):
        (own / "siril_cat1_healpix8_xpsamp" / f"siril_cat1_healpix8_xpsamp_{n}.dat").write_bytes(_xp_file(n))
    sirilcat.reset()
    st = sky_catalogue.status()
    assert (st.state, st.files, st.where, st.outside) == ("installed", 49, own, None)
    assert ge.status(ge.XP).state == "installed" and ge.status(ge.ASTRO).own
