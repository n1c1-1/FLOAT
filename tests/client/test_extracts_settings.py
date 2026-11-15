"""Settings › Catalogs & astrometry: the "Offline Gaia extracts" section, offscreen.

Folder, state of each extract, the two downloads run off the GUI thread, progress, cancel. The download itself is
replaced by a function that reports progress (its real behaviour: tests/test_gaia_extracts.py); nothing is written
outside a scratch folder.
"""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication, QLabel, QLineEdit, QProgressBar, QPushButton  # noqa: E402

from astro_gpu.client import i18n, prefs  # noqa: E402
from astro_gpu.client.services import extracts  # noqa: E402
from astro_gpu.client.services.worker import Gpu, Machine  # noqa: E402
from astro_gpu.client.settingsdlg import SettingsDialog  # noqa: E402
from astro_gpu.core import gaia_extracts as ge  # noqa: E402
from astro_gpu.core import sirilcat  # noqa: E402
from astro_gpu.core import sky_catalogue as sc  # noqa: E402

MACHINE = Machine([Gpu("cuda:0", "NVIDIA RTX 4060 Ti", 16.0)], 16, 64.0)
ASTRO_NAME = "siril_cat_healpix8_astro.dat"


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture()
def env(tmp_path, monkeypatch, qapp):
    monkeypatch.setenv("ASTRO_GPU_SETTINGS", str(tmp_path / "settings.json"))
    monkeypatch.setenv("ASTRO_GPU_HOME", str(tmp_path / "home"))
    (tmp_path / "nowhere").mkdir()
    for v in ("LOCALAPPDATA", "APPDATA", "XDG_CONFIG_HOME", "XDG_DATA_HOME"):
        monkeypatch.setenv(v, str(tmp_path / "nowhere"))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "userhome"))
    monkeypatch.setenv("ASTRO_GPU_GAIA_EXTRACTS", "")
    monkeypatch.setenv("ASTRO_GPU_SIRIL_CATALOGUES", "")
    prefs.load()
    prefs.set("spec_mode", False)
    i18n.set_language("en")
    sirilcat.reset()
    yield tmp_path
    i18n.set_language("en")
    sirilcat.reset()


def _texts(root) -> list[str]:
    return [w.text() for w in root.findChildren(QLabel) if w.isVisibleTo(root) and w.text()] + \
           [w.text() for w in root.findChildren(QPushButton) if w.isVisibleTo(root)]


def _gaia(shown: list[str]) -> list[str]:
    """The texts of this section only: the Asterion section after it says its own "Not installed"."""
    end = [i for i, t in enumerate(shown) if t in ("Asterion plate solver", "Solveur astrométrique Asterion")]
    return shown[:end[0]] if end else shown


def _page(qapp):
    d = SettingsDialog(None, MACHINE)
    d.show()
    d.show_page("catalogs")
    qapp.processEvents()
    return d, d.pages.currentWidget()


def _button(page, start):
    return next(b for b in page.findChildren(QPushButton) if b.text().startswith(start))


def _wait(qapp, cond, seconds=10.0):
    end = time.monotonic() + seconds
    while not cond() and time.monotonic() < end:
        qapp.processEvents()
        time.sleep(0.005)
    qapp.processEvents()
    assert cond()


def _tiny_astro(path: Path):
    h = bytearray(128)
    h[:48] = b"Siril test extract".ljust(48, b"\0")
    h[48:53] = bytes([3, 1, 1, 0, 0])
    rec = np.zeros(4, sirilcat.ASTRO_DT)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(h) + np.full(48, 4, "<u4").tobytes() + rec.tobytes())


def _fake_catalogue(folder: Path):
    """The 49 files of the catalogue as valid tiny extracts (positions and 48 compact chunks) in ``folder``."""
    import test_sky_catalogue as t
    for name, (sub, body) in t._bodies().items():
        (folder / sub).mkdir(parents=True, exist_ok=True)
        (folder / sub / name).write_bytes(body)


def test_the_section_shows_the_folder_the_state_and_the_two_downloads(env, qapp):
    d, page = _page(qapp)
    shown = _texts(page)
    assert "Offline star data" in shown and "Nothing is fetched while a project runs" in " ".join(shown)
    assert "online" not in " ".join(_gaia(shown))
    assert "Star catalogue" in shown and "Install (1.85 GB)" in shown
    assert "Positions and colours of the stars, 1.85 GB · needed by Astrometry and Color" in shown
    assert _gaia(shown).count("Not installed") == 2 and "Open folder" in shown        # the bright stars too
    assert not [t for t in shown if "Zenodo" in t or "Download positions" in t or "Download spectra" in t
                or "Star spectra" in t or "Star positions" in t]                       # the full extracts are not offered
    assert not [t for t in shown if "Gaia DR3 folder" in t or "12 GB" in t or "◇" in t]      # the old, unbuilt widgets
    edit = next(e for e in page.findChildren(QLineEdit) if e.text() == str(sc.folder()))
    assert edit.isReadOnly() and sc.folder() == env / "home" / "cache" / "gaia_extracts"
    assert not [b for b in page.findChildren(QProgressBar) if b.isVisibleTo(page)]            # idle: no bar
    b = _button(page, "Install (")
    assert b.isEnabled() and "cache folder" in b.toolTip()
    assert "full Gaia extracts are also read" in b.toolTip()             # said on the row too
    assert not sc.folder().exists()                                  # looking at the page writes nothing
    d.close()


def test_the_row_says_which_blocks_need_the_catalogue(env, qapp):
    needed = "Positions and colours of the stars, 1.85 GB · needed by Astrometry and Color"
    d, page = _page(qapp)
    shown = _gaia(_texts(page))
    assert shown.index("Star catalogue") < shown.index(needed)
    d.close()
    i18n.set_language("fr")
    d, page = _page(qapp)
    shown = _texts(page)
    assert "Catalogue d'étoiles" in shown and "Installer (1,85 Go)" in shown
    assert "Positions et couleurs des étoiles, 1,85 Go · nécessaire à l'astrométrie et à la couleur" in shown
    d.close()


def test_the_states_of_the_row_and_its_button(env, qapp):
    other = env / "nowhere" / "siril"                               # another program's data folder
    _tiny_astro(other / ASTRO_NAME)
    sirilcat.reset()
    d, page = _page(qapp)
    assert "Found in another program's data folder" not in " ".join(_texts(page))      # not complete: partly
    assert "Partly downloaded: 1 of 49 files" in _texts(page)                          # the positions only
    assert _button(page, "Install (").isEnabled()
    (other / ASTRO_NAME).unlink()
    # a download stopped half-way
    (sc.folder() / "float_xp_compact").mkdir(parents=True)
    (sc.folder() / "float_xp_compact" / "float_xp_compact_3.dat.part").write_bytes(b"x" * 3_000_000)
    sirilcat.reset()
    d.rebuild()
    qapp.processEvents()
    assert "Partly downloaded: 3 MB" in _texts(d.pages.currentWidget())
    # installed in our folder: greyed, the reason on hover
    (sc.folder() / "float_xp_compact" / "float_xp_compact_3.dat.part").unlink()
    _fake_catalogue(sc.folder())
    sirilcat.reset()
    d.rebuild()
    qapp.processEvents()
    page = d.pages.currentWidget()
    assert f"Installed: {sc.folder()}" in _texts(page) and _gaia(_texts(page)).count("Not installed") == 1
    b = _button(page, "Install (")
    assert not b.isEnabled() and b.toolTip() == f"Already installed: {sc.folder()}"
    d.close()


def test_a_catalogue_found_elsewhere_greys_the_button_and_says_where(env, qapp):
    elsewhere = env / "nowhere" / "siril"
    _fake_catalogue(elsewhere)
    sirilcat.reset()
    d, page = _page(qapp)
    assert f"Found in another program's data folder: {elsewhere}" in _texts(page)
    b = _button(page, "Install (")
    assert not b.isEnabled() and str(elsewhere) in b.toolTip()
    d.close()


def test_the_full_extracts_of_a_former_install_are_still_read(env, qapp):
    """48 sampled-spectra chunks and the positions in our folder: installed, button greyed, nothing to download."""
    import test_gaia_extracts as t
    own = sirilcat.own_dir()
    (own / "siril_cat1_healpix8_xpsamp").mkdir(parents=True)
    (own / ASTRO_NAME).write_bytes(t._astro_file(n=10))
    for n in range(48):
        (own / "siril_cat1_healpix8_xpsamp" / f"siril_cat1_healpix8_xpsamp_{n}.dat").write_bytes(t._xp_file(n))
    sirilcat.reset()
    d, page = _page(qapp)
    assert f"Installed: {own}" in _texts(page) and not _button(page, "Install (").isEnabled()
    d.close()


def test_a_download_runs_off_the_gui_thread_and_reports(env, qapp, monkeypatch):
    main = threading.get_ident()
    ran_in = []

    def fake(progress=None, cancel=None, **kw):
        ran_in.append(threading.get_ident())
        progress(0, 1000, "Checking the files")
        progress(400, 1000, "File 3 of 49: downloading 0.4 GB of 1.0 GB")
        progress(1000, 1000, "Done")
        _fake_catalogue(sc.folder())
        sirilcat.reset()
    monkeypatch.setattr(sc, "download", fake)
    d, page = _page(qapp)
    _button(page, "Install (").click()
    _wait(qapp, lambda: not extracts.job().running() and "Done." in _texts(d.pages.currentWidget()))
    assert ran_in and ran_in[0] != main
    page = d.pages.currentWidget()
    assert not [b for b in page.findChildren(QProgressBar) if b.isVisibleTo(page)]
    assert f"Installed: {sc.folder()}" in _texts(page) and not _button(page, "Install (").isEnabled()
    d.close()


def test_cancel_stops_the_download_and_closing_the_window_does_not(env, qapp, monkeypatch):
    started = threading.Event()
    stopped = []

    def slow(progress=None, cancel=None, **kw):
        progress(100, 1000, "File 1 of 49: downloading 0.1 GB of 0.8 GB")
        started.set()
        while not cancel.wait(0.01):
            pass
        stopped.append("catalogue")
        raise ge.Cancelled()
    monkeypatch.setattr(sc, "download", slow)
    d, page = _page(qapp)
    _button(page, "Install (").click()
    _wait(qapp, started.is_set)
    _wait(qapp, lambda: any("downloading 0.1 GB" in t for t in _texts(d.pages.currentWidget())))
    page = d.pages.currentWidget()
    bar = next(b for b in page.findChildren(QProgressBar))
    assert bar.isVisibleTo(page) and bar.value() == 100
    assert not _button(page, "Install (").isEnabled() and not _button(page, "Install bright").isEnabled()   # one at a time
    d.close()                                                                # the download goes on
    d.deleteLater()
    qapp.processEvents()
    assert extracts.job().running() and not stopped
    d2, page = _page(qapp)                                                   # a window opened later shows where it is
    assert any("downloading 0.1 GB" in t for t in _texts(page))
    next(b for b in page.findChildren(QPushButton) if b.text() == "Cancel").click()
    _wait(qapp, lambda: not extracts.job().running()
          and any(t.startswith("Cancelled.") for t in _texts(d2.pages.currentWidget())))
    assert stopped == ["catalogue"]
    assert _button(d2.pages.currentWidget(), "Install (").isEnabled()
    d2.close()


def test_an_error_is_shown_in_plain_words(env, qapp, monkeypatch):
    def refuse(progress=None, cancel=None, **kw):
        raise ge.DownloadError("Not enough free disk space in X: 1.9 GB needed")
    monkeypatch.setattr(sc, "download", refuse)
    d, page = _page(qapp)
    _button(page, "Install (").click()
    _wait(qapp, lambda: not extracts.job().running() and
          "Not enough free disk space in X: 1.9 GB needed" in _texts(d.pages.currentWidget()))
    d.close()


def test_the_real_job_fetches_the_pins_from_the_local_server_into_the_cache(env, qapp, monkeypatch):
    """The button to the files: the job off the GUI thread runs core/sky_catalogue.download for real, on a fake server."""
    import test_sky_catalogue as t
    server = t.Server(t._bodies())
    monkeypatch.setattr(sc, "pins", server.pins)
    try:
        d, page = _page(qapp)
        _button(page, "Install (").click()
        _wait(qapp, lambda: not extracts.job().running() and "Done." in _texts(d.pages.currentWidget()), 60)
        assert sc.status().state == "installed" and len(server.requests) == 49
        d.close()
    finally:
        server.close()


def test_the_bright_stars_have_their_row_and_install_from_it(env, qapp, monkeypatch):
    """The third row: the Hipparcos list and the blind solver's index, what they are for, installed or not with their
    size, Install run by the one download job off the GUI thread (core/bright_stars.py, replaced here)."""
    from astro_gpu.core import bright_stars as bs
    from astro_gpu.core import skycache
    d, page = _page(qapp)
    shown = _gaia(_texts(page))
    assert "Bright stars" in shown and "Find an unknown field; add the brightest stars Gaia lacks" in shown
    assert shown.count("Not installed") == 2
    b = _button(page, "Install bright stars (0.7 GB)")
    assert b.isEnabled() and "Hipparcos and Tycho-2" in b.toolTip() and "25 MB" in b.toolTip()
    main, ran_in = threading.get_ident(), []

    def fake(progress=None, cancel=None):
        ran_in.append(threading.get_ident())
        progress(5, 14, "Tycho-2 index: building, 3 of 35")
        for f in bs.FILES:
            skycache.save_npz(skycache.cache_dir() / f, x=np.zeros(1000))
        progress(14, 14, "Done")
    monkeypatch.setattr(bs, "download", fake)
    monkeypatch.setattr(sc, "download", lambda *a, **k: pytest.fail("the catalogue download ran"))
    b.click()
    _wait(qapp, lambda: not extracts.job().running() and "Done." in _texts(d.pages.currentWidget()))
    assert ran_in and ran_in[0] != main
    page = d.pages.currentWidget()
    assert any(t.startswith("Installed · ") for t in _texts(page))
    b = _button(page, "Install bright stars")
    assert not b.isEnabled() and b.toolTip() == "Already installed."
    (skycache.cache_dir() / bs.FILES[2]).unlink()
    d.rebuild()
    qapp.processEvents()
    assert "Partly installed: 2 of 3 files" in _texts(d.pages.currentWidget())
    d.close()


def test_the_section_has_its_french(env, qapp):
    i18n.set_language("fr")
    d, page = _page(qapp)
    shown = _texts(page)
    assert "Données d'étoiles hors ligne" in shown and "Catalogue d'étoiles" in shown and "Installer (1,85 Go)" in shown
    assert _gaia(shown).count("Non installé") == 2
    assert "Étoiles brillantes" in shown and "Installer étoiles brillantes (0,7 Go)" in shown
    from astro_gpu.core import bright_stars as bs
    assert d._bright_status_text(bs.Status("installed", 3, 3, 737_000_000)) == "Installé · 737 Mo"
    _fake_catalogue(env / "nowhere" / "siril")
    sirilcat.reset()
    d.rebuild()
    qapp.processEvents()
    assert any(t.startswith("Trouvé dans le dossier de données d'un autre programme") for t in _texts(d.pages.currentWidget()))
    assert _button(d.pages.currentWidget(), "Installer (").toolTip().startswith("Déjà installé : ")
    assert not i18n.missing(), sorted(i18n.missing())
    # the download's own texts
    ge.translate, ge.number = i18n._, i18n.number
    try:
        assert ge.human(1.85e9) == "1,9 Go"
        assert ge.human(5e8) == "500 Mo"
        for t in ("Checking the files", "A download is already running.", "Done",
                  "The disk is full: {why}", "The connection failed ({why}). What was downloaded is kept: start again to go on."):
            assert i18n._(t) != t
        assert "Fichier 3 sur 49" in ge._t("File {i} of {n}", i=3, n=49)
    finally:
        ge.translate, ge.number = (lambda s: s), (lambda x, d=1: f"{x:.{d}f}")
    d.close()


def test_the_progress_moves_with_byte_counts_above_four_gigabytes(env, qapp):
    """Issue #157: the signal and its slot were declared int, so a count past 2^31 raised OverflowError on every report
    and the bar and the text stayed where they were."""
    d, page = _page(qapp)
    j = extracts.job()
    for bar in page.findChildren(QProgressBar):
        bar.setVisible(True)
    j.progress.emit(5 * 2**30, 10 * 2**30, "5.0 GB of 10.0 GB")
    _wait(qapp, lambda: d.ex["msg"].text() == "5.0 GB of 10.0 GB")
    assert d.ex["bar"].value() == 500
    j._progress(9 * 2**30, 10 * 2**30, "9.0 GB of 10.0 GB")
    _wait(qapp, lambda: d.ex["bar"].value() == 900)
    d.close()
