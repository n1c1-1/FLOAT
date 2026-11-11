"""FLOAT's own star catalogue: what Settings downloads, where it goes, whether it is installed.

Never automatic: the download runs only when the user asks for it (Settings › Catalogs & astrometry), outside any run
(``online.guard``; a run forbids the network). The 49 files are the ones the installer fetches into ``{app}/sky``:

    siril_cat_healpix8_astro.dat                 star positions, cut to G < 16 (780 MB)
    float_xp_compact/float_xp_compact_<N>.dat    star colours, compact spectra, N = 0..47 (1.07 GB)

THE SINGLE SOURCE OF TRUTH is ``installer/lock/lock.json`` (part ``star_catalogue``: URL, size, SHA-256 and sub-folder of
every file). ``pins()`` reads it at run time where it is shipped with the program (``{root}/setup/lock.json`` of an
installed FLOAT, ``installer/lock/lock.json`` of a source checkout); else it reads ``resources/sky_data.json``, the same
part that ``installer/build.py catalogue`` writes beside the lock (tests/test_sky_catalogue.py checks they agree).

The files go to our own folder (``sirilcat.own_dir()``): positions at its top, the colours in ``float_xp_compact/``, where
``core/sirilcat.py`` reads them. The downloader is ``core/sky_fetch.py`` (the installer's too): ``<file>.part`` resumed with
an HTTP Range request, size and SHA-256 checked, a file already whole is skipped, a cancel keeps what was fetched. The full
Gaia extracts (a Siril project's, once offered here) are still read wherever they are found, so an existing install works.
Progress is handed on as byte counts, which pass 2^31 and must never go through a C int.
"""
from __future__ import annotations

import contextlib
import json
import shutil
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from . import online, sirilcat, sky_fetch
from .gaia_extracts import FILES, XP, Cancelled, DownloadError, _inside, _t, human

RESOURCE = Path(__file__).resolve().parents[1] / "resources" / "sky_data.json"
DISK_MARGIN = 200 << 20                                          # room left beyond the files themselves
KIND = "catalogue"                                               # the job's name for this download

_RUN = threading.Lock()


def lock_file() -> Path | None:
    """The lock shipped with the program: beside ``app`` in an installation, in ``installer/lock`` in a checkout."""
    root = Path(__file__).resolve().parents[2]
    for p in (root.parent / "setup" / "lock.json", root / "installer" / "lock" / "lock.json"):
        if p.is_file():
            return p
    return None


def pins() -> list[sky_fetch.Wheel]:
    """The files of the catalogue, pinned (URL, size, SHA-256, sub-folder), positions first."""
    lock = lock_file()
    part = None
    if lock is not None:
        with contextlib.suppress(OSError, ValueError):
            part = json.loads(lock.read_text(encoding="utf-8")).get("star_catalogue")
    if not part:
        part = json.loads(RESOURCE.read_text(encoding="utf-8"))
    return [sky_fetch.Wheel(**f) for f in part["files"]]


def total_bytes() -> int:
    return sum(w.size for w in pins())


def folder() -> Path:
    return sirilcat.own_dir()


@dataclass
class Status:
    state: str                  # "installed" | "partial" | "missing"
    files: int                  # complete files the engine finds (positions + colour chunks)
    total: int                  # files of the catalogue (49)
    part_bytes: int = 0         # bytes of unfinished downloads in our folder
    where: Path | None = None   # folder the catalogue is read from (the first the engine finds)
    outside: Path | None = None  # that folder when it is neither our cache folder nor the installation's ``sky`` folder


def status() -> Status:
    """What the engine finds (``sirilcat.locate``: our files or the full extracts alike), and what a download left."""
    found = sirilcat.locate()
    n_xp = len(found["xp"])
    files = (found["astro"] is not None) + n_xp
    part = 0
    for d in (folder(), folder() / sirilcat.COMPACT_DIR):
        if d.is_dir():
            for e in d.glob("*.part"):
                with contextlib.suppress(OSError):
                    part += e.stat().st_size
    complete = found["astro"] is not None and n_xp >= FILES[XP]
    state = "installed" if complete else "partial" if files or part else "missing"
    folders = sirilcat.installed_folders()
    where = folders[0] if folders else None
    mine = [folder()] + ([sirilcat.install_dir()] if sirilcat.install_dir() else [])
    outside = where if where is not None and not any(_inside(where, r) for r in mine) else None
    return Status(state, files, len(pins()), part, where, outside)


def download(progress: Callable[[int, int, str], None] | None = None, cancel: threading.Event | None = None,
             attempts: int = 3) -> None:
    """Fetch the files not whole yet into our folder. ``progress(done, total, text)``: bytes. Raises ``Cancelled`` or
    ``DownloadError``; always makes the engine look again for the files."""
    if online.offline():
        raise DownloadError(_t("The program is in offline mode: nothing is downloaded."))
    online.guard("the star catalogue")
    if not _RUN.acquire(blocking=False):
        raise DownloadError(_t("A download is already running."))
    cancel = cancel or threading.Event()
    try:
        items = pins()
        total = sum(w.size for w in items)
        dest = folder()
        if progress is not None:
            progress(0, total, _t("Checking the files"))
        dest.mkdir(parents=True, exist_ok=True)
        need = sum(w.size - _have(dest, w) for w in items) + DISK_MARGIN
        free = shutil.disk_usage(dest).free
        if free < need:
            raise DownloadError(_t("Not enough free disk space in {folder}: {need} needed, {free} free.",
                                   folder=dest, need=human(need), free=human(free)))

        def report(p: sky_fetch.Progress):
            if progress is not None:
                progress(int(p.overall_done), int(p.overall_total),
                         _t("File {i} of {n}", i=p.index, n=p.count)
                         + _t(": downloading {done} of {size}", done=human(p.file_done), size=human(p.wheel.size)))

        try:
            sky_fetch.download_all(items, dest, report, cancel, _ssl(), attempts=attempts)
        except sky_fetch.Cancelled as e:
            raise Cancelled() from e
        except sky_fetch.DownloadError as e:
            raise DownloadError(_error(e)) from e
        if progress is not None:
            progress(total, total, _t("Done"))
    except OSError as e:
        raise DownloadError(_t("Cannot write in {folder}: {why}", folder=folder(), why=e.strerror or e)) from e
    finally:
        _RUN.release()
        sirilcat.reset()


def _have(dest: Path, w: sky_fetch.Wheel) -> int:
    """Bytes of ``w`` already on the disk (the whole file, else its ``.part``)."""
    d = dest / w.dest if w.dest else dest
    for name in (w.filename, w.filename + ".part"):
        with contextlib.suppress(OSError):
            return min((d / name).stat().st_size, w.size)
    return 0


def _ssl():
    try:
        import certifi
        return sky_fetch.ssl_context(certifi.where())
    except ImportError:
        return sky_fetch.ssl_context(None)


def _error(e: sky_fetch.DownloadError) -> str:
    if e.kind == "hash":
        return _t("A file was damaged during the download (its checksum does not match the published one). It was "
                  "deleted; start again. {why}", why=e.detail)
    if e.kind == "disk":
        return _t("The disk is full: {why}", why=e.detail)
    return _t("The connection failed ({why}). What was downloaded is kept: start again to go on.", why=e.detail)
