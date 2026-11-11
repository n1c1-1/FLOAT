"""The Gaia DR3 extracts the engine reads offline: where they are looked for and whether they are installed.

The download is not here any more: Settings fetches FLOAT's own reduced catalogue (``core/sky_catalogue.py``). The full
extracts, HEALPix extracts of Gaia DR3 made by the Siril project (Zenodo, CC BY 4.0) that Settings used to download, are
still read by ``core/sirilcat.py`` wherever they are found, so an existing install keeps working; ``status`` recognises
both forms. They live in our folder, ``gaia_extracts`` of the cache (``sirilcat.own_dir()``):

    gaia_extracts/siril_cat_healpix8_astro.dat                                  star positions (1.5 GB)
    gaia_extracts/siril_cat1_healpix8_xpsamp/siril_cat1_healpix8_xpsamp_<N>.dat  star spectra, N = 0..47

The texts the download produces (progress, errors) are English templates passed through ``translate`` and their numbers
through ``number``: the client sets both to its language.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass
from pathlib import Path

from . import online, sirilcat

ASTRO = "astro"                                                 # star positions
XP = "xp"                                                       # star spectra
SUBDIR = {ASTRO: "", XP: "siril_cat1_healpix8_xpsamp"}
FILES = {ASTRO: 1, XP: 48}                                      # files a complete extract has


class DownloadError(Exception):
    """The download stopped; the message says why in plain words."""


class Cancelled(Exception):
    """The user cancelled; what was downloaded is kept for the next time."""


@dataclass
class Status:
    kind: str
    state: str                  # "installed" | "partial" | "missing"
    files: int                  # complete files the engine finds
    total: int                  # files a complete extract has
    part_bytes: int = 0         # bytes of unfinished downloads in our folder
    outside: Path | None = None  # folder of a copy found elsewhere (not ours), when the engine uses one
    own: bool = False           # every file the engine finds is in our folder
    g_max: float | None = None  # a cut extract (tools/catalogue_reduce.py): the faintest G it holds; None: full


# ------------------------------------------------------------------ texts

def translate(s: str) -> str:
    """The language of the texts (a hook: the client sets it to its translation function)."""
    return s


def number(x: float, digits: int = 1) -> str:
    """A number as text (a hook: the client sets it to write a decimal comma in French)."""
    return f"{x:.{digits}f}"


def _t(template: str, **fields) -> str:
    return translate(template).format(**fields)


def human(n: float) -> str:
    return f"{number(n / 1e9)} {translate('GB')}" if n >= 1e9 else f"{number(n / 1e6, 0)} {translate('MB')}"


# ------------------------------------------------------------------ what is installed

def folder() -> Path:
    return sirilcat.own_dir()


def target_dir(kind: str) -> Path:
    return folder() / SUBDIR[kind] if SUBDIR[kind] else folder()


def _inside(p: Path, root: Path) -> bool:
    try:
        p.resolve().relative_to(root.resolve())
        return True
    except (ValueError, OSError):
        return False


def status(kind: str) -> Status:
    """What the engine finds for this extract (``sirilcat.locate``), and what an unfinished download left."""
    found = sirilcat.locate()
    if kind == ASTRO:
        paths = [found["astro"].path] if found["astro"] is not None else []
    else:
        paths = [cf.path for cf in found["xp"].values()]
    # the installation's own ``sky`` folder is ours too (the setup fetched it): not "another program's data folder"
    roots = [folder()] + ([sirilcat.install_dir()] if sirilcat.install_dir() else [])
    mine = [p for p in paths if any(_inside(p, r) for r in roots)]
    outside = next((p.parent for p in paths if p not in mine), None)
    part = 0
    d = target_dir(kind)
    if d.is_dir():
        for e in d.iterdir():
            if e.name.endswith((".part", ".bz2")) and e.is_file() and (kind == XP) == ("xpsamp" in e.name):
                with contextlib.suppress(OSError):
                    part += e.stat().st_size
    total = FILES[kind]
    state = "installed" if len(paths) >= total else "partial" if paths or part else "missing"
    cut = sirilcat.limits()[kind]
    return Status(kind, state, len(paths), total, part, outside, len(mine) == len(paths), cut[1] if cut else None)


def _plain(e: BaseException) -> str:
    """The failure of a request in a few plain words."""
    code = getattr(getattr(e, "response", None), "status_code", None)
    if code:
        return _t("the server answered HTTP {code}", code=code)
    return online._plain(e)
