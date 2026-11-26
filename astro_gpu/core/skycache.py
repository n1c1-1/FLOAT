"""The sky catalogue cache: under ASTRO_GPU_HOME, by HEALPix sky tiles, offline first.

Where: ``$ASTRO_GPU_HOME/cache`` (default ``~/.astro_gpu/cache``). The cache an older version kept in
``~/.astro_gpu/cache`` whatever ASTRO_GPU_HOME said is still READ when ASTRO_GPU_HOME points elsewhere (never written,
moved or deleted), so that earlier downloads are not lost.

Gaia stars are cached by sky tile, not by query: a tile is a pixel of the HEALPix grid (nested scheme) and holds every
Gaia DR3 star of one magnitude range whose position (epoch 2016.0) falls in it. Any field inside tiles already fetched
needs no network: a re-stack of the same field, whose astrometric solution moves the query centres a little, or a crop.
A "layer" is a set of tiles of one magnitude range, one set of columns and one tile size, in its own folder with a
``layer.json``. Tile sizes:

    G < 16 (the astrometry's stars): level 5 (1.8 deg)      deeper (the catalogues stage's): level 6 (0.9 deg)

A query is served by any complete layer with the columns and a magnitude range that contain it (the catalogues stage's
deep layer serves the astrometry too). Tiles are no longer downloaded: the layers are those earlier versions fetched, read
as they are, and ``SkyUnavailable`` is raised for the tiles that are not there. The star data FLOAT reads now are its own
files (core/sky_catalogue.py).
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from .stage import StageError

# ------------------------------------------------------------------ folders

def home() -> Path:
    return Path(os.environ.get("ASTRO_GPU_HOME") or Path.home() / ".astro_gpu")


def cache_dir() -> Path:
    return home() / "cache"


_LEGACY: dict[tuple[str, str], list[Path]] = {}


def legacy_dirs() -> list[Path]:
    """Caches written by earlier versions (always ``~/.astro_gpu/cache``): read only."""
    old = Path.home() / ".astro_gpu" / "cache"
    key = (str(old), str(cache_dir()))
    if key not in _LEGACY:
        try:
            same = old.resolve() == cache_dir().resolve()
        except OSError:
            same = False
        _LEGACY[key] = [] if same else [old]
    return _LEGACY[key]


def listing(sub: str) -> dict[str, Path]:
    """{file name: its path} of a cache folder, the cache first, then earlier versions' caches (one directory read
    each, instead of one file test per star)."""
    out: dict[str, Path] = {}
    for d in reversed([cache_dir(), *legacy_dirs()]):
        try:
            with os.scandir(d / sub) as it:
                for e in it:
                    out[e.name] = Path(e.path)
        except OSError:
            continue
    return out


def find(rel: str) -> Path | None:
    """The file ``rel`` from the cache, else from an earlier version's cache, else from a folder of the star data
    (``sirilcat.data_folders``: ours, the installation's ``sky``); None when absent."""
    from . import sirilcat
    for d in [cache_dir(), *legacy_dirs(), *sirilcat.data_folders()]:
        p = d / rel
        if p.exists():
            return p
    return None


def _replace(tmp: Path, path: Path) -> None:
    """os.replace, patient with Windows: replacing a file another thread or process is replacing or reading at the
    same moment fails there with 'access denied'; the other writer wrote the same content."""
    for i in range(20):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if i == 19 or not path.exists():
                if path.exists():
                    break
                raise
            time.sleep(0.05 * (i + 1))
    try:
        tmp.unlink()
    except OSError:
        pass


def _tmp(path: Path, suffix: str) -> Path:
    return path.with_name(f"{path.stem}.{os.getpid()}.{threading.get_ident()}.tmp{suffix}")


def save_npz(path: Path, **arrays) -> None:
    """Written whole or not at all (a stopped run or two runs at once never leave half a file)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp(path, ".npz")
    np.savez(tmp, **arrays)
    _replace(tmp, path)


def save_npy(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp(path, ".npy")
    np.save(tmp, arr)
    _replace(tmp, path)


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp(path, ".json")
    tmp.write_text(json.dumps(obj, indent=1), encoding="utf-8")
    _replace(tmp, path)


# ------------------------------------------------------------------ HEALPix geometry

def _hp(level: int):
    from astropy_healpix import HEALPix
    return HEALPix(nside=2 ** level, order="nested")


def _sep(ra1, dec1, ra2, dec2):
    r = np.radians
    v = np.sin(r(dec1)) * np.sin(r(dec2)) + np.cos(r(dec1)) * np.cos(r(dec2)) * np.cos(r(np.subtract(ra1, ra2)))
    return np.degrees(np.arccos(np.clip(v, -1.0, 1.0)))


def tiles_in_cone(level: int, ra: float, dec: float, radius: float) -> np.ndarray:
    """Every tile of the level that the cone touches (exhaustive: a tile is kept when the cone's centre is in it or
    when its boundary, sampled finely, comes within the radius plus the sampling step)."""
    import astropy.units as u
    hp = _hp(level)
    res = hp.pixel_resolution.to_value(u.deg)
    cand = np.asarray(hp.cone_search_lonlat(ra * u.deg, dec * u.deg, (radius + 1.5 * res) * u.deg), np.int64)
    if not len(cand):
        return cand
    step = 16
    lon, lat = hp.boundaries_lonlat(cand, step=step)
    d = _sep(ra, dec, lon.to_value(u.deg), lat.to_value(u.deg)).min(1)
    centre = int(hp.lonlat_to_healpix(ra * u.deg, dec * u.deg))
    keep = (d <= radius + 1.6 * res / step) | (cand == centre)
    return np.unique(cand[keep])


def tile_of(level: int, ra: np.ndarray, dec: np.ndarray) -> np.ndarray:
    import astropy.units as u
    return np.asarray(_hp(level).lonlat_to_healpix(np.asarray(ra) * u.deg, np.asarray(dec) * u.deg), np.int64)


# ------------------------------------------------------------------ magnitude filters

def parse_filter(gfilter: str) -> tuple[float | None, float]:
    """A VizieR magnitude filter ('<18.5', '18.5..20.5') as (lower bound or None, upper bound): lo <= G < hi."""
    f = gfilter.strip()
    if ".." in f:
        lo, hi = f.split("..")
        return float(lo), float(hi)
    if f.startswith("<"):
        return None, float(f[1:].lstrip("="))
    raise ValueError(f"unsupported magnitude filter {gfilter!r}")


def filter_text(lo: float | None, hi: float) -> str:
    return f"<{hi:g}" if lo is None else f"{lo:g}..{hi:g}"


# ------------------------------------------------------------------ layers of tiles

class Layer:
    """Tiles of one magnitude range, one set of columns, one HEALPix level: ``cache/gaia_tiles/<name>/<tile>.npz``."""

    def __init__(self, cols: list[str], lo: float | None, hi: float, level: int, root: Path | None = None):
        self.cols = list(dict.fromkeys(cols))
        self.lo, self.hi, self.level = lo, hi, level
        sig = hashlib.sha1(",".join(self.cols).encode()).hexdigest()[:8]
        name = f"G{'' if lo is None else f'{lo:g}'}-{hi:g}_L{level}_{sig}"
        self.dir = (root or cache_dir() / "gaia_tiles") / name

    @classmethod
    def read(cls, d: Path) -> Layer | None:
        try:
            j = json.loads((d / "layer.json").read_text(encoding="utf-8"))
            lay = cls(j["cols"], j["lo"], j["hi"], j["level"], d.parent)
            lay.dir = d
            return lay
        except (OSError, ValueError, KeyError):
            return None

    def serves(self, cols: list[str], lo: float | None, hi: float) -> bool:
        lo_ok = self.lo is None or (lo is not None and self.lo <= lo + 1e-9)
        return set(cols) <= set(self.cols) and lo_ok and self.hi >= hi - 1e-9

    def path(self, tile: int) -> Path:
        return self.dir / f"{int(tile)}.npz"

    def missing(self, tiles) -> list[int]:
        return [int(t) for t in tiles if not self.path(t).exists()]

    def load(self, tile: int) -> dict:
        with np.load(self.path(tile)) as z:
            return {k: z[k] for k in z.files}


def layers() -> list[Layer]:
    root = cache_dir() / "gaia_tiles"
    if not root.is_dir():
        return []
    return [lay for d in sorted(root.iterdir()) if d.is_dir() and (lay := Layer.read(d)) is not None]


class SkyUnavailable(StageError):
    """What is asked of the sky is not on this computer (a run never fetches). A StageError: its message is the
    stage's failure, said as it is."""


def gaia_stars(cones: list[tuple[float, float, float]], cols: list[str], gfilter: str, log: Callable | None = None,
               progress: Callable[[float, str], None] | None = None, label: str = "Gaia stars") -> dict:
    """Every Gaia DR3 star (columns ``cols``, VizieR names, positions at epoch 2016.0) inside the union of the cones
    [(ra, dec, radius deg)] whose G passes ``gfilter`` ('<18.5' or '18.5..20.5'), from the cached tiles of an earlier
    version. Raises SkyUnavailable when tiles are missing."""
    lo, hi = parse_filter(gfilter)
    need = list(dict.fromkeys(list(cols) + ["RA_ICRS", "DE_ICRS", "Gmag"]))

    # a complete layer that holds the query (the smallest one: fewest stars to read)
    best = None
    first_miss = None
    for lay in layers():
        if lay.serves(need, lo, hi):
            ts = np.unique(np.concatenate([tiles_in_cone(lay.level, *c) for c in cones])) if cones else np.zeros(0, np.int64)
            miss = lay.missing(ts)
            if not miss:
                size = (lay.hi - (lay.lo if lay.lo is not None else 0.0), -lay.level)
                if best is None or size < best[0]:
                    best = (size, lay, ts)
            elif first_miss is None:
                first_miss = (len(miss), len(ts))
    if best is None:
        why = f"{first_miss[0]} of the {first_miss[1]} sky tiles of the field are" if first_miss else "the sky tiles of the field are"
        raise SkyUnavailable(f"{label}: {why} not on this computer (the offline star catalogue is not installed)")
    lay, ts = best[1], best[2]
    parts = [lay.load(t) for t in ts]
    out = {k: np.concatenate([p[k] for p in parts]) if parts else np.zeros(0) for k in need}
    g = out["Gmag"].astype(np.float64)
    keep = np.isfinite(g) & (g < hi)
    if lo is not None:
        keep &= g >= lo
    keep &= in_cones(out["RA_ICRS"], out["DE_ICRS"], cones)
    return {k: out[k][keep] for k in cols}


def _xyz(ra, dec) -> np.ndarray:
    r = np.radians
    ra, dec = np.asarray(ra, np.float64), np.asarray(dec, np.float64)
    return np.stack([np.cos(r(dec)) * np.cos(r(ra)), np.cos(r(dec)) * np.sin(r(ra)), np.sin(r(dec))], -1)


def in_cones(ra, dec, cones) -> np.ndarray:
    """Whether each position lies in the union of the cones [(ra, dec, radius deg)] (a k-d tree per radius: the
    catalogues stage asks for hundreds of small cones and millions of stars)."""
    from scipy.spatial import cKDTree
    ra, dec = np.asarray(ra, np.float64), np.asarray(dec, np.float64)
    inside = np.zeros(len(ra), bool)
    if not len(ra) or not cones:
        return inside
    p = _xyz(ra, dec)
    c = np.asarray(cones, np.float64)
    for r in np.unique(c[:, 2]):
        sel = c[c[:, 2] == r]
        chord = 2 * np.sin(np.radians(r) / 2) * (1 + 1e-12)
        d, _ = cKDTree(_xyz(sel[:, 0], sel[:, 1])).query(p, distance_upper_bound=chord)
        inside |= np.isfinite(d)
    return inside


# ------------------------------------------------------------------ Hipparcos (whole sky, V < 9: 22 000 stars)

HIP_FILE = "hipparcos/hip_main_v{vmax:g}.npz"


def hipparcos(cols: list[str], vmax: float) -> dict:
    """Every Hipparcos star brighter than V = vmax (VizieR column names), from the star data of this computer (the
    whole sky, V < 9: 22 000 stars). SkyUnavailable when the file is not there."""
    p = find(HIP_FILE.format(vmax=vmax))
    if p is not None:
        with np.load(p) as z:
            if set(cols) <= set(z.files):
                return {k: z[k] for k in cols}
    raise SkyUnavailable(f"the Hipparcos stars (V < {vmax:g}) are not on this computer")
