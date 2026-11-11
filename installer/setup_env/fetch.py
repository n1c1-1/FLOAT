"""Downloading the wheels of the lock file and the star catalogue: exact bytes (SHA-256), resumed after a cut, with a
progress report.

The downloader itself is ``astro_gpu/core/sky_fetch.py`` (standard library only), shared with FLOAT's Settings, which
fetches the same star catalogue files: staged beside this script by the build, else read from the source checkout. What
is here is the installer's own: the disk space check made before any download.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

#: disk room per byte of wheel at the peak: the wheel (1), what it unpacks to in uv's cache (1.8 measured on the CUDA
#: build) and, where hard links do not exist (exFAT), a second full copy for the installed files
DISK_FACTOR_LINKED = 2.9
DISK_FACTOR_COPIED = 4.7
DISK_MARGIN = 800 << 20


def _shared():
    here = Path(__file__).resolve().parent
    path = here / "sky_fetch.py"
    if not path.is_file():
        path = here.parents[1] / "astro_gpu" / "core" / "sky_fetch.py"
    spec = importlib.util.spec_from_file_location("sky_fetch", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["sky_fetch"] = mod                  # a dataclass looks its module up by name
    spec.loader.exec_module(mod)
    return mod


shared = _shared()
Wheel, Progress, DownloadError, Cancelled = shared.Wheel, shared.Progress, shared.DownloadError, shared.Cancelled
free_bytes, ssl_context, sha256_of = shared.free_bytes, shared.ssl_context, shared.sha256_of
download_one, download_all = shared.download_one, shared.download_all


def hard_links_work(folder: str | os.PathLike) -> bool:
    """Whether this volume has hard links: uv then installs without a second copy of everything."""
    probe = Path(folder) / ".link_probe"
    link = Path(folder) / ".link_probe2"
    try:
        probe.write_bytes(b"x")
        os.link(probe, link)
        return True
    except OSError:
        return False
    finally:
        for f in (link, probe):
            try:
                f.unlink()
            except OSError:
                pass


def needed_bytes(missing: list[Wheel], linked: bool = True) -> int:
    """The room these wheels need at the peak: downloaded, unpacked and installed. Nothing to install: none."""
    total = sum(w.size for w in missing)
    return int(total * (DISK_FACTOR_LINKED if linked else DISK_FACTOR_COPIED)) + DISK_MARGIN if total else 0
