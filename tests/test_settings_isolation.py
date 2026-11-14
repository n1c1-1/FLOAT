"""The tests never reach the user's real client settings (%APPDATA%/astro_gpu): tests/conftest.py gives every test process
(each xdist worker too) a private APPDATA before anything reads it, and refuses, in the process, any write to the real
folder. The window the user may have open rewrites that file at any time, so it is made unreachable rather than compared."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

import conftest as cf


def _real_file() -> Path:
    assert cf._REAL_APPDATA, "the real APPDATA was not recorded"
    return Path(cf._REAL_APPDATA) / "astro_gpu" / "settings.json"


def test_appdata_is_private_for_the_whole_process():
    assert Path(os.environ["APPDATA"]) == cf.PRIVATE_APPDATA
    assert cf.PRIVATE_APPDATA != Path(cf._REAL_APPDATA)
    assert not _real_file().is_relative_to(cf.PRIVATE_APPDATA)


def test_the_settings_file_is_never_the_real_one(monkeypatch):
    from astro_gpu.client import prefs
    monkeypatch.delenv("ASTRO_GPU_SETTINGS", raising=False)
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)       # as at collection time: the APPDATA branch
    f = prefs.settings_file()
    assert f is not None and f.is_relative_to(cf.PRIVATE_APPDATA)
    assert f != _real_file()


def test_a_write_to_the_real_settings_is_refused():
    real = _real_file()
    before = cf._VIOLATIONS[:]
    try:
        with pytest.raises(cf.UserFolderWrite):
            cf._audit("open", (str(real), "w", 0))
        with pytest.raises(cf.UserFolderWrite):
            cf._audit("os.rename", (str(real.with_name("settings.json.tmp")), str(real)))
        cf._audit("open", (str(real), "r", 0))                       # reading it stays allowed
    finally:
        cf._VIOLATIONS[:] = before                                   # these refusals are the test, not a violation


def test_a_subprocess_inherits_the_private_appdata():
    out = subprocess.run([sys.executable, "-c", "import os;print(os.environ['APPDATA'])"], capture_output=True,
                         text=True, check=True).stdout.strip()
    assert Path(out) == cf.PRIVATE_APPDATA
