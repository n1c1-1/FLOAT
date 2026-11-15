"""The tests never reach the user's real client settings (%APPDATA%/astro_gpu): tests/conftest.py gives every test process
(each xdist worker too) a private APPDATA before anything reads it, and refuses, in the process, any write to the real
folder. The window the user may have open rewrites that file at any time, so it is made unreachable rather than compared."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import conftest as cf
import pytest


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
        with pytest.raises(cf.UserFolderWrite):                     # nor read: his settings are not the test's input
            cf._audit("open", (str(real), "r", 0))
        with pytest.raises(cf.UserFolderWrite):                     # nor his templates listed
            cf._audit("os.scandir", (str(real.parent / "templates"),))
        with pytest.raises(cf.UserFolderWrite):                     # nor a project made in his Documents folder
            cf._audit("os.mkdir", (str(Path.home() / "Documents" / "FLOAT" / "x"), 0o777, None))
    finally:
        cf._VIOLATIONS[:] = before                                   # these refusals are the test, not a violation


def test_every_test_has_the_user_data_of_its_own(tmp_path):
    """The autouse fixture of tests/conftest.py: the settings file, the templates folder beside it, the recent projects
    in it and the cache of calibrated frames are the test's; going through them opens nothing outside tmp_path."""
    import sys

    from astro_gpu.client import prefs, templatefiles
    opened: list[str] = []

    def spy(event, args):
        if event in ("open", "os.listdir", "os.scandir", "os.mkdir", "os.rename", "os.replace") and args and \
                isinstance(args[0], (str, os.PathLike)):
            opened.append(os.path.abspath(os.fspath(args[0])))
    on = [True]
    sys.addaudithook(lambda e, a: spy(e, a) if on[0] else None)   # a hook cannot be taken away: switched off below
    try:
        assert prefs.settings_file() == tmp_path / "settings.json" and prefs.FILE == tmp_path / "settings.json"
        assert prefs.get("recent") == [] and prefs.get("templates") == []
        prefs.add_recent(str(tmp_path / "p"), "p")
        prefs.set("language", "fr")
        assert templatefiles.user_dir() == tmp_path / "templates"
        templatefiles.tabs()
        assert Path(prefs.get("cache_dir")).is_relative_to(tmp_path)
    finally:
        on[0] = False
    assert (tmp_path / "settings.json").is_file()
    real = os.path.normcase(str(_real_file().parent))
    assert opened and not [p for p in opened if os.path.normcase(p).startswith(real)]
    shipped = os.path.normcase(str(templatefiles.shipped_dir()))
    outside = [p for p in opened if not os.path.normcase(p).startswith((os.path.normcase(str(tmp_path)), shipped))]
    assert not outside, outside


def test_what_a_test_set_is_gone_in_the_next_one_a(tmp_path):
    from astro_gpu.client import prefs
    prefs.set("language", "fr")
    prefs.add_recent(str(tmp_path / "a"), "a")


def test_what_a_test_set_is_gone_in_the_next_one_b():
    from astro_gpu.client import prefs
    assert prefs.get("language") == "en" and prefs.recent() == []


def test_a_subprocess_inherits_the_private_appdata():
    out = subprocess.run([sys.executable, "-c", "import os;print(os.environ['APPDATA'])"], capture_output=True,
                         text=True, check=True).stdout.strip()
    assert Path(out) == cf.PRIVATE_APPDATA
