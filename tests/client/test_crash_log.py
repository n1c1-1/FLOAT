"""An unexpected error is added to the crash log in the application's own folder (beside its settings): its date, the
program's version, the page shown and the whole traceback, so that it can be read after the box is closed."""
from __future__ import annotations

import re
import sys

from astro_gpu import __version__
from astro_gpu.client import app as appmod
from astro_gpu.client import prefs


def test_a_crash_is_written_with_its_date_version_page_and_traceback(tmp_path, monkeypatch):
    monkeypatch.setenv("ASTRO_GPU_SETTINGS", str(tmp_path / "settings.json"))
    prefs.load()
    try:
        raise RuntimeError("the cause")
    except RuntimeError:
        import traceback
        text = "".join(traceback.format_exception(*sys.exc_info()))
    f = appmod.log_crash(text, appmod.page_shown(None))
    assert f == tmp_path / appmod.CRASH_LOG
    appmod.log_crash("second", "Inspect · Calibration")
    out = f.read_text(encoding="utf-8")
    heads = re.findall(r"^===== (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) · (.+)$", out, re.M)
    assert len(heads) == 2 and heads[0][1].endswith(f"{__version__} · Home") and heads[1][1].endswith("Calibration")
    assert "RuntimeError: the cause" in out and "Traceback" in out          # appended, the first one kept
