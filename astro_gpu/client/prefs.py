"""The user's settings, kept between runs: language, folders, compute, catalogs, display, spec mode, recent projects.

They live in %APPDATA%/astro_gpu/settings.json. ASTRO_GPU_SETTINGS names another file, so that tests and scripted
walkthroughs never touch the user's settings; under pytest, without it, nothing is read or written.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

ENV = "ASTRO_GPU_SETTINGS"
RECENT_MAX = 12

DEFAULTS = {
    "language": "en",
    "spec_mode": False,
    "recent": [],                     # [{"path", "name", "target"}], most recent first
    # folders and disk. No workspace (issue #31): a project lives where the user puts it
    # the folder the file dialogs open in: the last one used (issues #31, #32)
    "last_folder": "",
    # the user's own templates (issue #47): {key, tab, name, text, source, blocks}, made with Save as template…
    "templates": [],
    # set once the templates above are written as files, tab "My templates" (templatefiles.import_old); the entries stay
    "templates_moved": False,
    # the last export of each project, by its folder: format, size, content, folder (proposed at the next opening)
    "export_last": {},
    # the sections of Home (Settings › Folders), under the recent projects: each lists the projects of a folder and of
    # its sub-folders. "label" and "note" are English (translated where shown) or typed by the user; "open": shown open.
    # None at first (the night test pass: a folder of the development machine was shown to every new user)
    "home_sections": [],
    "cache_dir": (Path.home() / "astro_gpu_scratch").as_posix(),   # calibrated frames: <cache_dir>/<project>
    "temp_dir": (Path(tempfile.gettempdir()) / "astro_gpu").as_posix(),
    "cache_limit_gb": 200,
    # compute
    "device": "auto",                 # auto | cuda:<n> | xpu:<n> | mps | privateml:<n> | cpu
    # what the card test decided: {device (or "auto"): profile}, core/selftest.py; {}: not tested yet
    "compute_profiles": {},
    "threads": 0,                     # 0 = all cores but two
    "memory_limit_gb": 24,
    # catalogs and astrometry
    "solver": "internal",             # internal | astap | astrometry.net
    "survey_cache": "",
    # file formats
    "format": "FITS",                 # FITS | XISF
    "bits": "32-bit float",
    "compression": "None",
    "keep_frames": True,
    # display
    "auto_stretch": True,
    "units": "arcsec",                # arcsec | px
    # a change that would make computed and up-to-date results computed again waits for Replace / Try variant / Cancel
    "confirm_changes": True,
    # the window on a small screen (issue #44): auto | compact | comfortable; the user's panels per size range
    "layout_density": "auto",
    "layout": {},
    # master library
    "masters_dir": "",
    "master_tolerance_temp": 2.0,     # °C
    "master_tolerance_days": 180,
}

_data: dict = {}
FILE: Path | None = None
#: what went wrong with the settings file and is not said yet: (kind, detail), kind "unreadable" (detail: the name it
#: is kept under, "" when it could not be set aside) or "unwritable" (detail: the error)
_trouble: list[tuple[str, str]] = []
_listener = None
_write_failed = False                 # said once until a write succeeds again
_frozen = False                       # an unreadable file that could not be set aside: never written over


def settings_file() -> Path | None:
    """The settings file: ASTRO_GPU_SETTINGS, else %APPDATA%/astro_gpu/settings.json; None under pytest without
    ASTRO_GPU_SETTINGS (the tests never read or write the user's file)."""
    env = os.environ.get(ENV)
    if env:
        return Path(env)
    if "PYTEST_CURRENT_TEST" in os.environ:
        return None
    base = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
    return Path(base) / "astro_gpu" / "settings.json"


def watch(fn) -> None:
    """``fn(kind, detail)`` is told what goes wrong with the settings file (the window's banner): what happened before
    it was given at once, then each trouble as it comes. None: told no more."""
    global _listener
    _listener = fn
    while fn is not None and _trouble:
        _tell(*_trouble.pop(0))


def _tell(kind: str, detail: str) -> None:
    if _listener is None:
        _trouble.append((kind, detail))
        return
    try:
        _listener(kind, detail)
    except RuntimeError:                              # a window already gone
        pass


def _read_disk():
    """The settings file's object, None when there is none or it is not readable as one."""
    if FILE is None or not FILE.exists():
        return {}
    try:
        got = json.loads(FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return got if isinstance(got, dict) else None


def _set_aside() -> None:
    """An unreadable settings file kept as settings.json.bad (.bad2... when one is there: never written over), said."""
    global _frozen
    n, kept = 1, None
    while kept is None or kept.exists():
        kept = FILE.with_name(FILE.name + ".bad" + (str(n) if n > 1 else ""))
        n += 1
    try:
        os.replace(FILE, kept)
    except OSError:
        _frozen = True
        _tell("unreadable", "")
        return
    _tell("unreadable", kept.name)


def load():
    global _data, FILE, _frozen, _write_failed
    FILE = settings_file()
    _frozen = _write_failed = False
    _trouble.clear()                                  # those of a file read before
    _data = dict(DEFAULTS)
    _data["recent"] = []
    got = _read_disk()
    if got is None:
        _set_aside()
    else:
        _data.update(got)
    return _data


def system_language() -> str:
    """The language of the system's interface when the program has it ("fr"), else "en": the language of a first
    launch (the night test pass: a French user started in English and had to find Edit > Settings > Language)."""
    import locale
    try:
        import ctypes
        name = locale.windows_locale.get(ctypes.windll.kernel32.GetUserDefaultUILanguage(), "")
    except (AttributeError, OSError):                 # not Windows
        name = locale.getlocale()[0] or ""
    return "fr" if name.lower().startswith("fr") else "en"


def get(key):
    if not _data:
        load()
    return _data.get(key, DEFAULTS.get(key))


def set(key, value):                  # noqa: A001 - a settings store
    if not _data:
        load()
    _data[key] = value
    save(key)


def save(key=None):
    """Write the settings atomically (nothing when there is no settings file: pytest without ASTRO_GPU_SETTINGS). The
    file is read again first: another window of the program may have written it since, so its settings are kept and
    only ``key``, the one being changed, is this window's (all of them without a key). A failure is said once."""
    global _write_failed
    if FILE is None or _frozen:
        return
    disk = _read_disk()
    if disk is None:
        _set_aside()
        if _frozen:
            return
    elif key is not None:
        mine = _data.get(key)
        _data.update(disk)
        _data[key] = mine
    try:
        FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = FILE.with_name(FILE.name + ".tmp")
        tmp.write_text(json.dumps(_data, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, FILE)
        _write_failed = False
    except OSError as e:
        if not _write_failed:
            _write_failed = True
            _tell("unwritable", f"{type(e).__name__}: {e}")


def spec() -> bool:
    """Spec mode: shows what the engine still has to build (◇ marks), disabled."""
    return bool(get("spec_mode"))


def last_folder() -> str:
    """Where a file dialog opens: the last folder used in one (issues #31, #32: every dialog started from a fixed
    workspace, so the flats were looked for again from there after the lights), else the user's home folder."""
    cand = get("last_folder")
    return str(cand) if cand and Path(cand).is_dir() else str(Path.home())


def remember_folder(path) -> None:
    """A file or folder chosen in a dialog: its folder is where the next dialog opens."""
    if not path:
        return
    p = Path(path)
    folder = p if p.is_dir() else p.parent
    if str(folder) and folder.is_dir():
        set("last_folder", folder.as_posix())


def recent() -> list[dict]:
    """The recently opened projects, most recent first: [{"path", "name", "target"}]."""
    return [dict(r) for r in get("recent") or [] if isinstance(r, dict) and r.get("path")]


def add_recent(path: str, name: str, target: str = ""):
    path = Path(path).as_posix()
    rows = [r for r in recent() if Path(r["path"]).as_posix().lower() != path.lower()]
    set("recent", [{"path": path, "name": name, "target": target}] + rows[:RECENT_MAX - 1])


def drop_recent(path: str):
    path = Path(path).as_posix().lower()
    set("recent", [r for r in recent() if Path(r["path"]).as_posix().lower() != path])
