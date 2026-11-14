"""Start a usability session: the application's window, held by a background process, driven by do.py.

    python session.py --app <checkout dir> --dir <session dir> [--size 1250x685|1920x1080] [--lang fr|en]
                      [--project <path>] [--onscreen]

The window is not shown on the screen (unless --onscreen). The session has its own settings folder inside the session
dir, so it never reads or writes anyone's own settings or recent projects. It prints the first screenshot when ready.
A session with no command for 30 minutes ends by itself; `do.py <session dir> quit` ends it at once.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import do  # noqa: E402

GAIA = Path.home() / ".astro_gpu" / "cache" / "gaia_extracts"     # the star catalogue installed on this machine


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--app", required=True, help="the application's checkout folder")
    ap.add_argument("--dir", required=True, help="the session's folder (created)")
    ap.add_argument("--size", default="1920x1080")
    ap.add_argument("--lang", default="en", choices=("en", "fr"))
    ap.add_argument("--project", default="")
    ap.add_argument("--onscreen", action="store_true")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--idle-minutes", type=float, default=30.0, help="the session ends after this long without a "
                    "command (and stops any run it started)")
    a = ap.parse_args()
    app = Path(a.app).resolve()
    sdir = Path(a.dir).resolve()
    sdir.mkdir(parents=True, exist_ok=True)
    if not (app / "astro_gpu" / "client" / "app.py").exists():
        print(f"Not an application checkout: {app}")
        return 2
    if (sdir / "port").exists() and (sdir / "port").read_text(encoding="utf-8").strip():
        r = do.send(sdir, "shot", [], timeout=60)
        if r.get("shot"):
            print("A session is already running in this folder:")
            print(do.show(r))
            return 0
    w, h = (int(v) for v in a.size.lower().split("x"))
    appdata = sdir / "settings"
    settings = appdata / "astro_gpu" / "settings.json"
    settings.parent.mkdir(parents=True, exist_ok=True)
    if settings.exists():
        try:
            data = json.loads(settings.read_text(encoding="utf-8"))
        except ValueError:
            data = {}
    else:
        data = {}
    data["language"] = a.lang
    projects = sdir / "projects"                     # where the session's new projects go unless the tester browses
    projects.mkdir(exist_ok=True)
    data.setdefault("last_folder", projects.as_posix())
    settings.write_text(json.dumps(data, indent=1), encoding="utf-8")
    (sdir / "screen.json").write_text(json.dumps({"screens": [
        {"name": "tester", "x": 0, "y": 0, "width": w, "height": h, "logicalDpi": 96, "logicalBaseDpi": 96,
         "dpr": 1}]}), encoding="utf-8")
    (sdir / "port").write_text("", encoding="utf-8")
    env = dict(os.environ)
    env.update({"APPDATA": str(appdata), "ASTRO_GPU_SETTINGS": str(settings), "PYTHONPATH": str(app),
                "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"})
    env.pop("PYTEST_CURRENT_TEST", None)
    if GAIA.exists():
        env["ASTRO_GPU_GAIA_EXTRACTS"] = str(GAIA)
    if a.onscreen:
        env.pop("QT_QPA_PLATFORM", None)
    else:
        env["QT_QPA_PLATFORM"] = "offscreen:configfile=screen.json"     # relative: a drive's colon ends the option
        if Path("C:/Windows/Fonts").is_dir():
            env["QT_QPA_FONTDIR"] = "C:/Windows/Fonts"                   # the offscreen platform has no fonts else
    cmd = [a.python, str(HERE / "host.py"), "--app", str(app), "--dir", str(sdir), "--size", a.size,
           "--idle-minutes", str(a.idle_minutes)]
    if a.project:
        cmd += ["--project", str(Path(a.project).resolve())]
    if a.onscreen:
        cmd += ["--onscreen"]
    log = open(sdir / "host.log", "ab")
    kw = dict(cwd=str(sdir), env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, close_fds=True)
    if os.name == "nt":
        flags = 0x08000000 | 0x00000200          # no console window, a process group of its own
        try:
            proc = subprocess.Popen(cmd, creationflags=flags | 0x01000000, **kw)      # out of the caller's job
        except PermissionError:
            proc = subprocess.Popen(cmd, creationflags=flags, **kw)
    else:
        proc = subprocess.Popen(cmd, start_new_session=True, **kw)
    t0 = time.time()
    while time.time() - t0 < 180:
        if proc.poll() is not None:
            print(f"The application stopped while starting (see {sdir / 'host.log'}).")
            return 1
        try:
            port = (sdir / "port").read_text(encoding="utf-8").strip()
        except OSError:
            port = ""
        if port:
            break
        time.sleep(0.5)
    else:
        print("The application did not start within 3 minutes.")
        return 1
    time.sleep(2.0 if a.project else 1.0)
    r = do.send(sdir, "shot", [], timeout=120)
    print(f"Session started (window {a.size}, language {a.lang}).", flush=True)
    sys.stdout.buffer.write((do.show(r) + "\n").encode("utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
