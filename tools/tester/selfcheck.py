"""The tool's own check: a session from a blank start to a computed project, through every kind of gesture.

    python selfcheck.py --app <checkout> --frames <folder of a few lights> --computed <computed project folder>
                        --out <folder>

On a 1250x685 window, in English: New project, a tooltip on the palette, a Session dragged from the palette (seen
while held), frames dropped on it, its context menu, the device list, a Stacking placed, Run (the name dialog, its
folder answered), Stop, then a computed project opened from File > Open project (its dialog answered) and shown in
Inspect and Post. Each step checks what the screen says; eight screenshots are copied to <out>.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import do  # noqa: E402

FAILS: list[str] = []


def step(sdir, cmd, *args, expect=None, keep=None, out=None):
    r = do.send(sdir, cmd, [str(a) for a in args])
    text = do.show(r)
    print(f"> {cmd} {' '.join(map(str, args))}\n{text}\n", flush=True)
    if expect is not None:
        for e in ([expect] if isinstance(expect, str) else expect):
            if e not in text:
                FAILS.append(f"{cmd} {args}: expected {e!r}")
                print(f"!! expected {e!r}", flush=True)
    if keep and r.get("shot"):
        shutil.copyfile(r["shot"], Path(out) / keep)
    return text


def find(sdir, text) -> tuple[int, int]:
    """Where a button is (the check's own help: a tester reads it off the screenshot)."""
    r = do.send(sdir, "_find", [text])
    for n in r.get("notes", []):
        if n.startswith("at "):
            x, y = n[3:].split()
            return int(x), int(y)
    FAILS.append(f"button {text!r} not found")
    return 0, 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--app", required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--computed", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    import time
    s = out / time.strftime("session_%H%M%S")
    os.environ["TESTER_DEBUG"] = "1"                 # the session answers _find (where a button is)
    code = subprocess.call([sys.executable, str(HERE / "session.py"), "--app", a.app, "--dir", str(s),
                            "--size", "1250x685", "--lang", "en"])
    if code:
        print("the session did not start")
        return 1
    try:
        k = dict(out=out)
        step(s, "click", 150, 53, expect="Untitled", **k)                                  # Home: New project
        step(s, "mark", "project created", **k)
        step(s, "move", 20, 192, expect="Tooltip:", keep="1_tooltip_palette.png", **k)
        step(s, "press", 20, 192, **k)                                                     # a Session from the palette
        step(s, "move", 300, 260, **k)
        step(s, "move", 265, 305, keep="2_drag_held_over_place.png", **k)
        step(s, "release", 265, 305, expect="added", **k)
        step(s, "drop_files", 185, 275, a.frames, expect="lights added", keep="3_frames_dropped.png", **k)
        step(s, "click", 185, 265, "right", expect="On top: an open menu", keep="4_context_menu.png", **k)
        step(s, "key", "escape", **k)
        step(s, "click", 1090, 40, expect="On top: an open list", keep="5_combo_list.png", **k)
        step(s, "key", "escape", **k)
        step(s, "move", 20, 411, **k)                                                      # Stacking, double-clicked
        step(s, "click", 20, 411, "left", "double", expect="to run", **k)
        step(s, "choose_folder", s / "projects", **k)
        step(s, "click", 1203, 40, expect='dialog "Name the project"', **k)               # Run: the name first
        step(s, "click", *find(s, "Browse"), expect="was answered", **k)
        step(s, "click", *find(s, "Save"), expect='dialog "Run project"', **k)              # the run starts
        step(s, "wait", 15, expect=["Running", "Stop [button]"], keep="6_run_in_progress.png", **k)
        step(s, "click", *find(s, "Stop"), expect="Stop", **k)
        t = step(s, "wait", 15, expect="to run", keep="6b_run_stopped.png", **k)       # Run offered again
        if "Running" in t:
            FAILS.append("the run did not stop")
        step(s, "mark", "run started and stopped", **k)
        step(s, "key", "escape", **k)
        step(s, "choose_file", Path(a.computed) / "project.astroproj", **k)
        step(s, "key", "ctrl+o", expect="was answered", **k)                                # File > Open project…
        step(s, "wait", 3, **k)
        step(s, "key", "ctrl+2", **k)                                                       # Inspect
        step(s, "wait", 6, keep="7_inspect.png", **k)
        step(s, "key", "ctrl+3", **k)                                                       # Post
        step(s, "wait", 10, keep="8_post.png", **k)
        step(s, "mark", "computed project seen", **k)
    finally:
        step(s, "quit")
    subprocess.call([sys.executable, str(HERE / "stats.py"), str(s)])
    print("FAILED:\n" + "\n".join(FAILS) if FAILS else "All checks passed.")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
