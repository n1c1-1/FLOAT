"""Send one command to a running session and print what the screen shows after it.

    python do.py <session dir> <command> [args...]

Prints the path of the screenshot taken after the command, then what a person reads on screen: the window's title,
a tooltip, the notices, the status bar, what is on top (a dialog, a menu, a list).
"""
from __future__ import annotations

import json
import os
import socket
import sys
from pathlib import Path

PATH_ARGS = {"choose_file": 0, "choose_folder": 0, "drop_files": 2, "open": 0}


def send(session_dir, cmd: str, args: list[str], timeout: float = 240.0) -> dict:
    sdir = Path(session_dir)
    try:
        port = int((sdir / "port").read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return {"shot": "", "notes": ["No session is running in this folder: start one with session.py."]}
    if cmd in PATH_ARGS:
        k = PATH_ARGS[cmd]
        args = args[:k] + [os.path.abspath(a).replace("\\", "/") for a in args[k:]]
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=10) as s:
            s.settimeout(timeout)
            s.sendall((json.dumps({"cmd": cmd, "args": args}) + "\n").encode("utf-8"))
            data = b""
            while not data.endswith(b"\n"):
                chunk = s.recv(65536)
                if not chunk:
                    break
                data += chunk
    except OSError as e:
        return {"shot": "", "notes": [f"The session does not answer ({e}). It may have ended: start it again."]}
    try:
        return json.loads(data.decode("utf-8"))
    except ValueError:
        return {"shot": "", "notes": ["The session gave no answer."]}


def show(r: dict) -> str:
    lines = [r.get("shot", "")]
    if r.get("title"):
        lines.append("Window: " + r["title"])
    for d in r.get("on_top", []):
        if "popup" in d:
            lines.append("On top: an open " + d["popup"])
        else:
            lines.append("On top: dialog \"" + d.get("dialog", "") + "\"" +
                         (": " + d["text"] if d.get("text") else "") + (" (" + d["more"] + ")" if d.get("more") else ""))
    if r.get("tooltip"):
        lines.append("Tooltip: " + r["tooltip"])
    for n in r.get("notices", []):
        lines.append("Notice: " + n)
    if r.get("status"):
        lines.append("Status bar: " + "  |  ".join(r["status"]))
    for n in r.get("notes", []):
        lines.append("Note: " + n)
    return "\n".join(lines)


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    r = send(argv[0], argv[1], argv[2:])
    out = show(r)
    sys.stdout.buffer.write((out + "\n").encode("utf-8"))
    return 0 if r.get("shot") or argv[1] == "quit" else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
