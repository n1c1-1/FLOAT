"""The figures of a session from its journal.

    python stats.py <session dir>

Prints the number of commands, of clicks (double-clicks count once), of drags and keys, and the time from the
session's start to each mark.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path


def fmt(sec: float) -> str:
    sec = int(round(sec))
    return f"{sec // 3600}:{sec % 3600 // 60:02d}:{sec % 60:02d}" if sec >= 3600 else f"{sec // 60}:{sec % 60:02d}"


def main(argv):
    if not argv:
        print(__doc__)
        return 2
    path = Path(argv[0]) / "journal.jsonl"
    if not path.exists():
        print(f"No journal in {argv[0]}")
        return 1
    recs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            recs.append(json.loads(line))
        except ValueError:
            pass
    starts = [i for i, r in enumerate(recs) if r.get("cmd") == "start"]
    cmds = [r for r in recs if r.get("cmd") not in ("start", "quit", "_eval")]
    c = Counter(r["cmd"] for r in cmds)
    print(f"Sessions started: {len(starts)}")
    print(f"Commands: {len(cmds)}")
    print(f"Clicks: {c['click']}  (of which double: {sum(1 for r in cmds if r['cmd'] == 'click' and 'double' in r['args'])}, "
          f"right: {sum(1 for r in cmds if r['cmd'] == 'click' and 'right' in r['args'])})")
    print(f"Drags: {c['drag'] + c['press']}  Keys: {c['key']}  Typing: {c['type']}  Hovers: {c['move']}  "
          f"Wheel: {c['wheel']}  Files chosen: {c['choose_file'] + c['choose_folder']}  Drops: {c['drop_files']}  "
          f"Waits: {c['wait']}  Screenshots: {sum(1 for r in recs if r.get('shot'))}")
    if recs:
        from datetime import datetime
        first = datetime.strptime(recs[0]["time"], "%Y-%m-%d %H:%M:%S")
        last = datetime.strptime(recs[-1]["time"], "%Y-%m-%d %H:%M:%S")
        print(f"Duration: {fmt((last - first).total_seconds())} (from {recs[0]['time']})")
        n_cmd = 0
        n_click = 0
        for r in recs:
            if r.get("cmd") not in ("start", "quit", "_eval", "mark"):
                n_cmd += 1
                n_click += r.get("cmd") == "click"
            if r.get("cmd") == "mark":
                t = datetime.strptime(r["time"], "%Y-%m-%d %H:%M:%S")
                print(f"  {fmt((t - first).total_seconds()):>8}  after {n_cmd:3d} commands, {n_click:3d} clicks: "
                      f"{' '.join(r.get('args', []))}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
