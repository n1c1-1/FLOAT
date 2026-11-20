"""A snapshot (path, size, date) of a folder tree, to check that a bench wrote nothing into it.
    python tools/bench/rosette_reseau_A_snap.py FOLDER OUT.json  |  ... FOLDER OUT.json --compare BEFORE.json"""
import json
import os
import sys

root, out = sys.argv[1], sys.argv[2]
snap = {}
for d, ds, fs in os.walk(root):
    for n in fs + ds:
        p = os.path.join(d, n)
        try:
            s = os.stat(p)
        except OSError:
            continue
        snap[p] = [s.st_size, s.st_mtime_ns]
json.dump(snap, open(out, "w"))
print(len(snap), "entries")
if "--compare" in sys.argv:
    old = json.load(open(sys.argv[sys.argv.index("--compare") + 1]))
    diff = [p for p in set(old) | set(snap) if old.get(p) != snap.get(p)]
    print("changed, added or removed:", len(diff))
    for p in sorted(diff)[:20]:
        print("  ", p)
