"""The deliverables of rosette_reseau_A.py: the exports copied under their names and a 100 % crop of the same zone in each,
chosen where the dark rim around the shipped network's stars is deepest on nebula (a window of the export with the most
stars on nebula, weighted by the rim's depth).    python tools/bench/rosette_reseau_A_zoom.py"""
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rosette_reseau_A as R  # noqa: E402

W, H = 1000, 700
OFF = (37, 122)                                    # array pixel -> export pixel (the Output crop, then the Post crop)
prep = json.loads((R.done_dir("livre", "star_removal") / "prep_stars.json").read_text())
unit, bk = prep["unit"], np.array(prep["background"], np.float32)[:, None, None]
lin = np.load(R.upstream("render_simple") / "lin.npy").astype(np.float32)
stars = R.isolated_peaks(((lin - bk) / unit).mean(0), 100.0, 400)
S = np.load(R.done_dir("livre", "star_removal") / "starless.npy").astype(np.float32).mean(0)
m36, m1020 = R.ring(3, 6), R.ring(10, 20)
pts = []
for y, x in stars:
    c = S[y - 21:y + 22, x - 21:x + 22]
    a, b = float(c[m36].mean()), float(c[m1020].mean())
    if (b - bk.mean()) / unit > 2:
        pts.append((x - OFF[0], y - OFF[1], max(0.0, -100 * (a - b) / b)))
best = None
for y0 in range(0, 2716 - H, 50):
    for x0 in range(0, 3682 - W, 50):
        s = sum(w for x, y, w in pts if x0 + 60 < x < x0 + W - 60 and y0 + 60 < y < y0 + H - 60)
        if best is None or s > best[0]:
            best = (s, x0, y0)
_, x0, y0 = best
print("zone", x0, y0, W, H, "weighted stars", round(best[0], 1))
R.VIEW.mkdir(parents=True, exist_ok=True)
for net, name in (("livre", "rosette_reseau_livre"), ("A", "rosette_reseau_A")):
    src = R.BENCH / "export" / net
    shutil.copy(src / f"{name}.jpg", R.VIEW / f"{name}.jpg")
    shutil.copy(src / f"{name}_16bit.tif", R.VIEW / f"{name}.tif")
    jpg = cv2.imread(str(src / f"{name}.jpg"))
    tif = cv2.imread(str(src / f"{name}_16bit.tif"), cv2.IMREAD_UNCHANGED)
    cv2.imwrite(str(R.VIEW / f"{name}_zoom.jpg"), jpg[y0:y0 + H, x0:x0 + W], [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(str(R.VIEW / f"{name}_zoom.tif"), tif[y0:y0 + H, x0:x0 + W])
