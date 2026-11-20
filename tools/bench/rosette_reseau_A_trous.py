"""The 11 November measure of the hole around stars, on the linear images, and the x2 zone of the report's picture.

    python tools/bench/rosette_reseau_A_trous.py

For the bright stars on nebula: the level 0-4 px from the star's centre against the annulus 8-15 px out, in % of that
level (the image's linear units, sky included): the median over the stars and the share of stars more than 3 % below.
On the starless image of each network, and (control) on the image the Stars block was given. The zone is that of the
report's picture (export pixels 1283, 1098, 300 x 225), cut from the exports and enlarged x2."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rosette_reseau_A as R  # noqa: E402

prep = json.loads((R.done_dir("livre", "star_removal") / "prep_stars.json").read_text())
unit, bk = prep["unit"], np.array(prep["background"], np.float32)[:, None, None]
lin = np.load(R.upstream("render_simple") / "lin.npy").astype(np.float32)
stars = R.isolated_peaks(((lin - bk) / unit).mean(0), 100.0, 400)
c04, a815 = R.ring(0, 4.0001), R.ring(8, 15.0001)
imgs = {"entree": lin.mean(0)}
for net in ("livre", "A"):
    imgs[net] = np.load(R.done_dir(net, "star_removal") / "starless.npy").astype(np.float32).mean(0)
ann = np.array([imgs["livre"][y - 21:y + 22, x - 21:x + 22][a815].mean() for y, x in stars])
sel = (ann - bk.mean()) / unit > 1.0                 # on nebula: more than 1 noise unit over the sky around the star
res = {"stars_bright": len(stars), "on_nebula": int(sel.sum())}
for k, S in imgs.items():
    a = np.array([S[y - 21:y + 22, x - 21:x + 22][c04].mean() for y, x in stars])
    b = np.array([S[y - 21:y + 22, x - 21:x + 22][a815].mean() for y, x in stars])
    pct = (100 * (a - b) / b)[sel]
    res[k] = {"median_pct": round(float(np.median(pct)), 2), "share_below_3pct": round(float(100 * (pct < -3).mean()), 1),
              "p10_pct": round(float(np.percentile(pct, 10)), 2)}
    print(k, res[k], flush=True)
(R.BENCH / "mesures_trous.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
x0, y0, w, h = 1283, 1098, 300, 225
for net, name, out in (("livre", "rosette_reseau_livre", "rosette_trous_reseau_actuel_x2.png"),
                       ("A", "rosette_reseau_A", "rosette_trous_reseau_A_x2.png")):
    im = cv2.imread(str(R.BENCH / "export" / net / f"{name}.jpg"))[y0:y0 + h, x0:x0 + w]
    cv2.imwrite(str(R.VIEW / out), cv2.resize(im, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC))
