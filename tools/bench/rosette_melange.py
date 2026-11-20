"""The owner's Rosette through his Post settings with a mixed star network (issue #178), next to the shipped render.

    python tools/bench/rosette_melange.py --out DIR --weights mix.pt [--device cuda:0]

Reuses rosette_reseau_A (the Stars and Synthetic stars blocks from the project's upstream results, then Post with the
saved version, through the GUI's export code) with every path redirected into DIR. The shipped render and the shipped
Stars result are read in place (read only). Writes: the render, the two crops of the mix and the same two crops of the
shipped render, and the 11 November hole measure of the mix.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rosette_reseau_A as R  # noqa: E402

SHIPPED_JPG = Path("F:/astro_essais/A_REGARDER/rosette_reseau_A/rosette_reseau_livre.jpg")
SHIPPED_STARS = Path("F:/astro_essais/rosette_reseau_A/etoiles/livre/star_removal")
HOLES = (1283, 1098, 300, 225)      # export pixels: x, y, w, h (enlarged x2 bicubic)
ZOOM = (1200, 1250, 1000, 700)
OFF = (37, 122)                     # array pixel -> export pixel


def write(path: Path, img, params=()):
    ok, buf = cv2.imencode(path.suffix, img, list(params))
    assert ok
    buf.tofile(str(path))
    assert path.exists() and path.stat().st_size > 0, path


def read(path: Path):
    return cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)


def crops(img, out: Path, holes_name: str, zoom_name: str):
    x, y, w, h = HOLES
    c = cv2.resize(img[y:y + h, x:x + w], (2 * w, 2 * h), interpolation=cv2.INTER_CUBIC)
    write(out / holes_name, c)
    x, y, w, h = ZOOM
    write(out / zoom_name, img[y:y + h, x:x + w], [cv2.IMWRITE_JPEG_QUALITY, 95])


def holes_measure(net: str) -> dict:
    prep = json.loads((R.done_dir(net, "star_removal") / "prep_stars.json").read_text())
    unit, bk = prep["unit"], np.array(prep["background"], np.float32)[:, None, None]
    lin = np.load(R.upstream("render_simple") / "lin.npy").astype(np.float32)
    stars = R.isolated_peaks(((lin - bk) / unit).mean(0), 100.0, 400)
    c04, a815 = R.ring(0, 4.0001), R.ring(8, 15.0001)

    def prof(S):
        A = np.array([S[y - 21:y + 22, x - 21:x + 22][c04].mean() for y, x in stars])
        B = np.array([S[y - 21:y + 22, x - 21:x + 22][a815].mean() for y, x in stars])
        return A, B
    ship = np.load(SHIPPED_STARS / "starless.npy").astype(np.float32).mean(0)
    sel = (prof(ship)[1] - bk.mean()) / unit > 1.0
    res = {}
    for k, S in (("shipped", ship), (net, np.load(R.done_dir(net, "star_removal") / "starless.npy").astype(np.float32).mean(0))):
        A, B = prof(S)
        pct = (100 * (A - B) / B)[sel]
        res[k] = {"median_pct": round(float(np.median(pct)), 2), "share_below_3pct": round(float(100 * (pct < -3).mean()), 1),
                  "p10_pct": round(float(np.percentile(pct, 10)), 2), "stars": int(sel.sum())}
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args(argv)
    out = Path(a.out)
    work = out / "calcul"
    work.mkdir(parents=True, exist_ok=True)
    R.BENCH = work
    R.VIEW = out
    R.Cfg.scratch_dir = str(work / "scratch")
    R.WEIGHTS["melange"] = Path(a.weights)
    shutil.copy(R.SRC / "project.astroproj", work / "project.astroproj")
    R.cmd_stars(["melange"], a.device)
    R.cmd_post(["melange"], a.device)
    src = work / "export" / "melange" / "rosette_reseau_melange.jpg"
    assert src.exists(), src
    shutil.copy(src, out / "rosette_melange_50_50.jpg")
    crops(read(src), out, "rosette_melange_50_50_trous_x2.png", "rosette_melange_50_50_zoom.jpg")
    crops(read(SHIPPED_JPG), out, "rosette_reseau_actuel_trous_x2.png", "rosette_reseau_actuel_zoom.jpg")
    m = holes_measure("melange")
    (out / "mesure_trous.json").write_text(json.dumps(m, indent=1), encoding="utf-8")
    print("holes", m, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
