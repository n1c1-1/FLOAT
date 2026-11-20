"""Star removal, issue #178: the judges of stars_judge.py run on one more network, E3 (trained from synthetic stars, the chain's
own NAFNet and stretch, applied by the chain's tiled apply), with every output in one folder given by --out.

    python tools/bench/stars_e3.py judge  --out DIR [--only m27,ngc6543,rosette]   # core share, the images to look at
    python tools/bench/stars_e3.py nebula --out DIR                                # nebula alone, against the shipped network
    python tools/bench/stars_e3.py holes  --out DIR                                # the 11 November hole measure
    python tools/bench/stars_e3.py time   --out DIR                                # a 12 Mpx image on the processor

E3's weights: E3_WEIGHTS (default: the snapshot chosen on 9 November). Nothing is written outside --out: stars_judge's
output folders are redirected to it before any run. The owners' projects and the flow results are read only.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stars_judge as sj  # noqa: E402
from astro_gpu.stages import star_removal as sr  # noqa: E402

E3_WEIGHTS = Path(os.environ.get("E3_WEIGHTS", "F:/astro_essais/starRemoval/train/E3/snap/step_20000.pt"))
_shipped_get_net = sj.get_net


def get_net(name: str, dev):
    if name != "E3":
        return _shipped_get_net(name, dev)
    net = sr.NAFNet().to(dev)                     # the chain's definition, as sr.network builds it
    state = torch.load(E3_WEIGHTS, map_location=dev, weights_only=False)["net"]
    net.load_state_dict({k: v.float() for k, v in state.items()})
    return net.eval()


def _imwrite(path, img, params=None):
    """cv2.imwrite fails silently on a path with accents: encode, then write the bytes."""
    ok, buf = cv2.imencode(Path(path).suffix, img, params or [])
    Path(path).write_bytes(buf.tobytes())
    return ok


def redirect(out: Path) -> None:
    cv2.imwrite = _imwrite
    out.mkdir(parents=True, exist_ok=True)
    sj.OUT, sj.VIEW, sj.SN_TMP = out / "mesures", out / "images", out / "tmp"
    sj.get_net = get_net
    torch.set_num_threads(8)


def holes(out: Path) -> None:
    """rosette_trous_ref.py's measure for the flow's input: stars bright on nebula, 0-4 px against 8-15 px."""
    import rosette_reseau_A as R
    prep = json.loads((R.done_dir("livre", "star_removal") / "prep_stars.json").read_text())
    unit, bk = prep["unit"], np.array(prep["background"], np.float32)[:, None, None]
    lin = np.load(R.upstream("render_simple") / "lin.npy").astype(np.float32)
    O = (lin - bk) / unit
    stars = R.isolated_peaks(O.mean(0), 100.0, 400)
    c04, a815 = R.ring(0, 4.0001), R.ring(8, 15.0001)
    dev = torch.device("cpu")
    t = time.time()
    E3 = sr.apply(get_net("E3", dev), O, dev) * unit + bk
    sj.log(f"E3 on the flow input: {time.time() - t:.0f} s")
    imgs = {"entree": lin.mean(0), "livre": np.load(R.done_dir("livre", "star_removal") / "starless.npy").astype(np.float32).mean(0),
            "E3": E3.mean(0)}
    ann = np.array([imgs["livre"][y - 21:y + 22, x - 21:x + 22][a815].mean() for y, x in stars])
    sel = (ann - bk.mean()) / unit > 1.0
    res = {"stars_bright": len(stars), "on_nebula": int(sel.sum()), "unit": unit}
    for k, S in imgs.items():
        a = np.array([S[y - 21:y + 22, x - 21:x + 22][c04].mean() for y, x in stars])
        b = np.array([S[y - 21:y + 22, x - 21:x + 22][a815].mean() for y, x in stars])
        pct = (100 * (a - b) / b)[sel]
        res[k] = {"median_pct": round(float(np.median(pct)), 2), "share_below_3pct": round(float(100 * (pct < -3).mean()), 1),
                  "p10_pct": round(float(np.percentile(pct, 10)), 2), "median_sigma": round(float(np.median(((a - b) / unit)[sel])), 3)}
        sj.log(f"{k}: {res[k]}")
    (sj.OUT / "rosette_trous_E3.json").write_text(json.dumps(res, indent=1))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=("judge", "nebula", "holes", "time"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default="m27,ngc6543,rosette")
    a = ap.parse_args(argv)
    out = Path(a.out)
    redirect(out)
    sj.OUT.mkdir(parents=True, exist_ok=True)
    if a.cmd == "judge":
        return sj.judge(["E3"], "cpu", a.only.split(","))
    if a.cmd == "time":
        return sj.timing(["shipped", "E3"], "cpu", None)
    if a.cmd == "nebula":
        import stars_nebula as sn
        sn.NETS = ("shipped", "E3")
        sn.main("cpu")
        return 0
    holes(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
