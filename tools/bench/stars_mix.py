"""Star removal, issue #178: linear mixing of the weights of star-removal networks that share the chain's NAFNet
(w = (1 - a) W1 + a W2, as in weight interpolation between fine-tuned or independently trained models), judged with the
same judges as the pure networks: core share of M27 in the stars layer, nebula of NGC 6543 without its central star in the
stars layer, the Rosette hole around stars (11 November measure). No training. Every output goes in --out.

    python tools/bench/stars_mix.py measure --out DIR [--pairs shipped:A,shipped:E3,A:E3] [--alphas 0.25,0.5,0.75]
    python tools/bench/stars_mix.py export  --out DIR --nets shipped+A@0.5,...        # the crops of chosen mixes

Device: STARS_MIX_DEVICE (cpu by default). The weights: shipped (the bundle, read only), A, E3 (paths below, read only).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stars_e3 as e3  # noqa: E402  (its redirect: path-safe writer, outputs in --out)
import stars_judge as sj  # noqa: E402
from astro_gpu.stages import star_removal as sr  # noqa: E402

SRC = {"shipped": sr.MODEL, "A": Path("F:/astro_essais/stars_train/bank_A/best.pt"),
       "E3": Path("F:/astro_essais/starRemoval/train/E3/snap/step_20000.pt")}
DEV = torch.device(os.environ.get("STARS_MIX_DEVICE", "cpu"))
_STATE: dict = {}
REF: dict = {}                      # the shipped network's check values, set by its own measure


def state(name):
    if name not in _STATE:
        d = torch.load(SRC[name], map_location="cpu", weights_only=False)["net"]
        _STATE[name] = {k: v.float() for k, v in d.items()}
    return _STATE[name]


def mixed_state(spec: str):
    """'A' or 'shipped+E3@0.25' = 0.75 shipped + 0.25 E3."""
    if "+" not in spec:
        return state(spec)
    pair, a = spec.split("@")
    n1, n2 = pair.split("+")
    a = float(a)
    s1, s2 = state(n1), state(n2)
    assert s1.keys() == s2.keys()
    return {k: ((1 - a) * s1[k] + a * s2[k]) for k in s1}


def make_net(spec: str):
    net = sr.NAFNet().to(DEV)
    net.load_state_dict(mixed_state(spec))        # strict: same keys; LayerNorm only, no running statistics to mix
    return net.eval()


def sanity_buffers():
    net = sr.NAFNet()
    return [n for n, _ in net.named_buffers()]


class Cases:
    """The judges' inputs, built once."""

    def __init__(self):
        self.m27 = sj.build("m27", DEV)
        self.ngc = sj.build("ngc6543", DEV)
        O, T = self.ngc.O, self.ngc.target
        layer = (O - T).mean(0)
        stars = ndimage.binary_dilation(layer > 10, iterations=5)      # stars_nebula's zone, unchanged
        blur = ndimage.uniform_filter(T.mean(0), 15, mode="nearest")
        self.zone = (blur > 10) & ~stars
        self.ros = sj.build("rosette", DEV)
        import rosette_reseau_A as R
        prep = json.loads((R.done_dir("livre", "star_removal") / "prep_stars.json").read_text())
        self.unit, self.bk = prep["unit"], np.array(prep["background"], np.float32)[:, None, None]
        lin = np.load(R.upstream("render_simple") / "lin.npy").astype(np.float32)
        self.Oh = (lin - self.bk) / self.unit
        self.hstars = R.isolated_peaks(self.Oh.mean(0), 100.0, 400)
        self.c04, self.a815 = R.ring(0, 4.0001), R.ring(8, 15.0001)
        livre = np.load(R.done_dir("livre", "star_removal") / "starless.npy").astype(np.float32).mean(0)
        ann = np.array([livre[y - 21:y + 22, x - 21:x + 22][self.a815].mean() for y, x in self.hstars])
        self.sel = (ann - self.bk.mean()) / self.unit > 1.0

    def holes(self, S):
        c, a = self.c04, self.a815
        A = np.array([S[y - 21:y + 22, x - 21:x + 22][c].mean() for y, x in self.hstars])
        B = np.array([S[y - 21:y + 22, x - 21:x + 22][a].mean() for y, x in self.hstars])
        pct = (100 * (A - B) / B)[self.sel]
        return {"median_pct": round(float(np.median(pct)), 2), "share_below_3pct": round(float(100 * (pct < -3).mean()), 1),
                "p10_pct": round(float(np.percentile(pct, 10)), 2)}


def measure_one(C: Cases, spec: str) -> dict:
    net = make_net(spec)
    r = {"spec": spec}
    # NGC 6543 first: it carries the breakage check (the whole image is compared with the input)
    j = C.ngc
    S = sr.apply(net, j.O, DEV)
    fin = bool(np.isfinite(S).all())
    d = (S - j.O).mean(0) if fin else None
    r["finite"] = fin
    if fin:
        r["median_abs_change_sigma"] = round(float(np.median(np.abs(d))), 3)     # the sky must stay as it was
        r["mean_change_sigma"] = round(float(d.mean()), 3)
        r["frac_changed_over_1sigma"] = round(float((np.abs(d) > 1).mean()), 4)
    # broken: not finite, or the sky changed more than twice as much as with the shipped network (the reference of the
    # same call), or the mean level moved by more than twice the shipped one's (stars taken off) or went up (light added)
    ref = REF.get("ngc") or {"median_abs_change_sigma": 0.0, "mean_change_sigma": 0.0}
    broken = (not fin) or (bool(REF) and (r["median_abs_change_sigma"] > 2 * ref["median_abs_change_sigma"] or
                                          r["mean_change_sigma"] < 2 * ref["mean_change_sigma"] or r["mean_change_sigma"] > 1.0))
    r["broken"] = bool(broken)
    if spec == "shipped":
        REF["ngc"] = r
    if fin:
        L = j.O.mean(0)
        r["ngc6543"] = {**j.measure(S), "nebula_in_stars_pct": round(100 * float((L - S.mean(0))[C.zone].sum() / L[C.zone].sum()), 3)}
    if broken:
        del net
        return r
    j = C.m27
    S = sr.apply(net, j.O, DEV)
    S[:, ~j.valid] = j.O[:, ~j.valid]
    r["m27"] = {"core_share_in_stars": j.measure(S)["core_share_in_stars"]}
    S = sr.apply(net, C.Oh, DEV) * C.unit + C.bk
    r["rosette_holes"] = C.holes(S.mean(0))
    del net
    if DEV.type == "cuda":
        torch.cuda.empty_cache()
    return r


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("measure", "export"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--pairs", default="shipped:A,shipped:E3,A:E3")
    ap.add_argument("--alphas", default="0.25,0.5,0.75")
    ap.add_argument("--nets", default="")
    a = ap.parse_args(argv)
    out = Path(a.out)
    e3.redirect(out)
    sj.OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    if a.cmd == "export":
        names = a.nets.split(",")
        sj.get_net = lambda n, dev: make_net(n)
        sj.judge(names, DEV.type, ["ngc6543", "rosette"])
        return 0
    assert not sanity_buffers(), sanity_buffers()
    C = Cases()
    f = sj.OUT / "mix_results.json"
    res = json.loads(f.read_text()) if f.exists() else {}
    specs = ["shipped"]                  # first: the sanity check and the breakage reference
    for p in a.pairs.split(","):
        n1, n2 = p.split(":")
        specs += [f"{n1}+{n2}@{al}" for al in a.alphas.split(",")]
    wdir = out / "poids"
    wdir.mkdir(exist_ok=True)
    for s in specs:
        if s in res:
            continue
        res[s] = measure_one(C, s)
        if "+" in s:
            torch.save({"net": mixed_state(s)}, wdir / (s.replace("+", "_").replace("@", "_a") + ".pt"))
        f.write_text(json.dumps(res, indent=1))
        sj.log(f"{s}: {res[s]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
