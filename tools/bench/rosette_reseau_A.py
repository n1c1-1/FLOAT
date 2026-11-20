"""The owner's Rosette with the star network A next to the shipped one (issue #178): the dark holes around stars.

    python tools/bench/rosette_reseau_A.py stars  [--nets livre,A]     # the Stars block, then Synthetic stars, per network
    python tools/bench/rosette_reseau_A.py post   [--nets livre,A]     # Post with the project's saved composition, exported
    python tools/bench/rosette_reseau_A.py mesure                      # the ring around bright stars on nebula

The project's upstream results (the stack, the Gradient, the Deconvolution network, the Denoise, the Color, the Output)
are read in place from the owner's store, never written; every new result goes into the bench folder. The network A is
loaded into the Stars block in this process only (``star_removal.MODEL`` pointed at its weights): nothing of the
application changes.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

SRC = Path("E:/astro_float/Rosette")                 # read only
BENCH = Path("F:/astro_essais/rosette_reseau_A")
VIEW = Path("F:/astro_essais/A_REGARDER/rosette_reseau_A")
WEIGHTS = {"livre": None, "A": Path("F:/astro_essais/stars_train/bank_A/best.pt")}
WF = 3                                               # "Rosette 3 reseau": the Deconvolution network, then Denoise...
#: the results of that workflow's chain in the owner's store (their manifests name each other's keys)
UP = {"stack": "4297845b9624d4a1", "gradient": "563940c4ab8403e7", "deconv": "c70fae4c45274544",
      "denoise": "db2924c4be0898dd", "color": "66be55c35ae37c6e", "render_simple": "7d97dd93f1651764",
      "astrometry": "13b53cf20dc59b4a", "catalogs": "6fbe0e756da5b545"}


def upstream(stage: str) -> Path:
    return SRC / "store" / stage / UP[stage]


def done_dir(net: str, stage: str) -> Path:
    return BENCH / "etoiles" / net / stage


class Cfg:
    scratch_dir = str(BENCH / "scratch")
    target = None


class BenchProject:
    """What a stage's context asks of a project: where each stage's folder is (the owner's results for the blocks before,
    the bench's for those it makes), a folder for its own result, a place for scratch and the stop file."""

    def __init__(self, net: str, own_stage: str):
        self.net, self.own_stage = net, own_stage
        self.config = Cfg()
        self.logs = BENCH / "journaux"
        self.logs.mkdir(parents=True, exist_ok=True)
        self.scratch = BENCH / "scratch"

    def own_dir(self) -> Path:
        return done_dir(self.net, self.own_stage)

    def stage_dir(self, stage: str) -> Path:
        if stage in ("star_removal", "synth_stars"):
            return done_dir(self.net, stage)
        if stage in UP:
            return upstream(stage)
        return BENCH / "_absent" / stage              # never created: the stage is not there


def run_stage(net: str, stage_cls, params, device: str) -> Path:
    from astro_gpu.core.journal import Journal
    from astro_gpu.core.stage import Context
    proj = BenchProject(net, stage_cls.name)
    ctx = Context(proj, stage_cls(), Journal(proj.logs / net, echo=True), device)
    stage_cls().run(ctx, params)
    (proj.own_dir() / "manifest.json").write_text(json.dumps({"stage": stage_cls.name, "status": "done", "net": net,
                                                              "metrics": ctx.metrics}), encoding="utf-8")
    return proj.own_dir()


def cmd_stars(nets: list[str], device: str) -> int:
    from astro_gpu.stages import star_removal as sr
    from astro_gpu.stages.synth_stars import SynthStars, SynthStarsParams
    shipped = sr.MODEL
    for net in nets:
        sr.MODEL = WEIGHTS[net] or shipped             # this process only: A's weights, never the shipped file
        print(f"=== {net}: weights {sr.MODEL}", flush=True)
        t0 = time.time()
        d = run_stage(net, sr.StarRemoval, sr.StarRemovalParams(), device)
        print(f"Stars block done in {time.time() - t0:.0f} s -> {d}", flush=True)
        prep = json.loads((d / "prep_stars.json").read_text())
        print("unit", prep["unit"], "background", prep["background"], flush=True)
        t0 = time.time()
        d = run_stage(net, SynthStars, SynthStarsParams(), device)      # the project's setting: measured, defaults
        print(f"Synthetic stars done in {time.time() - t0:.0f} s -> {d}", flush=True)
    return 0


def cmd_post(nets: list[str], device: str) -> int:
    from astro_gpu import api
    from astro_gpu.client import postengine as pe
    from astro_gpu.client.services import render as rsv
    from astro_gpu.render.layers import LayerService
    VIEW.mkdir(parents=True, exist_ok=True)
    for net in nets:
        p = api.open_project(BENCH)                     # the copy of project.astroproj: its store is the bench's, empty
        p.scratch_dir = str(BENCH / "scratch")
        version = next(v for v in p.post["versions"] if v["id"] == p.post["active"])
        layers = pe.load_layers(version["layers"], {w.id for w in p.workflows})
        print(f"{net}: version {version['name']!r}, {len(layers)} layers, crop {version.get('crop')}", flush=True)

        def resolve(sid, _net=net):
            from astro_gpu.client.services.render import Folder
            if sid == WF:
                return Folder(upstream("render_simple"))
            if isinstance(sid, tuple) and sid[1] == WF:
                return Folder(done_dir(_net, "synth_stars" if sid[0] == "synth" else "star_removal"))
            return None                                 # the other workflows' layers are hidden in this version
        src = rsv.LayerSource(resolve, LayerService(device), rsv.bases_of(p))
        r = rsv.Renderer(src, device=device)
        out = BENCH / "export" / net
        out.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        crop = tuple(version["crop"]) if version.get("crop") else None
        name = "rosette_reseau_" + ("livre" if net == "livre" else net)
        files = rsv.run_export(r, rsv.ExportJob(layers=layers, folder=str(out), name=name, crop=crop,
                                                format=["tiff16", "jpeg"], wcs=False, project_name=p.name),
                               lambda f, m: None)
        print(f"{net}: exported in {time.time() - t0:.0f} s on {device}: {[Path(f).name for f in files]}", flush=True)
    return 0


# ---------------------------------------------------------------------------------------------- the measure
def isolated_peaks(L, lo, n, win=41, crowd=150):
    """Up to n isolated stars: local maxima of a luminance image (noise sigmas) above ``lo``, brightest first, whose
    neighbourhood (win x win) has few pixels above a tenth of the peak."""
    from scipy import ndimage
    mx = ndimage.maximum_filter(L, 31)
    ys, xs = np.nonzero((L == mx) & (L > lo))
    ok = (ys > 100) & (ys < L.shape[0] - 100) & (xs > 100) & (xs < L.shape[1] - 100)
    ys, xs = ys[ok], xs[ok]
    out = []
    for i in np.argsort(-L[ys, xs]):
        y, x = ys[i], xs[i]
        c = L[y - win // 2:y + win // 2 + 1, x - win // 2:x + win // 2 + 1]
        if (c > 0.1 * L[y, x]).sum() < crowd:
            out.append((int(y), int(x)))
            if len(out) >= n:
                break
    return out


def ring(r0, r1, R=21):
    yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
    r = np.hypot(xx, yy)
    return (r >= r0) & (r < r1)


NEB = 1.0


def cmd_mesure() -> int:
    """The starless level 3-6 px around bright stars on nebula, against the level 10-20 px out (the median over the stars of
    100 x (inner - outer) / outer, in the image's linear units, sky included). Same stars for both networks (found in the
    Output image, the one both started from). 'On nebula': the outer ring more than NEB noise units over the sky (the Rosette's faint emission is 1 to 4 of them)."""
    prep = json.loads((done_dir("livre", "star_removal") / "prep_stars.json").read_text())
    unit, bk = prep["unit"], np.array(prep["background"], np.float32)[:, None, None]
    lin = np.load(upstream("render_simple") / "lin.npy").astype(np.float32)
    L = ((lin - bk) / unit).mean(0)
    stars = isolated_peaks(L, 100.0, 400)                # bright: over 100 noise units
    m36, m1020 = ring(3, 6), ring(10, 20)
    res = {"unit": unit, "stars_found": len(stars)}
    sel = None
    for net in ("livre", "A"):
        S = np.load(done_dir(net, "star_removal") / "starless.npy").astype(np.float32).mean(0)
        a = np.array([S[y - 21:y + 22, x - 21:x + 22][m36].mean() for y, x in stars])
        b = np.array([S[y - 21:y + 22, x - 21:x + 22][m1020].mean() for y, x in stars])
        if sel is None:
            sel = (b - bk.mean()) / unit > NEB
        pct = 100 * (a - b) / b
        res[net] = {"on_nebula": int(sel.sum()), "median_pct": round(float(np.median(pct[sel])), 3),
                    "mean_pct": round(float(np.mean(pct[sel])), 3),
                    "p10_pct": round(float(np.percentile(pct[sel], 10)), 3),
                    "median_sigma": round(float(np.median((a - b)[sel] / unit)), 3),
                    "all_median_pct": round(float(np.median(pct)), 3)}
        for th in (0.5, 1, 2, 4):
            k = (b - bk.mean()) / unit > th
            res[net][f"median_pct_above_{th}"] = [int(k.sum()), round(float(np.median(pct[k])), 3) if k.any() else None]
        print(net, res[net], flush=True)
    res["stars"] = [[int(v) for v in s] for s in np.array(stars)[sel]][:50]
    (BENCH / "mesures.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["stars", "post", "mesure"])
    ap.add_argument("--nets", default="livre,A")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args(argv)
    nets = a.nets.split(",")
    if a.cmd == "stars":
        return cmd_stars(nets, a.device)
    if a.cmd == "post":
        return cmd_post(nets, a.device)
    return cmd_mesure()


if __name__ == "__main__":
    sys.exit(main())
