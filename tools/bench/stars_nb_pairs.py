"""The owner's reviewed star-removal pairs (livraison of 19 Nov 2026) as training data for the star network, in the
format of stars_train.py's pairs (stars_dataset_v2): <pair>/{input.npy, starless_fixed.npy (float16), ignore.npy,
meta.json} and a fields.json. Bench code: nothing is trained here.

    python tools/bench/stars_nb_pairs.py check24      # stars_train.fixes() on the delivered review TIFFs = the stored files?
    python tools/bench/stars_nb_pairs.py assembled    # a_verifier: original.npy -> cible.npy, as validated
    python tools/bench/stars_nb_pairs.py retouched    # a_retoucher: the owner's retouched TIFFs read back
    python tools/bench/stars_nb_pairs.py recover      # m97, ngc6543: the lost starting image recomputed, the shift undone

Writes only in F:/astro_essais/stars_dataset/paires_nb/ (see its LISEZMOI.txt); reads the delivery and the working
folders in place.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "bench"))

import stars_train as st  # noqa: E402

LIV = Path("F:/astro_essais/stars_dataset/livraison")
OUT = Path("F:/astro_essais/stars_dataset/paires_nb")
PN, COMBI = Path("F:/astro_essais/stars_dataset/_revue_pn"), Path("F:/astro_essais/stars_dataset/_revue_combi")


def log(msg):
    st.log(msg)


def check24() -> int:
    import tifffile
    stj = json.loads((st.REVIEW / "stretch.json").read_text())
    old = json.loads((st.DS / "fields.json").read_text())
    bad = 0
    for fld, s in sorted(stj.items()):
        if fld.startswith(st.NOT_REVIEWED):
            log(f"{fld}: not reviewed, left out (as fixes())")
            continue
        d = st.DS / s["pair"]
        sl = np.load(d / "starless.npy").astype(np.float32)
        fixed, changed = st.readback(sl, tifffile.imread(st.REVIEW / f"{fld}_starless.tif"), s)
        ex = np.load(d / "starless_fixed.npy")
        mine = fixed.astype(np.float16)
        same = bool(np.array_equal(ex, mine, equal_nan=True))
        mx = float(np.abs(ex.astype(np.float32) - mine.astype(np.float32)).max())
        o = old[fld]
        log(f"{fld}: identical={same} max|diff|={mx:.3g} changed {int(changed.sum())} (stored {o['changed_px']})")
        bad += (not same) or int(changed.sum()) != o["changed_px"]
    log(f"check24: {'ALL IDENTICAL' if not bad else str(bad) + ' DIFFER'}")
    return int(bool(bad))


def read_tif(path: Path) -> np.ndarray:
    """The first page only (an editor's layers follow it), (H, W, 3) uint16."""
    import tifffile
    t = tifffile.imread(path, key=0)
    if t.ndim != 3 or t.shape[2] < 3 or t.dtype != np.uint16:
        raise SystemExit(f"{path}: {t.shape} {t.dtype}, a 16 bit RGB image was expected")
    return t[:, :, :3]


def stretch_of(s: dict) -> float:
    return float(np.arcsinh((s["top"] + s["offset"]) / s["soft"]))


def stretch(v: np.ndarray, s: dict) -> np.ndarray:
    """Noise sigmas -> codes (float, rounded by the caller)."""
    return np.clip(np.arcsinh(np.clip(v + s["offset"], 0, None) / s["soft"]) / stretch_of(s), 0, 1) * 65535


def unstretch(codes: np.ndarray, s: dict) -> np.ndarray:
    return (s["soft"] * np.sinh(codes.astype(np.float32) / 65535 * stretch_of(s)) - s["offset"]).astype(np.float32)


def save_pair(name: str, inp: np.ndarray, tgt: np.ndarray, top: float, changed_px: int, extra: dict,
              held_out: bool = False, judge: bool = False) -> dict:
    """OUT/<name>/{input.npy (float32: m97's stars reach 2.8e6 sigmas, past float16), starless_fixed.npy (float16 when it
    holds, else float32), ignore.npy, meta.json}; the pair's fields.json entry (FWHM of its isolated stars)."""
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "input.npy", inp.astype(np.float32))
    tgt16 = tgt.astype(np.float16)
    np.save(d / "starless_fixed.npy", tgt16 if np.isfinite(tgt16).all() else tgt.astype(np.float32))
    H, W = inp.shape[1:]
    np.save(d / "ignore.npy", np.zeros((H, W), bool))
    fwhm, nfw = st.star_fwhm(inp, tgt, top)
    star_px = int(((inp - tgt).max(0) > 3.0).sum())            # the star layer: 3 noise sigmas above the target
    extra = {**extra, "star_layer_px": star_px}
    (d / "meta.json").write_text(json.dumps({"name": name, "field": name, "shape": list(inp.shape), "top": top,
                                             **extra}, indent=1), encoding="utf-8")
    fj = OUT / "fields.json"
    fields = json.loads(fj.read_text()) if fj.exists() else {}
    fields[name] = {"pair": name, "changed_px": int(changed_px), "ignored_px": 0, "fwhm_px": round(fwhm, 2),
                    "fwhm_stars": nfw, "top": top, "star_layer_px": star_px, **({"held_out": True} if held_out else {}),
                    **({"judge": True} if judge else {})}
    fj.write_text(json.dumps(fields, indent=1), encoding="utf-8")
    log(f"{name}: input {extra.get('input_source')} | target {extra.get('target_source')} | changed {changed_px} px "
        f"({100 * changed_px / (H * W):.2f} %), star layer {100 * star_px / (H * W):.2f} % | touched "
        f"{extra.get('tif_touched_px')} | FWHM {fwhm:.2f} px ({nfw} stars)")
    return fields[name]


ASSEMBLED = ("57_sho", "57_rgb", "58_hoo", "58_rgb", "62_ho", "26_rgb")


def assembled() -> int:
    """a_verifier: the owner validated the automatic targets as they are: input original.npy, target cible.npy (full
    precision). Checks that the delivered cible.tif is the stretched cible.npy (no retouch behind it)."""
    stj = json.loads((LIV / "a_evaluer/a_verifier/stretch.json").read_text())
    for n in ASSEMBLED:
        d = COMBI / n
        m = json.loads((d / "meta.json").read_text())
        s = stj[n]
        inp = np.load(d / "original.npy").astype(np.float32)
        tgt = np.load(d / "cible.npy").astype(np.float32)
        rs = np.load(d / "starless_reseau.npy").astype(np.float32)
        tif = read_tif(LIV / f"a_evaluer/a_verifier/{n}_cible.tif").astype(np.float32)
        code = np.round(stretch(tgt, s)).transpose(1, 2, 0)
        touched = int((np.abs(tif - code).max(2) > 4).sum())
        vs_net = int((np.abs(stretch(tgt, s) - stretch(rs, s)).max(0) > 4).sum())
        save_pair(n, inp, tgt, float(s["top"]), vs_net,
                  {"kind": "assembled", "input_source": "original.npy (assembled image with its stars)",
                   "target_source": "cible.npy (per-filter star removal, fused; validated as is)",
                   "tif_touched_px": touched, "changed_means": "target vs the network's starless, > 4 codes",
                   "unit": m["unit"], "background": m["background"], "offset": s["offset"], "soft": s["soft"]})
    return 0


#: a_retoucher: pair -> (kind, folder, input file, starting-image file), the file name of the owner's TIFF says the start
#: "starless": our network's starless image; "denoise_starnet": the reference star tool on FLOAT's Denoise of the input;
#: "starnet": the star tool on the original
RETOUCHED = {
    "34": ("starless", COMBI / "34", "original.npy", "starless_reseau.npy"),
    "64": ("starless", COMBI / "64", "original.npy", "starless_reseau.npy"),
    "m1": ("starless", PN / "m1", "input.npy", "starless.npy"),
    "m82": ("starless", PN / "m82", "input.npy", "starless.npy"),
    "m57": ("starless", PN / "m57", "input.npy", "starless.npy"),
    "m13": ("denoise_starnet", PN / "m13", "input.npy", None),
    "ngc6888": ("denoise_starnet", PN / "ngc6888", "input.npy", None),
    "m97": ("denoise_starnet", PN / "m97", "input.npy", None),
    "ngc6543": ("denoise_starnet", PN / "ngc6543", "input.npy", None),
    "ngc7662": ("starnet", PN / "ngc7662", "input.npy", None),
}
HELD = {"ngc6543": "judge"}          # the owner's plan: M27, the NGC 7000 dual-band field and NGC 6543 stay out of training
FLOAT_ROOT = os.environ.get("FLOAT_ROOT")      # the checkout whose Denoise made the delivered _denoise.tif files
PY = sys.executable
SCRATCH = Path(os.environ.get("NB_SCRATCH", "F:/astro_essais/stars_dataset/paires_nb/_denoise"))


def denoised_input(name: str, d: Path, fn: str) -> np.ndarray:
    """FLOAT's Denoise of the pair's input, float32 (step of the delivery's _denoise.tif, recomputed without the
    16 bit rounding); cached in SCRATCH/<name>.npy."""
    SCRATCH.mkdir(parents=True, exist_ok=True)
    out = SCRATCH / f"{name}.npy"
    if not out.exists():
        env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=os.environ.get("NB_CUDA", "0"))
        if FLOAT_ROOT:
            env["FLOAT_ROOT"] = FLOAT_ROOT
        r = subprocess.run([PY, str(Path(__file__).with_name("stars_nb_denoise.py")), str(d / fn),
                            str(d / "meta.json"), str(out)], env=env)
        if r.returncode:
            raise SystemExit(f"denoise of {name} failed")
    return np.load(out)


REPORT: dict = {}
STARTOOL_PY = os.environ.get("STARTOOL_PY")     # a python with onnxruntime; STARTOOL_WEIGHTS: the weights file
START = SCRATCH / "_startool_recalcul"


def recomputed_start(name: str) -> np.ndarray:
    """The star tool's output on the delivery's <name>_denoise.tif (the owner's starting image), (3, H, W) codes,
    recomputed with the delivery's own recipe into START (the owner's files are never written)."""
    import tifffile
    out = START / f"{name}_denoise_starnet.tif"
    if not out.exists():
        START.mkdir(parents=True, exist_ok=True)
        r = subprocess.run([STARTOOL_PY, str(Path(__file__).with_name("stars_nb_startool.py")),
                            str(LIV / f"a_evaluer/a_retoucher/done/{name}_denoise.tif"), str(out)],
                           env=dict(os.environ, CUDA_VISIBLE_DEVICES="-1"))
        if r.returncode:
            raise SystemExit(f"star tool on {name} failed")
    return tifffile.imread(out).transpose(2, 0, 1).astype(np.float32)


def fit_map(R: np.ndarray, C: np.ndarray, nb: int = 400, rounds: int = 4):
    """Per channel, a monotone map g with R ~ g(C): the medians of R in bins of C (quantile bins), made monotone, the
    pixels farther than 4 codes or 4 robust sigmas from it (the owner's local retouches) left out and the map refit.
    Returns [(x, g(x))] per channel, the inlier mask."""
    maps, inl = [], np.ones(R.shape[1:], bool)
    for c in range(3):
        x, y = C[c].ravel(), R[c].ravel()
        ok = np.ones(x.size, bool)
        for _ in range(rounds):
            q = np.unique(np.quantile(x[ok], np.linspace(0, 1, nb + 1)))
            b = np.clip(np.searchsorted(q, x, side="right") - 1, 0, len(q) - 2)
            gx, gy = [], []
            for k in range(len(q) - 1):
                m = (b == k) & ok
                if m.sum() >= 20:
                    gx.append(np.median(x[m]))
                    gy.append(np.median(y[m]))
            gx, gy = np.array(gx), np.maximum.accumulate(np.array(gy))
            res = y - np.interp(x, gx, gy)
            sig = 1.4826 * np.median(np.abs(res[ok] - np.median(res[ok])))
            ok = np.abs(res) <= max(4.0, 4 * sig)
        maps.append((gx, gy))
        inl &= ok.reshape(inl.shape)
    return maps, inl


def recover(name: str, codes: np.ndarray, s: dict):
    """The owner's retouched TIFF ``codes`` (3, H, W) against the star tool's recomputed start: method check (the start
    recomputed with the delivery's recipe, bit for bit where the owner did not retouch), a monotone per-channel map
    ``g`` (R ~ g(start)) on the untouched pixels, its test against known curves, and the target: the TIFF with g
    inverted when g departs from identity by more than 2 codes (else untouched: nothing to undo). Returns (target in
    sigmas, share-defining pixel count retouched, report)."""
    C = recomputed_start(name)
    R = codes.astype(np.float32)
    d = R - C
    same = float((d == 0).all(0).mean())
    maps, inl = fit_map(R, C)
    dev = [float(np.max(np.abs(gy - gx))) for gx, gy in maps]
    res = [float(np.std((R[c] - np.interp(C[c], *maps[c]))[inl])) for c in range(3)]
    xs = np.linspace(0, 1, 2001)
    ident = [float(np.sqrt(np.mean((np.interp(gx, gx, gy) - gx) ** 2))) for gx, gy in maps]
    g22 = [float(np.sqrt(np.mean((65535 * (gx / 65535) ** 2.2 - gy) ** 2))) for gx, gy in maps]
    srgb = [float(np.sqrt(np.mean((65535 * np.where(gx / 65535 <= 0.04045, gx / 65535 / 12.92,
                                                     ((gx / 65535 + 0.055) / 1.055) ** 2.4) - gy) ** 2))) for gx, gy in maps]
    # the editor stores 16 bit as 15 bit + 1: an untouched value is the start rounded to 32769 levels
    q = np.round(np.round(C / 65535 * 32768) / 32768 * 65535)
    exact15 = float((R == q).all(0).mean())
    retouched_px = int((np.abs(d).max(0) > 4).sum())
    rep = {"identical_px_share": round(same, 4), "identical_after_15bit_share": round(exact15, 4), "retouched_px": retouched_px,
           "retouched_share": round(retouched_px / d[0].size, 4), "map_max_dev_codes": [round(v, 1) for v in dev],
           "fit_residual_sd_codes": [round(v, 2) for v in res], "rms_vs_identity": [round(v, 1) for v in ident],
           "rms_vs_gamma2.2": [round(v, 0) for v in g22], "rms_vs_srgb_decode": [round(v, 0) for v in srgb],
           "map": "identity" if max(dev) <= 2 else "inverted"}
    if max(dev) > 2:
        R = np.stack([np.interp(R[c], maps[c][1], maps[c][0], left=None) for c in range(3)])   # g^-1 on the TIFF
        # keep the owner's retouch: where R differed from the start by more than the map predicts, R inverted stands
    return unstretch(R, s), retouched_px, rep


def retouched(only=None) -> int:
    stj = json.loads((LIV / "a_evaluer/a_retoucher/stretch.json").read_text())
    for name, (kind, d, fin, fstart) in RETOUCHED.items():
        if only and name not in only:
            continue
        s = stj[name]
        m = json.loads((d / "meta.json").read_text())
        inp = np.load(d / fin).astype(np.float32)
        conv = {"unit": m["unit"], "background": m["background"], "offset": s["offset"], "soft": s["soft"]}
        if kind == "starless":
            tif = read_tif(LIV / f"a_evaluer/a_retoucher/starless/{name}_starless.tif")
            sl = np.load(d / fstart).astype(np.float32)
            tgt, changed = st.readback(sl, tif, s)
            save_pair(name, inp, tgt, float(s["top"]), int(changed.sum()),
                      {"kind": kind, "input_source": f"{fin} (the pair's input)",
                       "target_source": f"{name}_starless.tif read back over {fstart} (>4 codes take the TIFF)", **conv},
                      held_out=name in HELD, judge=name in HELD)
        else:
            f = f"{name}_denoise_starnet.tif" if kind == "denoise_starnet" else f"{name}_starnet.tif"
            tif = read_tif(LIV / f"a_evaluer/a_retoucher/starless/{f}").astype(np.float32).transpose(2, 0, 1)
            codes = tif.copy()
            tgt = unstretch(tif, s)
            changed, src_start = None, ""
            if kind == "denoise_starnet":
                inp = denoised_input(name, d, fin)
                src = f"FLOAT Denoise of {fin}, float32"
                tgt, changed, rep = recover(name, codes, s)
                src_start = f"; start recomputed, owner's retouch on {changed} px, map {rep['map']}"
                REPORT[name] = rep
            else:
                src = f"{fin} (original, linear)"
                changed = int((np.abs(stretch(tgt, s) - stretch(inp, s)).max(0) > 4).sum())
            save_pair(name, inp, tgt, float(s["top"]), changed,
                      {"kind": kind, "input_source": src,
                       "target_source": f"{f} unstretched (16 bit precision){src_start}",
                       "changed_means": "target vs the star tool's start (recomputed) or vs input, > 4 codes", **conv},
                      held_out=name in HELD, judge=name in HELD)
        if name in REPORT:
            log(f"{name}: recovery {json.dumps(REPORT[name])}")
    return 0


def checkdenoise() -> int:
    """The recomputed (float32) Denoise of each denoise_starnet pair against the delivery's _denoise.tif: codes."""
    stj = json.loads((LIV / "a_evaluer/a_retoucher/stretch.json").read_text())
    for name, (kind, d, fin, _f) in RETOUCHED.items():
        if kind != "denoise_starnet" or not (SCRATCH / f"{name}.npy").exists():
            continue
        v = np.load(SCRATCH / f"{name}.npy")
        mine = np.round(stretch(v, stj[name])).transpose(1, 2, 0)
        ref = read_tif(LIV / f"a_evaluer/a_retoucher/done/{name}_denoise.tif").astype(np.float32)
        df = np.abs(mine - ref)
        log(f"{name}: recomputed Denoise vs delivered _denoise.tif: max {df.max():.0f} codes, share > 4 codes "
            f"{(df.max(2) > 4).mean():.4f}, share exact {(df.max(2) == 0).mean():.3f}")
    return 0


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.split(". ")[0])
    ap.add_argument("cmd", choices=("check24", "assembled", "retouched", "checkdenoise"))
    ap.add_argument("--only", default="")
    a = ap.parse_args(argv)
    return {"check24": check24, "assembled": assembled, "retouched": lambda: retouched(a.only.split(",") if a.only else None),
            "checkdenoise": checkdenoise}[a.cmd]()


if __name__ == "__main__":
    sys.exit(main())
