"""Star removal, issue #178: the judges of the shipped network and the candidates A and B (stars_bank.py).

    python tools/bench/stars_judge.py judge --nets shipped,A,B [--device cuda:0] [--only m27,ngc6543,comb56,...]
    python tools/bench/stars_judge.py time  --nets shipped,A,B --device cpu|cuda:0
    python tools/bench/stars_judge.py bench --nets shipped,A,B       # the bench's starless images (then stars_compare.py)

The judges (read-only on the owner's projects; numbers in F:/astro_essais/stars_juges_AB/<net>.json, the images to look
at in F:/astro_essais/A_REGARDER/reseau_etoiles_A_B/):
- m27: the owner's M27, the share of the nebula core that goes to the stars layer (the window and the core of the
  tester's recipe: rows 1300-1600, columns 2400-2760, core = the blurred starless image above half its maximum, the core
  taken from the stored starless image so that it is the same for every network);
- ngc6543: the held-out pair: the same share, against the owner's retouched target;
- comb56: an assembled multi-filter image held out from the training (a mono SHO M27, the starless target = each
  filter's starless image, re-fused): the starless image minus the target around the brightest isolated stars, by ring;
  a disc left by the network is a positive plateau out to about 60 px;
- rim_*: the dark rim 3-4 px from the stars on Rosette and NGC 7000, without and with the Deconvolution network:
  the starless level in a 3-4 px ring over the level in a 12-18 px ring, around the unsaturated stars of each image.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "bench"))

import stars_bank as sb  # noqa: E402
import stars_train as st  # noqa: E402
from astro_gpu.stages import star_removal as sr  # noqa: E402

OUT = Path("F:/astro_essais/stars_juges_AB")
VIEW = Path("F:/astro_essais/A_REGARDER/reseau_etoiles_A_B")
M27 = Path("G:/astro/2024-12_M27/M27 FLOAT/store/star_removal/900d75d60f7e6d3b")
ROSETTE = Path("E:/astro_float/Rosette/store/star_removal")
NGC7000 = Path("E:/astro_float/NGC7000_Thomas_auto/store/star_removal")
COMB56 = Path("F:/astro_essais/stars_dataset/_revue_nb/56_comb")
log = st.log
RINGS = [(4, 8), (8, 16), (16, 32), (32, 48), (48, 64), (64, 80)]


def get_net(name: str, dev):
    if name == "shipped":
        return sr.network(dev)
    f = sb.TRAIN / f"bank_{name}" / "best.pt"
    if not f.exists():                            # the timings do not need trained weights: the architecture alone
        log(f"{name}: no trained weights, the shipped ones (and B's added level at zero): timing only")
        return sb.build(name, dev).eval()
    return sb.load_net(f, name, dev)


# ---------------------------------------------------------------------------------------------- the images
def from_store(E: Path):
    """(O in noise sigmas, valid, the stored starless image in the same units): a Stars block result of a project."""
    prep = json.loads((E / "prep_stars.json").read_text())
    sl = np.load(E / "starless.npy").astype(np.float32)
    lin = sl + np.load(E / "stars.npy").astype(np.float32)
    bk = np.array(prep["background"], np.float32)[:, None, None]
    valid = np.load(E / "valid.npy") if (E / "valid.npy").exists() else np.ones(lin.shape[1:], bool)
    from_store.pedestal = float(np.mean(bk)) / prep["unit"]      # the sky level in sigmas, for the rim's percentages
    return (lin - bk) / prep["unit"], valid, (sl - bk) / prep["unit"]


def isolated_peaks(L, lo, hi, n, win=41, crowd=150):
    """Up to n unsaturated, isolated stars of a luminance image (peak between lo and hi sigmas), brightest first."""
    mx = ndimage.maximum_filter(L, 31)
    ys, xs = np.nonzero((L == mx) & (L > lo) & (L < hi))
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


def ring_masks(r0, r1, R=None):
    R = R or int(np.ceil(r1)) + 1
    yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
    r = np.hypot(xx, yy)
    return (r >= r0) & (r < r1)


def rim_of(S, stars, ped=0.0):
    """The starless level in the 3-4 px ring against the 12-18 px ring around each star. ``ped``: the sky level the image's
    working units were counted from (sigmas): the local level is the outer ring plus ped. Returns the median % of the local
    level over all the stars (the issue's measure), the median in sigmas, and the same on the stars whose outer ring is
    above 8 sigmas (nebula)."""
    L = S.mean(0)
    m34, mo = ring_masks(3, 4.0001, 19), ring_masks(12, 18, 19)
    pct, sig, neb = [], [], []
    for y, x in stars:
        c = L[y - 19:y + 20, x - 19:x + 20]
        a, b = float(c[m34].mean()), float(c[mo].mean())
        sig.append(a - b)
        if b + ped > 0.05:
            pct.append(100 * (a - b) / (b + ped))
        if b > 8:
            neb.append(100 * (a - b) / b)
    f = lambda v: round(float(np.median(v)), 4) if v else None   # noqa: E731
    return {"rim_pct_of_level": f(pct), "rim_sigma": f(sig), "rim_pct_nebula": f(neb), "n_stars": len(sig),
            "n_nebula": len(neb)}


def disc_of(S, T, stars):
    """(S - T) around each star, the median of each ring (sigmas), then the median over the stars."""
    D = (S - T).mean(0)
    rows = []
    for y, x in stars:
        c = D[y - 80:y + 81, x - 80:x + 81]
        yy, xx = np.mgrid[-80:81, -80:81]
        r = np.hypot(xx, yy)
        rows.append([float(np.median(c[(r >= a) & (r < b)])) for a, b in RINGS])
    med = np.median(np.array(rows), 0)
    return {"rings_px": [f"{a}-{b}" for a, b in RINGS], "median_S_minus_T_sigma": [round(float(v), 4) for v in med],
            "disc_10_60": round(float(np.mean([med[1], med[2], med[3], med[4]])), 4), "stars": len(stars)}


def core_share(O, S, core):
    o, s = O.mean(0), S.mean(0)
    return float(((o - s)[core]).sum() / o[core].sum())


class Judge:
    """One judge image: its input O, its ROIs to export, the measure of a starless image S."""

    def __init__(self, name, O, valid, rois, top_pct=99.95):
        self.name, self.O, self.valid, self.rois, self.top_pct = name, O, valid, rois, top_pct

    def measure(self, S):
        return {}


def build(name: str, dev):
    if name == "m27":
        O, valid, S0 = from_store(M27)
        w = np.s_[1300:1600, 2400:2760]
        sm = S0.mean(0)[w]
        blur = ndimage.uniform_filter(sm, 15, mode="nearest")
        core = blur > 0.5 * blur.max()
        j = Judge(name, O, valid, [("coeur", 1250, 1650, 2380, 2780, 2)])
        j.measure = lambda S: {"core_share_in_stars": round(core_share(O[:, w[0], w[1]], S[:, w[0], w[1]], core), 4),
                               "core_px": int(core.sum()),
                               "stored_shipped_share": round(core_share(O[:, w[0], w[1]], S0[:, w[0], w[1]], core), 4)}
        return j
    if name == "ngc6543":
        d = st.NB / "ngc6543"
        O, T = np.load(d / "input.npy").astype(np.float32), np.load(d / "starless_fixed.npy").astype(np.float32)
        blur = ndimage.uniform_filter(T.mean(0), 15, mode="nearest")
        core = blur > 0.2 * blur.max()                     # the nebula's body: the target's blurred image above a fifth of its maximum
        cy, cx = (int(v.mean()) for v in np.nonzero(core))
        y0, x0 = int(np.clip(cy - 200, 0, O.shape[1] - 400)), int(np.clip(cx - 200, 0, O.shape[2] - 400))
        j = Judge(name, O, np.ones(O.shape[1:], bool), [("coeur", y0, y0 + 400, x0, x0 + 400, 2)])
        j.target = T
        o = O.mean(0)

        def measure(S):
            s, t = S.mean(0), T.mean(0)
            return {"core_share_in_stars": round(core_share(O, S, core), 4),
                    "owner_target_share": round(core_share(O, T, core), 4), "core_px": int(core.sum()),
                    "core_mae_sigma": round(float(np.abs(s - t)[core].mean()), 4),
                    "core_mae_pct_of_level": round(100 * float(np.abs(s - t)[core].mean() / o[core].mean()), 3)}
        j.measure = measure
        return j
    if name == "comb56":
        O, T = np.load(COMB56 / "input.npy").astype(np.float32), np.load(COMB56 / "starless.npy").astype(np.float32)
        Lt = (O - T).mean(0)
        stars = isolated_peaks(Lt, 200, 1e12, 25, win=161, crowd=1500)
        y, x = stars[0]
        j = Judge(name, O, np.ones(O.shape[1:], bool), [("disque", y - 200, y + 200, x - 200, x + 200, 2)], top_pct=99.0)
        j.target = T
        j.measure = lambda S: {"disc": disc_of(S, T, stars), "rim_target": rim_of(T, stars, 0.0),
                               "rim": rim_of(S, stars, 0.0)}
        return j
    if name.startswith("rosette"):
        E = ROSETTE / ("05371932c85785f5" if name.endswith("deconv") else "3d5e079f33de24d5")
    else:
        E = NGC7000 / "e6449a4df328a726"
    O, valid, _ = from_store(E)
    if name == "ngc7000_deconv":                 # the Deconvolution network on the Stars block's input, plane by plane
        from astro_gpu.stages.deconv import BETA_WEIGHTS, network_plane
        from torch import nn
        net = sr.NAFNet(recompute=False)
        w = net.intro.out_channels
        net.intro, net.ending = nn.Conv2d(1, w, 3, padding=1), nn.Conv2d(w, 1, 3, padding=1)
        net.load_state_dict(torch.load(BETA_WEIGHTS, map_location="cpu", weights_only=False)["net"])
        net = net.to(dev).eval()
        fitted: list = []
        O = np.stack([network_plane(net, O[c], dev, lambda: None, fitted) for c in range(3)]).astype(np.float32)
        del net
    stars = isolated_peaks(O.mean(0), 30, 2e4, 200)
    lev = [float(O.mean(0)[y - 19:y + 20, x - 19:x + 20][ring_masks(12, 18, 19)].mean()) for y, x in stars]
    y, x = stars[int(np.argmax(lev))]
    j = Judge(name, O, valid, [("etoile", y - 100, y + 100, x - 100, x + 100, 4)])
    ped = from_store.pedestal
    j.measure = lambda S: {"rim": rim_of(S, stars, ped), "stars": len(stars), "pedestal_sigma": round(ped, 3)}
    return j


# ---------------------------------------------------------------------------------------------- the images to look at
def stretched(v, top):
    return np.clip(np.arcsinh((v + 3) / 3) / np.arcsinh((top + 3) / 3), 0, 1)


def export(j: Judge, kind: str, img: np.ndarray, tag: str) -> None:
    """One JPEG per ROI of ``img`` (3, H, W in sigmas), stretched by the judge image's own original: the same for every
    network."""
    VIEW.mkdir(parents=True, exist_ok=True)
    for roi, y0, y1, x0, x1, sc in j.rois:
        top = float(np.percentile(j.O[:, y0:y1, x0:x1].mean(0), j.top_pct))
        v = stretched(img[:, y0:y1, x0:x1], top).transpose(1, 2, 0)
        if sc > 1:
            v = cv2.resize(v, None, fx=sc, fy=sc, interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(VIEW / f"{j.name}_{roi}_{tag}.jpg"), (v[:, :, ::-1] * 255 + 0.5).astype(np.uint8),
                    [cv2.IMWRITE_JPEG_QUALITY, 92])


NAMES = ("m27", "ngc6543", "comb56", "rosette", "rosette_deconv", "ngc7000", "ngc7000_deconv")


def judge(nets, device, only) -> int:
    dev = torch.device(device)
    OUT.mkdir(parents=True, exist_ok=True)
    res = {n: json.loads((OUT / f"{n}.json").read_text()) if (OUT / f"{n}.json").exists() else {} for n in nets}
    for name in only:
        j = build(name, dev)
        if not (VIEW / f"{name}_{j.rois[0][0]}_original.jpg").exists():
            export(j, "o", j.O, "original")
            if hasattr(j, "target"):
                export(j, "t", j.target, "cible_proprietaire")
        for n in nets:
            net = get_net(n, dev)
            t = time.time()
            S = sr.apply(net, j.O, dev)
            S[:, ~j.valid] = j.O[:, ~j.valid]
            sec = time.time() - t
            res[n][name] = {**j.measure(S), "seconds": round(sec, 1), "device": device}
            export(j, "s", S, f"{n}_sans_etoiles")
            export(j, "l", j.O - S, f"{n}_etoiles")
            (OUT / f"{n}.json").write_text(json.dumps(res[n], indent=1))
            log(f"{name} {n}: {res[n][name]}")
            del net, S
            if dev.type == "cuda":
                torch.cuda.empty_cache()
        del j
    return 0


def timing(nets, device, only) -> int:
    dev = torch.device(device)
    if dev.type == "cpu":
        torch.set_num_threads(8)
    O, valid, _ = from_store(NGC7000 / "e6449a4df328a726")
    res = json.loads((OUT / "temps.json").read_text()) if (OUT / "temps.json").exists() else {}
    for n in nets:
        net = get_net(n, dev)
        sr.apply(net, O[:, :1024, :1024], dev)                           # warm-up
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t = time.time()
        sr.apply(net, O, dev)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        res.setdefault(n, {})[dev.type] = {"seconds": round(time.time() - t, 1), "shape": list(O.shape[1:]),
                                          "threads": torch.get_num_threads() if dev.type == "cpu" else None}
        log(f"{n} on {dev.type}: {res[n][dev.type]}")
        (OUT / "temps.json").write_text(json.dumps(res, indent=1))
    return 0


def bench(nets, device) -> int:
    """The bench's starless images of each network (stars_train.bench, for any candidate), stars_compare.py after."""
    from astropy.io import fits
    dev = torch.device(device)
    for n in nets:
        net = get_net(n, dev)
        tag = "shipped" if n == "shipped" else f"bank{n}"
        for case in ("heart", "iris", "m51", "m8", "rosette", "seestar"):
            try:
                O, _, (bk, unit, scale, nm) = st.bench_case(case)
            except Exception as e:  # noqa: BLE001
                log(f"{case}: left out ({type(e).__name__}: {e})")
                continue
            t = time.time()
            S = sr.apply(net, O, dev)
            S[:, nm <= 0] = O[:, nm <= 0]
            S = (S * unit + bk) / scale
            fits.PrimaryHDU(S.astype(np.float32)).writeto(st.COMPARE / case / f"net_{tag}_starless.fits", overwrite=True)
            log(f"{n} {case}: {O.shape[2]}x{O.shape[1]} in {time.time() - t:.1f} s")
        del net
    return 0


def control(nets, device, cases=("iris", "m51", "m8", "rosette", "seestar")) -> int:
    """The bench cases whose project is gone (only heart's remains, which stars_compare.py measures): their stored original
    through each network, as the Stars block does on an image without half-stacks (the sky's spread as the unit), the
    candidates against the shipped network. Light moved in the star pixels (where the shipped network took more than 3
    sigmas), as a share of the light it took there, and the change everywhere else (mean absolute, sigmas)."""
    from stars_compare import OUT as CMP, read_fits
    dev = torch.device(device)
    res = json.loads((OUT / "controle.json").read_text()) if (OUT / "controle.json").exists() else {}
    for case in cases:
        meta = json.loads((CMP / case / "meta.json").read_text())
        lin = read_fits(CMP / case / "input.fits") * meta["scale"]
        shown = np.ones(lin.shape[1:], bool)
        bk = np.array([np.median(lin[c][::7]) for c in range(3)], np.float32)[:, None, None]
        unit = sr.sky_unit(lin, shown)
        O = (lin - bk) / unit
        outs = {}
        for n in ["shipped"] + [x for x in nets if x != "shipped"]:
            net = get_net(n, dev)
            outs[n] = sr.apply(net, O, dev).mean(0)
            del net
            if dev.type == "cuda":
                torch.cuda.empty_cache()
        o, base = O.mean(0), outs["shipped"]
        star = (o - base) > 3
        flux = float((o - base)[star].sum())
        r = res.setdefault(case, {"shape": list(O.shape[1:]), "unit": round(float(unit), 4)})
        for n, S in outs.items():
            d = S - base
            r[n] = {"more_removed_pct_of_star_light": round(100 * float(np.clip(-d, 0, None)[star].sum()) / flux, 3),
                    "less_removed_pct_of_star_light": round(100 * float(np.clip(d, 0, None)[star].sum()) / flux, 3),
                    "off_star_mean_abs_change_sigma": round(float(np.abs(d[~star]).mean()), 4),
                    "off_star_median_change_sigma": round(float(np.median(d[~star])), 4)}
        (OUT / "controle.json").write_text(json.dumps(res, indent=1))
        log(f"{case}: {r}")
        del outs, O, lin
    return 0


def valprofile(nets, device) -> int:
    """The disc around bright stars on the bank's validation patches of the assembled fields (57_sho, 57_rgb, 58_hoo,
    58_rgb, 62_ho, 26_rgb): NOT held out (the candidates trained on the same fields; the shipped network did not), the
    plan's fallback to the held-out image. Median over the stars of each ring's median of (network - target), sigmas; the
    patches without and behind the Deconvolution network apart."""
    dev = torch.device(device)
    done = json.loads((sb.BANK / "progress.json").read_text())["done"]
    meta = json.loads((sb.BANK / "meta.json").read_text())
    inp, tgt = np.load(sb.BANK / "inp.npy", mmap_mode="r"), np.load(sb.BANK / "tgt.npy", mmap_mode="r")
    assembled = {"57_sho", "57_rgb", "58_hoo", "58_rgb", "62_ho", "26_rgb"}
    idx = [i for i in range(done - sb.NVAL, done) if meta["field"][i] in assembled]
    rings = [(4, 8), (8, 16), (16, 32), (32, 48), (48, 64)]
    yy, xx = np.mgrid[-64:65, -64:65]
    rr = np.hypot(xx, yy)
    res = json.loads((OUT / "profil_validation.json").read_text()) if (OUT / "profil_validation.json").exists() else {}
    for n in nets:
        net = get_net(n, dev)
        out = {"without": [], "behind_deconvolution": []}
        for i in idx:
            x = torch.from_numpy(np.asarray(inp[i], np.float32))[None].to(dev)
            with torch.no_grad(), torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                pred = net(x).float()
            pred, xl, tl = (sr.inv(v.float())[0].cpu().numpy() for v in (pred, x, torch.from_numpy(
                np.asarray(tgt[i], np.float32))[None].to(dev)))
            d = (pred - tl).mean(0)
            lay = (xl - tl).mean(0)
            for y, xx0 in isolated_peaks(lay, 100, 1e12, 5, win=129, crowd=1500):
                if 70 <= y < 512 - 70 and 70 <= xx0 < 512 - 70:
                    c = d[y - 64:y + 65, xx0 - 64:xx0 + 65]
                    out["behind_deconvolution" if meta["deconvolved"][i] else "without"].append(
                        [float(np.median(c[(rr >= a) & (rr < b)])) for a, b in rings])
        r = {}
        for k, v in out.items():
            if v:
                m = np.median(np.array(v), 0)
                r[k] = {"stars": len(v), "rings_px": [f"{a}-{b}" for a, b in rings],
                        "median_net_minus_target_sigma": [round(float(t), 4) for t in m],
                        "disc_8_64": round(float(m[1:].mean()), 4)}
        res[n] = r
        (OUT / "profil_validation.json").write_text(json.dumps(res, indent=1))
        log(f"{n}: {r}")
        del net
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=("judge", "time", "bench", "control", "valprofile"))
    ap.add_argument("--nets", default="shipped,A,B")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--only", default=",".join(NAMES))
    a = ap.parse_args(argv)
    nets = a.nets.split(",")
    if a.cmd == "judge":
        return judge(nets, a.device, a.only.split(","))
    if a.cmd == "time":
        return timing(nets, a.device, None)
    if a.cmd == "valprofile":
        return valprofile(nets, a.device)
    if a.cmd == "control":
        return control(nets, a.device)
    return bench(nets, a.device)


if __name__ == "__main__":
    sys.exit(main())
