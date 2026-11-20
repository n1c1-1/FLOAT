"""Issue #178: the nebula alone, without the stars. NGC 6543: the central star (its footprint in the owner's star layer =
input - retouched target, above 3 sigmas, dilated 5 px, with every other star's) is left out of the zone; the share of the
nebula's light that goes to the stars layer, for the owner's retouch (near 0), shipped, A, B and the reference tool. Also
the starless level of each network minus the shipped one on the nebula outside the stars (M27 and NGC 6543), in sigmas and
in % of the level: a real loss of nebula shows there, a display difference does not.

    python tools/bench/stars_nebula.py [--device cpu]     (the zone: A_REGARDER/.../ngc6543_zone_nebula.jpg)
The star footprint: the owner's star layer above 10 sigmas dilated 5 px; the nebula zone: the retouch smoothed above 10 sigmas.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stars_judge as sj  # noqa: E402
from astro_gpu.stages import star_removal as sr  # noqa: E402

NETS = ("shipped", "A", "B", "ref")


def run(n, O, dev):
    if n == "ref":
        return sj.ref_apply(O)[0]
    net = sj.get_net(n, dev)
    S = sr.apply(net, O, dev)
    del net
    return S


def main(device="cpu"):
    dev = torch.device(device)
    out = {}
    # ---- NGC 6543
    j = sj.build("ngc6543", dev)
    O, T = j.O, j.target
    L = O.mean(0)
    layer = (O - T).mean(0)
    stars = ndimage.binary_dilation(layer > 10, iterations=5)     # the owner's star layer above 10 sigmas, 5 px wider
    blur = ndimage.uniform_filter(T.mean(0), 15, mode="nearest")
    zone = (blur > 10) & ~stars                                    # the nebula: the retouch smoothed above 10 sigmas
    outside = (blur > 3) & ~stars
    S = {"owner": T}
    for n in NETS:
        S[n] = run(n, O, dev)
        sj.log(f"ngc6543 {n} done")
    r = {"zone_px": int(zone.sum()), "central_star_px": int(stars.sum()),
         "nebula_light_in_stars_layer_pct": {n: round(100 * float((L - s.mean(0))[zone].sum() / L[zone].sum()), 3)
                                            for n, s in S.items()}}
    base = S["shipped"].mean(0)
    lev = base[outside]
    r["vs_shipped_outside_stars"] = {n: {"median_sigma": round(float(np.median((s.mean(0) - base)[outside])), 4),
                                         "median_pct_of_level": round(100 * float(np.median(((s.mean(0) - base) / np.maximum(base, 1e-3))[outside & (base > 1)])), 3),
                                         "pixels": int(outside.sum()), "median_level_sigma": round(float(np.median(lev)), 2)}
                                     for n, s in S.items() if n not in ("shipped", "owner")}
    # sensitivity: the central star's footprint wider (15 px), and the starless against the owner's retouch (the truth), on the
    # nebula outside the stars and on the sky (retouch smoothed under 1 sigma): a loss of nebula is below the retouch on the
    # nebula, a pedestal is as much on the sky
    wide = ndimage.binary_dilation(layer > 10, iterations=15)
    zw = (blur > 10) & ~wide
    r["zone_px_wide15"] = int(zw.sum())
    r["nebula_light_in_stars_layer_pct_wide15"] = {n: round(100 * float((L - s.mean(0))[zw].sum() / L[zw].sum()), 3) for n, s in S.items()}
    sky = (blur < 1) & ~stars
    Tm = T.mean(0)
    r["vs_owner_retouch"] = {n: {"nebula_median_sigma": round(float(np.median((s.mean(0) - Tm)[outside])), 4),
                                 "nebula_median_pct_of_retouch": round(100 * float(np.median(((s.mean(0) - Tm) / np.maximum(Tm, 1e-3))[outside & (Tm > 1)])), 3),
                                 "sky_median_sigma": round(float(np.median((s.mean(0) - Tm)[sky])), 4), "sky_px": int(sky.sum())}
                              for n, s in S.items() if n != "owner"}
    r["vs_shipped_sky_median_sigma"] = {n: round(float(np.median((s.mean(0) - base)[sky])), 4) for n, s in S.items() if n not in ("shipped", "owner")}
    out["ngc6543"] = r
    # the zone to look at: the original stretched as the other exports, the zone tinted green, the stars' footprint red
    y0, y1, x0, x1, sc = j.rois[0][1:]
    top = float(np.percentile(O[:, y0:y1, x0:x1].mean(0), j.top_pct))
    v = sj.stretched(O[:, y0:y1, x0:x1], top).transpose(1, 2, 0) * 0.8
    z, st = zone[y0:y1, x0:x1], stars[y0:y1, x0:x1]
    v[z] = 0.6 * v[z] + 0.4 * np.array([0, 1, 0])
    v[st] = 0.6 * v[st] + 0.4 * np.array([1, 0, 0])
    v = cv2.resize(v, None, fx=sc, fy=sc, interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(sj.VIEW / "ngc6543_zone_nebula.jpg"), (np.clip(v[:, :, ::-1], 0, 1) * 255 + 0.5).astype(np.uint8))
    del S, O, T, j
    # ---- M27 (the owner's, read only)
    O, valid, S0 = sj.from_store(sj.M27)
    w = np.s_[1300:1600, 2400:2760]
    Ow = O[:, w[0], w[1]]
    sm = ndimage.uniform_filter(S0.mean(0)[w], 15, mode="nearest")
    stars = ndimage.binary_dilation((Ow - S0[:, w[0], w[1]]).mean(0) > 3, iterations=3)
    outside = (sm > 0.2 * sm.max()) & ~stars
    S = {n: run(n, O, dev)[:, w[0], w[1]].mean(0) for n in NETS}
    base = S["shipped"]
    out["m27"] = {"zone_px": int(outside.sum()), "vs_shipped_outside_stars": {
        n: {"median_sigma": round(float(np.median((s - base)[outside])), 4),
            "median_pct_of_level": round(100 * float(np.median(((s - base) / np.maximum(base, 1e-3))[outside & (base > 1)])), 3),
            "median_level_sigma": round(float(np.median(base[outside])), 2)} for n, s in S.items() if n != "shipped"}}
    (sj.OUT / "nebule.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[sys.argv.index("--device") + 1] if "--device" in sys.argv else "cpu")
