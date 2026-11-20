"""Issue #178: the 11 November hole measure (rosette_reseau_A_trous.py) for the reference tool, on the same input as the
"Rosette 3 reseau" flow's Stars block (the Output block's lin.npy), with the checks that its number is not an artefact:

* the stretch round-trip alone (stretch then unstretch, no network): must give back the input (max error reported);
* the tool's background pedestal (median of its starless minus the input's sky), removed before the measure;
* the hole also in noise units (a - b over the unit), which a uniform pedestal does not change, next to the % of level.

    python tools/bench/rosette_trous_ref.py            (writes F:/astro_essais/stars_juges_AB/rosette_trous_ref.json)
Needs the flow's results of the shipped and A networks already made by rosette_reseau_A.py (read, never rewritten).
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rosette_reseau_A as R  # noqa: E402
import stars_judge as sj  # noqa: E402

OUT = Path("F:/astro_essais/stars_juges_AB")
prep = json.loads((R.done_dir("livre", "star_removal") / "prep_stars.json").read_text())
unit, bk = prep["unit"], np.array(prep["background"], np.float32)[:, None, None]
lin = np.load(R.upstream("render_simple") / "lin.npy").astype(np.float32)
O = (lin - bk) / unit
stars = R.isolated_peaks(O.mean(0), 100.0, 400)
c04, a815 = R.ring(0, 4.0001), R.ring(8, 15.0001)

# the tool on the flow's input (the review stretch as in the judge), and the round-trip without it
S_sn, sec = sj.ref_apply(O)
sn = (S_sn * unit + bk).mean(0)
top = float(O.max())
k = np.arcsinh((top + 3) / 3)
y = np.clip(np.arcsinh(np.clip(O + 3, 0, None) / 3) / k, 0, 1)
rt = (3 * np.sinh(y * k) - 3) * unit + bk
ok = O >= -3                                      # the stretch has no value under -3 sigma: those pixels (3e-6 of them) are left out
rt_err = float(np.abs(rt - lin)[ok].max() / unit)
imgs = {"entree": lin.mean(0)}
for net in ("livre", "A"):
    imgs[net] = np.load(R.done_dir(net, "star_removal") / "starless.npy").astype(np.float32).mean(0)
# B has no flow result: the network (its own architecture, the judge's loader) on the same input, in the same units
import torch  # noqa: E402
from astro_gpu.stages import star_removal as sr  # noqa: E402
imgs["B"] = (sr.apply(sj.get_net("B", torch.device("cpu")), O, torch.device("cpu")) * unit + bk).mean(0)
imgs["ref"] = sn
imgs["aller_retour_seul"] = rt.mean(0)

# the pedestal: the tool starless minus the input where the shipped network left the image untouched (sky)
d = (sn - lin.mean(0))
quiet = np.abs(lin.mean(0) - imgs["livre"]) < 0.05 * unit     # where the shipped network did not touch the image: sky
ped = float(np.median(d[quiet]))
ped_vs_shipped = float(np.median((sn - imgs["livre"])[quiet]))
imgs["ref_sans_socle"] = sn - ped

ann = np.array([imgs["livre"][y0 - 21:y0 + 22, x - 21:x + 22][a815].mean() for y0, x in stars])
sel = (ann - bk.mean()) / unit > 1.0
res = {"stars_bright": len(stars), "on_nebula": int(sel.sum()), "unit": unit, "roundtrip_max_error_sigma": round(rt_err, 5),
       "ref_pedestal_sigma": round(ped / unit, 4), "ref_minus_shipped_on_untouched_sky_sigma": round(ped_vs_shipped / unit, 4),
       "ref_seconds": sec}
for kk, S in imgs.items():
    a = np.array([S[y0 - 21:y0 + 22, x - 21:x + 22][c04].mean() for y0, x in stars])
    b = np.array([S[y0 - 21:y0 + 22, x - 21:x + 22][a815].mean() for y0, x in stars])
    pct = (100 * (a - b) / b)[sel]
    sg = ((a - b) / unit)[sel]
    res[kk] = {"median_pct": round(float(np.median(pct)), 2), "share_below_3pct": round(float(100 * (pct < -3).mean()), 1),
               "p10_pct": round(float(np.percentile(pct, 10)), 2), "median_sigma": round(float(np.median(sg)), 3),
               "median_annulus_level_sigma": round(float(np.median(((b - bk.mean()) / unit)[sel])), 2)}
    print(kk, res[kk], flush=True)
(OUT / "rosette_trous_ref.json").write_text(json.dumps(res, indent=1))
print(json.dumps({k: v for k, v in res.items() if not isinstance(v, dict)}))
