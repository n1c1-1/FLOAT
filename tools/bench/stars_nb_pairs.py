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


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=("check24", "assembled", "retouched", "recover"))
    a = ap.parse_args(argv)
    return {"check24": check24}[a.cmd]()


if __name__ == "__main__":
    sys.exit(main())
