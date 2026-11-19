"""FLOAT's Denoise on one pair's linear input (the engine's single-image run replayed with the stage's own functions),
for stars_nb_pairs.py: <in.npy> (3, H, W) in noise sigmas, background 0 -> <out.npy> float32, same units.

    FLOAT_ROOT=<checkout whose stages/denoise.py is the version to use> python tools/bench/stars_nb_denoise.py \
        <in.npy> <meta.json> <out.npy>

The network runs on the device CUDA_VISIBLE_DEVICES leaves (CUDA_DEVICE_ORDER=PCI_BUS_ID, CUDA_VISIBLE_DEVICES=0 for the
second card); on an out-of-memory error it goes on on the processor.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.environ.get("FLOAT_ROOT") or str(Path(__file__).resolve().parents[2]))
from astro_gpu.stages.denoise import PASSES, apply_net, is_grey, keep_level, norm_params, pretrained  # noqa: E402


def run(dev, full, Wt, unit_ok=True):
    net = pretrained(dev, mono=False).eval()
    bk, sg, nm = norm_params(full, Wt)
    sg = (sg * np.sqrt(2)).astype(np.float32)
    nm = (nm / np.sqrt(2)).astype(np.float32)
    grey = (lambda a: np.repeat(a.mean(0, keepdims=True), 3, axis=0)) if is_grey(full) else (lambda a: a)
    with torch.no_grad():
        return keep_level(grey(apply_net(net, full, nm, bk, sg, dev, shifts=PASSES)), full, Wt)


def main(src, meta, dst):
    m = json.load(open(meta))
    v = np.load(src).astype(np.float32)
    u = np.float32(m["unit"])
    bkg = np.array(m["background"], np.float32)[:, None, None]
    full = v * u + bkg                      # the Stars block's input in its own units
    Wt = np.ones_like(full)
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    try:
        dF = run(dev, full, Wt)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        print("out of memory on the card: the processor", flush=True)
        dF = run(torch.device("cpu"), full, Wt)
    np.save(dst, ((dF - bkg) / u).astype(np.float32))
    print("denoised", dst, flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:4])
