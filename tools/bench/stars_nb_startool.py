"""The reference star-removal tool's onnx weights on one stretched 16 bit TIFF (the delivery's <t>_denoise.tif), the same
tiling as the delivery's: 512 px tiles, stride 256, sin^2 window, reflect padding, then min(output, input).

    STARTOOL_WEIGHTS=<weights.onnx> python tools/bench/stars_nb_startool.py <in.tif> <out.tif>

CPU only (onnxruntime). Writes <out.tif>, which must not exist.
"""
import os
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import tifffile

T, ST = 512, 256


def run(src: Path, dst: Path, threads: int = 12) -> None:
    if dst.exists():
        raise SystemExit(f"{dst} exists: nothing overwritten")
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    s = ort.InferenceSession(os.environ["STARTOOL_WEIGHTS"], so, providers=["CPUExecutionProvider"])
    w1 = np.sin(np.pi * (np.arange(T) + 0.5) / T).astype(np.float32)
    win = np.outer(w1, w1) ** 2 + 1e-3
    y = tifffile.imread(src).astype(np.float32) / 65535.0
    H, Wd = y.shape[:2]
    ph, pw = -(H - T) % ST, -(Wd - T) % ST
    pad = np.pad(y, ((T // 2, T // 2 + ph), (T // 2, T // 2 + pw), (0, 0)), mode="reflect")
    acc = np.zeros(pad.shape, np.float32)
    ws = np.zeros(pad.shape[:2], np.float32)
    for r in range(0, pad.shape[0] - T + 1, ST):
        for q in range(0, pad.shape[1] - T + 1, ST):
            acc[r:r + T, q:q + T] += s.run(None, {"x:0": pad[r:r + T, q:q + T][None]})[0][0] * win[..., None]
            ws[r:r + T, q:q + T] += win
    res = np.clip(np.minimum((acc / ws[..., None])[T // 2:T // 2 + H, T // 2:T // 2 + Wd], y), 0, 1)
    tifffile.imwrite(dst, np.round(res * 65535).astype(np.uint16), photometric="rgb")


if __name__ == "__main__":
    run(Path(sys.argv[1]), Path(sys.argv[2]))
