"""The reference star-removal weights (onnx, CPU only) on one stretched image: python stars_ref_worker.py in.npy out.npy
(in: float32 (H, W, 3) in [0, 1]; 512 px tiles, stride 256, sin^2 window, the output never above the input). Prints the
inference time. No torch here: it runs in its own environment."""
import os
import sys
import time

import numpy as np
import onnxruntime as ort

W = os.environ["REF_TOOL_WEIGHTS"]                         # the reference weights (onnx file), set by the caller
T, ST = 512, 256


def main(src, dst):
    so = ort.SessionOptions()
    so.intra_op_num_threads = 12
    s = ort.InferenceSession(W, so, providers=["CPUExecutionProvider"])
    w1 = np.sin(np.pi * (np.arange(T) + 0.5) / T).astype(np.float32)
    win = np.outer(w1, w1) ** 2 + 1e-3
    y = np.load(src).astype(np.float32)
    t0 = time.time()
    H, Wd = y.shape[:2]
    ph, pw = -(H - T) % ST, -(Wd - T) % ST
    pad = np.pad(y, ((T // 2, T // 2 + ph), (T // 2, T // 2 + pw), (0, 0)), mode="reflect")
    acc, ws = np.zeros(pad.shape, np.float32), np.zeros(pad.shape[:2], np.float32)
    for r in range(0, pad.shape[0] - T + 1, ST):
        for q in range(0, pad.shape[1] - T + 1, ST):
            acc[r:r + T, q:q + T] += s.run(None, {"x:0": pad[r:r + T, q:q + T][None]})[0][0] * win[..., None]
            ws[r:r + T, q:q + T] += win
    res = np.clip(np.minimum((acc / ws[..., None])[T // 2:T // 2 + H, T // 2:T // 2 + Wd], y), 0, 1)
    np.save(dst, res.astype(np.float32))
    print(f"{time.time() - t0:.1f}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
