"""The Stars mask part's search of a result's stars, in a process of its own:

    python -m astro_gpu.client.starsearch <stars luminance .npy> <out .npy>

A search of a 16 Mpx linear stars image takes about a minute and holds the interpreter for most of it (the detection's
deblending): run in the window's process, even in a thread, it froze the window that long. The window starts this
process (``search``), waits for it in a thread, and draws the mask without the stars meanwhile
(postengine.Composer.star_discs)."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np


def search(L: np.ndarray, work: Path) -> np.ndarray:
    """The stars of ``L`` (postengine.find_stars), found by this module in another process; ``work``: a folder for the
    two files. Blocks until found (call it from a thread: it waits on the process, the interpreter free meanwhile)."""
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    tag = f"{os.getpid()}_{id(L)}"
    src, out = work / f"in_{tag}.npy", work / f"out_{tag}.npy"
    np.save(src, np.ascontiguousarray(L, np.float32))
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "-1", "PYTHONIOENCODING": "utf-8"}
    pkg = str(Path(__file__).resolve().parents[2])           # this package first, whatever the environment says
    env["PYTHONPATH"] = pkg + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    kw = {"stdin": subprocess.DEVNULL, "stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL, "env": env}
    if os.name == "nt":
        kw["creationflags"] = 0x08000000                      # CREATE_NO_WINDOW: no console on the screen
    try:
        subprocess.run([sys.executable, "-m", "astro_gpu.client.starsearch", str(src), str(out)], check=True, **kw)
        return np.load(out)
    finally:
        for f in (src, out):
            try:
                f.unlink()
            except OSError:
                pass


def main(argv: list[str]) -> int:
    from .postengine import find_stars
    np.save(argv[1], find_stars(np.load(argv[0])))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
