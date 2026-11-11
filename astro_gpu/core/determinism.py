"""Run-to-run reproducibility: the same inputs give the same outputs.

On the GPU, sums that gather many contributions into one place (scatter-adds: drawing stars into an image,
accumulating gradients) are done in whatever order the threads finish, so the last bits of a result change from one
run to the next. Through a threshold or an iterative fit later on, those last bits could tip a decision and change a
result visibly. Inside ``deterministic()``, PyTorch uses its order-preserving versions of these operations, cuDNN its
deterministic algorithms, and cuBLAS a fixed workspace (``CUBLAS_WORKSPACE_CONFIG``, set when astro_gpu is
imported, before any GPU work).

PyTorch is imported by the functions, not by this module: the runner imports it (core/runner.py), and so did the
workers of the Calibration and Registration pools, which import the runner's module again when they start (a
spawned process imports its parent's main module) and never use PyTorch: about 700 MB reserved each for nothing
(issue #125; measured on 16 threads, pools of 12 through the runner: 956 MB a worker, 242 MB without PyTorch).

Random draws inside a result come from generators of their own seeded with a fixed value (``seeded_match`` for the
triangle matcher's RANSAC), never from the global ones: the same draws at every run, and no other code sees a change.
"""
from __future__ import annotations

import contextlib
import os
import threading

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


@contextlib.contextmanager
def deterministic(strict: bool = False):
    """Deterministic GPU algorithms inside the block; the previous settings restored after. ``strict``: an operation
    without a deterministic version raises an error instead of a warning (for code written to be fully reproducible)."""
    import torch
    prev = (torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled(),
            torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic)
    torch.use_deterministic_algorithms(True, warn_only=not strict)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(prev[0], warn_only=prev[1])
        torch.backends.cudnn.benchmark = prev[2]
        torch.backends.cudnn.deterministic = prev[3]


def warm_up() -> None:
    """Load what the first ``deterministic()`` block loads: in this PyTorch, ``use_deterministic_algorithms`` imports
    the compiler's configuration (torch._inductor, torch._dynamo), about a second once. The client calls it in a
    background thread at start, so that its first render does not wait for it."""
    import torch
    torch.use_deterministic_algorithms(torch.are_deterministic_algorithms_enabled(),
                                       warn_only=torch.is_deterministic_algorithms_warn_only_enabled())


#: the seed of the triangle matcher's random draws, the same at every match (``seeded_match``)
MATCH_SEED = 0
_SEED = threading.local()       # the generators of the match running in this thread (``seeded_match``), else none


class _SeededRandom:
    """``numpy.random`` as astroalign sees it: itself, but inside ``seeded_match`` (in that thread only) an unseeded
    ``default_rng()`` gives the block's generator and the legacy global draws (``shuffle``, ``rand``...) come from the
    block's own ``RandomState``, both seeded with MATCH_SEED. Nothing else changes, and numpy's global state is never
    read nor moved, so no other code sees a difference."""

    def __init__(self, random):
        self._random = random

    def __getattr__(self, name):
        state = getattr(_SEED, "state", None)
        if state is None:
            return getattr(self._random, name)
        gen, legacy = state
        if name == "default_rng":
            return lambda seed=None: gen if seed is None else self._random.default_rng(seed)
        if not name.startswith("_") and hasattr(legacy, name):
            return getattr(legacy, name)
        return getattr(self._random, name)


class _SeededNumpy:
    """numpy as astroalign's module holds it (its ``_np``), with ``_SeededRandom`` for its ``random``."""

    def __init__(self, np_):
        self._np, self.random = np_, _SeededRandom(np_.random)

    def __getattr__(self, name):
        return getattr(self._np, name)


@contextlib.contextmanager
def seeded_match(aa):
    """The triangle match of this block (``aa``: the astroalign module) draws from generators seeded with MATCH_SEED:
    the same stars give the same transform at every run. astroalign 2.6 shuffles the RANSAC's triangle matches with
    ``np.random.default_rng()``, unseeded, and has no parameter for a seed: the same stack solved twice used 7770 then
    7771 stars, its WCS a little different. Its module's numpy is seen through ``_SeededNumpy`` (set once, inert outside
    this block), not a change of numpy's global state or functions that other code and threads would see; it holds as
    long as astroalign draws through its ``_np``, by ``default_rng()`` or by the legacy global functions."""
    import numpy as np
    if getattr(aa, "_np", None) is np:
        aa._np = _SeededNumpy(np)
    old = getattr(_SEED, "state", None)
    _SEED.state = (np.random.default_rng(MATCH_SEED), np.random.RandomState(MATCH_SEED))
    try:
        yield
    finally:
        _SEED.state = old
