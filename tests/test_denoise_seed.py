"""The Adapted Denoise draws its tiles from a generator of its own seeded with the "Random seed" setting: the same
draws as torch's global generator seeded the same way (the results of before, byte for byte), but a draw made by other
code in the middle no longer shifts the tiles after it, and torch's global state is neither set nor read."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from astro_gpu.stages import denoise as dn


@pytest.fixture(autouse=True)
def _few_threads():
    n = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(n)


def _adapted(seed=3, steps=6):
    """The weights of a small network adapted on a small synthetic stack, as bytes; ``_adapted.moved``: whether torch's
    global random state changed during the adaptation."""
    rng = np.random.default_rng(0)
    A = rng.normal(1.0, 0.1, (3, 256, 256)).astype(np.float32)
    B = rng.normal(1.0, 0.1, (3, 256, 256)).astype(np.float32)
    bk, sg, nm = dn.norm_params(A, np.ones_like(A))
    g = torch.Generator().manual_seed(0)
    net = dn.UNet(w=(8, 16, 32))
    with torch.no_grad():
        for prm in net.parameters():
            prm.copy_(0.1 * torch.randn(prm.shape, generator=g))
    ctx = SimpleNamespace(check_stop=lambda: None, progress=lambda *a, **k: None, log=lambda *a, **k: None)
    before = torch.get_rng_state()
    dn.adapt(ctx, net, A, B, nm, bk, sg, torch.device("cpu"), steps, 1e-3, seed)
    _adapted.moved = not torch.equal(torch.get_rng_state(), before)
    return b"".join(v.numpy().tobytes() for v in net.state_dict().values())


def _hook(monkeypatch):
    """Other code drawing from torch's global generator between two tiles (at each tile's rotation)."""
    rot90 = torch.rot90

    def drawing(*a, **k):
        torch.rand(1)
        return rot90(*a, **k)
    monkeypatch.setattr(torch, "rot90", drawing)


def test_the_tiles_are_those_of_before_and_a_global_draw_no_longer_moves_them(monkeypatch):
    own = _adapted()
    assert not _adapted.moved                                   # the global state neither set nor drawn from
    assert _adapted() == own and _adapted(seed=4) != own
    with monkeypatch.context() as m:                            # the draws of before: torch's global generator
        m.setattr(dn, "_tile_generator", torch.manual_seed)
        assert _adapted() == own                                # the same results, byte for byte
        _hook(m)
        assert _adapted() != own                                # a global draw in the middle shifted the tiles
    _hook(monkeypatch)
    assert _adapted() == own                                    # it no longer does
