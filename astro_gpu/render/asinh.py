"""The colour-preserving arcsinh stretch (Lupton et al. 2004, PASP 116, 133; the reference program's asinh transformation):
pixel' = pixel * asinh(L * beta) / (L * asinh(beta)), L the luminance of the pixel once its black point is taken off:
the three channels take the same factor, so the hues of the linear image are kept. Shared by Post's Arcsinh stretch
filter (client/postfilters/tone.py) and the workflow layer's Arcsinh base stretch (render/simple.py)."""
from __future__ import annotations

import math

import torch


def asinh_factor(L: torch.Tensor, beta: float) -> torch.Tensor:
    """asinh(L beta) / (L asinh(beta)), its limit beta / asinh(beta) at L = 0, 1 at beta = 0."""
    if beta <= 1e-9:
        return torch.ones_like(L)
    a = math.asinh(beta)
    return torch.where(L > 1e-7, torch.asinh(L * beta) / (L.clamp(min=1e-7) * a), torch.full_like(L, beta / a))


def asinh_level(x: float, beta: float) -> float:
    """Where a grey level x (0..1) lands: asinh(x beta) / asinh(beta); x itself at beta = 0."""
    if beta <= 1e-9:
        return x
    return math.asinh(x * beta) / math.asinh(beta)


def strength_for(x: float, target: float) -> float:
    """The stretch factor beta that brings the grey level x (0 < x < 1) to ``target``: asinh(x beta) / asinh(beta) =
    target, found by bisection (the level rises with beta, from x to 1). 0 when x is already at or above the target (no
    stretch is needed)."""
    x = min(max(float(x), 1e-12), 1 - 1e-9)
    target = min(float(target), 1 - 1e-6)
    if target <= x:
        return 0.0
    lo, hi = math.log(1e-9), math.log(1e15)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if asinh_level(x, math.exp(mid)) < target:
            lo = mid
        else:
            hi = mid
    return math.exp(0.5 * (lo + hi))
