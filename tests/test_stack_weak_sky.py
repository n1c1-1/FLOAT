"""A mono stack of 3 to 5 frames with a weak sky (narrowband, calibrated with a dark that holds the bias): the automatic
percentile clipping, whose thresholds are fractions of the median, would reject the noise itself (89 % of a 4 frame SII
stack, 39 % of 3 red frames); the guard gives way to no rejection. And the one plane of a mono stack reports as one
colour: its noise and offsets are the plane's in the three, never zeros or NaN."""
from __future__ import annotations

import math

import numpy as np
import torch

from astro_gpu.core import rejection_classic as rc
from astro_gpu.core import stack_classic as sc

H, W = 64, 80


def _identity(n):
    c = np.zeros((n, 2, 10))
    c[:, 0, :3] = [W / 2, W / 2, 0]
    c[:, 1, :3] = [H / 2, 0, H / 2]
    return c


def _mono(n, level, sigma, seed=3):
    rng = np.random.default_rng(seed)
    return [(level + rng.normal(0, sigma, (H, W))).astype(np.float32) for _ in range(n)]


def _stack(n, level, sigma, **kw):
    frames = _mono(n, level, sigma)
    return sc.stack_classic(lambda i: frames[i], _identity(n), None, None, None, sc.ClassicParams(half_block=1),
                            "cpu", used=range(n), ch_of_k=[1, 1, 1, 1], H=H, W=W, mono=True, **kw)


def test_a_weak_sky_gets_no_percentile_clipping():
    res = _stack(4, level=4.0, sigma=8.0)
    assert res.choice.method == "none" and "too weak against the noise" in res.choice.reason
    assert res.rejected_pct == 0.0
    assert float(np.median(res.weight[1])) > 0                      # every point keeps its frames: coverage stays whole


def test_a_sky_above_the_noise_keeps_the_published_percentile_clipping():
    res = _stack(4, level=2000.0, sigma=8.0)
    assert res.choice.method == "percentile" and (res.choice.low, res.choice.high) == (0.2, 0.1)
    assert res.rejected_pct < 0.5                                    # the usual range of the rejection map: 0.1 to 0.5 %


def test_the_guard_only_touches_the_automatic_choice():
    v = torch.tensor(np.stack(_mono(4, 4.0, 8.0)).reshape(4, -1))
    by_hand = rc.choose(4, "percentile")
    assert rc.guard_percentile(by_hand, v) is by_hand               # the user named it: kept, as published
    auto = rc.choose(4)
    assert rc.guard_percentile(auto, v).method == "none"
    assert rc.guard_percentile(rc.choose(8), v).method == "winsorized"     # other methods are translation-invariant


def test_the_noise_and_offsets_of_a_mono_stack_are_the_planes_in_the_three_colours():
    res = _stack(4, level=500.0, sigma=8.0)
    assert all(res.noise[c] is not None and math.isfinite(res.noise[c]) for c in "RGB")
    assert res.noise["R"] == res.noise["G"] == res.noise["B"]
    assert len(set(res.rejected_pct_rgb)) == 1
    for f in res.frames:
        assert f["scale"][0] == f["scale"][1] == f["scale"][2] and f["offset"][0] == f["offset"][1] == f["offset"][2]


def test_only_monochrome_workflows_of_few_frames_change_key():
    from astro_gpu.stages.stack import Stack
    s = Stack()
    v = {"method": "classic"}
    assert s.mono_key_version(v, 4) == f"{s.key_version(v)}+weaksky1"
    assert s.mono_key_version(v, 3) != s.key_version(v)
    assert s.mono_key_version(v, 12) == s.key_version(v)
    assert s.key_version(v) == s.key_version({"method": "classic"})      # colour workflows: untouched
