"""The self-test of the card: nobody can try every card on the market, so the program tries the user's.

For each heavy operation of the engine (the drizzle deposit and the sampling of the prediction of the stack, the Denoise
network on one tile, the Stars network on one tile, the Post render), on a small synthetic input: the operation is run
on the card and on the CPU, the results are compared within a tolerance and both are timed. A chosen card computes:
a stage leaves it only when an operation it uses gives a wrong result there, or fails there (any exception,
out-of-memory included). The times are information, shown in Settings: a card timed slower keeps its stages, since a
test of seconds does not tell the speed of a real run (a 436-frame drizzle stack timed slower on the card ran there in
13.6 min, 25.7 min on the CPU). A stage that keeps the card runs exactly as it does without the self-test.

The operations run in a child process of their own (``python -m astro_gpu.core.selftest <device>``, one JSON line per
operation): a card that crashes or hangs takes the child with it, never the program, and the operations it did not
finish count as failed. ``run_test`` is that parent side (no Qt, no torch needed in the caller beyond the device list).

The result, a profile, is kept per device in the settings, with the card's name and driver (``client/services/selftest``
runs the test once in the background at first launch and when the card or its driver changed). ``cpu_jobs`` turns a
profile into the jobs of a plan that go to the CPU (core/runner2.py, the plan's ``cpu_jobs``).
"""
from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from . import devices

#: the version of the test, saved with its result: raised whenever what is measured or how it is judged changes, so
#: that a result saved by an older test is not used and the card is tested again. 1: no number saved (the first test).
#: 2: the stacks timed on what a real stack repeats, each way of stacking judged on its own operations. 3: a card
#: timed slower keeps its blocks. 4: a few values off where a rejection decides are not a wrong result.
TEST_VERSION = 4
#: the operations tested, in order, and the stage each one decides (Stack has three)
OPERATIONS = {"drizzle": "stack", "sampling": "stack", "classic": "stack", "denoise": "denoise", "stars": "star_removal",
              "post": "post"}
STAGES = ("stack", "denoise", "star_removal", "post")
#: the largest difference allowed between the card and the CPU, in the unit each operation returns its result in: the
#: image's own scale for the sums and the render (rounding of float32 and of the order of the additions), the noise of
#: the image for the networks (autocast on the card: bfloat16); set from what a sound card gives (see tests/test_selftest)
TOLERANCE = {"drizzle": 5e-3, "sampling": 5e-3, "classic": 5e-3, "denoise": 0.3, "stars": 0.05, "post": 8e-3}
#: the operations whose result holds yes/no decisions taken at a threshold: the Classic stack rejects a value or keeps
#: it (and clamps an interpolated one or not), the sampling gives a coverage or none where half of a sample's weight
#: is covered. A rounding difference can flip such a decision for a few samples on any two machines, and the value
#: decided then differs by far more than rounding (a value of 8 rejected moves their mean by a third of the noise),
#: while a device that computes wrong differs everywhere. So these are wrong when more than a very small share of
#: their values differ beyond the tolerance, not when one does: the tolerance is held against the DECISION_PERCENTILE
#: of the absolute differences (at most 1 value in 10 000 beyond it). Measured on a sound card: 16 values of
#: 2 272 704 off in the Classic stack, the largest by 0.0104, in isolated spots; the 99.99th percentile 2.4e-07.
#: The other operations (the deposit, the networks, the render) are continuous: their largest difference is judged.
DECISIONS = ("sampling", "classic")
DECISION_PERCENTILE = 99.99
#: seconds one operation may take before the card counts as gone (its child process is then ended)
OP_TIMEOUT = 300.0
#: the free share of the card's memory below which it is busy with something else: a test launched by itself waits
BUSY_FREE_SHARE = 0.7

#: the verdicts: the card is kept for 'ok', for 'absent' (a model this installation does not have: the stage
#: cannot run anywhere, and that says nothing about the card) and for 'slower' (the times are information, shown in
#: Settings: a test of seconds does not tell the speed of a real run); it is lost for 'wrong' and 'error'
OK, WRONG, SLOWER, ERROR, ABSENT = "ok", "wrong", "slower", "error", "absent"
#: why a block leaves the card, in English (translated in Settings and in the run window, client/locale)
REASONS = {WRONG: "the result was wrong on the card", ERROR: "the card failed"}
STAGE_ON_CPU = "the card test sends this block to the processor: {why}"


# ---------------------------------------------------------------------------------------------- the operations
# Each returns {name: array} in a unit where the tolerance means something (TOLERANCE), computed on ``dev``. The input is
# the same on every device (a fixed seed, built on the host).

#
# The operations of the Stack block are timed on what a real stack repeats, frame after frame, at a real frame's
# scale: a stack of hundreds of frames pays its start (the grids, the model, the kernels) and the way back of its sums
# once, so a timing where they weigh as much as the frames says nothing of it. Sizes, so that the three take seconds
# on a processor: 8 frames each (the start and the way back are then a small share). The drizzle: half the side of a
# 12 Mpx frame (3 Mpx; millions of samples a frame, sums far larger than a processor's caches, as the real ones).
# The Classic method costs ten times more a pixel: a quarter of that side (0.76 Mpx), where a pixel takes the time it
# takes on a larger frame (measured on a processor: 0.23 us at 0.76 Mpx, 0.21 us at 3 Mpx, 0.86 us on the 300 x 400
# frames of the first test, whose time was not the pixels').

#: the frames of the drizzle timings (deposit, sampling): their size (rows, columns), their number, the scale and
#: the drop of the output
DRIZZLE_SIZE, DRIZZLE_FRAMES, DRIZZLE_SCALE, DRIZZLE_DROP = (1424, 2128), 8, 1.5, 0.8
#: the frames of the Classic timing
CLASSIC_SIZE, CLASSIC_FRAMES = (712, 1064), 8

#: what a stack makes once, kept from a call to the next (never timed with the frames): the test's frames and sums
#: on the host, the grids and the model of a device
_ONCE: dict[tuple, Any] = {}


def _once(key: tuple, make: Callable[[], Any]) -> Any:
    if key not in _ONCE:
        _ONCE[key] = make()
    return _ONCE[key]


def _mosaics(seed: int, n: int, H: int, W: int, level: float, noise: float) -> np.ndarray:
    """``n`` calibrated mosaic frames (n, H, W) on the host, a few bad pixels in each (NaN, as the calibration marks
    them)."""
    def make():
        rng = np.random.default_rng(seed)
        a = rng.normal(level, noise, (n, H, W)).astype(np.float32)
        a[:, ::97, ::89] = np.nan
        return a
    return _once(("frames", seed, n, H, W), make)


def _grids(d, H: int, W: int):
    """The pixel grid of a frame on the device ``d`` (x, y) and the colour of each pixel (an RGGB pattern)."""
    import torch
    yy, xx = torch.meshgrid(torch.arange(H, device=d, dtype=torch.float32),
                            torch.arange(W, device=d, dtype=torch.float32), indexing="ij")
    return xx, yy, torch.tensor([[0, 1], [1, 2]], device=d)[yy.long() % 2, xx.long() % 2]


def _upload(a: np.ndarray, d):
    """A frame from the host to the device (a stack does it once per frame)."""
    import torch
    return torch.from_numpy(a).to(d)


def _download(t) -> np.ndarray:
    """A result from the device to the host (a stack does it once, at its end)."""
    return t.cpu().numpy()


def op_drizzle(dev: str) -> dict[str, np.ndarray]:
    """What a drizzle stack repeats (stages/stack.py, a pass of ``Stack.run``): each frame goes to the device and is
    dropped (``drizzle_deposit``) into sums that stay there; they come back once, after the last frame."""
    import torch

    from ..stages import stack
    (H, W), n, S = DRIZZLE_SIZE, DRIZZLE_FRAMES, DRIZZLE_SCALE
    frames = _mosaics(1, n, H, W, 1.0, 0.05)
    d = torch.device(dev)
    xx, yy, ch = _once(("grids", dev, H, W), lambda: _grids(d, H, W))
    Ho, Wo = int(H * S), int(W * S)
    num, den = torch.zeros(3 * Ho * Wo, device=d), torch.zeros(3 * Ho * Wo, device=d)
    for i in range(n):
        val = _upload(frames[i], d)
        dx, dy = 0.8 + 1.37 * i, 0.3 + 0.61 * i                        # where the frame is on the reference (dithered)
        stack.drizzle_deposit(num, den, val, (xx + dx) * S - 0.5, (yy + dy) * S - 0.5, 1.0, ~torch.isnan(val), ch,
                              DRIZZLE_DROP * S / 2, S, Ho, Wo)
    devices.synchronize(dev)
    return {"num": _download(num), "den": _download(den)}


def op_sampling(dev: str) -> dict[str, np.ndarray]:
    """What the drizzle's own rejection repeats (stages/stack.py, passes 2 and 3 of ``Stack.run``): the model of the
    stack (``Prediction``, made once in a pass and kept on the device) sampled where each frame's samples land; the
    last frame's values come back, once."""
    import torch

    from ..stages.stack import Prediction
    (H, W), n, S = DRIZZLE_SIZE, DRIZZLE_FRAMES, DRIZZLE_SCALE
    Ho, Wo = int(H * S), int(W * S)
    d = torch.device(dev)

    def sums():
        rng = np.random.default_rng(2)
        den = rng.uniform(0.8, 1.2, (3, Ho, Wo)).astype(np.float32)
        return den, den * rng.normal(1.0, 0.1, (3, Ho, Wo)).astype(np.float32), \
            den * rng.normal(1.1, 0.1, (3, Ho, Wo)).astype(np.float32)

    def model():
        den, num, sq = (_upload(a, d).reshape(-1) for a in _once(("sums", Ho, Wo), sums))
        return Prediction(num, den, sq, 20, Ho, Wo, d, torch.zeros(3, device=d))
    pred = _once(("model", dev, Ho, Wo), model)
    xx, yy, ch = _once(("grids", dev, H, W), lambda: _grids(d, H, W))
    for i in range(n):
        dx, dy = 0.8 + 1.37 * i, 0.3 + 0.61 * i
        pr = pred.sample((xx + dx) * S - 0.5, (yy + dy) * S - 0.5, ch)
    devices.synchronize(dev)
    return {k: _download(v / 20.0 if k == "cover" else v) for k, v in pr.items()}


def op_classic(dev: str) -> dict[str, np.ndarray]:
    """What a Classic stack repeats (core/stack_classic.py, called as stages/stack.py calls it, with the block's
    default settings): each frame goes to the device, is debayered and interpolated there and comes back to the host,
    then the frames are rejected and combined by bands of a real stack's size."""
    from ..stages.stack import StackParams, classic_params
    from . import stack_classic as sc
    (H, W), n = CLASSIC_SIZE, CLASSIC_FRAMES
    frames = _mosaics(3, n, H, W, 1000.0, 30.0)
    coef = np.zeros((n, 2, 10))
    coef[:, 0, 0], coef[:, 0, 1], coef[:, 1, 0], coef[:, 1, 2] = W / 2 + 0.4, W / 2, H / 2 - 0.3, H / 2
    coef[:, 0, 0] += 1.37 * np.arange(n)                               # where each frame is on the reference (dithered)
    coef[:, 1, 0] += 0.61 * np.arange(n)
    res = sc.stack_classic(lambda i: frames[i], coef, params=classic_params(StackParams()), device=dev,
                           used=list(range(n)), ch_of_k=[0, 1, 1, 2], H=H, W=W)
    return {"full": np.asarray(res.full) / 1000.0}


_NETS: dict[tuple[str, str], Any] = {}


def _net(kind: str, dev: str):
    """A network, loaded once per device (its loading is not what is timed)."""
    if (kind, dev) not in _NETS:
        import torch
        d = torch.device(dev)
        if kind == "denoise":
            from ..stages.denoise import pretrained
            net = pretrained(d)
            net.eval()
        else:
            from ..stages.star_removal import network
            net = network(d)
        _NETS[(kind, dev)] = net
    return _NETS[(kind, dev)]


def op_denoise(dev: str) -> dict[str, np.ndarray]:
    """The Denoise network over one tile; the result in units of the image's noise (its only scale)."""
    import torch

    from ..stages.denoise import apply_net
    rng = np.random.default_rng(3)
    x = rng.normal(1.0, 0.1, (3, 512, 512)).astype(np.float32)
    nm = np.ones((512, 512), np.float32)
    bk, sg = np.array([1.0, 1.0, 1.0], np.float32), np.array([0.1, 0.1, 0.1], np.float32)
    out = apply_net(_net("denoise", dev), x, nm, bk, sg, torch.device(dev))
    return {"image": ((out - bk[:, None, None]) / sg[:, None, None]).astype(np.float32)}


def op_stars(dev: str) -> dict[str, np.ndarray]:
    """The Stars network over one tile; the result in the network's own stretch (a star of 300 and a sky of 1 alike)."""
    import torch

    from ..stages.star_removal import TILE, apply, fwd
    rng = np.random.default_rng(4)
    img = rng.normal(0.0, 1.0, (3, TILE, TILE)).astype(np.float32)
    for y, x in ((60, 80), (130, 200), (190, 40)):
        img[:, y - 4:y + 4, x - 4:x + 4] += 300.0
    return {"image": fwd(apply(_net("stars", dev), img, torch.device(dev))).astype(np.float32)}


def _post_image() -> np.ndarray:
    rng = np.random.default_rng(5)
    img = np.abs(rng.normal(0.05, 0.01, (3, 1600, 2400))).astype(np.float32)
    img[:, 700:900, 1100:1300] += 0.3
    return img


def op_post(dev: str) -> dict[str, np.ndarray]:
    """The Post render of an image (0..1, the display's own scale)."""
    from ..render.simple import SimpleRenderEngine, SimpleSettings
    if ("post", dev) not in _NETS:
        _NETS[("post", dev)] = SimpleRenderEngine.from_array(_post_image(), device=dev)
    return {"image": _NETS[("post", dev)].render(SimpleSettings(), 1.0).final.cpu().numpy()}


OPS: dict[str, Callable[[str], dict[str, np.ndarray]]] = {
    "drizzle": op_drizzle, "sampling": op_sampling, "classic": op_classic, "denoise": op_denoise, "stars": op_stars,
    "post": op_post}


# ---------------------------------------------------------------------------------------------- one operation
def _timed(fn: Callable[[str], dict], dev: str) -> tuple[dict, float]:
    """The result and the seconds of ``fn(dev)``: the first call warms the device up (shaders, kernels, the network's
    loading: what a run pays once), the time is the best of two calls after it, so that another program's moment
    of load does not count against the card."""
    fn(dev)
    best, out = float("inf"), None
    for _ in range(2):
        t0 = time.perf_counter()
        out = fn(dev)
        devices.synchronize(dev)
        best = min(best, time.perf_counter() - t0)
    return out, best


def _differences(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> list[np.ndarray] | None:
    """The absolute differences of two results, array by array (inf where only one of them is a finite number);
    None when their shapes differ."""
    out = []
    for k in a:
        x, y = np.asarray(a[k], np.float64), np.asarray(b[k], np.float64)
        if x.shape != y.shape:
            return None
        d = np.abs(x - y)
        d[~np.isfinite(d)] = float("inf")
        out.append(d.ravel())
    return out


def compare(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> float:
    """The largest absolute difference between two results (inf where only one of them is not a finite number)."""
    d = _differences(a, b)
    return float("inf") if d is None else max((float(x.max()) for x in d if x.size), default=0.0)


def compare_but_a_few(a: dict[str, np.ndarray], b: dict[str, np.ndarray], percentile: float = DECISION_PERCENTILE
                      ) -> float:
    """The difference between two results that all but a few of their values stay under: the ``percentile`` of the
    absolute differences, all values together. A value that is a number on one side only is never one of the few:
    inf, as ``compare``."""
    d = _differences(a, b)
    worst = compare(a, b)
    if d is None or not np.isfinite(worst) or not sum(x.size for x in d):
        return worst
    return float(np.percentile(np.concatenate(d), percentile))


def check(name: str, card: str, op: Callable[[str], dict] | None = None, tolerance: float | None = None) -> dict:
    """One operation on ``card`` and on the CPU: {op, stage, verdict, card_s, cpu_s, diff, judged, tolerance, error}.
    ``diff``: the largest difference between the two results, kept for information; ``judged``: the difference held
    against the tolerance, the largest too, except for the operations of DECISIONS (``compare_but_a_few``)."""
    fn = op or OPS[name]
    tol = TOLERANCE.get(name, 5e-3) if tolerance is None else tolerance
    out: dict[str, Any] = {"op": name, "stage": OPERATIONS.get(name, name), "verdict": ERROR, "card_s": None,
                           "cpu_s": None, "diff": None, "judged": None, "tolerance": tol, "error": ""}
    try:
        ref, out["cpu_s"] = _timed(fn, "cpu")
    except FileNotFoundError as e:                     # a model this installation does not have
        return {**out, "verdict": ABSENT, "error": str(e)[:200]}
    except Exception as e:  # noqa: BLE001 - the operation cannot even run on the CPU: the card is not to blame
        return {**out, "verdict": ABSENT, "error": f"{type(e).__name__}: {e}"[:200]}
    try:
        got, out["card_s"] = _timed(fn, card)
    except Exception as e:  # noqa: BLE001 - any failure on the card, out of memory included: the CPU for this stage
        return {**out, "error": ("out of memory" if devices.is_out_of_memory(e) else f"{type(e).__name__}: {e}"[:200])}
    finally:
        devices.empty_cache(card)
    out["diff"] = compare(ref, got)
    out["judged"] = compare_but_a_few(ref, got) if name in DECISIONS else out["diff"]
    if out["judged"] > tol:
        out["verdict"] = WRONG
    elif out["card_s"] >= out["cpu_s"]:
        out["verdict"] = SLOWER
    else:
        out["verdict"] = OK
    return out


# ---------------------------------------------------------------------------------------------- the profile
def stack_operations(method: str, rejection: str = "own") -> tuple[str, ...]:
    """The operations tested that a stack of these settings repeats (stages/stack.py: ``method``,
    ``drizzle_rejection``): the Classic method its own; the drizzle its deposit, with the sampling of the prediction
    when it rejects on the raw pixels ('own'), or the Classic method's integration when the rejection comes from it
    ('integration'); by frame count ('by_count') the deposit alone."""
    if method == "classic":
        return ("classic",)
    return ("drizzle",) + {"own": ("sampling",), "integration": ("classic",)}.get(rejection, ())


#: every way of stacking (method, drizzle rejection): the Stack block is on the CPU as a whole only when none keeps the card
STACK_WAYS = (("drizzle", "own"), ("drizzle", "by_count"), ("drizzle", "integration"), ("classic", "own"))


def _lost(r: dict) -> bool:
    """Whether an operation's result takes the card away from what uses it: a wrong result, or a failure. Not a card
    timed slower (see the module's docstring)."""
    return r.get("verdict") in (WRONG, ERROR)


def judge(results: list[dict]) -> dict[str, str]:
    """{stage: 'card' | 'cpu'}: the card is kept for a stage unless an operation of it gave a wrong result or failed
    there (a stage not tested keeps it, as it would without the self-test; so does one timed slower). The Stack block has several ways of stacking, each with its own
    operations (``stack_operations``): an operation that lost sends to the CPU the stacks that use it and no other
    (``cpu_jobs``, ``stack_why``); the block is 'cpu' here only when no way of stacking keeps the card."""
    out = {s: "card" for s in STAGES}
    lost = {r["op"] for r in results if _lost(r)}
    for r in results:
        if _lost(r) and r["stage"] != "stack":
            out[r["stage"]] = "cpu"
    if all(lost.intersection(stack_operations(m, rej)) for m, rej in STACK_WAYS):
        out["stack"] = "cpu"
    return out


def make_profile(device: str, card: str, driver: str, results: list[dict], auto: bool = False) -> dict:
    """The profile saved in the settings: the card, its driver, what the test found per operation and per stage."""
    return {"device": device, "card": card, "driver": driver, "auto": auto, "version": TEST_VERSION,
            "when": time.strftime("%Y-%m-%d %H:%M"), "ops": {r["op"]: r for r in results}, "stages": judge(results)}


def profile_of(profiles: dict | None, device: str | None) -> dict | None:
    """The saved profile of ``device`` ('auto': the one made under Automatic), or None. A profile saved by another
    version of the test is not one: nothing reads it (a run, Post, Settings), and the card is tested again."""
    p = (profiles or {}).get(device or "auto")
    return p if p and p.get("version") == TEST_VERSION else None


def still_valid(profile: dict | None, card: str, driver: str) -> bool:
    """Whether a saved profile is for this card and this driver, by this version of the test (else the card is tested
    again)."""
    return (bool(profile) and profile.get("version") == TEST_VERSION and profile.get("card") == card
            and profile.get("driver") == driver and bool(card))


def stage_reason(profile: dict, stage: str) -> str:
    """Why a stage went to the CPU (English, one of REASONS), '' when it kept the card."""
    if (profile.get("stages") or {}).get(stage) != "cpu":
        return ""
    return next((REASONS[r["verdict"]] for r in lost_operations(profile, stage)), REASONS[ERROR])


def lost_operations(profile: dict | None, stage: str) -> list[dict]:
    """The results of the operations of ``stage`` that gave a wrong result or failed on the card, in the test's order."""
    ops = (profile or {}).get("ops") or {}
    return [ops[n] for n, s in OPERATIONS.items() if s == stage and n in ops and _lost(ops[n])]


def stage_on_cpu(profiles: dict | None, device: str | None, stage: str) -> bool:
    """Whether the profile of ``device`` sends ``stage`` to the CPU."""
    p = profile_of(profiles, device)
    return bool(p) and (p.get("stages") or {}).get(stage) == "cpu"


def cpu_jobs(profile: dict | None, plan: Any) -> dict[str, Any]:
    """{key: why} of the jobs of a plan that the profile sends to the CPU (their stage has 'cpu' in it). A job of the
    Stack block: {operation: why} of its operations that lost, since which of them the job uses depends on its
    settings, the automatic ones known only in the run (``stack_why``, core/runner2.py)."""
    if not profile:
        return {}
    from .workflow import base
    out: dict[str, Any] = {}
    lost = {r["op"]: STAGE_ON_CPU.format(why=REASONS[r["verdict"]]) for r in lost_operations(profile, "stack")}
    for j in plan.jobs:
        if base(j.stage) == "stack":
            if lost:
                out[j.key] = dict(lost)
            continue
        why = stage_reason(profile, base(j.stage))
        if why:
            out[j.key] = STAGE_ON_CPU.format(why=why)
    return out


def stack_why(lost: dict[str, str], method: str, rejection: str = "own") -> str:
    """Why a stack of these settings goes to the CPU, '' when it keeps the card: the reason of the first operation it
    uses among those that lost (``lost``: a Stack job's entry of ``cpu_jobs``)."""
    return next((lost[n] for n in stack_operations(method, rejection) if n in lost), "")


# ---------------------------------------------------------------------------------------------- the card
def _short(name: str) -> str:
    return " ".join(w for w in name.split() if w not in ("GeForce", "NVIDIA"))


def driver_version(device: str, name: str) -> str:
    """The driver of a card: what it says of itself, else the system; '' when not known. Runs small programs: not in
    the GUI thread."""
    kw: dict[str, Any] = {"capture_output": True, "text": True, "timeout": 15,
                          "creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)}
    if device.startswith("cuda"):
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"], **kw).stdout
            for line in out.splitlines():
                n, _, v = line.rpartition(",")
                if v.strip() and n.strip() and _short(n.strip()) == _short(name):
                    return v.strip()
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
    elif device.startswith("xpu"):
        try:
            import torch
            v = getattr(torch.xpu.get_device_properties(device), "driver_version", "")
            if v:
                return str(v)
        except Exception:  # noqa: BLE001 - not known
            pass
    if sys.platform == "win32":
        try:
            ps = ("Get-CimInstance Win32_VideoController | "
                  "ForEach-Object { $_.Name + '|' + $_.DriverVersion }")
            out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], **kw).stdout
            for line in out.splitlines():
                n, _, v = line.partition("|")
                if v.strip() and (n.strip() == name or _short(n.strip()) == _short(name)):
                    return v.strip()
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
    return ""


def card_identity(device: str) -> tuple[str, str]:
    """(the card's name, its driver) as a profile keeps them; ('', '') when this machine has no such device."""
    info = next((d for d in devices.list_devices() if d.device == device), None)
    if info is None:
        return "", ""
    return _short(info.name), driver_version(device, info.name)


def card_busy(device: str) -> bool:
    """Whether something else is using the card's memory: a test started by itself then waits for a quieter moment."""
    free, total = devices.free_memory(device), devices.total_memory(device)
    return free is not None and bool(total) and free < BUSY_FREE_SHARE * total


# ---------------------------------------------------------------------------------------------- the parent side
def run_test(device: str, *, on_step: Callable[[str, dict], None] | None = None,
             on_proc: Callable[[Any], None] | None = None, python: str | None = None) -> list[dict]:
    """Run the operations on ``device`` in a child process: the list of their results, a failed one for what the
    child did not finish (it crashed or hung: the card did). Raises RuntimeError when the child could not even start."""
    from .runner2 import device_env
    visible, inside = device_env(device, os.environ.get("CUDA_VISIBLE_DEVICES"))
    if inside == "cpu":
        return []
    root = str(Path(__file__).resolve().parents[2])
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": visible, "PYTHONUNBUFFERED": "1",
           "PYTHONPATH": root + os.pathsep + os.environ.get("PYTHONPATH", "")}
    cmd = [python or sys.executable, "-m", "astro_gpu.core.selftest", inside]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL, env=env,
                            cwd=root, text=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if on_proc is not None:
        on_proc(proc)
    lines: queue.Queue[str | None] = queue.Queue()

    def reader():
        for line in proc.stdout:                       # type: ignore[union-attr]
            lines.put(line)
        lines.put(None)

    threading.Thread(target=reader, name="selftest-reader", daemon=True).start()
    ready, got, why = False, {}, "the card test stopped: the card may have crashed"
    while len(got) < len(OPERATIONS):
        try:
            line = lines.get(timeout=OP_TIMEOUT)
        except queue.Empty:
            why = f"the card test stopped: no answer from the card in {int(OP_TIMEOUT)} s"
            break
        if line is None:
            break
        try:
            r = json.loads(line) if line.lstrip().startswith("{") else None
        except ValueError:
            r = None
        if not isinstance(r, dict):
            continue
        if r.get("ready"):
            ready = True
        elif r.get("op"):
            got[r["op"]] = r
            if on_step is not None:
                on_step(r["op"], r)
    if proc.poll() is None:
        proc.kill()
    if not ready:
        raise RuntimeError("the card test could not start")
    return [got.get(n) or {"op": n, "stage": s, "verdict": ERROR, "card_s": None, "cpu_s": None, "diff": None,
                           "tolerance": TOLERANCE.get(n), "error": why}
            for n, s in OPERATIONS.items()]


# ---------------------------------------------------------------------------------------------- the child process
def main(argv: list[str] | None = None) -> int:
    """The operations on ``device``, one JSON line each on the standard output (what ``run_test`` reads); for someone
    running it by hand, the same in plain words on the error output: the device, then each operation's times and
    verdict."""
    device = (argv or sys.argv[1:] or ["cpu"])[0]
    print(json.dumps({"ready": True}), flush=True)
    name = next((d.name for d in devices.list_devices() if d.device == device), "the processor in its place")
    print(f"card test on {device}: {name}", file=sys.stderr, flush=True)
    for op in OPERATIONS:
        r = check(op, device)
        print(json.dumps(r), flush=True)
        card, cpu = (f"{r[k]:8.3f} s" if r[k] is not None else "       -  " for k in ("card_s", "cpu_s"))
        print(f"{op:9s} card {card}   processor {cpu}   {r['verdict']}{'  ' + r['error'] if r['error'] else ''}",
              file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
