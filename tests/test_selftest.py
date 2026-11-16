"""The compute devices (core/devices.py), the card test (core/selftest.py), a stage sent to the CPU in a run (the plan's
``cpu_jobs``), the memory check of the Stack stage and the tiles of the networks on a small card. No card is used:
fake devices and small synthetic inputs on the CPU (the real test on a card: python -m astro_gpu.core.selftest)."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import torch

from astro_gpu.core import devices, runner2, selftest
from astro_gpu.core.stage import Context
from astro_gpu.stages import denoise as dn
from astro_gpu.stages import star_removal as sr

TESTS_DIR = os.path.dirname(__file__)
sys.path.insert(0, TESTS_DIR)
import test_runner2 as tr  # noqa: E402  (its toy stages and project)
from test_runner2 import proj, state  # noqa: E402, F401  (its fixtures)


# ---------------------------------------------------------------------------------------------- the devices
def test_device_names_and_kinds():
    assert all(devices.is_device(x) for x in ("cuda:0", "cuda", "xpu:1", "mps", "privateml:0"))
    assert not any(devices.is_device(x) for x in ("cpu", "gpu2", "auto", ""))
    assert devices._kind_of("AMD Radeon RX 7800 XT") == "amd"
    assert devices._kind_of("Intel(R) Arc(TM) A770") == "intel"
    assert devices._kind_of("NVIDIA GeForce RTX 4060 Ti") == "nvidia"
    assert devices._kind_of("Apple GPU") == "apple"


def test_a_cuda_card_of_amd_is_beta(monkeypatch):
    class Prop:
        name, total_memory = "AMD Radeon RX 7800 XT", 16 * 2 ** 30

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda i: Prop)
    monkeypatch.setattr(torch.version, "hip", "6.0", raising=False)
    d = devices.list_devices()[0]
    assert (d.device, d.kind, d.beta, round(d.memory_gb)) == ("cuda:0", "amd", True, 16)
    monkeypatch.setattr(torch.version, "hip", None, raising=False)


def test_an_nvidia_card_is_not_beta(monkeypatch):
    class Prop:
        name, total_memory = "NVIDIA GeForce RTX 4060 Ti", 8 * 2 ** 30

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda i: Prop)
    assert not devices.list_devices()[0].beta


def test_an_intel_card_is_found_and_beta(monkeypatch):
    class Prop:
        name, total_memory, driver_version = "Intel(R) Arc(TM) A770", 8 * 2 ** 30, "1.23"

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.xpu, "is_available", lambda: True)
    monkeypatch.setattr(torch.xpu, "device_count", lambda: 1)
    monkeypatch.setattr(torch.xpu, "get_device_properties", lambda i: Prop)
    monkeypatch.setattr(torch.xpu, "memory_reserved", lambda i: 2 * 2 ** 30)
    infos = devices.list_devices()
    assert [(d.device, d.kind, d.beta) for d in infos] == [("xpu:0", "intel", True)]
    assert devices.available("xpu:0") and devices.free_memory("xpu:0") == 6 * 2 ** 30


def test_device_env_of_the_other_cards():
    assert runner2.device_env("xpu:0") == ("-1", "xpu:0")
    assert runner2.device_env("mps") == ("-1", "mps")
    assert runner2.device_env("privateml:0") == ("-1", "privateml:0")


def test_a_stage_on_a_card_this_process_has_not_runs_on_the_cpu():
    ctx = Context.__new__(Context)
    ctx.device_name = "xpu:0"
    assert ctx.device.type == ("xpu" if devices.available("xpu:0") else "cpu")


def test_helpers_do_nothing_on_the_cpu(monkeypatch):
    # without a Memory limit (one left by an earlier test of the same process made the processor tell its room)
    monkeypatch.setattr(devices, "memory_limit", lambda: None)
    devices.empty_cache("cpu")
    devices.synchronize("cpu")
    assert devices.free_memory("cpu") is None and devices.total_memory("cpu") is None
    assert devices.usable("cpu") == "cpu"


def test_out_of_memory_is_recognised():
    assert devices.is_out_of_memory(torch.OutOfMemoryError("CUDA out of memory"))
    assert devices.is_out_of_memory(RuntimeError("PI_ERROR_OUT_OF_RESOURCES"))
    assert not devices.is_out_of_memory(RuntimeError("something else"))


# ---------------------------------------------------------------------------------------------- the verdicts
def fake_op(card_value=1.0, card_sleep=0.0, cpu_sleep=0.02, error=None):
    """An operation: on the 'cpu' it gives 1.0, on the card ``card_value``, or raises ``error``."""
    def op(dev):
        if dev == "cpu":
            time.sleep(cpu_sleep)
            return {"a": np.array([1.0])}
        time.sleep(card_sleep)
        if error is not None:
            raise error
        return {"a": np.array([card_value])}
    return op


def test_the_card_is_kept_when_it_is_right_and_faster():
    # the processor 0.3 s per call: a margin that a loaded machine (the parallel workers of a run) cannot eat up; 20 ms
    # gave "slower" in a level 3 run
    r = selftest.check("drizzle", "cuda:0", fake_op(cpu_sleep=0.3))
    assert r["verdict"] == selftest.OK and r["card_s"] < r["cpu_s"]


def test_a_wrong_result_or_a_nan_sends_the_stage_to_the_cpu():
    assert selftest.check("drizzle", "cuda:0", fake_op(card_value=1.5))["verdict"] == selftest.WRONG
    assert selftest.check("denoise", "cuda:0", fake_op(card_value=float("nan")))["verdict"] == selftest.WRONG


def test_a_card_that_is_slower_is_said_slower_and_keeps_its_blocks():
    """A short test does not tell the speed of a real run (a drizzle stack measured slower on the card runs there in
    half the processor's time): the times are kept as information, a chosen card computes."""
    r = selftest.check("post", "cuda:0", fake_op(card_sleep=0.05, cpu_sleep=0.0))
    assert r["verdict"] == selftest.SLOWER and r["card_s"] > r["cpu_s"]
    res = [{**ok(n), "verdict": selftest.SLOWER, "card_s": 2.0, "cpu_s": 1.0} for n in selftest.OPERATIONS]
    prof = selftest.make_profile("cuda:0", "x", "1", res)
    assert set(prof["stages"].values()) == {"card"} and prof["ops"]["post"]["verdict"] == selftest.SLOWER
    assert selftest.cpu_jobs(prof, StackPlan) == {}
    assert all(selftest.stage_reason(prof, s) == "" for s in selftest.STAGES)
    assert not any("slower" in t for t in selftest.REASONS.values())            # no such line in a run's log


N_VALUES = 2_272_704                                    # the values of the Classic operation's result


def differing(change):
    """An operation whose result on the card is the processor's (zeros) changed by ``change(array)``."""
    def op(dev):
        a = np.zeros(N_VALUES, np.float32)
        if dev != "cpu":
            change(a)
        else:
            time.sleep(0.05)                            # (the card is not slower)
        return {"full": a}
    return op


def _flips(a):
    a[::N_VALUES // 16][:16] = 0.01


def _on_the_card(a):
    """What the card gave for the Classic operation: 16 values of 2,272,704 off (4 over the tolerance, the largest
    0.010356), in 6 or 7 isolated spots; the rest equal to rounding (99.99th percentile 2.38e-07)."""
    a[:] = 2.0e-7
    spots = np.arange(16) * 90_001 + 676 * 1064
    a[spots[:4]] = (0.010356, 0.0093, 0.0071, 0.0052)
    a[spots[4:12]] = 0.002
    a[spots[12:]] = 0.0002


@pytest.mark.parametrize("name", ["classic", "sampling"])
def test_a_few_values_off_where_a_rejection_decides_are_not_a_wrong_result(name):
    """The card's Classic stack was found wrong for 16 values of 2.3 million: values at the rejection's threshold,
    kept on one machine and rejected on the other, everything else equal to rounding."""
    for change, largest in ((_flips, 0.01), (_on_the_card, 0.010356)):
        r = selftest.check(name, "cuda:0", differing(change))
        assert r["verdict"] == selftest.OK
        assert r["diff"] == pytest.approx(largest, rel=1e-4)             # the largest is kept, for information
        assert r["judged"] < 1e-6 < r["tolerance"]


def test_a_result_off_in_many_values_or_everywhere_is_wrong():
    def one_percent(a):
        a[::100] = 0.01

    def everywhere(a):
        a[:] = 0.006
    for change in (one_percent, everywhere):
        r = selftest.check("classic", "cuda:0", differing(change))
        assert r["verdict"] == selftest.WRONG and r["judged"] > r["tolerance"]

    def one_nan(a):
        a[5] = np.nan
    assert selftest.check("classic", "cuda:0", differing(one_nan))["verdict"] == selftest.WRONG   # never a rounding


@pytest.mark.parametrize("name", ["drizzle", "denoise", "stars", "post"])
def test_an_operation_with_no_such_decision_is_wrong_for_a_single_value(name):
    def one(a):
        a[7] = 10 * selftest.TOLERANCE[name]
    r = selftest.check(name, "cuda:0", differing(one))
    assert r["verdict"] == selftest.WRONG and r["judged"] == r["diff"]


def test_any_exception_or_out_of_memory_means_cpu():
    r = selftest.check("post", "cuda:0", fake_op(error=RuntimeError("the driver said no")))
    assert r["verdict"] == selftest.ERROR and "driver said no" in r["error"]
    r = selftest.check("post", "cuda:0", fake_op(error=torch.OutOfMemoryError("CUDA out of memory")))
    assert r["verdict"] == selftest.ERROR and r["error"] == "out of memory"


def test_a_model_that_is_not_there_says_nothing_about_the_card():
    def absent(dev):
        raise FileNotFoundError("the star removal network is missing")

    r = selftest.check("stars", "cuda:0", absent)
    assert r["verdict"] == selftest.ABSENT
    assert selftest.judge([{"stage": "star_removal", "verdict": r["verdict"], "op": "stars"}])["star_removal"] == "card"


def ok(name):
    return {"op": name, "stage": selftest.OPERATIONS[name], "verdict": selftest.OK, "card_s": 0.1, "cpu_s": 1.0,
            "diff": 0.0, "tolerance": 1e-3, "error": ""}


def test_judge_needs_every_operation_of_a_stage():
    res = [ok("drizzle"), ok("sampling"), {**ok("denoise"), "verdict": selftest.WRONG}, ok("stars"), ok("post")]
    stages = selftest.judge(res)
    assert stages["stack"] == "card" and stages["denoise"] == "cpu" and stages["post"] == "card"


def timed(name, card_s, cpu_s):
    """An operation's result with these times and the verdict the plain comparison gives them."""
    return {**ok(name), "card_s": card_s, "cpu_s": cpu_s, "verdict": selftest.SLOWER if card_s >= cpu_s else selftest.OK}


def test_a_drizzle_stack_stays_on_the_card_whatever_the_test_timed():
    """A 436-frame drizzle stack went to the processor (25.7 min, 13.6 min on the card): the test had timed the card
    slower. The figures of the card itself, with the test as it is now: still slower on the drizzle."""
    res = [timed("drizzle", 0.451, 0.40), timed("sampling", 0.056, 0.33), timed("classic", 5.10, 2.40), ok("denoise"),
           ok("stars"), ok("post")]
    prof = selftest.make_profile("cuda:0", "x", "1", res)
    assert prof["ops"]["drizzle"]["verdict"] == selftest.SLOWER and prof["stages"]["stack"] == "card"
    assert selftest.cpu_jobs(prof, StackPlan) == {}


def test_a_wrong_classic_operation_moves_the_stacks_that_use_it_and_no_other():
    res = [timed("drizzle", 0.451, 0.40), ok("sampling"), {**ok("classic"), "verdict": selftest.WRONG}, ok("denoise"),
           ok("stars"), ok("post")]
    prof = selftest.make_profile("cuda:0", "x", "1", res)
    assert prof["stages"]["stack"] == "card"                    # a way of stacking keeps the card: not the whole block
    assert set(selftest.cpu_jobs(prof, StackPlan)) == {"s"}
    bad = selftest.cpu_jobs(prof, StackPlan)["s"]
    assert selftest.stack_why(bad, "drizzle", "own") == ""
    assert selftest.stack_why(bad, "drizzle", "by_count") == ""
    assert "wrong" in selftest.stack_why(bad, "classic", "own")
    assert "wrong" in selftest.stack_why(bad, "drizzle", "integration")      # that rejection is the Classic method's


def test_each_way_of_stacking_is_judged_on_the_operations_it_uses():
    assert selftest.stack_operations("classic", "own") == ("classic",)
    assert selftest.stack_operations("drizzle", "own") == ("drizzle", "sampling")
    assert selftest.stack_operations("drizzle", "by_count") == ("drizzle",)
    assert selftest.stack_operations("drizzle", "integration") == ("drizzle", "classic")
    res = [ok("drizzle"), {**ok("sampling"), "verdict": selftest.ERROR}, ok("classic"), ok("denoise"), ok("stars"),
           ok("post")]
    bad = selftest.cpu_jobs(selftest.make_profile("cuda:0", "x", "1", res), StackPlan)["s"]
    assert "failed" in selftest.stack_why(bad, "drizzle", "own")
    assert selftest.stack_why(bad, "drizzle", "by_count") == "" and selftest.stack_why(bad, "classic", "own") == ""
    res = [{**ok("drizzle"), "verdict": selftest.WRONG}, ok("sampling"), ok("classic")]
    bad = selftest.cpu_jobs(selftest.make_profile("cuda:0", "x", "1", res), StackPlan)["s"]
    assert "wrong" in selftest.stack_why(bad, "drizzle", "by_count") and selftest.stack_why(bad, "classic", "own") == ""


def test_the_stack_block_is_on_the_processor_only_when_no_way_of_stacking_keeps_the_card():
    wrong = {"verdict": selftest.WRONG}
    res = [{**ok("drizzle"), **wrong}, ok("sampling"), {**ok("classic"), **wrong}]
    prof = selftest.make_profile("cuda:0", "x", "1", res)
    assert prof["stages"]["stack"] == "cpu"
    assert selftest.stage_reason(prof, "stack") == selftest.REASONS[selftest.WRONG]
    res = [{**ok("drizzle"), **wrong}, ok("sampling"), ok("classic")]
    assert selftest.judge(res)["stack"] == "card"


class J:
    def __init__(self, key, stage):
        self.key, self.stage = key, stage


class StackPlan:
    jobs = [J("s", "stack"), J("d", "denoise")]


def test_the_profile_keeps_the_card_and_driver_and_says_why():
    res = [{**ok("drizzle"), "verdict": selftest.ERROR}, ok("sampling"), {**ok("classic"), "verdict": selftest.ERROR},
           ok("denoise"), ok("stars"), ok("post")]
    prof = selftest.make_profile("cuda:0", "RTX 4060 Ti", "610.88", res, auto=True)
    assert json.loads(json.dumps(prof))["card"] == "RTX 4060 Ti"                # goes into the settings' JSON
    assert selftest.still_valid(prof, "RTX 4060 Ti", "610.88")
    assert not selftest.still_valid(prof, "RTX 4060 Ti", "620.00")              # a new driver: tested again
    assert not selftest.still_valid(prof, "RTX 3070", "610.88")
    assert not selftest.still_valid(None, "RTX 4060 Ti", "610.88")
    profiles = {"cuda:0": prof, "auto": prof}
    assert selftest.profile_of(profiles, "auto") is prof and selftest.profile_of(profiles, "cuda:0") is prof
    assert selftest.profile_of(profiles, "cuda:1") is None
    assert selftest.stage_reason(prof, "stack") == selftest.REASONS[selftest.ERROR]
    assert selftest.stage_reason(prof, "denoise") == ""
    assert selftest.stage_on_cpu(profiles, "auto", "stack") and not selftest.stage_on_cpu(profiles, "auto", "post")


def test_a_result_saved_by_an_older_test_is_not_used_and_the_card_is_tested_again():
    """A result three days old, saved by a test that judged otherwise, still sent the stacks to the processor."""
    res = [timed("drizzle", 0.1345, 0.1205), timed("sampling", 0.38, 0.63), timed("classic", 5.10, 2.40)]
    prof = selftest.make_profile("cuda:0", "RTX 4060 Ti", "610.88", res)
    assert prof["version"] == selftest.TEST_VERSION and selftest.still_valid(prof, "RTX 4060 Ti", "610.88")
    older = {**prof, "version": selftest.TEST_VERSION - 1}
    first = {k: v for k, v in prof.items() if k != "version"}                  # as the first test saved it
    for old in (older, first):
        assert not selftest.still_valid(old, "RTX 4060 Ti", "610.88")          # tested again
        profiles = {"cuda:0": old, "auto": old}
        assert selftest.profile_of(profiles, "cuda:0") is None and selftest.profile_of(profiles, "auto") is None
        assert selftest.cpu_jobs(selftest.profile_of(profiles, "cuda:0"), StackPlan) == {}     # a run: the card
        assert not selftest.stage_on_cpu(profiles, "auto", "stack")
    assert selftest.profile_of({"cuda:0": prof}, "cuda:0") is prof


def test_cpu_jobs_of_a_plan():
    res = [ok("drizzle"), {**ok("sampling"), "verdict": selftest.ERROR}, ok("denoise"),
           {**ok("stars"), "verdict": selftest.ERROR}, ok("post")]
    prof = selftest.make_profile("cuda:0", "x", "1", res)

    class P:
        jobs = [J("a", "stack"), J("b", "stack@2"), J("c", "denoise"), J("d", "star_removal")]

    jobs = selftest.cpu_jobs(prof, P)
    # a whole block: its reason; the Stack block: the reason of each of its operations that lost, the run chooses
    assert set(jobs) == {"a", "b", "d"} and "the card failed" in jobs["d"]
    assert set(jobs["a"]) == {"sampling"} and "the card failed" in jobs["b"]["sampling"]
    assert selftest.cpu_jobs(None, P) == {}


# ---------------------------------------------------------------------------------------------- the child process
class FakeProc:
    def __init__(self, lines):
        self.stdout = iter(lines)

    def poll(self):
        return 0

    def kill(self):
        pass


def _child(monkeypatch, lines):
    monkeypatch.setattr(selftest.subprocess, "Popen", lambda *a, **kw: FakeProc(lines))


def test_a_child_that_dies_counts_its_unfinished_operations_as_failed(monkeypatch):
    lines = [json.dumps({"ready": True}) + "\n", json.dumps(ok("drizzle")) + "\n"]
    _child(monkeypatch, lines)
    res = {r["op"]: r for r in selftest.run_test("xpu:0")}
    assert res["drizzle"]["verdict"] == selftest.OK
    assert res["denoise"]["verdict"] == selftest.ERROR and res["post"]["verdict"] == selftest.ERROR
    assert selftest.judge(list(res.values()))["denoise"] == "cpu"


def test_a_child_that_cannot_start_is_an_error_not_a_verdict(monkeypatch):
    _child(monkeypatch, [])
    with pytest.raises(RuntimeError):
        selftest.run_test("xpu:0")


def test_no_card_nothing_to_test(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    assert selftest.run_test("cuda:0") == []


# ---------------------------------------------------------------------------------------------- the operations, on the CPU
@pytest.mark.parametrize("op", ["drizzle", "sampling", "classic", "post"])
def test_operations_run_and_give_finite_numbers(op):
    out = selftest.OPS[op]("cpu")
    assert all(np.isfinite(v).all() for v in out.values())


def test_the_drizzle_timing_keeps_its_sums_on_the_device_over_the_frames_and_downloads_once(monkeypatch):
    """The timing deposited one small frame and brought 117 MB of sums back each time: the card looked slower than the
    processor on a stack that it does in half the time. A real stack uploads a frame, deposits it into sums that
    stay on the device, frame after frame, and brings them back once."""
    from astro_gpu.stages import stack
    monkeypatch.setattr(selftest, "DRIZZLE_SIZE", (120, 180))          # the shape of the work, not its size
    seen, events = [], []
    deposit, up, down = stack.drizzle_deposit, selftest._upload, selftest._download

    def spy(num, den, *a, **kw):
        seen.append((id(num), id(den), num.device.type))
        events.append("deposit")
        return deposit(num, den, *a, **kw)
    monkeypatch.setattr(stack, "drizzle_deposit", spy)
    monkeypatch.setattr(selftest, "_upload", lambda a, d: events.append("upload") or up(a, d))
    monkeypatch.setattr(selftest, "_download", lambda t: events.append("download") or down(t))
    out = selftest.op_drizzle("cpu")
    n = selftest.DRIZZLE_FRAMES
    assert n >= 8 and len(seen) == n and len(set(seen)) == 1          # every frame into the same sums, on the device
    assert events == ["upload", "deposit"] * n + ["download"] * len(out)        # a frame up, its deposit; one way back
    assert set(out) == {"num", "den"} and float(out["den"].max()) > 1.5         # several frames added up at a point


def test_the_timed_stacks_have_frames_of_a_real_size(monkeypatch):
    """Eight frames of 300 x 400 timed the start of the Classic stack, not what it repeats frame after frame."""
    from astro_gpu.core import stack_classic as sc
    seen = {}
    real = sc.stack_classic

    def spy(load, coef, **kw):
        seen.update(kw, shape=np.asarray(load(kw["used"][0])).shape)
        return real(load, coef, **{**kw, "used": kw["used"][:2]})    # (two frames here: what is asked, not its time)
    monkeypatch.setattr(sc, "stack_classic", spy)
    out = selftest.op_classic("cpu")
    H, W = selftest.CLASSIC_SIZE
    assert len(seen["used"]) == selftest.CLASSIC_FRAMES >= 8 and seen["shape"] == (H, W) == (seen["H"], seen["W"])
    assert H * W >= 700_000 and out["full"].shape == (3, H, W)     # where a pixel costs what it costs on a real frame
    assert seen["params"].band_pixels == sc.ClassicParams().band_pixels          # the bands of a real stack
    h, w = selftest.DRIZZLE_SIZE
    assert h * w >= 1_000_000


def test_the_sampling_timing_keeps_its_model_on_the_device_and_samples_it_for_each_frame(monkeypatch):
    from astro_gpu.stages import stack
    monkeypatch.setattr(selftest, "DRIZZLE_SIZE", (120, 180))
    made, sampled, events = [], [], []
    init, sample, down = stack.Prediction.__init__, stack.Prediction.sample, selftest._download
    monkeypatch.setattr(stack.Prediction, "__init__", lambda self, *a, **kw: made.append(1) or init(self, *a, **kw))
    monkeypatch.setattr(stack.Prediction, "sample",
                        lambda self, *a: sampled.append(id(self)) or events.append("sample") or sample(self, *a))
    monkeypatch.setattr(selftest, "_download", lambda t: events.append("download") or down(t))
    out = selftest.op_sampling("cpu")
    selftest.op_sampling("cpu")
    n = selftest.DRIZZLE_FRAMES
    assert len(made) == 1 and len(sampled) == 2 * n and len(set(sampled)) == 1   # one model, as in a pass of a stack
    assert events[:n + len(out)] == ["sample"] * n + ["download"] * len(out)


def test_what_is_built_once_in_a_stack_is_not_timed_at_each_call(monkeypatch):
    """The frames of the test are made once (the random numbers took longer than the deposit they were timed with)."""
    monkeypatch.setattr(selftest, "DRIZZLE_SIZE", (120, 180))
    selftest.op_drizzle("cpu")
    made = []
    monkeypatch.setattr(np.random, "default_rng", lambda *a: made.append(a) or pytest.fail("made again"))
    selftest.op_drizzle("cpu")
    assert made == []


def test_the_denoise_operation_runs_on_the_cpu():
    if not dn.MODEL.exists():
        pytest.skip("no denoising model here")
    assert np.isfinite(selftest.op_denoise("cpu")["image"]).all()


# ---------------------------------------------------------------------------------------------- a stage on the CPU in a run


def test_a_job_the_profile_sends_to_the_cpu_runs_there(tmp_path, monkeypatch, proj):  # noqa: F811
    plan = tr.plan_all(proj)
    job = plan.jobs[0]
    info = runner2.prepare_run(proj, plan, "cuda:0", {job.key: selftest.STAGE_ON_CPU.format(why="the card failed")})
    body = json.loads(open(info.plan_file, encoding="utf-8").read())
    assert body["cpu_jobs"] == {job.key: selftest.STAGE_ON_CPU.format(why="the card failed")}
    w = runner2._Worker(Path(proj.root), body, tr.RULES, False)
    assert w.device_of(job) == "cpu"
    assert w.device_of(plan.jobs[-1]) == w.device


@pytest.mark.parametrize("method, rejection, where", [("drizzle", "own", "card"), ("drizzle", "by_count", "card"),
                                                      ("classic", "own", "cpu"), ("drizzle", "integration", "cpu")])
def test_a_stack_job_goes_to_the_processor_only_for_the_operations_its_settings_use(proj, monkeypatch, method,  # noqa: F811
                                                                                    rejection, where):
    """The method of a stack is known in the run (Automatic: once the registration is there): the run judges there."""
    from types import SimpleNamespace
    plan = tr.plan_all(proj)
    job = next(j for j in plan.jobs if j.stage == "stack")
    why = selftest.STAGE_ON_CPU.format(why=selftest.REASONS[selftest.WRONG])
    info = runner2.prepare_run(proj, plan, "cuda:0", {job.key: {"classic": why}})
    body = json.loads(open(info.plan_file, encoding="utf-8").read())
    monkeypatch.setattr(runner2.Pipeline, "effective_params",
                        lambda self, st: (SimpleNamespace(method=method, drizzle_rejection=rejection), {}, {}))
    w = runner2._Worker(Path(proj.root), body, tr.RULES, False)
    assert w.device_of(job) == ("cpu" if where == "cpu" else w.device)
    assert w.cpu_why(job) == (why if where == "cpu" else "")


@pytest.mark.parametrize("op, moved", [("drizzle", True), ("classic", False)])
def test_the_log_of_a_run_says_why_a_stack_went_to_the_processor_and_nothing_for_a_slower_card(proj, op, moved):  # noqa: F811
    """A drizzle stack (the toy one: the default method): the card test found the result of ``op`` wrong on the card,
    every other operation slower there. It moves, and the log says why, only for an operation it uses."""
    plan = tr.plan_all(proj)
    job = next(j for j in plan.jobs if j.stage == "stack")
    res = [{**ok(n), "verdict": selftest.WRONG if n == op else selftest.SLOWER, "card_s": 2.0} for n in selftest.OPERATIONS]
    cpu = selftest.cpu_jobs(selftest.make_profile("cuda:0", "x", "1", res), plan)
    assert set(cpu) == {job.key}
    info = runner2.prepare_run(proj, plan, "cuda:0", cpu)
    body = json.loads(open(info.plan_file, encoding="utf-8").read())
    w = runner2._Worker(Path(proj.root), body, tr.RULES, False)
    w.device = "cuda:0"                                 # (the card is hidden from the tests: the worker would say cpu)
    for j in plan.jobs[:plan.jobs.index(job) + 1]:
        assert w._run_job_here(j) == "done"
    said = [e["msg"] for e in tr.events(proj, info) if e.get("type") == "log" and e.get("level") == "warning"]
    assert not any("slower" in m for m in said)
    assert w.device_of(job) == ("cpu" if moved else "cuda:0")
    assert said == ([selftest.STAGE_ON_CPU.format(why=selftest.REASONS[selftest.WRONG])] if moved else [])


# ---------------------------------------------------------------------------------------------- the memory of the Stack stage
def test_stack_memory_follows_the_output_pixels():
    assert devices.stack_memory(12.0) == pytest.approx(2.0)
    assert devices.stack_memory(48.0) == pytest.approx(8.0)


def test_a_stack_that_does_not_fit_goes_to_the_cpu(tmp_path, monkeypatch, proj):  # noqa: F811
    from astro_gpu import api
    from astro_gpu.core import estimate as est
    monkeypatch.setattr(api.devices, "available", lambda d: True)
    monkeypatch.setattr(api.devices, "free_memory", lambda d: 3 * 2 ** 30)
    monkeypatch.setattr(api, "_keys", lambda p: tr.K(p))
    monkeypatch.setattr(api, "_store", lambda p: None)
    monkeypatch.setattr(est, "facts", lambda keys, wf, store=None: est.Facts(grid_mpx=48.0))   # 8 GB for 3 GB free
    plan = tr.plan_all(proj)
    out = api._stack_on_cpu(proj, plan, "cuda:0")
    stacks = [j.key for j in plan.jobs if j.stage.split("@")[0] == "stack"]
    assert set(out) == set(stacks) and "needs about 8.0 GB" in next(iter(out.values()))
    monkeypatch.setattr(est, "facts", lambda keys, wf, store=None: est.Facts(grid_mpx=12.0))   # 2 GB: it fits
    assert api._stack_on_cpu(proj, plan, "cuda:0") == {}


# ---------------------------------------------------------------------------------------------- tiles on a small card
class EchoUNet(torch.nn.Module):
    """Stands for the denoising network: its output cancels the normalised input, so the tiles give the image back; it
    raises a memory error for tiles larger than ``limit``."""

    def __init__(self, limit=None):
        super().__init__()
        self.limit = limit

    def forward(self, x):
        if self.limit is not None and x.shape[-1] > self.limit:
            raise torch.OutOfMemoryError("CUDA out of memory")
        return torch.zeros(x.shape[0], 3, x.shape[2], x.shape[3], device=x.device)


def _denoise_input():
    x = np.random.default_rng(0).normal(1.0, 0.1, (3, 300, 400)).astype(np.float32)
    nm = np.ones((300, 400), np.float32)
    return x, nm, np.ones(3, np.float32), np.full(3, 0.1, np.float32)


def test_denoise_keeps_the_tile_on_a_4_gb_card(monkeypatch):
    seen = []
    orig = dn._apply_net
    monkeypatch.setattr(dn.devices, "free_memory", lambda d: 4 * 2 ** 30)
    monkeypatch.setattr(dn, "_apply_net", lambda *a, **kw: seen.append(a[6]) or orig(*a, **kw))
    x, nm, bk, sg = _denoise_input()
    out = dn.apply_net(EchoUNet(), x, nm, bk, sg, torch.device("cpu"))
    assert seen == [512] * len(dn.PASSES) and np.allclose(out, x, atol=1e-5)


def test_denoise_shrinks_the_tile_when_memory_is_short(monkeypatch):
    seen = []
    orig = dn._apply_net
    monkeypatch.setattr(dn.devices, "free_memory", lambda d: 200 * 2 ** 20)
    monkeypatch.setattr(dn, "_apply_net", lambda *a, **kw: seen.append(a[6]) or orig(*a, **kw))
    x, nm, bk, sg = _denoise_input()
    out = dn.apply_net(EchoUNet(), x, nm, bk, sg, torch.device("cpu"))
    assert seen[0] < 512 and np.allclose(out, x, atol=1e-5)


def test_denoise_halves_the_tile_on_a_memory_error():
    x, nm, bk, sg = _denoise_input()
    out = dn.apply_net(EchoUNet(limit=320), x, nm, bk, sg, torch.device("cpu"))    # a tile of 512+pad does not pass
    assert np.allclose(out, x, atol=1e-5)


def test_denoise_looks_at_stop_before_each_tile():
    """The night test pass: Stop waited for the end of a pass of the network (6 min on the processor)."""
    x, nm, bk, sg = _denoise_input()
    rows = []
    dn.apply_net(EchoUNet(), x, nm, bk, sg, torch.device("cpu"), tile=128, shifts=dn.PASSES[:1], check=lambda: rows.append(1))
    assert len(rows) == 12                                         # 300 x 400 in tiles of 128: 3 rows of 4, one look each

    def stop():
        raise RuntimeError("stop requested")
    with pytest.raises(RuntimeError, match="stop"):
        dn.apply_net(EchoUNet(), x, nm, bk, sg, torch.device("cpu"), tile=128, check=stop)


def test_the_denoise_error_passes_through_when_no_tile_helps():
    x, nm, bk, sg = _denoise_input()
    with pytest.raises(torch.OutOfMemoryError):
        dn.apply_net(EchoUNet(limit=16), x, nm, bk, sg, torch.device("cpu"))


class LatticeUNet(torch.nn.Module):
    """A network leaving a pattern on its own grid of 2 px: what the passes of the Denoise block are for (issue #84)."""

    def forward(self, x):
        h, w = x.shape[2:]
        sign = ((-1.0) ** torch.arange(h, device=x.device))[:, None] * ((-1.0) ** torch.arange(w, device=x.device))[None]
        return 0.1 * sign.expand(x.shape[0], 3, h, w).float()


def test_the_passes_of_the_denoise_block_cancel_the_pattern_of_its_network(monkeypatch):
    x, nm, bk, sg = _denoise_input()
    dev = torch.device("cpu")
    alone = dn._apply_net(LatticeUNet(), x, nm, bk, sg, dev, 512, 48)
    out = dn.apply_net(LatticeUNet(), x, nm, bk, sg, dev)
    assert out.shape == x.shape and out.dtype == x.dtype
    assert np.abs(alone - x)[..., 8:-8, 8:-8].max() > 5e-3
    assert np.abs(out - x)[..., 8:-8, 8:-8].max() < 1e-5                 # the pixels stay where they are: an image put back


def test_the_fast_pass_of_the_denoise_block_is_the_network_once(monkeypatch):
    seen = []
    orig = dn._apply_net
    monkeypatch.setattr(dn, "_apply_net", lambda *a, **kw: seen.append(a[6]) or orig(*a, **kw))
    x, nm, bk, sg = _denoise_input()
    dev = torch.device("cpu")
    out = dn.apply_net(LatticeUNet(), x, nm, bk, sg, dev, shifts=dn.PASSES[:1])
    assert seen == [512] and np.array_equal(out, orig(LatticeUNet(), x, nm, bk, sg, dev, 512, 48))
    assert dn.DenoiseParams().fast is False


# ---------------------------------------------------------------------------------------------- a very large stack
def test_denoise_sends_the_card_one_tile_at_a_time_whatever_the_image(monkeypatch):
    """Data set 20 (a 62 Mpx stack): the whole image went to the card at each pass, after a Stack that had filled it.
    Now only the tiles with their margins do, two arrays a tile (the image's, the noise map's), the same at the first
    tile and the last, and the cache of the blocks before is given back first."""
    calls, made = [], []
    orig = dn.torch.from_numpy
    monkeypatch.setattr(dn.devices, "empty_cache", lambda d=None: calls.append("empty"))
    monkeypatch.setattr(dn.devices, "free_memory", lambda d: calls.append("free") or 4 * 2 ** 30)
    monkeypatch.setattr(dn.torch, "from_numpy", lambda a: made.append(a.shape) or orig(a))

    class Counting(EchoUNet):
        def forward(self, x):
            calls.append(len(made))
            return super().forward(x)
    x, nm, bk, sg = _denoise_input()
    out = dn.apply_net(Counting(), x, nm, bk, sg, torch.device("cpu"), tile=128, pad=16)
    assert calls[:2] == ["empty", "free"] and np.allclose(out, x, atol=1e-5)
    per_tile = np.diff([0] + [c for c in calls if isinstance(c, int)])
    assert len(per_tile) == 4 * 3 * 4 and set(per_tile) == {2}           # 4 passes of 3 x 4 tiles (300 x 400, +1 px)
    assert max(s[-2] for s in made) <= 128 + 2 * 16 and max(s[-1] for s in made) <= 128 + 2 * 16


def test_the_denoise_progress_counts_the_tiles_and_never_goes_back():
    x, nm, bk, sg = _denoise_input()
    seen = []
    dn.apply_net(EchoUNet(), x, nm, bk, sg, torch.device("cpu"), tile=128, progress=lambda d, t: seen.append(d / t))
    assert len(seen) == 4 * 3 * 4 and seen == sorted(seen) and seen[-1] == 1.0
    seen.clear()                                     # a memory error starts again with smaller tiles: the bar waits
    dn.apply_net(EchoUNet(limit=320), x, nm, bk, sg, torch.device("cpu"), progress=lambda d, t: seen.append(d / t))
    assert seen and all(b > a for a, b in zip(seen, seen[1:])) and seen[-1] == 1.0


def _apply_net_before(net, x, nm, bk, sg, dev, tile=512, pad=48):
    """The passes as they were until data set 20: the whole image a tensor, the passes summed out of place."""
    total = 0
    for dy, dx in dn.PASSES:
        sx = torch.from_numpy(np.pad(x, ((0, 0), (dy, 0), (dx, 0)), mode="reflect"))
        sn = torch.from_numpy(np.pad(nm, ((dy, 0), (dx, 0)), mode="reflect"))[None]
        Hh, Ww = sx.shape[1:]
        out = np.zeros(sx.shape, np.float32)
        bk_t, sg_t = torch.as_tensor(bk), torch.as_tensor(sg)
        for y in range(0, Hh, tile):
            for x0 in range(0, Ww, tile):
                y1, x1, y2, x2 = max(0, y - pad), max(0, x0 - pad), min(Hh, y + tile + pad), min(Ww, x0 + tile + pad)
                ph, pw = y2 - y1, x2 - x1
                ph8, pw8 = (ph + 7) // 8 * 8, (pw + 7) // 8 * 8
                pt = torch.nn.functional.pad(sx[None, :, y1:y2, x1:x2], (0, pw8 - pw, 0, ph8 - ph), mode="reflect")
                pn = torch.nn.functional.pad(sn[None, :, y1:y2, x1:x2], (0, pw8 - pw, 0, ph8 - ph), mode="reflect")
                o = net(dn.to_input(pt, pn, bk_t, sg_t)).float()
                res = (((pt - bk_t.view(1, 3, 1, 1)) / sg_t.view(1, 3, 1, 1) + o) * sg_t.view(1, 3, 1, 1)
                       + bk_t.view(1, 3, 1, 1))[0, :, :ph, :pw]
                out[:, y:min(y + tile, Hh), x0:min(x0 + tile, Ww)] = \
                    res[:, y - y1:y - y1 + min(tile, Hh - y), x0 - x1:x0 - x1 + min(tile, Ww - x0)].numpy()
        total = total + out[:, dy:, dx:]
    return (total / len(dn.PASSES)).astype(x.dtype)


def test_denoise_gives_the_numbers_of_before():
    torch.manual_seed(0)
    net = dn.UNet(w=(8, 16, 16, 16)).eval()                  # a small network of random weights: every pixel moves
    x, nm, bk, sg = _denoise_input()
    nm = nm * np.random.default_rng(1).uniform(0.8, 1.2, nm.shape).astype(np.float32)
    with torch.no_grad():
        want = _apply_net_before(net, x, nm, bk, sg, torch.device("cpu"), tile=128, pad=16)
    out = dn.apply_net(net, x, nm, bk, sg, torch.device("cpu"), tile=128, pad=16)
    # on the processor the network runs in the channels-last layout, which sums in another order: the same numbers to
    # float rounding (under 1e-6 relative on a real stack), no longer bit for bit
    assert np.abs(out - x).max() > 1e-3 and np.allclose(out, want, rtol=1e-5, atol=1e-6)


class EchoNaf(torch.nn.Module):
    """Stands for the star network: the stretched input unchanged, a memory error above ``limit`` tiles in a batch."""

    def __init__(self, limit=None):
        super().__init__()
        self.limit, self.batches = limit, []

    def forward(self, x):
        self.batches.append(x.shape[0])
        if self.limit is not None and x.shape[0] > self.limit:
            raise torch.OutOfMemoryError("CUDA out of memory")
        return x


def _stars_img():
    return np.random.default_rng(1).normal(0.0, 1.0, (3, 300, 500)).astype(np.float32)


def test_the_stars_network_passes_fewer_tiles_on_a_small_card(monkeypatch):
    monkeypatch.setattr(sr.devices, "free_memory", lambda d: 200 * 2 ** 20)
    net = EchoNaf()
    out = sr.apply(net, _stars_img(), torch.device("cpu"))
    assert max(net.batches) < 16 and np.allclose(out, _stars_img(), atol=1e-3)


def test_the_stars_network_keeps_the_batch_on_a_4_gb_card(monkeypatch):
    monkeypatch.setattr(sr.devices, "free_memory", lambda d: 4 * 2 ** 30)
    net = EchoNaf()
    sr.apply(net, np.pad(_stars_img(), ((0, 0), (0, 0), (0, 3000)), mode="reflect"), torch.device("cpu"))
    assert max(net.batches) == 16


def test_the_stars_network_retries_with_a_smaller_batch_after_a_memory_error():
    net = EchoNaf(limit=4)
    img = np.pad(_stars_img(), ((0, 0), (0, 0), (0, 2000)), mode="reflect")          # more tiles than a batch of 4
    out = sr.apply(net, img, torch.device("cpu"))
    assert 4 in net.batches and np.allclose(out, img, atol=1e-3)




# ---------------------------------------------------------------------------------------------- the estimate of the stack's grid
def _register_result(tmp_path, fwhm):
    d = tmp_path / f"register{len(fwhm)}"
    d.mkdir()
    np.savez(d / "reg.npz", fwhm_moffat=np.array(fwhm, float), rms=np.ones(len(fwhm)))

    class Keys:
        def key(self, wf, stage):
            return "k"

    class Store:
        def done(self, stage, key):
            return True

        def dir(self, stage, key):
            return d
    return Keys(), Store()


def test_well_sampled_stars_measured_by_the_registration_make_the_stack_estimate_a_grid_of_scale_one(tmp_path):
    from astro_gpu.core import estimate as est
    keys, store = _register_result(tmp_path, [2.6, 2.4, 2.8])
    assert est._stars_well_sampled(keys, 1, store, ["register", "stack"])
    assert not est._stars_well_sampled(keys, 1, store, ["stack"])            # no registration in the chain
    assert not est._stars_well_sampled(keys, 1, None, ["register"])          # results not read
    keys, store = _register_result(tmp_path, [1.6, 1.7])
    assert not est._stars_well_sampled(keys, 1, store, ["register"])        # under-sampled: the stack may drizzle


def test_the_same_rows_are_rejected_on_every_device():
    from astro_gpu.core import rejection_classic as rc
    if not torch.cuda.is_available():
        pytest.skip("no card")
    g = torch.Generator().manual_seed(3)
    v = torch.randn(7, 3000, generator=g) * 0.05 + 1.0
    v[torch.rand(7, 3000, generator=g) < 0.03] += 1.0
    valid = torch.rand(7, 3000, generator=g) < 0.9
    ch = rc.choose(7)
    a = rc.reject(v, ch, valid)
    b = rc.reject(v.cuda(), ch, valid.cuda())
    assert all(bool((x == y.cpu()).all()) for x, y in zip(a, b))

