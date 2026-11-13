"""The Settings memory limit reaches the engine (ASTRO_GPU_MEMORY_LIMIT_GB): memory_budget follows it, the stages that
size their work by it take their low-memory ways, and the worker's children inherit it."""
from __future__ import annotations

import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from astro_gpu.core import devices, runner2
from astro_gpu.stages import calibrate as cal

GB = 2 ** 30


@pytest.fixture
def machine(monkeypatch):
    """A machine of 64 GB with 50 GB available, and a run that holds 2 GB."""
    pytest.importorskip("psutil")
    import psutil
    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(total=64 * GB, available=50 * GB))
    monkeypatch.setattr(devices, "_own_tree_bytes", lambda: 2 * GB)
    monkeypatch.delenv(devices.ENV_MEMORY_LIMIT, raising=False)


def test_without_a_limit_the_budget_is_the_machines(machine):
    fill, avail = devices.memory_budget()
    assert avail == 50 * GB and fill == int(50 * GB - 0.15 * 64 * GB)
    assert devices.memory_limit() is None


@pytest.mark.parametrize("bad", ["", "0", "-3", "abc"])
def test_a_missing_or_wrong_limit_is_no_limit(machine, monkeypatch, bad):
    monkeypatch.setenv(devices.ENV_MEMORY_LIMIT, bad)
    assert devices.memory_limit() is None and devices.memory_budget()[1] == 50 * GB


def test_the_limit_less_what_the_run_holds_caps_the_budget(machine, monkeypatch):
    monkeypatch.setenv(devices.ENV_MEMORY_LIMIT, "8")
    assert devices.memory_budget() == (6 * GB, 6 * GB)
    monkeypatch.setenv(devices.ENV_MEMORY_LIMIT, "100")                   # above the machine's: the machine's rules
    assert devices.memory_budget()[1] == 50 * GB
    monkeypatch.setenv(devices.ENV_MEMORY_LIMIT, "1")                     # already past it: nothing to fill
    assert devices.memory_budget() == (0, 0)


def test_the_cpu_tiles_know_the_limit_only_when_there_is_one(machine, monkeypatch):
    import torch
    cpu = torch.device("cpu")
    assert devices.free_memory(cpu) is None
    monkeypatch.setenv(devices.ENV_MEMORY_LIMIT, "8")
    assert devices.free_memory(cpu) == 6 * GB


def test_calibration_and_the_stack_take_their_low_memory_ways_under_a_low_limit(machine, monkeypatch):
    cache, workers, px = 18 * GB, 12, 2848 * 4256
    assert cal.memory_short(cache, workers, px) is None                   # 50 GB available: through the maps
    monkeypatch.setenv(devices.ENV_MEMORY_LIMIT, "8")
    assert cal.memory_short(cache, workers, px) == pytest.approx(6.0)     # plain writes, the room said in GB
    monkeypatch.setenv(devices.ENV_MEMORY_LIMIT, "24")
    assert cal.memory_short(cache, workers, px) is not None               # 22 GB to fill: 18 + 7 GB do not fit


def test_the_limit_reaches_the_worker_and_its_children(monkeypatch):
    monkeypatch.setenv(devices.ENV_MEMORY_LIMIT, "8")
    seen = {}

    class P:
        pid = 1

        def __init__(self, cmd, **kw):
            seen["env"] = kw["env"]

        def wait(self):
            return 0

    monkeypatch.setattr(subprocess, "Popen", P)
    runner2._spawn_worker(".", "plan.json", "0", os.devnull)
    assert seen["env"][devices.ENV_MEMORY_LIMIT] == "8"
    job = SimpleNamespace(key="k")
    obj = SimpleNamespace(root=".", plan_file=__file__, store=SimpleNamespace(forget=lambda: None))
    try:
        runner2._Worker._run_in_child(obj, job)
    except Exception:  # noqa: BLE001 - only the environment handed to the child matters here
        pass
    assert seen["env"][devices.ENV_MEMORY_LIMIT] == "8"


def test_a_real_child_process_reads_the_limit(monkeypatch):
    monkeypatch.setenv(devices.ENV_MEMORY_LIMIT, "8")
    out = subprocess.run([sys.executable, "-c", "from astro_gpu.core import devices; print(devices.memory_limit())"],
                         capture_output=True, text=True, env={**os.environ}).stdout.strip()
    assert out == str(8 * GB)
