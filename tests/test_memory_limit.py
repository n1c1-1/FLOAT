"""The Settings memory limit reaches the engine (ASTRO_GPU_MEMORY_LIMIT_GB): memory_budget follows it, the stages that
size their work by it take their low-memory ways, and the worker's children inherit it."""
from __future__ import annotations

import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
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


def test_the_pools_shrink_to_what_the_budget_holds_and_say_so(machine, monkeypatch):
    px = 2848 * 4256                                                      # 12 Mpx: 0.63 GB + 0.4 GB a worker
    lines: list[str] = []
    ctx = SimpleNamespace(log=lambda m, level="info": lines.append(m))
    assert cal.pool_size(ctx, 12, px) == 12 and lines == []               # 50 GB available: all of them
    monkeypatch.setenv(devices.ENV_MEMORY_LIMIT, "8")                     # 6 GB to fill: 6 workers
    assert cal.pool_size(ctx, 12, px) == 6
    assert lines == ["memory: 6 workers to stay within 6.0 GB (instead of 12)"]
    monkeypatch.setenv(devices.ENV_MEMORY_LIMIT, "2.5")                   # 0.5 GB to fill: one worker, never none
    assert cal.pool_size(ctx, 12, px) == 1
    monkeypatch.setattr(devices, "memory_budget", lambda: None)
    assert cal.pool_size(ctx, 12, px) == 12                               # memory unknown: as asked


def _fake_frames(tmp_path, n=9, shape=(70, 40)):
    rng = np.random.default_rng(3)
    paths = []
    for i in range(n):
        f = tmp_path / f"f{i}.npy"
        np.save(f, (rng.normal(1000, 30, shape) + (i == 4) * 5000 * (rng.random(shape) < 0.01)).astype(np.float32))
        paths.append(str(f))
    return paths


def test_the_masters_of_a_low_limit_are_the_same_to_the_bit_read_by_bands_from_disk(tmp_path, monkeypatch):
    """The darks' median and the flats' mean with the frames held whole, and with them on the disk (a limit that lets
    no stack in memory): the same arrays, bit for bit, the file left no trace."""
    paths = _fake_frames(tmp_path)
    monkeypatch.setattr(cal, "read_cfa", lambda p: np.load(p))
    logs: list[str] = []
    scratch = tmp_path / "scratch"
    ctx = SimpleNamespace(scratch=scratch, log=lambda m, level="info": logs.append(m),
                          progress=lambda *a, **k: None, project=None)
    fdark = np.full((70, 40), 100, np.float32)
    monkeypatch.setattr(devices, "memory_budget", lambda: (64 * GB, 64 * GB))
    dark_a = cal._median_stack_built(paths, ctx, "darks")
    flat_a = cal._master_flat_built(paths, fdark, ctx)
    assert logs == []
    monkeypatch.setattr(devices, "memory_budget", lambda: (1000, 1000))            # not even one stack fits
    dark_b = cal._median_stack_built(paths, ctx, "darks")
    flat_b = cal._master_flat_built(paths, fdark, ctx)
    assert dark_a.tobytes() == dark_b.tobytes() and flat_a.tobytes() == flat_b.tobytes()
    assert any("kept on the disk and read by bands" in m for m in logs)
    cal.sweep_frame_files()
    assert list(scratch.iterdir()) == []
    assert cal.band_threads(60, 4144) == 1                                          # one band at a time when short
