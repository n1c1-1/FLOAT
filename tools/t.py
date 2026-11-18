"""Run the tests at one of the four levels (tests/README.md has the table).

    python tools/t.py 1 tests/test_x.py [pytest args]   one file (or a few), in this process: under 1 min
    python tools/t.py 2 [pytest args]                    the fast suite, parallel: under 3 min (the default of ``pytest``)
    python tools/t.py 3 [pytest args]                    level 2 + the heavy tests (tests/levels.toml): under 20 min
    python tools/t.py 4 [pytest args]                    only the slow tests (real data sets): hours, never by default

Options (before the pytest arguments): ``-j N`` sets the number of parallel workers (default: the ASTRO_TEST_WORKERS
variable, else three quarters of the cores); ``-j 0`` runs in this process.

What it sets, and why: CUDA_VISIBLE_DEVICES=-1 (on this machine an EMPTY value does not hide the GPUs, only -1 does),
QT_QPA_PLATFORM=offscreen, one BLAS/OpenMP thread per worker so that the workers do not fight over the cores,
ASTRO_TEST_LEVEL for the time budget printed at the end and ASTRO_TEST_TIMEOUT, the seconds after which a test that
waits for ever is killed with its traceback (tests/conftest.py). Every pytest-xdist worker has its own
private ASTRO_GPU_HOME and temporary folder (tests/conftest.py).

The tests marked ``alone`` (Post's speed gate, tests/client/test_post_speed.py: times against thresholds) never run
beside the parallel workers: a parallel run gives them the processor the workers leave (2.5 s for a 1.6 s release, the
gate failed). They run after the parallel part, one at a time in one process, with the library threads as a user has
them, so that the gate is always checked, and checked on a quiet machine.
"""
from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: the -m of each level; level 2 is also the addopts of pyproject.toml, level 3 adds what tests/levels.toml sets aside
MARKS = {
    "1": "not level3 and not slow and not network",
    "2": "not level3 and not slow and not network",
    "3": "not slow and not network",
    "4": "slow and not network",
}
#: level 4 runs one test at a time (the slow tests use the machine's cores and disks by themselves)
PARALLEL = {"1": False, "2": True, "3": True, "4": False}
DIST = "load"
#: seconds one test may run before its process is killed with a traceback (tests/conftest.py); "0" = no limit
TIMEOUT = {"1": "120", "2": "120", "3": "600", "4": "0"}


def free_memory_gb() -> tuple[float, float]:
    """(memory that can still be committed, physical memory available), in GB; (inf, inf) when unknown (not Windows)."""
    if sys.platform != "win32":
        return float("inf"), float("inf")
    import ctypes

    class Status(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total_phys", ctypes.c_ulonglong),
                    ("avail_phys", ctypes.c_ulonglong), ("total_commit", ctypes.c_ulonglong),
                    ("avail_commit", ctypes.c_ulonglong), ("total_virtual", ctypes.c_ulonglong),
                    ("avail_virtual", ctypes.c_ulonglong), ("avail_extended", ctypes.c_ulonglong)]
    st = Status()
    st.length = ctypes.sizeof(Status)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
    return st.avail_commit / 2**30, st.avail_phys / 2**30


def workers(level: str, asked: int | None) -> int:
    if asked is not None:
        return asked
    if not PARALLEL[level]:
        return 0
    if os.environ.get("ASTRO_TEST_WORKERS"):
        return int(os.environ["ASTRO_TEST_WORKERS"])
    n = max(2, (os.cpu_count() or 4) * 3 // 4)          # 12 on 16 cores: the rest is for the engine's own workers and the OS
    # a worker imports torch + Qt: about 1.5 GB committed, up to 3 GB while it runs. When the machine is short of memory
    # to commit (a bench, a project run...), Windows fails the imports ("paging file is too small", WinError 1455)
    commit, phys = free_memory_gb()
    fit = max(2, int(min(commit / 3.0, phys / 2.0)))
    if fit < n:
        print(f"only {commit:.0f} GB can still be committed and {phys:.0f} GB are free: {fit} workers instead of {n}")
    return min(n, fit)


def main(argv: list[str]) -> int:
    if not argv or argv[0] not in MARKS:
        print(__doc__)
        return 2
    level, rest = argv[0], argv[1:]
    asked = None
    if rest[:1] == ["-j"] and len(rest) >= 2:
        asked, rest = int(rest[1]), rest[2:]
    if level == "1" and not any(not a.startswith("-") for a in rest):
        print("level 1 is one test file: python tools/t.py 1 tests/test_x.py   (the whole fast suite is level 2)")
        return 2
    n = workers(level, asked)
    if n > 0:
        try:
            import xdist  # noqa: F401
        except ImportError:
            print("pytest-xdist is not installed (python -m pip install pytest-xdist; with uv: uv pip install "
                  "pytest-xdist): running in this process")
            n = 0

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "-1"
    env["QT_QPA_PLATFORM"] = "offscreen"
    env["ASTRO_TEST_LEVEL"] = level
    env.setdefault("ASTRO_TEST_TIMEOUT", TIMEOUT[level])
    env.setdefault("PYTHONIOENCODING", "utf-8")
    if n > 0:       # one thread per worker: torch and BLAS pools of several threads in 12 workers slow each other down
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            env.setdefault(var, "1")

    low = os.environ.get("ASTRO_TEST_LOW_PRIORITY") == "1" and sys.platform == "win32"
    # ASTRO_TEST_LOW_PRIORITY=1: below-normal priority, the tests' workers included (a training shares the machine)
    flags = 0x4000 if low else 0
    if n == 0:
        cmd = [sys.executable, "-m", "pytest", "-m", MARKS[level], "-q", "-p", "no:cacheprovider"]
        cmd += (["-p", "no:xdist"] if _has_xdist() else []) + rest
        print(f"level {level}: one process  |  " + shlex.join(cmd[2:]), flush=True)
        return subprocess.call(cmd, cwd=ROOT, env=env, creationflags=flags)
    cmd = [sys.executable, "-m", "pytest", "-m", f"({MARKS[level]}) and not alone", "-q", "-p", "no:cacheprovider",
           "-n", str(n), "--dist", DIST] + rest
    print(f"level {level}: {n} workers  |  " + shlex.join(cmd[2:]), flush=True)
    first = subprocess.call(cmd, cwd=ROOT, env=env, creationflags=flags)
    # then the tests that time themselves, alone, the machine quiet (the library threads as a user has them)
    quiet = {k: v for k, v in env.items() if k not in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                                       "NUMEXPR_NUM_THREADS") or k in os.environ}
    cmd = [sys.executable, "-m", "pytest", "-m", f"({MARKS[level]}) and alone", "-q", "-p", "no:cacheprovider"]
    cmd += (["-p", "no:xdist"] if _has_xdist() else []) + rest
    print(f"level {level}, the timed tests alone: one process  |  " + shlex.join(cmd[2:]), flush=True)
    second = subprocess.call(cmd, cwd=ROOT, env=quiet, creationflags=flags)
    return combined(first, second)


def combined(*codes: int) -> int:
    """pytest's exit codes of the two parts as one: a failure in either; "no test collected" (5) only when neither part
    had any (a file named that holds no timed test is no failure)."""
    real = [c for c in codes if c != 5]
    if not real:
        return 5
    bad = [c for c in real if c != 0]
    return bad[0] if bad else 0


def _has_xdist() -> bool:
    try:
        import xdist  # noqa: F401
    except ImportError:
        return False
    return True


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
