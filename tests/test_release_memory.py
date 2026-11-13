"""devices.release_memory: the freed pages a step leaves in the worker go back after it (a drizzle stack left 7 GB)."""
import sys

import numpy as np
import psutil
import pytest

from astro_gpu.core import devices


def test_it_never_raises_and_keeps_the_arrays():
    a = np.arange(1000, dtype=np.float32)
    devices.release_memory()
    assert a.sum() == pytest.approx(499500.0)


@pytest.mark.skipif(sys.platform != "win32", reason="the working set is trimmed on Windows")
def test_the_working_set_is_trimmed():
    keep = np.ones(150_000_000 // 4, np.float32)           # touched: in the working set
    before = psutil.Process().memory_info().rss
    devices.release_memory()
    assert psutil.Process().memory_info().rss < before - 100e6
    assert keep[::1000].sum() > 0                          # still there (paged back on touch)
