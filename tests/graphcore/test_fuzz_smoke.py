"""Fuzz smoke: one seed, 300 random gestures from blank and from one fixture (the full fuzz is tools/fuzz/graph_fuzz.py,
run before each merge, 5 seeds x 2,000 gestures from every start)."""
import copy
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools", "fuzz"))
import graph_fuzz  # noqa: E402
from fixtures import Doc  # noqa: E402
from astro_gpu.graphcore import layout  # noqa: E402
from astro_gpu.graphcore.model import model_json  # noqa: E402

#: the smoke's time in units of the yardstick below: about 35 on a quiet machine (17 s for a yardstick of 0.5 s), the
#: bound a little under 3 times that: a gesture made several times slower fails, a loaded machine does not
UNITS = 100


def _yardstick() -> float:
    """The thread time of a fixed piece of the core's own work (a graph copied, serialised and laid out): it slows down
    as the fuzz does when the machine is loaded (shared cores, caches), so the bound follows the machine's state."""
    h = Doc.from_template("sho_rgb")
    t0 = time.thread_time()
    for _i in range(400):
        q = copy.deepcopy(h.project)
        model_json(q)
        layout(q, h.view)
    return time.thread_time() - t0


def test_fuzz_smoke_one_seed_300_gestures():
    # the thread's own time (not the process's: other threads of the process, a machine busy with other work), against
    # the yardstick measured before, between and after (the median: the load may change during the test)
    unit = [_yardstick()]
    t = 0.0
    for key in ("blank", "sho_rgb"):
        t0 = time.thread_time()
        digest = graph_fuzz.fuzz(1, key, 300)
        t += time.thread_time() - t0
        assert digest, f"fuzz failed from {key}"
        unit.append(_yardstick())
    unit = sorted(unit)[1]
    assert t < UNITS * unit, f"{t:.1f} s for {t / unit:.0f} units of {unit:.3f} s (at most {UNITS})"
