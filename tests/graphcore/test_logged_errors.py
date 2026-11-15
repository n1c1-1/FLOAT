"""An error the graph's checks pass over is still written to the log (the wiring the engine reads, left empty)."""
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from astro_gpu.graphcore import compile as cp  # noqa: E402
from astro_gpu.graphcore.model import blank_project  # noqa: E402


def test_a_graph_the_wiring_check_cannot_read_is_logged(monkeypatch, caplog):
    def broken(g, p):
        raise ValueError("unreadable")
    monkeypatch.setattr(cp, "compile_into", broken)
    with caplog.at_level(logging.WARNING, logger=cp.__name__):
        assert cp.engine_wiring(blank_project()) == {}             # the behaviour kept: nothing left out
    assert any("unreadable" in r.getMessage() for r in caplog.records)
