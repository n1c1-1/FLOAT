"""The page's file reader ends with its page: its worker thread held the page's bridge, and so the project, after the
project was closed (one more project kept in memory at each opening)."""
import gc
import time
import weakref

import shiboken6
from PySide6.QtWidgets import QApplication

from tests.client.graphview.test_gv_table import fwd, make


def test_the_reader_of_a_page_gone_ends_and_lets_its_project_go(qapp, tmp_path):
    from astro_gpu.client.graphview.bridge import Bridge
    from astro_gpu.client.graphview.document import MemoryDocument
    from astro_gpu.graphcore import api
    folder = fwd(make(tmp_path / "lights", 2, "L"))
    g, _r = api.apply(api.blank_project(), "add_row", kind="session")
    document = MemoryDocument(g)
    bridge = Bridge(document)
    reader = bridge.drops.reader
    bridge.drops.read(g.rows[0].nights[0][0], "lights", [folder])
    assert reader.wait()
    thread = reader._thread
    assert thread is not None and thread.is_alive()
    gone = weakref.ref(document)
    shiboken6.delete(bridge)                                   # the page and its bridge deleted (project closed)
    del bridge, document, reader
    t0 = time.perf_counter()
    while thread.is_alive() and time.perf_counter() - t0 < 5:
        QApplication.processEvents()
        time.sleep(0.01)
    gc.collect()
    assert not thread.is_alive()
    assert gone() is None
