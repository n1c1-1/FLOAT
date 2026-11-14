"""The table as the page hosts it (graphview/table_host.py) on the real core: built when first shown, the same document
as the graph, one undo step per gesture with the page's undo, the panel's cell, the Common link, another document."""
from __future__ import annotations

from PySide6.QtCore import Qt

from astro_gpu.graphcore.operations import COMMON
from tests.client.graphview.test_gv_table import Bench, make


def test_gv_host_builds_on_show_and_shares_the_document(qapp, tmp_path):
    from astro_gpu.client.graphview.bridge import Bridge
    from astro_gpu.client.graphview.document import MemoryDocument
    from astro_gpu.client.graphview.table_host import TableHost
    bridge = Bridge(MemoryDocument())
    host = TableHost(bridge)
    assert host.table is None                                         # lazily: nothing before it is shown
    host.resize(1046, 622)
    host.show()
    qapp.processEvents()
    t = host.table
    assert t is not None and t.bridge is bridge and t.grid.doc is bridge.doc
    assert t.grid.undo_delegated and not any(sc.isEnabled() for sc in t._shortcuts)    # Ctrl+Z is the page's
    host.hide()


def test_gv_host_undo_from_the_page_and_both_ways(qapp, tmp_path):
    night = make(tmp_path / "night" / "LIGHT", 4, "L")
    night2 = make(tmp_path / "night2", 3, "L")
    b = Bench(1366, blank=True, host=True)
    bridge = b.bridge
    n0 = b.steps()
    b.drop([night], ("body",))
    assert b.steps() == n0 + 1 and len(b.stacks()) == 1 and len(b.sessions()) == 1
    sid = b.sessions()[0]
    assert b.own(sid, "lights") == [str(night).replace("\\", "/")]
    rid = b.stacks()[0]["id"]
    b.drop([night2], ("ghost_s", rid))
    assert b.steps() == n0 + 2 and len(b.sessions()) == 2
    b.key(Qt.Key_Z, Qt.ControlModifier)                               # delegated: the grid leaves it to the page
    assert b.steps() == n0 + 2
    bridge.undo()                                                     # the page's Ctrl+Z
    assert len(b.sessions()) == 1
    bridge.undo()
    assert len(b.stacks()) == 0 and b.g.cell(("body",)) is not None   # the table follows
    bridge.redo()
    assert len(b.sessions()) == 1
    s0 = b.steps()
    b.g.act([("add_night", dict(row=b.stacks()[0]["id"]))])           # a table gesture = one step, seen by the graph
    assert b.steps() == s0 + 1 and len(bridge.graph.rows[0].nights) == 2
    bridge.act("add_night", row=bridge.graph.rows[0].id)              # a graph gesture: a row of the table
    assert len(b.sessions()) == 3
    bridge.undo()
    assert len(b.sessions()) == 2
    b.close()


def test_gv_host_cells_go_to_the_panel(qapp):
    b = Bench(1366, host=True)
    bridge = b.bridge
    chosen = []
    bridge.files_chosen.connect(lambda: chosen.append(bridge.file_cell))
    sid = b.sessions()[0]
    b.click(("box", sid, "lights"))
    assert bridge.file_cell == (sid, "lights") and chosen == [(sid, "lights")]
    cal = b.g.model.calibration_of(sid)
    b.click(("cal", sid))
    assert bridge.selected == cal and bridge.file_cell is None
    rid = b.stacks()[0]["id"]
    b.click(("setting", rid, "stk"))
    stk = b.g.block_of(b.g.cell(("setting", rid, "stk")))
    assert bridge.selected == stk and bridge.doc.graph.blocks[stk].kind == "stacking"
    b.click(("name", sid))
    assert bridge.selected == sid
    b.click(("stack", rid))
    assert bridge.selected == rid
    bridge.act("add_variant", block=stk)
    vs = b.g.model.variants(rid)
    assert len(vs) == 1 and b.g.cell(("variant", vs[0]["id"])) is not None
    b.click(("variant", vs[0]["id"]))
    assert bridge.selected == vs[0]["block"]
    bridge.undo()
    assert not b.g.model.variants(rid)
    b.key(Qt.Key_Escape)                                              # the selection cleared: nothing in the panel
    assert bridge.selected is None
    # « See the Common cell »: the Common cell chosen, its files shown
    assert bridge.go_common is not None
    bridge.go_common("darks")
    assert bridge.file_cell == (COMMON, "darks") and ("box", COMMON, "darks") in b.g.sel
    b.close()


def test_gv_host_columns_at_both_sizes(qapp):
    for width in (1366, 1920):
        b = Bench(width, host=True)
        g = b.g
        cols = list(g.xs)
        assert cols.index("bias") == cols.index("flats") + 1 and "flat_darks" not in g.xs
        assert not g.horizontalScrollBar().isVisible() and sum(w for _x, w in g.xs.values()) == g.viewport().width()
        assert min(w for _x, w in g.xs.values()) >= 70
        b.t.column("flat_darks", True)
        assert "flat_darks" in g.xs and sum(w for _x, w in g.xs.values()) == g.viewport().width()
        b.close()


def test_gv_host_common_setting_one_step(qapp):
    b = Bench(1366, host=True)
    n = b.steps()
    b.g.act([("set_common_setting", dict(kind="stacking", key="Rejection threshold", value="5 σ"))])
    assert b.steps() == n + 1
    t, _own, _k, _b = b.g.cell_text(b.g.cell(("setting", COMMON, "stk")))
    assert t == "drizzle auto · rejection at 5 σ"
    b.close()


def test_gv_host_another_document(qapp):
    from astro_gpu.client.graphview.document import MemoryDocument
    from astro_gpu.graphcore import api
    b = Bench(1366, host=True)
    old = b.host.table
    g, _r = api.apply(api.blank_project(), "add_row", kind="session")
    g, _r = api.apply(g, "add_night", row=g.rows[0].id)
    b.bridge.install(MemoryDocument(g))
    t = b.host.table
    assert t is not old and len([s for pl in t.grid.model.stacks() for s in pl["sessions"]]) == 2
    b.close()
