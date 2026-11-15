"""A Combination input is a role: a workflow of another filter wired into it is a normal choice, never judged; its
port says what is wired, as a fact."""
from __future__ import annotations

import numpy as np

from astro_gpu.core.project import stage_graph
from astro_gpu.core.workflow import chains
from tests.client.graphview.test_gv_table import Bench

stage_graph()
chains()


def frames(folder, n, flt):
    from astropy.io import fits
    folder.mkdir(parents=True, exist_ok=True)
    for k in range(n):
        h = fits.PrimaryHDU(np.zeros((4, 4), np.float32))
        h.header["FILTER"] = flt
        h.writeto(folder / f"L_{k}.fits")
    return folder


def test_a_combination_input_fed_by_another_filter_is_never_judged(qapp, tmp_path):
    red = frames(tmp_path / "r", 2, "R")
    b = Bench(1366, blank=True)
    b.drop([red], ("ghost_p",))
    g = b.doc.graph
    s = b.sessions()[0]
    row = next(r for r in g.rows if any(n[0] == s for n in r.nights))
    b.bridge.act("add_block", kind="stacking", row=row.id)
    b.bridge.act("add_combination", type="sho", name="Mix")
    g = b.doc.graph
    j = next(x for x, blk in g.blocks.items() if blk.kind == "combine")
    src = next(r for r in g.rows if r.id == row.id).blocks[-1]
    res = b.bridge.act("wire", src=src, src_port="image", dst=j, dst_port="S II")
    assert res.ok, res.reason                                  # a role: the user's choice
    g = b.doc.graph
    w = next(x for x in g.wires if tuple(x.dst) == (j, "S II"))
    name = next(r for r in g.rows if r.id == row.id).name
    tip = b.doc.wire_tip(w.id)
    assert "filter" not in tip and "check" not in tip            # the wire says nothing of filters
    assert b.doc.port_source(j, "S II") == name and b.doc.port_source(j, "Hα") == ""
    from astro_gpu.client.graphview.canvas import GraphCanvas
    cv = GraphCanvas(b.bridge)
    d = b.bridge.drawing()
    block_tip = cv.block_data(j, d.boxes[j])["tip"]
    assert f"Input S II ← {name}" in block_tip and "check the wire" not in block_tip
    cv.deleteLater()
    from astro_gpu.client.frames import check_lines
    data = type("D", (), {"checks": []})()
    assert check_lines(data) == []                             # the Health check says nothing of it
    b.close()
