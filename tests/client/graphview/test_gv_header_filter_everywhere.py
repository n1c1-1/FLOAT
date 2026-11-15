"""Every way of adding lights reads the filter their FITS header names (a beta tester: frames dropped on the Table's
« + Stack » row or on a Session block kept « none (colour camera) » while « Add files… » read it, so Stacking offered
colour debayering on monochrome narrowband frames). Flats whose header names another filter than the session's
lights are asked about, as lights are; a Combination input fed by frames of another filter is said on its wire, its
port and in the Health check, never refused. Only headers are read: the files' names say the opposite on purpose."""
from __future__ import annotations

import numpy as np

from astro_gpu.client.graphview import files
from astro_gpu.core.project import stage_graph
from astro_gpu.core.workflow import chains
from tests.client.graphview.test_gv_files import Toasts
from tests.client.graphview.test_gv_table import Bench, fwd

stage_graph()
chains()


def frames(folder, n, flt, kind="LIGHT", name="BLUE"):
    from astropy.io import fits
    folder.mkdir(parents=True, exist_ok=True)
    for k in range(n):
        h = fits.PrimaryHDU(np.zeros((4, 4), np.float32))
        h.header["FILTER"] = flt
        h.header["IMAGETYP"] = kind
        h.writeto(folder / f"{name}_{k}.fits")                 # a misleading name: never read
    return folder


def night_filter(b, block):
    return files.session_filter(b.doc.graph, block)


def test_lights_dropped_on_plus_stack_and_on_a_new_session_take_their_header_filter(qapp, tmp_path):
    ha, oiii = frames(tmp_path / "a", 3, "Ha"), frames(tmp_path / "b", 2, "OIII")
    b = Bench(1366, blank=True)
    b.drop([ha], ("ghost_p",))                                 # « + Stack »: a new stack
    s1 = b.sessions()[0]
    assert night_filter(b, s1) == "Ha"
    ha2 = frames(tmp_path / "c", 2, "H-alpha")
    b.drop([oiii, ha2], ("ghost_p",))                          # two folders: two sessions of a new stack, each its own
    new = b.sessions()[1:]
    assert sorted(night_filter(b, s) for s in new) == ["Ha", "OIII"]
    b.close()


def test_lights_dropped_on_an_empty_session_block_take_their_header_filter(qapp, tmp_path):
    red = frames(tmp_path / "r", 3, "Red")
    b = Bench(1366, blank=True)
    b.bridge.act("add_row", kind="session")
    s = next(x for x, blk in b.doc.graph.blocks.items() if blk.kind == "session")
    assert night_filter(b, s) == ""
    b.bridge.drops.drop(s, "lights", [fwd(red)])
    assert night_filter(b, s) == "R" and len(b.doc.files_of(s, "lights")[1]) == 3
    n = b.steps()
    b.bridge.undo()                                            # one undo step: frames and filter
    assert night_filter(b, s) == "" and not b.doc.files_of(s, "lights")[1] and b.steps() == n - 1
    b.close()


def test_a_filter_chosen_by_hand_is_kept_and_a_colour_session_asks(qapp, tmp_path):
    red, green = frames(tmp_path / "r", 2, "R"), frames(tmp_path / "g", 2, "G")
    b = Bench(1366, blank=True)
    b.bridge.act("add_row", kind="session")
    s = next(x for x, blk in b.doc.graph.blocks.items() if blk.kind == "session")
    b.bridge.drops.drop(s, "lights", [fwd(red)])
    b.bridge.act("set_night_filter", night=s, filter="")      # set back to « none » by hand, lights there
    toasts = Toasts(b.bridge)
    assert b.bridge.drops.drop(s, "lights", [fwd(green)]) is None          # asked, nothing added before
    assert "holds a colour camera's" in toasts.text and len(b.doc.files_of(s, "lights")[1]) == 2
    b.bridge.drops.link("filter_here")
    assert night_filter(b, s) == "" and len(b.doc.files_of(s, "lights")[1]) == 4
    b.close()


def test_flats_of_another_filter_are_asked_about(qapp, tmp_path):
    sii = frames(tmp_path / "l", 2, "SII")
    flat = frames(tmp_path / "f", 1, "OIII", kind="Master Flat", name="SII_flat")
    b = Bench(1366, blank=True)
    b.drop([sii], ("ghost_p",))
    s = b.sessions()[0]
    n = b.steps()
    b.drop([flat], ("box", s, "flats"))
    assert b.steps() == n and "These flats' header names filter OIII" in b.t.toast_text, b.t.toast_text
    b.t.link("filter_here")
    assert len(b.own(s, "flats")) == 1 and b.steps() == n + 1
    b.g.undo()
    toasts = Toasts(b.bridge)
    assert b.bridge.drops.drop(s, "flats", [fwd(flat)]) is None
    assert "These flats' header names filter OIII" in toasts.text and "filter_own" not in toasts.keys
    b.bridge.drops.link("cancel")
    assert not b.doc.files_of(s, "flats")[1] or b.doc.files_of(s, "flats")[0] != "own"
    good = frames(tmp_path / "f2", 1, "SII", kind="Master Flat", name="OIII_flat")
    b.bridge.drops.drop(s, "flats", [fwd(good)])                # its own filter: no question
    assert b.doc.files_of(s, "flats")[1] == files.engine_frames([fwd(good)])
    b.close()


def test_a_combination_input_fed_by_another_filter_is_said_not_refused(qapp, tmp_path):
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
    assert res.ok, res.reason                                  # accepted: the user may mean it
    w = next(x for x in b.doc.graph.wires if tuple(x.dst) == (j, "S II"))
    assert "Its frames' filter is R, this input is for SII" in b.doc.wire_tip(w.id)
    assert b.doc.port_warning(j, "S II") and not b.doc.port_warning(j, "Hα")
    assert [(x, p) for x, p, _t in files.channel_mismatches(b.doc.graph)] == [(j, "S II")]
    from astro_gpu.client.frames import check_lines
    data = type("D", (), {"checks": []})()
    model = type("M", (), {"_shown": b.doc.graph})()
    from astro_gpu.client.frames import channel_lines
    lines = check_lines(data, channel_lines(model))
    assert lines and lines[0][0] == "warn" and "“Mix” S II" in lines[0][1]
    b.close()
