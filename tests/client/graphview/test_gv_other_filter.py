"""Frames whose FITS header names another filter than the session or stack they are dropped on (Green frames dropped
into the stack R): the page made a session « R 2 » of them, filed under R. Now the drop says the two filters and the
user chooses: add here (named after the frames' own filter, never after the stack's), a session of their own filter
in a stack of its own, or cancel. Only the header's value is read, never a file or folder name."""
from __future__ import annotations

import numpy as np

from astro_gpu.core.project import stage_graph
from tests.client.graphview.test_gv_files import Toasts
from tests.client.graphview.test_gv_table import Bench, fwd

stage_graph()             # the engine's stages loaded at collection, not in the first test's time


def frames(folder, n, flt):
    from astropy.io import fits
    folder.mkdir(parents=True, exist_ok=True)
    for k in range(n):
        h = fits.PrimaryHDU(np.zeros((4, 4), np.float32))
        h.header["FILTER"] = flt
        h.writeto(folder / f"RED_{k}.fits")          # a misleading name: never read
    return folder


def r_stack():
    b = Bench(1366, blank=True)
    b.bridge.act("new_night", name="R", filter="R")
    sid = max(n["sid"] for n in b.doc.graph.nights)
    b.bridge.act("add_stack", name="R", nights=[sid])
    b.g.after_model()
    b.wait(0.02)
    return b


def test_gv_other_filter_as_a_new_session_of_the_stack(qapp, tmp_path):
    green = frames(tmp_path / "night", 3, "Green")
    b = r_stack()
    rid = b.stacks()[0]["id"]
    n0, ss0 = b.steps(), b.sessions()
    b.drop([green], ("ghost_s", rid))
    assert b.steps() == n0 and b.sessions() == ss0                    # nothing before the user's choice
    assert b.t.toast_text == "These frames' header names filter G, “R” is R: where do they go?"
    assert "filter_here" in b.t.toast_w.text() and "filter_own" in b.t.toast_w.text()
    b.t.link("filter_here")
    new = [s for s in b.sessions() if s not in ss0]
    assert len(new) == 1 and b.g.model.session_name(new[0]) == "G" and b.steps() == n0 + 1
    b.g.undo()
    b.drop([green], ("ghost_s", rid))
    assert "names filter G" in b.t.toast_text, b.t.toast_text
    b.t.link("filter_own")                                             # a G session in a stack of its own
    b.g.model.invalidate()
    g = [pl for pl in b.stacks() if pl["name"] == "G"]
    assert len(g) == 1 and len(b.own(g[0]["sessions"][0]["sid"], "lights")) == 3
    assert b.stacks()[0]["name"] == "R" and len(b.stacks()[0]["sessions"]) == 1
    b.drop([green], ("ghost_s", rid))
    b.t.link("cancel")
    assert len(b.stacks()[0]["sessions"]) == 1
    b.close()


def test_gv_other_filter_on_a_session_cell_and_block(qapp, tmp_path):
    green = frames(tmp_path / "night", 2, "G")
    red = frames(tmp_path / "red", 2, "R")
    b = r_stack()
    sid = b.sessions()[0]
    b.drop([red], ("box", sid, "lights"))                              # its own filter: no question
    assert len(b.own(sid, "lights")) == 2
    n0 = b.steps()
    b.drop([green], ("box", sid, "lights"))
    assert b.steps() == n0 and "names filter G" in b.t.toast_text
    b.t.link("filter_here")
    assert len(b.own(sid, "lights")) == 4
    toasts = Toasts(b.bridge)
    b.g.undo()
    assert b.bridge.drops.drop(sid, "lights", [fwd(green)]) is None    # a Session block's target
    assert "names filter G" in toasts.text and "filter_own" in toasts.keys
    b.bridge.drops.link("filter_own")
    b.g.after_model()
    b.g.model.invalidate()
    assert [pl["name"] for pl in b.stacks()] == ["R", "G"] and len(b.own(sid, "lights")) == 2
    b.close()
