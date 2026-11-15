"""A camera's JPEGs beside its raw frames, picked or dropped as single files: the graph page stores files only, and the
engine takes a file entry as it is (``list_frames``), so a JPEG stored beside raws would be read as one more frame
(the inventory then keeps the pattern of the most frames: the JPEGs and not the raws when they are more). Whatever the
filter, they are left out at the gesture, said once with « Take all »; the stored entries and the engine's listing
agree with what the toast says."""
from __future__ import annotations

from astro_gpu.core.project import list_frames, stage_graph
from tests.client.graphview.test_gv_files import Toasts, one_session_bridge, settle
from tests.client.graphview.test_gv_table import Bench, fwd, make

stage_graph()             # the engine's stages loaded at collection, not in the first test's time


def pairs(folder, n=8):
    make(folder, n, "DSC", "ARW")
    make(folder, n, "DSC", "JPG")
    return sorted(fwd(x) for x in folder.iterdir())


def engine(lst):
    return [fwd(x) for x in list_frames(lst)]


def test_gv_jpeg_single_files_dropped_on_a_session_are_left(qapp, tmp_path):
    from astro_gpu.client.graphview import files
    picked = pairs(tmp_path / "night")
    for i in (2, 3):                                     # the default filter, then « All images » (the dialog's)
        files.set_index(i)
        bridge, sid = one_session_bridge()
        toasts = Toasts(bridge)
        bridge.drops.drop(sid, "lights", picked)          # also the settings panel's rows (host.panel_drop)
        settle(bridge.drops.reader)
        lst = bridge.doc.files_of(sid, "lights")[1]
        assert len(lst) == 8 and all(x.endswith(".ARW") for x in lst) and len(engine(lst)) == 8
        assert toasts.text.startswith("8 ARW added · 8 JPG left") and "take_all" in toasts.keys
        assert bridge.drops.drop(sid, "lights", [x for x in picked if x.endswith(".JPG")]) is None   # alone, later:
        assert "8 JPG left" in toasts.text                                                         # still beside raws
        bridge.drops.link("take_all")                     # the user's explicit choice
        assert len(engine(bridge.doc.files_of(sid, "lights")[1])) == 16
    files.set_index(2)


def test_gv_jpeg_table_picker_folder_and_new_session(qapp, tmp_path):
    from astro_gpu.client.graphview import files
    picked = pairs(tmp_path / "night")
    files.set_index(3)                                    # « All images » chosen in the file dialog
    b = Bench(1366)
    sid = b.sessions()[0]
    b.g.choose_files = lambda title: list(picked)         # « Add… » of the cell: 16 files selected in the dialog
    b.g.picker(b.g.cell(("box", sid, "lights")))
    b.read()
    lst = b.own(sid, "lights")
    assert len(lst) == 8 and len(engine(lst)) == 8 and b.box(sid, "lights").n == 8
    assert "8 ARW added · 8 JPG left" in b.t.toast_text and "take_all" in b.t.toast_w.text()
    b.close()
    b = Bench(1366)                                       # a folder dropped
    sid = b.sessions()[0]
    b.drop([tmp_path / "night"], ("box", sid, "lights"))
    b.read()
    assert len(b.own(sid, "lights")) == 8 and "8 JPG left" in b.t.toast_text
    b.close()
    b = Bench(1366)                                       # loose files made a new session
    n = len(b.sessions())
    b.drop([tmp_path / "night" / x.rsplit("/", 1)[1] for x in picked], ("ghost_s", b.stacks()[0]["id"]))
    b.read()
    new = b.sessions()[n]
    assert len(b.own(new, "lights")) == 8 and len(engine(b.own(new, "lights"))) == 8 and "8 JPG left" in b.t.toast_text
    b.close()
    files.set_index(2)


def test_gv_jpeg_saved_beside_raws_is_said_with_a_one_click_removal(qapp, tmp_path):
    """A session saved since files only with JPEGs beside its raws: the table's cell and the files panel say it; one
    click takes the JPEGs out (one undo step); nothing is removed silently."""
    from astro_gpu.client.graphview.files_panel import FilesPanel
    picked = pairs(tmp_path / "night")
    b = Bench(1366)
    sid = b.sessions()[0]
    b.bridge.act("add_files", session=sid, kind="lights", paths=picked, mode="add")   # as saved before
    b.read()
    assert len(engine(b.own(sid, "lights"))) == 16                    # the engine would read 16 frames
    b.g.model.invalidate()
    w = b.g.model.box(sid, "lights").warn
    assert w is not None and w.text == "8 JPG beside raw frames: not usual, each read as one more frame"
    assert w.button == "Remove JPG" and len(b.own(sid, "lights")) == 16          # said, not removed
    p = FilesPanel(b.bridge, sid, "lights")
    assert not p.beside.isHidden() and p.b_beside.text() == "Remove JPG" and p.beside.text() == w.text
    n0 = b.steps()
    p.remove_beside()
    lst = b.own(sid, "lights")
    assert len(lst) == 8 and all(x.endswith(".ARW") for x in lst) and b.steps() == n0 + 1
    assert p.beside.isHidden()
    b.g.model.invalidate()
    assert b.g.model.box(sid, "lights").warn is None
    b.bridge.undo()
    b.g.model.invalidate()
    w = b.g.model.box(sid, "lights").warn
    b.g.click_zone(b.g.cell(("box", sid, "lights")), "button")       # the cell's own button
    assert len(b.own(sid, "lights")) == 8 and w.key[0] == "beside"
    b.close()
