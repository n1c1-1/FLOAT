"""The file-type filter (graphview/files.py), the background listing (table_reader.py), the drops on the graph's Session
targets (graph_drops.py) and the panel of a file cell (files_panel.py), on real temporary folders. The filter and
« Include » are resolved at drop time: a folder entry when the filter keeps what the engine reads there, else single
files; nothing about the filter is stored; the counts said are the engine's own (``list_frames``)."""
from __future__ import annotations

import time

from PySide6.QtCore import QItemSelectionModel
from PySide6.QtWidgets import QApplication

from astro_gpu.graphcore.operations import COMMON
from tests.client.graphview.test_gv_table import fwd, make


def settle(reader, timeout=30):
    for _i in range(3):
        assert reader.wait(timeout)
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < 0.05:
            QApplication.processEvents()
            time.sleep(0.005)


def one_session_bridge():
    from astro_gpu.client.graphview.bridge import Bridge
    from astro_gpu.client.graphview.document import MemoryDocument
    from astro_gpu.graphcore import api
    g, _r = api.apply(api.blank_project(), "add_row", kind="session")
    bridge = Bridge(MemoryDocument(g))
    return bridge, g.rows[0].nights[0][0]


class Toasts:
    """What bridge.rich_toast said last: its text and its link keys, as the page would show them."""

    def __init__(self, bridge):
        self.text_fn = self.links_fn = None
        bridge.rich_toast.connect(self._got)

    def _got(self, text, links):
        self.text_fn, self.links_fn = text, links

    @property
    def text(self):
        return self.text_fn() if callable(self.text_fn) else self.text_fn

    @property
    def keys(self):
        links = self.links_fn() if callable(self.links_fn) else (self.links_fn or [])
        return [k for _t, k in links]


# ----------------------------------------------------------------------------------------------- the filter
def test_gv_files_filter_by_extension_only(qapp):
    from astro_gpu.client import prefs
    from astro_gpu.client.graphview import files
    assert [n for n, _e in files.TYPES] == ["Camera raws", "FITS", "Raws and FITS", "All images"]
    assert files.index() == files.DEFAULT == 2
    assert files.accepts("x.CR3", 0) and not files.accepts("x.fit", 0) and files.accepts("x.FITS", 1)
    assert files.accepts("x.jpeg", 3) and not files.accepts("x.jpeg", 2) and files.accepts("x.fits.fz", 1)
    assert not any(files.accepts("x.txt", i) for i in range(len(files.TYPES)))   # the engine cannot read it
    assert not files.accepts("DARK_300s.txt", 2)                       # a name is never read, only the extension
    files.set_index(0)
    assert prefs.get("graph_file_filter") == "Camera raws" and files.index() == 0
    assert files.summary(["a.ARW", "b.arw", "c.jpg"]) == "2 ARW, 1 JPG"
    assert files.already("C:/a/b", ["C:/a/b"])                          # (a file read through a folder: the drops)
    assert not files.already("C:/a/b/c/d.fit", ["C:/a/b"])             # the engine reads a folder's own level only


def test_gv_files_reader_lists_direct_files(qapp, tmp_path):
    from astro_gpu.client.graphview.table_reader import Reader
    root = tmp_path / "night"
    make(root, 4, "L", "ARW")
    make(root, 2, "L", "JPG")
    make(root / "sub", 3, "S", "ARW")
    r = Reader()
    seen = []
    r.listen(seen.append)
    r.note(fwd(root), True)
    r.request(("s", "lights"), [fwd(root)])
    assert r.wait(10)
    fs = r.expand((fwd(root),))
    assert len(fs) == 4 and all(f.path.endswith(".ARW") for f in fs)   # the engine's rule: JPG beside raws aside
    assert len(r.left_out([fwd(root)])) == 2 and r.sub_folders([fwd(root)]) == 1
    assert seen and seen[-1].phase == "done" and not r.active_jobs()
    missing = fwd(tmp_path / "gone")
    r.note(missing, True)
    r.request(("s", "darks"), [missing])
    assert r.wait(10)
    assert r.unreadable([missing]) == [missing] and r.expand((missing,)) == []


# ----------------------------------------------------------------------------------------------- drops on a Session
def test_gv_files_graph_drop_filter_take_all_and_doubles(qapp, tmp_path):
    mixed = tmp_path / "mixed"
    make(mixed, 5, "IMG", "ARW")
    make(mixed, 5, "IMG", "JPG")
    bridge, sid = one_session_bridge()
    toasts = Toasts(bridge)
    drops = bridge.drops
    n0 = len(bridge.doc.document._undo)
    res = drops.drop(sid, "lights", [str(mixed)])
    assert res.ok and len(bridge.doc.document._undo) == n0 + 1        # one undo step, at once
    settle(drops.reader)
    assert toasts.text == "5 ARW added · 5 JPG left" and "take_all" in toasts.keys and "undo" in toasts.keys
    drops.link("take_all")
    settle(drops.reader)
    lst = bridge.doc.files_of(sid, "lights")[1]
    assert len(lst) == 10 and len(bridge.doc.document._undo) == n0 + 2  # files only: the 5 ARW, then the 5 JPG
    assert toasts.text == "All taken"
    assert drops.drop(sid, "lights", [str(mixed)]) is None            # already there: nothing added
    assert drops.drop(sid, "lights", [str(mixed / "IMG_0000.ARW")]) is None   # a file already there
    assert toasts.text == "Already there: nothing added" and len(bridge.doc.document._undo) == n0 + 2
    assert drops.reader.active_jobs() == []


def test_gv_files_graph_drop_sub_folders_and_empty_folder(qapp, tmp_path):
    only = tmp_path / "only_sub"
    make(only / "a", 3, "S", "ARW")
    make(only / "b", 3, "S", "ARW")
    bridge, sid = one_session_bridge()
    toasts = Toasts(bridge)
    drops = bridge.drops
    n0 = len(bridge.doc.document._undo)
    drops.drop(sid, "lights", [str(only)])
    settle(drops.reader)
    assert bridge.doc.files_of(sid, "lights")[0] == "none"           # no direct file: no empty entry left
    assert toasts.text == "No direct file · 2 sub-folders" and toasts.keys == ["include"]
    assert len(bridge.doc.document._undo) == n0
    drops.link("include")
    settle(drops.reader)
    src, lst = bridge.doc.files_of(sid, "lights")
    # each sub-folder's frames, stored one by one (files only); nothing stored about the choice
    assert src == "own" and lst == [fwd(only / d / f"S_{i:04d}.ARW") for d in "ab" for i in range(3)] and         len(bridge.doc.document._undo) == n0 + 1
    assert drops.reader.count(lst) == (6, True)
    assert "file_options" not in bridge.doc.view and not any("**" in x for x in lst)


def test_gv_files_a_filter_unlike_the_engine_rule_stores_single_files(qapp, tmp_path):
    from astro_gpu.client.graphview import files
    from astro_gpu.core.project import list_frames
    both = tmp_path / "both"
    make(both, 3, "R", "ARW")
    make(both, 2, "F", "fit")
    make(both, 1, "N", "txt")
    colour = tmp_path / "colour"
    make(colour, 4, "C", "JPG")
    mixed = tmp_path / "mixed"
    make(mixed, 2, "M", "ARW")
    make(mixed, 2, "M", "JPG")
    bridge, sid = one_session_bridge()
    toasts = Toasts(bridge)
    drops = bridge.drops
    files.set_index(0)                                                 # Camera raws: the engine would read the FITS too
    drops.drop(sid, "lights", [str(both)])
    lst = bridge.doc.files_of(sid, "lights")[1]
    assert sorted(lst) == sorted(fwd(both / f"R_{i:04d}.ARW") for i in range(3))
    assert toasts.text == "3 ARW added · 2 FIT left · 1 file not a frame, ignored"
    assert len(list_frames(lst)) == 3
    bridge.undo()
    files.set_index(3)                                                 # All images on a folder of JPEGs only: its files
    drops.drop(sid, "lights", [str(colour)])
    assert bridge.doc.files_of(sid, "lights")[1] == [fwd(colour / f"C_{i:04d}.JPG") for i in range(4)]
    drops.drop(sid, "lights", [str(mixed)])                            # All images beside raws: unlike the rule
    lst = bridge.doc.files_of(sid, "lights")[1]
    assert fwd(mixed) not in lst and sum(1 for x in lst if x.startswith(fwd(mixed) + "/")) == 4
    assert len(list_frames(lst)) == 8 and "file_options" not in bridge.doc.view
    assert drops.drop(sid, "lights", [str(tmp_path / "both" / "N_0000.txt")]) is None
    assert toasts.text == "Nothing to add: 1 file not a frame"


# ----------------------------------------------------------------------------------------------- the panel of a file cell
def test_gv_files_panel_entries_unfold_exclude_remove(qapp, tmp_path):
    from astro_gpu.client.graphview.files_panel import ENTRY, FILE, FOLDER, FilesPanel
    a = make(tmp_path / "a", 3, "L")
    b2 = make(tmp_path / "b", 2, "L")
    bridge, sid = one_session_bridge()
    bridge.drops.drop(sid, "lights", [str(a), str(b2)])
    settle(bridge.drops.reader)
    p = FilesPanel(bridge, sid, "lights")
    settle(bridge.drops.reader)
    p.refresh()
    # files only: each file stored on its own, shown under its folder (a heading of the display only)
    assert [r[0] for r in p.rows.rows] == [FOLDER, ENTRY, ENTRY, ENTRY, FOLDER, ENTRY, ENTRY]
    assert p.rows.rows[0][2].startswith(fwd(a)) and p.rows.rows[0][2].endswith("3")
    n0 = len(bridge.doc.document._undo)
    # a file: through the core, one undo step
    p.view.selectionModel().select(p.rows.index(len(p.rows.rows) - 1), QItemSelectionModel.ClearAndSelect)
    assert p.b_remove.isEnabled()
    p.remove()
    assert len(bridge.doc.files_of(sid, "lights")[1]) == 4 and len(bridge.doc.document._undo) == n0 + 1
    assert FILE not in [r[0] for r in p.rows.rows]


def test_gv_files_panel_inherited_cell_links_to_common(qapp, tmp_path):
    from astro_gpu.client.graphview.files_panel import FilesPanel
    darks = make(tmp_path / "darks", 2, "D")
    bridge, sid = one_session_bridge()
    bridge.act("set_common_files", kind="darks", paths=[fwd(darks)])
    asked = []
    p = FilesPanel(bridge, sid, "darks", go_common=asked.append)
    assert p.inherited and p.link is not None and not p.entries()
    p.link.click()
    assert asked == ["darks"]
    pc = FilesPanel(bridge, COMMON, "darks")
    settle(bridge.drops.reader)
    pc.refresh()
    assert not pc.inherited and pc.entries() == [fwd(darks)] and "2 files" in pc.summary.text()
