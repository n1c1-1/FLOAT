"""The file-type filter (graphview/files.py), the background listing (table_reader.py), the drops on the graph's Session
targets (graph_drops.py) and the panel of a file cell (files_panel.py), on real temporary folders."""
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
    assert [n for n, _e in files.TYPES] == ["Camera raws", "FITS", "Raws and FITS", "All images", "All files"]
    assert files.index() == files.DEFAULT == 2
    assert files.accepts("x.CR3", 0) and not files.accepts("x.fit", 0) and files.accepts("x.FITS", 1)
    assert files.accepts("x.jpeg", 3) and not files.accepts("x.jpeg", 2) and files.accepts("x.txt", 4)
    assert not files.accepts("DARK_300s.txt", 2)                       # a name is never read, only the extension
    files.set_index(0)
    assert prefs.get("graph_file_filter") == "Camera raws" and files.index() == 0
    assert files.summary(["a.ARW", "b.arw", "c.jpg"]) == "2 ARW, 1 JPG"
    assert files.already("C:/a/b", ["C:/a/b"]) and files.already("C:/a/b/c.fit", ["C:/a/b/**"])
    assert not files.already("C:/a/b/**", ["C:/a/b"])                   # read deeper: not a double


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
    assert len(fs) == 4 and all(f.path.endswith(".ARW") for f in fs)   # Raws and FITS; sub-folders not read
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
    assert len(lst) == 6 and len(bridge.doc.document._undo) == n0 + 2   # the folder + the 5 JPG as single files
    assert toasts.text == "All taken"
    assert drops.drop(sid, "lights", [str(mixed)]) is None            # already there: nothing added
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
    assert src == "own" and lst == [fwd(only) + "/**"] and len(bridge.doc.document._undo) == n0 + 1
    assert drops.reader.count(lst) == (6, True)
    assert bridge.doc.view["file_options"][f"{sid}|lights|{fwd(only)}/**"] == ["Raws and FITS", True]


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
    assert [r[0] for r in p.rows.rows] == [ENTRY, ENTRY]
    assert "2 entries · 5 files" == p.summary.text()
    assert "Raws and FITS · without sub-folders · 3 files" in p.rows.rows[0][2]
    p.toggle(fwd(a))
    kinds = [r[0] for r in p.rows.rows]
    assert kinds == [ENTRY, FOLDER, FILE, FILE, FILE, ENTRY]
    assert bridge.open_entries[(sid, "lights")] == {fwd(a)}           # kept over a redraw
    # one file of a folder: an exclusion, view side, no undo step
    n0 = len(bridge.doc.document._undo)
    first_file = p.rows.index(2)
    p.view.selectionModel().select(first_file, QItemSelectionModel.ClearAndSelect)
    assert p.b_remove.isEnabled()
    p.remove()
    assert bridge.doc.view["exclusions"][f"{sid}|lights"] == [p.rows.rows[2][3]]
    assert len(bridge.doc.document._undo) == n0 and "1 excluded" in p.summary.text()
    # an entry: through the core, one undo step
    p.view.selectionModel().select(p.rows.index(len(p.rows.rows) - 1), QItemSelectionModel.ClearAndSelect)
    p.remove()
    assert bridge.doc.files_of(sid, "lights")[1] == [fwd(a)] and len(bridge.doc.document._undo) == n0 + 1


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
