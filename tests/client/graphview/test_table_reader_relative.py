"""A project that holds its own frames names them relative to its folder: the Table reads them from there, whatever
folder the application runs in (they were all said unreadable, with a button that removed them)."""
import os

from astro_gpu.client.graphview.table_reader import Reader


def test_a_relative_entry_is_read_from_the_projects_folder(tmp_path, monkeypatch):
    root = tmp_path / "project"
    (root / "frames").mkdir(parents=True)
    (root / "frames" / "a.fit").write_bytes(b"x" * 10)
    monkeypatch.chdir(tmp_path)                                    # the application runs somewhere else
    r = Reader()
    r.base = lambda: root
    r.request("k", ["frames/a.fit", "frames/gone.fit"])
    assert r.wait()
    a, gone = r.expand(("frames/a.fit", "frames/gone.fit"))
    assert (a.path, a.size) == ("frames/a.fit", 10)                # the entry keeps the name the project gave it
    assert gone.size == 0                                          # a file really missing is still said so
    absolute = (root / "frames" / "a.fit").as_posix()
    r.request("k2", [absolute])
    assert r.wait() and r.expand((absolute,))[0].size == 10        # an absolute entry is read where it says


def test_without_a_project_folder_the_entry_is_read_as_written(tmp_path, monkeypatch):
    (tmp_path / "b.fit").write_bytes(b"x" * 4)
    monkeypatch.chdir(tmp_path)
    r = Reader()
    r.request("k", ["b.fit"])
    assert r.wait() and r.expand(("b.fit",))[0].size == 4
    assert os.path.isabs(r.on_disk(str(tmp_path / "b.fit")))


def test_a_relative_folder_held_by_a_session_is_counted_as_its_frames_on_the_graph(qapp, tmp_path, monkeypatch):
    """The graph's Session targets read a relative folder entry from the project's folder too (it was tested as a file
    where the application runs: one frame said)."""
    from tests.client.graphview.test_gv_files import one_session_bridge, settle
    from tests.client.graphview.test_gv_table import make
    from astro_gpu.client.graphview.graph_drops import GraphDrops
    root = tmp_path / "project"
    make(root / "frames", 3, "L", "fit")
    monkeypatch.chdir(tmp_path)                                    # the application runs somewhere else
    bridge, s = one_session_bridge()
    drops = GraphDrops(bridge)
    drops.reader.base = lambda: root
    drops.read(s, "lights", ["frames"])
    settle(drops.reader)
    assert drops.count(["frames"]) == (3, True)


# ------------------------------------------------------------------------------------------- the other places (13.18)
def _relative_session(tmp_path, monkeypatch, n=3, empty=None):
    """A Table whose session holds the relative folder entry "frames" of a project at tmp_path/project, the
    application running elsewhere."""
    from tests.client.graphview.test_gv_table import Bench, make
    root = tmp_path / "project"
    folder = make(root / "frames", n, "L", "fit")
    if empty is not None:
        (folder / f"L_{empty:04d}.fit").write_bytes(b"")
    monkeypatch.chdir(tmp_path)
    b = Bench(1366)
    b.reader.base = lambda: root
    sid = b.sessions()[0]
    assert b.bridge.act("add_files", quiet=True, session=sid, kind="lights", paths=["frames"], mode="replace").ok
    b.g.model.invalidate()
    b.bridge.drops.read(sid, "lights", ["frames"])
    b.read()
    return b, sid, root


def test_a_relative_folder_is_a_folder_for_the_page(qapp, tmp_path, monkeypatch):
    from astro_gpu.client.graphview import files
    root = tmp_path / "project"
    (root / "frames").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    r = Reader()
    r.base = lambda: root
    assert files.is_folder(r, "frames")                            # it was judged a file where the application runs


def test_a_frame_dropped_inside_a_relative_folder_entry_is_a_double(qapp, tmp_path, monkeypatch):
    from astro_gpu.client.graphview import files
    from tests.client.graphview.test_gv_table import make
    root = tmp_path / "project"
    make(root / "frames", 2, "L", "fit")
    monkeypatch.chdir(tmp_path)
    r = Reader()
    r.base = lambda: root
    dropped = (root / "frames" / "L_0001.fit").as_posix()
    assert files.already(dropped, {"frames"}, r)                    # it was added a second time


def test_a_relative_folder_listing_keeps_the_projects_form(tmp_path, monkeypatch):
    root = tmp_path / "project"
    (root / "frames").mkdir(parents=True)
    (root / "frames" / "a.fit").write_bytes(b"x" * 10)
    monkeypatch.chdir(tmp_path)
    r = Reader()
    r.base = lambda: root
    r.note("frames", True)
    r.request("k", ["frames"])
    assert r.wait()
    assert [f.path for f in r.expand(("frames",))] == ["frames/a.fit"]


def test_remove_in_a_relative_folder_keeps_the_list_relative(qapp, tmp_path, monkeypatch):
    """« Remove » on a file of a relative folder entry: the folder's other files are written as the project names
    them, so a project that holds its own frames stays movable (they were written as absolute paths)."""
    from PySide6.QtCore import QItemSelectionModel
    from astro_gpu.client.graphview.files_panel import FILE, FilesPanel
    from tests.client.graphview.test_gv_files import settle
    b, sid, _root = _relative_session(tmp_path, monkeypatch)
    p = FilesPanel(b.bridge, sid, "lights")
    settle(b.reader)
    p.refresh()
    p.toggle("frames")
    rows = [i for i, r in enumerate(p.rows.rows) if r[0] == FILE]
    assert len(rows) == 3
    p.view.selectionModel().select(p.rows.index(rows[1]), QItemSelectionModel.ClearAndSelect)
    p.remove()
    assert b.doc.files_of(sid, "lights")[1] == ["frames/L_0000.fit", "frames/L_0002.fit"]
    b.close()


def test_remove_unreadable_in_a_relative_folder_keeps_the_list_relative(qapp, tmp_path, monkeypatch):
    b, sid, _root = _relative_session(tmp_path, monkeypatch, empty=1)
    w = b.box(sid, "lights").warn
    assert w is not None and w.button == "Remove them"
    b.click(("warn", sid, "lights"), zone="button")
    b.read()
    assert b.own(sid, "lights") == ["frames/L_0000.fit", "frames/L_0002.fit"]
    b.close()


def test_copy_of_a_relative_entry_gives_where_it_is(qapp, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QApplication
    b, sid, root = _relative_session(tmp_path, monkeypatch)
    g = b.g
    g.set_focus(g.cell(("box", sid, "lights")))
    g.copy()
    urls = [u.toLocalFile() for u in QApplication.clipboard().mimeData().urls()]
    QApplication.clipboard().clear()
    assert urls == [(root / "frames").as_posix()]                   # it was "frames", read where the paste runs
    b.close()
