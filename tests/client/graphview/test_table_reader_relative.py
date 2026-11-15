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
    assert os.path.isabs(r._on_disk(str(tmp_path / "b.fit")))
