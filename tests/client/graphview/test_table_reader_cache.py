"""The listing cache of the graph page's reader follows the disk and the project: frames added to a folder already
listed are seen, another project naming the same relative folder gets its own listing, and a project opened in the
page starts from an empty cache (the cache was never emptied: counts stayed those of the first listing until the
application was restarted)."""
import os

from astro_gpu.client.graphview.table_reader import Reader
from tests.client.graphview.test_gv_table import fwd, make


def _count(r, key, sources):
    r.request(key, sources)
    assert r.wait()
    return r.count(sources)


def test_frames_added_to_a_folder_already_listed_are_seen(tmp_path):
    folder = fwd(make(tmp_path / "lights", 2, "L"))
    r = Reader()
    r.note(folder, True)
    assert _count(r, "k", [folder]) == (2, True)
    make(tmp_path / "lights", 3, "L")                                  # one more frame, written since
    st = os.stat(folder)
    os.utime(folder, (st.st_atime, st.st_mtime + 10))                  # (its folder's time moves on any disk)
    assert _count(r, "k", [folder]) == (3, True)


def test_another_project_naming_the_same_relative_folder_gets_its_own_listing(tmp_path, monkeypatch):
    a, b = tmp_path / "A", tmp_path / "B"
    make(a / "frames", 2, "L")
    make(b / "frames", 5, "L")
    monkeypatch.chdir(tmp_path)
    root = [a]
    r = Reader()
    r.base = lambda: root[0]
    r.note("frames", True)
    assert _count(r, "k", ["frames"]) == (2, True)
    root[0] = b                                                        # the page now shows the other project
    r.note("frames", True)
    assert _count(r, "k", ["frames"]) == (5, True)


def test_a_project_installed_in_the_page_starts_from_an_empty_cache(qapp, tmp_path):
    from astro_gpu.client.graphview.document import MemoryDocument
    from astro_gpu.graphcore import api
    from tests.client.graphview.test_gv_files import one_session_bridge
    folder = fwd(make(tmp_path / "lights", 2, "L"))
    bridge, sid = one_session_bridge()
    reader = bridge.drops.reader
    bridge.drops.read(sid, "lights", [folder])
    assert reader.wait() and reader.count([folder]) == (2, True)
    bridge.install(MemoryDocument(api.blank_project()))
    assert reader.known(folder) is None and reader.count([folder]) == (0, False)
