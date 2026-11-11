"""FLOAT's star catalogue as Settings downloads it (core/sky_catalogue.py): the pins and their single source, the state,
the download (resume, checksum, nothing fetched twice, byte counts past 4 GB) from a local fake server.

No network but 127.0.0.1: a small HTTP server in a thread answers with Range; the pins of the fake catalogue (a few KB per
file, the real layout: positions at the top, 48 compact chunks in ``float_xp_compact/``) are generated here.
"""
from __future__ import annotations

import hashlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest

from astro_gpu.core import gaia_extracts as ge
from astro_gpu.core import online, sirilcat, sky_catalogue, sky_fetch

ROOT = Path(__file__).resolve().parents[1]
ASTRO_NAME = "siril_cat_healpix8_astro.dat"
K = 4


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    """A private state folder and user home, no extract anywhere, the network allowed (only the local server)."""
    home = tmp_path / "userhome"
    home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.setenv("ASTRO_GPU_HOME", str(tmp_path / "aghome"))
    (tmp_path / "nowhere").mkdir()
    for v in ("LOCALAPPDATA", "APPDATA", "XDG_CONFIG_HOME", "XDG_DATA_HOME"):
        monkeypatch.setenv(v, str(tmp_path / "nowhere"))
    monkeypatch.setenv("ASTRO_GPU_GAIA_EXTRACTS", "")
    monkeypatch.setenv("ASTRO_GPU_SIRIL_CATALOGUES", "")
    monkeypatch.delenv("ASTRO_GPU_OFFLINE", raising=False)
    monkeypatch.setattr(sky_catalogue, "lock_file", lambda: None)
    online.reset()
    sirilcat.reset()
    yield
    online.reset()
    sirilcat.reset()


# ------------------------------------------------------------------ fake files of the real layout

def _header(kind: int, chunk: int | None, compact: bool = False) -> bytearray:
    h = bytearray(128)
    h[:48] = (b"FLOAT compact" if compact else b"Siril test extract").ljust(48, b"\0")
    h[48:53] = bytes([3, 1, kind, 1 if chunk is not None else 0, 1 if chunk is not None else 0])
    if chunk is not None:
        for off in (53, 57, 61):
            h[off:off + 4] = int(chunk).to_bytes(4, "little")
    if compact:
        h[65], h[66] = sirilcat.COMPACT_VERSION, K
        h[67:69] = len(sirilcat.XP_WAVELENGTHS).to_bytes(2, "little")
    return h


def _astro_file(n: int = 50) -> bytes:
    rng = np.random.default_rng(1)
    rec = np.zeros(n, sirilcat.ASTRO_DT)
    for name, dt in sirilcat.ASTRO_DT.fields.items():
        rec[name] = rng.integers(-1000, 1000, n).astype(dt[0])
    idx = np.where(np.arange(48) >= 5, n, 0).astype("<u4")
    return bytes(_header(sirilcat.TYPE_ASTRO, None)) + idx.tobytes() + rec.tobytes()


def _compact_file(chunk: int, n: int = 3) -> bytes:
    basis = np.zeros((len(sirilcat.XP_WAVELENGTHS), K), np.float32)
    rec = np.zeros(n, sirilcat.compact_dtype(K))
    return bytes(_header(sirilcat.TYPE_XPCOMPACT, chunk, True)) + basis.tobytes() + np.array([n], "<u4").tobytes() \
        + rec.tobytes()


def _bodies() -> dict[str, tuple[str, bytes]]:
    """name -> (sub-folder, bytes), the 49 files."""
    out = {ASTRO_NAME: ("", _astro_file())}
    for n in range(48):
        out[f"float_xp_compact_{n}.dat"] = ("float_xp_compact", _compact_file(n))
    return out


class Server:
    """Serves the bodies with Range. ``cut``: the first answer of the next request stops after that many bytes."""

    def __init__(self, bodies: dict[str, tuple[str, bytes]]):
        self.bodies, self.requests, self.cut = bodies, [], None
        outer = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def do_GET(self):
                name = self.path.lstrip("/")
                outer.requests.append((name, self.headers.get("Range")))
                body = outer.bodies[name][1]
                rng = self.headers.get("Range")
                start = int(rng.split("=")[1].split("-")[0]) if rng else 0
                part = body[start:]
                self.send_response(206 if start else 200)
                self.send_header("Content-Length", str(len(part)))
                if start:
                    self.send_header("Content-Range", f"bytes {start}-{len(body) - 1}/{len(body)}")
                self.end_headers()
                cut, outer.cut = outer.cut, None
                self.wfile.write(part if cut is None else part[:cut])
                self.wfile.flush()
                if cut is not None:
                    self.close_connection = True

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def pins(self, bad: str | None = None) -> list[sky_fetch.Wheel]:
        return [sky_fetch.Wheel("star_catalogue", "t", name, f"{self.base}/{name}",
                                "0" * 64 if name == bad else hashlib.sha256(b).hexdigest(), len(b), dest=sub)
                for name, (sub, b) in self.bodies.items()]

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture()
def server(monkeypatch):
    s = Server(_bodies())
    monkeypatch.setattr(sky_catalogue, "pins", s.pins)
    yield s
    s.close()


def _files_fetched(s: Server) -> list[str]:
    return [n for n, _r in s.requests]


# ------------------------------------------------------------------ the pins

def test_the_pins_of_the_package_agree_with_the_lock_of_the_installer():
    """One source of truth: installer/lock/lock.json; the package carries a copy written with it (build.py catalogue)."""
    lock = json.loads((ROOT / "installer" / "lock" / "lock.json").read_text(encoding="utf-8"))["star_catalogue"]
    copy = json.loads(sky_catalogue.RESOURCE.read_text(encoding="utf-8"))
    assert copy == lock
    assert len(lock["files"]) == 49 and lock["files"][0]["filename"] == ASTRO_NAME and lock["files"][0]["dest"] == ""
    assert {f["dest"] for f in lock["files"][1:]} == {sirilcat.COMPACT_DIR}
    assert sum(f["size"] for f in lock["files"]) == pytest.approx(1.85e9, rel=0.01)


def test_the_lock_beside_the_program_wins_and_the_copy_serves_without_it(tmp_path, monkeypatch):
    real = json.loads(sky_catalogue.RESOURCE.read_text(encoding="utf-8"))
    assert [w.filename for w in sky_catalogue.pins()] == [f["filename"] for f in real["files"]]    # no lock: the copy
    other = {"star_catalogue": {"version": "x", "files": [{**real["files"][0], "filename": "other.dat"}]}}
    lock = tmp_path / "lock.json"
    lock.write_text(json.dumps(other), encoding="utf-8")
    monkeypatch.setattr(sky_catalogue, "lock_file", lambda: lock)
    assert [w.filename for w in sky_catalogue.pins()] == ["other.dat"]
    lock.write_text("{ broken", encoding="utf-8")                       # an unreadable lock: the copy again
    assert len(sky_catalogue.pins()) == 49


def test_the_installer_and_settings_share_one_downloader():
    import sys
    sys.path.insert(0, str(ROOT / "installer" / "setup_env"))
    try:
        import fetch
    finally:
        sys.path.pop(0)
    assert fetch.download_one.__module__ == "sky_fetch" and fetch.Wheel.__module__ == "sky_fetch"
    assert (ROOT / "astro_gpu" / "core" / "sky_fetch.py").is_file()


# ------------------------------------------------------------------ the state

def test_the_states_missing_partial_installed(tmp_path, server):
    st = sky_catalogue.status()
    assert (st.state, st.files, st.total, st.where) == ("missing", 0, 49, None)
    own = sirilcat.own_dir()
    (own / sirilcat.COMPACT_DIR).mkdir(parents=True)
    (own / sirilcat.COMPACT_DIR / "float_xp_compact_3.dat").write_bytes(_bodies()["float_xp_compact_3.dat"][1])
    sirilcat.reset()
    st = sky_catalogue.status()
    assert (st.state, st.files, st.total) == ("partial", 1, 49)
    (own / sirilcat.COMPACT_DIR / "float_xp_compact_4.dat.part").write_bytes(b"x" * 1000)    # a stopped download
    sirilcat.reset()
    assert sky_catalogue.status().part_bytes == 1000
    sky_catalogue.download()
    st = sky_catalogue.status()
    assert (st.state, st.files, st.where, st.outside) == ("installed", 49, own, None)


def test_a_catalogue_found_elsewhere_is_installed_and_told_so(tmp_path, server):
    elsewhere = tmp_path / "elsewhere"
    for name, (sub, b) in _bodies().items():
        (elsewhere / sub).mkdir(parents=True, exist_ok=True)
        (elsewhere / sub / name).write_bytes(b)
    import os
    os.environ["ASTRO_GPU_GAIA_EXTRACTS"] = str(elsewhere)
    try:
        sirilcat.reset()
        st = sky_catalogue.status()
        assert st.state == "installed" and st.outside == elsewhere
    finally:
        os.environ["ASTRO_GPU_GAIA_EXTRACTS"] = ""


# ------------------------------------------------------------------ the download

def test_a_download_lands_where_the_reader_finds_it_and_a_second_one_fetches_nothing(server):
    seen: list[tuple[int, int, str]] = []
    sky_catalogue.download(lambda d, t, s: seen.append((d, t, s)))
    own = sirilcat.own_dir()
    assert (own / ASTRO_NAME).is_file() and (own / "float_xp_compact" / "float_xp_compact_47.dat").is_file()
    assert not list(own.rglob("*.part"))
    found = sirilcat.locate()
    assert found["astro"].path == own / ASTRO_NAME and sorted(found["xp"]) == list(range(48))
    assert ge.status(ge.XP).state == "installed" and sky_catalogue.status().state == "installed"
    total = sum(len(b) for _s, b in server.bodies.values())
    assert seen[0][2] == "Checking the files" and seen[-1] == (total, total, "Done")
    assert all(isinstance(d, int) and d <= t for d, t, _s in seen)
    assert any(s.startswith("File 2 of 49: downloading") for _d, _t, s in seen)
    assert len(server.requests) == 49
    sky_catalogue.download()
    assert len(server.requests) == 49                                      # nothing fetched again


def test_an_interrupted_file_resumes_from_its_part(server):
    server.cut = 100
    with pytest.raises(ge.DownloadError) as e:
        sky_catalogue.download(attempts=1)
    assert "start again to go on" in str(e.value)
    part = sirilcat.own_dir() / ASTRO_NAME
    assert not part.exists() and part.with_name(part.name + ".part").stat().st_size == 100
    assert sky_catalogue.status().part_bytes == 100
    sky_catalogue.download()
    assert (sirilcat.own_dir() / ASTRO_NAME).read_bytes() == server.bodies[ASTRO_NAME][1]
    assert server.requests[0] == (ASTRO_NAME, None) and server.requests[1] == (ASTRO_NAME, "bytes=100-")


def test_a_retry_after_a_passing_cut_goes_on_by_itself(server):
    server.cut = 100
    sky_catalogue.download(attempts=2)                                     # the second attempt continues the part
    assert sky_catalogue.status().state == "installed"


def test_a_wrong_checksum_is_refused_and_nothing_is_kept(server, monkeypatch):
    monkeypatch.setattr(sky_catalogue, "pins", lambda: server.pins(bad="float_xp_compact_0.dat"))
    with pytest.raises(ge.DownloadError) as e:
        sky_catalogue.download(attempts=1)
    assert "checksum" in str(e.value) and "start again" in str(e.value)
    d = sirilcat.own_dir() / "float_xp_compact"
    assert not (d / "float_xp_compact_0.dat").exists() and not (d / "float_xp_compact_0.dat.part").exists()
    assert sky_catalogue.status().state == "partial"          # the positions, fetched before, stay


def test_cancel_keeps_what_was_fetched_and_the_next_run_goes_on(server):
    cancel = threading.Event()

    def stop_in_the_third_file(done, total, text):
        if text.startswith("File 3 of 49"):
            cancel.set()
    with pytest.raises(ge.Cancelled):
        sky_catalogue.download(stop_in_the_third_file, cancel)
    assert (sirilcat.own_dir() / ASTRO_NAME).is_file() and sky_catalogue.status().state == "partial"
    fetched = len(server.requests)
    sky_catalogue.download()
    assert sky_catalogue.status().state == "installed"
    assert ASTRO_NAME not in _files_fetched(server)[fetched:]              # the file that was whole is not fetched again


def test_byte_counts_past_four_gigabytes_pass_as_python_ints(server, monkeypatch):
    big = 5 * 2**30

    def fake(items, folder, on_progress, cancel, context, attempts=3, pause=2.0):
        w = items[0]
        on_progress(sky_fetch.Progress(w, 1, 49, big // 2, big, 6 * 2**30, 1.0))
    monkeypatch.setattr(sky_fetch, "download_all", fake)
    seen = []
    sky_catalogue.download(lambda d, t, s: seen.append((d, t)))
    assert (big, 6 * 2**30) in seen and all(type(x) is int for pair in seen for x in pair)


def test_offline_and_a_second_download_say_it_in_plain_words(server, monkeypatch):
    monkeypatch.setenv("ASTRO_GPU_OFFLINE", "1")
    with pytest.raises(ge.DownloadError, match="offline mode"):
        sky_catalogue.download()
    assert server.requests == []
    monkeypatch.delenv("ASTRO_GPU_OFFLINE")
    assert sky_catalogue._RUN.acquire(blocking=False)
    try:
        with pytest.raises(ge.DownloadError, match="already running"):
            sky_catalogue.download()
    finally:
        sky_catalogue._RUN.release()


def test_not_enough_disk_space_is_said_before_any_fetch(server, monkeypatch):
    import collections
    usage = collections.namedtuple("usage", "total used free")
    monkeypatch.setattr(sky_catalogue.shutil, "disk_usage", lambda p: usage(10, 10, 1000))
    with pytest.raises(ge.DownloadError, match="Not enough free disk space"):
        sky_catalogue.download()
    assert server.requests == []


def test_a_run_cannot_download_the_catalogue(server):
    """The guard of a run (core/online.py) stops it before any request; the download is an action of Settings only."""
    online._FORBID.append("this run")
    try:
        with pytest.raises(online.NetworkForbidden):
            sky_catalogue.download()
    finally:
        online._FORBID.pop()
    assert server.requests == []
