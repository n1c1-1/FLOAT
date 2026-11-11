"""The star catalogue the installer fetches from FLOAT's data release: the lock's part, the setup step that fetches it into
the installation's ``sky`` folder (fake small files, the downloader patched: no network), what FLOAT then finds there,
and the rules of Setup for an update and an uninstall."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "installer" / "setup_env"))

import fetch  # noqa: E402
import setup_env  # noqa: E402

from astro_gpu.core import gaia_extracts as ge  # noqa: E402
from astro_gpu.core import sirilcat  # noqa: E402

LOCK = json.loads((ROOT / "installer" / "lock" / "lock.json").read_text(encoding="utf-8"))
ASTRO = "siril_cat_healpix8_astro.dat"
BASE = "https://github.com/n1c1-1/FLOAT/releases/download/sky-data-v1/"


# ------------------------------------------------------------------ the lock's part

def test_lock_star_catalogue_is_complete_and_exact():
    """49 assets of the release: the positions and the 48 compact spectra chunks, each with its URL, size, checksum and the
    folder it goes to (the sub-folder the reader expects for the spectra), each name once."""
    part = LOCK["star_catalogue"]
    files = part["files"]
    assert part["version"] == "sky-data-v1" and len(files) == 49
    assert files[0]["filename"] == ASTRO and files[0]["dest"] == ""
    assert [f["filename"] for f in files[1:]] == [f"float_xp_compact_{n}.dat" for n in range(48)]
    assert {f["dest"] for f in files[1:]} == {sirilcat.COMPACT_DIR}
    assert len({f["filename"] for f in files}) == 49
    for f in files:
        assert f["url"] == BASE + f["filename"] and f["name"] == "star_catalogue" and f["version"] == "sky-data-v1"
        assert len(f["sha256"]) == 64 and int(f["sha256"], 16) >= 0 and f["size"] > 0
        fetch.Wheel(**f)                                           # the keys the setup reads, nothing else
    assert 1.8e9 < sum(f["size"] for f in files) < 1.9e9


def test_lock_star_catalogue_sizes_are_the_local_files():
    """The sizes in the lock are those of the files the owner publishes (skipped where they are not at hand)."""
    spec = importlib.util.spec_from_file_location("float_build", ROOT / "installer" / "build.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not mod.CATALOGUE_ASTRO.is_file() or not mod.CATALOGUE_XP.is_dir():
        pytest.skip("the local catalogue files are not on this computer")
    part = mod.catalogue_part(hashed=lambda p: "0" * 64)
    mine = [(f["filename"], f["size"], f["url"], f["dest"]) for f in part["files"]]
    assert mine == [(f["filename"], f["size"], f["url"], f["dest"]) for f in LOCK["star_catalogue"]["files"]]


def test_notices_say_the_fetched_catalogue_for_this_lock():
    assert "star_catalogue" in LOCK                                # build.py passes this to the notices


# ------------------------------------------------------------------ fake files, a fake server

def astro_bytes() -> bytes:
    h = bytearray(128)
    h[:48] = b"Siril test extract".ljust(48, b"\0")
    h[48:53] = bytes([3, 1, sirilcat.TYPE_ASTRO, 0, 0])
    n = 20
    rec = np.zeros(n, sirilcat.ASTRO_DT)
    idx = np.where(np.arange(48) >= 5, n, 0).astype("<u4")
    return bytes(h) + idx.tobytes() + rec.tobytes()


def compact_bytes(chunk: int, k: int = 4) -> bytes:
    h = bytearray(128)
    h[:48] = sirilcat.COMPACT_TITLE.ljust(48, b"\0")
    h[48:53] = bytes([3, 1, sirilcat.TYPE_XPCOMPACT, 1, 1])
    for off in (53, 57, 61):
        h[off:off + 4] = chunk.to_bytes(4, "little")
    h[sirilcat.COMPACT_AT] = sirilcat.COMPACT_VERSION
    h[sirilcat.COMPACT_AT + 1] = k
    h[sirilcat.COMPACT_AT + 2:sirilcat.COMPACT_AT + 4] = len(sirilcat.XP_WAVELENGTHS).to_bytes(2, "little")
    basis = np.zeros((len(sirilcat.XP_WAVELENGTHS), k), "<f4")
    rec = np.zeros(2, sirilcat.compact_dtype(k))
    return bytes(h) + basis.tobytes() + np.array([2], "<u4").tobytes() + rec.tobytes()


class Served:
    """What the fake release serves: bytes by URL; ``cut``: the first answer to this URL stops after that many bytes."""

    def __init__(self):
        self.data: dict[str, bytes] = {}
        self.calls: list[tuple[str, int]] = []
        self.cut: dict[str, int] = {}


class Body:
    def __init__(self, data: bytes, start: int, cut: int | None):
        self.data, self.status, self.cut, self.pos = data[start:], 206 if start else 200, cut, 0

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self, n):
        if self.cut is not None and self.pos >= self.cut:
            raise ConnectionResetError("cut")
        end = self.pos + n if self.cut is None else min(self.pos + n, self.cut)
        out, self.pos = self.data[self.pos:end], end
        return out


@pytest.fixture
def served(monkeypatch):
    s = Served()

    def fake_open(url, start, context, timeout):
        s.calls.append((url, start))
        return Body(s.data[url], start, s.cut.pop(url, None))

    monkeypatch.setattr(fetch, "_open", fake_open)
    real = fetch.download_all
    monkeypatch.setattr(fetch, "download_all", lambda *a, **k: real(*a, attempts=1, pause=0))
    return s


@pytest.fixture
def world(tmp_path, monkeypatch, served):
    """An installation in tmp_path/app-parent with the lock of the 49 fake files, FLOAT's state elsewhere and no other
    catalogue on this computer; ``sirilcat.install_dir`` answers its ``sky`` folder as it does in an installed FLOAT."""
    root = tmp_path / "FLOAT"
    (root / "setup").mkdir(parents=True)
    home = tmp_path / "userhome"
    home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.setenv("ASTRO_GPU_HOME", str(tmp_path / "aghome"))
    (tmp_path / "nowhere").mkdir()
    for v in ("LOCALAPPDATA", "APPDATA", "XDG_CONFIG_HOME", "XDG_DATA_HOME"):
        monkeypatch.setenv(v, str(tmp_path / "nowhere"))
    monkeypatch.setenv("ASTRO_GPU_GAIA_EXTRACTS", "")
    monkeypatch.setenv("ASTRO_GPU_SIRIL_CATALOGUES", "")
    monkeypatch.setattr(sirilcat, "install_dir", lambda here=None: root / "sky")
    files = [(ASTRO, "", astro_bytes())] + [(f"float_xp_compact_{n}.dat", sirilcat.COMPACT_DIR, compact_bytes(n))
                                            for n in range(48)]
    entries = []
    for name, dest, data in files:
        served.data[BASE + name] = data
        entries.append({"name": "star_catalogue", "version": "sky-data-v1", "filename": name, "url": BASE + name,
                        "sha256": hashlib.sha256(data).hexdigest(), "size": len(data), "dest": dest})
    lock = root / "setup" / "lock.json"
    lock.write_text(json.dumps({"python": "3.12.13", "builds": {}, "star_catalogue": {"version": "sky-data-v1",
                                                                                        "files": entries}}), encoding="utf-8")
    sirilcat.reset()
    yield root, lock, entries
    sirilcat.reset()


class Ui:
    def __init__(self):
        self.stages: list[str] = []
        self.progress: list = []

    def stage(self, text): self.stages.append(text)
    def indeterminate(self): pass
    def download(self, p): self.progress.append(p)


def installer(world, lang="en"):
    root, lock, _ = world
    return setup_env.Installer(root, Ui(), lock, lang, {})


# ------------------------------------------------------------------ the fetch

def test_the_fetch_lands_where_float_finds_it(world, served):
    root, _, entries = world
    inst = installer(world)
    assert inst.fetch_catalogue() == ""
    assert (root / "sky" / ASTRO).is_file() and (root / "sky" / "float_xp_compact" / "float_xp_compact_47.dat").is_file()
    assert len(served.calls) == 49 and not list((root / "sky").rglob("*.part"))
    assert inst.ui.stages == [setup_env.TEXTS["en"]["catalogue_stage"]] and inst.ui.progress[-1].overall_done == sum(
        e["size"] for e in entries)
    sirilcat.reset()
    for kind in (ge.ASTRO, ge.XP):
        st = ge.status(kind)
        assert st.state == "installed" and st.files == st.total and st.own and st.outside is None
    assert sirilcat.install_dir() == root / "sky"


def test_a_second_run_fetches_nothing_and_does_not_read_the_files_again(world, served, monkeypatch):
    root, _, _ = world
    installer(world).fetch_catalogue()
    served.calls.clear()
    hashed = []
    real = fetch.sha256_of
    monkeypatch.setattr(fetch, "sha256_of", lambda p: hashed.append(p) or real(p))
    inst = installer(world)                                          # an update: a new run over the same folder
    hashed.clear()
    assert inst.fetch_catalogue() == ""
    assert served.calls == [] and hashed == []


def test_a_file_changed_after_it_was_checked_is_fetched_again(world, served):
    root, _, _ = world
    installer(world).fetch_catalogue()
    served.calls.clear()
    path = root / "sky" / "float_xp_compact" / "float_xp_compact_7.dat"
    data = bytearray(path.read_bytes())
    data[-1] ^= 1                                                    # same size, other bytes
    path.write_bytes(bytes(data))
    short = root / "sky" / "float_xp_compact" / "float_xp_compact_8.dat"
    short.write_bytes(short.read_bytes()[:-5])                       # a truncated one
    assert installer(world).fetch_catalogue() == ""
    assert sorted(u.rsplit("/", 1)[1] for u, _ in served.calls) == ["float_xp_compact_7.dat", "float_xp_compact_8.dat"]
    assert path.read_bytes() == served.data[BASE + "float_xp_compact_7.dat"]


def test_an_interrupted_file_resumes_and_the_summary_says_so(world, served):
    root, _, _ = world
    url = BASE + "float_xp_compact_3.dat"
    served.cut[url] = 300
    text = installer(world).fetch_catalogue()
    assert text.startswith("The star catalogue could not be downloaded") and "FLOAT starts all the same" in text
    part = root / "sky" / "float_xp_compact" / "float_xp_compact_3.dat.part"
    assert part.is_file() and part.stat().st_size == 300
    sirilcat.reset()
    assert ge.status(ge.XP).state == "partial"                       # FLOAT starts, the blocks are marked as today
    served.calls.clear()
    assert installer(world).fetch_catalogue() == ""
    assert (url, 300) in served.calls                                # continued where it stopped
    assert not part.exists()
    assert [u for u, _ in served.calls].count(BASE + ASTRO) == 0     # what was whole stays
    sirilcat.reset()
    assert ge.status(ge.XP).state == "installed"


def test_a_wrong_checksum_is_refused(world, served):
    root, lock, entries = world
    entries[5]["sha256"] = "0" * 64
    data = json.loads(lock.read_text(encoding="utf-8"))
    data["star_catalogue"]["files"] = entries
    lock.write_text(json.dumps(data), encoding="utf-8")
    inst = installer(world)
    text = inst.fetch_catalogue()
    assert text.startswith("A star catalogue file was damaged") and entries[5]["filename"] in text
    folder = root / "sky" / entries[5]["dest"]
    assert not (folder / entries[5]["filename"]).exists() and not (folder / (entries[5]["filename"] + ".part")).exists()


def test_not_enough_room_is_said_before_anything_is_fetched(world, served, monkeypatch):
    monkeypatch.setattr(fetch, "free_bytes", lambda p: 1000)
    text = installer(world).fetch_catalogue()
    assert text.startswith("The star catalogue was not downloaded: not enough disk space") and served.calls == []


def test_the_french_messages_exist_and_say_the_same_things(world, served):
    fr, en = setup_env.TEXTS["fr"], setup_env.TEXTS["en"]
    keys = [k for k in en if k.startswith("catalogue")]
    assert len(keys) == 6 and all(k in fr for k in keys)
    for k in keys:
        assert fr[k] != en[k] and ("{detail}" in en[k]) == ("{detail}" in fr[k]) and ("{need}" in en[k]) == ("{need}" in fr[k])
    served.cut[BASE + ASTRO] = 10
    assert installer(world, "fr").fetch_catalogue().startswith("Le catalogue d'étoiles n'a pas pu être téléchargé")


def test_a_cancel_is_said_and_keeps_the_part(world, served):
    root, _, _ = world
    inst = installer(world)
    inst.cancel.set()
    assert inst.fetch_catalogue() == setup_env.TEXTS["en"]["catalogue_stopped"]


def test_the_catalogue_is_the_last_step_and_its_failure_does_not_fail_the_installation(world, served, monkeypatch):
    """attempt() ends with the catalogue: the summary carries its message, the installation counts as ready."""
    inst = installer(world)
    steps = []
    for name in ("check_folder", "ensure_env", "download", "sync"):
        monkeypatch.setattr(inst, name, lambda *a, n=name: steps.append(n))
    monkeypatch.setattr(inst, "choose_build", lambda: "cpu")
    monkeypatch.setattr(inst, "missing_wheels", lambda w: [])
    monkeypatch.setattr(inst, "check_disk", lambda m: None)
    monkeypatch.setattr(inst, "smoke", lambda w: ({}, False))
    monkeypatch.setattr(inst, "finish", lambda *a: setattr(inst, "summary", "base") or steps.append("finish"))
    inst.lock["builds"] = {"cpu": []}
    served.cut[BASE + ASTRO] = 5
    inst.attempt()
    assert steps[-1] == "finish" and inst.summary.startswith("base\nThe star catalogue could not be downloaded")


def test_a_lock_without_a_catalogue_fetches_nothing(world, served):
    root, lock, _ = world
    lock.write_text(json.dumps({"python": "3.12.13", "builds": {}}), encoding="utf-8")
    assert installer(world).fetch_catalogue() == "" and served.calls == []


# ------------------------------------------------------------------ Setup's rules

def test_inno_setup_keeps_the_catalogue_on_update_and_removes_it_on_uninstall():
    text = (ROOT / "installer" / "float.iss").read_text(encoding="utf-8")
    sections = {}
    for block in text.split("\n["):
        head, _, body = block.partition("]")
        sections[head.strip("[")] = body
    line = 'Type: filesandordirs; Name: "{app}\\sky"'
    assert line in sections["UninstallDelete"]
    assert "sky" not in sections["InstallDelete"]                    # an update reuses what was fetched
    assert "sky" not in sections["Files"]                            # and the stage does not carry it
    size = int(next(ln for ln in text.splitlines() if ln.startswith("ExtraDiskSpaceRequired=")).split("=")[1])
    assert size >= 3221225472 + sum(f["size"] for f in LOCK["star_catalogue"]["files"])
