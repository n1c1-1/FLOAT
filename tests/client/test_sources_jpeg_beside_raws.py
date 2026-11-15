"""Sources / Inspect « Add files… », « Add folder… » and a drop on a kind's row: a camera's JPEGs beside its raw or FITS
frames are left (the engine reads a stored file as it is: each JPEG would be one more frame, counted and possibly
stacked instead of the raws), said with « Take all »; a session saved before with such JPEGs says it, with a button
that takes them out. The stored entries and the engine's listing agree with what is said."""
from __future__ import annotations

from pathlib import Path

from astro_gpu.core.project import list_frames
from tests.client.test_sources_import import close, fits_file, open_page, qapp, settle, texts  # noqa: F401


def night(folder: Path, n=8):
    for k in range(n):
        fits_file(folder / f"DSC{k:05d}.fits")
        (folder / f"DSC{k:05d}.JPG").write_bytes(bytes(64))
    return sorted(str(x) for x in folder.iterdir())


def test_sources_jpeg_beside_raws_left_taken_and_removed(qapp, tmp_path, monkeypatch):
    from astro_gpu.client.sources import SourcesView
    picked = night(tmp_path / "night")
    w, m, page = open_page(tmp_path, monkeypatch, sessions=1, workflows=("Full",))
    view = SourcesView(m)
    said: list = []
    view.toast.connect(said.append)
    try:
        sid = m.add_filter_session("")
        view.show_session(sid, "lights")
        assert view.add_files("lights", picked) == 8                       # the file dialog: 16 files selected
        stored = m.p.session(sid).lights
        assert len(stored) == 8 and all(x.endswith(".fits") for x in stored) and len(list_frames(stored)) == 8
        assert "8 JPG left: beside raw frames" in said[-1]
        settle(w, page)
        view.refresh()
        assert "Take all" in texts(view) and any("8 JPG left" in t for t in texts(view))
        view.take_all("lights")                                            # the user's own choice
        assert len(list_frames(m.p.session(sid).lights)) == 16
        settle(w, page)
        view.refresh()
        assert len(view.beside_raws("lights")) == 8 and "Remove JPG" in texts(view)
        assert not any("not frames the engine reads here" in t for t in view.anomalies("lights"))
        view.remove_beside_raws("lights")
        assert len(m.p.session(sid).lights) == 8 and not view.beside_raws("lights")
        sid2 = m.add_filter_session("")                                    # « Add folder… »
        view.show_session(sid2, "lights")
        assert view.add_files("lights", [str(tmp_path / "night")]) == 8 and "8 JPG left" in said[-1]
        assert len(list_frames(m.p.session(sid2).lights)) == 8
    finally:
        view.close()
        close(w, m, page)
