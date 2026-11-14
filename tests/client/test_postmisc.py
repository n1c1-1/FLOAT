"""Small defects of Post: the Shortcuts list, a Combine layers layer's default name, fresh layer ids, the export's history
file and the FITS HISTORY cards. Toy data, offscreen."""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")
from PySide6.QtGui import QKeySequence  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from astro_gpu import api  # noqa: E402
from astro_gpu.client.model import ProjectModel  # noqa: E402
from astro_gpu.client.services.worker import ApiWorker  # noqa: E402
from astro_gpu.core import project2  # noqa: E402
from astro_gpu.core.project import Target  # noqa: E402
from astro_gpu.render.layers import export_rgb, history_lines  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture()
def model(qapp, tmp_path):
    root = tmp_path / "P"
    p = project2.create(root, "P", Target(name="M 1"))
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "a1.ARW").write_bytes(b"\0" * 100)
    p.sessions.append(project2.Session(1, "night 1", lights=[str(raw)]))
    api.add_workflow(p, "S", "Full")
    api.add_workflow(p, "H", "Full")
    w = ApiWorker()
    m = ProjectModel(p, w, readonly=True)
    assert w.wait(120) and m.ready, m.error
    m.p.post = project2.new_post()
    yield m
    w.stop()


def _wf(m, name):
    return next(v for v in m.paths if v.name == name)


def test_a_combine_layers_default_name_follows_its_sides(model):
    s, h = _wf(model, "S"), _wf(model, "H")
    lid = model.add_post_layer(s.pid, "combined")
    assert model.post_layer(lid)["name"] == "S · Combine layers 2"   # (Post's default one is the first)
    model.set_layer_side(lid, "starless", h.pid)
    model.set_layer_side(lid, "stars", s.pid)
    assert model.post_layer(lid)["name"] == "H · Combine layers 2"
    model.set_post_layer(lid, name="My sky")                   # renamed by the user: kept
    model.set_layer_side(lid, "starless", s.pid)
    assert model.post_layer(lid)["name"] == "My sky"


def test_a_numbered_default_name_keeps_its_number(model):
    s, h = _wf(model, "S"), _wf(model, "H")
    model.add_post_layer(s.pid, "combined")
    lid = model.add_post_layer(s.pid, "combined")
    assert model.post_layer(lid)["name"] == "S · Combine layers 3"
    model.set_layer_side(lid, "starless", h.pid)
    assert model.post_layer(lid)["name"] == "H · Combine layers 3"


def test_a_new_layer_never_shares_the_id_of_an_existing_one(model):
    from astro_gpu.client import postengine
    s = _wf(model, "S")
    model.add_post_layer(s.pid, None)
    postengine._ids = __import__("itertools").count(1)         # a session whose counter did not see the loaded layers
    model.add_post_layer(s.pid, "combined")
    ids = [x["id"] for x in model.post_layers()]
    assert len(ids) > 1 and len(ids) == len(set(ids)), ids


def test_the_shortcuts_list_comes_from_the_actual_bindings(model, qapp):
    from astro_gpu.client import post3
    from astro_gpu.client.app import Main as MainWindow
    rows = dict(MainWindow.shortcut_rows(_Fake(MainWindow)))
    win = dict((w, k) for k, w in rows["Window"])
    assert win["Save"] == QKeySequence("Ctrl+S").toString(QKeySequence.NativeText)
    assert "Post" in rows and len(rows["Post"]) == len(post3.POST_KEYS)
    post = dict((w, k) for k, w in rows["Post"])
    assert post["Zoom 100 %"] == "1" and "Ctrl+0" in post["Fit"] and post["Delete"] == "Del"


class _Fake:
    """Just the menu bar of a window: the rows are read from its actions."""
    def __new__(cls, window_cls):
        from PySide6.QtGui import QAction
        from PySide6.QtWidgets import QMainWindow
        mw = QMainWindow()
        m = mw.menuBar().addMenu("File")
        a = QAction("Save", m)
        a.setShortcut(QKeySequence("Ctrl+S"))
        m.addAction(a)
        mw._keep = m
        mw.shortcut_rows = lambda: window_cls.shortcut_rows(mw)
        return mw


def test_every_post_key_has_a_french_text():
    from astro_gpu.client import post3
    cat = {}
    for f in (post3.__file__ and __import__("pathlib").Path(post3.__file__).parent / "locale" / "fr").glob("*.json"):
        cat.update(json.loads(f.read_text(encoding="utf-8")))
    assert [x[1] for x in post3.POST_KEYS if x[1] not in cat] == []
    assert "Shortcuts" in cat and "Window" in cat


def test_an_export_of_one_file_names_its_history_after_it(tmp_path):
    rgb = np.random.default_rng(0).random((3, 20, 30)).astype(np.float32)
    h = {"layers": ["Full"], "created": "x"}
    a = export_rgb(rgb, tmp_path, {"name": "v", "formats": ["fits32f"]}, history=h)
    b = export_rgb(rgb, tmp_path, {"name": "v", "formats": ["jpeg"]}, history=h)
    assert sorted(p.name for p in a) == ["v.fits", "v.fits.history.json"]
    assert sorted(p.name for p in b) == ["v.jpg", "v.jpg.history.json"]
    assert all(p.exists() for p in (*a, *b))


def test_the_curves_of_the_history_read_as_a_short_form():
    lay = {"name": "Full", "kind": "workflow", "pid": 1, "id": 1}
    flt = {"kind": "filter", "target": 1, "id": 2, "name": "Curves", "params": {"curves": {
        "RGB/K": {"pts": [[0.0, 0.0], [1.0, 1.0]], "interp": "Akima"}}}}
    text = "\n".join(history_lines({"project": "P", "created": "x", "layers": [flt, lay], "workflows": {}}))
    assert "curves RGB/K: 2 points, Akima" in text and "{" not in text


def test_a_long_filter_name_is_cut_at_words_with_the_continuation_indent():
    name = " ".join(["Starless"] * 6 + ["denoise"] * 6 + ["extraordinarily-long-name"] * 3 + ["end"] * 8)
    lay = {"name": "Full", "kind": "workflow", "pid": 1, "id": 1}
    flt = {"kind": "filter", "target": 1, "id": 2, "name": name, "params": {}}
    lines = history_lines({"project": "P", "created": "x", "layers": [flt, lay], "workflows": {}})
    cards = [x for x in lines if "Starless" in x or x.startswith("  ")]
    assert all(len(c) <= 72 for c in lines)
    assert all(c.startswith("  ") for c in cards[1:]) and len(cards) > 1
    assert " ".join(c.strip() for c in lines if "Starless" in c or c.startswith("  ")).count("extraordinarily-long-name") == 3
