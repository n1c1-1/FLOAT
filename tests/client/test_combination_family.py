"""The combination blocks are one family with one behaviour (the owner's rule): a combination of filters (S-H-O, H-O,
the recipes of stages/combine.py) and Combine layers (the stars put back on a starless image), and any block of the
family to come (graphcore model.JOIN_KINDS). The same rule is checked on each, so a user who knows one finds his way in
the other, and a new block of the family must pass the same tests.

The rules: placed only before Post; the same settings in their Inspect view and in Post, one source; a view of their own
in Post (the master layer); children treated alike; one name per thing and the block's shown name everywhere; one way
of naming a step (the setting, then its owner).

The window tests open a scratch copy of the owner's reference projects (their store read in place, nothing written in
them: tests/conftest.py scratch_project)."""
from __future__ import annotations

import os
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication  # noqa: E402

from astro_gpu.graphcore.model import JOIN_KINDS  # noqa: E402

FAMILY = tuple(JOIN_KINDS)


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def pump(s=0.0):
    t0 = time.perf_counter()
    while True:
        QApplication.processEvents()
        if time.perf_counter() - t0 >= s:
            break
        time.sleep(0.005)


# ---------------------------------------------------------------------------------------------- one name per thing
def test_the_starless_image_has_one_name_everywhere(qapp):
    """The Combine layers' input of the starless image was "Background" on the graph, "Starless" in its panel and
    "Starless image" in Post's row (French "Fond de ciel", "Sans étoiles"): the row's words everywhere."""
    from astro_gpu.client import i18n
    from astro_gpu.client.graphview.style import port_title
    from astro_gpu.client.postengine import SUB_NAMES
    for lang in ("en", "fr"):
        i18n.set_language(lang)
        try:
            row = i18n._(SUB_NAMES["starless"])
            assert port_title("Background") == port_title("Starless") == row, lang
            assert i18n._(SUB_NAMES["stars"]) == port_title("Stars")
        finally:
            i18n.set_language("en")


IC443 = "F:/astro_essais/tutoriels_ref/MONO/ic443"


def _whole(tmp_path, name):
    """The window with its real Graph, Inspect and Post pages on a scratch copy of a reference project."""
    from pathlib import Path

    from conftest import scratch_project
    from test_inspect import Harness

    from astro_gpu.client import app as appmod
    from astro_gpu.client import prefs
    src = Path(name)
    if not (src / "project.astroproj").is_file():
        pytest.skip(f"the project {src.name} is not here")
    os.environ["ASTRO_GPU_SETTINGS"] = str(tmp_path / "settings.json")
    prefs.load()
    appmod.apply_language("en")

    class Whole(Harness):
        def make_model(self):
            appmod.Main.make_model(self)

        def graph_page(self):
            return appmod.Main.graph_page(self)

        def inspect_page(self):
            return appmod.Main.inspect_page(self)
    w = Whole(readonly=True)
    w.resize(1500, 900)
    w.show()
    w.open_project(scratch_project(tmp_path / src.name, src))
    for _i in range(400):
        if w.s.ready:
            break
        w.worker.wait(5)
        time.sleep(0.05)
    assert w.s.ready, w.s.error
    pump(0.3)
    return w


def test_real_a_combination_is_named_by_its_block_s_title_in_post_and_in_the_panels(qapp, tmp_path):
    """IC 443's combinations were placed as S-H-O and made H-O (their workflows are stored "S-H-O"): the graph titles
    them H-O, Post's row of their Combine layers said "S-H-O · Combine layers", the Combine layers panel "S-H-O". The
    block's shown name everywhere; what the project stores stays as it was."""
    from astro_gpu.client.graphview.gestures import Gestures
    from astro_gpu.client.post3 import layer_name
    from astro_gpu.client.postengine import save_layers
    w = _whole(tmp_path, IC443)
    try:
        s = w.s
        w.set_mode(2)
        pump(1.0)
        pv = w._post_holder.page.view
        pv.rebuild()
        lay = next(x for x in pv.layers if x.get("kind") == "workflow" and x.get("part") == "combined"
                   and x.get("starless") == "output")
        assert lay["name"].startswith("S-H-O")                          # stored as it was
        title = s.shown_title(lay["pid"])
        assert title.startswith("H-O")
        row = next(r for r in pv.rows if r.lay["id"] == lay["id"] and r.lay["kind"] == "workflow")
        assert layer_name(lay) == row.n._full == title + " · Combine layers"
        assert all("retitle" not in d for d in save_layers(pv.layers))  # never written into the project
        # the Combine layers block's panel: the workflow it belongs to, its inputs, by the same names
        w.set_mode(0)
        pump(0.5)
        cl = next(k for k, b in w.graph.bridge.graph.blocks.items() if b.kind == "combine_layers")
        Gestures(w.graph).click(cl)
        pump(0.8)
        from PySide6.QtWidgets import QLabel
        texts = [x.text() for x in w.graph.panel.findChildren(QLabel) if x.isVisibleTo(w.graph.panel)]
        assert title in texts and not any(t.startswith("S-H-O") for t in texts), texts
        assert any("Starless image" in t for t in texts), texts
    finally:
        w.close()
