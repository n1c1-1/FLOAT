"""A project always opens (the engine owner's review of the format-3 file): a project whose graph this version cannot
compile opens, when the user accepts, from the definition kept at its last save (core/project2.WORKING_FILE), converted
to a graph as a file of the previous format, nothing written until Save; without that copy the message names the graph
and its cause. A save that has to write the definition instead of the graph says so. Every golden shape, and every
definition of the golden record when its folder is on this machine, saves as format 3 through the graph page."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "graphcore"))
import test_golden  # noqa: E402
from PySide6.QtWidgets import QApplication, QPushButton  # noqa: E402

from astro_gpu.client.model import ProjectModel  # noqa: E402
from astro_gpu.client.services.worker import ApiWorker  # noqa: E402
from astro_gpu.core import project2  # noqa: E402
from astro_gpu.core.project2 import Project2  # noqa: E402
from astro_gpu.graphcore.compile import compile_into  # noqa: E402
from astro_gpu.graphcore.convert import convert  # noqa: E402
from astro_gpu.graphcore.model import to_dict as graph_dict  # noqa: E402
from tests.client.test_swap_smoke import env, made_from, qapp, ready, settle, win  # noqa: E402,F401

GOLDEN = Path(os.environ.get("FLOAT_GOLDEN") or "C:/Users/Nicolas/sandbox/float_graphe_golden") / "defs"


def later_bug(monkeypatch):
    """The compile of this version fails on the file's graph (a later bug): the file is the one its save wrote."""
    def fails(d, root):
        raise project2.GraphUnreadable("RuntimeError: a later bug of the compile")
    monkeypatch.setattr(project2, "_from_graph", fails)


def file_format(root: Path):
    return json.loads((root / project2.FILE).read_text(encoding="utf-8"))["format"]


def saved_shape(root: Path, shape: str = "fan_in") -> Project2:
    p = test_golden.SHAPES[shape]()
    g = convert(p)
    compile_into(g, p)
    p.graph = graph_dict(g)
    project2.save(p, root)
    assert file_format(root) == project2.GRAPH_FORMAT
    return p


def test_model_a_graph_that_cannot_be_compiled_opens_from_the_copy_kept_at_its_last_save(qapp, tmp_path,  # noqa: F811
                                                                                        monkeypatch):
    root = tmp_path / "P"
    p = saved_shape(root)
    before = (root / project2.FILE).read_bytes()
    later_bug(monkeypatch)
    with pytest.raises(project2.GraphUnreadable) as e:
        project2.load(root)
    assert e.value.working is not None and e.value.working_current
    w = ApiWorker()
    m = ProjectModel(project2.load_working(root), w)
    try:
        assert w.wait(120) and m.ready
        assert {r.name for r in convert(p).rows} <= {r.name for r in m._g.rows}      # the graph drawn again
        assert [x.name for x in m.p.workflows] == [x.name for x in p.workflows]
        assert (root / project2.FILE).read_bytes() == before                         # nothing written
        m.edited()
        m.flush()                                    # this version's compile still fails: the definition, said
        assert file_format(root) == project2.FORMAT and m.save_note
        assert any(t.startswith("Saved in the previous format") for _t, t, _a in m.history)
        monkeypatch.undo()
        m.edited()
        m.flush()
        assert file_format(root) == project2.GRAPH_FORMAT and m.save_note == ""
        assert [x.name for x in project2.load(root).workflows] == [x.name for x in p.workflows]
    finally:
        m.close(save=False)
        w.stop()


@pytest.mark.parametrize("shape", list(test_golden.SHAPES))
def test_every_golden_shape_saves_as_format_3_through_the_graph_page(qapp, tmp_path, shape):  # noqa: F811
    root = tmp_path / "P"
    project2.save(test_golden.SHAPES[shape](), root)                    # a file of the previous format
    w = ApiWorker()
    m = ProjectModel(project2.load(root), w)
    try:
        assert w.wait(120) and m.ready
        m.edited()
        m.flush()
        assert file_format(root) == project2.GRAPH_FORMAT, m.p.graph_dropped
        assert m.save_note == ""
        assert project2.load(root).to_dict() == m.p.to_dict()
    finally:
        m.close(save=False)
        w.stop()


def golden_defs():
    return sorted(GOLDEN.glob("*.astroproj")) if GOLDEN.is_dir() else []


@pytest.mark.skipif(not golden_defs(), reason="no golden record on this machine")
def test_every_golden_definition_saves_as_format_3_as_the_graph_page_makes_its_graph(tmp_path):
    """The graph page's graph of a definition: converted, every default written, compiled into it; saved: format 3."""
    from astro_gpu import api
    from astro_gpu.client.graphdefaults import with_explicit_defaults
    from astro_gpu.client.i18n import _
    done = 0
    for f in golden_defs():
        try:
            p = Project2.from_dict(json.loads(f.read_text(encoding="utf-8")), tmp_path / f.stem)
        except project2.EarlierVersionProject:
            continue
        g = with_explicit_defaults(convert(p), _, p, api._ctx(p).rules())[0]
        compile_into(g, p)
        p.graph = graph_dict(g)
        project2.save(p)
        assert file_format(tmp_path / f.stem) == project2.GRAPH_FORMAT, (f.stem, p.graph_dropped)
        done += 1
    assert done > 20


# ---------------------------------------------------------------------------------------------- the main window
def banner_buttons(w) -> dict:
    box = w.banner.actions_box
    return {b.text(): b for b in (box.itemAt(i).widget() for i in range(box.count())) if isinstance(b, QPushButton)}


def test_the_window_offers_the_copy_kept_at_the_last_save_and_says_it_plainly_without_it(win, env, monkeypatch):  # noqa: F811
    root = made_from(win, env, "filters_min")
    win.save()
    settle(win)
    names = [x.name for x in win.s.p.workflows]
    assert file_format(root) == project2.GRAPH_FORMAT and (root / project2.WORKING_FILE).is_file()
    assert win.close_project(go_home=False)
    before = (root / project2.FILE).read_bytes()
    later_bug(monkeypatch)
    # without the copy: the plain message, naming the graph and its cause; nothing offered
    (root / project2.WORKING_FILE).replace(env["tmp"] / "kept_aside.json")
    assert not win.open_project(root)
    assert win.banner.isVisible() and not banner_buttons(win)
    assert win.banner.text.text().startswith("This project cannot be opened: its graph cannot be read by this version (")
    # with it: offered, opened from it on the user's word, the file left as it is until Save
    (env["tmp"] / "kept_aside.json").replace(root / project2.WORKING_FILE)
    assert not win.open_project(root) and win.project is None
    assert "The copy kept at its last save can be opened" in win.banner.text.text()
    banner_buttons(win)["Open copy"].click()
    ready(win)
    assert win.project is not None and [x.name for x in win.s.p.workflows] == names
    assert win.banner.isVisible() and win.banner.text.text().startswith("Opened from the copy kept at its last save")
    assert win.s.dirty and (root / project2.FILE).read_bytes() == before
    # saved while this version still cannot compile it: the definition is written, and the window says so
    win.save()
    settle(win)
    assert file_format(root) == project2.FORMAT
    assert win.banner.isVisible() and win.banner.text.text().startswith("Saved in the previous format")
    monkeypatch.undo()
    win.s.edited()
    win.save()
    settle(win)
    assert file_format(root) == project2.GRAPH_FORMAT
    assert [x.name for x in project2.load(root).workflows] == names
    assert not env["errors"], env["errors"]


def test_a_newer_version_s_file_is_said_in_french_too(win, env):  # noqa: F811
    from astro_gpu.client import app as appmod
    root = env["tmp"] / "newer"
    project2.save(test_golden.SHAPES["fan_in"](), root)
    d = json.loads((root / project2.FILE).read_text(encoding="utf-8"))
    d["format"] = project2.GRAPH_FORMAT + 1
    (root / project2.FILE).write_text(json.dumps(d), encoding="utf-8")
    assert not win.open_project(root)
    assert win.banner.text.text() == "This project was made with a newer FLOAT: open it with that version."
    appmod.apply_language("fr")
    assert not win.open_project(root)
    assert win.banner.text.text() == ("Ce projet a été fait avec une version plus récente de FLOAT : ouvrez-le avec "
                                      "celle-ci.")
    QApplication.processEvents()
