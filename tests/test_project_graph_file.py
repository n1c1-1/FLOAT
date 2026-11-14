"""The project file of the graph page (format 3: the stored graph and what is not graph): saved, then read back by the
engine (``Project2.load``, which compiles the definition from the graph), it gives the same definition and the same Keys
tables as before it was saved. On the golden shapes always, and on every definition of the golden record when its
folder is on this machine. A definition edited outside the graph page is saved as format 2 (the definition wins), and
a format-2 file is read as today."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests" / "graphcore"))
from astro_gpu.core import project2  # noqa: E402
from astro_gpu.core.project2 import Project2  # noqa: E402
from astro_gpu.graphcore.compile import _NoDisk, _key_rules, compile_into  # noqa: E402
from astro_gpu.graphcore.convert import convert  # noqa: E402
from astro_gpu.graphcore.model import to_dict as graph_dict  # noqa: E402
from astro_gpu.graphcore.proof import keys_table  # noqa: E402

GOLDEN = Path(os.environ.get("FLOAT_GOLDEN") or "C:/Users/Nicolas/sandbox/float_graphe_golden") / "defs"


def shapes():
    import test_golden
    return {k: f for k, f in test_golden.SHAPES.items()}


def golden_defs():
    return sorted(GOLDEN.glob("*.astroproj")) if GOLDEN.is_dir() else []


def round_trip(p: Project2, folder: Path):
    """``p`` with its graph, saved in ``folder``, read back: (the file's content, the project read)."""
    g = convert(p)
    q = Project2.from_dict(p.to_dict(), None)
    compile_into(g, q)
    assert q.to_dict() == p.to_dict(), "the graph does not compile to the definition"   # the golden proof's own gate
    p.graph = graph_dict(g)
    project2.save(p, folder)
    d = json.loads((folder / project2.FILE).read_text(encoding="utf-8"))
    return d, project2.load(folder)


def same(p: Project2, back: Project2):
    a, b = p.to_dict(), back.to_dict()
    assert a == b
    rules = _key_rules()
    assert keys_table(p, rules, _NoDisk()) == keys_table(back, rules, _NoDisk())


@pytest.mark.parametrize("shape", list(shapes()))
def test_a_shape_saved_with_its_graph_reads_back_the_same(shape, tmp_path):
    p = shapes()[shape]()
    d, back = round_trip(p, tmp_path / "P")
    assert d["format"] == project2.GRAPH_FORMAT and "graph" in d
    assert not any(k in d for k in project2.GRAPH_PARTS)           # nothing derived in the file
    same(p, back)
    assert back.graph == d["graph"]


@pytest.mark.skipif(not golden_defs(), reason="no golden record on this machine")
def test_every_golden_definition_saved_with_its_graph_reads_back_the_same(tmp_path):
    done, slow = 0, (0.0, "")
    for f in golden_defs():
        d0 = json.loads(f.read_text(encoding="utf-8"))
        try:
            p = Project2.from_dict(d0, tmp_path / f.stem)
        except project2.EarlierVersionProject:
            continue
        _d, back = round_trip(p, tmp_path / f.stem)
        assert _d["format"] == project2.GRAPH_FORMAT, f.stem          # never silently the definition (format 2)
        same(p, back)
        t0 = time.perf_counter()
        project2.load(tmp_path / f.stem)
        dt = time.perf_counter() - t0
        slow = max(slow, (dt, f.stem))
        done += 1
    assert done > 20
    print(f"\nslowest format-3 read: {slow[0] * 1000:.0f} ms ({slow[1]})")


def test_a_definition_edited_outside_the_page_is_saved_as_format_2(tmp_path):
    p = shapes()["fan_in"]()
    round_trip(p, tmp_path / "P")
    p = project2.load(tmp_path / "P")
    p.workflows[0].name = "renamed by a script"                     # the graph does not say this any more
    project2.save(p)
    d = json.loads((tmp_path / "P" / project2.FILE).read_text(encoding="utf-8"))
    assert d["format"] == project2.FORMAT and "graph" not in d
    assert project2.load(tmp_path / "P").workflows[0].name == "renamed by a script"


def test_a_plan_freezes_the_compiled_definition(tmp_path):
    """The definition a run's plan freezes (``to_dict``) is format 2, the same before and after the graph."""
    p = shapes()["variants_and_taps"]()
    before = json.dumps(p.to_dict(), sort_keys=True)
    _d, back = round_trip(p, tmp_path / "P")
    assert back.to_dict()["format"] == project2.FORMAT and "graph" not in back.to_dict()
    assert json.dumps(back.to_dict(), sort_keys=True).replace(str(tmp_path / "P"), "") == before


# ---------------------------------------------------------------------------------------------- a project always opens
def saved(folder: Path, shape: str = "fan_in"):
    """A shape saved with its graph in ``folder``: (the folder, the definition)."""
    p = shapes()[shape]()
    round_trip(p, folder)
    return folder, p


def edit_file(root: Path, change):
    f = root / project2.FILE
    d = json.loads(f.read_text(encoding="utf-8"))
    change(d)
    f.write_text(json.dumps(d, indent=2), encoding="utf-8")


def unreadable(root: Path) -> project2.GraphUnreadable:
    with pytest.raises(project2.GraphUnreadable) as e:
        project2.load(root)
    return e.value


def test_a_format_3_save_keeps_the_compiled_definition_as_a_working_file(tmp_path):
    root, p = saved(tmp_path / "P")
    w = json.loads((root / project2.WORKING_FILE).read_text(encoding="utf-8"))
    assert w["definition"] == project2.load(root).to_dict() == p.to_dict()
    assert w["project_file_sha256"] == hashlib.sha256((root / project2.FILE).read_bytes()).hexdigest()
    back = project2.load_working(root)
    assert back.graph is None
    same(p, back)


def test_a_graph_a_newer_version_wrote_opens_from_the_working_file(tmp_path):
    root, p = saved(tmp_path / "P")
    edit_file(root, lambda d: d["graph"]["rows"][0].update(warp=1))           # a field this version does not know
    e = unreadable(root)
    assert "“warp”" in e.cause and "newer FLOAT" in e.cause and "graph" in str(e)
    assert e.working == root / project2.WORKING_FILE and not e.working_current     # the file changed since that save
    same(p, project2.load_working(root))


def test_a_compile_that_fails_later_offers_the_working_file_of_that_very_save(tmp_path, monkeypatch):
    root, p = saved(tmp_path / "P")
    from astro_gpu.graphcore import compile as gc

    def later_bug(g, q):
        raise RuntimeError("a later bug of the compile")
    monkeypatch.setattr(gc, "compile_into", later_bug)
    e = unreadable(root)
    assert e.cause == "RuntimeError: a later bug of the compile" and e.working_current
    monkeypatch.undo()
    same(p, project2.load_working(root))


def test_without_the_working_file_the_error_names_the_graph_and_its_cause(tmp_path):
    root, _p = saved(tmp_path / "P")
    (root / project2.WORKING_FILE).replace(tmp_path / "kept_aside.json")
    edit_file(root, lambda d: d["graph"]["rows"][0]["blocks"].append("b999"))   # a block the graph does not have
    e = unreadable(root)
    assert e.working is None and e.cause == "KeyError: 'b999'"
    assert str(e).startswith("the graph of this project cannot be compiled")
    with pytest.raises(OSError):
        project2.load_working(root)


def test_what_the_engine_refuses_in_a_compiled_graph_is_said_of_the_graph(tmp_path):
    def swap(d):
        r = d["graph"]["rows"][0]
        r["blocks"][0], r["blocks"][1] = r["blocks"][1], r["blocks"][0]
    root, _p = saved(tmp_path / "P")
    edit_file(root, swap)
    assert unreadable(root).cause.startswith("the graph's row “Both nights”, block Registration: ")
    root, _p = saved(tmp_path / "Q")
    edit_file(root, lambda d: d["graph"]["nights"][0].update(filter="Zz"))
    assert unreadable(root).cause.startswith("the graph's night “N1”: filter 'Zz' is not one of")


def test_a_file_of_a_newer_format_says_a_newer_float_made_it(tmp_path):
    root, _p = saved(tmp_path / "P")
    edit_file(root, lambda d: d.update(format=project2.GRAPH_FORMAT + 1))
    with pytest.raises(project2.NewerVersionProject, match="made with a newer FLOAT"):
        project2.load(root)


def test_a_graph_written_as_the_definition_is_never_silent(tmp_path, caplog):
    root, _p = saved(tmp_path / "P")
    p = project2.load(root)
    p.workflows[0].name = "renamed by a script"
    with caplog.at_level(logging.WARNING, logger="astro_gpu.core.project2"):
        project2.save(p)
    said = [r.getMessage() for r in caplog.records if "graph dropped" in r.getMessage()]
    assert len(said) == 1 and "workflows differ" in said[0]
    assert "workflows differ" in p.graph_dropped and p.graph is None                   # dropped, said once
    caplog.clear()
    p.graph = {"rows": "not a graph"}
    with caplog.at_level(logging.WARNING, logger="astro_gpu.core.project2"):
        project2.save(p)
    assert p.graph_dropped.startswith("its graph does not compile (") and len(caplog.records) == 1
    project2.save(p)                                                                   # no graph: nothing to say
    assert p.graph_dropped == ""
