"""Save as writes the copy in the graph format with the same graph: its blocks, wires and view as they were, its
entries made absolute as the copy's definition gets them (a relative entry kept only when it still finds its file
from the new folder). The copy was written as a definition without its graph ("its graph compiles to another
definition"), and opened converted from it: what only the graph holds was lost without a word."""
from __future__ import annotations

import copy
import logging
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
pytest.importorskip("PySide6")
sys.path.insert(0, str(ROOT / "tests" / "graphcore"))
import test_h1_old_way as h1  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from astro_gpu import api  # noqa: E402
from astro_gpu.client import app as appmod  # noqa: E402
from astro_gpu.client.model import ProjectModel  # noqa: E402
from astro_gpu.client.steps import Steps  # noqa: E402
from astro_gpu.core import project2  # noqa: E402
from astro_gpu.graphcore import model as gm  # noqa: E402


class Win(appmod.Main):
    """The window, its Graph page a placeholder over the real model."""

    def graph_page(self):
        if self.s is None or self.s.p is not self.project.definition:
            self.s = ProjectModel(self.project.definition, self.worker, readonly=False)
        self.steps = Steps(self.s)
        return appmod.Placeholder("Graph")


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _open(root):
    w = Win()
    w.show()
    w.open_project(root)
    for _i in range(400):
        if w.s is not None and w.s.ready:
            break
        w.worker.wait(5)
        QApplication.processEvents()
        time.sleep(0.02)
    assert w.s.ready, w.s.error
    return w


def _absolute(d, old_root):
    """The graph dict with every entry made absolute from the old folder (what the copy's graph must hold)."""
    def ab(x):
        return x if Path(x).is_absolute() else (Path(old_root) / x).as_posix()
    d = copy.deepcopy(d)
    for b in d["blocks"].values() if isinstance(d["blocks"], dict) else d["blocks"]:
        for k, v in (b.get("files") or {}).items():
            b["files"][k] = [ab(x) for x in v]
    for k, v in d["common"]["files"].items():
        d["common"]["files"][k] = [ab(x) for x in v]
    for n in d.get("nights") or []:
        for k, v in (n.get("files") or {}).items():
            n["files"][k] = [ab(x) for x in v]
    return d


def test_save_as_keeps_the_graph_and_makes_its_entries_absolute(qapp, tmp_path, monkeypatch, caplog):
    root = tmp_path / "ws" / "Orig"
    w0 = h1.old_way(root, next(t for t in h1.GALLERY if t["key"] == "colour_min"), h1.FAKE[:1])
    project2.save(w0.p)
    w0.close()
    (root / "frames").mkdir()
    (root / "frames" / "a.fit").write_bytes(b"\0" * 64)
    w = _open(root)
    try:
        sid = next(b for b, x in w.s.stored.blocks.items() if x.kind == "session")
        assert w.s.apply([("add_files", {"session": sid, "kind": "lights", "paths": ["frames/a.fit"],
                                         "mode": "add"})]).ok                   # a frame the project holds itself
        before = gm.to_dict(w.s.stored)
        keys_before = {wf.id: api.stage_key(w.s.p, wf, "stack") for wf in w.s.p.workflows}
        target = tmp_path / "ws" / "Copy"
        target.mkdir()
        monkeypatch.setattr(appmod.QFileDialog, "getExistingDirectory", staticmethod(lambda *a, **k: str(target)))
        caplog.set_level(logging.WARNING)
        w.save_as()
        assert not [r for r in caplog.records if "graph dropped" in r.getMessage()]
        d = project2.read_json(target / project2.FILE) if hasattr(project2, "read_json") else None
        if d is None:
            import json
            d = json.loads((target / project2.FILE).read_text(encoding="utf-8"))
        assert d.get("format") == project2.GRAPH_FORMAT and d.get("graph")
        assert gm.to_dict(gm.from_dict(d["graph"])) == _absolute(before, root)
        copy_p = project2.load(target)
        lights = [x for s in copy_p.sessions for x in s.lights]
        assert (root / "frames" / "a.fit").as_posix() in lights
        assert {wf.id: api.stage_key(copy_p, wf, "stack") for wf in copy_p.workflows} == keys_before
    finally:
        w.close()


def test_the_first_naming_keeps_the_graph(qapp, tmp_path, monkeypatch, caplog):
    import json
    frame = tmp_path / "frames" / "a.fit"
    frame.parent.mkdir()
    frame.write_bytes(b"\0" * 64)
    w = Win()
    w.show()
    w.new_project()
    for _i in range(400):
        if w.s is not None and w.s.ready:
            break
        w.worker.wait(5)
        QApplication.processEvents()
        time.sleep(0.02)
    try:
        assert w.s.apply([("add_row", {"kind": "session", "files": {"lights": [frame.as_posix()]}})]).ok
        assert w.s.apply([("add_block", {"kind": "calibration",
                                         "at": {"type": "calib", "night": w.s.stored.rows[-1].nights[0][0]}})]).ok
        before = gm.to_dict(w.s.stored)
        target = tmp_path / "ws" / "Named"
        monkeypatch.setattr(appmod, "ask_project_name", lambda *a, **k: (target, "Named"))
        caplog.set_level(logging.WARNING)
        assert w.name_project()
        assert not [r for r in caplog.records if "graph dropped" in r.getMessage()]
        d = json.loads((target / project2.FILE).read_text(encoding="utf-8"))
        assert d.get("format") == project2.GRAPH_FORMAT
        assert gm.to_dict(gm.from_dict(d["graph"])) == before
    finally:
        w.close()
