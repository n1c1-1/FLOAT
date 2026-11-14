"""The bench of the legacy ports (test_gv_legacy_*.py): the behaviours today's graph page was tested for, checked on the
new page over a project model. A scratch project whose frames are empty files (nothing is computed), the project model
with the graph as its truth, the new page on it, shown offscreen; helpers to find a block by its view key and to drive
the page with real events (graphview/gestures.py)."""
from __future__ import annotations

import time
from pathlib import Path

from PySide6.QtWidgets import QApplication


def frames(folder: Path, names) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    for i, n in enumerate(names):
        (folder / n).write_bytes(b"\0" * (100 + i))
    return folder


def pump(n: int = 4):
    for _i in range(n):
        QApplication.processEvents()


class World:
    """A project model and the new graph page on it (1500 x 800, shown offscreen).

    ``workflows``: option templates of the engine (core.workflow.TEMPLATES) added before the model opens; ``filters``:
    one session per filter (per-session stacking), no workflow; ``nights``: sessions of one colour night each."""

    def __init__(self, tmp_path: Path, workflows=("Full",), lights=True, filters=(), nights=1, stack_mode=None,
                 size=(1500, 800), chains=None):
        from astro_gpu.client.graphview.gestures import Gestures
        from astro_gpu.client.graphview.page import GraphPage
        from astro_gpu.client.model import ProjectModel
        from astro_gpu.client.services.worker import ApiWorker
        from astro_gpu.core import project2
        from astro_gpu.core import workflow as wfl
        from astro_gpu.core.project import Target
        self.tmp = tmp_path
        p = project2.create(tmp_path / "P", "P", Target(name="M 31"))
        raw = tmp_path / "raw"
        self.raw = raw
        if filters:
            for i, f in enumerate(filters, 1):
                p.sessions.append(project2.Session(i, f, lights=[str(frames(raw / f, ["a1.fits"]))], filter=f))
            p.stack_mode = "per_session"
        else:
            for i in range(1, nights + 1):
                lt = [str(frames(raw / f"n{i}", [f"a{i}.ARW", f"b{i}.ARW"]))] if lights else []
                p.sessions.append(project2.Session(i, f"night {i}", lights=lt))
        if stack_mode:
            p.stack_mode = stack_mode
        for name in workflows:
            p.add_workflow(name, wfl.TEMPLATES[name])
        for name, blocks in (chains or {}).items():
            w = p.add_workflow(name, wfl.DEFAULT_OPTIONS)
            p.set_blocks(w.id, [wfl.node(b) for b in blocks])
        project2.save(p)
        self.w = ApiWorker()
        self.m = ProjectModel(project2.load(tmp_path / "P"), self.w)
        assert self.w.wait(120) and self.m.ready, self.m.error
        self.page = GraphPage(self.m)
        self.page.resize(*size)
        self.page.show()
        self.said = []
        self.page.toasted.connect(self.said.append)
        self.b = self.page.bridge
        self.g = Gestures(self.page)
        self.settle()

    # ------------------------------------------------------------------ time
    def settle(self, n: int = 6):
        for _i in range(n):
            self.w.wait(60)
            QApplication.processEvents()
            time.sleep(0.005)
        self.page.palette.finish()
        QApplication.processEvents()

    def close(self):
        self.page.close()
        self.page.deleteLater()
        self.m.close(save=False)
        self.w.stop()
        pump()

    # ------------------------------------------------------------------ the graph
    @property
    def graph(self):
        return self.b.graph

    @property
    def bm(self):
        return self.m.block_map

    def view(self, name):
        return next(q for q in self.m.paths if q.name == name)

    def block(self, key, p=None):
        """The block id of a view key ("fond", "calibration@2", "post" = the block holding the Output) in a workflow
        (its name, its view, its id; the current one by default)."""
        if isinstance(p, str):
            p = self.view(p)
        if p is None:
            p = self.m.path
        n = self.bm.node_of(p, key)
        return self.bm.block_of(n) if n is not None else None

    def kinds(self, kind):
        return [b for b, x in self.graph.blocks.items() if x.kind == kind]

    def one(self, kind):
        ids = self.kinds(kind)
        assert len(ids) == 1, (kind, ids)
        return ids[0]

    def choose(self, b):
        self.b.choose(b)
        self.settle(2)

    def card(self, kind):
        self.page.palette.finish()
        return self.page.palette.cards[kind]

    def template(self, key, **over):
        from astro_gpu.client.gallery import GALLERY
        t = dict(next(t for t in GALLERY if t["key"] == key), **over)
        self.page.template_chosen(t)
        self.settle()
        return t

    def steps(self) -> int:
        return len(self.m._undo)
