"""The application on the new graph page, end to end (the swap's smoke proof): the main window opens; a new project is
made from each template of the gallery; a project saved by today's FLOAT (format 2) opens, converted on open, its
definition unchanged; blocks chosen on the page show the real settings panel; a setting changed is undone and redone;
a block's image goes to Post and Post opens; a run on a tiny project shows the blocks' states and its progress over
the graph; a project saved, closed and opened again gives the same graph.

Nothing here writes outside the test's own folder: the study copies are copied first, their scratch folder moved into
the test's folder. A message box shown by the application fails the test (none may be shown but for a crash)."""
from __future__ import annotations

import json
import logging
import os
import shutil
import time
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from astro_gpu import api  # noqa: E402
from astro_gpu.client import app as appmod  # noqa: E402
from astro_gpu.client import prefs  # noqa: E402
from astro_gpu.core import project as project_mod  # noqa: E402
from astro_gpu.core.params import P, StageParams  # noqa: E402
from astro_gpu.core.project import list_frames  # noqa: E402
from astro_gpu.core.stage import Stage  # noqa: E402
from astro_gpu.core.store import KeyRules  # noqa: E402
from astro_gpu.core.workflow import Chains  # noqa: E402

HERE = Path(__file__).parent
SPEC = "test_swap_smoke:toy_rules"
#: a project saved by today's FLOAT, copied before it is opened (never opened where it is)
COPIES = Path("C:/Users/Nicolas/sandbox/float_graphe_etude/copies")
COPY = COPIES / "52_dslr_refractor_canon40da_rosette"


# ---------------------------------------------------------------------------------------------- toy stages (test_run's)
class InvParams(StageParams):
    max_lights: int = P(0, label="max lights", help="", ge=0)


class Inv(Stage):
    name = "inventory"
    title = "Inventory"
    Params = InvParams

    def run(self, ctx, p):
        files = list_frames(ctx.project.config.inputs.lights)
        ctx.save_json("frames.json", {"lights": [{"path": str(f)} for f in files]})
        ctx.metric("n_lights", len(files))


class ToyStackParams(StageParams):
    scale: float = P(1.5, label="Scale", help="", ge=1.0, le=3.0)
    workers: int = P(4, label="workers", help="", ge=1)
    exclude: list[str] = P([], label="exclude", help="", level="internal")


class ToyStack(Stage):
    name = "stack"
    title = "Stack"
    deps = ("inventory",)
    Params = ToyStackParams

    def run(self, ctx, p):
        n = len(ctx.load_json("inventory", "frames.json")["lights"])
        ctx.save_json("stack.json", {"n": n, "scale": p.scale})


class BParams(StageParams):
    steps: int = P(8, label="steps", help="", ge=1)
    delay: float = P(0.0, label="delay", help="seconds per step", ge=0)


class B(Stage):
    name = "denoise"
    title = "B"
    deps = ("stack",)
    Params = BParams

    def run(self, ctx, p):
        for i in range(p.steps):
            time.sleep(p.delay)
            ctx.progress((i + 1) / p.steps, f"step {i + 1}", force=True)
        ctx.save_json("b.json", {"steps": p.steps})


class CParams(StageParams):
    k: float = P(1.0, label="k", help="")


class C(Stage):
    name = "color"
    title = "C"
    deps = ("stack", "denoise")
    Params = CParams

    def run(self, ctx, p):
        ctx.save_json("c.json", {"n": ctx.load_json("stack", "stack.json")["n"]})


class E(Stage):
    name = "render_simple"
    title = "E"
    deps = ("color",)
    Params = CParams

    def run(self, ctx, p):
        ctx.save_json("e.json", ctx.load_json("color", "c.json"))


def toy_rule(options, end, reference):
    return ["inventory", "stack"] + (["denoise"] if options.get("denoise") else []) + ["color", "render_simple"]


def toy_rules() -> KeyRules:
    """Also built by the detached worker (plan.rules = 'test_swap_smoke:toy_rules')."""
    chains = Chains([Inv(), ToyStack(), B(), C(), E()], extra={"render_simple": ("denoise",)},
                    optional={"color": ("denoise",), "render_simple": ("denoise",)},
                    blocks={"calibration": ("inventory",), "stacking": ("stack",), "denoise": ("denoise",),
                            "color": ("color",), "output": ("render_simple",)}, rule=toy_rule)
    return KeyRules(chains, auto_fields={}, non_key={"stack": ("workers",)}, implied=lambda options, stage: {})


# ---------------------------------------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def qapp():
    a = QApplication.instance() or QApplication([])
    from astro_gpu.client.widgets import QSS
    a.setStyleSheet(QSS + appmod.EXTRA_QSS)
    return a


class Errors(logging.Handler):
    """The errors the application logged (a gesture or a view that failed is logged, never raised into Qt)."""

    def __init__(self):
        super().__init__(logging.ERROR)
        self.got = []

    def emit(self, record):
        self.got.append(self.format(record))


@pytest.fixture()
def env(tmp_path, monkeypatch, qapp):
    monkeypatch.setenv("ASTRO_GPU_SETTINGS", str(tmp_path / "settings.json"))
    prefs.load()
    appmod.apply_language("en")
    shown = []
    for kind in ("information", "warning", "critical", "question"):
        monkeypatch.setattr(QMessageBox, kind, lambda *a, _k=kind, **k: shown.append((_k, a[1:3])))
    monkeypatch.setattr(QMessageBox, "exec", lambda self: shown.append(("exec", self.text())))
    state = tmp_path / "state"
    monkeypatch.setattr(project_mod, "STATE", state)
    monkeypatch.setenv("ASTRO_GPU_HOME", str(state))
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(HERE), os.environ.get("PYTHONPATH", "")]))
    ws = tmp_path / "ws"
    ws.mkdir()
    prefs.set("last_folder", ws.as_posix())
    prefs.set("cache_dir", (tmp_path / "scratch").as_posix())
    prefs.set("device", "cpu")
    errors = Errors()
    logging.getLogger("astro_gpu").addHandler(errors)
    yield {"tmp": tmp_path, "shown": shown, "errors": errors.got}
    logging.getLogger("astro_gpu").removeHandler(errors)
    appmod.apply_language("en")
    assert not shown, f"a message box was shown: {shown}"


@pytest.fixture()
def win(env):
    w = appmod.Main()
    w.resize(1500, 900)
    w.show()
    assert w.worker.wait(60)
    yield w
    if w.project is not None:
        w.close_project(go_home=False, discard=True)
    w.close()
    w.deleteLater()
    settle(w)


def settle(w, n: int = 12):
    for _i in range(n):
        w.worker.wait(30)
        QApplication.processEvents()
        time.sleep(0.01)


def pump(cond, timeout: float = 90.0) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        QApplication.processEvents()
        if cond():
            return True
        time.sleep(0.005)
    return False


def ready(w, timeout: float = 120.0):
    assert pump(lambda: w.s is not None and w.s.ready, timeout), getattr(w.s, "error", "")
    settle(w)


def choose_every_block(w) -> list:
    """Each block of the graph chosen on the page: the settings panel shows it (its node, a title)."""
    from astro_gpu.client.panel import Panel
    page, s = w.graph, w.s
    shown = []
    for bid, blk in list(s.stored.blocks.items()):
        page.bridge.choose(bid)
        settle(w, 3)
        pn = page.panel
        assert isinstance(pn, Panel)
        node = pn.node
        if blk.kind in ("post",):
            assert node is not None and node.key == "final", (bid, blk.kind)
        elif blk.kind not in ("combine_layers",) and s.block_map.node(bid) is not None:
            assert node is not None and node is s.block_map.node(bid) or node.key == s.block_map.node(bid).key, \
                (bid, blk.kind, node)
        shown.append((blk.kind, getattr(node, "key", None)))
    return shown


def template(key: str) -> dict:
    from astro_gpu.client.gallery import GALLERY
    return next(t for t in GALLERY if t["key"] == key)


def made_from(w, env, key: str) -> Path:
    """A new project (as New project makes it, in the test's folder) made from a template of the gallery."""
    from astro_gpu.client.graphview.page import GraphPage
    root = env["tmp"] / f"p_{key}"
    assert w.create_project(root.as_posix(), key, "M 31")
    ready(w)
    page = w.graph
    assert isinstance(page, GraphPage) and page.is_blank()
    page.show_templates()
    settle(w, 3)
    page.template_chosen(template(key))
    settle(w)
    assert w.s.paths, key
    assert not page.is_blank()
    return root


def drawn(g) -> list:
    """What the graph draws, apart from its ids (the project file is today's: the graph is converted again when the
    project opens): each stack by its workflow, its blocks (kind, method, off, pause) and nights; taps; wires."""
    from astro_gpu.graphcore.model import Index
    ix = Index(g)

    def blk(b):
        x = g.blocks[b]
        return (x.kind, x.method, x.off, x.pause, getattr(x, "sid", None))

    def place(b):
        f = next((r for r in g.rows if b in r.blocks), None)
        return (f.wid, f.blocks.index(b)) if f is not None else blk(b)
    rows = sorted((r.wid if r.wid is not None else -1, r.name, [blk(b) for b in ix.effective(r.id)],
                   [(blk(a), blk(c)) for a, c in r.nights]) for r in g.rows)
    taps = sorted((t.wid, t.name, place(t.block)) for t in g.taps)
    wires = sorted((place(w.src[0]), str(w.src[1]), place(w.dst[0]), str(w.dst[1]), w.active) for w in g.wires)
    return [rows, taps, wires]


# ---------------------------------------------------------------------------------------------- the proof
def test_a_new_project_from_each_template_shows_its_blocks_and_the_panel(win, env):
    from astro_gpu.client.gallery import GALLERY
    for t in [t for t in GALLERY if t.get("copy") is None]:
        made_from(win, env, t["key"])
        shown = choose_every_block(win)
        assert any(k == "stacking" or k == "image" for k, _n in shown), (t["key"], shown)
        win.set_mode(1)                                   # Inspect, then back to the graph
        settle(win, 3)
        win.set_mode(0)
        settle(win, 3)
        assert win.close_project(go_home=False, discard=True)
    assert not env["errors"], env["errors"]


@pytest.mark.skipif(not COPY.is_dir(), reason="no study copy on this machine")
def test_a_project_of_todays_format_opens_converted_and_unchanged(win, env):
    from astro_gpu.core import project2
    root = env["tmp"] / "copy"
    root.mkdir()
    shutil.copy(COPY / "project.astroproj", root / "project.astroproj")
    d = json.loads((root / "project.astroproj").read_text(encoding="utf-8"))
    d["scratch_dir"] = (env["tmp"] / "scratch_copy").as_posix()    # its scratch folder in the test's folder
    (root / "project.astroproj").write_text(json.dumps(d, indent=2), encoding="utf-8")
    before = project2.load(root).to_dict()
    win.readonly = True                                    # nothing written in the copy either
    try:
        assert win.open_project(root)
        ready(win, 300)
        s = win.s
        assert s.stored.rows and s.graph_rebuilds == 0
        from astro_gpu.client.graphdefaults import strip_defaults
        after = strip_defaults(s.p.to_dict(), before)          # every default is written: compared without them
        for k in ("sessions", "excluded", "stack_mode", "workflows", "next_workflow_id", "post"):
            assert after[k] == before[k], k
        choose_every_block(win)
        win.set_mode(2)                                    # Post
        settle(win)
        assert win.post is not None
        win.set_mode(0)
        settle(win)
    finally:
        win.readonly = False
    assert not env["errors"], env["errors"]


def test_a_setting_undone_and_redone_and_a_block_to_post(win, env):
    made_from(win, env, "colour_min")
    s, page = win.s, win.graph
    q = s.paths[0]
    stacking = next(b for b, blk in s.stored.blocks.items() if blk.kind == "stacking")
    page.bridge.choose(stacking)
    settle(win, 3)
    assert page.panel.node is s.block_map.node(stacking)
    assert s.sel_step == "empilement" and s.path.pid == q.pid
    s.set_param(q, "stack", "kappa", 2.5)                  # the panel's own call
    settle(win, 3)
    assert s.p.workflow(q.pid).params["stack"]["kappa"] == 2.5
    page.bridge.undo()                                     # Ctrl+Z on the page
    settle(win, 3)
    assert s.p.workflow(q.pid).params["stack"]["kappa"] != 2.5      # back to the stage's default
    page.bridge.redo()
    settle(win, 3)
    assert s.p.workflow(q.pid).params["stack"]["kappa"] == 2.5
    gradient = next(b for b, blk in s.stored.blocks.items() if blk.kind == "gradient")
    n = len(s.paths)
    page.bridge.act("send_to_post", block=gradient)                # the block menu's To Post
    settle(win)
    assert len(s.paths) == n + 1
    sent = s.paths[-1]
    assert s.p.workflow(sent.pid).blocks[-1]["block"] == "output" and any(        # a workflow ending at the Gradient
        n["block"] == "gradient" for n in s.p.workflow(sent.pid).blocks)
    assert any(t.wid == sent.pid for t in s.stored.taps)                            # a tap of the graph
    win.set_mode(2)
    settle(win)
    post = win.post
    assert post is not None
    assert post.coming_layers() is not None              # Post reads the graph's layers to come
    win.set_mode(0)
    settle(win)
    page.bridge.undo()
    settle(win)
    assert len(s.paths) == n
    assert not env["errors"], env["errors"]


def toy_project(folder: Path):
    """A saved project of the toy graph: Plain (no denoise) and Trained (a slow denoise), three raws, on the CPU."""
    p = api.create_project(folder, "toy")
    api.use_rules(p, SPEC)
    raw = folder.parent / "raw" / "lights"
    raw.mkdir(parents=True)
    for i in range(3):
        (raw / f"IMG_{i:03d}.ARW").write_bytes(b"\0" * (100 + 10 * i))
    api.add_session(p, lights=[raw])
    api.add_workflow(p, "Plain", "Classic stack")
    w = api.add_workflow(p, "Trained", "Classic + denoise")
    api.set_param(p, [w.id], "denoise", "steps", 60)
    api.set_param(p, [w.id], "denoise", "delay", 0.05)
    p.devices["main"] = "cpu"
    api.save(p)
    return p


def test_a_run_shows_the_states_and_its_progress_on_the_graph(env):
    """As tests/client/test_run.py runs the toy project (its stages are this module's, read by the detached worker):
    the project model, the new graph page and the run controller watching the page's graph."""
    from PySide6.QtWidgets import QVBoxLayout, QWidget

    from astro_gpu.client.graphview.page import GraphPage
    from astro_gpu.client.logview import LogPanel
    from astro_gpu.client.model import ProjectModel
    from astro_gpu.client.runs import RunOverlay
    from astro_gpu.client.services.run import RUNNING, RunController
    from astro_gpu.client.services.worker import ApiWorker
    p = toy_project(env["tmp"] / "toy")
    worker = ApiWorker()
    s = ProjectModel(p, worker, readonly=True, gpu=False, device="cpu")
    assert worker.wait(120) and s.ready, s.error
    logp = LogPanel(s)
    ctl = RunController(s, None, lambda: logp)
    root = QWidget()
    try:
        v = QVBoxLayout(root)
        page = GraphPage(s)
        v.addWidget(page)
        root.resize(1400, 760)
        root.show()
        ctl.watch_graph(page.graph)
        for _i in range(10):
            QApplication.processEvents()
        bm = s.block_map
        assert any(n.paths and n.key == "debruitage" for n in bm.nodes)
        assert all(bm.state(n) != "done" for n in bm.nodes if n.paths and n.key not in ("session",))
        ov = page.graph.findChildren(RunOverlay)
        assert ov, "the run's overlay is not on the graph"
        ov = ov[0]
        d = ctl.launch()
        d.start()
        track = lambda: ctl.track                                                         # noqa: E731
        jb = lambda: next(j for j in track().jobs if j.stage == "denoise")                # noqa: E731
        assert pump(lambda: track() is not None and track().jobs and jb().state == RUNNING and jb().frac > 0.2, 120)
        items = [it for it in page.canvas.blocks.values() if getattr(it, "node", None) is not None]
        running = []

        def shown():
            running[:] = [it for it in items if ov.fraction_of(it.node) is not None]
            return bool(running)
        assert pump(shown, 15), [(it.node.key, s.block_map.state(it.node)) for it in items]
        assert [it.node.key for it in running] == ["debruitage"]
        f = ov.fraction_of(running[0].node)
        assert 0.0 < f < 1.0
        assert pump(lambda: s.block_map.state(running[0].node) == "running", 15)
        assert not ov.parent().grab().isNull()             # the overlay paints without error
        assert pump(lambda: track().finished and not ctl.running, 120)
        assert pump(lambda: all(s.block_map.state(n) == "done" for n in s.block_map.nodes
                                if n.paths and n.key not in ("session",) and not n.cut), 30),             [(n.key, s.block_map.state(n)) for n in s.block_map.nodes]
        assert ov.fraction_of(running[0].node) is None
    finally:
        root.close()
        ctl.close()
        s.close(save=False)
        worker.stop()
        logp.deleteLater()
        QApplication.processEvents()
    assert not env["errors"], env["errors"]


def test_saved_closed_and_opened_again_gives_the_same_graph(win, env):
    root = made_from(win, env, "filters_min")
    s = win.s
    q = s.paths[0]
    s.set_param(q, "stack", "kappa", 2.5)
    settle(win, 3)
    win.save()
    settle(win)
    assert not s.dirty
    from astro_gpu.graphcore.model import model_json
    graph, definition, stored = drawn(s.stored), s.p.to_dict(), model_json(s.stored)
    assert win.close_project(go_home=False)
    assert win.open_project(root)
    ready(win)
    again = win.s.p.to_dict()
    for k in ("sessions", "excluded", "stack_mode", "workflows", "next_workflow_id", "post"):
        assert again[k] == definition[k], k
    assert drawn(win.s.stored) == graph              # the same stacks, blocks, nights, taps and wires
    assert model_json(win.s.stored) == stored        # the file holds the graph (format 3): the very same one
    import json as _json
    assert _json.loads((root / "project.astroproj").read_text(encoding="utf-8"))["format"] == 3
    assert not env["errors"], env["errors"]
