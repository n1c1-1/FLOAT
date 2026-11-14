"""The settings panel, the project model and the page's selection, offscreen, on the test project with four workflows:
what tests/client/test_graph.py checked on today's page, checked again on the new page (graphview.GraphPage on the
project model, its blocks chosen through the page's bridge, the model's nodes from its BlockMap).

The test project (tests/conftest.py, TEST_PROJECT) is taken down to its Full workflow, computed (conftest.full_only);
the tests add Classic stack (not computed from Gradient on), Classic + denoise and Full twin in memory. The model is
read-only on the test project (its scratch copy reads the store through junctions; a test that saves works on a copy
of the definition in a temporary folder). ASTRO_GPU_SETTINGS points to a temporary file in every test; the user's
settings file is checked untouched."""
from __future__ import annotations

import json
import os
import re
import shutil
import threading
import time
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if Path("C:/Windows/Fonts").is_dir():
    os.environ.setdefault("QT_QPA_FONTDIR", "C:/Windows/Fonts")

from PySide6.QtCore import QEvent, QPoint, QRectF, Qt  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QAbstractButton,
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from astro_gpu import api  # noqa: E402
from astro_gpu.client import app as appmod  # noqa: E402
from astro_gpu.client import i18n, prefs  # noqa: E402
from astro_gpu.client.graphnode import block_state  # noqa: E402
from astro_gpu.client.graphview.page import GraphPage  # noqa: E402
from astro_gpu.client.graphview.style import PALETTE_W  # noqa: E402
from astro_gpu.client.i18n import hm  # noqa: E402
from astro_gpu.client.model import ProjectModel  # noqa: E402
from astro_gpu.client.panel import CompactRow, MethodCard  # noqa: E402
from astro_gpu.client.services.worker import ApiWorker  # noqa: E402
from astro_gpu.client.widgets import QSS  # noqa: E402
from astro_gpu.core import project2  # noqa: E402
from astro_gpu.core.project import Target  # noqa: E402
from conftest import TEST_PROJECT, full_only, needs_test_project, scratch_project  # noqa: E402  (tests/conftest.py)

F_PROJECT = TEST_PROJECT
pytestmark = needs_test_project
FOUR = (("Classic stack", "Classic stack", {"render_simple": {"linked": False}}),
        ("Classic + denoise", "Classic + denoise", None),
        ("Full twin", "Full", None))


def _tree_snapshot(folder: Path):
    return sorted((p.relative_to(folder).as_posix(), p.stat().st_mtime_ns, p.stat().st_size)
                  for p in folder.rglob("*") if p.is_file())


PROJECT_BEFORE = _tree_snapshot(F_PROJECT) if F_PROJECT.is_dir() else None


# ---------------------------------------------------------------------------------------------- fixtures
@pytest.fixture(autouse=True)
def env(tmp_path, monkeypatch, qapp):
    qapp.setStyleSheet(QSS + appmod.EXTRA_QSS)
    monkeypatch.setenv("ASTRO_GPU_SETTINGS", str(tmp_path / "settings.json"))
    prefs.load()
    appmod.apply_language("en")
    shown = []
    for kind in ("information", "warning", "critical"):
        monkeypatch.setattr(QMessageBox, kind, lambda *a, _k=kind, **k: shown.append((_k, a[1:3])))
    yield {"tmp": tmp_path, "shown": shown}       # (tests/conftest.py refuses any write into the user's own folders)
    appmod.apply_language("en")


def _open_four(folder: Path | None = None):
    p = full_only(api.open_project(folder or F_PROJECT))
    for name, template, params in FOUR:
        api.add_workflow(p, name, template, params)
    return p


@pytest.fixture(scope="module")
def world(qapp, tmp_path_factory):
    folder = scratch_project(tmp_path_factory.mktemp("graph") / "heart")
    p = _open_four(folder)
    w = ApiWorker()
    m = ProjectModel(p, w)
    assert not m.ready                                # the GUI thread did not wait for the disk
    assert w.wait(180) and m.ready, m.error
    yield {"p": p, "w": w, "m": m, "folder": folder}
    m.flush()
    w.stop()


def _events():
    QApplication.processEvents()
    QApplication.sendPostedEvents(None, QEvent.DeferredDelete)


def settle(w, page=None, n=12):
    for _i in range(n):
        w.wait(60)
        _events()
        if page is not None and page._pending:
            _events()
        time.sleep(0.01)
    w.wait(60)
    _events()


def _reset(m: ProjectModel):
    while m.can_undo():
        m.undo()
    m._redo.clear()
    m.scope, m.pending_change, m.current, m.sel_step = None, None, 0, "empilement"
    m.p.stack_mode = "together"
    m.flush()


@pytest.fixture()
def model(world):
    m = world["m"]
    _reset(m)
    yield m
    _reset(m)


@pytest.fixture()
def page(world, model, qapp):
    root = QWidget()
    root.setObjectName("root")
    v = QVBoxLayout(root)
    v.setContentsMargins(0, 0, 0, 0)
    pg = GraphPage(model)
    v.addWidget(pg)
    root.resize(1680 + PALETTE_W, 934)
    root.show()
    settle(world["w"], pg)
    pg.root = root
    yield pg
    root.close()
    root.deleteLater()
    qapp.processEvents()


def node(page, key, pid=None):
    m = page.s
    return m.block_map.node_of(m.view(pid) if pid else m.path, key)


def block_of(page, n):
    """The block id of a BlockMap node (Post for "final")."""
    if n.key == "final":
        return next(b for b, x in page.bridge.graph.blocks.items() if x.kind == "post")
    return page.s.block_map.block_of(n)


def select_node(page, n):
    """A block chosen on the page (a click): the model's selection and the panel follow. The page is shared by the
    module's tests: a block still chosen by an earlier test is chosen anew, so that its workflows are the scope."""
    b = block_of(page, n)
    if page.bridge.selected == b:
        page.bridge.choose(None)
    page.bridge.choose(b)
    for _i in range(3):
        _events()


def texts_of(root: QWidget) -> list[str]:
    out = []
    for w in [root] + root.findChildren(QWidget):
        if not w.isVisibleTo(root) and w is not root:
            continue
        if isinstance(w, QLabel):
            out.append(re.sub(r"<[^>]+>", "", w.text()))
        elif isinstance(w, QAbstractButton):
            out.append(w.text())
        elif isinstance(w, QComboBox):
            out += [w.itemText(i) for i in range(w.count())]
        if w.toolTip():
            out.append(w.toolTip())
    return [t.strip() for t in out if t and t.strip()]


NAMES = {"Full", "Classic stack", "Classic + denoise", "Full twin", "Raw stack"}


def untranslated(texts) -> list[str]:
    cat = i18n._cat
    return [t for t in texts if t in cat and cat[t] != t and t not in NAMES]


def cut_texts(root: QWidget) -> list[str]:
    out = []
    for sa in root.findChildren(QScrollArea):
        inner = sa.widget()
        if sa.isVisibleTo(root) and inner is not None and sa.horizontalScrollBarPolicy() == Qt.ScrollBarAlwaysOff:
            if inner.minimumSizeHint().width() > sa.viewport().width() + 1:
                out.append(f"{type(sa.parent()).__name__}: content {inner.minimumSizeHint().width()} px > "
                           f"{sa.viewport().width()} px")
    for w in root.findChildren(QWidget):
        if not w.isVisibleTo(root):
            continue
        if isinstance(w, QLabel) and not w.wordWrap() and w.text() and (w.pixmap() is None or w.pixmap().isNull()):
            if w.width() + 1 < w.sizeHint().width():
                out.append(w.text())
        elif isinstance(w, (QPushButton, QCheckBox)) and w.text():
            if w.width() + 1 < w.sizeHint().width():
                out.append(w.text())
    return out


def button(root: QWidget, start: str) -> QPushButton:
    return next(b for b in root.findChildren(QPushButton) if b.text().startswith(start) and b.isVisibleTo(root))


def _edit_kappa(page, value):
    settle(page.s.worker, page, 6)
    page.panel.expert = True
    page.panel.refresh(page.panel.node)
    _events()
    row = next(r for r in live_rows(page) if r.r["key"] == "stack.kappa")
    page.panel.set_setting(row.r, value)
    for _i in range(3):
        _events()
    return row.r


def live_rows(page) -> list[CompactRow]:
    return [r for r in page.panel.findChildren(CompactRow)
            if not (r.isHidden() and r.testAttribute(Qt.WA_WState_ExplicitShowHide))]


def method_cards(page):
    return [c for c in page.panel.findChildren(MethodCard) if c.isVisibleTo(page.panel)]


def stack_cards(page):
    from astro_gpu.client.panel import StackMethodCard
    return [c for c in page.panel.findChildren(StackMethodCard) if c.isVisibleTo(page.panel)]


def model_methods(block):
    from astro_gpu.client.model import methods_of
    return methods_of(block)


def computed_but_stars(m, p) -> bool:
    """Every stage of a workflow done, Stars/background apart (the test project's stars may be older than today's
    network: its key changed, it says "never run")."""
    return [s for s in m.pending(p) if s != "star_removal"] == []


def page_in(lang, model, w):
    """A page built in a language, as the window builds the project's screen again when the language changes."""
    appmod.apply_language(lang)
    root = QWidget()
    root.setObjectName("root")
    v = QVBoxLayout(root)
    v.setContentsMargins(0, 0, 0, 0)
    pg = GraphPage(model)
    v.addWidget(pg)
    root.resize(1680 + PALETTE_W, 934)
    root.show()
    settle(w, pg)
    pg.root = root
    return pg


# ---------------------------------------------------------------------------------------------- the model and its blocks
def test_the_model_does_not_wait_for_the_disk(qapp):
    """The worker warms the API; keys, states and costs come after readyChanged. (Its time in the GUI thread is not
    checked here: the conversion to the graph reads the frames there, a defect of the swap.)"""
    p = _open_four()
    w = ApiWorker()
    m = ProjectModel(p, w, readonly=True)
    assert not m.ready
    assert [v.name for v in m.paths] == ["Full", "Classic stack", "Classic + denoise", "Full twin"]
    assert m.all_stages() == [] and m.total_seconds() == 0.0 and m.state(None, "stack") == "pending"
    got = []
    m.readyChanged.connect(lambda: got.append(m.ready))
    assert w.wait(180)
    assert m.ready and got and got[-1]
    assert m.all_stages(m.paths[0])[:4] == ["inventory", "calibrate", "register", "stack"]
    m.close(save=False)
    w.stop()


def test_the_blocks_are_shared_as_the_engine_does(world, model, page):
    """Calibration to Color (Gradient included) shared by the 4 workflows, Denoise by 3, Stars by 2; the block that
    holds each workflow's Output is its last one (no Output block)."""
    bm = model.block_map
    by_key = {}
    for n in bm.nodes:
        if n.paths:
            by_key.setdefault(n.key, []).append(n)
    for key in ("alignement", "selection", "empilement", "astrometrie", "fond"):
        assert len(by_key[key]) == 1 and len(by_key[key][0].paths) == 4, key
    assert sorted(len(n.paths) for n in by_key["couleur"]) == [1, 3]       # after the Denoise, and one without it
    den = by_key["debruitage"]
    assert len(den) == 1 and {q.name for q in den[0].paths} == {"Full", "Classic + denoise", "Full twin"}
    stars = by_key["etoiles"]
    assert len(stars) == 1 and {q.name for q in stars[0].paths} == {"Full", "Full twin"}
    assert "post" not in by_key
    for q in model.paths:
        assert bm.node_of(q, "post").key != "post"
    assert model.block_keys(model.paths[0]) == ["calibration", "alignement", "selection", "empilement", "astrometrie",
                                                "fond", "debruitage", "couleur", "etoiles", "post"]
    bands = [b for b, _box in page.bridge.drawing().bands]
    assert bands == ["PREPROCESSING", "LINEAR PROCESSING", "COMPOSITION"][:len(bands)] and bands
    boxes = page.bridge.drawing().boxes
    assert boxes[block_of(page, node(page, "fond", 1))].x < boxes[block_of(page, node(page, "couleur", 1))].x


def test_states_come_from_the_api(world, model, page):
    api_states = api.states(model.p, update=False)
    for (wid, stage), st in api_states.items():
        assert model.state(model.view(wid), stage) == st.state
    bm = model.block_map
    full = model.paths[0]
    assert computed_but_stars(model, full) and full.rendered and not model.paths[1].computed
    assert bm.state(node(page, "empilement", 1)) == "done"
    assert bm.state(node(page, "fond", 1)) == "done" and not model.view(2).computed
    assert bm.state(bm.final) == "pending"                                     # not every layer is rendered
    model._states[(2, "gradient")] = api.StageState("failed", "boom", "k")
    assert block_state(model, model.view(2), "fond") == "failed"
    model._states[(2, "gradient")] = api.StageState("running", "", "k")
    assert block_state(model, model.view(2), "fond") == "running"
    model._sync()
    assert block_state(model, model.view(2), "fond") == "done"
    ses = next(b for b, x in page.bridge.graph.blocks.items() if x.kind == "session")
    assert re.match(r"\d+ lights · \d+ darks? · \d+ flats?", page.canvas.blocks[ses].d["summary"]), \
        page.canvas.blocks[ses].d["summary"]


def test_a_shared_stage_is_one_key(world, model):
    full, classic, cd, neural = model.paths
    assert [q.name for q in model.sharing(full, "stack")] == [q.name for q in model.paths]
    assert model.owner(neural, "stack") is full
    assert {q.name for q in model.sharing(full, "denoise")} == {"Full", "Classic + denoise", "Full twin"}
    assert model.stage_key(full, "stack") == api.stage_key(model.p, full.pid, "stack")
    assert model.upstream(full, "color") >= {"inventory", "calibrate", "register", "stack", "astrometry"}
    assert "star_removal" in model.downstream(full, "render_simple")
    assert [x for _p, x in model.pending_upto(classic, ["render_simple"])] == ["render_simple"]
    assert computed_but_stars(model, full)
    assert model.seconds("stack", full) > 0


# ---------------------------------------------------------------------------------------------- replace / variant
def test_a_change_on_a_shared_done_block_offers_replace_or_variant(world, model, page):
    select_node(page, node(page, "empilement"))
    settle(world["w"], page)
    assert model.scope == [v.pid for v in model.paths]
    r = _edit_kappa(page, 4.0)
    settle(world["w"], page)
    ch = model.pending_change
    assert ch and ch["step"] == "empilement" and ch["value"] == 4.0 and len(ch["pids"]) == 4
    rep = button(page.panel, "Replace")
    var = button(page.panel, "Try as variant")
    paths = [q for q in model.paths if q.pid in ch["pids"]]
    c_all = model.cost_of_change(paths, ch)
    c_one = model.cost_of_change([model.path], ch)
    assert c_all.seconds > c_one.seconds > 0
    assert hm(c_all.seconds) in rep.text() and hm(c_one.seconds) in var.text()
    assert all("kappa" not in (q.params.get("stack") or {}) or q.params["stack"]["kappa"] != 4.0 for q in model.paths)
    button(page.panel, "Cancel").click()
    from astro_gpu.stages.stack import StackParams
    assert model.pending_change is None and r["value"] == StackParams.model_fields["kappa"].default
    _edit_kappa(page, 4.0)
    button(page.panel, "Replace").click()
    settle(world["w"], page)
    assert all(q.params["stack"]["kappa"] == 4.0 for q in model.paths)
    assert model.state(model.paths[0], "stack") == "stale" and model.state(model.paths[0], "render_simple") == "stale"
    assert model.state(model.paths[0], "register") == "done"
    assert model.paths[0].computed is False and model.paths[0].rendered
    assert model.block_map.state(node(page, "empilement")) == "stale"
    page.refresh()
    settle(world["w"], page)
    assert page.canvas.blocks[block_of(page, node(page, "empilement"))].d["state"] == "stale"      # the graph too
    model.undo()
    settle(world["w"], page)
    assert computed_but_stars(model, model.paths[0]) and model.state(model.paths[0], "stack") == "done"


def test_try_as_variant_keeps_the_original_and_shares_the_blocks_before(world, model, page):
    select_node(page, node(page, "empilement", 1))
    _edit_kappa(page, 3.5)
    n_before = len(model.paths)
    button(page.panel, "Try as variant").click()
    settle(world["w"], page)
    assert len(model.paths) == n_before + 1
    q = model.paths[-1]
    full = model.view(1)
    assert q.variant_of == full.pid and q.name.startswith("Full · ") and "3,5" not in q.name
    assert q.params["stack"]["kappa"] == 3.5
    assert model.path is q and model.p.current == q.pid
    bm = model.block_map
    for key in ("alignement", "selection"):                      # shared up to the changed block
        assert bm.node_of(q, key) is bm.node_of(full, key)
    assert bm.node_of(q, "empilement") is not bm.node_of(full, "empilement")
    select_node(page, bm.node_of(q, "empilement"))
    settle(world["w"], page)
    assert any("Variant of" in t for t in texts_of(page.panel))
    assert button(page.panel, "Keep") and button(page.panel, "Discard")
    button(page.panel, "Discard").click()
    settle(world["w"], page)
    assert len(model.paths) == n_before and model.path.pid == full.pid
    select_node(page, node(page, "empilement", 1))
    _edit_kappa(page, 3.5)
    button(page.panel, "Try as variant").click()
    settle(world["w"], page)
    q = model.paths[-1]
    select_node(page, model.block_map.node_of(q, "empilement"))
    settle(world["w"], page)
    told = []
    page.toasted.connect(told.append)
    button(page.panel, "Keep").click()
    settle(world["w"], page)
    assert len(model.paths) == n_before and model.path.name == "Full" and model.path.params["stack"]["kappa"] == 3.5
    assert model.path.variant_of is None


def test_a_method_card_asks_the_same_question(world, model, page):
    """Gradient's method card follows the rule too: on a computed Gradient it offers Replace / variant."""
    select_node(page, node(page, "fond", 1))
    settle(world["w"], page)
    cards = method_cards(page)
    assert [c.ok for c in cards] == [True, True, True]
    other = next(c for c in cards if "Automatic" in c.findChildren(QLabel)[0].text())
    other.clicked.emit()
    settle(world["w"], page)
    assert model.pending_change and model.pending_change["key"] == "fond"
    assert "Method: Automatic" in model.pending_change["what"]
    button(page.panel, "Try as variant").click()
    settle(world["w"], page)
    q = model.paths[-1]
    assert q.opts["fond"] == "auto" and q.opts["etoiles"] == "retirer" and q.variant_of == 1
    assert q.name == "Full · automatic"


def test_a_method_card_says_the_duration_on_each_device_and_which_one_is_used(world, model, page):
    from astro_gpu.client.panel import durations_text
    w = world["w"]
    select_node(page, node(page, "debruitage", 1))
    settle(w, page)
    try:
        for gpu in (True, False):
            model.set_machine(gpu)
            settle(w, page)
            here = "gpu" if gpu else "cpu"
            cards = method_cards(page)
            assert len(cards) >= 2
            seen = 0
            for c in cards:
                lines = [lb.text() for lb in c.findChildren(QLabel) if "GPU" in lb.text() or "CPU" in lb.text()]
                if not lines:
                    continue
                seen += 1
                plain = re.sub(r"<[^>]+>", "", lines[0])
                assert "CPU" in plain and re.search(r"(GPU|CPU) (≈ )?[\d≤]", plain), plain
                if "GPU" in plain and "CPU" in plain:
                    bold = re.findall(r"<b[^>]*>(.*?)</b>", lines[0])
                    assert len(bold) == 1 and bold[0].startswith(here.upper()), (lines[0], here)
            assert seen >= 2
            ok = [m for m in model_methods("debruitage") if m["runs"] != "cpu" and not m["new"]]
            t = model.method_times(model.path, "debruitage", ok[0]["key"])
            assert set(t) == {"gpu", "cpu"} and t["gpu"][0] > 0 and t["cpu"][0] > t["gpu"][0] * 3, t
            assert durations_text(t, here).count("<b") == 1
        model.set_machine(False)
        for lang in ("en", "fr"):
            pg = page_in(lang, model, w)
            select_node(pg, node(pg, "debruitage", 1))
            settle(w, pg)
            assert not untranslated(texts_of(pg.panel))
            pg.root.close()
    finally:
        appmod.apply_language("en")
        model.set_machine(True)
        page.refresh()
        settle(w, page)
    assert durations_text(None, "gpu") == "…"
    assert re.sub(r"<[^>]+>", "", durations_text({"cpu": (90.0, True)}, "gpu")) == "CPU " + hm(90.0)
    assert "CPU ≈ " in durations_text({"cpu": (90.0, False)}, "cpu")


def test_the_durations_of_the_cards_are_read_by_the_worker_and_the_panel_follows(qapp, monkeypatch):
    seen = []
    orig = api.estimate

    def watch(*a, **k):
        seen.append(threading.current_thread() is threading.main_thread())
        return orig(*a, **k)
    monkeypatch.setattr(api, "estimate", watch)
    p = _open_four()
    w = ApiWorker()
    m = ProjectModel(p, w, readonly=True)
    assert w.wait(180) and m.ready, m.error
    try:
        v = m.path
        key = next(x["key"] for x in model_methods("debruitage") if not x["new"] and x["runs"] != "cpu")
        m._est.clear()
        fired = []
        m.costChanged.connect(lambda: fired.append(1))
        seen.clear()
        assert m.method_times(v, "debruitage", key) is None
        assert m.method_times(v, "debruitage", key) is None
        assert not any(seen)
        assert w.wait(180)
        assert fired and m.method_times(v, "debruitage", key) is not None
        assert seen and not any(seen), "the estimates were made in the GUI thread"
    finally:
        m.close(save=False)
        w.stop()


def test_undo_and_redo_restore_the_workflows(world, model, page):
    n0 = len(model.paths)
    p = model.add_path("Extra", {"debruitage": False, "etoiles": "garder", "fond": "simple"})
    assert len(model.paths) == n0 + 1 and model.can_undo()
    model.rename(p.pid, "Renamed")
    model.set_param(model.view(1), "stack", "kappa", 4.0)
    settle(world["w"], page)
    assert model.view(1).params["stack"]["kappa"] == 4.0
    assert model.undo() and model.view(1).params.get("stack", {}).get("kappa") != 4.0
    assert model.undo() and model.view(p.pid).name == "Extra"
    assert model.undo() and len(model.paths) == n0
    assert model.redo() and len(model.paths) == n0 + 1
    assert model.redo() and model.view(p.pid).name == "Renamed"
    assert model.redo() and model.view(1).params["stack"]["kappa"] == 4.0
    assert not model.redo()
    assert model.undo() and model.can_redo()
    with model.batch():
        model.rename(p.pid, "A")
        model.rename(p.pid, "B")
    assert model.view(p.pid).name == "B" and not model.can_redo()
    assert model.undo() and model.view(p.pid).name == "Renamed"
    depth = len(model._undo)
    model.rename(p.pid, "Renamed")
    assert len(model._undo) == depth
    # the computed results are never removed: a workflow deleted on the page comes back with its states
    row = page.bridge.row_of(block_of(page, node(page, "debruitage", 3)))
    assert page.bridge.act("delete_row", row=row.id).ok and model.view(3) is None
    page.bridge.undo()
    settle(world["w"], page)
    assert model.view(3) is not None and model.state(model.view(3), "stack") == "done"


def test_a_port_pulled_to_post_makes_a_raw_stack_workflow_without_star_sliders(world, model, page):
    """Stacking's port pulled onto Post: a workflow up to Stacking, its blocks shared (computed once), its Output its
    own; no star slider without a Stars/background block."""
    from astro_gpu.client.graphview.gestures import Gestures
    n0 = len(model.paths)
    st = block_of(page, node(page, "empilement", 1))
    post = next(b for b, x in page.bridge.graph.blocks.items() if x.kind == "post")
    Gestures(page).pull_wire(st, "image", post, "layers")
    settle(world["w"], page)
    assert len(model.paths) == n0 + 1
    q = model.paths[-1]
    assert q.name.startswith("Full · ") and model.block_keys(q)[-2:] == ["empilement", "post"]
    assert model.all_stages(q)[-1] == "render_simple"
    assert model.block_map.node_of(q, "empilement") is model.block_map.node_of(model.view(1), "empilement")
    ok, why, _need = model.slider_state(q, "star_strength")
    assert ok is False and "Stars/background block" in why
    assert model.slider_state(model.view(1), "star_strength", "combined")[0] is True


def test_duplicate_and_delete_a_workflow(world, model, page):
    """Duplicate keeps the options and the settings, at once (no window); Delete workflow at once, said."""
    from astro_gpu.client.graphview.gestures import Gestures
    n0 = len(model.paths)
    src = model.view(1)
    g = Gestures(page)
    page.canvas.setFocus()
    g.click(block_of(page, node(page, "etoiles", 1)))
    g.key("Ctrl+D", target=page.canvas)
    settle(world["w"], page)
    cp = model.paths[-1]
    assert len(model.paths) == n0 + 1 and cp.name != src.name
    # its options and settings kept; its Astrometry taken from the original it is aligned on (one plate solution)
    assert {k: v for k, v in cp.opts.items() if k != "cut"} == {k: v for k, v in src.opts.items() if k != "cut"}
    assert model.align_reference(cp) == src.pid
    r = page.bridge.row_of(block_of(page, model.block_map.node_of(cp, "post")))
    res = page.bridge.act("delete_row", row=r.id)
    settle(world["w"], page)
    assert res.ok and model.view(cp.pid) is None and len(model.paths) == n0


def test_clicking_selects_and_sets_the_scope(world, model, page):
    n = node(page, "debruitage")
    select_node(page, n)
    settle(world["w"], page)
    assert model.sel_step == "debruitage" and model.scope == [v.pid for v in n.paths] and len(n.paths) == 3
    assert model.path in n.paths
    n = next(x for x in model.block_map.nodes if x.key == "couleur" and len(x.paths) == 1)
    select_node(page, n)
    assert model.path is n.paths[0] and model.scope == [n.paths[0].pid]
    select_node(page, model.block_map.final)
    assert model.sel_step == "final" and model.scope is None
    settle(world["w"], page)
    assert "Layers" in texts_of(page.panel)
    opened = []
    page.openPost.connect(lambda: opened.append(1))
    page.canvas.inspect.emit(block_of(page, model.block_map.final))
    assert opened


# ---------------------------------------------------------------------------------------------- the panel
def test_the_panel_lists_the_settings_of_the_engine(world, model, page):
    select_node(page, node(page, "empilement"))
    settle(world["w"], page)
    rows = {r.r["key"]: r for r in live_rows(page)}
    assert {"stack.scale", "stack.pixfrac"} <= set(rows)
    sc = rows["stack.scale"].r
    assert sc["suggested"] in (1.0, 1.5, 2.0) and re.search(r"stars of [\d.]+ px", sc["reason"]), sc["reason"]
    assert sc["min"] == 1.0 and sc["max"] == 3.0
    assert sc["unit"] == "×" and not sc["changed"] and sc["type"] == "float"
    page.panel.expert = True
    page.panel.refresh(page.panel.node)
    keys = {r.r["key"] for r in live_rows(page)}
    assert {"stack.kappa", "stack.half_block"} <= keys
    assert "stack.exclude" not in keys and "stack.min_transparency" not in keys
    rows = {r.r["key"]: r for r in live_rows(page)}
    rows["stack.scale"].focused.emit(rows["stack.scale"].r)
    dt = texts_of(page.panel.details)
    assert any(t.startswith("Suggested: ") and sc["reason"][:12] in t for t in dt), dt
    page.panel.expert = False


def test_the_colour_block_has_one_factor_per_colour_and_no_choice_of_correction(world, model, page):
    select_node(page, node(page, "couleur"))
    settle(world["w"], page)
    rows = {r.r["key"]: r for r in live_rows(page)}
    assert "color.white_temp" in rows and "color.correction" not in rows


def test_a_settings_row_changes_marks_and_goes_back_to_the_suggestion(world, model, page):
    model.set_param(model.view(1), "gradient", "degree", 3)
    page.panel.expert = True
    try:
        select_node(page, node(page, "fond", 1))
        settle(world["w"], page)
        rows = {r.r["key"]: r for r in live_rows(page)}
        it = rows["gradient.degree"]
        assert it.r["changed"] and it.r["value"] == 3 and it.r["suggested"] == 2
        assert model.state(model.view(1), "gradient") == "stale"
        back = [b for b in it.findChildren(QPushButton) if b.property("iconRole") == "reset"][0]
        assert not back.isHidden() and "2" in back.toolTip()
        assert "e6a23c" in it.findChildren(QLabel)[0].styleSheet().lower()
        back.click()
        settle(world["w"], page)
        if model.pending_change:
            button(page.panel, "Replace").click()
            settle(world["w"], page)
        rows = {r.r["key"]: r for r in live_rows(page)}
        assert not rows["gradient.degree"].r["changed"] and rows["gradient.degree"].r["value"] == 2
        assert model.state(model.view(1), "gradient") == "done"
    finally:
        page.panel.expert = False


def test_the_deconvolution_block_has_its_method_and_its_settings(world, model, page):
    full = model.paths[0]
    with_block, notes = ProjectModel.with_tool(full.opts, "deconv")
    assert with_block["deconv"] and notes == []
    page.bridge.choose(block_of(page, node(page, "debruitage", 1)))
    page.palette.finish()
    page.palette.double_click("deconv")                        # the palette's Deconvolution, in Full
    settle(world["w"], page)
    try:
        select_node(page, node(page, "deconv", 1))
        settle(world["w"], page)
        assert len(method_cards(page)) == len([x for x in model_methods("deconv") if not x["new"]]) >= 1
        assert "Richardson-Lucy" in " ".join(texts_of(page.panel))
        assert "deconv.iterations" in {r.r["key"] for r in live_rows(page)}
    finally:
        _reset(model)
        settle(world["w"], page)
    assert not model.paths[0].opts.get("deconv")


def test_the_stacking_method_is_chosen_on_cards_and_shows_only_its_own_settings(world, model, page):
    select_node(page, node(page, "empilement"))
    settle(world["w"], page)
    page.panel.expert = True
    page.panel.refresh(page.panel.node)
    settle(world["w"], page)
    cards = stack_cards(page)
    assert [c.key for c in cards] == ["drizzle", "classic"] and [c.on for c in cards] == [True, False]
    keys = {r.r["key"] for r in live_rows(page)}
    assert {"stack.scale", "stack.pixfrac", "stack.kappa"} <= keys
    assert not any(k.startswith("stack.classic_") for k in keys) and "stack.method" not in keys
    assert "stack.rejection" not in keys
    done = model.state(model.view(1), "stack") == "done"
    cards[1].clicked.emit()
    settle(world["w"], page)
    ch = model.pending_change
    if done:
        assert ch and (ch["stage"], ch["name"], ch["value"]) == ("stack", "method", "classic")
        button(page.panel, "Replace").click()
        settle(world["w"], page)
    assert model.view(1).params["stack"]["method"] == "classic"
    assert [c.on for c in stack_cards(page)] == [False, True]
    keys = {r.r["key"] for r in live_rows(page)}
    assert {"stack.classic_debayer", "stack.classic_interpolation", "stack.classic_clamping",
            "stack.classic_rejection"} <= keys
    assert not keys & {"stack.scale", "stack.pixfrac", "stack.kappa", "stack.scale_rule"}
    select_node(page, node(page, "alignement"))
    settle(world["w"], page)
    assert "register.channel_align" not in {r.r["key"] for r in live_rows(page)}
    page.panel.expert = False
    model.undo()
    settle(world["w"], page)


def test_the_engines_method_is_an_automatic_outline_with_its_reason(world, model, page, monkeypatch):
    import copy

    from astro_gpu.client.widgets import ACCENT
    real = model.param_info
    reason = "{n} frames (drizzle needs 15): Classic"

    def fake(p, stage, name):
        info = real(p, stage, name)
        if (stage, name) == ("stack", "method") and info is not None:
            info = copy.copy(info)
            info.value, info.origin, info.reason, info.reason_args = "classic", "auto", reason, {"n": 12}
        return info
    monkeypatch.setattr(model, "param_info", fake)
    select_node(page, node(page, "empilement"))
    settle(world["w"], page)
    page.panel.refresh(page.panel.node)
    settle(world["w"], page)
    cards = {c.key: c for c in stack_cards(page)}
    assert not cards["drizzle"].on and not cards["drizzle"].recommended and cards["drizzle"].tag is None
    rec = cards["classic"]
    assert rec.recommended and rec.on and rec.tag.text() == "Automatic"
    assert ACCENT in rec.tag.styleSheet() and "border-radius" in rec.tag.styleSheet()
    assert rec.info.property("iconRole") == "info" and rec.info.toolTip() == "12 frames (drizzle needs 15): Classic"
    try:
        pg = page_in("fr", model, world["w"])
        select_node(pg, node(pg, "empilement"))
        settle(world["w"], pg)
        rec = {c.key: c for c in stack_cards(pg)}["classic"]
        assert rec.tag.text() == "Automatique" and rec.info.toolTip() == "12 images (le drizzle en demande 15) : Classique"
        pg.root.close()
    finally:
        appmod.apply_language("en")


def test_the_selection_rule_is_said_and_the_thresholds_act_only_with_the_thresholds_rule(world, model, page):
    model.scope = None
    select_node(page, node(page, "selection"))
    settle(world["w"], page)
    stored = model.path.params.setdefault("stack", {})
    before = stored.get("selection", "<unset>")

    def thresholds_editors():
        return [e for e in page.panel.findChildren(QDoubleSpinBox)
                if e.isVisibleTo(page.panel) and not isinstance(e.parent(), CompactRow)]

    def reason():
        return [b.text() for b in page.panel.findChildren(QLabel) if b.objectName() == "thresholds_reason"
                and b.isVisibleTo(page.panel)]
    try:
        for rule, said, live in (
                ("min_weight", "Every aligned frame is kept, except those weighing less than 5 % of the best one.", False),
                ("sota", "Frames far worse than the night's median (over 2.5 σ) in star size, roundness or "
                         "transparency are left out, then those weighing less than 5 % of the best one.", False),
                ("thresholds", "Frames under the minimum transparency or over the maximum star size below are left out.",
                 True)):
            model.set_param(model.path, "stack", "selection", rule)          # the rule chosen in the panel
            model._cache.clear()
            page.panel.refresh(page.panel.node)
            settle(world["w"], page)
            assert model.selection_rule() == rule and model.thresholds_active() is live
            # (k × σ: the transparency is said when the stack's settings use it)
            shown = [t.replace("\xa0", " ") for t in texts_of(page.panel)]           # (a no-break space before σ)
            assert any(t == said or (rule == "sota" and t.startswith(said[:60])) for t in shown), rule
            eds = thresholds_editors()
            assert len(eds) == 2 and all(e.isEnabled() is live for e in eds), rule
            assert reason() == ([] if live else ["These thresholds act only when the frame selection is “Thresholds”."])
            assert any(r.r["key"] == "stack.selection" for r in live_rows(page))
        model.set_param(model.path, "stack", "selection", "min_weight")
        model._cache.clear()
        try:
            pg = page_in("fr", model, world["w"])
            select_node(pg, node(pg, "selection"))
            settle(world["w"], pg)
            assert "Toutes les images alignées sont gardées, sauf celles qui pèsent moins de 5 % de la meilleure." \
                in texts_of(pg.panel)
            assert [b.text() for b in pg.panel.findChildren(QLabel) if b.objectName() == "thresholds_reason"
                    and b.isVisibleTo(pg.panel)] == ["Ces seuils n'agissent que si la sélection des images est sur « Seuils »."]
            pg.root.close()
        finally:
            appmod.apply_language("en")
    finally:
        if before == "<unset>":
            stored.pop("selection", None)
        else:
            stored["selection"] = before
        model._cache.clear()


def test_selection_thresholds_come_from_the_stack_settings(world, model, page):
    model.scope = None
    stored = model.path.params.setdefault("stack", {})
    before = stored.get("selection", "<unset>")
    stored["selection"] = "thresholds"
    model._cache.clear()
    try:
        select_node(page, node(page, "selection"))
        settle(world["w"], page)
        assert model.thresholds_active()
        t, f = model.thresholds_now()
        assert t == 60 and f > 0 and model.thresholds_unit() == "px"
        st, sf = model.suggested_thresholds()
        assert st == 60 and abs(sf - f) < 1e-6
        n = model.selection_total()
        assert n > 0 and model.kept_frames() == n and any(f"{n} of {n} frames kept" in t for t in texts_of(page.panel))
        page.panel.propose_selection(70, f)
        settle(world["w"], page)
        assert model.pending_change and model.pending_change["kind"] == "params"
        assert model.pending_change["items"][0] == ("stack", "min_transparency", 0.7)
        button(page.panel, "Cancel").click()
        model.request_selection(70, f)
        settle(world["w"], page)
        assert model.kept_frames() <= n and model.selection_total() == n
    finally:
        if before == "<unset>":
            stored.pop("selection", None)
        else:
            stored["selection"] = before
        model._cache.clear()


def test_spec_mode_shows_what_the_engine_does_not_have_disabled(world, model, page):
    select_node(page, node(page, "debruitage"))
    settle(world["w"], page)
    assert not any("◇" in t for t in texts_of(page.panel))
    assert len(method_cards(page)) == 2
    prefs.set("spec_mode", True)
    try:
        page.refresh()
        settle(world["w"], page)
        assert [c.ok for c in method_cards(page)] == [True, True]
        page.panel.expert = False
        select_node(page, node(page, "couleur"))
        settle(world["w"], page)
        new_rows = [r for r in live_rows(page) if r.r["kind"] == "new"]
        assert new_rows and not [b for r in new_rows for b in r.findChildren(QPushButton)
                                 if b.property("iconRole") == "reset" and b.isVisibleTo(r)]
        for r in new_rows:
            editors = r.findChildren(QComboBox) + r.findChildren(QCheckBox) + r.findChildren(QDoubleSpinBox) + \
                r.findChildren(QSpinBox)
            assert editors and not any(e.isEnabled() for e in editors), r.r["key"]
        assert "deconv" in page.palette.cards and "◇" not in page.palette.cards["deconv"].title.text()
        keys_before = {s: model.stage_key(model.paths[0], s) for s in model.all_stages(model.paths[0])}
        page.panel.set_setting(next(r for r in new_rows).r, 1)
        assert keys_before == {s: model.stage_key(model.paths[0], s) for s in model.all_stages(model.paths[0])}
        assert model.pending_change is None and not model.can_undo()
    finally:
        prefs.set("spec_mode", False)


def test_the_french_switch_translates_every_panel_and_the_page(world, model):
    prefs.set("spec_mode", True)
    page = None
    try:
        page = page_in("fr", model, world["w"])               # the window builds the project's screen again
        i18n.missing()
        for b in list(page.bridge.graph.blocks):
            page.bridge.choose(b)
            settle(world["w"], page, 4)
            assert not untranslated(texts_of(page.root)), (b, untranslated(texts_of(page.root)))
        select_node(page, node(page, "empilement"))
        _edit_kappa(page, 4.0)
        settle(world["w"], page)
        t = texts_of(page.root)
        assert not untranslated(t), untranslated(t)
        words = [x for x in i18n.missing() if re.sub(r"\{\w+\}|[\W\d]", "", x)]      # (a bare "{m0} · {m1}" says nothing)
        assert not words, sorted(words)
        assert "Remplacer" in " ".join(t) and "Essayer en variante" in " ".join(t)
        assert any(x.startswith("Suggéré : ") for x in t)
        button(page.panel, "Annuler").click()
        assert "Blocs" in t
    finally:
        appmod.apply_language("en")
        prefs.set("spec_mode", False)
        if page is not None:
            page.root.close()


def test_decimal_commas_in_french_reasons(world, model):
    try:
        page = page_in("fr", model, world["w"])
        select_node(page, node(page, "empilement"))
        settle(world["w"], page)
        r = next(x for x in live_rows(page) if x.r["key"] == "stack.scale").r
        page.root.close()
    finally:
        appmod.apply_language("en")
    assert re.search(r"\d,\d+ px", r["reason"]) and not re.search(r"\d\.\d", r["reason"]), r["reason"]


# ---------------------------------------------------------------------------------------------- GUI thread, saving
def test_nothing_slow_runs_in_the_gui_thread(qapp, monkeypatch):
    """Disk scans, describe, outputs, previews and warm run in the worker thread; the keys, states and costs the GUI
    asks for afterwards read caches, every block chosen on the new page."""
    from astro_gpu.core import store
    calls = []

    def watch(mod, name):
        orig = getattr(mod, name)

        def f(*a, **k):
            calls.append((name, threading.current_thread() is threading.main_thread()))
            return orig(*a, **k)
        monkeypatch.setattr(mod, name, f)
    for name in ("warm", "describe", "stage_outputs", "selection_preview", "frame_table", "scan_files"):
        watch(api, name)
    for name in ("list_frames", "file_id"):
        watch(store, name)
    p = _open_four()
    w = ApiWorker()
    m = ProjectModel(p, w, readonly=True)
    root = QWidget()
    root.setObjectName("root")
    v = QVBoxLayout(root)
    pg = GraphPage(m)
    v.addWidget(pg)
    root.resize(1680, 934)
    root.show()
    assert w.wait(180)
    settle(w, pg)
    assert api._MACHINE and any(c == "warm" and not main for c, main in calls)
    calls.clear()
    for b in list(pg.bridge.graph.blocks):
        pg.bridge.choose(b)
        QApplication.processEvents()
    select_node(pg, m.block_map.node_of(m.path, "empilement"))
    settle(w, pg)
    _edit_kappa(pg, 4.0)
    button(pg.panel, "Try as variant").click()
    m.add_path("X", {"debruitage": True, "etoiles": "retirer", "fond": "simple"})
    m.undo()
    m.cost()
    m.all_pending()
    settle(w, pg)
    main_calls = [c for c, main in calls if main]
    assert not main_calls, main_calls
    assert any(c in ("describe",) for c, main in calls if not main)
    m.close(save=False)
    w.stop()
    root.close()


def test_save_writes_only_the_definition_and_only_when_asked(qapp, tmp_path):
    root = tmp_path / "copy"
    root.mkdir()
    shutil.copy(F_PROJECT / "project.astroproj", root / "project.astroproj")
    p = api.open_project(root)
    w = ApiWorker()
    m = ProjectModel(p, w)
    assert w.wait(180)
    before = (root / "project.astroproj").read_bytes()
    seen = []
    m.dirtyChanged.connect(seen.append)
    assert not m.dirty
    m.add_path("Saved", {"debruitage": True, "etoiles": "garder", "fond": "simple"})
    time.sleep(0.45)
    qapp.processEvents()
    assert (root / "project.astroproj").read_bytes() == before                  # issue #1: nothing until Save
    assert m.dirty and seen == [True]
    m.rename(m.paths[-1].pid, "Last one")
    m.flush()
    assert [x.name for x in api.open_project(root).workflows][-1] == "Last one"
    assert not m.dirty and seen == [True, False]
    m.rename(m.paths[-1].pid, "Dropped")
    m.close(save=False)
    assert [x.name for x in api.open_project(root).workflows][-1] == "Last one"
    m = ProjectModel(api.open_project(root), w)
    assert w.wait(180)
    m.select(index=0)
    m.close()
    names = {f.name for f in root.iterdir()}
    # (the definition of today's format is kept once beside it before the first save of the new one)
    assert names <= {"project.astroproj", "project.astroproj.v2.bak", "state.json", "logs", "cache", "store", "inputs",
                     "scratch"}, names
    w.stop()


def test_shots_in_english_and_french_cut_no_text(world, model, tmp_path):
    """The page with the panel of Stacking, Denoise, Stars, Selection, Gradient and the variant's choice, in English
    and in French: no text cut, no English left in French."""
    out = Path(os.environ.get("ASTRO_GPU_SHOTS") or tmp_path / "shots")
    out.mkdir(parents=True, exist_ok=True)
    w = world["w"]
    cut = {}

    def shot(name):
        settle(w, page, 8)
        page.root.grab().save(str(out / f"{name}.png"))
        assert (out / f"{name}.png").stat().st_size > 5000
        c = cut_texts(page.root)
        if c:
            cut[name] = c
        if i18n.language() == "fr":
            assert not untranslated(texts_of(page.root)), (name, untranslated(texts_of(page.root)))
    page = None
    try:
        for lang in ("en", "fr"):
            _reset(model)
            page = page_in(lang, model, w)                    # the window builds the project's screen again
            shot(f"graph_{lang}")
            for key, name in (("empilement", "panel_shared"), ("debruitage", "panel_denoise"), ("etoiles", "panel_stars"),
                              ("selection", "panel_selection"), ("fond", "panel_gradient")):
                select_node(page, node(page, key))
                shot(f"{name}_{lang}")
            select_node(page, node(page, "empilement"))
            settle(w, page)
            _edit_kappa(page, 4.0)
            shot(f"variant_{lang}")
            button(page.panel, i18n._("Try as variant")).click()
            assert model.paths[-1].name == f"Full · {i18n._('Rejection threshold').lower()} 4"
            _reset(model)
            page.root.close()
    finally:
        appmod.apply_language("en")
    assert not cut, cut


def test_a_project_without_workflows_has_a_blank_graph_and_the_project_panel(qapp, tmp_path):
    p = project2.create(tmp_path / "P", "P", Target(name="M 1"))
    w = ApiWorker()
    m = ProjectModel(p, w, readonly=True)
    assert w.wait(120)
    root = QWidget()
    root.setObjectName("root")
    v = QVBoxLayout(root)
    pg = GraphPage(m)
    v.addWidget(pg)
    root.resize(1680, 934)
    root.show()
    settle(w, pg)
    assert pg.is_blank() and not [b for b, x in pg.bridge.graph.blocks.items() if x.kind != "post"]
    assert pg.canvas.blank.isVisible()
    assert [lb.text() for lb in pg.panel.findChildren(QLabel) if lb.text() == "Target"]   # the Project block's panel
    assert pg.palette.head.text() == "Blocks"
    assert m.total_seconds() == 0 and m.all_pending() == []
    m.close(save=False)
    w.stop()
    root.close()


def test_a_change_of_an_automatic_setting_is_seen_by_the_what_if(world, model, page):
    import copy
    full = model.view(1)
    stale = api.with_change(model.p, 1, {"kind": "param", "stage": "stack", "name": "scale", "value": 2.0})
    mine = model._changed(full, {"kind": "param", "stage": "stack", "name": "scale", "value": 2.0})
    assert api.stage_key(model.p, mine, "stack") != model.stage_key(full, "stack")
    assert api.stage_key(model.p, copy.deepcopy(stale), "stack") == api.stage_key(model.p, mine, "stack")
    for name, value in (("scale", 2.0), ("half_block", 20), ("min_transparency", 0.7)):
        ch = {"kind": "param", "stage": "stack", "name": name, "value": value}
        assert model.throws_away(model.paths, ch), name
        assert model.cost_with([full], ch) > 0, name
    ch = {"kind": "param", "stage": "gradient", "name": "degree", "value": 3}
    assert model.throws_away([full], ch) and model.cost_with([full], ch) > 0
    toasts = []
    page.toasted.connect(toasts.append)
    select_node(page, node(page, "empilement"))
    settle(world["w"], page)
    depth = len(model._undo)
    page.panel.propose("stack", "scale", 9.0, 1.5, "Drizzle scale: 1.5 → 9", "scale 9")
    assert toasts and "stack.scale" in toasts[0] and model.pending_change is None and len(model._undo) == depth
    assert model.view(1).params.get("stack", {}).get("scale") != 9.0


def test_no_setting_of_the_engine_is_out_of_reach(world, model, page):
    """Every setting a stage describes (but the ones the project or the options own) has a row in the panel of its
    block, its block chosen on the new page; no row is listed twice in a panel."""
    from astro_gpu.client.settings import CATALOG, HIDDEN, catalog_key, setting_method, stack_method

    def names(ck):
        return {it["name"] for _t, items in CATALOG.get(ck, []) for it in items}

    def another_methods(v, stage, name):
        block = {"gradient": "fond", "denoise": "debruitage"}.get(stage)
        return block is not None and name not in names(catalog_key(v, block)) and \
            any(name in names(ck) for ck in CATALOG if ck.startswith(block + ":"))
    want = set()
    for v in model.paths:
        for stage in model.all_stages(v):
            for info in api.describe(model.p, v.pid, stage):
                if info.level == "internal" or (stage, info.name) in HIDDEN:
                    continue
                m = setting_method({"stage": stage, "name": info.name})
                if m is not None and m != stack_method(model, v):
                    continue
                if another_methods(v, stage, info.name):
                    continue
                if (stage, info.name) == ("render_simple", "linked"):
                    continue
                want.add(f"{stage}.{info.name}")
    got = set()
    page.panel.expert = True
    try:
        for b in list(page.bridge.graph.blocks):
            page.bridge.choose(b)
            settle(world["w"], page, 5)
            keys = [r.r["key"] for r in live_rows(page)]
            assert len(keys) == len(set(keys)), (b, keys)
            got |= set(keys)
    finally:
        page.panel.expert = False
    assert not (want - got), sorted(want - got)


# ---------------------------------------------------------------------------------------------- pauses for review
_API_STATES = api.states


def pin(model, monkeypatch, *done):
    """The engine's states with these blocks computed and every other stage not, whatever is on the disk."""
    from astro_gpu.client.graphnode import ENGINE_BLOCK

    def states(p, update=True):
        out = dict(_API_STATES(p, update=False))
        block_of_ = {(w.id, st): ENGINE_BLOCK[b.block] for w in p.workflows for b in api.blocks(p, w.id)
                     for st in b.stages}
        for (wid, stage), st in out.items():
            if block_of_.get((wid, stage)) in done:
                out[(wid, stage)] = api.StageState("done", "", st.key)
            else:
                out[(wid, stage)] = api.StageState("pending", "never computed", st.key)
        return out
    monkeypatch.setattr(api, "states", states)
    model._sync(prefetch=False)
    model.pathsChanged.emit()


def test_every_block_has_review_pause_and_the_dot_shows(world, model, page, monkeypatch):
    """Every block but Post has Review pause in its menu (Post: Open Post only); chosen on a block shared by every
    workflow, every workflow gets the pause; the dot is drawn; chosen again it goes."""
    from astro_gpu.client.graphview.menu import menu
    pin(model, monkeypatch)
    settle(world["w"], page)
    doc = page.bridge.doc
    post = next(b for b, x in page.bridge.graph.blocks.items() if x.kind == "post")
    assert [e[0] for e in menu(doc, post) if e] == ["Open Post"]
    for b, x in page.bridge.graph.blocks.items():
        if x.kind in ("post",):
            continue
        texts = [e[0] for e in menu(doc, b) if e]
        assert "Review pause" in texts, x.kind
    st = block_of(page, node(page, "empilement"))
    assert len(node(page, "empilement").paths) == 4
    page.canvas.run_menu(st, "set_pause", {"block": st})
    settle(world["w"], page)
    assert all("stacking" in w.breakpoints for w in model.p.workflows)
    assert page.canvas.blocks[st].d["review"] == "pending"
    assert "Remove pause" in [e[0] for e in menu(doc, st) if e]
    page.canvas.run_menu(st, "set_pause", {"block": st})
    settle(world["w"], page)
    assert not any("stacking" in w.breakpoints for w in model.p.workflows)
    assert page.canvas.blocks[st].d["review"] is None


def test_the_dot_says_pending_paused_or_reviewed_and_the_graph_follows(world, model, page, monkeypatch):
    from astro_gpu.client.graphview.menu import menu
    p = model.view(1)
    pin(model, monkeypatch)
    model.set_breakpoint([p], "empilement")
    settle(world["w"], page)
    st = block_of(page, node(page, "empilement", 1))
    assert model.review_state(p, "empilement") == "pending" and page.canvas.blocks[st].d["review"] == "pending"
    pin(model, monkeypatch, "calibration", "alignement", "selection", "empilement")
    settle(world["w"], page)
    assert model.review_state(p, "empilement") == "paused"
    page.refresh()
    settle(world["w"], page)
    assert page.canvas.blocks[st].d["review"] == "paused"
    item = next(e for e in menu(page.bridge.doc, st) if e and e[0] == "Mark reviewed")
    assert item[3]                                                          # offered once it is paused
    page.canvas.run_menu(st, "mark_reviewed", {})
    settle(world["w"], page)
    assert model.review_state(p, "empilement") == "reviewed"
    page.refresh()
    settle(world["w"], page)
    assert page.canvas.blocks[st].d["review"] == "reviewed"
    state = json.loads((Path(world["folder"]) / "state.json").read_text(encoding="utf-8"))
    assert state["reviewed"] == [model.review_key(p, "empilement")]
    pin(model, monkeypatch)
    assert model.review_state(p, "empilement") == "pending"
    model._reviewed.clear()
    st_ = project2.read_state(world["folder"])
    st_.pop("reviewed", None)
    project2.write_state(world["folder"], st_)


def test_the_pause_is_stored_per_workflow_undone_copied_and_leaves_with_its_block(world, model, page):
    p, dn = model.view(1), model.view(3)
    assert "denoise" in model.all_stages(dn)
    model.set_breakpoint([p], "calibration")
    model.set_breakpoint([dn], "debruitage")
    model.set_breakpoint([dn], "empilement")
    assert model.p.workflow(3).breakpoints == ["stacking", "denoise"]
    assert model.breakpoints(dn) == ["empilement", "debruitage"] and model.breakpoints(model.view(2)) == []
    assert model.undo() and model.p.workflow(3).breakpoints == ["denoise"]
    assert model.redo() and model.p.workflow(3).breakpoints == ["stacking", "denoise"]
    # a copy keeps them; a block that leaves the copy takes its pause away, the original keeps its own
    model.set_breakpoint([p], "empilement")
    model.set_breakpoint([p], "fond")
    assert model.p.workflow(1).breakpoints == ["calibration", "stacking", "gradient"]
    page.bridge.act("duplicate_row", row=page.bridge.row_of(block_of(page, node(page, "fond", 1))).id)
    settle(world["w"], page)
    cp = model.paths[-1]                       # (the copy does not keep the pauses: a defect of the swap)
    model.set_breakpoint([cp], "empilement")
    model.set_breakpoint([model.view(cp.pid)], "fond")
    settle(world["w"], page)
    assert "gradient" in model.p.workflow(cp.pid).breakpoints and "stacking" in model.p.workflow(cp.pid).breakpoints
    page.remove_tool(node(page, "fond", cp.pid))
    settle(world["w"], page)
    assert "gradient" not in model.all_stages(model.view(cp.pid))
    assert "gradient" not in model.p.workflow(cp.pid).breakpoints and "stacking" in model.p.workflow(cp.pid).breakpoints
    assert model.p.workflow(1).breakpoints == ["calibration", "stacking", "gradient"]
    assert model.undo() and "gradient" in model.p.workflow(cp.pid).breakpoints
    with pytest.raises(ValueError):
        model.set_breakpoint([p], "sources")


def test_breakpoints_change_no_key_and_no_state(world, model, page):
    def snapshot():
        return ({(w.id, s): api.stage_key(model.p, w.id, s) for w in model.p.workflows for s in api.chain(model.p, w.id)},
                {k: (v.state, v.key, v.previous_key) for k, v in api.states(model.p, update=False).items()},
                {w.id: [(b.block, b.key) for b in api.blocks(model.p, w.id)] for w in model.p.workflows},
                [(j.key, j.stage) for j in model.plan_all().jobs])
    before = snapshot()
    model.set_breakpoint(model.paths, "empilement")
    model.set_breakpoint(model.paths, "post")
    model.set_breakpoint([model.view(1)], "selection")
    assert any(w.breakpoints for w in model.p.workflows)
    assert snapshot() == before
    while model.can_undo():
        model.undo()
    assert snapshot() == before and not any(w.breakpoints for w in model.p.workflows)


def test_selection_is_reviewed_with_its_thresholds_and_runs_to_registration(world, model, page):
    p = model.view(1)
    model.set_breakpoint([p], "selection")
    pause = model.pause_point()
    assert pause.block == "selection" and pause.pids == [1]
    plan, pause2 = model.plan_run()
    mine = [j for j in plan.jobs if 1 in j.wf]
    assert pause2.block == "selection" and (not mine or mine[-1].stage == "register")
    assert all(j.stage != "stack" for j in mine)
    k0 = model.review_key(p, "selection")
    model.apply_change([p], {"kind": "params", "items": [("stack", "min_transparency", 0.73)]})
    assert model.review_key(p, "selection") != k0
    model.apply_change([p], {"kind": "params", "items": [("stack", "min_transparency", None)]})
    assert model.review_key(p, "selection") == k0
    model.set_excluded([(1, "DSC00001.ARW")], True)
    assert model.review_key(p, "selection") != k0
    model.set_excluded([(1, "DSC00001.ARW")], False)
    assert model.review_key(p, "selection") == k0


def test_the_pause_in_french_on_the_graph(world, model, page, monkeypatch):
    from astro_gpu.client.graphview.menu import menu
    appmod.apply_language("fr")
    try:
        pin(model, monkeypatch)
        page.refresh()
        settle(world["w"], page)
        st = block_of(page, node(page, "empilement"))
        names = [e[0] for e in menu(page.bridge.doc, st) if e]
        assert "Pause vérif" in names and "Review pause" not in names
        for text in ("Review pause", "Mark reviewed", "Remove pause"):
            assert text in i18n._cat and i18n._cat[text] != text, text
    finally:
        appmod.apply_language("en")
        page.refresh()
        settle(world["w"], page)


# ---------------------------------------------------------------------------------------------- the view
def test_the_default_view_shows_the_session_at_any_width_and_post_when_it_fits(world, model, page):
    """The default view: the left end (the Session) in view at any width, in both languages; the whole chain to Post
    when the window is wide enough; when it is not, "Post →" at hand, its click shows Post."""
    c = page.canvas
    try:
        for lang in ("en", "fr"):
            appmod.apply_language(lang)
            page.refresh()
            for width in (1280, 1600, 1920 + PALETTE_W):
                page.root.resize(width, 934)
                c.auto_view()
                settle(world["w"], page)
                vp = c.mapToScene(c.viewport().rect()).boundingRect()
                ses = next(b for b, x in page.bridge.graph.blocks.items() if x.kind == "session")
                assert vp.contains(c.blocks[ses].sceneBoundingRect()), (lang, width)
                post = next(b for b, x in page.bridge.graph.blocks.items() if x.kind == "post")
                if c.b_post.isVisible():
                    c.b_post.click()
                    settle(world["w"], page, 3)
                    vp = c.mapToScene(c.viewport().rect()).boundingRect()
                    assert vp.contains(c.blocks[post].sceneBoundingRect()) and not c.b_post.isVisible()
                else:
                    assert vp.intersects(c.blocks[post].sceneBoundingRect()), (lang, width)
    finally:
        appmod.apply_language("en")
        page.root.resize(1680 + PALETTE_W, 934)
        page.refresh()
        settle(world["w"], page)


def test_a_band_one_column_wide_writes_its_title_on_two_lines(qapp):
    """GUI exploration 2: "TRAITEMENT LINÉAIRE" over one image block was cut; it wraps (a second line drawn)."""
    from PySide6.QtGui import QColor, QImage, QPainter

    from astro_gpu.client.graphview.items import BandItem
    i18n.set_language("fr")
    try:
        img = QImage(140, 80, QImage.Format_ARGB32)
        img.fill(QColor(0, 0, 0, 0))
        p = QPainter(img)
        BandItem(QRectF(0, 0, 120, 80), i18n._("Linear processing")).paint(p, None)
        p.end()
        text = QColor("#6b7280").rgb()
        rows = {y for y in range(80) for x in range(140) if img.pixel(x, y) == text}
        assert rows and max(rows) - min(rows) > 12, sorted(rows)
    finally:
        i18n.set_language("en")


def test_a_rebuild_keeps_where_the_graph_was_scrolled(world, page, qapp):
    """Issue #24: every change builds the graph again; built again, it stays where the user had scrolled."""
    c = page.canvas
    page.root.resize(700, 600)
    settle(world["w"], page)
    c.zoom_to(1.0)
    hb = c.horizontalScrollBar()
    try:
        if hb.maximum() == 0:
            pytest.skip("the graph fits the window: nothing to scroll")
        hb.setValue(hb.maximum() // 2)
        kept = hb.value()
        page.rebuild()
        assert hb.value() == kept
    finally:
        page.root.resize(1680 + PALETTE_W, 934)


def test_the_panel_built_again_keeps_the_setting_being_edited_focused(world, model, page, qapp):
    pn = page.panel
    n = node(page, "empilement")
    pn.refresh(n)
    qapp.processEvents()
    rows = [r for r in pn.findChildren(CompactRow) if getattr(r, "editor", None) is not None and r.editor.isEnabled()]
    if not rows:
        pytest.skip("no editable setting shown")
    row = rows[0]
    pn.focus_key = row.r["key"]
    row.editor.setFocus()
    qapp.processEvents()
    pn.refresh(n)
    for _i in range(5):
        qapp.processEvents()
    fw = QApplication.focusWidget()
    shown = [pn.bv.itemAt(i).widget() for i in range(pn.bv.count())]
    new = next(r for r in shown if isinstance(r, CompactRow) and r.r.get("key") == row.r["key"])
    assert fw is new.editor


# ---------------------------------------------------------------------------------------------- 2nd round: Post, menus
def test_round2_post_block_menu_opens_post_and_the_last_block_menu_puts_the_workflow_apart(world, model, page):
    from astro_gpu.client.graphview.menu import menu
    opened = []
    page.openPost.connect(lambda: opened.append(model.sel_step))
    post = next(b for b, x in page.bridge.graph.blocks.items() if x.kind == "post")
    page.canvas.run_menu(post, "inspect", {"block": post})
    assert opened and model.can_undo() is False                           # opening is never an undo step
    last = block_of(page, node(page, "post", 2))
    items = menu(page.bridge.doc, last)
    top = [e[0] for e in items if e and e[1] != "submenu"]
    sub = next(e for e in items if e and e[1] == "submenu" and e[0].startswith("Workflow"))
    assert "Delete workflow" in [x[0] for x in sub[2] if x] and "Delete workflow" not in top
    assert sum(t.startswith("Delete") for t in top) == 1


def test_round2_opening_a_default_recombine_in_post_is_no_undo_step(world, model, page):
    p = model.view(1)
    n = len(model._undo)
    lid = model.default_post_layer(p.pid)
    assert len(model._undo) == n and model.post_layer(lid)["name"] == f"{p.name} · Combine layers"


def test_round2_an_empty_click_deselects_and_del_and_d_do_nothing_then(world, model, page):
    from astro_gpu.client.graphview.gestures import Gestures
    g = Gestures(page)
    page.select(index=0, step="fond")
    settle(world["w"], page)
    assert page.bridge.selected is not None
    page.canvas.setFocus()
    g.key("Esc", target=page.canvas)
    settle(world["w"], page)
    assert page.bridge.selected is None and page.current_node() is None
    before = [list(model.block_keys(q)) for q in model.paths]
    g.key("Del", target=page.canvas)
    g.key("D", target=page.canvas)
    assert [list(model.block_keys(q)) for q in model.paths] == before


def test_round2_a_wire_chosen_takes_the_block_selection_and_goes_with_escape(world, model, page):
    from astro_gpu.client.graphview.gestures import Gestures
    g = Gestures(page)
    post = next(b for b, x in page.bridge.graph.blocks.items() if x.kind == "post")
    cv = page.canvas                                    # a wire to Post whose middle is its own (Post's wires from
    wire = next(w for w in page.bridge.graph.wires if w.dst[0] == post and cv.item_at(   # Combine layers run along)
        g._vp_pt(cv.wires[w.id].path().pointAtPercent(0.5))) is cv.wires[w.id])
    page.bridge.choose(block_of(page, node(page, "fond")))
    g.click_wire(wire.id)
    settle(world["w"], page)
    assert page.canvas.selected_wire == wire.id and page.bridge.selected is None
    g.key("Esc", target=page.canvas)
    assert page.canvas.selected_wire is None


def test_the_test_project_is_untouched():
    """Run last: nothing above wrote into the test project or its store."""
    assert _tree_snapshot(F_PROJECT) == PROJECT_BEFORE
