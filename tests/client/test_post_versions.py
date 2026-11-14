"""Several versions of Post in a project (issue #82): what the model keeps true whatever is done."""
import json
import random

import numpy as np
import pytest
from PySide6.QtCore import QEvent
from PySide6.QtWidgets import QApplication

from astro_gpu import api
from astro_gpu.client.model import ProjectModel
from astro_gpu.client.services.worker import ApiWorker
from astro_gpu.core import project2
from astro_gpu.core.project import Target
from astro_gpu.core.project2 import active_version, upgraded_post, versions_problem, workflow_images


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def two(qapp, tmp_path_factory):
    root = tmp_path_factory.mktemp("versions") / "P"
    p = project2.create(root, "P", Target(name="M 1"))
    raw = root.parent / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    (raw / "a1.ARW").write_bytes(b"\0" * 100)
    p.sessions.append(project2.Session(1, "night 1", lights=[str(raw)]))
    api.add_workflow(p, "Full", "Full")
    api.add_workflow(p, "Classic stack", "Classic stack")
    w = ApiWorker()
    m = ProjectModel(p, w, readonly=True)
    assert w.wait(120) and m.ready, m.error
    yield m
    m.close(save=False)                 # its timers go with it: left running they hold the event loop of the later tests
    w.stop()


@pytest.fixture()
def model(two):
    two.p.post = project2.new_post()
    yield two
    two.p.post = project2.new_post()


def by_name(m, name):
    return next(v for v in m.paths if v.name == name)


def text(v):
    return json.dumps(v, sort_keys=True)


def test_a_project_saved_before_the_versions_opens_with_its_composition_as_the_first():
    old = {"layers": [{"id": 3, "kind": "workflow", "pid": 1, "part": None}], "crop": [1, 2, 3, 4], "removed": [[2, None]]}
    post = upgraded_post(old)
    assert "layers" not in post and "crop" not in post and post["removed"] == [[2, None]]
    v = active_version(post)
    assert v["name"] == "Version 1" and v["layers"] == old["layers"] and v["crop"] == [1, 2, 3, 4]
    assert upgraded_post(post) == post
    assert upgraded_post(None) == project2.new_post()


def test_every_change_says_which_versions_it_touches(model):
    with pytest.raises(TypeError):
        model.set_post_layers([])
    with pytest.raises(TypeError):
        model._post_edit(lambda layers: None)
    with pytest.raises(ValueError):
        model.set_post_layers([], scope="some")


def test_a_new_version_is_a_copy_with_ids_of_its_own_and_is_active(model):
    full = by_name(model, "Full")
    lid = model.add_post_layer(full.pid, "stars")
    first = model.post_version()["id"]
    vid = model.new_post_version()
    post = model.p.post
    assert post["active"] == vid and len(post["versions"]) == 2 and model.post_version()["name"] == "Version 2"
    assert versions_problem(post) is None
    copy = model.post_version()["layers"]
    source = next(v for v in post["versions"] if v["id"] == first)["layers"]
    assert [x["name"] for x in copy] == [x["name"] for x in source] and lid not in [x["id"] for x in copy]


def test_a_setting_touches_the_active_version_only(model):
    full = by_name(model, "Full")
    model.add_post_layer(full.pid, "stars")
    model.new_post_version()
    other = next(v for v in model.p.post["versions"] if v["id"] != model.p.post["active"])
    before = text(other)
    lid = model.post_version()["layers"][0]["id"]
    model.set_post_layer(lid, opacity=40.0)
    assert model.post_layer(lid)["opacity"] == 40.0 and text(other) == before


def test_a_workflow_image_added_or_taken_out_touches_every_version(model):
    full = by_name(model, "Full")
    model.add_post_layer(full.pid, "stars")
    model.new_post_version()
    model.add_post_layer(full.pid, "starless")
    post = model.p.post
    assert len({text(sorted(workflow_images(v).items(), key=str)) for v in post["versions"]}) == 1
    assert (full.pid, "starless") in workflow_images(post["versions"][0]) and versions_problem(post) is None
    model.remove_post_part(full.pid, "stars")
    assert all((full.pid, "stars") not in workflow_images(v) for v in model.p.post["versions"])
    assert versions_problem(model.p.post) is None


def test_the_last_version_stays_and_names_do_not_repeat(model):
    only = model.post_version()["id"]
    model.delete_post_version(only)
    assert [v["id"] for v in model.p.post["versions"]] == [only]
    vid = model.new_post_version()
    model.rename_post_version(vid, "Version 1")
    assert model.post_version()["name"] == "Version 2"
    model.rename_post_version(vid, "  Bold  ")
    assert model.post_version()["name"] == "Bold" and model.post_version_label() == "Bold"
    model.delete_post_version(vid)
    assert model.p.post["active"] == only and model.post_version_label() == ""


def test_the_bin_is_one_step_of_undo_and_ctrl_z_brings_the_version_back(model):
    """#145: the bin deleted the version at once and for good: Ctrl+Z undid instead the last edit of the version shown
    after it. The delete is one step of Post's Undo, the version back with its layers and its crop, Ctrl+Y deletes it
    again; the only version's bin is greyed, its reason on hover."""
    from astro_gpu.client.post3 import PostPage3
    from astro_gpu.client.services.render import RenderWorker
    model.add_post_layer(by_name(model, "Full").pid, "stars")
    page = PostPage3(model, worker=RenderWorker(None, resolver=lambda pid: None, device="cpu"))
    v = page.view
    try:
        first = model.post_version()["id"]
        assert not v.b_version_del.isEnabled() and v.b_version_del.toolTip() == "The only version cannot be deleted"
        v.push()                                                   # an edit of the first version: a step of its own
        v.layers[0]["opacity"] = 50.0
        v._save()
        v._version_new()
        bold = model.p.post["active"]
        assert v._version == bold and v.b_version_del.isEnabled()
        v._set_crop([10, 10, 60, 60])
        v.layers[0]["opacity"] = 20.0
        v._save()
        kept = text(model.post_version())
        v.b_version_del.click()
        assert [x["id"] for x in model.p.post["versions"]] == [first] and v._version == first
        assert not v.b_version_del.isEnabled() and v.b_undo.isEnabled()
        v.do_undo()
        assert model.p.post["active"] == bold and v._version == bold and len(model.p.post["versions"]) == 2
        assert text(model.post_version()) == kept and v.crop == [10, 10, 60, 60] and v.layers[0]["opacity"] == 20.0
        v.do_redo()
        assert [x["id"] for x in model.p.post["versions"]] == [first] and v._version == first and v.crop is None
        v.do_undo()
        assert v._version == bold and v.crop == [10, 10, 60, 60]
        model.set_post_version(first)                              # the first version keeps its own step under it
        v.do_undo()
        assert v._version == first and v.layers[0]["opacity"] != 50.0
    finally:
        page.shutdown()
        page.deleteLater()
        # the thousands of widgets the page's rebuilds scheduled for deletion go now: left queued they were deleted
        # in the next test that spins the event loop, which starved its timers for seconds
        QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def test_the_save_says_when_the_versions_disagree_and_writes_all_the_same(model, monkeypatch):
    full = by_name(model, "Full")
    model.add_post_layer(full.pid, "stars")
    model.new_post_version()
    model.post_version()["layers"] = []
    saved, said = [], []
    monkeypatch.setattr(api, "save", lambda p: saved.append(True))
    monkeypatch.setattr(model, "log", lambda template, **a: said.append(a["problem"]))
    monkeypatch.setattr(model, "readonly", False)
    monkeypatch.setattr(model, "_dirty", True)
    model.flush()
    assert saved == [True] and "do not hold the same workflows" in said[0]


@pytest.mark.parametrize("seed", range(6))
def test_long_random_sequences_keep_the_versions_true(model, seed):
    """Graph actions, version actions, undo, save and reload, in any order."""
    r = random.Random(seed)
    m = model
    ids = [v.pid for v in m.paths]
    parts = ("stars", "starless", "combined")
    for step in range(120):
        act = r.choice(("add", "add", "remove", "new", "switch", "delete", "set", "set", "undo", "reload"))
        versions = m.p.post["versions"]
        before = {v["id"]: text(v) for v in versions if v["id"] != m.p.post["active"]}
        if act == "add":
            m.add_post_layer(r.choice(ids), r.choice(parts))
        elif act == "remove":
            m.remove_post_part(r.choice(ids), r.choice(parts))
        elif act == "new" and len(versions) < 5:
            m.new_post_version()
        elif act == "switch":
            m.set_post_version(r.choice(versions)["id"])
        elif act == "delete":
            m.delete_post_version(r.choice(versions)["id"])
        elif act == "set" and m.post_version()["layers"]:
            m.set_post_layer(r.choice(m.post_version()["layers"])["id"], opacity=float(r.randrange(100)))
            assert {v["id"]: text(v) for v in m.p.post["versions"] if v["id"] != m.p.post["active"]} == before, step
        elif act == "undo":
            snap = m._undo[-1][4] if m._undo else None
            if m.undo() and snap is not None:
                assert text(json.loads(snap)) == text(m.p.post), step
        elif act == "reload":
            stored = text(m.p.post)
            assert text(upgraded_post(json.loads(stored))) == stored, step
        assert versions_problem(m.p.post) is None, (step, act)
        active_version(m.p.post)


def test_the_graphs_undo_leaves_a_step_made_in_post_and_posts_undo_names_what_it_undid(model):
    """A version made in Post is one of Post's steps: Ctrl+Z in the graph leaves it (it took the version and its edits),
    and every Undo says what it undid."""
    from astro_gpu.client.post3 import PostPage3
    from astro_gpu.client.services.render import RenderWorker
    model.add_post_layer(by_name(model, "Full").pid, "stars")
    page = PostPage3(model, worker=RenderWorker(None, resolver=lambda pid: None, device="cpu"))
    v = page.view
    said = []
    v.toasted.connect(said.append)
    try:
        v._version_new()
        assert len(model.p.post["versions"]) == 2
        assert model.can_undo() and not model.can_undo(graph=True)
        assert model.undo(graph=True) is False and len(model.p.post["versions"]) == 2
        v.push()
        v.layers[0]["opacity"] = 40.0
        v.do_undo()
        assert said[-1].startswith("Undone: ") and "Opacity" in said[-1]
        v.do_redo()
        assert said[-1].startswith("Redone: ")
        v.do_undo()
        v.do_undo()                                                # the version is no step: Ctrl+Z stops at its first
        assert len(model.p.post["versions"]) == 2 and said[-1].startswith("Nothing to undo here")
    finally:
        page.shutdown()
        page.deleteLater()
        QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def test_an_undo_taken_while_post_is_shown_leaves_post_on_its_page_and_version(model):
    """A step undone from Post (a branch put back, say) must not select a block (Inspect opened empty) nor bring the
    version a step was made in back on screen."""
    full = by_name(model, "Full")
    first = model.p.post["active"]
    model.new_post_version()
    second = model.p.post["active"]
    model.add_post_layer(full.pid, "stars")                        # a step of the graph, made in the second version
    model.set_post_version(first)
    model.select(step="final")
    assert model.undo() and model.sel_step == "final"
    assert model.p.post["active"] == first
    assert model.can_redo() and second in [v["id"] for v in model.p.post["versions"]]


def test_a_name_another_version_has_is_refused_with_its_reason_not_taken_for_a_pick(model):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    from astro_gpu.client.post3 import PostPage3
    from astro_gpu.client.services.render import RenderWorker
    model.add_post_layer(by_name(model, "Full").pid, "stars")
    page = PostPage3(model, worker=RenderWorker(None, resolver=lambda pid: None, device="cpu"))
    v = page.view
    said = []
    v.toasted.connect(said.append)
    try:
        v._version_new()
        second = model.p.post["active"]
        ed = v.cb_version.lineEdit()
        ed.setFocus()
        ed.setText("Version 1")
        QTest.keyClick(ed, Qt.Key_Return)
        QApplication.processEvents()
        assert model.p.post["active"] == second and model.post_version()["name"] == "Version 2"
        assert v.cb_version.currentText() == "Version 2"
        assert said and said[-1].startswith("Name refused: ") and "Version 1" in said[-1]
    finally:
        page.shutdown()
        page.deleteLater()
        QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def test_the_delete_key_takes_out_the_selected_layer_and_undo_brings_it_back(model):
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QKeySequence, QShortcut
    from astro_gpu.client.post3 import PostPage3
    from astro_gpu.client.services.render import RenderWorker
    model.add_post_layer(by_name(model, "Full").pid, "stars")
    page = PostPage3(model, worker=RenderWorker(None, resolver=lambda pid: None, device="cpu"))
    v = page.view
    try:
        n = len(v.layers)
        assert n >= 2
        v.select(v.layers[0]["id"], "layer")
        keys = [sc for sc in v.findChildren(QShortcut) if sc.key() == QKeySequence("Delete")]
        assert len(keys) == 1
        keys[0].activated.emit()
        assert len(v.layers) == n - 1
        v.do_undo()
        assert len(v.layers) == n
    finally:
        page.shutdown()
        page.deleteLater()
        QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def test_spot_healing_switched_on_then_off_with_no_stroke_leaves_nothing(model):
    from astro_gpu.client.post3 import PostPage3
    from astro_gpu.client.services.render import RenderWorker
    model.add_post_layer(by_name(model, "Full").pid, "stars")
    page = PostPage3(model, worker=RenderWorker(None, resolver=lambda pid: None, device="cpu"))
    v = page.view
    try:
        n, steps = len(v.layers), len(v.undo)
        v.b_heal.setChecked(True)
        assert any(x.get("filter") == "heal" for x in v.layers)
        v.b_heal.setChecked(False)
        assert len(v.layers) == n and not any(x.get("filter") == "heal" for x in v.layers)
        assert len(v.undo) == steps and v.sel is not None
    finally:
        page.shutdown()
        page.deleteLater()
        QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def _page_of(model):
    from astro_gpu.client.post3 import PostPage3
    from astro_gpu.client.services.render import RenderWorker
    return PostPage3(model, worker=RenderWorker(None, resolver=lambda pid: None, device="cpu"))


def _done(page):
    page.shutdown()
    page.deleteLater()
    QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def _layers_of(model, vid):
    return json.loads(text(next(x for x in model.p.post["versions"] if x["id"] == vid)["layers"]))


def test_a_layer_duplicated_then_deleted_in_one_version_leaves_the_others_layers_and_filters_alone(model):
    """The data-loss sequence: Saturation in Version 1, a second version, the layer duplicated and its copy deleted
    there: Version 1 holds the layer and its filter exactly as it was, and Ctrl+Z there restores them as they were."""
    model.add_post_layer(by_name(model, "Full").pid, "stars")
    page = _page_of(model)
    v = page.view
    try:
        base = v.layers[-1]
        v.select(base["id"], "layer")
        fl = v.add_filter("saturation")
        key = next(iter(fl["params"]))
        fl["params"][key] = 1.5
        v._save()
        first = model.p.post["active"]
        one = _layers_of(model, first)
        names = [x["name"] for x in v.layers]
        v._version_new()
        second = model.p.post["active"]
        assert _layers_of(model, first) == one
        v.select(v.layers[-1]["id"], "layer")
        v.duplicate()
        v._save()
        assert _layers_of(model, first) == one                        # the copy is Version 2's alone
        copy_ = next(x for x in v.layers if x["kind"] != "filter" and x["name"] not in names)
        v.select(copy_["id"], "layer")
        v.delete()
        v._save()
        assert _layers_of(model, first) == one                        # nor does deleting it reach Version 1
        model.set_post_version(first)
        v._post_changed(None)
        assert [x["name"] for x in v.layers] == names
        kept = next(x for x in v.layers if x["kind"] == "filter")
        assert kept["params"][key] == 1.5
        v.select(base["id"], "layer")
        v.push()                                                      # Version 1's own edit, undone: exactly as before
        v.layers[-1]["opacity"] = 10.0
        v._save()
        v.do_undo()
        assert next(x for x in v.layers if x["kind"] == "filter")["params"][key] == 1.5
        assert _layers_of(model, first) == one
        assert second in [x["id"] for x in model.p.post["versions"]]
    finally:
        _done(page)


def test_what_spreads_between_versions_is_the_workflows_images_not_the_layers_made_of_them(model):
    full, classic = by_name(model, "Full"), by_name(model, "Classic stack")
    model.add_post_layer(full.pid, "stars")
    page = _page_of(model)
    v = page.view
    try:
        first = model.p.post["active"]
        v._version_new()
        second = model.p.post["active"]
        before = _layers_of(model, first)
        for op in ("duplicate", "filter", "opacity", "mode", "hide", "rename", "reorder", "delete_copy"):
            v.select(v.layers[-1]["id"], "layer")
            lay = v.layers[-1]
            if op == "duplicate":
                v.duplicate()
            elif op == "filter":
                v.add_filter("saturation")
            elif op == "opacity":
                v.push(); lay["opacity"] = 30.0
            elif op == "mode":
                v.push(); lay["mode"] = "Screen"
            elif op == "hide":
                v.push(); lay["visible"] = False
            elif op == "rename":
                v.push(); lay["name"] = "Other"
            elif op == "reorder" and len(v.layers) > 1:
                v.push(); v.layers.reverse()
            elif op == "delete_copy":
                extra = next((x for x in v.layers if "copy" in x["name"].lower()), None)
                if extra is not None:
                    v.select(extra["id"], "layer")
                    v.delete()
            v._save()
            assert _layers_of(model, first) == before, op
        # a workflow's image that is new to Post is shared: its first layer, in each version
        model.add_post_layer(classic.pid, None)
        for vid in (first, second):
            ver = next(x for x in model.p.post["versions"] if x["id"] == vid)
            assert any(x.get("pid") == classic.pid and x.get("part") is None for x in ver["layers"])
        assert versions_problem(model.p.post) is None
        # and one the graph takes out goes from both
        model.remove_post_part(classic.pid, None)
        for vid in (first, second):
            ver = next(x for x in model.p.post["versions"] if x["id"] == vid)
            assert not any(x.get("pid") == classic.pid and x.get("part") is None for x in ver["layers"])
    finally:
        _done(page)


def test_a_versions_only_image_layer_hidden_with_its_eye_stays_hidden_after_switching_away_and_back(model):
    page = _page_of(model)
    v = page.view
    try:
        model.add_post_layer(by_name(model, "Full").pid, None)
        first = model.p.post["active"]
        v._post_changed(None)
        for lay in v.layers:
            if lay["kind"] == "workflow":
                lay["visible"] = False
        v._save()
        hidden = _layers_of(model, first)
        v._version_new()
        second = model.p.post["active"]
        model.set_post_version(first)
        v._post_changed(None)
        assert [x["visible"] for x in v.layers if x["kind"] == "workflow"] == [False] * sum(
            1 for x in v.layers if x["kind"] == "workflow")
        v._save()
        assert _layers_of(model, first) == hidden and second != first
    finally:
        _done(page)


def test_the_enlarged_thumbnail_stays_on_the_screen():
    from PySide6.QtCore import QPoint, QRect
    from astro_gpu.client.post3 import PostView
    area = QRect(0, 0, 1920, 1080)
    assert PostView._big_place(QPoint(300, 200), 60, 180, 120, area) == (112, 200)         # room on the left: there
    x, y = PostView._big_place(QPoint(60, 1050), 60, 180, 120, area)                       # none: right of the thumbnail
    assert x == 128 and y + 120 <= 1080
    assert PostView._big_place(QPoint(1900, 10), 60, 180, 120, area)[0] + 180 <= 1920


def _two_versions(model):
    """Two layers of workflows' images (one the way an image opened from a file is), the first with its own opacity and
    a filter, then a second version; returns the page, the ids of the versions and the layers' state of the first."""
    model.add_post_layer(by_name(model, "Full").pid, "stars")
    model.add_post_layer(by_name(model, "Classic stack").pid, None)
    page = _page_of(model)
    v = page.view
    v.select(v.layers[-1]["id"], "layer")
    v.push()
    v.layers[-1]["opacity"] = 40.0
    v.add_filter("saturation")
    v._save()
    first = model.p.post["active"]
    v._version_new()
    return page, first, model.p.post["active"]


def test_duplicate_then_undo_leaves_the_other_version_whole_in_both_directions(model):
    page, first, second = _two_versions(model)
    v = page.view
    try:
        one, two = _layers_of(model, first), _layers_of(model, second)
        for lay in (v.layers[0], v.layers[-1]):                       # an image from a file, then the one with its filter
            v.select(lay["id"], "layer")
            v.duplicate()
            v._save()
            assert _layers_of(model, first) == one
            v.do_undo()
            v._save()
            assert _layers_of(model, first) == one and _layers_of(model, second) == two
        model.set_post_version(first)                                 # the other direction
        v._post_changed(None)
        v.select(v.layers[0]["id"], "layer")
        v.duplicate()
        v._save()
        assert _layers_of(model, second) == two
        v.do_undo()
        v._save()
        assert _layers_of(model, first) == one and _layers_of(model, second) == two
    finally:
        _done(page)


def test_duplicate_delete_the_copy_and_two_undos_repair_both_versions_exactly(model):
    page, first, second = _two_versions(model)
    v = page.view
    try:
        one, two = _layers_of(model, first), _layers_of(model, second)
        names = [x["name"] for x in v.layers]
        v.select(v.layers[-1]["id"], "layer")
        v.duplicate()
        copy_ = next(x for x in v.layers if x["kind"] != "filter" and x["name"] not in names)
        v.select(copy_["id"], "layer")
        v.delete()
        v._save()
        assert _layers_of(model, first) == one
        v.do_undo()
        v.do_undo()
        v._save()
        assert _layers_of(model, first) == one and _layers_of(model, second) == two
    finally:
        _done(page)


def test_a_layer_deleted_in_one_version_leaves_the_other_and_undo_puts_it_back_in_its_place(model):
    page, first, second = _two_versions(model)
    v = page.view
    try:
        one, two = _layers_of(model, first), _layers_of(model, second)
        file_layer = v.layers[0]                                      # the image of a workflow, at the top
        before = [x["id"] for x in v.layers]
        v.select(file_layer["id"], "layer")
        v.delete()
        v._save()
        assert _layers_of(model, first) == one
        assert file_layer["id"] not in [x["id"] for x in v.layers]
        v.do_undo()
        v._save()
        assert [x["id"] for x in v.layers] == before and _layers_of(model, second) == two
        assert _layers_of(model, first) == one
        # a layer under others: back under them, in its own version
        low = v.layers[-1]
        v.select(low["id"], "layer")
        v.delete()
        v._save()
        v.do_undo()
        v._save()
        assert [x["id"] for x in v.layers] == before and _layers_of(model, first) == one
    finally:
        _done(page)


def test_the_source_of_spot_healing_set_by_hand_belongs_to_its_version_and_its_undo(model):
    page, first, second = _two_versions(model)
    v = page.view
    try:
        model.set_post_version(first)
        v._post_changed(None)
        v._heal_src = (10.0, 12.0)                                    # Alt+click in the first version
        model.set_post_version(second)
        v._post_changed(None)
        assert v._heal_src is None and v._heal_off is None            # nothing carried to the other version
        model.set_post_version(first)
        v._post_changed(None)
        assert v._heal_src == (10.0, 12.0)
        v._heal_off = (3.0, 4.0)                                      # a spot was healed with it, then undone
        v.layers = [x for x in v.layers if x.get("filter") != "heal"]
        v._restored()
        assert v._heal_src is None and v._heal_off is None
    finally:
        _done(page)


def test_the_fits_export_through_the_render_service_carries_the_shots_facts(tmp_path):
    """The service's writer dropped them: the header had no OBJECT, DATE-OBS or EXPTIME."""
    from types import SimpleNamespace

    from astropy.io import fits

    from astro_gpu.client.services import render as rs
    p = project2.create(tmp_path / "P", "P", Target(name="Rosette Nebula"))
    cls = next(c for c in vars(rs).values() if isinstance(c, type) and "_write" in vars(c) and c.__module__ == rs.__name__)
    me = SimpleNamespace(project=p)
    rgb = np.random.default_rng(0).uniform(0, 1, (3, 8, 10)).astype(np.float32)
    meta = {"origin": (0, 0), "history": None, "facts": {"target": "Rosette Nebula", "date_obs": "2026-11-01T21:00:00",
                                                         "exposure_s": 3600.0}}
    files = cls._write(me, rgb, tmp_path / "out", {"name": "x", "formats": ["fits32f"]}, meta)
    h = fits.getheader(next(f for f in files if f.suffix == ".fits"))
    assert h["OBJECT"] == "Rosette Nebula" and h["EXPTIME"] == 3600.0 and h["DATE-OBS"].startswith("2026-11-01")


def test_posts_undo_does_not_take_back_a_to_post_made_in_the_graph(model):
    full, classic = by_name(model, "Full"), by_name(model, "Classic stack")
    model.add_post_layer(full.pid, "stars")
    page = _page_of(model)
    v = page.view
    said = []
    v.toasted.connect(said.append)
    try:
        first = model.p.post["active"]
        v._version_new()
        model.add_post_layer(classic.pid, None)                   # "To Post" in the graph, Version 2 shown
        v._post_changed(None)
        model.set_post_version(first)
        v._post_changed(None)
        assert not v.undo
        v.do_undo()
        assert said[-1].startswith("Nothing to undo here") and any(
            x.get("pid") == classic.pid for x in next(z for z in model.p.post["versions"] if z["id"] == first)["layers"])
    finally:
        _done(page)


def _page(model):
    from astro_gpu.client.post3 import PostPage3
    from astro_gpu.client.services.render import RenderWorker
    model.add_post_layer(by_name(model, "Full").pid, "stars")
    return PostPage3(model, worker=RenderWorker(None, resolver=lambda pid: None, device="cpu"))


def _close(page):
    page.shutdown()
    page.deleteLater()
    QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def test_ctrl_z_stops_at_the_versions_first_step_and_never_makes_a_version_go(model):
    """A new version, edits and a rename: Ctrl+Z undoes the rename and the edits, then says there is nothing left; the
    version stays; Ctrl+Z in it never reaches another version's steps."""
    page = _page(model)
    v = page.view
    said = []
    v.toasted.connect(said.append)
    try:
        v.push()
        v.layers[0]["opacity"] = 30.0                                   # a step of the first version
        v._version_new()
        second = v._version
        assert v.undo == []
        v.push()
        v.layers[0]["opacity"] = 60.0
        v.cb_version.setEditText("Mine")
        v._version_renamed()
        for _ in range(6):
            v.do_undo()
        assert len(model.p.post["versions"]) == 2 and v._version == second
        assert said[-1].startswith("Nothing to undo here")
        assert [x["name"] for x in model.p.post["versions"]][1] == "Version 2"
        assert v.layers[0]["opacity"] == 30.0                           # the copy's own start, not the first's steps
        first = model.p.post["versions"][0]["id"]
        model.set_post_version(first)
        v._post_changed(None)
        assert len(v.undo) == 1                                         # the first version keeps its own step
    finally:
        _close(page)


def test_a_version_never_takes_back_a_model_step_that_is_not_its_latest(model):
    """A version's rename is not undone once the workflows' Undo holds a later step made elsewhere."""
    page = _page(model)
    v = page.view
    said = []
    v.toasted.connect(said.append)
    try:
        v.cb_version.setEditText("Named")
        v._version_renamed()
        model.new_post_version()                                        # a later step of the model, not this list's
        v._post_changed(None)
        v.undo.append(v.undo[-1] if v.undo else None)
        v.undo = [x for x in v.undo if x is not None]
        v.do_undo()
        assert said[-1].startswith("Nothing to undo here")
        assert any(x["name"] == "Named" for x in model.p.post["versions"])
    finally:
        _close(page)


def test_every_post_step_updates_the_windows_undo_actions(model):
    """A crop made first leaves the Edit menu's Undo / Redo to be refreshed: Post tells the window at every step."""
    page = _page(model)
    v = page.view
    seen = []
    v.undoChanged.connect(lambda: seen.append(bool(v.undo)))
    try:
        v._set_crop([0, 0, 20, 20])
        assert seen and seen[-1] is True
        n = len(seen)
        v.do_undo()
        assert len(seen) > n and seen[-1] is False
    finally:
        _close(page)


def test_escape_in_the_versions_name_keeps_the_old_name_and_stays_in_post(model):
    """As a layer's name: Escape drops the typing; it does not reach the window's page-leaving shortcut."""
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QKeyEvent
    page = _page(model)
    v = page.view
    try:
        old = v.cb_version.currentText()
        ed = v.cb_version.lineEdit()
        ed.setText("Typed")
        so = QKeyEvent(QEvent.ShortcutOverride, Qt.Key_Escape, Qt.NoModifier)
        so.ignore()
        QApplication.sendEvent(ed, so)
        assert so.isAccepted()                                          # the shortcut is not offered the key
        QApplication.sendEvent(ed, QKeyEvent(QEvent.KeyPress, Qt.Key_Escape, Qt.NoModifier))
        assert v.cb_version.currentText() == old
        assert [x["name"] for x in model.p.post["versions"]] == [old]
    finally:
        _close(page)


def test_undoing_a_rename_gives_back_the_name_only_and_keeps_the_other_versions_edits(model):
    """Rename V2, add a filter in V1, back to V2, Ctrl+Z: V2's old name is back and V1 keeps its filter; the same for
    a delete brought back."""
    page = _page(model)
    v = page.view
    try:
        v._version_new()
        second = v._version
        first = next(x["id"] for x in model.p.post["versions"] if x["id"] != second)
        v.cb_version.setEditText("Mine")
        v._version_renamed()
        model.set_post_version(first)
        v._post_changed(None)
        v.add_filter("sharpen")
        n = len(v.layers)
        v._save()
        model.set_post_version(second)
        v._post_changed(None)
        v.do_undo()
        names = {x["id"]: x["name"] for x in model.p.post["versions"]}
        assert names[second] == "Version 2"
        assert len(next(x for x in model.p.post["versions"] if x["id"] == first)["layers"]) == n
        v.do_redo()
        assert {x["id"]: x["name"] for x in model.p.post["versions"]}[second] == "Mine"
        assert len(next(x for x in model.p.post["versions"] if x["id"] == first)["layers"]) == n
    finally:
        _close(page)


def test_undoing_a_workflow_step_made_from_post_keeps_the_other_versions_edits(model):
    """A layer's workflow renamed in the second version, then a filter added in the first: Ctrl+Z puts the name back
    and the first version keeps its filter (and the redo does the same)."""
    page = _page(model)
    v = page.view
    try:
        v._version_new()
        second = v._version
        first = next(x["id"] for x in model.p.post["versions"] if x["id"] != second)
        lay = next(x for x in v.layers if x["kind"] == "workflow" and x.get("part") is None
                   and x.get("name") == v.path(x["pid"]).name)
        pid = lay["pid"]
        was = model.view(pid).name
        v.rename(lay, "Elsewhere")
        assert model.view(pid).name == "Elsewhere"
        model.set_post_version(first)
        v._post_changed(None)
        v.add_filter("sharpen")
        n = len(v.layers)
        v._save()
        model.set_post_version(second)
        v._post_changed(None)
        v.do_undo()
        assert model.view(pid).name == was
        assert len(next(x for x in model.p.post["versions"] if x["id"] == first)["layers"]) == n
        v.do_redo()
        assert model.view(pid).name == "Elsewhere"
        assert len(next(x for x in model.p.post["versions"] if x["id"] == first)["layers"]) == n
    finally:
        _close(page)
