"""A combination's master layer on the owner's M27 (S-H-O). The test opens a copy of the project's definition alone
(no link to its store); the results are read where they are, read only, by the paths given below: the definition is
newer than them (its keys moved since they were computed), so its layers and its editor are given its S-H-O's
results by hand.

The master layer: the project opens with its Combine layers layer as the master layer, its filters as rows, its picture
made from the filters; its hues and intensities are the layer's own (one source for Post, the block's panel and
Inspect's editor), never outdate the block nor ask for a run; the balance stays the block's. The editor balances as the
run does (the same gains as Post), shows Draft and Applied alike at the same settings, has the Gradient's layout.
Also the plain cases' rule (client/postfusion.py)."""
from __future__ import annotations

import copy
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication  # noqa: E402

from astro_gpu.client import combineedit as ce  # noqa: E402
from astro_gpu.client import postengine as pe  # noqa: E402
from astro_gpu.client import postfusion  # noqa: E402
from astro_gpu.stages import recipes  # noqa: E402

M27 = Path("F:/astro_essais/tutoriels_ref/MONO/m27_v10")
#: the S-H-O's results, read where they are: the starless combination (balanced On signal, made before the balance's
#: maps were written) and its Output, the stars' combination's Output
RESULTS = {("fusion", 5): "store/combine_sho/d7eecb71bbbea37f", 5: "store/render_simple/d98430ad8cd1e148",
           4: "store/render_simple/fc11a390824d10e4"}
COMBINATION = M27 / RESULTS[("fusion", 5)]


def _resolver(sid):
    from astro_gpu.client.services.render import Folder
    rel = RESULTS.get(sid)
    if rel is None:
        return None
    d = M27 / rel
    return Folder(d, False, (d / "manifest.json").stat().st_mtime) if d.is_dir() else None


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


# ---------------------------------------------------------------------------------------------- the rule
def test_a_lightness_recipe_the_stars_and_a_block_after_the_combination_stay_plain():
    sho, lrgb = recipes.legacy_spec("sho"), recipes.legacy_spec("lrgb")
    ins = [{"role": r, "image": "starless"} for r in ("SII", "Ha", "OIII")]
    assert postfusion.plain_reason("combine_sho", sho, ins, ["combine_sho", "render_simple"]) == ""
    assert postfusion.plain_reason("combine_lrgb", lrgb, [], ["combine_lrgb", "render_simple"]) == "lightness"
    assert postfusion.plain_reason("combine_sho", sho, [dict(i, image="stars") for i in ins], []) == "stars"
    assert postfusion.plain_reason("combine_sho", sho, ins, ["combine_sho", "color", "render_simple"]) == "after"
    assert postfusion.plain_reason("combine_rgb", recipes.legacy_spec("rgb"), [], ["combine_rgb", "render_simple"]) == ""


def test_the_layer_s_hues_and_intensities_stand_in_for_the_block_s_and_a_change_splits_between_them():
    sho = recipes.legacy_spec("sho")
    stored = {"hue_oiii": 180.0, "weights": {"SII": 2.0}, "balance": "signal"}
    tune = {"hues": {"SII": 15.5, "Ha": 31.4, "OIII": 210.0}, "weights": {"SII": 1.0, "Ha": 0.5, "OIII": 1.0}}
    got = postfusion.tuned_stored("combine_sho", sho, stored, tune)
    assert got["hue_oiii"] == 210.0 and got["weights"] == {"Ha": 0.5} and got["balance"] == "signal"
    assert postfusion.tuned_stored("combine_sho", sho, stored, None) == stored
    part, rest = postfusion.split_change(sho, {"hue_oiii": 200.0, "weights": {"OIII": 0.7}, "balance": "sky"})
    assert part == {"hues": {"OIII": 200.0}, "weights": {"OIII": 0.7}} and rest == {"balance": "sky"}
    new = postfusion.new_tune("combine_sho", sho, stored, None, part)
    assert new == {"hues": {"SII": 15.5, "Ha": 31.4, "OIII": 200.0}, "weights": {"SII": 2.0, "Ha": 1.0, "OIII": 0.7}}


def test_the_description_is_the_combination_s_settings_and_its_matrix():
    from astro_gpu.stages.combine import hue_palette
    f = postfusion.describe("combine_sho", recipes.legacy_spec("sho"), {"hue_oiii": 180.0, "weights": {"SII": 2.0}},
                            {"SII": "S II", "Ha": "Hα", "OIII": "O III"}, {"weights": {"Ha": 0.5}})
    assert f["roles"] == ["SII", "Ha", "OIII"] and f["hues"]["OIII"] == 180.0
    assert f["weights"] == {"SII": 2.0, "Ha": 0.5, "OIII": 1.0}
    assert f["matrix"] == [[float(v) for v in row] for row in hue_palette((15.5, 31.4, 180.0))]
    assert f["hue_defaults"] == {"SII": 15.5, "Ha": 31.4, "OIII": 202.2} and f["palette"] == ""


# ---------------------------------------------------------------------------------------------- the owner's M27
def _open_copy(tmp: Path):
    """The project's definition copied alone, opened read only (nothing computed: the results come by hand)."""
    from test_inspect import Harness

    from astro_gpu.client import app as appmod
    from astro_gpu.client import prefs
    folder = tmp / "m27"
    folder.mkdir(parents=True, exist_ok=True)
    shutil.copy(M27 / "project.astroproj", folder / "project.astroproj")
    os.environ["ASTRO_GPU_SETTINGS"] = str(tmp / "settings.json")
    prefs.load()
    appmod.apply_language("en")

    class Whole(Harness):
        def make_model(self):
            appmod.Main.make_model(self)
    w = Whole(readonly=True)
    w.resize(1500, 900)
    w.show()
    w.open_project(folder)
    for _i in range(400):
        if w.s.ready:
            break
        w.worker.wait(5)
        time.sleep(0.05)
    assert w.s.ready, w.s.error
    return w


def _reduced(project, pid, stage, spec):
    _st, _spec, ins = postfusion.result_inputs(COMBINATION)
    return ce.reduced_inputs(ins, spec)


@pytest.fixture(scope="module")
def m27(qapp, tmp_path_factory):
    if not (M27 / "project.astroproj").is_file() or not COMBINATION.is_dir():
        pytest.skip("the project m27_v10 is not here")
    mp = pytest.MonkeyPatch()
    mp.setattr(ce, "_reduced_for", _reduced)
    mp.setattr(ce, "_result_dir_for", lambda project, pid, stage: str(COMBINATION))
    mp.setattr(ce, "_applied_for", lambda project, pid, stage, k, settings=None: ce.applied_small(
        str(COMBINATION), k, settings))
    w = _open_copy(tmp_path_factory.mktemp("m27"))
    from astro_gpu.client.post3 import PostView
    from astro_gpu.client.services.render import RenderWorker
    before = copy.deepcopy(w.s.p.post)
    pv = PostView(w.s, persist=False, worker=RenderWorker(w.s.p, resolver=_resolver, device="cpu"))
    yield w, pv, before
    pv.shutdown()
    w.close()
    mp.undo()


def _master(pv):
    pv._attach_fusion()
    return next(x for x in pv.layers if pe.is_group(x) and x.get("starless") == "output")


def _settle(w):
    from test_inspect import settle
    settle(w)


def _render(pv, level=2):
    from astro_gpu.client.services.render import Request
    pv._attach_fusion()
    if getattr(pv.worker, "renderer", None) is None:
        pv.worker._make()
    return pv.worker.renderer.run(Request(kind="half", layers=pe.snapshot(pv.layers), level=level))


def _editor(w, monkeypatch=None):
    from test_combine_editor import _open_combination, wait
    q, ed = _open_combination(w)
    if monkeypatch is not None:
        monkeypatch.setattr(ed, "has_result", lambda p=None: True)       # (its results are older than its keys here)
    wait(ed, 180)
    return q, ed


def _block(w, pid, stage):
    return copy.deepcopy(w.s.p.workflow(pid).params.get(stage) or {})


def _write(pv):
    """Post's layers into the project, as its saving timer does (the test's Post does not save by itself)."""
    pv.persist = True
    pv._save()
    pv.persist = False


def test_m27_opens_with_its_combination_as_the_master_layer_and_its_definition_unchanged(m27):
    w, pv, before = m27
    lay = _master(pv)
    assert pe.is_fusion(lay) and lay["fusion"]["roles"] == ["SII", "Ha", "OIII"]
    names = [(r.get("name"), d) for r, d in pe.rows(pv.layers)]
    i = names.index(("Starless image", 1))
    assert names[i + 1:i + 4] == [("S II", 2), ("Hα", 2), ("O III", 2)]
    assert w.s.p.post == before                                 # converted in memory only
    stored = next(x for x in before["versions"][0]["layers"] if x.get("part") == "combined")
    assert next(x for x in pe.save_layers(pv.layers) if x["id"] == lay["id"]) == stored
    st = lay["fusion"]["stage"]                                 # its first hues and intensities: the block's
    blk = _block(w, lay["pid"], st)
    assert lay["fusion"]["hues"]["OIII"] == float(blk.get("hue_oiii", 202.2)) and "tune" not in lay


def test_m27_s_picture_is_made_from_its_filters_with_the_run_s_balance(m27):
    w, pv, _before = m27
    res = _render(pv)
    assert res.rgb is not None and res.rgb.std() > 3
    cp = pv.worker.renderer.composer
    lay = _master(pv)
    cv = cp.canvas_of(pv.layers)
    kids = [cp.layer_rgb(pe.sub_layer(lay, pe.role_sub(r)), 2, (0, 0, *cv.size(2)), cv) for r in ("SII", "Ha", "OIII")]
    assert all(k.std() > 0.005 for k in kids)
    src = pv.worker.renderer.source
    full = src._fusion_full((pe.FUSION, lay["pid"]), lay["fusion"]["balance"])
    mat = np.asarray(lay["fusion"]["matrix"], np.float32)
    w8 = lay["fusion"]["weights"]
    mix = np.tensordot(mat, np.stack([full["planes"][r][::16, ::16] * w8[r] for r in ("SII", "Ha", "OIII")]), 1)
    drz = np.load(COMBINATION / "drz_full.npy", mmap_mode="r")[:, ::16, ::16]
    run = json.loads((COMBINATION / "manifest.json").read_text(encoding="utf-8"))["params"]
    if run.get("balance") == lay["fusion"]["balance"] and float(run.get("hue_oiii")) == lay["fusion"]["hues"]["OIII"]:
        assert np.abs(mix - drz).max() <= 2e-3 * float(np.abs(drz).max())


def test_a_hue_set_in_post_is_the_layer_s_a_step_of_post_and_never_outdates_the_block(m27, monkeypatch):
    w, pv, _before = m27
    s = w.s
    lay = _master(pv)
    st = lay["fusion"]["stage"]
    q, ed = _editor(w, monkeypatch)
    ed.set_intensity("SII", 1.5)                                       # a draft in Inspect
    assert ed.has_draft()
    said = []
    w.tune.toasted.connect(said.append)
    blk, state, key = _block(w, q.pid, st), s.state(q, st), s.stage_key(q, st)
    n, n_post = len(s._undo), len(pv.undo)
    pv.set_fusion(lay, {"hue_oiii": 180.0}, "O III hue")
    _write(pv)
    _settle(w)
    assert _block(w, q.pid, st) == blk and s.state(q, st) == state and s.stage_key(q, st) == key   # the block as it was
    assert len(s._undo) == n and len(pv.undo) == n_post + 1                                         # a step of Post
    lay = _master(pv)
    assert lay["tune"]["hues"]["OIII"] == 180.0 and lay["fusion"]["hues"]["OIII"] == 180.0
    assert not ed.has_draft() and "Draft dropped: changed in Post" in said
    assert ed.applied()["hue_oiii"] == 180.0                          # Inspect shows the layer's value
    assert pv.next_words("undo") == "O III hue"
    pv.do_undo()
    _write(pv)
    assert "tune" not in _master(pv) and ed.applied()["hue_oiii"] == float(blk.get("hue_oiii", 202.2))


def test_the_balance_set_in_post_is_the_block_s_and_needs_a_run(m27):
    w, pv, _before = m27
    s = w.s
    lay = _master(pv)
    st = lay["fusion"]["stage"]
    q = s.view(lay["pid"])
    key = s.stage_key(q, st)
    other = "sky" if lay["fusion"]["balance"] == "signal" else "signal"
    n = len(s._undo)
    pv.set_fusion(lay, {"balance": other}, "Balance")
    _settle(w)
    assert _block(w, q.pid, st).get("balance", "sky") == other and s.stage_key(q, st) != key and len(s._undo) == n + 1
    s.undo()
    _settle(w)
    assert s.stage_key(q, st) == key


def test_inspect_s_apply_of_a_hue_writes_the_layer_one_step_of_the_window_and_post_shows_it(m27, monkeypatch):
    w, pv, _before = m27
    s = w.s
    q, ed = _editor(w, monkeypatch)
    st = ed._stage()
    blk, key = _block(w, q.pid, st), s.stage_key(q, st)
    assert ed._render_settings()["keep_colours"] == 1.0              # the preview stretched as the master layer is
    ed.set_hue("OIII", 190.0)
    assert ed.cost_text() == "Post's layer only: nothing is computed again"
    n = len(s._undo)
    ed.apply()
    _settle(w)
    assert len(s._undo) == n + 1 and _block(w, q.pid, st) == blk and s.stage_key(q, st) == key
    assert (s.master_layer(q.pid) or {}).get("tune", {}).get("hues", {}).get("OIII") == 190.0
    assert _master(pv)["fusion"]["hues"]["OIII"] == 190.0
    s.undo()
    _settle(w)
    assert "tune" not in (s.master_layer(q.pid) or {})


def test_the_panel_s_route_an_intensity_goes_to_the_layer(m27):
    w, pv, _before = m27
    s = w.s
    lay = _master(pv)
    st = lay["fusion"]["stage"]
    q = s.view(lay["pid"])
    blk = _block(w, q.pid, st)
    s.apply_as({"kind": "combine", "values": {"weights": {"Ha": 0.8}}, "what": "", "short": ""}, "replace", [q])
    _settle(w)
    assert _block(w, q.pid, st) == blk and s.master_layer(q.pid)["tune"]["weights"]["Ha"] == 0.8
    s.undo()
    _settle(w)


def test_the_editor_balances_as_the_run_draft_and_applied_agree_and_its_page_is_the_gradient_s(m27, monkeypatch):
    from test_combine_editor import wait
    w, pv, _before = m27
    q, ed = _editor(w, monkeypatch)
    # the gains: the run's balance (measured once on the whole image: this result predates the maps), Post's own
    lay = _master(pv)
    _render(pv)
    post = pv.worker.renderer.source.fusion_gains(lay)
    assert {r: round(g, 6) for r, (g, _w) in ed.notes.items()} == {r: round(g, 6) for r, (g, _w) in post.items()}
    assert ed._bal_now()["from"] in ("run", "measured")
    # Draft and Applied at the same settings: the same picture
    ed.set_hue("OIII", 200.0)
    wait(ed, 120)
    ed.reset()
    wait(ed, 120)
    ap = ed.applied_img
    assert ap is not None and ap.get("img") is not None and np.abs(ap["img"] - ed.result).max() < 1e-6
    # the page: the draft bar under the settings, the panel at the right with the measurements, the image's tag
    bar, card = ed._ui["bar"], ed._ui["hue_OIII"]
    assert bar.mapTo(ed.page, bar.rect().topLeft()).y() > card.mapTo(ed.page, card.rect().topLeft()).y()
    texts = " ".join(x.text() for x in ed.drawer.findChildren(type(ed._ui["title"])))
    assert "Measurements" in texts and "S II" in texts and "×" in texts and "Draft" in texts
    assert ed._ui.get("undo") is not None
    ed.set_hue("OIII", 205.0)
    wait(ed, 120)
    ed._present()
    assert ed.preview_tag.startswith("Draft preview · reduced copy") and ed.preview_tag.endswith(" ms")
    ed.reset()


def test_at_1250_px_the_side_panel_is_the_editor_s(m27, monkeypatch):
    w, pv, _before = m27
    q, ed = _editor(w, monkeypatch)
    w.resize(1250, 685)
    for _i in range(20):
        QApplication.processEvents()
        time.sleep(0.02)
    assert not w.tune.panel.isVisible()
    assert ed.dock.state == "auto" and ed.drawer is not None
    w.resize(1500, 900)
    QApplication.processEvents()


def test_a_child_filter_its_eye_and_its_mask_are_post_s_steps_and_act_on_the_picture(m27):
    w, pv, _before = m27
    lay = _master(pv)
    plain = _render(pv).rgb.astype(np.float32)
    n = len(pv.undo)
    pv.sel = (lay["id"], "f:OIII")
    pv.add_filter("levels")
    fl = next(x for x in pv.layers if x["kind"] == "filter" and x.get("sub") == "f:OIII")
    assert fl["target"] == lay["id"] and len(pv.undo) == n + 1
    fl["params"].update(black=0.2)
    filt = _render(pv).rgb.astype(np.float32)
    assert np.abs(filt - plain).mean() > 0.5
    pv.set_shown(lay["id"], "f:OIII", False)
    assert len(pv.undo) == n + 2 and pv.next_words("undo") == "“O III” hidden"
    off = _render(pv).rgb.astype(np.float32)
    assert np.abs(off - plain).mean() > 0.5
    pv.do_undo()
    pv.do_undo()
    assert not any(x.get("sub") == "f:OIII" for x in pv.layers if x["kind"] == "filter")


def test_the_thumbnails_are_made_again_after_an_undo_of_a_hue(m27):
    w, pv, _before = m27
    lay = _master(pv)
    from PySide6.QtGui import QPixmap
    pm = QPixmap(54, 36)
    sig = pv._thumb_sig(lay)
    pv._lthumbs[lay["id"]] = (sig, pm)
    pv.set_fusion(lay, {"hue_oiii": 150.0}, "O III hue")
    assert lay["id"] in set(pv._stale_thumbs())
    pv._lthumbs[lay["id"]] = (pv._thumb_sig(_master(pv)), pm)
    pv.do_undo()
    assert lay["id"] in set(pv._stale_thumbs())                      # the picture of before: made again


def test_the_master_panel_shows_the_combination_s_settings_and_colours_greyed(m27):
    w, pv, _before = m27
    lay = _master(pv)
    pv.sel = None
    pv.select(lay["id"], "layer")
    from astro_gpu.client.post3 import ContSlider
    names = [x.name.text() for x in pv.findChildren(ContSlider) if x.isVisibleTo(pv.right)]
    assert {"S II hue", "Hα hue", "O III hue", "S II intensity", "Hα intensity", "O III intensity"} <= set(names)
    col = next(x for x in pv.findChildren(ContSlider) if x.name.text() == "Colours")
    assert not col.s.isEnabled()
    pv.select(lay["id"], "f:SII")
    names = [x.name.text() for x in pv.findChildren(ContSlider) if x.isVisibleTo(pv.right)]
    assert "S II intensity" in names and "Colours" not in names


def test_a_combination_post_cannot_make_stays_a_plain_layer_said_in_its_tooltip(m27, monkeypatch):
    w, pv, _before = m27
    real = w.s.fusion_of
    monkeypatch.setattr(w.s, "fusion_of", lambda pid, over=None, tune=False: (None, "after") if pid == 5 else
                        real(pid, over, tune))
    pv.sel = None
    pv.rebuild()
    lay = next(x for x in pv.layers if pe.is_group(x) and x.get("starless") == "output")
    assert not pe.is_fusion(lay)
    row = next(r for r in pv.rows if r.lay["id"] == lay["id"] and r.lay["kind"] == "workflow")
    assert row.toolTip() == "Blocks follow the combination in its workflow: Post shows its Output as it is"
    monkeypatch.undo()
    pv.rebuild()
    # a combination's Output shown as a plain layer (another S-H-O of the filters, as it comes): why, on its row
    out = pe.new_workflow_layer(6, "S-H-O coeur", True)
    pv.layers.insert(0, out)
    pv.rebuild()
    row = next(r for r in pv.rows if r.lay["id"] == out["id"])
    assert row.toolTip().startswith("The combination's Output as its run mixed it")
    pv.layers.remove(out)
    pv.rebuild()


def test_a_hue_drag_in_post_follows_the_slider_and_is_one_step(m27):
    w, pv, _before = m27
    lay = _master(pv)
    pv.sel = None
    pv.select(lay["id"], "layer")
    from astro_gpu.client.post3 import ContSlider
    sl = next(x for x in pv.findChildren(ContSlider) if x.name.text() == "Hα hue" and x.isVisibleTo(pv.right))
    n = len(pv.undo)
    for deg in (40.0, 50.0, 60.0):
        sl.moved.emit(deg)
        assert _master(pv)["fusion"]["hues"]["Ha"] == deg                 # the picture follows
    assert "tune" not in lay                                             # kept at the release
    sl.done.emit(60.0)
    assert len(pv.undo) == n + 1 and pv._live == {} and lay["tune"]["hues"]["Ha"] == 60.0
    pv.do_undo()


def test_a_slider_let_go_where_it_was_changes_nothing(m27):
    w, pv, _before = m27
    lay = _master(pv)
    pv.sel = None
    pv.select(lay["id"], "layer")
    from astro_gpu.client.post3 import ContSlider
    sl = next(x for x in pv.findChildren(ContSlider) if x.name.text() == "S II hue" and x.isVisibleTo(pv.right))
    n = len(pv.undo)
    sl.done.emit(16.0)                                 # 15.5 shown on a rail of whole degrees: a click on it
    assert len(pv.undo) == n and "tune" not in _master(pv)


def test_french_texts(m27, monkeypatch):
    from test_combine_editor import wait

    from astro_gpu.client import i18n
    w, pv, _before = m27
    i18n.set_language("fr")
    try:
        lay = _master(pv)
        pv.sel = None
        pv.select(lay["id"], "f:SII")
        pv.select(lay["id"], "layer")
        out = pe.new_workflow_layer(6, "S-H-O coeur", True)          # a combination's plain Output layer: its tooltip
        pv.layers.insert(0, out)
        pv.rebuild()
        pv.layers.remove(out)
        q, ed = _editor(w, monkeypatch)
        ed._lang = None
        ed._enter_page(q)
        wait(ed, 120)
        ed.set_hue("OIII", 199.0)
        wait(ed, 120)
        ed._sync()
        ed._present()
        ed.reset()
        assert not i18n._missing - {""}, " | ".join(sorted(i18n._missing))
    finally:
        i18n.set_language("en")
        pv.rebuild()
