"""A combination's master layer on the owner's M27 (S-H-O). The test opens a copy of the project's definition alone
(no link to its store); the results are read where they are, read only, by the paths given below: the definition is
newer than them (its keys moved since they were computed), so its layers and its editor are given its S-H-O's
results by hand.

The master layer: the project opens with its Combine layers layer as the master layer, its filters as rows, its picture
made from the filters; its hues and intensities are the layer's own in Post and the graph's settings in Inspect and the
block's panel (two sources, bridged by a copy: client/postfusion.py), never outdate the block nor ask for a run; the
balance stays the block's. The editor balances as the
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
def test_a_lightness_recipe_stays_plain_by_its_recipe_and_the_graph_says_the_rest():
    """The recipe's own reason is postfusion's; what the graph puts around a combination (a block after it, its depth) is
    the layer tree's (tests/graphcore/test_mix_tree.py), said on the node's row as a note (below)."""
    assert postfusion.plain_reason(recipes.legacy_spec("sho")) == ""
    assert postfusion.plain_reason(recipes.legacy_spec("lrgb")) == "lightness"
    assert postfusion.plain_reason(recipes.legacy_spec("rgb")) == ""


def test_every_reason_a_mix_is_fixed_has_its_note_and_its_tooltip_in_both_languages():
    import inspect
    import re

    from astro_gpu.client import i18n
    from astro_gpu.client.post3 import FIXED_NOTE, FIXED_TIPS
    from astro_gpu.graphcore import post as gp
    said = set(re.findall(r'return "(\w+)"', inspect.getsource(postfusion.plain_reason))) |         set(re.findall(r'"why": "(\w+)"', inspect.getsource(gp.mix_tree)))
    assert said == {"lightness", "after", "depth"} and said <= set(FIXED_TIPS)
    try:
        i18n.set_language("fr")
        assert all(i18n._(t) != t for t in (*FIXED_TIPS.values(), FIXED_NOTE))
        assert i18n._(FIXED_NOTE) == "Mélange figé"
    finally:
        i18n.set_language("en")


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
    mp.setattr(ce, "_applied_for", lambda project, pid, stage, k: ce.applied_small(str(COMBINATION), k))
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
    pv._attach_tree()
    return next(x for x in pv.layers if pe.is_group(x) and x.get("starless") == "output")


def _settle(w):
    from test_inspect import settle
    settle(w)


def _render(pv, level=2):
    from astro_gpu.client.services.render import Request
    pv._attach_tree()
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
    stored = {k: v for k, v in stored.items() if k != "sky_scale"}    # (a key of before, withdrawn: read past)
    got = next(x for x in pe.save_layers(pv.layers) if x["id"] == lay["id"])
    if all(t is None for t in (got.get("tunes") or {}).values()):         # (a nested mix's copy of the graph's
        got.pop("tunes", None)                                              # settings, none of its own: said at the next save)
    assert got == stored
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
    applied = ed.applied()
    blk, state, key = _block(w, q.pid, st), s.state(q, st), s.stage_key(q, st)
    n, n_post = len(s._undo), len(pv.undo)
    pv.set_fusion(lay, {"hue_oiii": 180.0}, "O III hue")
    _write(pv)
    _settle(w)
    assert _block(w, q.pid, st) == blk and s.state(q, st) == state and s.stage_key(q, st) == key   # the block as it was
    assert len(s._undo) == n and len(pv.undo) == n_post + 1                                         # a step of Post
    lay = _master(pv)
    assert lay["tune"]["hues"]["OIII"] == 180.0 and lay["fusion"]["hues"]["OIII"] == 180.0
    # Inspect's are the graph's settings: untouched, its draft kept (nothing goes from Post to the editor)
    assert ed.has_draft() and ed.applied() == applied and s.fusion_graph(q.pid) is None
    assert pv.next_words("undo") == "O III hue of “S-H-O”"                # the setting, then its owner
    pv.do_undo()
    _write(pv)
    assert "tune" not in _master(pv) and ed.applied() == applied
    ed.reset()


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


def test_inspect_s_apply_of_a_hue_sets_the_graph_s_settings_and_leaves_post_s_layer_as_it_is(m27, monkeypatch):
    """Inspect's Apply of a hue: the graph's settings of the block (one step of the window's Undo, nothing outdated,
    nothing to run); Post's layer, made before, keeps its own (an existing project's picture unchanged); a layer made
    now takes a copy; Post's Graph settings copies them into the layer, one step of Post's Undo."""
    from astro_gpu.client import i18n
    w, pv, _before = m27
    s = w.s
    q, ed = _editor(w, monkeypatch)
    st = ed._stage()
    blk, key, state = _block(w, q.pid, st), s.stage_key(q, st), s.state(q, st)
    before = copy.deepcopy(s.p.post)
    was = copy.deepcopy(_master(pv))
    ed.set_hue("OIII", 190.0)
    assert ed.cost_text() == "Graph settings: nothing is computed again · Post's layers keep theirs"
    i18n.set_language("fr")                                          # (said in the language shown, never kept from before)
    try:
        assert ed.cost_text().startswith("Réglages du graphe : rien n'est recalculé")
    finally:
        i18n.set_language("en")
    n = len(s._undo)
    ed.apply()
    _settle(w)
    assert len(s._undo) == n + 1 and _block(w, q.pid, st) == blk and s.stage_key(q, st) == key
    assert s.state(q, st) == state
    assert s.fusion_graph(q.pid)["hues"]["OIII"] == 190.0 and ed.applied()["hue_oiii"] == 190.0
    lay = _master(pv)
    assert pe.save_layers([lay]) == pe.save_layers([was]) and lay["fusion"]["hues"] == was["fusion"]["hues"]
    assert s.p.post == before                                        # the project's Post as it was saved
    made = pe.new_workflow_layer(q.pid, "S-H-O · Combine layers 2", True, "combined")
    made["starless"] = "output"
    s.with_graph_copy(made)                                          # a layer made now: the copy
    assert made["tune"]["hues"]["OIII"] == 190.0
    m = len(pv.undo)
    assert pv.graph_settings(lay["id"])                              # Graph settings: the copy, one step of Post
    lay = _master(pv)
    assert lay["tune"]["hues"]["OIII"] == 190.0 and lay["fusion"]["hues"]["OIII"] == 190.0 and len(pv.undo) == m + 1
    assert pv.next_words("undo") == "Graph settings on “S-H-O · Combine layers”"
    pv.do_undo()
    assert pe.save_layers([_master(pv)]) == pe.save_layers([was]) and len(pv.undo) == m
    s.undo()
    _settle(w)
    assert s.fusion_graph(q.pid) is None and ed.applied()["hue_oiii"] == was["fusion"]["hues"]["OIII"]


def test_the_panel_s_route_an_intensity_goes_to_the_graph_s_settings(m27):
    w, pv, _before = m27
    s = w.s
    lay = _master(pv)
    st = lay["fusion"]["stage"]
    q = s.view(lay["pid"])
    blk = _block(w, q.pid, st)
    s.apply_as({"kind": "combine", "values": {"weights": {"Ha": 0.8}}, "what": "", "short": ""}, "replace", [q])
    _settle(w)
    assert _block(w, q.pid, st) == blk and s.fusion_graph(q.pid)["weights"]["Ha"] == 0.8
    assert "tune" not in _master(pv)
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
    # the page: the draft bar at the top of the settings (a draft only), the panel at the right with the measurements, the image's tag
    bar, card = ed._ui["bar"], ed._ui["hue_OIII"]
    assert bar.mapTo(ed.page, bar.rect().topLeft()).y() < card.mapTo(ed.page, card.rect().topLeft()).y()
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


def test_a_combination_post_cannot_make_again_is_said_on_its_row_naming_the_block_that_fixes_it(m27, monkeypatch):
    from astro_gpu.client import i18n
    from astro_gpu.client.post3 import FIXED_NOTE
    w, pv, _before = m27
    real = w.s.post_tree
    for why, by in (("after", "Stars/background"), ("lightness", None), ("depth", "H-O")):
        def fixed(lay, live=None, why=why, by=by):
            tree = real(lay, live) if live else real(lay)
            if tree is not None and tree["kind"] == "layers":
                tree["sides"]["starless"] = {"kind": "fixed", "block": 1, "combination": "S-H-O", "by": by, "why": why}
            return tree
        monkeypatch.setattr(w.s, "post_tree", fixed)
        pv.sel = None
        pv.rebuild()
        lay = next(x for x in pv.layers if pe.is_group(x) and x.get("starless") == "output")
        assert not pe.is_fusion(lay)
        row = next(r for r in pv.rows if r.lay["id"] == lay["id"] and r.lay.get("sub") == "starless")
        assert row.note is not None and row.note.text() == FIXED_NOTE, why
        tip = row.note.toolTip()
        assert ("S-H-O" in tip and by in tip) if by else tip.startswith("A combination composed as lightness"), why
        try:
            i18n.set_language("fr")
            pv.rebuild()
            row = next(r for r in pv.rows if r.lay["id"] == lay["id"] and r.lay.get("sub") == "starless")
            assert row.note.text() == "Mélange figé" and (by is None or by in row.note.toolTip()), why
        finally:
            i18n.set_language("en")
    monkeypatch.undo()
    pv.rebuild()
    lay = next(x for x in pv.layers if pe.is_group(x) and x.get("starless") == "output")
    assert pe.is_fusion(lay) and all(r.note is None for r in pv.rows if r.lay["id"] == lay["id"])


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


def test_the_h_alpha_part_on_a_filter_s_mask_reads_the_combination_s_h_alpha(m27):
    w, pv, _before = m27
    lay = _master(pv)
    m = pe.new_mask()
    m["parts"] = [pe.new_part("halpha")]
    lay.setdefault("side_masks", {})["f:SII"] = m
    try:
        res = _render(pv)
        assert res.rgb is not None and not res.missing             # (it said: a file or folder is not found)
    finally:
        lay["side_masks"].pop("f:SII")
        if not lay["side_masks"]:
            lay.pop("side_masks")


def test_the_tab_stays_on_the_page_shown_while_a_draft_asks(m27, monkeypatch):
    w, pv, _before = m27
    w.set_mode(1)
    lit = []
    monkeypatch.setattr(w, "settle_drafts", lambda **k: lit.append([b.isChecked() for b in w.mode_btns]) or False)
    w.mode_btns[2].click()                                  # Post clicked, the draft's question answered Stay
    assert lit and lit[0][1] and not lit[0][2]             # while it asked: Inspect's tab lit, not Post's
    assert w.mode == 1 and w.mode_btns[1].isChecked() and not w.mode_btns[2].isChecked()


def test_the_balance_says_it_needs_a_run_in_post_and_in_the_editor(m27, monkeypatch):
    w, pv, _before = m27
    lay = _master(pv)
    from PySide6.QtWidgets import QLabel

    real_state = w.s.state

    def said(ran, state):
        monkeypatch.setattr(w.s, "combine_result_balance", lambda pid: ran)
        monkeypatch.setattr(w.s, "state", lambda q, st, *a, **k: state if st == lay["fusion"]["stage"] else
                            real_state(q, st, *a, **k))
        pv.sel = None
        pv.select(lay["id"], "layer")
        return [x.text() for x in pv.right.findChildren(QLabel) if x.isVisibleTo(pv.right)]
    other = "sky" if lay["fusion"]["balance"] == "signal" else "signal"
    # the balance computed: nothing said; the engine counting the block up to date (a result of an earlier version
    # computed under the former default, IC 443's): nothing said either; the block Outdated with another balance than
    # its result's: it needs a run
    assert not any(t.startswith("Needs a run") for t in said(lay["fusion"]["balance"], "stale"))
    assert not any(t.startswith("Needs a run") for t in said(other, "done"))
    assert any(t.startswith("Needs a run") for t in said(other, "stale"))
    monkeypatch.setattr(w.s, "state", real_state)
    q, ed = _editor(w, monkeypatch)
    ed.set_draft({"balance": "sky" if ed.applied().get("balance") == "signal" else "signal"})
    c = ed.cost_text()
    assert c.startswith(("Needs a run", "Computes again", "To compute")) and "nothing is computed" not in c
    ed.reset()


def test_shift_drags_the_hue_rail_ten_times_finer(qapp):
    from PySide6.QtCore import QEvent, QPointF, Qt
    from PySide6.QtGui import QMouseEvent
    from PySide6.QtWidgets import QSlider
    sl = QSlider(Qt.Horizontal)
    sl.setRange(0, 360)
    sl.resize(380, 24)
    sl.setValue(100)
    fd = ce.FineDrag(sl)
    sl.setSliderDown(True)

    def move(x, mods):
        e = QMouseEvent(QEvent.MouseMove, QPointF(x, 10), QPointF(x, 10), Qt.NoButton, Qt.LeftButton, mods)
        return fd.eventFilter(sl, e)
    assert move(100.0, Qt.ShiftModifier) and sl.value() == 100
    per = fd._per_px()
    move(100.0 + 50, Qt.ShiftModifier)
    assert sl.value() == int(round(100 + 50 * per / 10))
    assert not move(200.0, Qt.NoModifier)                  # without Shift: the slider's own drag
    sl.setSliderDown(False)


def test_judges_details_names_undo_tag_digits_intensity_tabs_gains_and_the_panel_kept(m27, monkeypatch):
    from test_combine_editor import wait

    from astro_gpu.client import i18n
    from astro_gpu.client.post3 import ContSlider
    from astro_gpu.graphcore import rules
    w, pv, _before = m27
    s = w.s
    q, ed = _editor(w, monkeypatch)
    # one name: the graph's block name in the editor's title, its draft's words, the path bar and Post's panel
    name = rules.title(s._g, s._combine_block(q), tr=i18n._)
    assert ed._ui["title"].text() == name and ed.title(q) == name
    from astro_gpu.client.graphnode import node_title
    assert node_title(s, ed.node) == name
    lay = _master(pv)
    assert pv._fusion_title(lay) == name
    # a hue dragged then let go: the draft's Undo in the panel at once
    ed.begin_drag()
    ed.set_hue("OIII", 150.0, undo=False)
    ed.end_drag()
    assert ed._ui["undo"].isEnabled()
    wait(ed, 120)
    # the single image's tag, as the Gradient's
    ed.set_compare("draft")
    ed._present()
    assert ed.shown_tag.startswith("Draft preview · reduced copy")
    ed.reset()
    # numbers the interface's way
    i18n.set_language("fr")
    try:
        assert ce._digits(0.0468, 3) == "0,0468"
    finally:
        i18n.set_language("en")
    # Intensity up to 10, in Post and Inspect, the same step
    assert ce.INTENSITY_MAX == 10.0 and ed._ui["int_SII"].maximum() == ce.INTENSITY_POSITIONS
    pv.sel = None
    pv.select(lay["id"], "layer")
    sl = next(x for x in pv.findChildren(ContSlider) if x.name.text() == "S II intensity" and x.isVisibleTo(pv.right))
    assert sl.hi == 10.0
    # the page tabs never take the arrow keys
    from PySide6.QtCore import Qt
    assert all(b.focusPolicy() == Qt.NoFocus for b in w.mode_btns)
    # a hue let go: the panel stays (the same slider under the pointer, not built again)
    hue = next(x for x in pv.findChildren(ContSlider) if x.name.text() == "O III hue" and x.isVisibleTo(pv.right))
    hue.moved.emit(120.0)
    hue.done.emit(120.0)
    for _i in range(5):
        QApplication.processEvents()
    assert hue.isVisibleTo(pv.right)
    pv.do_undo()
    # one gain figure: Post's filter panel and the editor's, the one the picture uses
    _render(pv)
    got = pv.worker.renderer.source.fusion_gains(_master(pv))
    assert {r: round(g, 6) for r, (g, _w) in ed.notes.items()} == {r: round(g, 6) for r, (g, _w) in got.items()}
    from astro_gpu.client import model as mmod

    class Out:                                       # (this copy has no store: its result, by hand)
        dir = str(COMBINATION)
        params = json.loads((COMBINATION / "manifest.json").read_text(encoding="utf-8"))["params"]
    monkeypatch.setattr(mmod.api, "stage_outputs", lambda *a, **k: Out)
    panel = w.s.fusion_gains(lay["pid"])
    assert {r: round(g, 6) for r, (g, _w, _f) in panel.items()} == {r: round(g, 6) for r, (g, _w) in got.items()}


def test_the_palette_field_follows_a_hue_let_go_in_place(m27):
    w, pv, _before = m27
    lay = _master(pv)
    pv.sel = None
    pv.select(lay["id"], "layer")
    from astro_gpu.client.post3 import ContSlider
    box = pv.palette_box[1]
    assert box.currentData() == lay["fusion"]["palette"]
    hue = next(x for x in pv.findChildren(ContSlider) if x.name.text() == "O III hue" and x.isVisibleTo(pv.right))
    hue.moved.emit(120.0)
    hue.done.emit(120.0)
    for _i in range(5):
        QApplication.processEvents()
    assert pv.palette_box[1] is box and box.currentData() == "" and box.currentText() == "Custom"
    pv.do_undo()


def test_an_intensity_let_go_where_it_was_when_the_panel_was_built_still_changes_the_block(m27):
    """The tester: S II intensity moved 1.0 -> 1.5 by the arrows, then a double-click back to its default: the slider
    showed 1, the combination kept 1.5, the picture did not change. A setting let go here leaves the panel as it is: the
    value it is compared with is the combination's now, not the one when the panel was built."""
    from astro_gpu.client.post3 import ContSlider
    w, pv, _before = m27
    lay = _master(pv)
    pv.sel = None
    pv.select(lay["id"], "layer")
    sl = next(x for x in pv.findChildren(ContSlider) if x.name.text() == "S II intensity" and x.isVisibleTo(pv.right))
    was = float(lay["fusion"]["weights"]["SII"])
    n = len(pv.undo)
    sl.set_value(was + 0.5)
    sl.moved.emit(was + 0.5)
    sl.done.emit(was + 0.5)
    lay = _master(pv)
    assert lay["fusion"]["weights"]["SII"] == pytest.approx(was + 0.5) and len(pv.undo) == n + 1
    assert any(x is sl for x in pv.findChildren(ContSlider))           # the panel kept (the slider under the pointer)
    sl.set_value(was)                                                  # back where it was when the panel was built
    sl.moved.emit(was)
    sl.done.emit(was)
    lay = _master(pv)
    assert lay["fusion"]["weights"]["SII"] == pytest.approx(was) and len(pv.undo) == n + 2
    pv.do_undo()
    pv.do_undo()
    assert "tune" not in _master(pv)


def test_the_intensities_are_set_to_the_hundredth_and_shift_drags_ten_times_finer(m27, monkeypatch):
    """The testers' look needed intensities finer than 0.05 near 1: one hundredth a step, in Post, in Inspect's editor
    and in the block's panel, and Shift while dragging ten times finer, as the hue rails."""
    from astro_gpu.client.combineedit import FineDrag
    from astro_gpu.client.post3 import ContSlider
    w, pv, _before = m27
    assert ce.INTENSITY_STEP == 0.01
    lay = _master(pv)
    pv.sel = None
    pv.select(lay["id"], "layer")
    for name in ("S II intensity", "S II hue"):
        sl = next(x for x in pv.findChildren(ContSlider) if x.name.text() == name and x.isVisibleTo(pv.right))
        assert sl.s.findChildren(FineDrag), name
    sl = next(x for x in pv.findChildren(ContSlider) if x.name.text() == "S II intensity" and x.isVisibleTo(pv.right))
    assert sl.law is not None and sl.s.maximum() == ce.INTENSITY_POSITIONS
    # the rail even by ratio (the brush size's law): 1 at its middle, 0 at its left end, 10 at its right; a tenth of
    # the rail moves the value by the same ratio anywhere, about 1.6 (a linear rail took 28 px from 1.00 to 2.95)
    assert ce.intensity_at(0.0) == 0.0 and ce.intensity_at(0.5) == 1.0 and ce.intensity_at(1.0) == 10.0
    assert ce.intensity_at(0.6) == pytest.approx(1.58, abs=0.01) and ce.intensity_at(0.4) == pytest.approx(0.63, abs=0.01)
    assert all(ce.intensity_at(ce.intensity_u(w)) == w for w in (0.0, 0.25, 1.0, 1.37, 4.2, 10.0))
    _q, ed = _editor(w, monkeypatch)
    it = ed._ui["int_SII"]
    assert it.maximum() == ce.INTENSITY_POSITIONS and it.findChildren(FineDrag)


def test_the_master_layer_s_saturation_goes_on_to_plus_four(m27):
    """The testers' look needed a Saturation filter on top of the layer's +2: its slider goes on to +4 (render/sliders:
    Saturation's own range), the same law to +2."""
    from astro_gpu.client.post3 import ContSlider
    w, pv, _before = m27
    lay = _master(pv)
    pv.sel = None
    pv.select(lay["id"], "layer")
    sl = next(x for x in pv.findChildren(ContSlider) if x.name.text() == "Saturation" and x.isVisibleTo(pv.right))
    assert sl.hi == 6
    sl.set_value(6)
    assert sl.val.text() == "+4.0"
    other = next(x for x in pv.findChildren(ContSlider) if x.name.text() == "Contrast" and x.isVisibleTo(pv.right))
    assert other.hi == 4


def test_a_change_in_post_computes_nothing_in_a_hidden_inspect_editor(m27, monkeypatch):
    """A release in Post was followed by a second render beside Post's: the combination's editor, its page built but
    not shown, made its preview again on every change of the master layer (the same processor). The editor follows
    nothing of Post now (the graph's settings, the block alone): a change in Post asks nothing of it; a grey filter's
    picture is made by the faster merge, the same values."""
    import numpy as np

    from astro_gpu.client.postengine import grey3
    w, pv, _before = m27
    q, ed = _editor(w, monkeypatch)
    ed.center(w.tune, None, None)                                     # another block: the page is not shown
    assert ed._page_ok() and not ed.page_shown()
    n = ed.computed
    lay = _master(pv)
    pv.set_fusion(lay, {"hue_oiii": 170.0}, "O III hue")
    pump_until = time.monotonic() + 2
    while time.monotonic() < pump_until:
        QApplication.processEvents()
    assert ed.computed == n
    pv.do_undo()
    x = np.random.default_rng(0).random((7, 9)).astype(np.float32)
    assert np.array_equal(grey3(x), np.repeat(x[..., None], 3, -1)) and grey3(x).flags["C_CONTIGUOUS"]


def test_the_tester_s_details_of_the_family_s_sliders(m27, monkeypatch):
    """Double-click puts the combination's Intensity and Hue back to their default in Inspect, as every slider; a
    slider without a step has its default as one of its positions (Saturation's 0 drew as a moved value on its 0-6
    range); Post's own step of a setting says what it became, as Inspect's Apply does."""
    from PySide6.QtCore import QEvent, QPointF, Qt
    from PySide6.QtGui import QMouseEvent

    from astro_gpu.client.post3 import ContSlider, step_words
    w, pv, _before = m27
    q, ed = _editor(w, monkeypatch)
    ed.set_intensity("SII", 2.0)
    sl = ed._ui["int_SII"]
    ev = QMouseEvent(QEvent.MouseButtonDblClick, QPointF(5, 5), QPointF(5, 5), Qt.LeftButton, Qt.LeftButton,
                     Qt.NoModifier)
    QApplication.sendEvent(sl, ev)
    assert ed.weight("SII") == 1.0
    sat = ContSlider("Saturation", 0, 6, 2.0, 2)
    assert sat.value() == 2.0 and sat.val.styleSheet() == ""
    lay = _master(pv)
    after = copy.deepcopy(lay)
    after["sliders"] = {**lay.get("sliders", {}), "saturation": 3.0}
    assert step_words([lay], [after]).endswith(": 0 → +1.0")
    ed.reset()


def test_the_bar_s_spot_healing_acts_on_the_selected_layer_of_either_block(m27):
    """Spot healing from Post's bar acts on the layer chosen, as Filter > Detail > Spot healing: an image of the stars
    and background block (Stars, Starless) or a filter of the combination gets its own healer, found again when it is
    chosen once more; another's healer is never used (the judge: with Stars chosen the spot went into S II's healer,
    with O III chosen the button opened the Stars healer)."""
    w, pv, _before = m27
    lay = _master(pv)
    keep = pe.snapshot(pv.layers)
    try:
        sides = ["stars", "starless", "f:SII", "f:OIII"]
        made = {}
        for side in sides:                                  # each child: a healer of its own, nested in it
            pv.b_heal.setChecked(False)
            pv.select(lay["id"], side)
            pv.b_heal.setChecked(True)
            h = pv._heal_layer()
            assert h is not None and (h.get("target"), h.get("sub")) == (lay["id"], side), side
            h["params"]["spots"] = [[10.0, 10.0, 5.0, 12.0, 0.0]]      # (not empty: kept when the tool is left)
            made[side] = h["id"]
        assert len(set(made.values())) == len(sides)
        for side in reversed(sides):                        # chosen again: its own healer, no new one
            pv.b_heal.setChecked(False)
            pv.select(lay["id"], side)
            pv.b_heal.setChecked(True)
            assert pv._heal_layer()["id"] == made[side], side
        assert sum(1 for x in pv.layers if x.get("filter") == "heal") == len(sides)
    finally:
        pv.b_heal.setChecked(False)
        pv.layers = pe.normalize(keep)
        pv.sel = None
        pv.rebuild()


def test_a_healer_s_spots_are_counted_in_undo_never_their_coordinates():
    """Undo / Redo of Spot healing says "1 spot added", "2 spots removed", "1 spot moved", never the coordinates of
    the spots; in French too."""
    from astro_gpu.client import i18n
    from astro_gpu.client.post3 import step_words
    h = pe.new_filter_layer("heal", None, None)
    one, two = [491.2, 1359.5, 40.0, 100.0, 0.0], [300.0, 200.0, 40.0, -100.0, 0.0]

    def said(a, b):
        x, y = copy.deepcopy(h), copy.deepcopy(h)
        x["params"]["spots"], y["params"]["spots"] = a, b
        return step_words([x], [y])
    assert said([], [one]).endswith(": 1 spot added")
    assert said([one], [one, two, two]).endswith(": 2 spots added")
    assert said([one, two], []).endswith(": 2 spots removed") and said([one], []).endswith(": 1 spot removed")
    assert said([one], [two]).endswith(": 1 spot moved")
    assert "[" not in said([], [one]) and "491" not in said([], [one])
    try:
        i18n.set_language("fr")
        assert said([], [one]).endswith("1 tache ajoutée") and said([one, two], []).endswith("2 taches retirées")
    finally:
        i18n.set_language("en")


def test_combine_layers_inspect_shows_the_block_alone_and_its_apply_leaves_post_s_layer(m27):
    """Combine layers' editor shows the block alone (its two images at its graph settings: no filter, mask or hidden
    side of Post's layer, whatever Post holds) and Apply sets the graph's settings: Post's layer keeps its own."""
    w, pv, _before = m27
    s = w.s
    lay = _master(pv)
    before = copy.deepcopy(s.p.post)
    bid = next(k for k, b in s._g.blocks.items() if b.kind == "combine_layers")
    s.sel_layer = lay["id"]
    s.select(index=s.index_of(lay["pid"]), step="composition")
    w.set_mode(1)
    _settle(w)
    ed = w.tune.layers_editor
    assert ed.page_shown() and ed.bid == bid
    one = ed._request()["layers"]
    assert len(one) == 1 and one[0]["id"] == -1 and one[0]["mask"] is None and not one[0].get("side_masks")
    assert all(one[0]["shown"].values()) and one[0]["fusion"]["hues"] == s.fusion_of(lay["pid"])[0]["hues"]
    # Post's filter on the layer, a mask, a side hidden: the editor's preview as it was
    f = pe.new_filter_layer("curves", lay["id"], None)
    pv.layers.insert(pv.layers.index(lay), f)
    lay["mask"] = pe.new_mask()
    lay["shown"] = {**lay["shown"], "stars": False}
    try:
        assert ed._request()["layers"] == one
    finally:
        pv.layers.remove(f)
        lay["mask"] = None
        lay["shown"] = {**lay["shown"], "stars": True}
    saved = pe.save_layers([lay])
    n = len(s._undo)
    ed.set_slider("saturation", 3.0)
    ed.apply()
    _settle(w)
    assert len(s._undo) == n + 1 and s.layers_graph(bid)["sliders"] == {"saturation": 3.0}
    assert pe.save_layers([_master(pv)]) == saved and s.p.post == before
    assert ed._request()["layers"][0]["sliders"] == {"saturation": 3.0}
    s.undo()
    _settle(w)
    assert s.layers_graph(bid)["sliders"] == {}
