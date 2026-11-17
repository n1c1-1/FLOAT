"""A combination's master layer on the owner's M27 (S-H-O, read through a scratch copy of its definition, nothing
computed, nothing written into it): the project opens with its Combine layers layer as the master layer, its filters
as rows, its picture made from the filters; the settings are one source with Inspect's editor, a change made in Post
is one step of the workflows' Undo and drops a draft of the block in Inspect; the layers stay a plain Combine layers
layer when the combination cannot be made again in Post. Also the plain cases' rule (client/postfusion.py)."""
from __future__ import annotations

import copy
import json
import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication  # noqa: E402

from astro_gpu.client import postengine as pe  # noqa: E402
from astro_gpu.client import postfusion  # noqa: E402
from astro_gpu.stages import recipes  # noqa: E402

M27 = "F:/astro_essais/tutoriels_ref/MONO/m27_v10"
#: the results Post reads, read only: the project's definition is newer than them (its keys moved since they were
#: computed), so its layers are given its S-H-O's results by hand: the starless combination (balanced On signal, made
#: before the balance's maps were written) and its Output, the stars' combination's Output
RESULTS = {("fusion", 5): "store/combine_sho/d7eecb71bbbea37f", 5: "store/render_simple/d98430ad8cd1e148",
           4: "store/render_simple/fc11a390824d10e4"}


def _resolver(sid):
    from pathlib import Path

    from astro_gpu.client.services.render import Folder
    rel = RESULTS.get(sid)
    if rel is None:
        return None
    d = Path(M27) / rel
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


def test_the_description_is_the_block_s_settings_and_its_matrix():
    from astro_gpu.stages.combine import hue_palette
    f = postfusion.describe("combine_sho", recipes.legacy_spec("sho"), {"hue_oiii": 180.0, "weights": {"SII": 2.0}},
                            {"SII": "S II", "Ha": "Hα", "OIII": "O III"}, {"weights": {"Ha": 0.5}})
    assert f["roles"] == ["SII", "Ha", "OIII"] and f["hues"]["OIII"] == 180.0
    assert f["weights"] == {"SII": 2.0, "Ha": 0.5, "OIII": 1.0}
    assert f["matrix"] == [[float(v) for v in row] for row in hue_palette((15.5, 31.4, 180.0))]
    assert f["hue_defaults"] == {"SII": 15.5, "Ha": 31.4, "OIII": 202.2} and f["palette"] == ""


# ---------------------------------------------------------------------------------------------- the owner's M27
@pytest.fixture(scope="module")
def m27(qapp, tmp_path_factory):
    import test_combine_editor as T
    if not os.path.isfile(os.path.join(M27, "project.astroproj")):
        pytest.skip("the project m27_v10 is not here")
    T.REAL["m27_v10"] = M27
    tmp = tmp_path_factory.mktemp("m27")
    w = T._real_window(tmp, "m27_v10")
    from astro_gpu.client.post3 import PostView
    from astro_gpu.client.services.render import RenderWorker
    before = copy.deepcopy(w.s.p.post)
    pv = PostView(w.s, persist=False, worker=RenderWorker(w.s.p, resolver=_resolver, device="cpu"))
    yield w, pv, before
    pv.shutdown()
    w.close()


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


def test_m27_opens_with_its_combination_as_the_master_layer_and_its_definition_unchanged(m27):
    w, pv, before = m27
    lay = _master(pv)
    assert pe.is_fusion(lay) and lay["fusion"]["roles"] == ["SII", "Ha", "OIII"]
    names = [(r.get("name"), d) for r, d in pe.rows(pv.layers)]
    i = names.index(("Starless image", 1))
    assert names[i + 1:i + 4] == [("S II", 2), ("Hα", 2), ("O III", 2)]
    assert w.s.p.post == before                                 # converted in memory only
    # its layer saved is the one the project holds: nothing of the master layer is written
    stored = next(x for x in before["versions"][0]["layers"] if x.get("part") == "combined")
    saved = next(x for x in pe.save_layers(pv.layers) if x["id"] == lay["id"])
    assert saved == stored


def test_m27_s_picture_is_made_from_its_filters(m27):
    w, pv, _before = m27
    res = _render(pv)
    assert res.rgb is not None and res.rgb.std() > 3
    cp = pv.worker.renderer.composer
    lay = _master(pv)
    cv = cp.canvas_of(pv.layers)
    kids = [cp.layer_rgb(pe.sub_layer(lay, pe.role_sub(r)), 2, (0, 0, *cv.size(2)), cv) for r in ("SII", "Ha", "OIII")]
    assert all(k.std() > 0.005 for k in kids)
    # the filters' balance is the run's: the maps it wrote give its image
    src = pv.worker.renderer.source
    full = src._fusion_full((pe.FUSION, lay["pid"]), lay["fusion"]["balance"])
    out = src.folder((pe.FUSION, lay["pid"]))
    meta = json.loads((out.path / "drz_meta.json").read_text(encoding="utf-8"))
    mat = np.asarray(lay["fusion"]["matrix"], np.float32)
    w8 = lay["fusion"]["weights"]
    mix = np.tensordot(mat, np.stack([full["planes"][r][::16, ::16] * w8[r] for r in ("SII", "Ha", "OIII")]), 1)
    drz = np.load(out.path / "drz_full.npy", mmap_mode="r")[:, ::16, ::16]
    if meta.get("balance", {}).get("how") == lay["fusion"]["balance"] or "balance" not in meta:
        assert np.abs(mix - drz).max() <= 2e-3 * float(np.abs(drz).max())


def test_a_hue_set_in_post_is_the_block_s_one_step_of_undo_and_drops_inspect_s_draft(m27, monkeypatch):
    w, pv, _before = m27
    s = w.s
    lay = _master(pv)
    q = s.view(lay["pid"])
    st = lay["fusion"]["stage"]
    from test_combine_editor import _open_combination
    q2, ed = _open_combination(w)
    assert q2.pid == q.pid
    monkeypatch.setattr(ed, "has_result", lambda p=None: True)        # (its results are older than its keys here)
    ed.set_intensity("SII", 1.5)                                       # a draft in Inspect
    assert ed.has_draft()
    said = []
    w.tune.toasted.connect(said.append)
    n = len(s._undo)
    old = (s.p.workflow(q.pid).params.get(st) or {}).get("hue_oiii")
    pv.set_fusion(lay, {"hue_oiii": 180.0}, "O III hue")
    _settle(w)
    assert (s.p.workflow(q.pid).params.get(st) or {}).get("hue_oiii") == 180.0
    assert len(s._undo) == n + 1
    assert not ed.has_draft() and "Draft dropped: changed in Post" in said
    assert ed.applied()["hue_oiii"] == 180.0                         # Inspect shows the same setting
    assert _master(pv)["fusion"]["hues"]["OIII"] == 180.0
    assert pv.next_words("undo")                                       # Post lists the step (the workflows' words)
    pv.do_undo()
    _settle(w)
    assert (s.p.workflow(q.pid).params.get(st) or {}).get("hue_oiii") == old
    assert _master(pv)["fusion"]["hues"]["OIII"] == (old if old is not None else 202.2)


def test_inspect_s_apply_shows_in_post(m27, monkeypatch):
    w, pv, _before = m27
    s = w.s
    from test_combine_editor import _open_combination
    q, ed = _open_combination(w)
    monkeypatch.setattr(ed, "has_result", lambda p=None: True)
    # its preview is stretched as the master layer is: the colour-preserving stretch (Colours at 1)
    assert ed._render_settings()["keep_colours"] == 1.0
    ed.set_intensity("OIII", 0.7)
    ed.apply()
    _settle(w)
    assert _master(pv)["fusion"]["weights"]["OIII"] == 0.7
    s.undo()
    _settle(w)
    assert _master(pv)["fusion"]["weights"]["OIII"] == 1.0


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
    assert len(pv.undo) == n + 2
    off = _render(pv).rgb.astype(np.float32)
    assert np.abs(off - plain).mean() > 0.5
    pv.do_undo()
    pv.do_undo()
    assert not any(x.get("sub") == "f:OIII" for x in pv.layers if x["kind"] == "filter")


def test_the_master_panel_shows_the_combination_s_settings_and_colours_greyed(m27, monkeypatch):
    w, pv, _before = m27
    lay = _master(pv)
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
    monkeypatch.setattr(w.s, "fusion_of", lambda pid, over=None: (None, "after"))
    pv.rebuild()
    lay = next(x for x in pv.layers if pe.is_group(x) and x.get("starless") == "output")
    assert not pe.is_fusion(lay)
    row = next(r for r in pv.rows if r.lay["id"] == lay["id"] and r.lay["kind"] == "workflow")
    assert row.toolTip() == "Blocks follow the combination in its workflow: Post shows its Output as it is"
    assert [r.get("name") for r, _d in pe.rows([lay])] == [lay["name"], "Stars", "Starless image"]
    monkeypatch.undo()
    pv.rebuild()


def test_french_texts(m27):
    from astro_gpu.client import i18n
    w, pv, _before = m27
    i18n.set_language("fr")
    try:
        lay = _master(pv)
        pv.select(lay["id"], "f:SII")
        pv.select(lay["id"], "layer")
        assert not i18n._missing - {""}, sorted(i18n._missing)[:5]
    finally:
        i18n.set_language("en")


def test_a_hue_drag_in_post_follows_the_slider_and_is_one_step(m27):
    w, pv, _before = m27
    s = w.s
    lay = _master(pv)
    pv.sel = None
    pv.select(lay["id"], "layer")
    from astro_gpu.client.post3 import ContSlider
    sl = next(x for x in pv.findChildren(ContSlider) if x.name.text() == "Hα hue" and x.isVisibleTo(pv.right))
    n = len(s._undo)
    st = lay["fusion"]["stage"]
    for deg in (40.0, 50.0, 60.0):
        sl.moved.emit(deg)
        assert _master(pv)["fusion"]["hues"]["Ha"] == deg                 # the picture follows
        assert (s.p.workflow(lay["pid"]).params.get(st) or {}).get("hue_ha") != deg   # the block: at the release
    sl.done.emit(60.0)
    _settle(w)
    assert len(s._undo) == n + 1 and pv._live == {}
    assert (s.p.workflow(lay["pid"]).params.get(st) or {}).get("hue_ha") == 60.0
    s.undo()
    _settle(w)
