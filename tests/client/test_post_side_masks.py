"""A mask on one side of a Combine layers layer: its Stars row's, or its Starless image row's, applied to that side's
image before the two are put together (as each layer of a reference image editor has its own mask). The whole layer's mask
is the layer row's, as before. An absent ``side_masks`` is no side mask: older compositions open and save unchanged."""
from __future__ import annotations

import copy
import json
import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

from astro_gpu import api  # noqa: E402
from astro_gpu.client import postengine as pe  # noqa: E402
from astro_gpu.client import prefs  # noqa: E402
from astro_gpu.client.postengine import Canvas, Composer, Provider  # noqa: E402

H, W = 40, 56


class Images(Provider):
    def __init__(self, parts):
        self.parts = parts

    def available(self, sid):
        return isinstance(sid, tuple) and sid[0] in ("stars", "synth", "starless", "combined", "workflow")

    def canvas(self, pids):
        return Canvas(W, H)

    def layer(self, lay, level, box, draft=False, stop=None):
        x0, y0, x1, y1 = box
        return self.parts[lay["part"]][y0:y1, x0:x1].copy()


def parts():
    rng = np.random.default_rng(3)
    return {"starless": rng.uniform(0.1, 0.6, (H, W, 3)).astype(np.float32),
            "stars": (rng.uniform(0.05, 1, (H, W, 3)) ** 3).astype(np.float32)}


def group(**kw):
    g = pe.new_workflow_layer(1, "Full · Combine layers", True, part="combined")
    g.update(kw)
    return g


def compose(layers, want=()):
    return Composer(Images(parts())).compose(pe.normalize(copy.deepcopy(layers)), 0, want=want)


def screen(a, b):
    t = 1.0 - np.clip(b, 0, 1)
    t *= 1.0 - a
    return 1.0 - t


def side(painted_disc=True):
    """A side mask whose one brush paints (adds) a disc: the side shows there only."""
    m = pe.new_mask()
    part = pe.new_part("brush")
    part["strokes"] = [(14.0, 20.0, 10.0, 100.0, True)]
    m["parts"] = [part]
    return m


def test_a_stars_side_mask_changes_only_the_stars_contribution_where_it_is_masked():
    p = parts()
    base = compose([group()]).rgb
    assert np.array_equal(base, np.clip(screen(p["starless"], p["stars"]), 0, 1))
    g = group(side_masks={"stars": side()})
    got = compose([g], want=[(g["id"], "stars")])
    m = got.masks[(g["id"], "stars")]
    assert 0 < m.min() + 1e-9 or m.min() == 0.0
    assert m.min() == 0.0 and m.max() == 1.0
    assert np.array_equal(got.rgb, np.clip(screen(p["starless"], p["stars"] * m[..., None]), 0, 1))
    out = m == 0                                       # outside the disc: the starless image alone
    assert np.allclose(got.rgb[out], np.clip(p["starless"], 0, 1)[out], atol=1e-6)
    inside = m == 1                                    # inside: untouched
    assert np.array_equal(got.rgb[inside], base[inside])


def test_a_starless_side_mask_changes_only_the_background():
    p = parts()
    base = compose([group()]).rgb
    g = group(side_masks={"starless": side()})
    got = compose([g], want=[(g["id"], "starless")])
    m = got.masks[(g["id"], "starless")]
    assert np.array_equal(got.rgb, np.clip(screen(p["starless"] * m[..., None], p["stars"]), 0, 1))
    out = m == 0
    assert np.allclose(got.rgb[out], np.clip(p["stars"], 0, 1)[out], atol=1e-6)          # the stars alone
    assert np.array_equal(got.rgb[m == 1], base[m == 1])


def test_an_inverted_feathered_side_mask_and_the_whole_layer_mask_stay_apart():
    p = parts()
    sm = side()
    sm["invert"], sm["feather"], sm["density"] = True, 3.0, 80.0
    g = group(side_masks={"stars": sm}, mask=pe.new_mask())
    g["mask"]["invert"] = False                           # a whole-layer mask that shows everywhere
    got = compose([g], want=[(g["id"], "stars"), g["id"]])
    m = got.masks[(g["id"], "stars")]
    assert 0 < m.min() < 1 or m.min() == 0.0
    assert np.array_equal(got.masks[g["id"]], np.ones((H, W), np.float32))
    assert np.array_equal(got.rgb, np.clip(screen(p["starless"], p["stars"] * m[..., None]), 0, 1))


def test_a_side_mask_follows_that_sides_filters_and_a_hidden_side_still_gives_its_mask():
    p = parts()
    g = group(side_masks={"stars": side()})
    lv = pe.new_filter_layer("levels", g["id"], "stars")
    lv["params"].update(white=0.5)
    got = compose([lv, g], want=[(g["id"], "stars")])
    m = got.masks[(g["id"], "stars")][..., None]
    x = pe.FILTERS["levels"]["fn"](p["stars"].copy(), lv["params"], None)
    assert np.array_equal(got.rgb, np.clip(screen(p["starless"], x * m), 0, 1))
    g["shown"]["stars"] = False
    got = compose([lv, g], want=[(g["id"], "stars")])
    assert np.array_equal(got.rgb, np.clip(p["starless"], 0, 1)) and (g["id"], "stars") in got.masks


def test_the_row_carries_its_side_mask_and_a_layer_without_one_has_none():
    g = group()
    assert pe.sub_row(g, "stars")["mask"] is None and pe.sub_row(g, "starless")["mask"] is None
    g["side_masks"] = {"starless": side()}
    assert pe.sub_row(g, "stars")["mask"] is None and pe.sub_row(g, "starless")["mask"] is g["side_masks"]["starless"]


def test_save_and_reopen_keep_side_masks_and_an_old_composition_is_unchanged():
    g = group()
    saved = pe.save_layers([g])
    assert "side_masks" not in saved[0]                                  # nothing written when there is none
    old = json.loads(json.dumps(saved))
    again = pe.save_layers(pe.load_layers(copy.deepcopy(old), {1}))
    assert json.dumps(again, sort_keys=True) == json.dumps(old, sort_keys=True)
    g["side_masks"] = {"stars": side()}
    g["side_masks"]["stars"]["feather"] = 5.0
    data = json.loads(json.dumps(pe.save_layers([g])))
    assert list(data[0]["side_masks"]) == ["stars"]
    back = pe.load_layers(copy.deepcopy(data), {1})
    assert back[0]["side_masks"]["stars"]["feather"] == 5.0
    assert json.dumps(pe.save_layers(back), sort_keys=True) == json.dumps(data, sort_keys=True)
    assert np.array_equal(compose([g]).rgb, compose(back).rgb)
    junk = copy.deepcopy(data)
    junk[0]["side_masks"] = {"stars": "x", "other": side(), "starless": None}
    assert "side_masks" not in pe.load_layers(junk, {1})[0]


def test_a_copy_of_the_layer_has_its_own_side_mask_and_part_ids():
    g = group(side_masks={"stars": side()})
    snap = pe.snapshot([g])[0]
    snap["side_masks"]["stars"]["parts"][0]["strokes"].append((1.0, 1.0, 2.0, 50.0, True))
    assert len(g["side_masks"]["stars"]["parts"][0]["strokes"]) == 1
    (c,) = pe.with_new_ids([g])
    assert c["side_masks"]["stars"]["parts"][0]["id"] != g["side_masks"]["stars"]["parts"][0]["id"]


# ---------------------------------------------------------------------------------------------- the page, on a project
@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture()
def project(qapp, tmp_path, monkeypatch):
    from astro_gpu.client.model import ProjectModel
    from astro_gpu.client.services.worker import ApiWorker
    from astro_gpu.core import project2
    from astro_gpu.core.project import Target
    monkeypatch.setenv("ASTRO_GPU_SETTINGS", str(tmp_path / "settings.json"))
    prefs.load()
    root = tmp_path / "P"
    p = project2.create(root, "P", Target(name="M 1"))
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "a1.ARW").write_bytes(b"\0" * 100)
    p.sessions.append(project2.Session(1, "night 1", lights=[str(raw)]))
    api.add_workflow(p, "Full", "Full")
    w = ApiWorker()
    m = ProjectModel(p, w, readonly=True)
    assert w.wait(120) and m.ready, m.error
    m.add_post_layer(m.paths[0].pid, "combined", keep_default=False)
    m._undo.clear()
    yield m
    w.stop()


def page_of(m):
    from astro_gpu.client.post3 import PostPage3
    from astro_gpu.client.services.render import RenderWorker
    return PostPage3(m, worker=RenderWorker(None, resolver=lambda pid: None, device="cpu"))


def test_the_mask_button_acts_on_the_chosen_row_per_version_with_undo_and_save(project):
    m = project
    page = page_of(m)
    v = page.view
    try:
        g = next(x for x in v.layers if pe.is_group(x))
        # the layer row: the whole layer's mask, as before
        v.select(g["id"], "layer")
        v.mask_button()
        assert g["mask"] is not None and "side_masks" not in g and v.sel == (g["id"], "mask")
        v.do_undo()
        g = next(x for x in v.layers if pe.is_group(x))
        assert g["mask"] is None
        # the Stars row: a mask of the stars side only
        v.select(g["id"], "stars")
        assert v.b_maskadd.isEnabled()
        v.mask_button()
        g = next(x for x in v.layers if pe.is_group(x))
        assert g["mask"] is None and set(g["side_masks"]) == {"stars"} and v.sel == (g["id"], "mask_stars")
        row = next(r for r in v.rows if r.lay.get("sub") == "stars")
        assert row.m is not None and next(r for r in v.rows if r.lay.get("sub") == "starless").m is None
        part = v.add_part(v.mask_owner(), "luminance")
        assert g["side_masks"]["stars"]["parts"][-1]["id"] == part["id"]
        # the Starless row: its own, apart
        v.select(g["id"], "starless")
        v.mask_button()
        assert set(g["side_masks"]) == {"stars", "starless"} and v.sel == (g["id"], "mask_starless")
        assert g["side_masks"]["starless"]["parts"] == [] and len(g["side_masks"]["stars"]["parts"]) == 1
        # undo takes the starless mask away, the stars one stays
        v.do_undo()
        g = next(x for x in v.layers if pe.is_group(x))
        assert set(g["side_masks"]) == {"stars"} and v.sel == (g["id"], "starless")
        v.do_redo()
        g = next(x for x in v.layers if pe.is_group(x))
        assert set(g["side_masks"]) == {"stars", "starless"}
        # saved into the active version, and the other version has its own
        v._save()
        saved = m.post_version()["layers"]
        assert set(next(x for x in saved if x.get("part") == "combined")["side_masks"]) == {"stars", "starless"}
        # deleting a side mask leaves the other, then none: the key goes with the last
        v.select(g["id"], "mask_starless")
        v.remove_mask(v.mask_owner())
        g = next(x for x in v.layers if pe.is_group(x))
        assert set(g["side_masks"]) == {"stars"} and v.sel == (g["id"], "starless")
        v.select(g["id"], "mask_stars")
        v.remove_mask(v.mask_owner())
        g = next(x for x in v.layers if pe.is_group(x))
        assert "side_masks" not in g
        # the real stars row has no mask of its own
        v.select(g["id"], "real")
        assert not v.b_maskadd.isEnabled()
    finally:
        page.shutdown()
        page.deleteLater()


def test_each_version_has_its_own_side_masks(project):
    m = project
    layers = copy.deepcopy(m.post_version()["layers"])
    g = next(x for x in layers if x.get("part") == "combined")
    g["side_masks"] = {"stars": side()}
    m.set_post_layers(layers, scope="active")
    first = m.post_version()["id"]
    m.new_post_version()                                                # a copy, then its own
    assert m.post_version()["id"] != first
    mine = next(x for x in m.post_version()["layers"] if x.get("part") == "combined")
    assert list(mine["side_masks"]) == ["stars"]
    layers = copy.deepcopy(m.post_version()["layers"])
    g = next(x for x in layers if x.get("part") == "combined")
    g["side_masks"] = {"starless": side()}
    m.set_post_layers(layers, scope="active")
    now = {v["id"]: next(x for x in v["layers"] if x.get("part") == "combined").get("side_masks") for v in
           m.p.post["versions"]}
    assert list(now[first]) == ["stars"] and list(now[m.post_version()["id"]]) == ["starless"]


def test_adding_a_mask_part_computes_only_that_part_and_the_pixels_are_the_same():
    """Each part's plane is kept by its settings and the image it reads: the n-th add draws one plane, not n."""
    g = group(mask=pe.new_mask())
    kinds = ["color", "luminance", "gradient", "color"]
    for i, k in enumerate(kinds):
        p = pe.new_part(k)
        if k == "color":
            p["hue"] = 40.0 * i
        g["mask"]["parts"].append(p)
    comp = Composer(Images(parts()))
    drawn = []
    for n in range(1, len(kinds) + 1):
        lay = copy.deepcopy(g)
        lay["mask"]["parts"] = lay["mask"]["parts"][:n]
        layers = pe.normalize([lay])
        before = comp.part_draws
        got = comp.compose(layers, 0, want=[layers[0]["id"]])
        drawn.append(comp.part_draws - before)
        fresh = Composer(Images(parts())).compose(copy.deepcopy(layers), 0, want=[layers[0]["id"]])
        assert np.array_equal(got.masks[layers[0]["id"]], fresh.masks[layers[0]["id"]])
        assert np.array_equal(got.rgb, fresh.rgb)
    assert drawn == [1, 1, 1, 1]
