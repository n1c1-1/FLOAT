"""The stars of a Combine layers layer are a group (issue #93): the synthetic stars and the real stars side by side in it,
each with its eye, the group in Screen over the starless image.

The constraint the group keeps: the synthetic stars REPLACE the real ones where their mask is open (Normal in the
group), never add to them. The pixels are checked against today's formula (until #93 the synthetic stars were a filter
of the stars: ``stars + (synth - stars) * mask``, then Screen on the starless image), bit for bit; then the eyes, the one
starting state of both methods, a composition saved before the group, the panel in French, and the Synthetic stars
added and removed from Post with their block in the graph, one Undo step each."""
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
from astro_gpu.client import i18n, prefs  # noqa: E402
from astro_gpu.client import postengine as pe  # noqa: E402
from astro_gpu.client.postengine import Canvas, Composer, Provider  # noqa: E402

H, W = 40, 56


class Images(Provider):
    """A Combine layers layer's images, made up: its starless image, its stars, the Synthetic stars block's stars."""

    def __init__(self, parts: dict):
        self.parts = parts

    def available(self, sid):
        return sid[0] in ("stars", "synth") and ("synth" in self.parts or sid[0] == "stars") if isinstance(sid, tuple) \
            else False

    def canvas(self, pids):
        return Canvas(W, H)

    def layer(self, lay, level, box, draft=False, stop=None):
        x0, y0, x1, y1 = box
        return self.parts[lay["part"]][y0:y1, x0:x1].copy()


def images(seed=1, synth=True):
    rng = np.random.default_rng(seed)
    out = {"starless": rng.uniform(0, 0.6, (H, W, 3)).astype(np.float32),
           "stars": (rng.uniform(0, 1, (H, W, 3)) ** 6).astype(np.float32)}
    if synth:
        out["synth"] = (rng.uniform(0, 1, (H, W, 3)) ** 5).astype(np.float32)
    return out


def screen(a, b):
    """Composer.blend's Screen, its operations in its order: (1 - b)(1 - a), b clipped."""
    t = 1.0 - np.clip(b, 0, 1)
    t *= 1.0 - a
    return 1.0 - t


def today(parts, m):
    """The composite of a Combine layers layer as Post made it before #93: its stars through the Synthetic stars filter
    (``Composer._apply``: ``under + (out - under) * mask``, out the block's stars; a mask None: out itself), put in Screen
    on its starless image; clipped as the composite is."""
    stars, synth = parts["stars"], parts["synth"]
    s2 = synth if m is None else stars + (synth - stars) * (1.0 * m)[..., None]
    return np.clip(screen(parts["starless"], s2), 0, 1).astype(np.float32)


def group(**kw):
    g = pe.new_workflow_layer(1, "Full · Combine layers", True, part="combined")
    g.update(kw)
    return g


def compose(layers, parts, want=()):
    return Composer(Images(parts)).compose(pe.normalize(copy.deepcopy(layers)), 0, want=want)


def painted(row, strokes):
    """The synthetic stars' brush (it subtracts: painted, the real stars come back) with these strokes."""
    row["mask"]["parts"][0]["strokes"] = [(x, y, r, 60.0, True) for x, y, r in strokes]
    return row


# ---------------------------------------------------------------------------------------------- the pixels
def test_the_group_gives_the_pixels_of_today_bit_for_bit():
    """The three equalities of #93. Synthetic stars hidden: the image of a workflow without the block; their mask fully
    open, both eyes on: today's with the filter fully open; any mask: today's with that mask. A mask saved None (the
    instrument's start until #93) is today's too."""
    parts = images()
    g = group()
    row = pe.new_synth_layer(g["id"])
    # 1. hidden: byte for byte the composition without the row, and the real stars in Screen
    hidden = compose([dict(row, visible=False), g], parts).rgb
    without = compose([g], images(synth=False)).rgb
    assert np.array_equal(hidden, without)
    assert np.array_equal(hidden, np.clip(screen(parts["starless"], parts["stars"]), 0, 1))
    # 2. the mask as it starts (open): today's with that mask, every real star replaced
    got = compose([row, g], parts, want=[row["id"]])
    m = got.masks[row["id"]]
    assert m.min() == 1.0 and np.array_equal(got.rgb, today(parts, m))
    assert np.abs(got.rgb - today(parts, None)).max() < 1e-6          # (a mask of ones: at most a rounding of float32)
    # 3. a mask painted (the real stars back in two discs) and feathered: today's with that mask
    painted(row, [(12, 10, 9), (40, 30, 12)])
    row["mask"]["feather"] = 3.0
    got = compose([row, g], parts, want=[row["id"]])
    m = got.masks[row["id"]]
    assert 0 < m.min() < 1 and m.max() == 1.0
    assert np.array_equal(got.rgb, today(parts, m))
    # a row saved None (the instrument's, before #93): today's, the block's stars as they are
    assert np.array_equal(compose([dict(row, mask=None), g], parts).rgb, today(parts, None))


def test_the_rows_of_the_group_act_on_what_is_under_them_in_it():
    """A filter above the synthetic stars acts on all the stars of the group; one under them (above the real stars) on
    the real stars alone, which the synthetic ones cover where their mask is open."""
    parts = images()
    g = group()
    row = painted(pe.new_synth_layer(g["id"]), [(20, 20, 14)])
    lv = pe.new_filter_layer("levels", g["id"], "stars")
    lv["params"].update(white=0.5)
    got = compose([lv, row, g], parts, want=[row["id"]])
    m = got.masks[row["id"]][..., None]
    levels = pe.FILTERS["levels"]["fn"]
    s2 = parts["stars"] + (parts["synth"] - parts["stars"]) * (1.0 * m)
    assert np.array_equal(got.rgb, np.clip(screen(parts["starless"], levels(s2.copy(), lv["params"], None)), 0, 1))
    under = compose([row, lv, g], parts).rgb                           # under the synthetic stars: the real ones only
    real = levels(parts["stars"].copy(), lv["params"], None)
    s2 = real + (parts["synth"] - real) * (1.0 * m)
    assert np.array_equal(under, np.clip(screen(parts["starless"], s2), 0, 1))


def test_the_eyes_of_the_group():
    """Real stars hidden: the synthetic ones alone in the group (where their mask is open); synthetic stars hidden: the
    real ones alone; the group hidden: the starless image alone."""
    parts = images()
    row = painted(pe.new_synth_layer(0), [(30, 18, 10)])
    for shown, want in (({"real": False}, "synth alone"), ({}, "real alone"), ({"stars": False}, "starless")):
        g = group(shown={"stars": True, "starless": True, **shown})
        row["target"] = g["id"]
        r = dict(row, visible=want != "real alone")
        got = compose([r, g], parts, want=[row["id"]])
        m = got.masks.get(row["id"])
        if want == "synth alone":
            zero = np.zeros_like(parts["stars"])
            s2 = zero + (parts["synth"] - zero) * (1.0 * m)[..., None]
            assert np.array_equal(got.rgb, np.clip(screen(parts["starless"], s2), 0, 1))
            assert m.min() == 0.0                                       # (closed somewhere: no star there at all)
        elif want == "real alone":
            assert np.array_equal(got.rgb, np.clip(screen(parts["starless"], parts["stars"]), 0, 1))
        else:
            assert np.array_equal(got.rgb, np.clip(parts["starless"], 0, 1))
    # the real stars' eye off without Synthetic stars (the block taken out since): the stars stay
    g = group(shown={"stars": True, "starless": True, "real": False})
    assert np.array_equal(compose([g], parts).rgb, np.clip(screen(parts["starless"], parts["stars"]), 0, 1))


def test_both_methods_start_with_the_same_open_mask():
    """Until #93 the image's shape started closed and the instrument's open: one start now, open, its brush taking the
    synthetic stars away where it paints."""
    made = {}
    for method in ("measured", "instrument"):
        g = group()
        layers = pe.with_synth_rows([g], {1: method})
        row = next(x for x in layers if pe.is_synth(x))
        assert row["target"] == g["id"] and row["sub"] == "stars" and row["visible"] and row["mode"] == "Normal"
        assert pe.with_synth_rows(layers, {1: method}) == layers                     # once
        assert [x for x in pe.with_synth_rows(layers, {}) if pe.is_synth(x)] == []   # the block gone: the row too
        made[method] = {k: v for k, v in row["mask"].items() if k != "parts"}, [
            {k: v for k, v in p.items() if k != "id"} for p in row["mask"]["parts"]]
        m = compose([row, g], images(), want=[row["id"]]).masks[row["id"]]
        assert m.min() == 1.0
    assert made["measured"] == made["instrument"]
    assert made["measured"][1][0]["src"] == "brush" and made["measured"][1][0]["op"] == "sub"


# ---------------------------------------------------------------------------------------------- saved compositions
def saved_before_93(gid=7):
    """A composition as Post saved it before #93 (postengine.save_layers): the Synthetic stars filter under the stars of
    the layer, its mask (the image shape's brush, painted in a corner) and its settings."""
    return [
        {"id": 20, "kind": "filter", "filter": "levels", "name": "Levels", "visible": True, "opacity": 100.0,
         "mode": "Normal", "params": {"black": 0.0, "white": 0.7, "gamma": 1.0}, "mask": None, "target": gid,
         "sub": "stars"},
        {"id": 21, "kind": "filter", "filter": "synth", "name": "Synthetic stars", "visible": True, "opacity": 80.0,
         "mode": "Normal", "params": {}, "target": gid, "sub": "stars",
         "mask": {"parts": [{"id": 30, "src": "brush", "op": "add", "strokes": [[6.0, 6.0, 14.0, 50.0, True]],
                             "size": 60.0, "hardness": 50.0, "paint": True}],
                  "feather": 2.0, "density": 100.0, "invert": False}},
        {"id": gid, "kind": "workflow", "pid": 1, "part": "combined", "name": "Full · Combine layers", "visible": True,
         "opacity": 100.0, "mode": "Normal", "base": None, "sliders": {}, "mask": None, "blend": "Screen",
         "shown": {"stars": True, "starless": True}},
    ]


def test_a_composition_saved_before_the_group_opens_as_it_with_the_same_image():
    """The filter under the stars becomes the Synthetic stars layer of the group, its mask and settings kept: the same
    pixels, and the same saved text (nothing to write until the user changes something)."""
    raw = saved_before_93()
    layers = pe.with_synth_rows(pe.load_layers(copy.deepcopy(raw), {1}), {1: "measured"})
    assert [(r.get("kind"), r.get("filter") or r.get("sub"), d) for r, d in pe.rows(layers)] == [
        ("workflow", None, 0), ("sub", "stars", 1), ("filter", "levels", 2), ("filter", "synth", 2),
        ("sub", "real", 2), ("sub", "starless", 1)]
    assert [pe.SUB_NAMES[r["sub"]] for r, _d in pe.rows(layers) if r["kind"] == "sub"] == [
        "Stars", "Real stars", "Starless image"]
    row = next(x for x in layers if pe.is_synth(x))
    assert row["opacity"] == 80.0 and row["mask"]["feather"] == 2.0 and row["mask"]["parts"][0]["strokes"] == [
        (6.0, 6.0, 14.0, 50.0, True)]
    # the two sides of a Combine layers layer are named now (issue #165): its starless image and its stars are its
    # workflow's, as they were; the rest of the saved text is the same
    sides = [dict(x, starless="starless", stars_pid=x["pid"]) if x.get("part") == "combined" else x for x in raw]
    assert json.dumps(pe.save_layers(layers), sort_keys=True) == json.dumps(sides, sort_keys=True)
    # the image: the levels on the stars after the synthetic ones replaced them through their mask at 80 %
    parts = images()
    got = compose(layers, parts, want=[row["id"]])
    m = got.masks[row["id"]]
    s2 = parts["stars"] + (parts["synth"] - parts["stars"]) * (0.8 * m)[..., None]
    lv = pe.FILTERS["levels"]["fn"](s2, layers[0]["params"], None)
    assert np.array_equal(got.rgb, np.clip(screen(parts["starless"], lv), 0, 1))
    # an older drag had put it on the starless image, or twice in the layer: one, on the stars
    odd = copy.deepcopy(raw)
    odd[1]["sub"] = "starless"
    odd.insert(1, dict(copy.deepcopy(raw[1]), id=22))
    fixed = pe.with_synth_rows(pe.load_layers(odd, {1}), {1: "measured"})
    assert [(x["id"], x["sub"]) for x in fixed if pe.is_synth(x)] == [(22, "stars")]


# ---------------------------------------------------------------------------------------------- the page, on a project
@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture()
def project(qapp, tmp_path, monkeypatch):
    """A project with the Full workflow (Stars/background, Combine layers) and its Combine layers layer in Post."""
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
    full = m.paths[0]
    m.add_post_layer(full.pid, "combined", keep_default=False)
    m._undo.clear()
    yield m
    w.stop()


def page_of(m):
    from astro_gpu.client.post3 import PostPage3
    from astro_gpu.client.services.render import RenderWorker
    return PostPage3(m, worker=RenderWorker(None, resolver=lambda pid: None, device="cpu"))


def blocks(m):
    return [n["block"] for n in m.graph(m.paths[0])]


def action(menu, text):
    return next(a for a in menu.actions() if a.text() == text)


def test_the_panel_shows_the_group_in_french_with_an_eye_each(project):
    m = project
    m.insert_block(m.paths[0], "synth")
    i18n.set_language("fr")
    page = page_of(m)
    v = page.view
    try:
        g = next(x for x in v.layers if pe.is_group(x))
        assert [(r.lay["kind"], r.lay.get("filter") or r.lay.get("sub"), r.depth) for r in v.rows] == [
            ("workflow", None, 0), ("sub", "stars", 1), ("filter", "synth", 2), ("sub", "real", 2),
            ("sub", "starless", 1)]
        assert [r.n.full_text() for r in v.rows][1:] == ["Étoiles", "Étoiles de synthèse", "Étoiles réelles",
                                                         "Image sans étoiles"]
        synth = next(r for r in v.rows if pe.is_synth(r.lay))
        assert synth.m is not None                                     # its mask beside it, as a layer's
        # the eyes: the real stars', the synthetic stars', the group's
        from astro_gpu.client.post3 import EyeBox
        next(r for r in v.rows if r.lay.get("sub") == "real").findChild(EyeBox).click()
        assert g["shown"]["real"] is False
        assert not next(r for r in v.rows if r.lay.get("sub") == "real").findChild(EyeBox).isChecked()
        next(r for r in v.rows if pe.is_synth(r.lay)).findChild(EyeBox).click()
        assert not v.layer(synth.lay["id"])["visible"]
        next(r for r in v.rows if r.lay.get("sub") == "stars").findChild(EyeBox).click()
        assert g["shown"]["stars"] is False and g["shown"]["starless"] is True
        # the group's menu: its Synthetic stars ticked; the row's own: Delete, no filter, no copy
        menu = v.layer_menu_of(pe.sub_row(g, "stars"))
        assert action(menu, "Étoiles de synthèse").isChecked()
        own = [a.text() for a in v.layer_menu_of(synth.lay).actions() if a.text()]
        assert "Supprimer" in own and "Dupliquer" not in own and "Ajouter un filtre" not in own
        v.select(synth.lay["id"], "layer")
        assert not v.b_dup.isEnabled()
    finally:
        i18n.set_language("en")
        page.shutdown()
        page.deleteLater()


def test_synthetic_stars_added_and_removed_from_post_with_their_block_one_undo_step_each(project):
    m = project
    page = page_of(m)
    v = page.view
    try:
        g = next(x for x in v.layers if pe.is_group(x))
        assert "synth" not in blocks(m) and not pe.has_synth(v.layers, g["id"])
        a = action(v.layer_menu_of(pe.sub_row(g, "stars")), "Synthetic stars")
        assert a.isEnabled() and not a.isChecked()
        a.trigger()
        b = blocks(m)
        assert b.index("synth") == b.index("stars") + 1                # after Stars/background, as the palette puts it
        row = next(x for x in v.layers if pe.is_synth(x))
        assert row["target"] == g["id"] and v.sel == (row["id"], "layer")
        v.do_undo()
        assert "synth" not in blocks(m) and not pe.has_synth(v.layers, g["id"])
        v.do_redo()
        assert "synth" in blocks(m) and pe.has_synth(v.layers, g["id"])
        # painted, then taken out from its own menu: one step brings back the block and the layer with its stroke
        row = next(x for x in v.layers if pe.is_synth(x))
        v.push()
        row["mask"]["parts"][0]["strokes"].append((10.0, 10.0, 30.0, 50.0, True))
        v._save()
        action(v.layer_menu_of(row), "Delete").trigger()
        assert "synth" not in blocks(m) and not pe.has_synth(v.layers, g["id"])
        assert not any(x.get("filter") == "synth" for x in m.post_version()["layers"])
        v.do_undo()
        assert "synth" in blocks(m)
        back = next(x for x in v.layers if pe.is_synth(x))
        assert back["id"] == row["id"] and back["mask"]["parts"][0]["strokes"] == [(10.0, 10.0, 30.0, 50.0, True)]
        v.do_undo()                                                    # the stroke, Post's own step
        assert next(x for x in v.layers if pe.is_synth(x))["mask"]["parts"][0]["strokes"] == []
        # the Delete button does the same
        v.select(back["id"], "layer")
        v.delete()
        assert "synth" not in blocks(m)
    finally:
        page.shutdown()
        page.deleteLater()


def test_a_project_saved_before_the_group_opens_unchanged(project):
    """Opened, the composition saved before #93 is not written: only a change of the user writes it."""
    m = project
    m.insert_block(m.paths[0], "synth")
    gid = next(x for x in m.post_version()["layers"] if x.get("part") == "combined")["id"]
    raw = saved_before_93(gid)
    m.post_version()["layers"] = copy.deepcopy(raw)
    before = json.dumps(m.p.post, sort_keys=True)
    page = page_of(m)
    v = page.view
    try:
        assert [(r.lay.get("filter") or r.lay.get("sub"), r.depth) for r in v.rows][1:] == [
            ("stars", 1), ("levels", 2), ("synth", 2), ("real", 2), ("starless", 1)]
        v.flush()
        assert json.dumps(m.p.post, sort_keys=True) == before
    finally:
        page.shutdown()
        page.deleteLater()
