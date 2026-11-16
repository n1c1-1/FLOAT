"""The base stretch of a workflow layer: Automatic (the screen transfer) or Arcsinh (the colour-preserving arcsinh
stretch, Lupton et al. 2004), applied to the linear image.

The defect: an object hundreds of times brighter than the sky (a planetary nebula) came out of the automatic stretch as
a flat disc at white, and nothing in Post could give its structure back, every filter acting after that stretch. On a
synthetic linear image whose core is 400 times the sky: the automatic stretch burns it, the Arcsinh keeps two levels of
the core apart and the colours of the linear image. Automatic leaves no trace in a saved composition; an Arcsinh layer
comes back as it was saved; the Output block's picture is Post's layer at its default settings for both; the choice is
one Undo step, named; the labels are in French.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication, QComboBox  # noqa: E402
from test_post import (  # noqa: E402, F401 - the fixtures and helpers of Post's tests
    DEV,
    Setup,
    _settings,
    named,
    near,
    preps,
    qapp,
    reference,
    shown,
    widgets_of,
)

from astro_gpu.client import i18n  # noqa: E402
from astro_gpu.client.postengine import (  # noqa: E402
    STRETCH_KEY,
    layer_stretch,
    load_layers,
    new_workflow_layer,
    output_stretch,
    save_layers,
    set_layer_stretch,
    snapshot,
)
from astro_gpu.client.services.render import Folder, LayerSource, Renderer, Request  # noqa: E402
from astro_gpu.render.layers import LayerService  # noqa: E402
from astro_gpu.render.simple import image_stats  # noqa: E402

CORE = (0.4, 0.9, 1.0)                                   # the colour of the core: a blue-green, as an O III core is
H, W = 400, 600


def bright_core_prep(d: Path) -> Path:
    """An Output block's folder: a sky of noise (deviation 1 around 0, its level taken off as the stage does) and a
    round core 400 times that: a ring at 100, a disc at 200 (two levels 2:1 apart) and a narrow peak at 400."""
    rng = np.random.default_rng(7)
    d.mkdir(parents=True, exist_ok=True)
    yy, xx = np.mgrid[:H, :W]
    r2 = (yy - H / 2) ** 2 + (xx - W / 2) ** 2
    level = np.where(r2 < 50 ** 2, 200.0, np.where(r2 < 100 ** 2, 100.0, 0.0)) + 200.0 * np.exp(-r2 / (2 * 3.0 ** 2))
    lin = (np.stack([c * level for c in CORE]) + rng.normal(0, 1, (3, H, W))).astype(np.float32)
    valid = np.ones((H, W), bool)
    prep = {"M": np.eye(3).tolist(), "crop": [0, W, 0, H], "stats": image_stats(lin, valid), "linked": True,
            "stretch": "auto", "denoised": False, "color": True, "exit": None, "drizzle_scale": 1.0}
    np.save(d / "lin.npy", lin)
    np.save(d / "valid.npy", valid)
    (d / "prep_simple.json").write_text(json.dumps(prep))
    return d


def zones():
    yy, xx = np.mgrid[:H, :W]
    r = np.hypot(yy - H / 2, xx - W / 2)
    return r < 100, (r > 10) & (r < 46), (r > 54) & (r < 96)          # the core, its disc at 200, its ring at 100


def u8(rgb: np.ndarray) -> np.ndarray:
    return np.clip(np.round(rgb * 255), 0, 255).astype(np.uint8)


def test_a_core_400_times_the_sky_is_burnt_by_automatic_and_kept_by_arcsinh(tmp_path):
    """Through the layer's render (the service Post renders its layers with), at the default settings."""
    folder = bright_core_prep(tmp_path / "out")
    svc = LayerService(DEV)
    core, disc, ring = zones()
    auto = u8(svc.render(folder, None, {}, 1.0).rgb)
    asinh = u8(svc.render(folder, None, {}, 1.0, stretch="arcsinh").rgb)
    white = lambda a: float((a >= 254).any(0)[core].mean())                 # noqa: E731
    assert white(auto) > 0.01, "the automatic stretch sends the core to white"
    assert white(asinh) < 0.001, f"Arcsinh keeps it under white ({white(asinh):.4f})"
    lum = lambda a, z: float(a[:, z].astype(np.float32).mean())             # noqa: E731
    apart = lum(asinh, disc) - lum(asinh, ring)
    assert apart > 10, f"two levels of the core 2:1 apart stay apart ({apart:.1f} levels of 255)"
    assert apart > 3 * (lum(auto, disc) - lum(auto, ring)), "further apart than the automatic stretch leaves them"
    # the sky starts where the automatic stretch puts it, as a share of the luminance's range: no black sky, no veil
    sky = lambda a: float(np.median(a[:, ~core]))                           # noqa: E731
    assert 0.5 * sky(auto) < sky(asinh) <= sky(auto) + 1


def test_arcsinh_keeps_the_hues_of_the_linear_image(tmp_path):
    """The three channels take the same factor: the ratios between the channels of an unclipped pixel are those of the
    linear image (the saturation left at 1: it moves them on purpose)."""
    folder = bright_core_prep(tmp_path / "out")
    svc = LayerService(DEV)
    y = svc.render(folder, {"saturation": 1.0}, {}, 1.0, stretch="arcsinh").rgb
    _core, disc, ring = zones()
    lum = json.loads((folder / "prep_simple.json").read_text())["stats"]["lum"]
    lin = np.load(folder / "lin.npy") - (lum["median"] - 2.8 * lum["mad"])  # the linear image, its black point taken off
    for zone in (disc, ring):
        px, was = y[:, zone], lin[:, zone]
        assert px.max() < 0.999                                             # unclipped
        r, g, b = (float(px[c].mean()) for c in range(3))
        assert r / b == pytest.approx(float(was[0].mean() / was[2].mean()), rel=0.005)
        assert g / b == pytest.approx(float(was[1].mean() / was[2].mean()), rel=0.005)
    # the automatic stretch does not keep them in a bright core (its channels go to white one after the other)
    a = svc.render(folder, {"saturation": 1.0}, {}, 1.0).rgb[:, disc]
    assert float(a[0].mean() / a[2].mean()) > 1.5 * CORE[0] / CORE[2]


def test_the_strength_darkens_the_sky_and_spreads_the_core(tmp_path):
    folder = bright_core_prep(tmp_path / "out")
    svc = LayerService(DEV)
    core, disc, ring = zones()
    mid = u8(svc.render(folder, None, {}, 1.0, stretch="arcsinh").rgb).astype(np.float32)
    low = u8(svc.render(folder, None, {STRETCH_KEY: 1.0}, 1.0, stretch="arcsinh").rgb).astype(np.float32)
    assert np.median(low[:, ~core]) < 0.6 * np.median(mid[:, ~core])
    assert low[:, disc].mean() - low[:, ring].mean() > mid[:, disc].mean() - mid[:, ring].mean()
    # without the choice the position is nothing: the automatic stretch has no strength
    a = svc.render(folder, None, {}, 1.0).rgb
    assert np.array_equal(svc.render(folder, None, {STRETCH_KEY: 1.0}, 1.0).rgb, a)


def test_automatic_leaves_no_field_and_arcsinh_comes_back_as_saved():
    lay = new_workflow_layer(1, "Full", True)
    before = json.dumps(save_layers([lay]), sort_keys=True)
    assert "stretch" not in before
    set_layer_stretch(lay, "arcsinh")
    lay["sliders"][STRETCH_KEY] = 1.25
    saved = json.loads(json.dumps(save_layers([lay])))
    back = load_layers(saved, {1})[0]
    assert layer_stretch(back) == "arcsinh" and back["sliders"][STRETCH_KEY] == 1.25
    assert save_layers([back]) == saved
    set_layer_stretch(back, "auto")                       # back to Automatic: the layer as it was before the choice
    back["id"] = lay["id"]
    assert json.dumps(save_layers([back]), sort_keys=True) == before
    # a composition saved before the choice opens unchanged, a value it does not know is Automatic
    old = json.loads(before)
    assert save_layers(load_layers(old, {1})) == old
    assert layer_stretch(load_layers([dict(old[0], stretch="other")], {1})[0]) == "auto"


def test_the_render_key_changes_only_when_the_stretch_is_not_automatic(preps):
    from astro_gpu.client.postengine import Canvas, Composer
    src = LayerSource(lambda pid: {1: Folder(preps["classic"])}.get(pid), LayerService(DEV))
    comp = Composer(src)
    lay = new_workflow_layer(1, "Full", True)
    cv = Canvas(64, 64, (0, 0))
    key = comp._key(lay, 1, None, cv)
    assert not any(isinstance(x, tuple) and x and x[0] == "stretch" for x in key)
    assert comp._key(dict(lay, stretch="arcsinh"), 1, None, cv)[:len(key)] == key
    assert comp._key(dict(lay, stretch="arcsinh"), 1, None, cv) != key
    # the stars of a Combine layers layer are not stretched by it: their image is kept as it is
    stars = dict(lay, part="stars")
    assert comp._key(dict(stars, stretch="arcsinh"), 1, None, cv) == comp._key(stars, 1, None, cv)


def test_the_output_blocks_picture_is_posts_default_layer_for_both_choices(preps):
    """Inspect renders the Output block's final image from Post's layer at its default settings; with Arcsinh chosen
    for that layer in Post, from the same layer with Arcsinh."""
    from astro_gpu.client.services.inspect import InspectService
    QApplication.instance() or QApplication([])
    post_layers = []
    model = NS(p=None, gpu=False, post_layers=lambda: post_layers)
    sv = InspectService(model)
    try:
        render = lambda layers: Renderer(LayerSource(lambda pid: {2: Folder(preps["classic"])}.get(pid),  # noqa: E731
                                                     LayerService(DEV))).run(
            Request(kind="half", layers=snapshot(layers), level=1)).rgb
        mine = new_workflow_layer(2, "Classic", True)
        mine["sliders"] = {"contrast": 3.0}                     # Post's own settings are not the Output block's
        post_layers[:] = save_layers([mine])
        plain = new_workflow_layer(2, "Classic", True)
        got = sv.final_layer(2, "Classic")
        assert {k: v for k, v in got.items() if k != "id"} == {k: v for k, v in plain.items() if k != "id"}
        assert sv.render_key(2, "k1") == "k1"                   # the picture kept before the choice is still the one
        auto = render([got])
        assert np.array_equal(auto, render([plain]))
        set_layer_stretch(mine, "arcsinh")
        mine["sliders"][STRETCH_KEY] = 1.5
        post_layers[:] = save_layers([mine])
        got = sv.final_layer(2, "Classic")
        default = dict(plain, stretch="arcsinh", sliders={STRETCH_KEY: 1.5})
        assert np.array_equal(render([got]), render([default]))
        assert not np.array_equal(render([got]), auto)
        assert sv.render_key(2, "k1") != "k1"
        # a Combine layers layer whose starless image is this Output: its choice too
        group = new_workflow_layer(2, "Classic", True, "combined")
        group.update(starless="output", stretch="arcsinh")
        assert output_stretch(save_layers([group]), 2) == {"stretch": "arcsinh", "sliders": {}}
        assert output_stretch(save_layers([group]), 3) == {}
    finally:
        sv.stop()


def _stretch_box(v):
    return next(c for c in widgets_of(v, QComboBox) if c.itemData(0) == "auto" and c.count() == 2)


def test_the_choice_is_in_the_layers_panel_one_undo_step_and_saved(preps, qapp):
    from astro_gpu.client.post3 import ContSlider
    s = Setup(preps, qapp)
    try:
        assert s.settle()
        v = s.view
        full = named(v, "Full")
        v.select(full["id"], "layer")
        box = _stretch_box(v)
        assert [box.itemText(i) for i in range(2)] == ["Automatic", "Arcsinh"] and box.currentData() == "auto"
        names = lambda: [x.name.text() for x in widgets_of(v, ContSlider)]         # noqa: E731
        assert names().count("Strength") == 1                   # the stars' own (greyed): none for Automatic
        assert names().index("Sky") < names().index("Contrast")
        before = shown(v).copy()
        v.set_stretch(full, "arcsinh")
        assert s.settle()
        full = named(v, "Full")
        assert layer_stretch(full) == "arcsinh"
        box = _stretch_box(v)
        assert box.currentData() == "arcsinh"
        assert names().count("Strength") == 2 and names().index("Strength") < names().index("Sky")   # right under it
        colours = next(x for x in widgets_of(v, ContSlider) if x.name.text() == "Colours")
        assert not colours.s.isEnabled() and colours.why.text() == "Arcsinh keeps the colours by itself"
        assert not near(shown(v), before) and near(shown(v), reference(s).rgb)
        saved = s.model.p.post["versions"][0]["layers"]
        assert next(x for x in saved if x["id"] == full["id"])["stretch"] == "arcsinh"
        assert v.next_words("undo") == "Stretch of “Full”: Arcsinh"
        v.do_undo()
        assert s.settle()
        assert layer_stretch(named(v, "Full")) == "auto" and "stretch" not in named(v, "Full")
        assert near(shown(v), before)
        assert "stretch" not in json.dumps(s.model.p.post["versions"][0]["layers"])
        assert v.next_words("redo") == "Stretch of “Full”: Arcsinh"
        v.do_redo()
        assert s.settle()
        assert layer_stretch(named(v, "Full")) == "arcsinh"
        # its Strength: a slider of the layer, named in the Undo words
        full = named(v, "Full")
        v.push()
        v._slider(full, STRETCH_KEY, 1.0)
        assert v.next_words("undo") == "Stretch strength of “Full”"
    finally:
        s.close()


def test_the_labels_are_french(preps, qapp):
    from astro_gpu.client.post3 import ContSlider
    i18n.set_language("fr")
    s = Setup(preps, qapp)
    try:
        assert s.settle()
        v = s.view
        full = next(x for x in v.layers if x["kind"] == "workflow" and x["pid"] == 1)
        v.select(full["id"], "layer")
        v.set_stretch(full, "arcsinh")
        assert s.settle()
        box = _stretch_box(v)
        assert [box.itemText(i) for i in range(2)] == ["Automatique", "Arcsinh"]
        from PySide6.QtWidgets import QLabel
        assert any(x.text() == "Étirement" for x in widgets_of(v, QLabel))
        assert [x.name.text() for x in widgets_of(v, ContSlider)].count("Intensité") == 2
        assert "Étirement de «" in v.next_words("undo") and v.next_words("undo").endswith(": Arcsinh")
        for text in ("gentler", "Stretch strength", "Arcsinh keeps the colours by itself"):
            assert i18n._(text) != text, text
    finally:
        s.close()
        i18n.set_language("en")
