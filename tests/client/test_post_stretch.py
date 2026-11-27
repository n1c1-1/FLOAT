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
        assert v.next_words("undo").startswith("Stretch strength of “Full”: ")       # (and what it became)
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
        names = [x.name.text() for x in widgets_of(v, ContSlider)]
        assert names.count("Intensité") == 1 and names.count("Force") == 1    # the Arcsinh's, the stars' own word
        assert "Étirement de «" in v.next_words("undo") and v.next_words("undo").endswith(": Arcsinh")
        for text in ("gentler", "Stretch strength", "Arcsinh keeps the colours by itself"):
            assert i18n._(text) != text, text
    finally:
        s.close()
        i18n.set_language("en")


# ------------------------------------------------------------------------------------------- the review's second round
def bright_stars_prep(d: Path) -> Path:
    """A Stars block's folder: a starless image (a sky of noise, a nebula at 100 with its top at 200 and a narrow peak
    at 300) and stars alone 50 times brighter than that top, away from the nebula; the statistics of the whole image (stars in), as the block
    writes them."""
    rng = np.random.default_rng(11)
    d.mkdir(parents=True, exist_ok=True)
    yy, xx = np.mgrid[:H, :W]
    r2 = (yy - H / 2) ** 2 + (xx - W / 2) ** 2
    level = np.where(r2 < 25 ** 2, 200.0, np.where(r2 < 60 ** 2, 100.0, 0.0))
    level = level + 100.0 * np.exp(-r2 / (2 * 3.0 ** 2))
    starless = (np.stack([0.9 * level, level, 0.95 * level]) + rng.normal(0, 1, (3, H, W))).astype(np.float32)
    stars = np.zeros((H, W), np.float32)
    n = 0
    while n < 40:
        x, y = rng.uniform(0, W), rng.uniform(0, H)
        if np.hypot(y - H / 2, x - W / 2) > 100:
            stars += 10000.0 * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * 2.0 ** 2))
            n += 1
    stars = np.stack([stars, stars, stars]).astype(np.float32)
    valid = np.ones((H, W), bool)
    np.save(d / "starless.npy", starless)
    np.save(d / "stars.npy", stars)
    np.save(d / "valid.npy", valid)
    (d / "prep_stars.json").write_text(json.dumps({"crop": [0, W, 0, H], "M": np.eye(3).tolist(), "drizzle_scale": 1.0,
                                                   "unit": 1.0, "background": [0, 0, 0],
                                                   "stats_full": image_stats(starless + stars, valid)}))
    return d


def test_stars_50_times_the_nebula_leave_its_top_in_the_upper_range(tmp_path):
    """The white point of the Arcsinh is the starless image's own (it is the image the stretch is applied to): read
    from the whole image, the stars set it and the nebula ended in the middle of the range. Through Post's renderer,
    a Combine layers layer at its default settings."""
    st = bright_stars_prep(tmp_path / "stars")
    fo = {("stars", 1): Folder(st)}
    r = Renderer(LayerSource(lambda sid: fo.get(sid), LayerService(DEV)))
    lay = new_workflow_layer(1, "Full", True, "combined")
    lay["stretch"] = "arcsinh"
    res = r.run(Request(kind="half", layers=snapshot([lay]), level=1))
    rgb = res.rgb.astype(np.float32)
    assert rgb.shape[:2] == (H // 2, W // 2)
    yy, xx = np.mgrid[:H // 2, :W // 2]
    rad = np.hypot(yy - H / 4, xx - W / 4)
    top = (rad > 5) & (rad < 10)                                            # the nebula's top (its inner disc, off the peak)
    lum = rgb @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    assert float(lum[top].mean()) > 0.85 * 255, f"the nebula's top is at {lum[top].mean():.0f} of 255"
    assert float((rgb[top] >= 254).any(-1).mean()) < 0.01               # and not burnt
    # the stars are not stretched by it: the same as on the automatic layer
    stars = lambda layer: r.run(Request(kind="half", level=1, layers=snapshot(                    # noqa: E731
        [dict(layer, shown={"stars": True, "starless": False})]))).rgb
    assert np.array_equal(stars(lay), stars(new_workflow_layer(1, "Full", True, "combined")))


def _sky(svc, folder, sliders) -> float:
    core = zones()[0]
    return 255 * float(np.median(svc.render(folder, None, sliders, 1.0, stretch="arcsinh").rgb[:, ~core]))


def test_one_step_of_the_strength_moves_the_sky_by_two_levels_at_most(tmp_path):
    """The Strength is linear in the level the sky lands at (each step of the first law was a factor on the stretch
    factor: the sky went from 35 to 10 of 255 in one notch on a planetary nebula, and could not be set at 25-30)."""
    from astro_gpu.render.sliders import STRENGTH_STEP
    folder = bright_core_prep(tmp_path / "out")
    svc = LayerService(DEV)
    sky = [_sky(svc, folder, {STRETCH_KEY: 2.0 + k * STRENGTH_STEP}) for k in range(-2, 3)]
    steps = np.diff(sky)
    assert (steps > 0.3).all() and (steps <= 2.0).all(), f"the sky per step of the Strength: {np.round(sky, 1)}"
    assert steps.max() - steps.min() < 0.3                              # the same amount at each step
    # over the slider: never more than two levels a step, and half the sky at half the position (above the start the
    # picture's scale grows too, the sky rises a little less than the position)
    start = _sky(svc, folder, {})
    assert _sky(svc, folder, {STRETCH_KEY: 1.0}) == pytest.approx(start / 2, abs=2.0)
    for pos in (0.5, 1.0, 3.0, 3.9):
        step = _sky(svc, folder, {STRETCH_KEY: pos + STRENGTH_STEP}) - _sky(svc, folder, {STRETCH_KEY: pos})
        assert 0.3 < step <= 2.0, (pos, step)


def test_sky_does_not_move_the_sky_under_arcsinh(tmp_path):
    """One knob for the sky: under Arcsinh the Strength sets it, the Sky slider's position is not read (it is greyed
    in the panel); under Automatic Sky sets it as before."""
    folder = bright_core_prep(tmp_path / "out")
    svc = LayerService(DEV)
    a = svc.render(folder, None, {}, 1.0, stretch="arcsinh").rgb
    assert np.array_equal(svc.render(folder, None, {"sky": 3.5}, 1.0, stretch="arcsinh").rgb, a)
    assert not np.array_equal(svc.render(folder, None, {STRETCH_KEY: 2.5}, 1.0, stretch="arcsinh").rgb, a)
    assert not np.array_equal(svc.render(folder, None, {"sky": 3.5}, 1.0).rgb, svc.render(folder, None, {}, 1.0).rgb)
    # the stars of a Combine layers layer keep their own stretch: their settings are those of an Automatic layer
    from astro_gpu.render.sliders import settings_at
    pos = {"sky": 3.5, "star_strength": 3.0}
    assert settings_at("stars", None, pos, False, "", "arcsinh") == settings_at("stars", None, pos, False)


#: the automatic stretch of ``bright_core_prep`` before the second round: the mean of each channel and five pixels
#: (row, column), in levels of 255
AUTO_MEANS = [72.44, 82.98, 85.89]
AUTO_PIXELS = [[165, 245, 255], [151, 238, 255], [151, 238, 255], [49, 56, 90], [71, 78, 23]]
AUTO_AT = ((200, 300), (200, 370), (200, 395), (40, 40), (350, 520))


def test_the_automatic_picture_is_the_one_it_was(tmp_path):
    folder = bright_core_prep(tmp_path / "out")
    a = u8(LayerService(DEV).render(folder, None, {}, 1.0).rgb)
    assert [round(float(a[c].mean()), 2) for c in range(3)] == pytest.approx(AUTO_MEANS, abs=0.02)
    got = np.array([a[:, y, x] for y, x in AUTO_AT], np.int16)
    assert np.abs(got - np.array(AUTO_PIXELS, np.int16)).max() <= 1, got.tolist()      # (a level of rounding apart)


def test_the_strength_counts_in_steps_and_sky_is_greyed_with_its_reason(preps, qapp):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    from astro_gpu.client.post3 import ContSlider
    from astro_gpu.render.sliders import STRENGTH_STEP
    s = Setup(preps, qapp)
    try:
        assert s.settle()
        v = s.view
        full = named(v, "Full")
        v.select(full["id"], "layer")
        by_name = lambda name: [x for x in widgets_of(v, ContSlider) if x.name.text() == name]    # noqa: E731
        assert by_name("Sky")[0].s.isEnabled()                              # Automatic: Sky sets the sky
        v.set_stretch(full, "arcsinh")
        assert s.settle()
        sky = by_name("Sky")[0]
        assert not sky.s.isEnabled() and sky.why.text() == "Arcsinh: Strength sets the sky"
        strength = v.stretch_strength
        assert strength.step == STRENGTH_STEP and strength.value() == 2.0
        assert strength.val.text() == "0"
        QTest.keyClick(strength.s, Qt.Key_Right)                            # an arrow key: one step
        assert s.settle()
        assert named(v, "Full")["sliders"][STRETCH_KEY] == pytest.approx(2.0 + STRENGTH_STEP)
        assert v.stretch_strength.val.text() == "+0.05"                     # the value says the step
        assert v.next_words("undo").startswith("Stretch strength of “Full”: ")       # (and what it became)
        v.do_undo()
        assert s.settle()
        assert STRETCH_KEY not in named(v, "Full")["sliders"]
    finally:
        s.close()


def test_the_second_rounds_labels_are_french():
    i18n.set_language("fr")
    try:
        from astro_gpu.render.simple import SimpleSettings
        from astro_gpu.render.sliders import ARCSINH_WHY
        tip = ("The level of the sky under the Arcsinh. In the middle the sky is where the Sky setting puts it within "
               "the range of the luminance. To the left the sky darkens and the bright areas spread apart; to the right "
               "the faint areas rise")
        for text in (*ARCSINH_WHY.values(), SimpleSettings.model_fields["stretch_strength"].json_schema_extra["help"],
                     tip):
            assert text and i18n._(text) != text, text
        assert i18n._(ARCSINH_WHY["sky"]) == "Arcsinh : Intensité règle le ciel"
    finally:
        i18n.set_language("en")


# --------------------------------------------------------------------------------------- the Sky slider's two scales
#: a layer as a project saved without a scale of the Sky slider holds it: the first scale, the one of every new layer
FIRST_LAYER = {"id": 3, "kind": "workflow", "pid": 1, "part": None, "name": "Full", "visible": True, "opacity": 100.0,
               "mode": "Normal", "base": None, "sliders": {}, "mask": None}


def _through_post(folder, lay) -> np.ndarray:
    r = Renderer(LayerSource(lambda sid: {1: Folder(folder)}.get(sid), LayerService(DEV)))
    return r.run(Request(kind="half", layers=snapshot([lay]), level=1)).rgb


def test_one_sky_table_and_a_stored_scale_of_before_is_read_past(tmp_path):
    """The Sky slider has one table: the sky at 0.25 by default, 0.05 to 0.37 over the slider. A layer made on 16 or
    17 November 2026 stored "sky_scale" (a second table, withdrawn: a compatibility patch): its file still loads, the
    key read past and never written again; the layer reads with the one table."""
    from astro_gpu.client.postengine import new_image_layer
    from astro_gpu.render.simple import SimpleSettings
    from astro_gpu.render.sliders import SKY_TABLE, settings_at
    assert SimpleSettings().sky == 0.25 == SKY_TABLE[2]
    assert settings_at("simple", None, {"sky": 0.0}).sky == pytest.approx(0.05)
    assert settings_at("simple", None, {"sky": 4.0}).sky == pytest.approx(0.37)
    old, = load_layers([dict(FIRST_LAYER, sky_scale=2)], {1})
    new = new_workflow_layer(1, "Full", True)
    assert "sky_scale" not in old and "sky_scale" not in new and "sky_scale" not in save_layers([old])[0]
    folder = bright_core_prep(tmp_path / "out")
    assert np.array_equal(_through_post(folder, old), _through_post(folder, dict(new, id=old["id"])))
    image = {"kind": "image", "key": "k", "name": "I", "sliders": {}, "sky_scale": 2}
    assert "sky_scale" not in load_layers([image], set())[0] and "sky_scale" not in new_image_layer("k", "I")


