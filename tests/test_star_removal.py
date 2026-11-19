"""The Stars block (stages/star_removal.py) and the rendering of its two images (render/stars.py): the network takes a
star off and leaves the sky, the two images add up to the image, each part renders, the star sliders move what they
say, and the workflows put the block after the Output."""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from astro_gpu.core import workflow as wfl
from astro_gpu.render import sliders as S
from astro_gpu.render.simple import SimpleRenderEngine, SimpleSettings, image_stats
from astro_gpu.render.stars import BLENDS, StarSettings, StarsRenderEngine, _morph, blend, resize_stars
from astro_gpu.stages import star_removal as sr

DEV = torch.device("cpu")


def moffat(h, w, x0, y0, amp, fwhm=3.0, beta=3.0):
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    a = fwhm / (2 * np.sqrt(2 ** (1 / beta) - 1))
    return amp * (1 + ((xx - x0) ** 2 + (yy - y0) ** 2) / a ** 2) ** -beta


@pytest.fixture(scope="module")
def net():
    return sr.network(DEV)


def test_the_network_takes_a_star_off_and_leaves_the_sky(net):
    rng = np.random.default_rng(1)
    h = w = 256
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    sky = 4.0 + 3.0 * np.exp(-((xx - 90) ** 2 + (yy - 160) ** 2) / (2 * 50.0 ** 2))       # a smooth nebula, noise units
    img = np.repeat(sky[None], 3, 0) + 0.1 * rng.standard_normal((3, h, w)).astype(np.float32)   # denoised
    star = moffat(h, w, 150.3, 100.6, 300.0)
    out = sr.apply(net, img + star[None], DEV)
    assert out.shape == img.shape and out.dtype == np.float32
    box = np.s_[:, 90:112, 140:162]
    left = float((out[box] - img[box]).sum()) / float((3 * star[90:112, 140:162]).sum())
    assert left < 0.1, f"{left:.0%} of the star left"
    far = np.s_[:, 180:240, 20:80]                    # the nebula far from the star: unchanged
    assert abs(float(np.median(out[far] - img[far]))) < 0.3


def test_a_tiny_image_is_padded_to_one_tile(net):
    out = sr.apply(net, np.ones((3, 40, 70), np.float32), DEV)
    assert out.shape == (3, 40, 70) and np.isfinite(out).all()


def test_the_channel_attention_averages_over_a_window_only_when_applied():
    """Issue #45: applied to more than the training size, the attention's average is local (Test-time Local Converter);
    within the window, and in training, it is the average over the whole tile, as the network was trained."""
    pool = sr._LocalPool(16)
    x = torch.zeros(1, 2, 64, 64)
    x[:, :, :, 32:] = 1.0
    pool.eval()
    local = pool(x)
    assert local.shape == x.shape
    assert local[0, 0, 32, 2] == pytest.approx(0.0) and local[0, 0, 32, 61] == pytest.approx(1.0)
    assert local[0, 0, 32, 32] == pytest.approx(0.5, abs=0.07)                      # an even window: half a pixel off
    assert pool(x[:, :, 24:40, 24:40]).shape == (1, 2, 1, 1)                        # within the window: the whole tile
    pool.train()
    assert pool(x).shape == (1, 2, 1, 1) and pool(x)[0, 0, 0, 0] == pytest.approx(0.5)


def test_the_local_average_is_exact_in_half_precision():
    """Sums over a big tile in bfloat16 would lose the small values; the average of a constant stays the constant."""
    pool = sr._LocalPool(32).eval()
    x = torch.full((1, 1, 512, 512), 3.0, dtype=torch.bfloat16)
    assert torch.allclose(pool(x).float(), torch.full((1, 1, 512, 512), 3.0), rtol=1e-3)


def test_the_window_follows_each_level_of_the_network():
    """The window is the training size in the pixels of each level, and the bundled network's weights still load."""
    net = sr.NAFNet()
    pools = [net.encs[0][0].sca[0], net.encs[3][0].sca[0], net.mid[0].sca[0], net.decs[0][0].sca[0], net.decs[3][0].sca[0]]
    assert [p.window for p in pools] == [sr.WINDOW, sr.WINDOW // 8, sr.WINDOW // 16, sr.WINDOW // 8, sr.WINDOW]
    assert not any("sca.0" in k for k in net.state_dict())                          # no weight added: the checkpoint loads


class _Tiles(torch.nn.Module):
    """Stands for the network: the input unchanged, the tiles it was given noted."""

    def __init__(self):
        super().__init__()
        self.sizes = []

    def forward(self, x):
        self.sizes.append(tuple(x.shape))
        return x


def test_a_large_image_goes_in_tiles_larger_than_the_largest_stars():
    """Issue #45: the tiles hold the halo of the largest stars with their core; the same memory at a time; the result
    is the image when the network changes nothing (the tiles add up)."""
    fake = _Tiles()
    img = np.random.default_rng(3).normal(0.0, 1.0, (3, 1300, 1500)).astype(np.float32)
    out = sr.apply(fake, img, DEV)
    assert {s[2:] for s in fake.sizes} == {(sr.TILE, sr.TILE)} and {s[0] for s in fake.sizes} == {1}
    assert sr.TILE > 280                                                            # the largest halos measured (M45)
    assert np.allclose(out, img, atol=1e-3)


def test_a_small_image_keeps_the_tiles_of_the_training_size():
    fake = _Tiles()
    sr.apply(fake, np.ones((3, 600, 1500), np.float32), DEV)
    assert {s[2:] for s in fake.sizes} == {(sr.NET_TILE, sr.NET_TILE)}


def test_a_small_card_takes_smaller_tiles_on_a_large_image(monkeypatch):
    monkeypatch.setattr(sr.devices, "free_memory", lambda d: 600 * 2 ** 20)
    fake = _Tiles()
    out = sr.apply(fake, np.ones((3, 1300, 1500), np.float32), DEV)
    assert all(s[2] < sr.TILE for s in fake.sizes) and np.allclose(out, 1.0, atol=1e-3)


def _field(seed=0, h=256, w=256, noise=5.0):
    """Two half-stacks of a field (sky, a nebula, stars, independent noise) and their full stack, the mean of both."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    sky = 100.0 + 60.0 * np.exp(-((xx - 120) ** 2 + (yy - 130) ** 2) / (2 * 20.0 ** 2))
    stars = sum(moffat(h, w, x, y, a) for x, y, a in ((60.3, 50.6, 900.0), (190.1, 200.4, 300.0), (150.7, 80.2, 120.0)))
    clean = np.stack([sky + stars, 0.8 * sky + stars, 0.6 * sky + stars]).astype(np.float32)
    A, B = (clean + noise * rng.standard_normal(clean.shape).astype(np.float32) for _ in range(2))
    return clean, A, B, (A + B) / 2


def test_a_single_image_is_measured_in_a_stacks_noise_unit():
    """Issue #174: an image from a file is read as the full stack of a stack: its unit is the one its half-stacks would
    give (it was the sky spread after Denoise, 2.3 times smaller on M27); a pedestal changes nothing, the unit follows
    the scale of the values; an image without noise keeps a defined unit."""
    _clean, A, _B, full = _field()
    W = np.ones_like(full)
    M = np.diag([0.7, 1.0, 1.6])
    half = sr.noise_unit(A, W, M)
    assert half == pytest.approx(5.0 * np.mean([0.7, 1.0, 1.6]), rel=0.15)      # the nebula widens it a little
    assert sr.noise_unit(full, W, M, single=True) == pytest.approx(half, rel=0.1)
    assert sr.noise_unit(full * 37.0 + 900.0, W, M, single=True) == pytest.approx(
        37.0 * sr.noise_unit(full, W, M, single=True), rel=1e-4)
    assert sr.noise_unit(np.ones_like(full), W, M, single=True) == 1.0


def test_a_pedestal_and_a_value_scale_give_the_same_starless(net):
    """Issue #174: the same image with a pedestal (a black point undone) and its values scaled (x 65535) gives the same
    starless image, up to that pedestal and scale: the network sees the same input."""
    _clean, _A, _B, full = _field(seed=1)
    W, M, shown = np.ones_like(full), np.eye(3), np.ones(full.shape[1:], bool)
    k, ped = 65535.0 / 4.0, 983.0
    one, bk1 = sr.remove_stars(net, full, shown, sr.noise_unit(full, W, M, single=True), DEV)
    two, bk2 = sr.remove_stars(net, full * k + ped, shown, sr.noise_unit(full * k + ped, W, M, single=True), DEV)
    assert np.allclose(bk2, bk1 * k + ped, rtol=1e-4)
    assert np.abs((two - ped) / k - one).max() < 1e-3 * float(np.abs(one).max())
    assert float((full - one)[:, 51, 60].mean()) > 0.5 * 900.0               # and the star is taken off


def _folder(tmp_path, h=64, w=96):
    """A star_removal folder: a sky, a nebula and two stars, the starless image and the stars."""
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    starless = np.stack([np.full((h, w), 2.0, np.float32) + 30 * np.exp(-((xx - 30) ** 2 + (yy - 30) ** 2) / 300)] * 3)
    starless += 0.5 * rng.standard_normal(starless.shape).astype(np.float32)
    stars = np.stack([moffat(h, w, 70, 20, 400.0), moffat(h, w, 50, 50, 80.0)]).sum(0)
    stars = np.repeat(stars[None], 3, 0).astype(np.float32)
    whole = starless + stars
    valid = np.ones((h, w), bool)
    np.save(tmp_path / "starless.npy", starless)
    np.save(tmp_path / "stars.npy", stars)
    np.save(tmp_path / "valid.npy", valid)
    prep = {"crop": [0, w, 0, h], "M": np.eye(3).tolist(), "drizzle_scale": 1.0, "unit": 1.0,
            "background": [0, 0, 0], "stats_full": image_stats(whole, valid)}
    (tmp_path / "prep_stars.json").write_text(json.dumps(prep))
    return whole, starless, stars


def test_each_part_renders_and_the_recombined_image_is_close_to_the_whole(tmp_path):
    whole, *_ = _folder(tmp_path)
    eng = StarsRenderEngine(tmp_path, device="cpu")
    s = StarSettings()
    parts = {p: eng.render(s, 1.0, part=p).final.numpy() for p in ("starless", "stars", "combined")}
    for a in parts.values():
        assert a.shape == (3, 64, 96) and 0 <= a.min() and a.max() <= 1
    # the stars on black; the starless image without them; recombined, the whole image's look (the same curve)
    assert parts["stars"][:, 0:5, 0:5].max() < 0.02
    assert parts["stars"][:, 20, 70].mean() > 0.9 and parts["starless"][:, 20, 70].mean() < 0.4
    ref = SimpleRenderEngine.from_array(whole)
    ref.prep["stats"] = image_stats(whole, np.ones(whole.shape[1:], bool))
    full = ref.render(SimpleSettings(), 1.0).final.numpy()
    assert np.abs(parts["combined"] - full).mean() < 0.03


def test_the_star_sliders_move_what_they_say(tmp_path):
    _folder(tmp_path)
    eng = StarsRenderEngine(tmp_path, device="cpu")

    def stars(**kw):
        return eng.render(StarSettings(**kw), 1.0, part="stars").final.numpy()
    base = stars()
    assert stars(star_strength=0.5).sum() < base.sum() < stars(star_strength=2.0).sum()
    assert stars(star_size=-2.0).sum() < base.sum() < stars(star_size=0.0).sum()       # -1 by default
    grey = stars(star_saturation=0.0)
    assert np.abs(grey - grey.mean(0, keepdims=True)).max() < 1e-5          # no colour left


def test_blends_and_the_size_change():
    a = torch.full((3, 4, 4), 0.5)
    b = torch.zeros((3, 4, 4))
    b[:, 1, 1] = 0.5
    assert torch.allclose(blend(a, b, "Screen")[:, 1, 1], torch.tensor(0.75))
    assert torch.allclose(blend(a, b, BLENDS[1])[:, 1, 1], torch.tensor(1.0))
    assert torch.allclose(blend(a, b, "Lighten")[:, 1, 1], torch.tensor(0.5))
    assert torch.equal(blend(a, b, "Screen")[:, 0, 0], a[:, 0, 0])            # black stars change nothing
    y = torch.zeros((3, 9, 9))
    y[:, 4, 4] = 1.0
    assert resize_stars(y, 1.0).sum() == 15 and torch.equal(resize_stars(y, -1.0), y)  # a cross: round, not a square;
    #                                         a star as narrow as the reduction is left alone (#169)
    assert resize_stars(y, 2.0)[0].bool().sum() == 21                                    # then a square: an octagon
    assert 3 < resize_stars(y, 0.5).sum() < 15                                 # half a pixel: half way


def test_smaller_stars_leave_the_grain_of_a_dense_field_as_it_is():
    """Issue #18: the stars shrink through a star mask (the reference image editor's Minimum through a selection of the stars,
    the reference processing suite's MorphologicalTransformation through a star mask): the faint grain between the stars of a dense
    field, above STAR_LEVEL but within its noise, is not eroded into a web of filaments."""
    g = torch.Generator().manual_seed(3)
    y = (0.04 + 0.03 * torch.randn((1, 96, 96), generator=g)).clamp(0, 1).repeat(3, 1, 1)
    yy, xx = torch.meshgrid(torch.arange(96.0), torch.arange(96.0), indexing="ij")
    star = 0.9 * torch.exp(-((yy - 48) ** 2 + (xx - 48) ** 2) / (2 * 2.0 ** 2))
    y = (y + star).clamp(0, 1)
    out = resize_stars(y, -1.0)
    far = ((yy - 48) ** 2 + (xx - 48) ** 2) > 12 ** 2
    moved = (out - y).abs()[:, far].mean() / (_morph(y, 1, False) - y).abs()[:, far].mean()
    assert moved < 0.1, moved                                                  # the grain as it was: under a tenth of
    #                                                                            what the erosion alone takes (#30: 6 %)
    assert out[:, 46:51, 46:51].sum() < 0.85 * y[:, 46:51, 46:51].sum()        # the star smaller


def test_a_wider_star_shrinks_whole_without_a_ring():
    """Issue #30: the sky of the star mask is measured with the stars clipped out, so a wider star's wings stand above
    it as its core does: the star shrinks whole, its radial profile still falls outward (no ring)."""
    yy, xx = torch.meshgrid(torch.arange(80.0), torch.arange(80.0), indexing="ij")
    r2 = (yy - 40) ** 2 + (xx - 40) ** 2
    y = (0.03 + 0.95 * torch.exp(-r2 / (2 * 3.0 ** 2))).clamp(0, 1).repeat(3, 1, 1)       # FWHM 7 px
    out = resize_stars(y, -2.0)[0]
    rr = r2.sqrt().round().long()
    prof = torch.stack([out[rr == k].mean() for k in range(0, 20)])
    assert torch.all(prof[1:] <= prof[:-1] + 1e-3), prof                       # falls outward: no ring
    assert (out - 0.03).sum() < 0.85 * (y[0] - 0.03).sum()                      # and it is smaller


def test_the_sliders_of_each_kind():
    ok = S.availability("combined", True)
    assert not any(ok[k] for k in ("sky", "contrast", "saturation", "star_strength", "star_small", "star_size",
                                   "star_color"))
    assert ok["noise"] == S.WHY_WHOLE
    stars = S.availability("stars", True)
    assert stars["sky"] == S.WHY_STARS_ONLY and not stars["star_size"]
    simple = S.availability("simple", True)
    assert simple["star_strength"] == S.WHY_STARS and not simple["noise"]
    assert S.availability("stacking", False)["star_color"] == S.WHY_EXIT_STARS
    st = S.settings_at("combined", None, {"star_strength": 0, "star_size": 4})
    assert isinstance(st, StarSettings) and st.star_strength == pytest.approx(0.25) and st.star_size == 1.0
    assert type(S.settings_at("simple", None, {"star_size": 0})) is SimpleSettings      # greyed: ignored
    with pytest.raises(ValueError):
        S.availability("engine", True)


def test_the_stars_block_comes_after_the_output_and_reads_the_old_options():
    full = {"denoise": "pretrained", "stars": "neural"}
    assert wfl.chain(full)[-2:] == ["render_simple", "star_removal"]
    for old in ("psf", "psf_neural"):
        assert wfl.check_options({**full, "stars": old})["stars"] == "neural"
    assert "star_removal" not in wfl.chain({"denoise": "pretrained"})
    with pytest.raises(ValueError):
        wfl.check_options({"stars": "neural"})                                # Stars needs Denoise
    o, notes = wfl.set_option({"denoise": None}, "stars", "neural")
    assert o["denoise"] == "pretrained" and notes == [wfl.NOTE_DENOISE_ADDED]
    c = wfl.chains()
    assert [b for b, _ in c.blocks_in(wfl.chain(full))][-2:] == ["stars", "output"]   # drawn before the Output


def _graded_field(n=60, seed=3):
    """n round stars of widths from 1.0 to 4.0 px (sigma), peaks 0.4 .. 0.9, on a 40-pixel grid of cells (touching
    neighbours' wings): their positions and sigmas."""
    yy, xx = torch.meshgrid(torch.arange(240.0), torch.arange(240.0), indexing="ij")
    rng = np.random.default_rng(seed)
    cells = [(cy, cx) for cy in range(15, 240, 30) for cx in range(15, 240, 30)][:n]
    sig = np.linspace(1.0, 4.0, len(cells))
    rng.shuffle(sig)
    y = torch.zeros(240, 240)
    for (cy, cx), s_ in zip(cells, sig):
        y += rng.uniform(0.4, 0.9) * torch.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * s_ * s_))
    return y.clamp(0, 1).repeat(3, 1, 1), cells, sig


def test_small_stars_is_the_share_of_the_stars_removed_the_smallest_first():
    """The setting is a share of the field's stars: at f about f of them are gone (a star is gone when its peak lost
    half), the smallest first and the biggest last; 0 changes nothing, 1 leaves nothing."""
    from astro_gpu.render.stars import fade_small_stars
    y, cells, sig = _graded_field()
    assert torch.equal(fade_small_stars(y, 0.0, 3.0), y)
    assert float(fade_small_stars(y, 1.0, 3.0).abs().max()) == 0.0
    order = np.argsort(sig)
    for f in (0.25, 0.5, 0.75):
        z = fade_small_stars(y, f, 3.0)
        gone = np.array([float(z[0, cy, cx]) < 0.5 * float(y[0, cy, cx]) for cy, cx in cells])
        assert abs(gone.mean() - f) < 0.1, (f, gone.mean())
    z = fade_small_stars(y, 0.5, 3.0)
    small = order[:len(cells) // 4]
    big = order[-len(cells) // 4:]
    assert all(float(z[0, cells[i][0], cells[i][1]]) < 0.5 * float(y[0, cells[i][0], cells[i][1]]) for i in small)
    assert all(float(z[0, cells[i][0], cells[i][1]]) > 0.7 * float(y[0, cells[i][0], cells[i][1]]) for i in big)
    assert (z <= y + 1e-6).all()                                                    # nothing is added: no disc


def test_small_stars_share_is_the_same_on_a_dense_field():
    """M27 (a dense field): touching faint stars, all above the stars' level, are ranked as the sparse ones are: the
    share removed is the setting, the large star among them stays."""
    from astro_gpu.render.stars import fade_small_stars, star_sizes
    yy, xx = torch.meshgrid(torch.arange(160.0), torch.arange(160.0), indexing="ij")
    rng = np.random.default_rng(5)
    y = sum(rng.uniform(0.2, 0.6) * torch.exp(-((yy - rng.uniform(5, 155)) ** 2 + (xx - rng.uniform(5, 155)) ** 2)
                                                / (2 * rng.uniform(0.9, 1.4) ** 2)) for _ in range(500))
    y = y + 0.9 * torch.exp(-((yy - 80) ** 2 + (xx - 80) ** 2) / (2 * 6.0 ** 2))
    y = y.clamp(0, 1).repeat(3, 1, 1)
    ys, xs, size = star_sizes(y, 3.0)
    z = fade_small_stars(y, 0.5, 3.0)
    gone = (z[0][ys, xs] < 0.5 * y[0][ys, xs]).float().mean()
    assert abs(float(gone) - 0.5) < 0.1
    assert float(z[0, 80, 80]) > 0.8 * float(y[0, 80, 80])                           # the large star stays


def test_a_star_size_is_half_its_width():
    from astro_gpu.render.stars import measure_fwhm, star_sizes
    yy, xx = torch.meshgrid(torch.arange(60.0), torch.arange(60.0), indexing="ij")
    y = (0.9 * torch.exp(-((xx - 30) ** 2 + (yy - 30) ** 2) / (2 * 2.0 ** 2))).repeat(3, 1, 1)
    _, _, size = star_sizes(y, 4.7)
    assert abs(float(size[0]) - 1.18 * 2.0) < 0.4                                   # FWHM / 2 = 1.18 sigma
    rng = np.random.default_rng(2)
    img = np.zeros((300, 300), np.float32)
    gy, gx = np.mgrid[0:300, 0:300]
    for _ in range(200):
        cy, cx, a = rng.uniform(10, 290), rng.uniform(10, 290), rng.uniform(0.1, 1)
        img += a * np.exp(-((gx - cx) ** 2 + (gy - cy) ** 2) / (2 * 2.0 ** 2))
    assert abs(measure_fwhm(torch.from_numpy(np.stack([img] * 3))) - 2.355 * 2.0) < 0.35   # the width at half height


def test_the_smallest_size_leaves_faint_stars_and_chains_as_they_are():
    """Issue #169: the reduction erodes only the stars wider than what it takes away (``wide_light``, the grey
    opening, place by place): a faint star and a chain of touching faint stars are the same to the pixel at the smallest size
    (no worm, no arc), a large star is narrower."""
    yy, xx = torch.meshgrid(torch.arange(120.0), torch.arange(120.0), indexing="ij")

    def star(cy, cx, s, a):
        return a * torch.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * s * s))

    y = star(20, 20, 0.9, 0.5) + sum(star(60, 15 + 2.0 * i, 0.9, 0.4) for i in range(6)) + star(90, 80, 3.5, 0.95) \
        + sum(star(90, 40 + 2.0 * i, 0.9, 0.4) for i in range(14))
    y = y.clamp(0, 1).repeat(3, 1, 1)
    out = resize_stars(y, -3.0)
    assert (out[:, :40, :] - y[:, :40, :]).abs().max() < 1e-3                    # the faint star: the same to the pixel
    assert (out[:, 45:75, :40] - y[:, 45:75, :40]).abs().max() < 1e-3          # the chain: no arc, no line
    assert (out[:, 80:100, 36:66] - y[:, 80:100, 36:66]).abs().max() < 1e-3      # a chain joined to the wide star: its
    #                                                                            narrow beads stay (not regrown from it)
    width = lambda im: int((im[0, 70:110, 70:120].max(0).values > 0.5 * im[0, 90, 80]).sum())    # width at half height
    assert width(out) < width(y) - 1, (width(out), width(y))                   # the large star shrinks


def test_the_smallest_size_leaves_a_faint_star_on_a_wide_one_as_it_was():
    """#174 (M27, a dense field): a faint star on the light of a wider one (its wing, the glow of a crowded field)
    is not thinned by the reduction: only the light wider than the erosion is eroded, the faint star's own light goes
    back on top as it was. A weight of the opening over the light (1 on the faint star's flanks, low on its core)
    eroded its flanks and spared its core: a fifth of its light lost and the rest a thinner core, faint stars drawn
    as broken chains."""
    yy, xx = torch.meshgrid(torch.arange(80.0), torch.arange(80.0), indexing="ij")

    def star(cy, cx, s, a):
        return a * torch.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * s * s))

    wide = star(40, 40, 8.0, 0.5)
    faint = star(40, 50, 1.0, 0.15) + star(41.5, 51.5, 1.0, 0.12)            # a faint pair of bumps on its wing
    own = (resize_stars((wide + faint).repeat(3, 1, 1), -1.0) - resize_stars(wide.repeat(3, 1, 1), -1.0))[0]
    win = (slice(34, 48), slice(44, 58))
    assert float(own[win].sum() / faint[win].sum()) > 0.97                   # its light kept (the old weight: 0.80)
    assert torch.equal(resize_stars(wide.repeat(3, 1, 1), 0.0), wide.repeat(3, 1, 1))
