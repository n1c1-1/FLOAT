"""The Align on block: a workflow's image brought onto another's grid by their stars or their plate solutions, of any size,
rotation and scale; the files it writes, the reasons it refuses."""
from __future__ import annotations

import json

import numpy as np
import pytest
from astropy.io import fits

from astro_gpu.core.journal import Journal
from astro_gpu.core.stage import Context, StageError
from astro_gpu.stages import align_on as ai
from tests.test_blocks43 import _Project

H, W = 400, 600


def field(n=60, seed=1):
    rng = np.random.default_rng(seed)
    return np.c_[rng.uniform(30, W + 60, n), rng.uniform(30, H + 60, n), rng.uniform(0.2, 1.0, n)]


def render(stars, shape, angle=0.0, scale=1.0, shift=(0.0, 0.0), sigma=1.6, seed=2):
    """An image of the stars seen through a similarity about the origin."""
    h, w = shape
    c, s = np.cos(np.radians(angle)) * scale, np.sin(np.radians(angle)) * scale
    x = c * stars[:, 0] - s * stars[:, 1] + shift[0]
    y = s * stars[:, 0] + c * stars[:, 1] + shift[1]
    rng = np.random.default_rng(seed)
    g = rng.normal(0.02, 0.0008, (h, w)).astype(np.float32)
    r = int(sigma * 5)
    for x0, y0, f in zip(x, y, stars[:, 2]):
        ix, iy = int(round(x0)), int(round(y0))
        if r <= ix < w - r and r <= iy < h - r:
            yy, xx = np.mgrid[iy - r:iy + r + 1, ix - r:ix + r + 1]
            g[iy - r:iy + r + 1, ix - r:ix + r + 1] += f * np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * sigma ** 2))
    return np.repeat(g[None], 3, axis=0), np.c_[x, y]


def positions(a):
    c = ai.stars_of(a)
    return c[:, :2]


def stack_dir(folder, img, halves=True, stretched=False):
    """A result folder as Stacking (or an Image block) writes it."""
    folder.mkdir(parents=True)
    np.save(folder / "drz_full.npy", img)
    if halves:
        np.save(folder / "drz_A.npy", img)
        np.save(folder / "drz_B.npy", img)
    np.save(folder / "drz_weight.npy", np.ones_like(img))
    np.save(folder / "saturation.npy", np.zeros((2,) + img.shape[1:], np.uint8))
    meta = {"scale": 1.0, "width": img.shape[2], "height": img.shape[1]}
    (folder / "drz_meta.json").write_text(json.dumps(meta | ({"stretched": True} if stretched else {})))
    return folder


def align(tmp_path, img, ref, method="stars", own_file="", ref_file="", name="out", **kw):
    """Run the block on ``img`` (an own stack) with ``ref`` as the reference's image: the folder it wrote."""
    own = stack_dir(tmp_path / f"own_{name}", img, **kw)
    rdir = stack_dir(tmp_path / f"ref_{name}", ref)
    out = tmp_path / name
    pr = _Project(tmp_path, {"stack": own, "align_on": out}, ["stack", "align_on"])
    st = ai.AlignOn()
    ctx = Context(pr, st, Journal(pr.logs), "cpu")
    st.run(ctx, ai.AlignOnParams(method=method, reference={"pid": 2, "dir": str(rdir), "wcs": ref_file, "path": own_file}))
    return out


@pytest.mark.parametrize("angle,scale,shift,shape", [(0, 1, (11.3, -7.8), (H, W)),
                                                     (32, 1, (150.0, -60.0), (H + 100, W + 100)),
                                                     (-8, 0.7, (40.0, 30.0), (360, 520))])
def test_an_image_lands_on_the_reference_through_its_stars(tmp_path, angle, scale, shift, shape):
    stars = field()
    ref, _ = render(stars, (H, W))
    img, _ = render(stars, shape, angle, scale, shift, seed=3)
    out = align(tmp_path, img, ref)
    got_img = np.load(out / "drz_full.npy")
    weight = np.load(out / "drz_weight.npy")
    assert got_img.shape == ref.shape and weight.shape == ref.shape
    want, got = positions(ref), positions(got_img)
    d, _i = __import__("scipy.spatial", fromlist=["cKDTree"]).cKDTree(want).query(got)
    assert (d < 1.2).sum() >= 0.7 * len(got) and len(got) >= 15
    valid = weight[0] > 0
    if scale != 1 or angle:
        assert not valid.all()                                    # outside the image: marked empty
    assert (got_img[:, ~valid] == 0).all()
    meta = json.loads((out / "drz_meta.json").read_text())
    assert (meta["width"], meta["height"], meta["aligned_on"]) == (W, H, 2)


def test_it_writes_the_files_of_the_stage_it_reads_on_the_references_grid(tmp_path):
    stars = field()
    ref, _ = render(stars, (H, W))
    img, _ = render(stars, (H + 50, W + 80), 5, 1.0, (20.0, 10.0), seed=3)
    out = align(tmp_path, img, ref)
    for f in ("drz_full.npy", "drz_A.npy", "drz_B.npy", "drz_weight.npy"):
        assert np.load(out / f).shape == (3, H, W), f
    sat = np.load(out / "saturation.npy")
    assert sat.shape == (2, H, W) and sat.dtype == np.uint8
    only_full = align(tmp_path, img, ref, name="o2", halves=False)       # an image from a file has no half-stacks
    assert not (only_full / "drz_A.npy").exists() and (only_full / "drz_full.npy").is_file()


def test_two_plate_solutions_are_used_when_both_files_have_one(tmp_path):
    from astropy.wcs import WCS

    def solved(crpix):
        w = WCS(naxis=2)
        w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        w.wcs.crval = [83.8, -5.4]
        w.wcs.crpix = crpix
        w.wcs.cdelt = [-0.001, 0.001]
        return w
    g = np.zeros((3, H, W), np.float32)
    g[:, 200, 300] = 1.0
    ref_f, img_f = tmp_path / "r.fits", tmp_path / "a.fits"
    fits.PrimaryHDU(g, solved([300, 200]).to_header()).writeto(ref_f)
    fits.PrimaryHDU(np.roll(g, (10, 20), (1, 2)), solved([320, 210]).to_header()).writeto(img_f)
    out = align(tmp_path, np.roll(g, (10, 20), (1, 2)), g, "plate", str(img_f), str(ref_f))
    a = np.load(out / "drz_full.npy")
    y, x = np.unravel_index(a[0].argmax(), a[0].shape)
    assert abs(x - 300) <= 1 and abs(y - 200) <= 1
    with pytest.raises(StageError, match="plate solution yet"):                  # a stack has none: no solving in the block
        align(tmp_path, g, g, "plate", "", str(ref_f), name="stack")
    plain = tmp_path / "plain.fits"
    fits.PrimaryHDU(g).writeto(plain)
    with pytest.raises(StageError, match="reference image has no plate solution"):
        align(tmp_path, g, g, "plate", str(img_f), str(plain), name="noref")
    with pytest.raises(StageError, match="this image has no plate solution"):
        align(tmp_path, g, g, "plate", str(plain), str(ref_f), name="noown")


def test_an_image_without_its_stars_is_said_plainly(tmp_path):
    ref, _ = render(field(), (H, W))
    blank = np.full_like(ref, 0.1)
    with pytest.raises(StageError, match="too few stars"):
        align(tmp_path, blank, ref)


def test_two_different_fields_are_refused(tmp_path):
    ref, _ = render(field(seed=1), (H, W))
    img, _ = render(field(seed=9), (H, W), seed=5)
    with pytest.raises(StageError, match="not the same field"):
        align(tmp_path, img, ref)


def test_a_block_without_its_reference_says_what_to_wire(tmp_path):
    img, _ = render(field(), (H, W))
    own = stack_dir(tmp_path / "own", img)
    pr = _Project(tmp_path, {"stack": own, "align_on": tmp_path / "out"}, ["stack", "align_on"])
    st = ai.AlignOn()
    with pytest.raises(StageError, match="reference to wire"):
        st.run(Context(pr, st, Journal(pr.logs), "cpu"), ai.AlignOnParams())


def test_a_stretched_image_keeps_the_files_an_output_writes(tmp_path):
    stars = field()
    ref, _ = render(stars, (H, W))
    img, _ = render(stars, (H, W), 3, 1.0, (4.0, 2.0), seed=3)
    own = stack_dir(tmp_path / "own", np.clip(img * 4, 0, 1).astype(np.float32), stretched=True)
    (own / "prep_simple.json").write_text(json.dumps({"source": "done.png"}))
    rdir = stack_dir(tmp_path / "ref", ref)
    out = tmp_path / "out"
    pr = _Project(tmp_path, {"stack": own, "align_on": out}, ["image", "align_on"])
    st = ai.AlignOn()
    st.run(Context(pr, st, Journal(pr.logs), "cpu"), ai.AlignOnParams(reference={"pid": 2, "dir": str(rdir)}))
    assert np.load(out / "lin.npy").shape == (3, H, W) and np.load(out / "valid.npy").shape == (H, W)
    prep = json.loads((out / "prep_simple.json").read_text())
    assert prep["stretched"] and prep["width"] == W and prep["source"] == "done.png"
    assert json.loads((out / "drz_meta.json").read_text())["stretched"]


def test_two_image_workflows_run_through_the_store_with_the_second_aligned_on_the_first(tmp_path):
    """The whole path of the block: keys, the reference's folder handed over by the overrides, the stage reading the image
    it follows, the aligned folder read by the stage after it."""
    from astro_gpu.core import project2
    from astro_gpu.core.stage import Context as Ctx
    from astro_gpu.core.store import Keys, Store, StoreProject
    from astro_gpu.core.workflow import TEMPLATES
    from astro_gpu.stages.image import ImageSource
    stars = field()
    ref, _ = render(stars, (H, W))
    img, _ = render(stars, (H + 40, W + 60), 6, 1.0, (30.0, 12.0), seed=3)
    for name, a in (("ref", ref), ("img", img)):
        fits.PrimaryHDU(a * 60000).writeto(tmp_path / f"{name}.fits")
    p = project2.create(tmp_path / "p", "p")
    a = p.add_workflow("ref", TEMPLATES["Classic stack"])
    b = p.add_workflow("img", TEMPLATES["Classic stack"])
    a.source, b.source = {"path": str(tmp_path / "ref.fits")}, {"path": str(tmp_path / "img.fits")}
    project2._add_align_on(p, b, a.id, "stars")
    keys = Keys(p)
    store = Store(p.root)

    def run(wf, stage, stage_obj):
        sp = StoreProject(keys, wf, stage)
        ctx = Ctx(sp, stage_obj, Journal(sp.logs), "cpu")
        stage_obj.run(ctx, type(stage_obj).Params(**keys.overrides(wf, stage)))
        done = store.dir(stage, keys.key(wf, stage))
        sp.own_dir().rename(done)
        (done / "manifest.json").write_text(json.dumps({"status": "done", "key": keys.key(wf, stage)}))
        return done
    run(a.id, "image", ImageSource())
    run(b.id, "image", ImageSource())
    assert keys.key(b.id, "align_on") != keys.key(b.id, "image")
    out = run(b.id, "align_on", ai.AlignOn())
    got = np.load(out / "drz_full.npy")
    assert got.shape == (3, H, W)
    d, _i = __import__("scipy.spatial", fromlist=["cKDTree"]).cKDTree(positions(np.load(
        store.dir("image", keys.key(a.id, "image")) / "drz_full.npy"))).query(positions(got))
    assert (d < 1.2).sum() >= 0.7 * len(d)
    after = StoreProject(keys, b.id, "astrometry")                          # what a block after it opens
    assert np.load(after.stage_dir("stack") / "drz_full.npy").shape == (3, H, W)
    assert json.loads((after.stage_dir("stack") / "drz_meta.json").read_text())["aligned_on"] == a.id
