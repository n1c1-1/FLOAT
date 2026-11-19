"""Issue #174: Denoise on a single image (the Image block's: no half-stacks). Pretrained only, the noise measured on the
image, the dither check skipped with a note in the log, Adapted refused with its reason; Denoise is placeable after an
Image (linear), not after a stretched one. A project of a workflow with the graph [Image, Denoise, Output]."""
from __future__ import annotations

import json

import numpy as np
import pytest
from astropy.io import fits

from astro_gpu.core import project2
from astro_gpu.core import workflow as wfl
from astro_gpu.core.journal import Journal
from astro_gpu.core.stage import Context, StageError
from astro_gpu.core.store import Keys, Store, StoreProject
from astro_gpu.stages import denoise as dn
from astro_gpu.stages.image import ImageSource

H, W = 224, 256


def noisy_image(sigma=0.004):
    rng = np.random.default_rng(5)
    a = 0.03 + sigma * rng.standard_normal((3, H, W)).astype(np.float32)
    yy, xx = np.mgrid[:H, :W]
    for _ in range(25):
        y0, x0 = rng.integers(10, H - 10), rng.integers(10, W - 10)
        a += rng.uniform(0.1, 0.6) * np.exp(-((yy - y0) ** 2 + (xx - x0) ** 2) / 4.0)
    return np.clip(a, 0, 1).astype(np.float32)


@pytest.fixture
def project(tmp_path):
    fits.PrimaryHDU(noisy_image()).writeto(tmp_path / "img.fits")
    p = project2.create(tmp_path / "p", "p")
    w = p.add_workflow("img", wfl.TEMPLATES["Classic stack"])
    w.source = {"path": str(tmp_path / "img.fits"), "stretched": False}
    w.blocks = [wfl.node("denoise", "pretrained"), wfl.node("output")]
    return p, w


def run(keys, store, wf, stage, obj, **over):
    sp = StoreProject(keys, wf, stage)
    ctx = Context(sp, obj, Journal(sp.logs), "cpu")
    values = {**keys.overrides(wf, stage), **over}
    obj.run(ctx, type(obj).Params(**values))
    done = store.dir(stage, keys.key(wf, stage))
    sp.own_dir().rename(done)
    (done / "manifest.json").write_text(json.dumps({"status": "done", "key": keys.key(wf, stage)}))
    return done


def test_a_single_image_is_denoised_with_its_own_noise_and_says_the_dither_check_is_skipped(project):
    p, w = project
    keys, store = Keys(p), Store(p.root)
    assert keys.chain(w.id)[:2] == ["image", "denoise"]
    run(keys, store, w.id, "image", ImageSource())
    done = run(keys, store, w.id, "denoise", dn.Denoise())
    out = np.load(done / "n2n_full.npy")
    assert out.shape == (3, H, W) and np.isfinite(out).all()
    metrics = {}
    log = (next(done.parent.parent.glob("**/*.log"), None) or done).name        # (the journal is read below)
    sp_logs = list(p.root.rglob("*.log")) + list(p.root.rglob("*.jsonl"))
    text = "\n".join(f.read_text(encoding="utf-8", errors="ignore") for f in sp_logs)
    assert "dither check is skipped" in text
    img = np.load(store.dir("image", keys.key(w.id, "image")) / "drz_full.npy")
    assert dn._noise(out[1]) < 0.8 * dn._noise(img[1])                      # the noise measured on the image: reduced


def test_adapted_is_refused_on_a_single_image_with_its_reason(project):
    p, w = project
    keys, store = Keys(p), Store(p.root)
    run(keys, store, w.id, "image", ImageSource())
    with pytest.raises(StageError) as e:
        run(keys, store, w.id, "denoise", dn.Denoise(), method="adapted")
    assert dn.WHY_NO_ADAPTED in str(e.value)


def test_denoise_is_placeable_after_a_linear_image_not_a_stretched_one():
    img = [wfl.node("output")]
    assert wfl.palette_problem(img, "image", "denoise") is None
    assert wfl.palette_problem(img, "stretched", "denoise") is not None
    assert wfl.palette_problem(img, "image", "deconv").template == wfl.WHY_HALVES      # still needs half-stacks


def test_a_stack_of_frames_keeps_its_denoise_untouched():
    """The single-image path is chosen by the stack's metadata only (source "image"): a stack's is not."""
    class Ctx:
        def load_json(self, st, name):
            return {"scale": 2.0, "source": "frames"}
    assert not dn.single_image(Ctx())
