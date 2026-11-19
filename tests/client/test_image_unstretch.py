"""Issue #174, the Image block with a file already stretched: the warning (and its badge), the exact undo from the header
done silently, the estimated inverse as a separate option greyed with its reason, the keys of existing projects unchanged.
Offscreen on scratch projects; the stage runs through the store on small synthetic images."""
from __future__ import annotations

import json
import os

import numpy as np
import pytest
from astropy.io import fits

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

from astro_gpu.core import project2  # noqa: E402
from astro_gpu.core import workflow as wfl  # noqa: E402
from astro_gpu.core.journal import Journal  # noqa: E402
from astro_gpu.core.stage import Context  # noqa: E402
from astro_gpu.core.store import Keys, Store, StoreProject  # noqa: E402
from astro_gpu.stages import unstretch as U  # noqa: E402
from astro_gpu.stages.image import ImageSource  # noqa: E402
from tests.client.test_image_source_fixes import model, qapp  # noqa: E402,F401
from tests.test_unstretch import linear_image, write_fits  # noqa: E402

MTF = "Linked MTF stretch (mid=0.250000, lo=0.002000, hi=1.000000)"


def stretched_file(path, line=MTF):
    lin = linear_image(sky=0.02, sigma=0.01)
    return lin, write_fits(path, U.mtf(lin, 0.25, 0.002, 1.0).astype(np.float32), [line] if line else [])


def autostretched(path, line=None):
    """A stack stretched as the reference program's automatic stretch does (the sky median at 0.25), with no history."""
    lin = linear_image(mono=True, sigma=0.004, sky=0.03)[0].astype(np.float64)
    lin = np.repeat(lin[None], 3, axis=0)
    med = float(np.median(lin))
    sig = 1.4826 * float(np.median(np.abs(lin - med)))
    lo = med - 2.8 * sig
    m = float(U.mtf(np.array(med - lo), 0.25, 0.0, 1.0))
    return lin, write_fits(path, U.mtf(lin, m, lo, 1.0).astype(np.float32), [line] if line else [])


def run_image(project, w):
    keys, store = Keys(project), Store(project.root)
    sp = StoreProject(keys, w.id, "image")
    ctx = Context(sp, ImageSource(), Journal(sp.logs), "cpu")
    ImageSource().run(ctx, ImageSource.Params(**keys.overrides(w.id, "image")))
    done = store.dir("image", keys.key(w.id, "image"))
    sp.own_dir().rename(done)
    (done / "manifest.json").write_text(json.dumps({"status": "done"}))
    text = "\n".join(f.read_text(encoding="utf-8", errors="ignore") for f in project.root.rglob("*.jsonl"))
    return np.load(done / "drz_full.npy"), text


def make(tmp_path, path, **source):
    p = project2.create(tmp_path / "p", "p")
    w = p.add_workflow("img", wfl.TEMPLATES["Classic stack"])
    w.source = {"path": str(path), "stretched": False, **source}
    return p, w


# ------------------------------------------------------------------------------------------------ the stage

def test_the_exact_undo_is_automatic_silent_and_one_log_line(tmp_path):
    lin, f = stretched_file(tmp_path / "s.fits")
    p, w = make(tmp_path, f)
    out, log = run_image(p, w)
    y = U.mtf(lin, 0.25, 0.002, 1.0)
    ok = (y > 0) & (y < 1)
    np.testing.assert_allclose(out[ok] / 65535.0, lin[ok], atol=3e-5)
    assert "stretch undone from the header" in log
    assert Keys(p).key_parts(w.id, "image")["x"][-1] == "unstretch-header"


def test_the_estimated_undo_is_asked_for(tmp_path):
    lin, f = autostretched(tmp_path / "s.fits")                       # no history: the exact one does not apply
    p, w = make(tmp_path, f, unstretch="estimated")
    out, log = run_image(p, w)
    assert "undone by estimate" in log
    assert Keys(p).key_parts(w.id, "image")["x"][-1] == "unstretch-estimated"
    e = out[1] / 65535.0
    A = np.stack([lin[1].ravel(), np.ones(lin[1].size)], axis=1)
    coef, *_ = np.linalg.lstsq(A, e.ravel(), rcond=None)
    assert np.std(e.ravel() - A @ coef) / coef[0] < 0.01


def test_keys_of_existing_projects_do_not_move(tmp_path):
    """A linear file, and a stretched file without a header's stretch, keep the key part they had (file id only)."""
    fits.PrimaryHDU(linear_image(sky=0.02, sigma=0.01)).writeto(tmp_path / "lin.fits")
    p, w = make(tmp_path, tmp_path / "lin.fits")
    k = Keys(p)
    fid = k.scan.image_id(str(tmp_path / "lin.fits"))
    assert k.key_parts(w.id, "image")["x"] == fid
    assert "unstretch" not in json.dumps(k.key_parts(w.id, "image")["p"])
    _l, f = stretched_file(tmp_path / "nohist.fits", line=None)
    p2, w2 = make(tmp_path / "two", f)
    assert Keys(p2).key_parts(w2.id, "image")["x"] == Keys(p2).scan.image_id(str(f))


def test_source_check_accepts_only_the_estimate_on_a_linear_image():
    project2.check_source({"path": "a.fits", "stretched": False, "unstretch": "estimated"})
    for bad in ({"path": "a.fits", "unstretch": "header"},
                {"path": "a.fits", "stretched": True, "unstretch": "estimated"}):
        with pytest.raises(ValueError):
            project2.check_source(bad)


# ------------------------------------------------------------------------------------------------ the block

def test_warning_and_options_of_the_image_block(model, tmp_path):
    """A stretched file with no usable header: a warning, the estimate offered; the exact header: no warning, the estimate
    greyed with its reason; a linear file: no warning; kind Stretched: the estimate greyed."""
    _l, bright = autostretched(tmp_path / "bright.fits")
    _l, exact = stretched_file(tmp_path / "exact.fits")
    _l, coarse = autostretched(tmp_path / "auto.fits", "Autostretch (shadows: -2.80, target bg: 0.25, linked)")
    fits.PrimaryHDU(linear_image(sky=0.02, sigma=0.01)).writeto(tmp_path / "lin.fits")
    q = model.add_image_workflow("Image", "")
    model.set_graph(q, [])
    model.set_image_file(q, str(bright))
    st = model.image_state(q)
    assert "% of white" in st["warning"] and st["estimate_off"] == "" and model.image_warning(q)
    model.set_image_file(q, str(coarse))
    st = model.image_state(q)
    assert "Autostretch" in st["warning"] and st["estimate_off"] == ""
    model.set_image_file(q, str(exact))
    st = model.image_state(q)
    assert st["warning"] == "" and "undone automatically" in st["estimate_off"]
    model.set_image_file(q, str(tmp_path / "lin.fits"))
    st = model.image_state(q)
    assert st["warning"] == "" and "does not look stretched" in st["estimate_off"]
    model.set_image_file(q, str(bright))
    model.set_image_file(q, unstretch="estimated")
    assert model.p.workflow(q.pid).source["unstretch"] == "estimated"
    assert "by estimate" in model.image_state(q)["warning"]
    model.set_image_file(q, stretched=True)                        # the kind Stretched: no stretch to undo
    assert "unstretch" not in model.p.workflow(q.pid).source
    assert "choose Linear first" in model.image_state(q)["estimate_off"]
    model.set_image_file(q, stretched=False)
    model.set_image_file(q, unstretch="estimated")
    model.set_image_file(q, str(exact))                            # a new file drops an estimate
    assert "unstretch" not in model.p.workflow(q.pid).source


def test_the_badge_of_the_image_block(model, tmp_path):
    """The warning marks the Image block (the badge heading its summary), as other warnings do."""
    from astro_gpu.client.graphview.page import GraphPage
    from tests.client.pagehelp import item
    _l, bright = autostretched(tmp_path / "bright.fits")
    _l, exact = stretched_file(tmp_path / "exact.fits")
    q = model.add_image_workflow("Image", str(bright))
    model.set_graph(q, [])
    page = GraphPage(model)
    try:
        assert item(page, key="image").d["warned"]
    finally:
        page.close()
    model.set_image_file(q, str(exact))
    page = GraphPage(model)
    try:
        assert not item(page, key="image").d["warned"]
    finally:
        page.close()


def test_denoise_adapted_is_greyed_on_an_image_with_its_reason(model, tmp_path):
    """Denoise after an Image: its method cards have Adapted greyed, the reason on hover; Pretrained stays."""
    from astro_gpu.client.graphview.page import GraphPage
    from astro_gpu.client.panel import MethodCard
    from astro_gpu.stages.denoise import WHY_NO_ADAPTED
    from tests.client.pagehelp import choose, item
    _l, f = stretched_file(tmp_path / "x.fits", line=None)
    q = model.add_image_workflow("Image", str(f))
    model.set_graph(q, [wfl.node("denoise", "pretrained"), wfl.node("output")])
    page = GraphPage(model)
    try:
        it = item(page, key="debruitage")
        choose(page, it.bid if hasattr(it, "bid") else next(b for b, x in page.canvas.blocks.items() if x is it))
        cards = page.panel.findChildren(MethodCard)
        assert len(cards) == 2
        pre, ada = [c for c in cards if c.ok], [c for c in cards if not c.ok]
        assert len(pre) == 1 and len(ada) == 1
        assert ada[0].toolTip() == WHY_NO_ADAPTED
    finally:
        page.close()


# ------------------------------------------------------------------------------------------------ colours calibrated (#174)

CALIBRATED = ("Photometric CC", "Photometric CC (algorithm: SPCC)", "Photometric Color Calibration",
              "Spectrophotometric Color Calibration", "Color Calibration")


@pytest.mark.parametrize("line", CALIBRATED)
def test_every_wording_of_a_colour_calibration_is_found_and_changes_the_key(tmp_path, line):
    _l, f = stretched_file(tmp_path / "c.fits", line=None)
    hdr = write_fits(tmp_path / "c2.fits", linear_image(sky=0.02, sigma=0.01), [line, "Background extraction (Correction: Subtraction)"])
    assert U.colour_line(hdr) == line
    p, w = make(tmp_path, hdr)
    assert "colour-calibrated" in Keys(p).key_parts(w.id, "image")["x"]
    out, log = run_image(p, w)
    assert "colours calibrated, from the header" in log
    assert json.loads((next((tmp_path / "p").rglob("drz_meta.json"))).read_text())["colour_calibrated"] == line


def test_without_such_a_line_nothing_changes(tmp_path):
    hdr = write_fits(tmp_path / "n.fits", linear_image(sky=0.02, sigma=0.01),
                     ["Background neutralization", "Background extraction (Correction: Subtraction)"])
    assert U.colour_line(hdr) == ""
    p, w = make(tmp_path, hdr)
    assert Keys(p).key_parts(w.id, "image")["x"] == Keys(p).scan.image_id(str(hdr))
    _o, log = run_image(p, w)
    assert "colours calibrated" not in log


def test_the_image_block_states_the_calibrated_colours(model, tmp_path):
    hdr = write_fits(tmp_path / "c.fits", linear_image(sky=0.02, sigma=0.01), ["Photometric CC"])
    plain = write_fits(tmp_path / "p.fits", linear_image(sky=0.02, sigma=0.01), [])
    q = model.add_image_workflow("Image", "")
    model.set_graph(q, [])
    model.set_image_file(q, str(plain))
    assert model.image_colours(q) == ""
    model.set_image_file(q, str(hdr))
    assert model.image_colours(q) == "Photometric CC"
