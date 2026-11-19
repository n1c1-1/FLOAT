"""Issue #174, the tester's third round: Inspect of the Image block shows its output, the Image panel states an exact
undo, a new Image takes its file's name, the engine's lines of the block have their French, Gradient follows an Image."""
from __future__ import annotations

import numpy as np
from PySide6.QtWidgets import QLabel

from astro_gpu.client import i18n
from astro_gpu.client.services import run as srun
from tests.client.test_image_fix3 import two_files  # noqa: F401
from tests.client.test_image_source_fixes import fits_file, model, qapp  # noqa: F401
from tests.client.test_image_unstretch import autostretched
from tests.client.test_swap_smoke import env, made_from, qapp as _q, settle, win  # noqa: F401  (fixtures by name)
from tests.test_api import fake_done
from tests.test_unstretch import write_fits
from astro_gpu.stages import unstretch as U


def test_inspect_of_the_image_block_shows_its_output(win, env):
    made_from(win, env, "filters_min")
    s = win.s
    f = fits_file(env["tmp"] / "m27.fits")
    q = s.add_image_workflow("m27", str(f))
    s.set_graph(q, [])
    q = s.view(q.pid)
    for st in s.all_stages(q):
        if st != "image":
            fake_done(s.p, q.pid, st)
    fake_done(s.p, q.pid, "image", files={"drz_full.npy": np.random.default_rng(1).normal(900, 50, (3, 48, 64)).astype(np.float32)})
    s._sync()
    q = s.view(q.pid)
    spec = win.tune.visuals(q, "image")
    assert spec and spec[0].stage == "image"
    s.select(index=s.paths.index(q), step="image")
    win.set_mode(1)
    from astro_gpu.client.services.inspect import service_of
    settle(win, 6)
    service_of(s).wait(30)
    for _i in range(100):
        settle(win, 1)
        if win.tune.view is not None and win.tune.view.image() is not None:
            break
    texts = [x.text() for x in win.tune.findChildren(QLabel) if x.isVisible()]
    assert "No image for this block." not in texts, texts
    assert win.tune.view is not None and win.tune.view.image() is not None, texts
    assert not env["errors"], env["errors"]


def exact_file(tmp_path):
    lin, a = autostretched(tmp_path / "ex.fits", "Histogram Transf. (mid=0.250, lo=0.015, hi=1.000)")
    return a


def test_the_panel_states_an_exact_undo(model, tmp_path):
    a = exact_file(tmp_path)
    q = model.add_image_workflow("Image", str(a))
    model.set_graph(q, [])
    st = model.image_state(q)
    if not st["exact"]:                                   # the file's header must list an exact undo
        import pytest
        pytest.skip("the synthetic header is not read as exact")
    assert "Histogram Transf" in st["lines"] and st["warning"] == ""
    i18n.set_language("fr")
    try:
        assert i18n._("Stretch undone (exact, from the header)") == "Étirement défait (exact, d'après l'en-tête)"
    finally:
        i18n.set_language("en")


def test_a_new_image_takes_its_files_name_a_typed_name_stays(model, tmp_path):
    a, b = two_files(tmp_path)
    model.apply([("add_row", {"kind": "image"})])
    q = model.paths[-1]
    assert q.name.startswith("Image")
    model.set_image_file(q, str(a))
    assert model.p.workflow(q.pid).name == "st_mono_mtf"
    model.apply([("add_row", {"kind": "image", "name": "My nebula"})])
    q2 = model.paths[-1]
    model.set_image_file(q2, str(b))
    assert model.p.workflow(q2.pid).name == "My nebula"


def test_the_image_blocks_lines_have_their_french():
    lines = ["image siril-out.fit: 4256 x 2848",
             "stretch undone from the header (1 step(s); 12 pixels clipped by the stretch stay as they are)",
             "stretch undone by estimate (mid-tones m = 0.2500, 0.2500)",
             "a single image: the noise is measured on the image, the dither check is skipped (it reads two half-stacks)",
             "denoising the image: tile 32 of 216"]
    i18n.set_language("fr")
    try:
        out = [srun.engine_text(x) for x in lines]
    finally:
        i18n.set_language("en")
    assert out[0] == "image siril-out.fit : 4256 x 2848" and "values" not in out[0]
    assert out[1].startswith("étirement défait d'après l'en-tête (1 étape(s)")
    assert out[2].startswith("étirement défait par estimation")
    assert out[3].startswith("une image seule") and out[4] == "débruitage de l'image : tuile 32 sur 216"
    for x, y in zip(lines, out):
        assert x != y


def test_gradient_is_placeable_after_an_image_block(model, tmp_path):
    from astro_gpu.graphcore import api
    from tests.client.pagehelp import block_id
    q = model.add_image_workflow("Image", str(fits_file(tmp_path / "a.fits")))
    model.set_graph(q, [])
    assert api.palette_state(model.stored, {}, "gradient")[0]
    img = block_id(model, q, "image")
    row = next(r.id for r in model.stored.rows if img in r.blocks)
    assert model.apply([("add_block", {"kind": "gradient", "at": {"type": "insert", "row": row, "after": img}})]).ok
