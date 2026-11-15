"""PNG and JPEG files in a folder with a letter outside the system code page (review 13.1).

On Windows ``cv2.imread`` / ``cv2.imwrite`` do not find "C:/…/Nébuleuse/x.png" (None / False, no error): the stages'
previews were not written, an Image block on a PNG failed, the PNG 16 export and the viewer failed. They go through
bytes now (io/cvfile.py).
"""
from __future__ import annotations

import numpy as np
import pytest

from astro_gpu.io import cvfile


@pytest.fixture
def folder(tmp_path):
    d = tmp_path / "Nébuleuse Pléiades"
    d.mkdir()
    return d


def test_read_and_write_go_through_any_path(folder):
    a = (np.arange(4 * 5 * 3) % 251).astype(np.uint8).reshape(4, 5, 3)
    cvfile.write(folder / "x.png", a)
    assert np.array_equal(cvfile.read(folder / "x.png"), a)
    b = np.arange(20, dtype=np.uint16).reshape(4, 5) * 3000
    cvfile.write(folder / "y.png", b)
    r = cvfile.read(folder / "y.png")
    assert r.dtype == np.uint16 and np.array_equal(r, b)
    with pytest.raises(OSError):
        cvfile.read(folder / "gone.png")
    (folder / "text.png").write_text("not an image")
    with pytest.raises(OSError):
        cvfile.read(folder / "text.png")


def test_a_stage_preview_is_written_there(folder):
    from astro_gpu.core.journal import Journal
    from astro_gpu.core.stage import Context
    from astro_gpu.stages.image import ImageSource

    class Pr:
        root, logs, scratch = folder, folder / "logs", folder / "scratch"

        def stage_dir(self, s):
            return folder / s

    Pr.logs.mkdir()
    ctx = Context(Pr(), ImageSource(), Journal(Pr.logs), "cpu")
    p = ctx.preview("quality", np.full((8, 8, 3), 0.5))
    assert p.is_file() and cvfile.read(p).shape == (8, 8, 3)


def test_the_image_block_post_and_viewer_read_a_png_there_and_the_exports_write(folder):
    from astro_gpu.client.imageview import read_file
    from astro_gpu.client.services.render import load_image
    from astro_gpu.render.layers import _write_png16
    from astro_gpu.stages.image import read_image
    rgb = np.zeros((4, 5, 3), np.uint8)
    rgb[..., 0] = 200                                           # red
    cvfile.write(folder / "red.png", rgb[..., ::-1])
    assert read_image(folder / "red.png")[0].max() == 200 and read_image(folder / "red.png")[2].max() == 0
    assert load_image(str(folder / "red.png"))[..., 0].max() == 200
    assert read_file(folder / "red.png")[0][..., 0].max() == 200
    _write_png16(folder / "export.png", np.full((4, 5, 3), 0.5, np.float32), b"")
    assert cvfile.read(folder / "export.png").dtype == np.uint16
