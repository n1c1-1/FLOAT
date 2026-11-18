"""A release while the previous one's full-resolution picture is being made: that picture is abandoned, and the new
screen picture no longer waits for it. A Stars block's image at full resolution is a few seconds of one call (the
owner's NGC 7000 with its stars shown: the next release's picture came 5.3 s after it, its own render 0.9 s); it is
now abandoned between its steps."""
from __future__ import annotations

import numpy as np
import pytest
from test_post import DEV, stars_prep

from astro_gpu.client.postengine import Superseded
from astro_gpu.render.layers import LayerService


class Asked(Exception):
    pass


def test_a_stars_image_is_abandoned_between_its_steps_and_unchanged_otherwise(tmp_path):
    _out, st = stars_prep(tmp_path / "w", 300, 420)
    svc = LayerService(DEV)
    sliders = {"star_strength": 1.3, "star_size": -1.0, "star_small": 2.0}
    plain = svc.render(st, None, sliders, 1.0, part="stars").rgb
    calls = []
    again = svc.render(st, None, dict(sliders, star_strength=1.31), 1.0, part="stars",
                       stop=lambda: calls.append(1)).rgb
    assert len(calls) >= 5                                  # asked between the steps: the tone, the small stars, the
    back = svc.render(st, None, sliders, 1.0, part="stars", stop=lambda: None).rgb    # size, the mask's passes
    assert np.array_equal(back, plain) and not np.array_equal(again, plain)

    def abandon():
        calls.append(1)
        if len(calls) > 2:
            raise Asked()
    calls.clear()
    with pytest.raises(Asked):
        svc.render(st, None, dict(sliders, star_strength=1.5), 1.0, part="stars", stop=abandon)
    assert np.array_equal(svc.render(st, None, sliders, 1.0, part="stars").rgb, plain)   # nothing half made kept


def test_post_s_stars_layer_passes_its_composition_s_stop_down(tmp_path):
    from astro_gpu.client.postengine import new_workflow_layer
    from astro_gpu.client.services.render import LayerSource
    _out, st = stars_prep(tmp_path / "w", 300, 420)

    class Folder:
        def __init__(self, path):
            self.path, self.mtime, self.previous, self.rendered = path, 0.0, False, True
    folders = {("stars", 1): Folder(st)}
    src = LayerSource(lambda pid: folders.get(pid), LayerService(DEV))
    lay = dict(new_workflow_layer(1, "Full", True), part="stars")
    src.canvas([("stars", 1)])
    with pytest.raises(Superseded):
        src.layer(lay, 0, (0, 0, 380, 292), stop=lambda: True)
