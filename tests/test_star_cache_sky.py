"""The stars layer's cache key: Sky is in it (the stars' stretch uses it), except for an image already stretched."""
import torch

from astro_gpu.render.stars import StarSettings, StarsRenderEngine


def _engine(stretched: bool):
    e = object.__new__(StarsRenderEngine)
    e.prep = {"stats_full": {"stretched": stretched}}
    e.calls = 0
    e._layer = lambda name, f, box: torch.zeros(3, 4, 4)

    def tone(x, s, f):
        e.calls += 1
        return x

    e.star_tone = tone
    return e


def test_sky_rebuilds_the_stars_of_a_linear_image():
    e = _engine(False)
    e._star_layer(StarSettings(sky=0.2), 1, None, "combined")
    e._star_layer(StarSettings(sky=0.2), 1, None, "combined")
    assert e.calls == 1
    e._star_layer(StarSettings(sky=0.3), 1, None, "combined")
    assert e.calls == 2


def test_sky_leaves_the_stars_of_a_stretched_image():
    e = _engine(True)
    e._star_layer(StarSettings(sky=0.2), 1, None, "combined")
    e._star_layer(StarSettings(sky=0.3), 1, None, "combined")
    assert e.calls == 1
    e._star_layer(StarSettings(star_strength=2.0), 1, None, "combined")
    assert e.calls == 2
