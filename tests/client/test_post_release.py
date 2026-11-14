"""Leaving Post hands back what it can rebuild (render engines, caches), and the caches are bounded."""
from __future__ import annotations

import time

from test_post import Setup, shown, post, preps, qapp, _settings  # noqa: F401 - the fixtures

from astro_gpu.client import postengine


def _wait(fn, timeout=30):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if fn():
            return True
        time.sleep(0.05)
    return fn()


def test_leaving_post_drops_the_engines_and_the_caches_and_coming_back_rebuilds_them(post):
    r = post.worker.renderer
    c = r.composer
    assert r.source.service.loaded() and c.images.bytes > 0
    before = shown(post.view).copy()
    post.page.release_memory()
    assert _wait(lambda: not r.source.service.loaded() and c.images.bytes == 0 and c.steps.bytes == 0)
    assert c.parts.bytes == 0 and not r.source._grids and not r.source._img
    post.view.request(final_only=True)                          # coming back: the next render opens them again
    assert post.settle() and r.source.service.loaded() and c.images.bytes > 0
    assert (shown(post.view) == before).all()                    # the same image


def test_leaving_post_keeps_its_caches_unless_memory_is_short(post, monkeypatch):
    from astro_gpu.core import devices
    r = post.worker.renderer
    monkeypatch.setattr(devices, "memory_budget", lambda: (8 << 30, 16 << 30))     # room enough: all kept
    post.page.leave()
    time.sleep(0.5)
    assert r.source.service.loaded() and r.composer.images.bytes > 0
    monkeypatch.setattr(devices, "memory_budget", lambda: (1 << 30, 2 << 30))      # short: handed back
    post.page.leave()
    assert _wait(lambda: not r.source.service.loaded() and r.composer.images.bytes == 0)


def test_a_request_after_a_release_is_not_undone_by_it(post):
    post.page.release_memory()
    post.view.request(final_only=True)
    assert post.settle()
    assert post.worker.renderer.source.service.loaded()


def test_the_caches_together_stay_in_the_default_budget():
    cap = postengine.default_cache_bytes()
    assert 300_000_000 <= cap <= 1_500_000_000
    c = postengine.Composer(None)
    assert c.images.cap + c.steps.cap + c.backdrops.cap + c.parts.cap <= cap
