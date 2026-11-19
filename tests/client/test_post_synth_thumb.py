"""The thumbnails of a Combine layers layer's rows are pictures, the Synthetic stars row's too (the tutorials' film: its
thumbnail stayed the empty placeholder, beside its white open mask, because the panel never asked for it once the group's
other thumbnails were there)."""
from __future__ import annotations

import copy
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from astro_gpu.client import postengine as pe  # noqa: E402
from astro_gpu.client.services.render import Renderer, Request  # noqa: E402
from tests.client.test_post_stars_group import (Images, group, images, page_of, project, qapp)  # noqa: E402,F401


def test_every_row_of_a_group_with_synthetic_stars_gets_a_thumbnail_that_is_a_picture():
    parts = images()
    rows = pe.with_synth_rows([group()], {1: "measured"})
    layers = pe.snapshot(pe.normalize(copy.deepcopy(rows)))
    gid = next(x["id"] for x in layers if pe.is_group(x))
    res = Renderer(Images(parts)).run(Request(kind="thumbs", layers=layers, need_thumbs=(gid,)))
    assert {gid, (gid, "stars"), (gid, "starless"), (gid, "synth")} <= set(res.layer_thumbs)
    for key, t in res.layer_thumbs.items():
        assert t.shape == (36, 54, 3) and t.std() > 1.0, key          # not one colour


def test_the_panel_asks_for_the_synthetic_stars_thumbnail_the_group_lacks(project):
    m = project
    m.insert_block(m.paths[0], "synth")
    v = page_of(m).view
    g = next(x for x in v.layers if pe.is_group(x))
    pm = v.thumb(g)
    v._lthumbs[g["id"]] = (v._thumb_sig(g), pm)
    for s in pe.subs_of(g):
        v._lthumbs[(g["id"], s)] = (None, pm)
    assert list(v._stale_thumbs()) == [g["id"]]                       # the synthetic stars' is the one missing
    v._lthumbs[(g["id"], "synth")] = (None, pm)
    assert list(v._stale_thumbs()) == []
