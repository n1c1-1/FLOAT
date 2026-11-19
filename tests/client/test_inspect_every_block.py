"""Issue #177 (and #174): Inspect of a block that has run shows its picture, whatever the block (the Synthetic stars
block, computed, said "No image for this block" like the Image block before it). The test goes over every block kind of
the palette (graphcore/rules.ORDER), on a project made of frames and on one made of an Image: each block is placed, every
stage is given a done result, and Inspect of the block offers at least one tab and does not say "No image for this
block". A block that has not run says "Not computed yet" (and how)."""
from __future__ import annotations

import numpy as np
import pytest
from PySide6.QtWidgets import QLabel

from astro_gpu.graphcore import rules
from tests.client.test_swap_smoke import env, made_from, qapp, settle, win  # noqa: F401  (fixtures by name)
from tests.test_api import fake_done

#: the blocks Inspect gives their own page (not tabs of pictures): frames, the selection charts, the layers editor
OWN_PAGE = {"session", "calibration", "registration", "selection", "crop", "stacking", "combine_layers", "post"}
KINDS = [k for k in rules.ORDER if k not in ("session",)]


def _fill(win, q):
    """Every stage of the workflow done, with a small linear image where the engine writes one."""
    s = win.s
    rng = np.random.default_rng(3)
    img = rng.normal(900, 40, (3, 48, 64)).astype(np.float32)
    files = {"drz_full.npy": img, "lin.npy": img, "stars.npy": img, "starless.npy": img, "deconv_full.npy": img,
             "n2n_full.npy": img, "valid.npy": np.ones((48, 64), np.float32)}
    for st in s.all_stages(q):
        fake_done(s.p, q.pid, st, files=files)
    s._sync()


def _blocks_of(win, kind):
    return [b for b, x in win.s.stored.blocks.items() if x.kind == kind]


def _place(win, kind):
    pal = win.graph.palette
    before = set(win.s.stored.blocks)
    pal.double_click(kind)
    settle(win, 4)
    return [b for b in win.s.stored.blocks if b not in before]


def _build(win, env, source):
    """A project with every block of the palette that its workflows take: from frames (the palette's cards), or from an
    Image (its row, the blocks after it)."""
    from tests.client.test_image_source_fixes import fits_file
    made_from(win, env, "colour_min")
    s = win.s
    if source == "image":
        q = s.add_image_workflow("m27", str(fits_file(env["tmp"] / "m27.fits")))
        s.set_graph(q, [])
        from tests.client.pagehelp import block_id
        last = block_id(s, s.view(q.pid), "image")
        row = next(r.id for r in s.stored.rows if last in r.blocks)
        for kind in ("astrometry", "gradient", "denoise", "color", "stars", "synth"):
            before = set(s.stored.blocks)
            res = s.apply([("add_block", {"kind": kind, "at": {"type": "insert", "row": row, "after": last}})])
            new = [b for b in s.stored.blocks if b not in before]
            if res.ok and new:
                last = new[0]
        settle(win, 4)
    else:
        for kind in KINDS:
            if not _blocks_of(win, kind):
                _place(win, kind)
    return s


def _addresses(win):
    s = win.s
    out = []
    for bid, blk in list(s.stored.blocks.items()):
        addr = s.block_map.address(bid)
        if addr is not None:
            out.append((blk.kind, addr))
    return out


@pytest.mark.parametrize("source", ["frames", "image"])
def test_inspect_offers_a_picture_for_every_block_kind(win, env, source):
    s = _build(win, env, source)
    for q in list(s.paths):
        _fill(win, q)
    settle(win, 6)
    seen, missing = set(), []
    for kind, (wid, key) in _addresses(win):
        seen.add(kind)
        if kind in OWN_PAGE:
            continue
        if not win.tune.visuals(s.view(wid), key.partition("~")[0]):
            missing.append((kind, key))
    assert not missing, f"no tab for the blocks: {missing}"
    assert {"gradient", "stars", "synth"} <= seen, seen


@pytest.mark.parametrize("source", ["frames", "image"])
def test_inspect_of_a_computed_block_shows_a_picture_never_the_empty_message(win, env, source):
    from astro_gpu.client.services.inspect import service_of
    s = _build(win, env, source)
    for q in list(s.paths):
        _fill(win, q)
    settle(win, 6)
    bad = []
    for kind, (wid, key) in _addresses(win):
        if kind in OWN_PAGE or kind in ("combine",):
            continue
        q = s.view(wid)
        s.select(index=s.paths.index(q), step=key)
        win.set_mode(1)
        settle(win, 4)
        service_of(s).wait(30)
        for _i in range(60):
            settle(win, 1)
            if win.tune.view is not None and win.tune.view.image() is not None:
                break
        texts = [x.text() for x in win.tune.findChildren(QLabel) if x.isVisible()]
        shown = win.tune.view is not None and win.tune.view.image() is not None
        if "No image for this block." in texts or "Not computed yet" in texts or not shown:
            bad.append((kind, key, texts[:4]))
    assert not bad, bad
    assert not env["errors"], env["errors"]


def test_a_block_not_run_says_not_computed_yet(win, env):
    s = _build(win, env, "frames")
    settle(win, 4)
    for kind, (wid, key) in _addresses(win):
        if kind not in ("synth", "stars"):
            continue
        s.select(index=s.paths.index(s.view(wid)), step=key)
        win.set_mode(1)
        settle(win, 6)
        texts = [x.text() for x in win.tune.findChildren(QLabel) if x.isVisible() and x.text()]
        assert "Not computed yet" in texts and "No image for this block." not in texts, (kind, texts)


def test_a_stretched_image_block_has_its_picture_too(win, env):
    from tests.client.test_image_source_fixes import fits_file
    made_from(win, env, "colour_min")
    q = win.s.add_image_workflow("m27", str(fits_file(env["tmp"] / "m27.fits")), stretched=True)
    assert win.tune.visuals(win.s.view(q.pid), "image")
