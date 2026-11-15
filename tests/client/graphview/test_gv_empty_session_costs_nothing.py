"""A session added with no lights to a computed project computes nothing again (a beta tester's empty session made the
whole project Outdated, 17 min to run, its target forgotten): the definition the engine reads, the keys and the states
stay as they were, the target is kept and the gesture's toast has nothing to compute. The session is drawn, and its
first lights bring it into its workflow, as any session. On the computed test project of test_gv_legacy_model."""
from __future__ import annotations

import numpy as np

from astro_gpu.core.store import InputScan, key_rules
from astro_gpu.graphcore.proof import keys_table
from tests.client.graphview.test_gv_legacy_model import (  # noqa: F401  (the fixtures are used by name)
    computed_but_stars,
    env,
    model,
    needs_test_project,
    page,
    settle,
    world,
)

pytestmark = needs_test_project


def _toast(page):
    t = page.canvas.toast
    return t.plain if t.isVisible() else ""


def _read(m):
    flows = m.p.to_dict()["workflows"]
    for d in flows:
        d["sessions"] = [s.id for s in m.p.sessions_of(d["id"])]
    return flows, keys_table(m.p, key_rules(), InputScan())


def test_an_empty_session_computes_nothing_again_and_counts_once_it_has_lights(world, model, page, tmp_path):
    from astropy.io import fits
    full = model.view(1)
    assert computed_but_stars(model, full)
    flows, keys = _read(model)
    states = {k: v.state for k, v in model._states.items()}
    target, seconds = model.p.target, model.total_seconds()
    sids = {s.id for s in model.p.sessions}
    row = next(r for r in page.bridge.graph.rows if r.nights and r.wid == full.pid)
    res = page.bridge.act("add_night", row=row.id)
    settle(world["w"], page)
    assert res.ok
    new = [s.id for s in model.p.sessions if s.id not in sids]
    assert len(new) == 1 and not model.p.session(new[0]).lights
    row = next(r for r in page.bridge.graph.rows if r.id == row.id)
    assert new[0] in [page.bridge.graph.blocks[s].sid for s, _c in row.nights]       # drawn in its stack
    assert _read(model) == (flows, keys)
    assert {k: v.state for k, v in model._states.items()} == states
    assert model.p.target == target and model.total_seconds() == seconds
    said = _toast(page)
    assert "To compute" not in said and "Computes again" not in said and "min to run" not in said, said
    # its first lights: it counts, its workflow computes again
    folder = tmp_path / "lights"
    folder.mkdir()
    fits.PrimaryHDU(np.zeros((8, 8), np.uint16)).writeto(folder / "l1.fits")
    assert model.add_session_entries(new[0], "lights", [str(folder / "l1.fits")])
    settle(world["w"], page)
    assert new[0] in [s.id for s in model.p.sessions_of(full.pid)]
    assert _read(model)[1] != keys
