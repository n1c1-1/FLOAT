"""The "Mono RGB" and "Mono LRGB" templates: a monochrome camera, each filter stacked on its own, the combination of the
filters, one Color block after it, Post showing the combination only. Offscreen; nothing is computed."""
from __future__ import annotations

import pytest

from astro_gpu.client import i18n
from astro_gpu.client.gallery import GALLERY, _view_opts, draw_graph
from astro_gpu.stages.combine import COMBINATIONS
from tests.client.test_joins165 import by_name, settle, world, qapp  # noqa: F401

KEYS = {"mono_rgb": ("rgb", ["R", "G", "B"]), "mono_lrgb": ("lrgb", ["L", "R", "G", "B"])}


def _make(world, key):
    m = world["m"]
    t = next(c for c in GALLERY if c["key"] == key)
    world["page"].make_from_template(t, "Colour", _view_opts(t))
    settle(world)
    return m, t


@pytest.mark.parametrize("key", KEYS)
def test_the_template_makes_its_rows_its_combination_and_its_color(world, key):
    kind, filters = KEYS[key]
    m, t = _make(world, key)
    assert [q.name for q in m.paths] == filters + ["Colour"]
    rows = [by_name(m, f) for f in filters]
    for q in rows:
        blocks = [n["block"] for n in m.graph(q)]
        assert "color" not in blocks and "stars" not in blocks
        assert all(b in blocks for b in ("calibration", "stacking", "gradient", "denoise")), blocks
        assert ("align_on" in blocks) == (q is not rows[0])
        assert ("astrometry" in blocks) == (q is rows[0])
        assert (m.align_reference(q) == rows[0].pid) == (q is not rows[0])
    fus = by_name(m, "Colour")
    f = m.p.workflow(fus.pid).source["fusion"]
    assert f["type"] == kind and [(i["role"], i["pid"]) for i in f["inputs"]] == [(r, by_name(m, r).pid) for r in filters]
    assert all(i.get("image", "output") == "output" for i in f["inputs"])
    assert [n["block"] for n in m.graph(fus)] == ["color"]
    gone = {int(a) for a, _b in m.p.post["removed"]}          # Post's default layers: the rows' are left out
    assert gone == {r.pid for r in rows} and fus.pid not in gone
    m.undo()
    settle(world)
    assert m.paths == [] and m.p.sessions == []


@pytest.mark.parametrize("key", KEYS)
def test_the_plan_lists_the_combination_then_color(world, key):
    from astro_gpu import api
    kind, _f = KEYS[key]
    m, _t = _make(world, key)
    fus = by_name(m, "Colour")
    plan = api.plan_all(m.p)
    stages = [st for pid, st in plan.targets if pid == fus.pid]
    cls = next(c for c in COMBINATIONS if c.kind == kind)
    assert stages[:2] == [cls.name, "color"], (stages, plan.skipped, plan.left_out)


def test_french_texts_and_card(world):
    i18n.set_language("fr")
    try:
        for key in KEYS:
            t = next(c for c in GALLERY if c["key"] == key)
            assert i18n._(t["name"]) != t["name"] and i18n._(t["text"]) != t["text"]
            assert not draw_graph(t, 400, 160, True).isNull()
    finally:
        i18n.set_language("en")


def test_a_missing_filter_gets_an_empty_session(world):
    m = world["m"]
    m.add_filter_session("R")
    m.add_filter_session("G")
    t = next(c for c in GALLERY if c["key"] == "mono_rgb")
    world["page"].make_from_template(t, "Colour", _view_opts(t))
    settle(world)
    assert sorted(m.filters()) == ["B", "G", "R"]
    assert [q.name for q in m.paths] == ["R", "G", "B", "Colour"]

