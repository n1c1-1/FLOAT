"""The Synthetic stars row of a Combine layers layer follows the workflow its STARS come from (its stars side), not
only the layer's own: a star row of a template, a combination of stars (the "· Stars" workflow of the narrow-band
templates). Matching, the block's place, render sources, and the add / Undo / remove steps from Post."""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from astro_gpu import api  # noqa: E402
from astro_gpu.client import i18n, prefs  # noqa: E402
from astro_gpu.client import postengine as pe  # noqa: E402
from tests.client.test_post_stars_group import action, page_of, qapp  # noqa: E402,F401


def group(pid=1, **kw):
    g = pe.new_workflow_layer(pid, "G", part="combined")
    g.update(kw)
    return g


def test_a_synth_row_is_put_in_each_layer_whose_stars_side_workflow_has_the_block():
    own, row, comb, unwired = group(1), group(1, stars_pid=2), group(1, stars_pid=3, stars_image="output"), \
        group(1, stars_pid=None)
    out = pe.with_synth_rows([own, row, comb, unwired], {2: "measured"})
    assert [x["target"] for x in out if pe.is_synth(x)] == [row["id"]]      # the layer's own workflow (1) has none
    out = pe.with_synth_rows([own, row, comb, unwired], {1: "measured", 3: "instrument"})
    assert sorted(x["target"] for x in out if pe.is_synth(x)) == sorted([own["id"], comb["id"]])
    # the row goes before its group, one per group
    assert out.index(next(x for x in out if pe.is_synth(x) and x["target"] == comb["id"])) < out.index(comb)
    assert len([x for x in pe.with_synth_rows(out, {1: "m", 3: "m"}) if pe.is_synth(x)]) == 2
    # a row whose stars side lost the block goes, the layer stays
    kept = pe.with_synth_rows(out, {1: "measured"})
    assert [x["target"] for x in kept if pe.is_synth(x)] == [own["id"]] and comb in kept
    assert pe.stars_side(own) == 1 and pe.stars_side(row) == 2 and pe.stars_side(unwired) is None


def test_the_synth_image_of_a_combination_of_stars_is_stretched_over_the_starless_combination():
    comb = group(1, stars_pid=3, stars_image="output")
    s, st = pe.sub_layer(comb, "synth"), pe.sub_layer(comb, "stars")
    assert (s["pid"], s["part"], s["stars_over"]) == (3, "synth", 1) and pe.source_id(s) == ("synth", 3)
    assert (st["pid"], st["part"], st["stars_over"]) == (3, None, 1) and pe.source_id(st) == 3
    row = group(1, stars_pid=2)                                              # a star row: its Stars block's images
    s = pe.sub_layer(row, "synth")
    assert (s["pid"], s["part"]) == (2, "synth") and "stars_over" not in s and pe.source_id(s) == ("synth", 2)
    own = pe.sub_layer(group(1, stars_pid=1), "synth")
    assert pe.source_id(own) == ("synth", 1) and pe.source_id(pe.sub_layer(group(1), "stars")) == ("stars", 1)


@pytest.fixture()
def two_rows(qapp, tmp_path, monkeypatch):  # noqa: F811
    """Two Full workflows: the first is a Combine layers layer's starless side, the second its stars side (a star row)."""
    from astro_gpu.client.model import ProjectModel
    from astro_gpu.client.services.worker import ApiWorker
    from astro_gpu.core import project2
    from astro_gpu.core.project import Target
    monkeypatch.setenv("ASTRO_GPU_SETTINGS", str(tmp_path / "settings.json"))
    prefs.load()
    p = project2.create(tmp_path / "P", "P", Target(name="M 1"))
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "a1.ARW").write_bytes(b"\0" * 100)
    p.sessions.append(project2.Session(1, "night 1", lights=[str(raw)]))
    api.add_workflow(p, "Full", "Full")
    api.add_workflow(p, "Clear", "Full")
    w = ApiWorker()
    m = ProjectModel(p, w, readonly=True)
    assert w.wait(120) and m.ready, m.error
    a, b = m.paths[:2]
    m.add_post_layer(a.pid, "combined", keep_default=False)
    lay = next(x for x in m.post_layers() if pe.is_group(x))
    assert m.set_layer_side(lay["id"], "stars", b.pid, "stars")
    m._undo.clear()
    yield m, a, b
    w.stop()


def blocks_of(m, pid):
    return [n["block"] for n in m.graph(m.view(pid))]


def key_of(m, pid, block):
    from astro_gpu.client.model import node_key
    return next(node_key(n) for n in m.graph(m.view(pid)) if n["block"] == block)


def test_the_block_of_a_star_row_gives_the_row_to_the_layer_that_takes_its_stars(two_rows):
    m, a, b = two_rows
    page = page_of(m)
    v = page.view
    try:
        g = next(x for x in v.layers if pe.is_group(x))
        assert g["pid"] == a.pid and g["stars_pid"] == b.pid and not pe.has_synth(v.layers, g["id"])
        act = action(v.layer_menu_of(pe.sub_row(g, "stars")), "Synthetic stars")
        assert act.isEnabled() and not act.isChecked()
        act.trigger()
        assert "synth" in blocks_of(m, b.pid) and "synth" not in blocks_of(m, a.pid)    # the stars side's workflow
        row = next(x for x in v.layers if pe.is_synth(x))
        assert row["target"] == g["id"] and m.synth_wanted() == {b.pid: "measured"}
        gid = g["id"]
        v.do_undo()                                                     # one step: the block and the row
        assert "synth" not in blocks_of(m, b.pid) and not pe.has_synth(v.layers, g["id"])
        v.do_redo()
        again = next(x for x in v.layers if pe.is_synth(x))
        assert again["target"] == gid == next(x for x in v.layers if pe.is_group(x))["id"]    # the same layer, the row back
        # the block taken out of the graph drops the row, the layer stays
        m.remove_node(m.view(b.pid), key_of(m, b.pid, "synth"))
        layers = pe.with_synth_rows(m.post_layers(), m.synth_wanted())
        assert not any(pe.is_synth(x) for x in layers) and any(pe.is_group(x) for x in layers)
    finally:
        page.shutdown()
        page.deleteLater()


def test_a_combination_of_stars_takes_the_block_when_its_inputs_bring_a_plate_solution(two_rows):
    m, a, b = two_rows
    f = m.add_fusion("H-O stars", "ho", [{"pid": a.pid, "role": "Ha", "channel": "L"},
                                         {"pid": b.pid, "role": "OIII", "channel": "L"}])
    assert m.palette_reason(m.view(f.pid), "synth") == ""
    m.insert_block(m.view(f.pid), "synth")
    assert blocks_of(m, f.pid) == ["synth"] and m.synth_wanted() == {f.pid: "measured"}
    lay = next(x for x in m.post_layers() if pe.is_group(x))
    assert m.set_layer_side(lay["id"], "stars", f.pid, "output")
    layers = pe.with_synth_rows(m.post_layers(), m.synth_wanted())
    assert [x["target"] for x in layers if pe.is_synth(x)] == [lay["id"]]


def test_without_a_plate_solution_the_palette_greys_it_with_its_reason_in_french(two_rows):
    m, a, b = two_rows
    for w in (a, b):                                   # no Astrometry in the channels: nothing to hand on
        for block in ("color", "astrometry"):
            m.remove_node(m.view(w.pid), key_of(m, w.pid, block))
    f = m.add_fusion("H-O stars", "ho", [{"pid": a.pid, "role": "Ha", "channel": "L"},
                                         {"pid": b.pid, "role": "OIII", "channel": "L"}])
    fv = m.view(f.pid)
    assert m.palette_reason(fv, "synth") == "the combination's inputs bring no plate solution"
    with pytest.raises(ValueError, match="plate solution"):
        m.insert_block(fv, "synth")
    i18n.set_language("fr")
    try:
        assert m.palette_reason(fv, "synth") == "les entrées de la combinaison n'apportent pas de solution astrométrique"
    finally:
        i18n.set_language("en")
