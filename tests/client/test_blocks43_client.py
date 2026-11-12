"""Issue #43 in the client: Session blocks before the chain, cut and greyed blocks, a workflow from an image, offscreen.

A scratch project whose frames are empty files (the keys read their names and sizes only): nothing is computed.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if Path("C:/Windows/Fonts").is_dir():
    os.environ.setdefault("QT_QPA_FONTDIR", "C:/Windows/Fonts")
pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication, QComboBox  # noqa: E402

from astro_gpu.client import i18n  # noqa: E402
from astro_gpu.client.blocks import Graph, node_title  # noqa: E402
from astro_gpu.client.graph import GraphPage  # noqa: E402
from astro_gpu.client.model import ProjectModel  # noqa: E402
from astro_gpu.client.services.worker import ApiWorker  # noqa: E402
from astro_gpu.core import project2  # noqa: E402
from astro_gpu.core import workflow as wfl  # noqa: E402
from astro_gpu.core.project import Target  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def frames(folder: Path, names) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    for i, n in enumerate(names):
        (folder / n).write_bytes(b"\0" * (100 + i))
    return folder


@pytest.fixture()
def world(qapp, tmp_path, monkeypatch):
    monkeypatch.setenv("ASTRO_GPU_SETTINGS", str(tmp_path / "settings.json"))
    i18n.set_language("en")
    raw = tmp_path / "raw"
    n1, n2 = frames(raw / "n1", ["a1.ARW", "a2.ARW"]), frames(raw / "n2", ["b1.ARW"])
    darks, flats = frames(raw / "darks", ["d1.ARW"]), frames(raw / "flats", ["f1.ARW"])
    p = project2.create(tmp_path / "P", "P", Target(name="M 31"))
    p.stack_mode = "per_session"
    p.sessions += [project2.Session(1, "night 1", lights=[str(n1)], darks=[str(darks)], flats=[str(flats)]),
                   project2.Session(2, "night 2", lights=[str(n2)], darks=[str(darks)])]
    p.add_workflow("Full", wfl.TEMPLATES["Full"])
    project2.save(p)
    w = ApiWorker()
    m = ProjectModel(p, w)
    assert w.wait(120)
    m._sync()
    page = GraphPage(m)
    page.resize(1500, 800)
    yield {"m": m, "w": w, "page": page, "raw": raw, "tmp": tmp_path}
    page.close()
    m.close(save=False)
    w.stop()


def settle(world):
    world["w"].wait(60)
    QApplication.processEvents()
    world["page"].refresh()


def test_two_sessions_are_drawn_before_the_chain_and_joined(world):
    m, page = world["m"], world["page"]
    g = Graph(m)
    assert [(s.ref, c.ref) for s, c in g.sessions] == [("session@1", "calibration@1"), ("session@2", "calibration@2")]
    assert [node_title(m, s) for s, _c in g.sessions] == ["night 1", "night 2"]
    assert m.display_keys(m.path)[:2] == ["alignement", "selection"]          # their Calibrations are not in the chain
    assert g.node_of(m.path, "calibration@2") is g.sessions[1][1]
    assert g.summary(g.sessions[1][0]).startswith("0 lights") or "light" in g.summary(g.sessions[1][0])
    assert m.shared_with(1, "darks") == [2] and m.shared_with(1, "flats") == []
    page.graph.rebuild()
    tips = [it.toolTip() for it in page.graph.scene().items()]
    assert any("join here" in t for t in tips)                               # the bar before Registration
    page.select_node(g.sessions[1][1])
    assert m.sel_step == "calibration@2"
    page.refresh()
    assert "night 2" in page.panel.hv.itemAt(0).layout().itemAt(0).widget().text()


def test_the_session_block_edits_its_frames_and_its_filter(world):
    m = world["m"]
    raw = world["raw"]
    bias = frames(raw / "bias", ["o1.ARW"])
    assert m.add_session_entries(2, "bias", [str(bias)]) == 1 and m.add_session_entries(2, "bias", [str(bias)]) == 0
    assert m.use_from_session(1, 2, "flats") and m.shared_with(2, "flats") == [1]
    assert m.set_session_filter(2, "Ha") and m.session(2).filter == "Ha"
    lights = m.session(2).lights[0]
    assert m.remove_session_entry(2, "lights", lights) and not m.session(2).lights   # the last light can go (#58)
    assert m.add_session_entries(2, "lights", [m.p.abspath(lights)]) == 1
    assert m.rename_session(2, "second night") and m.session(2).name == "second night"
    assert m.dirty and m.can_undo()                # its name and filter are undone (GUI exploration 2.3) ...
    assert m.undo() and m.session(2).name != "second night" and m.session(2).lights   # ... its frames are not
    assert m.redo() and m.session(2).name == "second night"
    sid = m.add_session("third", lights=[str(frames(raw / "n3", ["c1.ARW"]))])
    assert sid == 3 and m.separate_sessions() == [1, 2, 3]
    assert m.remove_session(3) and m.separate_sessions() == [1, 2]
    assert m.undo() and m.separate_sessions() == [1, 2, 3]            # a Session block removed: Ctrl+Z (#47)
    assert m.session(3).lights and m.redo() and m.separate_sessions() == [1, 2]
    world["page"].select(step="session@2")
    world["page"].refresh()
    combos = world["page"].panel.findChildren(QComboBox)
    assert any(cb.currentData() == "Ha" for cb in combos)


def test_switching_a_block_off_draws_it_so_refuses_what_cannot_and_undoes(world):
    """Issue #47: a block whose output is its input's kind can be switched off (drawn hatched, the data passes
    through); one the blocks after it need cannot, and the toast says why; Undo takes it back."""
    m, page = world["m"], world["page"]
    toasts = []
    page.toasted.connect(toasts.append)
    page.toggle_cut(Graph(m).node_of(m.path, "fond"), None)
    settle(world)
    assert m.is_cut(m.path, "fond") and "gradient" not in m.all_stages(m.path)
    assert Graph(m).node_of(m.path, "fond").cut and m.p.workflow(1).blocks is not None
    page.toggle_cut(Graph(m).node_of(m.path, "astrometrie"), None)          # Color needs it
    settle(world)
    assert not m.is_cut(m.path, "astrometrie") and "needed by Color" in toasts[-1]
    page.toggle_cut(Graph(m).node_of(m.path, "empilement"), None)           # it turns frames into an image
    assert not m.is_cut(m.path, "empilement") and toasts[-1].startswith("Stacking: frames → image")
    m.undo()
    settle(world)
    assert not m.is_cut(m.path, "fond")


def test_a_workflow_from_an_image_starts_on_its_own_lane(world):
    import numpy as np
    from astropy.io import fits
    m, page = world["m"], world["page"]
    img = world["tmp"] / "m42.fits"
    fits.PrimaryHDU(np.zeros((3, 8, 8), np.float32)).writeto(img)
    v = m.add_image_workflow("From an image", str(img))
    assert v.source["path"].endswith("m42.fits") and m.display_keys(v)[0] == "image"
    assert m.all_stages(v)[0] == "image"
    g = Graph(m)
    im = g.node_of(v, "image")
    assert im is not None and im.paths == [v] and v not in g.root.paths
    assert g.summary(im) == "m42.fits"
    page.select(index=m.paths.index(v), step="image")
    page.refresh()
    page.graph.rebuild()                                                      # drawn without error, its lane last


def test_filters_make_one_workflow_each_and_a_combination_in_the_composition_section(world):
    m, page = world["m"], world["page"]
    m.set_session_filter(1, "Ha")
    m.set_session_filter(2, "OIII")
    assert m.filters() == {"Ha": [1], "OIII": [2]}
    d = page.new_workflow()
    d.pick(next(c.t for c in d.cards if c.t["key"] == "filters_full"))
    assert d.ok.isEnabled() and "Empty Session blocks are made for SII" in d.duration.text()   # #47: made empty
    from astro_gpu.client.gallery import PALETTE_TIPS, SHO_DEFAULT
    from astro_gpu.stages import combine
    assert d.palette.currentData() == SHO_DEFAULT == combine.CombineSHO.title     # S II, H-alpha, O III first
    assert [d.palette.itemData(i) for i in range(d.palette.count())] == ["S-H-O", "H-O", "RGB", "LRGB"]   # no presets
    assert set(PALETTE_TIPS) == {c.title for c in combine.COMBINATIONS}           # each type says what it does
    d.palette.setCurrentIndex(d.palette.findData("H-O"))
    assert d.ok.isEnabled() and "Ha" in d.duration.text()
    d.ok.click()
    settle(world)
    names = [q.name for q in m.paths]
    assert names[-3:] == ["Ha", "OIII", "H-O"]
    ha, oiii, ho = m.paths[-3:]
    assert ha.sessions == [1] and oiii.sessions == [2] and m.is_mono(ha) and not m.is_mono(ho)
    assert ho.source["fusion"]["type"] == "ho" and "stars" not in ho.source["fusion"]
    assert m.display_keys(ho) == ["fusion", "etoiles"] and "couleur" not in m.display_keys(ha)   # the stars off once, after it (#165)
    assert "etoiles" not in m.display_keys(ha)
    assert m.align_reference(oiii) == ha.pid and m.align_reference(ha) is None      # the others on the first one's grid
    assert "align_on" in m.display_keys(oiii) and "align_on" not in m.display_keys(ha)
    g = Graph(m)
    fn = g.node_of(ho, "fusion")
    assert fn.handle and g.node_of(ho, "post") is fn and g.node_of(ho, "etoiles") is not None and not fn.waits
    assert [(id(a), r) for a, r, _i in g.fusion_inputs[id(fn)]] == [(id(g.leaves[ha.pid]), "Ha"),
                                                                    (id(g.leaves[oiii.pid]), "OIII")]   # named inputs
    page.graph.rebuild()
    assert page.graph.graph.node_of(ho, "fusion").col == page.graph._cx["f0"]          # before Combine
    d = page.new_workflow()                                                   # the same filters again: reused
    d.pick(next(c.t for c in d.cards if c.t["key"] == "filters_min"))
    d.ok.click()
    settle(world)
    assert [q.name for q in m.paths].count("Ha") == 1


def test_an_already_stretched_image_workflow_goes_to_post(world):
    import cv2
    import numpy as np
    m = world["m"]
    png = world["tmp"] / "done.png"
    cv2.imwrite(str(png), np.full((8, 8, 3), 100, np.uint8))
    v = m.add_image_workflow("Done", str(png), {"debruitage": False, "etoiles": "retirer", "fond": "simple"},
                             stretched=True)
    assert v.source["stretched"] and m.all_stages(v) == ["image", "star_removal"]
    assert m.display_keys(v) == ["image", "etoiles"] or m.display_keys(v) == ["image", "etoiles", "post"]
    g = Graph(m)
    assert g.summary(g.node_of(v, "image")).endswith("stretched")


def test_a_combination_panel_shows_its_named_inputs_and_its_settings(world):
    import re

    from PySide6.QtWidgets import QDoubleSpinBox, QLabel
    m, page = world["m"], world["page"]
    m.set_session_filter(1, "Ha")
    m.set_session_filter(2, "OIII")
    d = page.new_workflow()
    d.pick(next(c.t for c in d.cards if c.t["key"] == "filters_full"))
    d.palette.setCurrentIndex(d.palette.findData("H-O"))
    d.ok.click()
    settle(world)
    ha, oiii, ho = m.paths[-3:]
    assert m.all_stages(ho) == ["combine_ho", "render_simple", "star_removal"]          # the stars taken off once, after it
    page.graph.rebuild()
    page.select(index=m.paths.index(ho), step="fusion")
    page.refresh()
    settle(world)
    page.refresh()
    shown = {re.sub("<[^>]+>", "", x.text()) for x in page.panel.findChildren(QLabel)}
    assert {"Hα", "O III", "H-O"} <= shown                                               # its named inputs, its type
    assert {"Hα hue", "O III hue", "Colour smoothing"} <= shown                          # its settings, as any block's
    assert m.set_fusion_weight(m.view(ho.pid), "OIII", 0.5) is True                    # a setting of its block
    assert m.p.workflow(ho.pid).params["combine_ho"]["weights"] == {"OIII": 0.5}
    assert m.set_fusion_weight(m.view(ho.pid), "OIII", 1.0) is True                    # 1: as balanced, not stored
    assert "combine_ho" not in m.p.workflow(ho.pid).params
    page.refresh()
    assert len([s for s in page.panel.findChildren(QDoubleSpinBox) if s.maximum() == 5.0]) >= 2
    assert not any(c.findData("starless") >= 0 for c in page.panel.findChildren(QComboBox))   # no Split in a channel
    with pytest.raises(ValueError, match="Stars/background"):                          # its channel needs its Split
        m.set_fusion_image(m.view(ho.pid), "Ha", "starless")
