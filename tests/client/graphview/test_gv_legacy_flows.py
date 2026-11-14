"""What the tests of today's graph page checked on sessions, filters, combinations, Align on, blocks in parallel, the
composition and the Output, checked again on the new page over a project model (legacy.World: a scratch project whose
frames are empty files, nothing computed). PORT_TABLE.md maps each old test to its port here, to a test of the new page
that already checks it, or to a defect."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from tests.client.graphview.legacy import World, frames

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "graphcore"))
import invariants as inv  # noqa: E402


@pytest.fixture()
def world(qapp, tmp_path):
    made = []

    def make(**kw):
        w = World(tmp_path / f"w{len(made)}", **kw)
        made.append(w)
        return w
    yield make
    for w in made:
        w.close()


def row_named(W, name):
    return next(r for r in W.graph.rows if r.name == name)


def fits_of(folder: Path, flt: str) -> Path:
    from astropy.io import fits
    folder.mkdir(parents=True, exist_ok=True)
    fits.PrimaryHDU(np.zeros((8, 8), np.uint16), header=fits.Header({"FILTER": flt})).writeto(folder / "l1.fits")
    return folder / "l1.fits"


def a_fits(path: Path, shape=(3, 8, 8)) -> str:
    from astropy.io import fits
    fits.PrimaryHDU(np.zeros(shape, np.float32)).writeto(path)
    return str(path)


# ---------------------------------------------------------------------------------------------- sessions
def test_a_second_session_joins_the_chosen_stack_never_in_place_of_its_nights(world):
    """Decision 1: a double-click on the Session card adds the night to the chosen stack: the first night stays read,
    it says so; Ctrl+Z takes it back."""
    W = world()
    full = W.m.paths[0]
    W.choose(W.kinds("session")[0])
    W.g.double_click_palette("session")
    W.settle()
    assert [s.id for s in W.m.p.sessions_of(full.pid)] == [1, 2] and len(W.m.paths) == 1
    assert "Ctrl+Z" in W.said[-1]
    W.g.double_click_palette("session")                         # a third night: still all three
    W.settle()
    assert [s.id for s in W.m.p.sessions_of(full.pid)] == [1, 2, 3]
    W.b.undo()
    W.b.undo()
    W.settle()
    assert [s.id for s in W.m.p.sessions] == [1]


def test_a_session_chosen_with_a_combination_starts_a_stack_of_its_own(world):
    """Decision 1: a Combination chosen, the Session card makes a stack of its own."""
    W = world(workflows=(), filters=("Ha", "OIII"))
    W.template("filters_min")
    W.b.act("add_combination", type="ho")
    W.settle()
    W.choose(W.kinds("combine")[0])
    n = len(W.graph.rows)
    W.g.double_click_palette("session")
    W.settle()
    assert len(W.graph.rows) == n + 1 and W.graph.rows[-1].nights
    assert [len(row_named(W, x).nights) for x in ("Ha", "OIII")] == [1, 1]          # nothing else changed


@pytest.mark.parametrize("filters", [("Ha",), ("Ha", "OIII")])
def test_a_session_added_to_a_filter_flow_follows_the_filter_of_its_frames(world, tmp_path, filters):
    """The night test pass: a second session added to the H-alpha flow, given OIII frames, kept the H-alpha filter.
    Its frames' filter wins: it goes to the OIII flow (made when there is none, with the H-alpha flow's blocks)."""
    W = world(workflows=(), filters=filters, chains={f: ["calibration", "registration", "stacking", "astrometry"]
                                                     for f in filters})
    for q in list(W.m.paths):
        W.m.p.workflow(q.pid).sessions = [next(s.id for s in W.m.p.sessions if s.filter == q.name)]
    ha = W.view("Ha")
    sid = W.m.new_session(ha.pid)
    assert W.m.p.session(sid).filter == "Ha"                                  # joined Ha: took its filter
    W.m.add_session_entries(sid, "lights", [str(fits_of(tmp_path / "oiii2", "OIII"))])
    W.settle()
    assert W.m.p.session(sid).filter == "OIII"
    assert W.m.sessions_of(W.m.view(ha.pid)) == [1]
    oiii = [q for q in W.m.paths if q.source is None and sid in W.m.sessions_of(q)]
    assert len(oiii) == 1
    assert sorted(W.m.filters()) == ["Ha", "OIII"]


def test_names_are_never_empty_nor_another_s(world):
    """Decision 8: a session or a workflow cannot take an empty name or another's."""
    W = world(workflows=(), filters=("Ha", "OIII"))
    W.template("filters_min")
    ha = W.view("Ha")
    assert W.m.name_problem("OIII", sid=1) and W.m.name_problem("  ", sid=1) and not W.m.name_problem("Ha 2", sid=1)
    assert not W.m.rename_session(1, "OIII") and W.m.p.session(1).name == "Ha"
    assert W.m.name_problem("OIII", pid=ha.pid) and W.m.name_problem("", pid=ha.pid)
    W.page.rename(ha.pid, "OIII")                                             # the panel's rename: refused, said
    assert W.m.view(ha.pid).name == "Ha" and "refused" in W.page.canvas.toast.plain
    assert not W.m.rename(ha.pid, "OIII") and W.m.rename(ha.pid, "H-alpha") and W.m.view(ha.pid).name == "H-alpha"


def test_a_new_session_s_name_is_never_another_s(world):
    """GUI exploration 4: Session 1 renamed "Session 2" by the user: the next session is not "Session 2" too."""
    W = world()
    assert W.m.rename_session(1, "Session 2")
    sid = W.m.add_filter_session("")
    assert sid == 2 and W.m.p.session(sid).name == "Session 3"
    W.m.add_session()
    names = [s.name for s in W.m.p.sessions]
    assert len(set(names)) == len(names) == 3


def test_removing_a_stack_or_its_session_says_what_goes(world):
    """Two stacks: the Session removed takes its stack, and says so; Delete workflow takes the Session only it read;
    Ctrl+Z brings both back."""
    W = world(workflows=(), filters=("Ha", "OIII"))
    W.template("filters_min")
    r = W.g.click_cross(row_named(W, "Ha").nights[0][0])
    W.settle()
    assert [q.name for q in W.m.paths] == ["OIII"] and "Ha" in r["toast"] and "Ctrl+Z" in r["toast"]
    W.g.key("Ctrl+Z", target=W.page.canvas)
    W.settle()
    W.g.menu_pick(row_named(W, "OIII").blocks[0], "Delete workflow")
    W.settle()
    assert [q.name for q in W.m.paths] == ["Ha"] and "OIII" in W.page.canvas.toast.plain
    W.g.key("Ctrl+Z", target=W.page.canvas)
    W.settle()
    assert [s.id for s in W.m.p.sessions] == [1, 2] and len(W.m.paths) == 2


def test_a_session_calibration_shows_its_scope_and_has_its_buttons(world):
    """One Calibration for every session: switched off, every session's block looks so; its panel has Delete and
    Switch off. The Session panel has its Delete."""
    from PySide6.QtWidgets import QPushButton
    W = world()
    W.choose(W.kinds("session")[0])
    W.g.double_click_palette("session")
    W.settle()
    cals = W.kinds("calibration")
    assert len(cals) == 2
    W.choose(cals[0])
    texts = [b.text() for b in W.page.panel.findChildren(QPushButton)]
    assert "Delete" in texts and "Switch off" in texts
    W.g.key("D", target=W.page.canvas)
    W.settle()
    assert all(W.graph.blocks[c].off for c in cals)                           # together: every night's
    W.choose(W.kinds("session")[1])
    x = next(b for b in W.page.panel.findChildren(QPushButton) if b.text() == "Delete")
    assert x.isEnabled()


def test_the_deconvolution_and_synthetic_stars_cards_get_their_durations(world):
    """The night test pass: under each method of these two blocks, "…" for good: the worker never estimated them."""
    W = world()
    q = W.m.paths[0]
    W.m._prefetch_estimates(False)
    for block in ("deconv", "synth"):
        for meth in ("psf", "neural") if block == "deconv" else ("measured", "instrument"):
            got = W.m.method_times(q, block, meth)
            assert got is not None and set(got) <= {"gpu", "cpu"} and got, (block, meth, got)


def test_the_session_block_edits_its_frames_and_its_filter(world):
    """Issue #43: a Session's frames and filter edited (its name and filter undone, not its frames); a session added
    and removed is undone."""
    from PySide6.QtWidgets import QComboBox
    W = world(nights=2, stack_mode="per_session")
    raw = W.raw
    darks = frames(raw / "darks", ["d1.ARW"])
    assert W.m.add_session_entries(1, "darks", [str(darks)]) == 1
    bias = frames(raw / "bias", ["o1.ARW"])
    assert W.m.add_session_entries(2, "bias", [str(bias)]) == 1 and W.m.add_session_entries(2, "bias", [str(bias)]) == 0
    assert W.m.use_from_session(1, 2, "darks") and W.m.shared_with(2, "darks") == [1]
    assert W.m.set_session_filter(2, "Ha") and W.m.session(2).filter == "Ha"
    lights = list(W.m.session(2).lights)
    for e in lights:
        assert W.m.remove_session_entry(2, "lights", e)
    assert not W.m.session(2).lights                                                # the last light can go (#58)
    assert W.m.add_session_entries(2, "lights", [W.m.p.abspath(e) for e in lights]) >= 1
    assert W.m.rename_session(2, "second night") and W.m.session(2).name == "second night"
    assert W.m.dirty and W.m.can_undo()
    assert W.m.undo() and W.m.session(2).name != "second night" and W.m.session(2).lights
    assert W.m.redo() and W.m.session(2).name == "second night"
    W.settle()
    ses = next(b for b in W.kinds("session") if W.graph.blocks[b].sid == 2)
    W.choose(ses)
    assert any(cb.currentData() == "Ha" for cb in W.page.panel.findChildren(QComboBox))


# ---------------------------------------------------------------------------------------------- images
def test_a_workflow_from_an_image_starts_on_its_own_row(world, tmp_path):
    W = world()
    v = W.m.add_image_workflow("From an image", a_fits(tmp_path / "m42.fits"))
    W.settle()
    assert v.source["path"].endswith("m42.fits") and W.m.display_keys(v)[0] == "image"
    assert W.m.all_stages(v)[0] == "image"
    img = W.kinds("image")[0]
    r = W.b.row_of(img)
    assert r.name == "From an image" and not r.nights and r.blocks[0] == img
    W.choose(img)
    assert W.m.path.pid == v.pid and W.m.sel_step == "image"


def test_an_already_stretched_image_workflow_goes_to_post(world, tmp_path):
    import cv2
    W = world()
    png = tmp_path / "done.png"
    cv2.imwrite(str(png), np.full((8, 8, 3), 100, np.uint8))
    v = W.m.add_image_workflow("Done", str(png), {"debruitage": False, "etoiles": "retirer", "fond": "simple"},
                               stretched=True)
    assert v.source["stretched"] and W.m.all_stages(v) == ["image", "star_removal"]
    assert W.m.display_keys(v)[:2] == ["image", "etoiles"]


# ---------------------------------------------------------------------------------------------- combinations
def test_a_combination_is_placed_empty_and_waits_for_its_wires_and_one_grid(world):
    """Issue #165: a combination placed has empty inputs (no filter is read), the block waits, greyed (never red) with
    the reason, each input a hollow dot; wired on stacks that are not on one grid it waits for an Align on; a second
    combination is a block of its own; removed, the stacks it read are as they were."""
    W = world(workflows=(), filters=("Ha", "OIII"))
    W.template("filters_min")
    before = {n: list(row_named(W, n).blocks) for n in ("Ha", "OIII")}
    W.choose(None)
    W.g.double_click_palette("combine")
    W.settle()
    cb = W.kinds("combine")[0]
    d = W.page.canvas.blocks[cb].d
    assert "to wire" in d["grey"] and not d["broken"] and "Ctrl+Z" in W.said[-1]
    assert d["inputs"] and not any(wired for _p, _l, _y, wired, _c in d["inputs"])          # hollow dots
    assert "empty, pull a wire here" in W.page.canvas.blocks[cb].toolTip()
    W.b.act("set_combination_type", block=cb, type="ho")
    W.settle()
    for n, port in (("Ha", "Hα"), ("OIII", "O III")):
        r = W.b.act("wire", src=row_named(W, n).blocks[-1], src_port="image", dst=cb, dst_port=port)
        assert r.ok, W.b.say(r.reason)
    W.settle()
    assert not W.page.canvas.blocks[cb].d["grey"] or "grid" in W.page.canvas.blocks[cb].d["grey"]
    W.choose(None)
    W.g.double_click_palette("combine")                          # a second combination: its own block
    W.settle()
    assert len(W.kinds("combine")) == 2
    for c in W.kinds("combine"):
        W.b.act("remove_block", block=c)
    W.settle()
    assert not W.kinds("combine") and {n: list(row_named(W, n).blocks) for n in ("Ha", "OIII")} == before


def _crossing(a, b) -> bool:
    """Whether two wires (point lists) cross: a horizontal segment of one strictly across a vertical one of the other."""
    def segs(pts):
        return list(zip(pts, pts[1:]))
    for (x1, y1), (x2, y2) in segs(a):
        for (u1, v1), (u2, v2) in segs(b):
            for (h1, h2), (w1, w2) in ((((x1, y1), (x2, y2)), ((u1, v1), (u2, v2))),
                                       (((u1, v1), (u2, v2)), ((x1, y1), (x2, y2)))):
                if abs(h1[1] - h2[1]) < 1e-6 and abs(w1[0] - w2[0]) < 1e-6 and \
                        min(h1[0], h2[0]) + 1e-6 < w1[0] < max(h1[0], h2[0]) - 1e-6 and \
                        min(w1[1], w2[1]) + 1e-6 < h1[1] < max(w1[1], w2[1]) - 1e-6:
                    return True
    return False


@pytest.mark.parametrize("key", ["sho_rgb", "filters_full", "dual_x2"])
def test_the_wires_into_a_join_never_cross(world, key):
    """Issue #165: the wires into one combination (or Combine layers) never cross each other."""
    W = world(workflows=(), nights=0)
    W.template(key)
    d = W.b.drawing()
    joins = {b for b, x in W.graph.blocks.items() if x.kind in ("combine", "combine_layers")}
    assert joins
    for j in joins:
        into = [w.points for w in d.wires if w.dst == j]
        assert len(into) >= 2, (key, j)
        assert not any(_crossing(a, b) for i, a in enumerate(into) for b in into[i + 1:]), (key, j)


def test_branch_names_are_numbered_as_names_not_by_their_blocks_number(world):
    """GUI exploration 4: branches are named by their block's kind, numbered only to stay unique."""
    W = world(workflows=("Classic stack",))
    full = W.m.paths[0]
    names = [W.m.add_branch(W.m.view(full.pid), key, "gradient")[0].name for key in ("astrometrie", "fond", "fond")]
    assert names == [f"{full.name} · Gradient", f"{full.name} · Gradient 2", f"{full.name} · Gradient 3"]


def test_a_combination_panel_shows_its_named_inputs_and_its_settings(world):
    import re

    from PySide6.QtWidgets import QComboBox, QDoubleSpinBox, QLabel
    W = world(workflows=(), filters=("Ha", "OIII"))
    W.page.show_templates()
    card = next(c for c in W.page.templates.cards if c.t["key"] == "filters_full")
    card.set_palette("H-O")
    card.mousePressEvent(None)
    W.settle()
    ho = W.view("H-O")
    assert W.m.all_stages(ho) == ["combine_ho", "render_simple"]                    # no Stars/background after it
    W.choose(W.block("fusion", ho))
    shown = {re.sub("<[^>]+>", "", x.text()) for x in W.page.panel.findChildren(QLabel)}
    assert {"Hα", "O III", "H-O"} <= shown and {"Hα hue", "O III hue", "Colour smoothing"} <= shown
    assert W.m.set_fusion_weight(W.m.view(ho.pid), "OIII", 0.5) is True
    assert W.m.p.workflow(ho.pid).params["combine_ho"]["weights"] == {"OIII": 0.5}
    W.page.refresh()
    W.settle()
    assert len([s for s in W.page.panel.findChildren(QDoubleSpinBox) if s.maximum() == 5.0]) >= 2
    assert any(c.findData("starless") >= 0 and c.findData("stars") >= 0 for c in W.page.panel.findChildren(QComboBox))
    W.m.set_fusion_image(W.m.view(ho.pid), "Ha", "stars")                          # a row's stars alone are an image
    assert next(i for i in W.m.p.workflow(ho.pid).source["fusion"]["inputs"] if i["role"] == "Ha")["image"] == "stars"


def test_filters_make_one_workflow_each_and_their_combinations(world):
    """Filters on the sessions, the Filters template (H-O): one stack each, the others aligned on the first, the stars'
    combination and the nebula's, each row its own Stars/background, Post's layer puts the stars on."""
    W = world(workflows=(), filters=("Ha", "OIII"))
    W.page.show_templates()
    card = next(c for c in W.page.templates.cards if c.t["key"] == "filters_full")
    card.set_palette("H-O")
    card.mousePressEvent(None)
    W.settle()
    assert [q.name for q in W.m.paths][-4:] == ["Ha", "OIII", "H-O · Stars", "H-O"]
    ha, oiii, ho_stars, ho = W.m.paths[-4:]
    assert ha.sessions == [1] and oiii.sessions == [2] and W.m.is_mono(ha) and not W.m.is_mono(ho)
    assert ho.source["fusion"]["type"] == ho_stars.source["fusion"]["type"] == "ho"
    assert [i["image"] for i in ho.source["fusion"]["inputs"]] == ["starless"] * 2
    assert [i["image"] for i in ho_stars.source["fusion"]["inputs"]] == ["stars"] * 2
    assert W.m.display_keys(ho) == ["fusion"] and "couleur" not in W.m.display_keys(ha)
    assert "etoiles" in W.m.display_keys(ha)
    assert W.m.align_reference(oiii) == ha.pid and W.m.align_reference(ha) is None
    layers = [x for x in W.m.post_layers() if x.get("kind") == "workflow"]
    assert [(x["pid"], x["part"], x["starless"], x["stars_pid"], x["stars_image"]) for x in layers] == [
        (ho.pid, "combined", "output", ho_stars.pid, "output")]


@pytest.mark.parametrize("title", ["S-H-O", "H-O", "RGB", "LRGB"])
def test_every_filters_combination_takes_the_stars_off_per_filter_and_combines_them_like_the_nebula(world, title):
    """Each filter's row ends with its own Stars/background; the combination reads their starless images, a second
    block of the same type reads their stars (issue #165)."""
    from astro_gpu.client.gallery import PALETTES
    W = world(workflows=(), nights=0)
    W.template("filters_full", palette=title, filters=PALETTES[title])
    m = W.m
    fusions = [q for q in m.paths if q.source and q.source.get("fusion")]
    rows = [q for q in m.paths if q not in fusions]
    assert len(fusions) == 2 and len(rows) == len(PALETTES[title])
    assert [n["block"] for q in rows for n in m.graph(q)].count("stars") == len(rows)
    assert all([n["block"] for n in m.graph(f)] == [] for f in fusions)
    stars, nebula = fusions
    assert nebula.source["fusion"]["type"] == stars.source["fusion"]["type"]
    assert {i["image"] for i in nebula.source["fusion"]["inputs"]} == {"starless"}
    assert {i["image"] for i in stars.source["fusion"]["inputs"]} == {"stars"}
    assert [i["pid"] for i in nebula.source["fusion"]["inputs"]] == [i["pid"] for i in stars.source["fusion"]["inputs"]]
    assert all(not W.page.canvas.blocks[b].d["grey"].endswith("to wire") for b in W.kinds("combine"))


# ---------------------------------------------------------------------------------------------- Align on
def test_align_on_lists_every_other_stack_picks_one_refuses_a_loop_and_own_grid_undoes(world):
    """The Align on of a stack (issue #165): every other stack offered, one chosen in one undo step; a loop greyed
    with its reason; Own grid takes it back, Ctrl+Z gives it again."""
    W = world(workflows=(), filters=("Ha", "OIII", "SII"))
    W.template("filters_min", palette="S-H-O", filters=["SII", "Ha", "OIII"])
    sii, ha, o3 = (row_named(W, n) for n in ("SII", "Ha", "OIII"))
    r = W.g.menu_pick(o3.blocks[-1], "Ha")                     # Align on > Ha
    W.settle()
    assert row_named(W, "OIII").align["ref"] == ha.id
    texts = [t.replace("✓ ", "") for t, _on, _why in r["items"]]
    assert "SII" in texts and "Ha" in texts and "Own grid" in texts
    r = W.g.menu_pick(row_named(W, "Ha").blocks[-1], "OIII", open_only=True)
    loop = [(t, on, why) for t, on, why in r["items"] if t.endswith("OIII")]
    assert loop and not loop[0][1] and "loop" in loop[0][2]
    W.g.menu_pick(row_named(W, "OIII").blocks[-1], "Own grid")
    W.settle()
    assert row_named(W, "OIII").align is None
    W.g.key("Ctrl+Z", target=W.page.canvas)
    W.settle()
    assert row_named(W, "OIII").align["ref"] == ha.id


def test_align_images_is_gone_everywhere(world):
    W = world()
    s = W.m
    assert not hasattr(s, "align_images") and not hasattr(s, "align_group") and not hasattr(s, "plan_align")
    assert "align_images" not in W.page.palette.cards and all("align" not in (q.source or {}) for q in s.paths)
    from astro_gpu.client import graphnode
    assert "align_images" not in graphnode.BY_KEY and "align_on" in graphnode.BY_KEY


def test_inspect_reads_the_aligned_image_after_align_on(world):
    W = world(workflows=(), filters=("Ha", "OIII"))
    W.template("filters_full", palette="H-O", filters=["Ha", "OIII"])
    from astro_gpu.client.steps import Steps
    st = Steps(W.m)
    ha, oiii = W.view("Ha"), W.view("OIII")
    assert W.m.align_reference(oiii) == ha.pid
    assert st.read_stage(oiii, "stack") == "align_on" and st.read_stage(ha, "stack") == "stack"
    assert [x.stage for x in st.visuals(oiii, "align_on")] == ["align_on"]


# ---------------------------------------------------------------------------------------------- blocks in parallel (variants)
def test_a_variant_with_the_same_settings_computes_the_shared_block_once_and_keeps_its_own_settings(world):
    """A block in parallel (here a variant from Selection): the blocks before it are one computation; its settings
    stay its own."""
    from astro_gpu.core.store import Keys
    W = world(workflows=(), chains={"Main": ["calibration", "registration", "selection", "stacking", "astrometry"]})
    W.g.menu_pick(W.block("selection", W.view("Main")), "Variant here")
    W.settle()
    a, b = W.m.paths[0], W.m.paths[1]
    assert W.m.block_key(a, "alignement") == W.m.block_key(b, "alignement")
    assert W.m.stage_key(a, "stack") == W.m.stage_key(b, "stack")       # the same computation, once
    W.m.p.workflow(b.pid).params.setdefault("stack", {})["kappa"] = 2.5
    k = Keys(W.m.p)
    assert k.key(a.pid, "stack") != k.key(b.pid, "stack") and k.key(a.pid, "register") == k.key(b.pid, "register")


def test_a_variant_of_a_variant_has_a_name_of_its_own(world):
    W = world(workflows=(), chains={"Main": ["calibration", "registration", "selection", "stacking", "astrometry"]})
    W.g.menu_pick(W.block("empilement", W.view("Main")), "Variant here")
    W.settle()
    second = W.m.paths[1]
    W.g.menu_pick(W.block("empilement", second), "Variant here")
    W.settle()
    names = [q.name for q in W.m.paths]
    assert len(names) == 3 and len(set(names)) == 3 and all(n.startswith("Main") for n in names)


# ---------------------------------------------------------------------------------------------- Post and the composition
def test_the_layers_post_gives_a_stars_workflow_and_another(world):
    """Before Post is opened, the layers Post will give each workflow: with Stars, its Combine layers; without, its
    Output image; the block holding a workflow's Output is its last one (no Output block)."""
    W = world(workflows=("Full", "Classic stack"))
    full, classic = W.view("Full"), W.view("Classic stack")
    assert full.opts.get("etoiles") == "retirer" and classic.opts.get("etoiles") != "retirer"
    assert W.bm.to_post == {classic.pid}
    assert sorted(W.bm.post_layers) == sorted([("Full · Combine layers", full.pid), ("Classic stack", classic.pid)])
    assert W.bm.node_of(full, "post").key == "etoiles" and W.bm.node_of(classic, "post").key == "couleur"


def test_the_model_adds_changes_and_removes_a_post_layer_and_says_who_did(world):
    from astro_gpu.core.project2 import active_version
    W = world(workflows=("Full", "Classic stack"))
    model, full = W.m, W.view("Full")
    said = []
    model.postChanged.connect(said.append)
    me = object()
    lid = model.add_post_layer(full.pid, "stars", me)
    assert said == [me]
    layers = active_version(model.p.post)["layers"]
    assert [(x["pid"], x["part"], x["name"], x["mode"]) for x in layers][0] == (
        full.pid, "stars", "Full · Stars alone", "Screen")
    defaults = [x["id"] for x in layers[1:]]
    assert len(defaults) == 2
    top = model.add_post_layer(full.pid, "combined")
    assert [x["id"] for x in active_version(model.p.post)["layers"]] == [top, lid] + defaults
    assert model.post_layer(top)["blend"] == "Screen" and model.post_layer(top)["name"] == "Full · Combine layers 2"
    model.set_post_layer(top, me, blend="Linear dodge (add)", sliders={"star_strength": 3.0})
    assert model.post_layer(top)["blend"] == "Linear dodge (add)" and said[-1] is me
    model.remove_post_layer(lid, me)
    assert [x["id"] for x in active_version(model.p.post)["layers"]] == [top] + defaults
    assert model.post_layer(lid) is None
    assert ("Full · Combine layers 2", full.pid) in W.bm.post_layers


def test_removing_a_layer_takes_its_own_filters_with_it(world):
    from astro_gpu.client import postengine
    from astro_gpu.core.project2 import active_version
    W = world(workflows=("Full", "Classic stack"))
    model, full = W.m, W.view("Full")
    lid = model.add_post_layer(full.pid, "starless")
    keep = model.add_post_layer(full.pid, None)
    flt = postengine.new_filter_layer("saturation", lid)
    model.set_post_layers([flt] + model.post_layers(), scope="active")
    model.remove_post_layer(lid)
    ids = [x["id"] for x in active_version(model.p.post)["layers"]]
    assert ids[0] == keep and lid not in ids and flt["id"] not in ids


def test_an_image_layer_counts_among_post_s_layers(world):
    from astro_gpu.client import postengine
    W = world(workflows=("Full", "Classic stack"))
    model, full = W.m, W.view("Full")
    model.add_post_layer(full.pid, "combined")
    model.set_post_layers([postengine.new_image_layer("stack/abc", "Stack")] + model.post_layers(), scope="active")
    assert ("Stack", None) in W.bm.post_layers and len(W.bm.post_layers) == 4


def test_the_stars_panel_sends_each_image_to_post_and_the_stars_alone_can_be_taken_back(world):
    """Issue #29: the Stars/background block's panel sends each of its images to Post and makes a Combine layers; the
    stars alone sent first keep the layers Post gives the workflows; Out of Post takes them back with their filters."""
    from PySide6.QtWidgets import QPushButton

    from astro_gpu.client import postengine
    from astro_gpu.client.postengine import load_layers, save_layers
    W = world(workflows=("Full", "Classic stack"))
    model, full, classic = W.m, W.view("Full"), W.view("Classic stack")
    W.choose(W.block("etoiles", full))
    pn = W.page.panel
    buttons = {b.text().replace("&&", "&"): b for b in pn.findChildren(QPushButton)}
    assert "To Post" in buttons and "Combine layers" in buttons
    model.add_post_layer(full.pid, "stars")
    layers = model.post_layers()
    assert [(x["pid"], x.get("part")) for x in layers] == [(full.pid, "stars"), (classic.pid, None),
                                                           (full.pid, "combined")]
    W.page.refresh()
    W.settle()
    assert "Out of Post" in [b.text() for b in pn.findChildren(QPushButton)]
    lid = layers[0]["id"]
    model.set_post_layers(save_layers(postengine.normalize(load_layers(layers, {full.pid, classic.pid})
                                                           + [postengine.new_filter_layer("saturation", lid)])),
                          scope="active")
    assert model.remove_post_part(full.pid, "stars") == 1
    assert all(x.get("part") != "stars" and x.get("target") != lid for x in model.post_layers())


# ---------------------------------------------------------------------------------------------- the Output is never a block
def no_output_block(W, what):
    assert "output" not in {x.kind for x in W.graph.blocks.values()}, what
    assert all(it.d["title"] != "Output" for it in W.page.canvas.blocks.values()), what
    for q in W.m.paths:
        held = W.bm.node_of(q, "post")
        assert held is None or held.key != "post", (q.name, what)


PALETTE = ["calibration", "registration", "selection", "crop", "stacking", "astrometry", "gradient", "deconv",
           "denoise", "color", "stars", "synth"]


def place_everything(W, what):
    for r in list(W.graph.rows):
        for kind in PALETTE:
            res = W.b.act("add_block", quiet=True, kind=kind, row=r.id)
            if res.ok:
                W.settle(1)
                no_output_block(W, f"{kind} placed in {r.name} ({what})")


def test_no_template_and_no_block_placed_after_it_draws_an_output_block(world):
    """The Output is never a block of the graph (issues #17, #49): every template, then every block of the palette
    placed in each stack, one after the other."""
    from astro_gpu.client.gallery import GALLERY
    for t in [t for t in GALLERY if t.get("copy") is None and t["key"] not in ("image", "images")]:
        W = world(workflows=(), nights=0)
        W.template(t["key"])
        no_output_block(W, t["key"])
        place_everything(W, t["key"])
    assert "synth" in {x.kind for x in W.graph.blocks.values()}


def test_synthetic_stars_after_stars_and_background_draw_no_output_block_and_no_wire_under_a_block(world):
    """11 November: Stars/background, then Synthetic stars: the Output showed between them; the wires went behind the
    block. The block that holds the Output stays the Stars/background block's row's last one."""
    W = world(workflows=("Full",))
    W.choose(W.block("etoiles"))
    W.g.double_click_palette("synth")
    W.settle()
    q = W.m.paths[0]
    assert W.bm.node_of(q, "synth") is not None
    no_output_block(W, "Synthetic stars after Stars/background")
    d = W.b.drawing()
    assert not inv.i1(d) and not inv.i2(W.graph, d) and not inv.i3(W.graph, d)


def test_images_and_their_blocks_draw_no_output_block(world, tmp_path):
    import cv2
    W = world(workflows=(), nights=0)
    for n in ("a", "b"):
        png = tmp_path / f"{n}.png"
        cv2.imwrite(str(png), np.full((8, 8, 3), 100, np.uint8))
        q = W.m.add_image_workflow(n, str(png), stretched=True)
        W.m.set_graph(q, [])
    for n in ("c", "d"):
        W.m.add_image_workflow(n, a_fits(tmp_path / f"{n}.fits"))
    W.settle()
    no_output_block(W, "images")
    place_everything(W, "images")


def test_filter_workflows_and_their_combination_draw_no_output_block(world):
    W = world(workflows=(), filters=("Ha", "OIII"))
    W.template("filters_min")
    no_output_block(W, "two filters")
    W.b.act("add_combination", type="ho")
    W.settle()
    no_output_block(W, "their combination")


def test_no_wire_of_the_templates_passes_under_a_block_nor_goes_right_to_left(world):
    """The wires of the graph never pass under a block, never go right to left, no block covers another: every
    template of the gallery made on the page (the core's own templates: tests/graphcore/test_graph_core.py)."""
    from astro_gpu.client.gallery import GALLERY
    for t in [t for t in GALLERY if t.get("copy") is None]:
        W = world(workflows=(), nights=0)
        W.template(t["key"])
        d = W.b.drawing()
        assert not inv.i1(d) and not inv.i2(W.graph, d) and not inv.i3(W.graph, d), t["key"]
        W.page.set_display("folded")
        d = W.b.drawing()
        assert not inv.i1(d) and not inv.i2(W.graph, d), t["key"]


# ---------------------------------------------------------------------------------------------- what a block says
def test_every_block_says_what_it_does(world):
    W = world(workflows=("Full",))
    by = {it.d["kind"]: it for it in W.page.canvas.blocks.values()}
    assert "Where the workflows' images become layers" in by["post"].toolTip()
    assert "half-stacks" in by["denoise"].toolTip()
    assert "One night" in by["session"].toolTip()


def test_with_lights_a_workflow_s_wire_to_post_is_a_layer(world):
    W = world(workflows=("Classic stack",), lights=True)
    p = W.m.paths[0]
    assert p.pid in W.bm.to_post
    wire = next(w for w in W.graph.wires if w.dst[0] == W.one("post"))
    assert W.page.canvas.wires[wire.id].style_name == "port"
