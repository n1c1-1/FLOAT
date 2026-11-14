"""The combination blocks (stages/combine.py): one block per type (S-H-O, H-O, RGB, LRGB) on a shared base class, their
named inputs, their parameters, their keys, and the saved projects of the former single Combine channels block opened
with the block of their palette's type."""
from __future__ import annotations

import colorsys
import json

import numpy as np
import pytest

from astro_gpu import texts
from astro_gpu.core import project2
from astro_gpu.core import workflow as wfl
from astro_gpu.core.journal import Journal
from astro_gpu.core.stage import Context, StageError
from astro_gpu.core.store import Keys
from astro_gpu.stages import combine as C
from tests.joins import add_align_on
from tests.test_blocks43 import CLASSIC, FULL, _output, _Project, _split, _star, _wcs, nights, project  # noqa: F401

SHO = C.CombineSHO.ports


def _run(tmp_path, cls, params, name="out", said=None):
    """Run a combination's stage into ``tmp_path / name``; ``said``: a list given its log lines, then its metrics."""
    out = tmp_path / name
    pr = _Project(tmp_path, {cls.name: out}, [cls.name])
    st = cls()
    ctx = Context(pr, st, Journal(pr.logs), "cpu")
    if said is not None:
        log = ctx.log
        ctx.log = lambda msg, level="info": (said.append(msg), log(msg, level))[1]
    st.run(ctx, params)
    if said is not None:
        said.append(ctx.metrics)
    return out


def _hue(rgb) -> float:
    return 360 * colorsys.rgb_to_hsv(*(float(v) for v in rgb))[0]


def _sky_field(offsets: dict[str, float], noise: dict[str, float], H: int = 160, W: int = 192, seed: int = 3):
    """Channels on a sky of their own level and noise, a nebula (a wide blob, a different strength in each filter) in
    the middle: {port: (H, W)}, the nebula's mask."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W]
    neb = np.exp(-((xx - W / 2) ** 2 + (yy - H / 2) ** 2) / (2 * 22.0 ** 2)).astype(np.float32)
    amp = {"SII": 20.0, "Ha": 300.0, "OIII": 150.0, "L": 100.0, "R": 60.0, "G": 80.0, "B": 50.0}
    ch = {r: (offsets[r] + amp[r] * neb + rng.normal(0, noise[r], (H, W))).astype(np.float32) for r in offsets}
    return ch, neb > 0.02


# ---------------------------------------------------------------------------------------------- the classes

def test_four_blocks_on_one_base_class_each_with_its_named_inputs_parameters_and_version():
    assert [c.kind for c in C.COMBINATIONS] == ["sho", "ho", "rgb", "lrgb"]
    assert [c.title for c in C.COMBINATIONS] == ["S-H-O", "H-O", "RGB", "LRGB"]
    assert [c.ports for c in C.COMBINATIONS] == [("SII", "Ha", "OIII"), ("Ha", "OIII"), ("R", "G", "B"),
                                                  ("L", "R", "G", "B")]
    for c in C.COMBINATIONS:
        assert issubclass(c, C.Combination) and c.name == f"combine_{c.kind}" and c.version
        assert "run" not in vars(c)                                     # only its combine: the rest is shared
        assert {"inputs", "weights"} <= set(c.Params.model_fields)
        assert ("percentile" in c.Params.model_fields) == (not c.narrowband)      # the narrow-band types match to Hα
    sho = C.CombineSHO.Params()
    assert (sho.hue_sii, sho.hue_ha, sho.hue_oiii) == C.HUBBLE_HUES        # the measured hues, by default
    assert sho.lightness == "noise" and sho.smoothing == "auto" and sho.radius == 0
    ho = C.CombineHO.Params()
    assert (ho.hue_ha, ho.hue_oiii) == (0.0, 180.0) and not hasattr(ho, "hue_sii")
    assert not {"hue_r", "lightness", "smoothing"} & set(C.CombineRGB.Params.model_fields)
    assert wfl.BLOCK_TITLES["combine_ho"] == "H-O" and wfl.BLOCKS["combine_lrgb"] == ("combine_lrgb",)
    assert "fusion" not in wfl.BLOCKS and "fusion" not in wfl.ORDER


def test_the_hues_give_each_filter_its_colour_and_keep_grey_grey():
    """S-H-O's default hues (Rector et al. 2007; measured: tests/test_combine_hubble.py): equal channels give a grey,
    each filter alone its own hue. Hues 0/120/240 are the plain mapping; H-alpha red and S II green, the former HSO;
    the H-O defaults are H-alpha red, O III green and blue. Hues without a grey between them are refused."""
    mat = C.hue_palette(C.HUBBLE_HUES)
    one, nil = np.ones((1, 1), np.float32), np.zeros((1, 1), np.float32)
    assert np.allclose(C.mix({r: one for r in SHO}, SHO, mat)[:, 0, 0], 1.0)
    alone = {r: C.mix({x: one if x == r else nil for x in SHO}, SHO, mat)[:, 0, 0] for r in SHO}
    assert 0 <= _hue(alone["SII"]) < 30 and 30 <= _hue(alone["Ha"]) < 60 and 180 <= _hue(alone["OIII"]) < 240
    assert [round(_hue(alone[r])) for r in SHO] == [round(h) for h in C.HUBBLE_HUES]
    assert C.hue_palette((0, 120, 240)) == np.eye(3).tolist()
    assert C.hue_palette((120, 0, 240)) == [[0, 1, 0], [1, 0, 0], [0, 0, 1]]
    assert C.hue_palette((0, 180)) == [[1, 0], [0, 1], [0, 1]]
    for hues in ((0.0, 30.0, 60.0), (15.0, 45.0, 180.0), (0.0, 30.0)):
        with pytest.raises(ValueError, match="grey"):
            C.hue_palette(hues)


def test_a_combination_aligns_nothing_itself_and_needs_its_ports_and_one_grid(tmp_path):
    H, W = 60, 80
    ha = np.repeat((10 * _star(H, W, 40, 30))[None], 3, 0) + 0.01
    oiii = np.repeat((5 * _star(H, W, 40, 30))[None], 3, 0) + 0.01     # the same star, on the same grid
    _output(tmp_path / "ha", ha, _wcs([41.0, 31.0]))
    _output(tmp_path / "oiii", oiii, _wcs([44.0, 29.0]))
    ins = [{"role": "Ha", "channel": "L", "pid": 1, "dir": str(tmp_path / "ha"), "wcs": str(tmp_path / "ha" / "wcs.fits"),
            "grid": ["g", 1]},
           {"role": "OIII", "channel": "L", "pid": 2, "dir": str(tmp_path / "oiii"),
            "wcs": str(tmp_path / "oiii" / "wcs.fits"), "grid": ["g", 1]}]
    out = _run(tmp_path, C.CombineHO, C.CombineHO.Params(inputs=ins, smoothing="radius"))
    rgb = np.load(out / "drz_full.npy")
    assert rgb.shape == (3, H, W)
    for c in range(3):                                    # every channel peaks on the star
        assert np.unravel_index(np.argmax(rgb[c]), (H, W)) == (30, 40)
    meta = json.loads((out / "drz_meta.json").read_text())
    assert meta["source"] == "fusion" and meta["kind"] == "ho" and meta["narrowband"] and meta["sky_neutral"]
    assert (out / "wcs.fits").is_file()                                  # the first input's plate solution is kept
    with pytest.raises(StageError, match="OIII"):                         # a port without its channel
        _run(tmp_path, C.CombineHO, C.CombineHO.Params(inputs=ins[:1]), "out2")
    # inputs on grids that differ are refused, naming them: no hidden alignment
    apart = [ins[0], dict(ins[1], grid=["g", 2])]
    with pytest.raises(StageError, match=r"O III is not on the grid of Hα: place Align on after its Stacking"):
        _run(tmp_path, C.CombineHO, C.CombineHO.Params(inputs=apart), "out3")
    assert C.grid_problem([{"role": r, "grid": g} for r, g in (("SII", 1), ("Ha", 2), ("OIII", 3))], C.CombineSHO.ports) == (
        "Hα, O III are not on the grid of S II: place Align on after their Stacking")
    assert C.grid_problem([{"role": r, "grid": 1} for r in ("SII", "Ha", "OIII")], C.CombineSHO.ports) == ""
    for t in (C.TO_WIRE, C.GRID_ONE, C.GRID_MANY):                      # the interface translates these templates
        assert t in texts.WIRING_TEXTS


def test_rgb_and_lrgb_combine_as_they_did():
    one = np.ones((1, 1), np.float32)
    ch = {"R": one, "G": 0.5 * one, "B": 0.25 * one}
    assert np.allclose(C.CombineRGB().combine(None, None, ch, {}, None)[:, 0, 0], [1, 0.5, 0.25])
    lrgb = C.CombineLRGB().combine(None, None, {"L": 2 * one, "R": one, "G": one, "B": one}, {}, None)
    assert np.allclose(lrgb, 2.0)                         # the colour of RGB, the luminance of L


def test_each_channels_sky_goes_to_zero_so_that_an_empty_sky_is_grey_with_any_hues():
    """Background neutralisation: channels whose skies differ in level and noise (a weak S II, whose balance factor is
    the largest) give a combined sky at 0 in red, green and blue, whatever the hues."""
    valid = np.ones((160, 192), bool)
    off = {"SII": -0.7, "Ha": 2.5, "OIII": -0.3}
    sig = {"SII": 0.09, "Ha": 0.4, "OIII": 0.2}
    for roles, hues in ((SHO, C.HUBBLE_HUES), (SHO, (0, 120, 240)), (SHO, (120, 0, 240)), (("Ha", "OIII"), (0, 180))):
        ch, neb = _sky_field({r: off[r] for r in roles}, {r: sig[r] for r in roles})
        sky = C.neutralise_sky(ch, valid)
        assert all(abs(sky[r][0] - off[r]) < 3 * sig[r] / 10 + 0.05 for r in roles)      # the levels found
        rgb = C.mix({r: C.balance(ch[r], valid, 99.5)[0] for r in roles}, roles, C.hue_palette(hues))
        far = ~neb
        far[:, 60:132] = False                                               # the frame's two sides, no nebula
        for c in range(3):
            assert abs(float(np.mean(rgb[c][far]))) < 0.02 * float(rgb[c].max()), (hues, c)
    ch, neb = _sky_field(off, sig)                        # without it the channels carry the cast of their offsets
    raw = C.mix({r: C.balance(ch[r], valid, 99.5)[0] for r in off}, SHO, C.hue_palette(C.HUBBLE_HUES))
    assert abs(float(np.mean(raw[0][~neb]))) > 0.005 * float(raw[0].max())


def test_the_neutralised_sky_keeps_a_neutral_nebula_and_lrgb_stays_grey():
    """Channels whose sky is at 0 already (Wing's) come out as they were; the colour of an LRGB combination on a sky at 0
    does not blow its noise up through the luminance ratio."""
    valid = np.ones((160, 192), bool)
    mat = C.hue_palette(C.HUBBLE_HUES)
    ch, neb = _sky_field({r: 0.0 for r in SHO}, {"SII": 0.09, "Ha": 0.4, "OIII": 0.2})
    before = {r: C.balance(ch[r].copy(), valid, 99.5)[0] for r in SHO}
    C.neutralise_sky(ch, valid)
    a, b = C.mix(before, SHO, mat), C.mix({r: C.balance(ch[r], valid, 99.5)[0] for r in SHO}, SHO, mat)
    core = np.zeros_like(neb)
    core[70:90, 86:106] = True
    ra, rb = a[:, core].mean(1), b[:, core].mean(1)
    assert np.allclose(ra / ra.sum(), rb / rb.sum(), atol=0.01)              # the nebula's hue as before
    chans, _ = _sky_field({r: 0.0 for r in "LRGB"}, {"L": 0.2, "R": 0.3, "G": 0.3, "B": 0.3}, seed=5)
    C.neutralise_sky(chans, valid)
    rgb = C.CombineLRGB().combine(None, None, {r: C.balance(chans[r], valid, 99.5)[0] for r in chans}, {}, valid)
    far = ~neb
    far[:, 60:132] = False
    assert float(np.abs(rgb[:, far]).max()) < 0.25 * float(rgb.max())         # no speckle where the sky is noise


def test_the_sho_block_composes_its_parameters_and_shows_the_smoothing_it_used(tmp_path):
    """The stage removes the skies (Output preparation told); Auto smoothing is in its results (chroma_sigma), a
    radius set by hand is that radius, equal lightness changes the image; a port can read a starless image."""
    off = {"SII": -0.7, "Ha": 2.5, "OIII": -0.3}
    ch, _ = _sky_field(off, {"SII": 0.9, "Ha": 0.4, "OIII": 0.2})
    ch = {r: v / 1000 for r, v in ch.items()}            # the units of an image: 0 to 1
    ins = []
    for r in off:
        _output(tmp_path / r, np.repeat(ch[r][None], 3, 0))
        ins.append({"role": r, "channel": "L", "dir": str(tmp_path / r), "pid": 1})      # one grid: no alignment
    auto, fix, eq = [], [], []
    out = _run(tmp_path, C.CombineSHO, C.CombineSHO.Params(inputs=ins), "auto", auto)
    rgb = np.load(out / "drz_full.npy")
    assert abs(float(np.median(rgb[:, :, :40]))) < 0.05 * float(rgb.max())     # the side, away from the nebula
    assert auto[-1]["chroma_sigma"] > 0.5                                     # the radius Auto took, in its results
    fixed = _run(tmp_path, C.CombineSHO, C.CombineSHO.Params(inputs=ins, smoothing="radius", radius=3.0), "fixed", fix)
    none = _run(tmp_path, C.CombineSHO, C.CombineSHO.Params(inputs=ins, smoothing="radius"), "none")
    equal = _run(tmp_path, C.CombineSHO, C.CombineSHO.Params(inputs=ins, smoothing="radius", lightness="equal"), "eq", eq)
    assert fix[-1]["chroma_sigma"] == 3.0 and any("smoothed over 3.0 px" in str(x) for x in fix)
    assert any("equally" in str(x) for x in eq)
    imgs = [np.load(d / "drz_full.npy") for d in (out, fixed, none, equal)]
    assert all(not np.array_equal(imgs[i], imgs[j]) for i in range(4) for j in range(i + 1, 4))
    _split(tmp_path / "SII_split", np.repeat(ch["SII"][None], 3, 0), np.zeros((3,) + ch["SII"].shape))
    starless = [dict(i, image="starless", dir=str(tmp_path / "SII_split")) if i["role"] == "SII" else i for i in ins]
    _run(tmp_path, C.CombineSHO, C.CombineSHO.Params(inputs=starless), "starless")
    with pytest.raises(StageError, match="starless image of the SII channel"):
        _run(tmp_path, C.CombineSHO, C.CombineSHO.Params(inputs=[dict(starless[0], dir=str(tmp_path / "x"))] +
                                                         starless[1:]), "missing")


# ---------------------------------------------------------------------------------------------- workflows and keys

def _channels(tmp_path, nights, options=CLASSIC):  # noqa: F811 (the fixture's value)
    p = project(tmp_path / "p", nights, "per_session")
    p.sessions[0].filter, p.sessions[1].filter = "Ha", "OIII"
    p.workflow(1).sessions = [1]
    p.workflow(1).options = wfl.check_options(options)
    o = p.add_workflow("OIII", options)
    o.sessions = [2]
    return p, o


def test_a_combination_workflow_reads_its_channels_images_by_port(tmp_path, nights):  # noqa: F811 (the fixture)
    p, o = _channels(tmp_path, nights, FULL)
    add_align_on(p, o, 1, "stars")                                # the second channel on the first one's grid
    f = p.add_workflow("H-O", CLASSIC)
    f.source = {"fusion": {"type": "ho", "inputs": [{"pid": 1, "role": "Ha", "channel": "L"},
                                                    {"pid": o.id, "role": "OIII", "channel": "L"}]}}
    project2.save(p)
    q = project2.load(p.root)
    k = Keys(q)
    assert k.chain(f.id) == ["combine_ho", "render_simple"] and k.deps(f.id, "render_simple") == ("combine_ho",)
    assert k.cross_inputs(f.id, "combine_ho") == [(1, "render_simple"), (1, "astrometry"), (o.id, "render_simple"),
                                                  (o.id, "astrometry"), (1, "surveys")]       # + the solution it hands on
    ov = k.overrides(f.id, "combine_ho")
    assert ov["inputs"][1]["dir"].endswith(k.key(o.id, "render_simple")) and ov["inputs"][1]["role"] == "OIII"
    assert ov["inputs"][0]["grid"] == ov["inputs"][1]["grid"] == k.grid(1)    # one grid: the Align on block's
    before = k.key(f.id, "combine_ho")
    q.workflow(o.id).params["stack"] = {"kappa": 6}                         # a channel's workflow changes
    assert Keys(q).key(f.id, "combine_ho") != before
    q = project2.load(p.root)
    q.workflow(f.id).params["combine_ho"] = {"hue_oiii": 200.0, "weights": {"OIII": 0.5}}   # its own settings
    assert Keys(q).key(f.id, "combine_ho") != before
    q = project2.load(p.root)                                               # the starless image of a channel
    q.workflow(f.id).source["fusion"]["inputs"][0]["image"] = "starless"
    k2 = Keys(q)
    assert (1, "star_removal") in k2.cross_inputs(f.id, "combine_ho")
    assert (1, "render_simple") not in k2.cross_inputs(f.id, "combine_ho")
    assert k2.overrides(f.id, "combine_ho")["inputs"][0]["dir"].endswith(k2.key(1, "star_removal"))
    from astro_gpu.core import runner2
    stages = [j.stage for j in runner2.plan_workflows(project2.load(p.root), [f.id]).jobs]
    assert stages[-2:] == ["combine_ho", "render_simple"] and stages.count("render_simple") == 3
    assert "align_on" in stages and stages.index("align_on") < stages.index("combine_ho")
    for bad in ({"type": "ho", "inputs": [{"pid": 99, "role": "Ha"}]},                      # no such workflow
                {"type": "ho", "inputs": [{"pid": 1, "role": "SII"}]},                       # not one of its ports
                {"type": "ho", "inputs": [{"pid": 1, "role": "Ha"}, {"pid": o.id, "role": "Ha"}]},   # a port twice
                {"type": "xyz", "inputs": []}):
        q.workflow(f.id).source = {"fusion": bad}
        with pytest.raises(ValueError):
            q.validate()


def test_inputs_not_on_one_grid_leave_the_combination_out_with_what_follows_and_an_empty_port_waits(tmp_path, nights):  # noqa: F811
    from astro_gpu.core import runner2
    p, o = _channels(tmp_path, nights)
    f = p.add_workflow("H-O", CLASSIC)
    f.source = {"fusion": {"type": "ho", "inputs": [{"pid": 1, "role": "Ha"}, {"pid": o.id, "role": "OIII"}]}}
    project2.save(p)
    q = project2.load(p.root)                                                # saved with the joins free: nothing added
    assert not any(n["block"] == "align_on" for n in q.workflow(o.id).blocks or [])
    k = Keys(q)
    why = "O III is not on the grid of Hα: place Align on after its Stacking"
    assert k.wiring_problem(f.id, "combine_ho") == why
    plan = runner2.plan_workflows(q, [f.id])
    assert plan.left_out[f.id] == {"combine_ho": why, "render_simple": why}
    assert not any(j.stage in ("combine_ho", "render_simple") and j.workflow == f.id for j in plan.jobs)
    q.workflow(f.id).source["fusion"]["inputs"].pop()                        # a port with no workflow: the empty state
    k2 = Keys(q)
    assert k2.wiring_problem(f.id, "combine_ho") == "O III to wire"
    q.validate()
    add_align_on(q, q.workflow(o.id), 1, "stars")                  # the block on the first one: one grid
    q.workflow(f.id).source["fusion"]["inputs"].append({"pid": o.id, "role": "OIII", "channel": "L"})
    assert Keys(q).wiring_problem(f.id, "combine_ho") == ""
    two = q.workflow(1)                                                      # a variant sharing its Registration shares its grid
    var = q.add_workflow("variant", CLASSIC)
    var.sessions = list(two.sessions)
    assert Keys(q).grid(var.id) == Keys(q).grid(1)


def test_a_combination_with_its_stars_taken_off_after_it(tmp_path, nights):  # noqa: F811 (the fixture)
    p, o = _channels(tmp_path, nights)
    f = p.add_workflow("S-H-O", FULL)
    f.source = {"fusion": {"type": "sho", "inputs": [{"pid": 1, "role": "Ha", "channel": "L"},
                                                     {"pid": o.id, "role": "OIII", "channel": "L"}]}}
    project2.save(p)
    k = Keys(project2.load(p.root))
    assert k.chain(f.id) == ["combine_sho", "render_simple", "star_removal"]
    assert k.deps(f.id, "star_removal") == ("render_simple", "combine_sho")


# ---------------------------------------------------------------------------------------------- saved projects (#165)

def _saved(p, mutate):
    """The project's JSON, changed by ``mutate``, written back and opened."""
    d = json.loads((p.root / "project.astroproj").read_text(encoding="utf-8"))
    mutate(d)
    (p.root / "project.astroproj").write_text(json.dumps(d), encoding="utf-8")
    return project2.load(p.root)


def _former_fusion(d):
    d["workflows"].append({"id": 9, "name": "HOO", "options": CLASSIC, "params": {}, "source": {"fusion": {
        "palette": "HOO", "inputs": [{"pid": 1, "role": "Ha"}, {"pid": 1, "role": "OIII"}]}}})
    d["next_workflow_id"] = 10


def _aligned_image(d):
    d["workflows"][0]["source"] = {"path": "a.fits", "align": 2}


def _layer_without_sides(d):
    d["post"] = {"versions": [{"id": 1, "name": "V", "layers": [{"kind": "workflow", "pid": 1, "part": "combined", "id": 7}]}],
                 "active": 1}


@pytest.mark.parametrize("old", [_former_fusion, _aligned_image, _layer_without_sides])
def test_a_project_of_an_earlier_format_does_not_open_changed(tmp_path, nights, old):  # noqa: F811 (the fixture)
    """No conversion inside the program (issue #165): a former Combine channels, an image aligned by its source, a
    Combine layers layer without its stars side fail with a message that says to convert the project first; the file is
    left as it was."""
    p, _ = _channels(tmp_path, nights)
    project2.save(p)
    d = json.loads((p.root / "project.astroproj").read_text(encoding="utf-8"))
    old(d)
    f = p.root / "project.astroproj"
    f.write_text(json.dumps(d), encoding="utf-8")
    before = f.read_bytes()
    with pytest.raises(project2.EarlierVersionProject, match="saved by an earlier version of FLOAT: convert it first"):
        project2.load(p.root)
    assert f.read_bytes() == before
    from astro_gpu import texts
    assert texts.has_translation(project2.EarlierVersionProject.MESSAGE, "fr")      # the catalogue the window reads


def test_a_saved_project_opens_again_as_it_was(tmp_path, nights):  # noqa: F811 (the fixture)
    p, o = _channels(tmp_path, nights)
    f = p.add_workflow("H-O", CLASSIC)
    f.source = {"fusion": {"type": "ho", "inputs": [{"pid": 1, "role": "Ha"}, {"pid": o.id, "role": "OIII"}]}}
    project2.save(p)
    q = project2.load(p.root)
    assert q.to_dict() == p.to_dict() and q.workflow(o.id).blocks is None          # nothing is added to it


def test_a_loop_between_workflows_is_refused_and_a_combination_is_no_reference(tmp_path, nights):  # noqa: F811 (the fixture)
    p, o = _channels(tmp_path, nights)
    f = p.add_workflow("H-O", CLASSIC)
    f.source = {"fusion": {"type": "ho", "inputs": [{"pid": 1, "role": "Ha"}, {"pid": o.id, "role": "OIII"}]}}
    add_align_on(p, o, 1, "stars")
    p.validate()
    assert project2.join_problem(p, 1, o.id) == project2.JOIN_LOOP.format(src=o.name, target=p.workflow(1).name)
    assert project2.join_problem(p, 1, 1) == project2.JOIN_SELF
    assert project2.join_problem(p, 1, f.id, reference=True) == project2.JOIN_NO_STACK
    assert project2.join_problem(p, f.id, 1) == "" and project2.join_problem(p, 1, f.id) == project2.JOIN_LOOP.format(
        src=f.name, target=p.workflow(1).name)
    add_align_on(p, p.workflow(1), o.id, "stars")                             # 1 on o, o on 1
    with pytest.raises(ValueError, match="loop"):
        p.validate()
    p.workflow(1).blocks = None
    p.workflow(o.id).blocks[4]["reference"] = f.id                                       # a combination as a reference
    with pytest.raises(ValueError, match="no stack"):
        p.validate()


def test_a_combinations_output_frame_is_the_whole_frame(tmp_path):
    """The Output of a combination is not cropped (his decision of 13 November: the parts one filter does not cover show
    what happened; the user crops): the whole frame, as for a single stack."""
    from astro_gpu.stages.render_simple import RenderSimple, RenderSimpleParams
    H, W = 90, 120
    star = 10 * _star(H, W, 60, 45)
    ha = np.repeat(star[None], 3, 0) + 0.01
    oiii = np.repeat(0.5 * star[None], 3, 0) + 0.02
    _output(tmp_path / "ha", ha, _wcs([61.0, 46.0]))
    _output(tmp_path / "oiii", oiii, _wcs([61.0, 46.0]))
    va = np.ones((H, W), bool)
    va[:, :7] = False                                  # Hα: no data at the left
    va[H - 3:, W - 4:] = False                         # and a ragged corner at the bottom right
    va[H - 2, W - 3] = True
    vo = np.ones((H, W), bool)
    vo[:5] = False                                     # O III: no data at the top
    vo[:, W - 6:] = False                              # and at the right
    np.save(tmp_path / "ha" / "valid.npy", va)
    np.save(tmp_path / "oiii" / "valid.npy", vo)
    ins = [{"role": r, "channel": "L", "pid": i, "dir": str(tmp_path / r.lower()), "wcs": str(tmp_path / r.lower() / "wcs.fits"),
            "grid": ["g", 1]} for i, r in ((1, "Ha"), (2, "OIII"))]
    out = _run(tmp_path, C.CombineHO, C.CombineHO.Params(inputs=ins))
    rend = tmp_path / "render"
    pr = _Project(tmp_path, {"stack": out, "render_simple": rend}, ["combine_ho", "render_simple"])
    st = RenderSimple()
    ctx = Context(pr, st, Journal(pr.logs), "cpu")
    st.run(ctx, RenderSimpleParams())
    x0, x1, y0, y1 = ctx.metrics["crop"]
    assert (x0, x1, y0, y1) == (0, W, 0, H)                             # the whole frame, uncovered edges included


def test_the_bounds_of_the_match_are_fractions_of_the_inputs_full_scale(tmp_path):
    """Siril's images are 0 to 1, ours are in ADU: the bounds 1e-7 and 0.875 are taken of the Hα input's full scale."""
    ch, _ = _sky_field({"Ha": 0.0, "OIII": 0.0}, {"Ha": 0.4, "OIII": 0.2})        # amplitudes in the hundreds
    ins = []
    for r in ch:
        _output(tmp_path / r, np.repeat(ch[r][None], 3, 0))
        ins.append({"role": r, "channel": "L", "dir": str(tmp_path / r), "pid": 1, "full_scale": 1000.0})
    said = []
    _run(tmp_path, C.CombineHO, C.CombineHO.Params(inputs=ins), "scaled", said)
    line = next(x for x in said if "matched to Ha" in str(x))
    assert "between 0.0001 and 875:" in line and "x +" in line               # 1e-7 x 1000 and 0.875 x 1000
    a = float(line.split(": ")[1].split(" x")[0])
    assert 0.5 < a < 3.5                                                   # the nebulae are 300 and 150 high, the stars none


def test_each_filter_is_matched_linearly_to_the_halpha_on_the_pixels_between_the_bounds():
    """Siril's linear match: the fit ``Ha = a * filter + b`` on the pixels where the H-alpha lies in [low, high]; a
    pixel of the reference above the high bound (a bright star) is ignored."""
    rng = np.random.default_rng(4)
    n = 256
    yy, xx = np.mgrid[:n, :n].astype(np.float32)
    neb = np.exp(-((yy - 128) ** 2 + (xx - 128) ** 2) / (2 * 50.0 ** 2)).astype(np.float32)
    ha = 0.4 * neb + rng.normal(0, 0.002, (n, n)).astype(np.float32) + 0.001
    oiii = 0.04 * neb + rng.normal(0, 0.0005, (n, n)).astype(np.float32) + 0.0002
    ha[100:110, 100:110] = 0.95                         # a clipped star in the reference (over 0.875)
    oiii[100:110, 100:110] = 0.9                        # and a very different value in the other filter
    valid = np.ones((n, n), bool)
    a, b, used = C.linear_match(oiii, ha, valid)
    assert abs(a - 10) < 0.3 and abs(b - 0.001) < 0.002
    assert used < n * n - 100 + 1                       # the star's 100 pixels are out
    a2, _, _ = C.linear_match(oiii, ha, valid, high=1.0)
    assert a2 < 9                                       # with the star in, the fit is pulled away
    assert (C.MATCH_LOW, C.MATCH_HIGH, C.MATCH_REFERENCE) == (1e-7, 0.875, "Ha")
    with pytest.raises(StageError):
        C.linear_match(oiii, np.zeros_like(ha), valid)  # no reference pixel in the bounds


def test_a_combination_reads_the_plate_solutions_without_an_astropy_note(tmp_path):
    """The solution file is a header with no image: the combination reads it with core/imgutil.read_wcs, so no
    "more axes than the image" note reaches the run's log."""
    import warnings

    from astropy.wcs import FITSFixedWarning
    H, W = 60, 80
    for name in ("ha", "oiii"):
        _output(tmp_path / name, np.repeat((5 * _star(H, W, 40, 30))[None], 3, 0) + 0.01, _wcs([41.0, 31.0]))
    ins = [{"role": r, "channel": "L", "pid": i, "dir": str(tmp_path / r.lower()), "wcs": str(tmp_path / r.lower() / "wcs.fits"),
            "grid": ["g", 1]} for i, r in ((1, "Ha"), (2, "OIII"))]
    with warnings.catch_warnings():
        warnings.simplefilter("error", FITSFixedWarning)
        _run(tmp_path, C.CombineHO, C.CombineHO.Params(inputs=ins))


# ---------------------------------------------------------------------------------------------- the plate solution

def _solution_dir(root, crpix):
    """A folder with the files of an Astrometry result: its plate solution."""
    _output(root, np.zeros((3, 8, 8), np.float32) + 0.01, _wcs(crpix))
    return root


def test_a_combination_outputs_the_plate_solution_of_its_reference_row(tmp_path):
    from astropy.io import fits
    from astro_gpu.core.imgutil import read_wcs
    H, W = 60, 80
    ref = _wcs([41.0, 31.0])
    for name in ("ha", "oiii"):
        _output(tmp_path / name, np.repeat((5 * _star(H, W, 40, 30))[None], 3, 0) + 0.01, ref)
    sol = _solution_dir(tmp_path / "ast", [41.0, 31.0])
    # the Ha row is the reference; the O III row, aligned on it, has no Astrometry: its input carries no file
    ins = [{"role": "Ha", "channel": "L", "pid": 1, "dir": str(tmp_path / "ha"), "wcs": str(sol / "wcs.fits"), "grid": ["g", 1]},
           {"role": "OIII", "channel": "L", "pid": 2, "dir": str(tmp_path / "oiii"), "wcs": "", "grid": ["g", 1]}]
    out = _run(tmp_path, C.CombineHO, C.CombineHO.Params(inputs=ins, solution={"astrometry": str(sol)}))
    got, want = read_wcs(out / "wcs.fits"), read_wcs(sol / "wcs.fits")
    assert got.wcs.crpix.tolist() == want.wcs.crpix.tolist() and got.wcs.crval.tolist() == want.wcs.crval.tolist()
    assert got.wcs.cdelt.tolist() == want.wcs.cdelt.tolist()
    assert fits.getheader(out / "wcs.fits")["CTYPE1"] == "RA---TAN"
    # nothing solved: nothing to hand on
    ins[0]["wcs"] = ""
    assert not (_run(tmp_path, C.CombineHO, C.CombineHO.Params(inputs=ins), name="bare") / "wcs.fits").exists()


def _graph(w, drop=()):
    """A workflow's graph drawn from its options, without the blocks named."""
    w.blocks = wfl.check_blocks([n for n in wfl.with_output(wfl.from_options(w.options, w.end, "frames"), "frames")
                                 if n["block"] not in drop], "frames")
    w.end = None


def _with_colour(tmp_path, nights, drop=()):  # noqa: F811
    p, o = _channels(tmp_path, nights, FULL)
    _graph(p.workflow(1), drop)
    _graph(o, drop)
    add_align_on(p, o, 1, "stars")
    f = p.add_workflow("H-O", CLASSIC)
    f.source = {"fusion": {"type": "ho", "inputs": [{"pid": 1, "role": "Ha", "channel": "L"},
                                                    {"pid": o.id, "role": "OIII", "channel": "L"}]}}
    p.set_blocks(f.id, [wfl.node("color"), wfl.node("output")])
    return p, o, f


def test_a_colour_block_after_a_combination_reads_the_plate_solution_of_the_reference_row(tmp_path, nights):  # noqa: F811
    from astro_gpu.core import runner2
    from astro_gpu.core.store import StoreProject
    p, o, f = _with_colour(tmp_path, nights)
    k = Keys(p)
    assert k.chain(f.id) == ["combine_ho", "color", "render_simple"]
    sol = k.solution(f.id)
    assert sol["astrometry"] == (1, "astrometry") and sol["catalogs"] == (1, "catalogs")
    assert p.catalogs_wanted(1) and "catalogs" in k.chain(1)               # the reference row makes them for the Colour block
    assert k.deps(f.id, "color")[0] == "combine_ho" and k.wiring_problem(f.id, "color") == ""
    assert (1, "catalogs") in k.cross_inputs(f.id, "combine_ho")
    sp = StoreProject(k, f.id, "color")
    assert sp.stage_dir("catalogs") == sp.stage_dir("stack") == sp.dirs["combine_ho"]
    stages = [(j.run_wf, j.stage) for j in runner2.plan_workflows(p, [f.id]).jobs]
    assert stages.index((1, "catalogs")) < stages.index((f.id, "combine_ho")) < stages.index((f.id, "color"))
    # the solution is in the combination's key, through the stages it comes from
    before = k.key(f.id, "combine_ho")
    p.workflow(1).params["astrometry"] = {"min_stars": 8}
    assert Keys(p).key(f.id, "combine_ho") != before


def test_a_colour_block_after_a_combination_whose_inputs_have_no_solution_waits_with_its_reason(tmp_path, nights):  # noqa: F811
    from astro_gpu.core import runner2
    p, o, f = _with_colour(tmp_path, nights, drop=("astrometry", "color"))
    k = Keys(p)
    assert k.solution(f.id) == {}
    why = "the combination's inputs bring no plate solution"
    assert k.wiring_problem(f.id, "color") == why and k.wiring_problem(f.id, "combine_ho") == ""
    assert runner2.plan_workflows(p, [f.id]).left_out[f.id]["color"] == why


def test_a_colour_block_runs_on_a_combination_with_the_solution_it_handed_on(tmp_path):
    """The reference row's catalogue files come with the combination's image: Colour reads them as it would after an
    Astrometry block, on the combination's stage, and the field is the identified one."""
    from astro_gpu.stages.color import Color
    H, W = 60, 80
    for name in ("ha", "oiii"):
        _output(tmp_path / name, np.repeat((5 * _star(H, W, 40, 30))[None], 3, 0) + 0.01, _wcs([41.0, 31.0]))
    sol = _solution_dir(tmp_path / "ast", [41.0, 31.0])
    np.savez(sol / "gaia_xp.npz", x=np.zeros(0), y=np.zeros(0), xyz=np.zeros((0, 3)))      # no star with a spectrum
    (sol / "manifest.json").write_text(json.dumps({"metrics": {"solved": True}}), encoding="utf-8")
    ins = [{"role": r, "channel": "L", "pid": i, "dir": str(tmp_path / r.lower()), "wcs": str(sol / "wcs.fits"),
            "grid": ["g", 1]} for i, r in ((1, "Ha"), (2, "OIII"))]
    comb = _run(tmp_path, C.CombineHO, C.CombineHO.Params(inputs=ins, solution={"astrometry": str(sol), "catalogs": str(sol)}))
    assert (comb / "gaia_xp.npz").is_file() and (comb / "wcs.fits").is_file()
    out = tmp_path / "color"
    pr = _Project(tmp_path, {"combine_ho": comb, "stack": comb, "catalogs": comb, "color": out}, ["combine_ho", "color"])
    st = Color()
    ctx = Context(pr, st, Journal(pr.logs), "cpu")
    # past the catalogue and the field checks (read from the combination's folder): it is the star count that stops it
    with pytest.raises(StageError, match="stars with a spectrum|stars with spectra"):
        st.run(ctx, Color.Params(on_failure="stop"))
