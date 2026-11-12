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
        assert {"inputs", "weights", "percentile"} <= set(c.Params.model_fields)
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
    project2._add_align_on(p, o, 1, "stars")                                # the second channel on the first one's grid
    f = p.add_workflow("H-O", CLASSIC)
    f.source = {"fusion": {"type": "ho", "inputs": [{"pid": 1, "role": "Ha", "channel": "L"},
                                                    {"pid": o.id, "role": "OIII", "channel": "L"}]}}
    project2.save(p)
    q = project2.load(p.root)
    k = Keys(q)
    assert k.chain(f.id) == ["combine_ho", "render_simple"] and k.deps(f.id, "render_simple") == ("combine_ho",)
    assert k.cross_inputs(f.id, "combine_ho") == [(1, "render_simple"), (1, "astrometry"), (o.id, "render_simple"),
                                                  (o.id, "astrometry")]
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
    project2._add_align_on(q, q.workflow(o.id), 1, "stars")                  # the block on the first one: one grid
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


# ---------------------------------------------------------------------------------------------- saved projects

#: the former block's palettes → the type, and the settings the block gets (beside the smoothing)
EXPECTED = {"SHO Hubble": ("combine_sho", {}), "SHO": ("combine_sho", {"hue_sii": 0.0, "hue_ha": 120.0, "hue_oiii": 240.0}),
            "HSO": ("combine_sho", {"hue_sii": 120.0, "hue_ha": 0.0, "hue_oiii": 240.0}),
            "HOO": ("combine_ho", {}), "RGB": ("combine_rgb", {}), "LRGB": ("combine_lrgb", {})}


@pytest.mark.parametrize("palette", list(EXPECTED))
@pytest.mark.parametrize("stars", [None, {"mode": "palette"}, {"mode": "channel", "role": "Ha"}])
@pytest.mark.parametrize("graph", [False, True])
def test_a_saved_fusion_opens_as_the_block_of_its_palette(tmp_path, nights, palette, stars, graph):  # noqa: F811 (the fixture)
    """Each palette × option A (stars or not) × a workflow made before the graphs or after: the project opens with the
    block of its palette's type, the settings that reproduce it, its graph valid; the keys of the blocks before the
    combination (its channels' workflows) are those of the project without it."""
    p, o = _channels(tmp_path, nights, FULL if stars else CLASSIC)
    roles = C.FORMER_PALETTES[palette]
    ports = C.BY_KIND[roles[0]].ports
    for r in ports:                                                          # one channel workflow per port
        if r not in ("Ha", "OIII"):
            w = p.add_workflow(r, FULL if stars else CLASSIC)
            w.sessions = [1]
    pids = {"Ha": 1, "OIII": o.id} | {w.name: w.id for w in p.workflows if w.name in ports}
    project2.save(p)
    before = {pid: Keys(project2.load(p.root)).keys(pid) for pid in pids.values()}
    d = json.loads((p.root / "project.astroproj").read_text(encoding="utf-8"))
    old = {"palette": palette, "inputs": [{"pid": pids[r], "role": r, "channel": "L"} for r in ports],
           "weights": {ports[-1]: 0.5}}
    opts = wfl.check_options({**CLASSIC, "denoise": "pretrained", "stars": "neural"} if stars else CLASSIC)
    if stars:
        old |= {"stars": stars, "added": [{"pid": 1, "denoise": None, "stars": None}]}
    fid = d["next_workflow_id"]
    d["workflows"].append({"id": fid, "name": palette, "options": opts, "params": {"fusion": {"percentile": 99.0}},
                           "source": {"fusion": old},
                           "blocks": wfl.from_options(opts, None, "fusion") if graph else None})
    d["next_workflow_id"] = fid + 1
    (p.root / "project.astroproj").write_text(json.dumps(d), encoding="utf-8")
    q = project2.load(p.root)
    w = q.workflow(fid)
    stage, values = EXPECTED[palette]
    # option A (stars from a workflow or a channel): the inputs read their starless images, the stars side is a layer
    reads_starless = bool(stars) and stars["mode"] != "palette" and "Ha" in ports       # (the role must be a port)
    ins = [dict(i, image="starless") if reads_starless else i for i in old["inputs"]]
    assert w.source == {"fusion": {"type": stage.split("_")[1], "inputs": ins}}
    smooth = {"smoothing": "radius", "radius": 0.0} if C.BY_KIND[stage.split("_")[1]].narrowband and not stars else {}
    assert w.params == {stage: {"percentile": 99.0, **values, "weights": {ports[-1]: 0.5}, **smooth}}
    k = Keys(q)
    assert k.chain(fid) == [stage, "render_simple"] + (["star_removal"] if stars else [])
    k.key(fid, "render_simple")                                              # its settings valid, its key made
    assert {pid: k.keys(pid) for pid in pids.values()} == before            # the channels' results stay valid
    project2.save(q)                                                         # saved in the new form, opened again
    assert project2.load(p.root).workflow(fid).source == w.source


def test_a_former_fusion_reading_more_channels_than_its_palette_keeps_its_ports_only(tmp_path, nights):  # noqa: F811 (the fixture)
    p, o = _channels(tmp_path, nights)
    project2.save(p)
    d = json.loads((p.root / "project.astroproj").read_text(encoding="utf-8"))
    d["workflows"].append({"id": 9, "name": "HOO", "options": CLASSIC, "params": {}, "source": {"fusion": {
        "palette": "HOO", "inputs": [{"pid": 1, "role": "SII"}, {"pid": 1, "role": "Ha"}, {"pid": o.id, "role": "OIII"}]}}})
    d["next_workflow_id"] = 10
    (p.root / "project.astroproj").write_text(json.dumps(d), encoding="utf-8")
    assert [i["role"] for i in project2.load(p.root).workflow(9).source["fusion"]["inputs"]] == ["Ha", "OIII"]


# ---------------------------------------------------------------------------------------------- saved projects (#165)

def _saved(p, mutate):
    """The project's JSON, changed by ``mutate``, written back and opened."""
    d = json.loads((p.root / "project.astroproj").read_text(encoding="utf-8"))
    mutate(d)
    (p.root / "project.astroproj").write_text(json.dumps(d), encoding="utf-8")
    return project2.load(p.root)


def test_option_a_of_a_former_fusion_becomes_a_two_sided_layer(tmp_path, nights):  # noqa: F811 (the fixture)
    p, o = _channels(tmp_path, nights, FULL)
    project2.save(p)
    fid = p.next_workflow_id

    def old(d):
        d["workflows"].append({"id": fid, "name": "HOO", "options": CLASSIC, "params": {}, "source": {"fusion": {
            "palette": "HOO", "stars": {"mode": "channel", "role": "Ha"},
            "inputs": [{"pid": 1, "role": "Ha"}, {"pid": o.id, "role": "OIII"}]}}})
        d["next_workflow_id"] = fid + 1
        d["post"] = {"versions": [{"id": 1, "name": "V", "layers": [
            {"kind": "workflow", "pid": fid, "part": "combined", "id": 7},
            {"kind": "workflow", "pid": 1, "part": "combined", "id": 8}]}], "active": 1}
    q = _saved(p, old)
    assert [i["image"] for i in q.workflow(fid).source["fusion"]["inputs"]] == ["starless", "starless"]
    layers = {x["pid"]: x for x in q.post["versions"][0]["layers"]}
    assert (layers[fid]["starless"], layers[fid]["stars_pid"]) == ("output", 1)       # the combination's Output, Ha's stars
    assert (layers[1]["starless"], layers[1]["stars_pid"]) == ("starless", 1)         # migration 5: its own two sides
    again = project2.Project2.from_dict(q.to_dict(), q.root)                           # nothing moves a second time
    assert again.post == q.post and again.workflow(fid).source == q.workflow(fid).source


def test_the_alignment_a_combination_did_alone_is_made_explicit_once(tmp_path, nights):  # noqa: F811 (the fixture)
    p, o = _channels(tmp_path, nights)
    f = p.add_workflow("H-O", CLASSIC)
    f.source = {"fusion": {"type": "ho", "inputs": [{"pid": 1, "role": "Ha"}, {"pid": o.id, "role": "OIII"}]}}
    project2.save(p)
    q = _saved(p, lambda d: d.pop("join_format"))                                      # saved before the joins were free
    blocks = q.workflow(o.id).blocks
    assert [n["block"] for n in blocks][:5] == ["calibration", "registration", "selection", "stacking", "align_on"]
    align = blocks[4]
    assert (align["reference"], align["method"], align["off"]) == (1, "stars", False)
    assert q.workflow(1).blocks is None and Keys(q).grid(o.id) == Keys(q).grid(1)
    assert Keys(q).chain(o.id)[:6] == ["inventory", "calibrate", "register", "stack", "align_on", "astrometry"]
    project2.save(q)                                                                   # saved with the marker: free
    q.workflow(o.id).blocks = [n for n in q.workflow(o.id).blocks if n["block"] != "align_on"]
    project2.save(q)
    assert not any(n["block"] == "align_on" for n in project2.load(p.root).workflow(o.id).blocks)
    # a combination whose inputs come from one workflow (a dual-band image's channels) needs none
    r = p.add_workflow("one", CLASSIC)
    r.source = {"fusion": {"type": "ho", "inputs": [{"pid": 1, "role": "Ha"}, {"pid": 1, "role": "OIII"}]}}
    project2.save(p)
    assert _saved(p, lambda d: d.pop("join_format")).workflow(1).blocks is None


def test_align_images_of_an_earlier_version_becomes_an_align_on_block(tmp_path):
    from astropy.io import fits
    from astropy.wcs import WCS
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval, wcs.wcs.crpix, wcs.wcs.cdelt = [83.8, -5.4], [10, 10], [-0.001, 0.001]
    solved, plain = tmp_path / "a.fits", tmp_path / "b.fits"
    fits.PrimaryHDU(np.zeros((20, 20), np.float32), wcs.to_header()).writeto(solved)
    fits.PrimaryHDU(np.zeros((20, 20), np.float32)).writeto(plain)
    p = project2.create(tmp_path / "proj", "p")
    for name, path in (("ref", solved), ("onto solved", solved), ("onto plain", plain)):
        w = p.add_workflow(name, CLASSIC)
        w.source = {"path": str(path)}
    w1, w2, w3 = p.workflows
    w2.source["align"], w3.source["align"], w1.source["align"] = w1.id, w1.id, w1.id
    d = p.to_dict()
    d.pop("join_format")
    q = project2.Project2.from_dict(d, p.root)
    assert "align" not in q.workflow(w2.id).source
    first = lambda w: q.workflow(w.id).blocks[0]  # noqa: E731
    assert (first(w2)["block"], first(w2)["reference"], first(w2)["method"]) == ("align_on", w1.id, "plate")
    assert (first(w3)["reference"], first(w3)["method"]) == (w1.id, "stars")           # one header without a solution
    assert q.workflow(w1.id).blocks is None                                            # the reference itself: as it is
    with pytest.raises(ValueError, match="unknown keys"):
        project2.check_source({"path": "x.fits", "align": 1})


def test_a_loop_between_workflows_is_refused_and_a_combination_is_no_reference(tmp_path, nights):  # noqa: F811 (the fixture)
    p, o = _channels(tmp_path, nights)
    f = p.add_workflow("H-O", CLASSIC)
    f.source = {"fusion": {"type": "ho", "inputs": [{"pid": 1, "role": "Ha"}, {"pid": o.id, "role": "OIII"}]}}
    project2._add_align_on(p, o, 1, "stars")
    p.validate()
    assert project2.join_problem(p, 1, o.id) == project2.JOIN_LOOP.format(src=o.name, target=p.workflow(1).name)
    assert project2.join_problem(p, 1, 1) == project2.JOIN_SELF
    assert project2.join_problem(p, 1, f.id, reference=True) == project2.JOIN_NO_STACK
    assert project2.join_problem(p, f.id, 1) == "" and project2.join_problem(p, 1, f.id) == project2.JOIN_LOOP.format(
        src=f.name, target=p.workflow(1).name)
    project2._add_align_on(p, p.workflow(1), o.id, "stars")                             # 1 on o, o on 1
    with pytest.raises(ValueError, match="loop"):
        p.validate()
    p.workflow(1).blocks = None
    p.workflow(o.id).blocks[4]["reference"] = f.id                                       # a combination as a reference
    with pytest.raises(ValueError, match="no stack"):
        p.validate()
