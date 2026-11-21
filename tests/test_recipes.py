"""Combination recipes (stages/recipes.py): the TOML files of resources/combinations, read and checked; the four
types before recipes as four shipped recipes that run on their own blocks, with their keys (RGB and LRGB: the images
they gave before recipes; S-H-O and H-O: the equalisation and the plain composition); the recipe copied into the
project;
the folder's newer version told, applied only on demand; the two recipes added (HOO, HaRGB); the API."""
from __future__ import annotations

import hashlib
import json
import shutil

import numpy as np
import pytest

from astro_gpu import api
from astro_gpu.core import project2
from astro_gpu.core.store import Keys
from astro_gpu.stages import combine as C
from astro_gpu.stages import recipes as R
from tests.joins import add_align_on
from tests.test_blocks43 import CLASSIC, _output, nights  # noqa: F401 (the fixture)
from tests.test_combine import _channels, _run, _sky_field


def _sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


# ---------------------------------------------------------------------------------------------- the files

def test_the_folder_ships_six_recipes_and_the_four_types_are_their_frozen_content():
    found, bad = R.available()
    assert bad == [] and list(found) == ["hargb", "ho", "hoo", "lrgb", "rgb", "sho"]   # the files' order of names
    for kind, cls in C.BY_KIND.items():
        r = found[kind]
        assert R.is_legacy(r) and R.computational(r) == R.LEGACY[kind] and r["name"] == cls.title
        assert R.roles(r) == cls.ports and R.stage_for({"type": kind, "recipe": r}) == cls.name
        for name, q in r["parameters"].items():                      # the defaults are the blocks' own
            assert cls.Params.model_fields[name].default == q["default"], (kind, name)
        # (the blocks' own settings besides the recipe's: Stars, what a narrow-band type makes of the stars alone,
        # and Balance, how its filters are brought to the H-alpha)
        assert set(r["parameters"]) == set(cls.Params.model_fields) - set(C.CombinationParams.model_fields) - {
            "stars", "balance"}
    assert not R.is_legacy(found["hoo"]) and R.stage_for({"type": "hoo", "recipe": found["hoo"]}) == "combine_recipe"
    ex = R.load(R.FOLDER / "_example.toml")                           # the documentation is a valid recipe, not listed
    assert ex["id"] == "example" and "example" not in found
    for r in found.values():
        assert R.check(r) == r                                       # a copy reads back as it was written


@pytest.mark.parametrize("text, where", [
    ("id = \"x\"\nname = \n", "line 2"),                                                 # not TOML
    ("id = \"x\"\nname = \"X\"\nversion = 1\n[[inputs]]\nrole = \"Ha\"\ntype = \"mono\"\n[[steps]]\nop = \"magic\"\n"
     "[output]\nchannels = [\"Ha\"]\nmatrix = [[1], [1], [1]]\n", "line 8, steps[1].op: unknown operation"),
    ("id = \"x\"\nname = \"X\"\nversion = 1\n[[inputs]]\nrole = \"Ha\"\ntype = \"mono\"\nchannel = \"R\"\n",
     "line 7, inputs[1].channel: a monochrome image is read as its grey"),
    ("id = \"x\"\nname = \"X\"\nversion = 1\n[[inputs]]\nrole = \"Ha\"\ntype = \"mono\"\n[[inputs]]\nrole = \"R\"\n"
     "type = \"mono\"\n[[steps]]\nop = \"neutralise_sky\"\n[[steps]]\nop = \"linear_match\"\nreference = \"OIII\"\n",
     "line 14, steps[2].reference: OIII is not one of the inputs"),
    ("id = \"x\"\nname = \"X\"\nversion = 1\n[[inputs]]\nrole = \"Ha\"\ntype = \"mono\"\n[[steps]]\n"
     "op = \"neutralise_sky\"\n[output]\nchannels = [\"Ha\"]\nhues = [\"$hue\"]\n", "line 11, output.hues"),
    ("id = \"x\"\nname = \"X\"\nversion = 0\n", "line 3, version: version must be a whole number"),
])
def test_a_malformed_recipe_is_refused_naming_the_file_the_line_and_the_field(text, where):
    with pytest.raises(R.RecipeError) as e:
        R.parse(text, "bad.toml")
    assert str(e.value).startswith("bad.toml, ") and where in str(e.value), str(e.value)


def test_a_recipe_may_equalise_its_channels_to_a_reference():
    """The ``equalise`` operation: a reference input, the channels equalised ([] : all the others); the shipped
    narrow-band recipes use it, the reference the H-alpha."""
    head = ('id = "x"\nname = "X"\nversion = 1\n[[inputs]]\nrole = "Ha"\ntype = "mono"\n[[inputs]]\nrole = "OIII"\n'
            'type = "mono"\n[[steps]]\nop = "neutralise_sky"\n[[steps]]\nop = "equalise"\n')
    tail = '[output]\nchannels = ["Ha", "OIII"]\nhues = [0, 180]\n'
    r = R.parse(head + 'reference = "Ha"\n' + tail)
    assert r["steps"][1] == {"op": "equalise", "reference": "Ha", "channels": []}
    assert R.parse(head + 'reference = "Ha"\nchannels = ["OIII"]\n' + tail)["steps"][1]["channels"] == ["OIII"]
    for bad, where in ((tail, "reference is missing"), ('reference = "SII"\n' + tail, "SII is not one of the inputs"),
                       ('reference = "Ha"\nlow = 0.1\n' + tail, "steps[2].low: unknown setting"),
                       ('reference = "Ha"\nchannels = ["Ha"]\n' + tail, "inputs other than the reference")):
        with pytest.raises(R.RecipeError, match=where.replace("[", r"\[").replace("]", r"\]")):
            R.parse(head + bad, "bad.toml")
    found = R.available()[0]
    for rid in ("sho", "ho"):
        assert found[rid]["version"] == 2 and found[rid]["steps"][1] == {"op": "equalise", "reference": "Ha",
                                                                         "channels": []}
        assert found[rid]["output"]["lightness"] is None and found[rid]["output"]["strength"] == 1.0


def test_a_bad_file_never_hides_the_others(tmp_path):
    shutil.copy(R.FOLDER / "hoo.toml", tmp_path / "a_good.toml")
    (tmp_path / "b_broken.toml").write_text("id = [\n", encoding="utf-8")
    shutil.copy(R.FOLDER / "hoo.toml", tmp_path / "c_same_id.toml")
    (tmp_path / "_notes.toml").write_text("not a recipe", encoding="utf-8")         # documentation: not read
    found, bad = R.available(tmp_path)
    assert list(found) == ["hoo"]
    assert [e.source for e in bad] == ["b_broken.toml", "c_same_id.toml"]
    assert "already given by another file" in str(bad[1]) and ", line " in str(bad[1])
    listed = api.combination_recipes(tmp_path)
    assert [r["id"] for r in listed["recipes"]] == ["hoo"] and len(listed["refused"]) == 2
    assert R.available(tmp_path / "nowhere") == ({}, [])


# ---------------------------------------------------------------------------------------------- same images, same keys

def _cases(tmp_path):
    """The four blocks' settings on fixed synthetic channels: {case: (block, settings, the hash of drz_full.npy and of
    drz_meta.json they gave before recipes, float-main 72f8bcdc; None for the narrow-band types, whose recipes are
    at version 2: the filters equalised to the H-alpha and mixed as they are)}."""
    off = {"SII": -0.7, "Ha": 2.5, "OIII": -0.3, "L": 5.0, "R": 3.0, "G": -2.0, "B": 7.0}
    sig = {"SII": 0.9, "Ha": 0.4, "OIII": 0.2, "L": 0.02, "R": 0.03, "G": 0.03, "B": 0.03}
    ch, _ = _sky_field(off, sig)
    H, W = ch["Ha"].shape
    ins = {}
    for n, r in enumerate(off):
        _output(tmp_path / r, np.repeat((ch[r] / 1000)[None], 3, 0))
        m = np.zeros((2, H, W), np.uint8)
        m[0, 10 + 10 * n, 20] = 1
        np.save(tmp_path / r / "saturation.npy", m)
        ins[r] = {"role": r, "channel": "L", "pid": n + 1, "dir": str(tmp_path / r),
                  "saturation": str(tmp_path / r / "saturation.npy"), "full_scale": 1.0}
    _output(tmp_path / "dual", np.stack([ch["Ha"], ch["OIII"], 0.8 * ch["OIII"]]) / 1000)
    duo = [{"role": "Ha", "channel": "R", "pid": 9, "dir": str(tmp_path / "dual")},
           {"role": "OIII", "channel": "GB", "pid": 9, "dir": str(tmp_path / "dual")}]
    sho = [ins[r] for r in ("SII", "Ha", "OIII")]
    return {
        "sho": (C.CombineSHO, dict(inputs=sho), None, None),
        "sho_set": (C.CombineSHO, dict(inputs=sho, hue_sii=0.0, hue_ha=120.0, hue_oiii=240.0, weights={"SII": 0.6}),
                    None, None),
        "ho_dual": (C.CombineHO, dict(inputs=duo), None, None),
        "rgb": (C.CombineRGB, dict(inputs=[ins[r] for r in "RGB"], weights={"B": 1.3}), "3acb27703b91c40a",
                "c8276dcb9e39b2c2"),
        "lrgb": (C.CombineLRGB, dict(inputs=[ins[r] for r in "LRGB"]), "09b7231464b86dc6", "e34e6daaba3ad69a"),
    }


@pytest.mark.parametrize("case", ["sho", "sho_set", "ho_dual", "rgb", "lrgb"])
def test_the_four_types_give_their_images_with_or_without_the_recipe_copied_in(tmp_path, case):
    """The images of the RGB and LRGB blocks, byte for byte those they gave before recipes; every block's image with
    the shipped recipe copied in (``recipe``), the same as without."""
    cls, kw, full, meta = _cases(tmp_path)[case]
    out = _run(tmp_path, cls, cls.Params(**kw), case)
    got = (_sha(out / "drz_full.npy"), _sha(out / "drz_meta.json"))
    if full is not None:
        # the metadata as it was, with the balance's linear maps added since (stages/combine.Combination.steps)
        before = json.loads((out / "drz_meta.json").read_text(encoding="utf-8"))
        assert set(before.pop("balance")) == {"how", "maps", "gains", "noise"}
        (tmp_path / "plain.json").write_text(json.dumps(before, indent=1), encoding="utf-8")    # as the run writes
        assert (got[0], _sha(tmp_path / "plain.json")) == (full, meta)
    if cls.saturation_ports:
        assert _sha(out / "saturation.npy") == "59d9a35795cb2bc0"
    copied = _run(tmp_path, cls, cls.Params(**kw, recipe=R.available()[0][cls.kind]), case + "_copy")
    assert (_sha(copied / "drz_full.npy"), _sha(copied / "drz_meta.json")) == got


def test_the_stage_of_the_other_recipes_runs_a_shipped_recipe_the_same_way(tmp_path):
    """The other recipes' stage, its settings in ``values``, on a copy of the S-H-O recipe: the S-H-O block's image."""
    cls, kw, _full, _meta = _cases(tmp_path)["sho_set"]
    block = _run(tmp_path, cls, cls.Params(**kw), "block")
    vals = {k: kw[k] for k in ("hue_sii", "hue_ha", "hue_oiii")}
    out = _run(tmp_path, C.CombineRecipe, C.RecipeParams(inputs=kw["inputs"], weights=kw["weights"], values=vals,
                                                         recipe=R.available()[0]["sho"]), "generic")
    assert _sha(out / "drz_full.npy") == _sha(block / "drz_full.npy")


def _ho_project(tmp_path, nights):  # noqa: F811 (the fixture's value)
    p, o = _channels(tmp_path, nights)
    add_align_on(p, o, 1, "stars")                                # the second channel on the first one's grid
    f = p.add_workflow("H-O", CLASSIC)
    ins = [{"pid": 1, "role": "Ha", "channel": "L"}, {"pid": o.id, "role": "OIII", "channel": "L"}]
    f.source = {"fusion": {"type": "ho", "inputs": ins}}
    return p, o, f, ins


def test_the_recipe_is_copied_into_the_project_and_a_shipped_type_keeps_its_key(tmp_path, nights):  # noqa: F811
    p, o, f, ins = _ho_project(tmp_path, nights)
    k = Keys(p)
    before, parts = k.key(f.id, "combine_ho"), k.key_parts(f.id, "combine_ho")
    f.source = api.combination_source("ho", ins)                   # placed today: the recipe copied in
    assert f.source["fusion"]["recipe"] == R.available()[0]["ho"] and f.source["fusion"]["type"] == "ho"
    p.validate()
    k = Keys(p)
    assert k.chain(f.id) == ["combine_ho", "render_simple"]
    assert k.key(f.id, "combine_ho") == before and k.key_parts(f.id, "combine_ho") == parts   # no key moves
    assert k.overrides(f.id, "combine_ho")["recipe"]["id"] == "ho"
    project2.save(p)
    q = project2.load(p.root)
    assert q.workflow(f.id).source == f.source                     # the copy is part of the definition
    assert Keys(q).key(f.id, "combine_ho") == before


def test_a_newer_file_is_told_and_changes_nothing_until_applied(tmp_path, nights):  # noqa: F811
    folder = tmp_path / "recipes"
    shutil.copytree(R.FOLDER, folder)
    p, o, f, ins = _ho_project(tmp_path, nights)
    f.source = api.combination_source("ho", ins, folder)
    p.workflow(f.id).params["combine_ho"] = {"hue_oiii": 200.0, "weights": {"OIII": 0.5}}
    before = Keys(p).key(f.id, "combine_ho")
    st = api.combination_recipe_status(p, f.id, folder)
    assert (st["project"], st["folder"], st["newer"], st["changed"]) == (2, 2, False, False)
    text = (folder / "ho.toml").read_text(encoding="utf-8")
    assert text.count("version = 2") == 1 and text.count('reference = "Ha"') == 1
    (folder / "ho.toml").write_text(text.replace("version = 2", "version = 3").replace(
        'reference = "Ha"', 'reference = "Ha"\nchannels = ["OIII"]'), encoding="utf-8")
    st = api.combination_recipe_status(p, f.id, folder)
    assert (st["project"], st["folder"], st["newer"], st["changed"]) == (2, 3, True, True)
    assert Keys(p).key(f.id, "combine_ho") == before                # the file alone changes no result
    (folder / "ho.toml").rename(folder / "ho.off")
    assert api.combination_recipe_status(p, f.id, folder)["folder"] is None
    assert Keys(p).key(f.id, "combine_ho") == before                # nor its disappearance
    with pytest.raises(KeyError):
        api.apply_combination_recipe(p, f.id, folder)
    (folder / "ho.off").rename(folder / "ho.toml")
    api.apply_combination_recipe(p, f.id, folder)                   # the user's choice
    w = p.workflow(f.id)
    assert w.source["fusion"]["recipe"]["version"] == 3 and "combine_ho" not in w.params
    assert w.params["combine_recipe"] == {"values": {"hue_oiii": 200.0}, "weights": {"OIII": 0.5}}
    k = Keys(p)
    assert k.chain(f.id)[0] == "combine_recipe" and k.key(f.id, "combine_recipe") != before
    assert k.key_parts(f.id, "combine_recipe")["x"]["recipe"]["steps"][1]["channels"] == ["OIII"]
    assert k.key_parts(f.id, "combine_recipe")["p"] == {"values": {"hue_oiii": 200.0}, "weights": {"OIII": 0.5}}
    api.set_recipe_value(p, f.id, "hue_oiii", 180.0)               # back at the recipe's default: not stored
    assert p.workflow(f.id).params["combine_recipe"] == {"weights": {"OIII": 0.5}}
    with pytest.raises(ValueError, match="hue_oiii"):
        api.set_recipe_value(p, f.id, "hue_oiii", 400.0)
    with pytest.raises(ValueError, match="no parameter"):
        api.set_recipe_value(p, f.id, "hue_sii", 10.0)


def test_a_new_recipe_runs_on_the_recipe_stage_keyed_by_its_copy(tmp_path, nights):  # noqa: F811
    from astro_gpu.core import runner2
    p, o = _channels(tmp_path, nights)
    f = p.add_workflow("HOO", CLASSIC)
    f.source = api.combination_source("hoo", [{"pid": 1, "role": "Ha"}, {"pid": 1, "role": "OIII"}])
    p.validate()
    k = Keys(p)
    assert k.chain(f.id) == ["combine_recipe", "render_simple"] and k.wiring_problem(f.id, "combine_recipe") == ""
    ov = k.overrides(f.id, "combine_recipe")
    assert [i["channel"] for i in ov["inputs"]] == ["R", "GB"]    # the recipe's channels: one colour image read twice
    assert ov["recipe"]["id"] == "hoo"
    key = k.key(f.id, "combine_recipe")
    assert k.key_parts(f.id, "combine_recipe")["x"]["recipe"] == R.computational(f.source["fusion"]["recipe"])
    f.source["fusion"]["recipe"]["output"]["matrix"][2] = [0.0, 0.5]       # another recipe: another result
    f.source["fusion"]["recipe"]["version"] = 9                            # (the version alone keys nothing)
    k2 = Keys(p)
    assert k2.key(f.id, "combine_recipe") != key
    f.source["fusion"]["recipe"]["output"]["matrix"][2] = [0.0, 1.0]
    assert Keys(p).key(f.id, "combine_recipe") == key
    f.source["fusion"]["inputs"].pop()
    assert Keys(p).wiring_problem(f.id, "combine_recipe") == "O III to wire"
    f.source["fusion"]["inputs"].append({"pid": 1, "role": "OIII"})
    stages = [j.stage for j in runner2.plan_workflows(p, [f.id]).jobs]
    assert stages[-2:] == ["combine_recipe", "render_simple"]
    assert api._origin(p, f) == "palette"                          # a narrow-band palette: Post's H-alpha slider off
    for bad in ({"type": "hoo", "inputs": []},                             # a new recipe needs its copy
                {"type": "hoo", "recipe": dict(f.source["fusion"]["recipe"], steps=[{"op": "magic"}]), "inputs": []},
                {"type": "other", "recipe": f.source["fusion"]["recipe"], "inputs": []}):
        f.source = {"fusion": bad}
        with pytest.raises(ValueError):
            p.validate()


# ---------------------------------------------------------------------------------------------- the two new recipes

def test_hoo_reads_the_dual_band_image_twice_and_puts_o_iii_in_green_and_blue(tmp_path):
    ch, neb = _sky_field({"Ha": 2.5, "OIII": -0.3}, {"Ha": 0.4, "OIII": 0.2})
    _output(tmp_path / "dual", np.stack([ch["Ha"], ch["OIII"], ch["OIII"] + 0.1]) / 1000)
    ins = [{"role": r, "pid": 1, "dir": str(tmp_path / "dual")} for r in ("Ha", "OIII")]   # no channel: the recipe's
    said = []
    out = _run(tmp_path, C.CombineRecipe, C.RecipeParams(inputs=ins, recipe=R.available()[0]["hoo"]), "hoo", said)
    rgb = np.load(out / "drz_full.npy")
    assert np.array_equal(rgb[1], rgb[2])                          # O III, in green and blue
    assert any("OIII matched to Ha" in str(s) for s in said) and any("HOO (colour camera dual-band) from" in str(s)
                                                                       for s in said)
    core = neb & (np.hypot(*np.mgrid[:rgb.shape[1], :rgb.shape[2]] - np.array(rgb.shape[1:])[:, None, None] / 2) < 10)
    assert rgb[0][core].mean() > 0 and rgb[1][core].mean() > 0     # both lines in the nebula
    import json
    meta = json.loads((out / "drz_meta.json").read_text())
    assert meta["kind"] == "hoo" and meta["narrowband"]


def _star_field(H=320, W=320, n=60, seed=7):
    """Isolated Gaussian stars (sigma 1.5 px) of random brightness: positions, the image of unit stars."""
    rng = np.random.default_rng(seed)
    pts = []
    while len(pts) < n:
        x, y = rng.uniform(20, W - 20), rng.uniform(20, H - 20)
        if all((x - a) ** 2 + (y - b) ** 2 > 30 ** 2 for a, b in pts):
            pts.append((x, y))
    yy, xx = np.mgrid[0:H, 0:W]
    img = np.zeros((H, W), np.float32)
    amp = rng.uniform(0.02, 0.3, len(pts))
    for (x, y), a in zip(pts, amp):
        img += (a * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * 1.5 ** 2))).astype(np.float32)
    return pts, img


def test_hargb_takes_the_continuum_off_the_halpha_by_the_stars_and_adds_the_line_to_red_and_a_fifth_to_blue(tmp_path):
    """The stars have no line: their H-alpha flux is 0.08 of their red (the filters' scale); the nebula is a line the
    red records diluted. The scale is measured on the stars, the stars leave the H-alpha, its line goes to the red
    and a fifth of it to the blue (Waller 1990; the Telescope Live recombination)."""
    H = W = 320
    pts, stars = _star_field(H, W)
    rng = np.random.default_rng(1)
    yy, xx = np.mgrid[0:H, 0:W]
    neb = np.exp(-((xx - 160) ** 2 + (yy - 160) ** 2) / (2 * 40.0 ** 2)).astype(np.float32)
    noise = lambda s: rng.normal(0, s, (H, W)).astype(np.float32)  # noqa: E731
    img = {"R": 0.01 + stars + 0.004 * neb + noise(0.0005), "G": 0.02 + 0.9 * stars + noise(0.0005),
           "B": 0.005 + 0.7 * stars + noise(0.0005), "Ha": 0.002 + 0.08 * stars + 0.02 * neb + noise(0.0002)}
    for r, a in img.items():
        _output(tmp_path / r, np.repeat(a[None], 3, 0))
    ins = [{"role": r, "channel": "L", "pid": n, "dir": str(tmp_path / r)} for n, r in enumerate(img)]
    said = []
    hargb = R.available()[0]["hargb"]
    out = _run(tmp_path, C.CombineRecipe, C.RecipeParams(inputs=ins, recipe=hargb), "hargb", said)
    m = said[-1]
    assert abs(m["continuum_scale"] - 0.08) < 0.002 and m["continuum_stars"] >= 30
    rgb = np.load(out / "drz_full.npy")
    plain = _run(tmp_path, C.CombineRGB, C.CombineRGB.Params(inputs=ins[:3]), "rgb")
    base = np.load(plain / "drz_full.npy")                 # the RGB alone: the same red, green and blue
    line = rgb - base
    # green untouched, a fifth of the line in blue (up to the skies' levels, measured on another set of sky pixels:
    # the sky of the four channels, not of three)
    assert np.abs(line[1]).max() < 1e-4
    assert np.abs(line[2] - 0.2 * line[0]).max() < 1e-4
    on_stars = np.array([line[0][int(round(y)), int(round(x))] for x, y in pts])
    peaks = np.array([base[0][int(round(y)), int(round(x))] for x, y in pts])
    assert np.median(np.abs(on_stars) / peaks) < 0.05                         # the stars are taken off the line
    assert line[0][150:170, 150:170].mean() > 0.2                              # the nebula's line: 0.02 / 0.08 - 0.004
    fixed = dict(hargb, steps=[*hargb["steps"][:2], dict(hargb["steps"][2], scale=0.08)])
    out2 = _run(tmp_path, C.CombineRecipe, C.RecipeParams(inputs=ins, recipe=fixed), "fixed", said)
    assert said[-1]["continuum_scale"] == 0.08 and np.load(out2 / "drz_full.npy").shape == rgb.shape


def test_the_continuum_measurement_says_what_it_does_while_it_runs(tmp_path):
    """The detection of the stars takes seconds, minutes on an image of stars alone: the progress line and the log
    name the step, the channel, then the number of stars (issue #185: the bar stayed on "channel R" with nothing
    written)."""
    H = W = 320
    pts, stars = _star_field(H, W)
    img = {"R": 0.01 + stars, "G": 0.02 + 0.9 * stars, "B": 0.005 + 0.7 * stars, "Ha": 0.002 + 0.08 * stars}
    rng = np.random.default_rng(2)
    for r, a in img.items():
        _output(tmp_path / r, np.repeat((a + rng.normal(0, 0.0003, (H, W)).astype(np.float32))[None], 3, 0))
    ins = [{"role": r, "channel": "L", "pid": n, "dir": str(tmp_path / r)} for n, r in enumerate(img)]
    said, bar = [], []
    _run(tmp_path, C.CombineRecipe, C.RecipeParams(inputs=ins, recipe=R.available()[0]["hargb"]), "bar", said, bar)
    i = bar.index("Ha: measuring the continuum on the stars")
    assert bar[i - 1].startswith("channel ") and bar[i + 1].endswith("objects found")
    assert bar[i + 2].startswith("Ha: measuring the continuum on ") and bar[i + 2].endswith(" stars")
    assert said.index("Ha: detecting the stars of R to measure the continuum") < said.index(
        next(m for m in said if isinstance(m, str) and "'s continuum" in m))


def test_the_estimate_of_a_hargb_block_counts_its_continuum_measurement(tmp_path, nights):  # noqa: F811
    """The facts of a workflow whose recipe measures the continuum on the stars carry the Mpx of that step (issue
    #185); a recipe without one carries none, and the first costs more."""
    from astro_gpu.core import estimate as est
    p, o = _channels(tmp_path, nights)
    h = p.add_workflow("HaRGB", CLASSIC)
    h.source = api.combination_source("hargb", [{"pid": 1, "role": r} for r in ("R", "G", "B", "Ha")])
    o2 = p.add_workflow("HOO", CLASSIC)
    o2.source = api.combination_source("hoo", [{"pid": 1, "role": "Ha"}, {"pid": 1, "role": "OIII"}])
    p.validate()
    k = Keys(p)
    fh, fo = est.facts(k, h.id), est.facts(k, o2.id)
    assert fh.continuum_mpx == fh.grid_mpx > 0 and fo.continuum_mpx == 0.0
    assert (est.estimate("combine_recipe", fh, False, {}).seconds
            > est.estimate("combine_recipe", fo, False, {}).seconds)


def test_too_few_stars_for_the_continuum_say_so(tmp_path):
    ch, _ = _sky_field({"R": 3.0, "G": 1.0, "B": 2.0, "Ha": 2.5}, {"R": 0.03, "G": 0.03, "B": 0.03, "Ha": 0.4})
    for r, a in ch.items():
        _output(tmp_path / r, np.repeat((a / 1000)[None], 3, 0))
    ins = [{"role": r, "channel": "L", "pid": 1, "dir": str(tmp_path / r)} for r in ch]
    with pytest.raises(C.StageError, match="needs stars"):
        _run(tmp_path, C.CombineRecipe, C.RecipeParams(inputs=ins, recipe=R.available()[0]["hargb"]), "few")
    with pytest.raises(C.StageError, match="no recipe"):
        _run(tmp_path, C.CombineRecipe, C.RecipeParams(inputs=ins), "none")


def test_the_api_lists_the_recipes_for_the_graph():
    got = api.combination_recipes()
    assert got["refused"] == [] and [r["id"] for r in got["recipes"]] == ["hargb", "ho", "hoo", "lrgb", "rgb", "sho"]
    hoo = next(r for r in got["recipes"] if r["id"] == "hoo")
    assert hoo["name"] == "HOO (colour camera dual-band)" and hoo["version"] == 1 and hoo["narrowband"]
    assert [(i["role"], i["type"], i["channel"]) for i in hoo["inputs"]] == [("Ha", "colour", "R"), ("OIII", "colour", "GB")]
    sho = api.combination_recipe("sho")
    assert sho["parameters"]["hue_ha"]["default"] == 31.4 and sho["parameters"]["hue_ha"]["label"] == "Hα hue"
    with pytest.raises(KeyError):
        api.combination_recipe("nothing")


def test_a_project_saved_before_recipes_tells_its_type_s_recipe(tmp_path, nights):  # noqa: F811
    p, o, f, ins = _ho_project(tmp_path, nights)
    st = api.combination_recipe_status(p, f.id)
    assert (st["id"], st["name"], st["project"], st["folder"], st["newer"], st["changed"]) == ("ho", "H-O", 2, 2,
                                                                                                False, False)
    assert R.computational(st["copy"]) == R.LEGACY["ho"]
    api.set_recipe_value(p, f.id, "hue_oiii", 190.0)                       # the H-O block's own setting
    assert p.workflow(f.id).params["combine_ho"] == {"hue_oiii": 190.0}
