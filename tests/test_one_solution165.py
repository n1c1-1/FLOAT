"""One plate solution for aligned workflows (issue #165): the reference wire of an Align on block leaves from a block of the
reference workflow; from its Astrometry (or a block after it) it carries the plate solution, which Align on outputs with the
aligned image, so the blocks after it receive it through their own line. From the Stacking it brings none."""
from astro_gpu.core import project2, workflow as wfl
from astro_gpu.core.store import Keys, Store, StoreProject
from astro_gpu.core.workflow import TEMPLATES
from tests.joins import add_align_on


def two(tmp_path, wire_from="astrometry", own_astrometry=False, ref_astrometry=True):
    p = project2.create(tmp_path / "p", "p")
    a = p.add_workflow("ref", TEMPLATES["Classic stack"])
    b = p.add_workflow("img", TEMPLATES["Classic stack"])
    a.source, b.source = {"path": str(tmp_path / "ref.fits")}, {"path": str(tmp_path / "img.fits")}
    add_align_on(p, b, a.id, "stars")
    if not own_astrometry:
        b.blocks = [n for n in b.blocks if n["block"] != "astrometry"]
    a.blocks = wfl.check_blocks(wfl.with_output(wfl.from_options(a.options, a.end, "frames"), "frames"), "frames")
    a.end = None
    if not ref_astrometry:
        a.blocks = [n for n in a.blocks if n["block"] not in ("astrometry", "color")]
    if wire_from:
        next(n for n in b.blocks if n["block"] == "align_on")["reference_block"] = wire_from
    return p, a, b


def test_the_wire_from_the_astrometry_carries_the_solution_to_the_blocks_after_align_on(tmp_path):
    p, a, b = two(tmp_path)
    keys = Keys(p)
    ch = keys.chain(b.id)
    assert "astrometry" not in ch and "color" in ch and ch.index("align_on") < ch.index("color")
    assert keys.solution(b.id) == {d: (a.id, d) for d in ("astrometry", "reference", "catalogs", "surveys") if d in keys.chain(a.id)}
    assert (a.id, "astrometry") in keys.cross_inputs(b.id, "align_on")
    assert keys.cross_inputs(b.id, "color") == []                       # the block reads Align on's output, nothing else
    assert "align_on" in keys.deps(b.id, "color")
    assert keys.wiring_problem(b.id, "color") == ""
    sp = StoreProject(keys, b.id, "color")
    assert sp.stage_dir("catalogs") == sp.stage_dir("align_on") == sp.dirs["align_on"]


def test_the_reference_makes_the_catalogues_a_colour_block_after_align_on_reads(tmp_path):
    p, a, b = two(tmp_path)
    assert p.catalogs_wanted(a) and "catalogs" in Keys(p).chain(a.id)
    p2, a2, b2 = two(tmp_path / "x", wire_from=None)
    assert not p2.catalogs_wanted(a2)


def test_the_solution_changes_the_key_of_align_on(tmp_path):
    p, a, b = two(tmp_path)
    k1 = Keys(p).key(b.id, "align_on")
    q, a2, b2 = two(tmp_path / "y", wire_from=None)
    assert Keys(q).key(b2.id, "align_on") != k1


def test_an_astrometry_of_its_own_is_used(tmp_path):
    p, a, b = two(tmp_path, own_astrometry=True, wire_from=None)
    keys = Keys(p)
    assert keys.solution(b.id) == {}
    assert keys.wiring_problem(b.id, "color") == ""


def test_a_wire_from_the_stacking_brings_no_plate_solution(tmp_path):
    p, a, b = two(tmp_path, wire_from="stacking")
    keys = Keys(p)
    assert keys.solution(b.id) == {}
    assert keys.wiring_problem(b.id, "color") == "the reference brings no plate solution"
    assert keys.wiring_problem(b.id, "gradient") == ""         # its survey is optional
    sp = StoreProject(keys, b.id, "gradient")
    assert not sp.stage_dir("surveys").is_dir()                 # an optional stage that is not there stays absent


def test_a_reference_without_astrometry_brings_none(tmp_path):
    p, a, b = two(tmp_path, wire_from=None, ref_astrometry=False)
    assert Keys(p).wiring_problem(b.id, "color") == "the reference brings no plate solution"


def test_the_wire_from_a_block_after_the_astrometry_carries_it_too(tmp_path):
    p, a, b = two(tmp_path, wire_from="gradient")
    assert Keys(p).solution(b.id)


def test_align_on_hands_the_solution_on(tmp_path):
    from astro_gpu.stages.align_on import AlignOn
    import json

    class Ctx:
        def __init__(self, d):
            self.d, self.metrics = d, {}

        def out(self, name):
            return self.d / name

        def metric(self, k, v):
            self.metrics[k] = v
    src = tmp_path / "astro"
    src.mkdir()
    (src / "wcs.fits").write_bytes(b"w")
    (src / "preview.png").write_bytes(b"x")
    (src / "manifest.json").write_text(json.dumps({"metrics": {"scale_arcsec": 1.5}}), encoding="utf-8")
    out = tmp_path / "out"
    out.mkdir()
    c = Ctx(out)
    AlignOn.hand_on_solution(c, {"astrometry": str(src)})
    assert sorted(f.name for f in out.iterdir()) == ["wcs.fits"] and c.metrics == {"scale_arcsec": 1.5}


def test_the_graph_accepts_color_after_align_on_without_astrometry():
    g = [wfl.node("stacking"), wfl.node("align_on", reference=1, reference_block="astrometry"), wfl.node("color"),
         wfl.node("output")]
    assert wfl.graph_problem(g, "frames") is None
