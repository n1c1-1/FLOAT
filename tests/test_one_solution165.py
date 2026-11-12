"""One plate solution for aligned workflows (issue #165): a block after Align on reads the reference workflow's Astrometry
when its own workflow has none; an Astrometry placed in the aligned workflow is used as it is."""
from astro_gpu.core import project2, workflow as wfl
from astro_gpu.core.store import Keys, Store, StoreProject
from astro_gpu.core.workflow import TEMPLATES
from tests.joins import add_align_on


def two(tmp_path, own_astrometry=False, ref_astrometry=True):
    p = project2.create(tmp_path / "p", "p")
    a = p.add_workflow("ref", TEMPLATES["Classic stack"])
    b = p.add_workflow("img", TEMPLATES["Classic stack"])
    a.source, b.source = {"path": str(tmp_path / "ref.fits")}, {"path": str(tmp_path / "img.fits")}
    add_align_on(p, b, a.id, "stars")
    if not own_astrometry:
        b.blocks = [n for n in b.blocks if n["block"] != "astrometry"]
    if not ref_astrometry:
        a.blocks = wfl.check_blocks(wfl.with_output(wfl.from_options(a.options, a.end, "frames"), "frames"), "frames")
        a.blocks = [n for n in a.blocks if n["block"] not in ("astrometry", "color")]
    return p, a, b


def test_color_after_align_on_reads_the_reference_catalogs(tmp_path):
    p, a, b = two(tmp_path)
    keys = Keys(p)
    ch = keys.chain(b.id)
    assert "astrometry" not in ch and "color" in ch and ch.index("align_on") < ch.index("color")
    assert keys.borrowed(b.id, "color") == {"catalogs": (a.id, "catalogs")}
    assert (a.id, "catalogs") in keys.cross_inputs(b.id, "color")
    assert keys.wiring_problem(b.id, "color") == ""
    sp = StoreProject(keys, b.id, "color")
    assert sp.stage_dir("catalogs") == Store(p.root).dir("catalogs", keys.key(a.id, "catalogs"))


def test_an_astrometry_of_its_own_is_used(tmp_path):
    p, a, b = two(tmp_path, own_astrometry=True)
    keys = Keys(p)
    assert keys.borrowed(b.id, "color") == {}
    assert StoreProject(keys, b.id, "color").stage_dir("catalogs") == Store(p.root).dir("catalogs", keys.key(b.id, "catalogs"))


def test_without_astrometry_in_the_reference_the_block_is_left_out_with_the_reason(tmp_path):
    p, a, b = two(tmp_path, ref_astrometry=False)
    keys = Keys(p)
    assert keys.wiring_problem(b.id, "color") == "needs Astrometry in the reference workflow"
    assert keys.wiring_problem(b.id, "gradient") == ""         # its survey is optional


def test_the_graph_accepts_color_after_align_on_without_astrometry():
    g = [wfl.node("stacking"), wfl.node("align_on", reference=1), wfl.node("color"), wfl.node("output")]
    assert wfl.graph_problem(g, "frames") is None
