"""The proof of INTEGRATION §6 on synthetic definitions: each shape the golden record of existing projects holds (and
those it does not hold yet) converted to a graph, compiled back, and compared: (a) the definition, field by field,
(b) every stage key, block key and override, (c) the review keys, (d) the graph's rows against the drawing of today's
client. Frames are listed from the definition (no disk is read): each frame entry stands for three files."""
import copy
import os
import sys
from pathlib import Path

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from astro_gpu.core import store, workflow as wfl  # noqa: E402
from astro_gpu.core.project2 import Project2, Session, new_post  # noqa: E402
from astro_gpu.graphcore import proof  # noqa: E402
from astro_gpu.graphcore.compile import compile_into, engine_problems  # noqa: E402
from astro_gpu.graphcore.convert import Client, convert  # noqa: E402
from astro_gpu.graphcore.model import check, clone, from_dict, to_dict  # noqa: E402

DISK = "Q:/synthetic"          # never read: the frame lists come from the definition


@pytest.fixture(autouse=True)
def no_disk(monkeypatch):
    def session_frames(self, project, session):
        return {k: [(Path(f) / f"{k}_{j:03d}.fits", 1000 + 10 * i + j) for i, f in enumerate(getattr(session, k))
                    for j in range(3)] for k in store.KINDS}
    monkeypatch.setattr(store.InputScan, "session_frames", session_frames)
    monkeypatch.setattr(store.InputScan, "useful_areas", lambda self, project, sessions: [])
    monkeypatch.setattr(store, "file_id", lambda path: [Path(path).name.lower(), 4096, 2])
    monkeypatch.setattr(store, "content_sample", lambda path: "0123456789abcdef")


# ---------------------------------------------------------------------------------------------- builders
FRAMES = ("calibration", "registration", "selection", "stacking")


def nodes(*spec):
    """Nodes from "block", "block:method", "block!" (off), "gradient~2:rbf"..."""
    out = []
    for s in spec:
        off = s.endswith("!")
        s = s.rstrip("!")
        name, _c, method = s.partition(":")
        block = wfl.base(name)
        out.append(wfl.node(block, method or None, off, name))
    return out


def night(sid, name, filt="", darks=True, flats=True):
    s = Session(sid, name, lights=[f"{DISK}/{name}/lights"], filter=filt)
    if darks:
        s.darks = [f"{DISK}/{name}/darks"]
    if flats:
        s.flats = [f"{DISK}/{name}/flats"]
    return s


def project(name, sessions=(), mode="per_session"):
    return Project2(name=name, sessions=list(sessions), stack_mode=mode)


def flow(p, name, blocks=None, source=None, sessions=None, params=None, options=None, end=None, **kw):
    """A workflow: a graph (``blocks``, its Output added where it goes) or one made before it (``options``)."""
    w = p.add_workflow(name, options or dict(wfl.DEFAULT_OPTIONS), params or {}, end=end)
    w.source, w.sessions = copy.deepcopy(source), sessions
    if blocks is not None:
        p.set_blocks(w.id, blocks)
    for k, v in kw.items():
        setattr(w, k, v)
    return w


def loaded(p):
    """The definition as the current code loads it (checked, its values normalised)."""
    p.validate()
    return Project2.from_dict(p.to_dict())


def fan_in():
    """Per-night calibration, a workflow stacking two nights, its variant, per-night settings, a non-key setting
    differing on a shared block, an AUTO field stored at its default, a night calibrated apart, rows being built,
    frames excluded by hand, breakpoints shared and not."""
    p = project("fan-in", [night(1, "N1"), night(2, "N2"), night(3, "N3", flats=False)])
    full = nodes(*FRAMES, "astrometry", "gradient", "color", "denoise", "stars")
    flow(p, "Both nights", full, sessions=[1, 2], breakpoints=["stacking", "gradient", "output"],
         params={"calibrate@1": {"hot_sigma": 0.8}, "inventory@2": {"max_lights": 50},
                 "calibrate": {"workers": 4}, "register": {"workers": 2, "min_pairs": 12},
                 "stack": {"method": "drizzle", "min_transparency": 0.6, "kappa": 2.5}})
    flow(p, "Both nights · rbf", nodes(*FRAMES, "astrometry", "gradient:rbf", "color"), sessions=[1, 2],
         breakpoints=["stacking"],
         params={"calibrate@1": {"hot_sigma": 0.8}, "inventory@2": {"max_lights": 50}, "calibrate": {"workers": 4},
                 "register": {"workers": 6, "min_pairs": 12}, "stack": {"method": "drizzle", "min_transparency": 0.6,
                                                                       "kappa": 2.5}, "gradient": {"degree": 3}})
    flow(p, "Third night", [], sessions=[3])                                   # a row being built: no block
    flow(p, "All nights", nodes("calibration", "registration"))               # frames without Stacking
    flow(p, "Night 2 apart", nodes(*FRAMES, "astrometry", "gradient"), sessions=[1, 2],
         calibration={"2": {"fork": 5, "darks": [f"{DISK}/other darks"]}})
    p.excluded = [(1, "lights_001.fits")]
    return loaded(p)


def variants_and_taps():
    """One night calibrated with all: variants (another method, a switched-off block with its settings, a block placed
    twice, a trial pending), a block's image sent to Post, a duplicate, settings of a stage the chain has not, a frame
    block placed in parallel."""
    p = project("variants", [night(1, "Only night")], mode="together")
    base = nodes(*FRAMES, "astrometry", "gradient", "color", "denoise")
    flow(p, "Classic", base, params={"stack": {"pixfrac": 0.6}, "gradient": {"samples": 20}})
    flow(p, "Classic · Gradient", nodes(*FRAMES, "astrometry", "gradient"),
         params={"stack": {"pixfrac": 0.6}, "gradient": {"samples": 20}, "render_simple": {"linked": False}})
    flow(p, "Classic copy", base, params={"stack": {"pixfrac": 0.6}, "gradient": {"samples": 20}})
    a = flow(p, "No colour", nodes(*FRAMES, "astrometry", "gradient", "color!", "denoise"),
             params={"stack": {"pixfrac": 0.6}, "gradient": {"samples": 20}, "color": {"min_stars": 80},
                     "deconv": {"iterations": 50}})
    flow(p, "No colour · adapted", nodes(*FRAMES, "astrometry", "gradient", "color!", "denoise:adapted"),
         params={"stack": {"pixfrac": 0.6}, "gradient": {"samples": 20}, "color": {"min_stars": 80}},
         variant_of=a.id)
    flow(p, "Twice", nodes(*FRAMES, "astrometry", "gradient", "color", "gradient~2:rbf"),
         params={"stack": {"pixfrac": 0.6}, "gradient": {"samples": 20}, "gradient~2": {"degree": 1}})
    par = flow(p, "Parallel registration", nodes("calibration", "registration", "selection", "stacking"))
    par.blocks[1]["fork"] = par.id
    return loaded(p)


def combinations():
    """Mono filters aligned on one, a combination with Stars/background, a second combination of the stars alone
    ("· Stars"), one made before the free graph, inputs reading the starless image, channels named."""
    p = project("filters", [night(1, "SII", "SII"), night(2, "Ha", "Ha"), night(3, "OIII", "OIII")])
    chain = nodes(*FRAMES, "astrometry", "gradient", "denoise", "stars")
    s = flow(p, "SII", chain, sessions=[1])
    for sid, name in ((2, "Ha"), (3, "OIII")):
        g = nodes(*FRAMES) + [wfl.node("align_on", reference=s.id, reference_block="astrometry" if sid == 2 else None)] \
            + nodes("astrometry", "gradient", "denoise", "stars")
        flow(p, name, g, sessions=[sid])
    ins = [{"pid": 1, "role": "SII", "channel": "L"}, {"pid": 2, "role": "Ha", "channel": "L"},
           {"pid": 3, "role": "OIII", "channel": "L", "image": "starless"}]
    flow(p, "SHO", nodes("stars"), source={"fusion": {"type": "sho", "inputs": ins}},
         params={"combine_sho": {"lightness": "noise", "hue_sii": 12.0}})
    stars = [{"pid": i, "role": r, "channel": "L", "image": "stars"} for i, r in ((1, "SII"), (2, "Ha"), (3, "OIII"))]
    flow(p, "SHO · Stars", [], source={"fusion": {"type": "sho", "inputs": stars}})
    rgb = [{"pid": 1, "role": "R"}, {"pid": 2, "role": "G"}, {"pid": 3, "role": "B", "channel": "GB"}]
    flow(p, "RGB before", None, source={"fusion": {"type": "rgb", "inputs": rgb}})
    return loaded(p)


def images():
    """Image rows: two on one file (a variant), a stretched one with its stars taken off and one sent as it is (a tap),
    one without its file yet, one aligned on another's Astrometry; Post with two versions."""
    p = project("images")
    a = {"path": f"{DISK}/a.fits", "stretched": False}
    b = {"path": f"{DISK}/b.png", "stretched": True}
    first = flow(p, "A", nodes("astrometry", "gradient", "color"), source=a, params={"image": {"path": a["path"]}})
    flow(p, "A · rbf", nodes("astrometry", "gradient:rbf"), source=a)
    flow(p, "B stars", nodes("stars"), source=b)
    flow(p, "B", [], source=b)
    flow(p, "Empty", [], source={"path": "", "stretched": False})
    g = [wfl.node("align_on", "plate", reference=first.id, reference_block="astrometry")] + nodes("gradient", "color")
    flow(p, "C on A", g, source={"path": f"{DISK}/c.fits", "stretched": False}, params={"align_on": {"method": "plate"}})
    p.post = new_post()
    p.post["versions"][0]["layers"] = [{"id": 1, "kind": "workflow", "pid": 1, "part": None, "name": "A"}]
    p.post["versions"].append({"id": 2, "name": "Version 2",
                               "layers": [{"id": 2, "kind": "workflow", "pid": 1, "part": None, "name": "A"}]})
    return loaded(p)


def legacy():
    """Workflows made before the free graph (options only): mono (its chain has no Color), the Raw stack exit, a cut
    Astrometry with a stored zone, a cut Calibration."""
    p = project("legacy", [night(1, "Ha night", "Ha")], mode="together")
    flow(p, "Classic + denoise", None, options={**wfl.DEFAULT_OPTIONS, "denoise": "pretrained"})
    flow(p, "Raw stack", None, end="stacking")
    flow(p, "No astrometry", None, options={**wfl.DEFAULT_OPTIONS, "cut": ["astrometry"]},
         params={"stack": {"zone": [10, 10, 200, 200]}})
    flow(p, "No calibration", None, options={**wfl.DEFAULT_OPTIONS, "cut": ["calibration"]})
    return loaded(p)


def legacy_colour():
    """Workflows made before the free graph, a colour camera, two nights together: a tree of variants."""
    p = project("legacy colour", [night(1, "N1"), night(2, "N2")], mode="together")
    flow(p, "Classic", None)
    flow(p, "Auto", None, options={**wfl.DEFAULT_OPTIONS, "gradient": "automatic"})
    flow(p, "Full", None, options={**wfl.DEFAULT_OPTIONS, "denoise": "pretrained", "stars": "neural"},
         breakpoints=["stacking"])
    flow(p, "Full again", None, options={**wfl.DEFAULT_OPTIONS, "denoise": "pretrained", "stars": "neural"})
    return loaded(p)


SHAPES = {"fan_in": fan_in, "variants_and_taps": variants_and_taps, "combinations": combinations, "images": images,
          "legacy": legacy, "legacy_colour": legacy_colour}


# ---------------------------------------------------------------------------------------------- the proof
def _drawing(p):
    pytest.importorskip("PySide6")
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import graph_golden
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    return graph_golden.drawing_of(p)


@pytest.mark.parametrize("shape", list(SHAPES))
def test_round_trip_is_the_same_project(shape):
    from astro_gpu.core.store import InputScan, key_rules
    p = SHAPES[shape]()
    keys = proof.keys_table(p, key_rules(), InputScan())
    assert not [k for k in keys.values() if "error" in k]          # the shape is one the engine computes
    g, q, log = proof.round_trip(p)
    assert proof.definition_diff(p, q) == []
    assert proof.keys_diff(p, q, key_rules(), InputScan()) == []
    ra, rb = proof.review_keys(p, Client(p)), proof.review_keys(q, Client(q))
    assert ra == rb
    diffs, n = proof.drawing_check(p, g, q, _drawing(p))
    assert diffs == [] and n > 2, (diffs, log)
    assert to_dict(from_dict(to_dict(g))) == to_dict(g) == to_dict(clone(g))


def test_the_proof_sees_a_difference():
    """Not vacuous: one setting, one node name, one drawn share changed are each found."""
    from astro_gpu.core.store import InputScan, key_rules
    p = variants_and_taps()
    g, q, _log = proof.round_trip(p)
    q.workflow(1).params["stack"]["pixfrac"] = 0.5
    assert proof.definition_diff(p, q) and proof.keys_diff(p, q, key_rules(), InputScan())
    g, q, _log = proof.round_trip(p)
    q.workflow(6).blocks[-2]["name"] = "gradient~3"
    assert proof.definition_diff(p, q)
    g, q, _log = proof.round_trip(p)
    r = next(r for r in g.rows if r.wid == 2 or any(t.wid == 2 for t in g.taps))
    g.taps = [t for t in g.taps if t.wid != 3]                      # the duplicate drawn apart: a share less
    diffs, _n = proof.drawing_check(p, g, q, _drawing(p))
    assert diffs and r is not None


def test_convert_shapes():
    """What the converter makes of each shape: variants, taps, wires, legacy rows, per-row leftovers."""
    g = convert(variants_and_taps())
    by = {r.wid: r for r in g.rows}
    taps = {t.wid: t for t in g.taps}
    assert set(taps) == {2, 3}                                       # the branch sent to Post, the duplicate
    assert by[4].fork and by[5].fork and by[5].trial == 4            # variants; the trial pending
    assert by[4].extra == {"deconv": {"iterations": 50}}            # a stage the chain has not: kept
    off = next(b for b in by[4].blocks if g.blocks[b].kind == "color")
    assert g.blocks[off].off and g.blocks[off].params == {"color": {"min_stars": 80}}
    assert g.blocks[by[6].blocks[-1]].engine == "gradient~2"
    assert not by[7].fork                                            # a frame block in parallel: a stack of its own
    g = convert(fan_in())
    by = {r.wid: r for r in g.rows}
    assert by[2].extra == {"register": {"workers": 6}}               # a setting no key reads, on a shared block
    assert by[1].pauses == ["stacking", "gradient", "output"] and by[2].pauses is None
    night1 = g.blocks[by[1].nights[0][1]]
    assert night1.params == {"calibrate@1": {"hot_sigma": 0.8}} and by[1].calib == {"calibrate": {"workers": 4}}
    stack = next(b for b in by[1].blocks if g.blocks[b].kind == "stacking")
    assert g.blocks[stack].params == {"stack": {"method": "drizzle", "kappa": 2.5}}
    sel = next(b for b in by[1].blocks if g.blocks[b].kind == "selection")
    assert g.blocks[sel].params == {"stack": {"min_transparency": 0.6}}  # an AUTO default, kept as stored
    assert by[5].nights and g.blocks[by[5].nights[1][1]].own["fork"] == 5
    assert by[3].blocks == [] and by[4].all_nights
    g = convert(combinations())
    by = {r.wid: r for r in g.rows}
    assert by[2].align["ref"] == by[1].id and g.blocks[by[2].align["from"]].kind == "astrometry"
    assert by[3].align["from"] is None and by[3].align["from_name"] is None
    ins = [w for w in g.wires if w.dst[0] == by[4].blocks[0]]
    assert [(w.dst[1], w.src[1], w.image) for w in ins] == [("S II", "image", None), ("Hα", "image", None),
                                                            ("O III", "Starless", "starless")]
    assert by[6].legacy and by[6].legacy["frozen"] is False
    g = convert(legacy())
    by = {r.wid: r for r in g.rows}
    assert not any(g.blocks[b].kind == "color" for b in by[1].blocks)  # 5.9: a mono workflow made before
    assert any(g.blocks[b].kind == "crop" for b in by[3].blocks)       # 5.9: its zone


def test_compile_refreshes_the_same_objects():
    p = fan_in()
    g = convert(p)
    keep = (id(p), [id(w) for w in p.workflows], [id(s) for s in p.sessions])
    c = compile_into(g, p)
    assert keep == (id(p), [id(w) for w in p.workflows], [id(s) for s in p.sessions])
    assert c.map[g.rows[0].blocks[0]] == [(1, "registration"), (2, "registration")]
    assert proof.definition_diff(fan_in(), p) == []


def test_compile_numbers_new_rows_and_names_new_blocks():
    """A row and a block made by the core's gestures: the next free workflow id, the smallest free node name."""
    from astro_gpu.graphcore.operations import apply
    p = variants_and_taps()
    g = convert(p)
    row = next(r for r in g.rows if r.wid == 1)
    colour = next(b for b in row.blocks if g.blocks[b].kind == "color")
    g2, res = apply(g, "add_block", kind="gradient", at={"type": "insert", "row": row.id, "after": colour})
    assert res.ok, res.reason_text()
    q = Project2.from_dict(p.to_dict())
    c = compile_into(g2, q)
    new = [b for b in c.names if g2.blocks[b].kind == "gradient"]
    assert [c.names[b] for b in new] == ["gradient~2"]          # free in every chain it is in (not in "Twice")
    assert [n["name"] for n in q.workflow(1).blocks if n["block"] == "gradient"] == ["gradient", "gradient~2"]
    assert [n["name"] for n in q.workflow(6).blocks if n["block"] == "gradient"] == ["gradient", "gradient~2"]
    assert [w.id for w in q.workflows] == [w.id for w in p.workflows]
    g3, res = apply(g2, "duplicate_row", row=row.id)
    assert res.ok, res.reason_text()
    c = compile_into(g3, q)
    assert max(c.wids.values()) == p.next_workflow_id and q.next_workflow_id == p.next_workflow_id + 1


@pytest.mark.parametrize("shape", list(SHAPES))
def test_converted_graph_and_engine(shape):
    """The converted graph of each shape: what the core's validator and the engine say of it (step 3 closes the gaps
    listed in ECARTS_REGLES; here they are only recorded)."""
    g = convert(SHAPES[shape]())
    try:
        check(g)
        core = ""
    except Exception as e:  # noqa: BLE001
        core = str(e)
    assert isinstance(core, str)
    assert engine_problems(g) == []
