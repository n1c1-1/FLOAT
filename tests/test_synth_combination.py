"""The Synthetic stars block on the stars of a combination (the "· Stars" workflow of narrow-band templates): the rule,
the stage's input, the plate solution it needs, and the keys of every workflow without the block."""
from __future__ import annotations

import json

import numpy as np
import pytest

from astro_gpu.core import runner2
from astro_gpu.core import workflow as wfl
from astro_gpu.core.journal import Journal
from astro_gpu.core.stage import Context
from astro_gpu.core.store import Keys
from astro_gpu.stages.synth_stars import SynthStars
from tests.joins import add_align_on
from tests.test_blocks43 import CLASSIC, FULL, _Project, nights, project  # noqa: F401
from tests.test_combine import _channels, _graph

WHY = "the combination's inputs bring no plate solution"


def _stars_workflow(tmp_path, nights, drop=(), blocks=None):  # noqa: F811
    p, o = _channels(tmp_path, nights, FULL)
    _graph(p.workflow(1), drop)
    _graph(o, drop)
    add_align_on(p, o, 1, "stars")
    f = p.add_workflow("H-O · Stars", CLASSIC)
    f.source = {"fusion": {"type": "ho", "inputs": [{"pid": 1, "role": "Ha", "channel": "L"},
                                                    {"pid": o.id, "role": "OIII", "channel": "L"}]}}
    p.set_blocks(f.id, blocks or [wfl.node("output"), wfl.node("synth")])
    return p, o, f


def test_the_rule_allows_the_block_right_after_the_output_of_a_combination_and_nowhere_else():
    out = [wfl.node("output")]
    assert wfl.palette_problem(out, "fusion", "synth") is None
    assert wfl.places(out, "fusion", "synth") == [0, 1]                      # the Output is automatic: after it either way
    # the same block of a workflow from frames still needs a Stars/background block
    frames = wfl.from_options({"denoise": "pretrained", "stars": None})
    assert wfl.palette_problem(frames, "frames", "synth") is not None
    # an already stretched image has none
    assert wfl.palette_problem([node for node in wfl.from_options({"denoise": "pretrained", "stars": "neural"}, None, "stretched")],
                               "stretched", "synth") is not None
    # only Stars/background or Synthetic stars after the Output, and Synthetic stars once
    assert wfl.place_problem(out + [wfl.node("synth")], "fusion", "synth", 2) is not None


def test_the_combination_stars_workflow_with_the_block_reads_the_output_and_its_solution(tmp_path, nights):  # noqa: F811
    p, o, f = _stars_workflow(tmp_path, nights)
    k = Keys(p)
    assert k.chain(f.id) == ["combine_ho", "render_simple", "synth_stars"]
    d = k.deps(f.id, "synth_stars")
    assert d[0] == "render_simple" and "combine_ho" in d
    assert k.wiring_problem(f.id, "synth_stars") == ""
    stages = [(j.run_wf, j.stage) for j in runner2.plan_workflows(p, [f.id]).jobs]
    assert (f.id, "synth_stars") in stages and stages.index((f.id, "render_simple")) < stages.index((f.id, "synth_stars"))
    before = k.key(f.id, "synth_stars")
    p.workflow(f.id).params["render_simple"] = {"stretch": "arcsinh"} if False else p.workflow(f.id).params.get("render_simple", {})
    p.workflow(f.id).params["combine_ho"] = {"hue_oiii": 200.0}
    assert Keys(p).key(f.id, "synth_stars") != before                        # the stars it redraws changed


def test_without_a_plate_solution_the_block_waits_with_its_reason(tmp_path, nights):  # noqa: F811
    p, o, f = _stars_workflow(tmp_path, nights, drop=("astrometry", "color"))
    k = Keys(p)
    assert k.solution(f.id) == {}
    assert k.wiring_problem(f.id, "synth_stars") == WHY and k.wiring_problem(f.id, "combine_ho") == ""
    assert runner2.plan_workflows(p, [f.id]).left_out[f.id]["synth_stars"] == WHY


def test_the_stage_redraws_the_stars_of_the_combinations_output_in_its_layout(tmp_path):
    from tests.test_synth_stars import stars_alone
    s, valid, pts, fl = stars_alone()
    src, out = tmp_path / "render_simple", tmp_path / "synth_stars"
    src.mkdir()
    np.save(src / "lin.npy", s)
    np.save(src / "valid.npy", valid)
    prep = {"M": np.eye(3).tolist(), "crop": [0, 640, 0, 640], "stats": {"lum": {"median": 0.0}}, "drizzle_scale": 1.0}
    (src / "prep_simple.json").write_text(json.dumps(prep), encoding="utf-8")
    pr = _Project(tmp_path, {"render_simple": src, "synth_stars": out}, ["render_simple", "synth_stars"])
    st = SynthStars()
    ctx = Context(pr, st, Journal(pr.logs), "cpu")
    st.run(ctx, SynthStars.Params(method="measured"))
    assert (out / "lin.npy").is_file() and (out / "valid.npy").is_file() and (out / "prep_simple.json").is_file()
    assert not (out / "prep_stars.json").exists() and not (out / "stars.npy").exists()
    drawn = np.load(out / "lin.npy")
    assert drawn.shape == s.shape and ctx.metrics["stars_drawn"] + ctx.metrics.get("stars_kept", 0) == len(pts)
    assert json.loads((out / "prep_simple.json").read_text(encoding="utf-8"))["crop"] == prep["crop"]


def _fingerprint(p):
    k = Keys(p)
    return {w.id: {s: k.key(w.id, s) for s in k.chain(w.id)} | {"_chain": k.chain(w.id)} for w in p.workflows}


def test_workflows_without_the_block_keep_their_chain_and_keys(tmp_path, nights):  # noqa: F811
    """The same project with the block added on a combination: every other workflow, and the combination's own earlier
    stages, keep their keys; the block adds one stage and moves none."""
    p, o, f = _stars_workflow(tmp_path, nights, blocks=[wfl.node("output")])
    bare = _fingerprint(p)
    p.set_blocks(f.id, [wfl.node("output"), wfl.node("synth")])
    withb = _fingerprint(p)
    for wid, ch in bare.items():
        for s, key in ch.items():
            if s != "_chain":
                assert withb[wid][s] == key, (wid, s)
    assert withb[f.id]["_chain"] == bare[f.id]["_chain"] + ["synth_stars"]
    # a workflow from frames with Stars/background and the block: the block's deps are what they always were
    fr = Keys(p)
    p.set_blocks(1, wfl.from_options({"denoise": "pretrained", "stars": "neural"}) + [wfl.node("synth")])
    assert Keys(p).deps(1, "synth_stars")[0] == "star_removal"


def test_the_synthetic_stars_of_a_combination_are_stretched_as_the_real_ones_with_the_starless_statistics(tmp_path):
    """What the stage writes is an Output's layout: rendered over the starless combination's statistics (stats_from), the
    redrawn stars take the real stars' stretch, so they replace them without a jump in level."""
    from astro_gpu.render.layers import LayerService
    from tests.test_synth_stars import stars_alone
    s, valid, pts, fl = stars_alone(n=16, size=320)
    prep = {"M": np.eye(3).tolist(), "crop": [0, 320, 0, 320], "drizzle_scale": 1.0,
            "stats": {"lum": {"median": 5.0, "mad": 1.5, "p99995": 400.0}, "channels": [], "lum_p40": 5.0, "lum_p85": 6.0}}
    main, src, out = tmp_path / "main", tmp_path / "render_simple", tmp_path / "synth_stars"
    for d in (main, src):
        d.mkdir()
        np.save(d / "lin.npy", s if d is src else np.full_like(s, 5.0))
        np.save(d / "valid.npy", valid)
        (d / "prep_simple.json").write_text(json.dumps(prep), encoding="utf-8")
    pr = _Project(tmp_path, {"render_simple": src, "synth_stars": out}, ["render_simple", "synth_stars"])
    st = SynthStars()
    st.run(Context(pr, st, Journal(pr.logs), "cpu"), SynthStars.Params(method="measured"))
    svc = LayerService("cpu")
    real = svc.render(src, None, None, 0.5, stats_from=main).rgb
    drawn = svc.render(out, None, None, 0.5, stats_from=main).rgb
    assert real.shape == drawn.shape and drawn.max() > 0.3
    y, x = (pts[5] // 2).tolist()
    around = (slice(None), slice(y - 8, y + 9), slice(x - 8, x + 9))
    assert abs(float(drawn[around].sum()) / float(real[around].sum()) - 1) < 0.35       # the same level, star by star
    assert drawn[:, :20, :20].max() < 0.05 and real[:, :20, :20].max() < 0.05           # the empty sky: black for both
