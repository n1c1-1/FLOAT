"""The project model's facades (client/graphmodel.py, the graph as the truth) against the old model: the same call
gives the same compiled definition (Post included) and the same Keys tables as the old model gave (the comparison of
tests/graphcore/test_h1_old_way.py). Every gallery template made through the model as the graph page makes it gives
the old project too.

The old model went with the swap to the graph: what it reached was recorded before, by this file run with
``FLOAT_RECORD_OLD_WAY=1`` on the last commit that had it (tests/graphcore/old_way/): each state after each call, and
what the call returned. A replay makes the same calls on the model, each state compared with the recorded one.

Every model is read-only on its project folder (nothing is written there)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "graphcore"))
import test_h1_old_way as h1  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from astro_gpu.client.graphmodel import NOT_PORTED  # noqa: E402
from astro_gpu.client.model import ProjectModel  # noqa: E402
from astro_gpu.client.services.worker import ApiWorker  # noqa: E402
from astro_gpu.core import project2  # noqa: E402

RECORD = h1.RECORD


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture()
def recording(request):
    r = h1.Recording(f"test_graphmodel_facades__{request.node.name}")
    yield r
    r.save()


class Side:
    def __init__(self, root):
        self.w = ApiWorker()
        self.m = ProjectModel(project2.load(root), self.w, readonly=True)
        assert self.w.wait(120) and self.m.ready, self.m.error

    def close(self):
        self.m.close(save=False)
        self.w.stop()


@pytest.fixture()
def pair(qapp, tmp_path, recording):
    """pair(template key or None, sessions) -> the model opened on the template's saved project (recording: the old
    way's; else the new way's), its state checked against the recording."""
    made = []

    def make(key=None, sessions=()):
        root = tmp_path / "base"
        if key is None:
            w = h1.World(root, list(sessions))
        else:
            t = next(t for t in h1.GALLERY if t["key"] == key)
            w = (h1.old_way if RECORD else h1.new_way)(root, t, list(sessions))
        project2.save(w.p)
        w.close()
        side = Side(root)
        made.append(side)
        recording.check(side.m.p)
        return side.m
    yield make
    for s in made:
        s.close()


def view(m, name):
    return next(q for q in m.paths if q.name == name)


def both(rec, m, fn):
    """A gesture on the model; its project must be the one the old model reached: (the old call's answer, this one's)."""
    a = fn(m)
    old = rec.check(m.p, a)
    if not RECORD:
        assert m.graph_rebuilds == 0, "the definition was changed outside the graph"
    return old, h1._plain(a)


# ---------------------------------------------------------------------------------------------- blocks of a workflow
def test_a_block_inserted_where_post_and_the_palette_put_it(pair, recording):
    m = pair("colour_min", h1.FAKE[:1])
    a, b = both(recording, m, lambda m: m.insert_block(view(m, "Colour minimal"), "deconv"))
    assert a == b
    with pytest.raises(ValueError):                   # refused by both: Synthetic stars reads a Stars/background
        m.insert_block(view(m, "Colour minimal"), "synth")


def test_post_inserts_synthetic_stars_after_stars(pair, recording):
    m = pair("colour_full", h1.FAKE[:1])
    a, b = both(recording, m, lambda m: m.insert_block(view(m, "Colour full"), "synth"))     # Post's own call
    assert a == b
    a, b = both(recording, m, lambda m: m.synth_wanted())
    assert a == b


def test_a_block_taken_out_and_one_switched_off_and_on(pair, recording):
    m = pair("colour_full", h1.FAKE[:1])
    both(recording, m, lambda m: m.set_cut([view(m, "Colour full")], "fond", True))
    both(recording, m, lambda m: m.set_cut([view(m, "Colour full")], "fond", False))
    with pytest.raises(ValueError):                   # needed by Stars/background, for both
        m.set_cut([view(m, "Colour full")], "debruitage", True)
    both(recording, m, lambda m: m.remove_node(view(m, "Colour full"), "etoiles"))


def test_a_method_and_a_pause_on_a_block(pair, recording):
    m = pair("colour_min", h1.FAKE[:1])
    both(recording, m, lambda m: m.set_node_method([view(m, "Colour minimal")], "fond", "rbf"))
    both(recording, m, lambda m: m.set_breakpoint([view(m, "Colour minimal")], "empilement", True))
    both(recording, m, lambda m: m.set_breakpoint([view(m, "Colour minimal")], "empilement", False))


def test_settings_go_to_the_blocks_that_hold_them(pair, recording):
    m = pair("colour_min", h1.FAKE[:1])
    both(recording, m, lambda m: m.set_param(view(m, "Colour minimal"), "register", "min_pairs", 20))
    both(recording, m, lambda m: m.set_param(view(m, "Colour minimal"), "stack", "min_transparency", 0.6))
    both(recording, m, lambda m: m.set_param(view(m, "Colour minimal"), "stack", "white_balance", False))
    both(recording, m, lambda m: m.set_param(view(m, "Colour minimal"), "register", "min_pairs", None))
    a, b = both(recording, m, lambda m: m.set_trunk_param("calibrate", "hot_sigma", 6.0))
    assert a == b
    both(recording, m, lambda m: m.apply_change([view(m, "Colour minimal")], {
        "kind": "params", "items": [("stack", "min_transparency", 0.5), ("stack", "max_fwhm_ratio", 1.8)]}))
    both(recording, m, lambda m: m.apply_change([view(m, "Colour minimal")], {"kind": "opt", "key": "fond",
                                                                             "value": "rbf"}))
    with pytest.raises(ValueError):
        m.set_param(view(m, "Colour minimal"), "register", "no_such_setting", 1)


def test_try_as_variant_then_keep_or_discard(pair, recording):
    m = pair("colour_min", h1.FAKE[:1])
    ch = {"kind": "param", "stage": "register", "name": "min_pairs", "value": 20}
    a, b = both(recording, m, lambda m: m.add_variant(view(m, "Colour minimal"), ch, "Colour minimal · Min pairs").name)
    assert a == b
    both(recording, m, lambda m: m.keep_variant(view(m, a)))
    ch = {"kind": "param", "stage": "stack", "name": "min_transparency", "value": 0.7}
    a, b = both(recording, m, lambda m: m.add_variant(view(m, "Colour minimal"), ch, "Colour minimal · Selection").name)
    assert a == b                                     # named as today's branch names follow their last block
    both(recording, m, lambda m: m.drop_variant(view(m, a)))
    a, b = both(recording, m, lambda m: m.add_variant(view(m, "Colour minimal"), {"kind": "opt", "key": "fond",
                                                                                  "value": "rbf"},
                                                      "Colour minimal · RBF").name)
    assert a == b


def test_names_copies_and_a_block_to_post(pair, recording):
    m = pair("colour_full", h1.FAKE[:1])
    both(recording, m, lambda m: m.rename(view(m, "Colour full").pid, "Rosette"))
    assert m.rename(view(m, "Rosette").pid, "") is False
    both(recording, m, lambda m: m.send_to_post(view(m, "Rosette"), "fond", "Rosette · Gradient"))
    both(recording, m, lambda m: m.duplicate(view(m, "Rosette"), "Rosette copy"))
    both(recording, m, lambda m: m.remove_path(m.paths.index(view(m, "Rosette copy"))))


def test_a_combination_and_its_inputs(pair, recording):
    m = pair("filters_min", h1.FAKE)

    def add(m):
        ha, oiii = view(m, "Ha"), view(m, "OIII")
        return m.add_fusion("H-O", "ho", [{"pid": ha.pid, "role": "Ha", "channel": "R"},
                                          {"pid": oiii.pid, "role": "OIII", "channel": "GB"}]).name
    a, b = both(recording, m, add)
    assert a == b
    both(recording, m, lambda m: m.set_fusion_channel(view(m, "H-O"), "Ha", "L"))
    both(recording, m, lambda m: m.set_fusion_weight(view(m, "H-O"), "OIII", 1.5))
    both(recording, m, lambda m: m.clear_fusion_input(view(m, "H-O"), "OIII"))
    both(recording, m, lambda m: m.set_fusion_input(view(m, "H-O"), "OIII", view(m, "SII").pid))
    both(recording, m, lambda m: m.remove_fusion(view(m, "H-O")))


def test_align_on_another_stack(pair, recording):
    m = pair("filters_min", h1.FAKE)
    both(recording, m, lambda m: m.remove_node(view(m, "SII"), "align_on"))
    both(recording, m, lambda m: m.set_align_on(view(m, "SII"), view(m, "Ha").pid, method="plate"))
    both(recording, m, lambda m: m.set_align_reference(view(m, "SII"), view(m, "OIII").pid))


def test_images_and_new_workflows(pair, recording):
    m = pair("colour_min", h1.FAKE[:1])
    both(recording, m, lambda m: m.add_image_workflow("Image", "", None))
    both(recording, m, lambda m: m.set_image_file(view(m, "Image"), "D:/fake/stack.fits"))
    both(recording, m, lambda m: m.add_path("Second", {"debruitage": True, "etoiles": "garder", "fond": "simple"}))
    both(recording, m, lambda m: m.set_stack_mode("per_session"))


def test_sessions_and_their_files(pair, recording):
    m = pair("colour_min", h1.FAKE[:2])
    both(recording, m, lambda m: m.rename_session(1, "First night"))
    both(recording, m, lambda m: m.set_session_filter(2, "Ha"))
    both(recording, m, lambda m: m.remove_session_entry(1, "lights", "D:/fake/Night S/L0.fit"))
    both(recording, m, lambda m: m.use_from_session(1, 2, "darks"))
    both(recording, m, lambda m: m.add_filter_session("OIII"))


def test_sessions_coming_and_going(pair, recording):
    m = pair("colour_min", h1.FAKE[:2])
    a, b = both(recording, m, lambda m: m.new_session(view(m, "Colour minimal").pid))
    assert a == b
    both(recording, m, lambda m: m.remove_session(b))
    both(recording, m, lambda m: m.add_parallel_flow().name)


def test_a_filter_stack_deleted_with_the_session_only_it_reads(pair, recording):
    m = pair("filters_min", h1.FAKE)
    a, b = both(recording, m, lambda m: m.delete_workflow(view(m, "OIII")))
    assert a == b


# ---------------------------------------------------------------------------------------------- the templates
TEMPLATES = [t for t in h1.GALLERY if t.get("copy") is None]


@pytest.mark.skipif(RECORD, reason="the old way of these is test_h1_old_way's recording")
@pytest.mark.parametrize("sessions", [[], h1.FAKE], ids=["blank", "fake sessions"])
@pytest.mark.parametrize("t", TEMPLATES, ids=[t["key"] for t in TEMPLATES])
def test_every_template_made_through_the_model_is_the_old_project(qapp, tmp_path, t, sessions):
    """The gallery's choice on the graph page runs the template through the model (``apply_template``): it makes the
    project the old page's builder made, in one step."""
    rec = h1.template_recording(t, sessions)
    new = h1.World(tmp_path / "side", sessions)
    try:
        res = new.m.apply_template(t, h1.template_name(new.m, t), h1._view_opts(t))
        assert res.ok, res.reason
        rec.check(new.p)
        assert new.m.graph_rebuilds == 0
        assert len(new.m._undo) == 1                  # one step, as the old builders' batch
    finally:
        new.close()


def test_options_replaced_and_an_early_exit(pair, recording):
    m = pair("colour_min", h1.FAKE[:1])
    both(recording, m, lambda m: m.add_exit(view(m, "Colour minimal"), "empilement", "Raw stack"))
    both(recording, m, lambda m: m.set_opts(view(m, "Colour minimal"), m.with_tool(view(m, "Colour minimal").opts,
                                                                                   "debruitage")[0]))


def test_a_new_branch_from_a_block_and_beside_a_night_calibration(pair, recording):
    m = pair("colour_full", h1.FAKE[:1])
    a, b = both(recording, m, lambda m: (lambda r: (r[0].name, r[1]))(
        m.add_branch(view(m, "Colour full"), "astrometrie", "gradient")))
    assert a == b
    a, b = both(recording, m, lambda m: (lambda r: (r[0].name, r[1]))(
        m.add_branch(view(m, "Colour full"), "alignement", "stacking")))
    assert a == b
    a, b = both(recording, m, lambda m: (lambda r: (r[0].name, r[1]))(
        m.add_branch(view(m, "Colour full"), "sources", "calibration")))
    assert a == b


def test_every_facade_is_ported():
    assert NOT_PORTED == ()
