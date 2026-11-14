"""GraphEditing (client/graphmodel.py) against today's ProjectModel: the same call on the old model and on
``M(GraphEditing, ProjectModel)``, both opened on the same saved project, gives the same compiled definition (Post
included) and the same Keys tables (tests/graphcore/test_h1_old_way.py's comparison). Every gallery template built by
the gallery's own builder through M's facades gives the old project too.

Both models are read-only on one project folder (nothing is written there); what differs between them is said by
``differences`` (definition first, then keys)."""
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

from astro_gpu.client.graphmodel import NOT_PORTED, GraphEditing  # noqa: E402
from astro_gpu.client.model import ProjectModel  # noqa: E402
from astro_gpu.client.services.worker import ApiWorker  # noqa: E402
from astro_gpu.core import project2  # noqa: E402


class M(GraphEditing, ProjectModel):
    """Today's model with the graph as its truth (the swap of INTEGRATION §12 step 5, in a test)."""


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


class Side:
    def __init__(self, root, cls):
        self.w = ApiWorker()
        self.m = cls(project2.load(root), self.w, readonly=True)
        assert self.w.wait(120) and self.m.ready, self.m.error

    def close(self):
        self.m.close(save=False)
        self.w.stop()


@pytest.fixture()
def pair(qapp, tmp_path):
    """pair(template key or None, sessions) -> (old model, new model) on the same saved project."""
    made = []

    def make(key=None, sessions=()):
        root = tmp_path / "base"
        if key is None:
            w = h1.World(root, list(sessions))
        else:
            w = h1.old_way(root, next(t for t in h1.GALLERY if t["key"] == key), list(sessions))
        project2.save(w.p)
        w.close()
        old, new = Side(root, ProjectModel), Side(root, M)
        made.extend([old, new])
        assert h1.differences(old.m.p, new.m.p) == [], "opening the project changed its definition"
        return old.m, new.m
    yield make
    for s in made:
        s.close()


def view(m, name):
    return next(q for q in m.paths if q.name == name)


def both(old, new, fn):
    """The same gesture on both models; their projects must stay equal."""
    a, b = fn(old), fn(new)
    assert h1.differences(old.p, new.p) == []
    assert new.graph_rebuilds == 0, "the definition was changed outside the graph"
    return a, b


# ---------------------------------------------------------------------------------------------- blocks of a workflow
def test_a_block_inserted_where_post_and_the_palette_put_it(pair):
    old, new = pair("colour_min", h1.FAKE[:1])
    a, b = both(old, new, lambda m: m.insert_block(view(m, "Colour minimal"), "deconv"))
    assert a == b
    with pytest.raises(ValueError):                   # refused by both: Synthetic stars reads a Stars/background
        new.insert_block(view(new, "Colour minimal"), "synth")


def test_post_inserts_synthetic_stars_after_stars(pair):
    old, new = pair("colour_full", h1.FAKE[:1])
    a, b = both(old, new, lambda m: m.insert_block(view(m, "Colour full"), "synth"))     # Post's own call
    assert a == b
    assert old.synth_wanted() == new.synth_wanted()


def test_a_block_taken_out_and_one_switched_off_and_on(pair):
    old, new = pair("colour_full", h1.FAKE[:1])
    both(old, new, lambda m: m.set_cut([view(m, "Colour full")], "fond", True))
    both(old, new, lambda m: m.set_cut([view(m, "Colour full")], "fond", False))
    with pytest.raises(ValueError):                   # needed by Stars/background, for both
        new.set_cut([view(new, "Colour full")], "debruitage", True)
    both(old, new, lambda m: m.remove_node(view(m, "Colour full"), "etoiles"))


def test_a_method_and_a_pause_on_a_block(pair):
    old, new = pair("colour_min", h1.FAKE[:1])
    both(old, new, lambda m: m.set_node_method([view(m, "Colour minimal")], "fond", "rbf"))
    both(old, new, lambda m: m.set_breakpoint([view(m, "Colour minimal")], "empilement", True))
    both(old, new, lambda m: m.set_breakpoint([view(m, "Colour minimal")], "empilement", False))


def test_settings_go_to_the_blocks_that_hold_them(pair):
    old, new = pair("colour_min", h1.FAKE[:1])
    both(old, new, lambda m: m.set_param(view(m, "Colour minimal"), "register", "min_pairs", 20))
    both(old, new, lambda m: m.set_param(view(m, "Colour minimal"), "stack", "min_transparency", 0.6))
    both(old, new, lambda m: m.set_param(view(m, "Colour minimal"), "stack", "white_balance", False))
    both(old, new, lambda m: m.set_param(view(m, "Colour minimal"), "register", "min_pairs", None))
    a, b = both(old, new, lambda m: m.set_trunk_param("calibrate", "hot_sigma", 6.0))
    assert a == b
    both(old, new, lambda m: m.apply_change([view(m, "Colour minimal")], {
        "kind": "params", "items": [("stack", "min_transparency", 0.5), ("stack", "max_fwhm_ratio", 1.8)]}))
    both(old, new, lambda m: m.apply_change([view(m, "Colour minimal")], {"kind": "opt", "key": "fond",
                                                                         "value": "rbf"}))
    with pytest.raises(ValueError):
        new.set_param(view(new, "Colour minimal"), "register", "no_such_setting", 1)


def test_try_as_variant_then_keep_or_discard(pair):
    old, new = pair("colour_min", h1.FAKE[:1])
    ch = {"kind": "param", "stage": "register", "name": "min_pairs", "value": 20}
    a, b = both(old, new, lambda m: m.add_variant(view(m, "Colour minimal"), ch, "Colour minimal · Min pairs").name)
    assert a == b
    both(old, new, lambda m: m.keep_variant(view(m, a)))
    ch = {"kind": "param", "stage": "stack", "name": "min_transparency", "value": 0.7}
    a, b = both(old, new, lambda m: m.add_variant(view(m, "Colour minimal"), ch, "Colour minimal · Selection").name)
    assert a == b                                     # named as today's branch names follow their last block
    both(old, new, lambda m: m.drop_variant(view(m, a)))
    a, b = both(old, new, lambda m: m.add_variant(view(m, "Colour minimal"), {"kind": "opt", "key": "fond",
                                                                              "value": "rbf"}, "Colour minimal · RBF").name)
    assert a == b


def test_names_copies_and_a_block_to_post(pair):
    old, new = pair("colour_full", h1.FAKE[:1])
    both(old, new, lambda m: m.rename(view(m, "Colour full").pid, "Rosette"))
    assert new.rename(view(new, "Rosette").pid, "") is False
    both(old, new, lambda m: m.send_to_post(view(m, "Rosette"), "fond", "Rosette · Gradient"))
    both(old, new, lambda m: m.duplicate(view(m, "Rosette"), "Rosette copy"))
    both(old, new, lambda m: m.remove_path(m.paths.index(view(m, "Rosette copy"))))


def test_a_combination_and_its_inputs(pair):
    old, new = pair("filters_min", h1.FAKE)

    def add(m):
        ha, oiii = view(m, "Ha"), view(m, "OIII")
        return m.add_fusion("H-O", "ho", [{"pid": ha.pid, "role": "Ha", "channel": "R"},
                                          {"pid": oiii.pid, "role": "OIII", "channel": "GB"}]).name
    a, b = both(old, new, add)
    assert a == b
    both(old, new, lambda m: m.set_fusion_channel(view(m, "H-O"), "Ha", "L"))
    both(old, new, lambda m: m.set_fusion_weight(view(m, "H-O"), "OIII", 1.5))
    both(old, new, lambda m: m.clear_fusion_input(view(m, "H-O"), "OIII"))
    both(old, new, lambda m: m.set_fusion_input(view(m, "H-O"), "OIII", view(m, "SII").pid))
    both(old, new, lambda m: m.remove_fusion(view(m, "H-O")))


def test_align_on_another_stack(pair):
    old, new = pair("filters_min", h1.FAKE)
    both(old, new, lambda m: m.remove_node(view(m, "SII"), "align_on"))
    both(old, new, lambda m: m.set_align_on(view(m, "SII"), view(m, "Ha").pid, method="plate"))
    both(old, new, lambda m: m.set_align_reference(view(m, "SII"), view(m, "OIII").pid))


def test_images_and_new_workflows(pair):
    old, new = pair("colour_min", h1.FAKE[:1])
    both(old, new, lambda m: m.add_image_workflow("Image", "", None))
    both(old, new, lambda m: m.set_image_file(view(m, "Image"), "D:/fake/stack.fits"))
    both(old, new, lambda m: m.add_path("Second", {"debruitage": True, "etoiles": "garder", "fond": "simple"}))
    both(old, new, lambda m: m.set_stack_mode("per_session"))


def test_sessions_and_their_files(pair):
    old, new = pair("colour_min", h1.FAKE[:2])
    both(old, new, lambda m: m.rename_session(1, "First night"))
    both(old, new, lambda m: m.set_session_filter(2, "Ha"))
    both(old, new, lambda m: m.remove_session_entry(1, "lights", "D:/fake/Night S/L0.fit"))
    both(old, new, lambda m: m.use_from_session(1, 2, "darks"))
    both(old, new, lambda m: m.add_filter_session("OIII"))


def test_sessions_coming_and_going(pair):
    old, new = pair("colour_min", h1.FAKE[:2])
    a, b = both(old, new, lambda m: m.new_session(view(m, "Colour minimal").pid))
    assert a == b
    both(old, new, lambda m: m.remove_session(a))
    both(old, new, lambda m: m.add_parallel_flow().name)


def test_a_filter_stack_deleted_with_the_session_only_it_reads(pair):
    old, new = pair("filters_min", h1.FAKE)
    a, b = both(old, new, lambda m: m.delete_workflow(view(m, "OIII")))
    assert a == b


# ---------------------------------------------------------------------------------------------- the templates
TEMPLATES = [t for t in h1.GALLERY if t.get("copy") is None]


@pytest.mark.parametrize("sessions", [[], h1.FAKE], ids=["blank", "fake sessions"])
@pytest.mark.parametrize("t", TEMPLATES, ids=[t["key"] for t in TEMPLATES])
def test_every_template_built_through_the_facades_is_the_old_project(qapp, tmp_path, t, sessions):
    """The gallery's builder (make_from_template) calls the model's facades: through M it makes today's project."""
    old = h1.old_way(tmp_path / "side_old", t, sessions)
    new = h1.World(tmp_path / "side_new", sessions)
    new.m.close(save=False)
    new.m = M(new.p, new.w)
    assert new.w.wait(120)
    try:
        h1.FakePage(new.m).make_from_template(t, h1.template_name(new.m, t), h1._view_opts(t))
        assert h1.differences(old.p, new.p) == []
        assert new.m.graph_rebuilds == 0
        assert len(new.m._undo) == 1                  # one step, as the old builders' batch
    finally:
        old.close()
        new.close()


def test_options_replaced_and_an_early_exit(pair):
    old, new = pair("colour_min", h1.FAKE[:1])
    both(old, new, lambda m: m.add_exit(view(m, "Colour minimal"), "empilement", "Raw stack"))
    both(old, new, lambda m: m.set_opts(view(m, "Colour minimal"), m.with_tool(view(m, "Colour minimal").opts,
                                                                               "debruitage")[0]))


def test_a_new_branch_from_a_block_and_beside_a_night_calibration(pair):
    old, new = pair("colour_full", h1.FAKE[:1])
    a, b = both(old, new, lambda m: (lambda r: (r[0].name, r[1]))(
        m.add_branch(view(m, "Colour full"), "astrometrie", "gradient")))
    assert a == b
    a, b = both(old, new, lambda m: (lambda r: (r[0].name, r[1]))(
        m.add_branch(view(m, "Colour full"), "alignement", "stacking")))
    assert a == b
    a, b = both(old, new, lambda m: (lambda r: (r[0].name, r[1]))(
        m.add_branch(view(m, "Colour full"), "sources", "calibration")))
    assert a == b


def test_every_facade_is_ported():
    assert NOT_PORTED == ()
