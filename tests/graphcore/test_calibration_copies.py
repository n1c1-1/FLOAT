"""A night's Calibration copied into another night's (a variant of the whole stack, a night that variant follows, the
other nights of a branch from a night's Calibration, a duplicated row) holds everything the original holds: its
settings, its switch, its review pause, its engine settings and its own files (a variant tried from the Registration
calibrated with the default settings)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from fixtures import Doc  # noqa: E402
from test_points import ok, of_kind  # noqa: E402

FIELDS = ("settings", "off", "pause", "params", "own")


def _held(b):
    return {k: getattr(b, k) for k in FIELDS}


def _dress(doc, c, mark):
    """Calibration c given something of each kind it can hold."""
    b = doc.project.blocks[c]
    b.settings = {"note": mark}
    b.off, b.pause = True, True
    b.params = {"calibrate": {"sigma": 2.5}}
    b.own = {"fork": mark, "flats": [f"C:/own/{mark}.fit"]}
    return _held(b)


def _two_nights():
    doc = Doc.from_template("colour_minimal")
    r = doc.project.rows[0]
    ok(doc, "add_files", session=r.nights[0][0], kind="lights", paths=["C:/n/a.fit"])
    ok(doc, "add_night", row=r.id, files={"lights": ["C:/m/c.fit"]})
    r = doc.project.rows[0]
    assert all(c is not None for _s, c in r.nights)
    return doc, r


def test_a_whole_stack_variant_copies_each_nights_calibration():
    doc, r = _two_nights()
    want = [_dress(doc, c, 7 + i) for i, (_s, c) in enumerate(r.nights)]
    ok(doc, "try_variant", block=of_kind(doc, r, "registration"), change={"param": ["register", "min_pairs", 20]})
    v = doc.project.rows[-1]
    assert v.trial is not None and not v.fork
    assert [_held(doc.project.blocks[c]) for _s, c in v.nights] == want


def test_a_night_a_whole_variant_follows_copies_the_originals_calibration():
    from astro_gpu.graphcore.operations import _whole_variants_follow
    doc, r = _two_nights()
    ok(doc, "try_variant", block=of_kind(doc, r, "registration"), change={"param": ["register", "min_pairs", 20]})
    q = doc.project
    r, v = q.rows[0], q.rows[-1]
    want = _dress(doc, r.nights[1][1], 9)
    s, c = v.nights.pop()                       # the variant no longer draws that night: it is copied again
    q.blocks.pop(s), q.blocks.pop(c)
    _whole_variants_follow(q)
    assert len(v.nights) == 2 and _held(q.blocks[v.nights[1][1]]) == want


def test_a_branch_from_a_nights_calibration_copies_the_other_nights():
    doc, r = _two_nights()
    ok(doc, "set_calibrate_together", on=False)               # each night its own Calibration: a branch from one
    r = doc.project.rows[0]
    other = _dress(doc, r.nights[1][1], 5)
    ok(doc, "add_branch", block=r.nights[0][1], kind="calibration")
    v = doc.project.rows[-1]
    assert _held(doc.project.blocks[v.nights[1][1]]) == other
    mine = doc.project.blocks[v.nights[0][1]]                 # the branch's own: a new block, from its defaults
    assert (mine.settings, mine.off, mine.pause, mine.params) == ({}, False, False, {})
    assert mine.own == {"fork": v.wid}


def test_a_duplicated_row_copies_its_nights_calibration():
    doc, r = _two_nights()
    want = [_dress(doc, c, 3 + i) for i, (_s, c) in enumerate(r.nights)]
    ok(doc, "duplicate_row", row=r.id)
    d = next(x for x in doc.project.rows if x.id != r.id and x.nights and not x.trial)
    assert [_held(doc.project.blocks[c]) for _s, c in d.nights] == want
