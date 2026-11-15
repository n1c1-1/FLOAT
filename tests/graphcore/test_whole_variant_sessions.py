"""A variant of a whole stack draws its original's sessions (a beta tester's variant kept 4 lights after its original's
stack grew to 6, and the Table said its frames were listed twice, with a Remove): a session added to the original or
taken out of it is the variant's too; the same session drawn by both stacks is never a double."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from fixtures import Doc  # noqa: E402
from test_points import ok, of_kind  # noqa: E402
from test_engine_rules import compiled, engine_ok  # noqa: E402
from astro_gpu.graphcore import api  # noqa: E402


def sids(doc, r):
    return [doc.project.blocks[s].sid for s, _c in r.nights]


def test_a_whole_stack_variant_follows_its_originals_sessions():
    doc = Doc.from_template("colour_minimal")
    r = doc.project.rows[0]
    s0 = r.nights[0][0]
    ok(doc, "add_files", session=s0, kind="lights", paths=["C:/n/a.fit", "C:/n/b.fit"])
    reg = of_kind(doc, r, "registration")
    ok(doc, "try_variant", block=reg, change={"param": ["register", "min_pairs", 20]})
    v, r = doc.project.rows[-1], doc.project.rows[0]
    assert not v.fork and sids(doc, v) == sids(doc, r)
    assert not [w for w in api.warnings(doc.project) if "also in" in str(w[2])]     # one session, never a double
    ok(doc, "add_night", row=r.id, files={"lights": ["C:/m/c.fit"]})
    v, r = doc.project.rows[-1], doc.project.rows[0]
    assert len(r.nights) == 2 and sids(doc, v) == sids(doc, r)
    assert doc.project.blocks[v.nights[1][0]].files == doc.project.blocks[r.nights[1][0]].files
    p, _ = compiled(doc)
    assert p.workflows[1].sessions == p.workflows[0].sessions
    engine_ok(doc)
    ok(doc, "remove_night", night=r.nights[1][0])
    v, r = doc.project.rows[-1], doc.project.rows[0]
    assert len(v.nights) == 1 and sids(doc, v) == sids(doc, r)
    engine_ok(doc)
    doc.undo()
    v, r = doc.project.rows[-1], doc.project.rows[0]
    assert len(v.nights) == 2 and sids(doc, v) == sids(doc, r)
