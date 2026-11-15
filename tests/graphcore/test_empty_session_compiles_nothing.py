"""A session added with no lights yet changes nothing computed (a beta tester's empty session made the whole project
Outdated, 17 min to run, its target forgotten): the workflows read the sessions that have lights; the first lights of
the new one bring it in."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from fixtures import Doc  # noqa: E402
from test_engine_rules import compiled  # noqa: E402


def test_an_empty_session_is_left_out_of_the_workflows_until_it_has_lights():
    doc = Doc.from_template("colour_minimal")
    r = doc.project.rows[0]
    doc.apply("add_files", session=r.nights[0][0], kind="lights", paths=["C:/n/a.fit", "C:/n/b.fit"])
    p0, _ = compiled(doc)
    before = p0.workflows[0].sessions
    doc.apply("add_night", row=r.id)                                  # no lights yet
    p1, _ = compiled(doc)
    first = doc.project.blocks[r.nights[0][0]].sid
    assert p1.workflows[0].sessions in (before, [first]) and len(p1.sessions) == 2
    doc.apply("add_row", kind="session")                               # a stack of its own, empty
    p2, _ = compiled(doc)
    assert p2.workflows[0].sessions in (before, [first])
    r = doc.project.rows[0]
    doc.apply("add_files", session=r.nights[1][0], kind="lights", paths=["C:/m/c.fit"])
    p3, _ = compiled(doc)
    second = doc.project.blocks[r.nights[1][0]].sid
    assert second in (p3.workflows[0].sessions or [s.id for s in p3.sessions])
