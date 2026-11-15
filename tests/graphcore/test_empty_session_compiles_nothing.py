"""A session added with no lights yet changes nothing computed (a beta tester's empty session made the whole project
Outdated, 17 min to run, its target forgotten): the workflows read the sessions that have lights; the first lights of
the new one bring it in."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from fixtures import Doc  # noqa: E402
from test_engine_rules import compiled  # noqa: E402

from astro_gpu.graphcore.compile import _NoDisk, _key_rules  # noqa: E402
from astro_gpu.graphcore.proof import keys_table  # noqa: E402


def read(p):
    """What the engine computes from ``p``: each workflow's definition with the sessions it reads, and its keys."""
    flows = p.to_dict()["workflows"]
    for d in flows:
        d["sessions"] = [s.id for s in p.sessions_of(d["id"])]
    return flows, keys_table(p, _key_rules(), _NoDisk())


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


def test_an_empty_session_leaves_every_workflow_and_key_as_they_were_and_counts_once_it_has_lights():
    """A session added with none leaves what the workflows read and their
    keys as they were; the session is drawn all the same, and its first lights change the keys of its workflow."""
    doc = Doc.from_template("colour_complete")
    r = doc.project.rows[0]
    doc.apply("add_files", session=r.nights[0][0], kind="lights", paths=["C:/n/a.fit", "C:/n/b.fit"])
    p0, _ = compiled(doc)
    doc.apply("add_night", row=r.id)                                   # the empty session, in the same stack
    p1, _ = compiled(doc)
    assert read(p1) == read(p0)
    r = doc.project.rows[0]
    new = doc.project.blocks[r.nights[1][0]]
    assert len(r.nights) == 2 and not new.files.get("lights")         # drawn in the stack, with no lights
    assert new.sid is None or new.sid in [s.id for s in p1.sessions]
    doc.apply("add_files", session=r.nights[1][0], kind="lights", paths=["C:/m/c.fit"])
    p2, _ = compiled(doc)
    sid = doc.project.blocks[r.nights[1][0]].sid
    assert sid in [s.id for s in p2.sessions_of(p2.workflows[0].id)]  # its lights: it counts
    assert read(p2)[1] != read(p0)[1]
