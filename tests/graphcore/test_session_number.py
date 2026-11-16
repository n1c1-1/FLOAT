"""A new session's default name is "Session n", n the number the session gets in the project (its id), or the next
number no name has. A tutorial's second session, added to a workflow whose first session had been given a name, was
"Session 1": the number was the first one free, counted from 1."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from astro_gpu.graphcore import api  # noqa: E402
from astro_gpu.graphcore.model import check  # noqa: E402


def do(g, op, **kw):
    g, r = api.apply(g, op, **kw)
    assert r.ok, (op, kw, r.reason_text())
    check(g)
    return g


def sessions(g):
    return [g.blocks[s].name for r in g.rows for s, _c in r.nights]


def test_the_second_session_of_a_stack_is_session_2_whatever_the_first_is_called():
    g = do(api.blank_project(), "add_row", kind="session")
    first = g.rows[0].nights[0][0]
    g = do(g, "rename_row", row=g.rows[0].id, name="Colour minimal")       # a template's workflow
    g = do(g, "rename_night", night=first, name="29 Dec 2024")
    g = do(g, "add_block", kind="session", row=g.rows[0].id)
    assert sessions(g) == ["29 Dec 2024", "Session 2"]
    g = do(g, "add_files", session=g.rows[0].nights[1][0], kind="lights", paths=["/raw/n2/L_0.fits"])
    assert sessions(g) == ["29 Dec 2024", "Session 2"]            # its lights change no name
    g = do(g, "add_row", kind="session")                          # a stack of its own: the third session
    assert sessions(g) == ["29 Dec 2024", "Session 2", "Session 3"]


def test_a_number_a_name_already_has_is_skipped():
    g = do(api.blank_project(), "add_row", kind="session")
    g = do(g, "rename_row", row=g.rows[0].id, name="Colour minimal")
    g = do(g, "rename_night", night=g.rows[0].nights[0][0], name="Session 2")       # typed by the user
    g = do(g, "add_night", row=g.rows[0].id)
    assert sessions(g) == ["Session 2", "Session 3"]


def test_a_session_of_the_project_no_stack_draws_counts():
    g = do(api.blank_project(), "new_night", name="Ha", filter="Ha")
    g = do(g, "new_night")
    assert [n["name"] for n in g.nights] == ["Ha", "Session 2"]
    g = do(g, "add_row", kind="session")
    assert sessions(g) == ["Session 3"]
