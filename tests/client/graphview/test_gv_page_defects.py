"""What the ported tests found lost on the new graph page (its own side), checked fixed: renaming, the window's
selection, the Project block, the New workflow window, Align on for images, the page's words, the palette's lit places,
small screens, JPEG lights dropped on a Session."""
from __future__ import annotations

import pytest

from tests.client.graphview.legacy import World


@pytest.fixture()
def world(qapp, tmp_path):
    made = []

    def make(**kw):
        w = World(tmp_path / f"w{len(made)}", **kw)
        made.append(w)
        return w
    yield make
    for w in made:
        w.close()


# ---------------------------------------------------------------------------------------------- renaming
def test_f2_renames_the_workflow_of_the_chosen_block(world):
    W = world()
    st = W.block("empilement")
    W.g.click(st)
    W.settle(2)
    W.g.key("F2", target=W.page.canvas)
    ed = W.page.canvas.name_edit
    assert ed is not None
    ed.setText("Wide field")
    W.g.key("Return", target=ed)
    W.settle()
    assert [q.name for q in W.m.paths] == ["Wide field"]
    assert W.graph.rows[0].name == "Wide field"


def test_the_bridge_takes_an_operations_own_name(world):
    W = world()
    r = W.b.act("rename_row", row=W.graph.rows[0].id, name="Deep")
    W.settle()
    assert r.ok and W.graph.rows[0].name == "Deep"
