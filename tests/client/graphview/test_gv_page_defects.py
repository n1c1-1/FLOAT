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


# ---------------------------------------------------------------------------------------------- texts that overlap
def _titles_clear(cv):
    """No band title over a stack's name, a block or the Project block, as painted."""
    from PySide6.QtCore import QRectF

    from astro_gpu.client.graphview.items import CaptionItem
    others = [it.text_rect() for it in cv.scene().items() if isinstance(it, CaptionItem)]
    others += [QRectF(it.pos().x(), it.pos().y(), it.w, it.h) for it in cv.blocks.values()]
    others.append(cv.project_item.rect)
    return [(b.text, r.getRect()) for b in cv.bands for r in others if b.title_rect().intersects(r)]


@pytest.mark.parametrize("nights", [1, 2, 3])
def test_a_band_title_never_covers_the_stack_name_and_its_badge(world, nights):
    W = world(workflows=(), nights=nights)
    W.template("colour_full")
    cv = W.page.canvas
    for zoom in (0.5, 1.0, 2.0):
        cv.zoom_to(zoom)
        W.settle(2)
        assert not _titles_clear(cv), zoom
    assert any(x.d["kind"] != "card" for x in cv.blocks.values())
    W.page.set_display("folded")
    W.settle()
    assert not _titles_clear(cv)
