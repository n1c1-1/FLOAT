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


# ---------------------------------------------------------------------------------------------- the panel's kind rows
def _drop_on(widget, paths, leave=False):
    """A drop of files from the explorer on a widget, as Qt delivers it (enter, move, then drop or leave)."""
    from PySide6.QtCore import QMimeData, QPointF, Qt, QUrl
    from PySide6.QtGui import QDragEnterEvent, QDragLeaveEvent, QDragMoveEvent, QDropEvent
    from PySide6.QtWidgets import QApplication
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(p)) for p in paths])
    pt = widget.rect().center()
    enter = QDragEnterEvent(pt, Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier)
    QApplication.sendEvent(widget, enter)
    QApplication.sendEvent(widget, QDragMoveEvent(pt, Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier))
    hot = "dashed" in widget.styleSheet()
    if leave:
        QApplication.sendEvent(widget, QDragLeaveEvent())
    else:
        QApplication.sendEvent(widget, QDropEvent(QPointF(pt), Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier))
    return enter.isAccepted(), hot


def test_files_dropped_on_a_kind_row_of_the_session_panel_are_added_as_on_the_graph(world, tmp_path):
    from PySide6.QtWidgets import QLabel

    from tests.client.graphview.legacy import frames
    W = world()
    s = W.one("session")
    W.choose(s)
    W.settle()
    darks = frames(tmp_path / "darks", ["d1.ARW", "d2.ARW", "d3.ARW", "d1.JPG"])

    def row(kind_title):
        return next(x for x in W.page.panel.findChildren(QLabel)
                    if x.objectName() == "sectiontitle" and x.text().startswith(kind_title) and x.isVisible())
    ok, hot = _drop_on(row("Darks"), [darks], leave=True)
    assert ok and hot and "dashed" not in row("Darks").styleSheet()          # lit while over it, not after
    n0 = W.steps()
    ok, _hot = _drop_on(row("Darks"), [darks])
    W.settle()
    assert ok and W.steps() == n0 + 1
    got = W.b.doc.files_of(s, "darks")[1]
    assert len(got) == 3 and all(p.endswith(".ARW") for p in got)            # single files, the type chosen
    said = W.page.canvas.toast.plain
    assert "3 ARW added" in said and "1 JPG left" in said
    assert "take_all" in [k for _t, k in W.page.canvas.toast.links()]
    _drop_on(row("Darks"), [darks])
    W.settle()
    assert len(W.b.doc.files_of(s, "darks")[1]) == 3                        # no double
    assert "3 already there" in W.page.canvas.toast.plain


# ---------------------------------------------------------------------------------------------- New workflow
def _open_new_workflow(W):
    from astro_gpu.client.graphview.items import LinkItem
    from astro_gpu.client.gallery import TemplateGallery
    cv = W.page.canvas
    link = cv.new_link
    assert isinstance(link, LinkItem) and link.text == "New workflow" and link.toolTip()
    cv.centerOn(link)
    W.settle(2)
    W.g.press(cv.mapFromScene(link.sceneBoundingRect().center()))
    W.g.release(cv.mapFromScene(link.sceneBoundingRect().center()))
    W.settle(2)
    d = W.page._gallery
    assert isinstance(d, TemplateGallery) and d.isVisible()
    return d


def test_the_new_workflow_link_opens_the_window_of_templates_and_copies(world):
    W = world()
    d = _open_new_workflow(W)
    assert "copy" in d.tab_keys and "frames" in d.tab_keys
    n = len(W.m.paths)
    d.pick(next(c.t for c in d.cards if c.t.get("key") == "colour_min"))
    d.name.setText("Second")
    d._create()
    W.settle()
    assert len(W.m.paths) == n + 1 and "Second" in [q.name for q in W.m.paths]
    d = _open_new_workflow(W)
    d.tabs.setCurrentIndex(d.tab_keys.index("copy"))
    W.settle(2)
    d.pick(next(c.t for c in d.cards if c.t.get("copy") is not None))
    d.name.setText("Copied")
    d._create()
    W.settle()
    assert len(W.m.paths) == n + 2 and "Copied" in [q.name for q in W.m.paths]
    W.b.undo()
    W.settle()
    assert len(W.m.paths) == n + 1


def test_a_filters_template_is_offered_again_on_sessions_that_have_filters(world):
    W = world(workflows=(), filters=("Ha", "OIII"))
    W.template("filters_min")
    n = len(W.m.paths)
    assert n >= 2
    d = _open_new_workflow(W)
    t = next((c.t for c in d.cards if c.t.get("key") == "filters_min"), None)
    assert t is not None
    d.pick(t)
    d._create()
    W.settle()
    assert len(W.m.paths) > n
