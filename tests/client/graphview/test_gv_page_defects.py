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


# ---------------------------------------------------------------------------------------------- Align on
def test_an_image_workflow_is_aligned_from_its_menu_and_its_method_is_chosen(world, tmp_path):
    import numpy as np
    from astropy.io import fits
    W = world(workflows=("Classic stack",))
    files = []
    for n in ("a1", "a2"):
        f = tmp_path / f"{n}.fits"
        fits.PrimaryHDU(np.full((3, 32, 32), 100.0, np.float32)).writeto(f)
        files.append(str(f))
    W.page.place_images(files)
    W.settle()
    a1 = next(b for b, x in W.graph.blocks.items() if x.kind == "image" and W.b.row_of(b).name == "a1")
    r1 = W.b.row_of(a1)
    got = W.g.menu_pick(a1, "a2")                                       # Align on > a2
    W.settle()
    assert got["ok"] and W.b.row(r1.id).align and W.b.row(r1.id).align["ref"] == W.b.row_of(
        next(b for b, x in W.graph.blocks.items() if x.kind == "image" and W.b.row_of(b).name == "a2")).id
    items = dict((t, (on, tip)) for t, on, tip in W.g.menu_pick(a1, "", open_only=True)["items"])
    assert items["✓ Stars"][0] and items["Plate solutions"][0]          # an image may carry its plate solution
    got = W.g.menu_pick(a1, "Plate solutions")
    W.settle()
    assert got["ok"] and W.b.row(r1.id).align["method"] == "plate"
    st = W.one("stacking")                                              # a stack: plate greyed, its reason said
    got = W.g.menu_pick(st, "a1")
    W.settle()
    assert got["ok"] and W.b.row_of(st).align
    items = dict((t, (on, tip)) for t, on, tip in W.g.menu_pick(st, "", open_only=True)["items"])
    assert items["Plate solutions"] == (False, "the stack has no plate solution yet (Astrometry comes after)")


# ---------------------------------------------------------------------------------------------- words
def test_a_review_pause_says_its_block_and_workflows_in_its_toast_and_tooltip(world):
    W = world(workflows=("Full", "Classic stack"))
    st = W.block("empilement", "Full")
    W.g.menu_pick(st, "Review pause")
    W.settle()
    assert W.said[-1] == "Review pause after Stacking · 2 workflows · Ctrl+Z"      # shared by the two
    tip = W.page.canvas.blocks[st].toolTip()
    assert "Review point: the run stops here for your review." in tip
    gr = W.block("fond", "Full")
    W.g.menu_pick(gr, "Review pause")
    W.settle()
    assert W.said[-1] == "Review pause after Gradient · 2 workflows · Ctrl+Z"
    W.b.act("switch_off", block=gr)
    W.settle()
    assert W.said[-1] == "Gradient switched off · Full · Ctrl+Z"
    tip = W.page.canvas.blocks[gr].toolTip()
    assert "waits for the block to be switched on again" in tip and "the run stops here" not in tip
    assert all(len(line) <= 100 for line in tip.split("\n")), tip     # what it does wrapped (it was ~300)
    assert "Up to date" not in tip                                     # a block switched off is not computed


def test_a_session_whose_darks_folder_is_gone_says_so_under_its_name(world, tmp_path):
    W = world()
    W.m.p.session(1).darks = [(tmp_path / "darks_renamed").as_posix()]
    W.m.reload_sources()
    W.settle()
    W.page.refresh()
    W.settle()
    s = W.one("session")
    assert "darks not found" in W.page.canvas.blocks[s].d["summary"]


def test_an_image_block_says_its_file_stretched_or_to_choose(world, tmp_path):
    import numpy as np
    from astropy.io import fits
    from PIL import Image
    W = world(workflows=())
    f = tmp_path / "m42.fits"
    fits.PrimaryHDU(np.full((3, 32, 32), 100.0, np.float32)).writeto(f)
    png = tmp_path / "done.png"
    Image.fromarray(np.full((32, 32, 3), 120, np.uint8)).save(png)
    W.page.place_images([str(f), str(png)])
    W.settle()
    lines = sorted(it.d["summary"] for it in W.page.canvas.blocks.values() if it.d["kind"] == "image")
    assert lines == ["done.png  · stretched", "m42.fits"], lines
    W.template("image")
    lines = [it.d["summary"] for it in W.page.canvas.blocks.values() if it.d["kind"] == "image"]
    assert "file to choose" in lines, lines


def test_a_chain_wire_says_it_cannot_be_removed_and_an_output_port_shows_the_hand(world):
    from PySide6.QtCore import QPointF, Qt
    W = world()
    cv = W.page.canvas
    chain = [it for it in cv.wires.values() if it.style_name == "chain"]
    assert chain and all("cannot be removed" in it.tipfn() for it in chain)
    from astro_gpu.client.graphview.items import WireItem
    spots = [cv.mapFromScene(it.path().pointAtPercent(0.5)) for it in chain]
    pt = next(p for p in spots if isinstance(cv.item_at(p), WireItem))     # a spot of the wire, not a port
    W.g.press(pt)
    W.g.release(pt)
    W.settle(2)
    assert cv.toast.plain == "Chain wire: not removable" and cv.selected_wire is None
    st = cv.blocks[W.block("empilement")]
    port, y = st.d["outputs"][0][0], st.d["outputs"][0][1]
    st._grab_cursor(QPointF(st.w, y))
    assert st.cursor().shape() == Qt.OpenHandCursor
    st._grab_cursor(QPointF(st.w / 2, st.h / 2))
    assert not st.hasCursor()


# ---------------------------------------------------------------------------------------------- the palette's places
def test_a_palette_card_lights_its_workflows_places_and_another_stacks_under_the_pointer(world):
    from PySide6.QtCore import QMimeData, QPoint, Qt
    from PySide6.QtGui import QDragEnterEvent, QDragLeaveEvent, QDragMoveEvent
    from PySide6.QtWidgets import QApplication

    from astro_gpu.client.graphview.canvas import MIME_KIND
    W = world(workflows=(), filters=("Ha", "OIII", "SII", "L"))
    W.template("filters_min")
    cv = W.page.canvas
    rows = [r for r in W.graph.rows if r.nights]
    assert len(rows) >= 4
    first = rows[1]
    W.choose(next(b for b in first.blocks if W.graph.blocks[b].kind == "stacking"))
    mime = QMimeData()
    mime.setData(MIME_KIND, b"gradient")
    vp = cv.viewport()

    def lit_rows(at):
        QApplication.sendEvent(vp, QDragMoveEvent(at, Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier))
        return {g.row for g in cv.ghosts if g.isVisible() and g.place.lit and g.row is not None}
    far = QPoint(4, 4)
    QApplication.sendEvent(vp, QDragEnterEvent(far, Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier))
    assert lit_rows(far) == {first.id}                                   # the workflow the palette places in
    other = rows[3]
    y = cv.blocks[next(b for b in other.blocks if b in cv.blocks)].sceneBoundingRect().center().y()
    at = cv.mapFromScene(cv.mapToScene(far).x(), y)
    cv.centerOn(cv.mapToScene(far).x(), y)
    W.settle(1)
    at = cv.mapFromScene(cv.mapToScene(QPoint(4, 4)).x(), y)
    assert lit_rows(at) == {first.id, other.id}                          # and the stack under the pointer
    QApplication.sendEvent(vp, QDragLeaveEvent())
    W.settle(1)
    assert not cv.ghosts
