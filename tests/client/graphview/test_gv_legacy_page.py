"""What the tests of today's graph page checked on its free graphs (issue #47 and the GUI explorations), checked again on
the new page over a project model: the palette's places, blocks placed twice, pauses, wires to Post, templates on the
blank page, images and their names, the menu of another workflow's block, Post's layers, files added by mistake,
combinations wired from images. A scratch project whose frames are empty files: nothing is computed (legacy.World).
PORT_TABLE.md maps each old test to its port here, to a test of the new page that already checks it, or to a defect."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from tests.client.graphview.legacy import World, pump


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


def kinds_of(W, row):
    return [W.graph.blocks[b].kind for b in row.blocks]


def row_named(W, name):
    return next(r for r in W.graph.rows if r.name == name)


def a_fits(path: Path, shape=(3, 32, 32)) -> str:
    from astropy.io import fits
    fits.PrimaryHDU(np.full(shape, 100.0, np.float32)).writeto(path)
    return str(path)


# ---------------------------------------------------------------------------------------------- the palette
def test_the_palette_places_a_block_at_its_usual_place_and_twice(world):
    W = world()
    full = W.m.paths[0]
    W.choose(W.block("fond", full))
    W.g.double_click_palette("denoise")                         # a second Denoise, after the first
    W.settle()
    assert W.m.display_keys(W.m.view(full.pid))[-5:] == ["debruitage", "debruitage~2", "couleur", "etoiles", "post"]
    assert W.m.sel_step == "debruitage~2" and "denoise~2" in W.m.all_stages(W.m.view(full.pid))
    titles = [it.d["title"] for it in W.page.canvas.blocks.values() if it.d["kind"] == "denoise"]
    assert titles == ["Denoise", "Denoise 2"]
    assert W.page.panel.node is W.bm.node(W.block("debruitage~2", full))           # its panel: its own block
    assert not W.card("stacking").reason == "" and not W.card("calibration").enabled          # once per workflow
    W.b.undo()
    W.settle()
    assert "debruitage~2" not in W.m.display_keys(W.m.view(full.pid))


def test_a_block_placed_again_starts_from_its_defaults_and_one_switched_off_is_not_placed_twice(world):
    """The user's answers (GUI exploration, 7 November): a block placed starts from its defaults (9); a block switched
    off is not placed twice (4.4); Delete workflow at once, said with Ctrl+Z (10.1)."""
    W = world()
    full = W.m.paths[0]
    W.m.set_param(W.m.view(full.pid), "gradient", "degree", 4)
    W.settle()
    W.g.click_cross(W.block("fond", full))
    W.settle()
    assert "fond" not in W.m.display_keys(W.m.view(full.pid))
    W.choose(W.block("astrometrie", full))
    W.g.double_click_palette("gradient")
    W.settle()
    assert "fond" in W.m.display_keys(W.m.view(full.pid))
    assert W.m.p.workflow(full.pid).params.get("gradient", {}).get("degree") in (None, 2)     # the default, not 4
    reg = W.block("alignement", full)
    W.choose(reg)
    W.g.key("D", target=W.page.canvas)
    W.settle()
    assert W.graph.blocks[reg].off
    W.choose(reg)
    assert not W.card("registration").enabled                                     # not placed a second time
    r = W.b.act("delete_row", row=W.graph.rows[0].id)
    W.settle()
    assert r.ok and not W.m.paths and "Ctrl+Z" in W.said[-1]
    W.b.undo()
    W.settle()
    assert [q.name for q in W.m.paths] == ["Full"]


# ---------------------------------------------------------------------------------------------- pauses
def test_a_block_placed_twice_has_its_own_settings_and_its_own_pause(world):
    """GUI exploration 1.2: the second Gradient is a block of its own: its setting changes (the first keeps its own),
    it gets its pause for review, and Run to here aims at it."""
    W = world()
    full = W.m.paths[0]
    W.choose(W.block("fond", full))
    W.g.double_click_palette("gradient")
    W.settle()
    p = W.m.view(full.pid)
    assert "fond~2" in W.m.display_keys(p)
    W.m.set_param(p, "gradient~2", "degree", 4)
    assert W.m.p.workflow(full.pid).params.get("gradient~2", {}).get("degree") == 4
    assert W.m.p.workflow(full.pid).params.get("gradient", {}).get("degree") in (None, 2)     # its own default
    g2 = W.block("fond~2", full)
    W.g.menu_pick(g2, "Review pause")
    W.settle()
    p = W.m.view(full.pid)
    assert W.m.has_breakpoint(p, "fond~2") and not W.m.has_breakpoint(p, "fond")
    assert W.m.breakpoints(p) == ["fond~2"] and W.b.doc.review(g2) == "pending"
    W.m.p.validate()
    pause = W.m.pause_point()
    assert pause is not None and pause.block == "fond~2" and pause.all_stops() == [(p.pid, "fond~2")]
    plan, pause = W.m.plan_run()
    assert plan.jobs and pause.block == "fond~2"
    assert W.m.pause_name(p.pid, "fond~2") == "Gradient 2" and W.m.pause_name(p.pid, "fond") == "Gradient 1"
    from astro_gpu.client.services.run import pause_text
    assert pause_text(W.m, pause) == "Gradient 2"


def test_a_pause_stays_with_a_block_switched_off(world):
    """Round 2: switching a block off kept no pause (switching it on did not bring it back)."""
    W = world()
    p = W.m.paths[0]
    W.m.set_breakpoint([p], "fond", True)
    W.settle()
    g = W.block("fond", p)
    W.choose(g)
    W.g.key("D", target=W.page.canvas)
    W.settle()
    assert "gradient" in W.m.cut_blocks(W.m.view(p.pid))
    assert W.m.has_breakpoint(W.m.view(p.pid), "fond")             # kept with the block switched off
    assert "fond" not in W.m.breakpoints(W.m.view(p.pid))           # but no stop while it is off
    W.g.key("Ctrl+Z", target=W.page.canvas)
    W.settle()
    assert W.m.breakpoints(W.m.view(p.pid)) == ["fond"]


# ---------------------------------------------------------------------------------------------- deleting
def test_a_refused_delete_keeps_the_selection_and_frames_what_needs_it(world):
    """The play-test of issue #47: a refused delete keeps the block selected."""
    W = world()
    st = W.block("empilement")
    W.g.click(st)
    W.g.key("Del", target=W.page.canvas)
    W.settle(2)
    assert st in W.graph.blocks and W.b.selected == st and W.m.sel_step == "empilement"
    assert "Stacking" in W.page.canvas.toast.plain


def test_the_last_session_goes_with_its_workflow_and_ctrl_z_brings_everything_back(world):
    """The last Session takes its workflows from the frames with it (files kept); Ctrl+Z brings everything back."""
    W = world()
    n = len(W.m.paths)
    ses = W.kinds("session")[0]
    r = W.g.click_cross(ses)
    W.settle()
    assert r["ok"] and not W.m.p.sessions and not [q for q in W.m.paths if q.source is None]
    assert "Ctrl+Z" in W.page.canvas.toast.plain
    W.g.key("Ctrl+Z", target=W.page.canvas)
    W.settle()
    assert len(W.m.p.sessions) == 1 and len(W.m.paths) == n


def test_a_block_dragged_onto_the_palette_is_removed(world):
    """#54: every removable block has its cross; a placed block dragged onto the palette is removed."""
    W = world()
    crosses = {it.d["kind"] for it in W.page.canvas.blocks.values() if it.d["removable"]}
    assert {"gradient", "color", "astrometry", "denoise", "session"} <= crosses
    W.g.move_block(W.block("couleur"), None, to_palette=True)
    W.settle()
    assert "couleur" not in W.m.display_keys(W.m.paths[0])


# ---------------------------------------------------------------------------------------------- wires to Post
def test_a_wire_to_post_is_taken_away_with_del_and_stays_away(world):
    """#52: a workflow's wire to Post can be taken away (Del); a wire taken away stays away (Post gives no layer by
    itself any more), the graph built again keeps a chosen wire chosen; what is sent after still goes to Post."""
    W = world(workflows=("Classic stack",))
    post = W.one("post")
    wire = next(w for w in W.graph.wires if w.dst[0] == post)
    W.g.click_wire(wire.id)
    assert W.page.canvas.selected_wire == wire.id and W.page.canvas.wires[wire.id].isSelected() is not None
    W.page.canvas.rebuild()                                     # 2nd round: the same wire drawn again stays chosen
    assert W.page.canvas.selected_wire == wire.id
    W.page.canvas.setFocus()
    W.g.key("Del", target=W.page.canvas)
    W.settle()
    p = W.m.paths[0]
    assert not [w for w in W.graph.wires if w.dst[0] == post] and p.pid not in W.bm.to_post
    W.page.refresh()
    W.settle()
    assert p.pid not in W.bm.to_post and (p.pid, None) in W.m.post_removed()          # it stays away
    W.b.act("send_to_post", block=W.block("fond", p))                                 # a block's image, sent
    W.settle()
    sent = W.m.paths[-1]
    assert sent.pid in W.bm.to_post and p.pid not in W.bm.to_post
    W.m.add_post_layer(p.pid, None)                                                     # the wire given back
    W.settle()
    assert p.pid in W.bm.to_post and (p.pid, None) not in W.m.post_removed()


def test_post_saving_its_layers_keeps_the_workflows_not_computed_yet(world):
    """GUI exploration 4: Post holds layers only for the computed workflows; its saving a composition (a filter added)
    does not take the others' images out of Post."""
    W = world(workflows=("Full",))
    q, _key = W.m.add_branch(W.m.paths[0], "astrometrie", "gradient")
    W.settle()
    before = set(W.bm.to_post)
    assert q.pid in before
    post_page = type("PostView", (), {"is_post_page": True})()
    lay = {"id": 50, "kind": "filter", "filter": "curves", "name": "Curves", "visible": True, "opacity": 100.0,
           "mode": "Normal", "params": {}, "mask": None}
    W.m.set_post_layers([lay], post_page, scope="all")              # nothing computed: no workflow layer
    W.settle()
    assert not W.m.post_removed() and set(W.bm.to_post) == before


def test_a_port_pulled_again_says_its_image_is_a_layer_already_and_a_shared_block_is_named_by_its_owner(world):
    """5, 10: a block's image sent to Post makes a workflow named after the workflow and the block; pulled again it
    says its image is a layer already."""
    W = world(workflows=("Classic stack",))
    g = W.block("fond")
    W.g.pull_wire(g, "image", W.one("post"), "layers")
    W.settle()
    assert [q.name for q in W.m.paths] == ["Classic stack", "Classic stack · Gradient"]
    W.b.act("send_to_post", block=g)
    W.settle()
    assert len(W.m.paths) == 2 and "already" in W.page.canvas.toast.plain
    W.b.act("send_to_post", block=W.block("empilement", W.m.paths[0]))
    W.settle()
    assert W.m.paths[-1].name == "Classic stack · Stacking"


def test_escape_cancels_a_port_pulled_to_post(world):
    """GUI exploration 4: Escape while a port is pulled toward Post cancels it: released on Post it sends nothing."""
    W = world(workflows=("Classic stack",))
    n = len(W.m.paths)
    c = W.page.canvas
    a = W.g.port_point(W.block("fond"), "image", "out")
    b = W.g.block_point(W.one("post"))
    W.g.press(a)
    for pt in W.g._path(a, b):
        W.g.move(pt)
    assert c.gesture is not None and c.gesture.get("kind") == "wire"
    W.g.key("Esc", target=c)
    assert c.gesture is None and not c.ghosts
    W.g.release(b)
    W.settle()
    assert len(W.m.paths) == n and not W.graph.taps


def test_the_mouse_over_and_on_every_block_works(world):
    """Hovering a block and pressing it (its body, its output port) is what the user does first: no error, and a port
    pulled then dropped on nothing changes nothing."""
    from PySide6.QtCore import QEvent, QPointF
    from PySide6.QtWidgets import QGraphicsSceneHoverEvent
    W = world()
    before = W.m.p.to_dict()
    for b, it in list(W.page.canvas.blocks.items()):
        for pos in [QPointF(it.w / 2, it.h / 2)] + [QPointF(it.w, y) for _p, y, _c, _w in it.d["outputs"]]:
            ev = QGraphicsSceneHoverEvent(QEvent.GraphicsSceneHoverMove)
            ev.setPos(pos)
            it.hoverMoveEvent(ev)
    st = W.block("empilement")
    a = W.g.port_point(st, "image", "out")
    W.g.press(a)
    for pt in W.g._path(a, a + type(a)(0, 300)):
        W.g.move(pt)
    W.g.release(a + type(a)(0, 300))
    W.settle()
    assert W.m.p.to_dict() == before


# ---------------------------------------------------------------------------------------------- templates on a blank page
def test_a_filters_template_on_a_new_project_makes_its_empty_sessions(world):
    W = world(workflows=(), nights=0)
    W.page.show_templates()
    card = next(c for c in W.page.templates.cards if c.t["key"] == "filters_full")
    card.set_palette("H-O")
    card.mousePressEvent(None)
    W.settle()
    assert sorted(s.filter for s in W.m.p.sessions if s.filter) == ["Ha", "OIII"]
    assert [q.name for q in W.m.paths][-1] == "H-O"
    assert all("couleur" not in W.m.display_keys(q) for q in W.m.paths if W.m.is_mono(q))


def test_the_filters_card_makes_the_palette_chosen_on_it(world):
    W = world(workflows=(), nights=0)
    W.page.show_templates()
    card = next(c for c in W.page.templates.cards if c.t["key"] == "filters_full")
    card.set_palette("LRGB")
    card.mousePressEvent(None)
    W.settle()
    assert sorted(x.filter for x in W.m.p.sessions) == ["B", "G", "L", "R"]
    assert W.m.paths[-1].name == "LRGB" and (W.m.paths[-1].source or {}).get("fusion")


def test_the_dual_band_x2_card_offers_no_combination_choice(world):
    """The template is S-H-O by construction: its card has no list of combinations that would do nothing."""
    from PySide6.QtWidgets import QComboBox
    W = world(workflows=(), nights=0)
    W.page.show_templates()
    by_key = {c.t["key"]: c for c in W.page.templates.cards}
    assert not by_key["dual_x2"].findChildren(QComboBox)
    assert by_key["filters_full"].findChildren(QComboBox)


def test_the_template_cards_take_the_height_of_their_content_in_a_grid_that_wraps(world):
    """Issue #133: a card is as tall as the tallest of its row, its text whole, at the top; a tall page is empty below
    the cards; a wide page holds more columns, a narrow one fewer."""
    from PySide6.QtWidgets import QLabel, QWidget
    W = world(workflows=(), nights=0)
    W.page.resize(1600, 1000)
    W.page.show_templates()
    W.settle()

    def check():
        cards = W.page.templates.cards
        for c in cards:
            row = [x for x in cards if x.y() == c.y()]
            tallest = max(max(x.geometry().bottom() for x in r.findChildren(QWidget) if x.parent() is r) for r in row)
            assert c.height() - tallest < 30, (c.t["key"], c.height())
            for lb in c.findChildren(QLabel):
                if lb.wordWrap():
                    assert lb.height() >= lb.heightForWidth(lb.width()), (c.t["key"], lb.text())
            assert len({x.height() for x in row}) == 1
        return cards
    cards = check()
    room = W.page.templates.scroll.viewport().width()
    assert len({c.x() for c in cards}) == min(6, (room + 12) // (cards[0].width() + 12)) > 3, room
    assert max(c.geometry().bottom() for c in cards) < W.page.templates.scroll.viewport().height() * 0.9
    W.page.resize(700, 1000)
    W.settle()
    cards = check()
    assert len({c.y() for c in cards}) > 2 and len({c.x() for c in cards}) < 3


def test_no_template_asks_for_a_file_it_only_places_its_blocks(world, monkeypatch):
    """Issue #114: no template asks for anything on the blank page: its blocks are placed, its Image blocks wait for
    their file, "Several images" places one per picture's image, each a block of its own; Ctrl+Z gives the blank page."""
    from PySide6.QtWidgets import QFileDialog

    from astro_gpu.client.gallery import GALLERY
    asked = []
    for f in ("getOpenFileName", "getOpenFileNames", "getExistingDirectory"):
        monkeypatch.setattr(QFileDialog, f, lambda *a, f=f, **k: asked.append(f) or ("", ""))
    W = world(workflows=(), nights=0)
    for t in [t for t in GALLERY if t.get("copy") is None]:
        W.page.show_templates()
        next(c for c in W.page.templates.cards if c.t["key"] == t["key"]).mousePressEvent(None)
        W.settle()
        assert not asked, (t["key"], asked)
        assert W.m.paths and W.page.centre.currentWidget() is W.page.canvas, t["key"]
        if t["key"] in ("image", "images"):
            images = [q for q in W.m.paths if q.source is not None]
            assert len(images) == (3 if t["key"] == "images" else 1)
            assert all(W.m.p.workflow(q.pid).source["path"] == "" for q in images)
            assert len(W.kinds("image")) == len(images)                      # each a block of its own
        while not W.page.is_blank():
            assert W.b.undo().ok
            W.settle(2)


# ---------------------------------------------------------------------------------------------- images and their names
def test_images_are_numbered_by_their_workflows(world):
    """Issues #134, #155: Several images makes Image 1 to 3; an Image from the palette goes on with the series; one
    deleted frees its number, which the next one takes; one alone has no number; a name typed is kept."""
    W = world(workflows=(), nights=0)
    W.template("images")
    assert [q.name for q in W.m.paths] == ["Image 1", "Image 2", "Image 3"]
    W.choose(None)
    W.g.double_click_palette("image")
    W.settle()
    assert [q.name for q in W.m.paths] == ["Image 1", "Image 2", "Image 3", "Image 4"]
    W.b.act("delete_row", row=row_named(W, "Image 2").id)
    W.settle()
    W.g.double_click_palette("image")
    W.settle()
    assert [q.name for q in W.m.paths] == ["Image 1", "Image 3", "Image 4", "Image 2"]
    captions = {text for _r, text, _x, _y in W.b.drawing().captions}
    assert {"Image 1", "Image 2", "Image 3", "Image 4"} <= captions             # each row says its name
    while not W.page.is_blank():
        assert W.b.undo().ok
        W.settle(2)
    W.template("image")
    assert len(W.m.paths) == 1 and not W.m.paths[0].name[-1].isdigit()           # one alone: no number
    assert W.m.rename(W.m.paths[0].pid, "M 31")
    W.settle()
    W.g.double_click_palette("image")
    W.settle()
    assert [q.name for q in W.m.paths] == ["M 31", "Image 1"]                    # typed: kept


def test_two_images_without_a_file_are_two_blocks(world):
    """Decision 4: each Image placed is its own block, removed alone."""
    W = world()
    W.choose(None)
    W.g.double_click_palette("image")
    W.g.double_click_palette("image")
    W.settle()
    imgs = W.kinds("image")
    assert len(imgs) == 2
    W.g.click_cross(imgs[0])
    W.settle()
    assert W.kinds("image") == [imgs[1]] and len([q for q in W.m.paths if q.source is not None]) == 1


def test_an_image_placed_waits_for_its_file_and_is_not_run_without_it(world, tmp_path):
    """The third use: an Image is placed like a Session, its file chosen in its panel (no template window); without its
    file it is not run; a stretched one goes to Post as it is."""
    from astro_gpu import api
    W = world()
    W.choose(None)
    W.g.double_click_palette("image")
    W.settle()
    assert W.page.templates is None or W.page.centre.currentWidget() is W.page.canvas      # no template
    img = W.m.paths[-1]
    assert img.source == {"path": "", "stretched": False}
    assert W.m.graph(img) == [] and not W.m.has_data(img) and W.m.has_data()
    assert img.pid in api.plan_all(W.m.p).skipped                      # not run without its file
    W.m.set_image_file(img, a_fits(tmp_path / "stack.fits"))
    W.settle()
    img = W.m.view(img.pid)
    assert W.m.has_data(img) and img.source["path"].endswith("stack.fits")
    assert img.pid not in api.plan_all(W.m.p).skipped
    W.m.set_image_file(img, stretched=True)
    W.settle()
    assert W.m.view(img.pid).source["stretched"] and W.m.p.workflow(img.pid).blocks == []


def test_an_image_and_a_session_have_their_cross_and_ctrl_z_brings_either_back(world):
    W = world(workflows=("Classic stack",))
    W.choose(None)
    W.g.double_click_palette("image")
    W.settle()
    img = W.kinds("image")[0]
    assert W.page.canvas.blocks[img].d["removable"]
    W.g.click_cross(img)
    W.settle()
    assert all(q.source is None for q in W.m.paths)
    W.g.key("Ctrl+Z", target=W.page.canvas)
    W.settle()
    assert any(q.source is not None for q in W.m.paths)
    ses = W.kinds("session")[0]
    assert W.page.canvas.blocks[ses].d["removable"]


# ---------------------------------------------------------------------------------------------- the menu of another workflow
def test_the_menu_of_another_workflows_block_stays_open(world, monkeypatch):
    """Issue #154: a right-click on a block of a workflow that is not the current one makes it current; the menu stays
    open while the palette follows, and the palette's rebuild never shows a window of its own."""
    from PySide6.QtCore import QEvent, QObject
    from PySide6.QtWidgets import QMenu

    class Windows(QObject):
        seen: list = []

        def eventFilter(self, obj, e):                   # noqa: N802
            if e.type() == QEvent.Show and obj.isWidgetType() and obj.isWindow() and not isinstance(obj, QMenu):
                self.seen.append(obj)
            return False
    W = world(workflows=("Full", "Classic stack"))
    W.choose(W.block("fond", W.view("Full")))
    other = W.block("couleur", W.view("Classic stack"))
    spy = Windows()
    QApplication.instance().installEventFilter(spy)
    try:
        r = W.g.menu_pick(other, "", open_only=True)
        assert r["items"] and W.b.selected == other and W.m.path.pid == W.view("Classic stack").pid
        W.settle()
        assert not spy.seen, spy.seen
        assert W.page.palette._for_text.endswith("“Classic stack”")
    finally:
        QApplication.instance().removeEventFilter(spy)


# ---------------------------------------------------------------------------------------------- files added by mistake
def test_a_file_added_by_mistake_is_warned_and_taken_out_one_by_one(world, tmp_path):
    """Issue #58: a JPEG added among the raw files is warned on the Session block; one file taken out alone, a file of
    a folder chosen whole too (the folder becomes its other files); the last light can go."""
    from astro_gpu.core.project import type_warnings
    assert [w["key"] for w in type_warnings({"lights": ["a.ARW", "b.ARW", "c.JPG"], "flats": ["f.fits"]})] == \
        ["unread", "other_type"]
    W = world()
    raw = W.raw / "n1"
    jpg = tmp_path / "phone" / "DSC05869.JPG"
    jpg.parent.mkdir()
    jpg.write_bytes(b"\0" * 50)
    assert W.m.add_session_entries(1, "lights", [str(jpg)]) == 1
    W.settle()
    warns = W.m.file_warnings(1)
    assert [x["key"] for x in warns] == ["unread"] and warns[0]["name"] == "DSC05869.JPG"
    ses = W.kinds("session")[0]
    assert W.page.canvas.blocks[ses].d["warned"]
    assert W.m.remove_session_files(1, "lights", [jpg.as_posix()])
    W.settle()
    assert not W.m.file_warnings(1) and not W.page.canvas.blocks[ses].d["warned"]
    assert W.m.remove_session_files(1, "lights", [(raw / "a1.ARW").as_posix()])     # a file of a whole folder
    W.settle()
    assert [Path(e).name for e in W.m.session(1).lights] == ["b1.ARW"]
    assert W.m.remove_session_files(1, "lights", [(raw / "b1.ARW").as_posix()])     # the last light too
    assert W.m.session(1).lights == []


# ---------------------------------------------------------------------------------------------- combinations and images
def test_a_linear_image_is_chosen_as_a_fusion_channel_on_its_row(world, tmp_path):
    """Issue #55: a channel's row of a combination's panel chooses a linear Image's output; an image already stretched
    is listed greyed, with its reason."""
    from PySide6.QtWidgets import QComboBox
    W = world()
    W.choose(None)
    W.g.double_click_palette("image")
    W.settle()
    img = W.m.paths[-1]
    W.m.set_image_file(img, a_fits(tmp_path / "stack.fits"))
    stretched = W.m.add_image_workflow("Done", a_fits(tmp_path / "done.fits"), stretched=True)
    W.settle()
    W.b.act("add_combination", type="ho")
    W.settle()
    fu = W.m.paths[-1]
    assert fu.source["fusion"]["inputs"] == []                                   # placed empty
    offered = {q.pid: why for q, why in W.m.input_choices(fu, "OIII")}
    assert offered[img.pid] == "" and "stretch" in offered[stretched.pid]
    with pytest.raises(ValueError):
        W.m.set_fusion_input(fu, "OIII", stretched.pid)
    assert W.m.set_fusion_input(fu, "Ha", W.m.paths[0].pid)
    W.settle()
    W.choose(W.kinds("combine")[0])
    rows = [c for c in W.page.panel.findChildren(QComboBox) if c.findData(img.pid) >= 0]
    assert len(rows) == 2                                                        # Ha, OIII
    grey = rows[1].model().item(rows[1].findData(stretched.pid))
    assert not grey.isEnabled() and "stretch" in grey.toolTip()
    i = rows[1].findData(img.pid)
    rows[1].setCurrentIndex(i)
    rows[1].activated.emit(i)
    W.settle()
    fu = W.m.view(fu.pid)
    assert [x["pid"] for x in fu.source["fusion"]["inputs"]] == [W.m.paths[0].pid, img.pid]
    assert W.m.fusion_problem(fu) == ""
    cb = W.kinds("combine")[0]
    assert sorted(str(w.dst[1]) for w in W.graph.wires if w.dst[0] == cb) == ["Hα", "O III"]     # wired on the graph


def test_a_fusion_can_start_from_a_linear_image_alone(world, tmp_path):
    """Issue #55: no stack in the project: a linear Image gives a combination its channels (dual band)."""
    W = world(workflows=(), nights=0)
    W.choose(None)
    W.g.double_click_palette("image")
    W.settle()
    img = W.m.paths[-1]
    W.m.set_image_file(img, a_fits(tmp_path / "stack.fits"))
    W.settle()
    W.b.act("add_combination", type="ho")
    W.settle()
    fu = W.m.paths[-1]
    assert fu.source["fusion"]["inputs"] == []
    W.m.set_fusion_input(fu, "Ha", img.pid)                        # a colour image's channels: dual band by default
    W.m.set_fusion_input(W.m.view(fu.pid), "OIII", img.pid)
    f = W.m.view(fu.pid).source["fusion"]
    assert [(i["pid"], i["channel"]) for i in f["inputs"]] == [(img.pid, "R"), (img.pid, "GB")] and "stars" not in f
    assert W.m.fusion_problem(W.m.paths[-1]) == ""


def test_a_linear_image_takes_stars_without_denoise(world, tmp_path):
    """Issue #55: Stars on a linear Image, from the palette, needs no Denoise; the frames keep needing Denoise."""
    from astro_gpu.core import workflow as wfl
    W = world(workflows=(), nights=0)
    W.page.place_images([a_fits(tmp_path / "stack.fits")])
    W.settle()
    img = W.m.paths[-1]
    W.choose(W.kinds("image")[0])
    assert W.card("stars").enabled and not W.card("stars").reason
    W.g.double_click_palette("stars")
    W.settle()
    v = W.m.view(img.pid)
    assert "etoiles" in W.m.display_keys(v) and W.m.all_stages(v)[-2:] == ["render_simple", "star_removal"]
    assert "denoise" not in W.m.all_stages(v)
    with pytest.raises(wfl.BlockRefused, match="Denoise"):
        wfl.check_blocks([wfl.node("calibration"), wfl.node("registration"), wfl.node("selection"),
                          wfl.node("stacking"), wfl.node("output"), wfl.node("stars")], "frames")
