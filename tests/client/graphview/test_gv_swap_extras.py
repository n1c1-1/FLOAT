"""What today's page had and the swap put back, and the owner's "files only": the Combination card's flyout of types;
a dropped folder stored as its files; a project's folder entries converted to the files the engine reads there."""
from __future__ import annotations

from astro_gpu.client.graphview import files
from astro_gpu.client.graphview.palette import Palette


class Reader:
    def __init__(self):
        self.seen = {}

    def known(self, p):
        return self.seen.get(p)

    def note(self, p, d):
        self.seen[p] = d


def test_the_combination_card_offers_its_types_and_places_the_one_chosen(bridge):
    pal = Palette(bridge)
    m = pal.combination_types(pal.cards["combine"], run=False)
    names = [a.text() for a in m.actions()]
    assert names == ["S-H-O", "H-O", "RGB", "LRGB"]
    m.actions()[1].trigger()                                       # H-O
    assert bridge.combo_type == "ho" and pal.cards["combine"].title.text() == "H-O"
    pal.double_click("combine")
    js = [b for b in bridge.doc.graph.blocks.values() if b.kind == "combine"]
    assert len(js) == 1 and js[0].settings.get("type") == "ho"
    assert bridge.doc.undo().ok                                    # one step: the block and its type
    assert not any(b.kind == "combine" for b in bridge.doc.graph.blocks.values())


def test_a_dropped_folder_is_stored_as_its_files(tmp_path):
    d = tmp_path / "lights"
    d.mkdir()
    for n in ("b.fit", "a.fit", "c.fits", "notes.txt"):
        (d / n).write_bytes(b"\0" * 10)
    r = files.resolve(Reader(), [str(d)], i=files.index_of_name("FITS"))
    assert r.items == [f"{d.as_posix()}/{n}" for n in ("a.fit", "b.fit", "c.fits")]     # files, never the folder
    assert r.others == 1


def test_a_folder_entry_of_a_project_becomes_the_files_the_engine_reads(tmp_path):
    from astro_gpu import api
    from astro_gpu.core.store import InputScan
    from astro_gpu.graphcore.convert import files_only
    d = tmp_path / "raw"
    d.mkdir()
    for n in ("IMG_2.ARW", "IMG_1.ARW", "IMG_1.JPG"):                 # the camera's JPEG beside raws: not a frame
        (d / n).write_bytes(b"\0" * 20)
    p = api.create_project(tmp_path / "P", "P")
    api.add_session(p, lights=[d])
    before = InputScan().frames(p)
    assert files_only(p) == 1
    assert p.sessions[0].lights == [(d / "IMG_1.ARW").as_posix(), (d / "IMG_2.ARW").as_posix()]
    assert InputScan().frames(p) == before                         # the same frames: every key stays
    p.sessions[0].lights = [str(tmp_path / "unplugged")]           # a folder that cannot be read: kept as it is
    assert files_only(p) == 0 and p.sessions[0].lights == [str(tmp_path / "unplugged")]


def _stars_doc(qapp):
    from astro_gpu.client.graphview.bridge import Bridge
    from astro_gpu.client.graphview.document import MemoryDocument
    from tests.graphcore import fixtures
    b = Bridge(MemoryDocument(fixtures.Doc.from_template("colour_complete").project))
    stars = next(k for k, x in b.doc.graph.blocks.items() if x.kind == "stars")
    for k in [k for k, x in b.doc.graph.blocks.items() if x.kind == "combine_layers"]:   # its own Combine layers out
        assert b.act("remove_block", block=k).ok
    return b, stars


def test_the_stars_block_menu_sends_each_image_to_post_and_makes_combine_layers(qapp):
    from astro_gpu.client.graphview.menu import menu
    b, stars = _stars_doc(qapp)
    items = {it[0]: it for it in menu(b.doc, stars) if it is not None}
    for text in ("Starless to Post", "Stars to Post", "Combine layers"):
        assert text in items and items[text][3], (text, items.get(text))
    _t, op, args, _ok, _why = items["Starless to Post"]
    assert b.act(op, **args).ok
    post = next(k for k, x in b.doc.graph.blocks.items() if x.kind == "post")
    assert any(w.src == [stars, "Starless"] and w.dst[0] == post for w in b.doc.graph.wires)
    items = {it[0]: it for it in menu(b.doc, stars) if it is not None}
    assert "Starless out of Post" in items                        # and back out of it
    n = sum(x.kind == "combine_layers" for x in b.doc.graph.blocks.values())
    _t, op, args, _ok, _why = items["Combine layers"]
    assert b.act(op, **args).ok
    assert sum(x.kind == "combine_layers" for x in b.doc.graph.blocks.values()) == n + 1


def test_a_stars_image_pulled_into_the_band_at_its_right_makes_combine_layers(qapp):
    from PySide6.QtCore import QPointF

    from astro_gpu.client.graphview.canvas import GraphCanvas
    b, stars = _stars_doc(qapp)
    c = GraphCanvas(b)
    c.resize(1400, 800)
    c.show()
    c.rebuild()
    QApplication = __import__("PySide6.QtWidgets", fromlist=["QApplication"]).QApplication
    QApplication.processEvents()
    n = sum(x.kind == "combine_layers" for x in b.doc.graph.blocks.values())
    r = c.blocks[stars].sceneBoundingRect()
    st = dict(kind="wire", src=stars, port="Stars only", cands={}, hot=None, it=c.blocks[stars])
    c._end_wire(st, QPointF(r.right() + 400, r.bottom() + 300))
    assert sum(x.kind == "combine_layers" for x in b.doc.graph.blocks.values()) == n + 1
    c.close()
