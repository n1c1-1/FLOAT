"""The graph page (graphview/page.py) driven with REAL events through graphview/gestures.py: the validated scenarios
s01-s42 of the prototype that apply to the page, and the owner's decisions on it. Each test builds its graph with the
core's operations (no gesture), then makes one or two real gestures and checks what the user sees (items, toasts,
selection, the PREPROCESSING control) and the graph; every gesture's undo and redo give the graph back exactly."""
from __future__ import annotations

import time

import pytest
from PySide6.QtWidgets import QApplication

from astro_gpu.graphcore.operations import json_of

#: the scenarios' screen
W, H = 1920, 1080


# ======================================================================================================= the bench
def template(key):
    """A graph of the core's test templates, built by the core's operations."""
    from tests.graphcore import fixtures
    return fixtures.Doc.from_template(key).project


def pump(s=0.0):
    t0 = time.perf_counter()
    while True:
        QApplication.processEvents()
        if time.perf_counter() - t0 >= s:
            break
        time.sleep(0.005)


def fwd(p) -> str:
    return str(p).replace("\\", "/")


def make(folder, n, ext, prefix="IMG"):
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        (folder / f"{prefix}_{i:04d}.{ext}").write_bytes(bytes(64))
    return folder


class Bench:
    """The page at 1920 x 1080 on a graph held in memory, its gestures and what it shows."""

    def __init__(self, page):
        from astro_gpu.client.graphview.gestures import Gestures
        self.page = page
        self.b = page.bridge
        self.g = Gestures(page)
        self.links = []
        self.b.rich_toast.connect(lambda _t, ls: self.links.append(ls))

    # ---- documents
    def install(self, graph=None):
        from astro_gpu.client.graphview.document import MemoryDocument
        from astro_gpu.graphcore import api
        self.page.set_document(MemoryDocument(graph if graph is not None else api.blank_project()))
        pump()
        self.page.canvas.setFocus()
        return self

    def tpl(self, key):
        return self.install(template(key))

    def act(self, name, **kw):
        res = self.b.act(name, quiet=True, **kw)
        assert res.ok, self.b.say(res.reason)
        return res

    def setup(self, *steps):
        """Set-up steps of the core, then the history emptied: the test's gestures are the only undo steps."""
        for name, kw in steps:
            self.act(name, **kw)
        self.b.doc.document._undo.clear()
        pump()

    # ---- readings
    @property
    def graph(self):
        return self.b.graph

    @property
    def canvas(self):
        return self.page.canvas

    def js(self):
        return json_of(self.graph)

    def steps(self):
        return len(self.b.doc.document._undo)

    def row(self, name):
        return next(r for r in self.graph.rows if r.name == name)

    def ids(self, kind, row=None):
        """Block ids of a kind, rows in order, a row's nights then its chain."""
        rows = [self.row(row) if isinstance(row, str) else row] if row is not None else self.graph.rows
        out = []
        for r in rows:
            for b in [x for n in r.nights for x in n if x is not None] + list(r.blocks):
                if self.graph.blocks[b].kind == kind:
                    out.append(b)
        return out

    def one(self, kind, row=None, n=0):
        return self.ids(kind, row)[n]

    def post(self):
        return next(k for k, b in self.graph.blocks.items() if b.kind == "post")

    def post_wires(self):
        return [w for w in self.graph.wires if w.dst[0] == self.post()]

    def boxes(self, ids=None):
        d = self.b.drawing()
        return {k: (x.x, x.y, x.w, x.h) for k, x in d.boxes.items() if ids is None or k in ids}

    def overlaps(self):
        d = self.b.drawing()
        bs = list(d.boxes.values()) + list(d.cards.values())
        return [(a, c) for i, a in enumerate(bs) for c in bs[i + 1:]
                if a.x < c.x + c.w - 0.5 and c.x < a.x + a.w - 0.5 and a.y < c.y + c.h - 0.5 and c.y < a.y + a.h - 0.5]

    def toast(self):
        t = self.canvas.toast
        return t.plain if t.isVisible() else ""

    def link_keys(self):
        return [k for _t, k in self.canvas.toast.links()]

    def key(self, keys, target=None):
        w = target or self.canvas
        w.setFocus()
        return self.g.key(keys, w)

    def exact(self, before, after=None):
        """Ctrl+Z gives `before` back exactly, Ctrl+Y the state after the gesture (the page's keys)."""
        after = after if after is not None else self.js()
        self.key("Ctrl+Z")
        assert self.js() == before, "Ctrl+Z did not give the graph back exactly"
        self.key("Ctrl+Y")
        assert self.js() == after, "Ctrl+Y did not give the gesture's graph back exactly"

    def insert_after(self, kind, block):
        """The caret place of `kind` in the gap right after `block` (lit)."""
        box = self.b.drawing().boxes[block]
        for pl in self.b.places(kind=kind):
            r = pl.rect
            if pl.kind == "insert" and pl.lit and abs(r.y + r.h / 2 - (box.y + box.h / 2)) < box.h * 0.6 and \
                    box.x + box.w - 4 <= r.x <= box.x + box.w + 40:
                return pl
        raise AssertionError(f"no insert place for {kind} after {block}")

    def branch_under(self, kind, block):
        """The branch strip of `kind` under the column of `block` (the column the new block takes)."""
        box = self.b.drawing().boxes[block]
        cands = [pl for pl in self.b.places(kind=kind) if pl.kind == "branch" and abs(pl.rect.x - box.x) < 4 and
                 pl.rect.y > box.y + box.h - 2]
        assert cands, f"no branch strip under {block}"
        return min(cands, key=lambda pl: pl.rect.y)

    def on_screen(self, ident):
        """The WHOLE item of a block inside the viewport."""
        c = self.canvas
        view = c.mapToScene(c.viewport().rect()).boundingRect().adjusted(-0.5, -0.5, 0.5, 0.5)
        return view.contains(c.blocks[ident].sceneBoundingRect())


@pytest.fixture()
def page(qapp):
    from astro_gpu.client.graphview.document import MemoryDocument
    from astro_gpu.client.graphview.page import GraphPage
    p = GraphPage(MemoryDocument())
    p.resize(W, H)
    p.show()
    p.activateWindow()
    pump()
    yield p
    if p.bridge._drops is not None:
        p.bridge._drops.reader.wait(30)
    p.hide()
    p.deleteLater()
    pump()


#: a stack placed block by block: the Session comes alone, then its Calibration, Registration, Selection, Stacking
STACK = (("add_row", dict(kind="session")),) + tuple(
    ("add_block", dict(kind=k)) for k in ("calibration", "registration", "selection", "stacking"))


@pytest.fixture()
def B(page):                                                          # noqa: N802 - the scenarios' name
    return Bench(page)


def settle(B, timeout=30):                                            # noqa: N803
    """The background listing of the drops finished and its signals delivered."""
    reader = B.b.drops.reader
    for _i in range(3):
        assert reader.wait(timeout)
        pump(0.05)


# ======================================================================================================= s01-s05: Sessions
def test_s01_double_click_session_makes_a_stack(B):
    B.install()
    before = B.js()
    B.g.double_click_palette("session")
    assert len(B.graph.rows) == 1
    r = B.graph.rows[0]
    assert len(r.nights) == 1 and r.nights[0][1] is None
    assert r.blocks == []                                     # the Session alone: each next block placed by the user
    s = r.nights[0][0]
    assert B.canvas.blocks[s].d["title"] == "Session 1" and B.canvas.blocks[s].d["badge"] == "reference"
    assert not B.post_wires() and not B.overlaps()
    assert B.b.selected in B.canvas.blocks and B.canvas.blocks[B.b.selected].d["selected"]
    assert B.steps() == 1
    B.exact(before)


def test_s02_files_on_session_targets_count_on_the_block(B, tmp_path):
    B.install().setup(("add_row", dict(kind="session")))
    s = B.one("session")
    lights = make(tmp_path / "lights", 40, "fits", "L")
    darks = make(tmp_path / "darks", 12, "fits", "D")
    res = B.g.drop_files(s, "lights", sorted(str(p) for p in lights.iterdir()))
    assert res["targets"] == ["Lights", "Darks", "Flats", "Bias", "Flat darks"]
    assert "40 lights" in B.canvas.blocks[s].d["summary"]
    before = B.js()
    B.g.drop_files(s, "darks", sorted(str(p) for p in darks.iterdir()))
    assert "12 darks" in B.canvas.blocks[s].d["summary"] and B.steps() == 2
    assert not B.canvas.targets                                       # the targets go with the drop
    B.exact(before)


def test_s03_drag_session_onto_plus_session(B):
    B.install().setup(*STACK)
    lit = {pl["kind"] for pl in B.g.visible_places("session") if pl["lit"]}
    assert {"night", "row"} <= lit
    before = B.js()
    B.g.drag_palette("session", {"kind": "night", "lit": True})
    r = B.graph.rows
    assert len(r) == 1 and len(r[0].nights) == 2
    d = B.b.drawing()
    s1, s2 = (n[0] for n in r[0].nights)
    assert abs(d.boxes[s1].x - d.boxes[s2].x) < 1 and d.boxes[s2].y > d.boxes[s1].y       # the nights stacked
    assert any(w.style == "bar" for w in d.wires)                                          # fanning into Registration
    assert not B.canvas.ghosts and not B.overlaps()
    B.exact(before)


def test_s04_session_dragged_to_its_own_row(B):
    B.install().setup(("add_row", dict(kind="session")))
    B.setup(("add_night", dict(row=B.graph.rows[0].id)))
    s1, s2 = (n[0] for n in B.graph.rows[0].nights)
    before_boxes, before = B.boxes(), B.js()
    B.g.move_block(s2, {"kind": "row", "lit": True})
    assert len(B.graph.rows) == 2 and B.graph.rows[1].name == "Session 2"
    assert len(B.graph.rows[0].nights) == 1 and B.boxes()[s1] == before_boxes[s1]       # row 1 unchanged above
    B.exact(before)


def test_s05_back_onto_plus_session_two_undos_give_the_drawing_back(B):
    B.install().setup(("add_row", dict(kind="session")))
    B.setup(("add_night", dict(row=B.graph.rows[0].id)))
    step3, js3 = B.boxes(), B.js()
    s2 = B.graph.rows[0].nights[1][0]
    B.g.move_block(s2, {"kind": "row", "lit": True})
    assert len(B.graph.rows) == 2
    B.g.move_block(B.graph.rows[1].nights[0][0], {"kind": "night", "lit": True})
    assert len(B.graph.rows) == 1 and len(B.graph.rows[0].nights) == 2
    B.key("Ctrl+Z")
    B.key("Ctrl+Z")
    assert B.boxes() == step3 and B.js() == js3


# ======================================================================================================= s06-s10: chains
def test_s06_branch_strip_makes_a_variant_below(B):
    B.install().setup(*STACK)
    r = B.graph.rows[0]
    parent = list(r.blocks)
    before = B.js()
    B.g.drag_palette("selection", B.branch_under("selection", r.blocks[1]))
    assert len(B.graph.rows) == 2
    v = B.graph.rows[1]
    assert v.fork is not None and v.fork[0] == r.id and v.name == "Session 1 2"
    assert B.graph.rows[0].blocks == parent
    sel = next(b for b in v.blocks if B.graph.blocks[b].kind == "selection")
    d = B.b.drawing()
    assert d.boxes[sel].y > d.boxes[parent[1]].y and B.on_screen(sel)
    assert "like the mother" in B.canvas.blocks[sel].d["summary"]
    B.exact(before)


def test_s08_color_on_the_branch_after_gradient(B):
    B.tpl("colour_complete")
    r = B.graph.rows[0]
    gr = B.one("gradient")
    nxt = r.blocks[r.blocks.index(gr) + 1]
    n0 = len(B.graph.rows)
    before = B.js()
    B.g.drag_palette("color", B.branch_under("color", nxt))
    assert len(B.graph.rows) == n0 + 1
    assert sum(1 for b in B.graph.rows[0].blocks if B.graph.blocks[b].kind == "color") == 1
    v = next(x for x in B.graph.rows if x.fork)
    assert v.fork[0] == r.id
    d = B.b.drawing()
    assert d.boxes[v.blocks[0]].y > d.boxes[gr].y
    B.exact(before)


def test_s09_variant_here_then_insert_moves_only_the_variant(B):
    B.tpl("colour_complete")
    gr = B.one("gradient")
    n0 = len(B.graph.rows)
    res = B.g.menu_pick(gr, "Variant here")
    assert res["ok"] and len(B.graph.rows) == n0 + 1, res
    v = next(x for x in B.graph.rows if x.fork)
    g2 = next(b for b in v.blocks if B.graph.blocks[b].kind == "gradient")
    before_boxes, parent = B.boxes(), B.boxes(B.graph.rows[0].blocks)
    before = B.js()
    B.g.drag_palette("denoise", B.insert_after("denoise", g2))
    assert B.boxes(B.graph.rows[0].blocks) == parent                  # the parent row did not move
    v = next(x for x in B.graph.rows if x.fork)
    now = B.boxes()
    moved = [b for b in before_boxes if b in now and before_boxes[b] != now[b]]
    right = {B.post()} | {x for r in B.graph.rows if r.blocks and B.graph.blocks[r.blocks[0]].kind in
                          ("combine", "combine_layers") for x in r.blocks}
    assert moved and not [b for b in moved if b not in v.blocks and b not in right]
    assert not B.overlaps()
    B.exact(before)


def test_s10_to_post_adds_one_wire_no_row(B):
    B.tpl("colour_minimal")
    n_rows, n_wires = len(B.graph.rows), len(B.post_wires())
    before = B.js()
    res = B.g.menu_pick(B.one("stacking"), "To Post")
    assert res["ok"], res
    assert len(B.graph.rows) == n_rows and len(B.post_wires()) == n_wires + 1
    assert B.canvas.blocks[B.post()].d["summary"] == "2 layers"
    assert "to Post" in B.toast() and B.toast().endswith("Ctrl+Z")
    assert not B.overlaps()
    B.exact(before)


# ======================================================================================================= s11-s18: filters
def test_s11_badges_and_the_global_fold(B):
    from astro_gpu.client.graphview.items import StackItem
    B.tpl("sho_rgb")
    assert [r.name for r in B.graph.rows if r.nights] == ["SII", "Ha", "OIII", "RGB"]
    assert B.graph.reference == B.row("SII").id
    assert B.canvas.blocks[B.one("session", "SII")].d["badge"] == "reference"
    for name in ("Ha", "OIII", "RGB"):
        s = B.one("session", name)
        assert B.canvas.blocks[s].d["badge"] == "on the grid of “SII”"
        assert "on the grid of SII" in B.canvas.blocks[s].d["summary"]
    assert B.page.position() == "blocks" and B.page.positions["blocks"].isChecked()
    items = B.g.menu_pick(B.one("session", "SII"), "", open_only=True)["items"]
    assert items and not any("fold" in t.lower() for t, *_r in items)      # no fold of one stack: folding is global
    B.g.display("folded")
    assert B.b.view.get("folded") is True and B.page.position() == "folded" and B.page.positions["folded"].isChecked()
    B.canvas.auto_view()
    for r in [r for r in B.graph.rows if r.nights]:
        it = B.canvas.blocks[r.id]
        assert isinstance(it, StackItem) and it.d["title"] == r.name
        assert it.d["badge"] and it.d["badge"] == B.b.doc.row_badge(r.id)
    sr = B.canvas.scene().sceneRect()
    assert sr.width() * B.canvas.zoom <= B.canvas.viewport().width() + 2     # the folded graph fits the width


def test_s11b_a_stack_of_several_nights_shows_its_name_and_badge(B):
    B.tpl("sho_rgb")
    ha = B.row("Ha")
    B.setup(("add_night", dict(row=ha.id)))
    caps = [it for it in B.canvas.scene().items() if type(it).__name__ == "CaptionItem" and it.text == "Ha"]
    assert caps and caps[0].badge == "on the grid of “SII”"


def test_s12_reference_for_all_in_one_undo_step(B):
    B.tpl("sho_rgb")
    astro = B.one("astrometry")
    s = B.one("session", "RGB")
    before = B.js()
    res = B.g.menu_pick(s, "Reference for all")
    assert res["ok"], res
    assert B.graph.reference == B.row("RGB").id
    assert B.b.row_of(astro).name == "RGB"
    assert B.canvas.blocks[s].d["badge"] == "reference"
    assert B.canvas.blocks[B.one("session", "SII")].d["badge"] == "on the grid of “RGB”"
    B.exact(before)


def test_s13_variant_here_on_ha_gradient_dashed_wire(B):
    from PySide6.QtCore import Qt
    B.tpl("sho_rgb")
    ha = B.row("Ha")
    gr = B.one("gradient", "Ha")
    before = B.js()
    res = B.g.menu_pick(gr, "Variant here")
    assert res["ok"], res
    assert any(r.fork and r.fork[0] == ha.id for r in B.graph.rows)
    dashed = [w for w in B.b.drawing().wires if w.style == "port_inactive"]
    assert dashed and B.canvas.wires[dashed[0].id].pen().style() == Qt.DashLine
    B.exact(before)


def test_s14_click_on_the_dashed_wire_takes_the_variant(B):
    B.tpl("sho_rgb")
    B.setup(("add_variant", dict(block=B.one("gradient", "Ha"))))
    n = len(B.post_wires())
    w = next(w for w in B.b.drawing().wires if w.style == "port_inactive")
    before = B.js()
    B.g.click_wire(w.id)
    d = B.b.drawing()
    assert not any(x.id == w.id and x.style == "port_inactive" for x in d.wires)
    assert any(x.style == "port_inactive" for x in d.wires)              # Ha's own wire is the dashed one now
    assert len(B.post_wires()) == n
    B.exact(before)


def test_s16_session_dropped_on_the_empty_s_ii_port(B):
    B.tpl("sho_rgb")
    join = B.one("combine")
    B.setup(("set_combination_type", dict(block=join, type="ho")), ("set_combination_type", dict(block=join, type="sho")))
    assert not any(tuple(w.dst) == (join, "S II") for w in B.graph.wires)
    n = len(B.graph.rows)
    pl = next(p for p in B.g.visible_places("session") if p["kind"] == "port" and p["lit"] and "S II" in p["tip"])
    before = B.js()
    B.g.drag_palette("session", pl["i"])
    assert len(B.graph.rows) == n + 1
    r = B.graph.rows[-1]
    kinds = [B.graph.blocks[b].kind for b in r.blocks]
    # what the port needs, nothing more: the S-H-O is Combine layers' Background, a starless image
    assert r.name == "S II" and kinds == ["stacking", "denoise", "stars"], (r.name, kinds)
    assert any(tuple(w.dst) == (join, "S II") and w.src[1] == "Starless" for w in B.graph.wires)
    B.exact(before)


def test_s17_deleting_the_reference_stack(B):
    B.tpl("sho_rgb")
    sii = B.row("SII")
    before = B.js()
    res = B.g.menu_pick(sii.nights[0][0], "Delete workflow")
    assert res["ok"], res
    assert all(r.name != "SII" for r in B.graph.rows)
    assert B.graph.reference is not None and B.graph.reference != sii.id
    new_ref = B.b.row(B.graph.reference)
    assert B.canvas.blocks[new_ref.nights[0][0]].d["badge"] == "reference"
    assert any(B.b.doc.title(b) == "Astrometry" for b in B.b.drawing().boxes)
    assert B.toast()
    B.exact(before)


def test_s18_ctrl_d_duplicates_the_stack_below(B):
    B.tpl("sho_rgb")
    ha = B.row("Ha")
    B.g.select(ha.nights[0][0])
    before = B.js()
    B.key("Ctrl+D")
    copy = B.row("Ha copy")
    d = B.b.drawing()
    first = copy.nights[0][0] if copy.nights else copy.blocks[0]
    assert d.boxes[first].y > d.boxes[ha.nights[0][0]].y
    B.exact(before)
