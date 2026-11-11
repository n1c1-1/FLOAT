"""THE WIRES OF THE GRAPH NEVER PASS UNDER A BLOCK (11 November: the Stars/background block's two images went around the
Synthetic stars block in two arcs drawn behind it, and nothing left Synthetic stars). A wire runs from the block it
leaves to the block it enters; on its way it crosses the gaps, never another block's rectangle.

Synthetic stars after Stars/background: the stars enter Synthetic stars; two wires enter Combine layers' stars input, the
real stars from Stars/background and the synthetic ones from Synthetic stars (the two layers of the Stars group of its
layer in Post, issue #93: postengine.with_synth_rows); the starless image goes on to Combine layers clear of it."""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import QPointF, QRectF  # noqa: E402
from PySide6.QtWidgets import QGraphicsPathItem  # noqa: E402

from astro_gpu.client.graph import BH, BlockItem  # noqa: E402
from astro_gpu.core import workflow as wfl  # noqa: E402
from tests.client.test_fusion_sessions_client import done, open_page, qapp, settle  # noqa: E402, F401


def wires(view):
    """(the wire's item, its path in the scene) for each open path of the page: the wires, not the arrows' heads."""
    out = []
    for it in view.scene().items():
        if not isinstance(it, QGraphicsPathItem):
            continue
        path = it.mapToScene(it.path())
        if path.elementCount() < 2 or path.elementAt(0).x == path.elementAt(path.elementCount() - 1).x and \
                path.elementAt(0).y == path.elementAt(path.elementCount() - 1).y:
            continue
        out.append((it, path))
    return out


def blocks(view):
    """Each block's rectangle (its body, the ports' dots on its sides excepted)."""
    return [(it.node, QRectF(it.node.col, it.node.y - BH / 2, it.node.w, BH)) for it in view.scene().items()
            if isinstance(it, BlockItem)]


def under(view) -> list[str]:
    """The wires that pass under a block they neither leave nor enter: one line each."""
    bad = []
    rects = blocks(view)
    for it, path in wires(view):
        a, b = path.pointAtPercent(0), path.pointAtPercent(1)
        ends = [r for _n, r in rects if r.adjusted(-3, -3, 3, 3).contains(a) or r.adjusted(-3, -3, 3, 3).contains(b)]
        for n, r in rects:
            if r in ends:
                continue
            inner = r.adjusted(1, 1, -1, -1)
            steps = max(50, int(path.length() / 2))
            if any(inner.contains(path.pointAtPercent(path.percentAtLength(path.length() * i / steps)))
                   for i in range(steps + 1)):
                bad.append(f"{it.toolTip() or type(it).__name__} from {a.toPoint()} to {b.toPoint()} under {n.key}")
    return bad


def test_no_wire_of_the_templates_passes_under_a_block(qapp, tmp_path, monkeypatch):  # noqa: F811
    w, m, page = open_page(tmp_path, monkeypatch, workflows=tuple(wfl.TEMPLATES))
    try:
        settle(w, page)
        assert under(page.graph) == []
    finally:
        done(w, m, page)


def test_synthetic_stars_feed_combine_layers_and_no_wire_passes_under_it(qapp, tmp_path, monkeypatch):  # noqa: F811
    w, m, page = open_page(tmp_path, monkeypatch, workflows=("Full",))
    try:
        settle(w, page)
        m.insert_block(m.paths[0], "synth")
        settle(w, page)
        view = page.graph
        g = view.graph
        q = m.paths[0]
        stars, synth = g.node_of(q, "etoiles"), g.node_of(q, "synth")
        comp = g.node_of(q, "composition")
        assert synth is not None and comp is not None and synth.parent is stars

        def ends(path):
            return path.pointAtPercent(0), path.pointAtPercent(1)

        def on(pt: QPointF, n, side):                    # on the block's left or right side
            x = n.col if side == "in" else n.col + n.w
            return abs(pt.x() - x) < 2 and n.y - BH / 2 - 1 <= pt.y() <= n.y + BH / 2 + 1
        into_comp = [(it, a, b) for it, path in wires(view) for a, b in [ends(path)] if on(b, comp, "in")]
        assert any(on(a, synth, "out") for _it, a, _b in into_comp), "no wire from Synthetic stars into Combine layers"
        from_stars = [(it.toolTip(), b.y()) for it, a, b in into_comp if on(a, stars, "out")]
        assert len(from_stars) == 2, f"the starless image and the real stars do not both reach Combine layers: {from_stars}"
        # the real stars and the synthetic ones end on the same port: the stars input
        y_synth = next(b.y() for _it, a, b in into_comp if on(a, synth, "out"))
        assert sum(abs(y - y_synth) < 1 for _t, y in from_stars) == 1, "the real stars do not enter the stars input"
        assert any(on(a, stars, "out") and on(b, synth, "in") for _it, path in wires(view) for a, b in [ends(path)])
        assert under(view) == []
    finally:
        done(w, m, page)
