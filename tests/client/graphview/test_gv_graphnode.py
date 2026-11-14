"""graphnode.py: the block table and block answers of today's page (blocks.py and graph.py, gone with the swap to the
new page) keep today's colours; BlockMap gives each block of a compiled graph its Node and its address."""
from __future__ import annotations

import inspect
from types import SimpleNamespace

from astro_gpu.client import graphnode
from astro_gpu.client.graphview import style
from astro_gpu.client.widgets import ACCENT, BAD, OK, WARN


def test_the_page_colours_are_those_of_today():
    """The values today's page drew with (graph.py, before the swap)."""
    assert (style.CUT_COL, style.GREY_COL, style.BROKEN_COL) == ("#6b7280", "#c9a227", "#e05252")
    assert style.STATE_COL == {"done": OK, "pending": "#5d6470", "stale": WARN, "running": ACCENT, "failed": BAD,
                               "interrupted": WARN, "skipped": "#5d6470", "fallback": WARN}
    assert style.IMAGE_COL["starless"] == "#7aa2d6"
    assert style.IMAGE_COL["stars_only"] == "#e0c45c"
    assert style.PALETTE_W == 140


def test_graphnode_imports_neither_the_model_nor_the_graph_page():
    src = inspect.getsource(graphnode)
    head = src[:src.index("COLUMNS = ")]
    assert "from .model" not in head and "from .graph " not in head and "import graph" not in head


def test_blockmap_gives_each_block_its_node_and_address():
    from astro_gpu.graphcore.compile import Compiled
    from astro_gpu.graphcore.history import History
    h = History()
    h.apply("add_row", kind="session")                          # the Session comes alone: its blocks placed after
    rid = h.project.rows[0].id
    h.apply("add_block", kind="stacking", row=rid)
    h.apply("add_block", kind="gradient", row=rid)
    g = h.project
    ids = {x.kind: b for b, x in g.blocks.items()}
    c = Compiled(map={ids["session"]: [(1, "session")], ids["stacking"]: [(1, "stacking")],
                      ids["gradient"]: [(1, "gradient~2")]})
    p1 = SimpleNamespace(pid=1)
    bm = graphnode.BlockMap(c, SimpleNamespace(paths=[p1]), g)
    assert bm.node(ids["stacking"]).key == "empilement"
    assert bm.node(ids["gradient"]).key == "fond~2"
    assert bm.node(ids["session"]).sid == g.blocks[ids["session"]].sid
    assert bm.node(ids["stacking"]).paths == [p1]
    assert bm.address(ids["gradient"]) == (1, "fond~2")
    assert bm.address(ids["post"]) is None
    assert bm.node_of(p1, "empilement") is bm.node(ids["stacking"])
    assert bm.block_of(bm.node(ids["gradient"])) == ids["gradient"]
    assert bm.state(bm.node(ids["stacking"])) == "pending"        # a model without run answers: not computed
    assert bm.summary(bm.node(ids["gradient"]))


def test_blockmap_reads_the_run_answers_of_the_model():
    from astro_gpu.graphcore.compile import Compiled
    p1, p2 = SimpleNamespace(pid=1), SimpleNamespace(pid=2)
    states = {(1, "stack"): "done", (2, "stack"): "stale"}
    model = SimpleNamespace(paths=[p1, p2], all_stages=lambda p: ["stack"], state=lambda p, st: states[(p.pid, st)])
    bm = graphnode.BlockMap(Compiled(map={"b5": [(1, "stacking"), (2, "stacking")]}), model)
    assert bm.state(bm.node("b5")) == "stale"                   # the worst over its workflows
