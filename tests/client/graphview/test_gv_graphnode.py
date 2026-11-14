"""graphnode.py: the copies of today's block table and block answers stay equal to the originals (blocks.py, graph.py)
until the swap; BlockMap gives each block of a compiled graph its Node and its address."""
from __future__ import annotations

import inspect
from types import SimpleNamespace

from astro_gpu.client import blocks, graph, graphnode
from astro_gpu.client.graphview import style

CONSTANTS = ("COLUMNS", "OPTIONAL", "PHASES", "TOOL_COLUMN", "TITLES", "PORT_NAMES", "TOOLBOX", "WARN_MARK",
             "STAR_OUTPUTS", "ENGINE_BLOCK", "KEY_ENGINE_BLOCK", "SIDE_INPUTS", "TRUNK", "IMAGE_KEYS", "FRAME_KEYS",
             "STATE_WORD", "FALLBACK_WORD", "STATE_ORDER", "BREAK_COL", "PAUSE_COL", "REVIEWED_COL", "REVIEW_ORDER",
             "STEPS", "BY_KEY", "TOOLS")
FUNCTIONS = ("combine_kind", "combine_title", "port_name", "kbase", "knum", "block_title", "calibration_counts",
             "gone_kinds", "own_calibration_summary", "session_summary", "node_title", "review_title", "_safe_chain",
             "chain", "stages_of", "run_stages", "block_state", "block_fallback", "block_seconds", "fmt_int",
             "drizzle_auto", "method_summary")


def test_the_constants_are_those_of_blocks():
    for name in CONSTANTS:
        assert getattr(graphnode, name) == getattr(blocks, name), name


def test_the_functions_are_those_of_blocks():
    for name in FUNCTIONS:
        assert inspect.getsource(getattr(graphnode, name)) == inspect.getsource(getattr(blocks, name)), name
    a = {f.name: (f.type, f.default) for f in graphnode.Node.__dataclass_fields__.values()}
    b = {f.name: (f.type, f.default) for f in blocks.Node.__dataclass_fields__.values()}
    assert a == b


def test_the_page_colours_are_those_of_today():
    assert (style.CUT_COL, style.GREY_COL, style.BROKEN_COL) == (graph.CUT_COL, graph.GREY_COL, graph.BROKEN_COL)
    assert style.STATE_COL == graph.STATE_COL
    assert style.IMAGE_COL["starless"] == graph.IMAGE_COL["starless"]
    assert style.IMAGE_COL["stars_only"] == graph.IMAGE_COL["stars"]
    assert style.PALETTE_W == graph.PALETTE_W


def test_graphnode_imports_neither_the_model_nor_the_graph_page():
    src = inspect.getsource(graphnode)
    head = src[:src.index("COLUMNS = ")]
    assert "from .model" not in head and "from .graph " not in head and "import graph" not in head


def test_blockmap_gives_each_block_its_node_and_address():
    from astro_gpu.graphcore.compile import Compiled
    from astro_gpu.graphcore.history import History
    h = History()
    h.apply("add_row", kind="session")
    rid = h.project.rows[0].id
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
