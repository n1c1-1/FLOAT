"""Before any run nothing is "Up to date": not a Combine layers block's panel, not a Session block's tooltip (a beta
tester read "Up to date" on both before his first run, 3/3)."""
from __future__ import annotations

from PySide6.QtWidgets import QLabel

from tests.client.test_swap_smoke import env, made_from, qapp, settle, win  # noqa: F401  (fixtures by name)


def test_nothing_is_up_to_date_before_any_run(win, env):
    made_from(win, env, "sho_rgb")
    s, page = win.s, win.graph
    g = s.stored
    said = {}
    for b, blk in g.blocks.items():
        if blk.kind not in ("combine_layers", "session", "post"):
            continue
        page.bridge.choose(b)
        settle(win, 2)
        texts = [x.text() for x in page.panel.findChildren(QLabel) if x.isVisible()]
        tip = page.canvas.blocks[b].d.get("tip", "") if b in page.canvas.blocks else ""
        said[b] = (blk.kind, "Up to date" in texts, "Up to date" in tip)
    wrong = {b: v for b, v in said.items() if v[1] or v[2]}
    assert not wrong, wrong
    # a Session is never computed: its tooltip says no state at all (with its lights it said "Up to date")
    from astro_gpu.client.graphnode import STATE_WORD
    for b, blk in g.blocks.items():
        if blk.kind == "session" and b in page.canvas.blocks:
            tip = page.canvas.blocks[b].d.get("tip", "")
            assert not any(w in tip for w in STATE_WORD.values()), tip
    assert not env["errors"], env["errors"]
