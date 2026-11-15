"""The panel of a Stars/background block says where the graph's wires take its two images: a filter's starless image
wired into the S-H-O is not "unused" (a beta tester read "unused: drag to Post or Composition" on a wired block, 3/3)."""
from __future__ import annotations

from PySide6.QtWidgets import QLabel

from tests.client.test_swap_smoke import env, made_from, qapp, settle, win  # noqa: F401  (fixtures by name)


def test_a_stars_block_wired_into_the_combination_is_not_unused(win, env):
    made_from(win, env, "sho_rgb")
    s, page = win.s, win.graph
    g = s.stored
    stars = [b for b, x in g.blocks.items() if x.kind == "stars"]
    into_combo = [b for b in stars if any(w.src[0] == b and g.blocks[w.dst[0]].kind == "combine" for w in g.wires)]
    assert into_combo
    for b in into_combo:
        page.bridge.choose(b)
        settle(win, 3)
        texts = " ".join(x.text() for x in page.panel.findChildren(QLabel) if x.isVisible())
        wired = {str(w.src[1]) for w in g.wires if w.src[0] == b and w.active}
        line = {"Starless": "Starless image", "Stars only": "Stars alone"}
        for port, name in line.items():
            said = texts.split(f"<b>{name}</b>", 1)[1].split("<b>", 1)[0]
            assert ("unused" in said) == (port not in wired), (name, said)
        assert "Starless image</b><br>&nbsp;&nbsp;&nbsp;→ S-H-O" in texts, texts
    assert not env["errors"], env["errors"]
