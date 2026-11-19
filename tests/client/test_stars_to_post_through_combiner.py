"""A Stars/starless image that already reaches Post through a Combine layers block is never added a second time by the
panel's "To Post": the button is greyed, its reason on hover (a beta tester's M27: Post listed the stars, the starless
image and their combination, the Graph said 3 layers). Clicking one button leaves the other where it is."""
from __future__ import annotations

from PySide6.QtWidgets import QPushButton

from tests.client.test_swap_smoke import env, made_from, qapp, settle, win  # noqa: F401  (fixtures by name)
from tests.client.test_layer_stars_wire_kept import _block_of_layer, _layers, _stars_of


def _buttons(page):
    return {w.objectName(): w for w in page.panel.findChildren(QPushButton)
            if w.objectName().startswith("part_to_post_") and w.isVisible()}


def test_an_output_already_in_post_through_a_combiner_is_greyed_and_buttons_stay(win, env):
    made_from(win, env, "sho_rgb")
    s, page = win.s, win.graph
    over = next(x for x in _layers(s) if x.get("starless") == "output")
    cl = _block_of_layer(s, over["id"])
    other = next(q for q in s.paths if q.pid != over["pid"] and _stars_of(s, q.pid) is not None
                 and q.pid != over.get("stars_pid"))
    src = _stars_of(s, other.pid)
    page.bridge.act("wire", src=src[0], src_port="Stars only", dst=cl, dst_port="Stars")
    settle(win, 3)
    page.bridge.choose(src[0])
    settle(win, 3)
    btn = _buttons(page)
    stars, starless = btn["part_to_post_stars"], btn["part_to_post_starless"]
    assert not stars.isEnabled() and "Already in Post through" in stars.toolTip()
    assert starless.isEnabled()
    n_layers = len(s.post_layers())
    geo = stars.geometry().width(), starless.geometry().width()
    starless.click()                                     # the other image is the user's explicit choice
    settle(win, 3)
    page.bridge.choose(src[0])
    settle(win, 3)
    btn = _buttons(page)
    assert len(s.post_layers()) == n_layers + 1
    assert not btn["part_to_post_stars"].isEnabled()
    assert (btn["part_to_post_stars"].width(), btn["part_to_post_starless"].width()) == geo
    assert not env["errors"], env["errors"]
