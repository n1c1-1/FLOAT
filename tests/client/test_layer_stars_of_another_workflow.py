"""A Combine layers over a Stars/background's starless image takes the stars of another workflow (a beta tester: the
S-H-O's Combine layers refused the stars of an RGB workflow, « puts back the stars of its own Stars/background », and
picking them in its Stars list silently went back to its own). Post's layer and the graph's wire say the same, by the
wire and by the list, and they stay through the next gesture."""
from __future__ import annotations

from tests.client.test_layer_stars_wire_kept import _block_of_layer, _layers, _stars_of
from tests.client.test_swap_smoke import env, made_from, qapp, ready, settle, win  # noqa: F401  (fixtures by name)


def _stars_in(s, cl):
    return [w.src for w in s.stored.wires if w.dst == [cl, "Stars"] and w.active]


def test_a_starless_combine_layers_takes_another_workflows_stars(win, env):
    made_from(win, env, "sho_rgb")
    s, page = win.s, win.graph
    sii = next(q for q in s.paths if q.name == "SII")                  # the Combine layers of the SII workflow
    s.add_post_layer(sii.pid, "combined", None)
    settle(win, 3)
    over = next(x for x in _layers(s) if x.get("starless", "starless") != "output" and x.get("pid") is not None
                and _stars_of(s, x["pid"]) is not None)
    cl = _block_of_layer(s, over["id"])
    other = next(q for q in s.paths if q.pid != over["pid"] and _stars_of(s, q.pid) is not None)
    src = _stars_of(s, other.pid)
    res = page.bridge.act("wire", src=src[0], src_port="Stars only", dst=cl, dst_port="Stars")
    settle(win, 3)
    assert res is not None and res.ok, res.reason
    assert s.post_layer(over["id"])["stars_pid"] == other.pid
    assert _stars_in(s, cl) == [list(src)]
    page.bridge.act("rename_row", row=next(r.id for r in s.stored.rows if r.wid == s.paths[0].pid), name="Renamed")
    settle(win, 3)
    assert _stars_in(s, cl) == [list(src)] and s.post_layer(over["id"])["stars_pid"] == other.pid
    # its own stars again, written to Post's layer: the graph follows
    assert s.set_layer_side(over["id"], "stars", over["pid"], "stars")
    settle(win, 3)
    assert _stars_in(s, cl) == [list(_stars_of(s, over["pid"]))]
    assert s.set_layer_side(over["id"], "stars", other.pid, "stars")          # and another's
    settle(win, 3)
    assert _stars_in(s, cl) == [list(src)] and s.post_layer(over["id"])["stars_pid"] == other.pid
    assert not env["errors"], env["errors"]
