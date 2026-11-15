"""A wire drawn into the Stars input of the S-H-O's Combine layers from another workflow's stars (a beta tester's RGB
stars) is kept: Post's layer says it (the panel names it), the next gestures and a reopening keep it. A Stars input
Post cannot take is refused at the gesture with its reason, never dropped later."""
from __future__ import annotations

from astro_gpu.graphcore.model import model_json
from tests.client.test_swap_smoke import env, made_from, qapp, ready, settle, win  # noqa: F401  (fixtures by name)


def _layers(s):
    return [x for x in s.post_layers() if x.get("kind") == "workflow" and x.get("part") == "combined"]


def _block_of_layer(s, lid):
    g = s.stored
    return next(b for b, x in g.blocks.items() if x.kind == "combine_layers" and s.block_map.node(b) is not None
                and s.block_map.node(b).lid == lid)


def _stars_of(s, pid):
    """The Stars/background block whose stars a workflow gives (its own row's)."""
    from astro_gpu.graphcore.post import place_of
    return place_of(s.stored, pid, "stars")


def test_a_stars_wire_into_the_combinations_combine_layers_is_kept(win, env):
    root = made_from(win, env, "sho_rgb")
    s, page = win.s, win.graph
    over = next(x for x in _layers(s) if x.get("starless") == "output")      # the S-H-O's Combine layers
    cl = _block_of_layer(s, over["id"])
    other = next(q for q in s.paths if q.pid != over["pid"] and _stars_of(s, q.pid) is not None
                 and q.pid != over.get("stars_pid"))
    src = _stars_of(s, other.pid)
    res = page.bridge.act("wire", src=src[0], src_port="Stars only", dst=cl, dst_port="Stars")
    settle(win, 3)
    assert res is None or getattr(res, "ok", True), res

    def wired():
        g = s.stored
        return any(w.src == [src[0], "Stars only"] and w.dst == [cl, "Stars"] and w.active for w in g.wires)
    assert wired()
    lay = s.post_layer(over["id"])
    assert lay["stars_pid"] == other.pid and lay.get("stars_image", "stars") == "stars"
    assert s.block_map.node(cl).lid == over["id"]                # the panel shows that layer, its Stars side named
    # another gesture: the Combine layers are drawn again from Post, the wire with them
    q = s.paths[0]
    page.bridge.act("rename_row", row=next(r.id for r in s.stored.rows if r.wid == q.pid), name="Renamed")
    settle(win, 3)
    assert wired()
    win.save()
    settle(win)
    stored = model_json(s.stored)
    assert win.close_project(go_home=False)
    assert win.open_project(root)
    ready(win)
    s = win.s
    assert model_json(s.stored) == stored
    assert s.post_layer(over["id"])["stars_pid"] == other.pid
    # the panel's list picks the stars of another workflow: the graph's wire follows it (it "did not stick")
    third = next(q for q in s.paths if q.pid not in (over["pid"], other.pid) and _stars_of(s, q.pid) is not None)
    assert s.set_layer_side(over["id"], "stars", third.pid, "stars")
    settle(win, 3)
    src3 = _stars_of(s, third.pid)
    ins = [w.src for w in s.stored.wires if w.dst == [cl, "Stars"] and w.active]
    assert ins == [list(src3)], ins
    assert s.block_map.node(cl).lid == over["id"]
    assert not env["errors"], env["errors"]


def test_a_stars_wire_post_cannot_take_is_refused_at_the_gesture(win, env):
    made_from(win, env, "sho_rgb")
    s, page = win.s, win.graph
    over = next(x for x in _layers(s) if x.get("starless") == "output")      # the S-H-O's Combine layers
    cl = _block_of_layer(s, over["id"])
    g = s.stored
    q = next(q for q in s.paths if _stars_of(s, q.pid) is not None)
    row = next(r for r in g.rows if r.wid == q.pid)
    mid = next(b for b in row.blocks if g.blocks[b].kind == "denoise")     # an image no workflow ends with
    before = model_json(g)
    # while the wire is pulled the port is not lit: the page's check says the gesture's reason (the window fuzz: a lit
    # port, then the gesture refused)
    why = page.bridge.check_wire(mid, "image", cl, "Stars")
    assert why, why
    res = page.bridge.act("wire", src=mid, src_port="image", dst=cl, dst_port="Stars")
    settle(win, 3)
    assert model_json(s.stored) == before
    assert res is not None and not res.ok and res.reason, res
    assert s.post_layer(over["id"]).get("stars_pid") == over.get("stars_pid")
