"""A Combine layers added from the palette, then wired by hand: the stars of one channel into its Stars input, the
combination's Output into its Background (the tutorials' film: the block vanished at the second wire and Post went
back to its layers before). Post's layer says both sides, so the block the graph draws from Post again stays."""
from __future__ import annotations

from astro_gpu.graphcore.post import place_of
from tests.client.test_swap_smoke import env, made_from, qapp, ready, settle, win  # noqa: F401  (fixtures by name)


def _layers(s):
    return [x for x in s.post_layers() if x.get("kind") == "workflow" and x.get("part") == "combined"]


def _ok(res):
    assert res is None or getattr(res, "ok", True), getattr(res, "reason", res)


def test_a_combine_layers_wired_to_the_combinations_output_stays(win, env):
    root = made_from(win, env, "sho_rgb")
    s, page = win.s, win.graph
    first = _layers(s)[0]                                              # the template's: the S-H-O with the RGB stars
    first_cl = next(b for b, x in s.stored.blocks.items() if x.kind == "combine_layers")
    first_stars = [w.src for w in s.stored.wires if w.dst == [first_cl, "Stars"] and w.active]
    had = {x["id"] for x in _layers(s)}
    blocks = set(s.stored.blocks)
    _ok(page.bridge.act("add_block", kind="combine_layers"))
    settle(win, 3)
    cl = next(b for b in s.stored.blocks if b not in blocks and s.stored.blocks[b].kind == "combine_layers")
    new = next(x for x in _layers(s) if x["id"] not in had)             # the layer the new block is
    assert s.block_map.node(cl).lid == new["id"]
    sho = next(q for q in s.paths if s.stored.blocks[place_of(s.stored, q.pid, None)[0]].kind == "combine")
    oiii = next(q for q in s.paths if q.name == "OIII")
    assert new["pid"] not in (sho.pid, oiii.pid)                       # it came on the free Stars/background (RGB's)
    stars = place_of(s.stored, oiii.pid, "stars")
    _ok(page.bridge.act("wire", src=stars[0], src_port="Stars only", dst=cl, dst_port="Stars"))
    settle(win, 3)
    assert s.post_layer(new["id"])["stars_pid"] == oiii.pid
    out = place_of(s.stored, sho.pid, None)
    _ok(page.bridge.act("wire", src=out[0], src_port="image", dst=cl, dst_port="Background"))
    settle(win, 3)

    def both():
        g = s.stored
        return cl in g.blocks and \
            [w.src for w in g.wires if w.dst == [cl, "Background"] and w.active] == [[out[0], "image"]] and \
            [w.src for w in g.wires if w.dst == [cl, "Stars"] and w.active] == [list(stars)] and \
            [w.src for w in g.wires if w.dst == [first_cl, "Stars"] and w.active] == first_stars and \
            sum(x.kind == "combine_layers" for x in g.blocks.values()) == 2 and \
            s.block_map.node(cl).lid == new["id"] and s.block_map.node(first_cl).lid == first["id"]
    assert both()
    lay = s.post_layer(new["id"])                                      # the same layer of Post, its two sides said
    assert lay is not None and (lay["pid"], lay.get("starless")) == (sho.pid, "output")
    assert lay["stars_pid"] == oiii.pid and lay.get("stars_image", "stars") == "stars"
    assert len(_layers(s)) == len(had) + 1
    assert s.post_layer(first["id"])["stars_pid"] == first["stars_pid"]
    # another gesture, then the project saved and opened again: the block is still there with its two inputs
    page.bridge.act("rename_row", row=next(r.id for r in s.stored.rows if r.wid == oiii.pid), name="Renamed")
    settle(win, 3)
    assert both()
    win.save()
    settle(win)
    assert win.close_project(go_home=False)
    assert win.open_project(root)
    ready(win)
    s = win.s
    assert both()
    assert s.post_layer(new["id"])["pid"] == sho.pid
    assert not env["errors"], env["errors"]
