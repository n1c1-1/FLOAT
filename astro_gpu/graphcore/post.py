"""The graph's wires to Post as a projection of Post (INTEGRATION 4.7, engine review 3).

Post's layers are Post's (``Project2.post``, its versions); the graph draws a wire to Post for each image of the SHARED
set every version may show: ``S = ∪ workflow_images(version) ∪ defaults`` (``defaults``: the images Post shows by
default, which the client's model knows: a workflow not taken out, not feeding a combination, giving Post an image).
A layer deleted in ONE version (its ``removed`` list) never removes the wire while another version shows it; a wire
removed in the graph takes the image out of every version (``remove_post_part`` on all of them), a wire added puts it
in every version (``add_post_layer``). Combine layers are drawn from the active version (their own rows,
``with_layer_rows``): one per Combine layers layer of a workflow, the one Post shows by default included.

An image is ``(workflow id, part)``: part None = the workflow's Output; "starless" / "stars" = its Stars/background's
two images; "combined" = a Combine layers layer (not a wire)."""
from __future__ import annotations

from . import model as _model
from .model import Index, Row, Wire, POST_PORT, SYNTH_PORT, clone, new_id

PORT_OF_PART = {None: "image", "starless": "Starless", "stars": "Stars only"}
#: a Synthetic stars' image sent to Post is the image of a workflow of its own (a tap ending at the block)
PART_OF_PORT = {"image": None, "Starless": "starless", "Stars only": "stars", SYNTH_PORT: None}


def shared_images(post: dict, defaults=()) -> set:
    """S: the images some version shows, and those Post shows by default."""
    from ..core.project2 import workflow_images
    out = set()
    for v in (post or {}).get("versions") or []:
        out |= {(int(a), b) for a, b in workflow_images(v) if a is not None}
    return out | {(int(a), b) for a, b in defaults}


def _flows(g):
    ix = Index(g)
    out = {}
    for f in list(g.rows) + list(g.taps):
        if f.wid is None:
            continue
        if hasattr(f, "blocks"):
            chain = ix.effective(f.id)
        elif f.block in ix.row_of:
            eff = ix.effective(ix.row_of[f.block])
            chain = eff[:eff.index(f.block) + 1]
        else:
            continue
        out[f.wid] = chain
    return out


def place_of(g, wid, part, flows=None):
    """(block, out port) whose wire to Post carries a workflow's image, None when the graph draws none."""
    chain = (flows or _flows(g)).get(wid)
    if not chain:
        return None
    kinds = [g.blocks[b].kind for b in chain]
    if part in ("starless", "stars"):
        if "stars" not in kinds:
            return None
        return chain[kinds.index("stars")], PORT_OF_PART[part]
    if part is None:
        if g.blocks[chain[-1]].kind == "synth" and any(t.wid == wid for t in g.taps):
            return chain[-1], SYNTH_PORT              # Synthetic stars sent to Post: its tap's image
        for b in reversed(chain):
            if "image" in _ports(g, b):
                return b, "image"
        return None
    return None


def _ports(g, b):
    from .model import out_ports
    return out_ports(g.blocks[b])


def wired_images(g, c=None) -> set:
    """The images the graph's wires to Post carry: {(workflow id, part)}."""
    flows = _flows(g)
    post = Index(g).post
    out = set()
    for w in g.wires:
        if w.dst[0] != post or w.src[0] not in g.blocks:
            continue
        part = PART_OF_PORT.get(w.src[1], "?")
        if part == "?":
            continue
        if w.wid is not None and w.wid in flows:
            out.add((w.wid, part))
            continue
        for wid in flows:
            if place_of(g, wid, part, flows) == (w.src[0], w.src[1]):
                out.add((wid, part))
                break
    return out


def with_post_wires(g, shared: set):
    """The graph whose wires to Post are those of the shared set (the others to Post go; Combine layers' kept). A wire
    already there for an image of the set stays as it is (its id, its place): an Undo gives the graph back exactly,
    the projection drawn again over it changes nothing."""
    q = clone(g)
    flows = _flows(q)
    ix = Index(q)
    post = ix.post
    rows = {r.wid for r in q.rows if r.wid is not None}
    want = {}                                          # (block, port) -> the wid its wire names (None: its own)
    for wid, part in sorted(shared, key=lambda x: (x[0], str(x[1]))):
        at = place_of(q, wid, part, flows)
        if at is None or at in want:
            continue
        own = wid in rows and flows[wid][-1] in (at[0], *[b for b in flows[wid][-1:] if q.blocks[b].kind == "stars"])
        want[at] = None if own else wid
    keep, have = [], set()
    for w in q.wires:
        if w.dst[0] != post or q.blocks[w.src[0]].kind == "combine_layers":
            keep.append(w)
        elif tuple(w.src) in want and tuple(w.src) not in have:
            have.add(tuple(w.src))
            w.wid = want[tuple(w.src)]
            keep.append(w)
    for at, wid in want.items():
        if at not in have and at not in {tuple(w.src) for w in keep if w.dst[0] == post}:
            keep.append(Wire(new_id(q, "w"), list(at), [post, POST_PORT], True, wid))
    q.wires = keep
    return q


def stars_side(g, wid, flows=None):
    """The Stars/background whose images a workflow's Combine layers reads (its starless image, its real stars; the
    stars of a Synthetic stars after it join them in the Stars side): its last one; None without one."""
    chain = (flows or _flows(g)).get(wid)
    for b in reversed(chain or []):
        if g.blocks[b].kind == "stars":
            return b
    return None


def layer_rows(g) -> dict:
    """{block: [the Combine layers rows whose Background reads its image]}, in the graph's order: a Stars block's
    starless image, or a combination's Output (issue #165: the S-H-O over its "· Stars" twin). A Combine layers whose
    Background is not wired is not one of them."""
    out = {}
    heads = {r.blocks[0]: r for r in g.rows if r.blocks and g.blocks[r.blocks[0]].kind == "combine_layers"}
    for w in g.wires:
        r = heads.get(w.dst[0])
        if r is None or w.dst[1] != "Background" or w.src[1] not in ("Starless", "image") or w.src[0] not in g.blocks:
            continue
        out.setdefault(w.src[0], []).append(r.id)
    return out


def with_layer_rows(g, want: dict, sides: dict | None = None):
    """The graph whose Combine layers are Post's (INTEGRATION 4.7, drawn from the shown version): ``want`` {block: the
    count of Combine layers layers it is the background of}; a block holding fewer gets new ones, one holding more loses
    its last ones; the others are left as they are. A Stars block's new one reads its two images; another block's
    (a combination's Output) reads its image and, on its Stars side, ``sides[block]`` ([block, port] or None)."""
    sides = sides or {}
    from .model import new_block
    from .msg import name_of
    from .operations import free_name
    from .rules import LABEL
    g = _with_stars_group(g)
    have = layer_rows(g)
    if all(len(have.get(s, [])) == n for s, n in want.items()) and \
            all(s in want for s in have if s in g.blocks and g.blocks[s].kind == "stars"):
        return _to_post(_with_sides(g, sides))
    q = clone(g)
    post = Index(q).post
    for s, rows in have.items():
        extra = len(rows) - want.get(s, 0)
        if extra <= 0 or s not in q.blocks or (q.blocks[s].kind != "stars" and s not in want):
            continue
        for rid in rows[-extra:]:
            r = next(x for x in q.rows if x.id == rid)
            gone = set(r.blocks)
            q.rows.remove(r)
            q.wires = [w for w in q.wires if w.src[0] not in gone and w.dst[0] not in gone]
            for b in gone:
                q.blocks.pop(b, None)
    for s, n in want.items():
        if s not in q.blocks:
            continue
        for _i in range(n - len(have.get(s, []))):
            j = new_block(q, "combine_layers")
            q.rows.append(Row(new_id(q, "r"), free_name(q, name_of(LABEL["combine_layers"])), [j.id]))
            if q.blocks[s].kind == "stars":
                q.wires.append(Wire(new_id(q, "w"), [s, "Starless"], [j.id, "Background"]))
                q.wires.append(Wire(new_id(q, "w"), [s, "Stars only"], [j.id, "Stars"]))
            else:
                q.wires.append(Wire(new_id(q, "w"), [s, "image"], [j.id, "Background"]))
            q.wires.append(Wire(new_id(q, "w"), [j.id, "image"], [post, POST_PORT]))
    return _to_post(_with_sides(_with_stars_group(q), sides))


def _to_post(g):
    """Each Combine layers drawn for a layer of Post has its wire to Post (the layer it stands for): one made by a
    gesture after a Stars/background whose image did not go to Post, then saved, had none while Post held its layer."""
    post = Index(g).post
    heads = [b for rids in layer_rows(g).values() for rid in rids for r in g.rows if r.id == rid for b in r.blocks[:1]]
    missing = [j for j in heads if not any(w.src[0] == j and w.dst[0] == post for w in g.wires)]
    if post is None or not missing:
        return g
    q = clone(g)
    for j in missing:
        q.wires.append(Wire(new_id(q, "w"), [j, "image"], [post, POST_PORT]))
    return q


def _with_sides(g, sides):
    """The Stars side of each Combine layers wired from the image its layer names (over a combination's Output: the
    "· Stars" combination's Output, or the stars another workflow gives; over a Stars/background's starless image: its
    own stars, or another workflow's): Post's layer is what it shows, so a wire
    from another image into that side is replaced (the panel's choice of the stars "did not stick" while the port
    was wired)."""
    if not sides:
        return g
    have = layer_rows(g)
    add, drop = [], set()
    for s, src in sides.items():
        if src is None or src[0] not in g.blocks:
            continue
        for rid in have.get(s, []):
            j = next(r for r in g.rows if r.id == rid).blocks[0]
            into = [w for w in g.wires if w.dst == [j, "Stars"]]
            if any(w.src == list(src) and w.active for w in into):
                drop |= {w.id for w in into if w.active and w.src != list(src)}     # never two stars into one side
                continue
            drop |= {w.id for w in into if w.active}
            if not any(w.src == list(src) for w in into):
                add.append((list(src), j))
            else:                                       # drawn, not taken: taken now
                drop |= {w.id for w in into if w.src == list(src)}
                add.append((list(src), j))
    if not add and not drop:
        return g
    q = clone(g)
    q.wires = [w for w in q.wires if w.id not in drop]
    for src, j in add:
        q.wires.append(Wire(new_id(q, "w"), src, [j, "Stars"]))
    return q


def _synth_after(g, ix, s):
    """The Synthetic stars right after Stars/background ``s`` in its row, None."""
    r = ix.row_of.get(s)
    blocks = ix.row[r].blocks if r is not None else []
    i = blocks.index(s) if s in blocks else -1
    nxt = blocks[i + 1] if 0 <= i < len(blocks) - 1 else None
    return nxt if nxt is not None and g.blocks[nxt].kind == "synth" else None


def _with_stars_group(g):
    """Combine layers is ONE layer of the background and the stars, real and synthetic (Post's layer and its Stars
    group): its Background reads the Stars/background's starless image, its Stars the real stars and, with a Synthetic
    stars after that Stars/background (switched on), its Synthetic stars input the synthetic stars (Post's synthetic
    stars row). A wire never twice; the synthetic stars' wire goes with the block (or when it is switched off)."""
    ix = Index(g)
    heads = {r.blocks[0] for r in g.rows if r.blocks and g.blocks[r.blocks[0]].kind == "combine_layers"}
    if not heads:
        return g
    want = set()                                      # (src block, src port, Combine layers, its port)
    # a Combine layers whose Stars side reads the stars of another workflow (another Stars/background's, a
    # combination's Output: Post's layer names it) keeps them: its own block's stars are not drawn back beside them
    fed = {w.dst[0] for w in g.wires if w.dst[0] in heads and w.dst[1] == "Stars" and w.active
           and w.src[0] in g.blocks and g.blocks[w.src[0]].kind != "synth"}
    for w in g.wires:
        if w.dst[0] in heads and w.dst[1] == "Background" and w.src[1] == "Starless" and w.src[0] in g.blocks \
                and g.blocks[w.src[0]].kind == "stars":
            if w.dst[0] not in fed or any(x.dst == [w.dst[0], "Stars"] and x.src == [w.src[0], "Stars only"]
                                                and x.active for x in g.wires):
                want.add((w.src[0], "Stars only", w.dst[0], "Stars"))
            y = _synth_after(g, ix, w.src[0])
            if y is not None and not g.blocks[y].off:     # switched off: Post has no synthetic stars row
                want.add((y, SYNTH_PORT, w.dst[0], SYNTH_PORT))
    if _model.SYNTH_ON_STARS_SIDE:                    # the stars of another workflow (a "· Stars" combination, a star
        for w in g.wires:                             # row): its Synthetic stars after the block that gives them
            if w.dst[0] in heads and w.dst[1] == "Stars" and w.active and w.src[0] in g.blocks \
                    and g.blocks[w.src[0]].kind != "synth":
                y = _synth_after(g, ix, w.src[0])
                if y is not None and not g.blocks[y].off:
                    want.add((y, SYNTH_PORT, w.dst[0], SYNTH_PORT))
    drop, seen = [], set()
    for w in g.wires:
        if w.dst[0] not in heads:
            continue
        k = (*w.src, *w.dst)
        stale = w.src[0] in g.blocks and g.blocks[w.src[0]].kind == "synth" and k not in want
        if k in seen or stale:
            drop.append(w.id)
        seen.add(k)
    have = {(*w.src, *w.dst) for w in g.wires}
    add = sorted(k for k in want if k not in have)
    if not add and not drop:
        return g
    q = clone(g)
    q.wires = [w for w in q.wires if w.id not in drop]
    for s, sp, j, jp in add:
        q.wires.append(Wire(new_id(q, "w"), [s, sp], [j, jp]))
    return q


def post_actions(before, after, c=None) -> list[tuple]:
    """What Post does for a graph gesture that changed the wires to Post: ("add_post_layer", wid, part) for a wire
    added (it reaches every version), ("remove_post_part", wid, part) for one removed: run with EACH version active
    in turn (a version that had already deleted the layer itself does not pass the removal on to the others)."""
    a, b = wired_images(before), wired_images(after)
    return [("remove_post_part", wid, part) for wid, part in sorted(a - b, key=str)] + \
        [("add_post_layer", wid, part) for wid, part in sorted(b - a, key=str)]
