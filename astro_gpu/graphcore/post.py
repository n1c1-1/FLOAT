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

from .model import Index, Row, Wire, POST_PORT, clone, new_id

PART_OF_PORT = {"image": None, "Starless": "starless", "Stars only": "stars"}
PORT_OF_PART = {v: k for k, v in PART_OF_PORT.items()}


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
    """The graph whose wires to Post are those of the shared set (the others to Post go; Combine layers' kept)."""
    q = clone(g)
    flows = _flows(q)
    ix = Index(q)
    post = ix.post
    keep = [w for w in q.wires if w.dst[0] != post or q.blocks[w.src[0]].kind == "combine_layers"]
    rows = {r.wid for r in q.rows if r.wid is not None}
    have = {(tuple(w.src)) for w in keep if w.dst[0] == post}
    for wid, part in sorted(shared, key=lambda x: (x[0], str(x[1]))):
        at = place_of(q, wid, part, flows)
        if at is None or at in have:
            continue
        have.add(at)
        own = wid in rows and flows[wid][-1] in (at[0], *[b for b in flows[wid][-1:] if q.blocks[b].kind == "stars"])
        keep.append(Wire(new_id(q, "w"), list(at), [post, POST_PORT], True, None if own else wid))
    q.wires = keep
    return q


def stars_side(g, wid, flows=None):
    """The block whose two images (starless, stars alone) a workflow's Combine layers reads: its last Stars/background
    or Synthetic stars; None without one."""
    chain = (flows or _flows(g)).get(wid)
    for b in reversed(chain or []):
        if g.blocks[b].kind in ("stars", "synth"):
            return b
    return None


def layer_rows(g) -> dict:
    """{Stars block: [the Combine layers rows whose Background reads its starless image]}, in the graph's order. A
    Combine layers whose Background is not wired, or wired from another image, is not one of them."""
    out = {}
    heads = {r.blocks[0]: r for r in g.rows if r.blocks and g.blocks[r.blocks[0]].kind == "combine_layers"}
    for w in g.wires:
        r = heads.get(w.dst[0])
        if r is None or w.dst[1] != "Background" or w.src[1] != "Starless" or w.src[0] not in g.blocks:
            continue
        out.setdefault(w.src[0], []).append(r.id)
    return out


def with_layer_rows(g, want: dict):
    """The graph whose Combine layers are Post's (INTEGRATION 4.7, drawn from the shown version): ``want`` {Stars
    block: the count of Combine layers layers of its workflows}; a Stars block holding fewer gets new ones (both sides
    wired from it, its image to Post), one holding more loses its last ones. The others are left as they are."""
    from .model import new_block
    from .msg import name_of
    from .operations import free_name
    from .rules import LABEL
    have = layer_rows(g)
    if all(len(have.get(s, [])) == n for s, n in want.items()) and \
            all(s in want for s in have if s in g.blocks and g.blocks[s].kind in ("stars", "synth")):
        return g
    q = clone(g)
    post = Index(q).post
    for s, rows in have.items():
        extra = len(rows) - want.get(s, 0)
        if extra <= 0 or s not in q.blocks or q.blocks[s].kind not in ("stars", "synth"):
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
            q.wires.append(Wire(new_id(q, "w"), [s, "Starless"], [j.id, "Background"]))
            q.wires.append(Wire(new_id(q, "w"), [s, "Stars only"], [j.id, "Stars"]))
            q.wires.append(Wire(new_id(q, "w"), [j.id, "image"], [post, POST_PORT]))
    return q


def post_actions(before, after, c=None) -> list[tuple]:
    """What Post does for a graph gesture that changed the wires to Post: ("add_post_layer", wid, part) for a wire
    added (it reaches every version), ("remove_post_part", wid, part) for one removed: run with EACH version active
    in turn (a version that had already deleted the layer itself does not pass the removal on to the others)."""
    a, b = wired_images(before), wired_images(after)
    return [("remove_post_part", wid, part) for wid, part in sorted(a - b, key=str)] + \
        [("add_post_layer", wid, part) for wid, part in sorted(b - a, key=str)]
