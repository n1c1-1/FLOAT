"""I1–I6 of CONCEPTION §3 as executable property checks. Each returns a list of problems ("" lists = fine)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from astro_gpu.graphcore.layout import layout  # noqa: E402  (uncached: I6 must recompute)
from astro_gpu.graphcore.model import Index, JOIN_KINDS  # noqa: E402
from astro_gpu.graphcore.rules import title  # noqa: E402

EPS = 1e-6
INSERT = {"add_block", "add_night", "add_row", "add_variant", "duplicate_row", "split_night", "split_by"}
REMOVE = {"remove_block", "delete_block", "delete_row", "remove_night", "delete_night", "merge_nights"}
FOLD = {"fold_all", "unfold_all"}     # may move columns right of the band, never the order of rows
STILL = {"wire", "unwire", "send_to_post", "take_alternative", "set_setting", "apply_to_rows", "switch_off",
         "set_pause", "rename_row", "rename_night", "add_files", "remove_files", "use_for_all_nights",
         "set_common_files", "set_common_setting", "reset_own", "reset_own_files", "move_files",
         "set_combination_type", "set_mode", "select"}


def all_boxes(d):
    out = dict(d.boxes)
    out.update({"card:" + r: b for r, b in d.cards.items()})
    return out


def i1(d):
    bs = sorted(all_boxes(d).items(), key=lambda kv: kv[1].x)
    bad = []
    for i, (a, A) in enumerate(bs):
        for b, B in bs[i + 1:]:
            if B.x >= A.right - EPS:
                break
            if A.y < B.bottom - EPS and B.y < A.bottom - EPS:
                bad.append(f"I1 overlap {a} {b}")
    return bad


def _owner_boxes(p, d, b, ix):
    keys = {b}
    r = ix.row_of.get(b)
    if r and r in d.cards:
        keys.add("card:" + r)
    return keys


def i2(p, d):
    ix = Index(p)
    cell = 64.0
    buckets = {}
    for k, B in all_boxes(d).items():
        r = (B.x, B.right, B.y, B.bottom)
        for c in range(int(B.x // cell), int(B.right // cell) + 1):
            buckets.setdefault(c, []).append((k, r))
    bad = []
    for w in d.wires:
        skip = _owner_boxes(p, d, w.src, ix) | _owner_boxes(p, d, w.dst, ix)
        for (x1, y1), (x2, y2) in zip(w.points, w.points[1:]):
            if abs(x1 - x2) > EPS and abs(y1 - y2) > EPS:
                bad.append(f"I2 diagonal segment in {w.id}")
                continue
            lo_x, hi_x, lo_y, hi_y = min(x1, x2), max(x1, x2), min(y1, y2), max(y1, y2)
            seen = set()
            for c in range(int(lo_x // cell), int(hi_x // cell) + 1):
                for k, (bx, br, by, bb) in buckets.get(c, ()):
                    if k in skip or k in seen:
                        continue
                    seen.add(k)
                    # an axis-aligned segment crosses a box's interior
                    if lo_x < br - EPS and hi_x > bx + EPS and lo_y < bb - EPS and hi_y > by + EPS:
                        bad.append(f"I2 wire {w.id} crosses {k}")
                    elif hi_y - lo_y <= EPS and by + EPS < y1 < bb - EPS and lo_x < br - EPS and hi_x > bx + EPS:
                        bad.append(f"I2 wire {w.id} crosses {k}")
                    elif hi_x - lo_x <= EPS and bx + EPS < x1 < br - EPS and lo_y < bb - EPS and hi_y > by + EPS:
                        bad.append(f"I2 wire {w.id} crosses {k}")
    return bad


def i3(p, d):
    ix = Index(p)
    bad = []
    for w in d.wires:
        xs = [pt[0] for pt in w.points]
        if any(b < a - EPS for a, b in zip(xs, xs[1:])):
            bad.append(f"I3 leftward segment in {w.id}")
    for w in p.wires:
        cs = d.cells.get(w.src[0]) or d.cells.get("card:" + ix.row_of.get(w.src[0], ""))
        cd = d.cells.get(w.dst[0])
        sc = 5 if (w.src[0] not in d.cells) else cs[1]
        if cd is None or sc >= cd[1]:
            bad.append(f"I3 wire {w.id} not rightwards")
    return bad


def i4(before, after, op, acted):
    """Locality: before/after Drawings, op name, Result.acted = (row, line, col) (line: after for insertions,
    before for removals)."""
    if op in FOLD:
        rows_b = [o for o, _y in before.lines if not o.startswith("b")]
        rows_a = [o for o, _y in after.lines if not o.startswith("b")]
        return [] if rows_b == rows_a else [f"I4 {op} changed the order of rows"]
    if op not in INSERT | REMOVE | STILL:
        return []
    bad = []
    B, A = all_boxes(before), all_boxes(after)
    src = [k for k in B if k in A and before.cells.get(k, (None, 0))[0] is not None
           and before.cells[k][1] < 100]
    if op in STILL:
        for k in src:
            if B[k] != A[k] or before.cells[k] != after.cells[k]:
                bad.append(f"I4 {op} moved {k}")
        return bad
    if acted is None or acted[1] is None:
        return []
    line, col = acted[1], acted[2]
    sign = 1 if op in INSERT else -1
    ref = after if sign > 0 else before
    owner = ref.lines[line][0] if line < len(ref.lines) else None
    for k in src:
        dx, dy = (A[k].x - B[k].x) * sign, (A[k].y - B[k].y) * sign
        if dx < -EPS or dy < -EPS:
            bad.append(f"I4 {op} moved {k} {'up/left' if sign > 0 else 'down/right'}")
        bl, bc = before.cells[k]
        if bl < line or (bl == line and bc < col and before.lines[bl][0] == owner):
            if after.cells.get(k) != (bl, bc) or abs(A[k].y - B[k].y) > EPS:
                bad.append(f"I4 {op} moved {k} above/left of the gesture")
    return bad


def i5(after, focus, place, project=None):
    """Dropped = shown: the new block's cell (or its folded stack's card) is the place's (line, col)."""
    if place is None or not place.lit:
        return []
    cell = after.cells.get(focus)
    if cell is None and project is not None:
        cell = after.cells.get("card:" + str(Index(project).row_of.get(focus)))
    if cell != (place.line, place.col):
        return [f"I5 dropped at {(place.line, place.col)} shown at {cell}"]
    return []


def i6(doc, d, loaded):
    bad = []
    if layout(doc.project, doc.view) != d:
        bad.append("I6 layout not deterministic")
    if layout(loaded.project, loaded.view) != d:
        bad.append("I6 layout differs after save/load")
    return bad


def rows_drawn(p, d):
    """Every row of Project.rows (the only source of row lists) is visibly named, every block drawn or folded."""
    ix = Index(p)
    capt = {c[0] for c in d.captions}
    bad = []
    for r in p.rows:
        named = r.id in capt or r.id in d.cards or (len(r.nights) == 1) or (
            r.blocks and p.blocks[r.blocks[0]].kind in JOIN_KINDS and title(p, r.blocks[0], ix) == r.name)
        if not named:
            bad.append(f"row {r.name} not named in the drawing")
        for b in r.blocks + [x for n in r.nights for x in n if x is not None]:
            if b not in d.boxes and r.id not in d.cards:
                bad.append(f"block {b} of {r.name} not drawn")
    return bad


def static(doc, d):
    p = doc.project
    return i1(d) + i2(p, d) + i3(p, d) + rows_drawn(p, d)


def crossings(d):
    """Wire crossings of a Drawing: a horizontal segment of one wire strictly crossing a vertical one of another."""
    hs, vs = [], []
    for w in d.wires:
        for (x1, y1), (x2, y2) in zip(w.points, w.points[1:]):
            if abs(y1 - y2) <= EPS and abs(x1 - x2) > EPS:
                hs.append((y1, min(x1, x2), max(x1, x2), w.id))
            elif abs(x1 - x2) <= EPS and abs(y1 - y2) > EPS:
                vs.append((x1, min(y1, y2), max(y1, y2), w.id))
    n = 0
    for y, a, b, wid in hs:
        for x, c, e, vid in vs:
            if vid != wid and a + EPS < x < b - EPS and c + EPS < y < e - EPS:
                n += 1
    return n
