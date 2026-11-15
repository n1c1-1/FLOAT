"""The graph's gestures: what they share (the refusal, the outcome, the common helpers on rows, nights, blocks
and flows)."""
import copy
from dataclasses import dataclass
from .model import FILE_KINDS, FRAME_KINDS, Index, POST_PORT, SYNTH_PORT, Wire, edges, new_block, new_id, out_ports
from .msg import M, join, name_of
from .rules import LABEL
from ..core import workflow as wfl


COMMON = "common"


class Refusal(Exception):
    """An operation that cannot be done. msg: the reason, a message (see msg.py)."""

    def __init__(self, msg):
        super().__init__(msg)
        self.msg = msg


@dataclass
class Out:
    toast: object = ""        # a message
    focus: object = None
    new: object = None       # block whose cell is where the gesture acted (after)
    old: object = None       # block removed: its cell before
    style: object = None     # None | "need" | "off": how a need refusal is worded
    created: object = None   # {"row": id, "session": id, "block": id, "wire": id}: what the gesture made


def _flow_chain(q, ix, fid):
    if fid in ix.row:
        return ix.effective(fid)
    t = next((t for t in q.taps if t.id == fid), None)
    if t is None or t.block not in ix.row_of:
        return []
    eff = ix.effective(ix.row_of[t.block])
    return eff[:eff.index(t.block) + 1]


# ---------------------------------------------------------------------------------------------------- helpers
def _blk(q, b, kinds=None):
    if b not in q.blocks:
        raise Refusal("block not found")
    if kinds and q.blocks[b].kind not in kinds:
        raise Refusal(M("{lab} does not fit here", lab=LABEL.get(q.blocks[b].kind, "")))
    return q.blocks[b]


def _row(q, rid):
    for r in q.rows:
        if r.id == rid:
            return r
    raise Refusal("row not found")


def _names(q):
    return {r.name for r in q.rows} | {b.name for b in q.blocks.values() if b.name}


def free_name(q, base, first_plain=True):
    names = {r.name for r in q.rows}
    if first_plain and base not in names:
        return base
    n = 2
    while f"{base} {n}" in names:
        n += 1
    return f"{base} {n}"


def session_name(q):
    names, n = _names(q), 1
    while name_of("Session {n}", n=n) in names:
        n += 1
    return name_of("Session {n}", n=n)


def _night(q, name, files=None, calibration=True):
    """A night: its Session, with its Calibration or without (None: its lights taken as they are)."""
    s = new_block(q, "session", name=name, files=files)
    return [s.id, new_block(q, "calibration").id if calibration else None]


def _copy_calibration(q, c, to, sid=None):
    """Night Calibration c copied into Calibration `to` (a variant's, a branch's, a duplicate's night): all it holds,
    its settings, switch, review pause, engine settings and own files; `sid` the session it calibrates (None: a new
    one, numbered by the compile)."""
    a, b = q.blocks[c], q.blocks[to]
    b.settings, b.off, b.pause = copy.deepcopy(a.settings), a.off, a.pause
    b.method, b.params, b.own = a.method, copy.deepcopy(a.params), copy.deepcopy(a.own)
    if sid is not None:
        b.sid = sid


def _stack_calibrated(q, r):
    """Calibrated together, a stack's nights have the engine's one Calibration of the stack, or none: whether a night
    joining stack r must have one (None: each night its own, or no night yet: as the gesture says)."""
    if not q.calibrate_together or not r.nights:
        return None
    return r.nights[0][1] is not None


def _follow_stack(q, r, n):
    """A night joining stack r, calibrated together: its Calibration follows the stack's (one engine node for all its
    nights: made or taken away, switched as the stack's). -> a message when it changed, else ""."""
    want = _stack_calibrated(q, r)
    if want is None:
        return ""
    if want and n[1] is None:
        n[1] = new_block(q, "calibration", sid=q.blocks[n[0]].sid).id
        q.blocks[n[1]].off = _night_off(q, r)
        return "Calibration added: its stack is calibrated together"
    if not want and n[1] is not None:
        del q.blocks[n[1]]
        n[1] = None
        return "Calibration taken away: its stack is not calibrated"
    if n[1] is not None:
        q.blocks[n[1]].off = _night_off(q, r)
    return ""


def _to_post(q, b, port=None):
    post = Index(q).post
    port = port or out_ports(q.blocks[b])[0]
    if not any(w.src == [b, port] and w.dst[0] == post for w in q.wires):
        q.wires.append(Wire(new_id(q, "w"), [b, port], [post, POST_PORT]))


def _map_port(blk, port):
    ports = out_ports(blk)
    return port if port in ports else (ports[0] if ports else None)


def _copy_block(q, b):
    """A variant's copy of chain block b: its settings, its switch, its method and its engine settings (a Gradient
    variant of an RBF Gradient is an RBF Gradient with the same settings, as the original's panel shows them)."""
    o = q.blocks[b]
    return new_block(q, o.kind, settings=dict(o.settings), off=o.off, method=o.method,
                     params=copy.deepcopy(o.params)).id


def _after_subtree(q, rid):
    """Index in rows right after the rows of rid's subtree."""
    sub = set(Index(q).subtree(rid))
    return max(i for i, r in enumerate(q.rows) if r.id in sub) + 1


def delete_rows(q, rids):
    """Delete rows with their subtrees, blocks and wires; the rows aligned on a deleted one pass to the first of
    them (its own grid then), with the deleted target's Astrometry when it has none."""
    ix = Index(q)
    q._astro_moved = False
    gone = []
    for r in rids:
        for x in ix.subtree(r):
            if x not in gone:
                gone.append(x)
    left = [r for r in q.rows if r.id not in gone and r.align and r.align.get("ref") in gone]
    if left:
        heir = left[0]
        old = heir.align["ref"]
        heir.align = None
        for r in left[1:]:
            r.align["ref"], r.align["from"] = heir.id, None
            r.align["auto"] = True
        astro = [b for b in ix.row[old].blocks if q.blocks[b].kind == "astrometry"]             if old in ix.row and ix.row[old].nights and heir.nights else []
        if astro and not any(q.blocks[b].kind == "astrometry" for b in heir.blocks):
            ix.row[old].blocks.remove(astro[0])
            _insert_after_frames(q, heir, astro[0])
            q._astro_moved = True
    blocks = {b for r in gone for b in ix.row[r].blocks + [x for n in ix.row[r].nights for x in n if x is not None]}
    nw = len([w for w in q.wires if w.src[0] in blocks or w.dst[0] in blocks])
    q.wires = [w for w in q.wires if w.src[0] not in blocks and w.dst[0] not in blocks]
    for b in blocks:
        del q.blocks[b]
    q.rows = [r for r in q.rows if r.id not in gone]
    names = join([M("“{_name}”", _name=ix.row[r].name) for r in gone], ", ")
    done = (M("{names} deleted", names=names) if len(gone) == 1
            else M("{n} deleted: {names}", n=len(gone), names=names))
    if nw:
        done = join([done, M(" with {n} wire" if nw == 1 else " with {n} wires", n=nw)], "")
    return done


def _auto_align(q, r):
    """A new stack is aligned on the stack the others are aligned on (the reference), from its Astrometry when it
    has one; none when it is the first stack."""
    ref = q.reference
    if ref is not None and ref != r.id and ref in {x.id for x in q.rows}:
        r.align = new_align(ref)


def new_align(ref, block=None):
    return {"ref": ref, "from": block, "from_name": None, "ref_wid": None, "method": wfl.DEFAULT_METHOD["align_on"],
            "off": False,
            "fork": None, "params": {}, "auto": block is None}


def _night_off(q, r):
    """The switch of the Calibration of a night joining stack r: calibrated together, the stack's (one setting for
    all its nights)."""
    return bool(q.calibrate_together and r.nights and r.nights[0][1] is not None and q.blocks[r.nights[0][1]].off)


def _insert_after_frames(q, r, b):
    i = max([j for j, x in enumerate(r.blocks) if q.blocks[x].kind in FRAME_KINDS + ("image",)] or [-1]) + 1
    r.blocks.insert(i, b)


def detach(q, b, keep=False):
    """Take chain block b out of its row: its variants re-fork at its predecessor, its port wires pass to the
    predecessor of its row, an emptied variant row goes (its own variants take its place)."""
    ix = Index(q)
    rid = ix.row_of[b]
    r = ix.row[rid]
    i = ix.pos[b]
    pred = ix.pred(b)
    kids = [v for v in ix.children[rid] if ix.row[v].fork[1] == b]
    if i == 0 and kids and (len(r.blocks) > 1 or pred is None):
        raise Refusal("variants start from this block: delete them first")
    for w in list(q.wires):
        if w.src[0] == b:
            # a Synthetic stars' own image goes with it (nothing before it gives the synthetic stars)
            port = _map_port(q.blocks[pred], w.src[1]) if (i > 0 and pred and w.src[1] != SYNTH_PORT) else None
            if port is None:
                q.wires.remove(w)
            else:
                w.src = [pred, port]
    for v in kids:
        ix.row[v].fork = [ix.row_of[pred], pred] if pred else None
    r.blocks.remove(b)
    if not r.blocks and not r.nights:
        at = q.rows.index(r)
        q.rows = [x for x in q.rows if x.id not in kids]
        for k, v in enumerate(kids):
            q.rows.insert(at + 1 + k, ix.row[v])
        q.rows.remove(r)
    if not keep:
        q.wires = [w for w in q.wires if w.dst[0] != b]
        del q.blocks[b]


def would_cycle(q, src, dst):
    ix = Index(q)
    succ = {}
    for a, c in edges(q, ix):
        succ.setdefault(a, []).append(c)
    stack, seen = [dst], set()
    while stack:
        x = stack.pop()
        if x == src:
            return True
        if x not in seen:
            seen.add(x)
            stack += succ.get(x, [])
    return False


def fold_guard(before, q, view):
    """Folded band: a gesture that would fork a variant inside a stack is refused (its fork block is hidden in the
    card; unfolding the band by itself would move everything)."""
    if not view or view.get("folded") is not True:
        return
    from .layout import foldable
    ib, iq = Index(before), Index(q)
    for r in iq.row:
        if iq.row[r].nights and not foldable(q, iq, r) and (r not in ib.row or foldable(before, ib, r)):
            raise Refusal("Preprocessing is folded: unfold it first")


def _kind(k):
    if k not in FILE_KINDS:
        raise Refusal("unknown kind of files")


def _dedup(paths):
    out, seen = [], set()
    for x in paths:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _clone_block(q, b):
    """A copy of chain block b with all it holds (settings, engine settings, method, switch, pause); its node name is
    given by the compile in its own chain."""
    o = q.blocks[b]
    return new_block(q, o.kind, settings=dict(o.settings), off=o.off, pause=o.pause, method=o.method,
                     params=copy.deepcopy(o.params), source=copy.deepcopy(o.source)).id


def _copy_flow(dst, src):
    for k in ("all_nights", "sessions", "output", "extra", "unset", "pauses", "options", "end", "legacy"):
        setattr(dst, k, copy.deepcopy(getattr(src, k)))


def _file_name(path):
    return str(path).replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
