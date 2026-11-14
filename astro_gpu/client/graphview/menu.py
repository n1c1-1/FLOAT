"""The context menu of a block or a row: its items, each with whether the core accepts it now and why not.

An item is (text, action, args, enabled, why) or None (a separator). ``action`` is a core operation name (applied as
is), or a page action: "inspect" {block}, "rename" {row}, "mark_reviewed" {block}, "submenu" (args = its items).
Whether an operation is enabled is asked of the core by a dry run of it (nothing changes)."""
from __future__ import annotations

from ...graphcore.model import Index, out_ports
from ...graphcore.rules import off_reason
from ..i18n import _

FRAME_SOURCES = ("session", "calibration")


def _item(doc, text, op, args):
    """An item whose operation the core accepts, or greyed with its reason."""
    res = doc.try_many([(op, args)])
    return (text, op, args, res.ok, "" if res.ok else doc.say(res.reason))


def menu(doc, b):
    g = doc.graph
    ix = Index(g)
    if b in ix.row:                                     # a row (caption, card)
        return row_items(doc, ix, b)
    blk = g.blocks[b]
    k = blk.kind
    # a Combine layers block is a layer of Post: its Inspect is Post on that layer (graph.py: "Open Post")
    items = [(_("Open Post") if k in ("combine_layers", "post") else _("Inspect"), "inspect", {"block": b}, True, "")]
    if k != "post":
        text = _("Switch on (D)") if blk.off else _("Switch off (D)")
        why = off_reason(k)
        items.append((text, "switch_off", {"block": b}, False, doc.say(why)) if why else
                     _item(doc, text, "switch_off", {"block": b}))
        items.append((_("Remove pause") if blk.pause else _("Review pause"), "set_pause", {"block": b}, True, ""))
        items.append((_("Mark reviewed"), "mark_reviewed", {"block": b}, doc.review(b) == "paused",
                      "" if doc.review(b) == "paused" else _("after a stop on this block")))
        if out_ports(blk):
            items.append(_item(doc, _("To Post"), "send_to_post", {"block": b}))
        else:
            items.append((_("To Post"), "send_to_post", {"block": b}, False, _("lights first")))
        if k == "stars":
            items += stars_items(doc, ix, b)
        if b in ix.row_of and k not in FRAME_SOURCES + ("combine", "combine_layers", "image"):
            items.append(_item(doc, _("Variant here"), "add_variant", {"block": b}))
    if k == "calibration":
        on = g.calibrate_together
        items.append(_item(doc, _("Calibrate separately") if on else _("Calibrate together"),
                           "set_calibrate_together", {"on": not on}))
    if b in ix.row_of:
        items.append(None)
        items += row_items(doc, ix, ix.row_of[b])
    if k != "post":
        items.append(None)
        items.append(_item(doc, _("Delete"), "remove_block", {"block": b}))
    return items


def stars_items(doc, ix, b):
    """The Stars/background block's two images (today's menu, graph.py): each to Post as it is, or out of it again;
    a Combine layers block putting the stars back on the starless image."""
    from ...graphcore.model import POST_PORT
    g = doc.graph
    out = []
    for port, on, off in (("Starless", _("Starless to Post"), _("Starless out of Post")),
                          ("Stars only", _("Stars to Post"), _("Stars out of Post"))):
        w = next((w for w in g.wires if w.src == [b, port] and w.dst[0] == ix.post), None)
        out.append(_item(doc, off, "unwire", {"wire": w.id}) if w is not None else
                   _item(doc, on, "wire", {"src": b, "src_port": port, "dst": ix.post, "dst_port": POST_PORT}))
    out.append(_item(doc, _("Combine layers"), "add_block", {"kind": "combine_layers",
                                                             "at": {"type": "comp", "after": b}}))
    return out


def align_items(doc, ix, rid):
    """The Align on choices of a stack: on another row (each, refused ones greyed with the engine's reason), its own
    grid, switched off; "reference for all" in one gesture."""
    g = doc.graph
    r = ix.row[rid]
    a = r.align or {}
    out = [_item(doc, _("Reference for all"), "set_reference", {"row": rid})]
    others = [x for x in g.rows if x.id != rid]
    if others:
        out.append(None)
    for x in others:
        text = x.name if a.get("ref") != x.id else "✓ " + x.name
        out.append(_item(doc, text, "set_align", {"row": rid, "ref": x.id}))
    out.append(None)
    out.append(_item(doc, _("Own grid"), "own_grid", {"row": rid}))
    if r.align:
        out.append(_item(doc, _("Switch on") if a.get("off") else _("Switch off"), "set_align_off", {"row": rid}))
    return out


def row_items(doc, ix, rid):
    r = ix.row[rid]
    items = []
    if r.nights:
        items.append(_item(doc, _("Add session"), "add_night", {"row": rid}))
        items.append((_("Align on"), "submenu", align_items(doc, ix, rid), True, ""))
    sub = [(_("Rename"), "rename", {"row": rid}, True, ""),
           _item(doc, _("Duplicate (Ctrl+D)"), "duplicate_row", {"row": rid}),
           _item(doc, _("Delete workflow"), "delete_row", {"row": rid})]
    items.append((_("Workflow “{name}”").format(name=r.name), "submenu", sub, True, ""))
    return items
