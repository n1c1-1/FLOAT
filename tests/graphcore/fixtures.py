"""Project fixtures for the tests: small graphs built by the core's operations, so they are valid by construction.
(The product's templates come with the gallery; these only exercise the core.)"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from astro_gpu.graphcore.history import History  # noqa: E402
from astro_gpu.graphcore.model import Index  # noqa: E402
from astro_gpu.graphcore.operations import json_of  # noqa: E402


class Doc(History):
    """The History with the test fixtures: Doc.from_template(key)."""

    @classmethod
    def from_template(cls, key):
        return build(cls, key)

    @property
    def _json(self):
        return json_of(self.project)


def _ok(doc, op, **kw):
    r = doc.apply(op, **kw)
    if not r.ok:
        raise AssertionError(f"template step {op} {kw}: {r.reason_text()}")
    return r


def _stack(doc, name, kinds, kind="session"):
    """A stack placed block by block (a Session comes alone): its Calibration, Registration, Selection, Stacking to
    Post, then `kinds`."""
    _ok(doc, "add_row", kind=kind)
    rid = doc.project.rows[-1].id
    _ok(doc, "rename_row", row=rid, name=name)
    _ok(doc, "add_block", kind="calibration", at={"type": "calib", "night": doc.project.rows[-1].nights[0][0]})
    for k in ("registration", "selection", "stacking"):
        _ok(doc, "add_block", kind=k, at={"type": "insert", "row": rid, "after": _last(doc, rid)})
    _ok(doc, "send_to_post", block=_last(doc, rid))
    for k in kinds:                                  # appended in the order given (not the natural order)
        _ok(doc, "add_block", kind=k, at={"type": "insert", "row": rid, "after": _last(doc, rid)})
    return rid


def _last(doc, rid):
    blocks = Index(doc.project).row[rid].blocks
    return blocks[-1] if blocks else None


def _unpost(doc, rids):
    ix = Index(doc.project)
    for w in list(doc.project.wires):
        if w.dst[0] == ix.post and ix.row_of.get(w.src[0]) in rids:
            _ok(doc, "unwire", wire=w.id)


def _join(doc, kind, typ=None):
    _ok(doc, "add_block", kind=kind, at={"type": "comp"})
    j = doc.project.rows[-1].blocks[0]
    if typ:
        _ok(doc, "set_combination_type", block=j, type=typ)
    return j


def colour_minimal(doc):
    _stack(doc, "Session 1", ["astrometry", "gradient", "color"])


def colour_complete(doc):
    r = _stack(doc, "Session 1", ["astrometry", "gradient", "color", "denoise", "stars"])
    ef = _last(doc, r)
    _ok(doc, "add_block", kind="combine_layers", at={"type": "comp", "after": ef})


def filters_minimal(doc):
    ha = _stack(doc, "Ha", ["astrometry", "gradient"])
    o3 = _stack(doc, "OIII", ["gradient"])
    j = _join(doc, "combine", "ho")
    _ok(doc, "wire", src=_last(doc, ha), src_port="image", dst=j, dst_port="Hα")
    _ok(doc, "wire", src=_last(doc, o3), src_port="image", dst=j, dst_port="O III")
    _unpost(doc, [ha, o3])
    _ok(doc, "send_to_post", block=j)


def _filters(doc, rgb):
    k3 = ["gradient", "denoise", "stars"]
    rows = [_stack(doc, "SII", ["astrometry"] + k3 if rgb else k3), _stack(doc, "Ha", k3), _stack(doc, "OIII", k3)]
    j = _join(doc, "combine")
    for rid, port in zip(rows, ["S II", "Hα", "O III"]):
        _ok(doc, "wire", src=_last(doc, rid), src_port="Starless", dst=j, dst_port=port)
    if rgb:
        rvb = _stack(doc, "RGB", ["gradient", "denoise", "color", "stars"])
        stars = (_last(doc, rvb), "Stars only")
        rows.append(rvb)
    else:
        j2 = _join(doc, "combine")
        for rid, port in zip(rows, ["S II", "Hα", "O III"]):
            _ok(doc, "wire", src=_last(doc, rid), src_port="Stars only", dst=j2, dst_port=port)
        stars = (j2, "image")
    cc = _join(doc, "combine_layers")
    _ok(doc, "wire", src=j, src_port="image", dst=cc, dst_port="Background")
    _ok(doc, "wire", src=stars[0], src_port=stars[1], dst=cc, dst_port="Stars")
    _unpost(doc, rows)
    _ok(doc, "send_to_post", block=cc)


def filters_complete(doc):
    _filters(doc, False)


def sho_rgb(doc):
    _filters(doc, True)


def several_images(doc):
    for _i in range(3):
        _ok(doc, "add_row", kind="image")


def blank(doc):
    pass


TEMPLATES = {
    "colour_minimal": ("Colour minimal", "Session → … Stacking → Astrometry → Gradient → Color → Post",
                        colour_minimal),
    "colour_complete": ("Colour complete", "+ Denoise, Stars/background → Combine layers → Post", colour_complete),
    "filters_minimal": ("Filters minimal", "Ha (reference) and OIII → H-O → Post", filters_minimal),
    "filters_complete": ("Filters complete", "SII, Ha, OIII: S-H-O starless, S-H-O 2 stars → Combine layers",
                        filters_complete),
    "sho_rgb": ("SHO + RGB stars", "SII, Ha, OIII in S-H-O, stars of the RGB stack", sho_rgb),
    "several_images": ("Several images", "Image 1 to 3 → Post", several_images),
    "blank": ("Blank page", "Post only", blank),
}


def templates():
    return [(k, v[0], v[1]) for k, v in TEMPLATES.items()]


def build(cls, key):
    doc = cls()
    TEMPLATES[key][2](doc)
    doc._undo.clear()
    doc._redo.clear()
    return doc
