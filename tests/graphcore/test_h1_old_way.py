"""Hole H1 of the engine owner's review: graphs that do not come from a converted project. Every gallery template is
built the old way (GraphPage.make_from_template with ProjectModel, on a project with fake sessions or none) and the
new way (graphcore.templates as one apply_many, compiled into the same kind of project, then the same Post actions run
by the model's own Post functions), and the two projects are compared: the whole definition (to_dict, Post included)
and every key table. The commonest gestures (insert a block, try a variant, a block's image to Post, Align on, a
combination) are compared the same way, from a project built both ways.

Normalised (each cause given): the creation time stamp and the project folder (two projects); the ids of Post's
layers (a counter of the process, never read by a key) renumbered in their order."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if Path("C:/Windows/Fonts").is_dir():
    os.environ.setdefault("QT_QPA_FONTDIR", "C:/Windows/Fonts")
pytest.importorskip("PySide6")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from PySide6.QtWidgets import QApplication  # noqa: E402

from astro_gpu.client import i18n  # noqa: E402
from astro_gpu.client.gallery import GALLERY, _view_opts  # noqa: E402
from astro_gpu.client.graph import GraphPage  # noqa: E402
from astro_gpu.client.model import ProjectModel  # noqa: E402
from astro_gpu.client.services.worker import ApiWorker  # noqa: E402
from astro_gpu.core import project2  # noqa: E402
from astro_gpu.core.project import Target  # noqa: E402
from astro_gpu.graphcore import templates  # noqa: E402
from astro_gpu.graphcore.compile import _NoDisk, _key_rules, compile_into  # noqa: E402
from astro_gpu.graphcore.convert import convert  # noqa: E402
from astro_gpu.graphcore.model import Index  # noqa: E402
from astro_gpu.graphcore.operations import apply_many, apply  # noqa: E402
from astro_gpu.graphcore.proof import _first_diff, keys_table  # noqa: E402

FAKE = [("Night S", "SII"), ("Night H", "Ha"), ("Night O", "OIII")]


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


class FakePage:
    """The page's template builders without the page (they only call the model)."""
    make_from_template = GraphPage.make_from_template
    make_filters = GraphPage.make_filters

    def __init__(self, m):
        self.s = m
        self.said = []

    def select(self, **k):
        pass

    def toast(self, t):
        self.said.append(t)


class World:
    def __init__(self, root: Path, sessions):
        i18n.set_language("en")
        self.p = project2.create(root, "P", Target(name="M 31"))
        for sid, (name, flt) in enumerate(sessions, 1):
            s = project2.Session(sid, name, filter=flt)
            s.lights = [f"D:/fake/{name}/L{i}.fit" for i in range(3)]
            self.p.sessions.append(s)
        project2.save(self.p)
        self.w = ApiWorker()
        self.m = ProjectModel(self.p, self.w)
        self.w.wait(120)

    def close(self):
        self.m.close(save=False)
        self.w.stop()


def template_name(m, t):
    from astro_gpu.client.i18n import _
    return m.unique_name(_(t["palette"]) if t.get("kind") == "fusion" else _(t["name"]))


def old_way(root, t, sessions):
    w = World(root, sessions)
    name = template_name(w.m, t)
    FakePage(w.m).make_from_template(t, name, _view_opts(t))
    return w


def run_post(m, actions, wid_of):
    """A template's Post actions, by the model's own Post functions (as the old builders call them)."""
    keys = {}
    for a in actions:
        if a[0] == "add_layer":
            keys[a[3]] = m.add_post_layer(wid_of[a[1]], a[2], None)
        elif a[0] == "side":
            m.set_layer_side(keys[a[1]], a[2], wid_of[a[3]], a[4])
        elif a[0] == "remove_part":
            m.remove_post_part(wid_of[a[1]], a[2], None)
        elif a[0] == "removed":
            gone = {(int(x), y) for x, y in m.p.post.get("removed") or []} | {(wid_of[r], part) for r, part in a[1]}
            m.p.post = {**m.p.post, "removed": sorted([list(x) for x in gone], key=str)}


def new_way(root, t, sessions):
    """The template as graphcore operations on the converted (empty) project, compiled into it; Post's actions by
    the model; the graph kept on the world."""
    w = World(root, sessions)
    g = convert(w.p, block_keys=lambda wid: [])
    name = template_name(w.m, t)
    plan = templates.build(g, t, name, _view_opts(t))
    g2, res = apply_many(g, plan.steps)
    assert res.ok, (t["key"], res.reason_text())
    with w.m._edit():                       # one step, as the old builders' batch: Post's pruning runs at its end
        c = compile_into(g2, w.p)
        w.m._rebuild_views()
        run_post(w.m, plan.post(res.created["rows"]), c.wids)
    w.g, w.c = g2, c
    return w


def whole(p) -> dict:
    d = p.to_dict()
    d.pop("created", None)
    ids = {}
    for v in d["post"].get("versions") or []:
        for lay in v.get("layers") or []:
            ids.setdefault(lay["id"], len(ids) + 1)
    for v in d["post"].get("versions") or []:
        for lay in v.get("layers") or []:
            lay["id"] = ids[lay["id"]]
            if lay.get("target") in ids:
                lay["target"] = ids[lay["target"]]
    return d


def differences(a, b) -> list[str]:
    out = []
    d = _first_diff(whole(a), whole(b), "definition")
    if d:
        out.append(d)
    rules = _key_rules()
    d = _first_diff(_rooted(keys_table(a, rules, _NoDisk())), _rooted(keys_table(b, rules, _NoDisk())), "keys")
    if d:
        out.append(d)
    return out


def _rooted(table):
    """The key table with the project's folder named alike on both sides (the overrides hold its store's paths)."""
    import json
    return json.loads(json.dumps(table, default=str).replace("side_old", "side").replace("side_new", "side"))


TEMPLATES = [t for t in GALLERY if t.get("copy") is None]


@pytest.mark.parametrize("sessions", [[], FAKE], ids=["blank", "fake sessions"])
@pytest.mark.parametrize("t", TEMPLATES, ids=[t["key"] for t in TEMPLATES])
def test_every_template_built_the_new_way_is_the_old_project(qapp, tmp_path, t, sessions):
    old = old_way(tmp_path / "side_old", t, sessions)
    new = new_way(tmp_path / "side_new", t, sessions)
    try:
        assert differences(old.p, new.p) == []
    finally:
        old.close()
        new.close()


# ---------------------------------------------------------------------------------------------------- gestures
def _both(tmp_path, key, sessions=()):
    t = next(x for x in GALLERY if x["key"] == key)
    old, new = old_way(tmp_path / "side_old", t, list(sessions)), new_way(tmp_path / "side_new", t, list(sessions))
    assert differences(old.p, new.p) == []
    return old, new


def _gesture(new, steps):
    g2, res = apply_many(new.g, steps)
    assert res.ok, res.reason_text()
    with new.m._edit():
        new.c = compile_into(g2, new.p)
    new.m._rebuild_views()
    new.g = g2
    return res


def _row(g, name):
    return next(r for r in g.rows if r.name == name)


def _blk(g, r, kind):
    return next(b for b in Index(g).effective(r.id) if g.blocks[b].kind == kind)


def _view(m, name):
    return next(q for q in m.paths if q.name == name)


def test_insert_a_block_as_the_old_model_does(qapp, tmp_path):
    old, new = _both(tmp_path, "colour_min", FAKE[:1])
    try:
        q = _view(old.m, "Colour minimal")
        old.m.insert_block(q, "deconv")
        r = _row(new.g, "Colour minimal")
        at = {"type": "insert", "row": r.id, "after": _blk(new.g, r, "gradient")}
        _gesture(new, [("add_block", {"kind": "deconv", "at": at})])
        assert differences(old.p, new.p) == []
    finally:
        old.close()
        new.close()


def test_try_a_variant_as_the_old_model_does(qapp, tmp_path):
    old, new = _both(tmp_path, "colour_min", FAKE[:1])
    try:
        q = _view(old.m, "Colour minimal")
        ch = {"kind": "param", "stage": "register", "name": "min_pairs", "value": 20}
        old.m.add_variant(q, ch, "Colour minimal · Min pairs")
        r = _row(new.g, "Colour minimal")
        _gesture(new, [("try_variant", {"block": _blk(new.g, r, "registration"),
                                        "change": {"param": ["register", "min_pairs", 20]},
                                        "name": "Colour minimal · Min pairs"})])
        assert differences(old.p, new.p) == []
    finally:
        old.close()
        new.close()


def test_a_block_to_post_as_the_old_model_does(qapp, tmp_path):
    old, new = _both(tmp_path, "colour_full", FAKE[:1])
    try:
        q = _view(old.m, "Colour full")
        old.m.send_to_post(q, "fond", "Colour full · Gradient")
        r = _row(new.g, "Colour full")
        _gesture(new, [("send_to_post", {"block": _blk(new.g, r, "gradient"), "name": "Colour full · Gradient"})])
        assert differences(old.p, new.p) == []
    finally:
        old.close()
        new.close()


def test_align_on_as_the_old_model_does(qapp, tmp_path):
    old, new = _both(tmp_path, "filters_min", FAKE)
    try:
        ref = _view(old.m, "Ha")
        old.m.remove_node(_view(old.m, "SII"), "align_on")
        old.m.set_align_on(_view(old.m, "SII"), ref.pid, method="plate")
        g = new.g
        _gesture(new, [("own_grid", {"row": _row(g, "SII").id})])
        g = new.g
        _gesture(new, [("set_align", {"row": _row(g, "SII").id, "ref": _row(g, "Ha").id,
                                      "block": _blk(g, _row(g, "Ha"), "astrometry"), "method": "plate"})])
        assert differences(old.p, new.p) == []
    finally:
        old.close()
        new.close()


def test_a_combination_as_the_old_model_does(qapp, tmp_path):
    old, new = _both(tmp_path, "filters_min", FAKE)
    try:
        ha, oiii = _view(old.m, "Ha"), _view(old.m, "OIII")
        old.m.add_fusion("H-O", "ho", [{"pid": ha.pid, "role": "Ha", "channel": "R"},
                                       {"pid": oiii.pid, "role": "OIII", "channel": "GB"}])
        g = new.g
        _gesture(new, [("add_combination", {"type": "ho", "name": "H-O"}),
                       ("wire", lambda res: {"src": _row(g, "Ha").blocks[-1], "src_port": "image",
                                             "dst": res[0].created["block"], "dst_port": "Hα", "channel": "R"}),
                       ("wire", lambda res: {"src": _row(g, "OIII").blocks[-1], "src_port": "image",
                                             "dst": res[0].created["block"], "dst_port": "O III", "channel": "GB"})])
        assert differences(old.p, new.p) == []
    finally:
        old.close()
        new.close()


def test_post_wire_set_follows_every_version(qapp, tmp_path):
    """Engine review 3, with the model's Post: two versions; a layer deleted in version 2 only leaves the graph's
    wire; the wire removed in the graph takes the image out of both versions."""
    from astro_gpu.graphcore.post import post_actions, shared_images, wired_images
    old, new = _both(tmp_path, "colour_min", FAKE[:1])
    old.close()
    try:
        m, g, c = new.m, new.g, new.c
        wid = c.wids[_row(g, "Colour minimal").id]
        m.add_post_layer(wid, None, None)                     # Post holds the image (version 1)
        m.new_post_version()                                  # version 2, a copy, active
        class PostPage:
            is_post_page = True
        layers = [x for x in m.post_layers() if x.get("pid") != wid]
        m.set_post_layers(layers, PostPage(), scope="active")  # deleted in Post, in version 2 only
        assert not any(x.get("pid") == wid for x in m.post_version()["layers"])
        shared = shared_images(m.p.post, defaults=set())
        assert (wid, None) in shared                          # version 1 still shows it: the wire stays
        assert (wid, None) in wired_images(g, c)
        w = next(x for x in g.wires if g.blocks[x.dst[0]].kind == "post")
        g2, res = apply(g, "unwire", wire=w.id)
        assert res.ok
        acts = post_actions(g, g2, c)
        assert acts == [("remove_post_part", wid, None)]
        active = m.p.post["active"]
        for v in list(m.p.post["versions"]):                  # every version, each made active in turn
            m.set_post_version(v["id"])
            m.remove_post_part(wid, None, None)
        m.set_post_version(active)
        assert all(not any(x.get("pid") == wid for x in v.get("layers") or []) for v in m.p.post["versions"])
    finally:
        new.close()
