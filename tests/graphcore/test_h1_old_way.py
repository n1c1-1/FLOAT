"""Hole H1 of the engine owner's review: graphs that do not come from a converted project. Every gallery template is
built the new way (graphcore.templates as one apply_many, compiled into a project with fake sessions or none, then the
same Post actions run by the model's own Post functions) and compared with the project the old way made
(GraphPage.make_from_template with today's ProjectModel): the whole definition (to_dict, Post included) and every key
table. The commonest gestures (insert a block, try a variant, a block's image to Post, Align on, a combination) are
compared the same way, from a project built both ways.

The project model writes every setting of a block (at its default when nobody changed it): the definitions are
compared without the values stored at their static default, the key tables as they are (their overrides without those
values: the jobs get the same settings).

The old way's code went with the swap to the graph: its projects were recorded before, by this file run with
``FLOAT_RECORD_OLD_WAY=1`` on the last commit that had it (``old_way/<test>.json``: each state the old model reached,
its definition and what its call returned). A replay compares the new project with the recorded definition, and the
key tables of both computed now, in this process (a stage version bumped since changes both alike).

Normalised (each cause given): the creation time stamp and the project folder (two projects); the ids of Post's
layers (a counter of the process, never read by a key) renumbered in their order."""
from __future__ import annotations

import json
import os
import re
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
#: the old way's projects, recorded before its code went (see the module's text)
RECORDS = Path(__file__).with_name("old_way")
RECORD = os.environ.get("FLOAT_RECORD_OLD_WAY") == "1"


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


# ---------------------------------------------------------------------------------------------- the recordings
def _plain(v):
    """A call's answer as a recording keeps it (tuples as lists, views by their name)."""
    if hasattr(v, "pid") and hasattr(v, "name"):
        return v.name
    return json.loads(json.dumps(v, default=str))


class Recording:
    """The states the old model reached in one test, in order: ``check(p, value)`` records one (``RECORD``) or compares
    the project ``p`` with the next one recorded and gives back what the old call returned."""

    def __init__(self, name: str):
        self.path = RECORDS / (re.sub(r"[^A-Za-z0-9_.-]+", "_", name) + ".json")
        self.states: list = []
        self.at = 0
        if not RECORD:
            if not self.path.is_file():
                pytest.fail(f"no recording of the old way: {self.path.name}")
            self.states = json.loads(self.path.read_text(encoding="utf-8"))

    def check(self, p, value=None):
        if RECORD:
            self.states.append({"definition": p.to_dict(), "value": _plain(value)})
            return _plain(value)
        assert self.at < len(self.states), f"{self.path.name}: more states than the old way reached"
        rec = self.states[self.at]
        self.at += 1
        assert differences_to(rec["definition"], p) == []
        return rec["value"]

    def save(self):
        if RECORD:
            RECORDS.mkdir(exist_ok=True)
            self.path.write_text(json.dumps(self.states, indent=1, sort_keys=True, default=str), encoding="utf-8")
        else:
            assert self.at == len(self.states), f"{self.path.name}: the old way reached more states"


@pytest.fixture()
def recording(request):
    r = Recording(f"{request.module.__name__.rpartition('.')[2]}__{request.node.name}")
    yield r
    r.save()


def template_recording(t, sessions) -> Recording:
    """The recording of a template built the old way (``test_every_template...``'s)."""
    sid = "blank" if not sessions else "fake sessions"
    return Recording(f"test_h1_old_way__test_every_template_built_the_new_way_is_the_old_project[{t['key']}-{sid}]")


# ---------------------------------------------------------------------------------------------- the projects
class FakePage:
    """The old page's template builders without the page (they only call the model): recording only."""

    def __init__(self, m):
        from astro_gpu.client.graph import GraphPage      # the old page: there only on the commits that record
        self.s = m
        self.said = []
        self._page = GraphPage

    def make_from_template(self, t, name, opts):
        return self._page.make_from_template(self, t, name, opts)

    def make_filters(self, t, name, opts):
        return self._page.make_filters(self, t, name, opts)

    def select(self, **k):
        pass

    def toast(self, t):
        self.said.append(t)


class World:
    def __init__(self, root: Path, sessions, cls=None):
        i18n.set_language("en")
        self.p = project2.create(root, "P", Target(name="M 31"))
        for sid, (name, flt) in enumerate(sessions, 1):
            s = project2.Session(sid, name, filter=flt)
            s.lights = [f"D:/fake/{name}/L{i}.fit" for i in range(3)]
            self.p.sessions.append(s)
        project2.save(self.p)
        self.w = ApiWorker()
        self.m = (cls or ProjectModel)(self.p, self.w)
        self.w.wait(120)

    def close(self):
        self.m.close(save=False)
        self.w.stop()


def template_name(m, t):
    from astro_gpu.client.i18n import _
    return m.unique_name(_(t["palette"]) if t.get("kind") == "fusion" else _(t["name"]))


def old_way(root, t, sessions):
    """The template's project: recording, made by the old page's builder on today's model; else as the graph page
    makes it (the model's ``apply_template``, whose project test_graphmodel_facades checks against the recording)."""
    w = World(root, sessions)
    name = template_name(w.m, t)
    if RECORD:
        FakePage(w.m).make_from_template(t, name, _view_opts(t))
    else:
        res = w.m.apply_template(t, name, _view_opts(t))
        assert res.ok, res.reason
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


def _whole(d: dict) -> dict:
    d = json.loads(json.dumps(d, default=str))
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


def whole(p) -> dict:
    return _whole(p.to_dict())


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


def differences_to(recorded: dict, p) -> list[str]:
    """A recorded definition against the project ``p``: the definitions, then the key tables of both, the recorded
    one read in ``p``'s folder."""
    from astro_gpu.client.graphdefaults import strip_defaults, strip_override_defaults
    d = _first_diff(_whole(recorded), _whole(strip_defaults(p.to_dict(), recorded)), "definition")
    if d:
        return [d]
    q = project2.Project2.from_dict(json.loads(json.dumps(recorded)), p.root)
    rules = _key_rules()
    d = _first_diff(_rooted(strip_override_defaults(keys_table(q, rules, _NoDisk()), rules)),
                    _rooted(strip_override_defaults(keys_table(p, rules, _NoDisk()), rules)), "keys")
    return [d] if d else []


def _rooted(table):
    """The key table with the project's folder named alike on both sides (the overrides hold its store's paths)."""
    return json.loads(json.dumps(table, default=str).replace("side_old", "side").replace("side_new", "side"))


TEMPLATES = [t for t in GALLERY if t.get("copy") is None]


@pytest.mark.parametrize("sessions", [[], FAKE], ids=["blank", "fake sessions"])
@pytest.mark.parametrize("t", TEMPLATES, ids=[t["key"] for t in TEMPLATES])
def test_every_template_built_the_new_way_is_the_old_project(qapp, tmp_path, t, sessions, recording):
    w = (old_way if RECORD else new_way)(tmp_path / "side", t, sessions)
    try:
        recording.check(w.p)
    finally:
        w.close()


# ---------------------------------------------------------------------------------------------------- gestures
def _both(tmp_path, key, rec, sessions=()):
    """The template's project: recording, the old way's (recorded); else the new way's, checked against it."""
    t = next(x for x in GALLERY if x["key"] == key)
    w = (old_way if RECORD else new_way)(tmp_path / "side", t, list(sessions))
    rec.check(w.p)
    return w


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


def test_insert_a_block_as_the_old_model_does(qapp, tmp_path, recording):
    w = _both(tmp_path, "colour_min", recording, FAKE[:1])
    try:
        if RECORD:
            w.m.insert_block(_view(w.m, "Colour minimal"), "deconv")
        else:
            r = _row(w.g, "Colour minimal")
            at = {"type": "insert", "row": r.id, "after": _blk(w.g, r, "gradient")}
            _gesture(w, [("add_block", {"kind": "deconv", "at": at})])
        recording.check(w.p)
    finally:
        w.close()


def test_try_a_variant_as_the_old_model_does(qapp, tmp_path, recording):
    w = _both(tmp_path, "colour_min", recording, FAKE[:1])
    try:
        if RECORD:
            ch = {"kind": "param", "stage": "register", "name": "min_pairs", "value": 20}
            w.m.add_variant(_view(w.m, "Colour minimal"), ch, "Colour minimal · Min pairs")
        else:
            r = _row(w.g, "Colour minimal")
            _gesture(w, [("try_variant", {"block": _blk(w.g, r, "registration"),
                                          "change": {"param": ["register", "min_pairs", 20]},
                                          "name": "Colour minimal · Min pairs"})])
        recording.check(w.p)
    finally:
        w.close()


def test_a_block_to_post_as_the_old_model_does(qapp, tmp_path, recording):
    w = _both(tmp_path, "colour_full", recording, FAKE[:1])
    try:
        if RECORD:
            w.m.send_to_post(_view(w.m, "Colour full"), "fond", "Colour full · Gradient")
        else:
            r = _row(w.g, "Colour full")
            _gesture(w, [("send_to_post", {"block": _blk(w.g, r, "gradient"), "name": "Colour full · Gradient"})])
            w.m._write_branch_layer(_view(w.m, "Colour full · Gradient"))   # its layer, as the model writes it
        recording.check(w.p)
    finally:
        w.close()


def test_align_on_as_the_old_model_does(qapp, tmp_path, recording):
    w = _both(tmp_path, "filters_min", recording, FAKE)
    try:
        if RECORD:
            ref = _view(w.m, "Ha")
            w.m.remove_node(_view(w.m, "SII"), "align_on")
            w.m.set_align_on(_view(w.m, "SII"), ref.pid, method="plate")
        else:
            g = w.g
            _gesture(w, [("own_grid", {"row": _row(g, "SII").id})])
            g = w.g
            _gesture(w, [("set_align", {"row": _row(g, "SII").id, "ref": _row(g, "Ha").id,
                                        "block": _blk(g, _row(g, "Ha"), "astrometry"), "method": "plate"})])
        recording.check(w.p)
    finally:
        w.close()


def test_a_combination_as_the_old_model_does(qapp, tmp_path, recording):
    w = _both(tmp_path, "filters_min", recording, FAKE)
    try:
        if RECORD:
            ha, oiii = _view(w.m, "Ha"), _view(w.m, "OIII")
            w.m.add_fusion("H-O", "ho", [{"pid": ha.pid, "role": "Ha", "channel": "R"},
                                         {"pid": oiii.pid, "role": "OIII", "channel": "GB"}])
        else:
            g = w.g
            _gesture(w, [("add_combination", {"type": "ho", "name": "H-O"}),
                         ("wire", lambda res: {"src": _row(g, "Ha").blocks[-1], "src_port": "image",
                                               "dst": res[0].created["block"], "dst_port": "Hα", "channel": "R"}),
                         ("wire", lambda res: {"src": _row(g, "OIII").blocks[-1], "src_port": "image",
                                               "dst": res[0].created["block"], "dst_port": "O III", "channel": "GB"})])
        recording.check(w.p)
    finally:
        w.close()


@pytest.mark.skipif(RECORD, reason="the new way only")
def test_post_wire_set_follows_every_version(qapp, tmp_path):
    """Engine review 3, with the model's Post: two versions; a layer deleted in version 2 only leaves the graph's
    wire; the wire removed in the graph takes the image out of both versions."""
    from astro_gpu.graphcore.post import post_actions, shared_images, wired_images
    new = new_way(tmp_path / "side", next(x for x in GALLERY if x["key"] == "colour_min"), FAKE[:1])
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
