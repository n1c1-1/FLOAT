"""The stored model's fields are listed once, by their classes: every field of a block, row, wire, tap or project
survives a copy (clone) and a save and read (to_dict / from_dict), a field added later too; the files saved keep the
same keys in the same order, and the shipped templates read then written give the same bytes."""
import copy
import dataclasses
import glob
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from fixtures import Doc  # noqa: E402
from astro_gpu.graphcore import model as gm  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TEMPLATES = sorted(glob.glob(os.path.join(ROOT, "astro_gpu", "resources", "templates", "*", "*.astroproj")))

#: the keys a saved graph holds, in their order (the files written before must read and write the same)
KEYS = {
    "project": ["name", "target", "rows", "blocks", "wires", "common", "next_id", "nights", "excluded",
                "calibrate_together", "next_wid", "order", "taps"],
    "row": ["id", "name", "blocks", "nights", "fork", "wid", "all_nights", "sessions", "output", "extra", "unset",
            "pauses", "trial", "options", "end", "legacy", "calib_node", "calib_fork", "calib", "align"],
    "block": ["id", "kind", "settings", "off", "pause", "name", "files", "engine", "method", "fork", "params", "sid",
              "own", "source"],
    "wire": ["id", "src", "dst", "active", "wid", "channel", "image"],
    "tap": ["id", "block", "name", "wid", "all_nights", "sessions", "output", "extra", "unset", "pauses", "trial",
            "options", "end", "legacy", "auto"],
}


def _graph():
    doc = Doc.from_template("filters_complete")
    r = doc.project.rows[0]
    ok = doc.apply("send_to_post", block=next(b for b in r.blocks if doc.project.blocks[b].kind == "gradient"))
    assert ok.ok, ok.reason_text()
    return doc.project


def test_the_saved_keys_and_their_order_are_those_of_the_files_written_before():
    d = gm.to_dict(_graph())
    assert list(d) == KEYS["project"]
    assert list(d["rows"][0]) == KEYS["row"]
    assert list(next(iter(d["blocks"].values()))) == KEYS["block"]
    assert list(d["wires"][0]) == KEYS["wire"]
    assert d["taps"] and list(d["taps"][0]) == KEYS["tap"]


@pytest.mark.parametrize("path", TEMPLATES, ids=[os.path.basename(x) for x in TEMPLATES])
def test_a_shipped_template_read_then_written_gives_the_same_bytes(path):
    raw = open(path, "rb").read().replace(b"\r\n", b"\n")
    d = json.loads(raw)
    g = gm.from_dict(copy.deepcopy(d["graph"]))
    again = json.dumps({**d, "graph": gm.to_dict(g)}, indent=1, sort_keys=True, ensure_ascii=False) + "\n"
    assert again.encode("utf-8") == raw
    assert gm.model_json(gm.clone(g)) == gm.model_json(g)


def test_every_field_and_a_field_added_later_survive_clone_and_a_save_and_read(monkeypatch):
    p = _graph()
    added = {}
    for cls in (gm.Block, gm.Row, gm.Wire, gm.Tap, gm.Project):     # each class given one more field
        sub = dataclasses.make_dataclass(cls.__name__, [("later", object, dataclasses.field(default=None))],
                                         bases=(cls,))
        monkeypatch.setattr(gm, cls.__name__, sub)
        added[cls] = sub

    def again(x, n):
        return added[type(x)](**{f.name: getattr(x, f.name) for f in dataclasses.fields(x)}, later={"n": [n]})
    q = again(p, 0)
    q.rows = [again(r, i) for i, r in enumerate(p.rows)]
    q.blocks = {k: again(b, i) for i, (k, b) in enumerate(p.blocks.items())}
    q.wires = [again(w, i) for i, w in enumerate(p.wires)]
    q.taps = [again(t, i) for i, t in enumerate(p.taps)]
    for back in (gm.clone(q), gm.from_dict(json.loads(gm.model_json(q)))):
        assert back == q                                           # every field, the added one too
        assert back.later == {"n": [0]} and back.later is not q.later
        for a, b in zip(back.rows, q.rows):
            assert a.later == b.later and a.later is not b.later and a.blocks is not b.blocks
        assert all(back.blocks[k].later == b.later for k, b in q.blocks.items())
        assert [w.later for w in back.wires] == [w.later for w in q.wires]
        assert [t.later for t in back.taps] == [t.later for t in q.taps]
