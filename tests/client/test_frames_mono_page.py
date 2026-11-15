"""A workflow whose sessions name no filter and whose Inventory read monochrome frames (``api.frames_mono`` True) is
shown as the mono workflow it is: no debayering in its Stacking settings, Color greyed with the engine's reason in the
palette and on its placed block. False (colour frames) or None (no Inventory yet): the page as before."""
from __future__ import annotations

import pytest

from astro_gpu import api
from tests.client.graphview.legacy import World
from tests.client.pagehelp import qapp  # noqa: F401


@pytest.fixture()
def world(qapp, tmp_path):  # noqa: F811
    made = []

    def make(**kw):
        w = World(tmp_path / f"w{len(made)}", **kw)
        made.append(w)
        return w
    yield make
    for w in made:
        w.close()


def _stack_names(m, q) -> set:
    from astro_gpu.client.settings import step_settings
    return {(r.get("stage"), r.get("name")) for _t, rs in step_settings(m, q, "empilement") for r in rs}


def _read(W, answer, monkeypatch):
    """The engine's answer for every workflow, read at the next refresh of the states (never when drawn)."""
    calls = []

    def frames_mono(p, wf):
        calls.append(wf)
        return answer
    monkeypatch.setattr(api, "frames_mono", frames_mono)
    W.m._sync(False)
    W.settle()
    return calls


def test_the_engine_says_nothing_before_the_first_inventory_and_the_page_is_as_before(world):
    W = world(workflows=("Full",))
    m, doc = W.m, W.page.bridge.doc
    q = m.paths[0]
    m._sync(False)
    W.settle()
    assert m.__dict__["_mono_frames"] == {q.pid: None}      # the real call: nothing computed yet
    assert not m.is_mono(q) and doc.greyed(W.one("color")) == ""


@pytest.mark.parametrize("answer", [True, False, None])
def test_mono_frames_without_a_filter_show_a_mono_workflow(world, monkeypatch, answer):
    W = world(workflows=("Full",))
    m, doc = W.m, W.page.bridge.doc
    q = m.paths[0]
    assert not m.p.is_mono(q.pid)                      # its sessions name no filter
    before = _stack_names(m, q)
    calls = _read(W, answer, monkeypatch)
    assert q.pid in calls
    n = len(calls)
    color = W.one("color")
    W.choose(W.block("empilement"))
    ok, why = doc.palette_state("color")
    grey = doc.greyed(color)
    names = _stack_names(m, q)
    assert len(calls) == n                             # drawing and the panel never read it again
    assert not m.is_mono(q, frames=False)              # a chain rebuilt from its options: the definition alone
    if answer is True:
        assert m.is_mono(q)
        assert "colours" in grey and not ok and "colours" in why
        assert ("stack", "classic_debayer") not in names and ("stack", "white_balance") not in names
        assert names < before                          # the colour settings the stack had are gone
    else:
        assert not m.is_mono(q)
        assert grey == "" and (ok or "colours" not in why)
        assert names == before
