"""The projects computed before the key labels were frozen: their results are under keys made with EARLIER labels
(a new version of a stage changed its label then). When a stage has no result of today's key, the result of the same
stage whose recorded key parts are today's but for the labels (its own, and the keys of the results it read, each
found the same way) is the block's result, said "computed with an earlier version". The blocks after it, the runs and
Post read it; nothing is written in the project by that lookup (core/store.find_earlier).

On the toy stages of tests/test_api.py, computed, then their labels changed as the versions used to change them."""
from __future__ import annotations

import json
import shutil

from astro_gpu import api
from astro_gpu.core.store import Store
from tests.test_api import RULES, run_inproc, state, toy  # noqa: F401  (fixtures by name)
from tests.test_versions import _computed, _states, _strip_stamps


def _new_labels(monkeypatch, p, **labels: str) -> None:
    """The project was computed when these stages had another label and version (what a version bump did then)."""
    for stage, label in labels.items():
        for rules in (api._ctx(p).rules(), RULES):
            monkeypatch.setattr(type(rules.chains.stages[stage]), "key_label", label)
            monkeypatch.setattr(type(rules.chains.stages[stage]), "version", label)
    api.refresh(p)


def _files(p):
    return sorted((f.relative_to(p.root).as_posix(), f.stat().st_mtime_ns, f.stat().st_size)
                  for f in p.root.rglob("*") if f.is_file())


def _earlier_project(monkeypatch, p, **labels):
    """Computed, recorded as before the stamps and the recorded versions, then the labels of today."""
    before = _computed(p)
    _strip_stamps(p)
    api.save(p)
    _new_labels(monkeypatch, p, **labels)
    return before


def test_results_under_earlier_labels_are_the_blocks_results_and_nothing_is_written(monkeypatch, toy):  # noqa: F811
    p = toy
    before = _earlier_project(monkeypatch, p, stack="2")
    keys = {k: api.stage_key(p, *k) for k in before}
    moved = [k for k in before if keys[k] != before[k].key]
    assert sorted({s for _w, s in moved}) == ["b", "c", "e", "stack"]       # the stack's label and all that read it
    files = _files(p)
    st = api.states(p, update=False)
    assert {x.state for x in st.values()} == {"done"}                       # found: nothing to recompute
    assert all(st[k].key == keys[k] for k in st)                             # each under today's key
    assert st[(1, "stack")].earlier == ("1", "2") and st[(2, "stack")].earlier == ("1", "2")
    assert [k for k, x in st.items() if x.earlier] == [(1, "stack"), (2, "stack")]
    assert not api.plan_all(p) and api.cost(p).jobs == 0
    store = Store(p.root)
    # every reader gets the earlier result's folder: the blocks after it, Post (the Output's folder), Inspect
    assert api._dir(p, 2, "e") == store.path / "e" / before[(2, "e")].key
    assert api.stage_outputs(p, 2, "stack") is not None and api.stage_outputs(p, 1, "c") is not None
    assert not (store.path / "e" / keys[(2, "e")]).exists()
    assert not [i for i in api.cache_usage(p).unused if i.kind == "result"]    # in use: never offered for clearing
    assert _files(p) == files                                                # the lookup wrote nothing in the project
    # and the usual reading of the states neither: state.json already names these results, by their own keys
    assert {x.state for x in api.states(p).values()} == {"done"} and _files(p) == files
    assert api.stage_outputs(p, 2, "stack", "previous") is None              # it is the current result, not a previous one


def test_a_run_reads_them_and_a_setting_change_recomputes_only_what_follows(monkeypatch, toy):  # noqa: F811
    p = toy
    before = _earlier_project(monkeypatch, p, stack="2", b="3")
    assert set(_states(p).values()) == {"done"}
    api.set_param(p, 2, "c", "k", 2.0)
    plan = api.plan_all(p)
    assert [(j.stage, j.wf) for j in plan.jobs] == [("c", [2]), ("e", [2])]
    assert run_inproc(p, plan)[0] == "done"                                  # c read the earlier b and stack
    st = api.states(p)
    assert {x.state for x in st.values()} == {"done"} and not api.plan_all(p)
    c = Store(p.root).dir("c", st[(2, "c")].key)
    assert json.loads((c / "c.json").read_text())["steps"] == 10 and c.name == st[(2, "c")].key != before[(2, "c")].key


def test_a_result_with_other_settings_or_other_inputs_is_not_taken(monkeypatch, toy):  # noqa: F811
    p = toy
    before = _earlier_project(monkeypatch, p, stack="2")
    assert set(_states(p).values()) == {"done"}
    api.set_param(p, 2, "b", "steps", 12)
    st = api.states(p, update=False)
    assert [st[(2, s)].state for s in ("inventory", "stack")] == ["done", "done"]
    assert all(st[(2, s)].state == "stale" for s in ("b", "c", "e"))         # another setting: to compute
    # Outdated as any block whose setting changed: the result it had is still shown meanwhile
    assert st[(2, "b")].reason == api.REASON_SETTINGS and st[(2, "b")].previous_key == before[(2, "b")].key
    assert api.stage_outputs(p, 2, "b", "previous") is not None and api.stage_outputs(p, 2, "b") is None
    assert {st[(1, s)].state for s in ("inventory", "stack", "c", "e")} == {"done"}
    assert [j.stage for j in api.plan_all(p).jobs] == ["b", "c", "e"]


def test_recompute_current_version_of_a_result_found_under_an_earlier_label(monkeypatch, toy):  # noqa: F811
    p = toy
    before = _earlier_project(monkeypatch, p, stack="2")
    plan = api.plan_recompute(p, 1, "stacking")
    assert sorted((j.stage, tuple(j.wf)) for j in plan.jobs) == [
        ("b", (2,)), ("c", (1,)), ("c", (2,)), ("e", (1,)), ("e", (2,)), ("stack", (1, 2))]
    assert run_inproc(p, plan)[0] == "done"
    st = api.states(p)
    assert {x.state for x in st.values()} == {"done"} and all(x.earlier is None for x in st.values())
    store = Store(p.root)
    assert (store.path / "stack" / st[(1, "stack")].key / "manifest.json").is_file()   # under today's key now
    assert (store.path / "stack" / before[(1, "stack")].key).is_dir()                   # the earlier one is not touched
    assert not api.plan_all(p)


def test_a_result_made_again_for_its_lost_data_still_stands_for_the_earlier_one(monkeypatch, toy, tmp_path):  # noqa: F811
    """The inventory, found under an earlier label, lost its frames' place and is made again under today's key for
    one workflow's new stack: the other workflow's results, made from the earlier inventory, are still found."""
    p = toy
    _earlier_project(monkeypatch, p, inventory="2")
    assert set(_states(p).values()) == {"done"}
    moved = tmp_path / "moved"
    shutil.move(str(tmp_path / "raw" / "lights"), str(moved))
    p.sessions[0].lights = [str(moved)]
    api.set_param(p, 1, "stack", "kappa", 6.0)
    api.refresh(p, scan=True)
    plan = api.plan_all(p)
    assert [(j.stage, j.wf, j.redo) for j in plan.jobs] == [
        ("inventory", [1, 2], False), ("stack", [1], False), ("c", [1], False), ("e", [1], False)]
    assert run_inproc(p, plan)[0] == "done"
    assert set(_states(p).values()) == {"done"} and not api.plan_all(p)      # workflow 2's stack, b, c, e: still found
