"""A new version of a block's method is not a setting.

A result exists: it is used. A change of setting makes a block recompute; a new version of its method does not: the
result keeps its key, stays the block's result for the blocks after it and for Post, and says "computed with an earlier
version" (from, to). The user decides: "Recompute (current version)" recomputes the block with today's code and, as
for a setting change, what follows it. The new result takes the earlier one's place under the same key: each result
records the stamp of the results it read, and a block made from a result recomputed since is to recompute (an
interrupted recompute too). A result recorded before the stamps counts as matching.

The key holds a LABEL of the stage, frozen for ever (``KEY_LABELS`` below pins every one); the method's version lives
in the result's manifest.

On the toy stages of tests/test_api.py (inventory -> stack -> b (workflow 2 only) -> c -> e), run for real.
"""
from __future__ import annotations

import json

import pytest

from astro_gpu import api
from astro_gpu.core.store import Store, key_rules
from tests.test_api import RULES, run_inproc, state, toy  # noqa: F401  (fixtures by name)

WHY = ("a stage's key label never changes: it is part of the key of every result of the stage and of every stage "
       "after it, so every project of every user would recompute. To say that the method changed, raise the stage's "
       "`version` (the algorithm version): the results stay valid and say 'computed with an earlier version'.")

#: every stage's label in the keys, as it was the day the labels were frozen (16 November 2026)
KEY_LABELS = {
    "inventory": "3", "calibrate": "11", "register": "6", "stack": "8", "astrometry": "5", "reference": "2",
    "catalogs": "5", "surveys": "2", "color": "6", "denoise": "4-2ea85a0755", "gradient": "5", "render_simple": "13",
    "star_removal": "4-9b974a78cb", "sessions": "1", "image": "1", "align_on": "2", "combine_sho": "9",
    "combine_ho": "9", "combine_rgb": "3", "combine_lrgb": "3", "combine_recipe": "2", "deconv": "6",
    "synth_stars": "2"}
#: the labels that depend on the settings (and, for a monochrome workflow, on its count of light frames): every case
STACK_LABELS = {("bayer", "native", None): "8", ("classic", "native", None): "8+classic2",
                ("bayer", "integration", None): "8+classic2", ("bayer", "native", 5): "8+weaksky1",
                ("classic", "native", 5): "8+classic2+weaksky1", ("classic", "native", 6): "8+classic2",
                ("bayer", "native", 0): "8", ("bayer", "integration", 1): "8+classic2+weaksky1"}
DENOISE_LABELS = {("pretrained", None): "4-2ea85a0755", ("pretrained", 29): "4-2ea85a0755+mono-65c32fea3d",
                  ("pretrained", 30): "4-2ea85a0755", ("pretrained", 0): "4-2ea85a0755",
                  ("adapted", 5): "4-2ea85a0755", (None, 5): "4-2ea85a0755+mono-65c32fea3d"}


# ---------------------------------------------------------------------------------------------- the frozen labels

def test_the_key_label_of_every_stage_is_pinned():
    rules = key_rules()
    got = {n: rules.key_label(n) for n in rules.chains.stages}
    assert got == KEY_LABELS, f"{WHY} A new stage keeps the label '1' for ever and is added to this table."


def test_the_labels_that_follow_the_settings_are_pinned_too():
    rules = key_rules()
    for (method, rej, lights), label in STACK_LABELS.items():
        v = {"method": method, "drizzle_rejection": rej}
        assert rules.key_label("stack", v, lights) == label, f"stack {method}/{rej}/{lights}: {WHY}"
    for (method, lights), label in DENOISE_LABELS.items():
        v = {} if method is None else {"method": method}
        assert rules.key_label("denoise", v, lights) == label, f"denoise {method}/{lights}: {WHY}"
    assert rules.key_label("star_removal", {}, None) == "4-9b974a78cb", WHY


def test_a_new_version_moves_no_label_and_no_key(monkeypatch, toy):  # noqa: F811
    rules = key_rules()
    before = {n: rules.key_label(n) for n in rules.chains.stages}
    for st in rules.chains.stages.values():
        monkeypatch.setattr(type(st), "version", "999")
    assert {n: rules.key_label(n) for n in rules.chains.stages} == before
    assert all(rules.algorithm_version(n) == "999" for n in rules.chains.stages)
    # the network files' identity is part of the method's version only
    from astro_gpu.stages import denoise, stack
    monkeypatch.setattr(denoise, "MODEL_MONO_ID", "another")
    monkeypatch.setattr(stack, "CLASSIC_VERSION", "77")
    assert rules.key_label("denoise", {"method": "pretrained"}, 5) == "4-2ea85a0755+mono-65c32fea3d"
    assert rules.algorithm_version("denoise", {"method": "pretrained"}, 5).endswith("+mono-another")
    assert rules.key_label("stack", {"method": "classic"}, None) == "8+classic2"
    assert rules.algorithm_version("stack", {"method": "classic"}, None) == "999+classic77"
    # and in a project: the keys of the toy stages
    keys = {(w.id, s): api.stage_key(toy, w.id, s) for w in toy.workflows for s in api.chain(toy, w.id)}
    _bump(monkeypatch, toy, "stack", "2")
    assert {(w.id, s): api.stage_key(toy, w.id, s) for w in toy.workflows for s in api.chain(toy, w.id)} == keys


# ---------------------------------------------------------------------------------------------- helpers

def _bump(monkeypatch, p, stage: str, version: str) -> None:
    """The stage's method has a new version (in the rules the API reads and in those the runs of the tests use)."""
    for rules in (api._ctx(p).rules(), RULES):
        monkeypatch.setattr(type(rules.chains.stages[stage]), "version", version)
    api.refresh(p)


def _computed(p):
    status, _ = run_inproc(p, api.plan_all(p))
    assert status == "done"
    st = api.states(p)
    assert {x.state for x in st.values()} == {"done"} and all(x.earlier is None for x in st.values())
    return st


def _manifests(p):
    store = Store(p.root)
    return [d / "manifest.json" for sd in store.path.iterdir() if sd.is_dir() and not sd.name.startswith("_")
            for d in sd.iterdir() if d.is_dir() and not d.name.endswith(".tmp")]


def _strip_stamps(p) -> None:
    """The results as a version before the stamps recorded them."""
    for f in _manifests(p):
        m = json.loads(f.read_text(encoding="utf-8"))
        for k in ("stamp", "upstream", "replaced", "algorithm_version"):
            m.pop(k, None)
        f.write_text(json.dumps(m), encoding="utf-8")
    api.refresh(p)


def _states(p):
    return {k: v.state for k, v in api.states(p).items()}


# ---------------------------------------------------------------------------------------------- a new version

def test_a_new_version_leaves_every_block_done_and_says_which_one_is_earlier(monkeypatch, toy):  # noqa: F811
    p = toy
    before = _computed(p)
    _bump(monkeypatch, p, "b", "2")
    st = api.states(p)
    assert {x.state for x in st.values()} == {"done"}                       # nothing Outdated, nothing to recompute
    assert st[(2, "b")].earlier == ("1", "2") and st[(2, "b")].key == before[(2, "b")].key
    assert [k for k, x in st.items() if x.earlier] == [(2, "b")]           # the blocks after it are plain results
    assert not api.plan_all(p) and api.cost(p).jobs == 0                    # a Run recomputes nothing
    # the blocks after it, and Post, read it as any result
    assert api.stage_outputs(p, 2, "b") is not None and api.stage_outputs(p, 2, "e") is not None
    assert api._dir(p, 2, "e") == Store(p.root).dir("e", st[(2, "e")].key)
    assert not api.plan_to_block(p, 2, "output")


def test_a_result_recorded_before_the_stamps_is_valid_and_says_its_earlier_version(monkeypatch, toy):  # noqa: F811
    p = toy
    _computed(p)
    _strip_stamps(p)
    assert set(_states(p).values()) == {"done"} and not api.plan_all(p)
    _bump(monkeypatch, p, "stack", "7")                    # its version is read from the key parts it kept
    st = api.states(p)
    assert {x.state for x in st.values()} == {"done"} and not api.plan_all(p)
    assert st[(1, "stack")].earlier == ("1", "7") and st[(2, "stack")].earlier == ("1", "7")
    assert st[(2, "c")].earlier is None


def test_a_setting_change_still_makes_the_block_and_what_follows_recompute(monkeypatch, toy):  # noqa: F811
    p = toy
    _computed(p)
    _bump(monkeypatch, p, "b", "2")
    api.set_param(p, 2, "b", "steps", 12)
    st = api.states(p, update=False)
    assert [st[(2, s)].state for s in ("stack", "b", "c", "e")] == ["done", "stale", "stale", "stale"]
    assert st[(2, "b")].reason == api.REASON_SETTINGS and st[(2, "c")].reason == api.REASON_UPSTREAM
    assert [j.stage for j in api.plan_all(p).jobs] == ["b", "c", "e"]
    assert {st[(1, s)].state for s in ("inventory", "stack", "c", "e")} == {"done"}


# ---------------------------------------------------------------------------------------------- recompute

def test_recompute_current_version_recomputes_the_block_and_what_follows(monkeypatch, toy):  # noqa: F811
    p = toy
    before = _computed(p)
    _bump(monkeypatch, p, "b", "2")
    plan = api.plan_recompute(p, 2, "denoise")
    assert [(j.stage, j.wf, j.redo) for j in plan.jobs] == [("b", [2], True), ("c", [2], True), ("e", [2], True)]
    assert plan.seconds > 0                                                  # its cost, said before it runs
    assert all(j.key == before[(2, j.stage)].key for j in plan.jobs)       # the same keys: the results are replaced
    status, _ = run_inproc(p, plan)
    assert status == "done"
    st = api.states(p)
    assert {x.state for x in st.values()} == {"done"} and all(x.earlier is None for x in st.values())
    store = Store(p.root)
    assert store.manifest("b", st[(2, "b")].key)["version"] == "2"
    assert len(list(store.trash.iterdir())) == 3                            # the earlier results, kept in the trash
    assert not api.plan_all(p)
    # the block shared with another workflow: both workflows' followers are recomputed
    _bump(monkeypatch, p, "stack", "3")
    plan = api.plan_recompute(p, 1, "stacking")
    assert sorted((j.stage, tuple(j.wf)) for j in plan.jobs) == [
        ("b", (2,)), ("c", (1,)), ("c", (2,)), ("e", (1,)), ("e", (2,)), ("stack", (1, 2))]
    assert plan.jobs[0].stage == "stack" and all(j.redo for j in plan.jobs)


@pytest.mark.parametrize("old_manifests", [False, True])
def test_an_interrupted_recompute_leaves_what_follows_to_recompute(monkeypatch, toy, old_manifests):  # noqa: F811
    p = toy
    _computed(p)
    if old_manifests:
        _strip_stamps(p)                                   # a project computed before the stamps existed
    _bump(monkeypatch, p, "b", "2")
    plan = api.plan_recompute(p, 2, "denoise")
    plan.jobs = plan.jobs[:1]                              # the run stops once b is recomputed
    status, _ = run_inproc(p, plan)
    assert status == "done"
    st = api.states(p)
    assert st[(2, "b")].state == "done" and st[(2, "b")].earlier is None
    assert st[(2, "c")].state == "stale" and st[(2, "c")].reason == api.REASON_UPSTREAM_REDONE
    assert st[(2, "e")].state == "stale" and st[(2, "e")].reason == api.REASON_UPSTREAM_REDONE
    assert {st[(1, s)].state for s in ("inventory", "stack", "c", "e")} == {"done"}     # the other workflow: untouched
    # its earlier result is still the one shown meanwhile
    assert api.stage_outputs(p, 2, "c", "latest") is not None
    plan = api.plan_all(p)
    assert [(j.stage, j.wf, j.redo) for j in plan.jobs] == [("c", [2], True), ("e", [2], True)]
    assert run_inproc(p, plan)[0] == "done"
    assert set(_states(p).values()) == {"done"} and not api.plan_all(p)


def test_a_result_made_again_for_its_lost_data_keeps_what_follows_valid(toy, tmp_path):  # noqa: F811
    """The inventory made again because its frames moved is the same result: nothing after it is to recompute."""
    import shutil
    p = toy
    _computed(p)
    moved = tmp_path / "moved"
    shutil.move(str(tmp_path / "raw" / "lights"), str(moved))
    p.sessions[0].lights = [str(moved)]
    api.set_param(p, 1, "stack", "kappa", 6.0)
    api.refresh(p, scan=True)
    plan = api.plan_all(p)
    assert [(j.stage, j.rerun, j.redo) for j in plan.jobs][0] == ("inventory", True, False)
    assert run_inproc(p, plan)[0] == "done"
    assert set(_states(p).values()) == {"done"} and not api.plan_all(p)     # workflow 2's b, c, e: still valid


# ---------------------------------------------------------------------------------------------- unreadable

def test_an_earlier_result_today_s_code_cannot_read_asks_to_be_recomputed(monkeypatch, toy):  # noqa: F811
    p = toy
    st = _computed(p)
    _bump(monkeypatch, p, "b", "2")
    f = Store(p.root).dir("b", st[(2, "b")].key) / "b.json"
    f.write_text(json.dumps({"count": 10}), encoding="utf-8")              # what version 1 wrote: today's c cannot read it
    api.set_param(p, 2, "c", "k", 2.0)
    status, _ = run_inproc(p, api.plan_all(p))
    assert status == "failed"
    c = api.states(p)[(2, "c")]
    assert c.state == "failed" and "which this version may no longer read" in c.reason
    assert "the Denoise block" in c.reason and "1 -> 2" in c.reason and "“Recompute (current version)”" in c.reason


def test_the_words_of_an_unreadable_result_name_its_block_and_the_command():
    """The owner's HaRGB on the stars of an earlier Stars/background: the words name the block as the graph shows it
    (not its stage, "Star removal") and the command that recomputes it."""
    from astro_gpu.core.runner2 import unreadable_words
    msg = unreadable_words(key_rules().chains, "star_removal", "3-9b974a78cb", "6-9b974a78cb", "Exception: boom")
    assert msg.startswith("the result of the Stars/background block was computed with an earlier version "
                          "(3-9b974a78cb -> 6-9b974a78cb)")
    assert "“Recompute (current version)” on that block" in msg and msg.endswith("(Exception: boom)")


# ---------------------------------------------------------------------------------------------- the golden tool

def test_the_golden_tool_reports_an_earlier_version_apart_from_outdated(monkeypatch, toy):  # noqa: F811
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    try:
        import graph_golden
    finally:
        sys.path.pop(0)
    p = toy
    _computed(p)
    api.save(p)
    _bump(monkeypatch, p, "b", "2")
    monkeypatch.setattr("astro_gpu.core.project2.load", lambda folder: p)    # (the toy rules are tied to it)
    r = graph_golden.stale_project(p.root)
    assert r["stale"] == [] and r["done"] == r["stages"]
    assert [(x["workflow"], x["stage"], x["kind"], x["why"]) for x in r["earlier"]] == [
        (2, "b", "earlier version", "version 1 -> 2")]
    # a version alone is never "Outdated"; a setting is
    old = {"s": "b", "v": "1", "p": {}, "u": {"stack": "k"}, "x": None}
    assert graph_golden.stale_why(old, {**old, "v": "2"}) == ("earlier version", "version 1 -> 2")
    assert graph_golden.stale_why(old, {**old, "v": "2", "p": {"steps": 12}})[0] == "settings"


def test_recompute_of_a_calibration_makes_its_masters_again(tmp_path, monkeypatch):
    """Recompute (current version) on Calibration: the masters are made again, not copied from the earlier result
    (a new hot-pixel map with the same settings came out 12.5 % where it should be 3.3 %). A frame cache merely lost
    is still rebuilt with the result's own masters."""
    from types import SimpleNamespace

    from astro_gpu.core import runner2
    from astro_gpu.stages.calibrate import MASTER_FILES, MASTERS, CalibrateParams, result_params
    old = tmp_path / "old"
    old.mkdir()
    for n in MASTER_FILES:
        (old / n).write_bytes(b"x")
    params = CalibrateParams()
    store = SimpleNamespace(dir=lambda s, k: old,
                            manifest=lambda s, k: {"params": result_params(params), "metrics": {}, "version": "11"})
    monkeypatch.setattr(runner2, "read_manifest", lambda project, d: {})

    def give(redo: bool) -> bool:
        tmp = tmp_path / f"job_{redo}"
        tmp.mkdir()
        pipe = SimpleNamespace(project=SimpleNamespace(stage_dir=lambda s: tmp),
                               effective_params=lambda st: (params, None, None))
        worker = SimpleNamespace(store=store, _read_stamps=lambda job: {},
                                 journal=SimpleNamespace(emit=lambda *a, **k: None))
        st = SimpleNamespace(deps=(), version="12")
        runner2._Worker._give_masters(worker, pipe, st, SimpleNamespace(stage="calibrate", key="k", redo=redo))
        return (tmp / MASTERS).is_file()

    assert give(redo=False)              # a lost cache: the result's own masters, whatever version made them
    assert not give(redo=True)           # Recompute (current version): a full calibration
