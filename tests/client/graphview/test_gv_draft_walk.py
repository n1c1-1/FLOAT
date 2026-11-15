"""Every setting of every block's panel, on a computed colour template and a computed S-H-O: changed, it becomes a draft
(nothing outdated) or applies at once (it changes no result); the draft's Apply and Try variant never raise (a variant
the graph cannot make is greyed with its reason, a refusal is said); Parallel workers and the like change no key. The
synthetic worlds of tests/client/graphview/legacy.py, made "computed" as test_gv_loop_guard does."""
from __future__ import annotations

import pytest
from PySide6.QtWidgets import QMessageBox

from astro_gpu.client import prefs
from astro_gpu.client.blockdraft import DraftBar
from astro_gpu.client.panel import CompactRow, MethodCard
from tests.client.graphview.test_gv_loop_guard import _computed, _world
from tests.client.pagehelp import qapp  # noqa: F401  (the fixture, by name)



@pytest.fixture(autouse=True)
def _own_settings(tmp_path, monkeypatch):
    from astro_gpu.client import i18n
    monkeypatch.setenv("ASTRO_GPU_SETTINGS", str(tmp_path / "settings.json"))
    prefs.load()
    i18n.set_language("en")
    for kind in ("information", "warning", "critical", "question"):
        monkeypatch.setattr(QMessageBox, kind, lambda *a, **k: pytest.fail(f"a message box: {a[1:3]}"))


def _other(r):
    """Another valid value of a settings row, None when it has none."""
    v = r.get("value")
    if r.get("choices"):
        off = set((r.get("disabled_choices") or {}))
        return next((c for c, _t in r["choices"] if c != v and c not in off), None)
    if isinstance(v, bool):
        return not v
    if isinstance(v, (int, float)):
        lo, hi, step = r.get("min"), r.get("max"), r.get("step") or (1 if isinstance(v, int) else 0.1)
        for nv in (v + step, v - step, v + 2 * step):
            nv = int(round(nv)) if isinstance(v, int) else round(nv, 6)
            if (lo is None or nv >= lo) and (hi is None or nv <= hi) and nv != v:
                return nv
    return None


def _states(m):
    return {k: (v.state, v.key) for k, v in m._states.items()}


def _bar(panel):
    return next(iter(panel.findChildren(DraftBar)), None)


@pytest.mark.parametrize("kind", ["colour", "sho"])
def test_every_setting_of_every_block_is_a_draft_whose_apply_and_variant_never_raise(qapp, tmp_path, kind):
    W = _world(tmp_path, kind)
    try:
        m, pn = W.m, W.page.panel
        _computed(m)
        W.settle()
        told = []
        pn.toast.connect(told.append)
        seen = drafts = 0
        blocks = [b for b, blk in W.b.graph.blocks.items() if blk.kind not in ("session", "post")]
        for b in blocks:
            if b not in W.b.graph.blocks:
                continue
            W.choose(b)
            W.settle(2)
            pn.expert = True
            pn.refresh(pn.node)
            W.settle(2)
            pn.refresh(pn.node)
            node = pn.node
            if node is None:
                continue
            keys = [r.r["key"] for r in pn.findChildren(CompactRow) if r.r.get("kind") != "new"]
            for key in keys:
                for how in ("variant", "apply"):
                    row = next((r for r in pn.findChildren(CompactRow) if r.r.get("key") == key), None)
                    if row is None or row.r.get("locked"):
                        break
                    nv = _other(row.r)
                    if nv is None:
                        break
                    seen += 1
                    before, undo = _states(m), len(m._undo)
                    pn.set_setting(row.r, nv)
                    W.settle(1)
                    if not m.pending_change:                 # applied at once: nothing up to date was lost
                        if len(m._undo) > undo:
                            m.undo()
                            W.settle(1)
                        break
                    drafts += 1
                    assert _states(m) == before and len(m._undo) == undo, key     # a draft outdates nothing
                    bar = _bar(pn)
                    assert bar is not None and bar.what.text(), key
                    if how == "variant":
                        if bar.b_variant.isEnabled():
                            assert pn.draft_variant(), (key, told[-1:])
                        else:
                            assert bar.b_variant.toolTip().startswith("No variant: "), key
                            pn.draft_reset()
                    else:
                        pn.draft_apply()
                    W.settle(1)
                    if len(m._undo) > undo:
                        m.undo()
                        W.settle(1)
                    if m.pending_change:
                        pn.draft_reset()
            for card in [c for c in pn.findChildren(MethodCard) if c.ok and not c.on][:1]:
                card.clicked.emit()
                W.settle(1)
                if m.pending_change:
                    drafts += 1
                    assert pn.draft_variant() or pn.draft_reset()
                    W.settle(1)
                    if m.can_undo():
                        m.undo()
                        W.settle(1)
            pn.expert = False
        assert seen > 6 and drafts > 5, (seen, drafts)
    finally:
        W.close()


def test_settings_that_change_no_result_apply_at_once_and_keep_every_key(qapp, tmp_path):
    W = _world(tmp_path, "colour")
    try:
        m = W.m
        _computed(m)
        W.settle()
        q = m.paths[0]
        keys = {st: m.stage_key(q, st) for st in m.all_stages(q)}
        for stage, name in (("calibrate", "workers"), ("register", "workers"), ("stack", "workers")):
            if m.param_info(q, stage, name) is None:
                continue
            v = m.param_value(q, stage, name) or 1
            ch = dict(kind="param", stage=stage, name=name, value=int(v) + 1, suggested=None)
            assert not m.change_cost(ch, [q]).lost, (stage, name)
            m.apply_change([q], ch)
            W.settle(1)
            assert {st: m.stage_key(q, st) for st in m.all_stages(q)} == keys, (stage, name)
            m.undo()
            W.settle(1)
    finally:
        W.close()
