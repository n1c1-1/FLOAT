"""The question of a change of setting names everything it computes again, the workflows that read the one changed
included: a stacking setting of a filter of the S-H-O names the S-H-O too (a beta tester read "Computes again:
Stacking, Astrometry", the S-H-O left out, 3/3). Every stage is taken as computed here (no run in a unit test)."""
from __future__ import annotations

from tests.client.test_swap_smoke import env, made_from, qapp, settle, win  # noqa: F401  (fixtures by name)


def test_a_filters_setting_names_the_combination_reading_it(win, env, monkeypatch):
    made_from(win, env, "sho_rgb")
    s = win.s
    combo = next(q for q in s.paths if isinstance(q.source, dict) and "fusion" in q.source and q.name == "S-H-O")
    sii = next(q for q in s.paths if q.name == "SII")
    monkeypatch.setattr(type(s), "state", lambda self, p, stage: "done")
    ch = {"kind": "param", "stage": "stack", "name": "kappa", "value": 2.5, "suggested": None}
    cost = s.change_cost(ch, [sii])
    assert cost.lost, cost
    assert any(w == combo.pid for w, _st in cost.stages), cost.stages
    assert "S-H-O" in cost.blocks, cost.blocks
    assert not env["errors"], env["errors"]
