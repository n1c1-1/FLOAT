"""A change made in the graph's panel goes to the block the panel shows. A beta tester had the main SII Gradient chosen
on the graph (chosen before, and clicked again) while Inspect had shown the stacking of a variant meanwhile: Replace
changed the variant's Gradient, the model's scope being still the variant's."""
from __future__ import annotations

from astro_gpu.client.model import METHOD_OPT, method_of, method_value
from tests.client.test_swap_smoke import env, made_from, qapp, settle, win  # noqa: F401  (fixtures by name)


def test_a_method_goes_to_the_workflow_of_the_block_shown(win, env):
    made_from(win, env, "filters_min")
    s, page = win.s, win.graph
    a, b = s.paths[0], s.paths[1]
    bm = s.block_map
    ident = bm.block_of(bm.node_of(a, "fond"))
    page.bridge.choose(ident)
    settle(win, 3)
    s.select(s.paths.index(b), "fond")                   # Inspect showed the other workflow's block meanwhile
    s.scope = [b.pid]
    page.bridge.choose(ident)                            # the same block clicked again on the graph
    settle(win, 3)
    pn = page.panel
    assert pn.node is not None and a in pn.node.paths and b not in pn.node.paths
    before_b = method_of(b, "fond")
    assert method_of(a, "fond") != "auto"
    pn.propose_opt(METHOD_OPT["fond"], method_value("fond", "auto"), "Method: Automatic", "automatic")
    settle(win, 3)
    if s.pending_change:                                  # asked first when results would be computed again
        assert s.pending_change["pids"] == [a.pid]
        pn._replace()
        settle(win, 3)
    a, b = s.view(a.pid), s.view(b.pid)
    assert method_of(a, "fond") == "auto"
    assert method_of(b, "fond") == before_b
    assert not env["errors"], env["errors"]
