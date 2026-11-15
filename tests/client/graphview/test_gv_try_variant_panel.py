"""« Try variant » answered in the panel shows the variant's block of the same kind (a beta tester's settings panel went
blank, its side icons gone, until he clicked again). On the computed test project of test_gv_legacy_model."""
from __future__ import annotations

from tests.client.graphview.test_gv_legacy_model import (  # noqa: F401  (the fixtures are used by name)
    _edit_kappa,
    env,
    model,
    needs_test_project,
    node,
    page,
    select_node,
    settle,
    world,
)

pytestmark = needs_test_project


def test_try_variant_shows_the_variants_block(world, model, page):
    select_node(page, node(page, "empilement"))
    settle(world["w"], page)
    rows = {r.id for r in page.bridge.graph.rows}
    _edit_kappa(page, 4.0)
    settle(world["w"], page)
    assert model.pending_change
    assert page.answer_change("variant")
    settle(world["w"], page, 4)
    g = page.bridge.graph
    new = [r for r in g.rows if r.id not in rows]
    sel = page.bridge.selected
    assert new and sel in g.blocks and g.blocks[sel].kind == "stacking", sel
    assert page.panel.node is not None
    model.undo()
    settle(world["w"], page)


def test_try_variant_of_a_method_shows_the_variants_gradient(world, model, page):
    from PySide6.QtWidgets import QLabel

    from tests.client.graphview.test_gv_legacy_model import button, method_cards
    select_node(page, node(page, "fond", 1))
    settle(world["w"], page)
    other = next(c for c in method_cards(page) if "Automatic" in c.findChildren(QLabel)[0].text())
    other.clicked.emit()
    settle(world["w"], page)
    button(page.panel, "Try variant").click()
    settle(world["w"], page, 4)
    g = page.bridge.graph
    sel = page.bridge.selected
    assert sel in g.blocks and g.blocks[sel].kind == "gradient", sel
    assert page.panel.node is not None and page.panel.node.key == "fond"
    model.undo()
    settle(world["w"], page)
