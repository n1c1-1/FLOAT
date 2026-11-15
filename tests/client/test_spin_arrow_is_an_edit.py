"""A spin box's arrow (4.00 → 4.50) is an edit as typing then Enter is: the change is sent at once, so a computed
block's draft appears at once. It waited until the field was left."""
from __future__ import annotations

from PySide6.QtWidgets import QApplication, QDoubleSpinBox, QSpinBox

from astro_gpu.client.panel import CompactRow, Panel
from tests.client.test_decimal_fields_pass_through import _Panel, _row


def _app():
    return QApplication.instance() or QApplication([])


def _arrow(e, steps=1):
    e.stepBy(steps)                                     # what the arrow, the Up key and the focused wheel do
    for _i in range(3):
        QApplication.processEvents()


def test_an_arrow_sends_the_setting_at_once():
    _app()
    r = dict(_row(4.0, True), step=0.5, suggested=4.0)
    row = CompactRow(r)
    sent = []
    row.changed.connect(lambda r_, v: sent.append(v))
    assert isinstance(row.editor, QDoubleSpinBox)
    _arrow(row.editor)
    assert sent == [4.5]
    row = CompactRow(dict(_row(4, True), type="int", step=1, suggested=4))
    sent = []
    row.changed.connect(lambda r_, v: sent.append(v))
    assert isinstance(row.editor, QSpinBox)
    _arrow(row.editor, -1)
    assert sent == [3]


def test_an_arrow_on_a_threshold_or_a_recipe_parameter_proposes_at_once():
    _app()
    pn = _Panel()
    Panel.selection(pn, None)
    t = pn.root.findChildren(QDoubleSpinBox)[0]
    _arrow(t)
    assert pn.proposed and round(pn.proposed[-1][0], 4) == 62.7
    Panel.recipe_settings(pn, None)
    rec = [e for e in pn.root.findChildren(QDoubleSpinBox)][-1]
    n = len(pn.proposed)
    _arrow(rec)
    assert len(pn.proposed) == n + 1
