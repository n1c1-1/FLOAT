"""Passing through a decimal field (Tab, a click elsewhere) changes nothing: the field shows the decimals its value has,
and the end of an edit in which nothing was typed sends nothing. A value of 33.33 with a step of 1 showed 33.3 and a
Tab wrote 33.3: an automatic value became fixed, a computed block got a draft. Same in the Selection's thresholds and
in a combination recipe's parameters."""
from __future__ import annotations

from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QFocusEvent
from PySide6.QtWidgets import QApplication, QDoubleSpinBox, QLabel, QVBoxLayout, QWidget

from astro_gpu.client.panel import CompactRow, Panel


def _app():
    return QApplication.instance() or QApplication([])


def _pass_through(e):
    """The user tabs into the field and out of it, typing nothing."""
    QApplication.sendEvent(e, QFocusEvent(QEvent.FocusIn, Qt.TabFocusReason))
    QApplication.sendEvent(e, QFocusEvent(QEvent.FocusOut, Qt.TabFocusReason))
    QApplication.processEvents()


def _row(value, changed):
    return dict(key="x.y", name="y", label="Some value", type="float", value=value, suggested=33.33, step=1,
                min=0, max=100, kind="known", changed=changed, unit="", help="", level="normal")


def test_a_tab_through_a_decimal_setting_sends_nothing_and_shows_its_decimals():
    _app()
    for changed in (False, True):                       # automatic (the suggestion), then fixed by the user
        row = CompactRow(_row(33.33, changed))
        sent = []
        row.changed.connect(lambda r, v: sent.append(v))
        e = row.editor
        assert isinstance(e, QDoubleSpinBox)
        assert abs(e.value() - 33.33) < 1e-9             # shown 33.3 before
        _pass_through(e)
        assert sent == [], changed                       # a Tab wrote 33.3: no draft, an automatic value stays so
        e.setValue(40)                                   # a value the user changes is still sent
        e.editingFinished.emit()
        assert sent == [40.0]


class _Model:
    def thresholds_unit(self):
        return "px"

    def thresholds_active(self, p):
        return True

    def request_selection(self):
        pass

    def state(self, p, stage):
        return "todo"

    def recipe_parameters(self, p):
        return {"k": dict(label="Strength", value=0.333, min=0, max=5, step=0.1)}


class _Panel:
    """What Panel.selection and Panel.recipe_settings read of the panel."""

    def __init__(self):
        self.root = QWidget()
        self.bv = QVBoxLayout(self.root)
        self.s = _Model()
        self._unit_w = 40
        self.proposed = []

    def group_title(self, text):
        return QLabel(text)

    def thresholds_shown(self):
        return 61.7, 2.25                               # 61.7 % (a draft of 0.617), 2.25 px

    def propose_selection(self, t, f):
        self.proposed.append((t, f))

    def draft_recipe(self, name, default):
        return default

    def propose_recipe(self, *a):
        self.proposed.append(a)

    _sel = Panel._sel


def test_a_tab_through_a_selection_threshold_or_a_recipe_parameter_proposes_nothing():
    _app()
    pn = _Panel()
    Panel.selection(pn, None)
    eds = pn.root.findChildren(QDoubleSpinBox)
    assert [round(e.value(), 4) for e in eds] == [61.7, 2.25]
    for e in eds:
        _pass_through(e)
    assert pn.proposed == []
    Panel.recipe_settings(pn, None)
    rec = [e for e in pn.root.findChildren(QDoubleSpinBox) if e not in eds]
    assert len(rec) == 1 and abs(rec[0].value() - 0.333) < 1e-9
    _pass_through(rec[0])
    assert pn.proposed == []
