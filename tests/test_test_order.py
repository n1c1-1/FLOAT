"""ASTRO_TEST_ORDER (tests/conftest.py): the suite run in reverse or shuffled order, a module's tests kept together."""
from __future__ import annotations

from types import SimpleNamespace

import conftest as cf


def _items():
    return [SimpleNamespace(nodeid=f"tests/{m}.py::t{i}") for m in ("a", "b", "c") for i in range(3)]


def test_reverse_runs_the_modules_and_their_tests_last_to_first():
    got = [x.nodeid for x in cf.reorder(_items(), "reverse")]
    assert got == [x.nodeid for x in reversed(_items())]


def test_random_keeps_each_module_together_and_is_the_same_for_one_seed():
    a = [x.nodeid for x in cf.reorder(_items(), "random:7")]
    assert a == [x.nodeid for x in cf.reorder(_items(), "random:7")]
    assert sorted(a) == sorted(x.nodeid for x in _items())
    mods = [n.split("::")[0] for n in a]
    assert all(mods[i] == mods[i + 1] or mods[i + 1] not in mods[:i + 1] for i in range(len(mods) - 1))


def test_no_order_asked_keeps_the_collection_order():
    items = _items()
    assert cf.reorder(items, "") is items
