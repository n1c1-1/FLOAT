"""The new graph page's words: every English text it shows has its French, its catalogue changes no other view's
French, and the page shown in French says the owner's words."""
from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3] / "astro_gpu" / "client"
FR = ROOT / "locale" / "fr"


def _shown_texts() -> dict[str, str]:
    found: dict[str, str] = {}
    for f in sorted((ROOT / "graphview").glob("*.py")):
        for node in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
            if not (isinstance(node, ast.Call) and node.args):
                continue
            fn = node.func
            name = fn.id if isinstance(fn, ast.Name) else fn.attr if isinstance(fn, ast.Attribute) else ""
            args = node.args[1:3] if name == "plural" else node.args[:1] if name == "_" else []
            for a in args:
                if isinstance(a, ast.Constant) and isinstance(a.value, str) and a.value:
                    found.setdefault(a.value, f"{f.name}:{node.lineno}")
    from astro_gpu.client.graphview import page, style
    for _k, word, _r, tip in page.POSITIONS:
        found.setdefault(word, "POSITIONS")
        found.setdefault(tip, "POSITIONS")
    for title, _kinds in style.GROUPS:
        found.setdefault(title, "GROUPS")
    for t in style.KIND_TITLE.values():
        found.setdefault(t, "KIND_TITLE")
    return found


def _catalogue(skip=()):
    cat: dict[str, str] = {}
    for f in sorted(FR.glob("*.json")):
        if f.name not in skip:
            cat.update(json.loads(f.read_text(encoding="utf-8")))
    return cat


def test_every_text_of_the_page_has_its_french():
    cat = _catalogue()
    missing = {t: where for t, where in _shown_texts().items() if t not in cat}
    assert not missing, missing


def test_the_page_catalogue_changes_no_other_view():
    mine = json.loads((FR / "graphview.json").read_text(encoding="utf-8"))
    others = _catalogue(skip=("graphview.json",))
    changed = {k: (v, others[k]) for k, v in mine.items() if k in others and others[k] != v}
    assert not changed, changed
    assert all(isinstance(v, str) and v for v in mine.values())


def test_the_page_in_french(qapp):
    from astro_gpu.client import i18n
    from astro_gpu.client.graphview.gestures import pump
    from astro_gpu.client.graphview.page import GraphPage
    i18n.set_language("fr")
    try:
        p = GraphPage()
        p.resize(1200, 700)
        p.show()
        pump()
        assert [b.text() for b in p.positions.values()] == ["Blocs", "Replié", "Tableau"]
        p.bridge.act("add_row", kind="session")
        assert p.canvas.toast.plain.startswith("« Session 1 »") or "Session 1" in p.canvas.toast.plain
        p.palette.finish()
        assert p.palette.cards["stacking"].title.text() == "Empilement"
        p.close()
    finally:
        i18n.set_language("en")
