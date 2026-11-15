"""One language at a time: the main window is driven offscreen through Home, a blank project, the template gallery, every
template placed, the graph with every kind of block, every block's panel, the table, Inspect on a computed project, Post,
the report, the run window, the menus and the tooltips of the graph's items; every visible text is collected
(tests/client/langscan.py) and none may hold a word of the other language (the words come from the French catalogue:
its keys are the English texts, its values the French ones).

The names the user typed and proper names are allowed; so are the words both languages write the same way.
``SCAN_LANGS`` (default "fr,en") picks the languages, ``SCAN_REPORT`` a file the findings are written to."""
from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if Path("C:/Windows/Fonts").is_dir():
    os.environ.setdefault("QT_QPA_FONTDIR", "C:/Windows/Fonts")
pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication  # noqa: E402

from astro_gpu.client import app as appmod  # noqa: E402
from astro_gpu.client import i18n  # noqa: E402
from tests.client import langscan  # noqa: E402
from tests.client.test_swap_smoke import (  # noqa: E402,F401 - fixtures of the application's smoke test
    env,
    made_from,
    qapp,
    ready,
    settle,
    template,
    win,
)

#: names the scan's own projects carry (typed by the "user"): never flagged
MINE = {"Mine", "Scan", "M 31", "Untitled"} | {f"{p}{k}" for p in ("", "p_") for k in (
    "colour_min", "colour_full", "image", "filters_min", "filters_full", "sho_rgb", "images", "dual_x2")}
#: a text of the application that is allowed in the other language: proper names kept by design
KEPT = set()


class Scan:
    """What the walk found: ``seen`` {text: [steps]}, the findings {(text, word): step}."""

    def __init__(self, lang: str):
        self.lang = lang
        self.vocab = langscan.vocabulary(lang)
        self.mine = set(MINE)                          # the user's own names: the project's workflows and sessions
        self.found: dict[tuple[str, str], str] = {}
        self.count = 0

    def take(self, shown, step: str):
        for s in shown:
            self.count += 1
            if s.text in KEPT or re.match(r"^[A-Za-z]:[/\\]", s.text):     # a file's path is the user's
                continue
            for w in langscan.foreign(s.text, self.lang, self.mine, self.vocab):
                self.found.setdefault((s.text, w), f"{step} / {s.kind} {s.where}")

    def report(self) -> list[str]:
        return [f"{step}: {text!r}  ({word})" for (text, word), step in sorted(self.found.items())]


def _page_texts(w, scan: Scan, step: str):
    QApplication.processEvents()
    scan.take(langscan.texts(w), step)
    scan.take(langscan.menus(w), step + " menus")
    scan.take([langscan.Shown(w.status.text(), "label", None, "status")], step + " status")


def _every_block(w, scan: Scan, step: str):
    """Each block chosen: the settings panel it shows (the window as a whole is scanned by the callers)."""
    page = w.graph
    for bid in list(w.s.stored.blocks):
        page.bridge.choose(bid)
        settle(w, 1)
        scan.take(langscan.texts(page.panel), f"{step} block {w.s.stored.blocks[bid].kind}")
        scan.take([langscan.Shown(w.status.text(), "label", None, "status")], f"{step} status")


def _modes(w, scan: Scan, step: str):
    for mode, name in ((1, "Inspect"), (2, "Post"), (3, "Report"), (0, "Graph")):
        w.set_mode(mode)
        settle(w, 3)
        _page_texts(w, scan, f"{step} {name}")
    page = w.graph
    for key in ("table", "folded", "blocks"):
        page.set_display(key)
        settle(w, 2)
        _page_texts(w, scan, f"{step} {key}")


def _all_kinds(w, scan: Scan, step: str):
    """A drop of every kind of block of the palette where the graph lets it, one after the other."""
    page = w.graph
    done = []
    for kind in list(page.palette.cards):
        try:
            places = [p for p in page.bridge.places(kind=kind) if p.lit]
        except Exception:  # noqa: BLE001 - a kind with no place here
            places = []
        if not places:
            continue
        res = page.bridge.drop(places[0])
        settle(w, 2)
        if res.ok:
            done.append(kind)
    _page_texts(w, scan, f"{step} {len(done)} kinds placed")
    _every_block(w, scan, step + " kinds")
    return done


def _every_recipe(w, scan: Scan, step: str):
    """Each combination recipe of the application placed from the Combination card's list: its name in the list and
    its tooltip, the block, its panel."""
    from astro_gpu.graphcore import combos
    page = w.graph
    pal = page.palette
    m = pal.combination_types(pal.cards["combine"], run=False)
    scan.take([langscan.Shown(a.text(), "menu", None, "combination list") for a in m.actions()] +
              [langscan.Shown(a.toolTip(), "tooltip", None, "combination list") for a in m.actions()], step + " list")
    for rid in combos.offered():
        pal.set_combination_type(rid)
        before = {b for b, x in w.s.stored.blocks.items() if x.kind == "combine"}
        pal.double_click("combine")
        settle(w, 2)
        new = [b for b, x in w.s.stored.blocks.items() if x.kind == "combine" and b not in before]
        assert new, rid
        page.bridge.choose(new[0])
        settle(w, 1)
        scan.take(langscan.texts(page.panel), f"{step} {rid} panel")
        scan.take([langscan.Shown(page.canvas.blocks[new[0]].d["tip"], "tooltip", None, rid)], f"{step} {rid} tip")
    pal.set_combination_type(combos.DEFAULT)
    _page_texts(w, scan, step)


def walk(w, env, lang: str) -> Scan:
    scan = Scan(lang)
    w.change_language(lang)
    settle(w, 3)
    _page_texts(w, scan, "home")
    w.new_project()
    ready(w)
    page = w.graph
    _page_texts(w, scan, "blank")
    _modes(w, scan, "blank")
    page.show_templates()
    settle(w, 3)
    _page_texts(w, scan, "gallery page")
    from astro_gpu.client.gallery import GALLERY
    for t in GALLERY:
        page.show_templates()
        page.template_chosen(template(t["key"]))
        settle(w)
        _page_texts(w, scan, f"template {t['key']}")
        _every_block(w, scan, f"template {t['key']}")
        if t["key"] in ("filters_full", "colour_full"):
            _modes(w, scan, f"template {t['key']}")
            d = page.new_workflow()
            if d is not None:
                d.show()
                for i in range(d.tabs.count()):
                    d.tabs.setCurrentIndex(i)
                    for c in d.cards:
                        d.pick(c.t)
                        scan.take(langscan.texts(d), f"gallery window {t['key']} {c.t.get('key')}")
                d.close()
        if t["key"] == "filters_full":
            _all_kinds(w, scan, "every kind")
            _every_recipe(w, scan, "every recipe")
    return scan


@pytest.mark.parametrize("lang", os.environ.get("SCAN_LANGS", "fr,en").split(","))
def test_no_word_of_the_other_language_is_shown(win, env, lang):
    scan = walk(win, env, lang)
    got = scan.report()
    out = os.environ.get("SCAN_REPORT")
    if out:
        Path(out).write_text(f"{lang}: {scan.count} texts\n" + "\n".join(got), encoding="utf-8")
    assert scan.count > 200, scan.count
    assert not got, f"{len(got)} texts hold a word of the other language:\n" + "\n".join(got)


@pytest.mark.parametrize("first,then", [("fr", "en"), ("en", "fr")])
def test_a_change_of_language_leaves_no_word_of_the_first(win, env, first, then):
    """The screens are built in one language and the interface switched to the other: every screen shows the second
    one only (a text kept from the first is a screen that was not made again)."""
    win.change_language(first)
    made_from(win, env, "filters_full")
    _modes(win, Scan(first), "built")
    win.change_language(then)
    settle(win, 3)
    scan = Scan(then)
    _page_texts(win, scan, "switched")
    _every_block(win, scan, "switched")
    _modes(win, scan, "switched")
    got = scan.report()
    assert not got, f"{len(got)} texts kept from {first}:\n" + "\n".join(got)


@pytest.mark.parametrize("lang", ["fr", "en"])
def test_real_computed_project_shows_no_word_of_the_other_language(win, env, lang):
    """Inspect on a computed project (the test project, read only): every block of its graph in Inspect, then Post, the
    report and the graph with its blocks done."""
    from conftest import HAVE_TEST_PROJECT, TEST_PROJECT
    if not HAVE_TEST_PROJECT:
        pytest.skip("the test project is not on this machine")
    from astro_gpu.client.imageview import default_loader
    win.change_language(lang)
    win.readonly = True
    assert win.open_project(TEST_PROJECT)
    ready(win, 300)
    scan = Scan(lang)
    scan.mine |= {w.name for w in win.s.p.workflows} | {x.name for x in win.s.p.sessions} | {win.s.p.name}
    steps = []
    for bid in list(win.s.stored.blocks):
        n = win.s.block_map.node(bid)
        if n is not None and n.key not in steps:
            steps.append(n.key)
    for key in steps:
        win.s.select(step=key)
        win.set_mode(1)
        for _i in range(5):
            default_loader().wait(10)
            settle(win, 2)
        _page_texts(win, scan, f"inspect {key}")
    _modes(win, scan, "computed")
    _every_block(win, scan, "computed")
    got = scan.report()
    assert scan.count > 500
    assert not got, f"{len(got)} texts hold a word of the other language:\n" + "\n".join(got)
