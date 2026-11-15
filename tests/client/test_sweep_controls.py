"""Click every control: on a computed colour template and a computed S-H-O, in English and in French, every control a
hand can reach is driven once, under one oracle (tests/client/sweep.py): no uncaught exception, no message box, the
window answering, the definition as it was after Reset or Undo. A failure names its control by a readable path
("graph/Denoise/Method/Pretrained + adapted/variant"); what cannot be driven is printed in a "not driven" list.

- graph: every block chosen in turn, every control of its settings panel changed then Apply, again then Try variant,
  again then Reset; the question of a block left with a draft, each answer; every entry of every block's, wire's and
  the canvas' context menu (undone after); every palette card placed (undone after).
- table: every entry of every cell's context menu, every button of the table.
- inspect: every block of the band, each of its image tabs, every control of its settings panel as in the graph, every
  button of the page (the computed test project, a scratch copy).
- window: Home and every menu entry that needs no file (file dialogs answer "cancelled")."""
from __future__ import annotations

import os

import pytest

from astro_gpu.client import prefs
from astro_gpu.client.blockdraft import DRAFT_ANSWER_ENV, DraftBar
from tests.client.graphview.legacy import World
from tests.client.graphview.test_gv_loop_guard import _computed
from tests.client.pagehelp import qapp  # noqa: F401  (the fixture, by name)
from tests.client.sweep import Oracle, Sweep, controls, definition, differs, gestures

LANGS = ("en", "fr")
SHAPES = ("colour", "sho")


@pytest.fixture(autouse=True)
def _own_settings(tmp_path, monkeypatch):
    monkeypatch.setenv("ASTRO_GPU_SETTINGS", str(tmp_path / "settings.json"))
    monkeypatch.setenv(DRAFT_ANSWER_ENV, "discard")
    prefs.load()


@pytest.fixture()
def oracle(monkeypatch):
    o = Oracle(monkeypatch)
    yield o
    o.close()


def _lang(code):
    from astro_gpu.client import app as appmod
    appmod.apply_language(code)


def _world(tmp_path, shape):
    if shape == "colour":
        W = World(tmp_path / "w")
    else:
        W = World(tmp_path / "w", workflows=(), filters=("SII", "Ha", "OIII"), nights=0)
        W.template("sho_rgb")
    _computed(W.m)
    W.settle()
    return W


def _undo_to(m, n, settle):
    k = 0
    while len(m._undo) > n and k < 20:
        m.undo()
        settle()
        k += 1


def _bar(pn):
    return next((b for b in pn.findChildren(DraftBar) if b.isVisibleTo(pn)), None) or \
        next(iter(pn.findChildren(DraftBar)), None)


def _end(S: Sweep, *more: Sweep):
    import sys
    try:
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError):
        pass
    text = "\n".join(x.report() for x in (S, *more))
    print(text)
    fails = [f for x in (S, *more) for f in x.failures]
    if fails:
        print("\n".join(x.full() for x in (S, *more)))
    assert not fails, text


# ---------------------------------------------------------------------------------------------- a settings panel
def drive_panel(S: Sweep, m, get_panel, prefix: str, settle, restore=lambda: None):
    """Every control of the panel: changed then Apply, again then Try variant, again then Reset (its other entries or
    steps: then Apply; a gesture that makes no draft is undone at once); the definition checked after each answer and
    its Undo."""
    pn = get_panel()
    names = [n for n, _w in controls(pn, skip=pn.findChildren(DraftBar))]

    def find(name):
        p = get_panel()
        return dict(controls(p, skip=p.findChildren(DraftBar))).get(name) if p is not None else None

    for name in names:
        w = find(name)
        if w is None:
            S.skip(f"{prefix}/{name}", "gone once the panel was built again")
            continue
        gs = [g for g, _f in gestures(w)]
        if not gs:
            S.skip(f"{prefix}/{name}", f"no gesture for a {type(w).__name__}")
            continue
        for gi, gname in enumerate(gs):
            # the three answers on the control's first gesture; its other entries / steps: Apply (the time of a run)
            for how in (("apply", "variant", "reset") if gi == 0 else ("apply",)):
                w = find(name)
                do = dict(gestures(w)).get(gname) if w is not None else None
                if do is None:
                    break
                path = f"{prefix}/{name}/{gname}"
                before, n = definition(m), len(m._undo)
                S.drive(path, do)
                if not getattr(m, "pending_change", None):            # applied at once, or a gesture of the view
                    _undo_to(m, n, settle)
                    if definition(m) != before:
                        S.fail(path, differs("its Undo", before, m))
                    restore()
                    break
                bar = _bar(get_panel())
                if bar is None:
                    S.fail(path, "a draft without its bar")
                    get_panel().draft_reset()
                    settle()
                    break
                btn = {"apply": bar.b_apply, "variant": bar.b_variant, "reset": bar.b_reset}[how]
                if not btn.isEnabled():
                    if not btn.toolTip():
                        S.fail(f"{path}/{how}", "greyed without its reason")
                    btn = bar.b_reset
                S.drive(f"{path}/{how}", btn.click)
                if getattr(m, "pending_change", None):
                    S.fail(f"{path}/{how}", "the draft stays after its answer")
                    get_panel().draft_reset()
                    settle()
                _undo_to(m, n, settle)
                if definition(m) != before:
                    S.fail(f"{path}/{how}", differs("the answer and its Undo", before, m))
                restore()


def first_draft(m, get_panel, settle):
    """A draft made by the first control that makes one; its name, or None."""
    pn = get_panel()
    for name in [n for n, _w in controls(pn, skip=pn.findChildren(DraftBar))]:
        pn = get_panel()
        w = dict(controls(pn, skip=pn.findChildren(DraftBar))).get(name)
        if w is None:
            continue
        for _g, do in gestures(w)[:1]:
            n = len(m._undo)
            do()
            settle()
            if getattr(m, "pending_change", None):
                return name
            _undo_to(m, n, settle)
    return None


# ---------------------------------------------------------------------------------------------- the graph page
def _blocks(W):
    return [b for b, blk in W.graph.blocks.items() if blk.kind not in ("post",)]


def _title(W, b):
    blk = W.graph.blocks.get(b)
    if blk is None:
        return b
    from astro_gpu.graphcore.rules import LABEL
    return str(LABEL.get(blk.kind, blk.kind) if isinstance(LABEL, dict) else blk.kind)


@pytest.mark.parametrize("lang", LANGS)
@pytest.mark.parametrize("shape", SHAPES)
def test_sweep_graph_panels(qapp, tmp_path, oracle, monkeypatch, shape, lang):
    _lang(lang)
    W = _world(tmp_path, shape)
    S = Sweep(oracle, lambda: W.settle(1), f"graph[{shape},{lang}]")
    try:
        m = W.m
        get = lambda: W.page.panel                                # noqa: E731
        for b in _blocks(W):
            if b not in W.graph.blocks:
                continue
            t = _title(W, b)
            S.drive(f"{t}/choose", lambda b=b: W.choose(b))
            pn = get()
            if pn is None or pn.node is None:
                S.skip(t, "no settings panel for this block")
                continue
            pn.expert = True
            pn.refresh(pn.node)
            W.settle(1)
            drive_panel(S, m, get, t, lambda: W.settle(1), restore=lambda b=b: W.choose(b))
            # the question of a block left with a draft, each answer
            others = [x for x in _blocks(W) if x != b]
            for how in ("apply", "variant", "discard", "stay"):
                W.choose(b)
                n, before = len(m._undo), definition(m)
                made = first_draft(m, get, lambda: W.settle(1))
                if made is None:
                    S.skip(f"{t}/leave", "no control of the block makes a draft")
                    break
                monkeypatch.setenv(DRAFT_ANSWER_ENV, how)
                S.drive(f"{t}/leave with a draft/{how}", lambda: W.choose(others[0]))
                monkeypatch.setenv(DRAFT_ANSWER_ENV, "discard")
                if getattr(m, "pending_change", None):
                    get().draft_reset()
                    W.settle(1)
                _undo_to(m, n, lambda: W.settle(1))
                if definition(m) != before:
                    S.fail(f"{t}/leave with a draft/{how}", differs("its Undo", before, m))
            if get() is not None:
                get().expert = False
    finally:
        W.close()
    _end(S)


def _actions(menu, pre=""):
    out = []
    for a in menu.actions():
        if a.isSeparator():
            continue
        name = pre + " ".join(a.text().replace("&", "").split())
        if a.menu() is not None:
            out += _actions(a.menu(), name + "/")
        else:
            out.append((name, a))
    return out


@pytest.mark.parametrize("lang", LANGS)
@pytest.mark.parametrize("shape", SHAPES)
def test_sweep_graph_menus_and_palette(qapp, tmp_path, oracle, shape, lang):
    _lang(lang)
    W = _world(tmp_path, shape)
    S = Sweep(oracle, lambda: W.settle(1), f"menus[{shape},{lang}]")
    greyed = []
    try:
        m, cv = W.m, W.page.canvas
        targets = [("block", b) for b in list(cv.blocks)] + [("wire", w) for w in list(getattr(cv, "wires", {}))] + \
            [("canvas", None)]
        for what, ident in targets:
            def item():
                if what == "block":
                    return cv.blocks.get(ident)
                if what == "wire":
                    return cv.wires.get(ident)
                return None
            if what != "canvas" and item() is None:
                continue
            t = _title(W, ident) if what == "block" else what
            names = [(n, a.isEnabled(), a.toolTip()) for n, a in _actions(cv.build_menu(item()))]
            for name, ok, why in names:
                if not ok:
                    greyed.append(f"{t}/{name}" + ("" if why else " (no reason)"))
                    continue
                n, before = len(m._undo), definition(m)
                it = item()
                if what != "canvas" and it is None:
                    S.skip(f"{t}/{name}", "the item is gone after the previous entry and its Undo")
                    continue
                act = dict(_actions(cv.build_menu(it))).get(name)
                if act is None or not act.isEnabled():
                    S.skip(f"{t}/{name}", "the entry is gone once the menu is built again")
                    continue
                S.drive(f"{t}/{name}", act.trigger)
                ed = getattr(cv, "editor", None) or getattr(cv, "_editor", None)
                if ed is not None:
                    try:
                        ed.finish(cancel=True)
                    except (RuntimeError, AttributeError, TypeError):
                        pass
                _undo_to(m, n, lambda: W.settle(1))
                if definition(m) != before:
                    S.fail(f"{t}/{name}", differs("its Undo", before, m))
        # the palette: every card placed (a double click), then undone
        W.page.palette.finish()
        for kind, card in list(W.page.palette.cards.items()):
            if not card.isEnabled() or getattr(card, "ok", True) is False:
                continue
            n, before = len(m._undo), definition(m)
            S.drive(f"palette/{kind}", lambda kind=kind: W.g.double_click_palette(kind))
            _undo_to(m, n, lambda: W.settle(1))
            if definition(m) != before:
                S.fail(f"palette/{kind}", differs("its Undo", before, m))
    finally:
        W.close()
    print(f"greyed entries: {len(greyed)}; without a reason: " + "; ".join(g for g in greyed if "(no reason)" in g))
    _end(S)


# ---------------------------------------------------------------------------------------------- the table
@pytest.mark.parametrize("lang", LANGS)
@pytest.mark.parametrize("shape", SHAPES)
def test_sweep_table(qapp, tmp_path, oracle, shape, lang):
    _lang(lang)
    W = _world(tmp_path, shape)
    S = Sweep(oracle, lambda: W.settle(1), f"table[{shape},{lang}]")
    try:
        m = W.m
        S.drive("show", W.page.positions["table"].click)
        host = W.page.table_host
        if host is None or host.table is None:
            S.fail("show", "no table")
            return
        t = host.table
        grid = t.grid
        ids = [c.id for c in grid.pinned + grid.cells]
        for cid in ids:
            c = grid.cell(cid)
            if c is None:
                continue
            entries = [e for e in grid.menu_entries(c) if e is not None]
            for e in entries:
                text, ok = e[0], e[2]
                path = f"{'/'.join(str(x) for x in cid)}/{text}"
                if not ok:
                    continue
                c = grid.cell(cid)
                if c is None:
                    S.skip(path, "the cell is gone after the previous entry and its Undo")
                    break
                cb = next((x[1] for x in grid.menu_entries(c) if x is not None and x[0] == text and x[2]), None)
                if cb is None:
                    S.skip(path, "the entry is gone once the menu is built again")
                    continue
                n, before = len(m._undo), definition(m)
                S.drive(path, cb)
                ed = getattr(grid, "editor", None)
                if ed is not None:
                    try:
                        ed.hide()
                    except RuntimeError:
                        pass
                _undo_to(m, n, lambda: W.settle(1))
                if definition(m) != before:
                    S.fail(path, differs("its Undo", before, m))
        for name, w in controls(t, skip=[grid]):
            for gname, do in gestures(w):
                n, before = len(m._undo), definition(m)
                S.drive(f"button/{name}/{gname}", do)
                _undo_to(m, n, lambda: W.settle(1))
                if definition(m) != before:
                    S.fail(f"button/{name}/{gname}", differs("its Undo", before, m))
                if W.page.table_host is not host or not host.isVisible():
                    W.page.positions["table"].click()
                    W.settle(1)
    finally:
        W.close()
    _end(S)


# ---------------------------------------------------------------------------------------------- Inspect and the window
def _window(tmp_path_factory, qapp):
    from conftest import scratch_project

    from tests.client.test_inspect import open_window
    folder = scratch_project(tmp_path_factory.mktemp("sweep") / "proj")
    win = open_window(folder)
    win._sweep_root = folder
    return win


@pytest.mark.parametrize("lang", LANGS)
def test_sweep_inspect(qapp, tmp_path_factory, oracle, lang):
    from conftest import HAVE_TEST_PROJECT, NO_TEST_PROJECT
    if not HAVE_TEST_PROJECT:
        pytest.skip(NO_TEST_PROJECT)
    from PySide6.QtWidgets import QPushButton, QTabBar

    from tests.client.test_inspect import settle
    _lang(lang)
    win = _window(tmp_path_factory, qapp)
    S = Sweep(oracle, lambda: settle(win, 2), f"inspect[{lang}]")
    try:
        s = win.s
        win.set_mode(1)
        settle(win)
        strip = win.tune.strip

        def band():
            return [(" ".join(b.text().split()) or b.toolTip()[:30], b) for b in strip.findChildren(QPushButton)
                    if b.isVisibleTo(strip) and inside_layout(b, strip.h)]

        def inside_layout(b, lay):
            return any(lay.itemAt(i).widget() is b for i in range(lay.count()))

        names = [n for n, _b in band()]
        if not names:
            S.fail("band", "no block in Inspect's band")
        for name in names:
            def pick(name=name):
                if win.mode != 1:
                    win.set_mode(1)
                    settle(win, 2)
                b = dict(band()).get(name)
                if b is not None:
                    b.click()
            S.drive(f"{name}/choose", pick)
            step = s.sel_step
            def tabbar():
                return next((tb for tb in win.tune.findChildren(QTabBar) if tb.isVisibleTo(win.tune)), None)
            tb = tabbar()
            tabs = [(i, tb.tabText(i).strip()) for i in range(tb.count()) if tb.isTabEnabled(i)] if tb else []
            for i, tt in tabs:
                def tab(i=i):
                    t = tabbar()                             # built again by each tab shown
                    if t is not None and i < t.count():
                        t.setCurrentIndex(i)
                S.drive(f"{name}/tab {tt}", tab)

            def restore(step=step):
                if win.mode != 1 or s.sel_step != step:
                    s.select(step=step)
                    win.set_mode(1)
                    settle(win, 2)
            pn = win.tune.panel
            if pn is not None and pn.isVisibleTo(win.tune) and getattr(pn, "node", None) is not None:
                pn.expert = True
                pn.refresh(pn.node)
                settle(win, 2)
                drive_panel(S, s, lambda: win.tune.panel, name, lambda: settle(win, 2), restore)
                win.tune.panel.expert = False
            else:
                S.skip(f"{name}/panel", "no settings panel shown for this block")
            # the page's own buttons (the band and the panel apart)
            def page_controls():
                skip = [strip, win.tune.panel] + list(win.tune.findChildren(DraftBar))
                return dict(controls(win.tune, skip=skip))
            for bname in list(page_controls()):
                for gname in [g for g, _f in gestures(page_controls().get(bname) or strip)][:2]:
                    w = page_controls().get(bname)
                    do = dict(gestures(w)).get(gname) if w is not None else None
                    if do is None:
                        S.skip(f"{name}/page/{bname}/{gname}", "gone once the page was built again")
                        continue
                    n, before = len(s._undo), definition(s)
                    S.drive(f"{name}/page/{bname}/{gname}", do)
                    if getattr(s, "pending_change", None):
                        win.tune.panel.draft_reset()
                        settle(win, 2)
                    _undo_to(s, n, lambda: settle(win, 2))
                    if definition(s) != before:
                        S.fail(f"{name}/page/{bname}/{gname}", differs("its Undo", before, s))
                    restore()
    finally:
        win.close()
    _end(S)


@pytest.mark.parametrize("lang", LANGS)
def test_sweep_window_menus_and_home(qapp, tmp_path_factory, oracle, lang):
    from conftest import HAVE_TEST_PROJECT, NO_TEST_PROJECT
    if not HAVE_TEST_PROJECT:
        pytest.skip(NO_TEST_PROJECT)
    from tests.client.test_inspect import settle
    _lang(lang)
    win = _window(tmp_path_factory, qapp)
    win.build_menus()
    def settle_any():
        if win.project is None or getattr(win, "tune", None) is None:     # no project: the window's events alone
            for _i in range(6):
                qapp.processEvents()
            return
        settle(win, 2)
    S = Sweep(oracle, settle_any, f"window[{lang}]")
    #: entries that start a computation of the test project, or end the window: said, not driven
    starts = {"Run all", "Run to here", "Tout lancer", "Lancer jusqu'ici", "Quit", "Quitter"}
    try:
        from astro_gpu.client import app as appmod
        def entries():
            """Every entry of the menu bar by its path (the menus are built again by some of them)."""
            out = []
            for a in win.menuBar().actions():
                if a.menu() is None:
                    continue
                mn = " ".join(a.text().replace("&", "").split())
                a.menu().aboutToShow.emit()
                out += _actions(a.menu(), mn + "/")
            return out
        for name, _a in entries():
            mname = name.split("/")[0]
            if True:
                base = name.split("/")[-1].split("\t")[0].strip()
                a = dict(entries()).get(name)
                if a is None:
                    S.skip(name, "gone once the menus were built again")
                    continue
                if not a.isEnabled():
                    continue
                if base in starts or base.startswith(("Run", "Lancer", "Tout lancer")):
                    S.skip(name, "starts a computation, or ends the window")
                    continue
                if "Recent" in mname or name.count("/") > 1:
                    S.skip(name, "opens another project")
                    continue
                S.drive(name, a.trigger)
                if win.project is None:                           # Close project, New project: the project again
                    S.drive(f"{name}/reopened", lambda: win.open_project(win._sweep_root))
        # Home's own buttons
        S.drive("File/Home", win.go_home)
        home = win.centralWidget()
        for name, w in controls(home):
            txt = name.lower()
            if any(x in txt for x in ("open", "ouvrir", "recent", "récent")):
                S.skip(f"home/{name}", "opens a project (a file dialog or a recent project)")
                continue
            for gname, do in gestures(w)[:1]:
                S.drive(f"home/{name}/{gname}", do)
                if win.centralWidget() is not home:
                    win.go_home()
                    settle_any()
        appmod.apply_language("en")
    finally:
        try:
            win.close()
        except RuntimeError:
            pass
    _end(S)


def teardown_module(_m):
    from astro_gpu.client import i18n
    i18n.set_language("en")
    os.environ.pop(DRAFT_ANSWER_ENV, None)
