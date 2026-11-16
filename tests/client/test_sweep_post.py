"""Click every control of Post, on a scratch copy of the computed test project, in English and in French: each control
a hand can reach is driven once under the oracle of tests/client/sweep.py (no uncaught exception, no message box, the
window answering). This test only reports: every failure is collected and printed, the assertion comes at the end."""
from __future__ import annotations

import pytest

from astro_gpu.client import prefs
from astro_gpu.client.blockdraft import DRAFT_ANSWER_ENV
from tests.client.pagehelp import qapp  # noqa: F401  (the fixture, by name)
from tests.client.sweep import Oracle, Sweep, controls, definition, differs, gestures

LANGS = ("en", "fr")
#: gestures whose failure is a known defect of the application, checked on its own as an expected failure
KNOWN: set = set()


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


@pytest.mark.parametrize("lang", LANGS)
def test_sweep_post(qapp, tmp_path_factory, oracle, lang):
    from conftest import HAVE_TEST_PROJECT, NO_TEST_PROJECT, scratch_project
    if not HAVE_TEST_PROJECT:
        pytest.skip(NO_TEST_PROJECT)
    from astro_gpu.client import app as appmod
    from astro_gpu.client.post3 import PostPage3
    from tests.client.test_inspect import open_window, settle
    appmod.apply_language(lang)
    win = open_window(scratch_project(tmp_path_factory.mktemp("sweep_post") / "proj"))
    S = Sweep(oracle, lambda: settle(win, 2), f"post[{lang}]")
    try:
        s = win.s
        made = []                                   # the window's own Post (the test window mounts Inspect alone)
        S.drive("show", lambda: made.append(win.post_page()))
        page = made[0] if made and isinstance(made[0], PostPage3) else None
        if page is not None:
            page.resize(1500, 900)
            page.show()
            S.drive("shown", lambda: None)
        if page is None:
            S.fail("show", "no Post page")
        else:
            names = [n for n, _w in controls(page)]
            for name in names:
                w = dict(controls(page)).get(name)
                if w is None:
                    S.skip(name, "gone once the page was built again")
                    continue
                gs = [g for g, _f in gestures(w)][:3]
                if not gs:
                    S.skip(name, f"no gesture for a {type(w).__name__}")
                for gname in gs:
                    w = dict(controls(page)).get(name)
                    do = dict(gestures(w)).get(gname) if w is not None else None
                    if do is None:
                        break
                    n, m, before = len(s._undo), len(page.undo), definition(s)
                    S.drive(f"{name}/{gname}", do)
                    k = 0
                    while len(page.undo) > m and k < 20:     # Post's own steps, as the window undoes them on Post
                        page.do_undo()
                        settle(win, 2)
                        k += 1
                    while len(s._undo) > n and k < 40:
                        s.undo()
                        settle(win, 2)
                        k += 1
                    if definition(s) != before and f"{name}/{gname}" not in KNOWN:
                        S.fail(f"{name}/{gname}", differs("its Undo", before, s))
                    if not page.isVisible():
                        page.show()
                        settle(win, 2)
            page.shutdown()
            page.deleteLater()
    finally:
        appmod.apply_language("en")
        try:
            win.close()
        except RuntimeError:
            pass
    text = S.report()
    print(text)
    if S.failures:
        print(S.full())
    assert not S.failures, text
