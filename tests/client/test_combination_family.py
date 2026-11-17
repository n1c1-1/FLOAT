"""The combination blocks are one family with one behaviour (the owner's rule): a combination of filters (S-H-O, H-O,
the recipes of stages/combine.py) and Combine layers (the stars put back on a starless image), and any block of the
family to come (graphcore model.JOIN_KINDS). The same rule is checked on each, so a user who knows one finds his way in
the other, and a new block of the family must pass the same tests.

The rules: placed only before Post; the same settings in their Inspect view and in Post, one source; a view of their own
in Post (the master layer); children treated alike; one name per thing and the block's shown name everywhere; one way
of naming a step (the setting, then its owner).

The window tests open a scratch copy of the owner's reference projects (their store read in place, nothing written in
them: tests/conftest.py scratch_project)."""
from __future__ import annotations

import os
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication  # noqa: E402

from astro_gpu.graphcore.model import JOIN_KINDS  # noqa: E402

FAMILY = tuple(JOIN_KINDS)


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def pump(s=0.0):
    t0 = time.perf_counter()
    while True:
        QApplication.processEvents()
        if time.perf_counter() - t0 >= s:
            break
        time.sleep(0.005)


# ---------------------------------------------------------------------------------------------- one name per thing
def test_the_starless_image_has_one_name_everywhere(qapp):
    """The Combine layers' input of the starless image was "Background" on the graph, "Starless" in its panel and
    "Starless image" in Post's row (French "Fond de ciel", "Sans étoiles"): the row's words everywhere."""
    from astro_gpu.client import i18n
    from astro_gpu.client.graphview.style import port_title
    from astro_gpu.client.postengine import SUB_NAMES
    for lang in ("en", "fr"):
        i18n.set_language(lang)
        try:
            row = i18n._(SUB_NAMES["starless"])
            assert port_title("Background") == port_title("Starless") == row, lang
            assert i18n._(SUB_NAMES["stars"]) == port_title("Stars")
        finally:
            i18n.set_language("en")


IC443 = "F:/astro_essais/tutoriels_ref/MONO/ic443"


def _whole(tmp_path, name):
    """The window with its real Graph, Inspect and Post pages on a scratch copy of a reference project."""
    from pathlib import Path

    from conftest import scratch_project
    from test_inspect import Harness

    from astro_gpu.client import app as appmod
    from astro_gpu.client import prefs
    src = Path(name)
    if not (src / "project.astroproj").is_file():
        pytest.skip(f"the project {src.name} is not here")
    os.environ["ASTRO_GPU_SETTINGS"] = str(tmp_path / "settings.json")
    prefs.load()
    appmod.apply_language("en")

    class Whole(Harness):
        def make_model(self):
            appmod.Main.make_model(self)

        def graph_page(self):
            return appmod.Main.graph_page(self)

        def inspect_page(self):
            return appmod.Main.inspect_page(self)
    w = Whole(readonly=True)
    w.resize(1500, 900)
    w.show()
    w.open_project(scratch_project(tmp_path / src.name, src))
    for _i in range(400):
        if w.s.ready:
            break
        w.worker.wait(5)
        time.sleep(0.05)
    assert w.s.ready, w.s.error
    pump(0.3)
    return w


def test_real_a_combination_is_named_by_its_block_s_title_in_post_and_in_the_panels(qapp, tmp_path):
    """IC 443's combinations were placed as S-H-O and made H-O (their workflows are stored "S-H-O"): the graph titles
    them H-O, Post's row of their Combine layers said "S-H-O · Combine layers", the Combine layers panel "S-H-O". The
    block's shown name everywhere; what the project stores stays as it was."""
    from astro_gpu.client.graphview.gestures import Gestures
    from astro_gpu.client.post3 import layer_name
    from astro_gpu.client.postengine import save_layers
    w = _whole(tmp_path, IC443)
    try:
        s = w.s
        w.set_mode(2)
        pump(1.0)
        pv = w._post_holder.page.view
        pv.rebuild()
        lay = next(x for x in pv.layers if x.get("kind") == "workflow" and x.get("part") == "combined"
                   and x.get("starless") == "output")
        assert lay["name"].startswith("S-H-O")                          # stored as it was
        title = s.shown_title(lay["pid"])
        assert title.startswith("H-O")
        row = next(r for r in pv.rows if r.lay["id"] == lay["id"] and r.lay["kind"] == "workflow")
        assert layer_name(lay) == row.n._full == title + " · Combine layers"
        assert all("retitle" not in d for d in save_layers(pv.layers))  # never written into the project
        # the Combine layers block's panel: the workflow it belongs to, its inputs, by the same names
        w.set_mode(0)
        pump(0.5)
        cl = next(k for k, b in w.graph.bridge.graph.blocks.items() if b.kind == "combine_layers")
        Gestures(w.graph).click(cl)
        pump(0.8)
        from PySide6.QtWidgets import QLabel
        texts = [x.text() for x in w.graph.panel.findChildren(QLabel) if x.isVisibleTo(w.graph.panel)]
        assert title in texts and not any(t.startswith("S-H-O") for t in texts), texts
        assert any("Starless image" in t for t in texts), texts
    finally:
        w.close()


# ---------------------------------------------------------------------------------------------- one way of naming a step
M27 = "F:/astro_essais/tutoriels_ref/MONO/m27"
STEP = __import__("re").compile(r"^(?P<what>.+) of “(?P<owner>[^”]+)”(?P<change>: .+)?$")


def _owner(w, kind):
    """The name a step of the block of ``kind`` names it by: a combination by its block's title, Combine layers by
    its layer's name in Post (its row)."""
    from astro_gpu.client.graphnode import combine_kind
    from astro_gpu.client.post3 import layer_name, retitle
    s = w.s
    q = next(q for q in s.paths if combine_kind(q) and not all(i.get("image") == "stars"
                                                                for i in q.source["fusion"]["inputs"]))
    if kind == "combine":
        return q, s.shown_title(q.pid)
    lay = dict(s.master_layer(q.pid))
    retitle(s, lay)
    return q, layer_name(lay)


def _post_change(w, kind):
    """One setting of the block changed in Post's master layer: the words of Post's step."""
    from astro_gpu.client.post3 import ContSlider
    w.set_mode(2)
    pump(1.0)
    pv = w._post_holder.page.view
    q, _o = _owner(w, kind)
    lay = next(x for x in pv.layers if x.get("id") == w.s.master_layer(q.pid)["id"])
    pv.sel = None
    pv.select(lay["id"], "layer")
    pump(0.3)
    name = "S II hue" if kind == "combine" else "Strength"
    sl = next(x for x in pv.findChildren(ContSlider) if x.name.text() == name and x.isVisibleTo(pv.right))
    v = sl.value() + (20 if kind == "combine" else 0.5)
    sl.set_value(v)
    sl.moved.emit(v)
    sl.done.emit(v)
    pump(0.5)
    return name, pv.next_words("undo"), None


def _editor_change(w, kind):
    """One setting of the block changed in its Inspect editor and applied: the window's step and the notice."""
    said = []
    q, ed = _open_editor(w, kind)
    w.tune.toasted.connect(said.append)
    name = "S II intensity" if kind == "combine" else "Strength"
    if kind == "combine":
        ed.set_intensity("SII", ed.weight("SII") + 0.25)
    else:
        ed.set_slider("star_strength", float(ed.value("slider.star_strength")) + 0.5)
    ed.apply()
    pump(0.5)
    return name, w.s.next_words("undo"), said[-1] if said else ""


def _panel_change(w, kind):
    """One setting of the block changed in its graph panel and applied: the window's step and the notice."""
    from astro_gpu.client.graphview.gestures import Gestures
    from astro_gpu.client.post3 import ContSlider
    from astro_gpu.client.settings import DoubleSpinBox
    w.set_mode(0)
    pump(0.5)
    q, _o = _owner(w, kind)
    said = []
    pn = w.graph.panel
    pn.toast.connect(said.append)
    bid = next(k for k, b in w.graph.bridge.graph.blocks.items() if b.kind == kind)
    Gestures(w.graph).click(bid)
    pump(0.8)
    if kind == "combine":
        name = "S II intensity"
        sp = next(x for x in pn.findChildren(DoubleSpinBox) if x.objectName() == "intensity")
        pn.propose_weight(q, "SII", sp.value() + 0.25, sp.value())
    else:
        name = "Strength"
        sl = next(x for x in pn.findChildren(ContSlider) if x.name.text() == name and x.isVisibleTo(pn))
        sl.done.emit(sl.value() + 0.5)
    pump(0.3)
    assert w.s.pending_change is not None                         # a draft: its block has a result
    assert pn.draft_apply()
    pump(0.5)
    return name, w.s.next_words("undo"), said[-1] if said else ""


@pytest.mark.parametrize("kind", FAMILY)
def test_real_a_step_of_the_family_names_the_setting_then_its_owner(qapp, tmp_path, kind):
    """Undo and the notices said 'Strength of "S-H-O · Combine layers"' in Post but "Hα intensity" for the
    combination, "Strength: −0.5 → +1.0 · applied in S-H-O" from the graph's panel, "S-H-O draft: …" from Inspect: one
    way for every block of the family, the setting, then its owner, then what it became when said."""
    w = _whole(tmp_path, M27)
    try:
        _q, owner = _owner(w, kind)
        for change in (_post_change, _editor_change):
            name, words, notice = change(w, kind)
            m = STEP.match(words or "")
            assert m and m["what"] == name and m["owner"] == owner, (change.__name__, words)
            if notice is not None:
                assert notice.startswith(words) and "applied in" not in notice, notice
    finally:
        w.close()


def test_real_each_setting_of_a_layer_is_available_by_one_rule_in_post_and_in_its_panel(qapp, tmp_path):
    """Colours was enabled in the Combine layers panel and greyed in Post for the same layer (a combination's master
    layer keeps its colours by its stretch): one rule (post3.layer_slider_state) for every place that shows the
    layer's settings (Post, Inspect's editor; the graph's panel sums them up); each slider is available, or greyed
    with the same reason, in both."""
    from astro_gpu.client.post3 import ContSlider
    w = _whole(tmp_path, M27)
    try:
        w.set_mode(2)
        pump(1.0)
        pv = w._post_holder.page.view
        q, _o = _owner(w, "combine_layers")
        lay = next(x for x in pv.layers if x.get("id") == w.s.master_layer(q.pid)["id"])
        pv.sel = None
        pv.select(lay["id"], "layer")
        pump(0.3)
        post = {x.name.text(): (x.s.isEnabled(), x.why.text()) for x in pv.findChildren(ContSlider)
                if x.isVisibleTo(pv.right)}
        _q, ed = _open_editor(w, "combine_layers")
        panel = {x.name.text(): (x.s.isEnabled(), x.why.text()) for x in ed.page.findChildren(ContSlider)
                 if x.isVisibleTo(ed.page)}
        both = set(post) & set(panel)
        assert {"Colours", "Saturation", "Strength", "Sky"} <= both
        assert {k: post[k] for k in both} == {k: panel[k] for k in both}
        assert post["Colours"][0] is False
    finally:
        w.close()


# ---------------------------------------------------------------------------------------------- one source in Post


@pytest.mark.parametrize("kind", FAMILY)
def test_real_a_block_placed_on_its_own_is_wired_to_post_and_its_panel_says_what_it_waits_for(qapp, tmp_path, kind):
    """A Combine layers placed from the palette on its own came with no wire to Post and an empty panel, where a
    combination came wired to Post with its inputs to wire: both now as the combination, its panel saying what it waits
    for, its inputs not wired, its settings in one look and Adjust greyed with its reason."""
    from PySide6.QtWidgets import QLabel, QPushButton

    from astro_gpu.client.graphview.gestures import Gestures
    from astro_gpu.client.panel import ADJUST_UNWIRED
    w = _whole(tmp_path, M27)
    try:
        w.set_mode(0)
        pump(0.5)
        page = w.graph
        b = page.bridge
        before = set(b.graph.blocks)
        pl = next(x for x in b.places(kind=kind) if x.lit and x.op[1].get("at", {}).get("after") is None)
        Gestures(page).drag_palette(kind, pl)
        pump(0.8)
        new = [k for k in b.graph.blocks if k not in before and b.graph.blocks[k].kind == kind]
        assert len(new) == 1
        post = next(k for k, x in b.graph.blocks.items() if x.kind == "post")
        assert any(x.src[0] == new[0] and x.dst[0] == post for x in b.graph.wires)        # a layer of Post
        pn = page.panel
        texts = [x.text() for x in pn.findChildren(QLabel) if x.isVisibleTo(pn)]
        assert any(t.startswith("Waiting:") and "to wire" in t for t in texts), texts
        assert texts.count("not wired") >= 2, texts
        adj = next(x for x in pn.findChildren(QPushButton) if x.objectName() == "adjust")
        assert not adj.isEnabled() and adj.toolTip() == ADJUST_UNWIRED
    finally:
        w.close()


# ---------------------------------------------------------------------------------------------- one rule, every place


def test_real_the_stars_combination_s_setting_is_in_post_s_stars_section_one_source(qapp, tmp_path):
    """The stars' combination (Neutral or Palette, set in Inspect) had no place in Post: the master layer's Stars
    section and its Stars row show it, the block's setting (one source): a change there is the block's, Inspect's
    editor shows it, Undo takes it back."""
    from PySide6.QtWidgets import QComboBox

    from astro_gpu.client.combineedit import applied_values, stage_class
    w = _whole(tmp_path, M27)
    try:
        s = w.s
        w.set_mode(2)
        pump(1.0)
        pv = w._post_holder.page.view
        q, _o = _owner(w, "combine_layers")
        lay = next(x for x in pv.layers if x.get("id") == s.master_layer(q.pid)["id"])
        got = pv.stars_combination(lay)
        assert got is not None
        sq, stage, spec, _stored = got

        def stored():
            return applied_values(stage_class(stage), spec, s.combine_stored(sq.pid))["stars"]
        was = stored()
        for what in ("layer", "stars"):                             # the layer's Stars section, its Stars row
            pv.sel = None
            pv.select(lay["id"], what)
            pump(0.3)
            boxes = [x for x in pv.findChildren(QComboBox) if x.objectName() == "stars_fusion" and x.isVisibleTo(pv.right)]
            assert len(boxes) == 1 and boxes[0].currentData() == was, what
        other = "palette" if was == "neutral" else "neutral"
        n = len(s._undo)
        box = boxes[0]
        box.activated.emit(box.findData(other))
        pump(0.5)
        assert stored() == other and len(s._undo) == n + 1
        s.undo()
        pump(0.3)
        assert stored() == was
    finally:
        w.close()


# ---------------------------------------------------------------------------------------------- Inspect: one editor kind
def _open_editor(w, kind):
    """The block of ``kind`` chosen on the graph, then Inspect: its editor, its first preview made."""
    from astro_gpu.client.graphview.gestures import Gestures
    w.set_mode(0)
    pump(0.3)
    q, _o = _owner(w, kind)
    if kind == "combine_layers":
        w.s.sel_layer = w.s.master_layer(q.pid)["id"]
    bid = next(k for k, b in w.graph.bridge.graph.blocks.items() if b.kind == kind and
               (kind != "combine" or w.graph.bridge.row_of(k).name == q.name))
    Gestures(w.graph).click(bid)
    pump(0.5)
    w.set_mode(1)
    pump(0.5)
    ed = w.tune.layers_editor if kind == "combine_layers" else w.tune.comb_editor
    end = time.monotonic() + 240
    while time.monotonic() < end and (ed.busy or ed._want is not None or ed.result is None):
        pump(0.02)
    return q, ed


def _stored_setting(w, kind, q, ed):
    if kind == "combine":
        return dict(w.s.combine_stored(q.pid).get("weights") or {})
    return dict((w.s.master_layer(q.pid) or {}).get("sliders") or {})          # (the layer of the version shown)


def _draft_change(ed, kind):
    if kind == "combine":
        return ed.set_intensity("SII", ed.weight("SII") + 0.5)
    return ed.set_slider("saturation", 3.0)


@pytest.mark.parametrize("kind", FAMILY)
def test_real_inspect_tunes_every_block_of_the_family_as_a_draft_with_one_source(qapp, tmp_path, kind):
    """Inspect on Combine layers silently showed the S-H-O's editor: every block of the family has its editor, built
    on one base (blockeditor.DraftEditor, the Gradient editor's pattern): its page and draft bar, a preview on a reduced
    copy, Draft / Applied / Split; a change is a draft the project does not see until Apply, one step of the window's
    Undo named as every step of the family; its settings are those Post shows (one source)."""
    from astro_gpu.client.blockeditor import DraftEditor
    from astro_gpu.client.gradientedit import PREVIEW_PIXELS
    w = _whole(tmp_path, M27)
    try:
        s = w.s
        q, ed = _open_editor(w, kind)
        others = [x for x in w.tune.family_editors() if x is not ed]
        assert isinstance(ed, DraftEditor) and ed.page_shown() and not any(x.page_shown() for x in others)
        assert ed._ui["bar"].isVisible() and {"cmp_draft", "cmp_applied", "cmp_split"} <= set(ed._ui)
        assert ed.result is not None and ed.result.shape[1] * ed.result.shape[2] <= PREVIEW_PIXELS
        before = _stored_setting(w, kind, q, ed)
        n = len(s._undo)
        assert _draft_change(ed, kind) and ed.has_draft()
        assert _stored_setting(w, kind, q, ed) == before and len(s._undo) == n      # a draft: the project as it was
        assert ed.draft_pending() is not None                                        # leaving asks
        _q, owner = _owner(w, kind)
        ed.apply()
        pump(0.5)
        assert not ed.has_draft() and len(s._undo) == n + 1
        m = STEP.match(s.next_words("undo"))
        assert m and m["owner"] == owner, s.next_words("undo")
        after = _stored_setting(w, kind, q, ed)
        assert after != before and ed.applied() == ed.values()                      # the editor reads what it wrote
        s.undo()
        pump(0.3)
        assert _stored_setting(w, kind, q, ed) == before
    finally:
        w.close()


@pytest.mark.parametrize("kind", FAMILY)
def test_real_choosing_another_workflow_in_inspect_makes_nothing(qapp, tmp_path, kind):
    """With Combine layers' editor open, choosing OIII in Inspect's list of workflows added a Combine layers block
    wired to Post, its layer over the colour image, without Undo, "*" or question. A choice of workflow never makes a
    block: a workflow without the block shows its own view, for every block of the family."""
    w = _whole(tmp_path, M27)
    try:
        s = w.s
        q, ed = _open_editor(w, kind)
        assert ed.page_shown()
        blocks = sorted(w.graph.bridge.graph.blocks)
        layers = [x.get("id") for x in s.post_layers()]
        n, dirty = len(s._undo), s._dirty
        oiii = next(x for x in s.paths if x.name == "OIII")
        w.tune._strip_workflow(s.index_of(oiii.pid))
        pump(1.0)
        assert s.path.pid == oiii.pid and not ed.page_shown()
        # that workflow's own view, its last block: never an empty page (the owner's test: "Color full" chosen from
        # the combination's editor showed nothing under the path bar)
        assert s.sel_step not in ("fusion", "composition") and w.tune.center.count() > 0
        assert w.tune.view is not None or w.tune.center.itemAt(0).widget() is not None
        assert sorted(w.graph.bridge.graph.blocks) == blocks and [x.get("id") for x in s.post_layers()] == layers
        assert len(s._undo) == n and s._dirty == dirty
    finally:
        w.close()


@pytest.mark.parametrize("kind", FAMILY)
def test_real_try_variant_of_post_s_layers_alone_is_a_new_version_of_post(qapp, tmp_path, kind, monkeypatch):
    """Try variant was active for the combination and greyed for Combine layers for the same kind of change, and
    Combine layers' leave question offered Try variant, which applied the change. One rule: a draft of Post's layers
    alone goes into a new version of Post, the version shown untouched, one step of Undo; the leave question offers
    exactly what the bar offers."""
    from astro_gpu.client.blockdraft import DRAFT_ANSWER_ENV, settle
    w = _whole(tmp_path, M27)
    try:
        s = w.s
        q, ed = _open_editor(w, kind)
        versions = len(s.p.post["versions"])
        shown = s.post_version()["id"]
        before = _stored_setting(w, kind, q, ed)
        assert _draft_change(ed, kind)
        ed._sync()
        assert ed._ui["variant"].isEnabled() and "new version of Post" in ed._ui["variant"].toolTip()
        n = len(s._undo)
        monkeypatch.setenv(DRAFT_ANSWER_ENV, "variant")              # the leave question's Try variant: the bar's
        assert settle(ed, None)
        pump(0.5)
        assert len(s.p.post["versions"]) == versions + 1 and s.post_version()["id"] != shown and len(s._undo) == n + 1
        assert _stored_setting(w, kind, q, ed) != before and not ed.has_draft()
        s.set_post_version(shown)
        pump(0.3)
        assert _stored_setting(w, kind, q, ed) == before                # the version shown before: untouched
    finally:
        w.close()


def test_real_one_path_bar_one_link_and_one_name_for_every_block_of_the_family(qapp, tmp_path):
    """Inspect's path bar had no step for Combine layers (only "S-H-O › Post"), Post's "Open in Inspect" was under the
    combination's section only, and Combine layers had several names ("Combine layers 2" on the graph and in the
    outputs, "S-H-O · Combine layers" in Post): a step in the bar for each block, the link under each one's settings in
    Post, and one shown name everywhere (its layer of Post's)."""
    from PySide6.QtWidgets import QPushButton

    from astro_gpu.client.post3 import layer_name
    w = _whole(tmp_path, M27)
    try:
        s = w.s
        q, ed = _open_editor(w, "combine_layers")
        items = w.tune.strip.items(s, s.block_map)
        name = s.layers_title(ed.node)
        assert ("composition", name) in [(k, t) for k, t, _n in items]
        bid = next(k for k, b in w.graph.bridge.graph.blocks.items() if b.kind == "combine_layers")
        assert w.graph.bridge.doc.title(bid) == name == ed.heading()
        w.set_mode(2)
        pump(1.0)
        pv = w._post_holder.page.view
        lay = next(x for x in pv.layers if x.get("id") == s.master_layer(q.pid)["id"])
        assert layer_name(lay) == name
        pv.sel = None
        pv.select(lay["id"], "layer")
        pump(0.3)
        links = [b for b in pv.findChildren(QPushButton) if b.text() == "Open in Inspect  →" and b.isVisibleTo(pv.right)]
        assert len(links) == 2                                    # the combination's, Combine layers'
        links[-1].click()
        pump(0.8)
        assert w.mode == 1 and s.sel_step == "composition" and w.tune.layers_editor.page_shown()
    finally:
        w.close()


@pytest.mark.parametrize("kind", FAMILY)
def test_real_stay_keeps_the_editor_and_the_list_shows_the_workflow_edited(qapp, tmp_path, kind, monkeypatch):
    """With a draft, another workflow chosen in Inspect's list and the question answered Stay: the editor stays with
    its draft, and the list shows the workflow being edited again (it kept the one just chosen)."""
    from PySide6.QtWidgets import QComboBox

    from astro_gpu.client.blockdraft import DRAFT_ANSWER_ENV
    w = _whole(tmp_path, M27)
    try:
        s = w.s
        q, ed = _open_editor(w, kind)
        assert _draft_change(ed, kind)
        monkeypatch.setenv(DRAFT_ANSWER_ENV, "stay")
        oiii = next(x for x in s.paths if x.name == "OIII")
        box = next(c for c in w.tune.strip.findChildren(QComboBox) if c.findData(oiii.pid) >= 0)
        box.setCurrentIndex(box.findData(oiii.pid))                  # the user's choice in the list
        pump(0.8)
        assert s.path.pid == q.pid and ed.page_shown() and ed.has_draft()
        box = next(c for c in w.tune.strip.findChildren(QComboBox) if c.findData(oiii.pid) >= 0)
        assert box.currentData() == q.pid
        ed.reset()
    finally:
        w.close()
