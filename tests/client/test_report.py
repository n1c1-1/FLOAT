"""The Report tab and its night card (issue #81), and the mask brush's size (issue #80)."""
from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QObject, Signal  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from astro_gpu.client import i18n  # noqa: E402
from astro_gpu.client import report as rep  # noqa: E402
from astro_gpu.client.charts import RegistrationData  # noqa: E402
from astro_gpu.client.chartview import ChartView  # noqa: E402
from astro_gpu.client.post3 import BRUSH_MAX, BRUSH_MIN, brush_pos, brush_px  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    yield QApplication.instance() or QApplication([])


class Service(QObject):
    jobReady = Signal(str, int, object)
    jobFailed = Signal(str, int, str)

    def request_job(self, name, fn, *args):
        return 1


class Steps(QObject):
    changed = Signal()
    service = Service()

    def facts(self, p, key):
        return [("Value of " + key, "1")]


def data(n=12, timed=True, figures=True):
    t = 1000.0 + 15.0 * np.arange(n) if timed else np.full(n, np.nan)
    reg = RegistrationData(names=[f"f{i}" for i in range(n)], iso=[""] * n, secs=t, fwhm=np.full(n, 9.0),
                           transp=np.linspace(70, 100, n), nmatch=np.full(n, 50.0), nstars=np.full(n, 80.0),
                           rms=np.full(n, 0.2), aligned=np.ones(n, bool), dx=np.linspace(0, 5, n), dy=np.zeros(n), ref=2,
                           scale=2.0, roundness=np.full(n, 0.5))
    kept = np.ones(n, bool)
    kept[0] = False
    d = rep.ReportData(reg=reg, sessions=[("Night 1", list(range(6))), ("Night 2", list(range(6, n)))], kept=kept,
                       fwhm=np.linspace(2.0, 3.0, n), roundness=np.full(n, 0.9), sky=np.full(n, 800.0),
                       exposure=np.full(n, 15.0), settings=[("Stacking", "Kappa", 5.0, True), ("Stacking", "Method", "drizzle", False)],
                       astro=dict(focal_estimate_mm=272.4, field_deg=[7.33, 5.04]))
    if figures:
        rep.draw_figures(d, "M 45")
    return d


def page(monkeypatch, state="done", d=None):
    q = SimpleNamespace(pid=1, name="Full")
    acq: dict[str, str] = {}
    s = SimpleNamespace(paths=[q], all_stages=lambda p: ("stack",), info=lambda: dict(camera="A7S", n_lights=12, iso_gain=5000.0, n_darks=48, n_flats=62),
                        p=SimpleNamespace(acquisition=acq), target="M 45", name="Pleiades", block_keys=lambda p: ["alignement", "empilement"],
                        set_shooting_detail=lambda k, t: acq.__setitem__(k, t) if t else acq.pop(k, None))   # the model's
    monkeypatch.setattr(rep, "block_state", lambda *_a: state)
    p = rep.ReportPage(s, Steps())
    p.refresh()
    p.data = d if d is not None else data()
    p.build()
    return p


def text(p) -> str:
    return p.document().toPlainText()


def test_the_sizes_are_in_arcseconds_from_the_moffat_fit_and_the_kept_frames(qapp):
    rows = dict(rep.card_rows(data(figures=False), list(range(6))))
    assert rows["Frames kept"] == "5 / 6"
    assert rows["Total exposure"] and rows["Best FWHM"] == '4.2"'           # 2.0 x the 2.09 px of frame 1, the first kept


def test_the_card_counts_time_from_the_first_frame_never_the_clock(qapp):
    d = data(figures=False)
    x, minutes = rep.elapsed(d.reg, list(range(6)))
    assert minutes and x[0] == 0.0 and abs(x[-1] - 1.25) < 1e-9


def test_a_card_without_times_counts_the_frames(qapp):
    d = data(timed=False, figures=False)
    x, minutes = rep.elapsed(d.reg, list(range(6)))
    assert not minutes and list(x) == [1, 2, 3, 4, 5, 6]


def test_the_report_is_numbered_gives_each_block_and_the_recipe_and_a_stale_result_says_so(qapp, monkeypatch):
    t = text(page(monkeypatch, "stale"))
    assert "Processing report" in t and "Table 1. Acquisition" in t and "1. Registration" in t and "2. Stacking" in t
    assert "3. Recipe" in t and "4. Appendix" in t and "Value of alignement" in t and "Registration → Stacking" in t
    assert t.count("Outdated") == 2


def test_each_figure_has_its_number_and_caption(qapp, monkeypatch):
    t = text(page(monkeypatch))
    assert "Figure 1." in t and "Figure 2." in t and "Figure 3." in t and "Open circles" in t


def test_a_duration_is_exact_and_a_star_size_has_its_pixels():
    assert rep.exact_time(450.0) == "7 min 30 s" and rep.exact_time(3900.0) == "1 h 05" and rep.exact_time(480.0) == "8 min"
    assert rep.size_text(1.68, 6.37, '"', lambda x: f"{x:.1f}") == '10.7" (1.7 px)'


def test_the_english_text_has_no_space_before_its_colons(qapp, monkeypatch):
    p = page(monkeypatch)
    p.lang.setCurrentIndex(p.lang.findData("en"))
    assert "Camera: A7S" in p.details()


def test_every_part_can_be_copied_as_a_picture(qapp, monkeypatch):
    from PySide6.QtWidgets import QApplication
    p = page(monkeypatch)
    # in the order the report is read (#140): its first page, the night cards, then the sections
    names = [p.part.itemText(i) for i in range(p.part.count())]
    assert names[:4] == ["Session card", "Night card · Night 1", "Night card · Night 2", "The night, frame by frame · Night 1"]
    assert p.part.currentIndex() == 0 and "Table 1. Acquisition" in p._parts[0][1][1]
    for i in range(p.part.count()):
        p.part.setCurrentIndex(i)
        p.copy_image()
        assert not QApplication.clipboard().image().isNull()


def test_of_the_two_stacking_methods_only_the_one_that_ran_is_listed(qapp, monkeypatch):
    d = data()
    d.settings = [("Stacking", "Stacking method", "drizzle", False), ("Stacking", "Rejection (Classic)", "auto", False),
                  ("Stacking", "Rejection (drizzle)", "own", False)]
    t = text(page(monkeypatch, d=d))
    assert "Rejection (drizzle)" in t and "Rejection (Classic)" not in t


def test_the_report_uses_none_of_the_interactive_charts(qapp, monkeypatch):
    assert not page(monkeypatch).findChildren(ChartView)


def test_a_block_the_workflow_does_not_have_is_not_in_its_report(qapp, monkeypatch):
    assert "Value of fond" not in text(page(monkeypatch))


def _with_stars(monkeypatch, metrics):
    d = data()
    d.metrics = metrics
    p = page(monkeypatch, d=d)
    p.s.block_keys = lambda q: ["alignement", "etoiles"]
    monkeypatch.setattr(p.steps, "facts", lambda q, key: [] if key == "etoiles" else [("Value of " + key, "1")],
                        raising=False)
    return p


def test_the_stars_section_says_what_the_block_did(qapp, monkeypatch):
    """The Stars/background section was a title and nothing under it: it gives the method, the two images the block
    makes and the measure its result records."""
    t = text(_with_stars(monkeypatch, {"star_removal": {"grid_peaks": 2}}))
    section = t.split("2. Stars/background")[1].split("3. Recipe")[0]
    assert "Neural network" in section and "Starless image · Stars alone" in section
    assert "Network grid" in section and "taken off the starless image" in section
    i18n.set_language("fr")
    try:
        t = text(_with_stars(monkeypatch, {"star_removal": {"grid_peaks": 2}}))
        assert "Images produites" in t and "Image sans étoiles · Étoiles seules" in t and "Grille du réseau" in t
    finally:
        i18n.set_language("en")


def test_a_block_with_nothing_to_say_has_no_empty_numbered_section(qapp, monkeypatch):
    p = _with_stars(monkeypatch, {})
    t = text(p)
    assert "Stars/background" not in t.replace("Registration → Stars/background", "") and "2. Recipe" in t
    assert "Stars/background" not in [p.part.itemText(i).partition(". ")[2] for i in range(p.part.count())]


def test_nothing_computed_says_so(qapp, monkeypatch):
    p = page(monkeypatch, d=rep.ReportData())
    assert "Nothing computed yet" in text(p) and not p.card_ready()


@pytest.mark.parametrize("value,ok", [(5.0, True), ("drizzle", True), (r"F:\photos\raw", False), ("C:/a/b", False),
                                      ("/home/me/x.fits", False), (["a"], False), (None, False)])
def test_a_setting_that_names_a_place_stays_out(value, ok):
    assert rep.shareable(value) is ok


def test_the_exports_hold_the_report_and_no_path(qapp, monkeypatch, tmp_path):
    p = page(monkeypatch)
    assert not rep.PATHS.search(text(p))
    p.save_png(tmp_path / "r.png")
    p.save_pdf(tmp_path / "r.pdf")
    assert (tmp_path / "r.png").stat().st_size > 2000 and (tmp_path / "r.pdf").read_bytes()[:4] == b"%PDF"
    # the night test pass: 7 s without a sign, the file proposed in the program's own folder
    said, asked = [], []
    p.said.connect(said.append)
    monkeypatch.setattr(p.s, "root", tmp_path, raising=False)
    monkeypatch.setattr(rep.QFileDialog, "getSaveFileName",
                        lambda *a: (asked.append(a[2]), ((tmp_path / "x.pdf").as_posix(), ""))[1])
    p._ask_pdf()
    assert asked[0].startswith(tmp_path.as_posix()) and asked[0].endswith("_report.pdf")
    assert said[0].startswith("Writing x.pdf") and "x.pdf" in said[-1] and (tmp_path / "x.pdf").is_file()


def test_one_night_card_per_session_beside_the_image(qapp, monkeypatch, tmp_path):
    p = page(monkeypatch)
    assert p.card_ready()
    names = sorted(f.name for f in p.save_cards(tmp_path, "m45"))
    assert names == ["m45_night_card_1.png", "m45_night_card_2.png"]
    assert all((tmp_path / n).stat().st_size > 2000 for n in names)


def test_the_picture_lines_sum_every_session_the_picture_is_made_of():
    facts = {"stackcnt": 60, "exposure_s": 1500.0,
             "parts": [{"name": "Night 1", "frames": 20, "seconds": 300.0, "shared": 0},
                       {"name": "Night 2", "frames": 20, "seconds": 600.0, "shared": 0},
                       {"name": "Night 3", "frames": 20, "seconds": 600.0, "shared": 0}]}
    lines = rep.picture_lines(facts)
    assert lines[0] == "The picture: 60 frames · 25 min"
    assert lines[1:] == ["Night 1: 20 frames · 5 min", "Night 2: 20 frames · 10 min", "Night 3: 20 frames · 10 min"]
    assert rep.picture_lines({"stackcnt": 20, "exposure_s": 300.0, "parts": [{"name": "A", "frames": 20, "seconds": 300.0}]})         == ["The picture: 20 frames · 5 min"]
    assert rep.picture_lines(None) == [] and rep.picture_lines({"target": "x"}) == []


def test_the_cards_beside_an_image_say_what_the_whole_picture_is_made_of(qapp, monkeypatch, tmp_path):
    p = page(monkeypatch)
    facts = {"stackcnt": 60, "exposure_s": 900.0, "parts": [{"name": "A", "frames": 30, "seconds": 450.0, "shared": 0},
                                                              {"name": "B", "frames": 30, "seconds": 450.0, "shared": 0}]}
    plain = p.save_cards(tmp_path, "a")
    sized = [f.stat().st_size for f in plain]
    said = p.save_cards(tmp_path, "b", facts)
    assert len(said) == len(plain) == 2 and all(f.stat().st_size > 2000 for f in said)
    assert p.data.nights[0].foot and [f.read_bytes() for f in said] != [f.read_bytes() for f in plain] and sized


def test_the_picture_lines_and_the_loading_reason_are_in_french():
    i18n.set_language("fr")
    try:
        lines = rep.picture_lines({"stackcnt": 1, "exposure_s": 300.0, "parts": []})
        assert lines == ["L'image : 1 image · 5 min"]
        assert i18n._("Reading the registration result...") == "Lecture du résultat d'alignement..."
    finally:
        i18n.set_language("en")


def test_the_card_is_loading_not_missing_while_the_registration_is_read(qapp, monkeypatch):
    p = page(monkeypatch)
    p._token = None
    assert not p.card_loading()
    p._token = 7
    assert p.card_loading()
    p._ready("report", 7, rep.ReportData())
    assert not p.card_loading()


def test_the_brush_goes_to_1200_px_and_small_sizes_stay_on_the_slider():
    assert BRUSH_MAX == 1200.0 and brush_px(0.0) == BRUSH_MIN and brush_px(1.0) == 1200
    assert brush_px(brush_pos(60.0)) == 60
    assert brush_pos(60.0) > 0.4                         # not squeezed at the start of the rail


# ------------------------------------------------------------ the shooting details
def test_the_text_to_paste_says_what_is_known_in_the_language_of_the_forum(qapp, monkeypatch):
    p = page(monkeypatch)
    p.lang.setCurrentIndex(p.lang.findData("fr"))
    t = p.details()
    assert t.startswith("M 45") and "Appareil : A7S · ISO 5000" in t and "Poses : 11 × 15 s" in t and "Darks : 48" in t
    assert "Champ : 7,3° × 5,0°" in t and "Échantillonnage : 2,0\"/px" in t and "Traitement : FLOAT" in t
    assert "Monture" not in t and "Date" not in t                                      # nobody said
    p.lang.setCurrentIndex(p.lang.findData("en"))
    assert "Field: 7.3° × 5.0°" in p.details() and "Processing: FLOAT" in p.details()


def test_the_camera_clock_never_gives_the_date(qapp, monkeypatch):
    p = page(monkeypatch)
    assert not [r for r in rep.shot_rows(p.data, p.s.info(), {}, "en") if r[0] == "date" and r[2]]


def test_a_field_is_kept_in_the_project_and_goes_in_the_report(qapp, monkeypatch):
    p = page(monkeypatch)
    p.fields["instrument"].setText("Newton 130/900")
    p.set_field("instrument")
    assert p.s.p.acquisition == {"instrument": "Newton 130/900"} and "Newton 130/900" in text(p)
    assert "Instrument: Newton 130/900" in p.details()


def test_a_field_left_empty_is_forgotten(qapp, monkeypatch):
    p = page(monkeypatch)
    p.s.p.acquisition["mount"] = "EQ6"
    p.fields["mount"].setText("  ")
    p.set_field("mount")
    assert p.s.p.acquisition == {}


def test_an_empty_field_is_not_in_the_report(qapp, monkeypatch):
    assert "Mount" not in text(page(monkeypatch))


def test_every_section_says_what_it_is_for_and_every_figure_what_it_means(qapp, monkeypatch):
    p = page(monkeypatch)
    monkeypatch.setattr(p.steps, "facts", lambda q, key: [("Aligned frames", "30")], raising=False)
    t = text(p)
    assert "by matching their stars" in t and "Frames that could be aligned." in t


def test_the_reader_zooms_and_comes_back_to_the_width(qapp, monkeypatch):
    from PySide6.QtWidgets import QApplication
    from PySide6.QtPdfWidgets import QPdfView
    p = page(monkeypatch)
    p.resize(1200, 800)
    QApplication.processEvents()
    assert p.view.zoomMode() == QPdfView.ZoomMode.FitToWidth
    p.zoom_by(1.5)
    assert p.view.zoomMode() == QPdfView.ZoomMode.Custom
    was = p.view.zoomFactor()
    p.zoom_by(1.5)
    assert abs(p.view.zoomFactor() / was - 1.5) < 1e-6
    p.fit_width()
    assert p.view.zoomMode() == QPdfView.ZoomMode.FitToWidth


def test_the_text_and_the_curves_go_to_the_clipboard(qapp, monkeypatch):
    from PySide6.QtWidgets import QApplication
    p = page(monkeypatch)
    p.copy_details()
    assert QApplication.clipboard().text().startswith("M 45")


def test_the_recipe_gives_only_what_differs_from_the_defaults_and_the_appendix_all(qapp, monkeypatch):
    t = text(page(monkeypatch))
    recipe, appendix = t.split("4. Appendix")
    assert "Kappa" in recipe and "Method" not in recipe and "Method" in appendix


def test_the_details_stay_in_the_project_file(tmp_path):
    from astro_gpu.core import project2
    from astro_gpu.core.project import Target
    p = project2.create(tmp_path / "A", "A", Target())
    p.acquisition["mount"] = "EQ6"
    project2.save(p)
    assert project2.load(tmp_path / "A").acquisition == {"mount": "EQ6"}


def test_the_figures_have_a_decimal_comma_in_french():
    """The night test pass: the night card's axis read 0.5, 1.0, 1.5 where the rest of the program writes 0,5."""
    from matplotlib.figure import Figure
    i18n.set_language("fr")
    try:
        fig = Figure()
        ax = fig.add_subplot()
        ax.plot([0, 0.5, 1.5], [1, 2, 3])
        rep._png(fig)
        assert ax.xaxis.get_major_formatter()(0.5, 0) == "0,5"
    finally:
        i18n.set_language("en")


def test_the_session_card_quotes_only_what_was_measured():
    """The head of the report, made to be posted: its figures come from this session's results, and say how."""
    assert rep.sky_position(56.76679, 24.05996) == "03h47m04s +24°03′36″"
    assert rep.sky_position(83.82, -5.39) == "05h35m17s −05°23′24″"
    d = rep.ReportData(astro={"focal_estimate_mm": 893.7, "stars_used": 242, "residual_px": 0.189, "scale_arcsec": 1.9386},
                       metrics={"register": {"drift_px": 24.8, "transparency_p10_p90": [0.74, 1.02]},
                                "inventory": {"first_time": "2023-08-21T00:56:30", "last_time": "2023-08-21T00:59:20"},
                                "denoise": {"noise_reduction": 14.81}, "catalogs": {"stars_deep": 3203, "xp_spectra": 609},
                                "stack": {"channel_offsets": {"applied": True, "stars": 253,
                                                              "centre_px": {"R": [-0.471, 0.289], "B": [0.513, -0.304]}}}})
    got = {t: (v, h) for v, t, h in rep.highlights(d, "en")}
    assert got["Focal length, measured"][0] == "894 mm" and "242 stars, residual 0.37″" in got["Focal length, measured"][1]
    assert got["Field drift"][0] == "48″" and "17.0″ a minute" in got["Field drift"][1]
    assert got["Colors offset, corrected"][0] == "0.60 px" and got["Sky noise"][0] == "÷ 14.8"
    assert got["Gaia DR3 stars in the field"][0] == "3 203" and got["Transparency changes"][0] == "27 %"
    assert not rep.highlights(rep.ReportData(), "en")                      # nothing measured: nothing quoted


@pytest.mark.parametrize("image,sensor,line,table", [(2.0, 2.0, "2.00″/px", None),       # solved: under the title, once
                                                     (1.0, 2.0, "1.00″/px", "2.00″/px"),  # drizzled: two figures
                                                     (None, 2.0, None, "2.00″/px")])      # not solved: from the focal length
def test_the_first_page_gives_the_field_and_the_scale_once_written_one_way(qapp, monkeypatch, image, sensor, line, table):
    """#137: the scale was written 1.94″/px under the title and 1.9"/px in Table 1, and the field given in both."""
    d = data()
    d.reg.scale = sensor
    d.astro = dict(field_deg=[7.33, 5.04], **({"scale_arcsec": image} if image else {}))
    head = page(monkeypatch, d=d)._parts[0][1][1]
    under, tables = head.split("Table 1. Acquisition.")
    assert '"/px' not in head and head.count("7.3° × 5.0°") == 1 and "7.3° × 5.0°" in under
    assert (line is None or line in under) and (table is None or table in tables) and ("Sampling" in tables) == bool(table)
    assert head.count("″/px") == (line is not None) + (table is not None)


def test_without_a_scale_on_the_sky_the_star_size_is_given_in_pixels_saying_so(qapp, monkeypatch):
    """#135: no plate solving, no focal length: the stars' size is measured all the same, so the first page gives it,
    once, in pixels, and says why."""
    d = data()
    d.reg.scale, d.astro, d.metrics = None, {}, {"gradient": {"gradient_pct": 8.58, "samples_used": 261}}
    got = {t: (v, h) for v, t, h in rep.highlights(d, "en")}
    assert got["Star FWHM, measured"][0] == "2.5 px"
    assert "11 frames, best 2.1 px; in pixels: the scale on the sky is not known" in got["Star FWHM, measured"][1]
    head = page(monkeypatch, d=d)._parts[0][1][1]
    assert head.count("Star FWHM, measured") == 1 and head.count("2.5 px") == 1 and "Median FWHM" not in head


def test_a_smart_telescope_has_a_gain_and_the_bias_are_counted():
    from astro_gpu.client import report as rep
    d = rep.ReportData(gain=True)
    info = dict(camera="Seestar S50", iso_gain=80.0, n_darks=0, n_flats=30, n_bias=50)
    rows = {k: v for k, _w, v in rep.shot_rows(d, info, {}, "en")}
    assert rows["camera"] == "Seestar S50 · Gain 80" and rows["bias"] == "50" and rows["flats"] == "30"
    rows = {k: v for k, _w, v in rep.shot_rows(rep.ReportData(), info, {}, "fr")}
    assert rows["camera"] == "Seestar S50 · ISO 80"


def test_the_log_of_a_project_opened_again_is_its_last_run(tmp_path):
    from astro_gpu.client.logview import last_run_text
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "journal.md").write_text("## Run 1\n- a\n\n## Run 2 - stages: stack\n- b\n", encoding="utf-8")
    assert last_run_text(tmp_path) == "Run 2 - stages: stack\n- b"
    assert last_run_text(tmp_path / "none") == "" and last_run_text(None) == ""


def test_latest_reads_the_result_an_outdated_block_had_else_the_current_one(tmp_path):
    """The report of a project changed since its run (a night added) is made from the results it had."""
    from conftest import scratch_project

    from astro_gpu import api
    folder = scratch_project(tmp_path / "heart")
    p = api.open_project(folder)
    wf = p.workflows[0].id
    try:
        now = api.frame_table(p, wf)
    except api.NotDone:
        import pytest
        pytest.skip("the test project has no registration")
    assert [r.name for r in api.frame_table(p, wf, "latest").rows] == [r.name for r in now.rows]
    from astro_gpu.client.report import report_data
    d = report_data(p, wf, ("register", "stack"), "x")
    assert d.reg is not None and not d.outdated


def test_calibration_frames_two_sessions_share_are_counted_once():
    """Two sessions that use the same 10 darks and 30 flats: the report says 10 and 30, not 20 and 60."""
    from types import SimpleNamespace
    from astro_gpu.client import model as modelmod
    shared = {"darks": [f"/d/dark{i}.fits" for i in range(10)], "flats": [f"/d/flat{i}.fits" for i in range(30)]}
    files = {1: {"lights": [f"/a/l{i}.fits" for i in range(12)], **shared},
             2: {"lights": [f"/b/l{i}.fits" for i in range(12)], **shared}}
    me = SimpleNamespace(_facts=SimpleNamespace(inventory={}, files=files, register={}, stack={}), name="m", target="t",
                         sessions=[dict(lights=12, darks=10, flats=30, bias=0)] * 2)
    got = modelmod.ProjectModel.info(me)
    assert (got["n_lights"], got["n_darks"], got["n_flats"], got["n_bias"]) == (24, 10, 30, 0)


def test_a_run_that_ends_reads_the_report_again(qapp, monkeypatch):
    """The Report shown while a run ends: its figures, settings, night map and Outdated line follow, as its tables do
    (they stayed those of the reading before)."""
    p = page(monkeypatch)
    keys = {"stack": "k1"}
    p.s.state_info = lambda q, st: SimpleNamespace(state="done", key=keys.get(st), previous_key=None)
    asked = []
    monkeypatch.setattr(p.service, "request_job", lambda name, fn, *a: (asked.append(name), 7)[1])
    p.show()
    p.load()
    p._ready("report", 7, data())
    asked.clear()
    p.steps.changed.emit()                                       # outputs read again, the results as they were
    assert asked == []
    keys["stack"] = "k2"                                         # the run ended: another result
    p.steps.changed.emit()
    assert asked == ["report"]
    p.hide()


def test_a_failed_reading_says_so(qapp, monkeypatch):
    p = page(monkeypatch)
    p.load()
    p._failed("report", p._token, "the disk is gone")
    t = text(p)
    assert "Nothing computed yet" not in t and "the disk is gone" in t
    p.load()
    p._ready("report", p._token, data())                         # read again: the report, the reason gone
    assert "the disk is gone" not in text(p)


def test_every_value_of_the_french_report_is_in_french(qapp, monkeypatch, tmp_path):
    """The French report gave the settings' values as they are stored (pretrained, short, fallback, polynomial): every
    value it shows goes through the words the panels show for it. Walks the settings and the measures of the test
    project's results; a value the engine's catalogue knows must come out as the catalogue's French."""
    from conftest import HAVE_TEST_PROJECT, NO_TEST_PROJECT, scratch_project

    from astro_gpu import api, texts
    if not HAVE_TEST_PROJECT:
        pytest.skip(NO_TEST_PROJECT)
    p = api.open_project(scratch_project(tmp_path / "heart"))
    full = next(w for w in p.workflows if w.options.get("denoise") and w.options.get("stars"))
    d = rep.report_data(p, full.id, tuple(texts.stage_names()), "x")
    choices = [v for _b, _n, v, _c in d.settings if isinstance(v, rep.Choice)]
    assert {"pretrained", "short", "fallback", "polynomial"} <= set(choices)      # the test project's own
    i18n.set_language("fr")
    try:
        stored = []
        for v in choices:                                # a setting with choices: the panel's word for it
            en = texts.CHOICE_TEXTS.get(v, v)
            assert texts.has_translation(en, "fr"), f"no French for the choice {v!r} ({en})"
            assert rep.ReportPage._value(v) == rep.html.escape(texts.translate(en, "fr"))
            stored += [v] if texts.translate(en, "fr") == v and v in texts.CHOICE_TEXTS else []
        for stage, metrics in d.metrics.items():         # a measure that is a fixed word
            for key, v in metrics.items():
                if isinstance(v, str) and v in texts.VALUE_TEXTS:
                    assert texts.has_translation(texts.VALUE_TEXTS[v], "fr"), f"no French for {v!r} ({stage})"
                    assert texts.format_metric(key, v, "fr") == texts.translate(texts.VALUE_TEXTS[v], "fr")
        assert not stored, f"shown as stored: {stored}"
        t = text(page(monkeypatch, d=d))
        for word in ("pretrained", "short", "fallback", "polynomial", "continue"):
            assert not rep.re.search(rf"(?m)^{word}$", t), f"{word} is shown as it is stored"
        assert "Réseau pré-entraîné" in t and "Balance de repli" in t and "Polynôme" in t
    finally:
        i18n.set_language("en")


def test_a_typed_text_is_never_taken_for_a_choice(qapp):
    """A free text that reads like a stored choice (a target named "auto") stays what was typed."""
    assert rep.ReportPage._value("auto") == "auto" and rep.ReportPage._value(rep.Choice("auto")) == "Automatic"
