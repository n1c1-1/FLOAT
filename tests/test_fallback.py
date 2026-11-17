"""The declared fallbacks of the Color and Astrometry blocks: a long run left alone does not stop on a field the stars
cannot calibrate or identify. Color: "If calibration fails" (Fallback balance | Stop) and "Minimum stars"; Astrometry:
"If the field is not identified" (Continue without solution | Stop), and what the blocks after it do without a plate
solution. The run's state "done with fallback" (core/stage.Context.fallback) is in tests/test_runner2.py and
tests/client/test_fallback_state.py."""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from astro_gpu.core.stage import StageError
from astro_gpu.stages import astrometry as am
from astro_gpu.stages import color as cm

#: the colour of every star of the synthetic field, and its sky
STAR_RGB, SKY = np.array([0.5, 1.0, 0.8]), np.array([0.10, 0.20, 0.15])


def field(n_side: int = 12, size: int = 900, seed: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """A linear image (3, size, size) of stars on a grid, all of colour STAR_RGB, on a coloured sky; their (x, y)."""
    rng = np.random.default_rng(seed)
    img = SKY[:, None, None] + rng.normal(0, 0.002, (3, size, size))
    yy, xx = np.mgrid[:size, :size]
    xy = []
    for gx in np.linspace(90, size - 90, n_side):
        for gy in np.linspace(90, size - 90, n_side):
            x, y = gx + rng.uniform(-3, 3), gy + rng.uniform(-3, 3)
            box = (slice(int(y) - 12, int(y) + 13), slice(int(x) - 12, int(x) + 13))
            psf = np.exp(-((xx[box] - x) ** 2 + (yy[box] - y) ** 2) / (2 * 1.6 ** 2)) * rng.uniform(0.05, 0.4)
            img[:, box[0], box[1]] += STAR_RGB[:, None, None] * psf
            xy.append((x, y))
    return img.astype(np.float32), np.array(xy)


class Ctx:
    """What a stage reads and writes, in a folder per stage; the metrics of the stages before it given."""

    def __init__(self, root, given: dict | None = None):
        self.root, self.given = root, dict(given or {})
        self.metrics, self.logs, self.fallbacks, self.said = {}, [], [], []
        self.dir = root / "out"
        self.dir.mkdir(parents=True, exist_ok=True)

    def has(self, stage):
        return stage != "inventory"

    def metric_of(self, stage, key, default=None):
        return self.given.get((stage, key), default)

    def input(self, stage, name):
        p = self.root / stage / name
        if not p.exists():
            raise StageError(f"missing input {stage}/{name}: run stage '{stage}' first")
        return p

    def load(self, stage, name, mmap=False):
        return np.load(self.input(stage, name))

    def load_json(self, stage, name):
        return json.loads(self.input(stage, name).read_text(encoding="utf-8"))

    def out(self, name):
        return self.dir / name

    def save(self, name, a):
        np.save(self.dir / name, a)

    def save_json(self, name, obj):
        (self.dir / name).write_text(json.dumps(obj), encoding="utf-8")

    def metric(self, k, v):
        self.metrics[k] = v

    def log(self, msg, level="info"):
        self.logs.append((level, msg))

    def progress(self, frac, msg="", **k):
        self.said.append(msg)

    def manifest(self, stage):
        p = self.root / stage / "manifest.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

    def fallback(self, why):
        self.fallbacks.append(why)


def _xp(root, xy, n: int):
    """The catalogue's colour stars: ``n`` of the field's stars with a spectrum."""
    (root / "catalogs").mkdir(parents=True, exist_ok=True)
    np.savez(root / "catalogs" / "gaia_xp.npz", x=xy[:n, 0], y=xy[:n, 1], xyz=np.ones((n, 3)), source=np.arange(n),
             g=np.full(n, 12.0))


@pytest.fixture
def colour_field(tmp_path, monkeypatch):
    img, xy = field()
    monkeypatch.setattr(cm, "image_in", lambda ctx: SimpleNamespace(img=img))
    monkeypatch.setattr(cm, "saturation_map", lambda ctx, clipped=False: None)
    return img, xy


def _spectra_installed(monkeypatch, installed: bool):
    from astro_gpu.core import sirilcat
    monkeypatch.setattr(sirilcat, "locate", lambda *a, **k: {"astro": None, "xp": "extract" if installed else None})


# ---------------------------------------------------------------------------------------------- the fallback balance
def test_the_fallback_balance_makes_the_sky_neutral_and_the_field_stars_white(colour_field):
    """Background neutralisation then the grey world on the stars: the sky level of each channel, and the factors that
    make the stars' median colour white (green 1)."""
    img, xy = colour_field
    b = cm.fallback_balance(img, 5.0)
    assert np.allclose(b["background"], SKY, atol=2e-3), b
    assert b["stars"] >= 100                    # none at the top of the range: every star of the grid away from the edge
    assert b["k"][1] == 1.0 and abs(b["k"][0] * STAR_RGB[0] - 1) < 0.03 and abs(b["k"][2] * STAR_RGB[2] - 1) < 0.03, b
    flat = np.full_like(img, 0.1)                                      # no star: the camera's colours kept
    assert cm.fallback_balance(flat, 5.0)["k"].tolist() == [1.0, 1.0, 1.0]


# ---------------------------------------------------------------------------------------------- the Color block
def test_too_few_stars_with_spectra_take_the_fallback_balance(tmp_path, colour_field, monkeypatch):
    """A narrow field with 10 stars with a spectrum, the spectra installed: the fallback balance (the default), said as
    the block's fallback in plain words (the colours not calibrated, and why: the block's orange tooltip), and by its
    progress; the log says why without asking to install anything."""
    img, xy = colour_field
    _spectra_installed(monkeypatch, True)
    _xp(tmp_path, xy, 10)
    ctx = Ctx(tmp_path)
    cm.Color().run(ctx, cm.ColorParams(aperture=5.0))
    assert ctx.fallbacks == ["Colors not calibrated, fallback balance: only 10 stars with spectra in this field "
                             "(minimum 50)"]
    assert ctx.said == ["colors not calibrated: fallback balance"]          # never "white balance on … stars"
    assert ("info", "only 10 stars with spectra in this field (minimum 50)") in ctx.logs
    assert not any("install" in m for _, m in ctx.logs)
    M = np.load(ctx.out("M.npy"))
    assert np.allclose(M, np.diag(np.diag(M)))
    assert abs(M[0, 0] / M[1, 1] * STAR_RGB[0] - 1) < 0.03 and abs(M[2, 2] / M[1, 1] * STAR_RGB[2] - 1) < 0.03
    assert ctx.metrics["white_balance_stars"] >= 30 and "chromaticity_residual" not in ctx.metrics
    assert "stars" not in ctx.metrics                                       # no star calibration recorded


def test_the_output_records_calibrated_colours_only_without_the_fallback(tmp_path):
    """The Output's ``colour_calibrated`` (render_simple.colour_calibrated): a Color block done the usual way, yes; one
    done with its fallback balance, no, even with another Color block calibrated; no Color block, no."""
    from astro_gpu.stages.render_simple import colour_calibrated
    for stage, m in (("color", {"status": "done"}), ("color~2", {"status": "done", "fallback": cm.FALLBACK_FIELD})):
        (tmp_path / stage).mkdir()
        (tmp_path / stage / "manifest.json").write_text(json.dumps(m))
    ctx = Ctx(tmp_path)
    assert colour_calibrated(ctx, ["gradient", "color"]) is True
    assert colour_calibrated(ctx, ["gradient", "color~2"]) is False
    assert colour_calibrated(ctx, ["color", "color~2"]) is False
    assert colour_calibrated(ctx, ["gradient"]) is False


# ---------------------------------------------------------------------------------------------- saturated stars
def _clipped_field(size: int = 700, seed: int = 3):
    """A linear image file's field on the 16-bit scale (the Image block's, stages/image.py): a sky of 2000, 36 stars
    on a grid whose peaks go from 3000 to twice the range, clipped at 65535 as the camera clips; their (x, y) and
    peaks before the clip."""
    rng = np.random.default_rng(seed)
    img = 2000.0 + rng.normal(0, 20, (3, size, size))
    yy, xx = np.mgrid[:size, :size]
    xy, peaks = [], np.geomspace(3000, 130000, 36)
    for (gx, gy), a in zip([(gx, gy) for gx in np.linspace(100, size - 100, 6) for gy in np.linspace(100, size - 100, 6)],
                           peaks):
        x, y = gx + rng.uniform(-0.4, 0.4), gy + rng.uniform(-0.4, 0.4)
        img += a * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * 1.8 ** 2))[None]
        xy.append((x, y))
    return np.minimum(img, 65535.0).astype(np.float32), np.array(xy), peaks


def _at_top(img, xy, aperture=5.0):
    """Each star: a pixel within the aperture radius + 0.5 px of it at or above 60000 on the 16-bit scale, pixel by
    pixel."""
    yy, xx = np.mgrid[:img.shape[1], :img.shape[2]]
    top = (img >= 60000 / 65535 * 65535).any(0)
    return np.array([top[(xx - x) ** 2 + (yy - y) ** 2 < (aperture + 0.5) ** 2].any() for x, y in xy])


def test_a_star_is_saturated_as_siril_judges_it_without_a_saturation_map():
    """Without the stack's map (an image file), a star is saturated when a pixel of its aperture (centre within the
    radius + 0.5 px) is at or above 60000 / 65535 of the range, the default maximum of the photometry of the reference
    program's colour calibrations; every star below it is measured, however bright against the others (the rule it
    replaces, a peak above 45 % of the brightest pixels, left out the bright half of the field)."""
    img, xy, peaks = _clipped_field()
    table, why = cm.colour_star_table(img, xy[:, 0], xy[:, 1], 5.0)
    top = _at_top(img, xy)
    assert 3 < top.sum() < 33
    assert np.array_equal(why == "saturated", top), (why, peaks)
    assert set(why[~top]) == {""} and np.isfinite(table[~top]).all()
    assert cm.SIRIL_MAXVAL == 60000 / 65535 and cm.SIRIL_APERTURE_MARGIN == 0.5
    # the aperture as the reference program weighs it: a clipped pixel 5.4 px from a star is in a 5 px aperture,
    # one 5.6 px away is not
    clip = np.zeros(img.shape[1:], bool)
    clip[300, 305] = True
    assert cm.clipped_in_aperture(clip, np.array([299.6, 299.4]), np.array([300.0, 300.0]), 5.0).tolist() == [True,
                                                                                                             False]
    # an image already stretched holds 0-1: its range is 1
    assert cm.clip_map(img / 65535.0, 1.0).sum() == cm.clip_map(img).sum() > 0


def test_the_color_block_judges_saturation_on_the_stack_of_an_image_file(tmp_path, monkeypatch):
    """The Color block on an image file (no saturation map): the clipped pixels are those of the stack's own values
    (the blocks before it, Gradient, move them), and the stars below the maximum calibrate the colours."""
    img, xy, peaks = _clipped_field()
    (tmp_path / "stack").mkdir()
    np.save(tmp_path / "stack" / "drz_full.npy", img)
    (tmp_path / "stack" / "drz_meta.json").write_text(json.dumps({"source": "image", "factor": 65535.0}))
    monkeypatch.setattr(cm, "image_in", lambda ctx: SimpleNamespace(img=img - 1500.0))      # a sky taken off
    monkeypatch.setattr(cm, "saturation_map", lambda ctx, clipped=False: None)
    sat, clip = cm.star_saturation(Ctx(tmp_path))
    assert sat is None and np.array_equal(clip, cm.clip_map(img))
    _xp(tmp_path, xy, len(xy))
    ctx = Ctx(tmp_path)
    cm.Color().run(ctx, cm.ColorParams(aperture=5.0, min_stars=20))
    n_top = int(_at_top(img, xy).sum())
    assert ctx.fallbacks == [] and ctx.metrics["stars"] == 36 - n_top
    ctx = Ctx(tmp_path)                                      # too few: the reason counts the saturated stars
    cm.Color().run(ctx, cm.ColorParams(aperture=5.0, min_stars=35))
    assert ctx.fallbacks == [f"Colors not calibrated, fallback balance: only {36 - n_top} of the 36 stars with spectra "
                             f"in this field could be measured (minimum 35), {n_top} saturated"]


def test_stop_keeps_the_failure_and_install_is_said_only_when_the_spectra_are_missing(tmp_path, colour_field,
                                                                                    monkeypatch):
    img, xy = colour_field
    _xp(tmp_path, xy, 10)
    p = cm.ColorParams(aperture=5.0, on_failure="stop")
    _spectra_installed(monkeypatch, True)
    with pytest.raises(StageError, match=r"^only 10 stars with spectra in this field \(minimum 50\)$"):
        cm.Color().run(Ctx(tmp_path), p)
    _spectra_installed(monkeypatch, False)
    with pytest.raises(StageError, match=r"only 10 star spectra of this field are on this computer \(minimum 50\): "
                                         r"install the star spectra"):
        cm.Color().run(Ctx(tmp_path), p)
    # enough stars with a spectrum, too few measurable: neither
    _xp(tmp_path, np.r_[xy[:5], np.full((60, 2), 10.0)], 65)              # 60 at the edge
    with pytest.raises(StageError, match=r"^only [0-5] of the 65 stars with spectra in this field could be measured "
                                         r"\(minimum 50\)$"):
        cm.Color().run(Ctx(tmp_path), p)
    ctx = Ctx(tmp_path)                                                   # Minimum stars: its own threshold
    with pytest.raises(StageError, match=r"\(minimum 120\)"):
        cm.Color().run(ctx, p.model_copy(update={"min_stars": 120}))
    with pytest.raises(ValueError):
        cm.ColorParams(on_failure="carry on")


def test_a_field_not_identified_takes_the_fallback_balance_or_stops(tmp_path, colour_field):
    given = {("catalogs", "solved"): False}                 # Astrometry continued without a plate solution
    ctx = Ctx(tmp_path, given)                              # (no gaia_xp.npz: never read)
    cm.Color().run(ctx, cm.ColorParams(aperture=5.0))
    assert ctx.fallbacks == [cm.FALLBACK_FIELD] and ctx.out("M.npy").is_file()
    with pytest.raises(StageError, match="the field was not identified"):
        cm.Color().run(Ctx(tmp_path, given), cm.ColorParams(on_failure="stop"))


# ---------------------------------------------------------------------------------------------- Astrometry and after it
def _stack(root, n_stars: int, monkeypatch):
    """A stack whose detection gives ``n_stars`` stars (too few to identify a field)."""
    (root / "stack").mkdir(parents=True, exist_ok=True)
    np.save(root / "stack" / "drz_full.npy", np.zeros((3, 400, 600), np.float32))
    (root / "stack" / "drz_meta.json").write_text(json.dumps({"scale": 1.0, "width": 600, "height": 400}))
    o = np.zeros(n_stars, dtype=[("x", float), ("y", float), ("flux", float), ("flag", int)])
    o["x"], o["y"], o["flux"] = np.linspace(50, 550, n_stars), np.linspace(50, 350, n_stars), 1.0
    monkeypatch.setattr(am, "detect", lambda g: o)


def test_a_field_not_identified_continues_without_plate_solution_or_stops(tmp_path, monkeypatch):
    _stack(tmp_path, 3, monkeypatch)
    ctx = Ctx(tmp_path)
    am.Astrometry().run(ctx, am.AstrometryParams(ra=10.0, dec=20.0, scale=2.0))
    assert ctx.fallbacks == ["Field not identified: continued without plate solution"]
    assert ctx.metrics["solved"] is False and not ctx.out("wcs.fits").exists()
    assert any("only 3 stars found in the stack" in m for _, m in ctx.logs)        # why, in the log
    with pytest.raises(am.FieldNotIdentified, match="only 3 stars found in the stack"):
        am.Astrometry().run(Ctx(tmp_path), am.AstrometryParams(ra=10.0, dec=20.0, scale=2.0, on_failure="stop"))
    with pytest.raises(ValueError):
        am.AstrometryParams(on_failure="go on")


def test_without_plate_solution_each_block_after_takes_its_own_fallback(tmp_path, monkeypatch, colour_field):
    """The chain after an Astrometry that continued: Catalogs and the H-alpha survey give none and say why; Gradient
    chooses its samples without the survey (its fallback, when it would have used it); Color its fallback balance;
    Align on hands the reference's (absent) solution on, its stars unaffected; the Synthetic stars of the instrument
    without a focal length cannot run and say why (no fallback for them), the image's own shape can."""
    from astro_gpu.stages.align_on import AlignOn
    from astro_gpu.stages.catalogs import Catalogs, CatalogsParams
    from astro_gpu.stages.gradient import FALLBACK_NO_SURVEY, GradientParams, survey_unsolved
    from astro_gpu.stages.surveys import Surveys, SurveysParams
    from astro_gpu.stages.synth_stars import SynthStars, SynthStarsParams
    unsolved = {("astrometry", "solved"): False}
    ctx = Ctx(tmp_path / "catalogs_run", unsolved)
    Catalogs().run(ctx, CatalogsParams())
    assert ctx.metrics == {"solved": False} and not list(ctx.dir.iterdir())
    ctx = Ctx(tmp_path / "surveys_run", unsolved)
    Surveys().run(ctx, SurveysParams())
    survey = json.loads(ctx.out("survey.json").read_text())
    assert survey["survey"] == "none" and survey["solved"] is False and ctx.metrics["survey"] == "none"
    # Gradient reads that survey.json
    (tmp_path / "surveys").mkdir()
    (tmp_path / "surveys" / "survey.json").write_text(json.dumps(survey))
    g = Ctx(tmp_path)
    assert survey_unsolved(g, GradientParams()) and FALLBACK_NO_SURVEY.startswith("Field not identified")
    assert not survey_unsolved(g, GradientParams(avoid_emission=False))        # it would not have used it: no fallback
    (tmp_path / "surveys" / "survey.json").write_text(json.dumps({"survey": "none", "why": "not on this computer"}))
    assert not survey_unsolved(g, GradientParams())                           # no survey for another reason: as before
    # Color, through Catalogs
    ctx = Ctx(tmp_path / "color_run", {("catalogs", "solved"): False})
    cm.Color().run(ctx, cm.ColorParams(aperture=5.0))
    assert ctx.fallbacks == [cm.FALLBACK_FIELD]
    # Align on: the reference's Astrometry block without a solution; its stages' metrics handed on, no wcs.fits
    ref = tmp_path / "ref_astrometry"
    ref.mkdir()
    (ref / "manifest.json").write_text(json.dumps({"metrics": {"solved": False}}))
    ctx = Ctx(tmp_path / "align_run")
    AlignOn.hand_on_solution(ctx, {"astrometry": str(ref)})
    assert ctx.metrics == {"solved": False} and not ctx.out("wcs.fits").exists()
    # Synthetic stars of the instrument: the scale is the solution's, the focal length unknown
    ctx = Ctx(tmp_path / "synth_run", {("astrometry", "solved"): False})
    ctx.project = SimpleNamespace(config=SimpleNamespace(target=SimpleNamespace(pixel_um=None, focal_mm=None)))
    with pytest.raises(StageError, match="the field was not identified .* focal length is not known"):
        SynthStars.instrument(ctx, SynthStarsParams(method="instrument"))
    ctx.given[("inventory", "focal_mm")] = 400.0                                # a focal length: it can
    assert SynthStars.instrument(ctx, SynthStarsParams(method="instrument")).focal_mm == 400.0
