"""The stack's automatic gain (stages/stack.sky_gain): sessions of two exposures whose calibrated levels are off by a
dark error of their own (data set 59: every level below 0, the gain was negative and the stack stopped on a
ValidationError) give the gain by the photon transfer slope; frames that give none leave the default gain, with one
line in the log; single-exposure sessions keep the gain they had. Synthetic frame statistics, no frame cache."""
from __future__ import annotations

import json

import numpy as np
import pytest

from astro_gpu.core.frame_weights import robust_sigma
from astro_gpu.stages.stack import Stack, StackParams, log_unmeasured_gain, sky_gain, transfer_slope

GAIN, RN, SCALE = 0.39, 20.0, 0.25          # e- per cache ADU, cache ADU, cache ADU per raw ADU (data set 59's figures)
SAMPLES = 40_000                            # sky samples of a frame's statistic


class Ctx:
    """What sky_gain and the stack's auto() read: frame_stats, register's results, calibrate's metrics, the inventory."""

    def __init__(self, tmp, stats, exposures, dark_noise=None, given=None):
        self.tmp, self.logs = tmp, []
        np.save(tmp / "calibrate_frame_stats.npy", stats)
        n = len(stats)
        np.savez(tmp / "register_reg.npz", fwhm=np.full(n, 4.0), rms=np.full(n, 0.2), transp=np.ones(n))
        self.metrics = {("calibrate", "cache_scale"): SCALE}
        if dark_noise:
            self.metrics[("calibrate", "dark_frame_noise")] = dark_noise
        self.jsons = {("calibrate", "cache.json"): {},
                      ("inventory", "frames.json"): {"lights": [{"path": f"f{i}.fits", "exposure": e}
                                                                for i, e in enumerate(exposures)]}}
        self.project = type("P", (), {"overrides": staticmethod(lambda stage: dict(given or {}))})()

    def input(self, stage, name):
        return self.tmp / f"{stage}_{name}"

    def metric_of(self, stage, key, default=None):
        return self.metrics.get((stage, key), default)

    def load_json(self, stage, name):
        return json.loads(json.dumps(self.jsons[(stage, name)]))

    def log(self, msg, level="info"):
        self.logs.append((msg, level))


def frames(rng, exposures, sky_rate, zero):
    """frame_stats (n, 4, 2) of frames of these exposures: sky electrons (``sky_rate`` e-/s, a value a frame) as
    Poisson counts / GAIN, read noise RN, and the level's zero of each exposure ``zero[t]`` (the dark match's error),
    measured as calibrate measures them (median, 1.4826 x MAD)."""
    out = np.zeros((len(exposures), 4, 2))
    for i, (t, r) in enumerate(zip(exposures, sky_rate)):
        for k in range(4):
            v = rng.poisson(r * t, SAMPLES) / GAIN + rng.normal(0, RN, SAMPLES) + zero[t]
            out[i, k] = robust_sigma(v)
    return out


def old_gain(stats, dark_noise):
    """The automatic gain as stack.py computed it before (one-point photon transfer, every frame alike)."""
    m, s = stats[:, :, 0].mean(1), stats[:, :, 1].mean(1)
    if dark_noise:
        return round(float(np.nanmedian(m / np.maximum(s ** 2 - (dark_noise * SCALE) ** 2, 1e-3))), 4)
    return round(float(np.nanmedian(m / np.maximum(s ** 2, 1e-3))), 4)


def mixed_session(rng):
    """29 frames of 300 s and 18 of 60 s (data set 59's red), sky from 0.15 to 0.6 e-/s (moon, altitude), each exposure
    with its dark matched wrongly: the levels of 300 s 250 ADU below 0, those of 60 s 480 (as calibrated there)."""
    exp = [300.0] * 29 + [60.0] * 18
    rate = np.r_[rng.uniform(0.15, 0.6, 29), rng.uniform(0.6, 2.4, 18)]
    return exp, frames(rng, exp, rate, {300.0: -250.0, 60.0: -480.0})


def test_two_exposures_give_the_gain_by_the_transfer_slope(tmp_path):
    rng = np.random.default_rng(59)
    exp, st = mixed_session(rng)
    assert old_gain(st, None) < 0                                  # the failure: StackParams refused it
    g, how = sky_gain(Ctx(tmp_path, st, exp))
    assert how == "transfer"
    # tolerance from the noise: the slope's own standard error (the MAD sigma of 40 000 samples is good to about
    # 0.6 %, its variance to 1.2 %), 4 of them; and that error is small (under 3 % of the slope)
    m, v = st[:, :, 0].mean(1), st[:, :, 1].mean(1) ** 2
    slope, se = transfer_slope(m, v, np.array([0 if e == 300 else 1 for e in exp]))
    assert se < 0.03 * slope
    assert abs(1 / g - 1 / GAIN) < 4 * se + 1e-4
    assert g == pytest.approx(GAIN, rel=0.05)


def test_the_stack_auto_gives_it_and_it_validates(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from astro_gpu.core.params import Auto
    from astro_gpu.stages import stack
    rule = SimpleNamespace(scale=Auto(1.0, "rule"), pixfrac=Auto(1.0, "rule"), method=Auto("drizzle", "rule"))
    monkeypatch.setattr(stack, "drizzle_suggestion", lambda *a, **k: rule)
    exp, st = mixed_session(np.random.default_rng(1))
    auto = Stack().auto(Ctx(tmp_path, st, exp, dark_noise=80.0))
    assert auto["gain"].value == pytest.approx(GAIN, rel=0.05) and "photon transfer" in auto["gain"].reason
    StackParams(**{k: a.value for k, a in auto.items() if k in StackParams.model_fields})


def test_a_sky_without_spread_gives_no_gain_and_one_log_line(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from astro_gpu.core.params import Auto
    from astro_gpu.stages import stack
    rule = SimpleNamespace(scale=Auto(1.0, "rule"), pixfrac=Auto(1.0, "rule"), method=Auto("drizzle", "rule"))
    monkeypatch.setattr(stack, "drizzle_suggestion", lambda *a, **k: rule)
    rng = np.random.default_rng(3)
    exp = [300.0] * 6 + [60.0] * 4
    st = frames(rng, exp, np.full(10, 0.3), {300.0: -250.0, 60.0: -480.0})      # the same sky in every frame
    ctx = Ctx(tmp_path, st, exp)
    assert sky_gain(ctx) == (None, "mixed exposures")
    auto = Stack().auto(ctx)
    assert "gain" not in auto
    p = StackParams(**{k: a.value for k, a in auto.items() if k in StackParams.model_fields})
    assert p.gain == StackParams().gain
    log_unmeasured_gain(ctx, p)
    assert ctx.logs == [(f"automatic gain not measurable on these frames (mixed exposures): the noise model with its "
                         f"default gain of {p.gain:g} e-/ADU", "warning")]
    hand = Ctx(tmp_path, st, exp, given={"gain": 0.2})                         # set by hand: nothing to say
    log_unmeasured_gain(hand, p)
    assert not hand.logs


def test_the_log_line_is_translated():
    from astro_gpu.client import i18n
    from astro_gpu.client.services.run import engine_text
    i18n.set_language("fr")
    try:
        for why, fr in (("mixed exposures", "poses mélangées"), ("sky level not above zero", "zéro"),
                        ("outside the range of a sensor", "capteur")):
            out = engine_text(f"automatic gain not measurable on these frames ({why}): the noise model with its "
                              "default gain of 0.13 e-/ADU")
            assert out.startswith("gain automatique non mesurable") and fr in out and "0,13" in out
    finally:
        i18n.set_language("en")


@pytest.mark.parametrize("dark_noise", [None, 30.0])
def test_one_exposure_keeps_the_gain_it_had(tmp_path, dark_noise):
    """Single-exposure sessions: the same number as before, to the last digit (the dark match right or not)."""
    rng = np.random.default_rng(7)
    exp = [120.0] * 30
    for zero in (0.0, 40.0):
        st = frames(rng, exp, rng.uniform(0.5, 3.0, 30), {120.0: zero})
        g, _ = sky_gain(Ctx(tmp_path, st, exp, dark_noise=dark_noise))
        assert g == old_gain(st, dark_noise) and g > 0
    # and without the exposures (an inventory of an older format): the same
    ctx = Ctx(tmp_path, st, exp, dark_noise=dark_noise)
    ctx.jsons.pop(("inventory", "frames.json"))
    assert sky_gain(ctx)[0] == old_gain(st, dark_noise)


def test_one_exposure_below_zero_falls_back_instead_of_stopping(tmp_path):
    """The one-point estimate of a single exposure whose levels are below 0: no automatic gain, never an error."""
    rng = np.random.default_rng(11)
    exp = [300.0] * 12
    st = frames(rng, exp, rng.uniform(0.2, 0.5, 12), {300.0: -300.0})
    assert old_gain(st, None) < 0
    assert sky_gain(Ctx(tmp_path, st, exp)) == (None, "sky level not above zero")
