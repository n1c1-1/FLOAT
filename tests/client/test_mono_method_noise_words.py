"""A monochrome stack is not said in colour words: its method line names no demosaicing (a beta tester read "Classic
stack · RCD · Lanczos-4" on his mono stacks), and its noise is one value, not "R nan · G 0.470 · B nan"."""
from __future__ import annotations

from types import SimpleNamespace

from astro_gpu.client import steps as steps_mod
from astro_gpu.client.tune import _noise_text


class _Model:
    def __init__(self, mono, colour=False):
        self._mono, self._colour = mono, colour

    def is_mono(self, _p):
        return self._mono

    def colour_frames(self, _p):
        return self._colour


def _method(mono, colour=False):
    st = steps_mod.Steps.__new__(steps_mod.Steps)          # only what the method line reads
    st.s = _Model(mono, colour)
    out = SimpleNamespace(params={"method": "classic", "classic_debayer": "rcd", "classic_interpolation": "lanczos4"})
    st.outputs = lambda p, stage, which="current": out
    st.which_for = lambda p, stage: "current"
    return st._method_fact(SimpleNamespace(pid=1))[1]


def test_the_method_line_names_demosaicing_only_for_a_bayer_sensor():
    assert "RCD" in _method(mono=False)
    assert "RCD" not in _method(mono=True) and "Classic" in _method(mono=True)
    assert "RCD" not in _method(mono=False, colour=True)


def test_the_noise_of_a_stack_measured_on_one_channel_is_one_value():
    assert _noise_text({"R": float("nan"), "G": 0.4702, "B": None}) == "0.470 ADU"
    assert _noise_text({"R": 1.0, "G": 2.0, "B": 3.0}) == "R 1.000 ADU · G 2.000 ADU · B 3.000 ADU"
