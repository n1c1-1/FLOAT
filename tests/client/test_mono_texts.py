"""Inspect follows the frames: a monochrome stack is one plane, without colour wording."""
from __future__ import annotations

import numpy as np

from astro_gpu.client import settings, steps
from astro_gpu.client.services import inspect as si


class _P:
    def __init__(self, mono):
        self.mono = mono

    def is_mono(self, wid):
        return self.mono


def test_a_mono_stack_of_three_equal_planes_is_shown_as_one_plane():
    a = np.random.default_rng(0).random((1, 40, 50)).astype(np.float32).repeat(3, axis=0)
    assert si._one_plane(_P(True), 1, a).shape == (1, 40, 50)
    assert si._one_plane(_P(False), 1, a).shape == (3, 40, 50)
    b = a.copy()
    b[2] += 1
    assert si._one_plane(_P(True), 1, b).shape == (3, 40, 50)


def test_the_stack_note_does_not_claim_a_green_cast():
    assert "green" not in steps.STACK_NOTE and "colour" not in steps.STACK_MONO_NOTE.split(":")[0]


def test_a_mono_stack_has_no_white_balance_setting():
    assert ("stack", "white_balance") in settings.NO_COLOUR_STACK
