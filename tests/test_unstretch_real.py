"""Issue #174: the undo against the reference program itself. A small linear FITS (a real stack's sky level and dynamic:
sky near 0.0005 of white, bright stars) is stretched by the program's own command line, saved, and loaded here: the
history lines it writes are the ones the parser reads, and the exact undo brings the linear image back (error far under
the sky noise), or is refused for the reason the line gives. Skipped when the program is not installed."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from astro_gpu.stages import unstretch as U
from astro_gpu.stages.image import read_image

EXE = next((p for p in (shutil.which("siril-cli"), r"C:\Program Files\Siril\bin\siril-cli.exe") if p and Path(p).is_file()),
           None)
pytestmark = pytest.mark.skipif(EXE is None, reason="the reference program is not installed")

rng = np.random.default_rng(1740)


def stack(mono: bool, n=160):
    a = 0.0006 + 0.00015 * rng.standard_normal((1 if mono else 3, n, n))
    yy, xx = np.mgrid[:n, :n]
    for _ in range(12):
        y0, x0 = rng.integers(10, n - 10, 2)
        a += rng.uniform(0.05, 0.9) * np.exp(-((yy - y0) ** 2 + (xx - x0) ** 2) / 5.0)
    return np.clip(a, 1e-5, 1.0).astype(np.float32)


def stretch(tmp: Path, name: str, lin: np.ndarray, command: str) -> Path:
    src = tmp / f"{name}_lin.fits"
    # stored as a FITS stack is, bottom row first and no ROWORDER (issue #174): the program keeps the row order
    fits.PrimaryHDU(np.ascontiguousarray((lin if lin.shape[0] == 3 else lin[0])[..., ::-1, :])).writeto(src, overwrite=True)
    (tmp / f"{name}.ssf").write_text(f"requires 1.2.0\nload {src.stem}.fits\n{command}\nsave {name}_st\n")
    r = subprocess.run([EXE, "-d", str(tmp), "-s", f"{name}.ssf"], capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout[-600:]
    return tmp / f"{name}_st.fits"


def error(path: Path, lin: np.ndarray):
    """(analysis, 99th percentile and median error of the undo against the linear image, on the pixels the stretch did
    not clip; the highlights of a stretch that flattens them, the exponential form of the GHS, are where the inverse is
    least conditioned: their few pixels are the maximum, not the percentile)."""
    an = U.analyse(path)
    if not an.exact:
        return an, None, None
    a = read_image(path)
    y = a / U.scale_of(a)
    got = U.undo(y, an.ops, an.mono)
    want = np.broadcast_to(lin, got.shape)
    ok = ~np.any((y <= 0) | (y >= 1), axis=0)
    d = np.abs(got - want)[:, ok]
    return an, float(np.percentile(d, 99)), float(np.median(d))


NOISE = 0.00015


@pytest.mark.parametrize("mono,command", [
    (False, "mtf 0.0 0.2 1.0"),
    (True, "mtf 0.0 0.2 1.0"),
    (False, "asinh -human 10"),
    (True, "asinh 10"),
    (False, "asinh 12.5"),                                # written exactly as typed: the exact undo applies
    (True, "ght -D=3 -B=1 -SP=0.01"),
    (True, "ght -D=3 -B=0 -SP=0.01"),
    (True, "modasinh -D=3 -SP=0.01"),
])
def test_the_exact_undo_applies_to_what_the_program_writes(tmp_path, mono, command):
    lin = stack(mono)
    f = stretch(tmp_path, "c", lin, command)
    an, mx, med = error(f, lin)
    assert an.stretched and an.exact, (command, an.refusal)
    assert med < 1e-6 and mx < NOISE, (command, mx, med)


def test_refused_with_the_line_named(tmp_path):
    for mono, command, word in ((False, "autostretch -linked", "recipe"), (False, "autostretch", "recipe"),
                                (True, "autostretch", "recipe"), (False, "ght -D=3 -B=1 -SP=0.01", "colour")):
        lin = stack(mono)
        an = U.analyse(stretch(tmp_path, "r", lin, command))
        assert an.stretched and not an.exact and word in an.refusal[0], (command, an.refusal)


def test_a_black_point_the_program_rounds_away_is_refused(tmp_path):
    """"asinh 10 0.001" is written "offset: 0.0": the used value is not the written one, and it sits above the sky, so
    the stretch clipped the sky itself: refused, as the data shows it."""
    lin = stack(False)
    f = stretch(tmp_path, "o", lin, "asinh 10 0.001")
    assert "offset: 0.0" in " ".join(fits.getheader(f)["HISTORY"])
    an = U.analyse(f)
    assert not an.exact and "clipped the sky" in an.refusal[0]
