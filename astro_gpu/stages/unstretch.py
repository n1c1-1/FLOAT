"""An image already stretched, brought back to linear (issue #174).

Three questions, answered here from the file alone:

* ``analyse(path)``: is this image stretched, why do we think so, and can its stretch be undone exactly?
* ``undo(a, ops)``: the exact undo, from the stretches the file's header says were applied (FITS HISTORY lines, as the
  reference stacking program writes them: the formats are in ``_TABLE``);
* ``estimate(a)``: the estimated undo, for an image without such a header (a mid-tones transfer whose parameters are
  found from the image itself).

The header is read for what it says, never "to process": its lines are written "just for information and image tracking
purposes" by the program that writes them (V. Hourdin, discuss.pixls.us/t/history-in-fits-files/33770). We follow it
only when every line after the first stretch is one we can invert exactly; otherwise the exact undo is refused with the
line that stops it, and the estimated one stays offered.

Sources of the inverse formulas (each function's docstring names its own):

* mid-tones transfer function: ``MTF_pseudoinverse`` of the reference program's ``filters/mtf.c`` (exact except for the
  pixels the stretch clipped: its comment); the forward form is its ``MTF`` (same file);
* generalised hyperbolic stretch (GHS): D. Payne and M. Cranfield, https://ghsastro.co.uk (the equations of the
  transform and of its inverse, for b < -1... b > 0, and its arcsinh variant), as ``GHTsetup`` / ``GHT`` of ``filters/ght.c``;
* arcsinh stretch: ``asinhlut`` of ``filters/asinh.c`` (out = p * asinh(beta x) / (x asinh beta), x the luminance).
"""
from __future__ import annotations

import functools
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: the sky level (median / white) above which an image without a stretch in its header is called stretched: half the
#: sky target of an automatic stretch (0.25, ``AS_DEFAULT_TARGET_BACKGROUND``); the user's decision (issue #174)
SKY_LIMIT = 0.125

#: the substrings of the reference program's own test of "this image was stretched" (``fit_appears_stretched``), and the
#: form of its newer releases ("... MTF stretch")
STRETCH_WORDS = ("histogram transf.", "asinh", "midtones", "autostretch", "ghs", "mtf stretch", "autoghs")
#: lines of a spatial change, which a stretch's undo passes (the pixels move, the levels do not)
SPATIAL = re.compile(r"^(crop|resample|rotat|mirror|flip|binning)", re.I)

_NUM = r"([-+]?\d+(?:\.\d+)?)"


def _n(text: str) -> tuple[float, float]:
    """(value, step): a number as written, and the size of its last printed digit."""
    dec = len(text.split(".")[1]) if "." in text else 0
    return float(text), 10.0 ** -dec


@dataclass(frozen=True)
class Op:
    """One stretch the header says was applied. ``params``: ((name, value, step), ...); ``chan``: the one channel an
    unlinked stretch acted on (None: all); ``line``: the header line."""
    kind: str
    line: str
    params: tuple = ()
    chan: int | None = None
    flag: str = ""

    def p(self, over: dict | None = None) -> dict:
        d = {n: v for n, v, _s in self.params}
        d.update(over or {})
        return d


# ------------------------------------------------------------------------------------------------ the formulas

def mtf(x, m, lo, hi):
    """The mid-tones transfer function of the reference program (``MTF``, filters/mtf.c): x below lo gives 0, above hi
    gives 1, between them (m - 1) xp / ((2m - 1) xp - m), xp = (x - lo) / (hi - lo)."""
    xp = (x - lo) / (hi - lo)
    with np.errstate(divide="ignore", invalid="ignore"):
        y = ((m - 1.0) * xp) / ((2.0 * m - 1.0) * xp - m)
    return np.where(x <= lo, 0.0, np.where(x >= hi, 1.0, y))


def mtf_inverse(y, m, lo, hi):
    """The pseudo-inverse of the mid-tones transfer (``MTF_pseudoinverse``, filters/mtf.c, verbatim): exact for every
    pixel the stretch did not clip (its comment: "in the general case this cannot return the inverse of MTF() because
    MTF() may clip")."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return (((lo + hi) * m - lo) * y - lo * m + lo) / ((2.0 * m - 1.0) * y - m + 1.0)


def _setup(B, D, LP, SP, HP, kind):
    """The coefficients of the generalised hyperbolic stretch (``GHTsetup``, filters/ght.c; equations of
    https://ghsastro.co.uk): kind "normal", "inverse" (its inverse), "asinh" or "invasinh"."""
    c: dict[str, float] = {"a1": 0.0}
    log1p, exp, pw, lg = math.log1p, math.exp, math.pow, math.log
    if kind in ("normal", "inverse"):
        inv = kind == "inverse"
        if B == -1.0:
            qlp = -log1p(D * (SP - LP))
            q0 = qlp - D * LP / (1 + D * (SP - LP))
            qwp = log1p(D * (HP - SP))
            q1 = qwp + D * (1 - HP) / (1 + D * (HP - SP))
            q = 1 / (q1 - q0)
            if not inv:
                c.update(b1=(1 + D * (SP - LP)) / (D * q), a2=-q0 * q, b2=-q, c2=1 + D * SP, d2=-D,
                         a3=-q0 * q, b3=q, c3=1 - D * SP, d3=D,
                         a4=(qwp - q0 - D * HP / (1 + D * (HP - SP))) * q, b4=q * D / (1 + D * (HP - SP)))
            else:
                # the stretch at the symmetry point is -q0 q (a2 + b2 log 1); filters/ght.c writes q0 q here, which puts the
                # threshold on the wrong side: the inverse is the exact inverse of the stretch
                c.update(LPT=(qlp - q0) * q, SPT=-q0 * q, HPT=(qwp - q0) * q, b1=(1 + D * (SP - LP)) / (D * q),
                         a2=(1 + D * SP) / D, b2=-1 / D, c2=-q0, d2=-1 / q,
                         a3=-(1 - D * SP) / D, b3=1 / D, c3=q0, d3=1 / q,
                         a4=HP + (q0 - qwp) * (1 + D * (HP - SP)) / D, b4=(1 + D * (HP - SP)) / (q * D))
        elif B < 0.0:
            B = -B
            qlp = (1 - pw(1 + D * B * (SP - LP), (B - 1) / B)) / (B - 1)
            q0 = qlp - D * LP * pw(1 + D * B * (SP - LP), -1 / B)
            qwp = (pw(1 + D * B * (HP - SP), (B - 1) / B) - 1) / (B - 1)
            q1 = qwp + D * (1 - HP) * pw(1 + D * B * (HP - SP), -1 / B)
            q = 1 / (q1 - q0)
            if not inv:
                c.update(b1=D * pw(1 + D * B * (SP - LP), -1 / B) * q,
                         a2=(1 / (B - 1) - q0) * q, b2=-q / (B - 1), c2=1 + D * B * SP, d2=-D * B, e2=(B - 1) / B,
                         a3=(-1 / (B - 1) - q0) * q, b3=q / (B - 1), c3=1 - D * B * SP, d3=D * B, e3=(B - 1) / B,
                         a4=(qwp - q0 - D * HP * pw(1 + D * B * (HP - SP), -1 / B)) * q,
                         b4=D * pw(1 + D * B * (HP - SP), -1 / B) * q)
            else:
                c.update(LPT=(qlp - q0) * q, SPT=-q0 * q, HPT=(qwp - q0) * q,
                         b1=pw(1 + D * B * (SP - LP), 1 / B) / (q * D),
                         a2=(1 + D * B * SP) / (D * B), b2=-1 / (D * B), c2=-q0 * (B - 1) + 1, d2=(1 - B) / q,
                         e2=B / (B - 1),
                         a3=(D * B * SP - 1) / (D * B), b3=1 / (D * B), c3=1 + q0 * (B - 1), d3=(B - 1) / q,
                         e3=B / (B - 1),
                         a4=(q0 - qwp) / (D * pw(1 + D * B * (HP - SP), -1 / B)) + HP,
                         b4=1 / (D * pw(1 + D * B * (HP - SP), -1 / B) * q))
        elif B == 0.0:
            qlp = exp(-D * (SP - LP))
            q0 = qlp - D * LP * exp(-D * (SP - LP))
            qwp = 2 - exp(-D * (HP - SP))
            q1 = qwp + D * (1 - HP) * exp(-D * (HP - SP))
            q = 1 / (q1 - q0)
            if not inv:
                c.update(b1=D * exp(-D * (SP - LP)) * q, a2=-q0 * q, b2=q, c2=-D * SP, d2=D,
                         a3=(2 - q0) * q, b3=-q, c3=D * SP, d3=-D,
                         a4=(qwp - q0 - D * HP * exp(-D * (HP - SP))) * q, b4=D * exp(-D * (HP - SP)) * q)
            else:
                c.update(LPT=(qlp - q0) * q, SPT=(1 - q0) * q, HPT=(qwp - q0) * q,
                         b1=1 / (D * exp(-D * (SP - LP)) * q), a2=SP, b2=1 / D, c2=q0, d2=1 / q,
                         a3=SP, b3=-1 / D, c3=2 - q0, d3=-1 / q,
                         a4=(q0 - qwp) / (D * exp(-D * (HP - SP))) + HP, b4=1 / (D * exp(-D * (HP - SP)) * q))
        else:
            qlp = pw(1 + D * B * (SP - LP), -1 / B)
            q0 = qlp - D * LP * pw(1 + D * B * (SP - LP), -(1 + B) / B)
            qwp = 2 - pw(1 + D * B * (HP - SP), -1 / B)
            q1 = qwp + D * (1 - HP) * pw(1 + D * B * (HP - SP), -(1 + B) / B)
            q = 1 / (q1 - q0)
            if not inv:
                c.update(b1=D * pw(1 + D * B * (SP - LP), -(1 + B) / B) * q,
                         a2=-q0 * q, b2=q, c2=1 + D * B * SP, d2=-D * B, e2=-1 / B,
                         a3=(2 - q0) * q, b3=-q, c3=1 - D * B * SP, d3=D * B, e3=-1 / B,
                         a4=(qwp - q0 - D * HP * pw(1 + D * B * (HP - SP), -(B + 1) / B)) * q,
                         b4=D * pw(1 + D * B * (HP - SP), -(B + 1) / B) * q)
            else:
                c.update(LPT=(qlp - q0) * q, SPT=(1 - q0) * q, HPT=(qwp - q0) * q,
                         b1=1 / (D * pw(1 + D * B * (SP - LP), -(1 + B) / B) * q),
                         a2=1 / (D * B) + SP, b2=-1 / (D * B), c2=q0, d2=1 / q, e2=-B,
                         a3=-1 / (D * B) + SP, b3=1 / (D * B), c3=2 - q0, d3=-1 / q, e3=-B,
                         a4=(q0 - qwp) / (D * pw(1 + D * B * (HP - SP), -(B + 1) / B)) + HP,
                         b4=1 / (D * pw(1 + D * B * (HP - SP), -(B + 1) / B) * q))
        return c
    # the arcsinh variant
    s1 = math.sqrt(D * D * (SP - LP) ** 2 + 1)
    s2 = math.sqrt(D * D * (HP - SP) ** 2 + 1)
    qlp = -lg(D * (SP - LP) + s1)
    q0 = qlp - LP * D / s1
    qwp = lg(D * (HP - SP) + s2)
    q1 = qwp + (1 - HP) * D / s2
    q = 1 / (q1 - q0)
    c.update(b1=D / s1 * q, a2=-q0 * q, b2=-q, c2=-D, d2=D * D, e2=SP, a3=-q0 * q, b3=q, c3=D, d3=D * D, e3=SP,
             a4=(qwp - HP * D / s2 - q0) * q, b4=D / s2 * q)
    if kind == "invasinh":
        c["LPT"] = c["a1"] + c["b1"] * LP
        c["SPT"] = c["a2"] + c["b2"] * lg(c["c2"] * (SP - c["e2"]) + math.sqrt(c["d2"] * (SP - c["e2"]) ** 2 + 1))
        c["HPT"] = c["a4"] + c["b4"] * HP
    return c


def ght(x, B, D, LP, SP, HP, kind):
    """The generalised hyperbolic stretch of one channel (``GHT``, filters/ght.c, black point 0): ``kind`` "normal" is
    the stretch, "inverse" its inverse (``STRETCH_PAYNE_INVERSE``), "asinh" / "invasinh" the arcsinh form and its
    inverse. D is the amount as the reference program writes it in its log (``expm1`` of the amount asked)."""
    x = np.asarray(x, np.float64)
    if kind in ("normal", "inverse", "asinh"):
        x = np.maximum(x, 0.0)
    if D == 0.0:
        return x
    c = _setup(B, D, LP, SP, HP, kind)
    t = (c["LPT"], c["SPT"], c["HPT"]) if kind in ("inverse", "invasinh") else (LP, SP, HP)
    with np.errstate(all="ignore"):
        if kind == "asinh":
            def mid(i):
                u = x - c[f"e{i}"]
                return c[f"a{i}"] + c[f"b{i}"] * np.log(c[f"c{i}"] * u + np.sqrt(c[f"d{i}"] * u * u + 1.0))
            r1 = c["a1"] + c["b1"] * x
        elif kind == "invasinh":
            def mid(i):
                ex = np.exp((c[f"a{i}"] - x) / c[f"b{i}"])
                return c[f"e{i}"] - (ex - 1.0 / ex) / (2.0 * c[f"c{i}"])
            r1 = (x - c["a1"]) / c["b1"]
        else:
            fam = ("log" if B == -1.0 else "exp" if B == 0.0 else "pow") if kind == "normal" else \
                  ("exp" if B == -1.0 else "log" if B == 0.0 else "pow")

            def mid(i):
                a, b, cc, d = c[f"a{i}"], c[f"b{i}"], c[f"c{i}"], c[f"d{i}"]
                if fam == "log":
                    return a + b * np.log(cc + d * x)
                if fam == "exp":
                    return a + b * np.exp(cc + d * x)
                return a + b * np.power(cc + d * x, c[f"e{i}"])
            r1 = c["a1"] + c["b1"] * x if B == 0.0 else c["b1"] * x
        r4 = (x - c["a4"]) / c["b4"] if kind == "invasinh" else c["a4"] + c["b4"] * x
        return np.where(x < t[0], r1, np.where(x < t[1], mid(2), np.where(x < t[2], mid(3), r4)))


def undo_asinh(y, beta, offset, human, mono):
    """The inverse of the arcsinh stretch (``asinhlut``, filters/asinh.c): the stretch is, with p the channels after the
    black point (p = max(0, (v - offset) / (1 - offset))) and x their luminance (Rec.709 weights when ``human``, else
    one third each; the channel itself on a monochrome image), out = p * k(x), k(x) = asinh(beta x) / (x asinh(beta)).
    The luminance of out is therefore asinh(beta x) / asinh(beta), whence x = sinh(asinh(beta) Y) / beta, then
    p = out / k(x), then v = p (1 - offset) + offset. Exact except where the stretch clipped (p = 0, out = 1)."""
    y = np.clip(np.asarray(y, np.float64), 0.0, 1.0)
    if beta == 0.0:
        p = y
    else:
        w = np.array([1.0] if mono else ([0.2126, 0.7152, 0.0722] if human else [1 / 3] * 3))
        lum = np.tensordot(w, y[: len(w)], axes=1) if not mono else y[0]
        with np.errstate(all="ignore"):
            x = np.sinh(math.asinh(beta) * lum) / beta
            k = np.where(x > 0, np.arcsinh(beta * x) / (x * math.asinh(beta)), 0.0)
            p = np.where(k > 0, y / np.where(k > 0, k, 1.0), 0.0)
    return p * (1.0 - offset) + offset


# ------------------------------------------------------------------------------------------------ the header

@dataclass(frozen=True)
class Plan:
    """What the header says: ``stretch`` (a stretch line is in it), the stretches from the first on (``ops``), and
    ``refusal`` ((template, fields) of why they cannot be undone exactly, None when they can)."""
    stretch: bool
    ops: tuple
    refusal: tuple | None
    first: str = ""


def _is_stretch_line(line: str) -> bool:
    low = line.lower()
    if "linear bp" in low or "bp shift" in low:
        return False
    return any(w in low for w in STRETCH_WORDS)


_TABLE = (
    (re.compile(rf"^Histogram Transf\. \(mid={_NUM}, lo={_NUM}, hi={_NUM}\)$"), "mtf", ("mid", "lo", "hi"), ""),
    (re.compile(rf"^(Inverse midtones|Midtones) transfer \({_NUM}, {_NUM}, {_NUM}\)$"), "mtf", ("lo", "mid", "hi"), 1),
    (re.compile(rf"^Linked MTF stretch \(mid={_NUM}, lo={_NUM}, hi={_NUM}\)$"), "mtf", ("mid", "lo", "hi"), ""),
    (re.compile(rf"^Inverse MTF stretch \(lo={_NUM}, mid={_NUM}, hi={_NUM}\)$"), "mtfinv", ("lo", "mid", "hi"), ""),
    (re.compile(rf"^Unlinked MTF stretch \(lo={_NUM}, mid={_NUM}, hi={_NUM}\)$"), "mtf", ("lo", "mid", "hi"), ""),
    # the program's dialogs, written to the history at the undo point
    (re.compile(rf"^GHS pivot: {_NUM}, amount: {_NUM}, local: {_NUM} \[{_NUM} {_NUM}\]$"), "ghs", ("SP", "D", "B", "LP", "HP"), "normal"),
    (re.compile(rf"^GHS INV pivot: {_NUM}, amount: {_NUM}, local: {_NUM} \[{_NUM} {_NUM}\]$"), "ghs", ("SP", "D", "B", "LP", "HP"), "inverse"),
    (re.compile(rf"^GHS ASINH pivot: {_NUM}, amount: {_NUM} \[{_NUM} {_NUM}\]$"), "ghs", ("SP", "D", "LP", "HP"), "asinh"),
    (re.compile(rf"^GHS ASINH INV pivot: {_NUM}, amount: {_NUM} \[{_NUM} {_NUM}\]$"), "ghs", ("SP", "D", "LP", "HP"), "invasinh"),
    (re.compile(rf"^GHS LINEAR BP: {_NUM}$"), "bp", ("bp",), ""),
    # the program's commands
    (re.compile(rf"^GHS \(pivot: {_NUM}, amount: {_NUM}, local: {_NUM} \[{_NUM}, {_NUM}\]\)$"), "ghs", ("SP", "D", "B", "LP", "HP"), "normal"),
    (re.compile(rf"^Inverse GHS \(pivot: {_NUM}, amount: {_NUM}, local: {_NUM} \[{_NUM}, {_NUM}\]\)$"), "ghs", ("SP", "D", "B", "LP", "HP"), "inverse"),
    (re.compile(rf"^GHS asinh \(pivot: {_NUM}, amount: {_NUM} \[{_NUM}, {_NUM}\]\)$"), "ghs", ("SP", "D", "LP", "HP"), "asinh"),
    (re.compile(rf"^GHS inverse asinh \(pivot: {_NUM}, amount: {_NUM} \[{_NUM}, {_NUM}\]\)$"), "ghs", ("SP", "D", "LP", "HP"), "invasinh"),
    (re.compile(rf"^GHS BP shift \(new BP: {_NUM}\)$"), "bp", ("bp",), ""),
    (re.compile(rf"^Asinh stretch \(amount: {_NUM}, offset: {_NUM}, human: (yes|no)\)$"), "asinh", ("beta", "offset"), "cmd"),
    (re.compile(rf"^Asinh Transformation: \(stretch= *{_NUM}, bp= *{_NUM}\)$"), "asinh", ("beta", "offset"), "gui"),
)
_UNLINKED = re.compile(rf"^Unlinked MTF stretch \((.*)\)$")
_CHAN = re.compile(rf"Ch\s*(\d):\s*lo={_NUM},\s*mid={_NUM},\s*hi={_NUM}")


def _parse_line(line: str, channels: int):
    """([Op], None) for a stretch line, (None, refusal) when it cannot be undone exactly."""
    line = line.strip()
    um = _UNLINKED.match(line)
    if um and "Ch " in um.group(1):
        ops = []
        for ch, lo, mid, hi in _CHAN.findall(um.group(1)):
            (lo_v, lo_s), (m_v, m_s), (h_v, h_s) = _n(lo), _n(mid), _n(hi)
            ops.append(Op("mtf", line, (("lo", lo_v, lo_s), ("mid", m_v, m_s), ("hi", h_v, h_s)), chan=int(ch)))
        return (ops, None) if ops else (None, ("Unknown stretch line: {line}", {"line": line}))
    if " SAT" in line:
        return None, ("A saturation stretch cannot be undone from its header: {line}", {"line": line})
    if line.startswith("Autostretch") or line.startswith("AutoGHS"):
        return None, ("An automatic stretch keeps its recipe in the header, not the curve it drew: {line}",
                      {"line": line})
    for rx, kind, names, flag in _TABLE:
        m = rx.match(line)
        if not m:
            continue
        g = m.groups()
        flag_v = flag
        if flag == 1:                                         # "(Inverse midtones|Midtones) transfer"
            flag_v = "inverse" if g[0].startswith("Inverse") else ""
            g = g[1:]
        if kind == "asinh" and flag == "cmd":
            flag_v = "human" if g[2] == "yes" else "even"
            g = g[:2]
        if kind in ("ghs",) or (kind == "asinh" and flag == "gui"):
            if channels != 1:
                return None, ("A stretch of a colour image does not write its colour model in the header: {line}",
                              {"line": line})
        vals = [_n(x) for x in g]
        params = tuple((n, v, s) for n, (v, s) in zip(names, vals))
        if kind == "mtf" and flag_v == "inverse":
            kind = "mtfinv"
        if kind == "ghs" and "B" not in names:
            params += (("B", 0.0, 0.0),)
        return [Op(kind, line, params, flag=flag_v)], None
    return None, ("Unknown stretch line: {line}", {"line": line})


def plan_of(lines, channels: int) -> Plan:
    """The plan a header's history lines make: the first stretch line and every line after it (before it nothing
    changes the levels the way a stretch does)."""
    lines = [str(x).strip() for x in lines if str(x).strip()]
    first = next((i for i, x in enumerate(lines) if _is_stretch_line(x)), None)
    if first is None:
        return Plan(False, (), None)
    ops: list[Op] = []
    for line in lines[first:]:
        if SPATIAL.match(line):
            continue
        if _is_stretch_line(line):
            got, why = _parse_line(line, channels)
        elif "linear bp" in line.lower() or "bp shift" in line.lower():
            got, why = _parse_line(line, channels)
        else:
            got, why = None, ("A step after the stretch cannot be undone: {line}", {"line": line})
        if why is not None:
            return Plan(True, (), why, lines[first])
        ops += got
    return Plan(True, tuple(ops), None, lines[first])


def join_cards(cards) -> tuple:
    """The history lines of a header's HISTORY cards: a line longer than a card (72 characters) goes on the next cards,
    joined here while its parentheses are open."""
    out, cur = [], ""
    for c in cards:
        cur += str(c)
        if cur.count("(") <= cur.count(")"):
            out.append(cur)
            cur = ""
    if cur:
        out.append(cur)
    return tuple(out)


@functools.lru_cache(maxsize=64)
def _header(path: str, _mtime: int, _size: int):
    from astropy.io import fits
    try:
        with fits.open(path, memmap=True, lazy_load_hdus=True) as h:
            hd = next((x.header for x in h if int(x.header.get("NAXIS", 0)) >= 2), None)
            if hd is None:
                return (), 1
            lines = join_cards([str(x) for x in hd.get("HISTORY", [])])
            n3 = int(hd.get("NAXIS3", 1)) if int(hd.get("NAXIS", 2)) >= 3 else 1
            return lines, (3 if n3 == 3 else 1)
    except Exception:  # noqa: BLE001 - a file that cannot be read is said by the run
        return (), 1


def read_history(path: str | Path) -> tuple[tuple, int]:
    """((history lines), channels) of a FITS file's first image, from its header only (cached per file state)."""
    p = Path(str(path or ""))
    if p.suffix.lower() not in (".fits", ".fit", ".fts"):
        return (), 1
    try:
        st = p.stat()
    except OSError:
        return (), 1
    return _header(str(p), st.st_mtime_ns, st.st_size)


def plan_of_file(path: str | Path) -> Plan:
    """The plan of a FITS file's header (header only: cheap; the key of the Image block reads it)."""
    lines, ch = read_history(path)
    return plan_of(lines, ch)


def header_exact(path: str | Path) -> bool:
    """Whether the header lists stretches every one of which can be inverted (before the sky-level rounding check)."""
    pl = plan_of_file(path)
    return pl.stretch and pl.refusal is None


# ------------------------------------------------------------------------------------------------ the undo

def _undo_op(op: Op, a: np.ndarray, mono: bool, over: dict | None = None) -> np.ndarray:
    """One stretch undone on a (C, H, W) array of values in 0-1."""
    q = op.p(over)
    if op.chan is not None:
        out = a.copy()
        idx = range(a.shape[0]) if mono else [op.chan]
        for i in idx:
            out[i] = mtf_inverse(a[i], q["mid"], q["lo"], q["hi"])
        return out
    if op.kind == "mtf":
        return mtf_inverse(a, q["mid"], q["lo"], q["hi"])
    if op.kind == "mtfinv":                                   # the line is an inverse: its undo is the stretch
        return mtf(a, q["mid"], q["lo"], q["hi"])
    if op.kind == "bp":
        return a * (1.0 - q["bp"]) + q["bp"]
    if op.kind == "asinh":
        return undo_asinh(a, q["beta"], q["offset"], op.flag != "even", mono)
    if op.kind == "ghs":
        opposite = {"normal": "inverse", "inverse": "normal", "asinh": "invasinh", "invasinh": "asinh"}[op.flag]
        a = np.clip(a, 0.0, 1.0) if opposite in ("inverse", "invasinh") else a
        return np.stack([ght(ch, q["B"], q["D"], q["LP"], q["SP"], q["HP"], opposite) for ch in a])
    raise ValueError(op.kind)


def undo(a: np.ndarray, ops, mono: bool = False, over: dict[int, dict] | None = None) -> np.ndarray:
    """The image (C, H, W, values in 0-1) before the stretches ``ops`` (in the order they were applied): each one
    undone, last first. ``over``: {op index: {param: value}} (the rounding check's perturbations)."""
    a = np.asarray(a, np.float64)
    for i in range(len(ops) - 1, -1, -1):
        a = _undo_op(ops[i], a, mono, (over or {}).get(i))
    return a


def clipped(a: np.ndarray) -> int:
    """How many pixels of a stretched image sit at the stretch's limits (0 or 1): what the stretch clipped, which an
    undo cannot bring back."""
    return int(np.count_nonzero(np.any((a <= 0.0) | (a >= 1.0), axis=0)))


def _sky(x: np.ndarray):
    """(median, 1.4826 MAD) per channel of the finite values."""
    med, sig = [], []
    for ch in x:
        v = ch[np.isfinite(ch)]
        m = float(np.median(v)) if v.size else 0.0
        med.append(m)
        sig.append(1.4826 * float(np.median(np.abs(v - m))) if v.size else 0.0)
    return np.array(med), np.array(sig)


def rounding_refusal(a: np.ndarray, ops, mono: bool = False) -> tuple | None:
    """Whether a parameter written with few digits could move the linear sky level by more than its noise: each
    parameter is moved by half its last printed digit, the stretches undone again, and the sky level (median) of the
    result compared with its noise (1.4826 MAD). (template, fields) of the first that does, None when none does."""
    base = undo(a, ops, mono)
    med0, sig0 = _sky(base)
    for i, op in enumerate(ops):
        for name, val, step in op.params:
            if step <= 0:
                continue
            for sign in (-1.0, 1.0):
                with np.errstate(all="ignore"):
                    other = undo(a, ops, mono, {i: {name: val + sign * step / 2.0}})
                med1, _s = _sky(other)
                if np.any((sig0 > 0) & (np.abs(med1 - med0) > sig0)):
                    return ("{param} of \"{line}\" is written with too few digits: the sky level of the linear "
                            "image would move by more than its noise", {"param": name, "line": op.line})
    return None


# ------------------------------------------------------------------------------------------------ the estimate

def estimate_channel(y: np.ndarray) -> tuple[np.ndarray, float, float]:
    """A stretched channel (0-1) brought back to linear by an estimated mid-tones transfer: (linear, m, u).

    The reference program's automatic stretch (``find_linked_midtones_balance``, filters/mtf.c) puts the black point 2.8
    noise units under the sky median and the sky median at a target; so the stretched sky median s is
    MTF(2.8 u; m) and the stretched noise s_n is u MTF'(2.8 u; m), u the noise of the linear image in units of its own
    range, MTF'(x) = m (1 - m) / ((2m - 1) x - m)^2. Measured s and s_n give u and m (a bisection on u: m follows from
    the first equation); the channel is then ``mtf_inverse`` with that m on 0-1, linear to an offset and a scale,
    which the stages after it do not depend on (they measure the sky and the noise themselves). Raises ValueError
    when the sky and its noise cannot be measured."""
    v = y[np.isfinite(y)]
    s = float(np.median(v)) if v.size else 0.0
    sn = 1.4826 * float(np.median(np.abs(v - s))) if v.size else 0.0
    if not (0.0 < s < 1.0) or sn <= 0.0:
        raise ValueError("the sky level and its noise cannot be measured")

    def m_of(u):
        x = 2.8 * u
        return x * (s - 1.0) / (2.0 * s * x - s - x)

    def g(u):
        x, m = 2.8 * u, m_of(u)
        d = (2.0 * m - 1.0) * x - m
        return sn - u * m * (1.0 - m) / (d * d)

    lo, hi = 1e-9, 0.3
    glo, ghi = g(lo), g(hi)
    if not (np.isfinite(glo) and np.isfinite(ghi)) or glo * ghi > 0:
        raise ValueError("the stretch does not look like a mid-tones transfer")
    for _ in range(200):
        mid = math.sqrt(lo * hi)
        gm = g(mid)
        if glo * gm <= 0:
            hi, ghi = mid, gm
        else:
            lo, glo = mid, gm
    u = math.sqrt(lo * hi)
    m = m_of(u)
    if not (0.0 < m < 1.0):
        raise ValueError("the stretch does not look like a mid-tones transfer")
    return mtf_inverse(np.clip(y, 0.0, 1.0), m, 0.0, 1.0), m, u


def estimate(a: np.ndarray, mono: bool = False) -> tuple[np.ndarray, list]:
    """(C, H, W) 0-1 stretched -> (linear estimate, [(m, u) per channel])."""
    out, info = np.empty(a.shape, np.float64), []
    n = 1 if mono else a.shape[0]
    for i in range(n):
        out[i], m, u = estimate_channel(a[i])
        info.append((m, u))
    if mono:
        out[1:] = out[0]
    return out, info


# ------------------------------------------------------------------------------------------------ the analysis

@dataclass(frozen=True)
class Analysis:
    """What a file is: ``stretched`` and why (``why``: (template, fields) of the sign), the sky level (0-1, None when
    not measured), ``exact`` (its header gives an exact undo: done automatically), ``refusal`` (why it does not,
    when the header lists stretches), ``ops``, ``mono`` and ``scale`` (the range of the file's values)."""
    stretched: bool = False
    why: tuple | None = None
    sky: float | None = None
    exact: bool = False
    refusal: tuple | None = None
    ops: tuple = ()
    mono: bool = False
    scale: float = 1.0


def scale_of(a: np.ndarray) -> float:
    """The range an image's values are in (as ``stages/image.stretched`` reads it)."""
    top = float(a.max()) if a.size else 1.0
    return 65535.0 if top > 255.5 else (255.0 if top > 1.5 else 1.0)


def sky_level(a: np.ndarray) -> float:
    """The sky of an image: the median of its luminance over its white (p99.99), 0-1."""
    s = a[:, ::4, ::4] if a.ndim == 3 else a
    lum = np.mean(s, axis=0) if s.ndim == 3 else s
    lum = lum[np.isfinite(lum)]
    if not lum.size:
        return 0.0
    white = float(np.percentile(lum, 99.99))
    return float(np.median(lum) / white) if white > 0 else 0.0


@functools.lru_cache(maxsize=32)
def _analyse(path: str, _mtime: int, _size: int) -> Analysis:
    from .image import read_image, stretched_by_nature
    if stretched_by_nature(path):
        return Analysis(True, ("This picture is on 8 bits (or a PNG, a JPEG): it was stretched", {}))
    try:
        a = np.nan_to_num(read_image(path), nan=0.0, posinf=0.0, neginf=0.0)
    except Exception:  # noqa: BLE001 - its problem is said elsewhere (image_problem)
        return Analysis()
    lines, ch = read_history(path)
    mono = ch == 1
    scale = scale_of(a)
    pl = plan_of(lines, ch)
    sample = (a[:, ::4, ::4] / scale).astype(np.float64)
    if pl.stretch:
        why = ("Its header lists a stretch: {line}", {"line": pl.first})
        if pl.refusal is None:
            bad = rounding_refusal(sample, pl.ops, mono)
            if bad is None:
                return Analysis(True, why, sky_level(a), True, None, pl.ops, mono, scale)
            return Analysis(True, why, sky_level(a), False, bad, pl.ops, mono, scale)
        return Analysis(True, why, sky_level(a), False, pl.refusal, (), mono, scale)
    sky = sky_level(a)
    if sky > SKY_LIMIT:
        return Analysis(True, ("Its sky is at {pct} % of white: a linear image has it far lower",
                               {"pct": f"{100 * sky:.0f}"}), sky, False, None, (), mono, scale)
    return Analysis(False, None, sky, False, None, (), mono, scale)


def analyse(path: str | Path) -> Analysis:
    """The analysis of an image file, read once per file state."""
    p = Path(str(path or ""))
    try:
        st = p.stat()
    except OSError:
        return Analysis()
    return _analyse(str(p), st.st_mtime_ns, st.st_size)
