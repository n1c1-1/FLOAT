"""Simple rendering: the output of every workflow (the Output block), and of the Raw stack exit.

Works from the outputs of the ``render_simple`` stage: the linear image in display RGB with its sky removed (lin.npy,
the denoised stack when there is one), the same from the raw stack (lin_raw.npy, with denoise), the covered pixels
(valid.npy) and the statistics of the whole image (prep_simple.json). ``render(settings, scale, region)`` returns the
``Layers`` (the image, its scale, notes): a usual automatic stretch:

* the raw stack mixed back into the denoised one, or the faint areas smoothed (noise smoothing);
* a midtones transfer function from black = median - 2.8 MAD to white = the 99.995th percentile, the median sky at
  ``sky`` (linked channels: on the luminance; unlinked: each channel);
* the faint tones raised or lowered (``faint``): a generalised hyperbolic stretch (Payne and Cranfield) of the range
  above the sky, its black point and its symmetry point at the sky (the GHS users' placing: the symmetry point at the
  background's peak), on the luminance with the colour ratios kept: the tones just above the sky, the faint
  extensions, rise most, the sky stays where Sky puts it, the bright ones rise least;
* an S-curve for the contrast, then the saturation around the luminance.

In place of the midtones transfer function, a layer may take the colour-preserving arcsinh stretch (``stretch``
"arcsinh": Lupton et al. 2004, ``SimpleRenderEngine._arcsinh``), on the same linear data and with the same black
point: a logarithm for the bright tones, where the transfer function flattens an object hundreds of times brighter
than the sky near white. The faint tones, the contrast and the saturation follow it as they follow the automatic one.

Every statistic comes from prep_simple.json, measured once on the whole image at full resolution: a reduced preview,
the full image and a region of it share one curve, so a region is the crop of the full rendering.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from pydantic import field_validator

from ..core import devices
from ..core.params import ADVANCED, P, StageParams, check_choice
from . import ops
from .asinh import asinh_factor, asinh_level, strength_for

#: margin rendered around a region, in units of the smoothing radius (plus a few pixels)
REGION_PAD_SIGMAS = 4
#: the base stretches of a layer: the automatic screen transfer, the colour-preserving arcsinh
STRETCHES = ("auto", "arcsinh")


@dataclass
class Layers:
    """A rendering: the image (3, h, w) in display sRGB, 0..1; the scale it is at (1/f); notes on how it was made."""
    final: torch.Tensor
    scale: float
    info: dict


class SimpleSettings(StageParams):
    """The settings of a simple rendering. The defaults are the suggested rendering (Post's sliders at 2)."""

    sky: float = P(0.25, label="Sky brightness", ge=0.02, le=0.6, step=0.01,
                   help="Display level of the median sky after the stretch (0 = black, 1 = white). 0.25 by default: "
                        "the target of the published screen transfer function (the median brought to 0.25).",
                   tip="0.15-0.25 looks natural on screen.")
    contrast: float = P(0.0, label="Contrast", ge=-1.0, le=1.0, step=0.01,
                        help="S-curve after the stretch: above 0 the mid-tones are spread, below they are flattened.")
    saturation: float = P(1.2, label="Saturation", ge=0.0, le=3.0, step=0.05,
                          help="Colour saturation around the luminance, after the stretch. 1 = the colours of the "
                               "stretch.")
    noise_mix: float = P(1.0, label="Denoised share", level=ADVANCED, ge=0.0, le=1.0, step=0.05,
                         help="With Denoise: share of the denoised stack, the rest from the raw stack (finer grain, "
                              "more noise). 1 = the denoised stack alone.")
    smooth_px: float = P(0.0, label="Faint area smoothing", level=ADVANCED, ge=0.0, le=8.0, step=0.1, unit="px",
                         help="Smoothing of the faint areas (Gaussian radius); the bright areas keep full "
                              "resolution.")
    keep_colours: float = P(0.5, label="Keep the colours", ge=0.0, le=1.0, step=0.05,
                            help="How much of the stretch keeps the colours: 0 = each channel through the same curve "
                                 "(the faint colours turn grey, the bright cores white), 1 = the luminance stretched "
                                 "and the three channels scaled with it (Lupton et al. 2004: the colours of the linear "
                                 "image kept). 0.5 by default: the nebulae keep their colour and the bright cores "
                                 "their structure, without the strong reds of 1.")    # bench of 5 November
    faint: float = P(0.0, label="Faint tones", ge=-1.0, le=3.0, step=0.05,
                     help="Raises (above 0) or lowers (below 0) the faint tones just above the sky, without moving "
                          "the sky: a generalised hyperbolic stretch (D. Payne and M. Cranfield) of the range above the "
                          "sky, its symmetry point at the sky. 0 = the automatic stretch as it is, "
                          "by default; above 0 the faint extensions and outer arms come up.",
                     tip="The faint extensions of a nebula, the outer arms of a galaxy. Sky sets the black, this the "
                         "faint tones, as Blacks and Shadows are two controls in photo editors.")
    halpha: float = P(0.0, label="H-alpha", ge=-1.0, le=3.0, step=0.05,
                      help="The H-alpha emission, measured as the red above the continuum of the stars, added to the "
                           "red before the stretch (above 0) or taken out of it (below 0). 0 = the image as it is.")
    stretch: str = P("auto", label="Stretch", choices=list(STRETCHES),
                     help="The curve that brings the linear image to the screen. 'auto': the automatic screen "
                          "transfer (black just under the sky, the sky at Sky, white at the brightest pixels): fine "
                          "for most images, but an object hundreds of times brighter than the sky ends flat near "
                          "white. 'arcsinh': the colour-preserving arcsinh stretch (Lupton et al. 2004), a logarithm "
                          "for the bright tones: a bright core keeps its structure and its colours.")
    stretch_strength: float = P(1.0, label="Strength", ge=0.0, le=2.0, step=0.025,
                                help="Arcsinh: the level its sky lands at, as a multiple of the Sky setting (1: the sky "
                                     "at Sky within the range of the luminance). Below 1 the sky darkens and the bright "
                                     "areas spread apart; above, the faint areas rise and the bright ones flatten.")

    @field_validator("stretch")
    @classmethod
    def _stretch_known(cls, v: str) -> str:
        return check_choice("stretch", v, STRETCHES)


#: the smoothing of the H-alpha emission map (px at full resolution): its noise, not its structures
HALPHA_SIGMA = 1.5
#: Faint tones: the local intensity b of its stretch (1: a broad lift, strongest just above the sky) and its ln(D + 1)
#: per unit of the setting
FAINT_B, FAINT_STRETCH = 1.0, 1.0


def _interp(x: torch.Tensor, xs: torch.Tensor, ys: torch.Tensor) -> torch.Tensor:
    """Piecewise-linear interpolation of x on the increasing table (xs, ys)."""
    i = torch.searchsorted(xs, x.contiguous()).clamp(1, xs.numel() - 1)
    x0, x1, y0, y1 = xs[i - 1], xs[i], ys[i - 1], ys[i]
    return y0 + (y1 - y0) * ((x - x0) / (x1 - x0).clamp(min=1e-12))


def faint_tones(y: torch.Tensor, sky: float, k: float) -> torch.Tensor:
    """The display image (3, h, w) with its tones above ``sky`` through a generalised hyperbolic stretch (render/ghs,
    black and symmetry points at the sky, local intensity FAINT_B, ln(D + 1) = |k| FAINT_STRETCH; k < 0 its inverse),
    on the luminance with the colour ratios kept; the tones at or below the sky unchanged."""
    if k == 0:
        return y
    out = y * faint_ratio(y, sky, k)[None]
    return (out / out.amax(0, keepdim=True).clamp(min=1.0)).clamp(0, 1)


def faint_ratio(y: torch.Tensor, sky: float, k: float) -> torch.Tensor:
    """(h, w): the factor ``faint_tones`` scales each pixel's three channels by (the luminance raised or lowered
    over its own), before a pixel over 1 is brought back under it."""
    from .ghs import ghs_unit
    span = max(1.0 - sky, 1e-6)
    spz = 0.0
    D = math.expm1(abs(k) * FAINT_STRETCH)
    L = ops.luminance(y)
    z = ((L - sky) / span).clamp(0, 1)
    if k > 0:
        z2 = ghs_unit(z, D, FAINT_B, spz, 0.0, 1.0)
    else:                                       # the inverse stretch, read back from its table
        xs = torch.linspace(0, 1, 4097, dtype=torch.float64)
        ys = ghs_unit(xs.clone(), D, FAINT_B, spz, 0.0, 1.0)
        z2 = _interp(z.double(), ys.to(z.device), xs.to(z.device)).to(z.dtype)
    L2 = torch.where(L > sky, sky + span * z2, L)
    return L2 / L.clamp(min=1e-6)


def continuum_ratio(stars: np.ndarray) -> float:
    """The red-to-green ratio of the continuum, from the stars of the image (the continuum subtraction of narrow-band
    work, James et al. 2004, A&A 414, 23, with the stars as the continuum's colour): the sums of red and of green over
    the brightest star pixels of ``stars`` (3, H, W): an image of stars alone, or a high-pass of an image where stars
    outshine the nebula at their scale."""
    g = stars[1]
    thr = np.quantile(g[::4, ::4], 0.999)
    m = g > thr
    if m.sum() < 50 or float(g[m].sum()) <= 0:
        return 1.0
    return float(stars[0][m].sum() / g[m].sum())


def emission(x: torch.Tensor, alpha: float, k: float, f: int) -> torch.Tensor:
    """The linear image with its H-alpha emission (red above alpha times the green, smoothed, never negative) times k
    added to its red (k < 0: taken out, -1 leaves the continuum's red)."""
    if k == 0:
        return x
    e = ops.gaussian_blur((x[0] - alpha * x[1]).clamp(min=0), HALPHA_SIGMA / f)
    out = x.clone()
    out[0] = x[0] + k * e
    return out


def mtf(x, m: float):
    """Midtones transfer function (0 -> 0, m -> 0.5, 1 -> 1)."""
    return (m - 1) * x / ((2 * m - 1) * x - m)


def mtf_midtone(xm: float, target: float) -> float:
    """The midtone parameter that brings xm to target."""
    return xm * (1 - target) / (xm - 2 * target * xm + target)


def _anchors(st: dict) -> tuple[float, float, float]:
    """The anchors of a stretch from the statistics of the luminance or of a channel: the black point (median - 2.8
    MAD), the white point (the 99.995th percentile) and where the median sky sits between them (0..1)."""
    c0 = st["median"] - 2.8 * st["mad"]
    hi = max(st["p99995"], c0 + 1e-6)
    return c0, hi, min(max((st["median"] - c0) / (hi - c0), 1e-6), 1 - 1e-6)


#: the highest level the Strength may ask for the sky, within the range of the luminance
ARCSINH_SKY_MAX = 0.95


def arcsinh_strength(st: dict, s) -> float:
    """The stretch factor beta of a layer's Arcsinh stretch: the one that brings the median sky to the Sky setting
    times the Strength setting, as a share of the range from black to the white luminance (Strength 1: where the
    automatic stretch puts it: the two stretches start from the same sky). The Strength is so linear in the level the
    sky lands at: each of its steps moves the sky by the same amount. The rule is the statistics' alone: nothing is
    fitted on an image. 0 (no stretch) when the sky already sits at or above that level."""
    target = float(s.sky) * float(getattr(s, "stretch_strength", 1.0))
    return strength_for(_anchors(st["lum"])[2], min(max(target, 0.0), ARCSINH_SKY_MAX))


def arcsinh_curve(x: torch.Tensor, st: dict, s, linked: bool) -> tuple[torch.Tensor, float]:
    """The colour-preserving arcsinh stretch of a linear image (3, h, w) (Lupton et al. 2004, PASP 116, 133; the
    reference program's asinh transformation, render/asinh.py), before its scale, and the level its sky lands at. With
    the anchors of the automatic stretch: the black point at median - 2.8 MAD (linked: the luminance's; else each
    channel's own) and the data divided so that the luminance's 99.995th percentile is 1; then pixel * asinh(L beta) /
    (L asinh(beta)), L the luminance: the same factor on the three channels. beta is ``arcsinh_strength``."""
    c0, hi, xm = _anchors(st["lum"])
    if linked:
        xl = (x - c0).clamp(min=0)
    else:
        xl = torch.stack([(x[c] - _anchors(st["channels"][c])[0]).clamp(min=0) for c in range(3)])
    beta = arcsinh_strength(st, s)
    w = hi - c0
    return xl * (asinh_factor(ops.luminance(xl) / w, beta) / w)[None], asinh_level(xm, beta)


def arcsinh_headroom(st: dict, saturation: float, linked: bool = True) -> float:
    """By how much the Arcsinh stretch's picture is scaled down, as a whole, so that its brightest channel fits. The
    colours being kept, a channel stands above the luminance by its ratio to it in the stretched image too: with the
    luminance's white point at 1 a blue or red core passes 1. The white point of this stretch is therefore the largest
    of the three channels' own: each channel's 99.995th percentile above the black point, the statistic of the
    automatic stretch's white point (its own white point for the luminance, and for each channel when they are
    stretched apart), read from the statistics of the image the stretch is applied to; its ratio to the luminance's
    is carried through the Saturation setting (a channel c becomes L + saturation (c - L)). Never under 1: a grey
    image keeps the luminance's white point.

    The first rule measured the 99.995th percentile of the stretched picture's largest channel: faint, strongly
    coloured pixels set it (the remains of red stars at a fifth of the range, the sky's noise in one channel), which
    the stretch factor raises most: the nebula stayed in the lower half of the range and the scale grew with the
    Strength. A pixel still over 1 is divided by its largest channel, its colour kept (``SimpleRenderEngine._arcsinh``)."""
    c0, hi, _xm = _anchors(st["lum"])
    tops = []
    for ch in st.get("channels") or []:
        b0 = c0 if linked else _anchors(ch)[0]
        tops.append(ch["p99995"] - b0)
    ratio = max(max(tops, default=0.0) / (hi - c0), 1.0)
    return max(1.0 + max(float(saturation), 0.0) * (ratio - 1.0), 1.0)


def _smoothstep(x: torch.Tensor, a: float, b: float) -> torch.Tensor:
    t = ((x - a) / max(b - a, 1e-12)).clamp(0, 1)
    return t * t * (3 - 2 * t)


def _mix_stats(a: dict, b: dict | None, w: float) -> dict:
    """Statistics of w * a + (1 - w) * b, interpolated (the medians and spreads of the two stacks differ little)."""
    if b is None or w >= 1:
        return a
    return {k: w * a[k] + (1 - w) * b[k] for k in ("median", "mad", "p99995")}


class SimpleRenderEngine:
    """Renders a render_simple folder. Same interface as ``RenderEngine``: ``render(s, scale, region, fit)``,
    ``crop``, ``SC``, ``H``, ``W``, ``dev``; ``fit`` is accepted and ignored (the statistics are in the folder)."""

    def __init__(self, prep_dir: str | Path, device: str = "cuda:0"):
        self.dir = Path(prep_dir)
        self.dev = torch.device(devices.usable(device))
        self.prep = json.loads((self.dir / "prep_simple.json").read_text(encoding="utf-8"))
        self.M = np.array(self.prep["M"], np.float64)
        self.SC = float(self.prep.get("drizzle_scale", 1.0))
        self.crop = self.prep["crop"]                                    # x0, x1, y0, y1
        self._host = {"lin": torch.from_numpy(np.load(self.dir / "lin.npy"))}
        if (self.dir / "lin_raw.npy").exists():
            self._host["raw"] = torch.from_numpy(np.load(self.dir / "lin_raw.npy"))
        self.H, self.W = self._host["lin"].shape[-2:]
        self.valid_host = torch.from_numpy(np.load(self.dir / "valid.npy"))
        self.has_raw = "raw" in self._host
        self._proxy: dict = {}
        self.last_fit = None

    @classmethod
    def from_array(cls, rgb, device: str = "cpu", linked: bool = True, sample: int = 2) -> SimpleRenderEngine:
        """An engine on a linear image in memory (a stage result Post shows as a layer): (H, W), (H, W, 3) or (3, H, W),
        any dtype. The covered pixels are the finite ones that are not 0 in every channel; the statistics of the
        stretch are measured on every ``sample``-th pixel of them (the median and the spreads move by a fraction of a
        level)."""
        a = np.asarray(rgb)
        if a.ndim == 2:
            a = a[None]
        elif a.ndim == 3 and a.shape[0] not in (1, 3) and a.shape[2] in (1, 3):
            a = np.moveaxis(a, -1, 0)
        if a.ndim != 3 or a.shape[0] not in (1, 3):
            raise ValueError(f"expected an image (H, W), (H, W, 3) or (3, H, W), got {a.shape}")
        a = np.ascontiguousarray(a, np.float32)
        if a.shape[0] == 1:
            a = np.repeat(a, 3, 0)
        finite = np.isfinite(a).all(0)
        if not finite.all():
            a = np.where(finite[None], a, 0).astype(np.float32)
        valid = finite & (a != 0).any(0)
        if not valid.any():
            valid = finite
        eng = cls.__new__(cls)
        eng.dir = None
        eng.dev = torch.device(devices.usable(device))
        H, W = a.shape[1:]
        sl = (slice(None), slice(None, None, sample), slice(None, None, sample))
        v = valid[sl[1:]]
        eng.prep = {"M": np.eye(3).tolist(), "crop": [0, W, 0, H], "drizzle_scale": 1.0, "linked": bool(linked),
                    "stats": image_stats(a[sl], v if v.any() else np.ones_like(v)), "exit": "image"}
        eng.M = np.eye(3)
        eng.SC = 1.0
        eng.crop = eng.prep["crop"]
        eng._host = {"lin": torch.from_numpy(a)}
        eng.H, eng.W = H, W
        eng.valid_host = torch.from_numpy(valid)
        eng.has_raw = False
        eng._proxy = {}
        eng.last_fit = None
        return eng

    def continuum(self) -> float:
        """The continuum's red-to-green ratio of this image (``continuum_ratio``), measured once: on its high-pass (the
        image minus a 4 px blur), where the stars outshine the nebula."""
        if getattr(self, "_alpha", None) is None:
            import cv2
            a = self._host["lin"].numpy().astype(np.float32)
            hp = np.stack([a[c] - cv2.GaussianBlur(a[c], (0, 0), 4) for c in range(3)])
            self._alpha = continuum_ratio(np.maximum(hp, 0))
        return self._alpha

    @property
    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self._host.values()) + self.valid_host.numel()

    def fit_of(self, s=None, scale: float = 0.25) -> dict:
        return {"key": "", "f": max(1, int(round(1 / scale))), "stats": {}}

    # ------------------------------------------------------------------ helpers
    def _layer(self, name: str, f: int, box) -> torch.Tensor:
        """A layer on the device in float32, cut to the box (x0, x1, y0, y1) and reduced by f (box aligned on f)."""
        key = (name, f, box)
        if key in self._proxy:
            return self._proxy[key]
        x0, x1, y0, y1 = box
        t = (self.valid_host if name == "valid" else self._host[name])[..., y0:y1, x0:x1].to(self.dev)
        if name == "valid":
            t = t if f == 1 else F.avg_pool2d(t[None, None].float(), f)[0, 0] > 0.99
        else:
            t = t.float() if f == 1 else F.avg_pool2d(t.float()[None], f)[0]
        if f > 1 or box != (0, self.W, 0, self.H):      # the full resolution of the whole grid is not kept on the device
            if len(self._proxy) > 12:
                self._proxy.clear()
            self._proxy[key] = t
        return t

    # ------------------------------------------------------------------ public
    @torch.no_grad()
    def render(self, s: SimpleSettings, scale: float = 1.0, region=None, fit: dict | None = None, *,
               draft: bool = False, stop=None) -> Layers:
        """``draft`` and ``stop`` are accepted for the engine's interface: a simple rendering is quick and exact."""
        from ..core.determinism import deterministic
        with deterministic():
            return self._render(s, scale, region)

    def _geometry(self, scale: float, region):
        """(f, the region (x0, x1, y0, y1) on multiples of f, the box rendered around it with a margin for the
        smoothings)."""
        f = max(1, int(round(1 / scale)))
        Hf, Wf = self.H // f * f, self.W // f * f
        if region is None:
            x0, x1, y0, y1 = 0, Wf, 0, Hf
            box = (0, Wf, 0, Hf)
        else:
            x0, x1, y0, y1 = (int(round(float(v))) for v in region)
            x0, x1 = sorted((min(max(x0, 0), Wf), min(max(x1, 0), Wf)))
            y0, y1 = sorted((min(max(y0, 0), Hf), min(max(y1, 0), Hf)))
            x0, y0 = x0 // f * f, y0 // f * f
            x1, y1 = min(-(-x1 // f) * f, Wf), min(-(-y1 // f) * f, Hf)
            if x1 <= x0 or y1 <= y0:
                raise ValueError(f"empty region {list(region)} on a {self.W} x {self.H} grid")
            pad = -(-int(REGION_PAD_SIGMAS * 8.0 + 8) // f) * f          # the widest smoothing, 8 px
            box = (max(0, x0 - pad), min(Wf, x1 + pad), max(0, y0 - pad), min(Hf, y1 + pad))
        if box == (0, Wf, 0, Hf) and f == 1:
            box = (0, self.W, 0, self.H)
        return f, (x0, x1, y0, y1), box

    def _finish(self, y: torch.Tensor, f: int, reg, box, region, info: dict) -> Layers:
        """The rendering of the box, outside the covered pixels black, cut to the region."""
        x0, x1, y0, y1 = reg
        y[:, ~self._layer("valid", f, box)] = 0
        bx0, _, by0, _ = box
        y = y[:, (y0 - by0) // f:(y1 - by0) // f, (x0 - bx0) // f:(x1 - bx0) // f]
        if region is not None:
            info["region"] = [x0, x1, y0, y1]
        return Layers(y, 1.0 / f, info)

    @staticmethod
    def tone(x: torch.Tensor, st: dict, s: SimpleSettings, linked: bool) -> torch.Tensor:
        """The base stretch of a linear image, the automatic one (linked: on the luminance's statistics; else each
        channel on its own) or the Arcsinh, then the faint tones, the contrast and the saturation."""
        sky = s.sky                                # where the sky sits once stretched (Faint tones turns around it)
        arcsinh = getattr(s, "stretch", "auto") == "arcsinh" and not st.get("stretched")
        if st.get("stretched"):                    # an image already stretched (issue #43): as it is
            y = x.clamp(0, 1)
        elif arcsinh:
            y, sky = SimpleRenderEngine._arcsinh(x, st, s, linked)
        elif linked:
            y = SimpleRenderEngine._stretch(x, st["lum"], s.sky)
        else:
            y = torch.stack([SimpleRenderEngine._stretch(x[c], st["channels"][c], s.sky) for c in range(3)])
        k = 0.0 if st.get("stretched") or arcsinh else float(getattr(s, "keep_colours", 0.0))
        if k > 0:
            # the luminance stretched and the channels scaled by its gain (Lupton et al. 2004, PASP 116, 133): the
            # ratios of the linear image kept; the black point first, as the stretch has it
            lum = st["lum"]
            c0 = lum["median"] - 2.8 * lum["mad"]
            xl = (x - c0).clamp(min=0)
            L = ops.luminance(xl)
            gain = SimpleRenderEngine._stretch(L + c0, lum, s.sky) / L.clamp(min=1e-6)
            yk = xl * gain[None]
            m = yk.amax(0, keepdim=True).clamp(min=1.0)              # a channel over 1: the colour kept, scaled down
            y = (1 - k) * y + k * (yk / m)
        if getattr(s, "faint", 0.0) and not st.get("stretched"):    # an image already stretched keeps its tones
            y = faint_tones(y, sky, float(s.faint))
        return SimpleRenderEngine.finish(y, s)

    @staticmethod
    def finish(y: torch.Tensor, s: SimpleSettings) -> torch.Tensor:
        """The end of ``tone`` on a stretched image (3, h, w): the contrast's S-curve, then the saturation around the
        luminance."""
        if s.contrast != 0:
            y = (y + s.contrast * (3 * y * y - 2 * y ** 3 - y)).clamp(0, 1)
        Ly = ops.luminance(y)[None]
        return (Ly + s.saturation * (y - Ly)).clamp(0, 1)

    @staticmethod
    def colour_factor(x: torch.Tensor, st: dict, s: SimpleSettings) -> tuple[torch.Tensor, float]:
        """(the factor (h, w), the black point): ``tone``'s stretch of a linear colour image (3, h, w) as one factor
        per pixel, the same on its three channels: the colour-preserving stretch (Lupton et al. 2004, PASP 116, 133),
        the luminance (the luminance's statistics ``st``) brought to the screen and the channels scaled with it, by
        the automatic curve (``tone`` with Colours at 1) or the Arcsinh, a pixel over 1 brought under it, then the
        Faint tones, also a factor on the luminance. ``x``'s stretch is ``factor * (x - black point)`` (clipped at 0),
        which ``finish`` ends: a factor is linear, so the channels of a mix of images can be stretched before the mix
        as well as after it (Post's combination layer, client/postfusion.py)."""
        lum = st["lum"]
        c0, hi, xm = _anchors(lum)
        xl = (x - c0).clamp(min=0)
        L = ops.luminance(xl)
        if getattr(s, "stretch", "auto") == "arcsinh":
            room = arcsinh_headroom(st, float(s.saturation), True)
            beta = arcsinh_strength(st, s)
            w = hi - c0
            g = asinh_factor(L / w, beta) / w / room
            sky = asinh_level(xm, beta) / room
        else:
            g = SimpleRenderEngine._stretch(L + c0, lum, s.sky) / L.clamp(min=1e-6)
            sky = s.sky
        g = g / (xl * g[None]).amax(0).clamp(min=1.0)
        if getattr(s, "faint", 0.0):
            y = xl * g[None]
            r = faint_ratio(y, sky, float(s.faint))
            g = g * r / (y * r[None]).amax(0).clamp(min=1.0)
        return g, float(c0)

    def _render(self, s: SimpleSettings, scale: float, region) -> Layers:
        f, reg, box = self._geometry(scale, region)
        st = self.prep["stats"]
        x = self._layer("lin", f, box)
        linked = bool(self.prep.get("linked", True))
        mix = float(s.noise_mix) if self.has_raw else 1.0
        if mix < 1:
            x = mix * x + (1 - mix) * self._layer("raw", f, box)
        if s.smooth_px > 0:
            # faint areas smoothed, bright ones kept (decided on the denoised luminance, from its 40th to 85th
            # percentile)
            w = _smoothstep(ops.luminance(self._layer("lin", f, box)), st["lum_p40"], st["lum_p85"])[None]
            x = w * x + (1 - w) * ops.gaussian_blur(x, s.smooth_px / f)
        raw = self.prep.get("stats_raw")
        mixed = {"lum": _mix_stats(st["lum"], raw and raw["lum"], mix),
                 "channels": [_mix_stats(st["channels"][c], raw and raw["channels"][c], mix) for c in range(3)],
                 "stretched": bool(st.get("stretched"))}
        if s.halpha:
            x = emission(x, self.continuum(), float(s.halpha), f)
        y = self.tone(x, mixed, s, linked)
        return self._finish(y, f, reg, box, region, {"stretch": getattr(s, "stretch", "auto"), "linked": linked,
                                                     "noise_mix": mix})

    @staticmethod
    def _arcsinh(x: torch.Tensor, st: dict, s: SimpleSettings, linked: bool) -> tuple[torch.Tensor, float]:
        """The colour-preserving arcsinh stretch of a linear image (``arcsinh_curve``), scaled as a whole so that its
        brightest channel's white point fits (``arcsinh_headroom``), and the level its sky lands at. A pixel still over
        1 is divided by its largest channel, its colour kept (Lupton et al. 2004)."""
        room = arcsinh_headroom(st, float(s.saturation), linked)
        y, sky = arcsinh_curve(x, st, s, linked)
        y = y / room
        return (y / y.amax(0, keepdim=True).clamp(min=1.0)).clamp(0, 1), sky / room

    @staticmethod
    def _stretch(x: torch.Tensor, st: dict, target: float) -> torch.Tensor:
        """Linked or one channel: black at median - 2.8 MAD, white at the 99.995th percentile, the median at target."""
        c0, hi, xm = _anchors(st)
        m = mtf_midtone(xm, target)
        return mtf(((x - c0) / (hi - c0)).clamp(0, 1), m).clamp(0, 1)


def image_stats(lin: np.ndarray, shown: np.ndarray) -> dict:
    """The statistics a simple rendering needs, on the shown pixels of a linear (3, H, W) image: of the luminance
    and of each channel (median, MAD, 99.995th percentile), and the luminance's 40th and 85th percentiles."""
    def one(v: np.ndarray) -> dict:
        med = float(np.median(v))
        return {"median": med, "mad": 1.4826 * float(np.median(np.abs(v - med))),
                "p99995": float(np.percentile(v, 99.995))}
    L = (0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2])[shown]
    out = {"lum": one(L), "channels": [one(lin[c][shown]) for c in range(3)]}
    out["lum_p40"], out["lum_p85"] = (float(v) for v in np.percentile(L, [40, 85]))
    return out
