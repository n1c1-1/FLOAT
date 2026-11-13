"""The combination of filters (issue #43): the outputs of several workflows (one per filter of a monochrome camera, or
the channels of one colour workflow shot through a dual-band filter) combined into one colour image. One block per
type of combination, by what was photographed: S-H-O (S II, H-alpha, O III), H-O (H-alpha, O III), RGB (R, G, B),
LRGB (L, R, G, B). Each is a subclass of ``Combination``, which does what they share; a subclass declares its named
inputs (``ports``), its parameters and its own ``combine``. A type added later is a class added here.

A combination workflow starts from other workflows (``Workflow.source = {"fusion": {"type", "inputs"}}``): its stage
reads, for each port, the image of a channel's workflow (its Output's linear image, render_simple: lin.npy, the sky
taken off; or its Stars/background's starless image, when the user wants the palette made on starless channels) and
writes a stack, which its Output block renders as any stack (core/workflow.PROVIDES). The combination handles no star:
a Stars/background block after it, and the Combine layers block in Post, do (one block, one function).

1. Grid: the inputs must be on one pixel grid (those of the same workflow, a dual-band image's channels, are; images
   of other workflows are once an Align on block, placed after their Stacking, brought them onto the first one's). The
   combination aligns nothing itself: inputs not on one grid are refused, naming them.
2. Each channel is its input's luminance (a monochrome image's grey), or one of its colours (``R``, ``G``, ``B``), or
   ``GB`` (the mean of green and blue: the O III of a dual-band filter on a colour sensor, whose green and blue pixels
   both see the 501 nm line).
3. Each channel's sky is taken to 0 (``neutralise_sky``: the background neutralisation, below), then the channel is
   brought to the same scale. Narrow-band types: Siril's linear match (``linear_match``, below), each filter fitted
   linearly to the H-alpha's, so that a faint O III weighs as much as a strong H-alpha. RGB and LRGB: each channel
   divided by a high percentile of its signal (99.5 %, above its sky at 0). The user's weight of a channel scales it
   after.
4. The type's ``combine`` makes red, green and blue. The narrow-band types give each filter a hue (Rector, Levay,
   Frattare, English & Pu'uohau-Pummill 2007, AJ 133, 598, "Image-processing techniques for the creation of
   presentation-quality astronomical images": each filter's greyscale image is given its own hue and the colourised
   layers are added, the colour balance then set on the whole). In linear data, a channel tinted with a hue and the
   layers added is a matrix whose columns are the hues' colours (``hue_palette``); the colour balance is each
   column's weight, solved so that equal channels give a grey: a single filter's signal has exactly its own hue, a
   neutral input stays neutral. Hues 0, 120 and 240 are the plain mapping (S II red, H-alpha green, O III blue).
5. A narrow-band combination is composed as lightness and colour (``compose``): the lightness is the channels' mean
   weighted by their noise (or equal), the hues give only the colour, smoothed where the noisiest channel's grain would
   show: the noise of a weak filter, raised by the balance (M27's S II 15 times noisier than its H-alpha), no longer
   makes coloured blotches over the sky.

The S-H-O default hues are measured, not chosen (issue #163): five public SHO composites of the Hubble teams
(ESA/Hubble heic1501a, opo9544a, heic1007a, heic1520a, heic0707a: the Pillars of Creation 2014 and 1995, Mystic
Mountain, the Veil, Carina), their nebula's pixels (the sky and the stars masked) in Oklch (B. Ottosson 2020, the
perceptual space of hue, chroma and lightness): the warm areas' hue has its median at 57.1 degrees and its lower
quartile at 36.0, the blue areas' its median at 244.7 (pooled; the per-image medians 13-88 and 176-278). An H-alpha
signal alone lands on the warm median, an S II alone on the warm lower quartile (the S II-rich areas are the reddest
of the warm ones), an O III alone on the blue median. A hue here is on the HSV wheel of the linear RGB; Post shows the
combination's linear colour ratios as display values (render/simple: the colours kept), read as sRGB, so each target
is the HSV hue whose colour, at a display level of 0.6, has that Oklch hue: S II 15.5 degrees (Oklch 36.0), H-alpha
31.4 (57.1), O III 202.2 (244.7); from 0.3 to 0.9 they move by 4 degrees or less. The H-O defaults are the dual-band
filter's usual mapping (H-alpha red, O III cyan: hues 0 and 180).

Background neutralisation (step 3): an empty sky must map to (0, 0, 0) whatever the hues, so each channel's sky level
is measured on the same sky pixels in every channel and subtracted in the linear data, before the balance. It is the
first step of the narrow-band compositions (Rector et al. 2007 set the background of each filter's image to zero
before the layers are coloured and added; the background neutralisation of the usual processing programs is the same operation
on the colour channels). The input's own sky level is not enough: it is the median of the whole shown frame,
nebulosity included, so it differs from one filter to the other by what the nebula adds, and the balance divides it by
a factor ten to twenty times larger for a weak S II than for an H-alpha: a few tenths of a percent of offset in the S II
became a red or cyan cast of the whole sky (set 56: 8 times the sky's own noise).

Linear match (step 3 of the narrow-band types): Siril's, the ``linear_match reference low high`` command and the
Linear match button of its RGB compositing tool (src/filters/linear_match.c and find_linear_coeff in
src/algos/fitting.c of its source; siril.readthedocs.io, "Commands" and "RGB compositing"). For each channel, the
pixels where the reference's value lies in [low, high] (and is not 0, nor the channel's) are kept and the channel is
fitted to the reference by least squares, ``reference = a * channel + b``; the channel becomes ``a * channel + b``.
low and high are the RGB compositing tool's own, 1e-7 and 0.875 (the command has none; its Linear match dialog has 0
and 0.92), in the units of the image: fractions of the sensor's full scale above its dark level (``full_scale`` of the
reference's input: Siril's images are 0 to 1, ours are in ADU). The reference is the H-alpha, the strongest
signal: Siril's FAQ (siril.org/faq: linear match the other stacks, "pick the Ha stack as reference") and its
compositing tool, which takes the top layer loaded as the reference, the H-alpha loaded first for narrow-band images.
The fit's offset moves the sky, which is taken back to 0 after it (the compositions need an empty sky at 0).

The result's valid area is where every input has data. Its scale: the first input's (its level kept). Any workflow may
feed any port; a port with no workflow is the empty state, and the block waits for it.
"""
from __future__ import annotations

import colorsys
import json
from pathlib import Path
from typing import ClassVar

import numpy as np

from ..core.params import P, StageParams
from ..core.stage import Context, Stage, StageError

CHANNELS = ("L", "R", "G", "B", "GB")
#: what a port reads of its channel's workflow: its Output's image, or its Stars/background's starless image
IMAGES = {"output": ("render_simple", "lin.npy"), "starless": ("star_removal", "starless.npy")}
LUM = np.array([0.2126, 0.7152, 0.0722], np.float32)
#: the S-H-O hues measured on the Hubble teams' images (module docstring): S II red-orange, H-alpha copper, O III
#: blue; and the Oklch hues they land on in Post's display (the warm areas' lower quartile and median, the blue
#: areas' median)
HUBBLE_HUES = (15.5, 31.4, 202.2)
HUBBLE_OKLCH = (36.0, 57.1, 244.7)
#: the colour's floor, in units of its luminance's noise: below it the colour is grey, from 1 to 2 floors it comes in
CHROMA_FLOOR = 3.0
#: the share of the covered blocks taken for sky: the darkest fifth of the field (the usual rule of the background
#: samples: the faintest tiles of the frame, where the objects are not)
SKY_SHARE = 0.2
#: the side of the blocks whose medians find the sky (px): wide enough that a star or the noise does not move a block's
#: median, narrow enough that a faint nebula leaves its own blocks
SKY_BLOCK = 16
#: the bounds of the linear match (module docstring): Siril's RGB compositing tool's own
MATCH_LOW, MATCH_HIGH = 1e-7, 0.875
#: the narrow-band types' reference of the linear match: the H-alpha
MATCH_REFERENCE = "Ha"


# ---------------------------------------------------------------------------------------------- the shared steps

def hue_palette(hues: tuple[float, ...]) -> list[list[float]]:
    """The matrix of filters given hues (Rector et al. 2007): each channel tinted with its hue (degrees on the HSV
    colour wheel, its colour fully saturated, its brightest component 1), the layers added; each column then weighted
    so that equal channels give a grey (1, 1, 1): the colour balance, which keeps each hue as it is. Three hues solve
    it exactly; two (H-O) by least squares, exact when the hues face each other. Hues whose balance needs a weight of
    0 or less (no grey between them) are refused."""
    cols = np.array([colorsys.hsv_to_rgb((h % 360) / 360, 1.0, 1.0) for h in hues]).T
    try:
        w = np.linalg.solve(cols, np.ones(3)) if len(hues) == 3 else np.linalg.lstsq(cols, np.ones(3), rcond=None)[0]
    except np.linalg.LinAlgError:                      # e.g. three hues without blue
        w = np.zeros(len(hues))
    if not (w > 0).all():
        raise ValueError(f"the hues {hues} do not balance to a grey")
    return (cols * w[None]).round(6).tolist()


#: the ports' names as the interface writes them
PORT_NAMES = {"SII": "S II", "Ha": "Hα", "OIII": "O III"}


#: the reasons a combination waits (the interface translates the templates, filling the names in)
TO_WIRE = "{names} to wire"
GRID_ONE = "{names} is not on the grid of {first}: place Align on after its Stacking"
GRID_MANY = "{names} are not on the grid of {first}: place Align on after their Stacking"


def grid_problem(inputs: list[dict], ports: tuple[str, ...]) -> str:
    """Why a combination's inputs cannot be combined as they are ("" when they can): every input must be on the grid
    of the first port's (``inputs``: dicts with a ``role`` and a ``grid``, equal for two images on one grid). The
    combination aligns nothing: an Align on block placed after the Stacking of the others does."""
    ins = sorted((i for i in inputs if i["role"] in ports), key=lambda i: ports.index(i["role"]))
    if len(ins) < 2:
        return ""
    name = lambda r: PORT_NAMES.get(r, r)  # noqa: E731
    off = [name(i["role"]) for i in ins[1:] if i.get("grid") != ins[0].get("grid")]
    if not off:
        return ""
    return (GRID_MANY if len(off) > 1 else GRID_ONE).format(names=", ".join(off), first=name(ins[0]["role"]))


def channel_of(lin: np.ndarray, channel: str) -> np.ndarray:
    """One channel of an input's linear image (3, H, W): its luminance (L), a colour, or the mean of green and blue."""
    if channel == "L":
        return np.tensordot(LUM, lin, 1).astype(np.float32)
    if channel == "GB":
        return (0.5 * (lin[1] + lin[2])).astype(np.float32)
    return lin["RGB".index(channel)].astype(np.float32)


def linear_match(ch: np.ndarray, ref: np.ndarray, valid: np.ndarray, low: float = MATCH_LOW,
                 high: float = MATCH_HIGH) -> tuple[float, float, int]:
    """(a, b, the pixels used): Siril's linear match of a channel to a reference (module docstring), the least squares
    fit ``ref = a * ch + b`` on the pixels where ``ref`` lies in [low, high] and neither is 0."""
    m = valid & (ref >= low) & (ref <= high) & (ref != 0) & (ch != 0) & np.isfinite(ref) & np.isfinite(ch)
    n = int(m.sum())
    if n < 2:
        raise StageError("the channels cannot be matched: fewer than 2 pixels have a reference value between "
                         f"{low:g} and {high:g}")
    x, y = ch[m].astype(np.float64), ref[m].astype(np.float64)
    mx, my = x.mean(), y.mean()
    var = float(np.sum((x - mx) ** 2))
    if var <= 0:
        raise StageError("the channels cannot be matched: a channel is flat where the reference has data")
    a = float(np.sum((x - mx) * (y - my)) / var)
    return a, float(my - a * mx), n


def balance(ch: np.ndarray, valid: np.ndarray, pct: float) -> tuple[np.ndarray, float]:
    """A channel divided by the given percentile of its valid pixels above 0 (its sky is at 0)."""
    v = ch[valid]
    v = v[np.isfinite(v)]
    top = float(np.percentile(v, pct)) if v.size else 1.0
    top = top if top > 0 else 1.0
    return ch / np.float32(top), top


def block_medians(a: np.ndarray, b: int) -> np.ndarray:
    """The median of each b x b block of an image (the incomplete border dropped)."""
    h, w = a.shape[0] // b, a.shape[1] // b
    return np.median(a[:h * b, :w * b].reshape(h, b, w, b).transpose(0, 2, 1, 3).reshape(h, w, b * b), axis=2)


def sky_mask(chans: dict[str, np.ndarray], valid: np.ndarray, share: float = SKY_SHARE, b: int = SKY_BLOCK) -> np.ndarray:
    """(H, W) bool: the pixels of the blocks that are the sky of every channel at once. Each channel's block medians
    are put in units of their own robust spread (the median absolute deviation across the frame) and added: a block is
    sky when it is among the faintest ``share`` of the covered blocks in all the channels together, whichever filter
    is weak or strong. The same pixels serve every channel, so that the skies are compared where they are the same."""
    H, W = valid.shape
    b = max(2, min(b, H // 8, W // 8))
    ok = block_medians(valid.astype(np.float32), b) > 0.999
    z = 0.0
    for ch in chans.values():
        m = block_medians(np.where(valid & np.isfinite(ch), ch, 0).astype(np.float32), b)
        med = float(np.median(m[ok])) if ok.any() else 0.0
        spread = 1.4826 * float(np.median(np.abs(m[ok] - med))) if ok.any() else 0.0
        z = z + (m - med) / (spread or 1.0)
    out = np.zeros((H, W), bool)
    if not ok.any():
        return out
    keep = ok & (z <= np.percentile(z[ok], 100 * share))
    out[:keep.shape[0] * b, :keep.shape[1] * b] = np.kron(keep, np.ones((b, b), bool))
    return out & valid


def sky_level(ch: np.ndarray, sky: np.ndarray, clip: float = 3.0, iters: int = 5) -> tuple[float, float]:
    """(the level, the noise) of a channel's sky on the pixels of ``sky``: the median of the samples kept after
    clipping those further than ``clip`` robust sigmas from it (the stars' and nebula's remains), iterated."""
    v = ch[sky]
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0, 0.0
    med, sig = float(np.median(v)), 0.0
    for _ in range(iters):
        med = float(np.median(v))
        sig = 1.4826 * float(np.median(np.abs(v - med)))
        keep = np.abs(v - med) <= clip * max(sig, 1e-12)
        if keep.all():
            break
        v = v[keep]
    return med, sig


def neutralise_sky(chans: dict[str, np.ndarray], valid: np.ndarray) -> dict[str, tuple[float, float]]:
    """The background neutralisation (module docstring): every channel loses its sky level, in place, measured on the
    pixels that are sky in all of them together (``sky_mask``). Returns {role: (level removed, sky noise)}."""
    sky = sky_mask(chans, valid)
    out = {}
    for role, ch in chans.items():
        level, noise = sky_level(ch, sky)
        ch -= np.float32(level)
        out[role] = (level, noise)
    return out


def mix(chans: dict[str, np.ndarray], roles: tuple[str, ...], mat) -> np.ndarray:
    """(3, H, W): the balanced channels of ``roles``, in that order, through a matrix (rows R, G, B)."""
    return np.tensordot(np.asarray(mat, np.float32), np.stack([chans[r] for r in roles]), 1).astype(np.float32)


def lightness(chans: dict[str, np.ndarray], noise: dict[str, float], equal: bool = False) -> tuple[np.ndarray, float]:
    """(the lightness, its noise): the inverse-variance weighted mean of the balanced channels (each weighed by 1 over
    its sky noise squared, the minimum-variance combination of measures of one quantity: the noise-weighted luminance
    of narrow-band and LRGB work), or with ``equal`` their plain mean. Equal channels give that same value: a neutral
    source keeps its level."""
    w = {r: 1.0 if equal else 1.0 / max(noise[r], 1e-12) ** 2 for r in chans}
    tot = sum(w.values())
    L = sum(np.float32(w[r] / tot) * chans[r] for r in chans)
    noise_l = float(np.sqrt(sum((w[r] / tot * noise[r]) ** 2 for r in chans))) if equal else float(tot ** -0.5)
    return L.astype(np.float32), noise_l


def chroma_sigma(noise: dict[str, float], noise_l: float) -> float:
    """The Gaussian radius (px) of the colour's smoothing: where the noisiest channel's noise, averaged by the
    Gaussian (white noise: its standard deviation divided by 2 sqrt(pi) sigma), comes down to the lightness's own.
    The colour is then no noisier than the lightness that carries the detail; 0 below half a pixel."""
    s = max(noise.values()) / (2 * np.sqrt(np.pi) * max(noise_l, 1e-12))
    return float(s) if s >= 0.5 else 0.0


def compose(chans: dict[str, np.ndarray], roles: tuple[str, ...], mat, noise: dict[str, float],
            sigma: float | None = None, sky: np.ndarray | None = None, equal: bool = False) -> tuple[np.ndarray, dict]:
    """(3, H, W) of a narrow-band combination composed as luminance and chrominance (issue #163; the LRGB composition,
    the colour of one image and the lightness of another, R. Gendler, "Lessons from the Masters", Springer 2013, and
    Rector et al. 2007 for the colour layers): the lightness is the mean of the channels (``lightness``), the hues give
    only the colour, its ratios to its own luminance. The colour is smoothed at ``sigma`` px (None: ``chroma_sigma``,
    from the noise; 0: none), the lightness keeping every detail: the luminance-chrominance noise reduction, as the eye
    resolves colour much more coarsely than lightness (K. T. Mullen 1985, J. Physiol. 359, 381: the chromatic contrast
    sensitivity falls off at a few times lower spatial frequencies; the reason of the chroma subsampling of JPEG and
    video). Where the colour's luminance is under CHROMA_FLOOR times its noise (the sky at 0) the pixel is grey, from
    one to two floors the colour comes in: the noise of a weak filter, raised by the balance, no longer paints the sky.
    That noise is measured on the ``sky`` pixels (the channels' common sky, ``sky_mask``) of the smoothed colour's
    luminance: the residues of the channels' skies and their noise's grain, wider than one pixel, are what a white
    noise's formula would miss; without sky pixels, the formula. Returns (the image, notes for the log)."""
    rgb = mix(chans, roles, mat)
    L, noise_l = lightness(chans, noise, equal)
    lum_w = LUM.astype(np.float64) @ np.asarray(mat, np.float64)     # each channel's share of the colour's luminance
    noise_c = float(np.sqrt(sum((lum_w[i] * noise[r]) ** 2 for i, r in enumerate(roles))))
    sig = chroma_sigma({r: noise[r] for r in roles}, noise_l) if sigma is None else float(sigma)
    if sig > 0:
        import cv2
        rgb = np.stack([cv2.GaussianBlur(c, (0, 0), sig) for c in rgb])
        noise_c /= 2 * np.sqrt(np.pi) * sig
    lum = np.tensordot(LUM, rgb, 1)
    if sky is not None and sky.sum() > 1000:
        med, spread = sky_level(lum, sky)
        floor = abs(med) + CHROMA_FLOOR * max(spread, 1e-12)
    else:
        floor = CHROMA_FLOOR * max(noise_c, 1e-12)
    t = np.clip(lum / floor - 1, 0, 1)
    t = t * t * (3 - 2 * t)                             # from one to two floors, smoothly
    ratio = 1 + t[None] * (rgb / np.maximum(lum, floor)[None] - 1)
    return (ratio * L[None]).astype(np.float32), {"sigma": sig, "noise_l": noise_l, "floor": floor}


# ---------------------------------------------------------------------------------------------- the blocks

class CombinationParams(StageParams):
    #: the inputs, set by the project (not keyed: their workflows' keys are): [{"role" (the port), "channel", "pid",
    #: "image", "dir", "wcs", "grid"}], the first one the reference grid
    inputs: list[dict] = P([], label="Inputs", level="internal",
                           help="The images of the workflows combined, the port each one feeds, and its channel.")
    #: the user's weight of each channel ({port: weight}, 1 by default): a channel too faint for the balance (its noise
    #: raised with it) is lowered by hand
    weights: dict = P({}, label="Channel weights", level="internal",
                      help="Each channel's weight, after the balance (1: as balanced).")


class BalancedParams(CombinationParams):
    percentile: float = P(99.5, label="Channel balance", level="advanced", ge=90, le=99.99, step=0.1, unit="%",
                          help="Each channel is divided by this percentile of its signal before it is combined: the "
                               "filters weigh alike.")


class NarrowbandParams(CombinationParams):
    lightness: str = P("noise", label="Lightness", choices=["noise", "equal"],
                       help="How the filters make the lightness: weighted by their noise (the least noisy image), or "
                            "equally.")
    smoothing: str = P("auto", label="Colour smoothing", choices=["auto", "radius"],
                       help="The colour is smoothed, the lightness keeps its detail: Auto takes the radius from the "
                            "filters' noise (shown in the results); Radius takes the one set below.")
    radius: float = P(0.0, label="Smoothing radius", ge=0, le=50, step=0.5, unit="px",
                      help="The colour's smoothing radius when Colour smoothing is Radius; 0: none.")


class Combination(Stage):
    """What every type of combination shares: the named inputs read, checked to overlap, aligned on the first one's
    grid; the sky neutralised; the balance and the weights; the outputs, the previews and the metrics. A subclass
    declares ``kind`` (the source's type), ``ports`` (its named inputs, in the order its ``combine`` reads them) and
    ``narrowband`` (its result's origin for Post), and writes ``combine``."""

    kind: ClassVar[str]
    ports: ClassVar[tuple[str, ...]]
    narrowband: ClassVar[bool] = False
    Params = CombinationParams

    def combine(self, ctx: Context, p, chans: dict[str, np.ndarray], noise: dict[str, float],
                valid: np.ndarray) -> np.ndarray:
        """(3, H, W): the balanced channels by port (each sky at 0) made into red, green and blue (before the first
        channel's level is given back). ``noise``: each balanced channel's sky noise."""
        raise NotImplementedError

    def run(self, ctx: Context, p: CombinationParams) -> None:
        from astropy.io import fits
        from astropy.wcs import WCS
        if not p.inputs:
            raise StageError("no workflow to combine: choose the channels of the combination")
        missing = [r for r in self.ports if r not in {i["role"] for i in p.inputs}]
        if missing:
            raise StageError(f"{self.title} needs the {', '.join(missing)} channel(s)")
        ref: dict = {}
        chans: dict[str, np.ndarray] = {}
        valid_all = None
        tops: dict[str, float] = {}
        shapes: dict[str, list[int]] = {}
        inputs = [i for i in p.inputs if i["role"] in self.ports]
        bad = grid_problem(inputs, self.ports)
        if bad:
            raise StageError(bad)
        for n, inp in enumerate(inputs):
            d = Path(inp["dir"])
            image = inp.get("image", "output")
            name = IMAGES[image][1]
            if not (d / name).is_file():
                raise StageError(f"the starless image of the {inp['role']} channel is not computed: run its "
                                 "Stars/background" if image == "starless" else
                                 f"the output of the {inp['role']} channel is not computed")
            valid = np.load(d / "valid.npy")
            ch = channel_of(np.asarray(np.load(d / name, mmap_mode="r")), inp.get("channel", "L"))
            wcs_f = Path(inp.get("wcs") or "")
            if not ref:
                ref = {"shape": ch.shape, "wcs": WCS(fits.getheader(wcs_f)) if wcs_f.is_file() else None, "dir": d}
            shapes[inp["role"]] = list(ch.shape)
            valid_all = valid if valid_all is None else (valid_all & valid)
            chans[inp["role"]] = ch
            ctx.progress(0.6 * (n + 1) / len(inputs), f"channel {inp['role']}")
        # a safety net: the grids said alike, the sizes did not
        bad = grid_problem([{"role": r, "grid": s} for r, s in shapes.items()], self.ports)
        if bad:
            raise StageError(bad)
        if not valid_all.any():
            raise StageError("the channels do not overlap: no pixel where every one has data")
        skies = neutralise_sky(chans, valid_all)
        ctx.log("sky removed (level, noise): " + ", ".join(f"{r} {a:.4g}, {b:.3g}" for r, (a, b) in skies.items()))
        if self.narrowband:
            # Siril's linear match of every filter to the H-alpha; the result keeps the H-alpha's scale (top 1)
            ref_role = MATCH_REFERENCE if MATCH_REFERENCE in chans else next(iter(chans))
            ref_ch = chans[ref_role]
            full = float(next(i for i in inputs if i["role"] == ref_role).get("full_scale") or 1.0)
            for role in list(chans):
                tops[role] = 1.0
                if role == ref_role:
                    continue
                a, b, used = linear_match(chans[role], ref_ch, valid_all, MATCH_LOW * full, MATCH_HIGH * full)
                if not a > 0:
                    raise StageError(f"the {role} channel does not follow the {ref_role} one (a negative fit): "
                                     "check that they show the same field")
                chans[role] = (chans[role] * np.float32(a) + np.float32(b)).astype(np.float32)
                tops[role] = 1.0 / a
                ctx.log(f"{role} matched to {ref_role} on its pixels between {MATCH_LOW * full:.3g} and "
                        f"{MATCH_HIGH * full:.5g}: {a:.4g} x + {b:.3g} ({used} pixels)")
            skies = neutralise_sky(chans, valid_all)           # the fits' offsets taken off: the sky at 0 again
            level = 1.0
        else:
            level = None
            for role in list(chans):
                chans[role], tops[role] = balance(chans[role], valid_all, p.percentile)
                if level is None:
                    level = tops[role]
            skies = {r: (skies[r][0], skies[r][1] / tops[r]) for r in chans}
        for role in list(chans):
            wgt = float(p.weights.get(role, 1.0))
            if wgt != 1.0:                           # the user's weight: the balance's divisor scaled with it
                chans[role] *= np.float32(wgt)
                tops[role] /= wgt
                skies[role] = (skies[role][0], skies[role][1] * wgt)
                ctx.log(f"{role}: weight {wgt:g}")
        # each balanced channel's sky noise: the sky's, scaled as the channel was (the weight included)
        noise = {r: skies[r][1] for r in chans}
        rgb = self.combine(ctx, p, chans, noise, valid_all) * np.float32(level or 1.0)
        rgb[:, ~valid_all] = 0
        H, W = ref["shape"]
        ctx.save("drz_full.npy", rgb.astype(np.float32))
        ctx.save("drz_weight.npy", np.repeat(valid_all[None].astype(np.float32), 3, axis=0))
        meta = json.loads((ref["dir"] / "prep_simple.json").read_text(encoding="utf-8")) \
            if (ref["dir"] / "prep_simple.json").is_file() else {}
        ctx.save_json("drz_meta.json", {"scale": float(meta.get("drizzle_scale", 1.0)), "source": "fusion",
                                        "kind": self.kind, "narrowband": self.narrowband, "width": W, "height": H,
                                        "sky_neutral": True, "channels": {r: round(t, 4) for r, t in tops.items()}})
        if ref["wcs"] is not None:
            ref["wcs"].to_fits().writeto(ctx.out("wcs.fits"), overwrite=True)
        ctx.metric("channels", sorted(chans))
        ctx.log(f"{self.title} from {', '.join(chans)}; channel scales " + ", ".join(f"{r} {t:.4g}" for r, t in tops.items()))
        v = np.moveaxis(rgb, 0, -1)
        lo, hi = np.percentile(v[valid_all], 1), np.percentile(v[valid_all], 99.7)
        ctx.preview("fusion", np.clip((v - lo) / max(hi - lo, 1e-6), 0, 1) ** 0.5)


class NarrowbandCombination(Combination):
    """The narrow-band types: each filter given a hue, composed as lightness and colour (module docstring, 4-5). A
    subclass names its hue parameters ``hue_<port>``, in its ports' order."""

    narrowband = True

    def hues(self, p) -> tuple[float, ...]:
        return tuple(float(getattr(p, f"hue_{r.lower()}")) for r in self.ports)

    def combine(self, ctx, p, chans, noise, valid):
        try:
            mat = hue_palette(self.hues(p))
        except ValueError as e:
            raise StageError(f"{e}: choose hues around the colour wheel") from None
        sigma = None if p.smoothing == "auto" else p.radius
        rgb, notes = compose(chans, self.ports, mat, noise, sigma, sky=sky_mask(chans, valid),
                             equal=p.lightness == "equal")
        ctx.log(f"lightness from the channels {'weighted by their noise' if p.lightness == 'noise' else 'equally'} ("
                + ", ".join(f"{r} {(notes['noise_l'] / max(noise[r], 1e-12)) ** 2:.2f}" for r in chans)
                + "), colour from the hues " + ", ".join(f"{r} {h:g}" for r, h in zip(self.ports, self.hues(p)))
                + (f", smoothed over {notes['sigma']:.1f} px" if notes["sigma"] else ""))
        ctx.metric("chroma_sigma", round(notes["sigma"], 2))
        return rgb


class SHOParams(NarrowbandParams):
    hue_sii: float = P(HUBBLE_HUES[0], label="S II hue", ge=0, le=360, step=1, unit="°",
                       help="The colour of the S II filter (on the colour wheel: 0 red, 120 green, 240 blue). The "
                            "default is measured on the Hubble teams' images (red-orange).")
    hue_ha: float = P(HUBBLE_HUES[1], label="Hα hue", ge=0, le=360, step=1, unit="°",
                      help="The colour of the H-alpha filter (0 red, 120 green, 240 blue). The default is measured on "
                           "the Hubble teams' images (copper).")
    hue_oiii: float = P(HUBBLE_HUES[2], label="O III hue", ge=0, le=360, step=1, unit="°",
                        help="The colour of the O III filter (0 red, 120 green, 240 blue). The default is measured on "
                             "the Hubble teams' images (blue).")


class CombineSHO(NarrowbandCombination):
    name = "combine_sho"
    title = "S-H-O"
    kind = "sho"
    ports = ("SII", "Ha", "OIII")
    description = ("Combines the S II, H-alpha and O III filters into one colour image: each filter a hue, the lightness "
                   "from the filters weighted by their noise, aligned through their plate solutions.")
    Params = SHOParams
    version = "3"


class HOParams(NarrowbandParams):
    hue_ha: float = P(0.0, label="Hα hue", ge=0, le=360, step=1, unit="°",
                      help="The colour of the H-alpha filter (0 red, 120 green, 240 blue).")
    hue_oiii: float = P(180.0, label="O III hue", ge=0, le=360, step=1, unit="°",
                        help="The colour of the O III filter (180: green and blue).")


class CombineHO(NarrowbandCombination):
    name = "combine_ho"
    title = "H-O"
    kind = "ho"
    ports = ("Ha", "OIII")
    description = ("Combines the H-alpha and O III filters (two filters, or a dual-band image's channels) into one "
                   "colour image: each filter a hue, the lightness from the filters weighted by their noise.")
    Params = HOParams
    version = "3"


class CombineRGB(Combination):
    name = "combine_rgb"
    title = "RGB"
    kind = "rgb"
    ports = ("R", "G", "B")
    description = "Combines the red, green and blue filters into one colour image, aligned through their plate solutions."
    Params = BalancedParams
    version = "1"

    def combine(self, ctx, p, chans, noise, valid):
        return mix(chans, self.ports, np.eye(3))


class CombineLRGB(Combination):
    name = "combine_lrgb"
    title = "LRGB"
    kind = "lrgb"
    ports = ("L", "R", "G", "B")
    description = ("Combines the red, green and blue filters into the colour, the luminance filter into the detail: "
                   "each pixel's colour scaled so that its luminance is L's.")
    Params = BalancedParams
    version = "1"

    def combine(self, ctx, p, chans, noise, valid):
        rgb = np.stack([chans["R"], chans["G"], chans["B"]])
        lum = np.tensordot(LUM, rgb, 1)
        # the colour ratio only where the colour has signal: with the sky at 0 its luminance is noise around 0, whose
        # ratio to L's is meaningless; the floor is 3 robust sigmas of the colour's luminance over the frame
        med = float(np.median(lum))
        floor = max(1e-6, 3 * 1.4826 * float(np.median(np.abs(lum - med))))
        k = np.where(lum > floor, chans["L"] / np.maximum(lum, floor), 1.0).astype(np.float32)
        return (rgb * k[None]).astype(np.float32)


#: the types of combination, in the palette's order; adding a type is adding its class here
COMBINATIONS: tuple[type[Combination], ...] = (CombineSHO, CombineHO, CombineRGB, CombineLRGB)
BY_KIND: dict[str, type[Combination]] = {c.kind: c for c in COMBINATIONS}
#: the stages of the combinations (core/workflow: each the first stage of its workflow)
STAGES: tuple[str, ...] = tuple(c.name for c in COMBINATIONS)


def stage_of(kind: str) -> str:
    """The stage of a type of combination ("sho" → "combine_sho")."""
    return BY_KIND[kind].name


# ---------------------------------------------------------------------------------------------- saved projects

#: the palettes of the former single Combine channels block (stages/fusion.py until 2026-11-12) → (type, settings
#: that reproduce it): the SHO Hubble palette is the S-H-O defaults, the plain SHO hues 0/120/240, HSO the same with
#: H-alpha red and S II green, HOO the H-O defaults
FORMER_PALETTES: dict[str, tuple[str, dict]] = {
    "SHO Hubble": ("sho", {}),
    "SHO": ("sho", {"hue_sii": 0.0, "hue_ha": 120.0, "hue_oiii": 240.0}),
    "HSO": ("sho", {"hue_sii": 120.0, "hue_ha": 0.0, "hue_oiii": 240.0}),
    "HOO": ("ho", {}),
    "RGB": ("rgb", {}),
    "LRGB": ("lrgb", {}),
}


def upgraded_workflow(w: dict) -> dict:
    """A workflow of a saved project as it is now (its JSON, changed in place and returned): a Combine channels of the
    former single block becomes the block of its palette's type, its settings carried over (core/project2.from_dict).
    The source keeps its inputs that are ports of the type (in their order: the first stays the reference grid); the
    palette becomes the hues; the weights and the stored settings go to the block's parameters; a fusion without its
    stars taken off had no colour smoothing (smoothing Radius, 0), one with them had it from the noise (Auto, the
    default). Its option A (the channels' starless images, a star layer from the channels or another workflow) is
    made into inputs reading the starless images and a Combine layers layer with its stars side
    (core/project2._stars_of_former_fusions, which runs before this); a Stars/background block already after the
    combination stays."""
    f = (w.get("source") or {}).get("fusion") if isinstance(w.get("source"), dict) else None
    if not isinstance(f, dict) or "palette" not in f or f["palette"] not in FORMER_PALETTES:
        return w
    kind, values = FORMER_PALETTES[f["palette"]]
    cls = BY_KIND[kind]
    params = dict(w.get("params") or {})
    old = dict(params.pop("fusion", None) or {})
    mine = {k: v for k, v in old.items() if k in cls.Params.model_fields and k not in ("inputs", "weights")}
    mine.update(values)
    if f.get("weights"):
        mine["weights"] = {r: v for r, v in f["weights"].items() if r in cls.ports}
    if cls.narrowband and not f.get("stars"):
        mine.update(smoothing="radius", radius=0.0)
    if mine:
        params[cls.name] = mine
    w["params"] = params
    ins = [{"pid": i["pid"], "role": i["role"], "channel": i.get("channel", "L")}
           | ({"image": i["image"]} if i.get("image") in IMAGES else {}) for i in f.get("inputs") or []
           if i.get("role") in cls.ports]
    w["source"] = {"fusion": {"type": kind, "inputs": ins}}
    return w
