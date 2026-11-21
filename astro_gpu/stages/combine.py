"""The combination of filters (issue #43): the outputs of several workflows (one per filter of a monochrome camera, or
the channels of one colour workflow shot through a dual-band filter) combined into one colour image.

What a combination does is a recipe (stages/recipes.py): a TOML file of the application's folder
``astro_gpu/resources/combinations`` that names its inputs, the operations on them from a fixed vocabulary (the
background neutralisation, the equalisation, Siril's linear match, the continuum subtraction) and how they make
red, green and blue (a hue per filter or a matrix; composed as lightness and colour or mixed as they are; the colour smoothed and its
strength). ``Combination.run`` carries the recipe out with the functions of this module; the folder's
``_example.toml`` documents every setting. The recipe is copied into the project when the combination is placed: a
project computes the same whatever happens to the file.

The four types the application had before recipes, by what was photographed: S-H-O (S II, H-alpha, O III), H-O
(H-alpha, O III), RGB (R, G, B), LRGB (L, R, G, B), are the shipped recipes ``sho``, ``ho``, ``rgb``, ``lrgb``; each
keeps its block (a subclass of ``Combination``: its stage, its parameters, its ports), so that their keys and images
are those they had. Any other recipe runs on the stage ``combine_recipe`` (``CombineRecipe``), its parameters in
``values``. Below, the methods the four types use, which the recipes' operations are.

A combination workflow starts from other workflows (``Workflow.source = {"fusion": {"type", "inputs"}}``): its stage
reads, for each port, the image of a channel's workflow (its Output's linear image, render_simple: lin.npy, the sky
taken off; or its Stars/background's starless image, when the palette is made on starless channels, or its stars alone, for the
stars of a combination whose starless image is made the same way) and
writes a stack, which its Output block renders as any stack (core/workflow.PROVIDES). The combination handles no star:
a Stars/background block after it, and the Combine layers block in Post, do (one block, one function).

1. Grid: the inputs must be on one pixel grid (those of the same workflow, a dual-band image's channels, are; images
   of other workflows are once an Align on block, placed after their Stacking, brought them onto the first one's). The
   combination aligns nothing itself: inputs not on one grid are refused, naming them.
2. Each channel is its input's luminance (a monochrome image's grey), or one of its colours (``R``, ``G``, ``B``), or
   ``GB`` (the mean of green and blue: the O III of a dual-band filter on a colour sensor, whose green and blue pixels
   both see the 501 nm line).
3. Each channel's sky is taken to 0 (``neutralise_sky``: the background neutralisation, below), then the channel is
   brought to the scale of the type's reference (``match_reference``). The narrow-band types equalise each filter to
   the H-alpha (``equalise``, below: its median and its spread), so that a faint O III weighs as much as a strong
   H-alpha. The broadband ones use Siril's linear match (``linear_match``, below), each channel fitted linearly to
   the top layer of Siril's RGB compositing tool (its documentation:
   "The top layer loaded (usually Luminance or Red) is taken as the reference channel, and all other channels are
   automatically linearly matched to it", siril.readthedocs.io, "RGB compositing"): the red for RGB, the luminance
   for LRGB. The user's weight of a channel scales it after.
4. The type's ``combine`` makes red, green and blue. The narrow-band types give each filter a hue (Rector, Levay,
   Frattare, English & Pu'uohau-Pummill 2007, AJ 133, 598, "Image-processing techniques for the creation of
   presentation-quality astronomical images": each filter's greyscale image is given its own hue and the colourised
   layers are added, the colour balance then set on the whole). In linear data, a channel tinted with a hue and the
   layers added is a matrix whose columns are the hues' colours (``hue_palette``); the colour balance is each
   column's weight, solved so that equal channels give a grey: a single filter's signal has exactly its own hue, a
   neutral input stays neutral. Hues 0, 120 and 240 are the plain mapping (S II red, H-alpha green, O III blue).
5. The narrow-band types mix the equalised filters as they are, through that matrix (H-O, hues 0 and 180: red is
   the H-alpha, green and blue are the O III): Siril's RGB composition, which copies the layers and normalises
   nothing itself. A recipe may instead compose its channels as lightness and colour (``compose``: the lightness
   the channels' mean, the hues giving only the colour, smoothed, its strength raised); the two types did until
   their version 8, and no longer do: on a monochrome camera's H-alpha and O III the strength of 1.9 took the green
   to 0 over three quarters of the nebula's interior (2 % at strength 1) and the result was a flat saturated red; on
   a small nebula the S-H-O's sky came out in orange blotches.

Equalisation (step 3 of the narrow-band types): each filter is brought to the reference while linear, before the
composition, ``filter * mad(ref) / mad(filter) - mad(ref) / mad(filter) * median(filter) + median(ref)``: the formula
of Siril's H-alpha and O III extraction script (OSC_Extract_HaOIII) and of its RGB composition tutorial
(siril.org/tutorials/rgb_composition), the median and the median absolute deviation taken on the covered pixels. The
reference is the H-alpha, the strongest signal (siril.org/faq: "pick the Ha stack as reference"). Both statistics
are those of the bulk of the pixels: the stars, a few percent of the field, do not move them, so the filters come out
on one scale with or without their stars. The linear match was the narrow-band types' until their version 8; it is
a least squares fit, which the brightest pixels drive: on images with their stars the stars' ratio set the scale and
the O III of a supernova remnant was lowered (0.77) where its nebula asked for 3.8; on starless images of a small
nebula a weak S II was multiplied by 14 and its sky's residue covered the field. The setting Balance: On sky, the
default, is the equalisation described here; On signal fits the signal with that gain capped by the sky noises' ratio
(``signal_gain`` and ``Combination.steps``). The equalisation's offset moves the sky, which is taken back to 0 after
it.

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

Linear match (step 3 of the broadband types, and of the recipes that ask for it): Siril's, the
``linear_match reference low high`` command and the Linear match button of its RGB compositing tool (src/filters/linear_match.c and find_linear_coeff in
src/algos/fitting.c of its source; siril.readthedocs.io, "Commands" and "RGB compositing"). For each channel, the
pixels where the reference's value lies in [low, high] (and is not 0, nor the channel's) are kept and the channel is
fitted to the reference by least squares, ``reference = a * channel + b``; the channel becomes ``a * channel + b``.
low and high are the RGB compositing tool's own, 1e-7 and 0.875 (the command has none; its Linear match dialog has 0
and 0.92), in the units of the image: fractions of the sensor's full scale above its dark level (``full_scale`` of the
reference's input: Siril's images are 0 to 1, ours are in ADU). The reference is the H-alpha, the strongest
signal: Siril's FAQ (siril.org/faq: linear match the other stacks, "pick the Ha stack as reference") and its
compositing tool, which takes the top layer loaded as the reference, the H-alpha loaded first for narrow-band images.
The fit's offset moves the sky, which is taken back to 0 after it (the compositions need an empty sky at 0).

The broadband types (RGB, LRGB) give the colour as it was recorded; its balance is the Color block's, placed after the
combination (the templates Mono RGB and Mono LRGB): the photometric white balance of the stars, one factor per colour,
as Siril's and the reference processing suite's photometric calibrations of a monochrome camera's R, G and B make it
(Siril's RGB composition tutorial, siril.org/tutorials/rgb_composition: for RGB "no normalisation is needed", the
colour calibration after the composition). So the linear match only puts the channels on one scale; a factor it
leaves is taken off by the Color block's.

- RGB: the channels are the red, green and blue (the channel combination of the usual programs), nothing else.
- LRGB: the colour of RGB, the lightness of L, at every pixel (``compose`` with the lightness given): each pixel's
  colour ratios to their own luminance, times L; where the colour's luminance is under its noise floor the pixel is
  grey (L alone), so the depth of L is kept on the faint areas and the sky, where the colour has no signal. Siril
  substitutes the luminance for the lightness of the colour data (its rgbcomp command, HSL by default, HSV or CIE
  L*a*b*), the reference processing suite's LRGBCombination its CIE lightness: on a linear image the substitution
  that keeps each pixel's chromaticity is this scaling of its three channels. The L is on the colour's scale: the
  R, G and B are linearly matched to it (above). Siril and the photometric calibrations calibrate the colour on the
  RGB before L is added (siril.readthedocs.io, "Spectrophotometric Color Calibration": "We recommend processing only
  RGB with SPCC. The L layer must be added at a later stage"); here the Color block comes after the LRGB. Since L
  scales a pixel's three channels alike, the colour ratios the Color block measures and sets are those of the RGB,
  and its factors give the colour the order of the recommendation would give; only the lightness differs, by the
  luminance's share of the factors (``Y(M c) / Y(c)``, within the factors' spread around the grey's). The colour is
  not smoothed (Siril's composition has no chrominance smoothing; each filter's row has its own Denoise), and its
  strength is not raised: the subject's median Oklch chroma of professional broadband composites spreads over a
  factor five from one object to the next (0.015 to 0.077, nine ESO and ESA/Hubble images: DSS2 B and R plates,
  B V R or B V I), not one target, and ours lies within it.
- The stars' saturation: the combination hands on the saturation map of its colour channels' stacks (``saturation.npy``,
  the stack's own: a pixel near saturation in any of R, G, B), so that the Color block after it leaves out the
  saturated stars by the map, as after a Stacking, not by its estimate from the image's brightest pixels, which on
  a combination refused most of a field's stars (M8 LRGB: 160 of 195 Gaia XP stars, its calibration fell back;
  with the map 129 are measured). The L is not in it: it scales a pixel's three channels alike, so a star
  saturated in L keeps the colour ratios of its R, G and B.

Continuum subtraction (a recipe's ``continuum_subtract``, the HaRGB recipe): a narrow-band filter records its line
and the continuum under it; a broadband filter around the line records the continuum (and the line, diluted in its
width). The stellar method (W. H. Waller 1990, PASP 102, 1217; the standard procedure of H-alpha imaging, e.g.
Koopmann, Kenney & Young 2001, ApJS 135, 125; Spector, Finkelman & Brosch 2012, MNRAS 419, 2156): the field's stars
have no line emission, so the ratio of their fluxes in the two images is the scale of the continuum; the line image
minus the broadband image times that scale is the line alone (``continuum_scale``: aperture photometry of the
isolated, unsaturated stars, each with its local sky in an annulus, the median of their ratios). The line is then
divided by the scale: on the broadband image's scale, a star equal in both.

The stars alone (a second combination reading each filter's stars, the first one their starless images: the
template Filters full, and "<name> · Stars"): through the hues the stars take the palette's colours, magenta with the
H-alpha at a crimson hue, teal with the O III at 180 degrees. A narrow-band filter records a star's continuum in a few
nanometres: the ratio between filters says little of its colour, and the palette's hues say nothing of it. The
established practice gives a narrow-band image white stars, or the stars of red, green and blue filters when the
project has them (the template SHO + RGB stars). So a narrow-band combination whose inputs all read their stars alone
gives them neutral (``neutral_stars``; the setting Stars, Neutral by default): the filters equalised as for the
nebula, then their plain mean in red, green and blue, no weight fitted. Stars: Palette sends them through the hues, as
before the blocks' version 9. The broadband types keep the stars' colours as recorded: they are the true ones.

The result's valid area is where every input has data. Its scale: the first input's (its level kept). Any workflow may
feed any port; a port with no workflow is the empty state, and the block waits for it.
"""
from __future__ import annotations

import colorsys
import json
from pathlib import Path
from typing import ClassVar

import numpy as np

from pydantic import field_validator

from ..core.params import P, StageParams
from ..core.stage import Context, Stage, StageError
from . import recipes

CHANNELS = ("L", "R", "G", "B", "GB")
#: what a port reads of its channel's workflow: its Output's image, or its Stars/background's starless image or stars alone
IMAGES = {"output": ("render_simple", "lin.npy"), "starless": ("star_removal", "starless.npy"),
          "stars": ("star_removal", "stars.npy")}
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
#: the top of the display a colour may reach (``gamut_gain``): the brightest channel's shown value
DISPLAY_CAP = 0.9
#: the bounds of the linear match (module docstring): Siril's RGB compositing tool's own
MATCH_LOW, MATCH_HIGH = 1e-7, 0.875
#: the narrow-band types' reference of the equalisation: the H-alpha
MATCH_REFERENCE = "Ha"
#: what a narrow-band combination makes of the stars alone (module docstring): neutral, or through the palette's hues
STARS = ("neutral", "palette")
#: how a combination's ``equalise`` step brings a filter to its reference (module docstring, Balance): on the sky
#: (the medians and the median absolute deviations, the default), or on the signal
BALANCES = ("sky", "signal")
#: the balance on the signal: the reference's pixels fitted are those above its sky by SIGNAL_SIGMA times its sky
#: noise, up to SIGNAL_HIGH of its full scale
SIGNAL_SIGMA, SIGNAL_HIGH = 3.0, 0.92


# ---------------------------------------------------------------------------------------------- the shared steps

def hue_palette(hues: tuple[float, ...]) -> list[list[float]]:
    """The matrix of filters given hues (Rector et al. 2007): each channel tinted with its hue (degrees on the HSV
    colour wheel, its colour fully saturated, its brightest component 1), the layers added; each column then weighted
    so that equal channels give a grey (1, 1, 1): the colour balance, which keeps each hue as it is. Three hues solve
    it exactly; two (H-O) by least squares, exact when the hues face each other. Hues with no grey between them (a
    weight of 0 or less) are not balanced: the colourised layers are added as they are, in Rector et al.'s order
    (the colour balance of the whole is then the user's, in Post), scaled so that equal channels reach 1 in their
    strongest colour (``hues_neutral`` says which case it is)."""
    cols = np.array([colorsys.hsv_to_rgb((h % 360) / 360, 1.0, 1.0) for h in hues]).T
    try:
        w = np.linalg.solve(cols, np.ones(3)) if len(hues) == 3 else np.linalg.lstsq(cols, np.ones(3), rcond=None)[0]
    except np.linalg.LinAlgError:                      # e.g. three hues without blue
        w = np.zeros(len(hues))
    if not (w > 0).all():
        w = np.full(len(hues), 1.0 / float(cols.sum(axis=1).max()))
    return (cols * w[None]).round(6).tolist()


def hues_neutral(hues: tuple[float, ...]) -> bool:
    """True when equal channels give a grey through these hues (``hue_palette`` balances them), False when the hues
    have no grey between them and the layers are only added."""
    m = np.array(hue_palette(hues))
    return bool(np.allclose(m.sum(axis=1), 1.0, atol=1e-4))


#: the ports' names as the interface writes them
PORT_NAMES = {"SII": "S II", "Ha": "Hα", "OIII": "O III"}


#: the reasons a combination waits (the interface translates the templates, filling the names in)
TO_WIRE = "{names} to wire"
#: why a block that reads the plate solution after a combination has none: no input of the combination has one
NO_PLATE_SOLUTION = "the combination's inputs bring no plate solution"
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


def equalise(ch: np.ndarray, ref: np.ndarray, valid: np.ndarray) -> tuple[float, float]:
    """(a, b): the equalisation of a channel to a reference (module docstring): ``a * ch + b`` has the reference's
    median and its median absolute deviation, both measured on the covered pixels: a = mad(ref) / mad(ch),
    b = median(ref) - a * median(ch)."""
    m = valid & np.isfinite(ref) & np.isfinite(ch)
    if int(m.sum()) < 2:
        raise StageError("the channels cannot be equalised: fewer than 2 pixels where both have data")
    x, y = ch[m].astype(np.float64), ref[m].astype(np.float64)
    mx, my = float(np.median(x)), float(np.median(y))
    sx, sy = float(np.median(np.abs(x - mx))), float(np.median(np.abs(y - my)))
    if sx <= 0 or sy <= 0:
        raise StageError("the channels cannot be equalised: a channel is flat (its values have no spread)")
    a = sy / sx
    return a, my - a * mx


def signal_gain(ch: np.ndarray, ref: np.ndarray, valid: np.ndarray, noise_ref: float, full: float = 1.0) -> float | None:
    """The gain that fits a channel's signal onto the reference's (both with their sky at 0): the slope of the least
    squares fit ``ref = a * ch + b`` (``linear_match``) on the pixels where the reference is above its sky by
    SIGNAL_SIGMA times its sky noise (and under SIGNAL_HIGH of its full scale). None when no such pixels can be
    fitted or the fit is not positive: the channel does not show the reference's signal."""
    try:
        a, _b, _n = linear_match(ch, ref, valid, SIGNAL_SIGMA * noise_ref, SIGNAL_HIGH * full)
    except StageError:
        return None
    return a if np.isfinite(a) and a > 0 else None


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


def neutral_stars(chans: dict[str, np.ndarray], roles: tuple[str, ...]) -> np.ndarray:
    """(3, H, W): the stars alone of narrow-band filters made neutral (module docstring): the plain mean of the
    balanced channels of ``roles``, the same in red, green and blue."""
    grey = (sum(chans[r] for r in roles) / np.float32(len(roles))).astype(np.float32)
    return np.repeat(grey[None], 3, axis=0)


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


def _lum(r: np.ndarray) -> np.ndarray:
    """The luminance of colour planes (3, ...), summed pixel by pixel in a fixed order: the same value whatever the shape
    of the array (a matrix product rounds differently on the whole image and on a subset of its pixels)."""
    return LUM[0] * r[0] + LUM[1] * r[1] + LUM[2] * r[2]


def gamut_gain(ratio: np.ndarray, L: np.ndarray, gain: float) -> np.ndarray:
    """The colour ratios (3, H, W) with their departure from grey scaled by ``gain`` (the luminance kept), the scale
    lowered, pixel by pixel and down to 1 at the least, where the brightest channel would be shown above
    ``DISPLAY_CAP`` in Post's default display: a bright pixel (the core of a nebula) has little room left, and more
    colour there flattens it against the top of the display. This is gamut compression at constant lightness (the
    chroma scaled down to what the display can show: Morovic and Luo, "The fundamentals of gamut mapping: a survey",
    J. Imaging Sci. Technol. 45(3), 2001, "chroma clipping at constant lightness"; Ottosson 2021, "Gamut clipping in
    Oklab", the same rule in a perceptual space). The display is its midtones stretch put at the Post's rule: the
    median shown at 0.25, white at the 99.99th percentile of the brightest channel at gain 1."""
    def scaled(g):
        r = np.maximum(1 + g * (ratio - 1), 0)
        return r / np.maximum(_lum(r), 1e-6)[None]
    full = scaled(gain)
    lit = L > 0
    if lit.sum() < 1000:
        return full
    top = ratio.max(0) * L
    white = float(np.percentile(top[lit], 99.99))
    med = float(np.median(L[lit])) / max(white, 1e-12)
    if not 0 < med < 1 or white <= 0:
        return full
    m = float(((0.25 - 1) * med) / ((2 * 0.25 - 1) * med - 0.25))      # the mid point that shows the median at 0.25
    cap = white * float(((1 - m) - 1) * DISPLAY_CAP / ((2 * (1 - m) - 1) * DISPLAY_CAP - (1 - m)))   # its inverse
    cap = np.maximum(cap, top)                 # a pixel already above it keeps its own colour
    out_max = full.max(0) * L
    over = out_max > cap
    if not over.any():
        return full
    # the scale that puts the brightest channel at the cap: found by bisection on the (monotonic) channel, over the
    # pixels above it only (a nebula's bright core: a few percent of the image; the whole image took 2 s on a laptop)
    rs, Ls, cs = ratio[:, over], L[over], cap[over]
    lo, hi = np.ones_like(Ls), np.full_like(Ls, float(gain))
    for _ in range(8):
        mid = 0.5 * (lo + hi)
        r = np.maximum(1 + mid[None] * (rs - 1), 0)
        r = r / np.maximum(_lum(r), 1e-6)[None]
        ok = r.max(0) * Ls <= cs
        lo = np.where(ok, mid, lo)
        hi = np.where(ok, hi, mid)
    r = np.maximum(1 + lo[None] * (rs - 1), 0)
    full[:, over] = r / np.maximum(_lum(r), 1e-6)[None]
    return full


def compose(chans: dict[str, np.ndarray], roles: tuple[str, ...], mat, noise: dict[str, float],
            sigma: float | None = None, sky: np.ndarray | None = None, equal: bool = False,
            gain: float = 1.0, light: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
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
    noise's formula would miss; without sky pixels, the formula. ``light``: the lightness given (the L of an LRGB)
    instead of the channels' mean; its noise is then not known, so ``sigma`` must be given. Returns (the image, notes
    for the log)."""
    rgb = mix(chans, roles, mat)
    if light is None:
        L, noise_l = lightness(chans, noise, equal)
    elif sigma is None:
        raise ValueError("a lightness given needs the colour's smoothing radius")
    else:
        L, noise_l = light.astype(np.float32), float("nan")
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
    if gain != 1.0:                       # the colour strength: the departure from grey scaled, the luminance kept
        ratio = gamut_gain(ratio, L, gain)
    return (ratio * L[None]).astype(np.float32), {"sigma": sig, "noise_l": noise_l, "floor": floor}


#: the continuum's scale needs this many measured stars at the least
MIN_STARS = 5
#: the detection threshold of the stars (in units of the sky's noise) and their roundness (minor / major axis)
STAR_SIGMA, STAR_ROUND = 8.0, 0.6
#: the photometry's circle, in full widths at half maximum of the stars
APERTURE = 1.5


def continuum_scale(line: np.ndarray, cont: np.ndarray, valid: np.ndarray, top_line: float,
                    top_cont: float, say=None) -> tuple[float, int]:
    """(the scale, the stars used) of a continuum subtraction (module docstring): the median, over the field's stars,
    of their flux in the line image over their flux in the continuum image. The stars are found on the continuum
    image (SExtractor's detection, ``sep``, at STAR_SIGMA times the sky's noise; round, unflagged); each is measured
    in both images in a circle of APERTURE times the stars' full width at half maximum (near the best signal to noise
    of aperture photometry, S. B. Howell 1989, PASP 101, 616; the same circle in both images, so the share of a star
    it holds cancels in the ratio), its local sky the mean of an annulus from 1.25 to 2 radii (the nebula under the
    star taken off, which a line image has; narrow, so that the nebula's curvature biases it least: a wider one, 1.5
    to 3 radii, gave 3 % less on a synthetic nebula). Left out: a star with a neighbour within its annulus, one whose brightest pixel is above ``top_*`` in either image (saturated), one under 10 times its
    flux error. ``say(step, n)``, when given, is told when the detection is done (``"detected"``, the objects found) and
    before the photometry (``"measuring"``, the stars kept): the stretches of seconds, or minutes on an image of stars
    alone, where nothing else would move."""
    import sep

    from ..core.imgutil import sep_extract
    bad = ~valid | ~np.isfinite(cont) | ~np.isfinite(line)
    c = np.ascontiguousarray(np.where(bad, 0, cont), np.float32)
    ln = np.ascontiguousarray(np.where(bad, 0, line), np.float32)
    bkg = sep.Background(c, mask=bad, bw=64, bh=64)
    o = sep_extract(c - bkg.back(), STAR_SIGMA, err=bkg.globalrms, minarea=4, mask=bad, deblend_cont=0.005)
    if say:
        say("detected", len(o))
    o = o[(o["flag"] == 0) & (o["b"] >= STAR_ROUND * o["a"])]
    if len(o) < MIN_STARS:
        raise StageError(f"the continuum's scale needs stars: {len(o)} found on the continuum image (at least "
                         f"{MIN_STARS}); write the scale in the recipe")
    # the FWHM of a Gaussian of the objects' second moments
    r = max(float(APERTURE * 2.355 * np.median(np.sqrt(o["a"] * o["b"]))), 2.0)
    x, y = o["x"].astype(np.float64), o["y"].astype(np.float64)
    H, W = c.shape
    inside = (x > 2 * r) & (x < W - 1 - 2 * r) & (y > 2 * r) & (y < H - 1 - 2 * r)
    from scipy.spatial import cKDTree
    near = cKDTree(np.c_[x, y]).query_ball_point(np.c_[x, y], r=2 * r, return_length=True)
    keep = inside & (near <= 1)
    ri = int(np.ceil(r))
    for n in np.nonzero(keep)[0]:
        yi, xi = int(round(y[n])), int(round(x[n]))
        box = (slice(yi - ri, yi + ri + 1), slice(xi - ri, xi + ri + 1))
        if bad[box].any() or cont[box].max() > top_cont or line[box].max() > top_line:
            keep[n] = False
    x, y = x[keep], y[keep]
    if len(x) < MIN_STARS:
        raise StageError(f"the continuum's scale needs stars: {len(x)} isolated and unsaturated (at least "
                         f"{MIN_STARS}); write the scale in the recipe")
    if say:
        say("measuring", len(x))
    rms_l = float(sep.Background(ln, mask=bad, bw=64, bh=64).globalrms)
    fc, ec, _ = sep.sum_circle(c, x, y, r, err=float(bkg.globalrms), bkgann=(1.25 * r, 2 * r))
    fl, el, _ = sep.sum_circle(ln, x, y, r, err=rms_l, bkgann=(1.25 * r, 2 * r))
    ok = (fc > 10 * ec) & (fl > 10 * el)
    if ok.sum() < MIN_STARS:
        raise StageError(f"the continuum's scale needs stars: {int(ok.sum())} measured in both images (at least "
                         f"{MIN_STARS}); write the scale in the recipe")
    return float(np.median(fl[ok] / fc[ok])), int(ok.sum())


def recipe_value(p, spec: dict, name: str):
    """The value of a recipe's parameter: the block's own setting of that name (the four types' parameters), else the
    user's value in ``values`` (checked against the recipe), else the recipe's default."""
    q = spec["parameters"].get(name)
    if q is None:
        raise StageError(f"the recipe has no parameter {name!r}")
    if p is not None and name in getattr(type(p), "model_fields", {}) and name not in RecipeParams.model_fields:
        return getattr(p, name)
    v = ((getattr(p, "values", None) or {}) if p is not None else {}).get(name, q["default"])
    if "choices" in q:
        if v not in q["choices"]:
            raise StageError(f"{name}: {v!r} is not one of {', '.join(q['choices'])}")
        return v
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise StageError(f"{name}: {v!r} is not a number")
    if v < q.get("min", -np.inf) or v > q.get("max", np.inf):
        raise StageError(f"{name}: {v:g} is outside {q.get('min', -np.inf):g} to {q.get('max', np.inf):g}")
    return float(v)


def output_matrix(p, spec: dict) -> tuple[tuple[str, ...], list[list[float]]]:
    """(the channels, the matrix, rows R, G, B) of a recipe's ``output``: the hues' palette (``hue_palette``) or its
    matrix, with the block's settings (``recipe_value``)."""
    o = spec["output"]

    def get(v):
        return recipe_value(p, spec, v[1:]) if isinstance(v, str) and v.startswith("$") else v
    roles = tuple(o["channels"])
    if o["hues"] is not None:
        hues = tuple(float(get(h)) for h in o["hues"])
        try:
            return roles, hue_palette(hues)
        except ValueError as e:
            raise StageError(f"{e}: choose hues around the colour wheel") from None
    return roles, [[float(get(x)) for x in row] for row in o["matrix"]]


def recipe_output(ctx, p, spec: dict, chans: dict[str, np.ndarray], noise: dict[str, float],
                  valid: np.ndarray) -> np.ndarray:
    """(3, H, W): a recipe's ``output`` (stages/recipes.py) on the balanced channels: a hue per channel
    (``hue_palette``) or a matrix (``output_matrix``); mixed as they are, or composed as lightness and colour
    (``compose``)."""
    o = spec["output"]

    def get(v):
        return recipe_value(p, spec, v[1:]) if isinstance(v, str) and v.startswith("$") else v
    roles, mat = output_matrix(p, spec)
    if o["hues"] is not None:
        hues = tuple(float(get(h)) for h in o["hues"])
    if o["lightness"] is None:
        return mix(chans, roles, mat)
    smoothing, gain = get(o["smoothing"]), float(get(o["strength"]))
    if o["lightness"] == "channels":
        equal = get(o["weighting"]) == "equal"
        sigma = None if smoothing == "auto" else float(get(o["radius"]))
        rgb, notes = compose(chans, roles, mat, noise, sigma, sky=sky_mask(chans, valid), equal=equal, gain=gain)
        colour = ("the hues " + ", ".join(f"{r} {h:g}" for r, h in zip(roles, hues)) if o["hues"] is not None
                  else "the matrix")
        ctx.log(f"lightness from the channels {'equally' if equal else 'weighted by their noise'} ("
                + ", ".join(f"{r} {(notes['noise_l'] / max(noise[r], 1e-12)) ** 2:.2f}" for r in chans)
                + f"), colour from {colour}" + (f", smoothed over {notes['sigma']:.1f} px" if notes["sigma"] else ""))
        ctx.metric("chroma_sigma", round(notes["sigma"], 2))
        return rgb
    if smoothing == "auto":
        raise StageError("a lightness taken from an input needs a smoothing radius (its noise is not measured)")
    light = o["lightness"]
    rgb, notes = compose(chans, roles, mat, noise, sigma=float(get(o["radius"])), sky=sky_mask(chans, valid),
                         light=chans[light], gain=gain)
    ctx.log(f"lightness from {light}, colour from {', '.join(roles[:-1])} and {roles[-1]} (grey under "
            f"{notes['floor']:.3g}, the colour's sky noise floor)")
    return rgb


# ---------------------------------------------------------------------------------------------- the blocks

class CombinationParams(StageParams):
    #: the inputs, set by the project (not keyed: their workflows' keys are): [{"role" (the port), "channel", "pid",
    #: "image", "dir", "wcs", "grid"}], the first one the reference grid
    inputs: list[dict] = P([], label="Inputs", level="internal",
                           help="The images of the workflows combined, the port each one feeds, and its channel.")
    #: the user's weight of each channel ({port: weight}, 1 by default): a channel too faint for the balance (its noise
    #: raised with it) is lowered by hand
    #: set by the project (not keyed: the keys of the stages it holds are): {stage of the reference row's Astrometry
    #: block: its folder}, the plate solution handed on with the combination's image (empty: the inputs have none)
    solution: dict = P({}, label="Plate solution", level="internal",
                       help="The plate solution of the reference row, handed on with the combination's image.")
    weights: dict = P({}, label="Channel weights", level="internal",
                      help="Each channel's weight, after the balance (1: as balanced).")
    #: set by the project (not keyed here: the combination's source part of the key holds it, stages/recipes.py): the
    #: recipe copied into the project; empty for a project saved before recipes (its type's frozen recipe)
    recipe: dict = P({}, label="Recipe", level="internal",
                     help="The recipe of the combination, as copied into the project.")


def _stars_setting():
    return P(STARS[0], label="Stars", choices=list(STARS),
             help="What a combination of narrow-band filters makes of the stars alone (when every input reads its "
                  "stars): Neutral, the filters' plain mean, white stars; Palette, the stars through the filters' "
                  "hues, coloured like the nebula.")


class NarrowbandParams(CombinationParams):
    stars: str = _stars_setting()
    #: its default is no part of a key (core/store): a project saved before the setting keeps its results, said of an
    #: earlier version
    balance: str = P(BALANCES[0], label="Balance", choices=list(BALANCES),
                     help="How each filter is brought to the H-alpha before the hues. On sky: the filter's median "
                          "and spread brought to the H-alpha's, whatever its signal. On signal: the gain that fits the "
                          "filter's nebula onto the H-alpha's, where the H-alpha is 3 sigma above its sky, but never "
                          "more than the gain that makes the filter's sky noise equal to the H-alpha's: a weak filter "
                          "is not raised into its noise. The stars alone are always balanced on the sky.")

    @field_validator("balance")
    @classmethod
    def _known_balance(cls, v: str) -> str:
        if v not in BALANCES:
            raise ValueError(f"balance {v!r} is not one of {BALANCES}")
        return v

    @field_validator("stars")
    @classmethod
    def _known_stars(cls, v: str) -> str:
        if v not in STARS:
            raise ValueError(f"stars {v!r} is not one of {STARS}")
        return v


class RecipeParams(NarrowbandParams):
    values: dict = P({}, label="Recipe settings",
                     help="The recipe's parameters the user set, by name; the others at the recipe's defaults.")


class Combination(Stage):
    """What every combination shares: the recipe's named inputs read, checked to overlap, on the first one's grid; the
    recipe's steps (the skies neutralised, the linear match, the continuum subtraction); the weights; the recipe's
    output; the outputs, the previews and the metrics. The four types before recipes are subclasses (their stage,
    ``kind``, ``ports``, parameters), which run their type's recipe (``recipes.LEGACY``) or its copy in the project;
    ``CombineRecipe`` runs any other recipe."""

    kind: ClassVar[str]
    ports: ClassVar[tuple[str, ...]]
    narrowband: ClassVar[bool] = False
    #: the port every other channel is linearly matched to (module docstring, 3)
    match_reference: ClassVar[str] = MATCH_REFERENCE
    #: the ports whose stacks' saturation map the result hands on (module docstring, the broadband types)
    saturation_ports: ClassVar[tuple[str, ...]] = ()
    Params = CombinationParams

    def spec(self, p) -> dict:
        """The recipe the block runs: the project's copy (``p.recipe``), else its type's frozen one."""
        r = getattr(p, "recipe", None) if p is not None else None
        return r or recipes.legacy_spec(self.kind)

    def title_of(self, spec: dict) -> str:
        return self.title

    def combine(self, ctx: Context, p, chans: dict[str, np.ndarray], noise: dict[str, float],
                valid: np.ndarray) -> np.ndarray:
        """(3, H, W): the balanced channels by port (each sky at 0) made into red, green and blue by the recipe's
        output (before the first channel's level is given back). ``noise``: each balanced channel's sky noise."""
        return recipe_output(ctx, p, self.spec(p), chans, noise, valid)

    @staticmethod
    def stars_neutral(p, spec: dict, inputs: list[dict]) -> bool:
        """True when the combination gives neutral stars (module docstring): a narrow-band recipe, every input
        reading its stars alone, and the setting Stars left at Neutral."""
        return bool(spec["narrowband"]) and getattr(p, "stars", STARS[1]) == STARS[0] and             bool(inputs) and all(i.get("image") == "stars" for i in inputs)

    @staticmethod
    def balance_of(p, inputs: list[dict]) -> str:
        """How the recipe's ``equalise`` steps balance the channels: the block's setting Balance (the types without
        it: on the sky); a combination of the stars alone is always balanced on the sky (the stars have no nebula to
        fit, and theirs is untouched by the setting)."""
        if inputs and all(i.get("image") == "stars" for i in inputs):
            return BALANCES[0]
        return getattr(p, "balance", BALANCES[0])

    def saturation_of(self, inputs: list[dict], shape: tuple[int, int],
                      ports: tuple[str, ...] | None = None) -> np.ndarray | None:
        """The saturation map handed on (module docstring): the stacks' maps of ``ports`` (the recipe's
        ``saturation``; default ``saturation_ports``; each input's ``saturation``: its Stacking's or Align on's
        saturation.npy, layer 0 near saturation, 1 saturated) merged, a pixel flagged in any of them flagged; None when
        a port has none (the Color block then estimates it)."""
        maps = []
        for role in self.saturation_ports if ports is None else ports:
            f = next((i.get("saturation") or "" for i in inputs if i["role"] == role), "")
            if not f or not Path(f).is_file():
                return None
            m = np.load(f)
            if m.shape[-2:] != tuple(shape):
                return None
            maps.append(m)
        return np.maximum.reduce(maps) if maps else None

    def run(self, ctx: Context, p: CombinationParams) -> None:
        spec = self.spec(p)
        ports = recipes.roles(spec)
        title = self.title_of(spec)
        if not p.inputs:
            raise StageError("no workflow to combine: choose the channels of the combination")
        missing = [r for r in ports if r not in {i["role"] for i in p.inputs}]
        if missing:
            raise StageError(f"{title} needs the {', '.join(missing)} channel(s)")
        ref: dict = {}
        wcs_ref = None
        chans: dict[str, np.ndarray] = {}
        valid_all = None
        shapes: dict[str, list[int]] = {}
        inputs = [i for i in p.inputs if i["role"] in ports]
        bad = grid_problem(inputs, ports)
        if bad:
            raise StageError(bad)
        for n, inp in enumerate(inputs):
            d = Path(inp["dir"])
            image = inp.get("image", "output")
            name = IMAGES[image][1]
            if not (d / name).is_file():
                what = "starless image" if image == "starless" else "stars"
                raise StageError(f"the {what} of the {inp['role']} channel is not computed: run its Stars/background"
                                 if image != "output" else
                                 f"the output of the {inp['role']} channel is not computed")
            valid = np.load(d / "valid.npy")
            ch = channel_of(np.asarray(np.load(d / name, mmap_mode="r")),
                            inp.get("channel", recipes.default_channel(spec, inp["role"])))
            wcs_f = Path(inp.get("wcs") or "")
            if not ref:
                ref = {"shape": ch.shape, "dir": d}
            if wcs_ref is None and wcs_f.is_file():
                # the inputs are on one grid: the plate solution of the first input that has one (the reference row's)
                from ..core.imgutil import read_wcs  # imgutil loads OpenCV: not when the engine's API is imported
                wcs_ref = read_wcs(wcs_f)
            shapes[inp["role"]] = list(ch.shape)
            valid_all = valid if valid_all is None else (valid_all & valid)
            chans[inp["role"]] = ch
            ctx.progress(0.6 * (n + 1) / len(inputs), f"channel {inp['role']}")
        # a safety net: the grids said alike, the sizes did not
        bad = grid_problem([{"role": r, "grid": s} for r, s in shapes.items()], ports)
        if bad:
            raise StageError(bad)
        if not valid_all.any():
            raise StageError("the channels do not overlap: no pixel where every one has data")
        tops = {r: 1.0 for r in chans}
        how, notes, maps = self.balance_of(p, inputs), {}, {}
        skies = self.steps(ctx, spec, inputs, chans, valid_all, tops, how, notes, maps)
        noise = {r: float(n) for r, (_lv, n) in skies.items()}         # each balanced filter's sky noise, unweighted
        rgb = self.mix_balanced(ctx, p, spec, inputs, chans, skies, tops, valid_all)
        H, W = ref["shape"]
        ctx.save("drz_full.npy", rgb.astype(np.float32))
        ctx.save("drz_weight.npy", np.repeat(valid_all[None].astype(np.float32), 3, axis=0))
        sat = self.saturation_of(inputs, (H, W), tuple(spec["output"]["saturation"]))
        if sat is not None:
            from .stack import SATURATION
            ctx.save(SATURATION, sat)
        meta = json.loads((ref["dir"] / "prep_simple.json").read_text(encoding="utf-8")) \
            if (ref["dir"] / "prep_simple.json").is_file() else {}
        ctx.save_json("drz_meta.json", {"scale": float(meta.get("drizzle_scale", 1.0)), "source": "fusion",
                                        "kind": spec["id"], "narrowband": spec["narrowband"], "width": W, "height": H,
                                        "sky_neutral": True, "channels": {r: round(t, 4) for r, t in tops.items()},
                                        # each balanced channel as the linear map of the channels read (``steps``),
                                        # before the weights: Post's combination layer mixes them again from it
                                        "balance": {"how": how,
                                                    "maps": {r: {"coef": c, "offset": o} for r, (c, o) in maps.items()},
                                                    "gains": {r: [float(g), why] for r, (g, why) in notes.items()},
                                                    "noise": noise}})
        # the reference row's plate solution: the combination is on its grid and crops nothing, so it is the same
        # solution; the files of its Astrometry stages are handed on as Align on does, the blocks after read them
        from .align_on import AlignOn
        AlignOn.hand_on_solution(ctx, p.solution or {})
        if wcs_ref is not None:
            wcs_ref.to_fits().writeto(ctx.out("wcs.fits"), overwrite=True)
        ctx.metric("channels", sorted(chans))
        ctx.log(f"{title} from {', '.join(chans)}; channel scales " + ", ".join(f"{r} {t:.4g}" for r, t in tops.items()))
        v = np.moveaxis(rgb, 0, -1)
        lo, hi = np.percentile(v[valid_all], 1), np.percentile(v[valid_all], 99.7)
        ctx.preview("fusion", np.clip((v - lo) / max(hi - lo, 1e-6), 0, 1) ** 0.5)

    def mix_balanced(self, ctx: Context, p, spec: dict, inputs: list[dict], chans: dict[str, np.ndarray],
                     skies: dict[str, tuple[float, float]], tops: dict[str, float], valid: np.ndarray) -> np.ndarray:
        """(3, H, W): the channels the recipe's steps balanced (``steps``), each scaled by the user's weight, made into
        red, green and blue (the recipe's output, or the neutral stars), the pixels outside ``valid`` at 0. ``chans``,
        ``skies`` and ``tops`` are scaled in place by the weights. The run's own end, and the editor's preview
        (client/combineedit.py) on its reduced copy."""
        level = 1.0
        for role in list(chans):
            wgt = float(p.weights.get(role, 1.0))
            if wgt != 1.0:                           # the user's weight: the balance's divisor scaled with it
                chans[role] *= np.float32(wgt)
                tops[role] /= wgt
                skies[role] = (skies[role][0], skies[role][1] * wgt)
                ctx.log(f"{role}: weight {wgt:g}")
        # each balanced channel's sky noise: the sky's, scaled as the channel was (the weight included)
        noise = {r: skies[r][1] for r in chans}
        if self.stars_neutral(p, spec, inputs):
            mixed = tuple(spec["output"]["channels"])
            rgb = neutral_stars(chans, mixed)
            ctx.log(f"the stars alone: neutral, the plain mean of {', '.join(mixed)} (no hue)")
        else:
            rgb = self.combine(ctx, p, chans, noise, valid) * np.float32(level or 1.0)
        rgb[:, ~valid] = 0
        return rgb

    def steps(self, ctx: Context, spec: dict, inputs: list[dict], chans: dict[str, np.ndarray], valid: np.ndarray,
              tops: dict[str, float], balance: str = BALANCES[0],
              notes: dict | None = None, maps: dict | None = None) -> dict[str, tuple[float, float]]:
        """The recipe's steps on the channels, in place (``tops``: each channel's scale, 1 over the factor it was
        multiplied by). ``balance``: what an ``equalise`` step does (``BALANCES``). ``notes``, when given, receives
        each channel's gain and why ({role: (gain, "signal" | "noise" | "sky" | "match")}: the editor's line under a
        filter). ``maps``, when given, receives each balanced channel as a linear map of the channels read
        ({role: ({channel: coefficient}, offset)}): every step is linear in the channels (a level taken off, a gain
        and an offset, a channel minus a scaled other), so the balanced channels are these maps of the inputs (Post's
        combination layer mixes them again, client/postfusion.py). Returns the last neutralisation's {role: (level,
        noise)}."""
        notes = {} if notes is None else notes
        maps = {} if maps is None else maps
        maps.update({r: ({r: 1.0}, 0.0) for r in chans})

        def affine(role, a, b=0.0):                   # chans[role] became a * chans[role] + b
            c, o = maps[role]
            maps[role] = ({k: a * v for k, v in c.items()}, a * o + b)

        def off(levels):                              # each channel lost its level (``neutralise_sky``)
            for r, (lv, _n) in levels.items():
                affine(r, 1.0, -float(np.float32(lv)))
            return levels
        skies: dict[str, tuple[float, float]] = {}
        first = True
        for step in spec["steps"]:
            op = step["op"]
            if op == "neutralise_sky":
                skies = off(neutralise_sky(chans, valid))
                if first:
                    ctx.log("sky removed (level, noise): " + ", ".join(f"{r} {a:.4g}, {b:.3g}" for r, (a, b) in skies.items()))
                first = False
            elif op == "linear_match":
                # Siril's linear match of the channels to the reference; the result keeps the reference's scale
                ref_role = step["reference"]
                ref_ch = chans[ref_role]
                full = float(next(i for i in inputs if i["role"] == ref_role).get("full_scale") or 1.0)
                low, high = step["low"] * full, step["high"] * full
                for role in list(chans):
                    if role == ref_role or (step["channels"] and role not in step["channels"]):
                        continue
                    a, b, used = linear_match(chans[role], ref_ch, valid, low, high)
                    if not a > 0:
                        raise StageError(f"the {role} channel does not follow the {ref_role} one (a negative fit): "
                                         "check that they show the same field")
                    chans[role] = (chans[role] * np.float32(a) + np.float32(b)).astype(np.float32)
                    affine(role, float(np.float32(a)), float(np.float32(b)))
                    tops[role] = 1.0 / a
                    notes[role] = (a, "match")
                    ctx.log(f"{role} matched to {ref_role} on its pixels between {low:.3g} and "
                            f"{high:.5g}: {a:.4g} x + {b:.3g} ({used} pixels)")
                skies = off(neutralise_sky(chans, valid))          # the fits' offsets taken off: the sky at 0 again
            elif op == "equalise":
                # each channel brought to the reference's median and spread; the result keeps the reference's scale
                ref_role = step["reference"]
                if balance == BALANCES[1]:
                    # on the signal: the smaller of the gain that fits the channel's signal onto the reference's and
                    # of the one that makes their sky noises equal (the sky's noise, measured on the common sky by
                    # the neutralisation: ``sky_level``); no offset, the skies are at 0
                    skies = skies or off(neutralise_sky(chans, valid))
                    noise_ref = skies[ref_role][1]
                    full = float(next(i for i in inputs if i["role"] == ref_role).get("full_scale") or 1.0)
                    for role in list(chans):
                        if role == ref_role or (step["channels"] and role not in step["channels"]):
                            continue
                        by_noise = noise_ref / skies[role][1] if skies[role][1] > 0 and noise_ref > 0 else None
                        by_signal = signal_gain(chans[role], chans[ref_role], valid, noise_ref, full)
                        if by_noise is None and by_signal is None:
                            raise StageError(f"the {role} channel cannot be balanced on the {ref_role} one: no signal "
                                             "to fit and no sky noise to measure")
                        a = min(g for g in (by_signal, by_noise) if g is not None)
                        chans[role] = (chans[role] * np.float32(a)).astype(np.float32)
                        affine(role, float(np.float32(a)))
                        tops[role] = 1.0 / a
                        notes[role] = (a, "signal" if by_signal is not None and a == by_signal else "noise")
                        ctx.log(f"{role} balanced on the signal of {ref_role}: x {a:.4g} ("
                                + (f"its signal asks for {by_signal:.4g}" if by_signal is not None else
                                   "no signal to fit")
                                + (f", its sky noise allows {by_noise:.4g})" if by_noise is not None else ")"))
                    skies = off(neutralise_sky(chans, valid))
                    continue
                for role in list(chans):
                    if role == ref_role or (step["channels"] and role not in step["channels"]):
                        continue
                    a, b = equalise(chans[role], chans[ref_role], valid)
                    chans[role] = (chans[role] * np.float32(a) + np.float32(b)).astype(np.float32)
                    affine(role, float(np.float32(a)), float(np.float32(b)))
                    tops[role] = 1.0 / a
                    notes[role] = (a, "sky")
                    ctx.log(f"{role} equalised to {ref_role} (its median and its spread): {a:.4g} x + {b:.3g}")
                skies = off(neutralise_sky(chans, valid))          # the offsets taken off: the sky at 0 again
            elif op == "continuum_subtract":
                line, cont = step["line"], step["continuum"]
                full = {i["role"]: float(i.get("full_scale") or 1.0) for i in inputs}
                if step["scale"] == "stars":
                    # the detection of the stars takes seconds, minutes on an image of stars alone (no sky: its
                    # threshold falls to 0): the bar and the log say what is going on (issue #185)
                    ctx.progress(0.6, f"{line}: measuring the continuum on the stars", force=True)
                    ctx.log(f"{line}: detecting the stars of {cont} to measure the continuum")

                    def say(what, n, line=line):
                        ctx.progress(0.8 if what == "detected" else 0.85,
                                     f"{line}: {n} objects found" if what == "detected"
                                     else f"{line}: measuring the continuum on {n} stars", force=True)
                    mu, n = continuum_scale(chans[line], chans[cont], valid, MATCH_HIGH * full[line],
                                            MATCH_HIGH * full[cont], say)
                    ctx.log(f"{line}'s continuum: {cont} times {mu:.4g}, measured on {n} stars")
                    ctx.metric("continuum_stars", n)
                else:
                    mu = float(step["scale"])
                    ctx.log(f"{line}'s continuum: {cont} times {mu:.4g}, as the recipe says")
                if not mu > 0:
                    raise StageError(f"the {line} channel's continuum scale is not above 0: check that {line} and "
                                     f"{cont} show the same field")
                chans[line] = (chans[line] / np.float32(mu) - chans[cont]).astype(np.float32)
                cl, ol = maps[line]
                cc, oc = maps[cont]
                k = 1.0 / float(np.float32(mu))
                maps[line] = ({r: k * cl.get(r, 0.0) - cc.get(r, 0.0) for r in set(cl) | set(cc)}, k * ol - oc)
                tops[line] *= mu
                ctx.metric("continuum_scale", round(mu, 6))
                skies = off(neutralise_sky(chans, valid))
            else:                                   # recipes.parse admits no other
                raise StageError(f"unknown operation {op!r}")
        return skies


class NarrowbandCombination(Combination):
    """The narrow-band types (S-H-O, H-O): each filter equalised to the H-alpha, given a hue, the layers mixed as they
    are (module docstring, 3-5), their hue parameters ``hue_<port>``."""

    narrowband = True

    def hues(self, p) -> tuple[float, ...]:
        return tuple(float(getattr(p, f"hue_{r.lower()}")) for r in self.ports)


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
    description = ("Combines the S II, H-alpha and O III filters into one colour image: the S II and the O III "
                   "equalised to the H-alpha, then each filter a hue.")
    Params = SHOParams
    # 8: each filter equalised to the H-alpha (its median and its spread) and the hues mixed as they are, instead of
    # the linear match, the lightness weighted by the noise and the colour's strength of 1.9 (module docstring, 5)
    # 9: the stars alone come out neutral (the setting Stars), no longer through the hues
    # 10: the filters balanced on the signal by default (the setting Balance; On sky: the equalisation as before)
    # 11: On sky the default again (the default On signal of 10 was tuned on two test images); On signal a choice
    version = "11"
    key_label = "9"     # the keys' label, frozen (core/stage.Stage.key_label): never change it


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
                   "colour image: the O III equalised to the H-alpha, then each filter a hue (H-alpha red, O III "
                   "green and blue).")
    Params = HOParams
    # 8, 9, 10, 11: as the S-H-O block's
    version = "11"
    key_label = "9"     # the keys' label, frozen (core/stage.Stage.key_label): never change it


class CombineRGB(Combination):
    name = "combine_rgb"
    title = "RGB"
    kind = "rgb"
    ports = ("R", "G", "B")
    description = ("Combines the red, green and blue filters into one colour image, each linearly matched to the red; "
                   "the colour calibration after it sets the colour balance.")
    Params = CombinationParams
    match_reference = "R"
    saturation_ports = ("R", "G", "B")
    # 3: Siril's linear match to the red instead of a percentile; the stacks' saturation map handed on
    version = "3"
    key_label = "3"     # the keys' label, frozen (core/stage.Stage.key_label): never change it


class CombineLRGB(Combination):
    name = "combine_lrgb"
    title = "LRGB"
    kind = "lrgb"
    ports = ("L", "R", "G", "B")
    description = ("Combines the red, green and blue filters into the colour, the luminance filter into the detail: "
                   "at every pixel the colour of RGB, the lightness of L.")
    Params = CombinationParams
    match_reference = "L"
    saturation_ports = ("R", "G", "B")
    # 3: R, G and B linearly matched to L (Siril) instead of percentiles; L's lightness at every pixel, the colour grey
    # under its noise floor (before: the colour's own lightness there); the stacks' saturation map handed on
    version = "3"
    key_label = "3"     # the keys' label, frozen (core/stage.Stage.key_label): never change it


class CombineRecipe(Combination):
    """Any recipe other than the four types' frozen ones (stages/recipes.py): the recipe copied into the project runs,
    its parameters in ``values``; the key holds the recipe (core/store.source_part)."""

    name = recipes.RECIPE_STAGE
    title = "Combination"
    kind = ""
    ports = ()
    description = "Combines the inputs as the combination's recipe says (resources/combinations)."
    Params = RecipeParams
    # 2: a narrow-band recipe's stars alone come out neutral (the setting Stars), no longer through its hues
    # 3: a recipe's equalisation balances on the signal by default (the setting Balance)
    # 4: on the sky by default again, as the S-H-O block's 11
    version = "4"
    key_label = "2"     # the keys' label, frozen (core/stage.Stage.key_label): never change it

    def spec(self, p) -> dict:
        if p is None or not p.recipe:
            raise StageError("the combination has no recipe: place it again from the list of combinations")
        try:
            return recipes.check(p.recipe, "the combination's recipe")
        except recipes.RecipeError as e:
            raise StageError(str(e)) from None

    def title_of(self, spec: dict) -> str:
        return spec["name"]


#: the types of combination, in the palette's order; adding a type is adding its class here
COMBINATIONS: tuple[type[Combination], ...] = (CombineSHO, CombineHO, CombineRGB, CombineLRGB)
BY_KIND: dict[str, type[Combination]] = {c.kind: c for c in COMBINATIONS}
#: every combination block: the four types', then the one of the other recipes
ALL: tuple[type[Combination], ...] = (*COMBINATIONS, CombineRecipe)
#: the stages of the combinations (core/workflow: each the first stage of its workflow)
STAGES: tuple[str, ...] = tuple(c.name for c in ALL)


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
    palette becomes the hues; the weights and the stored settings the block still has go to its parameters. Its
    option A (the channels' starless images, a star layer from the channels or another workflow) is
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
    if mine:
        params[cls.name] = mine
    w["params"] = params
    ins = [{"pid": i["pid"], "role": i["role"], "channel": i.get("channel", "L")}
           | ({"image": i["image"]} if i.get("image") in IMAGES else {}) for i in f.get("inputs") or []
           if i.get("role") in cls.ports]
    w["source"] = {"fusion": {"type": kind, "inputs": ins}}
    return w
