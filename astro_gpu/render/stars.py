"""Rendering of the Stars block's two images (stages/star_removal.py): the starless image, the stars alone, and the two
recombined (the Stars and background block of Post).

The usual way (docs/chantiers/etoiles.md): the stars are taken off the linear image, each image is stretched on its
own, then the stars go back on top in Screen mode. The reference program's star recomposition tool stretches the starless image and
the star mask apart before combining them; the reference tool maker advises stretching both and recombining in Screen (Add only while
linear); in the reference image editor the stars layer goes over the starless one in Screen or Linear Dodge (Add). With the stars apart,
stretching them less is the star reduction: no reduction is needed on the recombined image.

* the starless image: the Output block's automatic stretch (render/simple.py) with the whole image's statistics (its
  own, without the stars' grain, would stretch it much harder: recombined, the image would not look like the Output
  block's), moved by the Sky, Contrast and Saturation sliders;
* the stars: black at 0 (their background is the starless image's), white and midtone those of the whole image, the
  midtone moved by ``star_strength``; the smallest stars taken away (``star_small``: the share of the stars removed, the smallest first, the
  star reduction of photo editors' tutorials; each star measured by the grey opening that takes half of its peak, the
  removal an opening by twice that radius, so that the wider light stays; on the stars alone, so the nebula's
  texture is never touched); their size by a round grey erosion (smaller) or dilation (larger), as the reference image editor's
  Minimum and Maximum filters with Preserve roundness (an octagon: a cross and a square in turn, the usual
  approximation of a disc), never a square; smaller on their light wider than the erosion only (the faint stars and
  the narrow light added back as they were: a faint star keeps its shape) and through a star mask (``star_mask``: the stars standing above
  their local background's noise), as those tutorials select the stars first, so that the grain between the stars of
  a dense field is not eroded into filaments; their colour by a saturation around the luminance;
* recombined: Screen 1 - (1 - a)(1 - b), Linear dodge (add) a + b (clipped), or Lighten max(a, b).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pydantic import field_validator

from ..core import devices
from ..core.params import P, check_choice
from . import ops
from .simple import Layers, SimpleRenderEngine, SimpleSettings, continuum_ratio, emission, mtf, mtf_midtone

#: how the stars go back on the starless image (the names of the reference image editor's modes)
BLENDS = ("Screen", "Linear dodge (add)", "Lighten")
#: what a rendering shows: the starless image, the stars alone, the two recombined
PARTS = ("starless", "stars", "combined")


class StarSettings(SimpleSettings):
    """The simple rendering's settings (for the starless image) and the stars' own. Their defaults are the suggested
    rendering (Post's sliders at 2)."""

    star_strength: float = P(0.6, label="Stars", ge=0.2, le=5.0, step=0.05,
                             help="How strongly the stars are stretched: 1 = the stretch of the whole image; less "
                                  "makes them fainter and smaller (the star reduction the separation allows), more "
                                  "brighter. 0.6 by default, with a size of -1: the nebula reads first.")  # the
    # user's choice of 5 November, checked on four of his projects
    star_size: float = P(-1.0, label="Star size", ge=-3.0, le=3.0, step=0.1, unit="px",
                         help="Below 0 the stars are shrunk by this radius (a grey erosion: each pixel takes the "
                              "darkest of its neighbours), above 0 grown (a grey dilation: the brightest).")
    star_small: float = P(0.0, label="Small stars", ge=0.0, le=2.0, step=0.05,
                          help="Share of the stars removed, the smallest first: 0 = none, half-way the smaller half, "
                               "the right end all of them. The size of a star is measured on the stars alone: how wide "
                               "a disc must be to take half of its peak away.")
    star_saturation: float = P(1.2, label="Star color", ge=0.0, le=3.0, step=0.05,
                               help="Colour saturation of the stars around their luminance, after their stretch.")
    blend: str = P("Screen", label="Blend", choices=list(BLENDS),
                   help="How the stars go back on the starless image: Screen (the usual way), Linear dodge (add) or "
                        "Lighten.")

    @field_validator("blend")
    @classmethod
    def _blend(cls, v: str) -> str:
        return check_choice("blend", v, BLENDS)


def is_stars_folder(folder: str | Path) -> bool:
    return (Path(folder) / "prep_stars.json").exists()


class StarsRenderEngine(SimpleRenderEngine):
    """Renders a star_removal folder: ``render(settings, scale, region, part=...)`` with part starless, stars or
    combined. Same interface as ``SimpleRenderEngine`` otherwise."""

    def __init__(self, folder: str | Path, device: str = "cuda:0"):
        self.dir = Path(folder)
        self.dev = torch.device(devices.usable(device))
        self.prep = json.loads((self.dir / "prep_stars.json").read_text(encoding="utf-8"))
        self.M = np.array(self.prep["M"], np.float64)
        self.SC = float(self.prep.get("drizzle_scale", 1.0))
        self.crop = self.prep["crop"]
        self._host = {k: torch.from_numpy(np.load(self.dir / f"{k}.npy")) for k in ("starless", "stars")
                      if (self.dir / f"{k}.npy").exists()}      # a Synthetic stars block: its stars alone
        self.H, self.W = self._host["stars"].shape[-2:]
        self.valid_host = torch.from_numpy(np.load(self.dir / "valid.npy"))
        self.has_raw = False
        self._proxy = {}
        self.last_fit = None

    @torch.no_grad()
    def render(self, s: StarSettings, scale: float = 1.0, region=None, fit: dict | None = None, *,
               draft: bool = False, stop=None, part: str = "combined") -> Layers:
        from ..core.determinism import deterministic
        if part not in PARTS:
            raise ValueError(f"unknown part {part!r} (allowed: {PARTS})")
        if not isinstance(s, StarSettings):
            s = StarSettings(**s.model_dump())
        with deterministic():
            f, reg, box = self._geometry(scale, region)
            linked = bool(self.prep.get("linked", True))
            a = b = None
            if stop is not None:
                stop()                                  # (it raises to abandon: Post's composition superseded)
            if part in ("starless", "combined"):
                x = self._layer("starless", f, box)
                if s.halpha:
                    x = emission(x, self.continuum(), float(s.halpha), f)
                st = self.prep["stats_full"]
                if getattr(s, "stretch", "auto") == "arcsinh":
                    st = self._starless_white(st)
                a = self.tone(x, st, s, linked)
            if part in ("stars", "combined"):
                b = self._star_layer(s, f, box, part, stop)
            y = a if b is None else b if a is None else blend(a, b, s.blend)
            return self._finish(y, f, reg, box, region, {"stretch": "auto", "linked": linked, "part": part})

    @torch.no_grad()
    def linear_stars(self, scale: float = 1.0, region=None) -> Layers:
        """The stars alone as the block computed them: its linear image, no stretch, none of Post's star settings, on
        the grid as ``render`` cuts it (the Stars mask part finds its stars there: client/postengine.find_stars)."""
        f, reg, box = self._geometry(scale, region)
        y = self._layer("stars", f, box).clone()
        return self._finish(y, f, reg, box, region, {"part": "linear"})

    def _starless_white(self, st: dict) -> dict:
        """The statistics the Arcsinh stretch of the starless image reads: the whole image's (``st``: the black point
        and the sky of the automatic stretch), with the white points of the starless image itself: the 99.995th
        percentile of its luminance and of each channel, the statistic of the automatic stretch's white point, measured
        on the image the stretch is applied to (every second pixel, as an image layer's statistics are). The whole
        image's white point is its stars': read by the Arcsinh, a logarithm up to it, stars 50 times the nebula left
        the nebula's top in the middle of the range. Measured once."""
        if st.get("stretched") or "starless" not in self._host:
            return st
        if getattr(self, "_white", None) is None:
            a = self._host["starless"][:, ::2, ::2].numpy()
            v = self.valid_host[::2, ::2].numpy()
            v = v if v.any() else np.ones_like(v)
            L = (0.2126 * a[0] + 0.7152 * a[1] + 0.0722 * a[2])[v]
            self._white = [float(np.percentile(x, 99.995)) for x in (L, *(a[c][v] for c in range(3)))]
        return {**st, "lum": {**st["lum"], "p99995": self._white[0]},
                "channels": [{**ch, "p99995": w} for ch, w in zip(st["channels"], self._white[1:])]}

    def _star_layer(self, s: StarSettings, f: int, box, part: str, stop=None) -> torch.Tensor:
        """The stretched stars of the grid at this scale and box (``star_tone``), kept from one render to the next while
        the settings that make them are the same: the starless image's sliders, a blend or an export (the same
        image at full size) take them as they are instead of making them again, which is the longest step of a
        render. One entry (the last), never mutated by what reads it."""
        # the Sky setting is in the stars' stretch (star_tone: the midtone puts the sky of the whole image there), except
        # for an image already stretched, whose stars are only scaled
        sky = None if self.prep["stats_full"].get("stretched") else s.sky
        key = (f, tuple(box) if box is not None else None, sky, s.star_strength, s.star_small, s.star_size,
               s.star_saturation)
        hit = getattr(self, "_star_cache", None)
        if hit is not None and hit[0] == key:
            y = hit[1]
        else:
            x = self._layer("stars", f, box)
            y = self.star_tone(x, s, f) if stop is None else self.star_tone(x, s, f, stop)
            self._star_cache = (key, y)
        return y.clone() if part == "stars" else y

    def star_fwhm(self) -> float:
        """The image's typical star width at half height (full-resolution pixels), measured once on the stars alone
        (``measure_fwhm``)."""
        if getattr(self, "_fwhm", None) is None:
            self._fwhm = measure_fwhm(self._host["stars"])
        return self._fwhm

    def continuum(self) -> float:
        """The continuum's red-to-green ratio (``continuum_ratio``), measured once on the stars alone."""
        if getattr(self, "_alpha", None) is None:
            self._alpha = continuum_ratio(np.maximum(self._host["stars"].numpy(), 0))
        return self._alpha

    def star_tone(self, x: torch.Tensor, s: StarSettings, f: int, stop=None) -> torch.Tensor:
        """The stars alone stretched: black at 0, the white and the midtone of the whole image (its sky at the
        Sky setting), the midtone divided by ``star_strength``; then their size and their colour. ``stop``: called
        between those steps (at full resolution each takes a second or more); it raises to abandon."""
        def check():
            if stop is not None:
                stop()
        st = self.prep["stats_full"]["lum"]
        if self.prep["stats_full"].get("stretched"):    # an image already stretched (issue #43): its stars as they are
            y = (x * float(s.star_strength)).clamp(0, 1)
        else:
            c0 = st["median"] - 2.8 * st["mad"]
            hi = max(st["p99995"], c0 + 1e-6)
            xm = min(max((st["median"] - c0) / (hi - c0), 1e-6), 1 - 1e-6)
            m = min(max(mtf_midtone(xm, s.sky) / s.star_strength, 1e-6), 0.5)
            y = mtf((x / (hi - c0)).clamp(0, 1), m).clamp(0, 1)
        v = float(s.star_small)
        check()
        if v > 0:                                  # the share of the stars removed, the smallest first (the range's end: all)
            y = fade_small_stars(y, min(1.0, v / 2.0), self.star_fwhm() / f, check)
            check()
        y = resize_stars(y, s.star_size / f, check)
        check()
        L = ops.luminance(y)[None]
        return (L + s.star_saturation * (y - L)).clamp(0, 1)


class CombinedStarsEngine(StarsRenderEngine):
    """The stars alone of a combination of stars (a second combination block of the same type as the starless image's,
    reading the filters' stars): its Output's linear image, stretched as ``star_tone`` stretches a Stars block's stars,
    with the statistics of the whole image the starless combination's Output gives (``stats_folder``), so that the stars
    go onto that image at their own level. ``render(..., part="stars")`` only."""

    def __init__(self, folder: str | Path, stats_folder: str | Path, device: str = "cuda:0"):
        self.dir = Path(folder)
        self.dev = torch.device(devices.usable(device))
        mine = json.loads((self.dir / "prep_simple.json").read_text(encoding="utf-8"))
        theirs = json.loads((Path(stats_folder) / "prep_simple.json").read_text(encoding="utf-8"))
        self.prep = {**mine, "stats_full": theirs["stats"], "linked": theirs.get("linked", True)}
        self.M = np.array(mine["M"], np.float64)
        self.SC = float(mine.get("drizzle_scale", 1.0))
        self.crop = mine["crop"]
        self._host = {"stars": torch.from_numpy(np.load(self.dir / "lin.npy"))}
        self.H, self.W = self._host["stars"].shape[-2:]
        self.valid_host = torch.from_numpy(np.load(self.dir / "valid.npy"))
        self.has_raw = False
        self._proxy = {}
        self.last_fit = None
        self.prep["stats_full"] = self._whole_stats(Path(stats_folder), theirs["stats"])

    def _whole_stats(self, stats_folder: Path, starless: dict) -> dict:
        """The statistics the stars are stretched with: those of the whole image, the starless combination's Output with
        these stars on it, as a Stars block's own stats_full are those of the Output it takes the stars off (its stars
        in them: the white point is the stars' own, p99.995 of the image they are on). The starless combination's alone
        have no star in them, so that white point is its nebula's and every star above it clipped (0.99 % of a
        picture's pixels against 0.14 % of a star row's). ``starless``: those, kept when the two images are not on one
        grid (nothing to add)."""
        from .simple import image_stats
        try:
            base = np.load(stats_folder / "lin.npy", mmap_mode="r")
            valid = np.load(stats_folder / "valid.npy")
            lin = np.asarray(base, np.float32) + self._host["stars"].numpy().astype(np.float32)
        except (OSError, ValueError):
            return starless
        if lin.shape != tuple(base.shape) or valid.shape != lin.shape[1:] or not valid.any():
            return starless
        return image_stats(lin, valid)


#: a typical star width when the stars cannot be measured (full-resolution pixels)
DEFAULT_FWHM = 3.0
#: the second moment per axis of a Gaussian kept above half its peak, in units of its sigma squared
CLIPPED_MOMENT = 2 * (1 - (1 + math.log(2)) / 2)


def measure_fwhm(stars: torch.Tensor, n: int = 300) -> float:
    """The median width at half height of the stars of a stars-alone image (3, H, W, linear): the local maxima of its
    luminance, the middle of them by brightness (neither the noise nor the saturated cores), each measured by the
    second moments of its 9 x 9 neighbourhood above half its peak (FWHM = 2.355 sigma)."""
    L = ops.luminance(stars.float().clamp(min=0))
    peaks = (L == F.max_pool2d(L[None, None], 7, 1, 3)[0, 0]) & (L > 0)
    ys, xs = torch.nonzero(peaks, as_tuple=True)
    keep = (ys >= 4) & (xs >= 4) & (ys < L.shape[0] - 4) & (xs < L.shape[1] - 4)
    ys, xs = ys[keep], xs[keep]
    if ys.numel() < 20:
        return DEFAULT_FWHM
    v = L[ys, xs]
    order = torch.argsort(v)
    mid = order[int(0.6 * len(order)):int(0.95 * len(order))]
    if mid.numel() > n:
        mid = mid[torch.linspace(0, mid.numel() - 1, n).long()]
    yy, xx = torch.meshgrid(torch.arange(-4, 5.0), torch.arange(-4, 5.0), indexing="ij")
    widths = []
    for i in mid.tolist():
        y, x = int(ys[i]), int(xs[i])
        w = L[y - 4:y + 5, x - 4:x + 5]
        w = (w - w.min()).clamp(min=0)
        w = torch.where(w >= 0.5 * w.max(), w, torch.zeros_like(w))
        tot = float(w.sum())
        if tot <= 0:
            continue
        cy, cx = float((w * yy).sum()) / tot, float((w * xx).sum()) / tot
        var = float((w * ((yy - cy) ** 2 + (xx - cx) ** 2)).sum()) / tot / 2
        # above half the peak only: a Gaussian's weighted second moment per axis there is 2 (1 - (1 + ln 2) / 2)
        # sigma^2 = 0.307 sigma^2 (its integral from 0 to the half-height radius)
        widths.append(2.355 * (var / CLIPPED_MOMENT) ** 0.5)
    return float(np.median(widths)) if len(widths) >= 10 else DEFAULT_FWHM


_CROSS = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)
_SQUARE = np.ones((3, 3), np.uint8)


def _step_cpu(y: torch.Tensor, grow: bool, cross: bool) -> torch.Tensor:
    """``_step`` on the processor by OpenCV's dilation / erosion (the borders replicated): the same values as the
    padded max_pool2d, which the processor runs far slower (the star sliders' renders took 15 to 25 s)."""
    a = np.ascontiguousarray(y.numpy())
    k = _CROSS if cross else _SQUARE
    op = cv2.dilate if grow else cv2.erode
    if a.ndim == 2:
        return torch.from_numpy(op(a, k, borderType=cv2.BORDER_REPLICATE))
    out = np.empty_like(a.reshape(-1, *a.shape[-2:]))
    for i, plane in enumerate(a.reshape(-1, *a.shape[-2:])):
        out[i] = op(plane, k, borderType=cv2.BORDER_REPLICATE)
    return torch.from_numpy(out.reshape(a.shape))


def _step(y: torch.Tensor, grow: bool, cross: bool) -> torch.Tensor:
    """One pixel of grey dilation (grow) or erosion: by a cross (the 4 neighbours) or a 3 x 3 square."""
    if y.device.type == "cpu" and y.dtype == torch.float32 and not y.requires_grad:
        return _step_cpu(y, grow, cross)
    z = y if grow else -y
    if cross:
        p = F.pad(z[None], (1, 1, 1, 1), mode="replicate")[0]
        z = torch.stack([z, p[..., :-2, 1:-1], p[..., 2:, 1:-1], p[..., 1:-1, :-2], p[..., 1:-1, 2:]]).amax(0)
    else:
        z = F.max_pool2d(F.pad(z[None], (1, 1, 1, 1), mode="replicate"), 3, 1, 0)[0]
    return z if grow else -z


def _morph(y: torch.Tensor, k: int, grow: bool) -> torch.Tensor:
    """A grey dilation (grow) or erosion of radius k pixels, round: k steps alternating a cross and a square (an
    octagon, the usual approximation of a disc; a square would make the stars square)."""
    for i in range(max(0, k)):
        y = _step(y, grow, cross=(i % 2 == 0))
    return y


#: the geodesic dilations of the reconstruction, per pixel of radius of the largest star kept whole (full resolution)
RECONSTRUCT_STEPS = 48


def _binary(a: torch.Tensor) -> bool:
    """Only 0 and 1 in it (a support, the seeds of a support)."""
    return bool(((a == 0) | (a == 1)).all())


def _reconstruct_binary(m: torch.Tensor, mask: torch.Tensor, steps: int) -> torch.Tensor:
    """``reconstruct`` of 0 / 1 planes on the processor, the same result: the dilations by the 3 x 3 square on 8-bit
    planes (a quarter of the memory), the growth seen in the count of set pixels (it only grows)."""
    shape = m.shape
    cur = np.ascontiguousarray(m.numpy().reshape(-1, *shape[-2:]).astype(np.uint8))
    lim = np.ascontiguousarray(mask.numpy().reshape(-1, *shape[-2:]).astype(np.uint8))
    for i in range(cur.shape[0]):
        a, b = cur[i], lim[i]
        n = cv2.countNonZero(a)
        for _ in range(max(1, steps)):
            a = cv2.min(cv2.dilate(a, _SQUARE, borderType=cv2.BORDER_REPLICATE), b)
            k = cv2.countNonZero(a)
            if k == n:
                break
            n = k
        cur[i] = a
    return torch.from_numpy(cur.astype(np.float32).reshape(shape))


def reconstruct(marker: torch.Tensor, mask: torch.Tensor, steps: int) -> torch.Tensor:
    """Grey reconstruction by dilation (L. Vincent, "Morphological grayscale reconstruction in image analysis",
    IEEE Trans. Image Processing 2, 1993): the marker dilated one pixel at a time, never above the mask, until it
    settles (or ``steps``)."""
    m = torch.minimum(marker, mask)
    if m.device.type == "cpu" and m.dtype == torch.float32 and _binary(m) and _binary(mask):
        return _reconstruct_binary(m, mask, steps)
    for _ in range(max(1, steps)):
        n = torch.minimum(_step(m, True, cross=False), mask)
        if torch.equal(n, m):
            break
        m = n
    return m


#: the level of the stretched stars above which a pixel belongs to a star (their support)
STAR_LEVEL = 0.04


def _go_on(check) -> None:
    if check is not None:
        check()


def star_sizes(y: torch.Tensor, fwhm: float, check=None):
    """The stars of a stars-alone image (3, h, w) and their sizes: the local maxima (7 x 7) of the luminance above
    STAR_LEVEL, and for each the radius of the grey opening that takes half of its peak away, from the openings by
    discs of 1 .. n pixels read at the peak (linearly between two radii). For a Gaussian star of width FWHM that is
    FWHM / 2 (exp(-r^2 / 2 sigma^2) = 1/2 at r = 1.18 sigma), so it is the star's own width, in pixels, and a star
    that no disc up to n takes half of (one wider than 2 n FWHM-wise: n is twice the typical star's FWHM, a star more
    than four times as wide as the field's) is given the size n + 1: the last. Returns (ys, xs, size), the sizes
    in pixels."""
    L = ops.luminance(y)
    pk = (L == F.max_pool2d(L[None, None], 7, 1, 3)[0, 0]) & (L > STAR_LEVEL)
    ys, xs = torch.nonzero(pk, as_tuple=True)
    n = max(2, int(math.ceil(2.0 * fwhm)))
    if ys.numel() == 0:
        return ys, xs, torch.zeros(0)
    peak = L[ys, xs]
    size = torch.full((ys.numel(),), float(n + 1))
    prev = torch.ones_like(peak)                       # the survival at radius 0
    done = torch.zeros(ys.numel(), dtype=torch.bool)
    eroded = L[None]
    for rho in range(1, n + 1):
        eroded = _step(eroded, False, cross=((rho - 1) % 2 == 0))
        opened = _morph(eroded, rho, True)[0]
        cur = opened[ys, xs] / peak
        cross = (~done) & (cur < 0.5)
        frac = ((prev - 0.5) / (prev - cur).clamp(min=1e-9)).clamp(0, 1)
        size = torch.where(cross, rho - 1 + frac, size)
        done |= cross
        prev = cur
        _go_on(check)
        if bool(done.all()):
            break
    return ys, xs, size


def fade_small_stars(y: torch.Tensor, f: float, fwhm: float, check=None) -> torch.Tensor:
    """The share f (0 none .. 1 all) of the field's stars removed, the smallest first, the biggest last, on the stars
    alone. The stars are those of ``star_sizes`` and are ranked by their size, the f of them that are the smallest are
    taken away; the same share of a dense field and of a sparse one.

    The removal of a star of size s, by the grey openings (Meyer 1979, Serra 1982), none of them tied to a connected
    zone (a fixed level joins all the stars of a dense field): the opening by a disc of the largest size taken away, r,
    takes the peaks of those stars, and leaves a plateau of about r where each was (a disc, brown on a coloured field:
    the artefact of a flat structuring element, which a paraboloid, the rolling ball of Sternberg 1983, does not cure:
    its cap stays as a wider dim star); a plateau of radius r cannot hold a disc of 2 r, so the opening by 2 r replaces
    them and keeps what is wider than the stars taken away: a large star's halo. The places of the removed stars are
    spread by r (a star is as wide as the disc that takes it out) and softened by sigma = r / 2.355, the Gaussian width
    of a star whose FWHM is r; a star that is kept has no place in it. All of them (f = 1): nothing is left."""
    if f <= 0:
        return y
    if f >= 1:
        return torch.zeros_like(y)
    ys, xs, size = star_sizes(y, fwhm, check)
    if ys.numel() == 0:
        return y
    gone = torch.argsort(size)[:int(round(f * ys.numel()))]
    if gone.numel() == 0:
        return y
    rho = max(1, int(math.ceil(float(size[gone].max()))))
    mark = torch.zeros(1, *y.shape[-2:])
    mark[0, ys[gone], xs[gone]] = 1.0
    first = torch.minimum(_morph(_morph(y, rho, False), rho, True), y)
    _go_on(check)
    base = torch.minimum(_morph(_morph(y, 2 * rho, False), 2 * rho, True), first)      # the opening by 2 r
    _go_on(check)
    w = ops.gaussian_blur(_morph(mark, 2 * rho, True), rho / 2.355)[0].clamp(0, 1)
    return y - w * (y - base)


def resize_stars(y: torch.Tensor, r: float, check=None) -> torch.Tensor:
    """The stars shrunk (r < 0) or grown (r > 0) by |r| pixels: the two whole radii around |r|, mixed by its
    fraction, so that the size moves smoothly with r."""
    if abs(r) < 1e-3:
        return y
    n, t = int(abs(r)), abs(r) - int(abs(r))
    if r > 0:
        lo = _morph(y, n, True)
        _go_on(check)
        return lo if t < 1e-3 else (1 - t) * lo + t * _morph(y, n + 1, True)
    # only the light wider than the erosion is eroded (``wide_light``), the narrower light (the faint stars, the beads of a
    # chain of touching ones, the top of a wide star) is added back as it was: an erosion of a faint star's light, or a
    # weight that erodes its flanks and spares its core (issue #169's ratio of the opening, 1 on a faint star's flanks and
    # low on its core), thins it into a line, a worm (#174, M27); its light on top of the eroded wide light keeps its shape
    w = wide_light(y, abs(r), check)
    lo = _morph(w, n, False)
    _go_on(check)
    out = lo if t < 1e-3 else (1 - t) * lo + t * _morph(w, n + 1, False)
    _go_on(check)
    # shrunk through a star mask, as the published star reductions do (the reference image editor: the Minimum filter through a
    # selection of the stars; the reference processing suite: MorphologicalTransformation through a star mask): an erosion of the faint
    # background between the stars draws a web of filaments on a dense field (issue #18)
    return y + (out - w) * star_mask(y, n + 1, check)


def wide_light(y: torch.Tensor, r: float, check=None) -> torch.Tensor:
    """The stars' light wider than a disc of r + 1 pixels (3, h, w): the grey opening of each channel by that disc, the
    first term of the image's decomposition into the opening and the top-hat (y = opening + top-hat; Meyer 1979, Serra
    1982: a granulometry). The opening keeps what contains the disc and removes what does not, place by place, and
    never adds light: a wide star stays whole but for its top, a faint star and the narrow beads of a chain of touching
    ones, attached to a wider star or not, are all in the top-hat (a reconstruction would regrow the whole connected
    chain from one wide part). The disc is the reduction's pixels plus the two a star needs to stay sampled across
    (Nyquist-Shannon): what is left of a star is never a one-pixel line."""
    rho = max(1, int(round(r + 1.0)))
    wide = _morph(_morph(y, rho, False), rho, True)
    _go_on(check)
    return torch.minimum(wide, y)


#: a star of the mask stands this many local noise units above its local background (the usual detection threshold)
MASK_SIGMAS = 3.0
#: the scale of the local background and noise (pixels of the grid rendered): wider than the stars, so that a star's
#: own light does not raise the background it is measured against (issue #30: its wings left the mask, a ring)
MASK_SCALE = 24.0
#: the clipping passes that take the stars out of that background (2 noise units, as the background estimators do)
MASK_PASSES, MASK_CLIP = 3, 2.0
#: the softened edge of the mask (pixels of the grid rendered)
MASK_FEATHER = 1.5


def star_mask(y: torch.Tensor, r: int, check=None) -> torch.Tensor:
    """Where the stars are (1, h, w in 0..1): the luminance above STAR_LEVEL and MASK_SIGMAS noise units above the
    local sky, the sky and its noise measured at MASK_SCALE with the stars clipped out (the iterative clipping of the
    astronomical background estimators, Bertin and Arnouts 1996), so that a star's wings stand above it as its core
    does (issue #30); on a dense field the faint grain between the stars stays under it (issue #18). Grown by r pixels
    (round) to hold the wings the reduction takes, its edge softened over MASK_FEATHER pixels."""
    L = ops.luminance(y)[None]
    bg = ops.gaussian_blur(L, MASK_SCALE)
    noise = 1.2533 * ops.gaussian_blur((L - bg).abs(), MASK_SCALE)          # the mean deviation, as a sigma
    for _ in range(MASK_PASSES):                                            # the stars out of the sky and its noise
        _go_on(check)
        lo = torch.minimum(L, bg + MASK_CLIP * noise)
        bg = ops.gaussian_blur(lo, MASK_SCALE)
        noise = 1.2533 * ops.gaussian_blur((lo - bg).abs(), MASK_SCALE)
    sup = ((L > STAR_LEVEL) & (L > bg + MASK_SIGMAS * noise)).float()
    grown = _morph(sup, r, True)
    return torch.maximum(ops.gaussian_blur(grown, MASK_FEATHER), grown)[0].clamp(0, 1)     # whole on the stars


def blend(a: torch.Tensor, b: torch.Tensor, mode: str) -> torch.Tensor:
    """The stars ``b`` over the starless image ``a``."""
    if mode == "Linear dodge (add)":
        return (a + b).clamp(0, 1)
    if mode == "Lighten":
        return torch.maximum(a, b)
    return 1 - (1 - a) * (1 - b)
