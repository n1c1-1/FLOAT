"""Inspect's pictures: what a view shows, the display stretch, loading a picture, the sky grid and the
catalogue stars over it, the cache of loaded pictures."""
from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ... import api
from ..stretch import mtf


# ---------------------------------------------------------------------------------------------- what a view shows
@dataclass(frozen=True)
class VisualSpec:
    """One picture of a block: where it comes from and how to read it."""

    key: str                 # the tab: "stack", "starless"...
    caption: str             # English, translated where shown
    kind: str                # array | preview | chart | render (the final image, rendered by the render worker)
    stage: str
    name: str = ""           # the array (without .npy) or the preview (without .jpg); "" for a chart or a render
    grid: bool = False       # on the stack grid: the RA / Dec readout
    linear: bool = False     # linear data: the automatic stretch applies
    spec: bool = False       # ◇: not in the engine yet (a disabled tab, in Spec mode only)
    sky: bool = False        # the astrometric solution is drawn over it (grid, catalogue stars, named objects)
    note: str = ""           # one line under the tabs: what this picture is for (English, translated where shown)
    camera: bool | None = None       # still in the camera's colours (the viewer's Neutral sky); None: guessed
    neutral: bool | None = None      # Neutral sky at first (display only, the user unticks it); None: off for a camera
    #                                  array (the stack as it is), the viewer's own rule otherwise
    compare: tuple = ()      # the pictures it is compared with (keys of ``Steps.picture``): the previous steps
    open_compare: bool = False       # opened compared with the first of them (split)
    same_stretch: bool = False       # the compared picture with the same stretch
    base: tuple = ()         # kind "color": the (stage, array) the colour matrix is applied to
    channel: int | None = None       # one channel of the array (the coverage)
    units: str = "adu"
    shown: str = ""          # its name in the viewer (the split's tag), when it is not the caption (English)
    compare_names: tuple = ()        # the names of the pictures of ``compare`` in the viewer, when not their captions
    backgrounds: bool = False        # the picture behind its overlay is the user's choice (``Steps.backgrounds``), no compare
    display: str = ""        # the viewer's kind of image (``ViewImage.kind``): "model" for a sky model, a gradient
    #                          surface, a star flat (linked, from zero: its colour is its own); "" a light
    previous: bool = False   # made from an Outdated result (its previous one): the lists mark it


@dataclass
class GridLine:
    """A line of the sky grid: a meridian ('ra') or a parallel ('dec'), as polylines in pixels of the stack grid (a line
    that leaves the field and comes back is several pieces)."""

    kind: str                                  # "ra" | "dec"
    label: str                                 # "02h40m", "+61°"
    pieces: list[np.ndarray] = field(default_factory=list)      # each (n, 2): x, y


@dataclass
class SkyData:
    """What the astrometric solution draws on the stack (made by the worker, drawn by ``steps.SkyOverlay``)."""

    lines: list[GridLine] = field(default_factory=list)
    stars: np.ndarray | None = None            # (n, 3): x, y, G magnitude of the catalogue stars, in pixels of the grid
    #: the named objects of the field (``annotate.Named``: centre, outline, designation, common name, kind) in pixels
    #: of the grid; ``source``: the catalogue they come from; ``why``: why there are none (no catalogue), English
    objects: list = field(default_factory=list)
    source: str = ""
    why: str = ""


@dataclass
class ShownImage:
    """What the worker hands the view for one picture. ``stretched`` is what is shown (the engine's preview as it is, or
    the automatic stretch of an array); ``linear`` is the same data without the stretch (None for a preview)."""

    key: tuple
    spec: VisualSpec
    stretched: Any = None                      # a PixelSource
    linear: Any = None                         # a PixelSource, or None
    lin: np.ndarray | None = None              # (3, H, W) float32, the data (ADU); None for a preview
    size: tuple[int, int] = (0, 0)             # (w, h) of the pixels shown
    grid: tuple[int, int] | None = None        # (w, h) of the stack grid the pixels cover (sky readout), or None
    origin: tuple[float, float] = (0.0, 0.0)   # the grid pixel of the first shown pixel (a cropped final image)
    sky: SkyData | None = None
    wcs: Any = None
    hist: dict[str, np.ndarray] = field(default_factory=dict)     # "stretched" / "linear" -> (3, 256) counts
    stretch: dict[str, list[float]] = field(default_factory=dict)  # black, white, balance per channel
    curves: Any = None                         # core.spectral.SpectralResponse, for a chart of the Color block
    nbytes: int = 0


class PixelSource:
    """A fixed picture (H x W x 3 uint8) and its half-resolution copy, an average of 2 x 2 pixels: what the viewer asks
    for when it shows the picture reduced (the viewer's own ``StaticSource`` drops three pixels in four, and the stars
    of a reduced picture would twinkle)."""

    def __init__(self, arr: np.ndarray):
        self.arr = np.ascontiguousarray(arr)
        self.half = reduce2(self.arr)
        self.version = 0

    def size(self):
        return self.arr.shape[1], self.arr.shape[0]

    def region(self, level, x0, y0, x1, y1):
        a = self.arr if level == 0 else self.half
        return a[y0:y1, x0:x1]

    @property
    def nbytes(self) -> int:
        return self.arr.nbytes + self.half.nbytes


def reduce2(a: np.ndarray) -> np.ndarray:
    """H x W x 3 uint8 -> half size, each pixel the mean of 2 x 2 (an odd last row or column is repeated)."""
    h, w = a.shape[:2]
    if h % 2 or w % 2:
        a = np.pad(a, ((0, h % 2), (0, w % 2), (0, 0)), mode="edge")
    s = a[::2, ::2].astype(np.uint16) + a[1::2, ::2] + a[::2, 1::2] + a[1::2, 1::2]
    return ((s + 2) >> 2).astype(np.uint8)


# ---------------------------------------------------------------------------------------------- the display stretch


def stretch_params(lin: np.ndarray) -> dict[str, np.ndarray]:
    """The automatic stretch of a (3, H, W) linear image, channel by channel: the black point sits 2.8 sigma under the
    median (sigma from the median absolute deviation), the white point at the 99.99th percentile, and the midtones put
    the median at 0.25. A sample of the image gives the numbers: the stretch is a display, never a measurement."""
    c = lin.shape[0]
    sub = lin[:, ::4, ::4].reshape(c, -1)
    sub = sub[:, np.isfinite(sub).all(0)] if not np.isfinite(sub).all() else sub
    med = np.median(sub, axis=1)
    mad = 1.4826 * np.median(np.abs(sub - med[:, None]), axis=1)
    bp = med - 2.8 * np.maximum(mad, 1e-7 * np.maximum(np.abs(med), 1.0))
    wp = np.maximum(np.percentile(sub, 99.99, axis=1), bp + 1e-6)
    mn = np.clip((med - bp) / (wp - bp), 1e-4, 0.99)
    return {"black": bp, "white": wp, "balance": mtf(0.25, mn)}


def neutral_params(lin: np.ndarray, ref: float = 99.5) -> dict[str, np.ndarray]:
    """The automatic stretch of an image still in the camera's colours (the engine applies the colour matrix at the
    output: every array Inspect shows is such an image). Each channel is first set so that its sky (the median) and
    its stars (the ``ref`` percentile) line up with the other channels, then one stretch is applied to all three, as
    the linked stretch of a calibrated image. The sky comes out neutral and the stars white, where a stretch per
    channel alone leaves the green of the sensor (it has twice as many green pixels) and of the light pollution.
    Display only: the data are never changed."""
    c = lin.shape[0]
    if c != 3:
        return stretch_params(lin)
    sub = lin[:, ::4, ::4].reshape(c, -1)
    sub = sub[:, np.isfinite(sub).all(0)] if not np.isfinite(sub).all() else sub
    med = np.median(sub, axis=1)
    hi = np.percentile(sub, ref, axis=1)
    span = np.maximum(hi - med, 1e-7 * np.maximum(np.abs(med), 1.0))
    x = ((sub - med[:, None]) / span[:, None]).mean(axis=0)
    m = float(np.median(x))
    mad = 1.4826 * float(np.median(np.abs(x - m)))
    bp = m - 2.8 * max(mad, 1e-7)
    wp = max(float(np.percentile(x, 99.99)), bp + 1e-6)
    mn = float(np.clip((m - bp) / (wp - bp), 1e-4, 0.99))
    bal = float(mtf(0.25, mn))
    return {"black": med + bp * span, "white": med + wp * span, "balance": np.array([bal] * c)}


def neutral_preview(a: np.ndarray) -> np.ndarray:
    """An engine preview of a frame in the camera's colours (H, W, 3 uint8), its sky made neutral for display: each
    channel is mapped piecewise linearly so that 0 stays 0, 255 stays 255 and its median goes to the median of the
    three."""
    med = np.median(a[::4, ::4].reshape(-1, 3), axis=0).astype(np.float64)
    target = float(med.mean())
    if target <= 0 or target >= 255 or (med.max() - med.min()) < 1:
        return a
    out = np.empty_like(a)
    x = np.arange(256, dtype=np.float64)
    for k in range(3):
        mk = min(max(med[k], 1.0), 254.0)
        lut = np.where(x <= mk, x * target / mk, target + (x - mk) * (255.0 - target) / (255.0 - mk))
        out[..., k] = np.clip(lut + 0.5, 0, 255).astype(np.uint8)[a[..., k]]
    return out


#: the engine previews of a frame in the camera's colours (before the colour calibration): shown with a neutral sky
CAMERA_PREVIEWS = {("calibrate", "calibrated_frame"), ("inventory", "middle_frame"), ("denoise", "before_after")}


def _lut(balance: float) -> np.ndarray:
    x = np.linspace(0.0, 1.0, 65536, dtype=np.float32)
    y = x if abs(balance - 0.5) < 1e-6 else mtf(balance, x)
    return (np.clip(y, 0.0, 1.0) * 255 + 0.5).astype(np.uint8)


def render8(lin: np.ndarray, black, white, balance) -> np.ndarray:
    """(3, H, W) float -> (H, W, 3) uint8: (x - black) / (white - black), then the midtones function (balance 0.5: none).
    Each channel has its own numbers (or one number for all, a scalar)."""
    c, h, w = lin.shape
    bp = np.broadcast_to(np.asarray(black, np.float64), (c,))
    wp = np.broadcast_to(np.asarray(white, np.float64), (c,))
    mb = np.broadcast_to(np.asarray(balance, np.float64), (c,))
    out = np.empty((h, w, 3), np.uint8)
    luts = {}
    for k in range(c):
        luts[k] = _lut(float(mb[k]))
    for r0 in range(0, h, 512):
        blk = lin[:, r0:r0 + 512]
        for k in range(c):
            x = (blk[k] - np.float32(bp[k])) * np.float32(65535.0 / (wp[k] - bp[k]))
            np.nan_to_num(x, copy=False, nan=0.0, posinf=65535.0, neginf=0.0)
            np.clip(x, 0, 65535, out=x)
            out[r0:r0 + 512, :, k] = luts[k][x.astype(np.uint16)]
    if c == 1:
        out[:, :, 1] = out[:, :, 2] = out[:, :, 0]
    return out


def histogram8(a: np.ndarray) -> np.ndarray:
    """(3, 256) counts of the values of an (H, W, 3) uint8 picture (every second pixel is enough)."""
    s = a[::2, ::2]
    return np.stack([np.bincount(s[:, :, k].ravel(), minlength=256) for k in range(3)])


def stretch_image(lin: np.ndarray, neutral: bool = True):
    """(stretched uint8, linear uint8, parameters): the automatic stretch (``neutral``: for an image in the camera's
    colours, ``neutral_params``; else one stretch per channel), and the data as stored (one black and one white for the
    three channels, no midtones), each ready to show."""
    sp = neutral_params(lin) if neutral else stretch_params(lin)
    stretched = render8(lin, sp["black"], sp["white"], sp["balance"])
    linear = render8(lin, float(sp["black"].min()), float(sp["white"].max()), 0.5)
    return stretched, linear, sp


# ---------------------------------------------------------------------------------------------- loading a picture
def _rgb_of_file(path: str) -> np.ndarray:
    from PIL import Image
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def _wcs_of(p, wid: int, which: str):
    out = api.stage_outputs(p, wid, "astrometry", which)
    if out is None or not (Path(out.dir) / "wcs.fits").is_file():
        return None
    import warnings

    from astropy.io import fits
    from astropy.wcs import WCS
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return WCS(fits.getheader(Path(out.dir) / "wcs.fits"))


# ---------------------------------------------------------------------------------------------- the sky grid
GRID_LINES = 6                                  # about this many meridians (and parallels) across the field


_DEC_STEPS = (1, 2, 5, 10, 15, 30, 60, 120, 300, 600)                         # arcmin


_RA_STEPS = (1, 2, 5, 10, 15, 30, 60, 120, 240, 300, 600, 900, 1800, 3600)    # seconds of time


CATALOGUE_G_MAX = 9.0                           # the catalogue stars drawn: brighter than this (Gaia G)


CATALOGUE_MAX = 1500


def _step(span: float, steps: tuple, target: int = GRID_LINES) -> float:
    """The smallest of the round ``steps`` that puts at most ``target`` lines across ``span`` (same unit)."""
    return next((s for s in steps if span / s <= target), steps[-1])


def ra_label(ra: float, step_s: float) -> str:
    """02h40m (seconds too when the step is under a minute)."""
    t = int(round((ra % 360.0) * 240.0)) % 86400
    h, m, s = t // 3600, t % 3600 // 60, t % 60
    return f"{h:02d}h{m:02d}m" if step_s >= 60 else f"{h:02d}h{m:02d}m{s:02d}s"


def dec_label(dec: float, step_arcmin: float) -> str:
    """+61° (arcminutes too when the step is under a degree)."""
    a = int(round(abs(dec) * 60.0))
    d, m = a // 60, a % 60
    sign = "+" if dec >= 0 else "−"
    return f"{sign}{d:02d}°" if step_arcmin >= 60 else f"{sign}{d:02d}°{m:02d}′"


def _pieces(x: np.ndarray, y: np.ndarray, w: int, h: int) -> list[np.ndarray]:
    """A polyline cut into the runs that lie within (a margin around) the picture."""
    m = 0.1 * max(w, h)
    ok = np.isfinite(x) & np.isfinite(y) & (x > -m) & (x < w + m) & (y > -m) & (y < h + m)
    out, i = [], 0
    while i < len(ok):
        if not ok[i]:
            i += 1
            continue
        j = i
        while j < len(ok) and ok[j]:
            j += 1
        if j - i >= 2:
            out.append(np.stack([x[i:j], y[i:j]], 1).astype(np.float32))
        i = j
    return out


def sky_grid(wcs, w: int, h: int) -> list[GridLine]:
    """The RA / Dec grid over a picture of w x h pixels of the grid ``wcs`` solves: round steps (the minutes of the
    right ascension, the degrees of the declination), each line sampled and projected back through the solution
    (distortion included)."""
    xs, ys = np.meshgrid(np.linspace(0, w - 1, 17), np.linspace(0, h - 1, 17))
    ra, dec = wcs.all_pix2world(xs, ys, 0)
    ok = np.isfinite(ra) & np.isfinite(dec)
    if ok.sum() < 8:
        return []
    ra, dec = ra[ok], dec[ok]
    ra0 = float(np.degrees(np.arctan2(np.sin(np.radians(ra)).mean(), np.cos(np.radians(ra)).mean()))) % 360.0
    dra = (ra - ra0 + 180.0) % 360.0 - 180.0
    r0, r1, d0, d1 = float(dra.min()), float(dra.max()), float(dec.min()), float(dec.max())
    lines: list[GridLine] = []
    dstep = _step((d1 - d0) * 60.0, _DEC_STEPS)
    pad_r, pad_d = 0.04 * (r1 - r0), 0.04 * (d1 - d0)
    rs = np.linspace(ra0 + r0 - pad_r, ra0 + r1 + pad_r, 64) % 360.0
    k0, k1 = int(np.ceil(d0 * 60.0 / dstep)), int(np.floor(d1 * 60.0 / dstep))
    for k in range(k0, k1 + 1):
        d = k * dstep / 60.0
        if abs(d) >= 90:
            continue
        x, y = wcs.all_world2pix(rs, np.full_like(rs, d), 0)
        pcs = _pieces(x, y, w, h)
        if pcs:
            lines.append(GridLine("dec", dec_label(d, dstep), pcs))
    if d1 < 88 and d0 > -88:                                       # at a pole the meridians all meet
        rstep = _step((r1 - r0) * 240.0, _RA_STEPS)                # seconds of time (1 degree = 240 s)
        ds = np.clip(np.linspace(d0 - pad_d, d1 + pad_d, 64), -89.9, 89.9)
        a0 = ra0 * 240.0
        k0, k1 = int(np.ceil((a0 + r0 * 240.0) / rstep)), int(np.floor((a0 + r1 * 240.0) / rstep))
        for k in range(k0, k1 + 1):
            a = (k * rstep / 240.0) % 360.0
            x, y = wcs.all_world2pix(np.full_like(ds, a), ds, 0)
            pcs = _pieces(x, y, w, h)
            if pcs:
                lines.append(GridLine("ra", ra_label(a, rstep), pcs))
    return lines


def catalogue_stars(p, wid: int, which: str, w: int, h: int) -> np.ndarray | None:
    """The brightest Gaia stars of the field, where the solution puts them (the Catalogs stage's own pixel positions):
    (n, 3) x, y, G, brightest first; None when the Catalogs result is not there."""
    out = api.stage_outputs(p, wid, "catalogs", which) or api.stage_outputs(p, wid, "catalogs")
    f = Path(out.dir) / "gaia_deep.npz" if out is not None else None
    if f is None or not f.is_file():
        return None
    with np.load(f) as z:
        g, x, y = z["g"], z["x"], z["y"]
    sel = np.flatnonzero((g < CATALOGUE_G_MAX) & (x >= 0) & (x < w) & (y >= 0) & (y < h))
    sel = sel[np.argsort(g[sel])][:CATALOGUE_MAX]
    return np.stack([x[sel], y[sel], g[sel]], 1).astype(np.float32)


def sky_data(p, wid: int, which: str, wcs, w: int, h: int, names: bool = True) -> SkyData | None:
    """The grid, the catalogue stars and the named objects of a picture of the stack (None without a solution)."""
    if wcs is None:
        return None
    try:
        lines = sky_grid(wcs, w, h)
    except (ValueError, ArithmeticError):                       # a degenerate solution: the picture alone
        lines = []
    try:
        stars = catalogue_stars(p, wid, which, w, h)
    except (OSError, ValueError, KeyError):
        stars = None
    out = SkyData(lines, stars)
    if names:
        from ..annotate import field_annotations
        ann = field_annotations(wcs, w, h, grid=False, names=True)
        out.objects, out.source, out.why = ann.objects, ann.source, ann.why
    return out


def add_sky(im: ShownImage, p, wid: int, spec: VisualSpec, which: str = "current") -> None:
    """The solution's grid and catalogue stars for a picture of the stack that was loaded without them (the same stack is
    shown by Stacking and by Astrometry: it is kept once, the solution is a small addition)."""
    if spec.sky and im.sky is None and im.lin is not None:
        w, h = im.size
        im.wcs = im.wcs or _wcs_of(p, wid, which) or _wcs_of(p, wid, "current")
        im.grid = im.grid or (w, h)
        im.sky = sky_data(p, wid, which, im.wcs, w, h)


def load_visual(p, wid: int, spec: VisualSpec, which: str = "current") -> ShownImage:
    """One picture of a done result, ready to show (worker thread). ``which``: current, or previous (the result an
    Outdated block had). Raises ``api.NotDone`` when the result or the file is not there."""
    out = api.stage_outputs(p, wid, spec.stage, which)
    if out is None:
        raise api.NotDone(f"the {spec.stage} result of workflow {wid} is not computed ({which})")
    key = (spec.stage, out.key, spec.name, which)
    if spec.kind == "chart":
        return ShownImage(key, spec, curves=api.spectral_response(p, wid, which))
    if spec.kind == "preview":
        f = out.previews.get(spec.name)
        if f is None:
            raise api.NotDone(f"{spec.stage} has no preview {spec.name!r}")
        a = _rgb_of_file(f)
        if (spec.stage, spec.name) in CAMERA_PREVIEWS:
            a = neutral_preview(a)
        src = PixelSource(a)
        return ShownImage(key, spec, stretched=src, size=src.size(), hist={"stretched": histogram8(a)},
                          wcs=_wcs_of(p, wid, which) if spec.grid else None, nbytes=src.nbytes)
    lin = np.asarray(api.stage_array(p, wid, spec.stage, spec.name, which=which), np.float32)
    if lin.ndim == 2:
        lin = lin[None]
    if lin.shape[0] == 1:
        lin = np.repeat(lin, 3, axis=0)
    h, w = lin.shape[1:]
    stretched, linear, sp = stretch_image(lin)
    s8, l8 = PixelSource(stretched), PixelSource(linear)
    grid = None
    wcs = None
    if spec.grid:
        wcs = _wcs_of(p, wid, which) or _wcs_of(p, wid, "current")
        grid = (w, h)
    sky = sky_data(p, wid, which, wcs, w, h) if spec.sky else None
    return ShownImage(key, spec, stretched=s8, linear=l8, lin=lin, size=(w, h), grid=grid, wcs=wcs, sky=sky,
                      hist={"stretched": histogram8(stretched), "linear": histogram8(linear)},
                      stretch={k: [float(x) for x in v] for k, v in sp.items()},
                      nbytes=lin.nbytes + s8.nbytes + l8.nbytes)


class ImageCache:
    """The last pictures, least recently used first out, within a budget of bytes (the arrays are big)."""

    def __init__(self, budget: int = 2_000_000_000):
        self.budget = budget
        self._d: OrderedDict[tuple, ShownImage] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: tuple) -> ShownImage | None:
        with self._lock:
            im = self._d.get(key)
            if im is not None:
                self._d.move_to_end(key)
            return im

    def put(self, im: ShownImage):
        with self._lock:
            self._d[im.key] = im
            self._d.move_to_end(im.key)
            while len(self._d) > 1 and sum(x.nbytes for x in self._d.values()) > self.budget:
                self._d.popitem(last=False)

    def clear(self):
        with self._lock:
            self._d.clear()

    def __len__(self) -> int:
        return len(self._d)

    def keys(self) -> list[tuple]:
        return list(self._d)
