"""The viewer's images: what the viewer shows, the alignment of two images, reading a file, a prepared image
and its tiles, thumbnails and the default stretch."""
from __future__ import annotations

import itertools
import math
import os
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PySide6.QtGui import QColor

from . import stretch as sm
from .widgets import ACCENT


TILE = 512
CANVAS_BG = QColor("#0f1115")
CANVAS_RGB = (15, 17, 21)               # the same, for a picture drawn where an aligned image has no pixel
CHANNEL_COLORS = ("#e06c6c", "#5fbf7f", "#5ea1ff")
MONO_COLOR = "#c8ccd4"
ZOOM_STEP = 1.25
ZOOM_PRESETS = (0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
MONO_FONT = "Consolas"
QSS = f"""
QFrame#ivbar {{ background: #1c1f26; border-bottom: 1px solid #2a2f38; }}
QFrame#ivread {{ background: #181b21; border-top: 1px solid #2a2f38; }}
QFrame#ivpanel {{ background: #1c1f26; border-left: 1px solid #2a2f38; }}
QFrame#ivseq {{ background: #1a1d23; border-top: 1px solid #2a2f38; }}
QFrame#ivsep {{ background: #353b47; }}
QFrame#ivalign {{ background: #1a1d23; border-bottom: 1px solid #2a2f38; }}
QLabel#ivalign {{ color: #9fc2ff; font-size: 8.5pt; }}
QLabel#ivcap {{ color: #7d8594; font-size: 8pt; }}
QLabel#ivval {{ color: #e8ebf0; font-size: 10pt; }}
QLabel#ivname {{ color: #ffffff; font-size: 10.5pt; font-weight: 600; }}
QLabel#ivread {{ color: #c9ced8; font-family: '{MONO_FONT}'; font-size: 9pt; }}
QPushButton#ivtool {{ padding: 3px 9px; }}
QPushButton#ivseg {{ padding: 3px 9px; border-radius: 0; }}
QPushButton#ivseg:checked {{ background: #2b3446; border-color: {ACCENT}; color: #ffffff; }}
QPushButton#ivstate {{ padding: 3px 12px; font-weight: 600; }}
QDoubleSpinBox#ivfield {{ min-width: 64px; }}
"""


# ---------------------------------------------------------------------------------------------- what the caller gives
@dataclass(eq=False)
class ViewImage:
    """One image for the viewer, as the caller describes it. Nothing is read until the viewer needs it (in a worker
    thread).

    ``source``: an array — (C, H, W) as the engine stores it, (H, W, C), (H, W) mono, or (H, W) Bayer mosaic with
    ``cfa`` — or a callable returning one, or a file path (.npy, .fits, .tif, .png, .jpg, or a camera raw).
    ``key``: the image's identity (the cache, Send to Post, the compare selector); default: the path, or the array.
    ``wcs``: an astropy WCS, a FITS header or the path of a wcs.fits: RA / Dec in the readout, and the FITS export.
    ``units``: "adu", "norm" (0-1), "count" (a count per pixel) or "" (a number without a unit); ``white``: the
    saturation level in those units (for the readout in the other unit; default 65535 ADU or 1). ``camera``: still
    in the camera's colours (Neutral sky on by default, unless ``neutral`` says otherwise); None: guessed from the sky
    (channel medians apart by more than 15 % and many noise sigmas: a camera's cast). ``neutral``: Neutral sky at
    first (a display option the user can tick or untick); None: as ``camera`` says (a mosaic: on).
    ``info``: a frame of a sequence, as the caller measured it: ``time`` (ISO), ``exposure`` (s), ``iso`` or
    ``gain``, ``temperature`` (°C), ``fwhm`` (px), ``fwhm_arcsec``, ``roundness`` (0-1), ``transparency`` (1 = 100 %),
    ``weight``, ``excluded`` (bool), ``reason`` (why the engine leaves it out), ``extra`` [(caption, text)].
    ``thumb``: a small (h, w, 3) uint8 picture or a callable returning one (the film strip). ``header``: FITS cards
    added on export."""

    source: Any
    name: str = ""
    key: Any = None
    wcs: Any = None
    units: str = "adu"
    white: float | None = None
    cfa: str = ""
    camera: bool | None = None
    info: dict = field(default_factory=dict)
    thumb: Any = None
    header: dict = field(default_factory=dict)
    #: already a picture for the eye (a final render, an engine preview): shown as it is, 0 to ``white`` linear, and
    #: read as levels without a unit
    stretched: bool = False
    #: Neutral sky at first: None follows ``camera``; False shows a camera-colour image as it is (its cast and all)
    neutral: bool | None = None
    #: what the image is, for its automatic stretch (``stretch.auto_for``): "" or "lights" (a light, a stack), "darks",
    #: "bias", "flat_darks", "flats", or "model" (a sky model, a gradient surface, a star flat). "Same stretch for all
    #: frames" keeps one stretch per kind, never across kinds.
    kind: str = ""
    #: where its pixels sit on a common grid, for the compare (``Placement``): two images placed on the same grid are
    #: drawn on each other by it; else by their astrometric solutions (``wcs``). A callable (w, h) -> Placement or None
    #: is called in the worker thread with the image's size.
    place: Any = None

    def identity(self):
        if self.key is not None:
            return self.key
        if isinstance(self.source, str | os.PathLike):
            return ("file", os.path.normcase(os.path.abspath(os.fspath(self.source))))
        return ("object", id(self.source))

    def white_level(self) -> float:
        if self.white:
            return float(self.white)
        return 1.0 if self.units == "norm" else 65535.0


@dataclass(frozen=True)
class Placement:
    """Where an image's pixels sit on a common grid (``ViewImage.place``). ``frame`` names the grid: two images placed
    on the same one are put on each other by their placements. ``m`` maps the image's coordinates to the grid's,
    X = m[0] x + m[1] y + m[2], Y = m[3] x + m[4] y + m[5], in pixels with the pixel edges on whole numbers (pixel i
    covers [i, i + 1)): a stack of drizzle scale s on the grid of its reference frame is (1/s, 0, 0, 0, 1/s, 0); a
    final image cut at (ox, oy) of that stack and shown at half size is, on the stack, (2, 0, ox, 0, 2, oy)."""

    frame: Any
    m: tuple = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)

    @classmethod
    def scaled(cls, frame, f: float = 1.0, ox: float = 0.0, oy: float = 0.0, fy: float | None = None) -> Placement:
        """X = ox + f x, Y = oy + fy y (fy: f by default)."""
        return cls(frame, (float(f), 0.0, float(ox), 0.0, float(f if fy is None else fy), float(oy)))

    def matrix(self) -> np.ndarray:
        a, b, c, d, e, f = self.m
        return np.array([[a, b, c], [d, e, f], [0.0, 0.0, 1.0]])

    def then(self, outer: Placement) -> Placement:
        """This placement, then ``outer`` (which places this one's grid on its own frame)."""
        return Placement(outer.frame, tuple(float(v) for v in (outer.matrix() @ self.matrix())[:2].ravel()))


@dataclass
class Alignment:
    """How a compared image is drawn on the grid of the image shown (display only; ``alignment``). ``T`` (2 x 3) maps a
    point of the shown image to the compared one's (pixel edges on whole numbers). ``how``: "grid" (both placed on one
    grid by the geometry of the results: drizzle scale, crop, half size), "sky" (by their astrometric solutions) or
    "width" (no common grid known: matched by their widths). ``residual``: for "sky", the largest distance, in pixels
    of the compared image, between the solutions and the affine map drawn."""

    T: np.ndarray
    how: str
    residual: float = 0.0
    #: for "sky": what the solutions add to the affine map (their distortions differ), sampled on an n x n grid of the
    #: shown image (x, y = linspace(0.5, size - 0.5, n)): (size, dx, dy), drawn by ``Display`` and read by ``apply``
    field: tuple | None = None

    def apply(self, x, y):
        t = self.T
        bx, by = t[0, 0] * x + t[0, 1] * y + t[0, 2], t[1, 0] * x + t[1, 1] * y + t[1, 2]
        if self.field is not None:
            rx, ry = self.field_at(np.asarray(x, float), np.asarray(y, float))
            bx, by = bx + rx, by + ry
            if np.ndim(bx) == 0:
                bx, by = float(bx), float(by)
        return bx, by

    def field_at(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The correction of the affine map at points of the shown image (bilinear between the grid's points)."""
        (w, h), fx, fy = self.field
        n = fx.shape[0]
        gx = np.clip((x - 0.5) / max(w - 1.0, 1e-9) * (n - 1), 0, n - 1)
        gy = np.clip((y - 0.5) / max(h - 1.0, 1e-9) * (n - 1), 0, n - 1)
        i0, j0 = np.minimum(np.floor(gy).astype(int), n - 2), np.minimum(np.floor(gx).astype(int), n - 2)
        ty, tx = gy - i0, gx - j0

        def lerp(f):
            return ((1 - ty) * ((1 - tx) * f[i0, j0] + tx * f[i0, j0 + 1])
                    + ty * ((1 - tx) * f[i0 + 1, j0] + tx * f[i0 + 1, j0 + 1]))
        return lerp(fx), lerp(fy)

    @property
    def scale(self) -> float:
        """Pixels of the compared image per pixel of the shown one."""
        return float(math.sqrt(abs(np.linalg.det(self.T[:, :2])))) or 1.0

    @property
    def angle(self) -> float:
        """The turn (degrees) of the compared image on the shown one."""
        return float(math.degrees(math.atan2(self.T[1, 0], self.T[0, 0])))

    def origin(self) -> tuple[float, float]:
        """Where the compared image's top-left corner falls on the shown one (its pixels)."""
        full = np.vstack([self.T, [0.0, 0.0, 1.0]])
        x, y, _w = np.linalg.solve(full, [0.0, 0.0, 1.0])
        return float(x), float(y)


def _identity(T: np.ndarray, tol: float = 1e-3) -> bool:
    return bool(np.allclose(T[:, :2], np.eye(2), atol=tol) and np.allclose(T[:, 2], 0.0, atol=0.02))


SKY_GRID = 17                           # the points of the shown image the solutions are read at (n x n)


def _sky_map(a: Prepared, b: Prepared) -> tuple[np.ndarray, float, tuple | None] | None:
    """The map from ``a`` to ``b`` their astrometric solutions give: the affine part (fitted on n x n points of ``a``),
    its largest error, and what the solutions add to it on that grid (their distortions differ: two programs' solutions
    of one field differ by a few pixels in the corners), None when under a tenth of a pixel. None when a solution cannot
    answer (no inverse, nothing in common)."""
    if not hasattr(b.wcs, "world_to_pixel_values") or not hasattr(a.wcs, "pixel_to_world_values"):
        return None
    n = SKY_GRID
    xs = np.linspace(0.5, a.width - 0.5, n)
    ys = np.linspace(0.5, a.height - 0.5, n)
    X, Y = (g.ravel() for g in np.meshgrid(xs, ys))
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            ra, dec = a.wcs.pixel_to_world_values(X - 0.5, Y - 0.5)
            bx, by = b.wcs.world_to_pixel_values(np.asarray(ra, float), np.asarray(dec, float))
        except Exception:  # noqa: BLE001 - a solution that cannot answer: no alignment by the sky
            return None
    bx, by = np.asarray(bx, float) + 0.5, np.asarray(by, float) + 0.5
    ok = np.isfinite(bx) & np.isfinite(by)
    if ok.sum() < 6:
        return None
    A = np.stack([X[ok], Y[ok], np.ones(int(ok.sum()))], 1)
    cx = np.linalg.lstsq(A, bx[ok], rcond=None)[0]
    cy = np.linalg.lstsq(A, by[ok], rcond=None)[0]
    res = float(np.hypot(A @ cx - bx[ok], A @ cy - by[ok]).max())
    field = None
    if res > 0.1 and ok.all():
        full = np.stack([X, Y, np.ones(X.size)], 1)
        field = ((float(a.width), float(a.height)), (bx - full @ cx).reshape(n, n).astype(np.float32),
                 (by - full @ cy).reshape(n, n).astype(np.float32))
    return np.array([cx, cy]), res, field


def alignment(a: Prepared, b: Prepared) -> Alignment | None:
    """How ``b`` (the compared image) is drawn on ``a`` (the one shown) so that a star sits at the same place in both:
    by their placements when both are on one grid (the geometry the results give: drizzle scale, crop, half size),
    else by their astrometric solutions when both have one, else matched by their widths (as the viewer always did).
    None: nothing to do (the same grid, the same size)."""
    pa, pb = a.place, b.place
    T, how, res, field = None, "", 0.0, None
    if pa is not None and pb is not None and pa.frame == pb.frame:
        T, how = (np.linalg.inv(pb.matrix()) @ pa.matrix())[:2], "grid"
    elif a.wcs is not None and b.wcs is not None:
        got = _sky_map(a, b)
        if got is not None:
            (T, res, field), how = got, "sky"
    if T is None:
        if a.size() == b.size():
            return None
        r = b.width / max(a.width, 1)
        T, how = np.array([[r, 0.0, 0.0], [0.0, r, 0.0]]), "width"
    if how != "width" and a.size() == b.size() and _identity(T) and res < 0.1:
        return None
    return Alignment(np.asarray(T, float), how, res, field)


RAW_SUFFIXES = {".arw", ".cr2", ".cr3", ".nef", ".dng", ".raf", ".orf", ".rw2", ".pef", ".srw"}


def read_file(path: str | os.PathLike, frame: bool = False) -> tuple[np.ndarray, str, Any]:
    """(array, Bayer pattern or "", WCS or None) of an image file (worker thread). ``frame``: a camera frame, whose
    FITS values are read in ADU as the engine reads them (io/frames.fits_adu: a master in floats on the 0-1 scale
    showed 0.008 where the engine and its white level say 508 ADU)."""
    p = Path(path)
    suf = p.suffix.lower()
    if suf == ".npy":
        return np.load(p), "", None
    if suf in (".fits", ".fit", ".fts"):
        import warnings

        from astropy.io import fits
        with fits.open(p, memmap=False) as hl:
            hdu = next((h for h in hl if getattr(h, "data", None) is not None), hl[0])
            data, hdr = np.asarray(hdu.data), hdu.header
        if frame:
            from ..io.frames import fits_adu
            data = fits_adu(data, hdr)
        cfa = str(hdr.get("BAYERPAT", "")).strip().upper() if data.ndim == 2 else ""
        wcs = None
        if "CTYPE1" in hdr:
            from astropy.wcs import WCS
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    wcs = WCS(hdr, naxis=2)
                except Exception:  # noqa: BLE001 - an unreadable solution: no sky readout
                    wcs = None
        return data, cfa, wcs
    if suf in (".tif", ".tiff"):
        import tifffile
        return tifffile.imread(p), "", None
    if suf in (".png", ".jpg", ".jpeg", ".bmp"):
        if suf == ".png":
            from ..io import cvfile                       # any path (review 13.1)
            a = cvfile.read(p)
            if a.ndim == 3:
                a = a[:, :, 2::-1] if a.shape[2] >= 3 else a
            return a, "", None
        from PIL import Image
        with Image.open(p) as im:
            return np.asarray(im.convert("RGB")), "", None
    if suf in RAW_SUFFIXES:
        from ..io.frames import read_cfa, read_info
        info = read_info(p)
        return read_cfa(p), info.cfa or "", None
    raise ValueError(f"cannot read {p.name}: unknown format")


def _wcs_of(w):
    """An astropy WCS from what the caller gave (a WCS, a header, a path, or a callable returning one of them: it is
    called in the worker thread), or None."""
    if callable(w) and not hasattr(w, "pixel_to_world_values"):
        try:
            w = w()
        except Exception:  # noqa: BLE001 - no solution: no sky readout
            return None
    if w is None:
        return None
    import warnings

    from astropy.wcs import WCS
    if isinstance(w, WCS) or hasattr(w, "pixel_to_world_values"):
        return w                                    # (an object with pixel_to_world_values only: the readout's)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            if isinstance(w, str | os.PathLike):
                from astropy.io import fits
                return WCS(fits.getheader(w), naxis=2)
            return WCS(w, naxis=2)
        except Exception:  # noqa: BLE001
            return None


# ---------------------------------------------------------------------------------------------- a prepared image
_UID = itertools.count(1)


class _Lru:
    """A thread-safe least-recently-used store of arrays within a byte budget."""

    def __init__(self, budget: int):
        self.budget = budget
        self._d: OrderedDict = OrderedDict()
        self._bytes = 0
        self._lock = threading.Lock()

    def get(self, key):
        with self._lock:
            v = self._d.get(key)
            if v is not None:
                self._d.move_to_end(key)
            return v

    def put(self, key, value: np.ndarray):
        with self._lock:
            old = self._d.pop(key, None)
            if old is not None:
                self._bytes -= old.nbytes
            self._d[key] = value
            self._bytes += value.nbytes
            while self._bytes > self.budget and len(self._d) > 1:
                _k, v = self._d.popitem(last=False)
                self._bytes -= v.nbytes

    def drop(self, pred):
        with self._lock:
            for k in [k for k in self._d if pred(k)]:
                self._bytes -= self._d.pop(k).nbytes

    def __len__(self):
        return len(self._d)


QTILES = _Lru(320 * 2 ** 20)            # the quantized tiles of every image shown (16 bits)
RTILES = _Lru(256 * 2 ** 20)            # the rendered tiles (8 bits), for the stretch of the moment


class Prepared:
    """An image ready to show: the data (read by the readout and the export), the display pyramid and the statistics.
    Made by ``prepare`` in a worker thread; read-only afterwards."""

    def __init__(self, vi: ViewImage, data: np.ndarray, levels: list[np.ndarray], factor: int, stats: sm.Stats,
                 cfa: str, wcs):
        self.uid = next(_UID)
        self.vi = vi
        self.key = vi.identity()
        self.name = vi.name
        self.data = data
        self.levels = levels
        self.factor = factor                         # image pixels per pixel of level 0 (2: a debayered mosaic)
        self.stats = stats
        self.cfa = cfa
        self.wcs = wcs
        if data.ndim == 2:
            self.height, self.width = data.shape
        else:
            self.height, self.width = data.shape[1:]
        self.channels = levels[0].shape[0]
        self.integer = bool(cfa or np.issubdtype(data.dtype, np.integer) or (
            isinstance(vi.source, np.ndarray) and np.issubdtype(vi.source.dtype, np.integer)))
        self.place: Placement | None = None          # where it sits on a common grid (``ViewImage.place``)

    # what the overlays read (``canvas.source.size()``)
    def size(self) -> tuple[int, int]:
        return self.width, self.height

    @property
    def nbytes(self) -> int:
        n = sum(lv.nbytes for lv in self.levels[1:]) + self.stats.hist.nbytes
        n += self.data.nbytes
        if self.levels[0] is not self.data and self.levels[0].base is not self.data:
            n += self.levels[0].nbytes
        return n

    def level_size(self, k: int) -> tuple[int, int]:
        lv = self.levels[k]
        return lv.shape[2], lv.shape[1]

    def tiles_of(self, k: int) -> tuple[int, int]:
        w, h = self.level_size(k)
        return -(-w // TILE), -(-h // TILE)

    def qtile(self, k: int, tx: int, ty: int) -> np.ndarray:
        """The quantized tile (C, th, tw) uint16 of level ``k`` (made once, kept while memory allows)."""
        key = (self.uid, k, tx, ty)
        q = QTILES.get(key)
        if q is None:
            lv = self.levels[k]
            blk = lv[:, ty * TILE:(ty + 1) * TILE, tx * TILE:(tx + 1) * TILE]
            q = sm.quantize(blk, self.stats.lo, self.stats.hi)
            QTILES.put(key, q)
        return q

    def value(self, x: int, y: int) -> tuple[list[float], str] | None:
        """The data at image pixel (x, y): the channel values, and the photosite's colour for a mosaic."""
        if not (0 <= x < self.width and 0 <= y < self.height):
            return None
        d = self.data
        if d.ndim == 2:
            return [float(d[y, x])], sm.cfa_letter(self.cfa, x, y) if self.cfa else ""
        return [float(v) for v in d[:, y, x]], ""

    def release(self):
        QTILES.drop(lambda k: k[0] == self.uid)


def planar(arr: np.ndarray) -> np.ndarray:
    """An image array as (C, H, W) float32, C 1 or 3 (an alpha channel dropped; not copied when it already is)."""
    a = np.asarray(arr)
    if a.ndim == 2:
        a = a[None]
    elif a.ndim == 3 and a.shape[0] not in (1, 3) and a.shape[2] in (1, 3, 4):
        a = a[:, :, :3].transpose(2, 0, 1) if a.shape[2] >= 3 else a.transpose(2, 0, 1)
    elif a.ndim != 3:
        raise ValueError(f"not an image: shape {a.shape}")
    if a.shape[0] not in (1, 3):
        a = a[:1] if a.shape[0] < 3 else a[:3]
    if a.dtype != np.float32 or not a.flags["C_CONTIGUOUS"]:
        a = np.ascontiguousarray(a, dtype=np.float32)
    return a


def _read(vi: ViewImage) -> tuple[np.ndarray, str, Any]:
    src = vi.source
    cfa, wcs = "" if vi.cfa == "RGB" else vi.cfa, None      # a frame already in colour (issue #57): no mosaic
    if isinstance(src, str | os.PathLike):
        arr, fcfa, wcs = read_file(src, frame=bool(vi.camera))
        cfa = cfa or fcfa
    elif callable(src):
        arr = src()
    else:
        arr = src
    arr = np.asarray(arr)
    return arr, "" if arr.ndim == 3 else cfa, wcs          # a mosaic is one plane: an image in colour has no pattern


def _to_side(img: np.ndarray, side: int) -> np.ndarray:
    """An (h, w) or (h, w, 3) uint8 picture as (h, w, 3), reduced so that its long side is at most ``side``."""
    import cv2
    if img.ndim == 2:
        img = np.repeat(img[:, :, None], 3, axis=2)
    h, w = img.shape[:2]
    f = side / max(h, w)
    if f < 1:
        img = cv2.resize(img, (max(1, round(w * f)), max(1, round(h * f))), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(img)


def quick_thumbnail(vi: ViewImage, side: int = 160) -> np.ndarray:
    """A thumbnail made straight from the source, for a film strip (worker thread): read, reduced, stretched
    automatically (neutral sky for a camera-colour image or a mosaic). ``ViewImage(..., thumb=partial(quick_thumbnail,
    vi))`` when the caller has no thumbnail of its own."""
    arr, cfa, _wcs = _read(vi)
    lv = sm.superpixel(arr, cfa) if arr.ndim == 2 and cfa else planar(arr)
    while max(lv.shape[1:]) > side * 2 and min(lv.shape[1:]) >= 4:
        lv = sm.reduce2(lv)
    stats = sm.compute_stats(lv, smooth=lv)
    camera = vi.camera if vi.camera is not None else (bool(arr.ndim == 2 and cfa) or sm.looks_camera(stats))
    st = sm.auto_for(stats, vi.kind, neutral=camera if sm.rule_of(vi.kind) == "light" else False)
    img = sm.render(sm.quantize(lv, stats.lo, stats.hi), sm.luts(st, stats))
    return _to_side(img, side)


def prepare(vi: ViewImage) -> Prepared:
    """Read (when needed) and prepare an image: the pyramid and the statistics (worker thread)."""
    arr, cfa, wcs = _read(vi)
    wcs = _wcs_of(vi.wcs) or wcs
    if arr.ndim == 2 and cfa:
        data = arr
        level0 = sm.superpixel(arr, cfa)
        factor = 2
    else:
        level0 = planar(arr)
        data = level0
        factor = 1
        cfa = ""
    levels = sm.pyramid(level0)
    stats = sm.compute_stats(level0, smooth=levels[-1])
    prep = Prepared(vi, data, levels, factor, stats, cfa, wcs)
    prep.place = _place_of(vi.place, prep.width, prep.height)
    return prep


def _place_of(place, w: int, h: int) -> Placement | None:
    """The placement the caller gave (a callable is called here, in the worker thread, with the image's size)."""
    if callable(place):
        try:
            place = place(w, h)
        except Exception:  # noqa: BLE001 - an unknown placement: the compare falls back on the solutions or the width
            return None
    return place if isinstance(place, Placement) else None


def thumbnail_of(prep: Prepared, st: sm.Stretch | None = None, side: int = 160) -> np.ndarray:
    """A small picture (h, w, 3) uint8 of a prepared image, its long side ``side``, stretched (automatic by default)."""
    st = st or default_stretch(prep)
    k = next((i for i, lv in enumerate(prep.levels) if max(lv.shape[1:]) <= max(side * 2, 64)), len(prep.levels) - 1)
    lv = prep.levels[k]
    img = sm.render(sm.quantize(lv, prep.stats.lo, prep.stats.hi), sm.luts(st, prep.stats))
    return _to_side(img, side)


def default_stretch(prep: Prepared, mode: str | None = None, linked: bool | None = None, neutral: bool | None = None,
                    vi: ViewImage | None = None) -> sm.Stretch:
    """The automatic stretch of an image, the one of its kind (``stretch.auto_for``); ``vi``: how the caller describes
    it now (default: as when it was read). ``mode``, ``linked``, ``neutral``: None for the kind's own."""
    v = vi or prep.vi
    if v.stretched and (mode or "mtf") == "mtf" and neutral is None:
        c = prep.channels
        white = v.white_level()
        return sm.Stretch("linear", np.zeros(c), np.full(c, white / 2), np.full(c, white), np.ones(c), np.zeros(c),
                          True, False, False)
    if neutral is None:
        if sm.rule_of(v.kind) != "light":
            neutral = False                     # a calibration frame or a model: no sky to make grey
        elif v.neutral is not None:
            neutral = bool(v.neutral)
        else:
            cam = v.camera
            neutral = bool(prep.cfa) or (cam if cam is not None else sm.looks_camera(prep.stats))
    return sm.auto_for(prep.stats, v.kind, mode, linked, neutral)


def kind_of(vi: ViewImage | None) -> str:
    """The kind of an image for "Same stretch for all frames" (one stretch per kind): its ``kind``, or "lights"."""
    k = str(vi.kind or "").lower() if vi is not None else ""
    return "lights" if k in ("", "light") else k
