"""The Align on block: a workflow's image brought onto the grid of another workflow's image, whatever their size, rotation
or scale. It differs from Registration: finished images (stacks, or imported images, possibly stretched and of
different instruments), not the frames of a night. It sits right after Stacking (or Image), and every later block reads
the aligned image, which has the reference's size: what overflows is cut, and where the image has no data is marked
invalid.

Two methods, chosen in the block, never guessed. ``stars`` (the default): the triangle matching of V. Valdes et al.
(PASP 107, 1119, 1995) and E. Groth (AJ 91, 1244, 1986), with the triangles of M. Beroiz, J. Cabral, B. Sanchez,
"Astroalign" (Astronomy and Computing 32, 100384, 2020), then the cubic polynomial of Registration on every star it
matched (a similarity alone with too few of them); the positions come from sep, on each image's own luminance: the
stretch of an image does not move its stars. ``plate``: both images carry a plate solution in their file, and the
reference grid's pixels go through the sky into the image (the reprojection of the FITS World Coordinate System,
Calabretta & Greisen 2002; the map computed on a coarse grid and interpolated, then a bicubic resampling). A stack has
no plate solution yet (Astrometry comes after): it is aligned through its stars.

The block writes the files of the stage it reads, each resampled on the reference grid with one map: the stack (full and
half-stacks, bicubic), its weights and saturation map (nearest, zero outside), and for a stretched image the Output
files an Image block writes.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.spatial import cKDTree

from ..core.params import P, StageParams
from ..core.stage import Context, Stage, StageError
from .register import SPARSE_MIN_MATCHES, TRIANGLE_STARS, _fit_from, _matched, _similarity, apply_poly, triangle_similarities

#: the scale two images may differ by (a triangle fitted to a far smaller one maps every star onto one point)
MAX_SCALE = 0.9
#: stars read per image
MAX_STARS = 2500
#: the share of the stars that can be matched (those landing inside the image, of the shorter list) that must be, or the
#: two images are not the same field: a similarity of a very different scale gathers stars by chance
MIN_MATCHED = 0.25


def plate_solution(path: str | Path):
    """The plate solution in a FITS header, or None (any other file, or a FITS without one)."""
    if Path(path).suffix.lower() not in (".fits", ".fit", ".fts"):
        return None
    import warnings

    from astropy.io import fits
    from astropy.wcs import WCS
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            wcs = WCS(fits.getheader(path)).celestial
        except Exception:  # noqa: BLE001 - a header astropy cannot read as a solution: none
            return None
    return wcs if wcs.has_celestial and wcs.wcs.ctype[0].strip() else None


def stars_of(a: np.ndarray) -> np.ndarray:
    """(x, y, flux) of an image's stars (3, H, W), the brightest first."""
    from .astrometry import detect
    g = np.ascontiguousarray(a.mean(0), np.float32)
    g = np.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
    try:
        o = detect(g)
    except Exception:  # noqa: BLE001 - sep refuses a flat image: no star in it
        return np.zeros((0, 3))
    cat = np.c_[o["x"], o["y"], o["flux"]]
    return cat[np.argsort(-cat[:, 2], kind="stable")][:MAX_STARS]


def _map_on_grid(fn, shape: tuple[int, int], step: int = 32):
    """(map_x, map_y) of a smooth mapping of the grid's pixels, computed on a coarse grid and interpolated."""
    H, W = shape
    ys = np.linspace(0, H - 1, max(4, int(np.ceil(H / step)) + 1))
    xs = np.linspace(0, W - 1, max(4, int(np.ceil(W / step)) + 1))
    gx, gy = np.meshgrid(xs, ys)
    ix, iy = fn(gx, gy)
    yy, xx = np.arange(H, dtype=np.float64), np.arange(W, dtype=np.float64)
    return (RectBivariateSpline(ys, xs, ix, kx=3, ky=3)(yy, xx).astype(np.float32),
            RectBivariateSpline(ys, xs, iy, kx=3, ky=3)(yy, xx).astype(np.float32))


def map_from_stars(ref: np.ndarray, cat: np.ndarray, shape: tuple[int, int], name: str, size: tuple[int, int]):
    """The map of the reference grid's pixels into the image (``size``: its own (H, W)), from their stars; (map, matched
    stars, residual px)."""
    H, W = shape
    if min(len(ref), len(cat)) < SPARSE_MIN_MATCHES:
        raise StageError(f"{name}: too few stars to align it. An image without its stars cannot be aligned by itself: "
                         "align the image that has them, then take them off (Stars/background after this block)")
    a, b = ref[:, :2], cat[:, :2]
    tree = cKDTree(b)
    s, t, _tri = triangle_similarities(a[:TRIANGLE_STARS], b[:TRIANGLE_STARS], max_scale=MAX_SCALE)
    za, zb = a[:, 0] + 1j * a[:, 1], b[:, 0] + 1j * b[:, 1]
    best, top = None, (0, 0.0)
    for s1, t1 in zip(s, t):
        z = s1 * za + t1
        i, _j, d = _matched(np.c_[z.real, z.imag], tree)
        score = (len(i), -float(np.sum(d ** 2)))
        if score > top:
            best, top = (s1, t1), score
    if best is not None:
        z = best[0] * za + best[1]
        inside = int(np.sum((z.real >= 0) & (z.real < size[1]) & (z.imag >= 0) & (z.imag < size[0])))
    if best is None or top[0] < SPARSE_MIN_MATCHES or top[0] < MIN_MATCHED * min(inside, len(cat)):
        raise StageError(f"{name}: its stars match none of the reference's: not the same field")
    s1, t1 = best
    for _ in range(3):
        z = s1 * za + t1
        i, j, d = _matched(np.c_[z.real, z.imag], tree)
        s1, t1 = _similarity(za[i], zb[j])
    z = s1 * za + t1
    fit = _fit_from(ref, cat, tree, W, H, np.c_[z.real, z.imag], 4.0)
    if fit is not None:
        coef, n, rms, _ratio = fit
        return _map_on_grid(lambda x, y: apply_poly(coef, x, y, W, H), shape), n, rms
    i, _j, d = _matched(np.c_[z.real, z.imag], tree)

    def similarity(x, y):
        m = s1 * (x + 1j * y) + t1
        return m.real, m.imag
    return _map_on_grid(similarity, shape), len(i), float(np.sqrt(np.mean(d ** 2)))


def reprojection_map(wcs_ref, wcs_in, shape: tuple[int, int], step: int = 32):
    """(map_x, map_y) float32 of the reference grid's pixels in the input image (cv2.remap): through the sky, on a
    coarse grid spanning the image, interpolated by a bicubic spline."""
    return _map_on_grid(lambda gx, gy: wcs_in.all_world2pix(*wcs_ref.all_pix2world(gx, gy, 0), 0), shape, step)


def warp(a: np.ndarray, wp, nearest: bool = False) -> np.ndarray:
    """An image resampled through a map of the reference grid's pixels (cv2.remap)."""
    import cv2  # OpenCV loaded when an image is resampled, not with the table
    mx, my = wp
    return cv2.remap(np.ascontiguousarray(a, np.float32), mx, my, cv2.INTER_NEAREST if nearest else cv2.INTER_CUBIC,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)


#: why a block has no reference yet (the runner leaves it and what follows out; the graph greys it)
NO_REFERENCE = "reference to wire"
NO_PLATE_STACK = "the stack has no plate solution yet (Astrometry comes after)"
#: the files of a stack the block resamples: (name, nearest neighbour)
RESAMPLED = (("drz_full.npy", False), ("drz_A.npy", False), ("drz_B.npy", False), ("drz_weight.npy", True))


class AlignOnParams(StageParams):
    method: str = P("stars", label="Align method", choices=["stars", "plate"],
                    help="How the image is brought onto the reference's grid: through their stars (the default, "
                         "whatever the images), or through their plate solutions (both images must carry one in "
                         "their file).")
    #: set by the project (not keyed: the reference workflow's keys are): {"pid": the reference workflow, "dir": the
    #: folder of the image it reads, "wcs": the file holding its plate solution, "path": this image's own file}
    reference: dict = P({}, label="Reference", level="internal",
                        help="The workflow whose image this one is aligned on.")


class AlignOn(Stage):
    name = "align_on"
    title = "Align on"
    description = ("Brings this workflow's image onto the grid of another workflow's image, through their stars or their "
                   "plate solutions; the blocks after it read the aligned image.")
    deps = ("stack",)
    Params = AlignOnParams
    version = "1"

    def run(self, ctx: Context, p: AlignOnParams) -> None:
        ref = p.reference or {}
        if not ref.get("dir"):
            raise StageError(NO_REFERENCE)
        rdir = Path(ref["dir"])
        if not (rdir / "drz_full.npy").is_file() or not (rdir / "drz_meta.json").is_file():
            raise StageError("the reference image is not computed: run its workflow first")
        rmeta = json.loads((rdir / "drz_meta.json").read_text(encoding="utf-8"))
        meta = ctx.load_json("stack", "drz_meta.json")
        full = np.nan_to_num(np.asarray(ctx.load("stack", "drz_full.npy")), nan=0.0, posinf=0.0, neginf=0.0)
        rimg = np.asarray(np.load(rdir / "drz_full.npy", mmap_mode="r"))
        H, W = rimg.shape[-2:]
        if p.method == "plate":
            if not ref.get("path"):
                raise StageError(NO_PLATE_STACK)
            wcs_a = plate_solution(ref["path"])
            wcs_r = plate_solution(ref["wcs"]) if ref.get("wcs") else None
            if wcs_a is None or wcs_r is None:
                what = "this image" if wcs_a is None else "the reference image"
                raise StageError(f"{what} has no plate solution in its file: align through the stars instead")
            wp = reprojection_map(wcs_r, wcs_a, (H, W))
            ctx.log("aligned through the plate solutions of the two files")
        else:
            wp, n, rms = map_from_stars(stars_of(rimg), stars_of(full), (H, W), "this image", full.shape[-2:])
            ctx.log(f"aligned through {n} stars, residual {rms:.2f} px")
            ctx.metric("matched_stars", int(n))
            ctx.metric("residual_px", round(float(rms), 3))
        valid = warp(np.ones(full.shape[-2:], np.float32), wp, nearest=True) > 0.5
        sdir = Path(ctx.project.stage_dir("stack"))
        for i, (name, nearest) in enumerate(RESAMPLED):
            if not (sdir / name).is_file():
                continue
            a = np.asarray(ctx.load("stack", name))
            a = np.stack([warp(c, wp, nearest) for c in a]) if a.ndim == 3 else warp(a, wp, nearest)
            if not nearest:
                a = np.nan_to_num(a, nan=0.0)
            a[..., ~valid] = 0
            ctx.save(name, a.astype(np.float32))
            ctx.progress((i + 1) / (len(RESAMPLED) + 1))
        if (sdir / "saturation.npy").is_file():
            s = np.asarray(np.load(sdir / "saturation.npy"))
            s = np.stack([warp(c.astype(np.float32), wp, True) for c in s]).astype(s.dtype)
            s[..., ~valid] = 0
            ctx.save("saturation.npy", s)
        out_meta = {**meta, "width": W, "height": H, "scale": float(rmeta.get("scale", meta.get("scale", 1.0))),
                    "aligned_on": ref.get("pid")}
        if meta.get("stretched"):
            from .image import save_stretched
            lin = np.clip(np.asarray(np.load(ctx.out("drz_full.npy"))), 0, 1)
            prep = ctx.load_json("stack", "prep_simple.json") if (sdir / "prep_simple.json").is_file() else {}
            save_stretched(ctx, lin, valid, str(prep.get("source", meta.get("file", ""))), 1.0, out_meta)
        else:
            ctx.save_json("drz_meta.json", out_meta)
        ctx.metric("width", W)
        ctx.metric("height", H)
        v = np.moveaxis(np.load(ctx.out("drz_full.npy")), 0, -1)
        lo, hi = np.percentile(v, 1), np.percentile(v, 99.5)
        ctx.preview("image", np.clip((v - lo) / max(hi - lo, 1e-6), 0, 1) ** 0.5)
