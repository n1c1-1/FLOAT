"""Gaia DR3 catalogues of the field: its bright stars and their XP spectra (colour).

Only the bright stars are needed (the colour calibration's XP stars, G 7.5 to 14, and their neighbours): they come from
the offline Gaia extract when it is installed (core/sirilcat), else from the sky cache's G < 16 layer, the
astrometry's. Nothing is downloaded here (a run never reaches the network: core/online.py): the star data are
installed before the run starts (core/sky_catalogue.py), and what is not on the disk makes the stage fail with its reason."""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..core import skycache
from ..core.imgutil import safe_world2pix
from ..core.params import ADVANCED, P, StageParams
from ..core.stage import Context, Stage, StageError


def _cache() -> Path:
    """Where new downloads go: $ASTRO_GPU_HOME/cache (earlier versions' ~/.astro_gpu/cache is read too)."""
    return skycache.cache_dir()


class CatalogsParams(StageParams):
    xp_gmin: float = P(7.5, label="Bright limit", level=ADVANCED, ge=3, le=15, step=0.5, unit="G mag",
                       help="Brightest Gaia stars whose XP spectrum is used for the colour calibration.")
    xp_gmax: float = P(14.0, label="Faint limit", level=ADVANCED, ge=8, le=17.5, step=0.5, unit="G mag",
                       help="Faintest Gaia stars whose XP spectrum is used for the colour calibration.")
    xp_max: int = P(4000, label="Star count", level=ADVANCED, ge=100, le=20000,
                    help="Maximum number of XP spectra read for the colour calibration (from the star catalogue on "
                         "this computer).")
    cell_deg: float = P(1.0, label="Cell size", level=ADVANCED, ge=0.1, le=5, unit="deg",
                        help="Size of the cones in which the Gaia stars are read.")


class Catalogs(Stage):
    name = "catalogs"
    title = "Gaia catalogues"
    description = ("The bright Gaia DR3 stars of the field: positions (propagated to the date of the images), "
                   "magnitudes and colours, plus the low-resolution XP spectra of a few thousand isolated stars used "
                   "to calibrate the colour of the sensor. Results are cached and shared between projects.")
    deps = ("inventory", "astrometry", "stack")
    Params = CatalogsParams
    version = "5"          # 2: the bright stars missing from Gaia are added from Hipparcos; 3: the bright stars only
    key_label = "5"     # the keys' label, frozen (core/stage.Stage.key_label): never change it
                           # (offline), the deep catalogue for the earlier Stars block left out; 4: the colour stars
                           # spread over the magnitudes (issue #12); 5: the spectra may come from the compact extract
                           # (core/sirilcat.py): rebuilt on a basis, their XYZ the sampled ones' to 1e-7, not bit-equal

    def run(self, ctx: Context, p: CatalogsParams) -> None:
        from ..core.imgutil import read_wcs
        if ctx.metric_of("astrometry", "solved") is False:
            # Astrometry went on without a plate solution (its fallback): no star of the field can be placed; the
            # blocks that read them take their own fallback (Color: its fallback balance)
            ctx.metric("solved", False)
            ctx.log("no plate solution (the field was not identified): no Gaia star placed on the image")
            return
        w = read_wcs(ctx.input("astrometry", "wcs.fits"))
        meta = ctx.load_json("stack", "drz_meta.json")
        W, H = meta["width"], meta["height"]
        scale = float(ctx.metric_of("astrometry", "scale_arcsec"))
        epoch = _epoch(ctx.metric_of("inventory", "first_time"))
        centres, radius = _cells(w, W, H, scale, p.cell_deg)
        fine = _cells(w, W, H, scale, p.cell_deg / 4)
        ctx.log(f"{len(centres)} sky cells of {p.cell_deg} deg, epoch {epoch:.2f}")

        # ---------------- the bright stars of the field
        cols = ["Source", "RA_ICRS", "DE_ICRS", "Gmag", "BPmag", "RPmag", "pmRA", "pmDE", "XPsamp"]
        c = _bright_stars(ctx, centres, radius, cols, fine)
        dt = epoch - 2016.0
        dec_e = c["DE_ICRS"] + np.nan_to_num(c["pmDE"]) / 3.6e6 * dt
        ra_e = c["RA_ICRS"] + np.nan_to_num(c["pmRA"]) / 3.6e6 * dt / np.cos(np.radians(c["DE_ICRS"]))
        x, y = safe_world2pix(w, ra_e, dec_e, W, H)
        inside = (x > -30) & (x < W + 30) & (y > -30) & (y < H + 30) & np.isfinite(c["Gmag"])
        deep = dict(source=c["Source"][inside], ra=ra_e[inside], dec=dec_e[inside], x=x[inside], y=y[inside],
                    g=c["Gmag"][inside], bp=c["BPmag"][inside], rp=c["RPmag"][inside], xp=c["XPsamp"][inside])
        pos16 = np.c_[c["RA_ICRS"][inside], c["DE_ICRS"][inside]]     # Gaia's own epoch: the XP extract is there
        add = _bright_missing(ctx, w, W, H, scale, epoch, deep)
        if add is not None:
            deep = {k: np.concatenate([deep[k], add[k].astype(deep[k].dtype)]) for k in deep}
            pos16 = np.r_[pos16, np.full((len(add["ra"]), 2), np.nan)]
        np.savez(ctx.out("gaia_deep.npz"), **deep)
        ctx.metric("stars_deep", int(inside.sum()))
        ctx.log(f"{inside.sum()} Gaia stars G < {BRIGHT_G} in the field")
        if inside.sum() < 100:
            raise StageError("almost no Gaia star in the field: the astrometric solution is probably wrong")

        # ---------------- XP spectra of isolated stars (colour calibration)
        # read once: indexing an npz file reads its array from the archive again at every access (in _isolated_xp's
        # loop over the candidates that cost 4-5 min of CPU per run)
        with np.load(ctx.out("gaia_deep.npz")) as z:
            d = {k: z[k] for k in z.files}
        from ..core import sirilcat
        cut = sirilcat.limits()["xp"]
        if cut is not None and p.xp_gmax > cut[1] + 0.02:
            ctx.log(f"the star spectra installed stop at G {cut[1]:.1f}: the colour stars between it and the faint "
                    f"limit, G {p.xp_gmax:g}, have none (the full star spectra: Settings › Catalogs & astrometry)",
                    "warning")
        sel = _isolated_xp(d, p, W, H, scale)
        ctx.log(f"{len(sel)} isolated stars with an XP spectrum selected for the colour calibration")
        rows = _xp_spectra(ctx, d["source"][sel].astype(np.int64), 0.85, 1.0, pos16[sel], d["g"][sel])
        # however few they are, the Color block judges them (its Minimum stars): it says why there are few, and takes
        # its fallback or stops
        keep = np.array([i for i, r in zip(sel, rows) if r is not None], int)
        got = [r for r in rows if r is not None]
        lam = got[0][0] if got else np.zeros(0)
        flux = np.stack([r[1] for r in got]) if got else np.zeros((0, 0))
        xyz = _xyz(lam, flux) if got else np.zeros((0, 3))
        np.savez(ctx.out("gaia_xp.npz"), source=d["source"][keep], x=d["x"][keep], y=d["y"][keep], g=d["g"][keep],
                 lam=lam, flux=flux, xyz=xyz)
        ctx.metric("xp_spectra", int(len(keep)))


#: the magnitude limit of the stars: the colour stars (to xp_gmax) and their neighbours
BRIGHT_G = 16.0
#: the wavelength grid the sampled XP spectra are brought to when those read do not share one (400-800 nm every 10 nm:
#: the compact extract's and the older cached spectra's)
XP_GRID = np.arange(400.0, 800.0 + 1e-6, 10.0)


def _bright_stars(ctx, centres, radius, cols, fine):
    """The bright stars of the field (G < BRIGHT_G), in the deep catalogue's columns: from the offline Gaia extract
    when it is installed (no network; it has no Gaia ids, colours or XP flags: ids are made negative and unique, the
    colours left unknown, every star a candidate for an XP spectrum, which the extract then has or not), else from the
    sky cache's bright layer (the astrometry's)."""
    from ..core import sirilcat
    if sirilcat.locate()["astro"] is None:
        ctx.log("offline Gaia extract not installed: the bright stars from the sky cache", "warning")
        return _query_cells(ctx, centres, radius, cols, f"<{BRIGHT_G:g}", "bright stars", 0.0, 0.5, fine)
    parts = []
    for ra, dec in np.asarray(centres, np.float64).reshape(-1, 2):
        r = sirilcat.astro_cone(float(ra), float(dec), float(radius), BRIGHT_G)
        if r is not None and len(r["Gmag"]):
            parts.append(r)
    ctx.progress(0.5, "Gaia bright stars: offline extract")
    if not parts:
        return {k: np.zeros(0) for k in cols}
    ra = np.concatenate([q["RA_ICRS"] for q in parts])
    dec = np.concatenate([q["DE_ICRS"] for q in parts])
    key = np.round(ra * 3.6e5).astype(np.int64) * 10 ** 9 + np.round(dec * 3.6e5).astype(np.int64)
    _, idx = np.unique(key, return_index=True)                     # duplicates from overlapping cones
    cat = {k: np.concatenate([q[k] for q in parts])[idx] for k in ("RA_ICRS", "DE_ICRS", "Gmag", "pmRA", "pmDE")}
    n = len(idx)
    complete = min(float(q["complete_to"]) for q in parts)
    ctx.log(f"{n} Gaia stars G < {BRIGHT_G:g} from the offline extract (complete to G {min(complete, BRIGHT_G):.1f})")
    return {**cat, "Source": -np.arange(1, n + 1, dtype=np.int64), "BPmag": np.full(n, np.nan),
            "RPmag": np.full(n, np.nan), "XPsamp": np.ones(n)}


def _epoch(t: str | None) -> float:
    """The observation date as a decimal year, for the stars' proper motions; 2016.0 (Gaia DR3's epoch) when there is
    none or it cannot be read. A date with a time zone ("Z", "+02:00") is taken in UTC (review 4.3: it stopped the
    block, and the colour calibration with it)."""
    if not t:
        return 2016.0
    try:
        dt = datetime.fromisoformat(str(t).strip().replace("Z", "+00:00"))
    except ValueError:
        return 2016.0
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    y0 = datetime(dt.year, 1, 1)
    return dt.year + (dt - y0).total_seconds() / (365.25 * 86400)


def _cells(w, W, H, scale, cell_deg):
    """Cone centres covering the frame (+ margin), laid out on the image grid (no RA wrap / pole issues)."""
    cell_px = cell_deg * 3600 / scale
    m = 0.05 * max(W, H)
    nx = max(1, int(np.ceil((W + 2 * m) / cell_px)))
    ny = max(1, int(np.ceil((H + 2 * m) / cell_px)))
    xs = -m + (np.arange(nx) + 0.5) * (W + 2 * m) / nx
    ys = -m + (np.arange(ny) + 0.5) * (H + 2 * m) / ny
    gx, gy = np.meshgrid(xs, ys)
    sky = w.pixel_to_world(gx.ravel(), gy.ravel())
    step = max((W + 2 * m) / nx, (H + 2 * m) / ny) * scale / 3600
    return np.c_[sky.ra.deg, sky.dec.deg], step * np.sqrt(2) / 2 * 1.05 + 0.01


def _cone_key(cols, gfilter, c, radius) -> str:
    """The cache key of one cell's cone in versions before the sky tiles (their files are still read)."""
    return hashlib.sha1(f"gaiadr3|{','.join(cols)}|{gfilter}|{c[0]:.5f}|{c[1]:.5f}|{radius:.4f}".encode()).hexdigest()


def _query_cells(ctx, centres, radius, cols, gfilter, label, p0, p1, fine=None):
    """Every Gaia star of the cells' cones. A cell an earlier version downloaded with exactly this cone is read from its
    file (a re-run of the same solution needs no network, same stars); the others come from the sky tiles cached by earlier
    versions (``core/skycache.py``) that the ``fine`` cones near them touch: the same cover of the frame with cells
    4 times smaller, which hugs the frame (a quarter fewer tiles to download than the 1-degree cones)."""
    files = [skycache.find(f"gaia/{_cone_key(cols, gfilter, c, radius)}.npz") for c in centres]
    parts = []
    for f in files:
        if f is not None:
            with np.load(f) as z:
                parts.append({k: z[k] for k in cols})
    if parts:
        ctx.log(f"Gaia {label}: {len(parts)} of the {len(centres)} cells read from the cache of an earlier version")
    missing = np.array([c for c, f in zip(centres, files) if f is None]).reshape(-1, 2)
    if len(missing):
        fc, fr = fine if fine is not None else (centres, radius)
        fc = np.asarray(fc, np.float64).reshape(-1, 2)
        # the fine cones within reach of a missing cell: they cover all of it (a point of the cell lies within the
        # cell's radius of its centre, and within the fine radius of a fine centre)
        near = skycache.in_cones(fc[:, 0], fc[:, 1], [(float(c[0]), float(c[1]), float(radius + fr)) for c in missing])
        cones = [(float(c[0]), float(c[1]), float(fr)) for c in fc[near]]

        def prog(f, msg):
            ctx.progress(p0 + (p1 - p0) * f, msg)
        # only the tiles on the disk (a run never downloads): the ones missing fail the stage with their reason
        parts.append(skycache.gaia_stars(cones, cols, gfilter, log=ctx.log, progress=prog, label=f"Gaia {label}"))
    ctx.progress(p1, f"Gaia {label}: read")
    c = {k: np.concatenate([q[k] for q in parts if len(q[k])]) if any(len(q[k]) for q in parts) else np.zeros(0)
         for k in cols}
    # duplicates from overlapping cones: exact position (Gaia ids must stay int64: as floats they collide)
    key = np.round(c["RA_ICRS"] * 3.6e5).astype(np.int64) * 10 ** 9 + np.round(c["DE_ICRS"] * 3.6e5).astype(np.int64)
    _, idx = np.unique(key, return_index=True)
    return {k: v[idx] for k, v in c.items()}


def _bright_missing(ctx, w, W, H, scale, epoch, deep, vmax=9.0):
    """The brightest stars of the sky are partly missing from Gaia DR3 (its detector saturates; Alnitak, by the
    Horsehead, is absent). Missing, such a star is not modelled: its core and its glare stay in the nebula layer.
    Hipparcos stars of the field with no Gaia counterpart are added. Their Gaia magnitude and colour come from V and
    B-V through relations fitted on the Hipparcos stars of the same field that Gaia does have (no assumed
    photometric transformation). Returns the entries to append to the deep catalogue, or None."""
    from scipy.spatial import cKDTree

    centre = w.pixel_to_world(W / 2, H / 2)
    radius = 0.5 * np.hypot(W, H) * scale / 3600 * 1.05
    cols = ["HIP", "RAICRS", "DEICRS", "Vmag", "B-V", "pmRA", "pmDE"]
    try:                                                     # the whole-sky list of the star data on this computer
        t = skycache.hipparcos(list(dict.fromkeys(cols + ["RAICRS", "DEICRS"])), vmax)
    except skycache.SkyUnavailable as e:
        ctx.log(f"{e}: bright stars missing from Gaia are not added", "warning")
        return None
    near = skycache._sep(centre.ra.deg, centre.dec.deg, t["RAICRS"], t["DEICRS"]) <= radius
    t = {k: t[k][near] for k in cols}
    if not len(t["HIP"]):
        return None
    dt = epoch - 1991.25
    dec = t["DEICRS"] + np.nan_to_num(t["pmDE"]) / 3.6e6 * dt
    ra = t["RAICRS"] + np.nan_to_num(t["pmRA"]) / 3.6e6 * dt / np.cos(np.radians(t["DEICRS"]))
    x, y = safe_world2pix(w, ra, dec, W, H)
    V, BV = t["Vmag"], np.nan_to_num(t["B-V"], nan=0.6)
    ok = (x > -30) & (x < W + 30) & (y > -30) & (y < H + 30) & np.isfinite(V)
    if not ok.any():
        return None
    tree = cKDTree(np.c_[deep["x"], deep["y"]])
    rpx = max(5.0 / scale, 2.0)
    gd, col = deep["g"], deep["bp"] - deep["rp"]
    pairs, missing = [], []
    for i in np.where(ok)[0]:
        nb = [j for j in tree.query_ball_point([x[i], y[i]], rpx) if gd[j] < V[i] + 1.0]
        if nb:
            j = min(nb, key=lambda j: gd[j])
            if np.isfinite(col[j]) and np.isfinite(t["B-V"][i]):
                pairs.append((BV[i], V[i], gd[j], col[j]))
        else:
            missing.append(i)
    if not missing:
        return None
    if len(pairs) >= 8:
        P_ = np.array(pairs)
        cg = np.polyfit(P_[:, 0], P_[:, 2] - P_[:, 1], 2)
        cc = np.polyfit(P_[:, 0], P_[:, 3], 2)
        how = f"relations fitted on {len(pairs)} stars of the field"
    else:                                   # too few: G - V of Riello et al. 2021, a mean colour relation
        cg = np.array([-0.2156, 0.01424, -0.02704])
        cc = np.array([0.0, 1.2, 0.08])
        how = "mean relations (too few Hipparcos stars matched in the field)"
    m = np.array(missing)
    g_new = V[m] + np.polyval(cg, BV[m])
    c_new = np.polyval(cc, BV[m])
    for i, gg in zip(m, g_new):
        ctx.log(f"HIP {int(t['HIP'][i])} (V = {V[i]:.2f}) is missing from Gaia: added with G = {gg:.2f} ({how})")
    ctx.metric("bright_stars_added", [int(t["HIP"][i]) for i in m])
    return dict(source=-t["HIP"][m].astype(np.int64), ra=ra[m], dec=dec[m], x=x[m], y=y[m], g=g_new,
                bp=g_new + c_new / 2, rp=g_new - c_new / 2, xp=np.zeros(len(m)))


def _isolated_xp(d, p, W, H, scale):
    from scipy.spatial import cKDTree
    g, x, y = d["g"], d["x"], d["y"]         # once: d may be an npz file, which reads an array at every access
    xp = np.nan_to_num(d["xp"]) == 1
    cand = np.where(xp & (g > p.xp_gmin) & (g < p.xp_gmax) & (x > 60) & (x < W - 60)
                    & (y > 60) & (y < H - 60))[0]
    tree = cKDTree(np.c_[x, y])
    keep = []
    rpx = 40.0 / scale                     # isolation: no star brighter than G + 3 within 40"
    for i in cand:
        nb = tree.query_ball_point([x[i], y[i]], rpx)
        if all(j == i or g[j] >= g[i] + 3 for j in nb):
            keep.append(i)
    keep = np.array(keep, int)
    keep = keep[np.argsort(g[keep])]
    if len(keep) <= p.xp_max:
        return keep
    # spread over the whole magnitude range, not the brightest only (issue #12): on a wide field taken with a camera
    # lens the brightest thousands are all saturated; evenly spaced in magnitude order, the unsaturated ones are there
    return keep[np.linspace(0, len(keep) - 1, p.xp_max).round().astype(int)]


def _xp_spectra(ctx, sources, p0, p1, pos16=None, gmag=None):
    """XP spectra of the stars, in this order: the cache (by Gaia id, kept by earlier versions), then the offline spectra
    extract when installed (by position at epoch 2016.0 and G: it has no ids). None for a star without its spectrum."""
    from ..core import sirilcat
    have = skycache.listing("xp")
    ids = [int(s) for s in sources]
    rows: list = [None] * len(ids)
    known = [False] * len(ids)
    for k, sid in enumerate(ids):
        cp = have.get(f"{sid}.npy") if sid > 0 else None
        if cp is not None:
            a = np.load(cp)
            known[k], rows[k] = True, ((a[0], a[1]) if a.size else None)
    todo = [k for k in range(len(ids)) if not known[k]]
    if todo and pos16 is not None and sirilcat.locate()["xp"]:
        sub = np.asarray(todo)
        got = sirilcat.xp_spectra(pos16[sub, 0], pos16[sub, 1], np.asarray(gmag)[sub], np.asarray(ids)[sub])
        n = 0
        for k, r in zip(todo, got):
            if r is not None:
                rows[k], known[k] = r, True
                n += 1
        ctx.log(f"{n} of {len(todo)} XP spectra read from the offline Gaia extract")
    ctx.progress(p1, f"XP spectra: {sum(known)} of {len(ids)}")
    if len({len(r[0]) for r in rows if r is not None}) > 1:
        ctx.log("XP spectra with different wavelength grids: all brought to 400-800 nm")
        rows = [None if r is None else (XP_GRID.copy(), np.interp(XP_GRID, r[0], r[1])) for r in rows]
    return rows


def _xyz(lam, flux):
    import colour
    cmfs = colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"]
    wl, xb = cmfs.wavelengths, cmfs.values
    s = (lam >= wl.min()) & (lam <= wl.max())
    xbar = np.stack([np.interp(lam[s], wl, xb[:, i]) for i in range(3)], 1)
    return flux[:, s] @ xbar * np.gradient(lam[s])[:, None].mean()
