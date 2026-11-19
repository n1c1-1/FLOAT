"""Stars redrawn on a stars-alone image (issue #77): detected in the image, each with its own flux and colour, and one
star shape for all of them: the shape measured at the centre of the field, or the instrument's theoretical one.

The theoretical shape is computed from the aperture (Fraunhofer: the squared modulus of the Fourier transform of the
pupil, with its central obstruction and spider), per colour and over the width of each colour band, then blurred by
the seeing. The spikes come out of the computation, and their strength follows each star's flux.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.signal import fftconvolve

#: the colour bands of the three channels, nm
BANDS = ((590.0, 680.0), (500.0, 570.0), (420.0, 500.0))
#: the wavelengths sampled in each band (the fringes of one wavelength do not show in a broad band)
BAND_SAMPLES = 5
#: the half-size of the stamps of the measured shape
R_MEASURED = 32
#: stars whose peak is above this share of the brightest are taken as saturated
SATURATED = 0.6
#: a star is detected this many noise units above the sky of the stars-alone image
DETECT_SIGMAS = 8.0
#: a candidate's width, roundness and sharpness may differ from the field's stars of the same brightness by this many
#: robust standard deviations (the 3 sigma limit of the star finders' acceptance ranges)
SHAPE_SIGMAS = 3.0
#: the brightness classes (quantiles of the candidates' peak) in which the field's stars are measured, at most
SHAPE_CLASSES = 10
#: candidates a brightness class needs for the field's stars to say what a star looks like; below this one class's
#: worth of candidates the shape is not judged at all
SHAPE_MIN_STARS = 100


@dataclass
class Instrument:
    """The optics of the theoretical shape. ``obstruction``: the central obstruction's diameter, a share of the
    aperture; ``spider_arms``: 0 for none, 4 gives a cross; ``seeing_px``: the width at half height of the final star."""

    diameter_mm: float
    focal_mm: float
    pixel_um: float
    obstruction: float = 0.0
    spider_arms: int = 0
    spider_mm: float = 1.0
    angle_deg: float = 0.0
    seeing_px: float = 2.5


def pupil(N: int, step: float, ins: Instrument) -> np.ndarray:
    """The aperture on an N x N grid of ``step`` mm: 1 where light passes, its edges softened over one grid step."""
    c = (np.arange(N) - N // 2) * step
    yy, xx = np.meshgrid(c, c, indexing="ij")
    r = np.hypot(yy, xx)
    a = np.clip((ins.diameter_mm / 2 - r) / step + 0.5, 0, 1)
    if ins.obstruction > 0:
        a *= np.clip((r - ins.obstruction * ins.diameter_mm / 2) / step + 0.5, 0, 1)
    for k in range(ins.spider_arms):
        t = math.radians(ins.angle_deg) + 2 * math.pi * k / ins.spider_arms
        along = xx * math.cos(t) + yy * math.sin(t)
        across = np.abs(-xx * math.sin(t) + yy * math.cos(t))
        a *= 1 - np.clip((ins.spider_mm / 2 - across) / step + 0.5, 0, 1) * (along > 0)
    return a.astype(np.float32)


def moffat(fwhm: float, r: int, beta: float = 3.0) -> np.ndarray:
    alpha = fwhm / (2 * math.sqrt(2 ** (1 / beta) - 1))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    k = (1 + (yy ** 2 + xx ** 2) / alpha ** 2) ** -beta
    return k / k.sum()


def theoretical_shape(ins: Instrument, R: int = 300) -> np.ndarray:
    """The star of this instrument (3, 2R+1, 2R+1), each colour summing to 1: the diffraction figure of the aperture,
    averaged over the colour band, on the pixel grid, blurred by the seeing."""
    lam_max = max(b[1] for b in BANDS) * 1e-6                     # mm
    lam_min = min(b[0] for b in BANDS) * 1e-6
    # the image's sample small enough for the pupil to fit twice in the grid, at the shortest wavelength
    os_ = max(3, int(math.ceil(ins.pixel_um * 1e-3 * 2 * ins.diameter_mm / (lam_min * ins.focal_mm))) | 1)
    N = min(8192, 2 * (R + 2) * os_)
    N += N % 2
    R = min(R, N // (2 * os_) - 2)
    delta = ins.pixel_um * 1e-3 / os_                             # mm on the sensor
    out = []
    for lo, hi in BANDS:
        total = 0.0
        for lam_nm in np.linspace(lo, hi, BAND_SAMPLES):
            lam = lam_nm * 1e-6
            step = lam * ins.focal_mm / (N * delta)               # mm of pupil per grid step
            p = pupil(N, step, ins)
            I = np.abs(np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(p)))) ** 2
            total = total + I / I.sum()
        m = N // 2
        half = os_ // 2
        cut = total[m - os_ * R - half:m + os_ * R + half + 1, m - os_ * R - half:m + os_ * R + half + 1]
        binned = cut.reshape(2 * R + 1, os_, 2 * R + 1, os_).sum((1, 3))
        out.append(binned)
    del lam_max
    k = moffat(ins.seeing_px, int(6 * ins.seeing_px) | 1)
    shapes = [np.clip(fftconvolve(c, k, mode="same"), 0, None) for c in out]
    return np.stack([s / s.sum() for s in shapes]).astype(np.float32)


def _noise(a: np.ndarray) -> float:
    return 1.4826 * float(np.median(np.abs(a - np.median(a))))


def _maxima(L: np.ndarray, sigma: float, window: int, T: float) -> tuple[np.ndarray, np.ndarray]:
    s = cv2.GaussianBlur(L, (0, 0), sigma)
    ys, xs = np.nonzero((s == cv2.dilate(s, np.ones((window, window)))) & (s > T))
    return ys, xs


def star_size(L: np.ndarray, ys: np.ndarray, xs: np.ndarray, r: int = 20) -> float:
    """The light's radius of the brightest stars below saturation (the rms radius of their median stamp, pixels): a
    defocused ring has a large one, whatever its hollow centre does to a measure at half height."""
    H, W = L.shape
    sat = SATURATED * float(L.max())
    keep = (ys > r) & (ys < H - r - 1) & (xs > r) & (xs < W - r - 1) & (L[ys, xs] < sat)
    order = np.argsort(-L[ys[keep], xs[keep]])[:300]
    st = np.array([L[y - r:y + r + 1, x - r:x + r + 1] for y, x in zip(ys[keep][order], xs[keep][order])], np.float32)
    if not len(st):
        return 1.5
    st = st / np.maximum(st.sum((1, 2), keepdims=True), 1e-9)
    m = np.clip(np.median(st, 0), 0, None)
    m = m - np.median(np.r_[m[0], m[-1], m[:, 0], m[:, -1]])
    m = np.clip(m, 0, None)
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    return float(np.sqrt((m * (yy ** 2 + xx ** 2)).sum() / max(m.sum(), 1e-9) / 2))


def detect(L: np.ndarray, valid: np.ndarray, margin: int = 34):
    """The stars of a stars-alone luminance: (ys, xs, peaks, threshold, size). Peaks are looked for at the scale of the
    stars, so that a defocused ring is one star and not several."""
    sub = L[::4, ::4]
    noise = _noise(sub)
    T = max(DETECT_SIGMAS * noise, float(np.quantile(sub, 0.99999)) * 4e-4)
    ys, xs = _maxima(L, 1.0, 5, T)
    wide_y, wide_x = _maxima(L, 6.0, 31, T * 0.1)                 # centred on a whole star, even a hollow one
    size = star_size(L, wide_y, wide_x)
    if size > 3.0:
        ys, xs = _maxima(L, 1.2 * size, int(2.5 * size) | 1, T * 0.1)
    good = cv2.erode(valid.astype(np.uint8), np.ones((81, 81))) > 0
    H, W = L.shape
    k = (ys > margin) & (ys < H - margin) & (xs > margin) & (xs < W - margin) & good[ys, xs]
    return ys[k], xs[k], L[ys[k], xs[k]], T, size


def _stamps(L: np.ndarray, ys: np.ndarray, xs: np.ndarray, R: int) -> np.ndarray:
    off = np.arange(-R, R + 1)
    return L[ys[:, None, None] + off[None, :, None], xs[:, None, None] + off[None, None, :]]


def measure(L: np.ndarray, ys: np.ndarray, xs: np.ndarray, size: float) -> dict[str, np.ndarray]:
    """What each candidate is made of, from its stamp minus the local sky (the median of its outer ring): ``sigma``
    the width (the Gaussian-windowed second moments, as the extractors' windowed centroid does), ``ell`` the
    elongation 1 - b/a of the same moments (the roundness limit of the star finders), ``sharp`` the share of the light
    in the centre (DAOFIND's sharpness: a star is peaked, a texture of the nebula is flat), ``peak`` the height above
    the sky and ``sky`` its noise. Candidates too near the edge have all of them at zero."""
    H, W = L.shape
    R = max(5, int(math.ceil(3 * size)))
    n = len(ys)
    out = {k: np.zeros(n, np.float32) for k in ("sigma", "ell", "sharp", "peak", "noise")}
    ok = (ys > R) & (ys < H - R - 1) & (xs > R) & (xs < W - R - 1)
    yy, xx = np.mgrid[-R:R + 1, -R:R + 1].astype(np.float32)
    rr = np.hypot(yy, xx)
    ring = (rr > R - 1.5) & (rr <= R)
    win = np.exp(-0.5 * (rr / size) ** 2)
    core = rr <= 0.5 * R + 0.5
    idx = np.nonzero(ok)[0]
    for a in range(0, len(idx), 20000):
        i = idx[a:a + 20000]
        P = _stamps(L, ys[i], xs[i], R)
        sky = np.median(P[:, ring], 1)
        out["noise"][i] = 1.4826 * np.median(np.abs(P[:, ring] - sky[:, None]), 1)
        Q = P - sky[:, None, None]
        m = np.clip(Q, 0, None) * win
        t = np.maximum(m.sum((1, 2)), 1e-9)
        cy = (m * yy).sum((1, 2)) / t
        cx = (m * xx).sum((1, 2)) / t
        syy = (m * yy ** 2).sum((1, 2)) / t - cy ** 2
        sxx = (m * xx ** 2).sum((1, 2)) / t - cx ** 2
        sxy = (m * yy * xx).sum((1, 2)) / t - cy * cx
        tr, d = sxx + syy, np.sqrt(np.maximum((sxx - syy) ** 2 + 4 * sxy ** 2, 0))
        a_ax = np.sqrt(np.maximum((tr + d) / 2, 1e-9))
        b_ax = np.sqrt(np.maximum((tr - d) / 2, 1e-9))
        out["sigma"][i] = np.sqrt(np.maximum(tr / 2, 0))
        out["ell"][i] = 1 - b_ax / a_ax
        out["peak"][i] = Q[:, R, R]
        out["sharp"][i] = Q[:, R, R] / np.maximum(np.clip(Q, 0, None)[:, core].sum(1), 1e-9)
    return out


def _outliers(v: np.ndarray, cls: np.ndarray, both: bool) -> np.ndarray:
    """True where ``v`` stands out from the candidates of its own brightness class, by more than SHAPE_SIGMAS of the
    class's robust deviation (1.4826 MAD): the field's own stars say what a star of that brightness looks like."""
    bad = np.zeros(len(v), bool)
    for c in np.unique(cls):
        m = cls == c
        med = float(np.median(v[m]))
        s = max(1.4826 * float(np.median(np.abs(v[m] - med))), 1e-6)
        bad[m] = (v[m] - med) > SHAPE_SIGMAS * s
        if both:
            bad[m] |= (med - v[m]) > SHAPE_SIGMAS * s
    return bad


def vet(L: np.ndarray, starless: np.ndarray | None, ys: np.ndarray, xs: np.ndarray, size: float) -> tuple[np.ndarray, dict]:
    """Which of the detected maxima are stars (a boolean per candidate) and how many fell for each reason.

    The detection takes any bump of the stars-alone image for a star, and the star network leaves some of the
    nebula's texture in it. What is kept follows the star finders' acceptance (DAOFIND's roundness and sharpness
    limits, the FWHM and roundness ranges of a point-source finder around the field's median, SExtractor's
    extendedness: a point source has the width of the field's stars): each candidate is measured (``measure``) and
    must stay within SHAPE_SIGMAS robust deviations of the field's stars of its own brightness for its width, its
    elongation (a blend of two touching stars is elongated: it is not drawn as one star) and its sharpness. It must
    also stand above the noise of the place: the sky's, and the structure of the STARLESS image at the scale of the
    stars (what the network may have left of it in the stars layer), as DETECT_SIGMAS of that noise: that is the
    block's own detection threshold (the one the candidates already passed against the sky's noise), applied against
    the noise of the place instead of the sky's, so no second constant exists; where the starless image has no
    structure beyond its usual pixel noise it changes nothing. The shape is not judged below SHAPE_MIN_STARS
    candidates, the statistical minimum for the field's stars to define what a star looks like. No value comes
    from an image: the limits are measured on the field's own stars."""
    n = len(ys)
    keep = np.ones(n, bool)
    why = {"stars_not_stellar": 0, "stars_on_texture": 0}
    if not n:
        return keep, why
    ms = measure(L, ys, xs, size)
    edge = ms["peak"] <= 0
    n_ok = int((~edge).sum())
    odd = np.zeros(n, bool)
    if n_ok >= SHAPE_MIN_STARS:
        classes = int(min(SHAPE_CLASSES, n_ok // SHAPE_MIN_STARS))
        cls = np.digitize(ms["peak"], np.quantile(ms["peak"][~edge], np.linspace(0, 1, classes + 1)[1:-1]))
        odd = _outliers(ms["sigma"], cls, True) | _outliers(ms["ell"], cls, False) | _outliers(ms["sharp"], cls, True)
        odd &= ~edge
    keep &= ~odd
    why["stars_not_stellar"] = int(odd.sum())
    if starless is not None:
        hp = starless - cv2.GaussianBlur(starless, (0, 0), 3.0 * size)
        rms = np.sqrt(cv2.GaussianBlur(hp * hp, (0, 0), 4.5 * size))
        # the starless image's own pixel noise, the field's usual value of that structure, is not texture
        local = np.sqrt(np.maximum(rms[ys, xs] ** 2 - float(np.median(rms[::8, ::8])) ** 2, 0.0))
        sky = float(np.median(ms["noise"][~edge])) if (~edge).any() else 0.0
        tex = ms["peak"] < DETECT_SIGMAS * np.sqrt(sky ** 2 + local ** 2)
        tex &= ~edge
        why["stars_on_texture"] = int((tex & keep).sum())
        keep &= ~tex
    return keep, why


def reaches(shape: np.ndarray, flux: np.ndarray, floor: float) -> np.ndarray:
    """The footprint radius of each star: where the drawn shape, at its flux, falls under ``floor`` at every angle
    (``_reach``, for many stars at once; the radius the shape measured on the field has, spikes of the real star
    excepted: they are its own, too)."""
    three = shape if shape.ndim == 3 else shape[None]
    env_img = three.max(0)
    R = env_img.shape[-1] // 2
    yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
    rings = np.rint(np.hypot(yy, xx)).astype(int).ravel()
    env = np.zeros(R + 2, np.float32)
    np.maximum.at(env, np.minimum(rings, R + 1), env_img.ravel())
    over = env[:R + 1, None] * np.asarray(flux, np.float32)[None, :] > floor
    last = np.where(over.any(0), R - np.argmax(over[::-1], 0), 2)
    return np.clip(last, 4, R)


def carried(s: np.ndarray, out: np.ndarray, ys, xs, peaks, star, shape, floor: float, size: float,
            starless: np.ndarray | None = None) -> tuple[np.ndarray, int, int]:
    """The light of the stars layer that is not a star's, to go on with the starless image unchanged: what is left
    of ``s`` (the stars layer) where ``out`` (the synthetic layer) does not hold it, round the candidates that are not
    stars. A candidate lying inside the footprint of a brighter star (``reaches``: the radius its own shape has at its
    flux, measured on the field) is a bead of that star's halo or spikes: it goes with the star, which the drawing
    replaces. Where the starless image itself has light (an extended object: its mean at the scale of the
    stars above its sky by DETECT_SIGMAS of that mean's own noise, the detection's threshold) the light the stars layer
    holds there is carried too (what is left of it once the drawn stars are taken off): it is the object's, left in the stars layer by
    the star network. Returns (light (3, H, W), candidates carried, candidates left with their star)."""
    from scipy.spatial import cKDTree
    H, W = s.shape[1:]
    rej = np.nonzero(~star)[0]
    st = np.nonzero(star)[0]
    flux = peaks[st] / max(float(np.max(shape)), 1e-9)
    rad = reaches(shape, flux, floor)
    tree = cKDTree(np.c_[ys[st], xs[st]])
    inside = np.zeros(len(rej), bool)
    near = tree.query_ball_point(np.c_[ys[rej], xs[rej]], float(rad.max()) if rad.size else 0.0) if rej.size else []
    for k, js in enumerate(near):
        for j in js:
            if peaks[st[j]] > peaks[rej[k]] and np.hypot(ys[st[j]] - ys[rej[k]], xs[st[j]] - xs[rej[k]]) <= rad[j]:
                inside[k] = True
                break
    free = rej[~inside]
    m = np.zeros((H, W), np.uint8)
    r = int(math.ceil(3.0 * size))
    for y, x in zip(ys[free].tolist(), xs[free].tolist()):
        cv2.circle(m, (x, y), r, 1, -1)
    m = cv2.GaussianBlur(m.astype(np.float32), (0, 0), max(1.0, size)).clip(0, 1)
    if starless is not None:
        sm = cv2.GaussianBlur(starless, (0, 0), 4.0 * size)
        sub = sm[::8, ::8]
        sky = float(np.median(sub))
        noise = max(_noise(sub), 1e-9)
        neb = cv2.GaussianBlur((sm > sky + DETECT_SIGMAS * noise).astype(np.float32), (0, 0), max(1.0, size)).clip(0, 1)
        m = np.maximum(m, neb)
    light = np.clip(s - out, 0, None) * m
    return light.astype(np.float32), int(free.size), int(inside.sum())


def _centre(p: np.ndarray, rad: float) -> tuple[float, float]:
    n = p.shape[0] // 2
    yy, xx = np.mgrid[-n:n + 1, -n:n + 1]
    q = np.clip(p - np.median(np.r_[p[0], p[-1], p[:, 0], p[:, -1]]), 0, None) * (np.hypot(yy, xx) < rad)
    t = max(float(q.sum()), 1e-9)
    return float((q * yy).sum() / t), float((q * xx).sum() / t)


def moved(p: np.ndarray, dy: float, dx: float) -> np.ndarray:
    """``p`` shifted by (-dy, -dx): a star (dy, dx) off the centre is brought to it."""
    M = np.float32([[1, 0, -dx], [0, 1, -dy]])
    return cv2.warpAffine(p, M, (p.shape[1], p.shape[0]), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT)


def measured_shape(L: np.ndarray, ys, xs, peaks, R: int = R_MEASURED, centre_share: float = 0.35) -> np.ndarray | None:
    """The median of the isolated unsaturated stars near the centre of the field, (2R+1, 2R+1), summing to 1."""
    H, W = L.shape
    yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
    rr = np.hypot(yy, xx)
    rc = np.hypot(ys - H / 2, xs - W / 2) / np.hypot(H / 2, W / 2)
    sat = SATURATED * float(L.max())
    done = np.zeros((H, W), bool)
    stamps = []
    for i in np.argsort(-peaks):
        y, x = int(ys[i]), int(xs[i])
        if rc[i] > centre_share or peaks[i] > sat or done[y, x]:
            continue
        done[max(0, y - 40):y + 40, max(0, x - 40):x + 40] = True
        p = np.asarray(L[y - R:y + R + 1, x - R:x + R + 1], np.float32)
        if p[rr > 20].max() > 0.03 * p.max():
            continue
        dy, dx = _centre(p, 6.0)
        st = moved(p, dy, dx)                           # the star brought to the stamp's centre
        st = st - np.median(st[rr > 28])
        stamps.append(st / max(float(st[rr < 24].sum()), 1e-9))
        if len(stamps) >= 300:
            break
    if len(stamps) < 5:
        return None
    m = np.median(np.array(stamps), 0)
    m = np.clip(m - np.median(m[rr > 28]), 0, None)
    return (m / m.sum()).astype(np.float32)


def flux_of(s: np.ndarray, y: int, x: int, r: int) -> np.ndarray:
    """The star's light per colour in an aperture of radius ~r, its sky taken on the stamp's edge."""
    st = s[:, y - r:y + r + 1, x - r:x + r + 1]
    bg = np.median(np.c_[st[:, 0], st[:, -1]], axis=1)[:, None, None]
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    return np.maximum(((st - bg) * (np.hypot(yy, xx) < r * 0.85)).sum((1, 2)), 0)


def saturated_fluxes(ys, xs, sat_mask, fluxes, gaia, reach: float = 4.0) -> tuple[np.ndarray, int]:
    """The saturated stars' light from the catalogue: the image's zero point on the unsaturated stars matched to
    Gaia, the colour of the unsaturated stars of the nearest BP-RP. ``gaia``: {x, y, g, bp, rp}. Returns the fluxes,
    and how many saturated stars the catalogue gave."""
    from scipy.spatial import cKDTree
    out = fluxes.copy()
    ok = np.isfinite(gaia["g"])                                  # (the offline extract gives no colour)
    gx, gy, g, col = gaia["x"][ok], gaia["y"][ok], gaia["g"][ok], (gaia["bp"] - gaia["rp"])[ok]
    tree = cKDTree(np.c_[gx, gy])
    d, j = tree.query(np.c_[xs, ys], distance_upper_bound=reach)
    hit = np.isfinite(d)
    ref = hit & ~sat_mask & (fluxes[1] > 0)
    if ref.sum() < 10:
        return out, 0
    zp = float(np.median(2.5 * np.log10(fluxes[1][ref]) + g[j[ref]]))
    cols = col[j[ref]]
    ratios = np.stack([fluxes[0][ref] / fluxes[1][ref], fluxes[2][ref] / fluxes[1][ref]])
    n = 0
    for i in np.nonzero(sat_mask & hit)[0]:
        near = np.argsort(np.abs(cols - col[j[i]]))[:30] if np.isfinite(col[j[i]]) else np.arange(len(cols))
        fg = 10 ** ((zp - g[j[i]]) / 2.5)
        out[:, i] = fg * float(np.median(ratios[0, near])), fg, fg * float(np.median(ratios[1, near]))
        n += 1
    return out, n


def _reach(shape: np.ndarray, flux: float, floor: float) -> int:
    """The radius a star of this flux needs: past it, the shape's light is under ``floor`` at every angle."""
    R = shape.shape[-1] // 2
    yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
    rings = np.rint(np.hypot(yy, xx)).astype(int).ravel()
    env = np.zeros(R + 2, np.float32)
    np.maximum.at(env, np.minimum(rings, R + 1), shape.ravel())
    over = np.nonzero(env[:R + 1] * flux > floor)[0]
    return int(np.clip(over.max() if over.size else 2, 4, R))


def _centres(L, ys, xs, size):
    r = max(6, int(2.5 * size))
    H, W = L.shape
    return [_centre(L[y - r:y + r + 1, x - r:x + r + 1], r) if r < y < H - r and r < x < W - r else (0.0, 0.0)
            for y, x in zip(ys.tolist(), xs.tolist())]


def _apertures(s, ys, xs, peaks, T, size):
    out = np.zeros((3, len(ys)), np.float32)
    H, W = s.shape[1:]
    for i, (y, x, p) in enumerate(zip(ys.tolist(), xs.tolist(), peaks.tolist())):
        r = min(max(int(3 * size), 14 if p > T * 20 else 7), y, x, H - y - 1, W - x - 1)
        out[:, i] = flux_of(s, y, x, r)
    return out


def from_the_image(s: np.ndarray, valid: np.ndarray, starless: np.ndarray | None = None) -> tuple[np.ndarray | None, dict]:
    """The stars redrawn with the shape measured at the centre of the field; the brightest and the saturated stay
    real, with all their light above the threshold and a very soft edge (their spikes are not in the shape's reach)."""
    L = s.mean(0)
    ys, xs, peaks, T, size = detect(L, valid)
    if len(ys):
        is_star, why = vet(L, None if starless is None else starless.mean(0), ys, xs, size)
    else:
        is_star, why = np.zeros(0, bool), {}
    shape = measured_shape(L, ys[is_star], xs[is_star], peaks[is_star]) if is_star.any() else None
    if shape is None:
        return None, {"stars_drawn": 0}
    rys, rxs, rpk = ys[~is_star], xs[~is_star], peaks[~is_star]
    ys, xs, peaks = ys[is_star], xs[is_star], peaks[is_star]
    real = (peaks >= np.quantile(peaks, 0.998)) | (peaks > SATURATED * float(L.max()))
    _n, lab = cv2.connectedComponents((L > T * 0.35).astype(np.uint8), connectivity=8)
    ids = np.unique(lab[ys[real], xs[real]])
    region = np.isin(lab, ids[ids > 0]).astype(np.uint8)
    mask = cv2.GaussianBlur(cv2.dilate(region, np.ones((31, 31))).astype(np.float32), (0, 0), 12)
    mask = np.maximum(mask, region.astype(np.float32))
    k = ~real
    fl = _apertures(s, ys[k], xs[k], peaks[k], T, size)
    drawn = draw(*L.shape, ys[k], xs[k], _centres(L, ys[k], xs[k], size), fl, shape, T * 0.05)
    out = drawn * (1 - mask) + s * mask
    light, n_carried, n_with = carried(s, out, np.r_[ys, rys], np.r_[xs, rxs], np.r_[peaks, rpk],
                                       np.r_[np.ones(len(ys), bool), np.zeros(len(rys), bool)], shape, T * 0.05, size,
                                       None if starless is None else starless.mean(0))
    return out, {"stars_drawn": int(k.sum()), "stars_kept": int(real.sum()), "stars_carried": n_carried,
                 "stars_with_their_star": n_with, **why, "carried": light}


def from_the_instrument(s: np.ndarray, valid: np.ndarray, ins: Instrument, gaia: dict | None,
                        starless: np.ndarray | None = None) -> tuple[np.ndarray | None, dict]:
    """Every detected star redrawn with the instrument's theoretical shape, the saturated ones from the catalogue."""
    L = s.mean(0)
    ys, xs, peaks, T, size = detect(L, valid)
    if not len(ys):
        return None, {"stars_drawn": 0}
    is_star, why = vet(L, None if starless is None else starless.mean(0), ys, xs, size)
    rys, rxs, rpk = ys[~is_star], xs[~is_star], peaks[~is_star]
    ys, xs, peaks = ys[is_star], xs[is_star], peaks[is_star]
    if not len(ys):
        return None, {"stars_drawn": 0}
    fl = _apertures(s, ys, xs, peaks, T, size)
    sat = peaks > SATURATED * float(L.max())
    n = 0
    if gaia is not None:
        fl, n = saturated_fluxes(ys, xs, sat, fl, gaia, max(4.0, 2.5 * size))
    shape = theoretical_shape(ins)
    drawn = draw(*L.shape, ys, xs, _centres(L, ys, xs, size), fl, shape, T * 0.05)
    light, n_carried, n_with = carried(s, drawn, np.r_[ys, rys], np.r_[xs, rxs], np.r_[peaks, rpk],
                                       np.r_[np.ones(len(ys), bool), np.zeros(len(rys), bool)], shape, T * 0.05, size,
                                       None if starless is None else starless.mean(0))
    return drawn, {"stars_drawn": len(ys), "stars_saturated": int(sat.sum()), "stars_from_catalogue": n,
                   "stars_carried": n_carried, "stars_with_their_star": n_with, **why, "carried": light}


def draw(H: int, W: int, ys, xs, centres, fluxes, shapes, floor: float) -> np.ndarray:
    """The stars on an empty image (3, H, W). ``shapes``: (3, k, k), or (k, k) for one shape in every colour."""
    out = np.zeros((3, H, W), np.float32)
    three = shapes if shapes.ndim == 3 else np.stack([shapes] * 3)
    R = three.shape[-1] // 2
    env = three.max(0)
    for i in np.argsort(fluxes.sum(0)):
        fl = fluxes[:, i]
        y, x = int(ys[i]), int(xs[i])
        r = _reach(env, float(fl.max()), floor)
        r = min(r, y, x, H - y - 1, W - x - 1)
        if r < 2:
            continue
        dy, dx = centres[i]
        for c in range(3):
            st = three[c, R - r:R + r + 1, R - r:R + r + 1]
            st = moved(st, -dy, -dx) if r <= 64 else st     # the centred shape taken to the star's place
            out[c, y - r:y + r + 1, x - r:x + r + 1] += fl[c] * st
    return out
