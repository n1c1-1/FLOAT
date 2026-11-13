"""The Classic stacking method's per-pixel work for the CPU, compiled by Numba: RCD demosaicing, bad-sample filling,
the saturation flags, and the resampling onto the reference grid with the normalisation. They compute what the PyTorch
code of core/debayer.py, core/resample.py and core/stack_classic.py computes (the same formulas in the same order, float32
where those are; the registration polynomial and the Lanczos weights in float64), one loop per task over all the cores
instead of generic array operations that each walk the 12 million samples and allocate a temporary array of their size
(RCD alone held about 40 arrays of a tile per tile of a batch). tests/test_classic_kernels.py keeps the two the same.

``cache=True``: compiled once, kept next to the module.
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

#: False: the callers use their PyTorch code on the CPU too (tests/test_classic_kernels.py compares the two)
ENABLED = True

F32 = np.float32
EPS = F32(1e-5)
EPSSQ = F32(1e-10)
HALF = F32(0.5)
QUART = F32(0.25)
THREE = F32(3.0)
SIX = F32(6.0)
ZERO = F32(0.0)
ONE = F32(1.0)
SCALE = F32(65536.0)
TILE = 194                       # core/debayer.py's TILE, TILE_BORDER (the tests check they agree)
TILE_BORDER = 9
STEP = TILE - 2 * TILE_BORDER
PAD = 4                          # the tile's arrays carry 4 zero samples around (the PyTorch code's _shifter)


# ------------------------------------------------------------------------------------------------ RCD
@njit(nogil=True, cache=True, error_model="numpy")
def _rcd_tile(cn, r0, c0, tr, tc, col, out, H, W):
    """rcd.cc on one tile: the tile at (r0, c0) of the normalised padded frame ``cn`` (real size tr x tc), its result
    (the scale of rcd.cc's output, x 65536) written to the inner STEP x STEP part at the frame position of ``out``."""
    T = TILE
    P = T + 2 * PAD
    o = PAD
    c = np.zeros((P, P), np.float32)
    for y in range(T):
        for x in range(T):
            c[o + y, o + x] = cn[r0 + y, c0 + x]
    v = np.zeros((P, P), np.float32)
    h = np.zeros((P, P), np.float32)
    # ---- step 1
    for y in range(T):
        yy = o + y
        for x in range(T):
            xx = o + x
            cv = c[yy, xx]
            a = (c[yy - 3, xx] - c[yy - 1, xx] - c[yy + 1, xx] + c[yy + 3, xx]) - THREE * (c[yy - 2, xx] + c[yy + 2, xx]) \
                + SIX * cv
            b = (c[yy, xx - 3] - c[yy, xx - 1] - c[yy, xx + 1] + c[yy, xx + 3]) - THREE * (c[yy, xx - 2] + c[yy, xx + 2]) \
                + SIX * cv
            v[yy, xx] = a * a
            h[yy, xx] = b * b
    VH = np.zeros((P, P), np.float32)
    for y in range(4, tr - 4):
        yy = o + y
        for x in range(4, tc - 4):
            xx = o + x
            V = max((v[yy - 1, xx] + v[yy, xx]) + v[yy + 1, xx], EPSSQ)
            Hs = max((h[yy, xx - 1] + h[yy, xx]) + h[yy, xx + 1], EPSSQ)
            VH[yy, xx] = V / (V + Hs)
    vhd = np.zeros((P, P), np.float32)
    for y in range(T):
        yy = o + y
        for x in range(T):
            xx = o + x
            n = QUART * ((VH[yy - 1, xx - 1] + VH[yy - 1, xx + 1]) + (VH[yy + 1, xx - 1] + VH[yy + 1, xx + 1]))
            d = VH[yy, xx]
            vhd[yy, xx] = n if abs(HALF - d) < abs(HALF - n) else d
    # ---- step 2: the low-pass at the red and blue samples
    lpf = np.zeros((P, P), np.float32)
    for y in range(2, tr - 2):
        yy = o + y
        for x in range(2, tc - 2):
            xx = o + x
            if col[y, x] == 1:
                continue
            lpf[yy, xx] = c[yy, xx] + HALF * (((c[yy - 1, xx] + c[yy + 1, xx]) + c[yy, xx - 1]) + c[yy, xx + 1]) + \
                QUART * (((c[yy - 1, xx - 1] + c[yy - 1, xx + 1]) + c[yy + 1, xx - 1]) + c[yy + 1, xx + 1])
    # ---- step 3: green at the red and blue samples
    g = c.copy()
    for y in range(4, tr - 4):
        yy = o + y
        for x in range(4, tc - 4):
            if col[y, x] == 1:
                continue
            xx = o + x
            cv = c[yy, xx]
            n1, s1, w1, e1 = c[yy - 1, xx], c[yy + 1, xx], c[yy, xx - 1], c[yy, xx + 1]
            n2, s2, w2, e2 = c[yy - 2, xx], c[yy + 2, xx], c[yy, xx - 2], c[yy, xx + 2]
            ns = abs(n1 - s1)
            we = abs(w1 - e1)
            n_grad = EPS + (ns + abs(cv - n2)) + (abs(n1 - c[yy - 3, xx]) + abs(n2 - c[yy - 4, xx]))
            s_grad = EPS + (ns + abs(cv - s2)) + (abs(s1 - c[yy + 3, xx]) + abs(s2 - c[yy + 4, xx]))
            w_grad = EPS + (we + abs(cv - w2)) + (abs(w1 - c[yy, xx - 3]) + abs(w2 - c[yy, xx - 4]))
            e_grad = EPS + (we + abs(cv - e2)) + (abs(e1 - c[yy, xx + 3]) + abs(e2 - c[yy, xx + 4]))
            lp = lpf[yy, xx]
            l2 = lp + lp
            n_est = n1 * l2 / (EPS + lp + lpf[yy - 2, xx])
            s_est = s1 * l2 / (EPS + lp + lpf[yy + 2, xx])
            w_est = w1 * l2 / (EPS + lp + lpf[yy, xx - 2])
            e_est = e1 * l2 / (EPS + lp + lpf[yy, xx + 2])
            v_est = (s_grad * n_est + n_grad * s_est) / (n_grad + s_grad)
            h_est = (w_grad * e_est + e_grad * w_est) / (e_grad + w_grad)
            g[yy, xx] = vhd[yy, xx] * (h_est - v_est) + v_est
    # ---- step 4.0 / 4.1: the diagonal high-pass at the odd columns, read at x | 1; the P / Q discrimination
    Pm = np.zeros((P, P), np.float32)
    Qm = np.zeros((P, P), np.float32)
    for y in range(3, tr - 3):
        yy = o + y
        for x in range(3, tc - 3):
            if x % 2 == 0:
                continue
            xx = o + x
            cv = c[yy, xx]
            a = (c[yy - 3, xx - 3] - c[yy - 1, xx - 1] - c[yy + 1, xx + 1] + c[yy + 3, xx + 3]) - \
                THREE * (c[yy - 2, xx - 2] + c[yy + 2, xx + 2]) + SIX * cv
            b = (c[yy - 3, xx + 3] - c[yy - 1, xx + 1] - c[yy + 1, xx - 1] + c[yy + 3, xx - 3]) - \
                THREE * (c[yy - 2, xx + 2] + c[yy + 2, xx - 2]) + SIX * cv
            Pm[yy, xx] = a * a
            Qm[yy, xx] = b * b
    Pr = np.zeros((P, P), np.float32)
    Qr = np.zeros((P, P), np.float32)
    for y in range(T):
        for x in range(T):
            Pr[o + y, o + x] = Pm[o + y, o + (x // 2) * 2 + 1]
            Qr[o + y, o + x] = Qm[o + y, o + (x // 2) * 2 + 1]
    PQ = np.zeros((P, P), np.float32)
    for y in range(4, tr - 4):
        yy = o + y
        for x in range(4, tc - 4):
            if col[y, x] == 1:
                continue
            xx = o + x
            p_stat = max((Pr[yy - 1, xx - 1] + Pr[yy, xx]) + Pr[yy + 1, xx + 1], EPSSQ)
            q_stat = max((Qr[yy - 1, xx + 1] + Qr[yy, xx]) + Qr[yy + 1, xx - 1], EPSSQ)
            PQ[yy, xx] = p_stat / (p_stat + q_stat)
    # ---- step 4.2: red at blue and blue at red
    opp = np.zeros((P, P), np.float32)
    for y in range(4, tr - 4):
        yy = o + y
        for x in range(4, tc - 4):
            if col[y, x] == 1:
                continue
            xx = o + x
            pn = QUART * (((PQ[yy - 1, xx - 1] + PQ[yy - 1, xx + 1]) + PQ[yy + 1, xx - 1]) + PQ[yy + 1, xx + 1])
            pd = PQ[yy, xx]
            pq = pn if abs(HALF - pd) < abs(HALF - pn) else pd
            gv = g[yy, xx]
            nw, ne, sw, se = c[yy - 1, xx - 1], c[yy - 1, xx + 1], c[yy + 1, xx - 1], c[yy + 1, xx + 1]
            d1 = abs(nw - se)
            d2 = abs(ne - sw)
            nw_grad = EPS + d1 + abs(nw - c[yy - 3, xx - 3]) + abs(gv - g[yy - 2, xx - 2])
            ne_grad = EPS + d2 + abs(ne - c[yy - 3, xx + 3]) + abs(gv - g[yy - 2, xx + 2])
            sw_grad = EPS + d2 + abs(sw - c[yy + 3, xx - 3]) + abs(gv - g[yy + 2, xx - 2])
            se_grad = EPS + d1 + abs(se - c[yy + 3, xx + 3]) + abs(gv - g[yy + 2, xx + 2])
            nw_est, ne_est = nw - g[yy - 1, xx - 1], ne - g[yy - 1, xx + 1]
            sw_est, se_est = sw - g[yy + 1, xx - 1], se - g[yy + 1, xx + 1]
            p_est = (nw_grad * se_est + se_grad * nw_est) / (nw_grad + se_grad)
            q_est = (ne_grad * sw_est + sw_grad * ne_est) / (ne_grad + sw_grad)
            opp[yy, xx] = gv + (pq * (q_est - p_est) + p_est)
    # ---- step 4.3: red and blue at the green samples; the result written out
    XR = np.zeros((P, P), np.float32)
    XB = np.zeros((P, P), np.float32)
    for y in range(T):
        for x in range(T):
            k = col[y, x]
            yy, xx = o + y, o + x
            if k == 0:
                XR[yy, xx] = c[yy, xx]
                XB[yy, xx] = opp[yy, xx]
            elif k == 2:
                XB[yy, xx] = c[yy, xx]
                XR[yy, xx] = opp[yy, xx]
    for y in range(STEP):
        gy = TILE_BORDER + r0 + y
        if gy >= H:
            break
        yy = o + TILE_BORDER + y
        for x in range(STEP):
            gx = TILE_BORDER + c0 + x
            if gx >= W:
                break
            xx = o + TILE_BORDER + x
            ty, tx = TILE_BORDER + y, TILE_BORDER + x
            gv = g[yy, xx]
            r_out = XR[yy, xx]
            b_out = XB[yy, xx]
            if col[ty, tx] == 1 and 4 <= ty < tr - 4 and 4 <= tx < tc - 4:
                n1g = EPS + abs(gv - g[yy - 2, xx])
                s1g = EPS + abs(gv - g[yy + 2, xx])
                w1g = EPS + abs(gv - g[yy, xx - 2])
                e1g = EPS + abs(gv - g[yy, xx + 2])
                vd = vhd[yy, xx]
                for which in range(2):
                    X = XR if which == 0 else XB
                    sn = abs(X[yy - 1, xx] - X[yy + 1, xx])
                    ew = abs(X[yy, xx - 1] - X[yy, xx + 1])
                    n_grad = n1g + sn + abs(X[yy - 1, xx] - X[yy - 3, xx])
                    s_grad = s1g + sn + abs(X[yy + 1, xx] - X[yy + 3, xx])
                    w_grad = w1g + ew + abs(X[yy, xx - 1] - X[yy, xx - 3])
                    e_grad = e1g + ew + abs(X[yy, xx + 1] - X[yy, xx + 3])
                    n_est = X[yy - 1, xx] - g[yy - 1, xx]
                    s_est = X[yy + 1, xx] - g[yy + 1, xx]
                    w_est = X[yy, xx - 1] - g[yy, xx - 1]
                    e_est = X[yy, xx + 1] - g[yy, xx + 1]
                    v_est = (n_grad * s_est + s_grad * n_est) / (n_grad + s_grad)
                    h_est = (e_grad * w_est + w_grad * e_est) / (e_grad + w_grad)
                    val = gv + (vd * (h_est - v_est) + v_est)
                    if which == 0:
                        r_out = val
                    else:
                        b_out = val
            r_out = r_out * SCALE
            b_out = b_out * SCALE
            gv = gv * SCALE
            out[0, gy, gx] = r_out if not r_out < ZERO else ZERO
            out[1, gy, gx] = gv if not gv < ZERO else ZERO
            out[2, gy, gx] = b_out if not b_out < ZERO else ZERO


@njit(nogil=True, parallel=True, cache=True, error_model="numpy")
def rcd_tiles(cn, col, nth, ntw, H, W, out):
    """RCD of the normalised padded frame ``cn`` ((nth STEP + 18, ntw STEP + 18), 0..1) by tiles of TILE, all the cores
    on the tiles; ``col`` (TILE, TILE) the colour of each tile pixel. Fills ``out`` (3, H, W), zeroed beforehand, but
    for its outer TILE_BORDER samples."""
    for t in prange(nth * ntw):
        ti, tj = t // ntw, t % ntw
        r0, c0 = ti * STEP, tj * STEP
        _rcd_tile(cn, r0, c0, min(H - r0, TILE), min(W - c0, TILE), col, out, H, W)


@njit(nogil=True, parallel=True, cache=True, error_model="numpy")
def bayer_border(raw, ch, border, rgb):
    """librtprocess bayerborder_demosaic (core/debayer._border) on the outer ``border`` pixels of ``rgb`` (3, H, W):
    each colour the mean of the samples of that colour of ``raw`` in the 3 x 3 neighbourhood, the sample itself where
    it is of that colour. ``ch``: the colour of each 2 x 2 position."""
    H, W = raw.shape
    for y in prange(H):
        edge_row = y < border or y >= H - border
        for x in range(W):
            if not edge_row and border <= x < W - border:
                continue
            k = ch[2 * (y % 2) + x % 2]
            for c in range(3):
                if k == c:
                    rgb[c, y, x] = raw[y, x]
                    continue
                num = F32(0.0)
                den = F32(0.0)
                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        yy, xx = y + dy, x + dx
                        if 0 <= yy < H and 0 <= xx < W and ch[2 * (yy % 2) + xx % 2] == c:
                            num += raw[yy, xx]
                            den += F32(1.0)
                rgb[c, y, x] = num / max(den, F32(1e-12))


# ------------------------------------------------------------------------------------------------ bad samples
@njit(nogil=True, parallel=True, cache=True, error_model="numpy")
def fill_bad_round(out, ok, idx, d, got_val, got):
    """One round of debayer.fill_bad on the missing samples ``idx`` (flat positions): the mean of the valid samples
    (``ok``) among the 8 at distance ``d`` (3 x 3 dilated), zero outside the frame; ``got`` 1 and the value in
    ``got_val`` where there is any."""
    H, W = out.shape
    for n in prange(len(idx)):
        i = idx[n]
        y, x = i // W, i % W
        s = F32(0.0)
        m = F32(0.0)
        for dy in (-d, 0, d):
            for dx in (-d, 0, d):
                if dy == 0 and dx == 0:
                    continue
                yy, xx = y + dy, x + dx
                if 0 <= yy < H and 0 <= xx < W and ok[yy, xx]:
                    s += out[yy, xx]
                    m += F32(1.0)
        if m > 0:
            got_val[n] = s / max(m, F32(1e-12))
            got[n] = 1
        else:
            got[n] = 0


# ------------------------------------------------------------------------------------------------ saturation flags
@njit(nogil=True, parallel=True, cache=True, error_model="numpy")
def saturation_flags(fr, satmap, valid_max, near_frac, spread, fl):
    """core/stack_classic's flag image: bit 0 where a sample within ``spread`` pixels (a square) is near saturation
    (above near_frac x the saturation level, or saturated), bit 1 where one is saturated (above ``valid_max``).
    ``satmap``: the saturation level, (1, 1) or (H, W)."""
    H, W = fr.shape
    sh, sw = satmap.shape
    tmp = np.zeros((H, W), np.uint8)
    for y in prange(H):
        for x in range(W):
            v = fr[y, x]
            s = satmap[np.int64(y) if sh > 1 else np.int64(0), np.int64(x) if sw > 1 else np.int64(0)]
            f = np.uint8(0)
            if v > valid_max:
                f = np.uint8(3)
            elif v > near_frac * s:
                f = np.uint8(1)
            tmp[y, x] = f
    # horizontal then vertical runs of the square
    hz = np.zeros((H, W), np.uint8)
    for y in prange(H):
        for x in range(W):
            f = np.uint8(0)
            for dx in range(-spread, spread + 1):
                xx = x + dx
                if 0 <= xx < W:
                    f |= tmp[y, xx]
            hz[y, x] = f
    for y in prange(H):
        for x in range(W):
            f = np.uint8(0)
            for dy in range(-spread, spread + 1):
                yy = y + dy
                if 0 <= yy < H:
                    f |= hz[yy, x]
            fl[y, x] = f


# ------------------------------------------------------------------------------------------------ resampling
@njit(nogil=True, cache=True, error_model="numpy")
def _poly(c, u, v):
    return (c[0] + c[1] * u + c[2] * v + c[3] * u * u + c[4] * u * v + c[5] * v * v + c[6] * u * u * u
            + c[7] * u * u * v + c[8] * u * v * v + c[9] * v * v * v)


@njit(nogil=True, cache=True, error_model="numpy")
def _source(cx, cy, ix, iy, X, Y, W, H, iters):
    """poly.poly_inverse at one reference position (float64): the inverse cubic ``ix``, ``iy`` then Newton steps."""
    u, v = X / (W / 2) - 1.0, Y / (H / 2) - 1.0
    x, y = _poly(ix, u, v), _poly(iy, u, v)
    for _ in range(iters):
        u, v = x / (W / 2) - 1.0, y / (H / 2) - 1.0
        rx, ry = X - _poly(cx, u, v), Y - _poly(cy, u, v)
        a = (cx[1] + 2 * cx[3] * u + cx[4] * v + 3 * cx[6] * u * u + 2 * cx[7] * u * v + cx[8] * v * v) * (2 / W)
        b = (cx[2] + cx[4] * u + 2 * cx[5] * v + cx[7] * u * u + 2 * cx[8] * u * v + 3 * cx[9] * v * v) * (2 / H)
        e = (cy[1] + 2 * cy[3] * u + cy[4] * v + 3 * cy[6] * u * u + 2 * cy[7] * u * v + cy[8] * v * v) * (2 / W)
        d = (cy[2] + cy[4] * u + 2 * cy[5] * v + cy[7] * u * u + 2 * cy[8] * u * v + 3 * cy[9] * v * v) * (2 / H)
        det = a * d - b * e
        x = x + (d * rx - b * ry) / det
        y = y + (a * ry - e * rx) / det
    return x, y


@njit(nogil=True, cache=True, error_model="numpy")
def _weights(f, w):
    """The 8 Lanczos-4 weights of the offsets -3 .. 4 for the fractional part f (float32 in, as the PyTorch code),
    normalised to a sum of 1 (OpenCV's interpolateLanczos4)."""
    fd = float(f)
    s1 = math.sin(math.pi * fd)
    a = math.pi * fd / 4.0
    sa, ca = math.sin(a), math.cos(a)
    tot = 0.0
    for j in range(8):
        o = j - 3
        t = o - fd
        if abs(t) >= 4.0:
            w[j] = 0.0
        elif t == 0.0:
            w[j] = 1.0
        else:
            sg = -1.0 if o % 2 == 0 else 1.0                  # sin(pi (o - f)) = -(-1)^o sin(pi f)
            s_t = sg * s1
            ang = o * (math.pi / 4.0)
            s_q = math.sin(ang) * ca - math.cos(ang) * sa      # sin(pi t / 4)
            w[j] = s_t * s_q * 4.0 / (math.pi * math.pi * t * t)
        tot += w[j]
    for j in range(8):
        w[j] = w[j] / tot


@njit(nogil=True, cache=True, error_model="numpy")
def _bilinear(img, c, h, w, x, y):
    """core/resample.bilinear_at at one position: the value (float32), NaN where the 2 x 2 samples are not all in."""
    x0f, y0f = math.floor(x), math.floor(y)
    fx, fy = F32(x - x0f), F32(y - y0f)
    x0, y0 = int(x0f), int(y0f)
    if not (x0 >= 0 and x0 + 1 <= w - 1 and y0 >= 0 and y0 + 1 <= h - 1):
        return F32(np.nan)
    return (img[c, y0, x0] * (ONE - fx) + img[c, y0, x0 + 1] * fx) * (ONE - fy) + \
        (img[c, y0 + 1, x0] * (ONE - fx) + img[c, y0 + 1, x0 + 1] * fx) * fy


@njit(nogil=True, parallel=True, cache=True, error_model="numpy")
def resample_pass1(img, cx, cy, ix, iy, W, H, guide_on, clamp_factor, out, low, fl, fo):
    """core/resample.resample, first half: ``out`` (3, Ho, Wo) the Lanczos-4 samples at the inverse-registered
    position of each output pixel (NaN where the 8 x 8 window leaves the frame), ``low`` (3, Ho, Wo) where the sample
    is under ``clamp_factor`` x the bilinear guide (``guide_on``); ``fo`` the flag image ``fl`` read at the
    (rounded) source position, 0 outside."""
    C, h, w = img.shape
    Ho, Wo = out.shape[1], out.shape[2]
    for Y in prange(Ho):
        wx = np.empty(8)
        wy = np.empty(8)
        for X in range(Wo):
            x, y = _source(cx, cy, ix, iy, float(X), float(Y), W, H, 3)
            xi, yi = int(np.rint(x)), int(np.rint(y))
            if 0 <= xi < w and 0 <= yi < h:
                fo[Y, X] = fl[yi, xi]
            else:
                fo[Y, X] = 0
            x0f, y0f = math.floor(x), math.floor(y)
            x0, y0 = int(x0f), int(y0f)
            ok = x0 - 3 >= 0 and x0 + 4 <= w - 1 and y0 - 3 >= 0 and y0 + 4 <= h - 1
            if not ok:
                for c in range(C):
                    out[c, Y, X] = F32(np.nan)
            else:
                _weights(F32(x - x0f), wx)
                _weights(F32(y - y0f), wy)
                for c in range(C):
                    acc = ZERO
                    for i in range(8):
                        row = ZERO
                        for j in range(8):
                            row += F32(wx[j]) * img[c, y0 - 3 + i, x0 - 3 + j]
                        acc += F32(wy[i]) * row
                    out[c, Y, X] = acc
            if guide_on:
                for c in range(C):
                    g = _bilinear(img, c, h, w, x, y)
                    low[c, Y, X] = 1 if out[c, Y, X] < clamp_factor * g else 0


@njit(nogil=True, parallel=True, cache=True, error_model="numpy")
def resample_pass2(img, cx, cy, ix, iy, W, H, out, low, guide_on, a, b, Q, scale):
    """The second half: the samples next to a ``low`` one (3 x 3 cross) take the bilinear guide where it is defined,
    then every colour c is normalised, (value - Q_c(a, b)) / scale_c as stack_classic does, in place."""
    C, h, w = img.shape
    Ho, Wo = out.shape[1], out.shape[2]
    for Y in prange(Ho):
        for X in range(Wo):
            for c in range(C):
                v = out[c, Y, X]
                if guide_on:
                    near = low[c, Y, X] != 0
                    if not near and Y > 0:
                        near = low[c, Y - 1, X] != 0
                    if not near and Y < Ho - 1:
                        near = low[c, Y + 1, X] != 0
                    if not near and X > 0:
                        near = low[c, Y, X - 1] != 0
                    if not near and X < Wo - 1:
                        near = low[c, Y, X + 1] != 0
                    if near:
                        x, y = _source(cx, cy, ix, iy, float(X), float(Y), W, H, 3)
                        g = _bilinear(img, c, h, w, x, y)
                        if np.isfinite(g):
                            v = g
                aa, bb = a[X], b[Y]
                q = Q[c]
                bg = q[0] + q[1] * aa + q[2] * bb + q[3] * aa * aa + q[4] * aa * bb + q[5] * bb * bb
                out[c, Y, X] = (v - bg) / scale[c]
