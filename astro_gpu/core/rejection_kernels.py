"""The winsorized sigma clipping for the CPU, compiled by Numba: one loop per column (pixel) that does the column's whole
clipping (its rounds and winsorizing steps) on its few values, instead of generic array operations that each walk the
whole band again and again (the rounds and steps of the slowest column setting the number of walks).

It computes what ``rejection_classic.winsorized_clip`` computes, in the same double precision and with the sums of a
column taken in the same order (frame order), so the rejections are the same; tests/test_rejection_classic.py keeps the
two the same on random stacks. ``cache=True``: compiled once, kept next to the module.
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange, set_num_threads


@njit(nogil=True, cache=True, inline="always")
def _std(w, act, n):
    """The standard deviation (n - 1) of the values of ``w`` flagged in ``act`` (0 under 2 values)."""
    if n < 2:
        return 0.0
    s = 0.0
    for i in range(w.shape[0]):
        if act[i]:
            s += w[i]
    mean = s / n
    q = 0.0
    for i in range(w.shape[0]):
        if act[i]:
            d = w[i] - mean
            q += d * d
    return np.sqrt(q / (n - 1))


@njit(nogil=True, parallel=True, cache=True)
def _winsorized(v, valid, low, high, max_rounds, max_winsor, lo, hi):
    N, P = v.shape
    for j in prange(P):
        x = np.empty(N, np.float64)
        w = np.empty(N, np.float64)
        act = np.empty(N, np.bool_)
        cand = np.empty(N, np.bool_)
        dev = np.empty(N, np.float64)
        n = 0
        for i in range(N):
            act[i] = valid[i, j]
            x[i] = np.float64(v[i, j])
            if act[i]:
                n += 1
        if n < 2:
            continue
        r = 0
        for _ in range(max_rounds):
            n = 0
            for i in range(N):
                if act[i]:
                    n += 1
            sigma = _std(x, act, n)
            # the median: the mean of the two middle values
            tmp = np.empty(n, np.float64)
            k = 0
            for i in range(N):
                if act[i]:
                    tmp[k] = x[i]
                    k += 1
            tmp.sort()
            m = 0.5 * (tmp[(n - 1) // 2] + tmp[n // 2])
            for i in range(N):
                w[i] = x[i]
            for _s in range(max_winsor):
                a = m - 1.5 * sigma
                b = m + 1.5 * sigma
                for i in range(N):
                    if act[i]:
                        w[i] = min(max(w[i], a), b)
                s0 = sigma
                sigma = 1.134 * _std(w, act, n)
                if not (abs(sigma - s0) > s0 * 0.0005):
                    break
            ncand = 0
            for i in range(N):
                c = act[i] and ((m - x[i] > sigma * low) or (x[i] - m > sigma * high))
                cand[i] = c
                if c:
                    ncand += 1
            allowed = max(n - r - 4, 0)
            if ncand > allowed:                         # The reference program's guard: the most deviant first
                for i in range(N):
                    dev[i] = abs(x[i] - m) if cand[i] else -1.0
                if allowed == 0:
                    for i in range(N):
                        cand[i] = False
                else:
                    srt = np.sort(dev)                  # ascending: the allowed-th largest
                    kth = srt[N - allowed]
                    greater = 0
                    for i in range(N):
                        if cand[i] and dev[i] > kth:
                            greater += 1
                    need = allowed - greater
                    seen = 0
                    for i in range(N):
                        if not cand[i]:
                            continue
                        if dev[i] > kth:
                            continue
                        if dev[i] == kth:
                            seen += 1
                            if seen <= need:
                                continue
                        cand[i] = False
                ncand = 0
                for i in range(N):
                    if cand[i]:
                        ncand += 1
            for i in range(N):
                if cand[i]:
                    if m - x[i] > sigma * low:
                        lo[i, j] = True
                    if x[i] - m > sigma * high:
                        hi[i, j] = True
                    act[i] = False
            r += ncand
            if not (ncand > 0 and n - ncand > 3):
                break


def winsorized(v: np.ndarray, valid: np.ndarray, low: float, high: float, max_rounds: int = 100,
               max_winsor: int = 200, workers: int | None = None):
    """(low, high) rejected, two (N, P) bool arrays, of the stack ``v`` (N frames, P pixels; float) where ``valid``."""
    if workers:
        set_num_threads(max(1, workers))
    lo = np.zeros(v.shape, np.bool_)
    hi = np.zeros(v.shape, np.bool_)
    _winsorized(np.ascontiguousarray(v), np.ascontiguousarray(valid), float(low), float(high), max_rounds, max_winsor,
                lo, hi)
    return lo, hi


NR_EPS = 1.0e-7                     # rejection_classic.NR_EPS


@njit(nogil=True, cache=True)
def _rofunc(y, n, b, tmp):
    """Numerical Recipes' rofunc for the first ``n`` values of ``y`` against their index: (sum_k k sgn(y_k - b k - a),
    a the median of y - b k, sum |y - a - b k|)."""
    for k in range(n):
        tmp[k] = y[k] - b * k
    t = np.sort(tmp[:n])
    aa = 0.5 * (t[(n - 1) // 2] + t[n // 2])
    abdev = 0.0
    f = 0.0
    for k in range(n):
        d = y[k] - (b * k + aa)
        abdev += abs(d)
        dd = d / abs(y[k]) if y[k] != 0 else d
        if abs(dd) > NR_EPS:
            f += k if d >= 0 else -k
    return f, aa, abdev


@njit(nogil=True, cache=True)
def _l1_line(y, n, tmp, max_iter):
    """(a, b, mean absolute deviation) of the line minimising the mean absolute deviation of y[:n] against the index
    (rejection_classic.l1_line, one column)."""
    nf = float(n)
    sx = sy = sxy = sxx = 0.0
    for k in range(n):
        sx += k
        sy += y[k]
        sxy += k * y[k]
        sxx += k * k
    det = nf * sxx - sx * sx
    det_ok = det != 0
    dets = det if det_ok else 1.0
    aa = (sxx * sy - sx * sxy) / dets
    bb = (nf * sxy - sx * sy) / dets if det_ok else 0.0
    chisq = 0.0
    for k in range(n):
        d = y[k] - (aa + bb * k)
        chisq += d * d
    sigb = np.sqrt(chisq / dets) if det_ok else 0.0
    b1 = bb
    f1 = _rofunc(y, n, b1, tmp)[0]
    b2 = bb + (3.0 * sigb if f1 >= 0 else -3.0 * sigb)
    f2 = _rofunc(y, n, b2, tmp)[0]
    last = b2 if (sigb > 0 and f1 != 0) else b1
    done = (sigb <= 0) or (b2 == b1) or (n < 3) or (f1 == 0)
    if (not done) and f2 == 0:
        done = True
    for _ in range(max_iter):                       # bracket the root of rofunc
        if done or not (f1 * f2 > 0):
            break
        bn = b2 + 1.6 * (b2 - b1)
        b1, f1, b2 = b2, f2, bn
        f2 = _rofunc(y, n, bn, tmp)[0]
        last = bn
        done = done or f2 == 0
    tol = 0.01 * sigb
    for _ in range(max_iter):                       # then bisect it
        if done or not (abs(b2 - b1) > tol):
            break
        bm = b1 + 0.5 * (b2 - b1)
        if bm == b1 or bm == b2:
            break
        f = _rofunc(y, n, bm, tmp)[0]
        done = done or f == 0
        if f * f1 > 0:
            b1, f1 = bm, f
        else:
            b2, f2 = bm, f
        last = bm
    b = last if sigb > 0 else bb
    _, a, abdev = _rofunc(y, n, b, tmp)
    adev = abdev / max(nf, 1.0)
    if n == 2:
        b = y[1] - y[0]
        a = y[0]
        adev = 0.0
    return a, b, adev


@njit(nogil=True, parallel=True, cache=True)
def _linear_fit(v, valid, low, high, max_rounds, max_iter, lo, hi):
    N, P = v.shape
    for j in prange(P):
        act = np.empty(N, np.bool_)
        s = np.empty(N, np.float64)
        tmp = np.empty(N, np.float64)
        val = np.empty(N, np.float64)
        n = 0
        for i in range(N):
            act[i] = valid[i, j]
            val[i] = np.float64(v[i, j])
            if act[i]:
                n += 1
        if n < 5:
            continue
        for _ in range(max_rounds):
            idx = np.empty(n, np.int64)
            k = 0
            for i in range(N):
                if act[i]:
                    idx[k] = i
                    k += 1
            x = np.empty(n, np.float64)
            for k in range(n):
                x[k] = val[idx[k]]
            order = np.argsort(x, kind="mergesort")
            for k in range(n):
                s[k] = x[order[k]]
            a, b, adev = _l1_line(s, n, tmp, max_iter)
            sigma = 2.0 * adev
            nc = 0
            if sigma > 0:
                for k in range(n):
                    line = a + b * k
                    if s[k] < line:
                        if (line - s[k]) / sigma >= low:
                            i = idx[order[k]]
                            lo[i, j] = True
                            act[i] = False
                            nc += 1
                    elif (s[k] - line) / sigma >= high:
                        i = idx[order[k]]
                        hi[i, j] = True
                        act[i] = False
                        nc += 1
            if not (nc > 0 and n - nc >= 3):
                break
            n -= nc


def linear_fit(v: np.ndarray, valid: np.ndarray, low: float, high: float, max_rounds: int = 100,
               workers: int | None = None):
    """(low, high) rejected, two (N, P) bool arrays: the linear fit clipping of rejection_classic.linear_fit_clip."""
    if workers:
        set_num_threads(max(1, workers))
    lo = np.zeros(v.shape, np.bool_)
    hi = np.zeros(v.shape, np.bool_)
    _linear_fit(np.ascontiguousarray(v), np.ascontiguousarray(valid), float(low), float(high), max_rounds, 100, lo, hi)
    return lo, hi
