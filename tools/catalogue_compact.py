"""Compact star spectra: the offline Gaia spectra extracts with each spectrum stored as its coordinates on a basis.

What the program computes from a star's sampled spectrum (343 samples, 336-1020 nm) is linear in it: the Catalogs block
its CIE 1931 XYZ (``stages/catalogs._xyz``, the only thing the Color block's white balance takes from the spectra), the
Inspect diagnostic of the sensor's response its projections on Gaussians every 15 nm (``core/spectral.py``, a
diagnostic that changes no colour). The basis: first the span of the three colour matching functions, so that every
star's XYZ is kept exactly whatever its spectrum (its 3 coordinates in float32); then the Karhunen-Loeve basis
(principal components) of what is orthogonal to it, the first K - 3 eigenvectors of the second-moment matrix of the
unit-norm spectra of a random sample of the catalogue (``--per-chunk`` records drawn in each chunk, seeded: the same
files give the same basis), in int16: constrained principal components (Takane & Shibayama 1991, Psychometrika 56,
97) of the KL compression of spectra (Connolly et al. 1995, AJ 110, 1071; Yip et al. 2004, AJ 128, 585). Plain
principal components or spectra binned to 16-48 bands were measured first and moved the white balance by up to 0.4 %
(K = 24) and 0.08 % (48 bands): the colour has to be in the basis, not approximated by it.

    python -m tools.catalogue_compact OUT [--src DIR] [--k 24] [--xp-gmax 14] [--xp-gmin G] [--chunks N ...]

From the spectra extracts (full or cut: ``tools/catalogue_reduce.py``), read in place, never written, this writes
``OUT/float_xp_compact/float_xp_compact_<N>.dat``: the same 48 level-1 chunks (fetched one by one), each pixel's stars
in their order, the G range said in the cut field (core/sirilcat.py describes the format). A file is written to
``<name>.part`` then renamed. The faint cut is widened by the match tolerance, as the cut extracts' is.

Measured on 4.1 M stars (G <= 14.011) of 9 chunks with a basis drawn from the 39 others: their XYZ differ from the
sampled spectra's by 1e-5 % at the 99th percentile, 0.009 % at most; the white balance of 4 fields (Heart, Rosette,
Seestar, NGC 7000) by 1e-7 at most, Delta E 2000 0.0000; the diagnostic's Gaussian projections by 0.5 % of the largest
in the median (K = 24). K = 24: 68 bytes a star, 1.07 GB for the 15.7 M stars to G 14.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from astro_gpu.core import sirilcat  # noqa: E402
from tools.catalogue_reduce import XP_MARGIN_MMAG, full_files  # noqa: E402

BLOCK = 200_000                         # records read at once (0.14 GB of sampled spectra)
K = 24                                  # coordinates a star
PER_CHUNK = 6000                        # records drawn in each chunk for the basis
SEED = 1


def _records(cf: sirilcat.CatFile) -> np.ndarray:
    n = int(cf.index()[-1])
    if not n:
        return np.zeros(0, cf.dtype)
    return np.memmap(cf.path, dtype=cf.dtype, mode="r", offset=sirilcat.HEADER + cf.extra + 4 * cf.n_index, shape=(n,))


def basis_from(paths: list[Path], lo: int, hi: int, k: int = K, per_chunk: int = PER_CHUNK, seed: int = SEED) -> np.ndarray:
    """The basis (343, k), orthonormal, of the spectra with lo <= G <= hi (mmag) of a random sample of the extracts
    ``paths``: the span of the three colour matching functions (``xyz_weights``), then the first k - 3 eigenvectors of
    the mean of f f^T over the unit-norm spectra f projected orthogonally to it, largest first (constrained principal
    components: Takane & Shibayama 1991, Psychometrika 56, 97); each vector signed so that its largest component is
    positive (the same files give the same basis)."""
    rng = np.random.default_rng(seed)
    C = np.zeros((len(sirilcat.XP_WAVELENGTHS),) * 2)
    n = 0
    for p in paths:
        cf = sirilcat.read_header(p)
        if cf is None or cf.kind != sirilcat.TYPE_XPSAMP:
            raise ValueError(f"{p} is not a complete sampled-spectra extract")
        mm = _records(cf)
        if not len(mm):
            continue
        rec = np.asarray(mm[np.sort(rng.choice(len(mm), min(per_chunk, len(mm)), replace=False))])
        del mm
        rec = rec[(rec["g"] >= lo) & (rec["g"] <= hi)]
        f = cf.spectra(rec)
        nrm = np.linalg.norm(f, axis=1)
        f = f[np.isfinite(nrm) & (nrm > 0)] / nrm[np.isfinite(nrm) & (nrm > 0), None]
        C += f.T @ f
        n += len(f)
    if n < k:
        raise ValueError(f"{n} spectra in the range: too few for a basis of {k}")
    Q0 = np.linalg.qr(xyz_weights())[0]             # the span of the colour the program computes, kept exactly
    P = np.eye(len(Q0)) - Q0 @ Q0.T                 # ... and the KL basis of what is orthogonal to it
    ev, V = np.linalg.eigh(P @ (C / n) @ P)
    V = np.hstack([Q0, V[:, ::-1][:, :k - 3]])
    V = np.linalg.qr(V)[0]                          # orthonormal to rounding (the first 3 span Q0's space)
    return V * np.sign(V[np.abs(V).argmax(0), np.arange(k)])


def xyz_weights() -> np.ndarray:
    """(343, 3): the CIE 1931 colour matching functions on the extract's wavelengths, as ``stages/catalogs._xyz`` takes
    them (zero outside their 360-830 nm): a star's XYZ is its spectrum times these."""
    import colour
    cm = colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"]
    lam = sirilcat.XP_WAVELENGTHS
    s = (lam >= cm.wavelengths.min()) & (lam <= cm.wavelengths.max())
    W = np.zeros((len(lam), 3))
    W[s] = np.stack([np.interp(lam[s], cm.wavelengths, cm.values[:, i]) for i in range(3)], 1)
    return W


def encode(flux: np.ndarray, basis: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(norm float32 (N,), the first 3 coordinates float32 (N, 3), the others int16 x 32767 (N, K - 3)) of the spectra
    ``flux`` (N, 343) (``CatFile.spectra`` rebuilds them). The colour's 3 in float32: in int16 their rounding alone
    moved the white balance of a field by 0.024 %."""
    nrm = np.linalg.norm(flux, axis=1)
    ok = np.isfinite(nrm) & (nrm > 0)
    c = np.zeros((len(flux), basis.shape[1]))
    c[ok] = (flux[ok] / nrm[ok, None]) @ basis
    return (np.where(ok, nrm, 0).astype("<f4"), c[:, :3].astype("<f4"),
            np.clip(np.round(c[:, 3:] * sirilcat.COEF_SCALE), -32767, 32767).astype("<i2"))


def compact_file(src: Path, dst: Path, basis: np.ndarray, lo_mmag: int | None, hi_mmag: int, log=print) -> dict:
    """Write to ``dst`` the compact extract of the spectra extract ``src``, its records with lo_mmag <= G <= hi_mmag
    (None: no bright cut; a cut source keeps its narrower range). Returns {"records", "kept", "bytes"}."""
    cf = sirilcat.read_header(src)
    if cf is None or cf.kind != sirilcat.TYPE_XPSAMP:
        raise ValueError(f"{src} is not a complete sampled-spectra extract")
    lo = sirilcat.NO_BRIGHT_CUT if lo_mmag is None else int(lo_mmag)
    hi = int(hi_mmag)
    if cf.cut is not None:
        lo, hi = max(lo, cf.cut[0]), min(hi, cf.cut[1])
    with open(src, "rb") as f:
        src_head = f.read(sirilcat.HEADER)
    k = basis.shape[1]
    idx = cf.index()
    n = int(idx[-1])
    head = bytearray(sirilcat.HEADER)
    head[:48] = (sirilcat.COMPACT_TITLE + b" Gaia DR3 XP spectra").ljust(48, b"\0")
    head[48:65] = src_head[48:65]                    # release, level, type, chunk fields: the source's
    head[50] = sirilcat.TYPE_XPCOMPACT
    a = sirilcat.COMPACT_AT
    head[a], head[a + 1] = sirilcat.COMPACT_VERSION, k
    head[a + 2:a + 4] = len(sirilcat.XP_WAVELENGTHS).to_bytes(2, "little")
    c = sirilcat.CUT_AT
    n_full = n if cf.cut is None else int.from_bytes(src_head[c + 20:c + 24], "little")
    head[c:c + 8] = sirilcat.CUT_MAGIC
    for off, v, signed in ((8, lo, True), (12, hi, True), (16, cf.cap(), False), (20, n_full, False)):
        head[c + off:c + off + 4] = int(v).to_bytes(4, "little", signed=signed)
    kept = np.zeros(cf.n_index, np.int64)
    part = dst.with_name(dst.name + ".part")
    dst.parent.mkdir(parents=True, exist_ok=True)
    mm = _records(cf)
    out_dt = sirilcat.compact_dtype(k)
    with open(part, "wb") as out:
        out.write(bytes(head))
        out.write(np.asarray(basis, "<f4").tobytes())
        out.write(b"\0" * (4 * cf.n_index))          # the index, written once the records are counted
        for b0 in range(0, n, BLOCK):
            rec = np.asarray(mm[b0:b0 + BLOCK])
            g = rec["g"].astype(np.int64)
            keep = (g >= lo) & (g <= hi)
            pix = np.searchsorted(idx, np.arange(b0, b0 + len(rec)), side="right")
            kept += np.bincount(pix[keep], minlength=cf.n_index)
            rec = rec[keep]
            o = np.zeros(len(rec), out_dt)
            for name in ("ra", "dec", "g"):
                o[name] = rec[name]
            o["norm"], o["xyz"], o["coef"] = encode(cf.spectra(rec), basis)
            out.write(o.tobytes())
        out.seek(sirilcat.HEADER + 4 * basis.size)
        out.write(np.cumsum(kept).astype("<u4").tobytes())
    del mm
    if sirilcat.read_header(part) is None:
        part.unlink()
        raise RuntimeError(f"{dst.name}: the compact file does not read back as an extract")
    os.replace(part, dst)
    kk = int(kept.sum())
    log(f"{dst.name}: {kk} of {n} stars, {dst.stat().st_size / 1e6:.0f} MB")
    return {"records": n, "kept": kk, "bytes": dst.stat().st_size}


def compact_all(src: Path, out: Path, k: int = K, xp_gmax: float = 14.0, xp_gmin: float | None = None,
                chunks: list[int] | None = None, basis: np.ndarray | None = None, per_chunk: int = PER_CHUNK,
                log=print) -> dict:
    """Every spectra extract of ``src`` (the layout of our own folder) as a compact one in ``out/float_xp_compact``, the
    basis drawn from all of them (``basis``: given); ``chunks``: those files only. Returns {name: compact_file's}."""
    files = full_files(src)["xp"]
    lo = sirilcat.NO_BRIGHT_CUT if xp_gmin is None else round(xp_gmin * 1000) - XP_MARGIN_MMAG
    hi = round(xp_gmax * 1000) + XP_MARGIN_MMAG
    if basis is None:
        basis = basis_from(files, lo, hi, k, per_chunk)
        log(f"basis of {k} drawn from {len(files)} files")
    res = {}
    for p in files:
        n = int(p.stem.rsplit("_", 1)[1])
        if chunks is not None and n not in chunks:
            continue
        name = f"{sirilcat.COMPACT_DIR}_{n}.dat"
        res[name] = compact_file(p, out / sirilcat.COMPACT_DIR / name, basis, None if xp_gmin is None else lo, hi, log)
    return res


def main(argv=None) -> int:
    from astro_gpu.core.sirilcat import own_dir
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out", type=Path, help="folder to write the compact extracts into (never the one read)")
    ap.add_argument("--src", type=Path, default=None, help="folder of the spectra extracts (default: the installed ones)")
    ap.add_argument("--k", type=int, default=K, help="coordinates a star")
    ap.add_argument("--xp-gmax", type=float, default=14.0, help="the stars down to this G")
    ap.add_argument("--xp-gmin", type=float, default=None, help="the stars from this G (default: all)")
    ap.add_argument("--per-chunk", type=int, default=PER_CHUNK, help="records drawn in each file for the basis")
    ap.add_argument("--chunks", type=int, nargs="+", default=None, help="these files only (N of _<N>.dat)")
    a = ap.parse_args(argv)
    src = a.src or own_dir()
    if a.out.resolve() == src.resolve():
        ap.error("the compact extracts go to another folder than the ones read")
    res = compact_all(src, a.out, a.k, a.xp_gmax, a.xp_gmin, a.chunks, per_chunk=a.per_chunk)
    print(f"{len(res)} files, {sum(r['bytes'] for r in res.values()) / 1e9:.3f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
