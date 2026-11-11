"""Cut extracts: the offline Gaia extracts holding only the stars the program reads, for the installer to fetch.

The full extracts (core/sirilcat.py) are 1.5 GB of star positions and 21 GB of spectra; the program reads the positions
of the stars brighter than G 16 (the astrometry's catalogue stops there, the Catalogues block's bright stars too) and
the spectra of the colour stars (Catalogues › Faint limit, G 14 by default). This writes the same files, the same names
and chunks (one spectra file per level-1 pixel, fetched one by one), with only the records in a G range: each pixel's
records in their order, the index counted again, the header the full file's plus the field that says the cut
(sirilcat: ``FLOATcut``), so that the engine reads them unchanged and knows how deep they go.

    python -m tools.catalogue_reduce OUT [--src DIR] [--astro-gmax 16] [--xp-gmax 14] [--xp-gmin G] [--only astro|xp]

The full files are read in place, never written; a file is written to ``<name>.part`` then renamed (a stopped run
leaves no file that looks complete). The faint cut is strict for the positions (G < 16: what ``astro_cone`` keeps) and
widened by the spectra's match tolerance (``xp_spectra``: a star is matched to a record within 0.01 mag of its G) plus
a millimag of rounding, so that every record a match can pick is there.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from astro_gpu.core import sirilcat  # noqa: E402

BLOCK = 1_000_000                       # records read at once (0.7 GB of spectra)
XP_MARGIN_MMAG = 11                     # xp_spectra's 0.01 mag tolerance, and a millimag of rounding


def reduce_file(src: Path, dst: Path, lo_mmag: int | None, hi_mmag: int, log=print) -> dict:
    """Write to ``dst`` the extract ``src`` holding only the records with lo_mmag <= G <= hi_mmag (mmag; None: no
    bright cut). Returns {"records", "kept", "bytes"}. A file already cut keeps its narrower range."""
    cf = sirilcat.read_header(src)
    if cf is None:
        raise ValueError(f"{src} is not a complete extract")
    lo = sirilcat.NO_BRIGHT_CUT if lo_mmag is None else int(lo_mmag)
    hi = int(hi_mmag)
    if cf.cut is not None:
        lo, hi = max(lo, cf.cut[0]), min(hi, cf.cut[1])
    with open(src, "rb") as f:
        head = bytearray(f.read(sirilcat.HEADER))
    idx = cf.index()
    n = int(idx[-1])
    head[sirilcat.CUT_AT:sirilcat.CUT_AT + 8] = sirilcat.CUT_MAGIC
    for off, v, signed in ((8, lo, True), (12, hi, True), (16, cf.cap(), False),
                           (20, n if cf.cut is None else int.from_bytes(head[sirilcat.CUT_AT + 20:sirilcat.CUT_AT + 24],
                                                                       "little"), False)):
        head[sirilcat.CUT_AT + off:sirilcat.CUT_AT + off + 4] = int(v).to_bytes(4, "little", signed=signed)
    data0 = sirilcat.HEADER + 4 * cf.n_index
    kept = np.zeros(cf.n_index, np.int64)
    part = dst.with_name(dst.name + ".part")
    dst.parent.mkdir(parents=True, exist_ok=True)
    mm = np.memmap(src, dtype=cf.dtype, mode="r", offset=data0, shape=(n,)) if n else np.zeros(0, cf.dtype)
    with open(part, "wb") as out:
        out.write(bytes(head))
        out.write(b"\0" * (4 * cf.n_index))          # the index, written once the records are counted
        for a in range(0, n, BLOCK):
            rec = np.asarray(mm[a:a + BLOCK])
            g = rec["g"].astype(np.int64)
            keep = (g >= lo) & (g <= hi)
            # the pixel of each record: the first pixel whose cumulative count passes it
            pix = np.searchsorted(idx, np.arange(a, a + len(rec)), side="right")
            kept += np.bincount(pix[keep], minlength=cf.n_index)
            out.write(rec[keep].tobytes())
        out.seek(sirilcat.HEADER)
        out.write(np.cumsum(kept).astype("<u4").tobytes())
    del mm
    if sirilcat.read_header(part) is None:
        part.unlink()
        raise RuntimeError(f"{dst.name}: the cut file does not read back as an extract")
    os.replace(part, dst)
    k = int(kept.sum())
    log(f"{dst.name}: {k} of {n} stars kept (G {'-inf' if lo == sirilcat.NO_BRIGHT_CUT else lo / 1000} to "
        f"{hi / 1000}), {dst.stat().st_size / 1e6:.0f} MB")
    return {"records": n, "kept": k, "bytes": dst.stat().st_size}


def full_files(src: Path) -> dict:
    """{"astro": Path | None, "xp": [Path, ...]}: the extract files of the folder ``src`` (the layout of our own)."""
    astro = sorted(src.glob("siril_cat*_healpix*_astro*.dat"))
    xp = sorted((src / "siril_cat1_healpix8_xpsamp").glob("siril_cat*_healpix*_xpsamp*.dat"),
                key=lambda p: int(p.stem.rsplit("_", 1)[1]))
    return {"astro": astro[0] if astro else None, "xp": xp}


def reduce_all(src: Path, out: Path, astro_gmax: float = 16.0, xp_gmax: float = 14.0, xp_gmin: float | None = None,
               only: str | None = None, chunks: list[int] | None = None, log=print) -> dict:
    """Cut every extract of ``src`` into ``out`` (the same layout): positions G < astro_gmax, spectra up to xp_gmax
    (and from xp_gmin) widened by the match tolerance; ``chunks``: those spectra files only (a test). Returns {name:
    reduce_file's answer}."""
    files = full_files(src)
    res = {}
    if only in (None, "astro") and files["astro"] is not None:
        p = files["astro"]
        res[p.name] = reduce_file(p, out / p.name, None, round(astro_gmax * 1000) - 1, log)
    if only in (None, "xp"):
        lo = None if xp_gmin is None else round(xp_gmin * 1000) - XP_MARGIN_MMAG
        for p in files["xp"]:
            if chunks is not None and int(p.stem.rsplit("_", 1)[1]) not in chunks:
                continue
            res[p.name] = reduce_file(p, out / p.parent.name / p.name, lo, round(xp_gmax * 1000) + XP_MARGIN_MMAG, log)
    return res


def main(argv=None) -> int:
    from astro_gpu.core.sirilcat import own_dir
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out", type=Path, help="folder to write the cut extracts into (never the one read)")
    ap.add_argument("--src", type=Path, default=None, help="folder of the full extracts (default: the installed ones)")
    ap.add_argument("--astro-gmax", type=float, default=16.0, help="positions: the stars brighter than this G")
    ap.add_argument("--xp-gmax", type=float, default=14.0, help="spectra: the stars down to this G")
    ap.add_argument("--xp-gmin", type=float, default=None, help="spectra: the stars from this G (default: all)")
    ap.add_argument("--only", choices=("astro", "xp"), default=None)
    ap.add_argument("--chunks", type=int, nargs="+", default=None, help="these spectra files only (N of _<N>.dat)")
    a = ap.parse_args(argv)
    src = a.src or own_dir()
    if a.out.resolve() == src.resolve():
        ap.error("the cut extracts go to another folder than the full ones")
    res = reduce_all(src, a.out, a.astro_gmax, a.xp_gmax, a.xp_gmin, a.only, a.chunks)
    print(f"{len(res)} files, {sum(r['bytes'] for r in res.values()) / 1e9:.3f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
