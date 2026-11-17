"""The Stars block: a network takes the stars off the image, which gives two images, the starless image and the stars
alone (docs/chantiers/etoiles.md).

The network is NAFNet (Chen, Chu, Zhang, Sun 2022, arXiv:2204.04676), an image restoration network, trained once on
our stacks: starless images corrected by hand, plus stars drawn on top of them (a Moffat core, an r^-3 aureole,
diffraction spikes, saturation), where the truth under each star is exact (tools/bench/stars_train.py, run "mixed").
On the bench it leaves less of the faint stars than the reference tool and takes the spikes and halos of the bright
stars off, the background unchanged (docs/POINT_MATINEE_2026-11-05.md).

It works on what it was trained on: the Output block's linear image (render_simple: the stack with its sky gradient
removed, denoised, white balanced, its sky taken off), each channel minus its background, over one noise unit (the
noise of the half-stacks before denoising, through the colour matrix, averaged over the three colours; a linear Image
has no half-stacks: the spread of its own sky). The network sees a fixed asinh stretch of it, undone exactly on the
way out.

Outputs, in the units of the Output block's lin.npy (so that starless + stars is that image, to the pixel):

* starless.npy, the image without its stars; stars.npy, the stars alone (the image minus the starless one); the
  faint halo the network leaves around the bright stars is moved from the first to the second (Star halos: Remove,
  take_off_halos);
* valid.npy, the covered pixels (the Output block's), and prep_stars.json: the useful frame and the statistics of
  the whole image's stretch (render/stars.py stretches both images with it in Post).
"""
from __future__ import annotations

import hashlib
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ..core import devices
from ..core.imgutil import asinh_preview
from pydantic import field_validator

from ..core.params import P, StageParams, check_choice
from ..core.stage import Context, Stage

MODEL = Path(__file__).resolve().parents[1] / "resources" / "stars" / "starless_nafnet.pt"
_MODEL_READ: dict[tuple, str] = {}


def model_id() -> str:
    """What the stage's version says of the network's file: the start of its hash, "missing" without the file. Read
    from the file as it is now (hashed once for a path, a size and a date), not once when the application starts: a
    file put in place while the application runs is in the next plan, as the run's worker, a new process, reads it
    (the application kept "missing", and every run failed on a key that was not the planned one until it was started
    again)."""
    try:
        st = MODEL.stat()
    except OSError:
        return "missing"
    k = (str(MODEL), st.st_size, st.st_mtime_ns)
    got = _MODEL_READ.get(k)
    if got is None:
        try:
            got = hashlib.sha256(MODEL.read_bytes()).hexdigest()[:10]
        except OSError:                  # gone since, or still being written: as without it, read again when asked
            return "missing"
        _MODEL_READ.clear()
        _MODEL_READ[k] = got
    return got
#: the network's stretch: it sees asinh(v / STRETCH) / SCALE, v in noise units
STRETCH, SCALE = 3.0, 10.0
#: the size the network was trained on, and its tiles on an image that is smaller than a big tile
NET_TILE, NET_OVERLAP = 256, 64
#: the tiles it works on: larger than the largest star halos (M45: 280 px), so that the tile holding a star's core also
#: holds its halo. Tiles of the training size cut such a halo along their boundary: the tile with the core takes it off,
#: its neighbour, which only sees halo, leaves it (issue #45; 1024/256 on M45: 13 s, 256/64: 9 s, on an RTX 4060 Ti)
TILE, OVERLAP = 1024, 256
#: its channel attention averages over this window instead of over the whole tile, the training size. It is how the
#: network's authors run it on more than its training size: Test-time Local Converter (Chu, Chen, Chen, Lu 2022, "Improving
#: Image Restoration by Revisiting Global Information Aggregation", arXiv:2112.04491; NAFNetLocal in
#: github.com/megvii-research/NAFNet). Their default is 1.5 x the training size; their reported range starts at the
#: training size, which on M45 leaves no ridge on the tile bands where 384 leaves a few (max 3 against 17 noise units)
WINDOW = NET_TILE
#: tiles of NET_TILE through the network at once (a big tile counts for its area), and the card memory one of them takes:
#: 1.4 GB measured for 16 on 8 November 2026, in bfloat16; a 4 GB card holds the whole batch
BATCH, TILE_GB = 16, 0.09


#: the star halos' choices (take_off_halos)
HALOS = ("remove", "keep")


class StarRemovalParams(StageParams):
    """The network is applied as it was trained (its outputs are adjusted in Post); the halos it leaves are taken off
    or kept."""

    halos: str = P(HALOS[0], label="Star halos", choices=list(HALOS), key_neutral=HALOS[0],
                   help="The network takes the core of each star off but leaves the outer part of the wide halo of the "
                        "bright stars in the starless image: a faint disc around them. Remove: what the network leaves "
                        "around the image's own isolated bright stars is measured by brightness, and each bright star's "
                        "share, never more than its own extended profile, is moved to the stars image. Keep: the "
                        "network's result as it is.",
                   tip="Remove by default. Keep when a halo is part of what you want to show.")

    @field_validator("halos")
    @classmethod
    def _known(cls, v: str) -> str:
        return check_choice("halos", v, HALOS)


class StarRemoval(Stage):
    name = "star_removal"
    title = "Star removal"
    description = ("A network trained on our stacks takes the stars off the Output block's image: the starless image "
                   "and the stars alone, two images that Post combines (the Combine layers block) or takes as "
                   "they are.")
    deps = ("render_simple", "stack")
    Params = StarRemovalParams

    #: the keys' label: FROZEN, never change it (core/stage.Stage.key_label). It was the version with the start of
    #: the network file's hash; the file's identity is now part of the algorithm version only (``version``)
    key_label = "4-9b974a78cb"

    @property
    def version(self) -> str:            # 2: the network's grid notched (core/periodic.py, issue #28); 3: big tiles (#45);
        #                                  4: a starless fusion's stretch (gone with the fusion's star layer, 2026-11-12)
        #                                  5: the halos the network leaves taken off (take_off_halos)
        return f"5-{model_id()}"         # with the network's file as it is now (model_id)

    def run(self, ctx: Context, p: StarRemovalParams) -> None:
        from .denoise import norm_params
        from .gradient import stack_image
        ctx.progress(0.02, "loading the image")
        prep = ctx.load_json("render_simple", "prep_simple.json")
        lin = ctx.load("render_simple", "lin.npy").astype(np.float32)
        valid = ctx.load("render_simple", "valid.npy")
        M = np.asarray(prep["M"], np.float64)
        x0, x1, y0, y1 = prep["crop"]
        shown = np.zeros_like(valid)
        shown[y0:y1, x0:x1] = True
        shown &= valid
        if (ctx.project.stage_dir("stack") / "drz_A.npy").is_file():
            _bk, sg, _nm = norm_params(stack_image(ctx, "drz_A.npy"), ctx.load("stack", "drz_weight.npy", mmap=True))
            unit = float(np.mean(np.sqrt((M ** 2) @ (np.asarray(sg, np.float64) ** 2))))
        else:                                            # a linear Image (issue #55): no half-stack to read it on
            unit = sky_unit(lin, shown)
        bk = np.array([np.median(lin[c][shown][::7]) for c in range(3)], np.float32)[:, None, None]
        ctx.log(f"noise unit {unit:.4g}, background {', '.join(f'{v:.3g}' for v in bk.ravel())}")
        dev = ctx.device
        net = network(dev)
        t0 = time.time()
        starless = apply(net, (lin - bk) / unit, dev, ctx) * unit + bk
        ctx.log(f"stars taken off in {time.time() - t0:.1f} s on {dev}")
        if np.array_equal(lin[0, ::97, ::97], lin[1, ::97, ::97]) and np.array_equal(lin[2, ::97, ::97],
                                                                                    lin[1, ::97, ::97]):
            starless = np.repeat(starless.mean(0, keepdims=True), 3, axis=0)   # a monochrome image stays grey (#43)
        starless[:, ~valid] = lin[:, ~valid]
        # the network's own grid (its checkerboard, at the multiples of 1/16 px^-1) notched where it stands out
        from ..core.periodic import lattice_frequencies, remove_periodic
        starless, taken = remove_periodic(starless, lattice_frequencies(16), valid)
        ctx.metric("grid_peaks", len(taken))
        if taken:
            ctx.log(f"the network's grid: {len(taken)} peaks notched (strongest {max(t[3] for t in taken):.0f} x the "
                    "spectrum around)")
        ctx.progress(0.86, "star halos")
        starless, halos = halos_step(lin, starless, valid, p.halos, ctx.log)
        if halos is not None:
            ctx.metric("halo_stars", len(halos["stars"]))
        stars = lin - starless
        ctx.progress(0.9, "statistics")
        ctx.save("starless.npy", starless)
        ctx.save("stars.npy", stars)
        np.save(ctx.out("valid.npy"), valid)
        out = {"crop": prep["crop"], "M": prep["M"], "drizzle_scale": prep.get("drizzle_scale", 1.0),
               "width": prep.get("width"), "height": prep.get("height"), "unit": unit, "background": bk.ravel().tolist(),
               "stats_full": prep["stats"]}
        if prep.get("origin") == "palette":              # a narrow-band combination's: Post's sliders know it (#163)
            out["origin"] = "palette"
        ctx.save_json("prep_stars.json", out)
        ctx.progress(0.95, "preview")
        h, w = lin.shape[1:]
        cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
        s = min(800, h, w) // 2
        box = (slice(None), slice(max(0, cy - s), cy + s), slice(max(0, cx - s), cx + s))
        a, b = lin[box], starless[box]                   # one stretch for both: before and after side by side
        ctx.preview("starless", asinh_preview(np.concatenate([a, np.zeros((3, a.shape[1], 6), np.float32), b], 2)))



class _LayerNorm2d(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.w, self.b = nn.Parameter(torch.ones(c)), nn.Parameter(torch.zeros(c))

    def forward(self, x):
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        return (x - mu) / (var + 1e-6).sqrt() * self.w[:, None, None] + self.b[:, None, None]


class _LocalPool(nn.Module):
    """The channel attention's average. In training, over the whole tile, as published. Applied, over a ``window`` around
    each position when the features are larger than that (Test-time Local Converter, see WINDOW): a tile larger than the
    training size then sees what the network was trained on."""

    def __init__(self, window):
        super().__init__()
        self.window = window

    def forward(self, x):
        h, w = x.shape[2:]
        if self.training or (self.window >= h and self.window >= w):
            return x.mean((2, 3), keepdim=True)
        k1, k2 = min(self.window, h), min(self.window, w)
        s = nn.functional.pad(x.float().cumsum(2).cumsum(3), (1, 0, 1, 0))            # float32: bfloat16 cannot sum
        o = (s[:, :, k1:, k2:] + s[:, :, :-k1, :-k2] - s[:, :, :-k1, k2:] - s[:, :, k1:, :-k2]) / (k1 * k2)
        dh, dw = h - o.shape[2], w - o.shape[3]
        return nn.functional.pad(o, (dw // 2, dw - dw // 2, dh // 2, dh - dh // 2), mode="replicate").to(x.dtype)


class _NAFBlock(nn.Module):
    def __init__(self, c, window=WINDOW):
        super().__init__()
        self.n1, self.n2 = _LayerNorm2d(c), _LayerNorm2d(c)
        self.c1 = nn.Conv2d(c, 2 * c, 1)
        self.c2 = nn.Conv2d(2 * c, 2 * c, 3, padding=1, groups=2 * c)
        self.sca = nn.Sequential(_LocalPool(window), nn.Conv2d(c, c, 1))
        self.c3 = nn.Conv2d(c, c, 1)
        self.c4 = nn.Conv2d(c, 2 * c, 1)
        self.c5 = nn.Conv2d(c, c, 1)
        self.beta = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, inp):
        x = self.c2(self.c1(self.n1(inp)))
        a, b = x.chunk(2, 1)
        x = a * b
        x = self.c3(x * self.sca(x))
        y = inp + x * self.beta
        a, b = self.c4(self.n2(y)).chunk(2, 1)
        return y + self.c5(a * b) * self.gamma


class NAFNet(nn.Module):
    """NAFNet (Chen, Chu, Zhang, Sun 2022, arXiv:2204.04676), as published (github.com/megvii-research/NAFNet).
    ``recompute``: in training, the blocks' activations are recomputed on the way back instead of kept (less memory,
    about a third slower)."""

    def __init__(self, width=32, enc=(2, 2, 4, 8), mid=12, dec=(2, 2, 2, 2), recompute=False):
        super().__init__()
        self.recompute = recompute
        self.intro = nn.Conv2d(3, width, 3, padding=1)
        self.ending = nn.Conv2d(width, 3, 3, padding=1)
        self.encs, self.downs, self.ups, self.decs = (nn.ModuleList() for _ in range(4))
        c, win = width, WINDOW                                        # win: WINDOW in this level's own pixels
        for n in enc:
            self.encs.append(nn.Sequential(*[_NAFBlock(c, win) for _ in range(n)]))
            self.downs.append(nn.Conv2d(c, 2 * c, 2, 2))
            c, win = 2 * c, win // 2
        self.mid = nn.Sequential(*[_NAFBlock(c, win) for _ in range(mid)])
        for n in dec:
            self.ups.append(nn.Sequential(nn.Conv2d(c, 2 * c, 1, bias=False), nn.PixelShuffle(2)))
            c, win = c // 2, win * 2
            self.decs.append(nn.Sequential(*[_NAFBlock(c, win) for _ in range(n)]))
        self.pad = 2 ** len(enc)

    def _run(self, m, v):
        if self.recompute and self.training:
            from torch.utils.checkpoint import checkpoint
            return checkpoint(m, v, use_reentrant=False)
        return m(v)

    def forward(self, inp):
        H, W = inp.shape[2:]
        ph, pw = (-H) % self.pad, (-W) % self.pad
        x = nn.functional.pad(inp, (0, pw, 0, ph))
        x0 = x
        x = self.intro(x)
        skips = []
        for e, dn in zip(self.encs, self.downs):
            x = self._run(e, x)
            skips.append(x)
            x = dn(x)
        x = self._run(self.mid, x)
        for d, up, sk in zip(self.decs, self.ups, reversed(skips)):
            x = self._run(d, up(x) + sk)
        return (self.ending(x) + x0)[:, :, :H, :W]


def sky_unit(lin: np.ndarray, shown: np.ndarray) -> float:
    """The noise unit of an image without half-stacks: the spread of its sky over the shown pixels, in the colours of
    ``lin`` (its colour matrix is applied already), averaged over the three colours; 1 for an image without noise."""
    from .render_simple import sky_spread
    return float(np.mean([sky_spread(lin[c][shown][::7]) for c in range(3)])) or 1.0


# ----------------------------------------------------------------------------------------------------- the star halos
# The network works in noise units of the stack before Denoise: the outer wing of a bright star, a few hundredths of
# that unit per pixel, is taken for sky and stays in the starless image, a faint disc of 50 to 100 px around each bright
# star once denoised (m27_v10, 17 November 2026: 98 % of the star gone within 12 px, a disc of 2 to 4 sky sigmas left
# out to 50 px, fading by 100 px).
# The established remedy of deep imaging (Slater, Harding & Mihos 2009, PASP 121:1267; Infante-Sainz et al. 2020,
# MNRAS 491:5317): the extended profile of the stars is built from the image's own bright stars (their radial profiles
# normalised on their core and combined robustly, out to where they meet the sky) and scaled to each bright star.
# What the network leaves of it is measured the same way, on the same isolated stars, by classes of core level (it does
# not grow in proportion to the star: the network's stretch); each bright star gets the part its class leaves, never
# more than its own scaled profile, moved from the starless image to the stars image (starless + stars is still the
# image). Measured star by star instead (the model minus what the stars image holds at each radius), the model's own
# error within the core, a few per cent of a star thousands of sigmas bright, dwarfed the few sigmas left: dark rings
# and holes on the bench.

#: the widest halo measured, px, and the sky ring beyond it against which the profile stars are measured
HALO_R, HALO_RING = 200, 32
#: the extended profile is the clipped mean of at most HALO_STARS stars, and needs at least HALO_FEWEST
HALO_STARS, HALO_FEWEST = 40, 5
#: the candidates: their core HALO_DETECT sky sigmas above the stars image's pedestal; a candidate is treated when what
#: is left around it reaches HALO_LEVEL sky sigma (fainter: nothing measurable)
HALO_DETECT, HALO_LEVEL = 100.0, 0.5
#: what the network leaves is measured by classes of HALO_CLASS profile stars of neighbouring core levels
HALO_CLASS = 8


def halos_step(lin: np.ndarray, starless: np.ndarray, valid: np.ndarray, how: str, log=None):
    """The starless image with the halos the network left taken off ("remove"), or as it is ("keep": the very array).
    Returns (starless, what take_off_halos says, None with "keep")."""
    if how != "remove":
        return starless, None
    t0 = time.time()
    out, info = take_off_halos(lin, starless, valid)
    if log is not None:
        if info.get("why"):
            log(f"star halos: none taken off ({info['why']})")
        else:
            log(f"star halos: the profile of {info['psf_stars']} stars (FWHM {info['fwhm']:.1f} px, out to "
                f"{info['r_max']} px), what the network leaves in {info['classes']} classes of "
                f"{info['profile_stars']} stars, taken off {len(info['stars'])} stars in {time.time() - t0:.1f} s")
    return out, info


def _sky_noise(img: np.ndarray, valid: np.ndarray) -> float:
    """The sky noise per pixel: the spread of the image minus its median over 60 px, at one pixel in 4 (1.4826 MAD)."""
    a, v = img[::4, ::4], valid[::4, ::4]
    from scipy import ndimage as ndi
    d = (a - ndi.median_filter(a, 15))[v]
    if d.size < 100:
        return 0.0
    return 1.4826 * float(np.median(np.abs(d - np.median(d))))


def _ring_medians(v: np.ndarray, r: np.ndarray, nb: int) -> np.ndarray:
    """The median of the values ``v`` in each ring of 1 px (``r``: the ring of each value), nb rings; nan when empty."""
    keep = r < nb
    v, r = v[keep], r[keep]
    vs = v[np.lexsort((v, r))]
    cnt = np.bincount(r, minlength=nb)[:nb]
    start = np.cumsum(cnt) - cnt
    out = np.full(nb, np.nan)
    ok = cnt > 0
    out[ok] = vs[start[ok] + cnt[ok] // 2]
    return out


def _box(y: int, x: int, R: int, H: int, W: int):
    """The box of radius R around (y, x) clipped to the image, and the distance of each of its pixels to the centre."""
    y0, y1, x0, x1 = max(0, y - R), min(H, y + R + 1), max(0, x - R), min(W, x + R + 1)
    yy, xx = np.ogrid[y0 - y:y1 - y, x0 - x:x1 - x]
    return (slice(y0, y1), slice(x0, x1)), np.sqrt(yy * yy + xx * xx)


def take_off_halos(lin: np.ndarray, starless: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, dict]:
    """The starless image (3, H, W) with the halos the network left taken off, and what was done
    ({"psf_stars", "classes", "profile_stars", "fwhm", "r_max", "stars": [(y, x)], "profile", "left", "levels"}, or
    {"why": ...} when nothing was).

    Rules, all read on the image itself (the sky sigma: the starless image's noise per pixel):

    * the stars image ``lin - starless`` sits on a pedestal (the faint stars of the field the network takes off too):
      the median over 240 px at one pixel in 16;
    * the candidates: the maxima of the stars image HALO_DETECT sky sigmas above it; their FWHM, the median of their
      half-maximum widths; their core, the rings from 2 to 4 FWHM (where the network takes nearly everything off),
      its level the mean of the stars image there;
    * the profile stars: the candidates whose box (HALO_R + the sky ring) lies in the covered image, with no brighter
      star in it, on a flat sky (in the starless image, the rings from HALO_R / 2 out within one sky sigma of the sky
      ring: a nebula under the star would bend its profiles, a linear gradient does not, the rings' medians cancel
      it). Ring medians throughout: the fainter stars in a ring do not count;
    * the extended profile: the input image's rings minus its sky ring, normalised on the core, on the HALO_STARS
      brightest profile stars, 3-sigma clipped mean ring by ring; beyond the core made non-increasing and cut where it
      first meets the sky (falls to 0);
    * what the network leaves: the starless image's rings minus its sky ring, on every profile star, the median of
      each class of HALO_CLASS stars of neighbouring core levels; inside the core, its value at the core's edge; cut
      where it meets the sky beyond its peak;
    * each candidate whose core lies in the covered image, with no brighter star within 8 FWHM (crowded) and whose core
      follows the extended profile (their ratio varies by less than 2 over the core: a saturation wider than the
      core does not): what its class leaves (between classes on the log of the core level; never extrapolated above
      the brightest class, in proportion below the faintest), never more than its own extended profile (scaled on its
      core), when that reaches HALO_LEVEL sky sigma; taken off the starless image ring by ring.
    """
    from scipy import ndimage as ndi
    C, H, W = lin.shape
    grey = all(np.array_equal(a[c, ::97, ::97], a[0, ::97, ::97]) for a in (lin, starless) for c in range(C))
    chans = [0] if grey else list(range(C))
    S = [lin[c] - starless[c] for c in chans]
    sig = [_sky_noise(starless[c], valid) for c in chans]
    if min(sig) <= 0:
        return starless, {"why": "no sky noise"}
    ped = [ndi.median_filter(s[::16, ::16], 15) for s in S]

    def ped_at(i, y, x):
        return float(ped[i][min(y // 16, ped[i].shape[0] - 1), min(x // 16, ped[i].shape[1] - 1)])
    lum = S[0] if grey else np.mean(S, 0)
    sl = float(np.mean(sig))
    sm = ndi.gaussian_filter(lum, 1.0)
    base = float(np.median(np.mean([p for p in ped], 0)))
    lab, n = ndi.label((sm > base + HALO_DETECT * sl) & valid)
    if n == 0:
        return starless, {"why": "no bright star"}
    pos = np.array(ndi.maximum_position(sm, lab, range(1, n + 1)), int)
    peak = sm[pos[:, 0], pos[:, 1]] - np.array([np.mean([ped_at(i, y, x) for i in range(len(chans))])
                                                  for y, x in pos])
    order = np.argsort(-peak)
    pos, peak = pos[order], peak[order]
    # the FWHM: the median half-maximum width of the candidates
    widths = []
    _bx, r15 = _box(15, 15, 15, 31, 31)
    i15 = r15.astype(int).ravel()
    rc = np.bincount(i15, r15.ravel()) / np.bincount(i15)              # the mean radius of each ring
    for y, x in pos[:300]:
        bx, rr = _box(y, x, 15, H, W)
        if rr.shape != r15.shape:
            continue
        pl = np.mean([ped_at(i, y, x) for i in range(len(chans))])
        prof = _ring_medians((lum[bx] - pl).ravel(), i15, 16)
        below = np.nonzero(prof < prof[0] / 2)[0]
        if below.size and below[0] > 0:
            k = below[0]
            widths.append(2 * np.interp(prof[0] / 2, [prof[k], prof[k - 1]], [rc[k], rc[k - 1]]))
    if not widths:
        return starless, {"why": "no star width"}
    fwhm = float(np.median(widths))
    r_lo, r_hi = int(round(2 * fwhm)), max(int(round(4 * fwhm)), int(round(2 * fwhm)) + 2)
    core = slice(r_lo, r_hi + 1)
    RB = HALO_R + HALO_RING
    gy, gx = np.mgrid[-RB:RB + 1, -RB:RB + 1]
    gr = np.hypot(gy, gx)
    ring_y, ring_x = gy[gr >= HALO_R], gx[gr >= HALO_R]                # the sky ring of a star

    def sky_ring(y, x):
        """The pixels of the sky ring around (y, x) that lie in the covered image (None: less than half of it)."""
        ay, ax = ring_y + y, ring_x + x
        k = (ay >= 0) & (ay < H) & (ax >= 0) & (ax < W)
        ay, ax = ay[k], ax[k]
        k = valid[ay, ax]
        return (ay[k], ax[k]) if k.sum() >= 0.5 * ring_y.size else None

    def core_of(y, x, ring=False):
        """A star's core in each channel of the stars image, above its pedestal (``ring``: on its sky ring, else the
        pedestal's map) (None: not measurable: near the edge, outside the covered image)."""
        if not (r_hi < y < H - r_hi - 1 and r_hi < x < W - r_hi - 1):
            return None
        bx, rr = _box(y, x, r_hi + 1, H, W)
        if not valid[bx].all():
            return None
        if ring:
            at = sky_ring(y, x)
            if at is None:
                return None
            under = [float(np.median(s[at])) for s in S]
        else:
            under = [ped_at(i, y, x) for i in range(len(S))]
        ri = rr.astype(int).ravel()
        return [_ring_medians((s[bx] - u).ravel(), ri, r_hi + 1) for s, u in zip(S, under)]

    # the profile stars: the extended profile on the input image, and what the network left on the starless image
    L = lin[0] if grey else lin.mean(0)
    Ls = starless[0] if grey else starless.mean(0)
    from scipy.spatial import cKDTree
    tree = cKDTree(pos)
    profs, lefts, levels = [], [], []
    for (y, x), pk in zip(pos, peak):
        if not (RB <= y < H - RB and RB <= x < W - RB):
            continue
        if np.any(peak[tree.query_ball_point((y, x), RB)] > pk):
            continue
        bx, rr = _box(y, x, RB, H, W)
        if not valid[bx].all():
            continue
        ring = rr >= HALO_R
        z = Ls[bx]
        sky = np.median(z[ring])
        outer = [np.median(z[(rr >= a) & (rr < a + HALO_R // 8)]) for a in range(HALO_R // 2, HALO_R, HALO_R // 8)]
        if max(abs(v - sky) for v in outer) > sl:
            continue
        cores = core_of(y, x, ring=True)
        if cores is None:
            continue
        ri = rr.astype(int).ravel()
        c = L[bx]
        prof = _ring_medians((c - np.median(c[ring])).ravel(), ri, HALO_R)
        norm = float(np.mean(prof[core]))
        if norm <= 0:
            continue
        if len(profs) < HALO_STARS:
            profs.append(prof / norm)
        z = [starless[c][bx] for c in chans]
        lefts.append([_ring_medians((a - np.median(a[ring])).ravel(), ri, HALO_R) for a in z])
        levels.append(float(np.mean([np.mean(p[core]) for p in cores])))
    if len(profs) < HALO_FEWEST:
        return starless, {"why": f"{len(profs)} isolated bright stars on a flat sky, {HALO_FEWEST} needed"}
    # the extended profile: their 3-sigma clipped mean, ring by ring, non-increasing beyond the core, to where it meets
    # the sky
    profs = np.array(profs)
    med = np.median(profs, 0)
    spread = 1.4826 * np.median(np.abs(profs - med), 0)
    kept = np.abs(profs - med) <= 3 * spread
    P = np.where(kept, profs, 0).sum(0) / np.maximum(kept.sum(0), 1)
    P[r_hi:] = np.minimum.accumulate(ndi.median_filter(P, 9, mode="nearest")[r_hi:])
    out_r = np.nonzero((P <= 0) & (np.arange(HALO_R) > r_hi))[0]
    r_max = int(out_r[0]) if out_r.size else HALO_R
    P[r_max:] = 0.0
    # what the network leaves, by class of core level: HALO_CLASS stars per class, the median of their rings, from the
    # brightest; within the core, the value at its outer edge; to where it meets the sky
    order = np.argsort(levels)[::-1]
    groups = [order[k:k + HALO_CLASS] for k in range(0, len(order), HALO_CLASS)]
    if len(groups) > 1 and len(groups[-1]) < HALO_CLASS:
        groups[-2] = np.concatenate(groups[-2:])
        groups.pop()
    centre, templates = [], []
    for g in groups:
        centre.append(float(np.median(np.log([levels[k] for k in g]))))
        t = []
        for i in range(len(chans)):
            m = np.nan_to_num(np.median([lefts[k][i] for k in g], 0))
            m = ndi.median_filter(m, 9, mode="nearest")
            m[:r_lo] = m[r_lo]
            top = int(np.argmax(m))
            zero = np.nonzero(m[top:] <= 0)[0]                         # where it meets the sky, beyond its peak
            if zero.size:
                m[top + zero[0]:] = 0.0
            t.append(np.maximum(m, 0.0))
        templates.append(t)
    centre = np.array(centre[::-1])                                     # faintest class first
    templates = np.array(templates[::-1])                               # (class, channel, radius)

    def template(level):
        """What the network leaves around a star of this core level, per channel: between the classes on the log of
        the level; above the brightest, the brightest's (never extrapolated); below the faintest, the faintest's in
        proportion to the level."""
        lv = np.log(max(level, 1e-12))
        if lv <= centre[0]:
            return templates[0] * np.exp(lv - centre[0])
        if lv >= centre[-1]:
            return templates[-1]
        k = int(np.searchsorted(centre, lv)) - 1
        w = (lv - centre[k]) / (centre[k + 1] - centre[k])
        return (1 - w) * templates[k] + w * templates[k + 1]

    # each bright star
    take = [np.zeros((H, W), np.float32) for _ in chans]
    done = []
    centres = np.arange(HALO_R) + 0.5
    faint = 0
    for (y, x), pk in zip(pos, peak):
        if faint >= 50:
            break                                                       # 50 in a row with nothing left: the rest are fainter
        if np.any(peak[tree.query_ball_point((y, x), 2 * r_hi)] > pk):
            continue                                                    # crowded: a brighter star over its core
        cores = core_of(y, x)
        if cores is None:
            continue
        q = [p[core] / P[core] for p in cores]
        if not all(a.min() > 0 and a.max() < 2 * a.min() for a in q):
            continue                                                    # its core does not follow the profile
        scale = [float(np.median(a)) for a in q]
        t = template(float(np.mean([np.mean(p[core]) for p in cores])))
        if max(float(a.max()) / s for a, s in zip(t, sig)) < HALO_LEVEL:
            faint += 1
            continue                                                    # nothing measurable left
        faint = 0
        rem = [np.minimum(t[i], scale[i] * P) for i in range(len(chans))]   # never more than its own wing
        if max(float(r.max()) / s for r, s in zip(rem, sig)) < HALO_LEVEL:
            continue
        ext = max(int(np.nonzero(r > 0)[0][-1]) + 1 if r.any() else 0 for r in rem)
        bx, rr = _box(y, x, ext, H, W)
        inside = (rr < ext) & valid[bx]
        for i, r in enumerate(rem):
            take[i][bx] += np.where(inside, np.interp(rr, centres[:ext], r[:ext], left=r[0], right=0.0),
                                    0).astype(np.float32)
        done.append((int(y), int(x)))
    out = starless.copy()
    for i, c in enumerate(chans):
        out[c] -= take[i]
    if grey:
        out[1:] = out[0]
    return out, {"psf_stars": len(profs), "classes": len(groups), "profile_stars": len(levels), "fwhm": fwhm,
                 "r_max": r_max, "stars": done, "profile": P.tolist(), "left": templates.tolist(),
                 "levels": np.exp(centre).tolist()}


def fwd(v):
    """The network's stretch (numpy or torch), v in noise units."""
    mod = np if isinstance(v, np.ndarray) else torch
    return mod.asinh(v / STRETCH) / SCALE


def inv(y):
    mod = np if isinstance(y, np.ndarray) else torch
    return mod.sinh(y * SCALE) * STRETCH


def network(dev) -> NAFNet:
    """The bundled network (resources/stars), in float32."""
    if not MODEL.exists():
        raise FileNotFoundError(f"the star removal network is missing: {MODEL}")
    net = NAFNet().to(dev)
    state = torch.load(MODEL, map_location=dev, weights_only=False)["net"]
    net.load_state_dict({k: v.float() for k, v in state.items()})
    net.eval()
    return net


@torch.no_grad()
def apply(net, img: np.ndarray, dev, ctx: Context | None = None, tile: int = TILE, overlap: int = OVERLAP,
          batch: int = BATCH) -> np.ndarray:
    """The network over a whole (3, H, W) image in noise units, in tiles larger than the largest stars (an image smaller
    than such a tile: tiles of the training size), blended with a cosine window; returns the starless image in the same
    units. On a card with little free memory the tiles go through fewer at a time, then are cut smaller; a memory error
    does the same (the batch size changes nothing in the result, the tile a little)."""
    if min(img.shape[1:]) < tile:
        tile, overlap = min(tile, NET_TILE), min(overlap, NET_OVERLAP)
    tile_gb = TILE_GB * (tile / NET_TILE) ** 2
    fits = devices.tile_that_fits(tile, dev, tile_gb)
    if fits < tile:
        overlap, tile_gb = overlap * fits // tile, tile_gb * (fits / tile) ** 2
        tile = fits
    batch = max(1, batch * NET_TILE ** 2 // tile ** 2)                # the same memory whatever the tile
    free = devices.free_memory(dev)
    if free is not None:
        batch = max(1, min(batch, int(free / 2 ** 30 / (tile_gb * 1.5))))
    while True:
        try:
            return _apply(net, img, dev, ctx, tile, overlap, batch)
        except Exception as e:  # noqa: BLE001 - only a memory error is retried
            if not devices.is_out_of_memory(e):
                raise
            devices.empty_cache(dev)
            if batch > 1:
                batch //= 2
            elif tile // 2 >= devices.SMALLEST_TILE:
                tile, overlap = tile // 2, overlap // 2
            else:
                raise


def _apply(net, img: np.ndarray, dev, ctx: Context | None, tile: int, overlap: int, batch_size: int) -> np.ndarray:
    img = np.asarray(img, np.float32)
    C, H, W = img.shape
    if H < tile or W < tile:                                    # a tiny image: padded to one tile
        ph, pw = max(0, tile - H), max(0, tile - W)
        return _apply(net, np.pad(img, ((0, 0), (0, ph), (0, pw)), mode="reflect"), dev, ctx, tile, overlap,
                      batch_size)[:, :H, :W]
    step = tile - overlap
    out = np.zeros_like(img, np.float32)
    wsum = np.zeros((H, W), np.float32)
    win1 = np.sin(np.linspace(0, np.pi, tile + 2)[1:-1]) ** 2 + 1e-3
    win = np.outer(win1, win1).astype(np.float32)
    ys = list(range(0, H - tile + 1, step)) + ([H - tile] if (H - tile) % step else [])
    xs = list(range(0, W - tile + 1, step)) + ([W - tile] if (W - tile) % step else [])
    for i, y in enumerate(ys):
        batch = [img[:, y:y + tile, x:x + tile] for x in xs]
        for j in range(0, len(batch), batch_size):
            b = torch.from_numpy(np.ascontiguousarray(np.stack(batch[j:j + batch_size]))).to(dev)
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                r = net(fwd(b)).float()
            r = inv(r).cpu().numpy()
            for k, x in enumerate(xs[j:j + batch_size]):
                out[:, y:y + tile, x:x + tile] += r[k] * win
                wsum[y:y + tile, x:x + tile] += win
        if ctx is not None:
            ctx.check_stop()
            ctx.progress(0.05 + 0.8 * (i + 1) / len(ys), f"taking the stars off: row {i + 1}/{len(ys)}")
    return out / wsum
