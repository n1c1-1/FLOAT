"""Star removal, issue #178: two candidate networks (A, B) trained on one fixed bank of examples.

    python tools/bench/stars_bank.py profile                 # the share of the example generation in a training step
    python tools/bench/stars_bank.py prep                    # the choices written in the bank's folder (inputs, FWHM)
    python tools/bench/stars_bank.py write [--n 20000]       # the bank (resumable)
    python tools/bench/stars_bank.py check0                  # B's output at step 0 against the shipped network
    python tools/bench/stars_bank.py train --arch A --hours 3 --lr 1e-4
    python tools/bench/stars_bank.py train --arch B --hours 3 --lr 1e-4

The data are stars_train.training_pairs() (the old reviewed fields and the 19 Nov pairs, the judge pair and the
held-out fields left out) with the choices of ``prep``: the denoised input for m1, m82, m57, the FWHM of the other fields
for m97; the old and the new fields drawn about half each. An example is made as stars_train.train makes one (a tile of
a pair, synthetic stars on 60 % of them, the Deconvolution network's pass on 50 %, input and target alike), written once
as 512 px patches: input, target, loss mask. The patches are stored in the network's own stretch (stars_train.fwd), in
float16: the stars of some fields reach millions of sigmas, past float16 in the linear data.

A: the shipped NAFNet fine-tuned, on 256 px crops of the bank. B: the same network with one more, coarser level after the
middle blocks (1/32), joined back through a projection that starts at zero (B starts equal to the shipped network),
on the whole 512 px patches. Both from the shipped weights, same bank, same time.

Files: F:/astro_essais/banque_reseau_etoiles/ (the bank), F:/astro_essais/stars_train/bank_<A|B>/ (weights, history).
"""
from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "bench"))

import stars_train as st  # noqa: E402
from astro_gpu.stages import star_removal as sr  # noqa: E402
from astro_gpu.stages.star_removal import MODEL, NAFNet, _NAFBlock, fwd, inv  # noqa: E402

BANK = Path("F:/astro_essais/banque_reseau_etoiles")
TRAIN = st.TRAIN
PATCH = 512
DENOISED = ("m1", "m82", "m57")                  # their input is input_denoised.npy (all other inputs are denoised)
NVAL = 400                                       # the last examples of the bank, never trained on
SYNTHETIC, DECONV = 0.6, 0.5                     # as stars_train.train's default and the plan's share
log = st.log


# ---------------------------------------------------------------------------------------------- the networks
def set_windows(net: NAFNet, base: int) -> None:
    """The channel attention's window of each level, in the level's own pixels (base at full size)."""
    for lv, m in enumerate(net.encs):
        for b in m:
            b.sca[0].window = base >> lv
    for b in net.mid:
        b.sca[0].window = base >> len(net.encs)
    for i, m in enumerate(net.decs):
        for b in m:
            b.sca[0].window = base >> (len(net.encs) - 1 - i)
    if hasattr(net, "mid5"):
        for b in net.mid5:
            b.sca[0].window = base >> (len(net.encs) + 1)


class NAFNetWide(NAFNet):
    """The shipped NAFNet and one coarser level after its middle blocks: the middle output is downsampled to 1/32,
    goes through ``extra`` NAFBlocks, is upsampled, and is added back through a 1x1 projection that starts at zero, so
    the network starts exactly equal to the shipped one while its receptive field reaches 2x as far."""

    def __init__(self, extra: int = 4, width=32, enc=(2, 2, 4, 8), **kw):
        super().__init__(width=width, enc=enc, **kw)
        c = width * 2 ** len(enc)
        win = sr.WINDOW // 2 ** (len(enc) + 1)
        self.down5 = nn.Conv2d(c, 2 * c, 2, 2)
        self.mid5 = nn.Sequential(*[_NAFBlock(2 * c, win) for _ in range(extra)])
        self.up5 = nn.Sequential(nn.Conv2d(2 * c, 4 * c, 1, bias=False), nn.PixelShuffle(2))
        self.proj5 = nn.Conv2d(c, c, 1)
        nn.init.zeros_(self.proj5.weight)
        nn.init.zeros_(self.proj5.bias)
        self.pad = 2 ** (len(enc) + 1)

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
        x = x + self.proj5(self.up5(self._run(self.mid5, self.down5(x))))
        for d, up, sk in zip(self.decs, self.ups, reversed(skips)):
            x = self._run(d, up(x) + sk)
        return (self.ending(x) + x0)[:, :, :H, :W]


def build(arch: str, dev, recompute=False, init=True):
    """The network of a candidate, from the shipped weights (all existing weights loaded), windows set for its patch."""
    net = (NAFNet if arch == "A" else NAFNetWide)(recompute=recompute)
    if init:
        state = {k: v.float() for k, v in torch.load(MODEL, map_location="cpu", weights_only=False)["net"].items()}
        miss, unexp = net.load_state_dict(state, strict=False)
        assert not unexp and all(k.split(".")[0] in ("down5", "mid5", "up5", "proj5") for k in miss), (miss, unexp)
    set_windows(net, 256 if arch == "A" else 512)
    return net.to(dev)


def load_net(path, arch: str, dev):
    net = build(arch, dev, init=False)
    net.load_state_dict({k: v.float() for k, v in torch.load(path, map_location=dev, weights_only=False)["net"].items()})
    return net.eval()


# ---------------------------------------------------------------------------------------------- the data
def noise_bias(inp, tgt):
    from stars_nb_pairs import noise_check
    return noise_check(inp, tgt)


def prep() -> int:
    """m1, m82, m57: the denoised input's sky offset against the target; m97: the FWHM of the input's own isolated stars."""
    from scipy import ndimage
    BANK.mkdir(parents=True, exist_ok=True)
    nb = json.loads((st.NB / "fields.json").read_text())
    out = {"denoised": {}}
    for n in DENOISED:
        t = np.load(st.NB / n / "starless_fixed.npy", mmap_mode="r")
        for fn in ("input.npy", "input_denoised.npy"):
            a = np.load(st.NB / n / fn, mmap_mode="r")
            hf_in, hf_t, bias = noise_bias(a, t)
            # calm sky: the quietest 40 % (as noise_check), minus the stars: median of (input - target) per colour
            sl = (slice(None), slice(0, None, 3), slice(0, None, 3))
            aa, tt = np.asarray(a[sl], np.float32), np.asarray(t[sl], np.float32)
            g = ndimage.uniform_filter(tt.mean(0), 9)
            m = (g < np.percentile(g, 40)) & ((aa - tt).max(0) < 3.0)
            med = [float(np.median((aa[c] - tt[c])[m])) for c in range(3)]
            out["denoised"].setdefault(n, {})[fn] = {"median_input_minus_target_by_colour": [round(v, 4) for v in med],
                                                     "smooth_sky_bias_target_minus_input": round(bias, 4),
                                                     "input_hf_noise": round(hf_in, 4)}
            log(f"{n} {fn}: median(input - target) on calm sky {[round(v, 4) for v in med]} sigmas; "
                f"smooth bias {bias:+.4f}; input noise {hf_in:.3f}")
    # m97: its stars, measured on the input itself (isolated, unsaturated, azimuthal profile)
    A = np.load(st.NB / "m97" / "input.npy").astype(np.float32)
    top = nb["m97"]["top"]
    L = A.mean(0)
    bg = ndimage.zoom(ndimage.median_filter(L[::4, ::4], 9), 4, order=1)[:L.shape[0], :L.shape[1]]
    L = L - bg
    pk = (L == ndimage.maximum_filter(L, 21)) & (L > 30) & (L < 0.25 * top)
    ys, xs = np.nonzero(pk[25:-25, 25:-25])
    ys, xs = ys + 25, xs + 25
    yy, xx = np.mgrid[-12:13, -12:13]
    fw = []
    for y, x in zip(ys[np.argsort(-L[ys, xs])[:600]], xs[np.argsort(-L[ys, xs])[:600]]):
        c = L[y - 12:y + 13, x - 12:x + 13]
        if (c > 0.3 * c[12, 12]).sum() > 60:
            continue
        w = c.clip(0) * (np.hypot(xx, yy) <= 8)
        cx, cy = (w * xx).sum() / w.sum(), (w * yy).sum() / w.sum()
        b = np.floor(np.hypot(xx - cx, yy - cy) * 2).astype(int)
        prof = np.array([c[b == i].mean() if (b == i).any() else np.nan for i in range(16)])
        ok = prof < 0.5 * prof[0]
        if np.isnan(prof).any() or not ok.any() or not ok[1:].any():
            continue
        k = int(np.argmax(ok))
        f = (prof[k - 1] - 0.5 * prof[0]) / (prof[k - 1] - prof[k] + 1e-9)
        fw.append(2 * ((k - 1) / 2 + 0.25 + f * 0.5))
    fw = np.array(fw)
    others = sorted(p["fwhm"] for p in st.training_pairs(nb=True) if p["field"] != "m97")
    med_other = float(np.median(others))
    small = int((fw < 2).sum())
    out["m97"] = {"measured_n": len(fw), "median": round(float(np.median(fw)), 2),
                  "share_below_2px": round(small / len(fw), 2), "p10_p90": [round(float(v), 2) for v in
                                                                             np.percentile(fw, [10, 90])],
                  "stored_in_fields_json": nb["m97"]["fwhm_px"], "median_other_pairs": round(med_other, 2)}
    # the isolated stars split in two populations (about 1 px and 4-6 px): not a reliable single value
    out["fwhm"] = {"m97": round(med_other, 2)}
    log(f"m97 FWHM on the input's own stars: n {len(fw)}, median {np.median(fw):.2f}, {small / len(fw):.0%} under 2 px, "
        f"10-90 % {np.percentile(fw, [10, 90]).round(2)} -> unreliable; used: median of the other pairs {med_other:.2f}")
    (BANK / "choix.json").write_text(json.dumps(out, indent=1))
    return 0


def bank_pairs(small_ok: bool = False):
    """training_pairs with the choices, a group tag (old / new) and the pairs too small for a patch left out."""
    choix = json.loads((BANK / "choix.json").read_text())
    new = set(json.loads((st.NB / "fields.json").read_text()))
    out, dropped = [], []
    for p in st.training_pairs(nb=True):
        p = dict(p)
        p["group"] = "new" if p["field"] in new and p["name"] in new else "old"
        if p["name"] in DENOISED:
            p["input"] = st.NB / p["name"] / "input_denoised.npy"
        if p["name"] in choix["fwhm"]:
            p["fwhm"] = choix["fwhm"][p["name"]]
        H, W = np.load(p["input"], mmap_mode="r").shape[1:]
        if min(H, W) < PATCH + 2 * st.DEC_PAD + 1:
            dropped.append(p["name"])
            continue
        out.append(p)
    return out, dropped


class Tiles(st.Tiles):
    """stars_train.Tiles with the field drawn half from the old fields, half from the new ones."""

    def __init__(self, pairs, seed):
        super().__init__(pairs, seed)
        self.groups = {g: sorted({p["field"] for p in pairs if p["group"] == g}) for g in ("old", "new")}

    def get(self, n, P=PATCH):
        xin, tgt, msk, fw, top = [], [], [], [], []
        while len(xin) < n:
            g = self.groups["new" if self.rng.random() < 0.5 else "old"]
            idx = self.by_field[g[self.rng.integers(len(g))]]
            i = idx[self.rng.integers(len(idx))]
            A, T, I = self.arr[i]
            H, W = A.shape[1:]
            y, x = int(self.rng.integers(0, H - P)), int(self.rng.integers(0, W - P))
            a = np.asarray(A[:, y:y + P, x:x + P], np.float32)
            inside = np.abs(a).max(0) > 0
            m = inside & (~np.asarray(I[y:y + P, x:x + P]) if I is not None else inside)
            if m.mean() < 0.5:
                continue
            t = np.asarray(T[:, y:y + P, x:x + P], np.float32)
            k = int(self.rng.integers(4))
            a, t, m = np.rot90(a, k, (1, 2)), np.rot90(t, k, (1, 2)), np.rot90(m, k)
            if self.rng.random() < 0.5:
                a, t, m = a[:, :, ::-1], t[:, :, ::-1], m[:, ::-1]
            xin.append(np.ascontiguousarray(a))
            tgt.append(np.ascontiguousarray(t))
            msk.append(np.ascontiguousarray(m))
            fw.append(self.pairs[i]["fwhm"])
            top.append(self.pairs[i]["top"])
            self.last_field = getattr(self, "last_field", []) + [self.pairs[i]["name"]]
        return np.stack(xin), np.stack(tgt), np.stack(msk)[:, None], np.array(fw), np.array(top)


def make_batch(data, dec, gen, dev, n, P, timers=None, got=None):
    """n examples as stars_train.train makes them (tiles, synthetic stars, the deconvolution pass), centred on P px."""
    def tick(k, t0):
        if timers is not None:
            if dev.type == "cuda":
                torch.cuda.synchronize()
            timers[k] = timers.get(k, 0.0) + time.time() - t0
        return time.time()
    SIZE = P + 2 * st.DEC_PAD
    t0 = time.time()
    if got is None:
        data.last_field = []
        got = data.get(n, SIZE)
    a, t, m, fw, top = got
    a, t, m = (torch.from_numpy(v).to(dev) for v in (a, t, m))
    m = m.float()
    level = torch.from_numpy(top).float().to(dev)[:, None, None, None]
    t0 = tick("tiles", t0)
    pick = torch.rand(n, generator=gen, device=dev) < SYNTHETIC
    if pick.any():
        k = int(pick.sum())
        pk = pick.cpu().numpy()
        stars = st.synthetic_stars(k, fw[pk], top[pk], gen, dev, SIZE)
        t0 = tick("synthetic", t0)
        base = torch.where((torch.rand(k, 1, 1, 1, generator=gen, device=dev) < 0.5), t[pick], a[pick])
        sat = torch.from_numpy(top[pk]).float().to(dev)[:, None, None, None] \
            * (0.5 + 0.5 * torch.rand(k, 1, 1, 1, generator=gen, device=dev))
        a[pick] = torch.minimum(base + stars, torch.maximum(base, sat))
        level[pick] = sat
    behind = torch.rand(n, generator=gen, device=dev) < DECONV
    t0 = time.time()
    if behind.any():
        a[behind], t[behind] = dec(a[behind], t[behind], level[behind])
    tick("deconvolution", t0)
    a, t, m = st.centre(a, P).contiguous(), st.centre(t, P).contiguous(), st.centre(m, P).contiguous()
    return a, t, m, behind


# ---------------------------------------------------------------------------------------------- step 1
def profile(device: str, steps: int = 40) -> int:
    """Where a training step of stars_train.train (batch 8, 256 px, synthetic 0.6, deconvolved 0.5) spends its time."""
    dev = torch.device(device)
    torch.backends.cudnn.benchmark = True
    pairs, _ = bank_pairs()
    data = Tiles(pairs, 0)
    dec = st.Deconvolved(dev)
    gen = torch.Generator(device=dev)
    gen.manual_seed(0)
    net = build("A", dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-5)
    timers, tt = {}, 0.0
    for i in range(steps + 30):
        if i == 30:
            timers, tt = {}, 0.0
        a, t, m, _ = make_batch(data, dec, gen, dev, 8, 256, timers)
        t0 = time.time()
        net.train()
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            pred = net(fwd(a)).float()
        loss = ((pred - fwd(t)).abs() * m).sum() / (m.sum() * 3 + 1)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        tt += time.time() - t0
    tot = sum(timers.values()) + tt
    log(f"step = {tot / steps:.3f} s: " + ", ".join(f"{k} {v / steps:.3f} s ({v / tot:.0%})" for k, v in timers.items())
        + f", network step {tt / steps:.3f} s ({tt / tot:.0%}); generation share {(tot - tt) / tot:.0%}")
    return 0


# ---------------------------------------------------------------------------------------------- the bank
def write(n: int, device: str, chunk: int = 8) -> int:
    from numpy.lib.format import open_memmap
    dev = torch.device(device)
    torch.backends.cudnn.benchmark = True
    BANK.mkdir(parents=True, exist_ok=True)
    pairs, dropped = bank_pairs()
    log(f"{len(pairs)} pairs ({sum(p['group'] == 'old' for p in pairs)} old, {sum(p['group'] == 'new' for p in pairs)} new), "
        f"left out as smaller than a patch: {dropped}")
    shapes = {"inp.npy": (n, 3, PATCH, PATCH), "tgt.npy": (n, 3, PATCH, PATCH), "msk.npy": (n, PATCH, PATCH)}
    state = BANK / "progress.json"
    done = json.loads(state.read_text())["done"] if state.exists() else 0
    mm = {}
    for fn, sh in shapes.items():
        dt = np.float16 if fn != "msk.npy" else np.uint8
        mm[fn] = open_memmap(BANK / fn, mode="r+" if (BANK / fn).exists() else "w+", dtype=dt, shape=sh)
    meta = json.loads((BANK / "meta.json").read_text()) if (BANK / "meta.json").exists() else \
        {"field": [None] * n, "deconvolved": [False] * n}
    dec = st.Deconvolved(dev)
    t0, start = time.time(), done
    c = done // chunk
    SIZE = PATCH + 2 * st.DEC_PAD
    ahead: queue.Queue = queue.Queue(maxsize=3)

    def reader():                                          # the tiles read from the disk while the card works
        d0, c0 = done, c
        while d0 < n:
            k0 = min(chunk, n - d0)
            data = Tiles(pairs, 1000 + c0)                 # reseeded by chunk: the bank resumes where it stopped
            data.last_field = []
            ahead.put((data, k0, data.get(k0, SIZE)))
            d0 += k0
            c0 += 1

    threading.Thread(target=reader, daemon=True).start()
    while done < n:
        data, k, got = ahead.get()
        gen = torch.Generator(device=dev)
        gen.manual_seed(1000 + c)
        a, t, m, behind = make_batch(data, dec, gen, dev, k, PATCH, got=got)
        mm["inp.npy"][done:done + k] = fwd(a).cpu().numpy().astype(np.float16)
        mm["tgt.npy"][done:done + k] = fwd(t).cpu().numpy().astype(np.float16)
        mm["msk.npy"][done:done + k] = m[:, 0].cpu().numpy().astype(np.uint8)
        meta["field"][done:done + k] = data.last_field
        meta["deconvolved"][done:done + k] = behind.cpu().tolist()
        done += k
        c += 1
        if c % 25 == 0 or done == n:
            for v in mm.values():
                v.flush()
            (BANK / "meta.json").write_text(json.dumps(meta))
            state.write_text(json.dumps({"done": done}))
            el = time.time() - t0
            got = done - start
            log(f"{done}/{n} examples, {el / 60:.1f} min (about {(n - done) * el / max(1, got) / 60:.0f} min left)")
    return 0


# ---------------------------------------------------------------------------------------------- B at step 0
def check0(device: str) -> int:
    dev = torch.device(device)
    ref = sr.network(dev).float()
    for p in ref.parameters():
        p.requires_grad_(False)
    b = build("B", dev).eval()
    set_windows(b, 256)                                # the same windows as the shipped network, for the comparison
    n = sum(p.numel() for p in b.parameters()) / 1e6
    worst = 0.0
    d = np.load(st.NB / "57_sho" / "input.npy", mmap_mode="r")
    for y, x in ((100, 100), (1500, 2000), (3000, 500)):
        v = torch.from_numpy(np.asarray(d[:, y:y + 512, x:x + 512], np.float32))[None].to(dev)
        with torch.no_grad():
            worst = max(worst, float((ref(fwd(v)) - b(fwd(v))).abs().max()))
    v = torch.randn(1, 3, 256, 256, device=dev)
    with torch.no_grad():
        worst = max(worst, float((ref(v) - b(v)).abs().max()))
    log(f"B at step 0 against the shipped network: max difference {worst:.2e} (B {n:.1f} M parameters, "
        f"shipped {sum(p.numel() for p in ref.parameters()) / 1e6:.1f} M)")
    return 0


# ---------------------------------------------------------------------------------------------- training
class Bank:
    def __init__(self):
        self.inp = np.load(BANK / "inp.npy", mmap_mode="r")
        self.tgt = np.load(BANK / "tgt.npy", mmap_mode="r")
        self.msk = np.load(BANK / "msk.npy", mmap_mode="r")
        done = json.loads((BANK / "progress.json").read_text())["done"]    # the bank may be stopped short of its size
        self.n = done - NVAL                                # the last NVAL written are the validation's

    def batch(self, idx, P, rng, augment=True):
        a, t, m = [], [], []
        for i in idx:
            y, x = (int(rng.integers(0, PATCH - P + 1)), int(rng.integers(0, PATCH - P + 1))) if P < PATCH else (0, 0)
            sl = (slice(None), slice(y, y + P), slice(x, x + P))
            aa, tt, mm = (np.asarray(self.inp[i][sl]), np.asarray(self.tgt[i][sl]),
                          np.asarray(self.msk[i][y:y + P, x:x + P]))
            if augment:
                k = int(rng.integers(4))
                aa, tt, mm = np.rot90(aa, k, (1, 2)), np.rot90(tt, k, (1, 2)), np.rot90(mm, k)
                if rng.random() < 0.5:
                    aa, tt, mm = aa[:, :, ::-1], tt[:, :, ::-1], mm[:, ::-1]
            a.append(np.ascontiguousarray(aa))
            t.append(np.ascontiguousarray(tt))
            m.append(np.ascontiguousarray(mm))
        return np.stack(a), np.stack(t), np.stack(m)[:, None]


def train(arch: str, hours: float, device: str, lr: float, batch: int, tag: str, val_n: int = 200) -> int:
    dev = torch.device(device)
    torch.backends.cudnn.benchmark = True
    out = TRAIN / tag
    out.mkdir(parents=True, exist_ok=True)
    P = 256 if arch == "A" else 512
    bank = Bank()
    torch.manual_seed(0)
    net = build(arch, dev)
    log(f"{tag}: arch {arch}, {sum(p.numel() for p in net.parameters()) / 1e6:.1f} M parameters, patch {P}, "
        f"batch {batch}, lr {lr}, bank {bank.n} train / {NVAL} validation, {hours} h")
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4, betas=(0.9, 0.9))
    q: queue.Queue = queue.Queue(maxsize=8)
    stop = threading.Event()

    def loader(seed):
        rng = np.random.default_rng(seed)
        while not stop.is_set():
            idx = rng.integers(0, bank.n, batch)
            q.put(bank.batch(idx, P, rng))

    threads = [threading.Thread(target=loader, args=(s,), daemon=True) for s in range(3)]
    for th in threads:
        th.start()
    vidx = np.arange(bank.n, bank.n + NVAL, NVAL // val_n)    # val_n validation examples, whole patches, no augmentation
    va = bank.batch(vidx, PATCH, np.random.default_rng(0), augment=False)
    deconv_flag = np.array(json.loads((BANK / "meta.json").read_text())["deconvolved"])[vidx]

    def validate():
        net.eval()
        al, st_, sd = [], [], []
        with torch.no_grad():
            for j in range(0, len(vidx), 4):
                x, tt, m = (torch.from_numpy(v[j:j + 4].astype(np.float32)).to(dev) for v in va)
                with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                    pred = net(x).float()
                err = (pred - tt).abs() * m
                al.append(float(err.sum() / (m.sum() * 3 + 1)))
                star = ((x - tt).abs().max(1, keepdim=True)[0] > 0.01) & (m > 0)
                e = (pred - tt).abs().max(1, keepdim=True)[0]
                for b in range(len(x)):
                    s = star[b]
                    (sd if deconv_flag[j + b] else st_).append(float(e[b][s].mean()) if s.any() else 0.0)
        net.train()
        return float(np.mean(al)), float(np.mean(st_)), float(np.mean(sd))

    hist, best, run, it = [], None, None, 0
    t0, lastval, warm = time.time(), time.time(), 200
    v0 = validate()
    log(f"step 0: validation {v0[0]:.5f}, on the stars {v0[1]:.5f}, behind a deconvolution {v0[2]:.5f}")
    hist.append({"step": 0, "minutes": 0, "val": v0[0], "val_stars": v0[1], "val_stars_deconv": v0[2]})
    net.train()
    while True:
        frac = (time.time() - t0) / (hours * 3600)
        if frac >= 1.0:
            break
        for g in opt.param_groups:
            g["lr"] = lr * min(1.0, (it + 1) / warm) * 0.5 * (1 + np.cos(np.pi * frac))
        a, t, m = (torch.from_numpy(v.astype(np.float32)).to(dev, non_blocking=True) for v in q.get())
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            pred = net(a).float()
        loss = ((pred - t).abs() * m).sum() / (m.sum() * 3 + 1)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 0.1)
        opt.step()
        run = loss.item() if run is None else 0.99 * run + 0.01 * loss.item()
        it += 1
        if time.time() - lastval > 600:
            lastval = time.time()
            v = validate()
            score = v[1] + v[2]
            hist.append({"step": it, "minutes": round((time.time() - t0) / 60, 1), "train": round(run, 5), "val": v[0],
                         "val_stars": v[1], "val_stars_deconv": v[2], "lr": opt.param_groups[0]["lr"]})
            (out / "history.json").write_text(json.dumps(hist, indent=1))
            torch.save({"net": net.state_dict(), "opt": opt.state_dict(), "it": it}, out / "net.pt")
            if best is None or score < best:
                best = score
                torch.save({"net": net.state_dict(), "it": it}, out / "best.pt")
            log(f"step {it}: train {run:.5f}, validation {v[0]:.5f}, on the stars {v[1]:.5f}, behind a deconvolution "
                f"{v[2]:.5f}, lr {opt.param_groups[0]['lr']:.1e}, {(time.time() - t0) / it:.2f} s/step, queue {q.qsize()}")
    v = validate()
    hist.append({"step": it, "minutes": round((time.time() - t0) / 60, 1), "train": round(run, 5), "val": v[0],
                 "val_stars": v[1], "val_stars_deconv": v[2], "final": True})
    (out / "history.json").write_text(json.dumps(hist, indent=1))
    torch.save({"net": net.state_dict(), "opt": opt.state_dict(), "it": it}, out / "net.pt")
    if best is None or v[1] + v[2] < best:
        torch.save({"net": net.state_dict(), "it": it}, out / "best.pt")
    stop.set()
    log(f"done: {it} steps in {(time.time() - t0) / 60:.0f} min, final validation {v}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=("profile", "prep", "write", "check0", "train"))
    ap.add_argument("--arch", choices=("A", "B"), default="A")
    ap.add_argument("--hours", type=float, default=3.0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--val-n", type=int, default=200)
    a = ap.parse_args(argv)
    if a.cmd == "prep":
        return prep()
    if a.cmd == "profile":
        return profile(a.device)
    if a.cmd == "write":
        return write(a.n, a.device)
    if a.cmd == "check0":
        return check0(a.device)
    return train(a.arch, a.hours, a.device, a.lr, a.batch, a.tag or f"bank_{a.arch}", a.val_n)


if __name__ == "__main__":
    sys.exit(main())
