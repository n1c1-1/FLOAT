"""The narrow-band retouch dataset of the star removal network: pairs (the image the Stars block receives, the network's
starless image) of real mono narrow-band fields, to retouch by hand (the network leaves a faint disc around the bright
stars of mono layers) and to train on again (tools/bench/stars_train.py reads them like the old fields).

    python tools/bench/stars_nb_dataset.py run [--fields 62,56,57,58,64] [--device cpu|cuda]

Two kinds of pairs, from the five banque projects named in FIELDS (read only: nothing is ever written there):

A, per filter: the filter's image where the "Filters full" template puts its Stars block (the Denoise block's
   Output: render_simple's linear image; the template's blocks b10, b21, b32), and the network's result on it.
B, after the combination: the fused image (the combination block run on the filters WITH their stars, the inputs of
   the template's Combine block), and the network's result on it. The fusion's coefficients go to
   <key>_comb_coeffs.json (per filter the affine map of the inputs that the engine measured, its weight, the
   output matrix, the sky noises), so that a later step fuses the per-filter retouches exactly as this fusion did.
   The check: the coefficients alone, reapplied to the filters with their stars, give the engine's fused image; and
   reapplied to the network's per-filter starless images they give what the engine's own mixing code
   (Combination.mix_balanced) gives of the same frozen maps.

How: each project's definition is copied (project.astroproj and state.json only) to
F:/astro_essais/stars_dataset_nb_work/<key>/; its store and inputs are READ where they are (an overlay on
core.store.Store: a result is read from the scratch store, else from the project's; everything written goes to the
scratch store; a write inside the project is refused). The stages from Gradient on run through the engine's own run
(api.plan_to_block, api.start_run), the Stars block added to each filter's workflow (api.set_option). A plan that
would run a stage before Gradient is refused: calibration, alignment and stacking are never recomputed. The overlay
reaches the run's worker processes through a sitecustomize in the work folder (PYTHONPATH).

Output (F:/astro_essais/stars_dataset/_revue_nb/), the format of the old _revue and of tools/bench/stars_dataset.py
``review``: <key>_<filter>_{original,starless}.tif (16 bit, the stretch y = asinh((v + OFFSET) / SOFT) /
asinh((top + OFFSET) / SOFT) of v in noise sigmas, background 0), <key>_comb_{original,starless}.tif, stretch.json,
and for each pair a folder <pair>/{input.npy, starless.npy, meta.json} (linear, noise sigmas, float32, 3 x H x W: what
fixes() and training_pairs() of stars_train.py read, with DS and REVIEW set to this folder). The unit and the
background that bring a pair back to the Output's own units are in its meta.json.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BANQUE = Path("F:/astro_banque")
WORK = Path("F:/astro_essais/stars_dataset_nb_work")
OUT = Path("F:/astro_essais/stars_dataset/_revue_nb")
OFFSET, SOFT = 3.0, 3.0                      # the review stretch of tools/bench/stars_dataset.py
#: the projects, nothing guessed: {key: (folder, the filters, the combination's type or None)}
FIELDS = {
    "56": ("56_mono_newton_asi1600mm_m27_sho_rgb", ("SII", "Ha", "OIII"), "sho"),
    "57": ("57_mono_newton_asi1600mm_m8_lrgb_sho", ("SII", "Ha", "OIII"), "sho"),
    "58": ("58_mono_newton_asi1600mm_ic443_hoo_rgb", ("Ha", "OIII"), "ho"),
    "62": ("62_mono_lco04m_sq38_ngc3372_hau", ("Ha", "OIII"), "ho"),
    "64": ("64_smart_seestar_s30_m16_dualband", (), None),
}
#: the stages before Gradient: never computed here
BEFORE_GRADIENT = ("inventory", "calibrate", "register", "stack", "astrometry", "surveys", "align_on", "reference",
                   "image")
COMBINES = ("combine_sho", "combine_ho", "combine_rgb", "combine_lrgb", "combine_recipe")


def log(msg: str) -> None:
    print(f"{datetime.now():%H:%M:%S} {msg}", flush=True)


# ---------------------------------------------------------------------------------------------- the overlay
def install_overlay(src: str, dst: str) -> None:
    """In this process: ``core.store.Store`` reads a result from the scratch project ``dst`` first, then from the
    project ``src``; what it writes (``own_dir``, ``tmp_dir``, ``trash``) is the scratch's; the scratch's inputs
    folder is the source's; and no write inside ``src`` is possible."""
    import builtins
    import io

    from astro_gpu.core import project2, runner2
    from astro_gpu.core import store as smod
    srcp, dstp = Path(os.path.abspath(src)), Path(os.path.abspath(dst))
    src_store = srcp / smod.STORE
    if getattr(smod.Store, "_nb_overlay", False):
        return
    smod.Store._nb_overlay = True
    orig_dir = smod.Store.dir

    def dir_(self, stage, key):
        own = self.own_dir(stage, key)                       # a result of this key made here
        if (own / "manifest.json").is_file():
            return own
        d = orig_dir(self, stage, key)                       # (or an earlier result's folder in the scratch)
        if (d / "manifest.json").is_file():
            return d
        s = src_store / smod.base(stage) / self.found_key(stage, key)
        return s if (s / "manifest.json").is_file() else d
    smod.Store.dir = dir_
    orig_find = smod.find_earlier

    def find(keys, store, seen=None):                        # the earlier results of the project's own store
        t = smod.Store.__new__(smod.Store)
        t.path, t._usable = src_store, {}
        got = orig_find(keys, t, seen)
        # a result of the same key made here stands for itself, not the project's earlier one
        return {(b, k): v for (b, k), v in got.items() if not (dstp / smod.STORE / b / k / "manifest.json").is_file()}
    smod.find_earlier = find
    runner2.find_earlier = find
    from astro_gpu import api
    api.find_earlier = find
    prop = project2.Project2.inputs_dir
    project2.Project2.inputs_dir = property(
        lambda self, _f=prop.fget: srcp / "inputs" if os.path.normcase(os.path.abspath(self.root)) ==
        os.path.normcase(str(dstp)) else _f(self))

    def refuse(what, *paths):
        g = os.path.normcase(str(srcp))
        for p in paths:
            if isinstance(p, (str, bytes, os.PathLike)):
                s = os.path.normcase(os.path.abspath(os.fspath(p)))
                if s == g or s.startswith(g + os.sep):
                    raise PermissionError(f"stars_nb_dataset: {what} inside the project read in place, refused: {p}")
    o_open = builtins.open

    def g_open(file, mode="r", *a, **k):
        if isinstance(file, (str, bytes, os.PathLike)) and any(c in mode for c in "wax+"):
            refuse("open for writing", file)
        return o_open(file, mode, *a, **k)
    builtins.open = g_open
    io.open = g_open

    def wrap(mod, name, paths_of):
        fn = getattr(mod, name)

        def inner(*a, **k):
            refuse(f"{mod.__name__}.{name}", *paths_of(a, k))
            return fn(*a, **k)
        setattr(mod, name, inner)
    for n in ("remove", "unlink", "rmdir", "mkdir", "utime", "truncate", "makedirs", "removedirs"):
        if hasattr(os, n):
            wrap(os, n, lambda a, k: a[:1])
    for n in ("replace", "rename"):
        wrap(os, n, lambda a, k: a[:2])
    for n in ("copy", "copy2", "copyfile", "move", "copytree"):
        wrap(shutil, n, lambda a, k: a[1:2])
    wrap(shutil, "rmtree", lambda a, k: a[:1])


def write_sitecustomize(folder: Path) -> None:
    """The run's worker processes import this at start: the same overlay as the driver's."""
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "sitecustomize.py").write_text(
        "import os, sys\n"
        "if os.environ.get('FLOAT_NB_SRC'):\n"
        f"    sys.path.insert(0, {(ROOT / 'tools' / 'bench').as_posix()!r})\n"
        f"    sys.path.insert(0, {ROOT.as_posix()!r})\n"
        "    import stars_nb_dataset as _m\n"
        "    _m.install_overlay(os.environ['FLOAT_NB_SRC'], os.environ['FLOAT_NB_DST'])\n", encoding="utf-8")


# ---------------------------------------------------------------------------------------------- the engine's run
def follow(api, p, info, poll: float = 5.0, grace: float = 300.0) -> str:
    offset, last = info.offset, time.time()
    while True:
        evs, offset = api.read_events(p, offset, info.run_id)
        for e in evs:
            last = time.time()
            t = e.get("type")
            if t == "stage_start":
                log(f"  start {e.get('stage')} (workflows {e.get('wf')})")
            elif t == "stage_end":
                log(f"  end   {e.get('stage')} {e.get('status')} in {float(e.get('duration') or 0):.1f} s"
                    + (f" -- {e.get('error')}" if e.get("error") else ""))
            elif t == "log" and e.get("level") in ("warning", "error"):
                log(f"  {e.get('level')}: [{e.get('stage')}] {str(e.get('msg'))[:300]}")
            elif t == "run_end":
                return str(e.get("status"))
        if api.running(p) is None and time.time() - last > grace:
            return "lost"
        time.sleep(poll)


def device_check(device: str) -> str:
    import torch
    if device == "cpu":
        if torch.cuda.is_available():
            raise SystemExit("cpu asked and torch.cuda.is_available() is True: set CUDA_VISIBLE_DEVICES=-1")
        return "cpu"
    names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    log(f"CUDA devices: {names}")
    if len(names) != 1 or "4060" not in names[0]:
        raise SystemExit("the GPU run needs CUDA_DEVICE_ORDER=PCI_BUS_ID and CUDA_VISIBLE_DEVICES=0 (the 4060 Ti alone)")
    return "cuda:0"


def setup(key: str):
    """The scratch project of a field, opened through the API with the overlay on."""
    name = FIELDS[key][0]
    src, dst = BANQUE / name, WORK / key
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy(src / "project.astroproj", dst / "project.astroproj")
    if (src / "state.json").is_file():
        shutil.copy(src / "state.json", dst / "state.json")
    os.environ["FLOAT_NB_SRC"], os.environ["FLOAT_NB_DST"] = str(src), str(dst)
    site = WORK / "_site"
    write_sitecustomize(site)
    rest = [x for x in os.environ.get("PYTHONPATH", "").split(os.pathsep) if x and x != str(site)]
    os.environ["PYTHONPATH"] = os.pathsep.join([str(site), *rest])
    install_overlay(str(src), str(dst))
    from astro_gpu import api
    return api, api.open_project(dst), src, dst


def plan_to(api, p, wfs: list, block: str, device: str):
    """The plan of "run to this block" for a workflow, with the done results from Gradient on that were computed by an
    earlier version of their method recomputed in their place (what the GUI's Recompute does, limited to those
    stages: the ones before Gradient are never redone), and what is made from them."""
    from astro_gpu.core import runner2 as r2
    from astro_gpu.core.store import earlier_version
    kw = api._plan_kw(p, device, None)
    keys, store = kw["keys"], kw["store"]
    redo = set()
    for wf in wfs:
        w = p.workflow(wf)
        for s in keys.chain(w):
            if s.split("#")[0].split("@")[0].split("~")[0] in BEFORE_GRADIENT:
                continue
            k = keys.key(w, s)
            m = store.manifest(s, k)
            if store.is_done(s, k, m) and earlier_version(keys, w, s, m) is not None:
                redo.add(k)
    planner = r2.Planner(keys.project, redo=redo, **kw)
    targets = []
    for o in p.workflows:
        try:
            chain = keys.chain(o)
        except ValueError:
            continue
        for s in chain:
            try:
                k = keys.key(o, s)
            except ValueError:
                continue
            if (k in redo and planner.done(s, k)) or planner.upstream.redone(o, s):
                targets.append((o.id, s))
    for wf in wfs:                                                 # what is not computed yet
        targets += [t for t in api.plan_to_block(p, wf, block, device).targets if t not in targets]
    return api._stamp(p, planner.plan(targets)), redo


def run_to(api, p, wfs: list, block: str, device: str) -> float:
    """Run the blocks up to ``block`` of these workflows; refused when a stage before Gradient would run."""
    plan, redo = plan_to(api, p, wfs, block, device)
    early = [j.stage for j in plan.jobs if j.stage.split("#")[0].split("@")[0].split("~")[0] in BEFORE_GRADIENT]
    if early:
        raise SystemExit(f"refused: the plan would compute {', '.join(early)}")
    log(f"workflows {wfs}, plan to {block}: {len(plan.jobs)} jobs: {', '.join(j.stage for j in plan.jobs) or 'none'}"
        f" ({len(redo)} results of an earlier version)")
    if not plan.jobs:
        return 0.0
    t = time.time()
    info = api.start_run(p, plan, device)
    log(f"run {info.run_id} on {info.device}, pid {info.pid}")
    status = follow(api, p, info)
    api.refresh(p)
    if status != "done":
        raise SystemExit(f"the run ended {status}")
    return time.time() - t


# ---------------------------------------------------------------------------------------------- the pairs
def pair_arrays(lin: np.ndarray, starless: np.ndarray, prep: dict):
    """The network's own units (stages/star_removal.py: the image minus its background, over the noise unit): the
    pair as the dataset keeps them, noise sigmas around a background of 0."""
    bk = np.asarray(prep["background"], np.float32)[:, None, None]
    unit = float(prep["unit"])
    return (((lin - bk) / unit).astype(np.float32), ((starless - bk) / unit).astype(np.float32),
            {"unit": unit, "background": [float(x) for x in bk.ravel()]})


def save_pair(name: str, a: np.ndarray, b: np.ndarray, conv: dict, extra: dict) -> None:
    import tifffile
    OUT.mkdir(parents=True, exist_ok=True)
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "input.npy", a)
    np.save(d / "starless.npy", b)
    top = float(a.max())
    meta = {"name": name, "field": name, "shape": list(a.shape), "top": top, "offset": OFFSET, "soft": SOFT,
            "units": "noise sigmas of the Stars block's input (the network's own units), background 0",
            "to_output_units": "v * unit + background (render_simple's linear image)", **conv, **extra}
    (d / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    st = np.arcsinh((top + OFFSET) / SOFT)
    for kind, v in (("original", a), ("starless", b)):
        y = np.arcsinh(np.clip(v + OFFSET, 0, None) / SOFT) / st
        tifffile.imwrite(OUT / f"{name}_{kind}.tif", np.round(np.clip(y, 0, 1) * 65535).astype(np.uint16)
                         .transpose(1, 2, 0), photometric="rgb")
    sf = OUT / "stretch.json"
    stretch = json.loads(sf.read_text()) if sf.exists() else {}
    stretch[name] = {"pair": name, "top": top, "offset": OFFSET, "soft": SOFT, "white_balance": None,
                     "white_balanced": False}
    sf.write_text(json.dumps(stretch, indent=1), encoding="utf-8")
    log(f"{name}: {a.shape[2]}x{a.shape[1]} written")


def read_star_removal(api, p, wf):
    """(the Stars block's input in the Output's units, its starless image, the block's prep, the Output's folder, the
    Stars block's folder) of a workflow."""
    so = api.stage_outputs(p, wf, "star_removal")
    ro = api.stage_outputs(p, wf, "render_simple")
    if so is None or ro is None:
        raise SystemExit(f"workflow {wf}: the Output or the Stars block is not computed")
    lin = np.load(Path(ro.dir) / "lin.npy").astype(np.float32)
    sl = np.load(Path(so.dir) / "starless.npy").astype(np.float32)
    prep = json.loads((Path(so.dir) / "prep_stars.json").read_text(encoding="utf-8"))
    return lin, sl, prep, Path(ro.dir), Path(so.dir)


# ---------------------------------------------------------------------------------------------- the fusion's coefficients
def block_of(stage: str):
    from astro_gpu.stages import combine as cb
    return next(c for c in (cb.CombineSHO, cb.CombineHO, cb.CombineRGB, cb.CombineLRGB, cb.CombineRecipe)
                if getattr(c, "name", None) == stage)()


def stage_params(p, wf, stage: str):
    from astro_gpu.core.store import Keys
    keys = Keys(p)
    return keys.rules.params(stage)(**keys._key_values(wf, stage)[1])


def coefficients(p, wf, stage: str, comb_dir: Path) -> dict:
    """The fusion's coefficients as the engine measured them (drz_meta.json's balance), with what the output needs:
    the weights, the recipe's output matrix, the channel each filter is read through."""
    from astro_gpu.stages import combine as cb
    bal = json.loads((comb_dir / "drz_meta.json").read_text(encoding="utf-8"))["balance"]
    params = stage_params(p, wf, stage)
    spec = block_of(stage).spec(params)
    if spec["output"]["lightness"] is not None:
        raise SystemExit(f"the recipe {spec['id']!r} composes lightness and colour: not linear, no frozen coefficients")
    roles, mat = cb.output_matrix(params, spec)
    ports = list(cb.recipes.roles(spec))
    given = {i["role"]: i.get("channel") for i in wf.source["fusion"]["inputs"]}
    return {"stage": stage, "recipe": spec["id"], "inputs": ports,
            "channel_read": {r: given.get(r) or cb.recipes.default_channel(spec, r) for r in ports},
            "balance_mode": bal["how"],
            "maps": {r: {"coef": m["coef"], "offset": m["offset"]} for r, m in bal["maps"].items()},
            "gains": bal["gains"], "sky_noise": bal["noise"],
            "weights": {r: float(params.weights.get(r, 1.0)) for r in ports},
            "output_channels": list(roles), "output_matrix": [[float(x) for x in row] for row in mat],
            "formula": "balanced[r] = weight[r] * (sum_k coef[r][k] * channel(input k) + offset[r]); "
                       "rgb = output_matrix @ balanced[output_channels]; 0 outside valid"}


def apply_coefficients(co: dict, imgs: dict, valid: np.ndarray) -> np.ndarray:
    """(3, H, W): the fusion of ``imgs`` {role: (3, H, W)} from the coefficients alone (the json's content)."""
    from astro_gpu.stages.combine import channel_of
    ch = {r: channel_of(np.asarray(imgs[r], np.float32), co["channel_read"][r]).astype(np.float64) for r in co["inputs"]}
    bal = {}
    for r in co["output_channels"]:
        m = co["maps"][r]
        bal[r] = co["weights"][r] * (sum(float(c) * ch[k] for k, c in m["coef"].items()) + float(m["offset"]))
    rgb = np.tensordot(np.asarray(co["output_matrix"], np.float64), np.stack([bal[r] for r in co["output_channels"]]), 1)
    rgb[:, ~valid] = 0
    return rgb.astype(np.float32)


def engine_fusion(co: dict, imgs: dict, valid: np.ndarray, params) -> np.ndarray:
    """The same fusion by the engine's own mixing code (``Combination.mix_balanced``): the channels read as the engine
    reads them, the recorded maps applied in float32 as its steps do, then its weights and its output."""
    from astro_gpu.stages import combine as cb

    class Ctx:
        def log(self, *a, **k):
            pass
    block = block_of(co["stage"])
    chans = {r: cb.channel_of(np.asarray(imgs[r], np.float32), co["channel_read"][r]) for r in co["inputs"]}
    new = {}
    for r in co["inputs"]:
        m = co["maps"][r]
        v = np.zeros_like(chans[r])
        for k, c in m["coef"].items():
            v += np.float32(c) * chans[k]
        new[r] = (v + np.float32(m["offset"])).astype(np.float32)
    skies = {r: (0.0, float(co["sky_noise"][r])) for r in co["inputs"]}
    return block.mix_balanced(Ctx(), params, block.spec(params), [{"role": r} for r in co["inputs"]], new, skies,
                              {r: 1.0 for r in co["inputs"]}, valid)


def check(p, wf, co: dict, comb_dir: Path, filt: dict, valid: np.ndarray, starless: dict) -> dict:
    """The two checks of the module docstring; returns the figures."""
    engine = np.load(comb_dir / "drz_full.npy").astype(np.float32)
    d1 = float(np.abs(apply_coefficients(co, filt, valid) - engine).max())
    a = apply_coefficients(co, starless, valid)
    b = engine_fusion(co, starless, valid, stage_params(p, wf, co["stage"]))
    d2 = float(np.abs(a - b).max())
    return {"with_stars_vs_engine_max_abs": d1, "with_stars_rel": d1 / float(np.abs(engine).max()),
            "starless_vs_engine_code_max_abs": d2, "starless_rel": d2 / max(float(np.abs(b).max()), 1e-30)}


# ---------------------------------------------------------------------------------------------- the field
def do_field(key: str, device: str) -> dict:
    name, filters, kind = FIELDS[key]
    t0 = time.time()
    api, p, src, dst = setup(key)
    field = key
    log(f"{name}: workflows {[(w.id, w.name, (w.source or {}).get('fusion', {}).get('type')) for w in p.workflows]}")
    comb = next((w for w in p.workflows if (w.source or {}).get("fusion") and w.source["fusion"]["type"] == kind),
                None) if kind else None
    if kind and comb is None:
        raise SystemExit(f"no {kind} combination in the project")
    fw = {i["role"]: p.workflow(i["pid"]) for i in comb.source["fusion"]["inputs"]} if comb else {}
    if comb is None:                                        # a colour project: its one workflow
        fw = {"image": p.workflow(p.current)}
    for w in fw.values():                                   # the Stars block where the template has it
        api.set_option(p, w.id, "stars", "neural")
    if comb is not None:
        api.set_option(p, comb.id, "stars", "neural")
    api.save(p)
    secs = run_to(api, p, [w.id for w in [*fw.values(), *([comb] if comb else [])]], "stars", device)
    out = {"field": field, "run_s": round(secs, 1)}
    filt, starless, valid = {}, {}, None
    for r, w in fw.items():
        lin, sl, prep, rdir, sdir = read_star_removal(api, p, w.id)
        a, b, conv = pair_arrays(lin, sl, prep)
        save_pair(f"{field}_{r}" if comb else f"{field}_{w.name.replace(' ', '_')}", a, b, conv, {"kind": "A", "filter": r, "workflow": w.name,
                                                                   "source": name})
        filt[r], starless[r] = lin, sl
    if comb is not None:
        lin, sl, prep, rdir, sdir = read_star_removal(api, p, comb.id)
        a, b, conv = pair_arrays(lin, sl, prep)
        save_pair(f"{field}_comb", a, b, conv, {"kind": "B", "workflow": comb.name, "source": name,
                                                "filters": list(fw)})
        stage = next(s for s in COMBINES if api.stage_outputs(p, comb.id, s) is not None)
        comb_dir = Path(api.stage_outputs(p, comb.id, stage).dir)
        co = coefficients(p, comb, stage, comb_dir)
        valid = np.load(comb_dir / "drz_weight.npy")[0] > 0
        co["valid_file"] = f"{field}_comb_valid.npz"
        np.savez_compressed(OUT / co["valid_file"], valid=valid)
        co["check"] = check(p, comb, co, comb_dir, filt, valid, starless)
        (OUT / f"{field}_comb_coeffs.json").write_text(json.dumps(co, indent=1), encoding="utf-8")
        log(f"coefficient check: {co['check']}")
        out["check"] = co["check"]
    out["total_s"] = round(time.time() - t0, 1)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=("run",))
    ap.add_argument("--fields", default="62")
    ap.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    a = ap.parse_args(argv)
    sys.path.insert(0, str(ROOT))
    os.environ.setdefault("ASTRO_GPU_HOME", str(WORK / "_home"))
    dev = device_check(a.device)
    for key in a.fields.split(","):
        log(f"{key}: {do_field(key, dev)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
