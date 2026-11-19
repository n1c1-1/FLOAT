"""Planetary-nebula / cluster review pairs for the star removal network, from public archive frames of a robotic
telescope network: the frames are downloaded, then stacked, combined and passed through the Stars block by the
engine, as the GUI does, and the pair (the image the Stars block receives, its starless image) is written to be
reviewed and retouched by hand (the compact bright cores that the network takes for stars).

    python tools/bench/stars_pn_sources.py download [--targets m57,...]
    python tools/bench/stars_pn_sources.py build    --targets m57          (the projects, graph printed)
    python tools/bench/stars_pn_sources.py run      --targets m57 --device cuda|cpu
    python tools/bench/stars_pn_sources.py pairs    --targets m57          (the Output files; run does it too)

Nothing is written outside F:/astro_essais/stars_dataset/_sources_pn/ and _revue_pn/. The archive's public API needs
no login; the frames are the reduced level (calibrated: bias, dark, flat, with their WCS), so the projects' Calibration
blocks are cut (the engine's path for frames that are calibrated already).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = Path("F:/astro_essais/stars_dataset/_sources_pn")
LCO = SRC / "lco"
PROJECTS = SRC / "projets"
OUT = Path("F:/astro_essais/stars_dataset/_revue_pn")
API = "https://archive-api.lco.global/frames/"
MAX_PER_FILTER = 40

#: the sets, nothing guessed: key -> (title, how to ask the archive, telescope, instrument, the night, the filters
#: (the optical element of the archive = the FITS FILTER), the combination)
SETS = {
    "m57": ("M57", {"covers": "POINT(283.396 33.029)"}, "ogg", "0m4b", "sq30", "2026-07-24",
            ("H-Alpha", "OIII"), "hoo"),
    "ngc6543": ("NGC 6543", {"covers": "POINT(269.639 66.633)"}, "tfn", "0m4b", "sq33", "2024-07-27",
                ("B", "V", "rp"), "bvr"),
    "m97": ("M97", {"covers": "POINT(168.699 55.019)"}, "tfn", "0m4a", "kb25", "2019-04-30",
            ("B", "V", "rp"), "bvr"),
    "ngc7662": ("NGC 7662", {"OBJECT": "NGC 7662"}, "tfn", "0m4b", "kb24", "2022-11-23",
                ("B", "V", "rp"), "bvr"),
    "m13": ("M13", {"covers": "POINT(250.4217 36.4613)"}, "ogg", "0m4b", "sq30", "2024-07-26",
            ("B", "V", "rp"), "bvr"),
    "m1": ("M1", {"covers": "POINT(83.633 22.015)"}, "cpt", "0m4a", "kb84", "2019-12-16",
           ("B", "V", "rp"), "bvr"),
    "m82": ("M82", {"covers": "POINT(148.968 69.680)"}, "ogg", "0m4c", "sq40", "2024-06-11",
            ("B", "V", "rp"), "bvr"),
    "ngc6888": ("NGC 6888", {"covers": "POINT(303.05 38.35)"}, "elp", "0m4a", "sq31", "2025-05-21",
                ("SII", "H-Alpha", "OIII"), "sho"),
}


def log(msg: str) -> None:
    print(f"{datetime.now():%H:%M:%S} {msg}", flush=True)


# ---------------------------------------------------------------------------------------------- the archive
def query(**kw) -> list[dict]:
    base = dict(public="true", reduction_level=91, configuration_type="EXPOSE", limit=100)
    base.update(kw)
    url, out = API + "?" + urllib.parse.urlencode(base), []
    while url:
        for k in range(5):
            try:
                with urllib.request.urlopen(url, timeout=90) as r:
                    d = json.load(r)
                break
            except Exception as e:  # noqa: BLE001 - the archive is shared: wait and retry
                log(f"archive: {e}; retry {k + 1}")
                time.sleep(10 * (k + 1))
        else:
            raise SystemExit("the archive does not answer")
        out += d["results"]
        url = d.get("next")
    return out


def pick(key: str) -> dict[str, list[dict]]:
    """{filter: frames} of a set: the night's frames of the telescope and instrument, each filter at its commonest
    exposure, the first MAX_PER_FILTER in time order."""
    title, ask, site, tel, ins, night, filters, _ = SETS[key]
    d0 = datetime.fromisoformat(night)
    rows = query(**ask, start=(d0 - timedelta(days=1)).strftime("%Y-%m-%d"),
                 end=(d0 + timedelta(days=2)).strftime("%Y-%m-%d"))
    rows = [r for r in rows if r["site_id"] == site and r["telescope_id"] == tel and r["instrument_id"] == ins
            and r["DAY_OBS"] == night]
    out = {}
    for f in filters:
        fr = [r for r in rows if r["primary_optical_element"] == f]
        if not fr:
            raise SystemExit(f"{key}: no frame of the filter {f}")
        exps: dict[int, int] = {}
        for r in fr:
            exps[round(r["exposure_time"])] = exps.get(round(r["exposure_time"]), 0) + 1
        top = max(exps, key=exps.get)
        fr = sorted((r for r in fr if round(r["exposure_time"]) == top), key=lambda r: r["DATE_OBS"])
        out[f] = fr[:MAX_PER_FILTER]
    return out


def download(key: str) -> dict:
    sel = pick(key)
    got = {}
    for f, rows in sel.items():
        d = LCO / key / f
        d.mkdir(parents=True, exist_ok=True)

        def one(r, d=d):
            dst = d / r["filename"]
            if dst.is_file() and dst.stat().st_size > 0:
                return dst.stat().st_size
            tmp = dst.with_suffix(dst.suffix + ".part")
            for k in range(4):
                try:
                    with urllib.request.urlopen(r["url"], timeout=180) as resp, open(tmp, "wb") as fh:
                        while True:
                            b = resp.read(1 << 20)
                            if not b:
                                break
                            fh.write(b)
                    os.replace(tmp, dst)
                    return dst.stat().st_size
                except Exception as e:  # noqa: BLE001
                    log(f"{r['filename']}: {e}; retry {k + 1}")
                    time.sleep(5 * (k + 1))
            raise SystemExit(f"{r['filename']}: not downloaded")
        with ThreadPoolExecutor(3) as ex:
            sizes = list(ex.map(one, rows))
        got[f] = (len(rows), sum(sizes) / 1e6)
        log(f"{key} {f}: {len(rows)} frames, {sum(sizes) / 1e6:.0f} MB, exposure {rows[0]['exposure_time']} s")
    return got


# ---------------------------------------------------------------------------------------------- the project
#: the template's palette of each combination, the views' options of the Mono RGB template (denoise on, stars kept)
PALETTE = {"hoo": "H-O", "bvr": "RGB", "sho": "S-H-O"}
OPTS = dict(debruitage=True, etoiles="garder", fond="simple")


def header_filters(folder: Path) -> set[str]:
    """The FITS FILTER keyword of every frame of a folder, as written."""
    from astropy.io import fits
    out = set()
    for f in sorted(folder.glob("*.fz")):
        with fits.open(f) as h:
            out.add(str(next((x.header["FILTER"] for x in h if "FILTER" in x.header), "")).strip())
    return out


def print_graph(p, key: str) -> None:
    """The graph built: its blocks, in each row's order, and their inputs (the nights' folders, the wires)."""
    from astro_gpu.core import project2
    d = json.loads(json.dumps(project2.file_dict(p), default=str))
    g = d["graph"]
    log(f"graph of {key}: format {d['format']}, {len(g['blocks'])} blocks")
    for r in g["rows"]:
        log(f"  row {r['id']} \"{r['name']}\" (workflow {r.get('wid')}):")
        for night, calib in r["nights"]:
            sid = g["blocks"][night]["sid"]
            s = p.session(sid)
            cal = g["blocks"][calib] if calib else None
            log(f"    input {night} Session \"{s.name}\" filter {s.filter}: {len(s.lights)} lights ({Path(s.lights[0]).parent.name if s.lights else ''})"
                + (f"; Calibration {calib} {'CUT (calibrated frames)' if cal['off'] else 'on'}" if cal else ""))
        for b in r["blocks"]:
            blk = g["blocks"][b]
            extra = f" method {blk['method']}" if blk.get("method") else ""
            log(f"    block {b} {blk['kind']}{extra}{' OFF' if blk.get('off') else ''}")
        if r.get("align"):
            log(f"    align on {r['align'].get('ref')} from {r['align'].get('from')}")
    for w in g["wires"]:
        log(f"  wire {w['src']} -> {w['dst']} channel {w.get('channel')}")


def build(key: str) -> Path:
    """The project of a set, made as the GUI's template gallery makes it: the Mono template of the combination's type
    (one row per filter, the combination after them), the night folders of each filter added to its session, the
    Calibration blocks cut (the frames are calibrated), a branch with a Stars block after each filter's Denoise, and
    a Stars block after the combination (which reads the filters with their stars)."""
    title, _ask, _site, _tel, _ins, _night, filters, kind = SETS[key]
    root = PROJECTS / key
    if (root / "project.astroproj").exists():
        raise SystemExit(f"{root}: a project is already there; nothing overwritten")
    sys.path.insert(0, str(ROOT))
    from PySide6.QtWidgets import QApplication

    from astro_gpu import api
    from astro_gpu.client import i18n
    from astro_gpu.client.model import ProjectModel
    from astro_gpu.client.services.worker import ApiWorker
    from astro_gpu.core import project2
    from astro_gpu.core.project import Target
    from astro_gpu.io.frames import FILTER_NAMES
    QApplication.instance() or QApplication([])
    i18n.set_language("en")
    roles = [FILTER_NAMES[f.lower()] for f in filters]
    for f, r in zip(filters, roles):                                   # the filter is the header's, nothing guessed
        seen = header_filters(LCO / key / f)
        if seen != {f}:
            raise SystemExit(f"{key}/{f}: the frames' FILTER keyword says {seen}")
        log(f"{key}/{f}: FILTER {seen} -> the engine's filter {r}")
    root.mkdir(parents=True, exist_ok=True)
    p = project2.create(root, title, Target(name=title))
    w = ApiWorker()
    m = ProjectModel(p, w)
    w.wait(120)
    try:
        t = dict(kind="fusion", mono_rgb=True, filters=roles, palette=PALETTE[kind])
        res = m.apply_template(t, title, OPTS)
        if not res.ok:
            raise SystemExit(f"the template is refused: {res.reason}")
        sid = {s.filter: s.id for s in p.sessions}
        for f, r in zip(filters, roles):
            notes: list = []
            n = m.add_session_entries(sid[r], "lights", [(LCO / key / f).as_posix()], notes)
            log(f"{key}: {n} lights added to the session of {r} {notes or ''}")
        rows = [v for v in m.paths if v.name in roles]
        m.set_cut(rows, "calibration", True)
        for r in roles:                          # the filter's own Stars block: a branch after its Denoise
            v = next(x for x in m.paths if x.name == r)
            made, key_ = m.add_branch(v, "debruitage", "stars")
            log(f"{key}: branch \"{made.name}\" (Stars block {key_}) after the Denoise of {r}")
        comb = next(x for x in m.paths if x.name == title)   # the combination reads the filters WITH their stars
        m.insert_block(comb, "stars", method="neural")
        m.flush(force=True)
        api.save(p)
        print_graph(p, key)
    finally:
        m.close(save=False)
        w.stop()
    return root


# ---------------------------------------------------------------------------------------------- the run
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


def follow(api, p, info, poll: float = 5.0, grace: float = 300.0) -> tuple[str, list]:
    offset, last, errs = info.offset, time.time(), []
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
                if e.get("error"):
                    errs.append(str(e.get("error")))
            elif t == "log" and e.get("level") in ("warning", "error"):
                log(f"  {e.get('level')}: [{e.get('stage')}] {str(e.get('msg'))[:300]}")
            elif t == "run_end":
                return str(e.get("status")), errs
        if api.running(p) is None and time.time() - last > grace:
            return "lost", errs
        time.sleep(poll)


def has_stars(w) -> bool:
    return any(b.get("block") == "stars" for b in (w.blocks or []))


def run(key: str, device: str) -> dict:
    sys.path.insert(0, str(ROOT))
    from astro_gpu import api
    dev = device_check(device)
    p = api.open_project(PROJECTS / key)
    t0, out = time.time(), {"key": key, "device": dev, "stages": {}}
    log(f"{key}: workflows {[(w.id, w.name) for w in p.workflows]}")
    for w in p.workflows:
        if not has_stars(w):
            continue
        plan = api.plan_to_block(p, w.id, "stars", dev)
        log(f"workflow {w.id} {w.name}: plan to Stars, {len(plan.jobs)} jobs: "
            + ", ".join(j.stage for j in plan.jobs))
        if not plan.jobs:
            continue
        t = time.time()
        info = api.start_run(p, plan, dev)
        log(f"run {info.run_id} on {info.device}, pid {info.pid}")
        status, errs = follow(api, p, info)
        api.refresh(p)
        out["stages"][w.name] = {"status": status, "seconds": round(time.time() - t, 1), "errors": errs}
        if status != "done":
            log(f"{key}: workflow {w.name} ended {status}: {errs}")
            return {**out, "status": status}
    out.update(status="done", seconds=round(time.time() - t0, 1))
    return out


# ---------------------------------------------------------------------------------------------- the pairs
OFFSET, SOFT = 3.0, 3.0                      # the review stretch of the narrow-band dataset (stretch.json)


def read_star_removal(api, p, wf):
    """(the Stars block's input in the Output's units, its starless image, the block's prep) of a workflow."""
    import numpy as np
    so = api.stage_outputs(p, wf, "star_removal")
    ro = api.stage_outputs(p, wf, "render_simple")
    if so is None or ro is None:
        raise SystemExit(f"workflow {wf}: the Output or the Stars block is not computed")
    lin = np.load(Path(ro.dir) / "lin.npy").astype("float32")
    sl = np.load(Path(so.dir) / "starless.npy").astype("float32")
    prep = json.loads((Path(so.dir) / "prep_stars.json").read_text(encoding="utf-8"))
    return lin, sl, prep


def save_pair(name: str, label: str, a, b, conv: dict, extra: dict) -> None:
    """The pair as the narrow-band review keeps it: the stretched 16 bit TIFFs, the linear arrays in noise sigmas
    in <name>/, the stretch in stretch.json. A name already taken stops everything (nothing is overwritten)."""
    import numpy as np
    import tifffile
    files = [OUT / f"{label}_original.tif", OUT / f"{label}_starless_reseau.tif", OUT / name]
    for f in files:
        if f.exists():
            raise SystemExit(f"{f} already exists: nothing overwritten")
    OUT.mkdir(parents=True, exist_ok=True)
    d = OUT / name
    d.mkdir(parents=True)
    np.save(d / "input.npy", a)
    np.save(d / "starless.npy", b)
    top = float(a.max())
    meta = {"name": name, "field": label, "shape": list(a.shape), "top": top, "offset": OFFSET, "soft": SOFT,
            "units": "noise sigmas of the Stars block's input (the network's own units), background 0",
            "to_output_units": "v * unit + background (render_simple's linear image)", **conv, **extra}
    (d / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    st = np.arcsinh((top + OFFSET) / SOFT)
    for f, v in ((files[0], a), (files[1], b)):
        y = np.arcsinh(np.clip(v + OFFSET, 0, None) / SOFT) / st
        tifffile.imwrite(f, np.round(np.clip(y, 0, 1) * 65535).astype(np.uint16).transpose(1, 2, 0), photometric="rgb")
    sf = OUT / "stretch.json"
    stretch = json.loads(sf.read_text()) if sf.exists() else {}
    stretch[name] = {"pair": name, "top": top, "offset": OFFSET, "soft": SOFT, "white_balance": None,
                     "white_balanced": False}
    sf.write_text(json.dumps(stretch, indent=1), encoding="utf-8")
    log(f"{name}: {a.shape[2]}x{a.shape[1]} written as {label}_original.tif / _starless_reseau.tif")


def pairs(key: str) -> None:
    import numpy as np
    sys.path.insert(0, str(ROOT))
    from astro_gpu import api
    from astro_gpu.io.frames import FILTER_NAMES
    title, _a, _s, _t, _i, _n, filters, _k = SETS[key]
    p = api.open_project(PROJECTS / key)
    for w in p.workflows:
        if not has_stars(w):
            continue
        lin, sl, prep = read_star_removal(api, p, w.id)
        bk = np.asarray(prep["background"], np.float32)[:, None, None]
        unit = float(prep["unit"])
        a, b = ((lin - bk) / unit).astype(np.float32), ((sl - bk) / unit).astype(np.float32)
        conv = {"unit": unit, "background": [float(x) for x in bk.ravel()]}
        flt = next((f for f in filters if w.name.startswith(FILTER_NAMES[f.lower()] + " ·")), None)
        name = key if flt is None else f"{key}_{flt}"
        save_pair(name, name, a, b, conv, {"kind": "B" if flt is None else "A", "workflow": w.name,
                                           "target": title, "filters": list(filters), "set": key})


# ---------------------------------------------------------------------------------------------- main
def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=("list", "download", "build", "graph", "run", "pairs"))
    ap.add_argument("--targets", default=",".join(SETS))
    ap.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    a = ap.parse_args(argv)
    keys = a.targets.split(",")
    if a.cmd == "list":
        for k in keys:
            for f, rows in pick(k).items():
                log(f"{k} {f}: {len(rows)} frames of {rows[0]['exposure_time']} s, {rows[0]['DATE_OBS']} .. "
                    f"{rows[-1]['DATE_OBS']}")
    elif a.cmd == "download":
        for k in keys:
            download(k)
    elif a.cmd == "build":
        for k in keys:
            build(k)
    elif a.cmd == "graph":
        sys.path.insert(0, str(ROOT))
        from astro_gpu import api
        for k in keys:
            print_graph(api.open_project(PROJECTS / k), k)
    elif a.cmd == "run":
        for k in keys:
            log(f"{k}: {run(k, a.device)}")
    else:
        for k in keys:
            pairs(k)
    return 0


if __name__ == "__main__":
    sys.exit(main())
