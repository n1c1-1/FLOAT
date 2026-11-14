"""Golden record of existing FLOAT projects: every cache key and state the current code gives them, read-only.

    python tools/graph_golden.py record <root>... --out <dir> [--recent] [--guard <folder>]... [--origins <json>]
    python tools/graph_golden.py stable <dir>
    python tools/graph_golden.py check <dir> [--out <json>] [--only <name part>]...

It never writes in a project folder (nor in the folders of the frames a project reads). A later change of the
graph model can be checked against this record: it must show zero difference.

``record``: for each ``project.astroproj`` found under the roots (depth <= 3) it loads the definition (never saves),
and writes ``<out>/<safe name>.json``: the keys of every stage of every workflow (real stage versions, and the same
with the versions patched to a constant, so that a later version bump does not stale it), the block keys, the states
(``api.states(update=False)``), the drawing the current client infers (``client.blocks.Graph``), the frame listings
the keys were made from (``<out>/frames/``) and a copy of the definition (``<out>/defs/``).

Project folders are read-only. Three layers hold that:
  1. a write guard (every open for writing, os.replace/rename/remove/mkdir/utime..., Popen) raises and records a
     violation for any path under the roots, the project folders and the folders of the frames they list;
  2. a file-system snapshot (names, sizes, mtimes of the project folder's top level and of each folder under it, plus
     state.json) before and after each project: any difference aborts the whole run;
  3. no worker, no warm-up (``api.warm`` and the model's jobs are not started), ``update=False`` everywhere, the
     settings file replaced by a temporary copy.
``stable`` records twice in two fresh processes and says what differs.
``check`` takes every recorded project through the graph model (astro_gpu/graphcore: convert, then compile into a
copy of the definition, in one process) and compares the definitions, the keys, the states, the review keys and the
graph's rows against the recorded drawing; read-only under the same guard; and the converted graph must be one the
core keeps as a whole (operations.legality: the engine's rules on every row, the joins' ports, the compiled
definition). A project is PROVEN when every session's frame list is non-empty on both sides, or every workflow from
frames reads a non-empty frame list on both sides (a session with no lights in its definition reads none); a project
of images only: when its image files are there; a project with no session and no workflow is EMPTY (no key to
compare); else its definition is only shown equal.
"""
from __future__ import annotations

import argparse
import builtins
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
MAX_DEPTH = 3
SKIP_DIRS = {"store", "logs", "_trash", "cache", "scratch", "export", "exports", "_scratch", "drafts", "__pycache__"}


# ---------------------------------------------------------------------------------------------- the write guard
class WriteGuard:
    """Raises (and records) on any write under the guarded folders."""

    def __init__(self) -> None:
        self.roots: list[str] = []
        self.violations: list[str] = []
        self.active = False
        self._orig: dict = {}

    @staticmethod
    def norm(p) -> str:
        return os.path.normcase(os.path.abspath(os.fspath(p)))

    def add(self, p) -> None:
        n = self.norm(p)
        if n not in self.roots:
            self.roots.append(n)

    def guarded(self, p) -> bool:
        if not self.active:
            return False
        try:
            s = self.norm(p)
        except (TypeError, ValueError):
            return False
        return any(s == g or s.startswith(g + os.sep) for g in self.roots)

    def refuse(self, what: str, *paths) -> None:
        hit = [str(p) for p in paths if isinstance(p, (str, bytes, os.PathLike)) and self.guarded(p)]
        if hit:
            msg = f"{what} {hit}"
            self.violations.append(msg)
            raise PermissionError(f"graph_golden write guard: {msg}")

    def install(self) -> None:
        g = self
        o = self._orig
        o["open"] = builtins.open

        def _open(file, mode="r", *a, **k):
            if isinstance(file, (str, bytes, os.PathLike)) and any(c in mode for c in "wax+"):
                g.refuse("open for writing", file)
            return o["open"](file, mode, *a, **k)
        builtins.open = _open
        io.open = _open

        o["os.open"] = os.open

        def _osopen(path, flags, *a, **k):
            if flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC):
                g.refuse("os.open for writing", path)
            return o["os.open"](path, flags, *a, **k)
        os.open = _osopen

        def wrap(mod, name, paths_of):
            fn = getattr(mod, name)

            def inner(*a, **k):
                g.refuse(f"{mod.__name__}.{name}", *paths_of(a, k))
                return fn(*a, **k)
            setattr(mod, name, inner)
        first = lambda a, k: a[:1]                    # noqa: E731
        two = lambda a, k: a[:2]                      # noqa: E731
        for n in ("remove", "unlink", "rmdir", "mkdir", "utime", "chmod", "truncate", "makedirs", "removedirs"):
            if hasattr(os, n):
                wrap(os, n, first)
        for n in ("replace", "rename", "link", "symlink"):
            wrap(os, n, two)
        for n in ("copy", "copy2", "copyfile", "move", "copytree"):
            wrap(shutil, n, lambda a, k: a[1:2])
        wrap(shutil, "rmtree", first)

        class NoPopen(subprocess.Popen):
            def __init__(self, *a, **k):
                g.violations.append(f"subprocess.Popen {str(a[:1])[:120]}")
                raise PermissionError("graph_golden: no process may start")
        subprocess.Popen = NoPopen
        self.active = True


GUARD = WriteGuard()


# ---------------------------------------------------------------------------------------------- snapshots
def snapshot(folder: Path) -> dict:
    """Names, sizes and mtimes of a project folder's top level and of the entries of each folder under it."""
    out: dict = {}

    def stat_row(e: os.DirEntry) -> list:
        try:
            st = e.stat(follow_symlinks=False)
            return [e.is_dir(follow_symlinks=False), st.st_size, st.st_mtime_ns]
        except OSError:
            return [None, None, None]
    try:
        with os.scandir(folder) as it:
            top = sorted(it, key=lambda e: e.name)
    except OSError as e:
        return {"error": str(e)}
    for e in top:
        row = stat_row(e)
        out[e.name] = row
        if row[0]:
            try:
                with os.scandir(e.path) as it2:
                    for e2 in sorted(it2, key=lambda x: x.name):
                        out[f"{e.name}/{e2.name}"] = stat_row(e2)
            except OSError:
                out[f"{e.name}/"] = "unreadable"
    return out


def snapshot_state(folder: Path) -> dict:
    try:
        st = (folder / "state.json").stat()
        return {"state.json": [st.st_size, st.st_mtime_ns]}
    except OSError:
        return {"state.json": None}


def diff(a: dict, b: dict) -> list[str]:
    return [f"{k}: {a.get(k)} -> {b.get(k)}" for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)]


# ---------------------------------------------------------------------------------------------- finding projects
def find_projects(root: Path, depth: int = MAX_DEPTH) -> list[Path]:
    """Folders holding a project.astroproj, root at depth 0, found by listing only (no descent into a project)."""
    out: list[Path] = []

    def walk(d: Path, n: int) -> None:
        try:
            if (d / "project.astroproj").is_file():
                out.append(d)
                return
            if n >= depth:
                return
            with os.scandir(d) as it:
                subs = sorted((e for e in it if e.is_dir(follow_symlinks=False)), key=lambda e: e.name)
        except OSError:
            return
        for e in subs:
            if e.name in SKIP_DIRS or e.name.startswith("."):
                continue
            walk(Path(e.path), n + 1)
    walk(root, 0)
    return out


def raw_entries(folder: Path) -> list[Path]:
    """The folders and files a definition lists as frames, absolute (read from the JSON alone)."""
    try:
        d = json.loads((folder / "project.astroproj").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    out = []
    for s in d.get("sessions") or []:
        for k in ("lights", "darks", "flats", "bias", "flat_darks"):
            for f in s.get(k) or []:
                if isinstance(f, str) and f:
                    out.append(Path(f) if os.path.isabs(f) else folder / f)
    return out


def safe_name(folder: Path) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", folder.as_posix()).strip("_")[:150]


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, (set, frozenset)):
        return sorted((jsonable(v) for v in x), key=lambda v: json.dumps(v, sort_keys=True))
    if isinstance(x, Path):
        return x.as_posix()
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    return repr(x)


# ---------------------------------------------------------------------------------------------- recording one project
def _keys_tables(project, scan, rules, patched: bool) -> dict:
    from astro_gpu.core.store import KeyRules, Keys
    orig = KeyRules.version
    if patched:
        KeyRules.version = lambda self, stage, values=None, mono_lights=None: "V"      # noqa: E731
    try:
        keys = Keys(project, rules, scan)
        out: dict = {}
        for w in project.workflows:
            row: dict = {"stage_keys": {}, "block_keys": {}, "errors": {}}
            try:
                chain = keys.chain(w)
            except Exception as e:  # noqa: BLE001
                row["errors"]["chain"] = f"{type(e).__name__}: {e}"
                out[str(w.id)] = row
                continue
            row["chain"] = list(chain)
            for s in chain:
                try:
                    row["stage_keys"][s] = keys.key(w, s)
                except Exception as e:  # noqa: BLE001
                    row["errors"][s] = f"{type(e).__name__}: {e}"
            try:
                for b, stages in rules.chains.blocks_in(chain):
                    try:
                        row["block_keys"][b] = {"stages": list(stages), "key": list(keys.block_key(w, b))}
                    except Exception as e:  # noqa: BLE001
                        row["errors"]["block:" + b] = f"{type(e).__name__}: {e}"
            except Exception as e:  # noqa: BLE001
                row["errors"]["blocks_in"] = f"{type(e).__name__}: {e}"
            out[str(w.id)] = row
        return out
    finally:
        KeyRules.version = orig


def _states(project) -> dict:
    from astro_gpu import api
    return {f"{wid}|{stage}": {"state": st.state, "reason": st.reason, "key": st.key,
                               "previous_key": st.previous_key, "fallback": st.fallback}
            for (wid, stage), st in api.states(project, update=False).items()}


def _frames(project, scan) -> dict:
    return {"per_session": {str(s.id): scan.frames(project, s.id) for s in project.sessions},
            "per_workflow": {str(w.id): scan.frames(project, None, project.sessions_of(w)) for w in project.workflows},
            "useful_areas": {str(w.id): scan.useful_areas(project, project.sessions_of(w)) for w in project.workflows},
            "excluded_paths": scan.excluded_paths(project)}


def _raws(project) -> dict:
    from astro_gpu.core.store import KINDS
    out, missing = {}, 0
    for s in project.sessions:
        row = {}
        for k in KINDS:
            ents = []
            for f in getattr(s, k):
                a = project.abspath(f)
                ok = Path(a).exists()
                missing += not ok
                ents.append({"path": Path(a).as_posix(), "exists": ok})
            row[k] = ents
        out[str(s.id)] = row
    return {"sessions": out, "missing_entries": missing}


def _identity_checks(project, rules, scan) -> dict:
    """Questions 3 and 6 of the integration notes: do the keys move when a legacy (options-only) workflow is given its
    graph, or when ``sessions None`` is written as the explicit full list?"""
    from astro_gpu.core import workflow as wfl
    from astro_gpu.core.project2 import Project2
    from astro_gpu.core.store import Keys
    ref = {w.id: Keys(project, rules, scan).keys(w) for w in project.workflows}
    out: dict = {"legacy_graph_same_keys": {}, "explicit_sessions_same_keys": {}}

    def cmp(q, w, label):
        try:
            got = Keys(q, rules, scan).keys(q.workflow(w.id))
            return {"same": got == ref[w.id],
                    "diff": {s: [ref[w.id].get(s), got.get(s)] for s in sorted(set(got) | set(ref[w.id]))
                             if got.get(s) != ref[w.id].get(s)}}
        except Exception as e:  # noqa: BLE001
            return {"same": None, "error": f"{type(e).__name__}: {e}"}
    for w in project.workflows:
        if w.blocks is None:
            q = Project2.from_dict(project.to_dict(), project.root)
            try:
                q.set_blocks(w.id, wfl.graph_of(q.workflow(w.id)))
                out["legacy_graph_same_keys"][str(w.id)] = cmp(q, w, "legacy")
            except Exception as e:  # noqa: BLE001
                out["legacy_graph_same_keys"][str(w.id)] = {"same": None, "error": f"{type(e).__name__}: {e}"}
        if w.sessions is None and project.sessions:
            q = Project2.from_dict(project.to_dict(), project.root)
            q.workflow(w.id).sessions = [s.id for s in q.sessions]
            out["explicit_sessions_same_keys"][str(w.id)] = cmp(q, w, "sessions")
    return out


def _stub_worker():
    """No job ever runs: nothing is read in the background, nothing is warmed."""
    from PySide6.QtCore import QObject, Signal

    class W(QObject):
        done = Signal(str, int, object)
        failed = Signal(str, int, str)

        def submit(self, *a, **k):
            return 0

        def wait(self, *a, **k):
            return True
    return W()


def _drawing(folder: Path) -> dict:
    """What client/blocks.Graph infers from a project the model opened read-only, without facts (no warm-up: the
    frames' headers are not read, so no 'broken' reasons, no colour notes) and without the sky/cost lookups."""
    from astro_gpu.core import project2
    return drawing_of(project2.load(folder))


def drawing_of(p) -> dict:
    """The drawing of ``_drawing`` for a definition already loaded (or built in memory)."""
    from astro_gpu.client import blocks as cb
    from astro_gpu.client.model import ProjectModel
    m = ProjectModel(p, _stub_worker(), readonly=True, gpu=False)
    m.sky_missing = lambda *a, **k: ""                  # needs the cost plan: not part of the structure
    m.image_problem = lambda *a, **k: ""                # opens the image file
    m.ready = True
    m._sync(prefetch=False)
    g = cb.Graph(m)
    idx = {id(n): i for i, n in enumerate(g.nodes)}

    def ix(n):
        return idx.get(id(n)) if n is not None else None
    nodes = [{"i": i, "key": n.key, "ref": n.ref, "parent": ix(n.parent), "children": [ix(c) for c in n.children],
              "paths": [q.pid for q in n.paths], "sid": n.sid, "lid": n.lid, "slot": n.slot, "num": n.num,
              "after": n.after, "handle": n.handle, "cut": n.cut, "grey": n.grey, "broken": n.broken,
              "waits": n.waits, "missing": n.missing, "bk": jsonable(n._bk)} for i, n in enumerate(g.nodes)]
    return {
        "nodes": nodes, "root": ix(g.root), "final": ix(g.final),
        "forks": [{"node": i, "key": n.key, "children": [ix(c) for c in n.children]}
                  for i, n in enumerate(g.nodes) if len(n.children) > 1],
        "leaves": {str(k): ix(v) for k, v in g.leaves.items()},
        "by_path": {str(pid): {k: ix(n) for k, n in row.items()} for pid, row in g.by_path.items()},
        "sessions": [[ix(a), ix(b)] for a, b in g.sessions],
        "calib_forks": {str(k): [ix(n) for n in v] for k, v in g.calib_forks.items()},
        "compositions": [ix(n) for n in g.compositions],
        "direct": [[ix(n), part] for n, part in g.direct],
        "to_post": sorted(g.to_post), "post_layers": jsonable(g.post_layers),
        "awaits_post": sorted(g.awaits_post),
        "fusion_inputs": {str(ix(n)): [[ix(a), r, im] for a, r, im in g.fusion_inputs[id(n)]]
                          for n in g.nodes if id(n) in g.fusion_inputs},
        "align_refs": {str(ix(n)): ix(g.align_refs[id(n)]) for n in g.nodes if id(n) in g.align_refs},
        "composition_inputs": {str(ix(n)): [[ix(a), side] for a, side in g.composition_inputs[id(n)]]
                               for n in g.nodes if id(n) in g.composition_inputs},
        "model_state": {"current": m.current, "sel_step": m.sel_step, "views": [v.pid for v in m.paths]},
        "limits": "model opened without warm-up: no facts (colour notes, session problems); sky_missing and "
                  "image_problem left empty",
    }


def record_project(folder: Path, out: Path, origin: str = "") -> dict:
    from astro_gpu.core import project2
    from astro_gpu.core.store import InputScan, key_rules
    raw = (folder / "project.astroproj").read_bytes()
    d = json.loads(raw.decode("utf-8"))
    (out / "defs" / f"{safe_name(folder)}.astroproj").write_bytes(raw)        # kept even when it does not load
    rec: dict = {"source": folder.as_posix(), "format": d.get("format"), "name": d.get("name"),
                 "definition_sha1": hashlib.sha1(raw).hexdigest()}
    if origin:
        rec["migrated_copy"] = True
        rec["original_path"] = origin
    p = project2.load(folder)
    rules = key_rules()
    scan = InputScan()
    rec["workflow_ids"] = [w.id for w in p.workflows]
    rec["stack_mode"] = p.stack_mode
    rec["sessions"] = [{"id": s.id, "name": s.name, "filter": s.filter,
                        "n": {k: len(getattr(s, k)) for k in ("lights", "darks", "flats", "bias", "flat_darks")},
                        "left_out": {k: len(v) for k, v in s.left_out.items()}} for s in p.sessions]
    rec["excluded"] = [list(x) for x in p.excluded]
    rec["workflows"] = {str(w.id): {
        "name": w.name, "options": jsonable(w.options), "blocks": jsonable(w.blocks), "end": w.end,
        "sessions": w.sessions, "variant_of": w.variant_of, "source": jsonable(w.source),
        "calibration": jsonable(w.calibration), "breakpoints": jsonable(w.breakpoints),
        "params": jsonable(w.params)} for w in p.workflows}
    rec["raws"] = _raws(p)
    rec["keys"] = _keys_tables(p, scan, rules, patched=False)
    rec["keys_constant_version"] = _keys_tables(p, scan, rules, patched=True)
    rec["states"] = _states(p)
    rec["checks"] = _identity_checks(p, rules, scan)
    frames = _frames(p, scan)
    rec["frames_sha1"] = hashlib.sha1(json.dumps(frames, sort_keys=True).encode()).hexdigest()
    try:
        rec["drawing"] = _drawing(folder)
    except Exception as e:  # noqa: BLE001
        import traceback
        rec["drawing"] = {"error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-1500:]}
    sums = {
        "legacy_workflows": [w.id for w in p.workflows if w.blocks is None and w.source is None],
        "sessions_none": [w.id for w in p.workflows if w.sessions is None],
        "variants": [w.id for w in p.workflows if w.variant_of is not None],
        "combinations": [w.id for w in p.workflows if isinstance(w.source, dict) and "fusion" in w.source],
        "images": [w.id for w in p.workflows if isinstance(w.source, dict) and "fusion" not in w.source],
        "forks": len(rec["drawing"].get("forks", [])),
        "drawing_error": rec["drawing"].get("error"),
        "missing_raws": rec["raws"]["missing_entries"],
        "key_errors": sum(len(r["errors"]) for r in rec["keys"].values()),
        "identity_diffs": [k for t in rec["checks"].values() for k, v in t.items() if v.get("same") is not True],
    }
    rec["summary"] = sums
    name = safe_name(folder)
    (out / f"{name}.json").write_text(json.dumps(rec, indent=1, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    (out / "frames" / f"{name}.json").write_text(json.dumps(frames, sort_keys=True), encoding="utf-8")
    return {"name": name, "summary": sums, "workflows": len(p.workflows), "sessions": len(p.sessions)}


# ---------------------------------------------------------------------------------------------- record
def cmd_record(args) -> int:
    out = Path(args.out).resolve()
    roots = [Path(r) for r in args.roots]
    settings = Path(os.environ.get("APPDATA", "")) / "astro_gpu" / "settings.json"
    sections, recent = [], []
    try:
        st = json.loads(settings.read_text(encoding="utf-8"))
        sections = st.get("home_sections") or []
        recent = [r.get("path") for r in st.get("recent") or [] if r.get("path")]
    except (OSError, ValueError):
        pass
    origins: dict = json.loads(Path(args.origins).read_text(encoding="utf-8")) if args.origins else {}
    found: list[tuple[str, Path]] = []
    index: dict = {"roots": {}, "home_sections": sections, "recent": recent, "projects": {}, "failed": {}, "skipped": []}
    for r in roots:
        if not r.is_dir():
            index["roots"][r.as_posix()] = {"found": 0, "note": "root absent"}
            continue
        ps = find_projects(r)
        index["roots"][r.as_posix()] = {"found": len(ps)}
        found += [(r.as_posix(), q) for q in ps]
    for sec in sections:
        fol = sec.get("folder")
        if fol and Path(fol).is_dir():
            ps = find_projects(Path(fol))
            index["roots"]["home:" + fol] = {"found": len(ps)}
            found += [("home:" + fol, q) for q in ps]
        else:
            index["roots"]["home-section:" + str(sec.get("label"))] = {"found": 0, "note": "section has no folder"}
    if args.recent:
        n = 0
        for r in recent:
            q = Path(r)
            try:
                if (q / "project.astroproj").is_file():
                    found.append(("recent", q))
                    n += 1
                else:
                    index["skipped"].append({"path": r, "why": "no project.astroproj (absent drive, v1 or not a project)"})
            except OSError as e:
                index["skipped"].append({"path": r, "why": str(e)})
        index["roots"]["recent"] = {"found": n}
    seen: set[str] = set()
    todo = []
    for label, q in found:
        k = os.path.normcase(os.path.abspath(q))
        if k not in seen:
            seen.add(k)
            todo.append((label, q))
    for _l, q in todo:
        GUARD.add(q)
        for e in raw_entries(q):
            GUARD.add(e)
    for r in roots:
        GUARD.add(r)
    for r in args.guard or []:
        GUARD.add(r)
    no = os.path.normcase(str(out))
    for g in GUARD.roots:
        if no == g or no.startswith(g + os.sep):
            print(f"refused: the output folder {out} is inside a read-only folder {g}", file=sys.stderr)
            return 2
    for sub in ("", "frames", "defs"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="graph_golden_"))
    if settings.exists():
        shutil.copy2(settings, tmp / "settings.json")
    os.environ["ASTRO_GPU_SETTINGS"] = str(tmp / "settings.json")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    sys.path.insert(0, str(REPO))
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication(sys.argv[:1])
    try:
        import psutil
        psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
    except Exception:  # noqa: BLE001
        pass
    from astro_gpu import api  # noqa: F401  (everything is imported before the guard is installed)
    from astro_gpu.client import model  # noqa: F401
    from astro_gpu.stages import v2  # noqa: F401
    GUARD.install()
    aborted = 0
    for label, q in todo:
        before, before_state = snapshot(q), snapshot_state(q)
        n_viol = len(GUARD.violations)
        try:
            index["projects"][q.as_posix()] = {"root": label, **record_project(q, out, origins.get(q.as_posix(), ""))}
        except BaseException as e:  # noqa: BLE001
            import traceback
            index["failed"][q.as_posix()] = {"root": label, "error": f"{type(e).__name__}: {str(e)[:300]}",
                                             "detail": getattr(e, "detail", ""), "trace": traceback.format_exc()[-800:]}
        d = diff(before, snapshot(q)) + diff(before_state, snapshot_state(q))
        if d or len(GUARD.violations) != n_viol:
            print(f"ABORT: the read-only project {q} changed or a write was attempted: {d[:5]} "
                  f"{GUARD.violations[n_viol:][:5]}", file=sys.stderr)
            index["ABORTED_AT"] = q.as_posix()
            index["write_proof"] = {"differences": d, "violations": GUARD.violations}
            aborted = 1
            break
    index.setdefault("write_proof", {"differences": [], "violations": GUARD.violations, "guarded": GUARD.roots,
                                     "snapshots": "top level + one level down + state.json, before/after each project"})
    (out / "_index.json").write_text(json.dumps(index, indent=1, ensure_ascii=False), encoding="utf-8")
    (out / "_roots.json").write_text(json.dumps({"roots": [r.as_posix() for r in roots], "recent": bool(args.recent),
                                                  "origins": str(Path(args.origins).resolve()) if args.origins else "",
                                                  "guard": list(args.guard or [])}),
                                     encoding="utf-8")
    print(json.dumps({"recorded": len(index["projects"]), "failed": len(index["failed"]), "aborted": bool(aborted)}))
    return aborted


# ---------------------------------------------------------------------------------------------- stable
def cmd_stable(args) -> int:
    out = Path(args.out).resolve()
    cfg = json.loads((out / "_roots.json").read_text(encoding="utf-8"))
    runs = []
    for tag in ("_stable_a", "_stable_b"):
        d = out / tag
        cmd = [sys.executable, str(Path(__file__).resolve()), "record", *cfg["roots"], "--out", str(d)]
        if cfg.get("recent"):
            cmd.append("--recent")
        if cfg.get("origins"):
            cmd += ["--origins", cfg["origins"]]
        for g in cfg.get("guard") or []:
            cmd += ["--guard", g]
        r = subprocess.run(cmd, env={**os.environ, "PYTHONUTF8": "1"}, capture_output=True, text=True)
        print(tag, r.returncode, r.stdout.strip()[-200:], r.stderr.strip()[-300:])
        if r.returncode:
            return 1
        runs.append(d)
    names = sorted(p.name for p in runs[0].glob("*.json") if p.name != "_index.json")
    diffs = [n for n in names if json.loads((runs[0] / n).read_text(encoding="utf-8")) !=
             json.loads((runs[1] / n).read_text(encoding="utf-8"))]
    diffs += [f"frames/{p.name}" for p in sorted((runs[0] / "frames").glob("*.json"))
              if p.read_bytes() != (runs[1] / "frames" / p.name).read_bytes()]
    vs_main = [n for n in names if (out / n).exists() and json.loads((out / n).read_text(encoding="utf-8")) !=
               json.loads((runs[0] / n).read_text(encoding="utf-8"))]
    print(json.dumps({"files": len(names), "different_between_two_runs": diffs, "different_from_main_record": vs_main}))
    return 1 if diffs else 0


# ---------------------------------------------------------------------------------------------- check
def _keys_diff_modulo_defaults(p, q, rules, scan) -> list[str]:
    """Every stage key and block key as they are; the overrides without the values stored at their static default
    (client/graphdefaults.strip_override_defaults: the jobs get the same settings)."""
    from astro_gpu.client.graphdefaults import strip_override_defaults
    from astro_gpu.graphcore import proof
    a = strip_override_defaults(proof.keys_table(p, rules, scan), rules)
    b = strip_override_defaults(proof.keys_table(q, rules, scan), rules)
    return [d for wid in a if (d := proof._first_diff(a[wid], b.get(wid), f"workflow {wid}"))]


def check_project(folder: Path, rec: dict) -> dict:
    """One recorded project through the graph model, in this process: its definition loaded, converted, compiled
    into a copy; then (a) the definitions, (b) every stage key, block key and override, (c) the states and the
    review keys, (d) the graph's rows against the drawing recorded from today's client."""
    from astro_gpu.core import project2
    from astro_gpu.core.store import InputScan, key_rules
    from astro_gpu.graphcore import proof
    from astro_gpu.graphcore.convert import Client
    from astro_gpu.graphcore.model import check as core_check, Invalid
    from astro_gpu.graphcore.rules import structure
    from astro_gpu.graphcore.msg import say
    p = project2.load(folder)
    rules, scan = key_rules(), InputScan()
    base = p
    files = None
    if os.environ.get("FLOAT_FILES_ONLY") == "1":
        # files only (the graph page stores no folder): each folder entry replaced by the files the engine reads there;
        # the definitions compared from that point, the keys, states and frames against the project as it is
        from astro_gpu.graphcore.convert import files_only
        base = project2.Project2.from_dict(p.to_dict(), p.root)
        files = files_only(base)
    g, q, log = proof.round_trip(base)
    out: dict = {"workflows": len(p.workflows), "sessions": len(p.sessions), "convert_log": log}
    if files is not None:
        out["kinds_made_files"] = files
    if os.environ.get("FLOAT_EXPLICIT_DEFAULTS") == "1":
        # the file says everything (client/graphdefaults.py): every default written on the graph, compiled again;
        # the definitions compared without the values written at their static default, the keys as they are
        from astro_gpu.client.graphdefaults import strip_defaults, with_explicit_defaults
        from astro_gpu.graphcore.compile import compile_into
        g, out["defaults_written"] = with_explicit_defaults(g, like=base)
        q = project2.Project2.from_dict(base.to_dict(), p.root)
        compile_into(g, q)
        stripped = project2.Project2.from_dict(strip_defaults(q.to_dict(), base.to_dict()), p.root)
        out["definition"] = proof.definition_diff(base, stripped)
    else:
        out["definition"] = proof.definition_diff(base, q)
    if os.environ.get("FLOAT_EXPLICIT_DEFAULTS") == "1":
        out["keys"] = _keys_diff_modulo_defaults(p, q, rules, scan)
    else:
        out["keys"] = proof.keys_diff(p, q, rules, scan)
    out["states"] = proof.states_diff(p, q)
    cl_p, cl_q = Client(p), Client(q)
    ra, rb = proof.review_keys(p, cl_p), proof.review_keys(q, cl_q)
    out["reviews"] = [f"review {k}: {ra.get(k)} -> {rb.get(k)}" for k in sorted(set(ra) | set(rb))
                      if ra.get(k) != rb.get(k)]
    dr = rec.get("drawing") or {}
    if "nodes" in dr:
        out["drawing"], out["drawn_blocks"] = proof.drawing_check(p, g, q, dr, cl_q)
    else:
        out["drawing"] = [f"no recorded drawing: {dr.get('error')}"]
    frames_p = {s.id: scan.frames(p, s.id) for s in p.sessions}
    frames_q = {s.id: scan.frames(q, s.id) for s in q.sessions}
    images = [w.source["path"] for w in p.workflows if isinstance(w.source, dict) and w.source.get("path")]
    out["frames_nonempty"] = bool(p.sessions) and all(frames_p.values()) and all(frames_q.values())
    # every workflow from frames reads frames on both sides (a session with no lights in the definition reads none,
    # whatever is on the disk): its keys were compared on real frames
    flows = [w for w in p.workflows if w.source is None]
    out["workflow_frames_nonempty"] = bool(flows) and all(
        scan.frames(x, None, x.sessions_of(w.id)) for x in (p, q) for w in flows)
    out["empty_definition"] = not p.sessions and not p.workflows
    out["images_present"] = bool(images) and all(Path(p.abspath(x)).exists() for x in images)
    out["frames_equal"] = frames_p == frames_q
    try:
        core_check(g)
        out["core_check"] = ""
    except Invalid as e:
        out["core_check"] = str(e)
    out["core_structure"] = [say(m) for m in structure(g)]
    # the converted graph as a whole is one the core keeps (its legality: the engine's rules on every row, the
    # joins' ports, the compiled definition as the engine reads it): no gesture would be refused for its state
    from astro_gpu.graphcore.operations import legality
    try:
        why = legality(g)
        out["legal"] = [say(why)] if why else []
    except Exception as e:  # noqa: BLE001
        out["legal"] = [f"{type(e).__name__}: {e}"]
    return out


def cmd_check(args) -> int:
    gold = Path(args.golden).resolve()
    index = json.loads((gold / "_index.json").read_text(encoding="utf-8"))
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    tmp = gold / "_check_settings"
    tmp.mkdir(exist_ok=True)
    settings = Path(os.environ.get("APPDATA", "")) / "astro_gpu" / "settings.json"
    if settings.exists():
        shutil.copy2(settings, tmp / "settings.json")
    os.environ["ASTRO_GPU_SETTINGS"] = str(tmp / "settings.json")
    sys.path.insert(0, str(REPO))
    try:
        import psutil
        psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
    except Exception:  # noqa: BLE001
        pass
    from astro_gpu import api  # noqa: F401
    from astro_gpu.graphcore import compile as _c, convert as _v, proof as _p  # noqa: F401
    from astro_gpu.stages import v2  # noqa: F401
    todo = []
    for src, row in index["projects"].items():
        q = Path(src)
        recf = gold / f"{row['name']}.json"
        if args.only and not any(o in row["name"] for o in args.only):
            continue
        todo.append((q, recf))
        GUARD.add(q)
        for e in raw_entries(q):
            GUARD.add(e)
        try:
            d = json.loads((q / "project.astroproj").read_text(encoding="utf-8"))
            for w in d.get("workflows") or []:
                s = w.get("source")
                if isinstance(s, dict) and s.get("path"):
                    GUARD.add(Path(s["path"]) if os.path.isabs(s["path"]) else q / s["path"])
        except (OSError, ValueError):
            pass
    no = os.path.normcase(str(gold))
    for g in GUARD.roots:
        if no == g or no.startswith(g + os.sep):
            print(f"refused: the golden folder {gold} is inside a read-only folder {g}", file=sys.stderr)
            return 2
    GUARD.install()
    report: dict = {"projects": {}, "failed": {}}
    for q, recf in todo:
        before, before_state = snapshot(q), snapshot_state(q)
        n_viol = len(GUARD.violations)
        try:
            rec = json.loads(recf.read_text(encoding="utf-8"))
            report["projects"][q.as_posix()] = check_project(q, rec)
        except BaseException as e:  # noqa: BLE001
            import traceback
            report["failed"][q.as_posix()] = {"error": f"{type(e).__name__}: {str(e)[:300]}",
                                              "trace": traceback.format_exc()[-1500:]}
        d = diff(before, snapshot(q)) + diff(before_state, snapshot_state(q))
        if d or len(GUARD.violations) != n_viol:
            report["ABORTED_AT"] = q.as_posix()
            report["write_proof"] = {"differences": d, "violations": GUARD.violations[n_viol:]}
            print(f"ABORT: the read-only project {q} changed or a write was attempted", file=sys.stderr)
            break
    report.setdefault("write_proof", {"differences": [], "violations": GUARD.violations})
    gate = ("definition", "keys", "states", "reviews", "drawing", "legal")
    summary = {"proven": [], "proven_images": [], "empty": [], "definition_equal_only": [], "differing": {}}
    for src, r in report["projects"].items():
        bad = {k: r[k][0] for k in gate if r.get(k)}
        if bad:
            summary["differing"][src] = bad
        elif r["frames_nonempty"] or r["workflow_frames_nonempty"]:
            summary["proven"].append(src)
        elif not r["sessions"] and r["images_present"]:
            summary["proven_images"].append(src)
        elif r["empty_definition"]:
            summary["empty"].append(src)          # no session, no workflow: no key to compare
        else:
            summary["definition_equal_only"].append(src)
    report["summary"] = summary
    out = Path(args.out) if args.out else gold / "_check.json"
    GUARD.active = False
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps({"proven": len(summary["proven"]), "proven_images": len(summary["proven_images"]),
                      "empty": len(summary["empty"]),
                      "definition_equal_only": len(summary["definition_equal_only"]),
                      "differing": len(summary["differing"]), "failed": len(report["failed"]),
                      "aborted": "ABORTED_AT" in report}))
    return 1 if summary["differing"] or report["failed"] or "ABORTED_AT" in report else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="every recorded project converted to a graph and compiled back: no difference")
    c.add_argument("golden")
    c.add_argument("--out")
    c.add_argument("--only", action="append", help="only the projects whose recorded name holds this text")
    r = sub.add_parser("record")
    r.add_argument("roots", nargs="*")
    r.add_argument("--out", required=True)
    r.add_argument("--recent", action="store_true", help="also the Home 'recent projects' of the settings")
    r.add_argument("--guard", action="append", help="more folders to keep read-only")
    r.add_argument("--origins", help="JSON {copy folder: original folder} of definitions migrated on a copy")
    s = sub.add_parser("stable")
    s.add_argument("out")
    a = ap.parse_args()
    return {"record": cmd_record, "stable": cmd_stable, "check": cmd_check}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
