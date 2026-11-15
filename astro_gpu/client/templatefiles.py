"""Templates are blank project files (issue #167): a folder per tab, one project file per template.

Two roots are read, in the same tabs: the folder shipped with the application (``astro_gpu/resources/templates``,
read-only) and the user's own, next to the application's other user settings (``<settings folder>/templates``). A
sub-folder that exists in only one of them is a tab of its own. Nothing here knows Qt.

A template file is a project file (format 3: its graph) without frames, results or paths, and a ``template`` entry
that the gallery shows: ``{"card": {...}}`` (its name, its text, and the recipe that adds the same workflow to a
project that already holds frames) and, for a file saved by the user, its ``text``. Nothing is deduced from a file's name
or a folder's: a tab is a folder, a card is a file."""
from __future__ import annotations

import copy
import json
from pathlib import Path

from . import prefs

#: a template file's extension, the project file's
SUFFIX = ".astroproj"
#: the tabs the application ships, in order; any other tab follows, by name
SHIPPED_TABS = ("Color", "Mono")
#: the views' options a user's template gets (a recipe from its graph alone)
USER_OPTS = dict(debruitage=False, etoiles="garder", fond="simple")


def shipped_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "resources" / "templates"


def user_dir() -> Path | None:
    """The user's folder, next to the settings file; None where the settings are not kept (the tests)."""
    return prefs.FILE.parent / "templates" if prefs.FILE is not None else None


def _folders(root: Path | None) -> list[Path]:
    return sorted(p for p in root.iterdir() if p.is_dir()) if root is not None and root.is_dir() else []


def tabs() -> list[str]:
    """The tabs: the shipped ones first (Color, Mono), then the others by name, a folder of either root one tab."""
    names = {p.name for p in _folders(shipped_dir())} | {p.name for p in _folders(user_dir())}
    return [n for n in SHIPPED_TABS if n in names] + sorted(n for n in names if n not in SHIPPED_TABS)


def files(tab: str) -> list[tuple[Path, bool]]:
    """The template files of a tab: (path, the user's), the shipped ones first, each root's by name."""
    out = []
    for root, mine in ((shipped_dir(), False), (user_dir(), True)):
        d = root / tab if root is not None else None
        if d is not None and d.is_dir():
            out += [(f, mine) for f in sorted(d.glob("*" + SUFFIX))]
    return out


def read(path: Path) -> dict | None:
    """A template file's content; None when it is not a project file this program reads."""
    try:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return d if isinstance(d, dict) and isinstance(d.get("graph"), dict) else None


def card(path: Path, tab: str, mine: bool) -> dict | None:
    """What the gallery shows of a template file, as its cards have always been: key, tab, name, text, and the recipe
    (``opts``, ``kind``, ``source``, ``blocks``...) that adds its workflow to a project holding frames."""
    d = read(path)
    if d is None:
        return None
    info = d.get("template") if isinstance(d.get("template"), dict) else {}
    c = copy.deepcopy(info.get("card") or {})
    stem = Path(path).stem
    c.setdefault("name", stem)
    c.setdefault("text", str(info.get("text") or ""))
    c.setdefault("opts", dict(USER_OPTS))
    c["key"] = c.get("key") or f"user:{tab}/{stem}"
    if mine:
        c["key"] = f"user:{tab}/{stem}"
    c.update(tab=tab, file=Path(path).as_posix(), user=mine)
    if not any(k in c for k in ("kind", "blocks", "source")) and not info.get("card", {}).get("opts"):
        c.update(recipe(d["graph"]))                    # no recipe stored: it is read from the file's graph
    return c


#: the blocks of a row that are not a workflow's own chain (they are the project's: sessions, calibration, outputs)
_NOT_CHAIN = ("session", "calibration", "output", "align_on", "post")


def recipe(graph: dict) -> dict:
    """What adds a template's workflow to a project that already holds frames, read from the template's graph: the
    source of its first workflow and its blocks, in the nodes the gallery adds (``blocks``). A graph without a
    workflow adds nothing."""
    rows = graph.get("rows") or []
    blocks = graph.get("blocks") or {}
    kinds = [(blocks[b].get("kind"), blocks[b].get("method")) for b in (rows[0].get("blocks") or []) if b in blocks]         if rows else []
    if kinds and kinds[0][0] == "image":
        return {"source": "image", "blocks": []}
    nodes = [{"block": "calibration", "name": "calibration", "method": None, "off": False}] if kinds else []
    nodes += [{"block": k, "name": k, "method": m, "off": False} for k, m in kinds if k and k not in _NOT_CHAIN]
    return {"source": "frames", "blocks": nodes}


def cards(tab: str) -> list[dict]:
    return [c for f, mine in files(tab) if (c := card(f, tab, mine)) is not None]


def all_cards() -> list[dict]:
    return [c for t in tabs() for c in cards(t)]


def shipped_cards() -> list[dict]:
    """The shipped templates only, in the order of the tabs."""
    return [c for t in SHIPPED_TABS for f, mine in files(t) if not mine and (c := card(f, t, False)) is not None]


# ---------------------------------------------------------------------------------------------- blank, and saved
def blank(d: dict) -> dict:
    """A project file's content without what belongs to the work: the frames of its sessions, the files of its Image
    blocks, what was calculated, the names and dates. What stays is the graph and the settings."""
    d = copy.deepcopy(d)
    for k in ("created", "name", "scratch_dir", "reference_image", "description", "current_block"):
        d[k] = ""
    d["target"] = {}
    d["render_bases"] = {}
    d["acquisition"] = {}
    g = d.get("graph") or {}
    g["target"] = ""
    g["excluded"] = []
    g["common"] = {"files": {}, "settings": g.get("common", {}).get("settings", {})}
    for n in g.get("nights") or []:
        n["files"] = {}
        n["left_out"] = {}
    for b in (g.get("blocks") or {}).values():
        if b.get("files") is not None:
            b["files"] = {}
        if b.get("kind") == "image":
            b.get("settings", {}).pop("Image file", None)
    d["graph"] = g
    for k in ("sessions", "workflows", "next_workflow_id", "excluded", "stack_mode"):
        d.pop(k, None)                                  # compiled from the graph when the file is read
    return d


def save_user(d: dict, tab: str, name: str, text: str = "", card_data: dict | None = None) -> Path:
    """The blank project ``d`` as a template of the user's, in the folder ``tab``, under ``name``. An existing file of
    that name is replaced. -> the file."""
    root = user_dir()
    if root is None:
        raise OSError("no folder for the user's templates")
    safe = lambda s: "".join(c for c in s.strip() if c not in '\\/:*?"<>|').strip(" .")      # noqa: E731
    tab, name = safe(tab), safe(name)
    if not tab or not name:
        raise ValueError("a template needs a tab and a name")
    out = blank(d)
    out["template"] = {"text": text.strip(), **({"card": card_data} if card_data else {})}
    f = root / tab / (name + SUFFIX)
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(out, indent=1, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return f


def delete_user(path: str | Path) -> bool:
    """A template file of the user's goes (a shipped one never does)."""
    f = Path(path).resolve()
    root = user_dir()
    if root is None or root.resolve() not in f.parents or f.suffix != SUFFIX or not f.is_file():
        return False
    f.unlink()
    return True


def project_file(path: str | Path) -> dict:
    """The content of the project file a template makes: the template's, without its gallery entry."""
    d = read(Path(path))
    if d is None:
        raise ValueError("not a template file")
    d = copy.deepcopy(d)
    d.pop("template", None)
    return d


# ------------------------------------------------------------------------- the templates kept in the settings
#: the tab of the templates the user made before they were files
OLD_TAB = "Mine"
#: the settings entry that says the old templates were written as files
MOVED = "templates_moved"


def _from_old(entry: dict) -> dict:
    """The blank project file of a template kept in the settings ({name, text, source, blocks}): the workflow the
    gallery made of it on an empty project, its graph as the application writes one."""
    import tempfile
    from ..core import project2
    from ..core.project import Target
    from ..graphcore import model as gm
    from ..graphcore import templates
    from ..graphcore.compile import compile_into
    from ..graphcore.convert import convert
    from ..graphcore.operations import apply_many
    t = {"key": entry.get("key") or "", "name": entry["name"], "source": entry.get("source") or "frames",
         "blocks": [dict(n) for n in entry.get("blocks") or []]}
    with tempfile.TemporaryDirectory(prefix="float_old_template_") as tmp:
        p = project2.create(Path(tmp), "Untitled", Target())
        g = convert(p, block_keys=lambda wid: [])
        plan = templates.build(g, t, entry["name"], dict(USER_OPTS))
        g2, res = apply_many(g, plan.steps)
        if not res.ok:
            raise ValueError(res.reason_text())
        compile_into(g2, p)
        p.graph = gm.to_dict(g2)
        d = project2.file_dict(p)
    if d.get("format") != project2.GRAPH_FORMAT:
        raise ValueError("its graph does not compile")
    return d


def import_old() -> list[Path]:
    """At start: each template the settings keep under the old key is written once as a blank project file in the
    user's folder, tab "Mine"; the settings entry stays as it is and ``templates_moved`` says it is done. An entry
    that cannot be written is left out (it is logged), never written approximately. -> the files written."""
    import logging
    out: list[Path] = []
    if user_dir() is None or prefs.get(MOVED):
        return out
    for e in prefs.get("templates") or []:
        try:
            if not isinstance(e, dict) or not str(e.get("name") or "").strip():
                continue
            name = str(e["name"]).strip()
            safe = "".join(c for c in name if c not in '\\/:*?"<>|').strip(" .")
            if safe and (user_dir() / OLD_TAB / (safe + SUFFIX)).exists():
                continue
            out.append(save_user(_from_old(e), OLD_TAB, name, text=str(e.get("text") or "")))
        except Exception:  # noqa: BLE001 - one template that cannot be turned into a file does not stop the others
            logging.getLogger(__name__).exception("the template %r could not be written as a file", e.get("name") if isinstance(e, dict) else e)
    prefs.set(MOVED, True)
    return out
