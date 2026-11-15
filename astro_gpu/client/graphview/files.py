"""The file-type filter of the imports, and the logic of a drop of files and folders, shared by the table and by the
graph's Session targets (graph_drops.py).

The filter and « Include » are choices of the drop, resolved AT DROP TIME; nothing about the filter is stored. FILES
ONLY: a dropped or picked folder is listed at that moment (its first level; « Include »: its sub-folders too), the filter
keeps its frames and they are stored one by one; the page never stores a folder (the table shows the files under their
folder, a display only). Types the engine cannot read are never added. The counts said are those of the files stored.

Matching is by the engine's own frame kinds (``frame_kind``: by extension only, case-insensitive): nothing else of a
name is read, nothing is sorted or proposed from a name, a folder or a date. The filter chosen last is remembered in
the client's settings (prefs, "graph_file_filter") and is the one applied to drops.
"""
from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass, field

from ...core.project import COLOUR_EXT, FITS_EXT, RAW_EXT, frame_kind, list_frames
from .. import prefs
from ..i18n import _, plural

#: (English name, the engine's frame kinds it keeps), in the order of the dialog and of the combo. There is no « All
#: files »: the engine reads frames only, so it could add nothing more than « All images »
TYPES = [("Camera raws", ("raw",)), ("FITS", ("fits",)), ("Raws and FITS", ("raw", "fits")),
         ("All images", ("raw", "fits", "colour"))]
#: the extensions of each kind, for the file dialog's patterns
KIND_EXT = {"raw": sorted(RAW_EXT), "fits": sorted(FITS_EXT) + [".fz"], "colour": sorted(COLOUR_EXT)}
DEFAULT = 2
PREF = "graph_file_filter"
TIP = "Only the files of this type (by their extension) are taken at import; the others are left out."


# ---------------------------------------------------------------------------------------------------- the filter
def index() -> int:
    """The filter in force (the last one chosen)."""
    return index_of_name(prefs.get(PREF) or "")


def set_index(i: int):
    if 0 <= i < len(TYPES):
        prefs.set(PREF, TYPES[i][0])


def filter_name(i: int) -> str:
    """The English name of filter i (shown through _)."""
    return TYPES[i][0]


def index_of_name(name: str) -> int:
    for i, (n, _e) in enumerate(TYPES):
        if n == name:
            return i
    return DEFAULT


def extension(path: str) -> str:
    return os.path.splitext(path)[1].lstrip(".").lower()


def accepts(path: str, i: int | None = None) -> bool:
    """Whether the filter keeps this file: a frame of one of its kinds (never a file the engine cannot read)."""
    k = frame_kind(path)
    return bool(k) and k in TYPES[index() if i is None else i][1]


def describe(source: str, is_folder: bool) -> str:
    """The tooltip line of an entry: "folder · its own files" / "file"."""
    return _("folder · without sub-folders") if is_folder else _("file")


def dialog_filters() -> list[str]:
    """The file dialog's name filters, in order."""
    out = []
    for name, kinds in TYPES:
        exts = [e for k in kinds for e in KIND_EXT[k]]
        pats = " ".join(f"*{e} *{e.upper()}" for e in exts)
        out.append(f"{_(name)} ({pats})")
    return out


def index_of_filter(text: str) -> int:
    """Index of the dialog's selected name filter (as the dialog gives it back)."""
    for i, f in enumerate(dialog_filters()):
        if f == text or text.startswith(_(TYPES[i][0]) + " ("):
            return i
    return index()


def summary(paths) -> str:
    """"214 ARW" / "5 ARW, 3 FIT": counts by extension (upper case), most numerous first."""
    c = Counter(extension(f).upper() or _("no extension") for f in paths)
    return ", ".join(f"{n} {e}" for e, n in c.most_common())


# ---------------------------------------------------------------------------------------------------- drops (shared)
def is_folder(reader, p: str) -> bool:
    d = reader.known(p)
    return os.path.isdir(p) if d is None else d


def already(p: str, existing) -> bool:
    """No silent doubles: the same entry, or a file the engine already reads through a folder entry (its own level;
    a JPEG beside raws is not read there, so it is not a double)."""
    if p in existing:
        return True
    if os.path.isdir(p):
        return False
    parent = os.path.dirname(p).replace("\\", "/")
    return any(parent == e.rstrip("/") and p in engine_frames([e]) for e in existing)


def without_doubles(res, existing) -> int:
    """The entries, sub-folders and left-out files of a drop already in the list taken out. -> how many entries were
    doubles."""
    existing = set(existing)
    new = [p for p in res.items if not already(p, existing)]
    ignored, res.items = len(res.items) - len(new), new
    res.subs = [x for x in res.subs if not _sub_taken(x, existing)]
    res.left = [x for x in res.left if x not in existing]       # left out by the filter, but taken before
    return ignored


def _sub_taken(folder: str, existing) -> bool:
    """A sub-folder already taken: its entry, or (files only) every frame at its level already in the list."""
    if folder in existing:
        return True
    fs, _ds, bad = _level(folder)
    frames = [f for f in fs if frame_kind(f)]
    return not bad and bool(frames) and all(already(f, existing) for f in frames)


def normalise(p: str) -> str:
    p = str(p).replace("\\", "/")
    return p.rstrip("/") if len(p) > 3 else p


def note_paths(reader, paths) -> tuple[int, int]:
    """Tell the reader which dropped paths are folders (a stat each, immediate). -> (folders, files). The folder of
    the first one is where the next file dialog opens (a drop is a use: « Add lights » proposed the home folder)."""
    if paths:
        prefs.remember_folder(list(paths)[0])
    nd = nf = 0
    for p in paths:
        d = os.path.isdir(p)
        reader.note(p, d)
        nd += d
        nf += not d
    return nd, nf


def engine_frames(entries) -> list[str]:
    """What the engine will read of these entries (core ``list_frames``), forward slashes."""
    return [str(p).replace("\\", "/") for p in list_frames(list(entries))]


def _level(folder: str):
    """(files, sub-folders) of a folder's first level, names sorted; ([], [], True) when it cannot be read."""
    fs, ds = [], []
    try:
        with os.scandir(folder) as it:
            for e in sorted(it, key=lambda x: x.name):
                try:
                    (ds if e.is_dir(follow_symlinks=False) else fs).append(e.path.replace("\\", "/"))
                except OSError:
                    continue
    except OSError:
        return [], [], True
    return fs, ds, False


def frame_folders(folder: str) -> list[str]:
    """Every folder below this one (all levels, depth first, by name) holding at least one frame at its own level."""
    out, todo = [], [folder]
    while todo:
        d = todo.pop()
        fs, ds, _bad = _level(d)
        if d != folder and any(frame_kind(f) for f in fs):
            out.append(d)
        todo.extend(reversed(ds))
    return out


@dataclass
class Resolution:
    """What a drop stores and what it says: ``items`` the files to store (one by one), ``left`` frames the filter left out (« Take all »), ``others`` files the engine cannot
    read (never added), ``subs`` sub-folders holding frames (« Include »), ``unreadable`` folders that cannot be read,
    ``empty`` folders that gave nothing."""
    items: list = field(default_factory=list)
    left: list = field(default_factory=list)
    elsewhere: list = field(default_factory=list)      # lights already in another session of the stack (not added)
    elsewhere_in: list = field(default_factory=list)   # the names of those sessions
    others: int = 0
    subs: list = field(default_factory=list)
    unreadable: list = field(default_factory=list)
    empty: list = field(default_factory=list)

    def frames(self) -> list[str]:
        return engine_frames(self.items)


def _folder_of(p: str) -> str:
    return os.path.normcase(os.path.dirname(normalise(p)))


def beside_raws(paths, existing=()) -> list[str]:
    """The colour images (JPEG, PNG, TIFF) of ``paths`` whose folder holds raw or FITS frames among ``paths`` or among
    ``existing`` (the entries the list already holds): the engine's own rule for a folder (``list_frames``: a camera's
    JPEGs beside its raws are not more frames, issue #57), kept for single files. The engine takes a file entry as it
    is, so a JPEG stored beside raws would be read as a frame: the inventory then keeps the pattern of the most frames
    and drops the others, the raws when the JPEGs are more."""
    raw = {_folder_of(p) for p in list(paths) + list(existing) if frame_kind(p) in ("raw", "fits")}
    return [p for p in paths if frame_kind(p) == "colour" and _folder_of(p) in raw]


def stack_lights(graph, session=None, row=None) -> dict[str, str]:
    """{a light's path, case-folded: the name of the session holding it} of the OTHER sessions of the stack(s) drawing
    ``session`` (or of stack ``row``): the same night added twice is counted twice on the screen, though the engine
    reads each file once."""
    out: dict[str, str] = {}
    for r in graph.rows:
        ids = [s for s, _c in r.nights]
        if (session is not None and session in ids) or (row is not None and r.id == row):
            for s in ids:
                b = graph.blocks.get(s)
                if s == session or b is None:
                    continue
                for f in ((b.files or {}).get("lights") or []):
                    out.setdefault(os.path.normcase(normalise(f)), b.name or s)
    return out


def elsewhere(res: Resolution, others: dict[str, str]) -> None:
    """The lights of a drop already in another session of the stack (``stack_lights``) taken out of its items: said,
    not added (« Add anyway » adds them)."""
    hit = [p for p in res.items if os.path.normcase(normalise(p)) in others]
    if hit:
        gone = set(hit)
        res.items = [p for p in res.items if p not in gone]
        res.elsewhere = hit
        res.elsewhere_in = sorted({others[os.path.normcase(normalise(p))] for p in hit})


def elsewhere_text(res: Resolution) -> str:
    """« 60 already in “Session 1”: not added », "" when none."""
    if not res.elsewhere:
        return ""
    return _("{n} already in {sessions}: not added").format(
        n=len(res.elsewhere), sessions=", ".join(f"“{s}”" for s in res.elsewhere_in))


def header_filter(paths) -> str:
    """The filter the FITS header of the first FITS frame of ``paths`` names (its FILTER value, read as the engine
    reads it: io.frames.frame_filter); "" for raws or no filter. Never a file's or a folder's name."""
    from ...io.frames import frame_filter
    first = next((p for p in paths if frame_kind(p) == "fits"), None)
    return frame_filter(first) if first is not None else ""


def session_filter(graph, session) -> str:
    """The filter stored for a Session block's night ("" for a colour camera's, or none stored)."""
    b = graph.blocks.get(session)
    sid = getattr(b, "sid", None)
    e = next((n for n in graph.nights if n.get("sid") == sid), None) if sid is not None else None
    return str((e or {}).get("filter") or "")


def stack_filter(graph, row) -> str:
    """The one filter of a stack's nights, "" when they name none or several."""
    r = next((x for x in graph.rows if x.id == row), None)
    flts = {session_filter(graph, s) for s, _c in (r.nights if r is not None else [])}
    return next(iter(flts)) if len(flts) == 1 else ""


def free_session_name(graph, base: str) -> str:
    """``base``, else « base 2 », « base 3 »…: a name no session has (the name of the frames' own filter)."""
    names = {b.name for b in graph.blocks.values() if b.kind == "session"} | {n.get("name") for n in graph.nights}
    k, name = 2, base
    while name in names:
        name, k = f"{base} {k}", k + 1
    return name


def own_filter_steps(graph, flt: str, items) -> list:
    """« New {f} session »: a stack of its own for frames whose header names filter ``flt``: its night (that filter,
    named after it), its stack, then its lights."""
    from ...graphcore.operations import _next_sid
    sid, name = _next_sid(graph), free_session_name(graph, flt)
    return [("new_night", dict(name=name, filter=flt)),
            ("add_stack", dict(name=name, nights=[sid])),
            ("add_files", lambda outs: dict(session=outs[-1].created["session"], kind="lights", paths=list(items),
                                            mode="add"))]


def filter_question(flt: str, other: str, where: str) -> tuple[str, list]:
    """(text, links) of frames whose header names another filter than the session or stack they are dropped on: the
    user chooses, nothing is added before."""
    return (_("These frames' header names filter {f}, “{s}” is {g}: where do they go?").format(f=flt, s=where, g=other),
            [(_("Add here"), "filter_here"), (_("New {f} session").format(f=flt), "filter_own"), (_("Cancel"), "cancel")])


def beside_note(stored) -> tuple[list, str, str]:
    """A list saved with colour images beside its raws (before they were left at the gesture): (those images, the
    notice, the button's label), or ([], "", "") when it holds none. Said, never removed without the user's click."""
    aside = beside_raws(list(stored), list(stored))
    if not aside:
        return [], "", ""
    typ = ", ".join(sorted({extension(f).upper() for f in aside}))
    return (aside, plural(len(aside), "{n} {type} beside raw frames: not usual, read as one more frame",
                          "{n} {type} beside raw frames: not usual, each read as one more frame", type=typ),
            _("Remove {type}").format(type=typ))


def resolve(reader, paths, i: int | None = None, deep: bool = True, existing=()) -> Resolution:
    """A drop resolved now: each folder's first level read (scandir, names only), the filter applied, the engine's
    rule compared (see the module's text). ``deep`` False: the sub-folders are not looked for (« Include » itself).
    Whatever the filter, a colour image beside raw or FITS frames of its folder (taken in the same gesture, or already
    in ``existing``, the list it goes to) is left out (``beside_raws``): « Take all » is the only way to store it."""
    r = _resolve(reader, paths, index() if i is None else i, deep)
    aside = set(beside_raws(r.items, existing))
    if aside:
        r.items = [p for p in r.items if p not in aside]
        r.left += [p for p in sorted(aside) if p not in r.left]
    return r


def _resolve(reader, paths, i: int, deep: bool) -> Resolution:
    r = Resolution()
    for p in paths:
        p = normalise(p)
        if is_folder(reader, p):
            fs, ds, bad = _level(p)
            if bad:
                r.unreadable.append(p)
                continue
            readable = [f for f in fs if frame_kind(f)]
            chosen = [f for f in readable if accepts(f, i)]
            if not chosen and readable and i == DEFAULT:
                # the default type takes what the engine reads in this folder: its colour images (JPEG, TIFF...)
                # when it holds no raw nor FITS (issue #57), never a camera's JPEGs beside its raws
                read = set(engine_frames([p]))
                chosen = [f for f in readable if f in read]
            r.others += len(fs) - len(readable)
            r.left += [f for f in readable if f not in chosen]
            if chosen:                                    # each file stored on its own: never the folder
                for f in chosen:
                    r.items.append(f)
                    reader.note(f, False)
            else:
                r.empty.append(p)
            if deep and ds:
                r.subs += frame_folders(p)
        elif frame_kind(p):
            if accepts(p, i):
                r.items.append(p)
                reader.note(p, False)
            else:
                r.left.append(p)
        else:
            r.others += 1
    return r


def merge(parts) -> Resolution:
    """One Resolution of several (the groups of a drop made into sessions), for its toast."""
    r = Resolution()
    for x in parts:
        r.items += x.items
        r.left += x.left
        r.others += x.others
        r.subs += x.subs
        r.unreadable += x.unreadable
        r.empty += x.empty
        r.elsewhere += x.elsewhere
        r.elsewhere_in = sorted(set(r.elsewhere_in) | set(x.elsewhere_in))
    return r


def include(reader, subs, i: int | None = None) -> Resolution:
    """« Include »: the frames of each sub-folder holding some, kept by the filter, each file stored on its own."""
    return resolve(reader, subs, i, deep=False)


def drop_sentence(base_text, res: Resolution, mode, ignored=0, taken=None):
    """The toast of a drop: what is really read (``list_frames`` of the stored entries), what was left out, the
    sub-folders not read, the doubles ignored."""
    fs = res.frames() if taken is None else taken
    text = base_text() if callable(base_text) else base_text
    not_read = len(res.subs)
    if res.left and fs:
        src = "{taken} replace the previous ones · {left} left" if mode == "replace" else "{taken} added · {left} left"
        parts = [_(src).format(taken=summary(fs), left=summary(res.left))]
    elif res.left:
        parts = [_("Nothing taken · {left} left").format(left=summary(res.left))]
    elif not fs and not_read:
        parts = [plural(not_read, "No direct file · {n} sub-folder", "No direct file · {n} sub-folders")]
        not_read = 0
    elif not fs:
        parts = [_("Empty folder: nothing to read")]
    elif not_read:
        src = "{taken} replace the previous ones" if mode == "replace" else "{taken} added"
        parts = [_(src).format(taken=summary(fs))]
    else:
        parts = [text]
    if not_read:
        parts.append(plural(not_read, "{n} sub-folder not read", "{n} sub-folders not read"))
    if res.others:
        parts.append(plural(res.others, "{n} file not a frame, ignored", "{n} files not frames, ignored"))
    if res.unreadable:
        parts.append(_("unreadable folder"))
    if ignored:
        parts.append(plural(ignored, "{n} already there, ignored", "{n} already there, ignored"))
    parts.append(elsewhere_text(res))
    return " · ".join(p for p in parts if p)


def drop_links(res: Resolution, mode, undo=True):
    """The links of a drop's toast: [(label, key)]: Undo, (Rather add), Take all when frames were left out, Include
    when sub-folders hold frames."""
    links = [(_("Undo"), "undo")] if undo else []
    if mode == "replace":
        links.append((_("Rather add"), "rather_add"))
    if res.left:
        links.append((_("Take all"), "take_all"))
    if res.subs:
        links.append((_("Include"), "include"))
    if res.elsewhere:
        links.append((_("Add anyway"), "add_anyway"))
    return links


def nothing_to_take(res: Resolution, ignored):
    """(text, links) of a drop that adds nothing: frames all left out by the filter, sub-folders only, doubles, lights
    of another session of the stack, or nothing the engine can read."""
    if res.left or res.subs:
        return drop_sentence("", res, "add", ignored, taken=[]), drop_links(res, "add", undo=False)
    if res.elsewhere:
        return elsewhere_text(res), drop_links(res, "add", undo=False)
    if ignored:
        return _("Already there: nothing added"), []
    if res.others:
        return plural(res.others, "Nothing to add: {n} file not a frame", "Nothing to add: {n} files not frames"), []
    if res.empty:
        return _("Empty folder: nothing to read"), []
    return _("Nothing to add"), []


#: the first line of a drop's toast, by file kind: (added to a session, replace a session's, added to Common,
#: replace Common's, for n boxes); each a (one, many) pair of plural()
_ADDED = {
    "lights": ("{n} light added to “{s}”", "{n} lights added to “{s}”"),
    "darks": ("{n} dark added to “{s}”", "{n} darks added to “{s}”"),
    "flats": ("{n} flat added to “{s}”", "{n} flats added to “{s}”"),
    "bias": ("{n} bias added to “{s}”", "{n} biases added to “{s}”"),
    "flat_darks": ("{n} flat dark added to “{s}”", "{n} flat darks added to “{s}”")}
_REPLACE = {
    "lights": ("{n} light replaces those of “{s}”", "{n} lights replace those of “{s}”"),
    "darks": ("{n} dark replaces those of “{s}”", "{n} darks replace those of “{s}”"),
    "flats": ("{n} flat replaces those of “{s}”", "{n} flats replace those of “{s}”"),
    "bias": ("{n} bias replaces those of “{s}”", "{n} biases replace those of “{s}”"),
    "flat_darks": ("{n} flat dark replaces those of “{s}”", "{n} flat darks replace those of “{s}”")}
_COMMON_ADDED = {
    "darks": ("{n} dark added to Common", "{n} darks added to Common"),
    "flats": ("{n} flat added to Common", "{n} flats added to Common"),
    "bias": ("{n} bias added to Common", "{n} biases added to Common"),
    "flat_darks": ("{n} flat dark added to Common", "{n} flat darks added to Common")}
_COMMON_REPLACE = {
    "darks": ("{n} dark replaces Common's", "{n} darks replace Common's"),
    "flats": ("{n} flat replaces Common's", "{n} flats replace Common's"),
    "bias": ("{n} bias replaces Common's", "{n} biases replace Common's"),
    "flat_darks": ("{n} flat dark replaces Common's", "{n} flat darks replace Common's")}
_BOXES = {
    "lights": ("{n} lights for {k} boxes", "{n} lights for {k} boxes"),
    "darks": ("{n} darks for {k} boxes", "{n} darks for {k} boxes"),
    "flats": ("{n} flats for {k} boxes", "{n} flats for {k} boxes"),
    "bias": ("{n} biases for {k} boxes", "{n} biases for {k} boxes"),
    "flat_darks": ("{n} flat darks for {k} boxes", "{n} flat darks for {k} boxes")}
def drop_toast(kind, owner_name, n, mode, n_boxes=1) -> str:
    """The first line of a drop's toast (« 214 lights added to “Ha” »): ``n`` the frames the engine will read of the
    entries stored. owner_name: the session's name, or None for the Common row."""
    if n_boxes > 1:
        return plural(n, *_BOXES[kind], k=n_boxes)
    if owner_name is None:
        pair = (_COMMON_REPLACE if mode == "replace" else _COMMON_ADDED).get(kind, _ADDED[kind])
        return plural(n, *pair, s=_("Common"))
    return plural(n, *(_REPLACE if mode == "replace" else _ADDED)[kind], s=owner_name)
