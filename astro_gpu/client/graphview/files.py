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
    new = [p for p in res.items if not already(p, existing)]
    ignored, res.items = len(res.items) - len(new), new
    res.subs = [x for x in res.subs if x not in existing]
    res.left = [x for x in res.left if x not in existing]       # left out by the filter, but taken before
    return ignored


def normalise(p: str) -> str:
    p = str(p).replace("\\", "/")
    return p.rstrip("/") if len(p) > 3 else p


def note_paths(reader, paths) -> tuple[int, int]:
    """Tell the reader which dropped paths are folders (a stat each, immediate). -> (folders, files)."""
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
    others: int = 0
    subs: list = field(default_factory=list)
    unreadable: list = field(default_factory=list)
    empty: list = field(default_factory=list)

    def frames(self) -> list[str]:
        return engine_frames(self.items)


def resolve(reader, paths, i: int | None = None, deep: bool = True) -> Resolution:
    """A drop resolved now: each folder's first level read (scandir, names only), the filter applied, the engine's
    rule compared (see the module's text). ``deep`` False: the sub-folders are not looked for (« Include » itself)."""
    i = index() if i is None else i
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
    return links


def nothing_to_take(res: Resolution, ignored):
    """(text, links) of a drop that adds nothing: frames all left out by the filter, sub-folders only, doubles, or
    nothing the engine can read."""
    if res.left or res.subs:
        return drop_sentence("", res, "add", ignored, taken=[]), drop_links(res, "add", undo=False)
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
