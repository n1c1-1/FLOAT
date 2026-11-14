"""The file-type filter of the imports, and the logic of a drop of files and folders, shared by the table and by the
graph's Session targets (graph_drops.py).

Matching is by EXTENSION only, case-insensitive: nothing else of a name is read, nothing is sorted or proposed from a
name, a folder or a date. The filter chosen last is remembered in the client's settings (prefs, "graph_file_filter")
and is the one applied to drops. A dropped folder gives only its own files; « Include » reads all its levels (its
entry then ends with ``MARK``). Each folder entry keeps the filter it was taken with in the document's view
(``view["file_options"]``), so that it reopens with the same files whatever the filter set later.
"""
from __future__ import annotations

import os
from collections import Counter

from .. import prefs
from ..i18n import _, plural

CAMERA = ("arw", "cr2", "cr3", "nef", "raf", "orf", "rw2", "dng", "pef")
FITS = ("fit", "fits", "fts")
IMAGES = CAMERA + FITS + ("tif", "tiff", "jpg", "jpeg", "png")
#: (English name, extensions | None = everything), in the order of the dialog and of the combo
TYPES = [("Camera raws", CAMERA), ("FITS", FITS), ("Raws and FITS", CAMERA + FITS), ("All images", IMAGES),
         ("All files", None)]
DEFAULT = 2
PREF = "graph_file_filter"
RECURSIVE_DEFAULT = False         # a dropped folder gives only its own files; « Include » reads all levels
MARK = "/**"                      # a folder entry read at all levels ends with this (the core's lists take plain paths)
TIP = "Only the files of this type (by their extension) are taken at import; the others are left out."
OPTIONS = "file_options"          # view key: {"owner|kind|entry": [filter name, all levels]}


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
    exts = TYPES[index() if i is None else i][1]
    return exts is None or extension(path) in exts


def base(source: str):
    """(path, all levels?) of an entry."""
    if source.endswith(MARK):
        return source[:-len(MARK)], True
    return source, False


def describe(source: str, is_folder: bool, name: str) -> str:
    """The tooltip line of an entry: "folder · Raws and FITS · without sub-folders"."""
    if not is_folder:
        return _("file")
    levels = _("all sub-folders") if base(source)[1] else _("without sub-folders")
    return _("folder · {filter} · {levels}").format(filter=_(name), levels=levels)


def dialog_filters() -> list[str]:
    """The file dialog's name filters, in order."""
    out = []
    for name, exts in TYPES:
        pats = " ".join(f"*.{e} *.{e.upper()}" for e in exts) if exts else "*"
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
    return os.path.isdir(base(p)[0]) if d is None else d


def already(p: str, existing) -> bool:
    """No silent doubles: the same folder (already read at least as deep), or a path inside a folder already read at
    all levels, is not added again."""
    bp, rp = base(p)
    for e in existing:
        be, re_ = base(e)
        if (bp == be and (re_ or not rp)) or (re_ and bp.startswith(be.rstrip("/") + "/")):
            return True
    return False


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


def split_loose(reader, paths):
    """Folders go as they are (listed in the background, with the type filter); loose files are filtered at once by
    their extension. -> (entries taken, loose files left out)."""
    items, left, i = [], [], index()
    for p in paths:
        p = normalise(p)
        if is_folder(reader, p):
            items.append(p + MARK if RECURSIVE_DEFAULT else p)
            reader.note(items[-1], True)
            reader.opts[items[-1]] = i
        elif accepts(p, i):
            items.append(p)
            reader.note(p, False)
        else:
            left.append(p)
    return items, left


def drop_sentence(reader, base_text, items, loose, mode, ignored=0):
    """The toast of a drop: the usual one, or (once listed) what was taken and left out, the sub-folders not read,
    doubles ignored. A function of the reader's state: it follows the listing."""
    def fn():
        text = base_text() if callable(base_text) else base_text
        fs = reader.expand(tuple(items))
        if fs is None:
            return text
        dirs = [x for x in items if is_folder(reader, x)]
        not_read = reader.sub_folders([x for x in dirs if not base(x)[1]])
        included = reader.sub_folders([x for x in dirs if base(x)[1]])
        left = [f.path for f in reader.left_out(items)] + list(loose)
        if left and fs:
            src = ("{taken} replace the previous ones · {left} left" if mode == "replace" else
                   "{taken} added · {left} left")
            parts = [_(src).format(taken=summary([f.path for f in fs]), left=summary(left))]
        elif left:
            parts = [_("Nothing taken · {left} left").format(left=summary(left))]
        elif not fs and not_read:
            parts = [plural(not_read, "No direct file · {n} sub-folder", "No direct file · {n} sub-folders")]
            not_read = 0
        elif not fs:
            parts = [_("Empty folder: nothing to read")]
        elif not_read:
            src = "{taken} replace the previous ones" if mode == "replace" else "{taken} added"
            parts = [_(src).format(taken=summary([f.path for f in fs]))]
        else:
            parts = [text]
        lost = sum(1 for f in fs if f.mtime == 0.0 and f.size == 0)
        if lost:
            parts.append(plural(lost, "{n} not found", "{n} not found"))
        if not_read:
            parts.append(plural(not_read, "{n} sub-folder not read", "{n} sub-folders not read"))
        if included:
            parts.append(plural(included, "{n} sub-folder included", "{n} sub-folders included"))
        if reader.unreadable(items):
            parts.append(_("unreadable folder"))
        if ignored:
            parts.append(plural(ignored, "{n} already there, ignored", "{n} already there, ignored"))
        return " · ".join(p for p in parts if p)
    return fn


def drop_links(reader, items, loose, mode):
    """The links of a drop's toast: [(label, key)]: Undo, (Rather add), Take all when files were left out, Include
    when sub-folders were not read. A function: it follows the listing."""
    def f():
        links = [(_("Undo"), "undo")]
        if mode == "replace":
            links.append((_("Rather add"), "rather_add"))
        if reader.left_out(items) or loose:
            links.append((_("Take all"), "take_all"))
        if reader.sub_folders([x for x in items if is_folder(reader, x) and not base(x)[1]]):
            links.append((_("Include"), "include"))
        return links
    return f


def empty_folders(reader, items) -> list:
    """The folder entries whose listing is complete and holds NO direct matching file: they leave no empty entry."""
    return [x for x in items if is_folder(reader, x) and reader.expand((x,)) == []]


def save_options(reader, view: dict, owner, kind, items, name=None):
    """Keep in the document's view the filter and the all-levels choice each dropped folder was taken with."""
    rec = view.setdefault(OPTIONS, {})
    for it in items:
        if is_folder(reader, it):
            rec[f"{owner}|{kind}|{it}"] = [name or filter_name(reader.opts.get(it, DEFAULT)), base(it)[1]]


def option_name(reader, view: dict, owner, kind, entry) -> str:
    r = view.get(OPTIONS, {}).get(f"{owner}|{kind}|{entry}")
    return r[0] if r else filter_name(reader.opts.get(entry, DEFAULT))


def restore_options(reader, view: dict, owner, kind, src):
    """Before a list is asked of the reader: each folder entry is read with the filter it was taken with."""
    for it in src:
        r = view.get(OPTIONS, {}).get(f"{owner}|{kind}|{it}")
        if r:
            reader.opts[it] = index_of_name(r[0])


def nothing_to_take(loose, ignored):
    """(text, links) of a drop that adds nothing: files all left out by the filter, doubles, or nothing at all."""
    if loose:
        return _("Nothing taken · {left} left").format(left=summary(loose)), [(_("Take all"), "take_all")]
    return (_("Already there: nothing added") if ignored else _("Nothing to add")), []


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
#: the file kinds, lower case, for the sentences said while a folder is still being listed
KIND_WORD = {"lights": "lights", "darks": "darks", "flats": "flats", "bias": "bias", "flat_darks": "flat darks"}


def drop_toast(reader, kind, owner_name, paths, mode, n_boxes=1):
    """The first line of a drop's toast (« 214 lights added to “Ha” »), a function of what the reader knows so far.
    owner_name: the session's name, or None for the Common row."""
    def fn():
        n, ok = reader.count(paths)
        if not ok:
            nd = sum(1 for p in paths if is_folder(reader, p))
            if nd:
                what = plural(nd, "{n} folder ({kind})", "{n} folders ({kind})", kind=_(KIND_WORD[kind]))
            else:
                what = plural(len(paths), "{n} file ({kind})", "{n} files ({kind})", kind=_(KIND_WORD[kind]))
            if n_boxes > 1:
                return _("{what} for {k} boxes").format(what=what, k=n_boxes)
            if owner_name is None:
                return _("{what} for Common").format(what=what)
            return _("{what} for “{s}”").format(what=what, s=owner_name)
        if n_boxes > 1:
            return plural(n, *_BOXES[kind], k=n_boxes)
        if owner_name is None:
            pair = (_COMMON_REPLACE if mode == "replace" else _COMMON_ADDED).get(kind, _ADDED[kind])
            return plural(n, *pair, s=_("Common"))
        return plural(n, *(_REPLACE if mode == "replace" else _ADDED)[kind], s=owner_name)
    return fn
