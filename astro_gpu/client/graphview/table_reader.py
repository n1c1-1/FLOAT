"""The background listing of the folders and files dropped in the table or on a Session: names, sizes and modification
times only (os.scandir; no file is opened, nothing is read from a header, nothing is guessed from a name).

``Reader`` is a cache filled by one worker thread, never part of the model nor of the undo history. The interface
asks ``request(key, sources)`` and reads ``count``, ``expand``, ``left_out``, ``sub_folders``, ``unreadable`` at once
(what is known so far); listeners are told of the progress from the worker thread (connect them to a Qt signal).
A folder entry is listed as the engine reads it (core/project.py ``list_frames``): its first level only, its frames,
the colour images set aside when it holds raw or FITS frames (``left_out``).
"""
from __future__ import annotations

import os
import threading
import time
from collections import namedtuple
from dataclasses import dataclass, field

from ...core.project import frame_kind

#: one listed file: its path (forward slashes), size in bytes, modification time, the folder entry it came from
FileInfo = namedtuple("FileInfo", "path size mtime root")


def list_folder(path: str, counter=None, stop=None, accept=None, left_out=None, report=None,
                recursive=True) -> list[FileInfo]:
    """Listing of a folder (scandir only: names, sizes, times). ``accept(name) -> bool`` is the file-type filter
    (default: every file); the files it refuses go to ``left_out``; ``report`` gets the number of sub-folders met
    ("sub") and "unreadable" when the folder itself cannot be read. Depth first, alphabetical: the same folder always
    lists the same."""
    out: list[FileInfo] = []
    todo = [path]
    root = path.replace("\\", "/")
    ok = accept or (lambda name: True)
    while todo:
        if stop and stop():
            break
        d = todo.pop()
        try:
            with os.scandir(d) as it:
                entries = sorted(it, key=lambda e: e.name)
        except OSError:
            if report is not None and d == path:
                report["unreadable"] = True
            continue
        subs = []
        for e in entries:
            try:
                if e.is_dir(follow_symlinks=False):
                    if recursive:
                        subs.append(e.path)
                    if report is not None:
                        report["sub"] = report.get("sub", 0) + 1
                else:
                    st = e.stat()
                    f = FileInfo(e.path.replace("\\", "/"), st.st_size, st.st_mtime, root)
                    if ok(e.name):
                        out.append(f)
                        if counter:
                            counter(len(out))
                    elif left_out is not None:
                        left_out.append(f)
            except OSError:
                continue
        todo.extend(reversed(subs))
    return out


@dataclass
class Job:
    key: object
    sources: tuple
    t0: float = field(default_factory=time.perf_counter)
    phase: str = "list"            # list | done | stopped
    listed: int = 0
    stop: bool = False

    @property
    def active(self):
        return self.phase == "list"

    def seconds(self):
        return time.perf_counter() - self.t0


class Reader:
    """The listings of the dropped entries, by one worker thread."""

    def __init__(self, delay: float = 0.0):
        self.delay = delay
        # the listings are kept by where an entry is on the disk (``_key``: the entry and its place), so that another
        # project naming the same relative entry, or the same project moved, never gets them
        self._folders: dict[tuple, list[FileInfo]] = {}
        self._is_folder: dict[tuple, bool] = {}         # entry -> is a folder (noted at drop time)
        self._infos: dict[tuple, FileInfo] = {}
        self._mtimes: dict[tuple, float] = {}          # folder -> its modification time when it was listed
        self._jobs: dict[object, Job] = {}
        self._queue: list[Job] = []
        self._cond = threading.Condition()
        self._thread = None
        self._listeners: list = []
        self._gen = 0
        self._cache: dict[tuple, tuple] = {}
        self._busy = False
        self._left: dict[tuple, list] = {}              # folder -> files of its level the engine does not read
        self._reports: dict[tuple, dict] = {}           # folder -> {"sub": n, "unreadable": bool}
        self.base = None                                # () -> the project's folder: where a relative entry is read

    def on_disk(self, s: str) -> str:
        """Where an entry is on the disk: one written relative to the project is read from the project's folder
        (a project that holds its own frames), never from the folder the application happens to run in. Every
        place of the page that looks at an entry on the disk asks here."""
        if os.path.isabs(s) or self.base is None:
            return s
        root = self.base()
        return os.path.join(str(root), s) if root else s

    def _key(self, s: str) -> tuple:
        """What a listing is kept under: the entry as written and where it is on the disk."""
        return s, os.path.normcase(os.path.abspath(self.on_disk(s)))

    def forget(self):
        """Every listing dropped (another project in the page): the running ones are stopped."""
        for t in list(self._jobs.values()):
            t.stop = True
        with self._cond:
            self._queue.clear()
        self._jobs.clear()
        for d in (self._folders, self._is_folder, self._infos, self._mtimes, self._left, self._reports, self._cache):
            d.clear()
        self._gen += 1

    def _changed(self, s: str) -> bool:
        """A folder listed before whose modification time moved since (frames added or taken out): its listing is
        dropped, to be made again."""
        k = self._key(s)
        if k not in self._folders or k not in self._mtimes:
            return False
        try:
            now = os.stat(self.on_disk(s)).st_mtime
        except OSError:
            now = None
        if now == self._mtimes[k]:
            return False
        for f in self._folders.pop(k):
            self._infos.pop(self._key(f.path), None)
        for d in (self._mtimes, self._left, self._reports):
            d.pop(k, None)
        self._gen += 1
        return True

    @staticmethod
    def _as_entry(entry: str, infos) -> list:
        """Files listed in a folder entry written relative to the project, written the same way (the entry's first
        level): a « Remove » that writes them back keeps the project movable."""
        root = entry.replace("\\", "/").rstrip("/")
        return [f._replace(path=f"{root}/{os.path.basename(f.path)}", root=root) for f in infos]

    # --------------------------------------------------------------- called from the interface thread
    def listen(self, fn):
        """``fn(job)`` is called from the worker thread on progress (at most 10 times a second) and at the end."""
        if fn not in self._listeners:
            self._listeners.append(fn)

    def unlisten(self, fn):
        self._listeners = [x for x in self._listeners if x != fn]

    def note(self, path: str, is_folder: bool):
        self._is_folder[self._key(path)] = is_folder

    def known(self, path: str):
        """True / False when the path was noted as a folder / a file, None when never seen."""
        return self._is_folder.get(self._key(path))

    def job(self, key) -> Job | None:
        return self._jobs.get(key)

    def active_jobs(self):
        return [t for t in list(self._jobs.values()) if t.active]

    def request(self, key, sources) -> Job | None:
        """List the entries of a box (cheap when nothing changed): -> its Job, or None when everything is known."""
        sources = tuple(sources)
        t = self._jobs.get(key)
        if not (t and t.active):
            for s in sources:
                self._changed(s)                        # a folder that moved since its listing: listed again
        fresh = self.expand(sources) is not None
        if t and t.sources == sources and (t.active or (t.phase == "done" and fresh)):
            return t
        if t:
            t.stop = True
        if fresh:
            self._jobs.pop(key, None)
            return None
        t = Job(key, sources)
        self._jobs[key] = t
        with self._cond:
            self._queue.append(t)
            if self._thread is None:
                self._thread = threading.Thread(target=self._loop, daemon=True, name="graphview-reader")
                self._thread.start()
            self._cond.notify()
        return t

    def stop(self, key):
        t = self._jobs.get(key)
        if t and t.active:
            t.stop = True

    def count(self, sources) -> tuple[int, bool]:
        """(files known so far, complete?) at once from what was noted at the drop."""
        n, ok = 0, True
        for s in sources:
            k = self._key(s)
            if k in self._folders:
                n += len(self._folders[k])
            elif self._is_folder.get(k) is False or k in self._infos:
                n += 1
            else:
                ok = False
        return n, ok

    def expand(self, sources) -> list[FileInfo] | None:
        """Every file of the entries, or None while one of them is not listed yet."""
        ks = [self._key(s) for s in sources]
        key = (tuple(ks), self._gen)
        hit = self._cache.get(key)
        if hit is not None:
            return hit[0]
        out = []
        for k in ks:
            if k in self._folders:
                out.extend(self._folders[k])
            elif k in self._infos:
                out.append(self._infos[k])
            else:
                out = None
                break
        if len(self._cache) > 64:
            self._cache.clear()
        self._cache[key] = (out,)
        return out

    def left_out(self, sources) -> list:
        """Files of the (listed) folders' level that the engine does not read (not frames, colour beside raws)."""
        return [f for s in sources for f in self._left.get(self._key(s), [])]

    def sub_folders(self, sources) -> int:
        return sum(self._reports.get(self._key(s), {}).get("sub", 0) for s in sources)

    def unreadable(self, sources) -> list:
        return [s for s in sources if self._reports.get(self._key(s), {}).get("unreadable")]

    def adopt(self, infos):
        """Files already listed (left out by the filter, then taken): known as single files from now on."""
        for f in infos:
            k = self._key(f.path)
            self._infos[k] = f
            self._is_folder[k] = False
        self._gen += 1

    def wait(self, timeout=30.0) -> bool:
        """Block until no listing is running (tests)."""
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < timeout:
            if not any(t.active for t in list(self._jobs.values())) and not self._queue and not self._busy:
                return True
            time.sleep(0.005)
        return False

    # --------------------------------------------------------------- the worker thread
    def _signal(self, t: Job, force=False):
        now = time.perf_counter()
        if force or now - getattr(t, "_last", 0) >= 0.1:
            t._last = now
            for fn in list(self._listeners):
                try:
                    fn(t)
                except RuntimeError:                      # a listener's Qt object is gone
                    self._listeners = [x for x in self._listeners if x is not fn]

    def _loop(self):
        while True:
            with self._cond:
                while not self._queue:
                    self._cond.wait()
                t = self._queue.pop(0)
                self._busy = True
            try:
                self._run(t)
            except Exception:                                   # noqa: BLE001 - the reader never kills the window
                t.phase = "stopped"
            finally:
                self._busy = False
                self._signal(t, True)

    def _run(self, t: Job):
        base_n = 0
        for s in t.sources:
            if t.stop:
                break
            k = self._key(s)
            if k in self._folders:
                base_n += len(self._folders[k])
                continue
            if k in self._infos and k not in self._folders:
                base_n += 1
                continue
            path = self.on_disk(s)
            is_folder = self._is_folder.get(k)
            if is_folder is None:
                is_folder = os.path.isdir(path)
            if is_folder:
                try:
                    mtime = os.stat(path).st_mtime        # taken before the listing: a frame written meanwhile
                except OSError:                           # moves it again
                    mtime = None
                def counter(n, b=base_n):
                    t.listed = b + n
                    self._signal(t)
                left, report = [], {}
                fs = list_folder(path, counter, lambda: t.stop, lambda name: bool(frame_kind(name)), left, report,
                                 recursive=False)
                if t.stop:
                    break
                if path != s:                     # a relative folder: its files written as the project names them
                    fs, left = self._as_entry(s, fs), self._as_entry(s, left)
                if any(frame_kind(f.path) != "colour" for f in fs):   # the engine's rule: colour beside raws aside
                    left += [f for f in fs if frame_kind(f.path) == "colour"]
                    fs = [f for f in fs if frame_kind(f.path) != "colour"]
                self._left[k], self._reports[k], self._mtimes[k] = left, report, mtime
                for f in fs:
                    self._infos[self._key(f.path)] = f
                self._folders[k] = fs
                base_n += len(fs)
                if self.delay:
                    time.sleep(self.delay)
            else:
                try:
                    st = os.stat(path)
                    f = FileInfo(s.replace("\\", "/"), st.st_size, st.st_mtime, os.path.dirname(s).replace("\\", "/"))
                except OSError:
                    f = FileInfo(s.replace("\\", "/"), 0, 0.0, os.path.dirname(s).replace("\\", "/"))
                self._infos[k] = f
                base_n += 1
                t.listed = base_n
                self._signal(t)
            self._gen += 1
        t.listed = base_n
        t.phase = "stopped" if t.stop else "done"
