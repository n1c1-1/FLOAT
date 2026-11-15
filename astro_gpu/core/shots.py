"""The same shot found twice: a light file that is in two sessions (or two folders), as the same path or as a copy.

A copy is the same shot when its name, its size and the capture date of its own header are those of an earlier file
(nothing else is guessed: a file without a date of its own is never taken for another's copy). The first file in the
given order wins; the others are not stacked."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Hashable, Iterable

_DATES: dict[tuple[str, int, int], str | None] = {}


def date_of(path: Path, size: int | None = None) -> str | None:
    """A frame's own capture date, read once per file state (path, size, modification time)."""
    from ..io.frames import shot_date
    try:
        st = path.stat()
    except OSError:
        return None
    k = (str(path), st.st_size, st.st_mtime_ns)
    if k not in _DATES:
        if len(_DATES) > 50000:
            _DATES.clear()
        _DATES[k] = shot_date(path)
    return _DATES[k]


def repeated(groups: Iterable[tuple[Hashable, list[tuple[Path, int]]]],
             date: Callable[[Path], str | None] | None = None) -> dict[tuple[Any, Path], tuple[Any, Path]]:
    """{(owner, file): (first owner, first file)} of each file that is the same shot as an earlier one, from the ordered
    ``(owner, [(file, size)])`` groups. Headers are read only for files sharing a name and a size."""
    date = date or date_of
    seen_path: dict[str, tuple[Any, Path]] = {}
    seen_key: dict[tuple[str, int], list[tuple[Any, Path]]] = {}
    out: dict[tuple[Any, Path], tuple[Any, Path]] = {}
    for owner, files in groups:
        for f, size in files:
            pk = os.path.normcase(os.path.normpath(str(f)))
            if pk in seen_path:                      # the very same file again
                out[(owner, f)] = seen_path[pk]
                continue
            first = None
            earlier = seen_key.get((f.name, size))
            if earlier:
                d = date(f)
                if d is not None:
                    first = next((e for e in earlier if date(e[1]) == d), None)
            if first is not None:
                out[(owner, f)] = first
                seen_path[pk] = first
                continue
            seen_path[pk] = (owner, f)
            seen_key.setdefault((f.name, size), []).append((owner, f))
    return out
