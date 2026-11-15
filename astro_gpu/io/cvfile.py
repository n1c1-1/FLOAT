"""PNG and JPEG files read and written by OpenCV through bytes, so that any path works.

``cv2.imread`` and ``cv2.imwrite`` take the path as a narrow string: on Windows a folder with a letter outside the
system code page ("Nébuleuse", "Pléiades") is not found, without an error (``imread`` gives None, ``imwrite`` False;
review 13.1). The file is read whole as bytes and decoded (``cv2.imdecode``), or encoded and its bytes written
(``cv2.imencode``), Python opening the file.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np


def read(path: str | os.PathLike, flags: int | None = None) -> np.ndarray:
    """The image of a PNG, JPEG or other file OpenCV decodes, as ``cv2.imread(path, flags)`` gives it (BGR order, its
    own depth with ``cv2.IMREAD_UNCHANGED``). OSError when the file cannot be opened or decoded."""
    import cv2
    data = np.fromfile(str(path), np.uint8)
    a = cv2.imdecode(data, cv2.IMREAD_UNCHANGED if flags is None else flags) if data.size else None
    if a is None:
        raise OSError(f"{Path(path).name}: the image cannot be decoded")
    return a


def write(path: str | os.PathLike, img: np.ndarray, params: list[int] | None = None) -> None:
    """Writes ``img`` (BGR order, as ``cv2.imwrite``) in the format of the path's suffix. OSError when it cannot be
    encoded or written."""
    import cv2
    p = Path(path)
    ok, buf = cv2.imencode(p.suffix.lower() or ".png", np.ascontiguousarray(img), params or [])
    if not ok:
        raise OSError(f"{p.name}: the image cannot be encoded as {p.suffix or 'PNG'}")
    buf.tofile(str(p))
