"""The viewer's loading and display: the loader thread, one image on screen in tiles, its name tag."""
from __future__ import annotations

import heapq
import itertools
import math
import threading
import time
from collections import OrderedDict

import numpy as np
from PySide6.QtCore import QObject, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter
from PySide6.QtWidgets import QApplication

from . import stretch as sm
from .icons import draw
from .imageview_images import TILE, CANVAS_RGB, ViewImage, Alignment, _UID, RTILES, Prepared, prepare, thumbnail_of


# ---------------------------------------------------------------------------------------------- the loader
class Unavailable(Exception):
    """Raised by a source that knows why its image is not there: the text (already in the language of the
    interface) is what the view shows, alone."""


class Loader(QObject):
    """Prepares images in worker threads, the most urgent first, and keeps the last ones (least recently used out,
    within a byte budget). Answers come as signals in the GUI thread. One loader is shared by default
    (``default_loader``): the same stack shown in two views is read once."""

    ready = Signal(object, object)            # key, Prepared
    failed = Signal(object, str)              # key, reason
    thumbReady = Signal(object, object)       # key, (h, w, 3) uint8

    def __init__(self, budget: int = 3 * 2 ** 30, threads: int = 2, parent=None):
        super().__init__(parent)
        self.budget = budget
        self._cache: OrderedDict = OrderedDict()
        self._bytes = 0
        self._cond = threading.Condition()
        self._heap: list = []
        self._queued: dict = {}                    # (kind, key) -> (priority, seq)
        self._running: set = set()
        self._seq = itertools.count()
        self._stop = False
        self.thumbs: dict = {}                     # key -> thumbnail
        self.unavailable: set = set()              # keys whose source raised Unavailable (its text is the message)
        self.loads = 0                             # images prepared (the tests count them)
        self.load_threads: set[int] = set()        # the threads that did it
        self._threads = [threading.Thread(target=self._run, daemon=True, name=f"imageview-loader-{i}")
                         for i in range(threads)]
        for t in self._threads:
            t.start()

    # ------------------------------------------------------------ GUI thread
    def cached(self, key) -> Prepared | None:
        with self._cond:
            p = self._cache.get(key)
            if p is not None:
                self._cache.move_to_end(key)
            return p

    def request(self, vi: ViewImage, priority: int = 0):
        """Prepare ``vi`` (0: now, 1, 2...: ahead of time). Answer: ``ready`` or ``failed``; nothing when cached."""
        self._submit("load", vi, priority)

    def request_thumb(self, vi: ViewImage, priority: int = 10):
        key = vi.identity()
        if key in self.thumbs:
            return
        if isinstance(vi.thumb, np.ndarray):
            self.thumbs[key] = np.ascontiguousarray(vi.thumb[:, :, :3]) if vi.thumb.ndim == 3 else vi.thumb
            self.thumbReady.emit(key, self.thumbs[key])
            return
        if callable(vi.thumb):
            self._submit("thumb", vi, priority)

    def forget_requests(self, keep=()):
        """Drop the waiting requests made ahead of time (priority above 0), but those of the keys in ``keep``."""
        keep = set(keep)
        with self._cond:
            for (kind, key), (pr, _s) in list(self._queued.items()):
                if pr > 0 and key not in keep and kind == "load":
                    del self._queued[(kind, key)]

    def busy(self) -> bool:
        with self._cond:
            return bool(self._queued or self._running)

    def wait(self, timeout: float = 60.0) -> bool:
        """Process events until every request is answered (tests and scripted walkthroughs)."""
        end = time.monotonic() + timeout
        app = QApplication.instance()
        while time.monotonic() < end:
            app.processEvents()
            if not self.busy():
                app.processEvents()
                return True
            time.sleep(0.003)
        return False

    def clear(self):
        with self._cond:
            for p in self._cache.values():
                p.release()
            self._cache.clear()
            self._bytes = 0
            self.thumbs.clear()

    def stop(self):
        with self._cond:
            self._stop = True
            self._cond.notify_all()

    def _submit(self, kind: str, vi: ViewImage, priority: int):
        key = vi.identity()
        with self._cond:
            if kind == "load" and key in self._cache:
                return
            if (kind, key) in self._running:
                return
            old = self._queued.get((kind, key))
            if old is not None and old[0] <= priority:
                return
            seq = next(self._seq)
            self._queued[(kind, key)] = (priority, seq)
            heapq.heappush(self._heap, (priority, seq, kind, key, vi))
            self._cond.notify()

    # ------------------------------------------------------------ worker threads
    def _next(self):
        with self._cond:
            while True:
                if self._stop:
                    return None
                while self._heap:
                    pr, seq, kind, key, vi = heapq.heappop(self._heap)
                    if self._queued.get((kind, key)) == (pr, seq):
                        del self._queued[(kind, key)]
                        self._running.add((kind, key))
                        return kind, key, vi
                self._cond.wait()

    def _run(self):
        while True:
            job = self._next()
            if job is None:
                return
            kind, key, vi = job
            try:
                if kind == "load":
                    prep = prepare(vi)
                    self.load_threads.add(threading.get_ident())
                    self.loads += 1
                    self._keep(key, prep)
                    self._emit(self.ready, key, prep)
                    if key not in self.thumbs and vi.thumb is None:
                        th = thumbnail_of(prep)
                        self.thumbs[key] = th
                        self._emit(self.thumbReady, key, th)
                else:
                    th = np.ascontiguousarray(np.asarray(vi.thumb())[:, :, :3])
                    self.thumbs[key] = th
                    self._emit(self.thumbReady, key, th)
            except Exception as e:  # noqa: BLE001 - every failure goes to the view as text
                if kind == "load":
                    if isinstance(e, Unavailable):          # the caller's own words (translated), shown alone
                        self.unavailable.add(key)
                        self._emit(self.failed, key, str(e))
                    else:
                        self.unavailable.discard(key)
                        self._emit(self.failed, key, f"{type(e).__name__}: {e}")
            finally:
                with self._cond:
                    self._running.discard((kind, key))

    def _keep(self, key, prep: Prepared):
        with self._cond:
            old = self._cache.pop(key, None)
            if old is not None:
                self._bytes -= old.nbytes
            self._cache[key] = prep
            self._bytes += prep.nbytes
            while self._bytes > self.budget and len(self._cache) > 1:
                _k, p = self._cache.popitem(last=False)
                self._bytes -= p.nbytes
                p.release()

    @staticmethod
    def _emit(sig, *args):
        try:
            sig.emit(*args)
        except RuntimeError:                          # the loader went with the application
            pass


# ---------------------------------------------------------------------------------------------- one image on screen
class Display:
    """A prepared image shown through one stretch: the rendered tiles and the picture of the visible part."""

    def __init__(self, prep: Prepared, st: sm.Stretch):
        self.uid = next(_UID)
        self.prep = prep
        self.version = 0
        self._comp = None
        self._thumb = None
        self.align: Alignment | None = None          # drawn on another image's grid (a compared image)
        self.frame: tuple[int, int] | None = None    # then the size of that image: the area it is drawn on
        self.stretch = st
        self.set_stretch(st)

    def set_alignment(self, al: Alignment | None, frame: tuple[int, int] | None = None):
        """Draw this image on the grid of another one (``al.T`` maps that grid onto this image), ``frame`` its size;
        None: on its own grid."""
        self.align = al
        self.frame = (int(frame[0]), int(frame[1])) if al is not None and frame is not None else None
        if self.frame is None:
            self.align = None
        self._comp = None
        self._thumb = None

    def size(self) -> tuple[int, int]:
        """The size of the area it is drawn on: its own, or the grid it is aligned on."""
        return self.frame if self.frame is not None else self.prep.size()

    def set_stretch(self, st: sm.Stretch):
        st = st.adapted(self.prep.channels)
        old = self.version
        self.stretch = st
        self.lut = sm.luts(st, self.prep.stats)
        self.version += 1
        self._comp = None
        self._thumb = None
        RTILES.drop(lambda k, u=self.uid, v=old: k[0] == u and k[1] <= v)

    def level_for(self, s_dev: float) -> int:
        """The pyramid level nearest the screen's resolution at ``s_dev`` device pixels per image pixel."""
        v = 1.0 / max(s_dev * self.prep.factor, 1e-9)
        k = int(round(math.log2(v))) if v > 1.0 else 0
        return max(0, min(k, len(self.prep.levels) - 1))

    def _tile(self, k: int, tx: int, ty: int) -> np.ndarray:
        key = (self.uid, self.version, k, tx, ty)
        t = RTILES.get(key)
        if t is None:
            t = sm.render(self.prep.qtile(k, tx, ty), self.lut)
            RTILES.put(key, t)
        return t

    def composite(self, k: int, tx0: int, ty0: int, tx1: int, ty1: int) -> np.ndarray:
        """The tiles [tx0, tx1) x [ty0, ty1) of level ``k`` side by side (made in parallel threads)."""
        key = (k, tx0, ty0, tx1, ty1, self.version)
        if self._comp is not None and self._comp[0] == key:
            return self._comp[1]
        coords = [(tx, ty) for ty in range(ty0, ty1) for tx in range(tx0, tx1)]
        missing = [c for c in coords if RTILES.get((self.uid, self.version, k) + c) is None]
        if len(missing) > 1:
            list(sm.pool("render").map(lambda c: self._tile(k, *c), missing))
        w, h = self.prep.level_size(k)
        x1, y1 = min(w, tx1 * TILE), min(h, ty1 * TILE)
        shape = (y1 - ty0 * TILE, x1 - tx0 * TILE) + ((3,) if self.prep.channels == 3 else ())
        out = np.empty(shape, np.uint8)
        for tx, ty in coords:
            t = self._tile(k, tx, ty)
            oy, ox = (ty - ty0) * TILE, (tx - tx0) * TILE
            out[oy:oy + t.shape[0], ox:ox + t.shape[1]] = t
        self._comp = (key, out)
        return out

    def view(self, s_dev: float, cx: float, cy: float, W: int, H: int, clip: tuple[int, int] | None = None):
        """The visible part of the image in device pixels: (picture, u0, v0), its top-left corner in the area, or
        None. ``s_dev``: device pixels per image pixel; (cx, cy): the image point at the centre of the W x H area;
        ``clip``: only the columns [u0, u1)."""
        if self.align is not None:
            return self._view_aligned(s_dev, cx, cy, W, H, clip)
        import cv2
        p = self.prep
        k = self.level_for(s_dev)
        F = p.factor * (1 << k)
        lw, lh = p.level_size(k)
        ex0, ex1 = W / 2 - cx * s_dev, W / 2 + (lw * F - cx) * s_dev
        ey0, ey1 = H / 2 - cy * s_dev, H / 2 + (lh * F - cy) * s_dev
        u0, u1 = max(0, math.floor(ex0)), min(W, math.ceil(ex1))
        v0, v1 = max(0, math.floor(ey0)), min(H, math.ceil(ey1))
        if clip is not None:
            u0, u1 = max(u0, clip[0]), min(u1, clip[1])
        if u1 <= u0 or v1 <= v0:
            return None
        lx0 = ((u0 - W / 2) / s_dev + cx) / F
        lx1 = ((u1 - W / 2) / s_dev + cx) / F
        ly0 = ((v0 - H / 2) / s_dev + cy) / F
        ly1 = ((v1 - H / 2) / s_dev + cy) / F
        ntx, nty = p.tiles_of(k)
        tx0 = max(0, int(math.floor(lx0 - 1)) // TILE)
        tx1 = min(ntx, int(math.ceil(lx1 + 1)) // TILE + 1)
        ty0 = max(0, int(math.floor(ly0 - 1)) // TILE)
        ty1 = min(nty, int(math.ceil(ly1 + 1)) // TILE + 1)
        comp = self.composite(k, tx0, ty0, tx1, ty1)
        s_lvl = s_dev * F
        bx = ((u0 + 0.5 - W / 2) / s_dev + cx) / F - 0.5 - tx0 * TILE
        by = ((v0 + 0.5 - H / 2) / s_dev + cy) / F - 0.5 - ty0 * TILE
        m = np.array([[1.0 / s_lvl, 0.0, bx], [0.0, 1.0 / s_lvl, by]])
        interp = cv2.INTER_NEAREST if s_lvl >= 0.9999 else cv2.INTER_LINEAR
        buf = cv2.warpAffine(comp, m, (u1 - u0, v1 - v0), flags=interp | cv2.WARP_INVERSE_MAP,
                             borderMode=cv2.BORDER_REPLICATE)
        return buf, u0, v0

    def _view_aligned(self, s_dev: float, cx: float, cy: float, W: int, H: int, clip):
        """``view`` for an image drawn on another one's grid (``align``): (cx, cy) and the area are that grid's; each
        screen pixel takes this image's pixel ``align.T`` maps it on (bilinear below 1:1, display only). Where this
        image has no pixel the canvas's background shows."""
        import cv2
        p = self.prep
        fw, fh = self.frame
        T = self.align.T
        ex0, ex1 = W / 2 - cx * s_dev, W / 2 + (fw - cx) * s_dev
        ey0, ey1 = H / 2 - cy * s_dev, H / 2 + (fh - cy) * s_dev
        u0, u1 = max(0, math.floor(ex0)), min(W, math.ceil(ex1))
        v0, v1 = max(0, math.floor(ey0)), min(H, math.ceil(ey1))
        if clip is not None:
            u0, u1 = max(u0, clip[0]), min(u1, clip[1])
        if u1 <= u0 or v1 <= v0:
            return None
        lin = self.align.scale                                   # this image's pixels per pixel of the grid
        k = self.level_for(s_dev / lin)
        F = p.factor * (1 << k)
        ax = (u0 + 0.5 - W / 2) / s_dev + cx                     # the grid point of the first screen pixel
        ay = (v0 + 0.5 - H / 2) / s_dev + cy
        # screen pixel (u, v) -> grid (ax + u / s, ay + v / s) -> this image (T) -> its level-k pixel index
        m = np.array([[T[0, 0] / (s_dev * F), T[0, 1] / (s_dev * F), (T[0, 0] * ax + T[0, 1] * ay + T[0, 2]) / F - 0.5],
                      [T[1, 0] / (s_dev * F), T[1, 1] / (s_dev * F), (T[1, 0] * ax + T[1, 1] * ay + T[1, 2]) / F - 0.5]])
        du, dv = u1 - u0, v1 - v0
        corners = m @ np.array([[0, du, 0, du], [0, 0, dv, dv], [1, 1, 1, 1]], float)
        field = self.align.field
        pad = 1.0 + (max(float(np.abs(field[1]).max()), float(np.abs(field[2]).max())) / F if field is not None
                     else 0.0)
        lw, lh = p.level_size(k)
        ntx, nty = p.tiles_of(k)
        tx0 = max(0, int(math.floor(corners[0].min() - pad)) // TILE)
        tx1 = min(ntx, int(math.ceil(corners[0].max() + pad)) // TILE + 1)
        ty0 = max(0, int(math.floor(corners[1].min() - pad)) // TILE)
        ty1 = min(nty, int(math.ceil(corners[1].max() + pad)) // TILE + 1)
        if tx1 <= tx0 or ty1 <= ty0 or corners[0].max() < -1 or corners[1].max() < -1 or \
                corners[0].min() > lw or corners[1].min() > lh:
            bg = np.empty((dv, du, 3) if p.channels == 3 else (dv, du), np.uint8)
            bg[...] = CANVAS_RGB if p.channels == 3 else CANVAS_RGB[0]
            return bg, u0, v0                                    # nothing of this image here
        comp = self.composite(k, tx0, ty0, tx1, ty1)
        m[0, 2] -= tx0 * TILE
        m[1, 2] -= ty0 * TILE
        interp = cv2.INTER_NEAREST if s_dev * F / lin >= 0.9999 else cv2.INTER_LINEAR
        border = CANVAS_RGB if comp.ndim == 3 else CANVAS_RGB[0]
        if field is None:
            buf = cv2.warpAffine(comp, m, (du, dv), flags=interp | cv2.WARP_INVERSE_MAP,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=border)
            return buf, u0, v0
        # the solutions' own distortions: the affine map corrected pixel by pixel (bilinear between the grid's points)
        (fw_, fh_), fx, fy = field
        n = fx.shape[0]
        gx = np.clip((ax + np.arange(du) / s_dev - 0.5) / max(fw_ - 1.0, 1e-9) * (n - 1), 0, n - 1)
        gy = np.clip((ay + np.arange(dv) / s_dev - 0.5) / max(fh_ - 1.0, 1e-9) * (n - 1), 0, n - 1)
        mgx = np.ascontiguousarray(np.broadcast_to(gx.astype(np.float32)[None, :], (dv, du)))
        mgy = np.ascontiguousarray(np.broadcast_to(gy.astype(np.float32)[:, None], (dv, du)))
        rx = cv2.remap(fx, mgx, mgy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        ry = cv2.remap(fy, mgx, mgy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        uu = np.arange(du, dtype=np.float32)[None, :]
        vv = np.arange(dv, dtype=np.float32)[:, None]
        mapx = (m[0, 0] * uu + m[0, 1] * vv + m[0, 2] + rx / F).astype(np.float32)
        mapy = (m[1, 0] * uu + m[1, 1] * vv + m[1, 2] + ry / F).astype(np.float32)
        buf = cv2.remap(comp, mapx, mapy, interp, borderMode=cv2.BORDER_CONSTANT, borderValue=border)
        return buf, u0, v0

    def release(self):
        """Forget the rendered tiles (the image is no longer shown)."""
        RTILES.drop(lambda k, u=self.uid: k[0] == u)
        self._comp = None

    def thumbnail(self, side: int) -> np.ndarray:
        if self._thumb is None or self._thumb[0] != side:
            if self.align is not None:                       # drawn on another grid: its picture there
                fw, fh = self.frame
                s = side / max(fw, fh)
                w, h = max(1, round(fw * s)), max(1, round(fh * s))
                img = np.empty((h, w, 3), np.uint8)
                img[...] = CANVAS_RGB
                got = self.view(s, fw / 2, fh / 2, w, h)
                if got is not None:
                    buf, u0, v0 = got
                    if buf.ndim == 2:
                        buf = np.repeat(buf[:, :, None], 3, axis=2)
                    img[v0:v0 + buf.shape[0], u0:u0 + buf.shape[1]] = buf
                self._thumb = (side, img)
            else:
                self._thumb = (side, thumbnail_of(self.prep, self.stretch, side))
        return self._thumb[1]


def to_qimage(buf: np.ndarray) -> QImage:
    """An (h, w, 3) or (h, w) uint8 array as a QImage over the same memory (keep the array while it is drawn)."""
    buf = np.ascontiguousarray(buf)
    h, w = buf.shape[:2]
    fmt = QImage.Format_RGB888 if buf.ndim == 3 else QImage.Format_Grayscale8
    return QImage(buf.data, w, h, buf.strides[0], fmt)


TAG_BG = QColor(0, 0, 0, 170)


def tag(p: QPainter, pt: QPointF, text: str, right: bool = False, color: str = "#ffffff", bg: QColor = TAG_BG,
        role: str | None = None):
    """A small label on the image (rounded, dark); ``role``: an icon on the side it points at."""
    m = p.fontMetrics()
    w = m.horizontalAdvance(text) + 14 + (18 if role else 0)
    r = QRectF(pt.x() - w if right else pt.x(), pt.y(), w, max(m.height() + 6, 20))
    p.setPen(Qt.NoPen)
    p.setBrush(bg)
    p.drawRoundedRect(r, 4, 4)
    p.setPen(QColor(color))
    if role:
        ix = r.left() + 4 if right else r.right() - 20
        draw(p, QRectF(ix, r.center().y() - 8, 16, 16), role)
        p.drawText(r.adjusted(20, 0, 0, 0) if right else r.adjusted(0, 0, -20, 0), Qt.AlignCenter, text)
    else:
        p.drawText(r, Qt.AlignCenter, text)
    return r
