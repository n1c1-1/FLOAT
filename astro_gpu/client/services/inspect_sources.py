"""Inspect's sources: what the viewer shows for a block's image, a calibrated frame and its masters, where
a picture sits on the stack, an image sent to Post, a key resolved to its source."""
from __future__ import annotations

import os
import re
import threading
from collections import OrderedDict
from pathlib import Path

import numpy as np

from ... import api
from .worker import norm_path
from .inspect_display import neutral_preview, CAMERA_PREVIEWS, _rgb_of_file, _wcs_of
from .inspect_frames import inventory_headers


# ---------------------------------------------------------------------------------------------- the viewer's sources
# The common viewer (imageview.ImageView) reads its images in its own worker threads: a ``ViewImage`` source may be a
# callable, called there. The callables below read the engine's results (nothing runs in the GUI thread). A result
# that is not there raises ``Unavailable`` with the words the view shows.

def diag(name: str):
    """A diagnostic of the engine (``api.calibration_frames``, ``registration_table``, ``stack_diagnostics``,
    ``color_fit``), or None while the engine does not have it."""
    return getattr(api, name, None)


def block_of(stage: str) -> str:
    """The title of the block a stage belongs to (English; translated where shown)."""
    from ..graphnode import STEPS
    return next((b["title"] for b in STEPS if stage in b["stages"]), stage)


def not_yet(stage: str) -> str:
    """The words for a diagnostic the engine does not give yet, or that this result does not hold (translated)."""
    from ..i18n import _
    return _("To recompute: {block}").format(block=_(block_of(stage)))


_MISSING = re.compile(r"^(.*): missing: recompute (.*) to get it\.?$")


def missing_text(entry: str) -> str:
    """One ``missing`` entry of a diagnostic ("rejection maps: missing: recompute Stacking to get it") in the
    interface's language: "Rejection maps: not in this result. Recompute Stacking to get them." """
    from ..i18n import _
    m = _MISSING.match(str(entry).strip())
    if m is None:
        return _("Not in this result.")
    what, stage = m.group(1).strip(), m.group(2).strip()
    w = _(what)
    return _("{what}: older result · recompute {stage}").format(what=w[:1].upper() + w[1:], stage=_(stage))


def missing_of(obj, word: str = "") -> list[str]:
    """The ``missing`` entries of a diagnostic (those naming ``word``), in the interface's language."""
    return [missing_text(m) for m in (getattr(obj, "missing", None) or []) if word in str(m)]


def not_computed(stage: str) -> str:
    from ..i18n import _
    return _("{block}: not computed here").format(block=_(block_of(stage)))


def _unavailable(text: str):
    from ..imageview import Unavailable
    return Unavailable(text)


def _w(which, stage: str) -> str:
    """``which`` is one word for every stage, or {stage: current | previous} (a picture made of several results)."""
    return which.get(stage, "current") if isinstance(which, dict) else which


def _array(p, wid: int, stage: str, name: str, which) -> np.ndarray:
    which = _w(which, stage)
    try:
        return np.asarray(api.stage_array(p, wid, stage, name, which=which))
    except api.NotDone:
        if api.stage_outputs(p, wid, stage, which) is None:
            raise _unavailable(not_computed(stage)) from None
        raise _unavailable(not_yet(stage)) from None


def array_source(p, wid: int, stage: str, name: str, which: str = "current", channel: int | None = None):
    """An array of a done result (one channel of it with ``channel``)."""
    def load():
        a = _array(p, wid, stage, name, which)
        if channel is not None and a.ndim == 3:
            return a[channel]
        return _one_plane(p, wid, a)
    return load


def _one_plane(p, wid: int, a: np.ndarray) -> np.ndarray:
    """A monochrome sensor's stack is stored as three equal planes (the engine's files are all (3, H, W)): shown as the
    one plane it is, with one column of statistics and no channels."""
    try:
        mono = a.ndim == 3 and a.shape[0] == 3 and p.is_mono(wid)
    except (KeyError, AttributeError):
        return a
    if mono and np.array_equal(a[0, ::7, ::7], a[1, ::7, ::7]) and np.array_equal(a[0, ::7, ::7], a[2, ::7, ::7]):
        return a[:1]
    return a


def preview_source(p, wid: int, stage: str, name: str, which: str = "current"):
    """An engine preview (a JPEG), the ones still in the camera's colours with a neutral sky."""
    def load():
        out = api.stage_outputs(p, wid, stage, _w(which, stage))
        f = out.previews.get(name) if out is not None else None
        if f is None:
            raise _unavailable(not_computed(stage) if out is None else not_yet(stage))
        a = _rgb_of_file(f)
        return neutral_preview(a) if (stage, name) in CAMERA_PREVIEWS else a
    return load


def apply_matrix(M: np.ndarray, a: np.ndarray) -> np.ndarray:
    """The colour matrix of the Color stage on a (3, H, W) image in the camera's colours: linear display RGB."""
    M = np.asarray(M, np.float32)
    out = np.empty_like(a, dtype=np.float32)
    for c in range(3):
        out[c] = M[c, 0] * a[0] + M[c, 1] * a[1] + M[c, 2] * a[2]
    return out


def grey_sky(a: np.ndarray) -> np.ndarray:
    """The sky of a colour image set grey by an offset per channel (its median brought to the mean of the three), as a
    background neutralisation does: the colours of the stars and the nebulae are not touched (no gain)."""
    if a.ndim != 3 or a.shape[0] != 3:
        return a
    med = np.array([float(np.median(a[c, ::4, ::4])) for c in range(3)])
    shift = med.mean() - med
    for c in range(3):
        a[c] += np.float32(shift[c])
    return a


def color_source(p, wid: int, base: tuple[str, str], which: str = "current"):
    """An image of the workflow with the colour matrix of its Color result applied (the colours the stage measured on
    the Gaia stars), its sky set grey (``grey_sky``: the light pollution's colour is the Gradient block's)."""
    def load():
        M = _array(p, wid, "color", "M", which)
        return grey_sky(apply_matrix(M, _array(p, wid, base[0], base[1], which).astype(np.float32, copy=False)))
    return load


def _sky_model(p, wid: int, which: str, H: int, W: int, entry: str = "gradient"
               ) -> tuple[np.ndarray, np.ndarray]:
    """(the sky variation the Gradient block removes, (3, H, W) in the colours it was modelled in, read back as the
    stage reads it; the sky level it keeps, (3,)). ``entry``: the block's node in the graph."""
    import json

    from ...stages.gradient import INFO_FILE, SKY_FILE, read_grid
    out = api.stage_outputs(p, wid, entry, _w(which, entry))
    if out is None:
        raise _unavailable(not_computed("gradient"))
    info = json.loads((Path(out.dir) / INFO_FILE).read_text(encoding="utf-8"))
    return read_grid(np.load(Path(out.dir) / SKY_FILE), info, H, W), np.asarray(info["level"], np.float32)


def calibrated_source(p, wid: int, base: tuple[str, str], which: str = "current"):
    """An image of the workflow in calibrated colours: the colour matrix of its Color result applied, nothing else (its
    sky stays where the step left it: an image whose gradient is removed keeps its zero point)."""
    def load():
        M = _array(p, wid, "color", "M", which)
        return apply_matrix(M, _array(p, wid, base[0], base[1], which).astype(np.float32, copy=False))
    return load


def input_ops(p, wid: int, entry: str) -> list[str]:
    """The image blocks a block reads before its own work, in the graph's order (``stages.gradient.image_ops``: the
    chain's entries it depends on, of the kinds its rule applies): what stands between the stack and its input."""
    from ...core.workflow import READ_KINDS, READERS, base
    try:
        chain = api.chain(p, wid)
        mine = set(api.deps(p, wid, entry))
    except (KeyError, ValueError):
        return []
    kinds = READ_KINDS[READERS.get(base(entry), "all")]
    return [x for x in chain if x in mine and base(x) in kinds]


def _image_through(p, wid: int, entries: list[str], which) -> np.ndarray:
    """The stack with the image blocks ``entries`` applied in order, as ``stages.gradient.image_in`` reads it: a Gradient
    takes its sky variation off, a Color applies its matrix, a Deconvolution or a Denoise gives its image."""
    from ...core.workflow import base
    a = _array(p, wid, _stk(p, wid, which), "drz_full", which).astype(np.float32)
    for e in entries:
        b = base(e)
        if b == "gradient":
            a -= _sky_model(p, wid, which, *a.shape[1:], entry=e)[0]
        elif b == "color":
            a = apply_matrix(_array(p, wid, e, "M", which), a)
        elif b == "deconv":
            a = _array(p, wid, e, "deconv_full", which).astype(np.float32)
        elif b == "denoise":
            a = _array(p, wid, e, "n2n_full", which).astype(np.float32)
    return a


def input_source(p, wid: int, stage: str = "gradient", which: str = "current"):
    """What a block reads: the stack with the image blocks before it in the graph applied (``input_ops``)."""
    def load():
        return _image_through(p, wid, input_ops(p, wid, stage), which)
    return load


def gradient_source(p, wid: int, stage: str = "gradient", which: str = "current"):
    """The stack with its sky gradient removed, as the later blocks read it: what the Gradient block reads (the blocks
    before it in the graph applied), its sky variation taken off (the level stays)."""
    def load():
        return _image_through(p, wid, [*input_ops(p, wid, stage), stage], which)
    return load


def removed_source(p, wid: int, stage: str = "gradient", which: str = "current"):
    """What the Gradient block took away: its sky model (the variation, on the level it keeps), in the camera's
    colours."""
    def load():
        stack = _array(p, wid, _stk(p, wid, which), "drz_full", which)
        var, level = _sky_model(p, wid, which, *stack.shape[1:], entry=stage)
        var += level[:, None, None]
        return var
    return load


#: the masters the calibration keeps (arrays of its result): key, file, caption
MASTERS = [("master_dark", "master_dark", "Master dark"), ("master_flat", "master_flat", "Master flat"),
           ("master_bias", "master_bias", "Master bias"), ("master_flat_dark", "master_flat_dark", "Master dark flat"),
           ("badmask", "badmask", "Defective pixels")]


_masters_lock = threading.Lock()


_masters: OrderedDict = OrderedDict()                     # calibrate dir -> (dark, flat, bad), the last two


def _calib_masters(cal_dir: str):
    with _masters_lock:
        hit = _masters.get(cal_dir)
        if hit is not None:
            return hit
    d = Path(cal_dir)
    m = (np.load(d / "master_dark.npy").astype(np.float32), np.load(d / "master_flat.npy").astype(np.float32),
         np.load(d / "badmask.npy").astype(bool))
    with _masters_lock:
        _masters[cal_dir] = m
        while len(_masters) > 2:
            _masters.popitem(last=False)
    return m


def forget_masters(root) -> None:
    """The masters read from a project's results go with it (closed): a gigabyte for a large sensor, kept until the
    program ended."""
    top = os.path.normcase(os.path.abspath(str(root)))
    with _masters_lock:
        for k in [k for k in _masters if os.path.normcase(os.path.abspath(k)).startswith(top + os.sep)]:
            del _masters[k]


def calibrated_frame(p, wid: int, path: str, which: str = "current") -> tuple[np.ndarray, str, str]:
    """One light frame as the calibration made it: (mosaic in ADU above the dark, Bayer pattern, how it was got); a frame
    already in colour: (its image (H, W, 3), no pattern, how it was got). From the calibrated cache when it is on disk,
    else made again from the stage's own masters with its formula ((raw − master dark) / master flat, defective pixels
    left empty)."""
    import json

    from ...io.frames import read_cfa
    from ...stages.calibrate import VALID_MAX
    which = _w(which, "calibrate")
    out = api.stage_outputs(p, wid, "calibrate", which)
    if out is None:
        raise _unavailable(not_computed("calibrate"))
    d = Path(out.dir)
    info = json.loads((d / "cache.json").read_text(encoding="utf-8"))
    cfa = str(info.get("cfa") or "")
    white = float(info.get("white_level") or 65535.0) - float(info.get("dark_level") or 0.0)
    cache = Path(info.get("path") or "")
    n, H, W = int(info["n"]), int(info["height"]), int(info["width"])
    colour = bool(info.get("rgb_path"))                    # frames already in colour (issue #57): their three planes
    if colour:
        cache = Path(info["rgb_path"])
    idx = None
    if cache.is_file() and cache.stat().st_size == n * (3 if colour else 1) * H * W * 2:
        inv = inventory_headers(p, wid, "lights")
        order = list(inv)
        norm = norm_path(path)
        idx = order.index(norm) if norm in order and len(order) == n else None
    if colour:
        return _calibrated_colour(cache, info, d, path, idx, white), "", "cache" if idx is not None else "masters"
    if idx is not None:
        mm = np.memmap(cache, dtype=np.float16, mode="r", shape=(n, H, W))
        x = np.asarray(mm[idx], np.float32)
        sat = x >= VALID_MAX                                   # the saturation marks of the cache
        x /= float(info.get("scale") or 1.0)
        x[sat] = white
        how = "cache"
    else:
        dark, flat, bad = _calib_masters(str(d))
        raw = read_cfa(path).astype(np.float32)
        x = (raw - dark) / flat
        x[bad] = np.nan
        how = "masters"
    return np.nan_to_num(x, nan=0.0, posinf=white, neginf=0.0), cfa, how


def _calibrated_colour(cache: Path, info: dict, d: Path, path: str, idx: int | None, white: float) -> np.ndarray:
    """A frame already in colour as the calibration made it (issue #57): (H, W, 3) in ADU above the dark, from its
    three planes in the cache (``idx``: its rank there), else made again colour by colour from the masters (a joined
    set keeps its masters on the luminance only: the same for the three colours)."""
    from ...io.frames import read_colour
    from ...stages.calibrate import MASTER_RGB, VALID_MAX
    n, H, W = int(info["n"]), int(info["height"]), int(info["width"])
    if idx is not None:
        mm = np.memmap(cache, dtype=np.float16, mode="r", shape=(n, 3, H, W))
        x = np.array(mm[idx], np.float32)
        sat = x >= VALID_MAX
        x /= float(info.get("scale") or 1.0)
        x[sat] = white
    else:
        dark, flat, bad = _calib_masters(str(d))
        if all((d / f).is_file() for f in MASTER_RGB):
            dark, flat = (np.load(d / f).astype(np.float32) for f in MASTER_RGB)
        x = (read_colour(path) - dark) / flat
        x[:, bad] = np.nan
    return np.moveaxis(np.nan_to_num(x, nan=0.0, posinf=white, neginf=0.0), 0, -1)


def calframe_source(p, wid: int, path: str, which: str = "current"):
    def load():
        return calibrated_frame(p, wid, path, which)[0]
    return load


def wcs_source(p, wid: int, which: str = "current"):
    """The astrometric solution of the workflow, read in the viewer's thread (None without one)."""
    def load():
        return _wcs_of(p, wid, _w(which, "astrometry")) or _wcs_of(p, wid, "current")
    return load


class ScaledWcs:
    """The solution of the stack for a reduced, cropped picture of its grid (the final image): pixel (x, y) of the
    picture is grid pixel (ox + (x + 0.5) f − 0.5, oy + (y + 0.5) f − 0.5). Only what the readout and the compare
    ask."""

    def __init__(self, wcs, origin, f: float):
        self.wcs, self.ox, self.oy, self.f = wcs, float(origin[0]), float(origin[1]), float(f)

    def pixel_to_world_values(self, x, y):
        f = self.f
        return self.wcs.pixel_to_world_values(self.ox + (np.asarray(x) + 0.5) * f - 0.5,
                                              self.oy + (np.asarray(y) + 0.5) * f - 0.5)

    def world_to_pixel_values(self, ra, dec):
        gx, gy = self.wcs.world_to_pixel_values(ra, dec)
        f = self.f
        return (np.asarray(gx) + 0.5 - self.ox) / f - 0.5, (np.asarray(gy) + 0.5 - self.oy) / f - 0.5


# ---------------------------------------------------------------------------------------------- where a picture sits
def _stk(p, wid: int, which="current") -> str:
    """The stage whose image the blocks after Stacking read (Chains.resolve): the aligned image when the workflow has an
    Align on block done, else the stack (issue #165)."""
    return "align_on" if api.stage_outputs(p, wid, "align_on", _w(which, "align_on")) is not None else "stack"


def _stack_meta(p, wid: int, which: str, stage: str | None = None) -> tuple[str, float, int, int]:
    """(the registration's key, the drizzle scale, the stack grid's width and height) of a workflow's stack result:
    its grid is the reference frame of that registration enlarged by the scale (stage stack: X = (x + 0.5) s - 0.5)."""
    import json
    stage = stage or _stk(p, wid, which)
    st = api.stage_outputs(p, wid, stage, which) or api.stage_outputs(p, wid, stage)
    if st is None:
        raise LookupError("no stack result")
    meta = json.loads((Path(st.dir) / "drz_meta.json").read_text(encoding="utf-8"))
    reg = (st.manifest.get("inputs") or {}).get("register") or ""
    if stage == "align_on":                           # the reference's grid: a frame of its own, its zone is none
        return str(st.key), float(meta.get("scale") or 1.0), int(meta["width"]), int(meta["height"])
    if not reg:
        out = api.stage_outputs(p, wid, "register", which) or api.stage_outputs(p, wid, "register")
        reg = out.key if out is not None else ""
    W, H = int(meta["width"]), int(meta["height"])
    sensor = meta.get("sensor") or [W, H]
    s = float(meta.get("scale") or (W / float(sensor[0])))
    return str(reg or st.key), s, W, H


def _stack_zone(p, wid: int, which: str, stage: str | None = None) -> tuple[int, int]:
    """Where the stack starts on the reference frame (a Crop block's zone: issue #79), in its pixels."""
    import json
    stage = stage or _stk(p, wid, which)
    if stage == "align_on":
        return 0, 0
    st = api.stage_outputs(p, wid, stage, which)
    meta = json.loads((Path(st.dir) / "drz_meta.json").read_text(encoding="utf-8")) if st is not None else {}
    zone = meta.get("zone") or [0, 0]
    return int(zone[0]), int(zone[1])


#: the blocks in which the Crop block looks for the sky, in pixels of the image it shows
SKY_BLOCK = 32


def crop_frame(p, wid: int) -> dict:
    """What the Crop block shows (issue #79): the reference frame in colour, stretched (``rgb``, ``step`` sensor pixels
    per pixel of it), and where the sky is (``sky``, one value per block of ``SKY_BLOCK`` pixels of ``rgb``)."""
    import json

    from ..stretch import superpixel
    out = api.stage_outputs(p, wid, "register", "current")
    if out is None:
        raise _unavailable(not_computed("register"))
    ref = int(np.load(Path(out.dir) / "reg.npz")["ref"])
    inv = api.stage_outputs(p, wid, "inventory", "current")
    lights = json.loads((Path(inv.dir) / "frames.json").read_text(encoding="utf-8"))["lights"]
    a, cfa, _how = calibrated_frame(p, wid, lights[ref]["path"])
    a = np.nan_to_num(np.asarray(a, np.float32))
    rgb = a if a.ndim == 3 else np.moveaxis(superpixel(a, cfa), 0, 2)
    step = round(a.shape[1] / rgb.shape[1])
    level = np.median(rgb, axis=(0, 1))
    spread = 1.4826 * np.median(np.abs(rgb - level), axis=(0, 1))
    shown = np.arcsinh(np.clip((rgb - level) / np.maximum(spread, 1e-6) + 2.0, 0, None) / 3.0) / 6.0
    grey = rgb.mean(2)
    H, W = grey.shape
    n, m = max(1, H // SKY_BLOCK), max(1, W // SKY_BLOCK)
    blocks = grey[:n * SKY_BLOCK, :m * SKY_BLOCK].reshape(n, -1, m, SKY_BLOCK).mean((1, 3))
    low = float(np.quantile(blocks, .1))
    noise = 1.4826 * float(np.median(np.abs(blocks[blocks <= np.median(blocks)] - low)))
    return {"rgb": (np.clip(shown, 0, 1) * 255).astype(np.uint8), "step": step, "size": (W * step, H * step),
            "sky": blocks <= low + 3 * max(noise, 1e-6)}


def stack_place(p, wid: int, which="current", stage: str | None = None):
    """Where a picture on the stack grid of a workflow sits (``imageview.Placement``), read in the viewer's thread: on
    the grid of the registration's reference frame, at the drizzle scale. Its size (w, h) tells a reduced copy of the
    grid (an engine preview of it) from the grid itself."""
    from ..imageview import Placement
    which = _w(which, "stack")

    def place(w: int, h: int):
        reg, s, W, H = _stack_meta(p, wid, which, stage)
        fx, fy = (W / float(w), H / float(h)) if (w, h) != (W, H) else (1.0, 1.0)
        x0, y0 = _stack_zone(p, wid, which, stage)
        return Placement.scaled(("reference frame", reg), fx / s, float(x0), float(y0), fy=fy / s)
    return place


def reference_frame_place(p, wid: int, which="current"):
    """The reference frame of the registration (calibrated, on the sensor): the grid the stacks are placed on."""
    from ..imageview import Placement
    which = _w(which, "stack")

    def place(w: int, h: int):
        reg, _s, _W, _H = _stack_meta(p, wid, which)
        return Placement(("reference frame", reg))
    return place


def render_place(p, wid: int, stack_which: str, grid, origin, size):
    """Where a final image sits: a part of the stack grid (``grid`` its size there, ``origin`` its corner: the render
    canvas), reduced to ``size`` (the render's half size); then the stack grid's own placement (``stack_which``: the
    stack result the render was made from, current or previous)."""
    from ..imageview import Placement
    on_stack = stack_place(p, wid, stack_which)

    def place(w: int, h: int):
        f = float(grid[0]) / float(size[0] or w)
        fy = float(grid[1]) / float(size[1] or h)
        sx, sy = (float(v) for v in origin)
        _reg, _s, W, H = _stack_meta(p, wid, _w(stack_which, "stack"))
        return Placement.scaled(None, f, sx, sy, fy=fy).then(on_stack(W, H))
    return place


def preview_wcs_source(p, wid: int, which_ast: str, stage: str, name: str, which="current"):
    """The solution of the stack for an engine preview of its grid (a reduced copy: the H-alpha survey), read in the
    viewer's thread: the preview's size against the stack grid's gives the reduction."""
    def load():
        w = _wcs_of(p, wid, which_ast) or _wcs_of(p, wid, "current")
        out = api.stage_outputs(p, wid, stage, _w(which, stage))
        f = out.previews.get(name) if out is not None else None
        if w is None or f is None:
            return None
        from PIL import Image
        with Image.open(f) as im:
            pw = im.size[0]
        _reg, _s, W, _H = _stack_meta(p, wid, _w(which, "stack"))
        return ScaledWcs(w, (0.0, 0.0), W / float(pw))
    return load


def final_wcs_source(p, wid: int, which: str, grid, origin, size):
    """The sky readout of a final image (``grid``: the canvas size on the stack grid, ``size``: the picture's)."""
    def load():
        w = _wcs_of(p, wid, which) or _wcs_of(p, wid, "current")
        if w is None or not grid or not size or not size[0]:
            return None
        return ScaledWcs(w, origin, grid[0] / size[0])
    return load


def light_thumb(p, path: str, cfa: str = ""):
    """The film strip's thumbnail of a light, made from the raw with the viewer's automatic stretch (a neutral
    sky)."""
    from ..imageview import ViewImage, quick_thumbnail

    def load():
        return quick_thumbnail(ViewImage(path, cfa=cfa, camera=True))
    return load


# ---------------------------------------------------------------------------------------------- Send to Post
def post_key(vi, wid: int | None) -> str:
    """The key Post keeps for an image Inspect sends (``PostPage3.add_image_layer``): "<stage>/<store key>/<array>"
    for an array of a result (Post reads it from the store), else "inspect:{...}" or "file:<path>", which
    ``resolve_key`` turns into the source again when the project is opened again."""
    import json
    ident = vi.identity()
    if isinstance(ident, tuple) and ident[:1] == ("file",):
        return "file:" + str(vi.source)
    if isinstance(ident, tuple) and ident[:1] == ("inspect",):
        kind = ident[1]
        if kind == "array" and ident[3] is None and len(ident[5]) == 1:
            stage, key = ident[5][0]
            return f"{stage}/{key}/{ident[2]}"
        if kind == "master":
            return f"calibrate/{ident[2]}/{ident[3]}"
        if kind == "calframe":
            return "inspect:" + json.dumps({"wid": wid, "kind": "calframe", "path": ident[3]})
        if kind in ("array", "preview", "color", "gradient", "removed", "calibrated", "input"):
            return "inspect:" + json.dumps({"wid": wid, "kind": kind, "name": ident[2], "channel": ident[3],
                                            "base": list(ident[4]), "stages": [s for s, _k in ident[5]]})
        return "inspect:" + json.dumps({"wid": wid, "kind": str(kind)})
    return "inspect:" + json.dumps({"wid": wid, "kind": "other", "name": vi.name})


def _debayered(load, cfa: str = ""):
    """A source for Post: a Bayer mosaic made a colour image (one pixel per 2 x 2 photosites)."""
    from ..stretch import superpixel

    def get():
        a = np.asarray(load())
        return superpixel(a, cfa) if a.ndim == 2 and cfa else a
    return get


def post_source(vi, neutral: bool = False):
    """What Post reads for an image of a viewer: its source, read in Post's render worker (a raw or calibrated mosaic
    made a colour image). ``neutral``: the viewer showed it with a neutral sky (still in the camera's colours), so Post
    stretches its channels apart (``PostView.add_image_layer``)."""
    from ..imageview import _read

    def load():
        arr, cfa, _wcs = _read(vi)
        return _debayered(lambda: arr, cfa)()
    load.neutral = bool(neutral)
    return load


def shown_neutral(viewer) -> bool:
    """Whether a viewer shows its image with a neutral sky now (its stretch's Neutral sky)."""
    try:
        st = viewer.stretch()
    except (AttributeError, RuntimeError):
        return False
    return bool(getattr(st, "neutral", False))


def resolve_key(p, key: str):
    """The source of an image layer of Post from its key (``post_key``), when the project is opened again; None when
    it cannot be made again (Post says the image is missing). Called in Post's render worker."""
    import json
    if key.startswith("file:"):
        path = key[5:]
        if not Path(path).is_file():
            return None
        from ..io.frames import read_cfa, read_info
        return _debayered(lambda: read_cfa(path), read_info(path).cfa or "")
    if not key.startswith("inspect:"):
        return None
    try:
        d = json.loads(key[len("inspect:"):])
    except ValueError:
        return None
    wid, kind = d.get("wid"), d.get("kind")
    if wid is None:
        return None
    which = {s: _which_done(p, wid, s) or "current" for s in d.get("stages") or []}
    if kind == "calframe":
        w = _which_done(p, wid, "calibrate") or "current"

        def cal():
            x, cfa, _how = calibrated_frame(p, wid, d["path"], w)
            return _debayered(lambda: x, cfa)()
        return cal
    stage = (d.get("stages") or [""])[0]
    if kind == "array":
        return array_source(p, wid, stage, d.get("name") or "", which, d.get("channel"))
    if kind == "preview":
        return preview_source(p, wid, stage, d.get("name") or "", which)
    if kind == "color":
        return color_source(p, wid, tuple(d.get("base") or ("stack", "drz_full")), which)
    if kind == "gradient":
        return gradient_source(p, wid, stage, which)
    if kind == "input":
        return input_source(p, wid, stage, which)
    if kind == "calibrated":
        return calibrated_source(p, wid, tuple(d.get("base") or ("background", "astro")), which)
    if kind == "removed":
        return removed_source(p, wid, stage, which)
    if kind in ("rejection_low", "rejection_high"):
        fn = diag("stack_diagnostics")
        w = _which_done(p, wid, "stack") or "current"
        return (lambda: getattr(fn(p, wid, w), kind)) if fn is not None else None
    return None


# The engine's diagnostics (``diag``) take ``which``: an Outdated block shows the result it had. Each answer is read
# whole in the worker; what an older result did not keep comes back as None (the view says "recompute ... to get it").

def _try(fn, *a, **k):
    """A diagnostic, or None when the result is not there or predates it."""
    try:
        return fn(*a, **k)
    except (api.NotDone, OSError, ValueError, KeyError, IndexError, TypeError):
        return None


def _which_done(p, wid: int, stage: str) -> str | None:
    """current when the workflow's result of ``stage`` is done, else previous when it had one, else None."""
    for w in ("current", "previous"):
        if api.stage_outputs(p, wid, stage, w) is not None:
            return w
    return None
