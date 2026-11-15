"""Writes the shipped templates (astro_gpu/resources/templates/<Tab>/<name>.astroproj) from the construction below.

A template is a blank project file: the graph a gallery template makes on a project without frames (empty Session blocks
where it needs filters), the project's own format, plus a ``template`` entry that holds what the gallery's card shows
(its name, its text, and the recipe that adds the same workflow to a project that already holds frames).

The files are the reference: this construction is only kept here, to write them again when a default changes
(``python tools/make_templates.py``); tests/graphcore/test_template_files.py checks the files against it.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FULL = dict(debruitage=True, etoiles="retirer", fond="simple")
CLASSIC_DN = dict(debruitage=True, etoiles="garder", fond="simple")
CLASSIC = dict(debruitage=False, etoiles="garder", fond="simple")
SHIPPED = ROOT / "astro_gpu" / "resources" / "templates"
SUFFIX = ".astroproj"


def construction() -> list[dict]:
    """Today's templates, sorted by the equipment they are built for: Color (a colour camera, or an image) and Mono
    (a monochrome camera and its filters)."""
    from astro_gpu.client.gallery import PALETTES, SHO_DEFAULT
    return [
        dict(key="colour_min", tab="Color", name="Colour minimal", opts=CLASSIC,
             text="Calibration, registration, selection and stacking, then astrometry, the sky gradient and the colours. "
                  "The quickest complete workflow, without a neural network."),
        dict(key="colour_full", tab="Color", name="Colour full", opts=FULL, recommended=True,
             text="The same, then denoise on the two half-stacks and the stars taken off: the starless image and the stars "
                  "come to Post as two layers."),
        dict(key="image", tab="Color", name="Image", opts=CLASSIC, source="image", plain=True, blocks=[],
             text="One image to Post as a layer, a FITS, TIFF, PNG or JPEG file: blocks are added from the palette."),
        dict(key="images", tab="Color", name="Several images", opts=CLASSIC, source="image", plain=True, blocks=[],
             images=3,
             text="Several images to Post, one layer each: each Image block takes its file in the graph."),
        dict(key="dual_x2", tab="Color", name="Dual-band x2", opts=FULL, kind="fusion", palette=SHO_DEFAULT,
             filters=["O III / H-alpha", "O III / S II"], dual_x2=True,
             text="A colour camera with two dual-band filters (O III/H-alpha and O III/S II) and a session for the stars."),
        dict(key="sho_rgb", tab="Color", name="SHO + RGB stars", opts=FULL, kind="fusion", palette=SHO_DEFAULT,
             filters=PALETTES[SHO_DEFAULT], rgb_stars=True,
             text="The nebula from the narrow bands, the stars from a colour camera: each filter's workflow and a colour "
                  "workflow take their stars off, the others aligned on the first filter; the combination reads the "
                  "starless images, and a Combine layers block puts the colour stars back on its image."),
        dict(key="filters_min", tab="Mono", name="Filters minimal", opts=CLASSIC, kind="filters", palette="H-O",
             filters=PALETTES["H-O"],
             text="A monochrome camera: each filter gets its own workflow, the others aligned on the first one, and each "
                  "result is a layer of Post."),
        dict(key="filters_full", tab="Mono", name="Filters full", opts=FULL, kind="fusion", palette=SHO_DEFAULT,
             filters=PALETTES[SHO_DEFAULT],
             text="Each filter's workflow with denoise and its stars taken off, the others aligned on the first one; the "
                  "combination of the filters chosen here makes the colour nebula from their starless images, a second one "
                  "of the same type makes the colour stars from their stars, and a Combine layers block puts them back."),
        dict(key="mono_rgb", tab="Mono", name="Mono RGB", opts=CLASSIC_DN, kind="fusion", palette="RGB",
             filters=list(PALETTES["RGB"]), mono_rgb=True,
             text="A monochrome camera with red, green and blue filters: each filter stacked on its own, the three combined "
                  "into the colour image, which is calibrated once, stars kept."),
        dict(key="mono_lrgb", tab="Mono", name="Mono LRGB", opts=CLASSIC_DN, kind="fusion", palette="LRGB",
             filters=list(PALETTES["LRGB"]), mono_rgb=True,
             text="A monochrome camera with a luminance filter and red, green and blue ones: each filter stacked on its "
                  "own, the colour from R, G and B and the detail from L, calibrated once, stars kept."),
    ]


def _renumber(d: dict) -> dict:
    """Post's layer ids are a counter of the process: renumbered in their order of appearance."""
    ids: dict = {}
    for v in d["post"].get("versions") or []:
        for lay in v.get("layers") or []:
            ids.setdefault(lay["id"], len(ids) + 1)
    for v in d["post"].get("versions") or []:
        for lay in v.get("layers") or []:
            lay["id"] = ids[lay["id"]]
            if lay.get("target") in ids:
                lay["target"] = ids[lay["target"]]
    return d


def build(t: dict) -> dict:
    """The blank project file's content of template ``t``."""
    from astro_gpu.client import i18n
    from astro_gpu.client.gallery import _view_opts
    from astro_gpu.client.model import ProjectModel
    from astro_gpu.client.services.worker import ApiWorker
    from astro_gpu.core import project2
    from astro_gpu.core.project import Target
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    i18n.set_language("en")
    root = Path(tempfile.mkdtemp(prefix="float_template_"))
    p = project2.create(root, "Untitled", Target())
    w = ApiWorker()
    m = ProjectModel(p, w)
    w.wait(120)
    try:
        from astro_gpu.client.i18n import _
        name = m.unique_name(_(t["palette"]) if t.get("kind") == "fusion" else _(t["name"]))
        res = m.apply_template(t, name, _view_opts(t))
        if not res.ok:
            raise RuntimeError(f"{t['key']}: {res.reason}")
        m.flush(force=True)
        d = project2.file_dict(p)
    finally:
        m.close(save=False)
        w.stop()
    d = json.loads(json.dumps(d, default=str))
    d["created"] = ""
    d["scratch_dir"] = ""
    d["name"] = ""
    d = _renumber(d)
    card = {k: v for k, v in t.items() if k not in ("tab",)}
    d["template"] = {"card": card}
    return d


def file_of(t: dict) -> Path:
    return SHIPPED / t["tab"] / (t["name"] + SUFFIX)


def write_json(path: Path, d: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(d, indent=1, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    for t in construction():
        write_json(file_of(t), build(t))
        print("written", file_of(t).relative_to(ROOT).as_posix())


if __name__ == "__main__":
    main()
