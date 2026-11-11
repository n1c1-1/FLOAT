"""Crops and optimises the raw shots of tools/guide_shots.py into docs/images/guide/ (PNG, each under 250 KB).

    python tools/guide_crop.py RAW_FOLDER
"""
import sys
from pathlib import Path

from PIL import Image, ImageDraw

OUT = Path(__file__).resolve().parents[1] / "docs" / "images" / "guide"
#: shot -> (file, box, width to scale to or 0, [boxes painted with the page colour])
SHOTS = {
    "1_home": ("1-home.png", (0, 20, 640, 340), 0, [(482, 40, 640, 68)]),
    "2_gallery": ("2-templates.png", (0, 30, 1210, 590), 0, []),
    "3_frames": ("3-frames.png", (0, 30, 1600, 372), 0, []),
    "4_run": ("4-run.png", (0, 40, 740, 303), 0, []),
    "5_graph": ("5-graph.png", (0, 30, 1600, 580), 1280, []),
    "7_post": ("6-post.png", (0, 30, 1600, 870), 1000, []),
}


def main(raw: Path):
    OUT.mkdir(parents=True, exist_ok=True)
    for f in sorted(raw.glob("*.png")):
        key = f.stem.split("_en")[0].split("_", 1)[1]
        if key not in SHOTS:
            continue
        name, box, width, paint = SHOTS[key]
        im = Image.open(f).convert("RGB")
        for b in paint:
            draw = ImageDraw.Draw(im)
            draw.rectangle(b, fill=im.getpixel((b[2] + 2, b[1] + 2)))
        im = im.crop(box)
        if width:
            im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
        im = im.quantize(256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.FLOYDSTEINBERG)
        im.save(OUT / name, optimize=True)
        print(name, im.size, (OUT / name).stat().st_size // 1024, "KB")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
