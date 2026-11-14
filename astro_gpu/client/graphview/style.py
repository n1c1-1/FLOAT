"""The look of the graph page: the client's own colours, style sheet, icons and words (widgets.py, icons.py,
graphnode.py), plus the few constants of today's graph page (graph.py) the new items paint with, copied so that the
page does not import the old one (tests/client/graphview/test_graphnode.py checks they stay equal)."""
from __future__ import annotations

from ..graphnode import BREAK_COL, BY_KEY, FALLBACK_WORD, PAUSE_COL, REVIEWED_COL, STATE_WORD  # noqa: F401
from ..i18n import _
from ..icons import IconLabel, app_icon, draw, icon, icon_only, pixmap, set_icon  # noqa: F401
from ..widgets import ACCENT, BAD, DISABLED, OK, QSS, SHARED, WARN, Box, ElidedLabel, clear, hline, label, para  # noqa: F401

# -- graph.py
CUT_COL, GREY_COL = "#6b7280", "#c9a227"
BROKEN_COL = "#e05252"
STATE_COL = {"done": OK, "pending": "#5d6470", "stale": WARN, "running": ACCENT, "failed": BAD,
             "interrupted": WARN, "skipped": "#5d6470", "fallback": WARN}
#: the colours of the Stars block's two images: their ports and their wires (graph.py IMAGE_COL, by the core's names)
IMAGE_COL = {"starless": "#7aa2d6", "stars_only": "#e0c45c", "": "#9aa3b2"}
WIRE_COL = "#9aa3b2"
PALETTE_W = 140
PALETTE_FOLDED_W = 46      # folded or automatic: the strip of the blocks' icons (graph.py PALETTE_FOLDED_W)
#: the block a pane shows is "never computed" in this state: its state band is not drawn on a Session or Post
PENDING = "pending"

#: a core block kind -> the client's icon role
ROLE = {"session": "block_session", "image": "block_image", "calibration": "block_calibration",
        "registration": "block_alignment", "selection": "block_selection", "crop": "block_crop",
        "stacking": "block_stacking", "astrometry": "block_astrometry", "gradient": "block_gradient",
        "deconv": "deconvolve", "denoise": "block_denoise", "color": "block_colour", "stars": "block_stars",
        "synth": "block_stars", "combine": "block_fusion", "combine_layers": "block_recombine", "post": "layers"}
#: the palette's groups (headings shown in capitals), named as today's palette names them (graph.py PALETTE_GROUPS:
#: "FRAMES", one word in each language)
GROUPS = [("Sources", ["session", "image"]),
          ("FRAMES", ["calibration", "registration", "selection", "crop", "stacking"]),
          ("Image", ["astrometry", "gradient", "deconv", "denoise", "color"]),
          ("Post", ["stars", "synth"]),
          ("Composition", ["combine", "combine_layers"])]
#: a core block kind -> its key in the client's block table (graphnode.BY_KEY)
CLIENT_KEY = {"session": "session", "image": "image", "calibration": "calibration", "registration": "alignement",
              "selection": "selection", "crop": "crop", "stacking": "empilement", "astrometry": "astrometrie",
              "gradient": "fond", "deconv": "deconv", "denoise": "debruitage", "color": "couleur", "stars": "etoiles",
              "synth": "synth", "combine": "fusion", "combine_layers": "composition"}
#: the block names of the palette (English; shown through _)
KIND_TITLE = {"session": "Session", "image": "Image", "calibration": "Calibration", "registration": "Registration",
              "selection": "Selection", "crop": "Crop", "stacking": "Stacking", "astrometry": "Astrometry",
              "gradient": "Gradient", "deconv": "Deconvolution", "denoise": "Denoise", "color": "Color",
              "stars": "Stars/background", "synth": "Synthetic stars", "combine": "Combination",
              "combine_layers": "Combine layers", "post": "Post"}
#: Post: what it does (graph.py POST_WHAT)
POST_WHAT = "Where the workflows' images become layers: stretched, balanced, blended and exported."


def kind_title(kind: str) -> str:
    return _(KIND_TITLE.get(kind, kind))


def kind_what(kind: str) -> str:
    """What a block kind does, in the client's words (the block table), translated."""
    if kind == "post":
        return _(POST_WHAT)
    step = BY_KEY.get(CLIENT_KEY.get(kind, ""))
    return _(step["what"]) if step else ""


def state_word(state: str) -> str:
    return _(STATE_WORD.get(state, "")) if state in STATE_WORD else ""
