"""Plain records shared by the core and the view (no Qt). Coordinates in scene pixels, y down.
Texts are messages (see msg.py): the *_text methods write them in a language."""
from dataclasses import dataclass, field

from .msg import say


@dataclass(frozen=True)
class Box:
    x: float
    y: float
    w: float
    h: float

    @property
    def right(self):
        return self.x + self.w

    @property
    def bottom(self):
        return self.y + self.h


@dataclass
class Path:
    """A drawn wire. id: stored wire id ("w7") or a derived one ("c:b3>b4" chain, "n:b1>b2" night,
    "f:b5>b9" fork). style: "chain" | "night" | "fork" | "port" | "port_inactive" | "bar" (fan-in bar).
    colour: "" | "starless" | "stars_only". points: Manhattan polyline [(x, y), ...]."""
    id: str
    points: list
    style: str
    src: str = ""
    dst: str = ""
    colour: str = ""


@dataclass
class Drawing:
    """Everything the view draws. boxes: {block id: Box} (Post included; blocks folded in a card absent).
    cards: {row id: Box} (folded stacks). captions: [(row id, text, x, y)] (text baseline-left).
    bands: [(label, Box)]. wires: [Path]. ports: {(block id, port, "in"|"out"): (x, y)}.
    lines: [(owner, y)] owner = row id (chain line) or session id (night line), top to bottom.
    cells: {block id | "card:<row id>": (line index | None, logical column)} (None = composition area).
    extras: {"project": Box (the Project block), "new_workflow": Box (the link under the last row)}.
    size: (w, h)."""
    boxes: dict = field(default_factory=dict)
    cards: dict = field(default_factory=dict)
    captions: list = field(default_factory=list)
    bands: list = field(default_factory=list)
    wires: list = field(default_factory=list)
    ports: dict = field(default_factory=dict)
    lines: list = field(default_factory=list)
    cells: dict = field(default_factory=dict)
    extras: dict = field(default_factory=dict)
    size: tuple = (0, 0)


@dataclass
class Place:
    """A drop place for a drag (§2 placement grammar). kind: "insert" | "branch" | "night" | "row" | "comp" |
    "port". rect: Box of the ghost. line: index into Drawing.lines where the block will appear (None for
    "night"/"row"/"comp"/"port" when the line does not exist yet: then it is the index it will have).
    col: global column index the new block takes. lit: True if the drop is accepted; reason: refusal
    message when not lit. tip: message shown on hover. op: (operation name, kwargs) to pass to
    apply(graph, name, **kwargs)."""
    kind: str
    rect: Box
    line: object
    col: object
    lit: bool
    reason: object
    tip: object
    op: tuple

    def reason_text(self, tr=None):
        return say(self.reason, tr)

    def tip_text(self, tr=None):
        return say(self.tip, tr)


@dataclass
class Result:
    """Outcome of apply / apply_many. ok False: nothing changed, reason = refusal message.
    toast: message for the toast ("… · Ctrl+Z" is appended by the view). focus: id to select
    (block or row) or None. acted: (row id, line index, col) of where the gesture acted, or None
    (used by the invariants I4/I5). created: what the gesture made, {"rows", "blocks", "wires": [new ids],
    "row", "session", "block": the first new one of each kind or None} (None for refusals, undo, view ops)."""
    ok: bool
    reason: object = ""
    toast: object = ""
    focus: object = None
    acted: object = None
    created: object = None
    dependants: object = None   # a refused remove_block: the blocks after it that need it ([] when none can help)

    def reason_text(self, tr=None):
        return say(self.reason, tr)

    def toast_text(self, tr=None):
        return say(self.toast, tr)
