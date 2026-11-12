"""A helper of the tests of the joins (issue #165): an Align on block placed in a workflow's graph."""
from astro_gpu.core import workflow as wfl


def add_align_on(p, w, reference: int, method: str) -> None:
    """An Align on block on ``reference`` right after the workflow's Stacking (or first, from an image)."""
    if w.blocks is None:
        w.blocks = wfl.check_blocks(wfl.with_output(wfl.from_options(w.options, w.end, wfl.source_of(w)), wfl.source_of(w)),
                                    wfl.source_of(w))
        w.end = None
    at = 0
    if wfl.source_of(w) == "frames":
        at = next(i for i, n in enumerate(w.blocks) if n["block"] == "stacking") + 1
    w.blocks.insert(at, wfl.node("align_on", method, reference=reference))
    w.options = wfl.options_of(w.blocks, wfl.source_of(w))
