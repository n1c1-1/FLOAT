"""The contract of the graph core: everything a view or the client may call. Qt-free, pure functions on a stored
graph (model.Project) and a view state (operations.new_view()); no history, no file format.

Ids are strings: blocks "b<n>", rows "r<n>", wires "w<n>". A night is named by its Session block id.
Block kinds (keys, the engine's block names): session calibration registration selection crop stacking image
astrometry gradient deconv denoise color stars synth combine combine_layers post.
Ports: out "image" (every block giving an image), Stars/background out "Starless" / "Stars only"; Combination in =
its type's names ("S II", "Hα", "O III" | "Hα", "O III" | "R", "G", "B" | "L", "R", "G", "B"); Combine layers in
"Background", "Stars"; Post in "layers" (many wires). Combination types: sho ho rgb lrgb.
File kinds: lights darks flats bias flat_darks.

Texts: Result.reason / .toast and Place.reason / .tip are messages (msg.py), written by `reason_text(tr)`,
`toast_text(tr)`, `tip_text(tr)`; the query functions below that return text take `tr` (the client's translation
function; English when None).

Operations: apply(graph, name, **kwargs) -> (graph2, Result), argument names fixed:
  add_block(kind, at=None, row=None)   at = a place spec (Place.op carries it); at None = natural place
                                       (double-click) in `row` (or the first row)
  move_block(block, at)                remove_block(block)            (alias delete_block)
  add_row(kind, files=None, name=None) kind "session" | "image"       delete_row(row)
  add_night(row, files=None, name=None)  files = {kind: [paths]}      remove_night(night)  (alias delete_night)
  move_night(night, row=None, before=None)  row None = its own new row; before = a session of `row` (reorder)
  merge_nights(a, b)                   split_night(night, lights)     lights = list of paths taken out
  add_variant(block)                   take_alternative(wire)         set_reference(row)
  set_combination_type(block, type)    type "sho" | "ho" | "rgb" | "lrgb"
  wire(src, src_port, dst, dst_port)   unwire(wire)                   send_to_post(block)
  switch_off(block)                    set_pause(block)
  rename_row(row, name)                rename_night(night, name)      duplicate_row(row)
  set_setting(block, key, value)       apply_to_rows(block, key)
  add_files(session, kind, paths, mode="add")   mode "add" | "replace"; a following box starts from Common's list
  remove_files(session, kind, paths)   use_for_all_nights(session, kind)
  -- Common: Project.common = {"files": {kind: [paths]}, "settings": {block kind: {key: v}}}
     Session.files[kind] absent = follows Common; [] = own "none". Effective = own, else Common, else default.
  set_common_files(kind, paths, mode="add")      set_common_setting(kind, key, value)
  reset_own(block, keys=None)          reset_own_files(session, kind)
  move_files(src_session, src_kind, dst_session, dst_kind, paths=None, copy=False)
                                       src_session/dst_session may be COMMON; paths None = the whole list
  split_by(session, groups)            groups = [{"into": "here"|"common"|"night"|"row", "files": {kind: [paths]},
                                       "name": str|None}]; the listed paths leave the session's lists first
  merge_rows(src, dst)                 src's nights join dst; src's row (chain, variants, wires) goes
  set_align(row, ref, block=None)      the row's Align on target (a row or tap; None: to choose), from `block`
                                       of the target (its Stacking/Image by default; its Astrometry brings the plate)
  own_grid(row)                        no Align on            set_align_off(row)   switch it off / on
  set_reference(row)                   "reference for all": every other stack aligned on this one
  set_calibrate_together(on)           one calibration for all the nights of a stack, or each night its own
  -- step 3b (what the current graph page does)
  try_variant(block, change, name=None)  change {"param": [stage, name, value]} | {"method": m} | {"off": bool}
                                       | {"setting": [key, value]}: a variant row on trial (Row.trial = the
                                       original's workflow id); before any fork point, a whole copy of the stack
                                       drawing the same nights
  keep_variant(row)  discard_variant(row)
  set_param(block, stage, name, value)  the engine's setting by its real key on the block owning it (value None: back
                                       to the suggestion); set_row_param(row, stage, name, value) for calibrate /
                                       inventory (the stack's), render_simple (its Output), align_on
  set_method(block, method)            add_combination(type="sho", name=None)
  wire(..., channel=None)              set_input(block, port, channel=None, weight=None, image=None)
  add_stars_combination(block)         "<name> · Stars": the same type reading the stars alone
  new_night(name=None, filter="")      a session no stack draws yet
  add_stack(name, nights, blocks=None, align="auto", all_nights=False)   a stack drawing these session ids
  share_night(night, row)              the same session drawn in another stack (files and name kept equal)
  set_night_filter(night, filter)      send_to_post(block, name=None): mid-row, a tap; remove_tap(tap)
  set_own_calibration(block, kind, paths, mode="add")   follow_session(block, kind)   remove_own_calibration(block)
  Templates: graphcore.templates.build(graph, t, name, opts) -> Plan (Plan.steps for apply_many, Plan.post(rows)).
  Post's wires: graphcore.post (shared_images, wired_images, with_post_wires, post_actions).
  fold_all() unfold_all() set_mode(mode) select(id)   view only: they change the view dict, never the graph
apply_many(graph, [(name, kwargs | f(results_so_far) -> kwargs), ...]) = one step, all or nothing.
"""
from .shapes import Box, Path, Drawing, Place, Result  # noqa: F401
from .model import (Project, Block, Row, Wire, Index, FILE_KINDS, FRAME_KINDS, JOIN_KINDS, COMBO_PORTS,  # noqa: F401
                    COMBO_LABEL, blank_project, check, clone, to_dict, from_dict)
from .msg import M, say  # noqa: F401
from .operations import apply, apply_many, new_view, COMMON, try_op  # noqa: F401
from .rules import LABEL, SETTINGS, FILE_LABEL, defaults, setting_spec  # noqa: F401
from . import rules as _rules


def layout(project, view, tr=None):
    """Pure: the whole Drawing of the graph page for this model and view state. Same input, same output."""
    from .layout import layout as _l
    from .judge import memo
    return memo("layout", project, (view.get("folded") is True, id(tr)), lambda: _l(project, view, tr))


def places(project, view, kind=None, block=None):
    """Every drop place for a palette kind being dragged (kind=...) or a placed block being dragged
    (block=id). Lit and refused ones; nothing moves. -> [Place]."""
    from .placement import places as _p
    return _p(project, view, kind=kind, block=block)


def palette_state(project, view, kind, tr=None):
    """-> (enabled, reason text): a card is greyed when no place is lit, with its natural place's reason."""
    from .placement import palette_state as _s
    ok, why = _s(project, view, kind)
    return ok, say(why, tr)


def title(project, b, tr=None):
    """Block title: "Gradient 2", "S-H-O", "Session 1" (a one-night row's Session shows the row name).
    A row id gives the row name."""
    if b in {r.id for r in project.rows}:
        return next(r.name for r in project.rows if r.id == b)
    return _rules.title(project, b, tr=tr)


def summary(project, b, tr=None):
    """One-line summary under the title, or the greyed reason when greyed. A frames row id gives its stack card
    line "3 sessions · 78 lights · drizzle auto"."""
    if b in {r.id for r in project.rows}:
        return _rules.card_summary(project, b, tr)
    from .judge import evaluation
    return _rules.summary(project, b, info=evaluation(project), tr=tr)


def greyed(project, b, tr=None):
    """"" when the block's needs are met, else its reason ("Astrometry is needed first", "downstream of Gradient")."""
    from .judge import evaluation
    return _rules.greyed(project, b, info=evaluation(project), tr=tr)


def wire_tip(project, w, tr=None):
    """Type text of a wire (stored id or derived Path id), e.g. "linear image · without stars"."""
    from .judge import evaluation
    return _rules.wire_tip(project, w, info=evaluation(project), tr=tr)


def row_badge(project, r, tr=None):
    """Under a stack's name: "reference" | "on the grid of “SII”" | "grid to choose" | "Align on off" | ""."""
    row = next((x for x in project.rows if x.id == r), None)
    if row is None or not row.nights:
        return ""
    a = row.align
    if a:
        if a.get("off"):
            return say("Align on off", tr)
        names = {x.id: x.name for x in project.rows} | {t.id: t.name for t in project.taps}
        if a.get("ref") in names:
            return say(M("on the grid of “{_ref}”", _ref=names[a["ref"]]), tr)
        return say("grid to choose", tr)
    return say("reference", tr) if project.reference == r else ""


def wire_check(project, view, src, src_port, dst, dst_port, tr=None):
    """"" when wire(src, src_port, dst, dst_port) would be accepted, else its reason (nothing changes)."""
    return say(try_op(project, "wire", {"src": src, "src_port": src_port, "dst": dst, "dst_port": dst_port}, view), tr)


def project_state(project, tr=None):
    """Project block colour: ("none" | "partial" | "complete", tip text)."""
    return _rules.project_state(project, tr)


def warnings(project, scope=None, tr=None):
    """Warnings decidable from the model alone (no reading): scope None, a row id or a session id ->
    [(session id, kind, text, (op, kwargs) | None)]."""
    return [(s, k, say(m, tr), fix) for s, k, m, fix in _rules.warnings(project, scope)]


stacks = _rules.stacks
effective = _rules.effective
followers = _rules.followers
files_of = _rules.files_of
differs = _rules.differs


def own_keys(project, b):
    """Keys block b holds of its own (unlike Common / default) -> [key]."""
    return list(project.blocks[b].settings)
