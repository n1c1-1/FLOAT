"""The window's sides on the new graph page, hosted as today's page hosts them (graph.py ``GraphPage``): the settings
panel (``panel.Panel``, the same widget built the same way, with its signals wired to the same answers), its ``Dock``
(docked, folded to a strip of icons, automatic, hidden with an arrow), the palette's ways of showing with the arrow of a
hidden palette, and the "+ N min to run" said after a change.

The panel is given the page's own nodes (graphnode ``Node``, from the project model's ``BlockMap``; ``panel_node``):
the chosen block's, Post's (``final``, every workflow), a Combine layers block's (``composition``, its workflow, its
layer of Post when known) and the Project's. The scene's block items carry the same node as ``node`` for the run
overlay. Nothing here reads today's ``blocks.Graph``.
"""
from __future__ import annotations

import logging
import time

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QToolButton, QWidget

from ..i18n import _, hm
from .bridge import PROJECT

log = logging.getLogger(__name__)


class SidesHost:
    """Mixed in ahead of ``QWidget`` by ``GraphPage``: what the page hosts beside the graph (see the module's text)."""

    # ------------------------------------------------------------------ the palette over the page
    def _build_palette_room(self, body, h):
        """The palette is a child of the body placed over its slot: automatic, it opens over the graph (graph.py)."""
        from ..layout import ARROW_H, ARROW_W
        from . import style
        self.palette.setParent(body)
        # weakly: the palette holding the page's method kept every closed project's page and model alive
        import weakref
        fit = weakref.WeakMethod(self._place_palette)
        self.palette.on_fit = lambda: fit() and fit()()
        self.palette.chosen.connect(self._palette_chosen)
        self.palette_slot = QWidget()                    # the room it takes in the page (graph.py palette_slot)
        self.palette_slot.setFixedWidth(self.palette.slot_width())
        h.addWidget(self.palette_slot)
        self.left_arrow = QToolButton(body)
        style.set_icon(self.left_arrow, "chevron_right", 16, "dim")
        self.left_arrow.setFixedSize(ARROW_W, ARROW_H)
        self.left_arrow.setToolTip(_("Show the blocks again"))
        self.left_arrow.setStyleSheet("QToolButton { background: #1c1f26; border: 1px solid #2a2f38; padding: 0; "
                                      "border-top-right-radius: 5px; border-bottom-right-radius: 5px; }")
        self.left_arrow.clicked.connect(self.sidesShown.emit)
        self.left_arrow.hide()
        self.palette.show()

    def _palette_chosen(self, st):
        self.sideChosen.emit("left", st)

    def _place_palette(self):
        body = getattr(self, "body", None)
        if body is None or getattr(self, "palette_slot", None) is None:
            return
        self.palette_slot.setFixedWidth(self.palette.slot_width())
        self.palette.setGeometry(0, 0, self.palette.width(), body.height())
        self.palette.raise_()
        self.left_arrow.move(0, (body.height() - self.left_arrow.height()) // 2)
        self.left_arrow.raise_()

    def _show_palette(self, on: bool):
        """The palette and its slot shown (a graph position, its side not hidden) or not (Table, the gallery, Tab)."""
        on = on and self.left_state != "hidden"
        self.palette.setVisible(on)
        self.palette_slot.setVisible(on)
        self.left_arrow.setVisible(self.left_state == "hidden")
        self._place_palette()

    # ------------------------------------------------------------------ the settings panel and its dock
    def _build_panel(self, body, h):
        """Today's settings panel and its dock, built as graph.py builds them (only on a project model)."""
        from ..layout import Dock, fold_button
        from ..panel import Panel
        pn = Panel(self.s, self.steps)
        h.addWidget(pn)
        self.panel = pn
        self.dock = Dock(body, pn, lambda: h.addWidget(pn), lambda: h.removeWidget(pn), pn.folds.items,
                         _("Block settings"))
        h.insertWidget(h.indexOf(pn), self.dock.strip)
        self.dock.chosen.connect(lambda st: self.sideChosen.emit("right", st))
        self.dock.arrow.clicked.connect(self.sidesShown.emit)
        pn.sectionsChanged.connect(self.dock.refresh_icons)
        pn.corner = fold_button(pn, self.dock.fold, _("Fold to icons"))
        pn.changed.connect(self.request_refresh)
        pn.openTune.connect(self.openBlock.emit)
        pn.openPost.connect(self.openPost.emit)
        pn.variantMade.connect(self.variant_made)
        pn.addSession.connect(self.addSession.emit)
        pn.renamed.connect(self.rename)
        pn.selectBlock.connect(lambda key: self.select(step=key))
        pn.restoreBlock.connect(self.restore_block)
        pn.toast.connect(self.toast)
        pn.removeTool.connect(lambda: self.remove_tool(self.current_node(), None))
        pn.saveTemplate.connect(self.save_template)
        pn.openLayer.connect(self.openLayer.emit)
        pn.cutToggled.connect(self.toggle_cut)
        pn.drop_files = self.panel_drop
        self._panel_timer = QTimer(self, singleShot=True, interval=0)
        self._panel_timer.timeout.connect(self.refresh_panel)
        self._build_confirm(pn)

    def panel_drop(self, sid, kind, paths):
        """Files dropped on a kind's row of a Session's panel: resolved as a drop on that Session's target of the graph
        (graph_drops: the type filter, « n added · n left · Take all », no double, files stored one by one)."""
        g = self.bridge.graph
        block = next((b for b, blk in g.blocks.items() if blk.kind == "session" and getattr(blk, "sid", None) == sid),
                     None)
        if block is None:
            return None
        return self.bridge.drops.drop(block, kind, paths)

    def panel_later(self, *_a):
        if getattr(self, "panel", None) is not None:
            self._panel_timer.start()

    def refresh_panel(self):
        """The panel shows the block chosen on the page (today's node of it), or the wire to Post chosen."""
        pn = getattr(self, "panel", None)
        if pn is None:
            return
        try:
            node = self.panel_node()
            w = self.canvas.selected_wire
            if node is None and w is not None and self.wire_panel(w):
                return
            pn.refresh(node)
        except Exception:  # noqa: BLE001 - the panel follows the page; it never takes the page down
            log.exception("graph page: the settings panel could not show the block chosen")

    def panel_node(self):
        """The node the panel shows for the block chosen on the page: its BlockMap node; Post's (``final``, every
        workflow); a Combine layers block's (``layer_node``); the Project's; None."""
        s, b = self.s, self.bridge
        sel = b.selected
        if sel == PROJECT or (self.is_blank() and sel is None):
            return self._project_node()
        if sel is None or s is None:
            return None
        blk = b.block(sel)
        if blk is not None and blk.kind == "combine_layers":
            return self.layer_node(sel)
        if blk is not None and blk.kind == "post":
            from ..graphnode import Node
            return Node("final", list(getattr(s, "paths", None) or []))
        return self.current_node()

    # ------------------------------------------------------------------ a Combine layers block: its layer of Post
    def _layer_source(self, block):
        """The workflow whose images a Combine layers block combines: that of the block wired to its first input."""
        bm = self._block_map()
        if bm is None:
            return None
        g = self.bridge.graph
        for w in sorted((w for w in g.wires if w.dst[0] == block), key=lambda w: str(w.dst[1]) != "Background"):
            n = bm.node(w.src[0])
            if n is not None and n.paths:
                return n.paths[0]
        return None

    def layer_node(self, block):
        """The node of a Combine layers block: ``composition`` over the workflow whose images it combines, with its
        layer of Post (``lid``) when the BlockMap gives one (SWAP_NOTES: the graph does not tell it yet). None when no
        workflow feeds it yet."""
        bm = self._block_map()
        n = bm.node(block) if bm is not None else None
        if n is not None and n.paths:
            return n
        p = self._layer_source(block)
        if p is None:
            return None
        from ..graphnode import Node
        return Node("composition", [p], lid=getattr(n, "lid", None))

    def layer_id(self, block):
        """The Post layer a Combine layers block opens (graph.py ``open_layer``): its own, else the one Post gives its
        workflow by default, made a layer without an undo step (decision 7); None without a project model."""
        s = self.s
        if s is None:
            return None
        n = self.layer_node(block)
        if n is not None and n.lid is not None:
            return n.lid
        p = (n.paths[0] if n is not None and n.paths else None) or self._layer_source(block)
        if p is not None and hasattr(s, "default_post_layer"):
            return s.default_post_layer(p.pid)
        return None

    # ------------------------------------------------------------------ a wire to Post chosen (graph.py show_wire)
    def wire_panel(self, wire_id) -> bool:
        """A wire to Post chosen: the layer it makes, by its name, what it is, Rename, Out of Post, Open Post (today's
        wire panel). False when the wire does not go to Post."""
        s, b = self.s, self.bridge
        w = next((x for x in b.graph.wires if x.id == wire_id), None)
        bm = self._block_map()
        if w is None or bm is None or b.block(w.dst[0]) is None or b.block(w.dst[0]).kind != "post":
            return False
        n = bm.node(w.src[0])
        p = n.paths[0] if n is not None and n.paths else None
        if p is None:
            return False
        from ..post3 import tr_name
        part = {"Starless": "starless", "Stars only": "stars"}.get(str(w.src[1]))   # the Stars block's two ports
        lay = next((x for x in s.post_layers() if x.get("kind") == "workflow" and x.get("pid") == p.pid
                    and x.get("part") == part), None)
        unwire = (lambda: b.act("unwire", wire=wire_id))
        if part is None:
            name = tr_name(str((lay or {}).get("name") or p.name))
            note = _("The image of “{name}”, a layer of Post. Del or Out of Post takes it out; Ctrl+Z brings it "
                     "back.").format(name=p.name)
            row = b.row_of(w.src[0])
            actions = [(_("Rename"), (lambda: self.canvas.rename(w.src[0], row.id)) if row is not None else
                        (lambda: None), _("The workflow and its layer")),
                       (_("Out of Post"), unwire, _("Its image is no longer a layer"))]
        else:
            from ..graphnode import STAR_OUTPUTS
            word = _(dict(STAR_OUTPUTS)[part])
            name = tr_name(str((lay or {}).get("name") or f"{p.name} · {word}"))
            note = _("{image} of “{name}”, sent to Post as it is: a layer. Del or Out of Post takes it out; Ctrl+Z "
                     "brings it back.").format(image=word, name=p.name)
            actions = [(_("Out of Post"), unwire, _("Its image is no longer a layer"))]
        actions.append((_("Open Post"), self.openPost.emit, _("Post, where the layer is set")))
        self.panel.show_wire(name, note, actions)
        return True

    # ------------------------------------------------------------------ the panel's gestures (graph.py's answers)
    def variant_made(self, pid, orig):
        """The variant is shown at the same block, side by side with the workflow it comes from."""
        s = self.s
        self.select(index=s.index_of(pid))
        q = s.view(pid)
        if q is not None:
            self.toast(_('Variant "{name}" created').format(name=q.name) + " · Ctrl+Z")
        self.variantMade.emit(pid, orig)

    def rename(self, pid, name):
        why = self.s.name_problem(name, pid=pid)          # decision 8: never empty, never another's name
        if why:
            self.toast(_("Name refused: {why}").format(why=why))
            return
        self.s.rename(pid, name)

    def _selected_block(self):
        sel = self.bridge.selected
        if sel is not None and self.bridge.row(sel) is not None:
            sel = self._card_block(sel)
        return sel if sel in self.bridge.graph.blocks else None

    def toggle_cut(self):
        """The panel's Switch off / Switch on: the page's D on the block chosen (one undo step, refused with why)."""
        b = self._selected_block()
        if b is not None:
            self.bridge.act("switch_off", block=b)

    def restore_block(self, key):
        """A greyed block's "Switch on Astrometry": the block it needs, in the chosen workflow, switched on again."""
        bm, s = self._block_map(), self.s
        n = bm.node_of(s.path, key) if bm is not None and s.path is not None else None
        b = bm.block_of(n) if n is not None else None
        if b is not None and self.bridge.block(b) is not None and self.bridge.block(b).off:
            self.bridge.act("switch_off", block=b)

    def save_template(self, pid):
        """Save template… (issue #47): the workflow's graph, under a name, in a tab of the gallery (graph.py)."""
        from ..gallery import TABS, make_template, save_user_template, template_dialog
        s = self.s
        p = s.view(pid)
        if p is None:
            return
        src = s.source_kind(p)
        if src == "fusion":
            self.toast(_("Combination: no template"))
            return
        got = template_dialog(self, p.name)
        if got is None:
            return
        name, tab = got
        what = _(dict(TABS)[tab])
        t = save_user_template(make_template(name, tab, _("Made from the workflow “{name}”.").format(name=p.name),
                                             src, s.graph(p)))
        s.log('Template "{name}" saved', name=t["name"])
        self.toast(_("Template “{name}” saved · {tab}").format(name=t["name"], tab=what))

    def _post_changed(self, source=None):
        """Post's composition changed: the page follows, not for a slider of the panel (it stays under the mouse)."""
        if source is None or source is not getattr(self, "panel", None):
            self.request_refresh()

    # ------------------------------------------------------------------ "+ N min to run" (graph.py refresh)
    def said(self, *_a):
        """A gesture's own message was just said: the time it adds is not said over it."""
        self._said_at = time.monotonic()

    def cost_toast(self):
        """After a change that adds 30 s or more to the run, the toast says how much (today's page, through the
        model's cost): never without frames or an image, never for a run that is refused, never over the gesture's
        own message."""
        s = self.s
        if s is None or not getattr(s, "ready", False):
            return
        try:
            sec = s.total_seconds()
            last = self._last
            said = time.monotonic() - getattr(self, "_said_at", -9.0) < 1.5
            refused = bool(getattr(s, "run_problems", lambda: [])())
            if last is not None and s.paths and len(s.paths) == last[1] and sec - last[0] >= 30 and \
                    s.has_data() and not said and not refused:
                t = hm(sec - last[0])                    # "≤ 1 min" is already an estimate: one sign only
                self.toast(_("+ {t} to run").format(t=("" if s.cost().measured or t.startswith("≤") else "≈ ") + t))
            self._last = (sec, len(s.paths))
        except Exception:  # noqa: BLE001 - an estimate is a word of the toast: never a failure of the page
            log.exception("graph page: the run's estimate could not be said")

    # ------------------------------------------------------------------ a change waiting for its confirmation
    def _build_confirm(self, pn):
        """A change that would compute again results computed and up to date waits for Replace / Try variant / Cancel:
        a change of the panel at the panel's foot, a gesture of the graph in the page's toast; Enter answers the
        default (Try variant, else Replace), Esc cancels; choosing another block meanwhile asks first."""
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QKeySequence, QShortcut
        self._confirm_keys = []
        for key, how in ((Qt.Key_Return, "default"), (Qt.Key_Enter, "default"), (Qt.Key_Escape, "cancel")):
            sc = QShortcut(QKeySequence(key), self)
            sc.setContext(Qt.WidgetWithChildrenShortcut)
            sc.setEnabled(False)
            sc.activated.connect(lambda h=how: self.answer_change(h))
            self._confirm_keys.append(sc)
        self.bridge.hold = self._hold_selection
        pn.answered.connect(self._change_answered)
        pn.changed.connect(self._confirm_keys_follow)
        if hasattr(self.s, "changeAsked"):
            self.s.changeAsked.connect(self._gesture_asked)
        self.canvas.toast.link.connect(self._confirm_link)

    def _confirm_keys_follow(self, *_a):
        """Enter and Esc answer the question only while one waits (else they are the page's and the fields' own)."""
        s = self.s
        on = bool(getattr(s, "pending_change", None) or getattr(s, "pending_gesture", None))
        for sc in getattr(self, "_confirm_keys", ()):
            sc.setEnabled(on)

    def answer_change(self, how: str) -> bool:
        """The change waiting answered: "default" (Enter), "replace", "variant", "cancel" (Esc). False: none waits."""
        s = self.s
        if s is None:
            return False
        if getattr(s, "pending_change", None) and self.panel is not None:
            done = self.panel.answer(how)
            self._confirm_keys_follow()
            return done
        if getattr(s, "pending_gesture", None):
            g = s.pending_gesture
            if how == "default":
                how = "variant" if g.get("variant") else "replace"
            self._answer_gesture(how)
            return True
        self._confirm_keys_follow()
        return False

    def _hold_selection(self, ident) -> bool:
        """Another block chosen while a change of the panel waits: the choice stays, the panel asks to answer first;
        the block chosen is shown once the change is answered (never a silent loss of what was typed)."""
        s, pn = self.s, self.panel
        if s is None or pn is None or not getattr(s, "pending_change", None) or self._mirroring:
            return False
        pn.pending_next = (ident,)                       # (None: nothing chosen, a click on the empty page)
        pn.refresh(pn.node)
        self.toast(_("Answer first: Replace, Try variant or Cancel"))
        return True

    def _change_answered(self):
        """The change of the panel answered: the block chosen meanwhile is chosen now."""
        pn = self.panel
        self._confirm_keys_follow()
        if pn is None or pn.pending_next is None:
            return
        (ident,), pn.pending_next = pn.pending_next, None
        if ident is None or ident == PROJECT or ident in self.bridge.graph.blocks or self.bridge.row(ident) is not None:
            self.bridge.choose(ident)

    def _confirm_toast(self):
        """The toast the question is asked in: the table's while the table is shown, else the canvas'."""
        if self.position() == "table" and getattr(self, "table_toast", None) is not None:
            t = self.table_toast
            if not getattr(self, "_table_toast_wired", False):
                t.link.connect(self._confirm_link)
                self._table_toast_wired = True
            return t
        return self.canvas.toast

    def _gesture_asked(self, note: str = ""):
        """A gesture of the graph waits (``pending_gesture``): what it computes again and the time, with Replace, Try
        variant (when the gesture can be made on a copy of its block), Cancel and Do not ask, in the page's toast,
        which stays until the answer."""
        s = self.s
        g = getattr(s, "pending_gesture", None)
        if g is None:
            return
        links = [(_("Replace"), "confirm_replace")]
        if g.get("variant"):
            links.append((_("Try variant"), "confirm_variant"))
        links.append((_("Cancel"), "confirm_cancel"))
        if not note:
            links.append((_("Do not ask"), "confirm_never"))
        text = " · ".join(x for x in (note, g.get("what"), g["cost"].text()) if x)
        self._confirm_toast().say_links(text, links, lambda: getattr(self.s, "pending_gesture", None) is not None)
        self.said()
        self._confirm_keys_follow()

    def _confirm_link(self, key):
        if key == "confirm_replace":
            self._answer_gesture("replace")
        elif key == "confirm_variant":
            self._answer_gesture("variant")
        elif key == "confirm_cancel":
            self._answer_gesture("cancel")
        elif key == "confirm_never":                     # the preference off; the question still waits for its answer
            from .. import prefs
            prefs.set("confirm_changes", False)
            self._gesture_asked(_("Changes apply at once from now on (Edit › Settings › General asks again)"))

    def _answer_gesture(self, how: str):
        """Replace: the gesture made (one Undo step; Ctrl+Z gives the results back at once); Try variant: made on a copy
        of its block, the computed chain kept; Cancel: nothing changes."""
        s = self.s
        g, s.pending_gesture = getattr(s, "pending_gesture", None), None
        self._confirm_keys_follow()
        if g is None:
            return None
        if how == "cancel":
            self.canvas.toast.hide()
            if getattr(self, "table_toast", None) is not None:
                self.table_toast.hide()
            return None
        steps = g["variant"] if how == "variant" and g.get("variant") else g["steps"]
        with s.confirmed():
            res = self.bridge.act_many(steps, quiet=how == "replace")
        if res.ok and how == "replace":
            said = self.bridge.say(res.toast) if res.toast else _("Changed")
            self.bridge.toast.emit(_("{what} · results computed again · Ctrl+Z gives them back").format(what=said))
        if res.ok:
            self.bridge.acted.emit(steps[0][0], res)
        return res

