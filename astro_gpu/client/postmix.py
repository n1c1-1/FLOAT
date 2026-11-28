"""The composition's mixes of a layer's tree (#181): one recursive renderer for every combination Post mixes again, at
any depth (client/postfusion.py: the layer tree and its paths; client/postengine.py: the layers and the composition).

A mix is a combination of filters (S-H-O, H-O...): each of its inputs is a node (its path names it in every saved key),
a plane of its own (the filter, balanced as its run balanced it) or another combination's picture (a combination inside
a combination: the filter reads that combination's Output through a channel). The nodes have an eye, a mask and filters
of their own, which act on the node's image before its parent mixes it.

One stretch for the whole mix (render/simple.SimpleRenderEngine.colour_factor): the Output of a combination is linear in
its balanced filters (the run's own maps, tests/client/test_post_nest_premises.py), so the planes of every leaf (a
plane that reads no other combination: ``postfusion.leaf_plan``) are stretched by ONE factor, from the statistics of the
outermost mix, and every picture under it is the same sum of stretched leaves as the linear mix is of linear ones:
``y = F * (x - c0)`` at every depth. A role that reads a nested combination A through a channel is, in the stretched
domain, ``sum_k coef_k * s_k + F * (offset + c0 * (sum coef_k - 1))`` with ``s_k`` A's picture read through the channel
(or a plane read of the combination's other inputs): the same affine map the run applied to the linear planes.

The stars side made of the stars' combination is stretched as a whole, as the engine does (the stretch of its
Output with the starless combination's statistics: render/stars.CombinedStarsEngine, which is not one factor for
palette stars): its filters are its balanced planes times one per-pixel factor ``G`` (the engine's picture over the
linear mix, in luminance), so that nothing is changed at the defaults; what the user changes in it (an eye, a mask, a
filter, a hue) is added to the engine's picture through the same sums. The planes of the stars mix are not stretched
apart: "Stretched with the stars mix".

The images of the filters are kept as ever: the leaf planes, linear and apart from hues and intensities (``_leaf_lin``),
the stretched ones with the factor (``_mix_parts``), the statistics of the outermost mix (``_fusion_stats``), a side's
image before the end of its stretch by all it depends on (``_mix_key``); the filters' steps by digests (``_node_sig``),
the same ones the plan makes without computing (``_mix_sig``)."""
from __future__ import annotations

import numpy as np

from . import postengine as pe
from . import postfusion as pf


class MixRendering:
    """The Composer's part that makes the mixes of the layers' trees."""

    # ------------------------------------------------------------ the tree
    @staticmethod
    def _nlay(lay: dict, node: dict) -> dict:
        """What the provider is asked about a combination: the layer, its workflow and description in place of the layer's."""
        return dict(lay, pid=node["pid"], fusion=node["fusion"])

    def _form(self, lay: dict, node: dict) -> dict:
        """How the roles of a combination read its inputs ({"maps", "channels"}: ``Provider.fusion_form``), kept by its
        result and balance."""
        sid = (pe.FUSION, node["pid"])
        key = (node["pid"], self.provider.version(sid), node["fusion"].get("balance"))
        hit = self._forms.get(key)
        if hit is None:
            hit = self._forms[key] = self.provider.fusion_form(self._nlay(lay, node))
        return hit

    def _leaves(self, lay: dict, root: str) -> list[tuple[str, str, str]]:
        """The planes the mix at ``root`` is made of (``postfusion.leaf_plan``)."""
        return pf.leaf_plan(pe.mix_at(lay, root), root, lambda n: self._form(lay, n))

    @staticmethod
    def _under(lay: dict, root: str) -> list[str]:
        """The paths of the nodes under a side (its own mix's filters, and theirs)."""
        return [p for p in pe.nodes_of(lay) if p not in ("", *pe.SUBS) and pf.root_of(p) == root]

    def _struct(self, node: dict) -> tuple:
        """What the planes of a mix depend on of its tree (its results, filters and balance), not its hues or intensities."""
        return (node["pid"], self.provider.version((pe.FUSION, node["pid"])), tuple(node["fusion"]["roles"]),
                node["fusion"].get("balance"),
                tuple((r, self._struct(c)) for r, c in sorted(pf.mix_kids(node).items())))

    def _mix_plain(self, lay: dict, own: list, root: str) -> bool:
        """A mix whose nodes are all shown, none with a filter or a mask: its picture is the one the engine makes."""
        under = set(self._under(lay, root))
        shown, sides = lay.get("shown") or {}, lay.get("side_masks") or {}
        return all(shown.get(x, True) for x in under) and not any(x in sides for x in under) and \
            not any(fl.get("sub") in under and fl["visible"] for fl in own)

    # ------------------------------------------------------------ the planes
    def _leaf_lin(self, lay, root, level, box, canvas=None, cache=True) -> tuple[np.ndarray, np.ndarray]:
        """(L, h, w): the planes of the mix's leaves (``postfusion.leaf_plan``), linear, before their intensities, in
        its order; and the covered pixels. Kept apart from the hues and intensities (a hue changed reads nothing again)."""
        node = pe.mix_at(lay, root)
        leaves = self._leaves(lay, root)
        cache = cache and level > 0                       # (at full resolution: as ``_mix_parts``)
        key = ("lin", root, self._struct(node), tuple(leaves), level,
               None if canvas is not None and level > 0 else tuple(box),
               (canvas.W, canvas.H, canvas.origin) if canvas is not None else None)
        holder = (lay["id"], (pe.FUSION, "lin", root), level)
        hit = self.images.get(key) if cache and canvas is not None else None
        if hit is None:
            nodes = {p: n for p, n in pe.nodes_of(lay).items() if n.get("kind") == "mix" and pf.root_of(p) == root}
            got: dict = {}
            valid = None
            for path, role, kind in leaves:
                if (path, kind) not in got:
                    nl = self._nlay(lay, nodes[path])
                    if kind == "bal":
                        planes, v = self.provider.fusion_planes(nl, level, box)
                        valid = np.asarray(v, bool) if valid is None else valid & np.asarray(v, bool)
                    else:
                        planes = self.provider.fusion_reads(nl, level, box)
                    got[(path, kind)] = planes
            if valid is None:                              # (a mix of nested mixes only: any of its planes says it)
                _p, v = self.provider.fusion_planes(self._nlay(lay, node), level, box)
                valid = np.asarray(v, bool)
            hit = np.empty((len(leaves) + 1, *valid.shape), np.float32)
            for i, (path, role, kind) in enumerate(leaves):
                hit[i] = got[(path, kind)][role]
            hit[-1] = valid
            if cache and canvas is not None:
                self.images.put(key, hit, holder)
        elif cache:
            self.images.hold(key, holder)
        return hit[:-1], hit[-1] > 0.5

    @staticmethod
    def _stats_at(canvas, draft: bool, level: int = 0) -> int:
        """The level the statistics of a mix are measured at: ``stats_level``, the same for every level and region rendered;
        a draft (a quick look while a slider moves: a hue or an intensity changes them at each move) measures them on its
        own picture, and the exact picture at the release the exact statistics."""
        return max(pe.stats_level(canvas.W, canvas.H), int(level)) if draft else pe.stats_level(canvas.W, canvas.H)

    def _fusion_stats(self, lay, root, canvas, draft: bool = False, level: int = 0) -> dict:
        """The statistics of the mix at ``root`` (render/simple.image_stats), measured once on the whole canvas at the
        measure level (``_stats_at``): the same for every level and region rendered."""
        node = pe.mix_at(lay, root)
        at = self._stats_at(canvas, draft, level)
        key = (root, pe.nest_core(node), self._struct(node), canvas.W, canvas.H, canvas.origin, at)
        hit = self._fstats.get(key)
        if hit is None:
            from ..render.simple import image_stats
            W, H = canvas.size(at)
            lin, valid = self._leaf_lin(lay, root, at, (0, 0, W, H), canvas)
            x = self._lin_mix(lay, root, lin)
            hit = image_stats(x, valid if valid.any() else np.ones_like(valid))
            self._fstats[key] = hit
            while len(self._fstats) > 16:
                self._fstats.pop(next(iter(self._fstats)))
        return hit

    def _lin_mix(self, lay, root, lin: np.ndarray) -> np.ndarray:
        """(3, h, w): the mix's linear picture (its Output, as the run mixed it) from its leaves' planes."""
        import torch
        idx = {leaf: i for i, leaf in enumerate(self._leaves(lay, root))}
        T = torch.from_numpy(np.ascontiguousarray(lin))
        y, _pre = self._eval(lay, pe.mix_at(lay, root), root, lambda p, r, k: T[idx[(p, r, k)]], 1.0, 0.0)
        return y.numpy()

    # ------------------------------------------------------------ the stretch
    def _parts_key(self, lay, root, level, box, canvas, draft: bool = False) -> tuple:
        """What the stretched planes of a mix (``_mix_parts``) depend on: the layer as ``_key`` says it, its sliders left
        out, and in their place what the colour-preserving stretch reads of its settings (the starless side's); the stars
        side's, the engine's picture of the stars."""
        node = pe.mix_at(lay, root)
        if root == "starless":
            return self._key_of(lay, "starless", level, box, canvas, {}) + (
                pe._stretch_reads(self.provider.fusion_settings(lay)), self._struct(node)) + (
                ("draft",) if draft else ())
        return (self._key_of(lay, "stars", level, box, canvas), pe.nest_core(node), self._struct(node))

    def _key_of(self, lay, path, level, box, canvas, sliders=None) -> tuple:
        sub = pe.sub_layer(lay, path)
        if sliders is not None:
            sub = dict(sub, sliders=sliders)
        return self._key(sub, level, box, canvas)

    def _mix_parts(self, lay, root, level, box, canvas, cache, draft: bool = False) -> np.ndarray:
        """(L + 2, h, w): the mix's leaves stretched (each its balanced linear plane, minus the black point, times the
        factor: before its intensity), the factor, the covered pixels (1, 0 elsewhere). The factor is the
        colour-preserving stretch of the mix (render/simple.SimpleRenderEngine.colour_factor) on the outermost mix's
        statistics: the same on every leaf, so the mix of the stretched leaves is the stretched mix. The stars side's
        planes are those times ``G``, the engine's stars picture over the linear mix. Made with the processor's threads
        (torch), from the planes kept apart from the hues (``_leaf_lin``)."""
        # at full resolution not kept: four or five planes of the whole image would push out of the cache the images a
        # change does not touch (the other side, the stretched planes), and a hue changes them all anyway
        cache = cache and level > 0
        key = ("parts", root) + self._parts_key(lay, root, level, box, canvas, draft)
        holder = (lay["id"], (pe.FUSION, "parts", root), level)
        hit = self.images.get(key) if cache else None
        if hit is not None:
            self.images.hold(key, holder)
            return hit
        import torch

        from ..render.simple import SimpleRenderEngine
        lin, valid = self._leaf_lin(lay, root, level, box, canvas, cache)
        n, h, w = lin.shape
        L = torch.from_numpy(np.ascontiguousarray(lin))
        x = torch.from_numpy(self._lin_mix(lay, root, lin))
        vf = torch.from_numpy(valid.astype(np.float32))
        out = torch.empty((n + 2, h, w), dtype=torch.float32)
        if root == "starless":
            st = self._fusion_stats(lay, root, canvas, draft, level)
            F, c0 = SimpleRenderEngine.colour_factor(x, st, self.provider.fusion_settings(lay))
            torch.sub(L, float(c0), out=out[:n])
        else:
            E = self.layer_rgb(pe.sub_layer(lay, "stars"), level, box, canvas, cache)
            lx = 0.2126 * x[0] + 0.7152 * x[1] + 0.0722 * x[2]
            lE = torch.from_numpy(np.ascontiguousarray(pe.lum(E)))
            F = torch.where(lx > 1e-9, lE / lx.clamp(min=1e-9), torch.zeros_like(lx))
            out[:n] = L
        del x
        out[:n].mul_((F * vf)[None])
        out[n] = F
        out[n + 1] = vf
        hit = out.numpy()
        if cache:
            self.images.put(key, hit, holder)
        return hit

    def _c0(self, lay, root, canvas, draft: bool = False, level: int = 0) -> float:
        """The black point the stretch subtracts (the starless side's, raised by its Black point setting as
        ``colour_factor`` raises it; the stars side's has none of its own)."""
        if root != "starless":
            return 0.0
        from ..render.simple import _anchors
        black = float(getattr(self.provider.fusion_settings(lay), "black", 0.0))
        return float(_anchors(self._fusion_stats(lay, root, canvas, draft, level)["lum"], black)[0])

    def _eval(self, lay, node, path, leaf, one, c0, hook=None, rec=None):
        """(y, pre) of the mix at ``path``: its picture (3, h, w), made of its roles' images, and each role's image before
        its own filters. ``leaf(path, role, kind)``: a plane of the mix's leaves, stretched (``one`` the factor times
        the covered pixels, ``c0`` the black point) or linear (``one`` 1, ``c0`` 0); ``hook``: what the node's eye, its
        filters and mask do to a role's image (``hide``, ``apply``), None: nothing; ``rec``: {path: the image of every
        role before its filters}, to be filled."""
        import torch
        f = node["fusion"]
        roles = f["roles"]
        w = f.get("weights") or {}
        mat = torch.from_numpy(np.asarray(f["matrix"], np.float32))
        kids = pf.mix_kids(node)
        info = self._form(lay, node) if kids else None
        der = pf.derived(node, info) if kids else {}
        pics: dict = {}
        pre: dict = {}
        for r in roles:
            cp = pf.child(path, r)
            if hook is not None and hook.hidden(cp):
                pre[r] = None
                continue
            if r in der:
                coef, off = der[r]
                acc, tot = None, 0.0
                for k, a in coef.items():
                    if k in kids:
                        if k not in pics:
                            pics[k] = self._eval(lay, kids[k], pf.child(path, k), leaf, one, c0, hook, rec)[0]
                        row = pf.channel_row(info["channels"].get(k, "L"))
                        y = pics[k]
                        s = float(row[0]) * y[0] + float(row[1]) * y[1] + float(row[2]) * y[2]
                    else:
                        s = leaf(path, k, "read")
                    acc = float(a) * s if acc is None else acc + float(a) * s
                    tot += float(a)
                q = acc + one * (float(off) + c0 * (tot - 1.0))
            else:
                q = leaf(path, r, "bal")
            wr = float(w.get(r, 1.0))
            p = q if wr == 1.0 else wr * q + ((wr - 1.0) * c0) * one
            pre[r] = p
            if rec is not None:
                rec[cp] = p
        done = []
        for r in roles:
            p = pre[r]
            if hook is not None:
                p = hook.apply(pf.child(path, r), path, p, pre)
            done.append(p)
        K = torch.stack(done)
        n, h, w2 = K.shape
        y = (mat @ K.reshape(n, -1)).reshape(3, h, w2)
        if c0 != 0.0:                                     # (a row that does not sum to 1 takes the black point back)
            lost = 1.0 - mat.sum(1)
            for c in range(3):
                if abs(float(lost[c])) > 1e-7:
                    y[c].sub_(one * (c0 * float(lost[c])))
        return y, pre

    def _pass(self, lay, root, level, abox, canvas, cache, draft: bool = False):
        """What a render of the mix at ``root`` over ``abox`` works from: (its stretched planes cut to the box, ``leaf``,
        the factor times the covered pixels, the black point, the covered pixels)."""
        import torch
        W, H = canvas.size(level)
        whole = (0, 0, W, H)
        parts = self._mix_parts(lay, root, level, whole if level > 0 else abox, canvas, cache, draft)
        if level > 0 and abox != whole:
            x0, y0, x1, y1 = abox
            parts = parts[:, y0:y1, x0:x1]
        idx = {leaf: i for i, leaf in enumerate(self._leaves(lay, root))}
        n = len(idx)
        T = torch.from_numpy(np.ascontiguousarray(parts))
        return T, (lambda p, r, k: T[idx[(p, r, k)]]), T[n] * T[n + 1], self._c0(lay, root, canvas, draft, level), T[n + 1]

    def _mix_end(self, lay, root, y, valid) -> np.ndarray:
        """The end of the stretch on a mix's picture (Contrast, Saturation: render/simple.SimpleRenderEngine.finish,
        which never changes ``y``), outside the covered pixels black; (h, w, 3). The stars side's picture is as the engine
        made it, no end of its own."""
        import torch
        if root == "starless":
            from ..render.simple import SimpleRenderEngine
            y = SimpleRenderEngine.finish(y, self.provider.fusion_settings(lay))
        y.mul_(torch.from_numpy(np.asarray(valid >= 0.5, np.float32))[None])
        return y.permute(1, 2, 0).contiguous().numpy()

    def _default(self, lay, root, level, abox, canvas, cache, draft: bool = False):
        """(the mix's picture as its run made it, before the end of its stretch (3, h, w), clamped; the covered pixels;
        {path: the image of each node before its filters}): all nodes shown, none with a filter."""
        T, leaf, one, c0, vf = self._pass(lay, root, level, abox, canvas, cache, draft)
        rec: dict = {}
        y, _pre = self._eval(lay, pe.mix_at(lay, root), root, leaf, one, c0, None, rec)
        y = y.clamp(min=0)
        return y, vf, rec

    def _fusion_rgb(self, lay, level, abox, canvas, cache, draft: bool = False) -> np.ndarray:
        """The image of a node of a layer's tree (``sub_layer``: ``node``) over the box at the level: a side that is a
        mix, its picture with all its nodes as they are (cached as a workflow layer's image is), or a node, its grey
        image as its mix makes it. A node's image is not kept: it is one plane of the planes kept, made again at once, and
        it took the room of the images that are long to make."""
        node = lay.get("node") or "starless"
        root = pf.root_of(node)
        W, H = canvas.size(level)
        whole = (0, 0, W, H)
        want = whole if level > 0 else abox
        cache_parts, cache = cache, cache and node == root
        if draft and level > 0 and self.images.get(self._key(lay, level, abox, canvas)) is None:
            cache = False                                 # (a quick look with its own statistics: never kept as the exact picture)
        key = self._key(lay, level, abox, canvas)
        holder = (lay["id"], (pe.FUSION, node), level)
        a = self.images.get(key) if cache else None
        if a is None:
            y, vf, rec = self._default(lay, root, level, want, canvas, cache_parts, draft)
            if node == root:
                a = self._mix_end(lay, root, y, vf.numpy() if hasattr(vf, "numpy") else vf)
            else:
                a = pe.grey3(rec[node].numpy())
            self._drawn += 1
            if cache:
                self.images.put(key, a, holder)
        elif cache:
            self.images.hold(key, holder)
        if level > 0 and abox != whole:
            x0, y0, x1, y1 = abox
            a = a[y0:y1, x0:x1]
        return a

    # ------------------------------------------------------------ digests
    def _node_sig(self, lay, own, path, level, abox, canvas) -> str:
        """What the image of a node of the tree depends on before its own filters: its key and, for a combination, what
        its inputs' eyes and filters do (the same digest in the plan and in the composition: no pixel is read)."""
        sig = pe._sig(self._key(pe.sub_layer(lay, path), level, abox, canvas))
        n = pe.mix_at(lay, path)
        if n is None:
            return sig
        return pe._sig(sig, [self._after_sig(lay, own, pf.child(path, r), level, abox, canvas)
                             for r in n["fusion"]["roles"]])

    def _after_sig(self, lay, own, path, level, abox, canvas) -> str:
        """What a node's image depends on after its eye and its filters (the steps' digests); its mask is not in it (the
        layer's settings say it)."""
        if not (lay.get("shown") or {}).get(path, True):
            return pe._sig("off", path)
        sig = self._node_sig(lay, own, path, level, abox, canvas)
        for fl in own:
            if fl.get("sub") == path and fl["visible"]:
                sig = pe._sig(sig, pe._settings_of(fl))
        return sig

    def _mix_sig(self, lay, own, root, level, abox, canvas, draft=False) -> str:
        """What a side that is a mix depends on (``_mix_side``), made without computing it."""
        if root == "starless":
            base = pe._sig(self._key(pe.sub_layer(lay, "starless"), level, abox, canvas))
        else:
            base = self._img_sig(pe.sub_layer(lay, "stars"), level, abox, canvas, draft)
        if self._mix_plain(lay, own, root):
            return base
        n = pe.mix_at(lay, root)
        return pe._sig(base, [self._after_sig(lay, own, pf.child(root, r), level, abox, canvas)
                              for r in n["fusion"]["roles"]])

    def _mix_key(self, lay, own, root, level, abox, canvas, draft) -> tuple:
        """What a side's picture before the end of its stretch depends on (``_mix_blend``): the stretched planes
        (``_parts_key``), each node's eye, filters and mask, the level, the box, a draft; and Contrast and Saturation when
        a node's mask reads the picture (the filters mixed, the stretch ended)."""
        W, H = canvas.size(level)
        whole = (0, 0, W, H)
        sides = lay.get("side_masks") or {}
        reads = []

        def per(path):
            out = []
            for r in pe.mix_at(lay, path)["fusion"]["roles"]:
                cp = pf.child(path, r)
                side = sides.get(cp)
                if any(p.get("src") in pe.READS_IMAGE and p.get("from") == "picture" for p in (side or {}).get("parts") or ()):
                    reads.append(cp)
                out.append((cp, (lay.get("shown") or {}).get(cp, True),
                            pe._settings_of({"side_masks": {cp: side}}) if side else "",
                            [(pe._settings_of(fl), fl["visible"]) for fl in own if fl.get("sub") == cp],
                            per(cp) if pe.mix_at(lay, cp) is not None else None))
            return out
        state = per(root)
        fin = pe._finish_reads(self.provider.fusion_settings(lay)) if reads and root == "starless" else None
        return (pe._sig(self._parts_key(lay, root, level, whole if level > 0 else abox, canvas, draft), state, fin),
                level, tuple(abox), bool(draft))

    # ------------------------------------------------------------ a side that is a mix
    def _mix_side(self, lay, own, root, level, abox, canvas, ctx, cache, draft, check, keep_mask, where, want, thumbs):
        """A side's picture (its mix of the layer's tree): each node through its eye, its filters (bottom first) and its
        mask, then the mix; all as the engine made it when nothing is changed."""
        if self._mix_plain(lay, own, root):
            return self.layer_rgb(pe.sub_layer(lay, root), level, abox, canvas, cache, draft, check)
        under = self._under(lay, root)
        # kept, by all it depends on (``_mix_key``), when none of its masks or histograms is asked for: a change above
        # it (a filter on the layer, its mask, the other side) finds it again; kept before the end of the stretch, so
        # that Contrast and Saturation find it too when no mask reads the picture. Its mask thumbnails with it.
        mine = {(lay["id"], p) for p in under} | {fl["id"] for fl in own if fl.get("sub") in under}
        key = None
        if cache and not (mine & set(want)) and not (mine & set(ctx.hist_ids)):
            key = ("fusion side", root) + self._mix_key(lay, own, root, level, abox, canvas, draft)
            hit = self.images.get(key)
            if hit is not None and (not thumbs or key in self._side_thumbs):
                self.images.hold(key, (lay["id"], (pe.FUSION, "side", root), level))
                for k, t in self._side_thumbs.get(key, ()) if thumbs else ():
                    keep_mask(lay, None, k, thumb=t)
                return self._mix_end(lay, root, pe._torch_of(hit.y), hit.valid)
        made = []

        def keep(owner, m, k=None):
            t = keep_mask(owner, m, k)
            if t is not None:
                made.append((owner["id"] if k is None else k, t))
        y, valid = self._mix_blend(lay, own, root, level, abox, canvas, ctx, cache, draft, check, keep, where, want,
                                   thumbs)
        held = pe._SideHeld(y.numpy(), np.asarray(valid) >= 0.5) if key is not None else None
        holder = (lay["id"], (pe.FUSION, "side", root), level)
        # at full resolution kept only where there is room, never instead of another image: the screen's images
        # (the other side, the stretched planes) are the ones a release needs first
        if key is not None and not self._stars_missing and (level > 0 or self.images.room(holder) >= held.nbytes):
            self.images.put(key, held, holder)
            if thumbs:
                self._side_thumbs[key] = made
                self._side_thumbs.move_to_end(key)
                while len(self._side_thumbs) > pe.THUMB_ITEMS:
                    self._side_thumbs.popitem(last=False)
        return self._mix_end(lay, root, y, valid)

    def _mix_blend(self, lay, own, root, level, abox, canvas, ctx, cache, draft, check, keep_mask, where, want, thumbs):
        """``_mix_side``'s work: (the mix before the end of its stretch, a tensor (3, h, w); the covered pixels)."""
        import torch
        hw = (abox[3] - abox[1], abox[2] - abox[0])
        node = pe.mix_at(lay, root)
        T, leaf, one, c0, vf = self._pass(lay, root, level, abox, canvas, cache, draft)
        sides, shown = lay.get("side_masks") or {}, lay.get("shown") or {}
        memo: dict = {}
        comp = self
        rec: dict = {}
        stars = root == "stars"

        def default_of(path):
            """A mix's picture as its run made it, before the end of its stretch, over the box: numpy (h, w, 3)."""
            if path not in memo:
                if path == root and level > 0 or stars and path == root:
                    memo[path] = comp.layer_rgb(pe.sub_layer(lay, root), level, abox, canvas, cache, draft, check)
                else:
                    y, _pre = comp._eval(lay, pe.mix_at(lay, path), path, leaf, one, c0, None)
                    memo[path] = comp._mix_end(lay, "stars" if stars else "starless", y.clamp(min=0), vf.numpy())
            return memo[path]

        class Hook:
            """What a node's eye, filters and mask do to its image before its parent mixes it."""

            @staticmethod
            def hidden(cp):
                return not shown.get(cp, True)

            @staticmethod
            def apply(cp, parent, p, pre):
                side = sides.get(cp)
                chain = [fl for fl in own if fl.get("sub") == cp]
                if p is None:                              # its eye off: out of the mix; a mask asked for is shown
                    if side is not None and ((lay["id"], cp) in want or thumbs):
                        z = np.zeros((*hw, 3), np.float32)
                        src, sigs = sources(cp, parent, z, pe._sig("off", hw), pre)
                        keep_mask(lay, comp.mask({"mask": side}, level, abox, z, canvas, ctx, src, sigs), (lay["id"], cp))
                    return torch.zeros(hw, dtype=torch.float32)
                if not chain and side is None:
                    return p
                x = pe.grey3(p.numpy())
                sig = comp._node_sig(lay, own, cp, level, abox, canvas)
                for fl in chain:
                    if check():
                        raise pe.Superseded()
                    if not fl["visible"]:
                        if fl["id"] in want or (thumbs and fl["mask"] is not None):
                            keep_mask(fl, comp.mask(fl, level, abox, x, canvas, ctx, sigs={None: sig}))
                        continue
                    ctx.stats_key = pe._sig(ctx.row_edit, fl["id"])
                    x, fm, sig = comp._step(fl, x, sig, where, want, thumbs, level, abox, canvas, ctx)
                    keep_mask(fl, fm)
                if side is not None:
                    src, sigs = sources(cp, parent, x, sig, pre)
                    m = comp.mask({"mask": side}, level, abox, x, canvas, ctx, src, sigs)
                    keep_mask(lay, m, (lay["id"], cp))
                    x = x * m[..., None]
                return torch.from_numpy(np.ascontiguousarray(pe.lum(x)))

        def sources(cp, parent, x, sig, pre):
            """What the parts of a node's mask that read an image read, each made when a part reads it: the node itself
            (the default, as every filter's), the picture (its mix as it is), another filter of the mix (stretched), a
            filter of its own; and what each depends on (``mask``'s ``sigs``)."""
            makers = {"self": lambda: x, "picture": lambda: default_of(parent)}
            sigs = {"self": sig, "picture": pe._sig(comp._key(pe.sub_layer(lay, parent), level, abox, canvas))}
            above = pe.mix_at(lay, parent)
            for o in above["fusion"]["roles"]:
                op = pf.child(parent, o)
                if op != cp:
                    makers[op] = lambda op=op, o=o: pe.grey3(pre[o].numpy()) if pre.get(o) is not None else \
                        np.zeros((*hw, 3), np.float32)
                    sigs[op] = comp._node_sig(lay, own, op, level, abox, canvas)
            inner = pe.mix_at(lay, cp)
            if inner is not None:
                for o in inner["fusion"]["roles"]:
                    op = pf.child(cp, o)
                    makers[op] = lambda op=op: pe.grey3(rec[op].numpy()) if rec.get(op) is not None else \
                        np.zeros((*hw, 3), np.float32)
                    sigs[op] = comp._node_sig(lay, own, op, level, abox, canvas)
            return pe._LazySources(makers), sigs

        hook = Hook()
        y, _pre = self._eval(lay, node, root, leaf, one, c0, hook, rec)
        if stars:                                          # the engine's picture, and what was changed in it
            E = torch.from_numpy(np.ascontiguousarray(
                self.layer_rgb(pe.sub_layer(lay, "stars"), level, abox, canvas, cache, draft, check))).permute(2, 0, 1)
            y0, _p = self._eval(lay, node, root, leaf, one, c0, None)
            y = E + y - y0
        y.clamp_(min=0)
        return y, vf.numpy()

    def _side_sources(self, lay, sub, side, x, got, level, abox, canvas, cache, draft: bool = False) -> dict:
        """What a part of a side's mask that reads an image may read, the same choice as a node's mask: the image itself
        first (what it read before the choice), the picture (the filters mixed, without the stars), each filter of the
        starless mix and of the mix of this side (stretched)."""
        out = {"self": x}
        out["picture"] = got[0] if got else x                 # (the starless image comes first in the group)
        wanted = {p.get("from") for p in side.get("parts") or []}
        for root in ("starless", "stars"):
            node = pe.mix_at(lay, root)
            if node is None or root not in ("starless", sub):
                continue
            paths = {r: pf.child(root, r) for r in node["fusion"]["roles"]}
            if wanted & set(paths.values()):
                _y, _vf, rec = self._default(lay, root, level, abox, canvas, cache, draft)
                for r, cp in paths.items():
                    if rec.get(cp) is not None:
                        out[cp] = pe.grey3(rec[cp].numpy())
        return out

    def _fusion_halpha_of(self, lay, level, abox, canvas, cache):
        """The H-alpha map of a layer whose starless side is a combination with an H-alpha filter (a function, computed
        when a mask asks for it), None for any other layer: the nearest mix of the tree that holds the filter, its
        balanced image through the rule of the emission map of Post's other layers (render/layers.LayerService.
        _halpha: 0 at its median, 1 at its 97th percentile, both measured on the whole field, smoothly between)."""
        if not pe.is_group(lay):
            return None
        hit = None
        for p in sorted(self._under(lay, "starless") + ["starless"], key=pf.depth_of):
            n = pe.mix_at(lay, p)
            if n is not None and "Ha" in n["fusion"]["roles"]:
                hit = n
                break
        if hit is None:
            return None

        def fn():
            import torch

            from ..render.simple import _smoothstep              # the map's curve (render/layers)
            at = pe.stats_level(canvas.W, canvas.H)
            W, H = canvas.size(at)
            v = self._plane(lay, hit, "Ha", at, (0, 0, W, H), canvas, cache)
            vs = v[0][v[1]]
            q0, q1 = (float(x) for x in np.percentile(vs, [50, 97])) if vs.size else (0.0, 1.0)
            Wl, Hl = canvas.size(level)
            whole = (0, 0, Wl, Hl)
            e, _valid = self._plane(lay, hit, "Ha", level, whole if level > 0 else abox, canvas, cache)
            if level > 0 and abox != whole:
                x0, y0, x1, y1 = abox
                e = e[y0:y1, x0:x1]
            return _smoothstep(torch.from_numpy(np.ascontiguousarray(e)), q0, q1).numpy()
        return fn

    def _plane(self, lay, node, role, level, box, canvas, cache):
        """(a combination's balanced filter (h, w), before its intensity; the covered pixels), kept like the leaves'."""
        cache = cache and level > 0
        sid = (pe.FUSION, node["pid"])
        key = ("plane", sid, self.provider.version(sid), role, node["fusion"].get("balance"), level,
               None if level > 0 else tuple(box), (canvas.W, canvas.H, canvas.origin))
        holder = (lay["id"], (pe.FUSION, "plane", role), level)
        hit = self.images.get(key) if cache else None
        if hit is None:
            planes, valid = self.provider.fusion_planes(self._nlay(lay, node), level, box)
            hit = np.empty((2, *np.asarray(valid).shape), np.float32)
            hit[0] = planes[role]
            hit[1] = np.asarray(valid, np.float32)
            if cache:
                self.images.put(key, hit, holder)
        elif cache:
            self.images.hold(key, holder)
        return hit[0], hit[1] > 0.5
