"""To Post on a Denoise block of a channel row: the branch it makes gives a layer to the version of Post shown, once computed
(offscreen, scratch project, nothing computed: the branch is marked rendered)."""
from __future__ import annotations

from astro_gpu.client.blocks import Graph
from astro_gpu.client.post3 import PostPage3
from astro_gpu.client.services.render import RenderWorker
from astro_gpu.core import workflow as wfl

from test_fusion_sessions_client import CHAIN, open_page, qapp, settle  # noqa: F401


def _rendered(m):
    for q in m.paths:
        q.rendered = True


def test_to_post_on_a_denoise_block_of_a_channel(qapp, tmp_path, monkeypatch):
    w, m, page = open_page(tmp_path, monkeypatch, filters=("Ha", "OIII"))
    page3 = None
    try:
        for q in list(m.paths):
            m.p.set_blocks(q.pid, [wfl.node(b) for b in CHAIN + ["gradient", "denoise", "stars"]])
        m._sync()
        settle(w, page)
        chans = [{"pid": q.pid, "role": q.name, "channel": "L"} for q in m.paths]
        m.add_fusion("H-O", "ho", chans)
        settle(w, page)
        _rendered(m)
        page3 = PostPage3(m, worker=RenderWorker(None, resolver=lambda sid: None, device="cpu"))
        v = page3.view
        v.sync_paths(initial=True)
        v._version_new()
        ha = m.paths[0]
        page.send_to_post(Graph(m).node_of(m.view(ha.pid), "debruitage"))      # right-click Denoise > To Post
        settle(w, page)
        sent = m.paths[-1]
        assert sent.pid in Graph(m).to_post
        _rendered(m)                                          # the run computed the branch
        m.pathsChanged.emit()
        assert any(x.get("pid") == sent.pid for x in v.layers)
        first = m.p.post["active"]
        v._save()
        assert any(x.get("pid") == sent.pid for x in m.post_version()["layers"])
        m.set_post_version(first)
    finally:
        if page3 is not None:
            page3.shutdown()
            page3.deleteLater()
        page.close()
        m.close(save=False)
        w.stop()
