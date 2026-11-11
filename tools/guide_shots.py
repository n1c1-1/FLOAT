"""The screenshots of the README's "First steps" guide, offscreen, in English, 1600 x 900.

    python tools/guide_shots.py --demo "F:/FLOAT demo" --out FOLDER

``--demo`` is a folder with neutral names: ``frames/{lights,darks,flats}`` (a few raw frames), ``Heart Nebula`` (a computed
project whose sources are those frames), and it receives its drafts in a sibling folder "<demo> drafts", the project made from scratch. The
shots are raw (whole window); ``tools/guide_crop.py`` crops them. CPU only, a private settings file and state folder.
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import client_walkthrough as cw  # noqa: E402


def main():
    import faulthandler
    faulthandler.dump_traceback_later(900, exit=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    demo, out = Path(a.demo), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="guide_"))
    cw.setup_environment(tmp)
    draft = demo.parent / (demo.name + " drafts")
    draft.mkdir(exist_ok=True)
    walk = cw.Walk(out, demo, True, tmp)
    from astro_gpu.client import prefs
    prefs.set("home_sections", [{"label": "Projects", "folder": demo.as_posix(), "open": True}])
    prefs.set("last_folder", draft.as_posix())
    w = walk.w
    w.resize(1600, 900)
    w.show_home()
    walk.pump(2.0)
    walk.shot("1_home")
    # a project from scratch: the template, the frames, the plan
    w.new_project()
    walk.wait_ready()
    walk.pump(0.3)
    w.graph.show_templates()
    walk.settle()
    walk.shot("2_gallery")
    next(c for c in w.graph.templates.cards if c.t["key"] == "colour_full").mousePressEvent(None)
    walk.settle()

    def named(d):
        d.location = draft
        d.name.setText("Heart Nebula")
        d.ok.click()
    walk.modal(w.save, named, "Name the project")
    walk.wait_ready()
    sid = w.s.p.sessions[0].id
    for kind in ("lights", "darks", "flats"):
        w.s.add_session_entries(sid, kind, [str(demo / "frames" / kind)])
    walk.wait_ready()
    walk.select("sources")
    w.set_mode(1)
    w.frames.show_page(0)
    walk.settle()
    walk.until(lambda: w.frames.data is not None, 120, "the sessions table")
    walk.pump(1.0)
    walk.shot("3_frames")
    w.set_mode(0)
    walk.select("empilement", w.s.paths[0].pid)
    w.run_to_here()
    walk.until(lambda: w.run.dialog is not None and w.run.dialog.isVisible(), 30, "the plan dialog")
    walk.pump(1.0)
    walk.shot("4_run", w.run.dialog)
    w.run.dialog.close()
    # the computed project
    w.open_project(demo / "Heart Nebula")
    walk.wait_ready()
    w.set_mode(0)
    walk.select("post", w.s.paths[0].pid)
    walk.pump(1.0)
    walk.shot("5_graph")
    walk.inspect("empilement", w.s.paths[0].pid, "6_inspect")
    w.set_mode(2)
    walk.settle()
    walk.until(lambda: w.render is not None and not w.render.busy(), 300, "the Post render")
    walk.shot("7_post")
    print("problems:", walk.problems)
    print("messages:", walk.messages[:5])


if __name__ == "__main__":
    main()
