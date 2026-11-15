"""« Add lights » proposed the home folder after files were dropped, and the name proposed at the first Run was the
frames' folder's name (« lights »). A drop is a use: the next dialog opens where it came from; a project's name is
never taken from a folder or a file: the field is empty and focused unless the user gave a name or a target."""
from __future__ import annotations

from types import SimpleNamespace

from astro_gpu.core.project import stage_graph
from astro_gpu.core.workflow import chains
from tests.client.graphview.test_gv_files import one_session_bridge, settle
from tests.client.graphview.test_gv_table import fwd, make

stage_graph()             # the engine's stages loaded at collection, not in the first test's time
chains()


def test_gv_a_drop_is_where_the_next_dialog_opens(qapp, tmp_path):
    from astro_gpu.client import prefs
    night = make(tmp_path / "night" / "lights", 2, "L")
    bridge, sid = one_session_bridge()
    bridge.drops.drop(sid, "lights", [fwd(night / "L_0000.fit"), fwd(night / "L_0001.fit")])
    settle(bridge.drops.reader)
    assert prefs.last_folder() == fwd(night)


def test_gv_no_project_name_from_a_folder(qapp, tmp_path):
    from astro_gpu.client.app import Main
    from astro_gpu.client.home import NewProjectDialog
    lights = make(tmp_path / "lights", 1, "L")
    p = SimpleNamespace(sessions=[SimpleNamespace(lights=[fwd(lights / "L_0000.fit")])], workflows=[],
                        abspath=lambda e: e)
    s = SimpleNamespace(name="", p=p, target_state=lambda: SimpleNamespace(name=""))
    assert Main._suggested_name(SimpleNamespace(s=s)) == ""
    s.target_state = lambda: SimpleNamespace(name="M 31")
    assert Main._suggested_name(SimpleNamespace(s=s)) == "M 31"           # what the user gave
    d = NewProjectDialog(str(tmp_path), "")
    d.show()
    qapp.processEvents()
    import datetime
    proposed = "Project " + datetime.date.today().isoformat()     # never an object's name the frames may not be of
    assert d.name.text() == proposed and d.name.selectedText() == proposed and d.focusWidget() is d.name
    assert d.ok.isEnabled()                        # a real default name, selected: Save is not greyed
    d.reject()
