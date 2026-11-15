"""The stop cross of a reading and the « Rather add » link take back the drop that made them, never another gesture:
the cross of a reading started by the project itself (a folder entry it holds) undid the last real edit, whatever it
was, and « Rather add » undid whatever came last."""
from tests.client.graphview.test_gv_table import Bench, fwd, make, tree  # noqa: F401 - the fixture


def test_the_stop_of_a_reading_no_drop_started_keeps_the_last_gesture(qapp, tmp_path):
    folder = fwd(make(tmp_path / "lights", 3, "L"))
    b = Bench(1366, delay=0.8)
    sid = b.sessions()[0]
    rid = b.stacks()[0]["id"]
    b.reader.note(folder, True)
    assert b.bridge.act("add_files", quiet=True, session=sid, kind="lights", paths=[folder], mode="replace").ok
    assert b.bridge.act("rename_row", quiet=True, row=rid, name="Kept").ok          # the last real edit
    b.g.after_model()                                                                 # its reading starts
    b.wait(0.1)
    assert "stop" in b.g.zones(b.g.cell(("box", sid, "lights")))
    n0 = b.steps()
    b.click(("box", sid, "lights"), zone="stop")
    b.reader.delay = 0.0
    b.read()
    assert b.steps() == n0 and b.stacks()[0]["name"] == "Kept" and b.own(sid, "lights") == [folder]
    b.close()


def test_rather_add_after_another_gesture_takes_nothing_back(qapp, tree):
    b = Bench(1920)
    sid = b.sessions()[0]
    rid = b.stacks()[0]["id"]
    b.drop([tree["flats"]], ("box", sid, "flats"))
    b.read()
    b.drop([tree["flats2"]], ("box", sid, "flats"), fx=0.8)                          # a replacement
    b.read()
    assert "rather_add" in b.t.toast_w.text()
    assert b.bridge.act("rename_row", quiet=True, row=rid, name="Kept").ok            # a gesture made since
    b.g.after_model()
    n0 = b.steps()
    b.t.link("rather_add")
    b.read()
    assert b.stacks()[0]["name"] == "Kept" and b.steps() == n0
    b.close()
