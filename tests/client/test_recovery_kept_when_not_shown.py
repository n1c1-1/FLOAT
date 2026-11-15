"""A recovery after a crash that fails to be shown keeps its copy: the copy is this window's only once the project's
screen is built. It was marked this window's before the build, so the close that follows the failure erased it and
the work not saved was lost for good; now the copy is still offered."""
from __future__ import annotations

from PySide6.QtWidgets import QApplication

from astro_gpu.client import app as appmod
from astro_gpu.client import drafts
from tests.client.test_new_page_asks_to_save import qapp, win  # noqa: F401 - the fixtures


def test_a_recovery_that_cannot_be_shown_keeps_its_copy(win, tmp_path, monkeypatch):
    from astro_gpu.core import project2
    from astro_gpu.core.project import Target
    root = tmp_path / "ws" / "Crashed"
    p = project2.create(root, "Crashed", Target(name="M 31"))
    project2.save(p)
    p.target.name = "M 42"                                           # the edit the crash left unsaved
    drafts.write_recovery(root, "Crashed", p.to_dict())
    rec = next(r for r in drafts.recoveries() if str(r["root"]).lower() == root.as_posix().lower())

    def broken(self):
        raise RuntimeError("a widget failed")
    monkeypatch.setattr(appmod.Main, "build_project", broken)
    win.reopen_recovered(rec)
    QApplication.processEvents()
    assert win.project is None                                       # not shown, said in the banner
    assert any(str(r["root"]).lower() == root.as_posix().lower() for r in drafts.recoveries())   # the copy is kept
