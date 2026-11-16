"""A folder that holds no project, given to the window to open (the command line, a usability session's --project):
the notice names the folder, never the whole path of the user's disk (it was read on a tutorial's screen)."""
from __future__ import annotations

from tests.client.test_swap_smoke import env, qapp, settle, win  # noqa: F401  (fixtures by name)


def test_the_notice_of_a_folder_without_project_names_the_folder_only(win, env):
    folder = env["tmp"] / "deep" / "My night"
    folder.mkdir(parents=True)
    assert win.open_project(folder.as_posix()) is False
    settle(win, 3)
    said = win.banner.text.text()
    assert win.banner.isVisible() and said == "No project in this folder: My night", said
    assert win.banner.text.toolTip() == folder.as_posix()        # where it is: on hover
