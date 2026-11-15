"""The settings file: an unreadable one is kept aside (settings.json.bad, never overwritten) and said once in the
window's banner, a valid JSON that is not an object does not stop the start, a failed write is said, and two windows
writing the same file keep each other's changes (the key being changed wins, the others are read from the disk)."""
from __future__ import annotations

import json
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from astro_gpu.client import prefs  # noqa: E402


@pytest.fixture()
def said(tmp_path, monkeypatch):
    """The settings in tmp_path; what the settings say of their file, collected."""
    monkeypatch.setenv(prefs.ENV, str(tmp_path / "settings.json"))
    prefs.watch(None)                                  # a window of an earlier test is told nothing more
    out = []
    yield out
    prefs.watch(None)


def test_an_unreadable_file_is_kept_aside_and_said_once(tmp_path, said):
    f = tmp_path / "settings.json"
    f.write_text("{ not json", encoding="utf-8")
    (tmp_path / "settings.json.bad").write_text("an older one", encoding="utf-8")
    prefs.load()
    assert prefs.get("language") == "en"                                     # the defaults
    assert (tmp_path / "settings.json.bad").read_text(encoding="utf-8") == "an older one"   # never overwritten
    assert (tmp_path / "settings.json.bad2").read_text(encoding="utf-8") == "{ not json"
    prefs.watch(lambda kind, detail: said.append((kind, detail)))
    assert said == [("unreadable", "settings.json.bad2")]
    prefs.set("language", "fr")                                              # a new file, the bad one kept
    assert json.loads(f.read_text(encoding="utf-8"))["language"] == "fr"
    assert said == [("unreadable", "settings.json.bad2")]                    # said once


def test_a_json_that_is_not_an_object_does_not_stop_the_start(tmp_path, said):
    f = tmp_path / "settings.json"
    f.write_text("[1, 2]", encoding="utf-8")
    prefs.load()
    assert prefs.get("recent") == [] and prefs.recent() == []
    assert (tmp_path / "settings.json.bad").read_text(encoding="utf-8") == "[1, 2]"
    prefs.watch(lambda kind, detail: said.append(kind))
    assert said == ["unreadable"]


def test_a_failed_write_is_said(tmp_path, said, monkeypatch):
    prefs.load()
    prefs.watch(lambda kind, detail: said.append(kind))

    def refuse(*_a):
        raise PermissionError("Access is denied")
    monkeypatch.setattr(prefs.os, "replace", refuse)
    prefs.set("language", "fr")
    prefs.set("units", "px")
    assert said == ["unwritable"]                                            # once, not at every change
    monkeypatch.undo()
    monkeypatch.setenv(prefs.ENV, str(tmp_path / "settings.json"))
    prefs.set("units", "arcsec")
    assert json.loads((tmp_path / "settings.json").read_text(encoding="utf-8"))["units"] == "arcsec"


def test_two_windows_keep_each_other_s_changes(tmp_path, said):
    f = tmp_path / "settings.json"
    prefs.load()
    prefs.set("language", "fr")
    disk = json.loads(f.read_text(encoding="utf-8"))                          # another window adds a template
    disk["templates"] = [{"key": "mine", "name": "Mine"}]
    f.write_text(json.dumps(disk), encoding="utf-8")
    prefs.set("units", "px")                                                 # this window changes another setting
    after = json.loads(f.read_text(encoding="utf-8"))
    assert after["templates"] == [{"key": "mine", "name": "Mine"}] and after["units"] == "px"
    assert after["language"] == "fr" and prefs.get("templates") == [{"key": "mine", "name": "Mine"}]


def test_the_window_says_it_in_its_banner_in_both_languages(tmp_path, said):
    from astro_gpu.client import app as appmod
    QApplication.instance() or QApplication([])
    (tmp_path / "settings.json").write_text("{ not json", encoding="utf-8")
    prefs.load()
    appmod.apply_language("fr")
    try:
        w = appmod.Main()
        w.show()
        QApplication.processEvents()
        assert w.banner.isVisible() and "settings.json.bad" in w.banner.text.text()
        assert "réglages" in w.banner.text.text()
        for kind, detail in (("unwritable", "PermissionError: Access is denied"), ("unreadable", "")):
            w._settings_trouble(kind, detail)
            assert "réglages" in w.banner.text.text()
        from astro_gpu.client import i18n
        assert not i18n.missing(), sorted(i18n.missing())
        w.close()
        w.deleteLater()
        QApplication.processEvents()
    finally:
        appmod.apply_language("en")
