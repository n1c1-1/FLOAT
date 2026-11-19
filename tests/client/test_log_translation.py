"""Every line the engine writes to the log reaches the French window in French (a new line cannot escape).

The stages write their log lines as f-strings (``ctx.log(f"...")``, ``ctx.progress(f, f"...")``, ``ctx.fallback``,
``journal.emit("log", msg=...)``); the window translates them by ``engine_text`` (the fixed sentences by the
catalogue, the ones with numbers by ``PATTERNS``). This test reads every such call of the engine, writes the line with
sample numbers and asks ``engine_text`` for its French: a line it leaves as it is has no translation."""
from __future__ import annotations

import ast
import os
import re
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from astro_gpu.client import app as appmod  # noqa: E402
from astro_gpu.client import i18n  # noqa: E402
from astro_gpu.client.services.run import engine_text  # noqa: E402

ROOT = Path(__file__).parents[2] / "astro_gpu"
#: the engine's folders whose messages reach the user
FOLDERS = ("stages", "core", "render")
#: names of the log calls and of the position of their message
LOG_CALLS = {"log": 0, "fallback": 0, "progress": 1, "warn": 0}


def _parts(node: ast.AST) -> list | None:
    """A message expression as a list of its literal texts and of its fields (a spec: the format of the field)."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value]
    if isinstance(node, ast.JoinedStr):
        out: list = []
        for v in node.values:
            if isinstance(v, ast.Constant):
                out.append(str(v.value))
            else:
                spec = ""
                if v.format_spec is not None:
                    spec = "".join(str(x.value) for x in v.format_spec.values if isinstance(x, ast.Constant))
                out.append(("field", spec))
        return out
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        a, b = _parts(node.left), _parts(node.right)
        return a + b if a is not None and b is not None else None
    return None


def _key(parts: list) -> str:
    """The catalogue's key of a line: its text with ``{}`` for each field (locale/fr/log.json)."""
    return "".join(p if isinstance(p, str) else "{}" for p in parts)


def _sample(parts: list) -> str:
    """The line with sample numbers in its fields."""
    return "".join(p if isinstance(p, str) else
                   ("7.25" if re.search(r"\.\d+[fgeE]|[fge]$|%$", p[1]) else "7") for p in parts)


def log_lines() -> list[tuple[str, int, str, str]]:
    """(file, line, key, sample) of every log message of the engine written as a literal."""
    found = []
    for folder in FOLDERS:
        for f in sorted((ROOT / folder).rglob("*.py")):
            tree = ast.parse(f.read_text(encoding="utf-8"))
            for n in ast.walk(tree):
                if not isinstance(n, ast.Call):
                    continue
                name = n.func.attr if isinstance(n.func, ast.Attribute) else getattr(n.func, "id", "")
                msg = None
                if name == "emit" and n.args and isinstance(n.args[0], ast.Constant) and n.args[0].value == "log":
                    msg = next((k.value for k in n.keywords if k.arg == "msg"), None)
                elif name in LOG_CALLS and isinstance(n.func, ast.Attribute) and \
                        isinstance(n.func.value, ast.Name) and n.func.value.id == "ctx":
                    i = LOG_CALLS[name]
                    msg = n.args[i] if len(n.args) > i else next((k.value for k in n.keywords if k.arg == "msg"), None)
                parts = _parts(msg) if msg is not None else None
                if parts and _key(parts).strip():
                    found.append((f.relative_to(ROOT).as_posix(), n.lineno, _key(parts), _sample(parts)))
    return found


def test_every_log_line_of_the_engine_has_its_french():
    appmod.apply_language("fr")
    try:
        lines = log_lines()
        assert len(lines) > 100                       # the scan reads the engine, not nothing
        bad = [f"{f}:{ln}  {k}" for f, ln, k, s in lines
               if k not in i18n._cat and engine_text(s) == s]
    finally:
        appmod.apply_language("en")
    assert not bad, f"{len(bad)} log lines stay in English in the French window:\n" + "\n".join(bad)


def test_the_log_catalogue_keeps_the_fields_of_each_line():
    """A French line has as many fields as its English one (``{}`` in order, or ``{0}``... in another order)."""
    import json
    for k, v in json.loads((ROOT / "client" / "locale" / "fr" / "log.json").read_text(encoding="utf-8")).items():
        n = k.count("{}")
        assert v.count("{}") == n or all(f"{{{i}}}" in v for i in range(n)), k
        assert v.strip() == v and "  " not in v, k


def test_the_french_window_translates_the_lines_of_the_m27_run():
    appmod.apply_language("fr")
    try:
        assert engine_text("sky gradient: fitted on 3203 of 3551 sky samples; 22.01 across the field on a sky of "
                           "1002.18") == ("gradient du ciel : ajusté sur 3203 échantillons de ciel sur 3551 ; "
                                          "22,01 sur le champ pour un ciel de 1002,18")
        assert engine_text("stars taken off in 182.3 s on cpu") == "étoiles retirées en 182,3 s sur processeur"
        assert engine_text("noise unit 13.53, background 1.25, 2.5, 3.75") ==             "unité de bruit 13,53, fond 1,25, 2,5, 3,75"
        assert engine_text("the network's grid: 17825 notches placed in the image's windows") ==             "la grille du réseau : 17825 filtres placés dans les fenêtres de l'image"
        assert engine_text("simple output 10 x 20 from stack, useful frame x 0-10, y 0-20; sky: sky level of the "
                           "Gradient block") .endswith("niveau du ciel du bloc Gradient")
        # a line in English stays as it is in English
        appmod.apply_language("en")
        assert engine_text("stars taken off in 182.3 s on cpu") == "stars taken off in 182.3 s on cpu"
    finally:
        appmod.apply_language("en")


def test_the_log_of_a_project_opened_again_is_in_the_language_of_the_window(tmp_path):
    from astro_gpu.client.logview import last_run_text
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "journal.md").write_text(
        "## Run 2026-11-19 10:00:00 - stages: star_removal\n"
        "- 2026-11-19 10:01:00 [star_removal] stars taken off in 182.3 s on cpu\n"
        "- 2026-11-19 10:02:00 [star_removal] a line nobody knows 1.5\n", encoding="utf-8")
    appmod.apply_language("fr")
    try:
        text = last_run_text(tmp_path)
        assert "[star_removal] étoiles retirées en 182,3 s sur processeur" in text
        assert "a line nobody knows 1.5" in text
    finally:
        appmod.apply_language("en")
    assert "stars taken off in 182.3 s on cpu" in last_run_text(tmp_path)
