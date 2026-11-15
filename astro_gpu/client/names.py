"""The names the application gives (a workflow made from a template, a numbered block, a session, a copy, a variant,
the name of a combination), and how they get their language.

A default name is stored in a neutral form, English ("Colour full", "Combine layers 2", "Ha · Denoise", "Ha copy"):
what the project holds never depends on the language the project was made in. It is shown in the interface's language
while it is still the default: ``shown("Colour full")`` is "Couleur complet" in French. A name the user typed is shown
as typed: only a name made of the application's own words (a block, a template, a combination, a copy, a number) is
translated, and each part of "Ha · Denoise" on its own ("Ha" is not one of those words).

A project saved by an earlier version holds default names in the language they were made in ("Couleur complet",
"Combiner calques 2"): ``shown`` reads them in any language the interface has, so they follow the interface too, and
``neutral`` (what is stored from now on) gives them their English form.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache

from . import i18n

SEP = " · "
#: the texts that make a copy's name, by their English source: the name inside is converted on its own
_COPY = ("{_name} copy", "{name} (copy)")
#: words of the application that are names on their own, besides the blocks', the templates' and the combinations'
_WORDS = ("Session", "Image", "Images", "Stars", "Raw stack", "variant", "Starless", "Stars only", "New workflow",
          "Project", "Untitled")


@lru_cache(maxsize=1)
def _catalogues() -> dict[str, dict[str, str]]:
    """{language: {English text: its translation}} for every language the interface has but English."""
    out = {}
    for code in i18n.LANGUAGES:
        if code == "en":
            continue
        cat: dict[str, str] = {}
        for f in sorted((i18n.LOCALE / code).glob("*.json")):
            cat.update(json.loads(f.read_text(encoding="utf-8")))
        out[code] = cat
    return out


@lru_cache(maxsize=1)
def _bases() -> frozenset[str]:
    """The English names the application gives, whole: its blocks, templates, combinations, ports and a few words."""
    from ..graphcore import combos
    from ..graphcore.model import LAYER_PORTS, PAIR_PORTS
    from ..graphcore.rules import LABEL
    from .gallery import GALLERY, PALETTES
    from .graphnode import TITLES
    out = set(LABEL.values()) | combos.titles() | set(LAYER_PORTS) | set(PAIR_PORTS) | set(TITLES.values())
    out |= {t["name"] for t in GALLERY} | set(PALETTES) | set(_WORDS)
    return frozenset(out)


@lru_cache(maxsize=1)
def _reverse() -> dict[str, str]:
    """{a name in some language: its English form}, for the names above that a language writes differently."""
    out: dict[str, str] = {}
    for cat in _catalogues().values():
        for en in sorted(_bases()):
            tr = cat.get(en)
            if tr and tr != en:
                out.setdefault(tr, en)
    return out


@lru_cache(maxsize=1)
def _copies() -> list[tuple[str, list[re.Pattern]]]:
    """(the English template of a copy's name, the patterns that read it in English and in each language)."""
    out = []
    for tmpl in _COPY:
        forms = {tmpl} | {cat.get(tmpl, tmpl) for cat in _catalogues().values()}
        rxs = []
        for f in sorted(forms):
            hole = "{_name}" if "{_name}" in f else "{name}"
            rxs.append(re.compile("^" + re.escape(f).replace(re.escape(hole), "(?P<n>.+)") + "$"))
        out.append((tmpl, rxs))
    return out


#: English names of the application's words that are shown otherwise: the engine's and the graph core's own word, and
#: the name projects made before stored ("Ha · Color"), is shown and stored as the interface writes it
LEGACY = {"Color": "Colour"}


def _word(text: str, to: str) -> str | None:
    """A name of the application in language ``to``, or None when ``text`` is not one."""
    en = LEGACY.get(text) or (text if text in _bases() else _reverse().get(text))
    if en is None:
        return None
    return en if to == "en" else _catalogues().get(to, {}).get(en, en)


def convert(name: str, to: str) -> str:
    """``name`` with its default parts written in language ``to`` ("en", "fr"); anything else as it is."""
    if not name or not isinstance(name, str):
        return name
    if SEP in name:
        return SEP.join(convert(p, to) for p in name.split(SEP))
    for tmpl, rxs in _copies():
        m = next((m for m in (rx.match(name) for rx in rxs) if m), None)
        if m:
            form = tmpl if to == "en" else _catalogues().get(to, {}).get(tmpl, tmpl)
            return form.replace("{_name}", convert(m.group("n"), to)).replace("{name}", convert(m.group("n"), to))
    m = re.fullmatch(r"(.+?)((?: \d+)+)", name)             # "Denoise 2", "Colour minimal 2 2"
    if m:
        w = _word(m.group(1), to)
        if w is not None:
            return w + m.group(2)
    return _word(name, to) or name


def shown(name: str) -> str:
    """A name as the interface shows it: a default name in the interface's language, a typed name as typed."""
    return convert(name, i18n.language())


def neutral(name: str) -> str:
    """A name as it is stored: a default name in English, a typed name as typed."""
    return convert(name, "en")


def _install():
    from ..graphcore import msg
    msg.NAMES = shown


_install()
