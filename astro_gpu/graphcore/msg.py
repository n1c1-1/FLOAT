"""Messages: every text the core produces for a person is a message, never a finished sentence.

A message is an English source string, or a pair (template, fields) such as ("{n} lights added", {"n": 3}).
`say(message, tr)` writes it in a language: `tr` maps an English source string to its translation (the client's
`_`); with no `tr` the English is written as it stands. String fields are translated too (they name blocks, ports,
settings), unless the field's name starts with an underscore: that is text typed by a person (a name), kept as is.
A field can itself be a message. An empty message is "" (nothing to say)."""
from contextvars import ContextVar


def M(template, **fields):
    """A message with fields."""
    return (template, fields)


def _same(s):
    return s


def _same_name(s):
    return s


#: how a name typed by a person is written in a message: as it is, unless the client says better (a default name the
#: application gave is shown in the interface's language: client/names.py sets this)
NAMES = _same_name


def say(m, tr=None):
    """The text of a message in the language of `tr` ("" for no message)."""
    return _render(m, tr, lambda s: NAMES(s))


def _render(m, tr, names):
    if not m:
        return ""
    tr = tr or _same
    if isinstance(m, str):
        return tr(m)
    template, fields = m
    out = {}
    for k, v in fields.items():
        if isinstance(v, tuple):
            v = _render(v, tr, names)
        elif isinstance(v, str) and not k.startswith("_"):
            t = _render(v, tr, names)
            v = t if t != v else names(v)          # a name no catalogue knows: a default name of the application?
        elif isinstance(v, str):
            v = names(v)
        out[k] = v
    return tr(template).format(**out)


def join(messages, sep=" · "):
    """Several messages as one, in order, separated by `sep` (empty ones left out)."""
    parts = [m for m in messages if m]
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    return (sep.join("{m%d}" % i for i in range(len(parts))), {f"m{i}": m for i, m in enumerate(parts)})


def typed(text):
    """A message that is text typed by a person: never translated."""
    return ("{_text}", {"_text": text})


#: the language a message of one call is written in, set by apply_many (a default name is not: see ``name_of``)
TR = ContextVar("graphcore_tr", default=None)


def name_of(template, **fields):
    """A default name to store (a row or session name): always in English, the neutral form. The client shows it in
    its language while it is the default (client/names.py), so what a project holds never depends on the language it
    was made in."""
    return _render((template, fields), None, _same_name)
