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


def say(m, tr=None):
    """The text of a message in the language of `tr` ("" for no message)."""
    if not m:
        return ""
    tr = tr or _same
    if isinstance(m, str):
        return tr(m)
    template, fields = m
    out = {}
    for k, v in fields.items():
        if isinstance(v, tuple) or (isinstance(v, str) and not k.startswith("_")):
            v = say(v, tr)
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


#: the language names a stored default name (Session 2, Stars copy) is written in, set by apply_many for one call
TR = ContextVar("graphcore_tr", default=None)


def name_of(template, **fields):
    """A default name to store (a row or session name): written in the language of the current call."""
    return say((template, fields), TR.get())
